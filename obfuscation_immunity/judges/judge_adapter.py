"""
The judge adapter on verl's own actor model, trained by verl's own update.

An adapter judge's weights are a second LoRA adapter, ψ ("judge"), on the
actor's model next to verl's own, π ("default"): no second model. It is
added as verl builds the model, before FSDP wraps it
(`JudgeAdapterEngine._build_lora_module`), so FSDP shards ψ and reduces its
gradients as it does π's. Actor and judge take turns on the same engine:
verl's update of the actor, then verl's update of the judge, through the
same pipeline (common.trainer `PairTrainer.update_on_partition`). For the
judge's turn the engine switches (`use_judge_adapter`):

                      thinks with   learns   reference for the KL
    actor             π             π        the base model
    standalone judge  ψ             ψ        the base model
    stacked judge     π + ψ         ψ        π

and steps ψ with its own optimizer and learning-rate schedule, built by
verl from the actor's optimizer config with the judge's learning rate. ψ
starts at zero. verl's checkpoint holds ψ with π; the judge's optimizer is
saved beside it.
"""

import copy
import dataclasses
import functools
import shutil
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from omegaconf import OmegaConf
from peft import LoraConfig, PeftModel, get_peft_model_state_dict
from safetensors.torch import save_file
from verl.single_controller.base.decorator import Dispatch, register
from verl.workers.config.optimizer import build_optimizer
from verl.workers.engine.fsdp.transformer_impl import FSDPEngineWithLMHead
from verl.workers.engine_workers import ActorRolloutRefWorker, TrainingWorker

ACTOR_ADAPTER_NAME = "default"
"""The actor's LoRA adapter, as verl names it."""

JUDGE_ADAPTER_NAME = "judge"
"""The judge's LoRA adapter, next to the actor's on the same model."""

JUDGE_ADAPTER_CONFIG_KEY = "judge_adapter"
"""Where the judge adapter's settings sit in `actor_rollout_ref`."""


@dataclass(frozen=True)
class JudgeAdapterSettings:
    """
    The judge adapter's settings, as verl's workers read them.

    Attributes:
        rank: ψ's LoRA rank; its α/r is the actor adapter's.
        lr: ψ's learning rate.
        stacked: Whether the judge thinks with π + ψ, rather than ψ alone.
    """

    rank: int
    lr: float
    stacked: bool


def adapter_parameters(
    model: PeftModel, adapter_name: str
) -> dict[str, torch.nn.Parameter]:
    """
    Lists one adapter's parameters.

    Args:
        model: The actor's peft model.
        adapter_name: ACTOR_ADAPTER_NAME or JUDGE_ADAPTER_NAME.

    Returns:
        The adapter's parameters, by name.
    """
    marker = f".{adapter_name}."
    return {
        name: parameter
        for name, parameter in model.named_parameters()
        if marker in name
    }


def switch_adapters(model: PeftModel, active: list[str], learning: str) -> None:
    """
    Switches adapters on and sets which one learns.

    peft switches `requires_grad` with the adapters; this sets it back so
    that only `learning`'s parameters take a gradient.

    Args:
        model: The actor's peft model.
        active: The adapters the model thinks with.
        learning: The adapter whose parameters learn.
    """
    model.base_model.set_adapter(active)
    for adapter_name in (ACTOR_ADAPTER_NAME, JUDGE_ADAPTER_NAME):
        for parameter in adapter_parameters(model, adapter_name).values():
            parameter.requires_grad_(adapter_name == learning)


def add_judge_adapter(model: PeftModel, rank: int) -> None:
    """
    Adds the judge adapter, at zero, next to the actor's, switched off.

    It covers the actor adapter's modules with its α/r and dtype; only the
    actor's adapter stays on and learns.

    Args:
        model: The actor's peft model.
        rank: The judge adapter's LoRA rank.
    """
    actor_config = model.peft_config[ACTOR_ADAPTER_NAME]
    assert isinstance(actor_config, LoraConfig)
    judge_config = copy.deepcopy(actor_config)
    judge_config.r = rank
    judge_config.lora_alpha = actor_config.lora_alpha * rank // actor_config.r
    model.add_adapter(JUDGE_ADAPTER_NAME, judge_config)
    actor_parameters = adapter_parameters(model, ACTOR_ADAPTER_NAME)
    dtype = next(iter(actor_parameters.values())).dtype
    for parameter in adapter_parameters(model, JUDGE_ADAPTER_NAME).values():
        parameter.data = parameter.data.to(dtype)
    switch_adapters(model, [ACTOR_ADAPTER_NAME], ACTOR_ADAPTER_NAME)


def stacked_lora(
    actor_config: LoraConfig,
    actor_weights: dict[str, torch.Tensor],
    judge_config: LoraConfig,
    judge_weights: dict[str, torch.Tensor],
) -> tuple[LoraConfig, dict[str, torch.Tensor]]:
    """
    Stacks the actor's and the judge's adapters into one LoRA.

    A's rows and B's columns side by side: exact, since both adapters have
    the same α/r.

    Args:
        actor_config: π's LoRA config.
        actor_weights: π's weights, in PEFT's format.
        judge_config: ψ's LoRA config.
        judge_weights: ψ's weights, in PEFT's format.

    Returns:
        The stacked LoRA's config and weights.
    """
    weights = {
        key: torch.cat(
            [weight, judge_weights[key]], dim=0 if "lora_A" in key else 1
        )
        for key, weight in actor_weights.items()
    }
    config = copy.deepcopy(actor_config)
    config.r = actor_config.r + judge_config.r
    config.lora_alpha = actor_config.lora_alpha * config.r // actor_config.r
    return config, weights


def gathered_adapter(model: PeftModel, adapter_name: str) -> dict:
    """
    Gathers one adapter's weights whole, on every rank.

    FSDP holds them as shards; gathering is collective, so every rank calls
    it, in the same order.

    Args:
        model: The actor's peft model.
        adapter_name: ACTOR_ADAPTER_NAME or JUDGE_ADAPTER_NAME.

    Returns:
        The adapter's weights, in PEFT's format, on the CPU.
    """
    weights = get_peft_model_state_dict(model, adapter_name=adapter_name)
    return {
        key: (
            weight.full_tensor() if hasattr(weight, "full_tensor") else weight
        )
        .detach()
        .cpu()
        for key, weight in weights.items()
    }


class JudgeAdapterEngine(FSDPEngineWithLMHead):
    """
    verl's FSDP engine, with the judge adapter on its model.

    Attributes:
        settings: The judge adapter's settings.
        judging: Whether the judge's turn is on.
        actor_optimizer: verl's optimizer, kept aside during the judge's turn.
        actor_lr_scheduler: verl's schedule, kept aside likewise.
        judge_optimizer: ψ's optimizer, once built.
        judge_lr_scheduler: ψ's schedule, once built.
    """

    settings: JudgeAdapterSettings
    judging: bool
    actor_optimizer: Any
    actor_lr_scheduler: Any
    judge_optimizer: Any
    judge_lr_scheduler: Any

    def __init__(self, *args: Any, settings: JudgeAdapterSettings, **kwargs):
        """
        Builds verl's engine and keeps the judge adapter's settings.

        Args:
            *args: verl's arguments for the engine.
            settings: The judge adapter's settings.
            **kwargs: verl's keyword arguments for the engine.
        """
        super().__init__(*args, **kwargs)
        self.settings = settings
        self.judging = False
        self.actor_optimizer = self.actor_lr_scheduler = None
        self.judge_optimizer = self.judge_lr_scheduler = None

    @property
    def peft_model(self) -> PeftModel:
        """The actor's peft model, inside FSDP's wrapping."""
        return getattr(self.module, "_fsdp_wrapped_module", self.module)

    def _build_lora_module(self, module: Any) -> PeftModel:
        """
        Builds verl's LoRA, then adds the judge adapter, switched off.

        Args:
            module: The base model.

        Returns:
            The peft model, both adapters on it, the actor's on.
        """
        module = super()._build_lora_module(module)
        add_judge_adapter(module, self.settings.rank)
        return module

    def judge_adapters(self) -> list[str]:
        """
        Lists the adapters the judge thinks with.

        Returns:
            π and ψ for the stacked judge; ψ alone otherwise.
        """
        if self.settings.stacked:
            return [ACTOR_ADAPTER_NAME, JUDGE_ADAPTER_NAME]
        return [JUDGE_ADAPTER_NAME]

    def build_judge_optimizer(self) -> None:
        """Builds ψ's optimizer and schedule with verl's, the first time."""
        if self.judge_optimizer is not None:
            return
        judge_config = dataclasses.replace(
            self.optimizer_config, lr=self.settings.lr
        )
        parameters = adapter_parameters(self.peft_model, JUDGE_ADAPTER_NAME)
        self.judge_optimizer = build_optimizer(
            list(parameters.values()), judge_config
        )
        self.judge_lr_scheduler = self._build_lr_scheduler(self.judge_optimizer)

    def use_judge_adapter(self, judging: bool) -> None:
        """
        Switches the model and the optimizer to the judge's turn, or back.

        Args:
            judging: True for the judge's turn, False for the actor's.
        """
        if judging == self.judging:
            return
        if judging:
            self.build_judge_optimizer()
            switch_adapters(
                self.peft_model, self.judge_adapters(), JUDGE_ADAPTER_NAME
            )
            self.actor_optimizer = self.optimizer
            self.actor_lr_scheduler = self.lr_scheduler
            self.optimizer = self.judge_optimizer
            self.lr_scheduler = self.judge_lr_scheduler
        else:
            switch_adapters(
                self.peft_model, [ACTOR_ADAPTER_NAME], ACTOR_ADAPTER_NAME
            )
            self.optimizer = self.actor_optimizer
            self.lr_scheduler = self.actor_lr_scheduler
        self.judging = judging

    def disable_adapter(self) -> Any:
        """
        Switches to the reference for the KL loss.

        Returns:
            For the stacked judge's turn, a context with π alone on; else
            verl's, with every adapter off.
        """
        if self.judging and self.settings.stacked:
            return self.actor_adapter_alone()
        return super().disable_adapter()

    @contextmanager
    def actor_adapter_alone(self) -> Generator[None, None, None]:
        """
        Switches π alone on, then the judge's adapters back on.

        Yields:
            Nothing: the model is the reference meanwhile.
        """
        switch_adapters(
            self.peft_model, [ACTOR_ADAPTER_NAME], JUDGE_ADAPTER_NAME
        )
        try:
            yield
        finally:
            switch_adapters(
                self.peft_model, self.judge_adapters(), JUDGE_ADAPTER_NAME
            )

    def write_served_adapter(self, directory: Path, stale: Path) -> None:
        """
        Writes the LoRA vLLM serves the judge with, from the first rank.

        It is ψ alone, or π and ψ stacked for the stacked judge, in bf16.

        Args:
            directory: Where to write it.
            stale: An older one to remove.
        """
        model = self.peft_model
        actor_config = model.peft_config[ACTOR_ADAPTER_NAME]
        judge_config = model.peft_config[JUDGE_ADAPTER_NAME]
        assert isinstance(actor_config, LoraConfig)
        assert isinstance(judge_config, LoraConfig)
        judge_weights = gathered_adapter(model, JUDGE_ADAPTER_NAME)
        config, weights = copy.deepcopy(judge_config), judge_weights
        if self.settings.stacked:
            config, weights = stacked_lora(
                actor_config,
                gathered_adapter(model, ACTOR_ADAPTER_NAME),
                judge_config,
                judge_weights,
            )
        if self.rank != 0:
            return
        config.save_pretrained(str(directory))
        save_file(
            {
                key: weight.to(torch.bfloat16).contiguous()
                for key, weight in weights.items()
            },
            directory / "adapter_model.safetensors",
        )
        shutil.rmtree(stale, ignore_errors=True)

    def judge_optimizer_file(self, directory: str) -> Path:
        """
        Gives this rank's file for ψ's optimizer in a checkpoint.

        Args:
            directory: verl's checkpoint directory for the actor.

        Returns:
            The file.
        """
        return Path(directory) / f"judge_optimizer_rank_{self.rank}.pt"

    def save_judge_optimizer(self, directory: str) -> None:
        """
        Saves ψ's optimizer and schedule beside verl's checkpoint.

        Args:
            directory: verl's checkpoint directory for the actor.
        """
        if self.judge_optimizer is None:
            return
        torch.save(
            {
                "optimizer": self.judge_optimizer.state_dict(),
                "lr_scheduler": self.judge_lr_scheduler.state_dict(),
            },
            self.judge_optimizer_file(directory),
        )

    def load_judge_optimizer(self, directory: str) -> None:
        """
        Loads ψ's optimizer and schedule, if the checkpoint has them.

        Args:
            directory: verl's checkpoint directory for the actor.
        """
        saved_file = self.judge_optimizer_file(directory)
        if not saved_file.exists():
            return
        self.build_judge_optimizer()
        saved = torch.load(saved_file, weights_only=False)
        self.judge_optimizer.load_state_dict(saved["optimizer"])
        self.judge_lr_scheduler.load_state_dict(saved["lr_scheduler"])


class JudgeAdapterTrainingWorker(TrainingWorker):
    """verl's training worker, whose engine carries the judge adapter."""

    def __init__(self, config: Any, settings: JudgeAdapterSettings):
        """
        Builds verl's worker, then swaps in the judge adapter's engine.

        The model is built later (`reset`), so it is built with ψ.

        Args:
            config: verl's training worker config.
            settings: The judge adapter's settings.
        """
        super().__init__(config)
        self.engine = JudgeAdapterEngine(
            model_config=self.model_config,
            engine_config=self.engine_config,
            optimizer_config=self.optimizer_config,
            checkpoint_config=self.checkpoint_config,
            settings=settings,
        )


class JudgeAdapterWorker(ActorRolloutRefWorker):
    """verl's actor worker, with the judge adapter on the actor's model."""

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self) -> None:
        """Builds verl's actor and rollout, the actor with the judge adapter."""
        settings_values = OmegaConf.to_container(
            self.config[JUDGE_ADAPTER_CONFIG_KEY]
        )
        assert isinstance(settings_values, dict)
        settings = JudgeAdapterSettings(**settings_values)
        self.actor_worker_cls = functools.partial(
            JudgeAdapterTrainingWorker, settings=settings
        )
        super().init_model()

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def use_judge_adapter(self, judging: bool) -> None:
        """
        Switches the actor's engine to the judge's turn, or back.

        Args:
            judging: True for the judge's turn, False for the actor's.
        """
        self.actor.engine.use_judge_adapter(judging)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def write_served_adapter(self, directory: str, stale: str) -> None:
        """
        Writes the LoRA vLLM serves the judge with.

        Args:
            directory: Where to write it.
            stale: An older one to remove.
        """
        self.actor.engine.write_served_adapter(Path(directory), Path(stale))

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def save_checkpoint(
        self,
        local_path: str,
        hdfs_path: str | None = None,
        global_step: int = 0,
        max_ckpt_to_keep: int | None = None,
    ) -> None:
        """
        Saves verl's checkpoint, then ψ's optimizer beside it.

        Args:
            local_path: verl's checkpoint directory for the actor.
            hdfs_path: verl's remote copy, if any.
            global_step: The step saved.
            max_ckpt_to_keep: How many checkpoints verl keeps.
        """
        super().save_checkpoint(
            local_path, hdfs_path, global_step, max_ckpt_to_keep
        )
        self.actor.engine.save_judge_optimizer(local_path)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def load_checkpoint(
        self,
        local_path: str,
        hdfs_path: str | None = None,
        del_local_after_load: bool = False,
    ) -> None:
        """
        Loads verl's checkpoint, then ψ's optimizer beside it.

        Args:
            local_path: verl's checkpoint directory for the actor.
            hdfs_path: verl's remote copy, if any.
            del_local_after_load: Whether verl deletes the local copy.
        """
        self.actor.engine.load_judge_optimizer(local_path)
        super().load_checkpoint(local_path, hdfs_path, del_local_after_load)
