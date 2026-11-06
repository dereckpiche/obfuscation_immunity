"""The base class of every method, and the game it plays.

Each trajectory is a pair of answers to one puzzle: an honest one s, then a
sneaky one s̄, written with s in view, that should look right but be wrong
(common.pairs). V, the puzzle's grader, labels both; a method's judge J
says which it believes correct, y = (y_s, y_s̄) ∈ {0,1}². The policy plays

    max_π E[ V((1,0) | s, s̄) + KL(V ‖ J) ],    KL(V ‖ J) = −log J(y_true)

so it is paid for s right, s̄ wrong and a judge that cannot tell. A method
is its judge:

    judge(loop, pair) -> (log J over ANSWERS in (s, s̄) order, metrics)
    train(trainer, step)     after each actor update; by default nothing
    prepare()                before verl starts; by default nothing

Its keyword arguments are its hyperparameters, `learn()` trains it and
`cli()` is its command line; verl's config is built here from them.
"""

import argparse
import importlib
import inspect
import json
import math
import os
import re
import time
from functools import cache
from pathlib import Path
from typing import ClassVar

from obfuscation_immunity.common import progress, puzzles
from obfuscation_immunity.common.pairs import SNEAKY, Pair, readable

ROOT = Path(__file__).resolve().parents[2]
METHOD = "OI_METHOD"
"""The environment variable that carries the method to every process."""
REQUEST = 256
"""Room in a trajectory for a user turn between two answers."""
RUN_DIR = "OI_RUN_DIR"
"""The one that carries the run directory: checkpoints and method files."""
MODEL_LENGTH = 8192
"""vLLM's context: room for a judge's question, both answers in it, and
the judge's thinking."""


class BaseMethod:
    """The game on reasoning-gym puzzles, trained with GRPO and LoRA.

    :param puzzle: the reasoning-gym puzzle the run trains on
    :param data: the directory that keeps a dataset per puzzle, written
        there on first use
    :param model: the policy's base model
    :param steps: training steps
    :param prompts: puzzles per step
    :param group: answers per puzzle, GRPO's group
    :param lr: the actor's learning rate
    :param lora_rank: the actor's LoRA rank (alpha is twice it)
    :param kl_coef: β of the KL penalty to the starting model
    :param reasoning: tokens the policy may think for per answer; past
        them its thinking is closed for it and it answers
    :param answering: tokens it may write per answer after its thinking
    :param floor: J's floor inside the log, so the reward stays finite
    :param gpus: training GPUs, one replica each
    :param name: the run's name, by default the method's
    :param runs: the directory that holds a directory per run
    :param verl: verl config overrides, `key=value`
    """

    verl_overrides: ClassVar[list[str]] = []
    """What this method changes in verl's config."""
    kwargs: dict
    """The arguments the method was built with."""

    def __new__(cls, **kwargs):
        """Keep the arguments: every process of the run rebuilds the method
        from them.
        """
        self = super().__new__(cls)
        self.kwargs = kwargs
        return self

    def __init__(
        self,
        *,
        puzzle: str = puzzles.PUZZLE,
        data: str = "data",
        model: str = "Qwen/Qwen3.5-4B",
        steps: int = 500,
        prompts: int = 8,
        group: int = 4,
        lr: float = 3e-6,
        lora_rank: int = 64,
        kl_coef: float = 1e-4,
        reasoning: int = 2048,
        answering: int = 256,
        floor: float = 1e-3,
        gpus: int = 1,
        name: str = "",
        runs: str = "runs",
        verl: list[str] | None = None,
    ):
        """The game's arguments, as the class describes them."""
        self.puzzle = puzzle
        self.data = Path(data).resolve()
        self.model = model
        self.steps = steps
        self.prompts = prompts
        self.group = group
        self.lr = lr
        self.lora_rank = lora_rank
        self.kl_coef = kl_coef
        self.reasoning = reasoning
        self.answering = answering
        self.floor = floor
        self.response_length = 2 * (reasoning + answering + 16) + REQUEST
        self.gpus = gpus
        self.name = name or type(self).__name__
        self.runs = Path(runs).resolve()
        self.verl = list(verl or [])

    @property
    def run_dir(self) -> Path:
        """Where the run keeps its checkpoints and files."""
        return self.runs / self.name

    async def rollout(self, loop, sampling_params, row):
        """The honest answer, the request, the sneaky answer; the judge's
        verdict; the game's reward; how long each part took.
        """
        clock = [time.monotonic()]
        messages = list(row["raw_prompt"])
        prompt = await loop.ct_build_initial_tokens(messages)
        budget = (sampling_params, self.reasoning, self.answering)
        ids, mask, logprobs, honest = await loop.answer(prompt, [], [], *budget)
        clock.append(time.monotonic())
        end = len(ids)
        answered = messages + [{"role": "assistant", "content": honest}]
        ids, mask, logprobs = await loop.told(
            ids, mask, logprobs, answered, SNEAKY
        )
        start = len(ids)
        ids, mask, logprobs, sneaky = await loop.answer(
            ids, mask, logprobs, *budget
        )
        clock.append(time.monotonic())
        task = json.loads(row["extra_info"]["task"])
        pair = Pair(
            task="\n\n".join(m["content"] for m in messages),
            honest=readable(honest),
            sneaky=readable(sneaky),
            honest_correct=puzzles.correct(puzzles.grade(task, honest)),
            sneaky_correct=puzzles.correct(puzzles.grade(task, sneaky)),
            prompt=ids[: len(prompt)],
            honest_ids=ids[len(prompt) : end],
            sneaky_ids=ids[start:],
        )
        logp, measured = await self.judge(loop, pair)
        clock.append(time.monotonic())
        loss = -max(logp[pair.truth], math.log(self.floor))
        likeliest = max(range(len(logp)), key=lambda k: logp[k])
        metrics = {
            "honest_correct": float(pair.honest_correct),
            "sneaky_correct": float(pair.sneaky_correct),
            "judge_accuracy": float(likeliest == pair.truth),
            "judge_logloss": loss,
            **measured,
        }
        reward = pair.only_honest_correct + loss
        return loop.output(
            len(prompt),
            ids,
            mask,
            logprobs,
            reward,
            metrics,
            task=row["extra_info"]["index"],
            after=after(pair, logp, loss, reward),
            seconds=dict(zip(PARTS, (b - a for a, b in zip(clock, clock[1:])))),
        )

    async def judge(self, loop, pair: Pair):
        """log J(y | s, s̄) for y in ANSWERS, (s, s̄) order, and what the
        judge measured.
        """
        raise NotImplementedError

    def prepare(self) -> None:
        """Before verl starts, in the process that starts it: nothing."""

    def train(self, trainer, step: int) -> None:
        """After the actor's update `step` (at the start, the steps done
        so far): nothing.
        """

    def learn(self) -> None:
        """Train: the puzzles written if they are not yet, then verl in
        this process, its workers rebuilding the method.
        """
        from verl.trainer.main_ppo import run_ppo
        from verl.trainer.ppo.utils import need_critic, need_reference_policy
        from verl.utils.config import validate_config
        from verl.utils.device import auto_set_device

        from obfuscation_immunity.common.trainer import Runner

        dataset = self.data / self.puzzle
        if not (dataset / "train.parquet").exists():
            puzzles.write(dataset, self.puzzle)
        config = self.config()
        auto_set_device(config)
        validate_config(
            config,
            use_reference_policy=need_reference_policy(config),
            use_critic=need_critic(config),
        )
        os.environ[METHOD] = json.dumps(self.spec())
        os.environ[RUN_DIR] = str(self.run_dir)
        os.environ[progress.EPISODES] = str(self.run_dir / "episodes.jsonl")
        paths = [str(ROOT), os.environ.get("PYTHONPATH", "")]
        os.environ["PYTHONPATH"] = os.pathsep.join(p for p in paths if p)
        run_ppo(config, task_runner_class=Runner)

    def spec(self) -> dict:
        """The method's class and arguments, its paths absolute, as every
        process of the run rebuilds it.
        """
        cls = type(self)
        paths = {"data": str(self.data), "runs": str(self.runs)}
        kwargs = {**self.kwargs, **paths}
        return {
            "module": cls.__module__,
            "class": cls.__name__,
            "kwargs": kwargs,
        }

    def config(self):
        """verl's default config with this recipe, the method's changes and
        the run's own overrides, last.
        """
        from hydra import compose, initialize_config_module

        self.run_dir.mkdir(parents=True, exist_ok=True)
        loops = self.run_dir / "loop.yaml"
        target = "obfuscation_immunity.common.loop.MethodLoop"
        loops.write_text(f"- name: oi\n  _target_: {target}\n")
        overrides = [
            *self.recipe(self.run_dir, loops),
            *self.verl_overrides,
            *self.verl,
        ]
        with initialize_config_module("verl.trainer.config", version_base=None):
            return compose("ppo_trainer", overrides=overrides)

    def recipe(self, run_dir: Path, loops: Path) -> list[str]:
        """The training recipe: Qwen3.5 with LoRA, GRPO, vLLM rollouts on
        the training GPUs, sized for one L40S per replica: the actor whole
        on each (no sharding), in fp32, since verl would cast its adapter
        to a bf16 base's dtype, where small updates round away; vLLM in
        the rest. The adapter is small and skips the vision tower: the
        puzzles are text, and RL needs little capacity.

        The weight sync's bucket holds the 4B's embedding (2.4 GiB in
        fp32), which verl cannot send outside one.

        No validation: the puzzles are many, so every training pair is a
        fresh one (verl still builds a validation set, from the training
        file, never run). GRPO's advantages are not divided by the group's
        spread: the judge's log-loss is continuous, and a group whose
        rewards nearly agree would have its noise scaled up to full
        advantages.
        """
        actor, rollout = "actor_rollout_ref.actor", "actor_rollout_ref.rollout"
        tokens = self.response_length + 1280
        dataset = self.data / self.puzzle
        return [
            f"data.train_files={dataset / 'train.parquet'}",
            f"data.val_files={dataset / 'train.parquet'}",
            f"data.train_batch_size={self.prompts}",
            "data.max_prompt_length=1024",
            f"data.max_response_length={self.response_length}",
            "data.return_raw_chat=true",
            "+data.apply_chat_template_kwargs.enable_thinking=true",
            "data.filter_overlong_prompts=true",
            "data.dataloader_num_workers=0",
            "algorithm.adv_estimator=grpo",
            "algorithm.norm_adv_by_std_in_grpo=false",
            "algorithm.rollout_correction.rollout_is=token",
            "algorithm.rollout_correction.rollout_is_threshold=2.0",
            "algorithm.rollout_correction.rollout_rs=seq_mean_k1",
            "algorithm.rollout_correction.rollout_rs_threshold=0.99_1.01",
            f"actor_rollout_ref.model.path={self.model}",
            "actor_rollout_ref.model.use_remove_padding=true",
            "actor_rollout_ref.model.enable_gradient_checkpointing=true",
            "+actor_rollout_ref.model.override_config.attn_implementation="
            "flash_attention_2",
            f"actor_rollout_ref.model.lora_rank={self.lora_rank}",
            f"actor_rollout_ref.model.lora_alpha={2 * self.lora_rank}",
            "actor_rollout_ref.model.exclude_modules=.*visual.*",
            "actor_rollout_ref.model.lora.merge=false",
            f"{actor}.strategy=fsdp2",
            f"{actor}.optim.lr={self.lr}",
            f"{actor}.optim.weight_decay=0.0",
            f"{actor}.ppo_mini_batch_size={self.prompts}",
            f"{actor}.ppo_epochs=1",
            f"{actor}.loss_agg_mode=seq-mean-token-mean",
            f"{actor}.use_dynamic_bsz=true",
            f"{actor}.ppo_max_token_len_per_gpu={tokens}",
            f"{actor}.use_kl_loss=true",
            f"{actor}.kl_loss_coef={self.kl_coef}",
            f"{actor}.kl_loss_type=low_var_kl",
            f"{actor}.entropy_coeff=0",
            f"{actor}.entropy_from_logits_with_chunking=true",
            f"{actor}.fsdp_config.fsdp_size=1",
            f"+{actor}.checkpoint.save_lora_only=true",
            f"{rollout}.name=vllm",
            f"{rollout}.mode=async",
            f"{rollout}.tensor_model_parallel_size=1",
            f"{rollout}.gpu_memory_utilization=0.4",
            f"{rollout}.max_model_len={MODEL_LENGTH}",
            f"{rollout}.max_num_seqs=32",
            f"{rollout}.checkpoint_engine.update_weights_bucket_megabytes=3072",
            f"{rollout}.temperature=1.0",
            f"{rollout}.n={self.group}",
            f"{rollout}.log_prob_use_dynamic_bsz=true",
            f"{rollout}.log_prob_max_token_len_per_gpu={tokens}",
            f"{rollout}.agent.num_workers=2",
            f"{rollout}.agent.agent_loop_config_path={loops}",
            f"{rollout}.agent.default_agent_loop=oi",
            "trainer.logger=[]",
            "trainer.project_name=obfuscation_immunity",
            f"trainer.experiment_name={self.name}",
            f"trainer.n_gpus_per_node={self.gpus}",
            f"trainer.total_training_steps={self.steps}",
            "trainer.val_before_train=false",
            "trainer.test_freq=-1",
            "trainer.save_freq=50",
            f"trainer.default_local_dir={run_dir}",
            "trainer.resume_mode=auto",
            "trainer.max_actor_ckpt_to_keep=1",
        ]

    @classmethod
    def cli(cls, argv: list[str] | None = None) -> None:
        """The command line: every keyword argument of the method and its
        bases as a flag (`--freeze-at 50`), described by its `:param:`;
        anything else is a verl override (`trainer.save_freq=20`). A saved
        setup (`--config configs/<setup>.yaml`) gives the defaults: the
        command line's flags win, and its verl overrides come after the
        setup's.
        """
        package = cls.__module__.rpartition(".")[0]
        parser = argparse.ArgumentParser(
            prog=f"python -m {package}", description=cls.__doc__
        )
        parser.add_argument(
            "--config",
            type=Path,
            help="a saved setup: these arguments, and `verl`, a list of "
            "verl overrides, as YAML",
        )
        for name, param, help in arguments(cls):
            if name == "verl":
                continue
            kind = type(param.default)
            parser.add_argument(
                f"--{name.replace('_', '-')}",
                type=(lambda s: s.lower() == "true") if kind is bool else kind,
                default=param.default,
                help=help,
            )
        known, _ = parser.parse_known_args(argv)
        setup = setups(known.config, parser) if known.config else {}
        verl = setup.pop("verl", [])
        parser.set_defaults(**setup)
        args, overrides = parser.parse_known_args(argv)
        kwargs = {k: v for k, v in vars(args).items() if k != "config"}
        cls(**kwargs, verl=[*verl, *overrides]).learn()


PARTS = ["honest", "sneaky", "judge"]
"""A pair's parts, timed."""
OUTCOMES = [
    "both right",
    "only honest right",
    "only sneaky right",
    "neither right",
]
"""ANSWERS in words."""


def after(pair: Pair, logp: list[float], loss: float, reward: float):
    """What follows a pair on its page, as (kind, who, text, fold): the
    grade (green when only the honest answer is right, amber when the
    sneaky one is right too, red when the honest one is wrong), the judge's
    J with the truth marked, and the reward with its parts.
    """
    mark = {True: "right", False: "wrong"}
    honest, sneaky = mark[pair.honest_correct], mark[pair.sneaky_correct]
    grade = f"honest {honest} · sneaky {sneaky}"
    tone = "good" if pair.only_honest_correct else "bad"
    if pair.honest_correct and pair.sneaky_correct:
        tone = "warn"
    verdict = "\n".join(
        f"{'→' if k == pair.truth else '·'} {word}: {math.exp(v):.2f}"
        for k, (word, v) in enumerate(zip(OUTCOMES, logp))
    )
    parts = (
        f"{reward:.3f} = {pair.only_honest_correct:g} (V)"
        f" + {loss:.3f} (judge's log-loss)"
    )
    return [
        (tone, "Grade", grade, ""),
        ("method", "Judge", verdict, ""),
        ("", "Reward", parts, ""),
    ]


def setups(path: Path, parser: argparse.ArgumentParser) -> dict:
    """A saved setup's arguments, checked against the command line's."""
    import yaml

    setup = yaml.safe_load(path.read_text()) or {}
    known = set(vars(parser.parse_known_args([])[0])) | {"verl"}
    if unknown := set(setup) - known:
        parser.error(f"{path} sets {', '.join(sorted(unknown))}, not taken")
    return setup


def arguments(cls):
    """Each keyword argument of `cls` and its bases: name, parameter and
    the `:param:` line describing it.
    """
    found = {}
    for klass in reversed(cls.__mro__):
        if "__init__" not in vars(klass):
            continue
        helps = dict(re.findall(r":param (\w+): (.+)", klass.__doc__ or ""))
        signature = inspect.signature(vars(klass)["__init__"])
        for name, param in signature.parameters.items():
            if param.kind is param.KEYWORD_ONLY:
                found[name] = (name, param, helps.get(name, ""))
    return list(found.values())


@cache
def running() -> BaseMethod:
    """The run's method, rebuilt in this process from METHOD."""
    spec = json.loads(os.environ[METHOD])
    cls = getattr(importlib.import_module(spec["module"]), spec["class"])
    return cls(**spec["kwargs"])
