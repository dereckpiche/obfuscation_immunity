import copy
import os
import shutil
import tempfile
from pathlib import Path

import torch
from torch.utils.checkpoint import checkpoint

from obfuscation_immunity.common import workers
from obfuscation_immunity.common.base_method import RUN_DIR, BaseMethod
from obfuscation_immunity.common.pairs import (
    ANSWERS,
    Pair,
    asked,
    reasoned,
    unshuffled,
)


class CoevolvingJudge(BaseMethod):
    """A judge that co-evolves with the actor (research note loopholes.md).

    The judge J_{ψ,π} is the actor's own model with a second LoRA adapter,
    ψ, switched on next to the actor's π. Asked the pair's question, it
    thinks (z), then its next-token probabilities over the four answers
    are J, and it plays the other side of the game:

        max_π min_ψ E[ V((1,0) | s, s̄) + KL(V ‖ J_{ψ,π}) ]

    With one thought z per pair, J(y) is read as p(y | q, z), and ψ
    descends E_z[−log p(y_true | q, z)], a bound on −log J(y_true): the
    gradient of the verdict, plus the score-function term that teaches
    the thinking, (log p(y_true | q, z) − b) ∇ log p(z | q), b a running
    mean. ψ's gradient is taken as each pair is judged, with the π that
    wrote it, and ψ steps on their mean after the actor's update. After K
    steps ψ is frozen; nothing else changes. ψ starts at zero, so at first
    J_{ψ,π} is the actor judging itself, as in ActorJudge.

    The judge thinks in vLLM, with π and ψ stacked into one adapter
    (`write`, served by coevolving_judge.server), and is read on the
    training workers, which are idle while vLLM generates: no extra GPU,
    and ψ never reaches the actor's optimizer. ψ is saved to the run
    directory after each step and read back on a restart.

    :param freeze_at: K, the steps after which ψ is updated
    :param psi_rank: ψ's LoRA rank; π's and ψ's together are at most 512,
        vLLM's largest
    :param psi_lr: ψ's AdamW learning rate
    :param judging: tokens the judge may think for before its verdict
    """

    verl_overrides = ["actor_rollout_ref.rollout.gpu_memory_utilization=0.35"]
    """Room on the GPU for the judge's backward pass during rollouts."""

    def __init__(
        self,
        *,
        freeze_at: int = 100,
        psi_rank: int = 64,
        psi_lr: float = 1e-4,
        judging: int = 1024,
        **kwargs,
    ):
        """The game's arguments, and ψ's."""
        super().__init__(**kwargs)
        self.freeze_at = freeze_at
        self.psi_rank = psi_rank
        self.psi_lr = psi_lr
        self.judging = judging

    def prepare(self) -> None:
        """Serve the judge's adapter next to π in vLLM."""
        from obfuscation_immunity.coevolving_judge import server

        server.register()

    async def judge(self, loop, pair: Pair):
        """J_{ψ,π} thinks in vLLM, then is read on a training worker,
        where the pair also gives ψ its gradient.
        """
        ids, letters, honest_first = asked(loop.tokenizer, pair)
        adapter = await workers.on_a_worker(current)
        full, thought = await reasoned(loop, ids, self.judging, adapter=adapter)
        order = unshuffled(list(range(len(ANSWERS))), honest_first)
        truth = order[pair.truth]
        logp = await workers.on_a_worker(
            self.read, full, len(ids), thought, letters, truth
        )
        return unshuffled(logp, honest_first), {"judge_thinking": thought}

    def train(self, trainer, step: int) -> None:
        """After the actor's update: ψ's step on every worker at once."""
        workers.on_every_worker(
            trainer.actor_rollout_wg.workers, self.update, step
        )

    def read(self, model, ids, start, length, letters, truth) -> list[float]:
        """On a worker: log J of answers A to D with π and ψ on, after the
        question and the `length` tokens of thinking from `start`; while
        ψ learns, its gradient is added to the sum kept aside. ψ's
        gradient is taken alone, so π's parameters and their gradients are
        untouched.
        """
        psi = self.psi(model)
        learn = not PSI["frozen"]
        keep = len(ids) - start + 1 if learn else 1
        with (
            adapters(model, psi, learn),
            torch.set_grad_enabled(learn),
            workers.last_logits(model, keep),
        ):
            input_ids = torch.tensor([ids], device=PSI["pairs"].device)
            logits = model(input_ids=input_ids, use_cache=False).logits[0]
            logp = torch.log_softmax(logits[-1, letters].float(), -1)
            if learn:
                thought = input_ids[0, start : start + length]
                thinking = logprob(logits[:length], thought)
                verdict = logp[truth]
                gain = verdict.item() - PSI.setdefault(
                    "baseline", verdict.item()
                )
                loss = -(verdict + gain * thinking)
                grads = torch.autograd.grad(loss, list(psi.values()))
                for name, grad in zip(psi, grads):
                    PSI["grads"][name] += grad
                PSI["pairs"] += 1
                PSI["baseline"] += 0.1 * gain
        return logp.detach().tolist()

    def update(self, model, step: int) -> None:
        """On every worker, after the actor's update `step`: ψ's AdamW
        step on the mean gradient over all workers' pairs while step ≤ K,
        then ψ saved; frozen once K steps are done; π and ψ written as the
        judge's adapter, since π moved.
        """
        psi = self.psi(model)
        grads, pairs = PSI["grads"], PSI["pairs"]
        if torch.distributed.is_initialized():
            torch.distributed.all_reduce(pairs)
            for grad in grads.values():
                torch.distributed.all_reduce(grad)
        if pairs.item() > 0 and step <= self.freeze_at:
            for name, parameter in psi.items():
                parameter.grad = grads[name] / pairs
            PSI["optimizer"].step()
            PSI["optimizer"].zero_grad(set_to_none=True)
            save(model)
        for grad in grads.values():
            grad.zero_()
        pairs.zero_()
        PSI["frozen"] = step >= self.freeze_at
        PSI["adapter"] = write(model, step)

    def psi(self, model) -> dict[str, torch.Tensor]:
        """ψ's parameters; the first time, ψ is added next to π, on π's
        modules with π's α/r, zero or as saved, switched off, with its
        optimizer.
        """
        if "psi" not in model.peft_config:
            pi = model.peft_config["default"]
            config = copy.deepcopy(pi)
            config.r = self.psi_rank
            config.lora_alpha = pi.lora_alpha * self.psi_rank // pi.r
            model.add_adapter("psi", config)
            load(model)
            psi = parameters(model)
            for parameter in psi.values():
                parameter.requires_grad_(False)
            device = next(iter(psi.values())).device
            PSI.update(
                optimizer=torch.optim.AdamW(list(psi.values()), lr=self.psi_lr),
                grads={n: torch.zeros_like(p) for n, p in psi.items()},
                pairs=torch.zeros((), device=device),
                frozen=False,
            )
        return parameters(model)


CHUNK = 128
"""Rows of logits per chunk in `logprob`."""
PSI: dict = {}
"""In a training worker's process: ψ's optimizer, its gradients kept
aside and how many pairs they sum over, the running mean of the verdict's
log-probability, whether ψ is frozen, and the judge's adapter."""


class adapters:
    """π and ψ both on, ψ learning if `learn`, then π alone on again. π's
    parameters keep their settings: FSDP sets itself up on the first pass.
    Learning puts the model in train mode, where gradient checkpointing
    applies (Qwen3.5 has no dropout), else ψ's backward holds every
    activation of the pair.
    """

    def __init__(self, model, psi: dict, learn: bool):
        """For `model`, its ψ, and whether ψ learns."""
        self.model, self.psi, self.learn = model, psi, learn

    def __enter__(self):
        """π and ψ on."""
        self.training = self.model.training
        self.model.train(self.learn)
        self.model.base_model.set_adapter(["default", "psi"])
        for parameter in self.psi.values():
            parameter.requires_grad_(self.learn)

    def __exit__(self, *exc):
        """π alone on, as before."""
        self.model.base_model.set_adapter("default")
        for parameter in self.psi.values():
            parameter.requires_grad_(False)
        self.model.train(self.training)


def logprob(logits: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
    """Σ log p(token), each row of `logits` scoring one token, in fp32, a
    chunk of rows at a time: a thousand rows' log-softmax over the
    vocabulary is gigabytes, so each chunk's is recomputed for the
    backward rather than kept.
    """
    total = logits.new_zeros((), dtype=torch.float32)
    for rows, wanted in zip(logits.split(CHUNK), tokens.split(CHUNK)):
        total = total + checkpoint(chosen, rows, wanted, use_reentrant=False)
    return total


def chosen(rows: torch.Tensor, wanted: torch.Tensor) -> torch.Tensor:
    """Σ log p(token) for one chunk of rows, in fp32."""
    logp = torch.log_softmax(rows.float(), -1)
    return logp.gather(-1, wanted[:, None]).sum()


def parameters(model) -> dict[str, torch.Tensor]:
    """ψ's parameters, by name."""
    return {n: p for n, p in model.named_parameters() if ".psi." in n}


def current(model) -> str:
    """On a worker: the judge's adapter, for the weights of the moment."""
    return PSI["adapter"]


def first() -> bool:
    """Whether this is the first worker, the one that writes files."""
    return not (
        torch.distributed.is_initialized() and torch.distributed.get_rank()
    )


def write(model, step: int) -> str:
    """π and ψ stacked into one adapter, in PEFT's format, for vLLM to
    serve the judge with: A's rows and B's columns side by side, exact
    since both have the same α/r. Written by the first worker into the
    node's temporary directory (the run is on one node); the one before
    last is removed. π is whole on every worker (FSDP does not shard it).
    """
    from peft import get_peft_model_state_dict
    from safetensors.torch import save_file

    folder = Path(tempfile.gettempdir()) / Path(os.environ[RUN_DIR]).name
    out = folder / f"judge-{step}"
    if not first():
        return str(out)
    pi = get_peft_model_state_dict(model, adapter_name="default")
    psi = get_peft_model_state_dict(model, adapter_name="psi")
    stacked = {
        key: torch.cat(
            [whole(a), whole(psi[key])], dim=0 if "lora_A" in key else 1
        )
        .to(torch.bfloat16)
        .contiguous()
        for key, a in pi.items()
    }
    config = copy.deepcopy(model.peft_config["default"])
    rank = config.r + model.peft_config["psi"].r
    config.lora_alpha = config.lora_alpha * rank // config.r
    config.r = rank
    config.save_pretrained(out)
    save_file(stacked, out / "adapter_model.safetensors")
    shutil.rmtree(folder / f"judge-{step - 2}", ignore_errors=True)
    return str(out)


def whole(tensor: torch.Tensor) -> torch.Tensor:
    """A parameter as one local tensor, if FSDP holds it as a DTensor."""
    tensor = tensor.detach()
    return tensor.full_tensor() if hasattr(tensor, "full_tensor") else tensor


def saved() -> Path:
    """Where ψ is kept between steps and restarts."""
    return Path(os.environ.get(RUN_DIR, ".")) / "psi.safetensors"


def save(model) -> None:
    """Write ψ, from the first worker only."""
    if not first():
        return
    from peft import get_peft_model_state_dict
    from safetensors.torch import save_file

    state = get_peft_model_state_dict(model, adapter_name="psi")
    save_file({k: v.detach().contiguous() for k, v in state.items()}, saved())


def load(model) -> None:
    """Read ψ back, if a previous run saved it."""
    if saved().exists():
        from peft import set_peft_model_state_dict
        from safetensors.torch import load_file

        set_peft_model_state_dict(model, load_file(saved()), adapter_name="psi")
