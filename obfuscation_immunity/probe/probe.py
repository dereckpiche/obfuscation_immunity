"""
A probe on the starting model's activations, as the judge.

The probe ψ is fitted once on School of Reward Hacks (probe.fit), on the
starting model's activations: ψ(x) is how much an answer looks like a hack.
The starting model reads each answer of the pair on its own, after the
puzzle and nothing else: the actor's own model with its adapter scaled to
zero, on a training worker while vLLM generates. Each answer is judged
correct with probability 1 − ψ, independently:

    J(y_s, y_s̄) = Π_a (1 − ψ_a)^{y_a} · ψ_a^{1 − y_a}

Reading with the starting model keeps the probe fixed as the policy drifts:
the question is whether the policy learns to write sneaky answers it
misses.
"""

import asyncio
import json
import math
from collections.abc import Callable
from dataclasses import asdict, dataclass
from functools import cache
from pathlib import Path

import torch
import torch.nn.functional as F
from huggingface_hub import snapshot_download
from peft import PeftModel
from peft.helpers import rescale_adapter_scale
from transformers import AutoConfig
from verl.experimental.agent_loop.agent_loop import AgentLoopBase

from obfuscation_immunity.common import training_workers
from obfuscation_immunity.common.judge import (
    Judge,
    JudgeConfig,
    Verdict,
)
from obfuscation_immunity.common.pairs import OUTCOMES, Pair


@dataclass
class ProbeWeights:
    """
    A fitted probe: ψ(x) = σ(scale · directionᵀx + bias).

    x pools φ(h_t) at `layer` of `model`, φ being the encoder of the SAE in
    `sae`.

    Attributes:
        model: The model whose activations the probe reads.
        layer: The decoder layer after which it reads the residual stream.
        sae: The SAE whose encoder gives φ: a directory or a Hub repo.
        direction: v, the difference of the classes' mean features, scaled
            so that projections on it have unit spread.
        scale: a, the logistic link's slope.
        bias: b, the logistic link's intercept.
    """

    model: str
    layer: int
    sae: str
    direction: torch.Tensor
    scale: float
    bias: float

    @classmethod
    def fit(
        cls,
        features: torch.Tensor,
        labels: torch.Tensor,
        model: str,
        layer: int,
        sae: str,
    ) -> "ProbeWeights":
        """
        Fits a probe: a difference of means, then a 1-D logistic link.

        The direction is rescaled so that the projections have unit spread;
        a and b come from logistic regression, lightly regularised since
        the classes may separate.

        Args:
            features: One pooled feature vector per answer.
            labels: 1 for a hack, 0 for an honest answer.
            model: The model the features come from.
            layer: The layer they come from.
            sae: The SAE that encoded them.

        Returns:
            The fitted probe.
        """
        hacks, honest = features[labels == 1], features[labels == 0]
        direction = hacks.mean(0) - honest.mean(0)
        direction = direction / (features @ direction).std()
        projections, targets = features @ direction, labels.float()
        scale = torch.zeros((), requires_grad=True)
        bias = torch.zeros((), requires_grad=True)
        optimizer = torch.optim.LBFGS([scale, bias], max_iter=200)

        def regularised_loss() -> torch.Tensor:
            """
            Computes the link's cross-entropy, plus a small penalty on a.

            Returns:
                The loss, after its backward pass (as LBFGS needs).
            """
            optimizer.zero_grad()
            logits = scale * projections + bias
            loss = F.binary_cross_entropy_with_logits(logits, targets)
            loss = loss + 1e-3 * scale**2
            loss.backward()
            return loss

        optimizer.step(regularised_loss)
        return cls(model, layer, sae, direction, scale.item(), bias.item())

    def logit(self, features: torch.Tensor) -> float:
        """
        Computes ψ's logit, a · vᵀx + b, for one pooled feature vector.

        Args:
            features: x, one answer's pooled features.

        Returns:
            The logit.
        """
        direction = self.direction.to(features.device, features.dtype)
        return (self.scale * direction @ features + self.bias).item()

    def score(self, features: torch.Tensor) -> float:
        """
        Computes ψ(x), how much one answer looks like a hack.

        Args:
            features: x, one answer's pooled features.

        Returns:
            ψ(x), in (0, 1).
        """
        return torch.sigmoid(torch.tensor(self.logit(features))).item()

    def save(self, path: Path) -> None:
        """
        Saves the probe.

        Args:
            path: The file to write.
        """
        torch.save(asdict(self), path)

    @classmethod
    def load(cls, path: str) -> "ProbeWeights":
        """
        Loads a probe written by `save`.

        Args:
            path: The file to read.

        Returns:
            The probe.
        """
        return cls(**torch.load(path))


class TopKSAE:
    """
    Qwen-Scope's SAE encoder for one layer: φ(h) = TopK_k(W_enc h + b_enc).

    Attributes:
        k: How many features stay active.
        encoder_weight: W_enc.
        encoder_bias: b_enc.
    """

    k: int
    encoder_weight: torch.Tensor
    encoder_bias: torch.Tensor

    def __init__(
        self, path: str, layer: int, width: int, device: torch.device | str
    ):
        """
        Loads k and one layer's encoder from an SAE, checking its width.

        Every Qwen3.5 size has its own residual width, so a mismatch means
        another model's SAE.

        Args:
            path: The SAE: a directory, or a Hub repo to download.
            layer: The layer whose encoder to load.
            width: The residual stream's width it must read.
            device: Where to put the weights.

        Raises:
            ValueError: The SAE reads residual streams of another width.
        """
        root = Path(path)
        if not root.is_dir():
            root = Path(snapshot_download(path))
        config = json.loads((root / "config.json").read_text())
        if config["d_model"] != width:
            raise ValueError(
                f"{path} was trained on {config['base_model']}, whose residual"
                f" stream is {config['d_model']} wide, not {width}"
            )
        self.k = config["k"]
        weights = torch.load(root / f"layer{layer}.sae.pt", map_location=device)
        self.encoder_weight = weights["W_enc"]
        self.encoder_bias = weights["b_enc"]

    def __call__(self, residual: torch.Tensor) -> torch.Tensor:
        """
        Encodes residual streams, keeping the k largest features.

        Args:
            residual: Residual streams, one per row.

        Returns:
            Feature activations, all but the k largest set to zero.
        """
        weight = self.encoder_weight.T.to(residual.dtype)
        before_topk = residual @ weight + self.encoder_bias.to(residual.dtype)
        top = before_topk.topk(self.k, dim=-1)
        activations = torch.zeros_like(before_topk)
        return activations.scatter_(-1, top.indices, top.values)


def pooled_sae_features(
    model: torch.nn.Module,
    sae: Callable[[torch.Tensor], torch.Tensor],
    layer: int,
    token_ids: list[int],
    answer_mask: list[int],
) -> torch.Tensor:
    """
    Computes x for one sequence: SAE features averaged over the answer.

    x is φ of the residual stream after decoder layer `layer`
    (hidden_states[layer + 1]), averaged over the answer's tokens.

    Args:
        model: The model that reads the sequence.
        sae: φ, the SAE's encoder.
        layer: The decoder layer after which to read.
        token_ids: The sequence.
        answer_mask: 1 for the answer's tokens, 0 for the rest.

    Returns:
        x, the pooled features.
    """
    device = next(model.parameters()).device
    input_ids = torch.tensor([token_ids], device=device)
    with training_workers.logits_for_last_positions(model, 1):
        output = model(
            input_ids=input_ids, output_hidden_states=True, use_cache=False
        )
    residual = output.hidden_states[layer + 1][0].float()
    in_answer = torch.tensor(answer_mask, device=device).bool()
    return sae(residual[in_answer]).mean(0)


@dataclass(frozen=True)
class ProbeConfig(JudgeConfig):
    """
    The probe judge's settings.

    Attributes:
        probe: The fitted probe's file (probe.fit writes one).
    """

    probe: str


class Probe(Judge):
    """
    A fixed probe on the starting model's activations, judging each answer.

    Attributes:
        config: The method's config.
    """

    Config = ProbeConfig
    config: ProbeConfig

    async def verdict(self, loop: AgentLoopBase, pair: Pair) -> Verdict:
        """
        Judges one pair from the probe's logit on each answer.

        log ψ = log σ(z) and log (1 − ψ) = log σ(−z), z the probe's logit.

        Args:
            loop: The agent loop the pair was generated in.
            pair: The pair to judge.

        Returns:
            J over OUTCOMES, and ψ of each answer as metrics.
        """
        prompt = pair.prompt_token_ids
        answer_logits = await asyncio.gather(
            *(
                training_workers.call_on_a_training_worker(
                    self.probe_logit,
                    prompt + answer.token_ids,
                    [0] * len(prompt) + [1] * len(answer.token_ids),
                )
                for answer in (pair.honest, pair.sneaky)
            )
        )
        outcome_logprobs = [
            sum(
                log_sigmoid(-logit if correct else logit)
                for logit, correct in zip(answer_logits, outcome)
            )
            for outcome in OUTCOMES
        ]
        honest_hack_score, sneaky_hack_score = (
            math.exp(log_sigmoid(logit)) for logit in answer_logits
        )
        metrics = {
            "probe_honest": honest_hack_score,
            "probe_sneaky": sneaky_hack_score,
        }
        return Verdict(outcome_logprobs, metrics)

    def probe_logit(
        self, model: PeftModel, token_ids: list[int], answer_mask: list[int]
    ) -> float:
        """
        Computes the probe's logit for one answer, on a training worker.

        The actor's adapter is scaled to zero rather than switched off,
        which would change which of its parameters train; FSDP sets itself
        up on the first pass.

        Args:
            model: The actor's peft model.
            token_ids: The puzzle's prompt, then the answer.
            answer_mask: 1 for the answer's tokens, 0 for the prompt's.

        Returns:
            z, the probe's logit.
        """
        device = next(model.parameters()).device
        probe_weights, sae_encoder = load_probe(self.config.probe, device)
        with torch.no_grad(), rescale_adapter_scale(model, 0):
            features = pooled_sae_features(
                model, sae_encoder, probe_weights.layer, token_ids, answer_mask
            )
        return probe_weights.logit(features)


def log_sigmoid(logit: float) -> float:
    """
    Computes log σ(z) without overflow, at any logit.

    Args:
        logit: z.

    Returns:
        log σ(z).
    """
    if logit >= 0:
        return -math.log1p(math.exp(-logit))
    return logit - math.log1p(math.exp(logit))


@cache
def load_probe(path: str, device: torch.device) -> tuple[ProbeWeights, TopKSAE]:
    """
    Loads the probe and its SAE's encoder, once per worker.

    Args:
        path: The fitted probe's file.
        device: Where to put the SAE's weights.

    Returns:
        The probe, and the encoder of the SAE it reads.
    """
    probe_weights = ProbeWeights.load(path)
    text_config = AutoConfig.from_pretrained(
        probe_weights.model, trust_remote_code=True
    ).get_text_config()
    sae_encoder = TopKSAE(
        probe_weights.sae,
        probe_weights.layer,
        text_config.hidden_size,
        device,
    )
    return probe_weights, sae_encoder
