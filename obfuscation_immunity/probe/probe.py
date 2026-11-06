import asyncio
import math
from functools import cache

import torch

from obfuscation_immunity.common import workers
from obfuscation_immunity.common.base_method import BaseMethod
from obfuscation_immunity.common.pairs import ANSWERS, Pair


class Probe(BaseMethod):
    """A probe on the starting model's activations as the judge.

    The probe ψ is fitted once on School of Reward Hacks (probe.fit), on
    the starting model's activations: ψ(x) is how much an answer looks like
    a hack. The starting model reads each answer of the pair on its own,
    after the puzzle and nothing else: the actor's own model with its
    adapter scaled to zero, on a training worker while vLLM generates.
    Each answer is
    judged correct with probability 1 − ψ, independently:

        J(y_s, y_s̄) = Π_a (1 − ψ_a)^{y_a} · ψ_a^{1 − y_a}

    Reading with the starting model keeps the probe fixed as the policy
    drifts: the question is whether the policy learns to write sneaky
    answers it misses.

    :param probe: the fitted probe (probe.fit writes one)
    """

    def __init__(self, *, probe: str = "probe.pt", **kwargs):
        """The game's arguments, and the probe's file."""
        super().__init__(**kwargs)
        self.probe = probe

    async def judge(self, loop, pair: Pair):
        """ψ of each answer, read by the starting model; J from them, as
        log ψ = log σ(z) and log (1 − ψ) = log σ(−z) of ψ's logit z.
        """
        logits = await asyncio.gather(
            *(
                workers.on_a_worker(
                    self.read,
                    pair.prompt + answer,
                    [0] * len(pair.prompt) + [1] * len(answer),
                )
                for answer in (pair.honest_ids, pair.sneaky_ids)
            )
        )
        logp = [
            sum(logsigmoid(-z if y else z) for z, y in zip(logits, outcome))
            for outcome in ANSWERS
        ]
        honest, sneaky = (math.exp(logsigmoid(z)) for z in logits)
        return logp, {"probe_honest": honest, "probe_sneaky": sneaky}

    def read(self, model, ids, mask) -> float:
        """On a worker: ψ's logit for one answer. The adapter is scaled to
        zero rather than switched off, which would change which of its
        parameters train, and FSDP sets itself up on the first pass.
        """
        from peft.helpers import rescale_adapter_scale

        from obfuscation_immunity.probe.fit import features

        weights, sae = fitted(self.probe, next(model.parameters()).device)
        with torch.no_grad(), rescale_adapter_scale(model, 0):
            x = features(model, sae, weights.layer, ids, mask)
        return weights.logit(x)


def logsigmoid(z: float) -> float:
    """log σ(z), without overflow."""
    return -math.log1p(math.exp(-z)) if z >= 0 else z - math.log1p(math.exp(z))


@cache
def fitted(path: str, device) -> tuple:
    """The probe and its SAE's encoder, read once per worker. probe.fit is
    imported here, so that running it imports it once.
    """
    from transformers import AutoConfig

    from obfuscation_immunity.probe.fit import TopKSAE, Weights

    weights = Weights.load(path)
    config = AutoConfig.from_pretrained(weights.model).get_text_config()
    return weights, TopKSAE(
        weights.sae, weights.layer, config.hidden_size, device
    )
