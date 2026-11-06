import asyncio
import math
import uuid

from obfuscation_immunity.common.base_method import BaseMethod
from obfuscation_immunity.common.pairs import (
    Pair,
    asked,
    reasoned,
    unshuffled,
)


class ActorJudge(BaseMethod):
    """The actor as its own judge: the baseline for CoevolvingJudge.

    The judge is the policy in training, with its current weights, as the
    run's vLLM serves it; nothing trains it, it follows the actor because
    it is the actor. Asked the pair's question, it thinks, then its
    next-token probabilities over the four answers, renormalised, are J.

    :param judging: tokens the judge may think for before its verdict
    """

    verl_overrides = ["actor_rollout_ref.rollout.logprobs_mode=raw_logprobs"]
    """Raw logprobs: a token forced by `allowed_token_ids` keeps its own
    probability."""

    def __init__(self, *, judging: int = 1024, **kwargs):
        """The game's arguments, and the judge's thinking budget."""
        super().__init__(**kwargs)
        self.judging = judging

    async def judge(self, loop, pair: Pair):
        """Ask the actor; J, the mass it put on the four answers and how
        long it thought.
        """
        ids, letters, honest_first = asked(loop.tokenizer, pair)
        ids, thought = await reasoned(loop, ids, self.judging)
        logp = await asyncio.gather(*(self.ask(loop, ids, k) for k in letters))
        total = logsumexp(logp)
        shown = [v - total for v in logp]
        measured = {
            "judge_answer_mass": math.exp(total),
            "judge_thinking": thought,
        }
        return unshuffled(shown, honest_first), measured

    async def ask(self, loop, ids: list[int], letter: int) -> float:
        """log p(letter | the question and the judge's thinking): the
        actor's one token, forced.
        """
        reply = await loop.server_manager.generate(
            request_id=uuid.uuid4().hex,
            prompt_ids=ids,
            sampling_params={
                "max_tokens": 1,
                "allowed_token_ids": [letter],
                "logprobs": True,
            },
        )
        return reply.log_probs[0]


def logsumexp(values: list[float]) -> float:
    """log Σ exp(v), without underflow."""
    top = max(values)
    return top + math.log(sum(math.exp(v - top) for v in values))
