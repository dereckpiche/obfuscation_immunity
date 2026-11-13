"""
The actor judge: the actor, judging its own pairs.

The judge is the policy in training, with its current weights, as the run's
vLLM serves it; nothing trains it, it follows the actor because it is the
actor. Asked the pair's question, it thinks, within its own budget as an
answerer, then states the probability of each outcome: that is J.
"""

from verl.experimental.agent_loop.agent_loop import AgentLoopBase

from obfuscation_immunity.common.judge import (
    Judge,
    JudgeConfig,
    Verdict,
)
from obfuscation_immunity.common.pairs import Pair
from obfuscation_immunity.judges.text_judge import ask_text_judge


class ActorJudge(Judge):
    """
    The actor, judging its own pairs through the run's vLLM.

    Attributes:
        config: The method's config.
    """

    Config = JudgeConfig
    config: JudgeConfig

    async def verdict(self, loop: AgentLoopBase, pair: Pair) -> Verdict:
        """
        Asks the actor to judge one pair.

        Args:
            loop: The agent loop the pair was generated in.
            pair: The pair to judge.

        Returns:
            J as the actor stated it, in (s, s̄) order.
        """
        turn = await ask_text_judge(loop, pair)
        return turn.verdict
