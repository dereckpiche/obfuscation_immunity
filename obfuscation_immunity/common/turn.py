"""
One turn of a model: a single try, in the strict format, never forced.

The model thinks, then replies in its exact format, within a budget its
prompt states (common.chat_format). There is no second try: a turn out of
the format, or out of budget, ends the game (common.loop). No token is
ever written in the model's place: every token it trains on is its own,
and the chat template's turn markers and the user turns are context.
"""

import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from verl.experimental.agent_loop.agent_loop import AgentLoopBase

from obfuscation_immunity.common.chat_format import ChatFormat

FORMAT_STAKE = 0.5
"""Earned by a turn in the format; lost by any other."""


@dataclass
class Turn:
    """
    One turn of a model.

    Attributes:
        token_ids: What the model generated.
        logprobs: The sampler's log-probability of each token.
        problem: What was wrong with the turn (common.chat_format); empty
            if it was in the format.
        weight_versions: verl's record of the weights that wrote it.
    """

    token_ids: list[int]
    logprobs: list[float]
    problem: str = ""
    weight_versions: dict[str, Any] = field(default_factory=dict)

    @property
    def in_format(self) -> bool:
        """Whether the turn ended itself, in the format."""
        return not self.problem

    @property
    def stake(self) -> float:
        """The turn's local reward: `FORMAT_STAKE`, lost if out of format."""
        return FORMAT_STAKE if self.in_format else -FORMAT_STAKE


async def generate_turn(
    loop: AgentLoopBase,
    chat: ChatFormat,
    prompt_ids: list[int],
    sampling_params: dict[str, Any],
    max_tokens: int,
    reply_in_format: Callable[[str], bool],
) -> Turn:
    """
    Generates one turn through the run's vLLM and checks its format.

    Args:
        loop: The agent loop; it holds the tokenizer and the vLLM client.
        chat: The model's turn markers.
        prompt_ids: The tokens the model continues.
        sampling_params: vLLM's sampling parameters.
        max_tokens: How many tokens the turn may take.
        reply_in_format: Tells whether a reply is in its exact format.

    Returns:
        The turn.
    """
    output = await loop.server_manager.generate(
        request_id=uuid.uuid4().hex,
        prompt_ids=prompt_ids,
        sampling_params={**sampling_params, "max_tokens": max_tokens},
    )
    token_ids = list(output.token_ids)
    return Turn(
        token_ids=token_ids,
        logprobs=list(output.log_probs or []),
        problem=chat.problem(loop.tokenizer, token_ids, reply_in_format),
        weight_versions=dict(output.extra_fields),
    )
