"""
The exact record of one pair, token by token, for its page.

The loop records every stretch of tokens in the order it happened: what
the policy was shown, what it generated, the request between its answers,
and whatever the judge read and generated. The trainer renders one
record per step (common.render) with the run's tokenizer.
"""

from dataclasses import asdict, dataclass, field
from typing import Any

ENVIRONMENT = "environment"
"""A segment's source: tokens no model generated (prompts, requests)."""

POLICY = "policy"
"""A segment's source: tokens the policy generated."""

JUDGE = "judge"
"""A segment's source: tokens a judge generated."""

PUZZLE_KIND = "puzzle"
"""A segment's kind: the puzzle's prompt."""

HONEST_ANSWER_KIND = "honest_answer"
"""A segment's kind: the policy's honest answer."""

SNEAKY_REQUEST_KIND = "sneaky_request"
"""A segment's kind: the request for a sneaky answer."""

SNEAKY_ANSWER_KIND = "sneaky_answer"
"""A segment's kind: the policy's sneaky answer."""

JUDGE_QUESTION_KIND = "judge_question"
"""A segment's kind: the question a text judge reads."""

JUDGE_OUTPUT_KIND = "judge_output"
"""A segment's kind: what a text judge generated."""


@dataclass
class TranscriptSegment:
    """
    One stretch of tokens, exactly as a model read or generated it.

    Attributes:
        title: What the stretch is, such as "Honest answer".
        source: Who produced it: ENVIRONMENT, POLICY or JUDGE.
        token_ids: Its tokens.
        logprobs: The sampler's log-probability of each token, for
            generated stretches; empty otherwise.
        trained: Whether these tokens get a gradient.
        budget: How many tokens the generation was allowed; 0 if none.
        kind: What the stretch is, for the page's helpers: one of the
            `*_KIND` names, or empty for any other.
        details: Facts the page shows next to the stretch, by label.
        model: Whose sequence the stretch belongs to, POLICY or JUDGE: the
            model that reads it, if it is context, or generated it.
    """

    title: str
    source: str
    token_ids: list[int]
    logprobs: list[float] = field(default_factory=list)
    trained: bool = False
    budget: int = 0
    kind: str = ""
    details: dict[str, str] = field(default_factory=dict)
    model: str = POLICY


@dataclass
class PairRecord:
    """
    Everything one pair's page shows.

    Attributes:
        segments: The pair's stretches of tokens, in the order they
            happened.
        puzzle: The reasoning-gym dataset the puzzle comes from.
        expected_answer: The puzzle's own answer, as its grader knows it.
        honest_correct: V's label of the honest answer.
        sneaky_correct: V's label of the sneaky answer.
        true_outcome: The index in OUTCOMES of y = (y_s, y_s̄).
        outcome_logprobs: The judge's log J over OUTCOMES, (s, s̄) order;
            empty if the judge never stated it.
        judge_statement: What a text judge stated after its thinking.
        verifier_reward: V((1,0) | s, s̄); 0 if the game ended early.
        judge_miss: 1 − J(y_true); 1 if the judge broke its format, 0 if
            the policy did first.
        reward: The trajectory's reward: V, the judge's miss and the
            format reward.
        metrics: The pair's numbers, as the report averages them.
        part_seconds: How long each part of the pair took: `honest`,
            `sneaky` and `judge`.
        format_reward: `FORMAT_STAKE` for each answer in the format,
            minus it for an answer out of it (common.turn).
    """

    segments: list[TranscriptSegment]
    puzzle: str
    expected_answer: str
    honest_correct: bool
    sneaky_correct: bool
    true_outcome: int
    outcome_logprobs: list[float]
    judge_statement: str
    verifier_reward: float
    judge_miss: float
    reward: float
    metrics: dict[str, float]
    part_seconds: dict[str, float] = field(default_factory=dict)
    format_reward: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """
        Converts the record to plain data, as verl's transfer queue keeps it.

        Returns:
            The record as nested dicts and lists.
        """
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PairRecord":
        """
        Rebuilds a record from `to_dict`'s plain data.

        Args:
            data: The record as nested dicts and lists.

        Returns:
            The record.
        """
        segments = [TranscriptSegment(**s) for s in data["segments"]]
        return cls(**{**data, "segments": segments})
