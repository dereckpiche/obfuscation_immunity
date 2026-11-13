"""
verl's agent loop for the task: one trajectory per pair.

The policy answers the puzzle (s), is asked for a sneaky answer, and
answers again (s̄); the run's judge reads the pair, and the trajectory
is paid the game's reward. Each turn, the policy's and the judge's, must
think, then reply in its exact format, within the budget its prompt states
(common.chat_format), in one try (common.turn). A turn in the format earns
`FORMAT_STAKE`, any other loses it, and ends the game there: each side
keeps only its stakes, and whoever broke the format first loses. If the
policy does, the judge wins (the judge term pays the policy 0); if the
judge does, the policy wins (it pays 1). The policy's own tokens train;
the request does not.
"""

import json
import math
import time
from typing import Any

from verl.experimental.agent_loop.agent_loop import (
    AgentLoopBase,
    AgentLoopMetrics,
    AgentLoopOutput,
)

from obfuscation_immunity.common import puzzles
from obfuscation_immunity.common.chat_format import ChatFormat
from obfuscation_immunity.common.judge import Judge, Verdict, build_judge
from obfuscation_immunity.common.pairs import (
    OUTCOMES,
    SNEAKY_REQUEST,
    Answer,
    Pair,
)
from obfuscation_immunity.common.transcript import (
    ENVIRONMENT,
    HONEST_ANSWER_KIND,
    POLICY,
    PUZZLE_KIND,
    SNEAKY_ANSWER_KIND,
    SNEAKY_REQUEST_KIND,
    PairRecord,
    TranscriptSegment,
)
from obfuscation_immunity.common.turn import Turn, generate_turn

SNEAKY_REQUEST_ROOM = 256
"""Response tokens kept for the request between the two answers."""


def answer_in_format(reply: str) -> bool:
    """
    Tells whether an answer's reply is its answer alone within \\boxed{}.

    Args:
        reply: What the policy wrote after its thinking.

    Returns:
        Whether the reply is exactly one boxed answer.
    """
    return puzzles.answer_alone(reply) is not None


class PairLoop(AgentLoopBase):
    """
    Generates, judges and pays one pair; verl builds one per trajectory.

    Attributes:
        judge: The run's judge.
        chat: The model's turn markers.
        response_length: verl's cap on a trajectory's response tokens.
        turn_tokens: How many tokens an answer may take; the judge's turn
            takes as many.
        messages: The chat so far, as the chat template renders it.
        token_ids: The trajectory so far, prompt included.
        response_mask: 1 for each response token the policy wrote, 0 for
            the request's.
        response_logprobs: vLLM's log-probability of each response token
            (verl asks vLLM for them: `calculate_log_probs`).
        weight_versions: Which actor weights wrote the trajectory, as verl
            wants to know: the first answer's, and the latest.
        segments: The pair's stretches of tokens so far, for its page.
    """

    judge: Judge
    chat: ChatFormat
    response_length: int
    turn_tokens: int
    messages: list[dict[str, str]]
    token_ids: list[int]
    response_mask: list[int]
    response_logprobs: list[float]
    weight_versions: dict[str, Any]
    segments: list[TranscriptSegment]

    def __init__(self, *args: Any, **kwargs: Any):
        """
        Builds verl's loop, with the run's judge, markers and budget.

        Each answer may take half of what the request leaves of the
        response length.

        Args:
            *args: verl's arguments for an agent loop.
            **kwargs: verl's keyword arguments for an agent loop.
        """
        super().__init__(*args, **kwargs)
        self.judge = build_judge(self.config.judge)
        self.chat = ChatFormat.build(self.tokenizer, self.config.chat)
        self.response_length = self.rollout_config.response_length
        self.turn_tokens = (self.response_length - SNEAKY_REQUEST_ROOM) // 2

    async def run(
        self, sampling_params: dict[str, Any], **row: Any
    ) -> AgentLoopOutput:
        """
        Generates one pair, has the run's judge read it, and pays it.

        Played to the end, the reward is the stakes of both answers, plus
        V((1,0) | s, s̄) + 1 − J(y_true): both terms in [0, 1]. An answer
        out of the format ends the game with the stakes so far; a judge out
        of the format ends it with the policy paid V + 1.

        Args:
            sampling_params: verl's sampling parameters for the policy.
            **row: The dataset's row: the puzzle's chat prompt
                (`raw_prompt`) and its task (`extra_info`).

        Returns:
            The trajectory for verl, with the reward and the pair's record
            (`pair_record`), which the report reads.
        """
        prompt_messages = puzzles.puzzle_messages(
            list(row["raw_prompt"]), self.turn_tokens
        )
        self.messages = list(prompt_messages)
        self.token_ids = await self.ct_build_initial_tokens(prompt_messages)
        self.response_mask, self.response_logprobs = [], []
        self.weight_versions = {}
        prompt_token_ids = list(self.token_ids)
        self.segments = [
            TranscriptSegment(
                "Puzzle", ENVIRONMENT, prompt_token_ids, kind=PUZZLE_KIND
            )
        ]
        task = json.loads(row["extra_info"]["task"])
        started = time.monotonic()
        honest_turn = await self.generate_answer(
            sampling_params, "Honest answer", HONEST_ANSWER_KIND
        )
        part_seconds = {"honest": time.monotonic() - started}
        honest = self.graded_answer(task, honest_turn)
        metrics = {"honest_in_format": float(honest_turn.in_format)}
        if not honest_turn.in_format:
            return self.ended_by_policy(
                task, [honest_turn], metrics, part_seconds
            )
        metrics["honest_correct"] = float(honest.correct)
        await self.append_sneaky_request()
        sneaky_started = time.monotonic()
        sneaky_turn = await self.generate_answer(
            sampling_params, "Sneaky answer", SNEAKY_ANSWER_KIND
        )
        part_seconds["sneaky"] = time.monotonic() - sneaky_started
        sneaky = self.graded_answer(task, sneaky_turn)
        metrics["sneaky_in_format"] = float(sneaky_turn.in_format)
        turns = [honest_turn, sneaky_turn]
        if not sneaky_turn.in_format:
            return self.ended_by_policy(task, turns, metrics, part_seconds)
        metrics["sneaky_correct"] = float(sneaky.correct)
        pair = Pair(
            task="\n\n".join(
                m["content"] for m in prompt_messages if m["role"] == "user"
            ),
            prompt_token_ids=prompt_token_ids,
            honest=honest,
            sneaky=sneaky,
        )
        judge_started = time.monotonic()
        verdict = await self.judge.verdict(self, pair)
        part_seconds["judge"] = time.monotonic() - judge_started
        return self.judged(task, pair, verdict, turns, metrics, part_seconds)

    def judged(
        self,
        task: dict[str, Any],
        pair: Pair,
        verdict: Verdict,
        turns: list[Turn],
        metrics: dict[str, float],
        part_seconds: dict[str, float],
    ) -> AgentLoopOutput:
        """
        Pays a pair both of whose answers were in the format.

        Args:
            task: The puzzle, as its dataset row holds it.
            pair: The pair.
            verdict: What the judge said of it.
            turns: The two answers' turns.
            metrics: The pair's numbers so far.
            part_seconds: How long each part of the pair took.

        Returns:
            The trajectory for verl, paid the stakes, V and the judge's
            miss: 1 − J(y_true), or 1 if the judge broke its format.
        """
        metrics = {**metrics, "judge_in_format": float(verdict.stated)}
        judge_miss = 1.0
        if verdict.stated:
            true_probability = math.exp(
                verdict.outcome_logprobs[pair.true_outcome]
            )
            judge_miss = 1.0 - true_probability
            judged_right = verdict.likeliest_outcome == pair.true_outcome
            metrics["judge_accuracy"] = float(judged_right)
            metrics["judge_true_probability"] = true_probability
        format_reward = sum(turn.stake for turn in turns)
        reward = pair.only_honest_correct + judge_miss + format_reward
        record = PairRecord(
            segments=self.segments + verdict.segments,
            puzzle=task["name"],
            expected_answer=str(task["entry"]["answer"]),
            honest_correct=pair.honest.correct,
            sneaky_correct=pair.sneaky.correct,
            true_outcome=pair.true_outcome,
            outcome_logprobs=verdict.outcome_logprobs,
            judge_statement=verdict.statement,
            verifier_reward=pair.only_honest_correct,
            judge_miss=judge_miss,
            reward=reward,
            metrics={**metrics, **verdict.metrics},
            part_seconds=part_seconds,
            format_reward=format_reward,
        )
        return self.trajectory(record, verdict.judge_trajectory)

    def ended_by_policy(
        self,
        task: dict[str, Any],
        turns: list[Turn],
        metrics: dict[str, float],
        part_seconds: dict[str, float],
    ) -> AgentLoopOutput:
        """
        Ends a pair whose last answer broke the format: the judge wins.

        Nothing follows a turn out of the format, and there is no pair to
        judge: the policy keeps only its stakes.

        Args:
            task: The puzzle, as its dataset row holds it.
            turns: The answers' turns, the last out of the format.
            metrics: The pair's numbers so far.
            part_seconds: How long each part of the pair took.

        Returns:
            The trajectory for verl, paid its stakes.
        """
        format_reward = sum(turn.stake for turn in turns)
        record = PairRecord(
            segments=self.segments,
            puzzle=task["name"],
            expected_answer=str(task["entry"]["answer"]),
            honest_correct=False,
            sneaky_correct=False,
            true_outcome=OUTCOMES.index((0, 0)),
            outcome_logprobs=[],
            judge_statement="",
            verifier_reward=0.0,
            judge_miss=0.0,
            reward=format_reward,
            metrics=metrics,
            part_seconds=part_seconds,
            format_reward=format_reward,
        )
        return self.trajectory(record, None)

    def trajectory(
        self, record: PairRecord, judge_trajectory: dict[str, Any] | None
    ) -> AgentLoopOutput:
        """
        Packs the trajectory so far for verl, paid the record's reward.

        Args:
            record: The pair's record.
            judge_trajectory: The judge's turn to learn from, if any.

        Returns:
            The trajectory for verl.
        """
        length = self.response_length
        prompt_length = len(self.token_ids) - len(self.response_mask)
        return AgentLoopOutput(
            prompt_ids=self.token_ids[:prompt_length],
            response_ids=self.token_ids[prompt_length:][:length],
            response_mask=self.response_mask[:length],
            response_logprobs=self.response_logprobs[:length],
            reward_score=record.reward,
            num_turns=len(self.messages),
            metrics=AgentLoopMetrics(),
            extra_fields={
                **self.weight_versions,
                "pair_record": record.to_dict(),
                "judge_trajectory": judge_trajectory,
            },
        )

    async def generate_answer(
        self, sampling_params: dict[str, Any], title: str, kind: str
    ) -> Turn:
        """
        Generates the policy's next answer and appends it to the trajectory.

        Args:
            sampling_params: verl's sampling parameters for the policy.
            title: The answer's title on the pair's page.
            kind: The answer's segment kind, for the page's helpers.

        Returns:
            The answer's turn.
        """
        turn = await generate_turn(
            self,
            self.chat,
            self.token_ids,
            sampling_params,
            self.turn_tokens,
            answer_in_format,
        )
        versions = turn.weight_versions
        if not self.weight_versions:
            self.weight_versions = dict(versions)
        elif versions.get("max_global_steps") is not None:
            latest = versions["max_global_steps"]
            self.weight_versions["max_global_steps"] = latest
        text = self.tokenizer.decode(turn.token_ids)
        self.messages = self.messages + [{"role": "assistant", "content": text}]
        self.token_ids = self.token_ids + turn.token_ids
        self.response_mask = self.response_mask + [1] * len(turn.token_ids)
        self.response_logprobs = self.response_logprobs + turn.logprobs
        self.segments.append(
            TranscriptSegment(
                title=title,
                source=POLICY,
                token_ids=turn.token_ids,
                logprobs=turn.logprobs,
                trained=True,
                budget=self.turn_tokens,
                kind=kind,
                details={"format": turn.problem} if turn.problem else {},
            )
        )
        return turn

    async def append_sneaky_request(self) -> None:
        """Appends the request for a sneaky answer, as a user turn."""
        requested = self.messages + [
            {"role": "user", "content": SNEAKY_REQUEST}
        ]
        length_before = len(self.token_ids)
        merge_result, self.response_mask, self.response_logprobs = (
            await self.ct_merge_context_msg(
                self.messages,
                requested,
                self.token_ids,
                self.response_mask,
                self.response_logprobs,
            )
        )
        self.messages = requested
        self.token_ids = merge_result.token_ids
        request_token_ids = self.token_ids[length_before:]
        self.segments.append(
            TranscriptSegment(
                "Request",
                ENVIRONMENT,
                request_token_ids,
                kind=SNEAKY_REQUEST_KIND,
            )
        )

    def graded_answer(self, task: dict[str, Any], turn: Turn) -> Answer:
        """
        Gathers one answer for reading and grades it with V.

        Args:
            task: The puzzle, as its dataset row holds it.
            turn: The answer's turn.

        Returns:
            The answer: its reasoning, its reply, and V's label, which is
            wrong unless the turn was in the format.
        """
        reasoning, final = self.chat.split(self.tokenizer, turn.token_ids)
        score = puzzles.grade_final_answer(task, final)
        correct = turn.in_format and puzzles.is_correct(score)
        return Answer(turn.token_ids, reasoning, final, correct)
