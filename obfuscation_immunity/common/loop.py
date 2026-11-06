"""verl's agent loop for every method, and the token bookkeeping a
rollout needs: the policy's capped answers, a user turn between them, the
trajectory as verl takes it.
"""

import uuid

from verl.experimental.agent_loop.agent_loop import (
    AgentLoopBase,
    AgentLoopMetrics,
    AgentLoopOutput,
)

from obfuscation_immunity.common import progress
from obfuscation_immunity.common.base_method import running


def thinking(tokenizer, ids: list[int], prompt: int, mask: list[int]) -> int:
    """Policy tokens between a `<think>` and the next `</think>`; the
    opening tag may come from the prompt's template.
    """
    start, end = tokenizer.convert_tokens_to_ids(["<think>", "</think>"])
    inside, count = False, 0
    for token, own in zip(ids, [0] * prompt + mask):
        if token in (start, end):
            inside = token == start
        elif inside and own:
            count += 1
    return count


def extend(ids, mask, logprobs, tokens, own: int):
    """Tokens, mask and logprobs with `tokens` appended: a turn the policy
    wrote (own 1, its logprobs) or context (own 0, logprob 0).
    """
    values = getattr(tokens, "log_probs", None)
    tokens = list(getattr(tokens, "token_ids", tokens))
    if values or logprobs:
        logprobs = logprobs + list(values or [0.0] * len(tokens))
    return ids + tokens, mask + [own] * len(tokens), logprobs


class MethodLoop(AgentLoopBase):
    """verl's agent loop for any method: the method writes the trajectory.
    The helpers below are the token bookkeeping it needs.
    """

    def __init__(self, *args, **kwargs):
        """verl's loop, for one trajectory, with the response length it
        allows.
        """
        super().__init__(*args, **kwargs)
        self.response_length = self.rollout_config.response_length
        self.versions = {}

    async def run(self, sampling_params, **row) -> AgentLoopOutput:
        """The method's rollout of one row."""
        return await running().rollout(self, sampling_params, row)

    async def write(self, ids: list[int], sampling_params, tokens: int):
        """The policy's next turn after `ids`, within `tokens`. verl wants
        to know which weights wrote the trajectory: the first turn's, and
        the latest.
        """
        params = {**sampling_params, "max_tokens": max(tokens, 1)}
        turn = await self.server_manager.generate(
            request_id=uuid.uuid4().hex, prompt_ids=ids, sampling_params=params
        )
        if not self.versions:
            self.versions = dict(turn.extra_fields)
        elif turn.extra_fields.get("max_global_steps") is not None:
            self.versions["max_global_steps"] = turn.extra_fields[
                "max_global_steps"
            ]
        return turn

    async def answer(
        self, ids, mask, logprobs, sampling_params, reasoning, answering
    ):
        """The policy's answer after `ids`: it thinks until `</think>`, for
        at most `reasoning` tokens; if it is still thinking then, its
        thinking is closed for it (context, mask 0); then it answers within
        `answering`. The tokens, mask and logprobs with the answer appended,
        and what the policy wrote.
        """
        start = len(ids)
        close, end = self.tokenizer.convert_tokens_to_ids(
            ["</think>", "<|im_end|>"]
        )
        stop = {**sampling_params, "stop_token_ids": [close, end]}
        thought = await self.write(ids, stop, reasoning)
        ids, mask, logprobs = extend(ids, mask, logprobs, thought, 1)
        if end not in thought.token_ids:
            if close not in thought.token_ids:
                text = "\n</think>\n\n"
                closing = self.tokenizer.encode(text, add_special_tokens=False)
                ids, mask, logprobs = extend(ids, mask, logprobs, closing, 0)
            reply = await self.write(ids, sampling_params, answering)
            ids, mask, logprobs = extend(ids, mask, logprobs, reply, 1)
        return ids, mask, logprobs, self.tokenizer.decode(ids[start:])

    async def told(self, ids, mask, logprobs, messages, text: str):
        """Tokens, mask and logprobs with a user turn appended (mask 0)."""
        merged, mask, logprobs = await self.ct_merge_context_msg(
            messages,
            messages + [{"role": "user", "content": text}],
            ids,
            mask,
            logprobs,
        )
        return merged.token_ids, mask, logprobs

    def output(
        self, prompt: int, ids, mask, logprobs, reward, metrics, **extra
    ):
        """The trajectory as verl takes it; the pair, with its reward, the
        method's numbers, its transcript and `extra`, goes to the episode
        log the run's report reads (common.progress).
        """
        length = self.response_length
        output = AgentLoopOutput(
            prompt_ids=ids[:prompt],
            response_ids=ids[prompt:][:length],
            response_mask=mask[:length],
            response_logprobs=logprobs[:length] or None,
            reward_score=reward,
            num_turns=2,
            metrics=AgentLoopMetrics(),
        )
        progress.append(
            {
                "reward": reward,
                "metrics": metrics,
                "output_tokens": sum(mask),
                "reasoning_tokens": thinking(self.tokenizer, ids, prompt, mask),
                "input_tokens": len(ids) - sum(mask),
                "transcript": self.tokenizer.decode(ids),
                **extra,
            }
        )
        output.extra_fields.update(
            {
                **self.versions,
                "turn_scores": [],
                "tool_rewards": [],
            }
        )
        return output
