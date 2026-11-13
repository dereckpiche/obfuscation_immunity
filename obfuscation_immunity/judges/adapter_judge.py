"""
A judge that is a LoRA adapter on the actor's model, trained by verl.

Asked the pair's question q, the judge writes o: its thinking, within the
actor's budget, then the one line of the probability of each outcome, J_o
(judges.text_judge). Its weights are a
judge adapter ψ on the actor's own model (judges.judge_adapter): stacked
on the actor's adapter π (judges.stacked_judge), or alone on the base
model (judges.standalone_judge). It plays the other side of the game:

    max_π min_ψ E[ V((1,0) | s, s̄) + 1 − J(y_true) ]

Its reward is J_o(y_true), in [0, 1] as the policy's terms are, plus
`FORMAT_STAKE` if its turn was in the format; a turn out of it ends the
game with −`FORMAT_STAKE` alone. J(y_true) is linear in J, not a proper
score: the judge does best by putting all its mass on the outcome it finds
likeliest, not by stating what it believes. A judge that is not
`verified` is paid its format stake alone: it learns the format and
nothing else, so its judgement stays fixed. verl trains ψ on it right
after the actor's update, with the actor's own algorithm and safeguards:
GRPO within each puzzle, PPO's clipping, the KL loss, the correction of the
rollout and training mismatch, gradient clipping. After K steps ψ is
frozen (never, if K is -1); nothing else changes.

After each step, ψ (with π, when stacked) is written for vLLM
(`served_adapter_dir`): a pair generated with the actor's weights of step
s is judged with the judge's of step s (judges.judge_server).
"""

import math
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

from omegaconf import DictConfig, open_dict
from verl.experimental.agent_loop.agent_loop import AgentLoopBase

from obfuscation_immunity.common.judge import (
    JUDGE_PARTITION,
    Judge,
    JudgeConfig,
    Verdict,
)
from obfuscation_immunity.common.pairs import Pair
from obfuscation_immunity.judges.judge_adapter import (
    JUDGE_ADAPTER_CONFIG_KEY,
    JudgeAdapterWorker,
)
from obfuscation_immunity.judges.text_judge import JudgeTurn, ask_text_judge


@dataclass(frozen=True)
class AdapterJudgeConfig(JudgeConfig):
    """
    An adapter judge's settings.

    Attributes:
        freeze_at: K, the actor updates after which the judge adapter stops
            learning; -1 for never.
        adapter_rank: The judge adapter's LoRA rank; the LoRA vLLM serves
            the judge with is at most vLLM's `max_lora_rank`.
        adapter_lr: The judge adapter's learning rate.
        run_dir: The run's directory; its name keeps the served LoRAs of
            runs apart.
        verified: Whether the judge is paid J(y_true) against V's labels;
            if not, only its format stake.
    """

    freeze_at: int
    adapter_rank: int
    adapter_lr: float
    run_dir: str
    verified: bool


def served_adapter_dir(run_dir: str, step: int) -> Path:
    """
    Gives the directory of the LoRA vLLM serves the judge with at a step.

    It is in the node's temporary directory: the run is on one node.

    Args:
        run_dir: The run's directory.
        step: The step the weights come from.

    Returns:
        The directory.
    """
    return Path(tempfile.gettempdir()) / Path(run_dir).name / f"judge-{step}"


def judge_score(turn: JudgeTurn, pair: Pair, verified: bool) -> float:
    """
    Scores the judge's turn: its reward.

    Args:
        turn: The judge's turn.
        pair: The pair it judged.
        verified: Whether the judge is paid J(y_true).

    Returns:
        J(y_true) + `FORMAT_STAKE` if it stated J in the format, without
        J(y_true) unless `verified`; else −`FORMAT_STAKE`.
    """
    if not turn.verdict.stated or not verified:
        return turn.turn.stake
    true_logprob = turn.verdict.outcome_logprobs[pair.true_outcome]
    return math.exp(true_logprob) + turn.turn.stake


class AdapterJudge(Judge):
    """
    A judge adapter ψ that reads the pair as text and learns its log score.

    A subclass says whether ψ is stacked on the actor's adapter (`STACKED`).

    Attributes:
        config: The judge's config.
    """

    Config = AdapterJudgeConfig
    config: AdapterJudgeConfig
    actor_worker_class = JudgeAdapterWorker
    STACKED: ClassVar[bool]
    """Whether the judge thinks with π + ψ, rather than ψ alone."""

    def before_verl_starts(self, config: DictConfig) -> None:
        """
        Gives verl's workers the judge adapter's settings and vLLM its LoRA.

        Args:
            config: The run's whole config.
        """
        from obfuscation_immunity.judges import judge_server

        judge_server.use_two_lora_vllm_servers()
        with open_dict(config):
            config.actor_rollout_ref[JUDGE_ADAPTER_CONFIG_KEY] = (
                self.judge_adapter_settings()
            )

    def judge_adapter_settings(self) -> dict[str, Any]:
        """
        Gives the judge adapter's settings, as verl's workers read them.

        Returns:
            judges.judge_adapter `JudgeAdapterSettings`'s fields.
        """
        return {
            "rank": self.config.adapter_rank,
            "lr": self.config.adapter_lr,
            "stacked": self.STACKED,
        }

    async def verdict(self, loop: AgentLoopBase, pair: Pair) -> Verdict:
        """
        Asks the judge to judge one pair; keeps its turn while it learns.

        The pair was generated with the actor's weights of some step; the
        judge answers with its own of that step, and learns from the pair
        if it learns at the next.

        Args:
            loop: The agent loop the pair was generated in.
            pair: The pair to judge.

        Returns:
            J as the judge stated it, in (s, s̄) order, with its turn to
            learn from while it learns.
        """
        step = loop.weight_versions.get("max_global_steps") or 0
        learning = self.learns_at(step + 1)
        turn = await ask_text_judge(
            loop,
            pair,
            judge_learns=learning,
            lora_dir=str(served_adapter_dir(self.config.run_dir, step)),
        )
        if learning:
            turn.verdict.judge_trajectory = {
                "prompt_ids": turn.question_token_ids,
                "response_ids": turn.turn.token_ids,
                "response_mask": [1] * len(turn.turn.token_ids),
                "response_logprobs": turn.turn.logprobs,
                "score": judge_score(turn, pair, self.config.verified),
            }
        return turn.verdict

    def learns_at(self, step: int) -> bool:
        """
        Tells whether the judge adapter learns at a step.

        Args:
            step: An actor update, counted from 1.

        Returns:
            Whether the judge adapter is updated with it: always if K is
            -1, else up to K.
        """
        return self.config.freeze_at < 0 or step <= self.config.freeze_at

    def on_trainer_ready(self, trainer: Any) -> None:
        """
        Writes the judge's LoRA for the first rollouts.

        Args:
            trainer: The run's common.trainer `PairTrainer`.
        """
        self.write_served_adapter(trainer, trainer.global_steps)

    def after_actor_update(self, trainer: Any, metrics: dict) -> None:
        """
        Trains ψ with verl on the step's judge turns, until K.

        The actor's engine takes the judge's turn for the update; then the
        judge's LoRA is written for the next rollouts.

        Args:
            trainer: The run's common.trainer `PairTrainer`.
            metrics: verl's numbers for the step, to add the judge's to.
        """
        step = trainer.global_steps
        if self.learns_at(step):
            trainer.actor_rollout_wg.use_judge_adapter(True)
            try:
                judge_metrics = trainer.update_on_partition(JUDGE_PARTITION)
            finally:
                trainer.actor_rollout_wg.use_judge_adapter(False)
            metrics.update(
                {
                    f"judge/{name}": value
                    for name, value in judge_metrics.items()
                }
            )
        self.write_served_adapter(trainer, step)

    def write_served_adapter(self, trainer: Any, step: int) -> None:
        """
        Writes the judge's LoRA of a step, and removes the one before last.

        Args:
            trainer: The run's common.trainer `PairTrainer`.
            step: The step the weights come from.
        """
        run_dir = self.config.run_dir
        trainer.actor_rollout_wg.write_served_adapter(
            str(served_adapter_dir(run_dir, step)),
            str(served_adapter_dir(run_dir, step - 2)),
        )
