"""
The judge: what a run tries against obfuscation, on a fixed task.

Each trajectory is a pair of answers to one puzzle: an honest one s, then a
sneaky one s̄, written with s in view, that should look right but be wrong
(common.loop). V, the puzzle's grader, labels both; a judge J says which
it believes correct, y = (y_s, y_s̄) ∈ {0,1}². The policy plays

    max_π min_J E[ V((1,0) | s, s̄) + 1 − J(y_true) ]

so it is paid for s right, s̄ wrong and a judge that cannot tell; both
terms are in [0, 1]. Runs differ only in their judge, a `Judge` subclass
built (`build_judge`) from the config's `judge` (configs/judge/), checked
against its typed `Config`.
"""

from dataclasses import dataclass, field
from typing import Any, ClassVar

from hydra.utils import get_class
from omegaconf import DictConfig, OmegaConf
from verl.experimental.agent_loop.agent_loop import AgentLoopBase

from obfuscation_immunity.common.pairs import Pair
from obfuscation_immunity.common.transcript import TranscriptSegment

JUDGE_PARTITION = "judge"
"""The transfer queue partition the judge's own trajectories go to."""


@dataclass
class Verdict:
    """
    What a judge says about one pair.

    Attributes:
        outcome_logprobs: log J(y) for each y in OUTCOMES, in (s, s̄) order;
            empty if the judge stated none.
        metrics: The judge's own numbers for this pair; the trainer logs
            their means as `game/<name>`.
        statement: What a text judge stated after its thinking.
        segments: What the judge read and generated, token by token, for
            the pair's page.
        judge_trajectory: The judge's own turn, for verl to train the
            judge's adapter on (it goes to `JUDGE_PARTITION`):
            `prompt_ids` (its question), `response_ids`, `response_mask`,
            `response_logprobs` and its `score`; None if the judge does not
            learn from this pair.
        stated: Whether the judge stated J in its format; if not, the game
            ends and the policy wins.
    """

    outcome_logprobs: list[float]
    metrics: dict[str, float] = field(default_factory=dict)
    statement: str = ""
    segments: list[TranscriptSegment] = field(default_factory=list)
    judge_trajectory: dict[str, Any] | None = None
    stated: bool = True

    @property
    def likeliest_outcome(self) -> int:
        """The index in OUTCOMES of the outcome J finds likeliest."""
        logprobs = self.outcome_logprobs
        return max(range(len(logprobs)), key=lambda k: logprobs[k])


@dataclass(frozen=True)
class JudgeConfig:
    """What every judge takes: nothing yet; a judge's own config adds to it."""


class Judge:
    """
    A judge J of the pair; the task stays the same.

    Attributes:
        config: The judge's config, checked against its `Config`.
    """

    Config: ClassVar[type[JudgeConfig]] = JudgeConfig
    """The dataclass the judge's config is checked against."""
    actor_worker_class: ClassVar[type | None] = None
    """verl's actor worker class for the run, if not verl's own."""
    config: JudgeConfig

    def __init__(self, config: JudgeConfig):
        """
        Keeps the judge's config.

        Args:
            config: The judge's config, already checked.
        """
        self.config = config

    async def verdict(self, loop: AgentLoopBase, pair: Pair) -> Verdict:
        """
        Judges one pair.

        Args:
            loop: The agent loop the pair was generated in; it holds the
                tokenizer and the vLLM client.
            pair: The pair to judge.

        Returns:
            log J over OUTCOMES, in (s, s̄) order, and what the judge
            measured.

        Raises:
            NotImplementedError: Every judge defines its own.
        """
        raise NotImplementedError

    def before_verl_starts(self, config: DictConfig) -> None:
        """
        Changes what verl builds, before it builds it; by default nothing.

        Args:
            config: The run's whole config.
        """

    def on_trainer_ready(self, trainer: Any) -> None:
        """
        Acts once verl's workers are built and synced; by default nothing.

        Args:
            trainer: The run's common.trainer `PairTrainer`; its
                `global_steps` are the steps already done (none, unless the
                run resumed).
        """

    def after_actor_update(self, trainer: Any, metrics: dict) -> None:
        """
        Acts after the actor's update, each step; by default nothing.

        It runs before verl saves its checkpoint and syncs vLLM.

        Args:
            trainer: The run's common.trainer `PairTrainer`.
            metrics: verl's numbers for the step, to add to.
        """


def build_judge(node: DictConfig) -> Judge:
    """
    Builds the judge the config names, its config checked by type.

    A missing key, an unknown one or a wrong type fails here, before
    anything runs.

    Args:
        node: The config's `judge`: `_target_`, the judge's class, and
            the fields of its `Config`.

    Returns:
        The judge.
    """
    judge_class: type[Judge] = get_class(node["_target_"])
    values = {k: v for k, v in node.items() if k != "_target_"}
    schema = OmegaConf.structured(judge_class.Config)
    config = OmegaConf.to_object(OmegaConf.merge(schema, values))
    assert isinstance(config, judge_class.Config)
    return judge_class(config)


def whole_batches(
    listed: dict[str, dict[str, Any]], batch_multiple: int
) -> list[str]:
    """
    Picks the trajectories of a partition that make whole batches, oldest first.

    Args:
        listed: The partition's keys and their tags (`global_steps`: the
            step that generated each).
        batch_multiple: The batch size verl trains on without padding.

    Returns:
        The oldest keys, as many as the largest multiple of the batch size
        there are; none if there is not yet a whole batch.
    """
    oldest_first = sorted(listed, key=lambda key: listed[key]["global_steps"])
    whole = len(oldest_first) // batch_multiple * batch_multiple
    return oldest_first[:whole]
