"""
The run's report: a few lines per step on the console, the same in wandb.

verl's own logger is off (`trainer.logger: []`): its hundred-odd numbers
bury the game. Each step reports these groups, and nothing else:

    puzzles          the game: the reward, V's labels, which turns kept
                     the format, the judge's accuracy and J(y_true), the
                     judge's own numbers; the tokens the policy wrote,
                     thought and read; and `pair`, one random pair of the
                     step as a page (common.render)
    training         the update: KL to the reference, entropy, pg_loss,
                     grad_norm, clip fraction, ppo_kl, the advantages'
                     range, the learning rate, the rollout/training
                     mismatch, and the share of GRPO groups whose rewards
                     differ (`signal`)
    judge_training   the same for a judge verl trains, while it learns
    time_and_memory  where the step's time went, a pair's parts on
                     average, peak memory
"""

import random
import re
from typing import Any, cast

import wandb
from omegaconf import DictConfig, OmegaConf
from transformers import PreTrainedTokenizerBase

from obfuscation_immunity.common import render
from obfuscation_immunity.common.chat_format import ChatFormat
from obfuscation_immunity.common.transcript import (
    POLICY,
    PUZZLE_KIND,
    PairRecord,
)

TRAINING_KEYS = {
    "kl": "actor/kl_loss",
    "entropy": "actor/entropy",
    "pg_loss": "actor/pg_loss",
    "grad_norm": "actor/grad_norm",
    "clip_fraction": "actor/pg_clipfrac",
    "ppo_kl": "actor/ppo_kl",
    "advantage_max": "critic/advantages/max",
    "advantage_min": "critic/advantages/min",
    "learning_rate": "actor/lr",
    "mismatch": "rollout_corr/kl",
}
"""The training group's numbers, by the name of verl's metric."""

JUDGE_TRAINING_KEYS = {
    name: f"judge/{key}" for name, key in TRAINING_KEYS.items()
}
"""The judge training group's numbers, by the name of verl's metric."""

TIME_AND_MEMORY_KEYS = {
    "generation_seconds": "timing_s/gen",
    "training_seconds": "timing_s/update_actor",
    "judge_training_seconds": "timing_s/judge_update",
    "step_seconds": "timing_s/step",
    "gpu_allocated_gb": "actor/perf/max_memory_allocated_gb",
    "gpu_reserved_gb": "actor/perf/max_memory_reserved_gb",
    "cpu_used_gb": "actor/perf/cpu_memory_used_gb",
}
"""The time and memory group's numbers, by the name of verl's metric."""

CONSOLE_FORMATS = {
    "reward": "reward {:.2f}",
    "tokens_written": "out {:.0f}",
    "tokens_thinking": "think {:.0f}",
    "tokens_read": "in {:.0f}",
    "kl": "kl {:.4f}",
    "entropy": "entropy {:.3f}",
    "pg_loss": "pg_loss {:.4f}",
    "grad_norm": "grad_norm {:.2f}",
    "clip_fraction": "clip {:.3f}",
    "ppo_kl": "ppo_kl {:.4f}",
    "advantage_max": "adv_max {:.2f}",
    "advantage_min": "adv_min {:.2f}",
    "learning_rate": "lr {:.1e}",
    "mismatch": "mismatch {:.4f}",
    "signal": "signal {:.2f}",
    "generation_seconds": "generation {:.0f}s",
    "training_seconds": "training {:.0f}s",
    "judge_training_seconds": "judge training {:.0f}s",
    "step_seconds": "step {:.0f}s",
    "gpu_allocated_gb": "gpu {:.1f}GB",
    "gpu_reserved_gb": "reserved {:.1f}GB",
    "cpu_used_gb": "cpu {:.0f}GB",
    "honest_seconds": "honest {:.0f}s",
    "sneaky_seconds": "sneaky {:.0f}s",
    "judge_seconds": "judge {:.0f}s",
}
"""How each number is written on the console; others get 3 decimals."""


def mean(values: list[float]) -> float:
    """
    Averages a non-empty list.

    Args:
        values: The numbers.

    Returns:
        Their mean.
    """
    return sum(values) / len(values)


def means_by_name(found: list[dict[str, float]]) -> dict[str, float]:
    """
    Averages every number in a list of named numbers, by name.

    Args:
        found: Numbers by name; a name may be missing from some.

    Returns:
        Each name's mean over the dicts that have it.
    """
    names = dict.fromkeys(name for numbers in found for name in numbers)
    return {
        name: mean([numbers[name] for numbers in found if name in numbers])
        for name in names
    }


def thinking_length(token_ids: list[int], think_end: int) -> int:
    """
    Counts a generation's thinking tokens, its closing tag included.

    Args:
        token_ids: The generation.
        think_end: The id of the tag that closes the thinking.

    Returns:
        The tokens up to that tag; all of them if it never closed.
    """
    if think_end in token_ids:
        return token_ids.index(think_end) + 1
    return len(token_ids)


def policy_token_counts(record: PairRecord, think_end: int) -> dict[str, float]:
    """
    Counts the tokens the policy wrote, thought and read in one pair.

    Args:
        record: The pair's record.
        think_end: The id of the tag that closes the thinking.

    Returns:
        `tokens_written`, `tokens_thinking` and `tokens_read` (the puzzle).
    """
    written = [s for s in record.segments if s.source == POLICY]
    return {
        "tokens_written": float(sum(len(s.token_ids) for s in written)),
        "tokens_thinking": float(
            sum(thinking_length(s.token_ids, think_end) for s in written)
        ),
        "tokens_read": float(
            sum(
                len(s.token_ids)
                for s in record.segments
                if s.kind == PUZZLE_KIND
            )
        ),
    }


def groups_with_signal(records: list[PairRecord]) -> float:
    """
    Measures the share of GRPO groups whose rewards differ.

    A group is the pairs of one puzzle; one whose rewards all agree has no
    advantage and teaches nothing.

    Args:
        records: The step's pairs.

    Returns:
        The share of groups with more than one reward.
    """
    rewards_by_puzzle: dict[tuple[int, ...], set[float]] = {}
    for record in records:
        puzzle = tuple(record.segments[0].token_ids)
        rewards_by_puzzle.setdefault(puzzle, set()).add(round(record.reward, 6))
    return mean(
        [float(len(rewards) > 1) for rewards in rewards_by_puzzle.values()]
    )


def step_report(
    verl_metrics: dict[str, float], records: list[PairRecord], think_end: int
) -> dict[str, dict[str, float]]:
    """
    Gathers the step's numbers into the report's groups.

    Args:
        verl_metrics: verl's numbers for the step.
        records: The step's pairs.
        think_end: The id of the tag that closes the thinking, to count
            thinking tokens.

    Returns:
        The numbers by group; a group with nothing to report is left out.
    """
    report = {}
    if records:
        puzzles = {"reward": mean([r.reward for r in records])}
        puzzles |= means_by_name([r.metrics for r in records])
        puzzles |= means_by_name(
            [policy_token_counts(r, think_end) for r in records]
        )
        report["puzzles"] = puzzles
    training = {
        name: verl_metrics[key]
        for name, key in TRAINING_KEYS.items()
        if key in verl_metrics
    }
    if records:
        training["signal"] = groups_with_signal(records)
    if training:
        report["training"] = training
    judge_training = {
        name: verl_metrics[key]
        for name, key in JUDGE_TRAINING_KEYS.items()
        if key in verl_metrics
    }
    if judge_training:
        report["judge_training"] = judge_training
    time_and_memory = {
        name: verl_metrics[key]
        for name, key in TIME_AND_MEMORY_KEYS.items()
        if key in verl_metrics
    }
    time_and_memory |= means_by_name(
        [
            {f"{part}_seconds": s for part, s in r.part_seconds.items()}
            for r in records
        ]
    )
    if time_and_memory:
        report["time_and_memory"] = time_and_memory
    return report


def console_lines(step: int, report: dict[str, dict[str, float]]) -> list[str]:
    """
    Writes the step's report as one console line per group.

    Args:
        step: The step's number.
        report: The step's numbers by group.

    Returns:
        The lines.
    """
    lines = []
    for group, numbers in report.items():
        text = "  ".join(
            CONSOLE_FORMATS.get(name, name + " {:.3f}").format(value)
            for name, value in numbers.items()
        )
        lines.append(f"step {step:>4}  {group:<15}  {text}")
    return lines


class Progress:
    """
    The run's console lines and wandb run.

    Attributes:
        run: The wandb run; a restarted run resumes it.
        page_meta: What each pair's page says it is, after its puzzle.
        chat_markers: The setup's turn markers (`chat:`), read with the
            tokenizer at the first step.
    """

    run: wandb.Run
    page_meta: dict[str, str]
    chat_markers: dict[str, str]

    def __init__(self, config: DictConfig):
        """
        Opens the run's wandb run, named after the experiment.

        Args:
            config: The run's whole config, logged with the run.
        """
        name = config.trainer.experiment_name
        self.run = wandb.init(
            project=config.trainer.project_name,
            name=name,
            id=re.sub(r"[^A-Za-z0-9_-]", "-", name),
            resume="allow",
            config=cast(dict[str, Any], OmegaConf.to_container(config)),
        )
        print(f"wandb {self.run.url}", flush=True)
        self.chat_markers = cast(
            dict[str, str], OmegaConf.to_container(config.chat)
        )
        self.page_meta = {
            "judge": config.judge._target_.rsplit(".", 1)[-1],
            "model": config.actor_rollout_ref.model.path.rsplit("/", 1)[-1],
            "run": name,
        }

    def log_step(
        self,
        step: int,
        verl_metrics: dict[str, float],
        records: list[PairRecord],
        tokenizer: PreTrainedTokenizerBase,
    ) -> None:
        """
        Reports one step: its console lines, its numbers and a pair's page.

        Args:
            step: The step's number.
            verl_metrics: verl's numbers for the step.
            records: The step's pairs.
            tokenizer: The run's tokenizer, to render the pair's page.
        """
        chat = ChatFormat.build(tokenizer, self.chat_markers)
        report = step_report(verl_metrics, records, chat.think_close_id)
        for line in console_lines(step, report):
            print(line, flush=True)
        logged: dict[str, object] = {
            f"{group}/{name}": value
            for group, numbers in report.items()
            for name, value in numbers.items()
        }
        if records:
            meta = {**self.page_meta, "step": str(step)}
            page = render.pair_page_html(
                random.choice(records), tokenizer, chat, meta
            )
            logged["puzzles/pair"] = wandb.Html(page, inject=False)
        self.run.log(logged, step=step)

    def finish(self) -> None:
        """
        Closes the wandb run and its service.

        Done here rather than at exit: Ray may close the service's pipe
        before wandb's own exit hook runs.
        """
        self.run.finish()
        wandb.teardown()
