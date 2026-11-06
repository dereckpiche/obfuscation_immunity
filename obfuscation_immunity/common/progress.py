"""A few lines per step on the console, and the same numbers in wandb.

Every finished pair is appended to a JSON-lines episode log (`append`);
at each training step the trainer hands over verl's numbers and the pairs
that arrived since the last step are averaged. These groups reach the
console and wandb, nothing else:

    puzzles                  the game: the reward, honest_correct,
                             sneaky_correct, the judge's accuracy and
                             log-loss, the method's own numbers; then
                             tokens written, thinking and read; and `pair`,
                             one random pair per step as a page
                             (common.render), with the grade, the judge's
                             verdict and the reward
    training                 the update: kl to the reference, entropy,
                             pg_loss, grad_norm, clip fraction, ppo_kl, the
                             advantages' range, learning rate,
                             rollout/training mismatch, and
                             groups_with_signal, the share of GRPO groups
                             whose rewards differ
    time_and_memory          where the step's time went, a pair's parts
                             (the honest answer, the sneaky one, the
                             judge) on average, peak memory
"""

import json
import os
import random
import re

import wandb

from obfuscation_immunity.common import render

EPISODES = "OI_EPISODES"
"""The environment variable that names the episode log."""
BEHAVIOUR = {
    "output_tokens": "out {:.0f}",
    "reasoning_tokens": "think {:.0f}",
    "input_tokens": "in {:.0f}",
}
TRAINING = {
    "kl": ("actor/kl_loss", "kl {:.4f}"),
    "entropy": ("actor/entropy", "entropy {:.3f}"),
    "pg_loss": ("actor/pg_loss", "pg_loss {:.4f}"),
    "grad_norm": ("actor/grad_norm", "grad_norm {:.2f}"),
    "clip_fraction": ("actor/pg_clipfrac", "clip {:.3f}"),
    "ppo_kl": ("actor/ppo_kl", "ppo_kl {:.4f}"),
    "advantage_max": ("critic/advantages/max", "adv_max {:.2f}"),
    "advantage_min": ("critic/advantages/min", "adv_min {:.2f}"),
    "learning_rate": ("actor/lr", "lr {:.1e}"),
    "mismatch": ("rollout_corr/kl", "mismatch {:.4f}"),
}
TIME_AND_MEMORY = {
    "generation_seconds": ("timing_s/gen", "generation {:.0f}s"),
    "training_seconds": ("timing_s/update_actor", "training {:.0f}s"),
    "step_seconds": ("timing_s/step", "step {:.0f}s"),
    "gpu_allocated_gb": ("actor/perf/max_memory_allocated_gb", "gpu {:.1f}GB"),
    "gpu_reserved_gb": (
        "actor/perf/max_memory_reserved_gb",
        "reserved {:.1f}GB",
    ),
    "cpu_used_gb": ("actor/perf/cpu_memory_used_gb", "cpu {:.0f}GB"),
}
FORMATS = {
    "reward": "reward {:.2f}",
    "honest_seconds": "honest {:.0f}s",
    "sneaky_seconds": "sneaky {:.0f}s",
    "judge_seconds": "judge {:.0f}s",
    "groups_with_signal": "signal {:.2f}",
    **BEHAVIOUR,
    **{name: fmt for name, (_, fmt) in (TRAINING | TIME_AND_MEMORY).items()},
}


def append(record: dict) -> None:
    """Append one finished pair to the episode log, if there is one."""
    path = os.environ.get(EPISODES)
    if path:
        with open(path, "a") as file:
            file.write(json.dumps(record) + "\n")


class EpisodeLog:
    """Reads the episode log from where the previous read stopped."""

    def __init__(self, path: str):
        """Start at the end of whatever the file already holds."""
        self.path = path
        self.offset = os.path.getsize(path) if os.path.exists(path) else 0

    def drain(self) -> list[dict]:
        """The records appended since the last call, oldest first."""
        if not os.path.exists(self.path):
            return []
        with open(self.path) as file:
            file.seek(self.offset)
            lines = file.readlines()
            self.offset = file.tell()
        return [json.loads(line) for line in lines if line.strip()]


def mean(values: list[float]) -> float:
    """The average of a non-empty list."""
    return sum(values) / len(values)


def averages(found: list[dict]) -> dict[str, float]:
    """The mean of every number in a list of name -> value dicts, by name,
    over the dicts that have it.
    """
    names = dict.fromkeys(k for f in found for k in f)
    return {n: mean([f[n] for f in found if n in f]) for n in names}


def rollout(records: list[dict]) -> dict[str, float]:
    """The game's averages: the reward, the game's and the method's
    numbers, then the behaviour.
    """
    numbers = {"reward": mean([r["reward"] for r in records])}
    numbers |= averages([r["metrics"] for r in records])
    return numbers | {k: mean([r[k] for r in records]) for k in BEHAVIOUR}


def signal(records: list[dict]) -> float:
    """The share of the step's GRPO groups (pairs of one puzzle) whose
    rewards differ: the others have no advantage and teach nothing.
    """
    groups = {}
    for r in records:
        groups.setdefault(r["task"], set()).add(round(r["reward"], 6))
    return mean([float(len(rewards) > 1) for rewards in groups.values()])


def parts(records: list[dict]) -> dict[str, float]:
    """A pair's parts' mean seconds, as `<part>_seconds`."""
    timed = [r["seconds"] for r in records if "seconds" in r]
    return {f"{k}_seconds": v for k, v in averages(timed).items()}


def report(verl: dict, records: list[dict]) -> dict[str, dict[str, float]]:
    """The step's numbers by group; a group with nothing to report is left
    out.
    """
    groups = {"puzzles": rollout(records)} if records else {}
    training = {k: verl[key] for k, (key, _) in TRAINING.items() if key in verl}
    if records:
        training["groups_with_signal"] = signal(records)
    if training:
        groups["training"] = training
    memory = TIME_AND_MEMORY.items()
    found = {k: verl[key] for k, (key, _) in memory if key in verl}
    found |= parts(records)
    if found:
        groups["time_and_memory"] = found
    return groups


def lines(step: int, groups: dict[str, dict[str, float]]) -> list[str]:
    """One console line per group."""
    out = []
    for group, numbers in groups.items():
        text = "  ".join(
            FORMATS.get(name, name + " {:.3f}").format(value)
            for name, value in numbers.items()
        )
        out.append(f"step {step:>4}  {group:<17}  {text}")
    return out


def subtitle(step: int, record: dict) -> str:
    """The line under a pair's title: when, and how much it wrote."""
    return (
        f"step {step} · {record['output_tokens']:,} tokens written, "
        f"{record['reasoning_tokens']:,} of them thinking"
    )


def pages(records: list[dict], step: int) -> dict[str, str]:
    """One random pair of the step as a page, as `puzzles/pair`."""
    if not records:
        return {}
    pick = random.choice(records)
    page = render.page(
        pick["transcript"], "Pair", subtitle(step, pick), pick["after"]
    )
    return {"puzzles/pair": page}


class Progress:
    """The run's console lines and wandb run."""

    def __init__(self, project: str, name: str, config: dict, episodes: str):
        """Open the wandb run (a restarted run resumes it) and the log."""
        self.run = wandb.init(
            project=project,
            name=name,
            id=re.sub(r"[^A-Za-z0-9_-]", "-", name),
            resume="allow",
            config=config,
        )
        print(f"wandb {self.run.url}", flush=True)
        self.log = EpisodeLog(episodes)

    def step(self, step: int, verl: dict) -> None:
        """Report the step: its console lines, its numbers and pages."""
        records = self.log.drain()
        groups = report(verl, records)
        for printed in lines(step, groups):
            print(printed, flush=True)
        logged = {
            f"{group}/{name}": value
            for group, numbers in groups.items()
            for name, value in numbers.items()
        }
        logged |= {
            key: wandb.Html(page, inject=False)
            for key, page in pages(records, step).items()
        }
        self.run.log(logged, step=step)

    def finish(self) -> None:
        """Close the wandb run and its service now: Ray may close the
        service's pipe before wandb's own exit hook would.
        """
        self.run.finish()
        wandb.teardown()
