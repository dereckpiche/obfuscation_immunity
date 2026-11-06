"""Reasoning-gym puzzles answered in text: the tasks every method trains
on, and their verifier V.

The model reads a puzzle and writes its answer between <answer> and
</answer>; the puzzle's own dataset scores it. A run trains on one kind of
puzzle, so the policy has the time to learn it. The default, zebra puzzles,
is hard to verify: checking an answer means checking every clue, about the
work of finding it; and it is learnable: the 2B went from 0% to 70% right
in its first 20 steps. A method writes its puzzles on first use (`write`).
"""

import json
import re
from functools import cache
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import reasoning_gym

PUZZLE = "zebra_puzzles"
"""The default puzzle, a reasoning-gym dataset name."""

SYSTEM = (
    "Solve the puzzle. Think briefly, then write your final answer alone "
    "between <answer> and </answer>, in the format the puzzle asks for."
)

ANSWER = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)


@cache
def dataset(name: str, seed: int = 0, size: int = 1):
    """A reasoning-gym dataset."""
    return reasoning_gym.create_dataset(name, seed=seed, size=size)


def answer(text: str) -> str:
    """The last answer the text gives after its reasoning; empty if none."""
    found = ANSWER.findall(text.rpartition("</think>")[2])
    return found[-1].strip() if found else ""


def grade(task: dict, text: str) -> float:
    """V: the puzzle's score of the text's answer, in [0, 1]."""
    given = answer(text)
    if not given:
        return 0.0
    entry = task["entry"]
    return float(dataset(task["name"]).score_answer(answer=given, entry=entry))


def correct(score: float) -> bool:
    """A solution is correct when its score is full."""
    return score >= 1.0


def tasks(name: str, n: int, seed: int) -> list[dict]:
    """`n` puzzles of one kind."""
    entries = dataset(name, seed, n)
    keys = ("question", "answer", "metadata")
    return [
        {"name": name, "entry": {k: entries[i][k] for k in keys}}
        for i in range(n)
    ]


def row(task: dict, index: int) -> dict:
    """One parquet row in the layout verl's dataset reads."""
    return {
        "data_source": "puzzles",
        "prompt": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": task["entry"]["question"]},
        ],
        "reward_model": {"style": "rule", "ground_truth": ""},
        "extra_info": {
            "index": index,
            "task": json.dumps(task, default=str),
        },
    }


def write(out: Path, name: str, n: int = 2000, seed: int = 0) -> None:
    """`n` puzzles as verl's parquet file, `out`/train.parquet."""
    out.mkdir(parents=True, exist_ok=True)
    rows = [row(task, i) for i, task in enumerate(tasks(name, n, seed))]
    pq.write_table(pa.Table.from_pylist(rows), out / "train.parquet")
    print(f"{n} puzzles -> {out / 'train.parquet'}")
