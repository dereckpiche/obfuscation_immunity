"""
Reasoning-gym puzzles answered in text: the task, and its verifier V.

The model reads a puzzle and, after its thinking, replies with its answer
alone within \\boxed{}, as Qwen3.5's card recommends for math; the puzzle's own
dataset scores it.
The system prompt, with the budget, is the loop's (`puzzle_messages`); a
dataset row holds the puzzle alone, and an older row's system turn is
dropped. A run trains on one kind of puzzle, so the policy has the time to
learn it. The default, zebra puzzles, is hard to verify: checking an answer
means checking every clue, about the work of finding it; and it is
learnable: the 2B went from 0% to 70% right in its first 20 steps on the
4 × 4 ones. A run uses 3 × 3 ones (configs/train.yaml `puzzle.settings`),
answerable within its budget, and writes them on first use
(`write_puzzles`).
"""

import json
import os
import re
from functools import cache
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import reasoning_gym

SYSTEM_PROMPT = (
    "Solve the puzzle. You have {budget} tokens for your thinking and your "
    "reply together; a reply cut off at that limit fails, so think briefly. "
    "Please reason step by step, and put your final answer within "
    "\\boxed{{}}: after your thinking, reply with only \\boxed{{your "
    "answer}}, with nothing before or after it, not even a space. A reply "
    "in any other form fails, and you get no second try."
)
"""What the policy is told before every puzzle, with its budget."""

ANSWER_ALONE = re.compile(r"\\boxed\{([^{}\s](?:[^{}]*[^{}\s])?)\}")
"""A reply that is one boxed answer and nothing else, not even a space."""


@cache
def puzzle_dataset(
    name: str, seed: int = 0, size: int = 1, **settings: Any
) -> Any:
    """
    Creates a reasoning-gym dataset, once per process.

    Args:
        name: The reasoning-gym dataset's name, such as "zebra_puzzles".
        seed: The seed its puzzles are drawn with.
        size: How many puzzles it holds.
        **settings: The dataset's own settings, such as a zebra puzzle's
            `num_people`; its defaults if none.

    Returns:
        The dataset; it also grades answers to its puzzles.
    """
    return reasoning_gym.create_dataset(name, seed=seed, size=size, **settings)


def puzzle_messages(
    prompt: list[dict[str, str]], budget: int
) -> list[dict[str, str]]:
    """
    Builds the policy's chat for a puzzle: the system prompt, then the puzzle.

    Args:
        prompt: The dataset row's chat prompt; any system turn is dropped.
        budget: The tokens the policy's first try may take.

    Returns:
        The chat messages.
    """
    system = {"role": "system", "content": SYSTEM_PROMPT.format(budget=budget)}
    return [system] + [m for m in prompt if m["role"] != "system"]


def answer_alone(final: str) -> str | None:
    """
    Reads a reply that is one boxed answer and nothing else.

    Args:
        final: What the policy wrote after its thinking.

    Returns:
        The answer inside `\\boxed{}`; None if the reply is anything else,
        even a space more.
    """
    match = ANSWER_ALONE.fullmatch(final)
    return None if match is None else match.group(1)


def grade_final_answer(task: dict[str, Any], final: str) -> float:
    """
    Scores a final answer with the puzzle's own grader, V.

    Args:
        task: The puzzle, as `sample_tasks` gives it.
        final: What the policy wrote after its thinking.

    Returns:
        The grader's score in [0, 1]; 0 unless the reply is one boxed
        answer and nothing else.
    """
    answer = answer_alone(final)
    if answer is None:
        return 0.0
    dataset = puzzle_dataset(task["name"])
    return float(dataset.score_answer(answer=answer, entry=task["entry"]))


def is_correct(score: float) -> bool:
    """
    Tells whether a grader's score counts as correct.

    Args:
        score: A score from `grade_final_answer`.

    Returns:
        True for full marks only.
    """
    return score >= 1.0


def sample_tasks(
    name: str, count: int, seed: int, **settings: Any
) -> list[dict[str, Any]]:
    """
    Draws puzzles of one kind, each with what grading needs.

    Args:
        name: The reasoning-gym dataset's name.
        count: How many puzzles to draw.
        seed: The seed they are drawn with.
        **settings: The dataset's own settings.

    Returns:
        One task per puzzle: the dataset's name, and the puzzle's question,
        answer and metadata.
    """
    entries = puzzle_dataset(name, seed, count, **settings)
    keys = ("question", "answer", "metadata")
    return [
        {"name": name, "entry": {k: entries[i][k] for k in keys}}
        for i in range(count)
    ]


def parquet_row(task: dict[str, Any], index: int) -> dict[str, Any]:
    """
    Lays one puzzle out as a row of verl's dataset.

    Args:
        task: The puzzle, as `sample_tasks` gives it.
        index: Its position in the file.

    Returns:
        The row: the chat prompt, and the task itself in `extra_info` for
        grading.
    """
    return {
        "data_source": "puzzles",
        "prompt": [{"role": "user", "content": task["entry"]["question"]}],
        "reward_model": {"style": "rule", "ground_truth": ""},
        "extra_info": {
            "index": index,
            "task": json.dumps(task, default=str),
        },
    }


def write_puzzles(
    path: Path, name: str, count: int = 2000, seed: int = 0, **settings: Any
) -> None:
    """
    Writes puzzles of one kind as verl's parquet file.

    It is written aside and moved into place, so that runs starting at once
    never read half a file.

    Args:
        path: The parquet file to write; its directory is created.
        name: The reasoning-gym dataset's name.
        count: How many puzzles to write.
        seed: The seed they are drawn with.
        **settings: The dataset's own settings.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tasks = sample_tasks(name, count, seed, **settings)
    rows = [parquet_row(task, index) for index, task in enumerate(tasks)]
    aside = path.with_name(f"{path.name}.{os.getpid()}")
    pq.write_table(pa.Table.from_pylist(rows), aside)
    aside.replace(path)
    print(f"{count} puzzles -> {path}")
