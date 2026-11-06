import asyncio
import json
import math
from types import SimpleNamespace

import pytest

from obfuscation_immunity import ActorJudge, CoevolvingJudge, Probe
from obfuscation_immunity.common import base_method, pairs, progress, puzzles
from obfuscation_immunity.common.base_method import BaseMethod
from obfuscation_immunity.common.loop import MethodLoop

METHODS = [ActorJudge, CoevolvingJudge, Probe]


def test_a_puzzle_survives_its_row_and_grades_exactly():
    for task in puzzles.tasks(puzzles.PUZZLE, 10, seed=0):
        row = puzzles.row(task, 0)
        task = json.loads(row["extra_info"]["task"])
        right = f"</think><answer>{task['entry']['answer']}</answer>"
        assert puzzles.grade(task, right) == 1.0
        assert puzzles.grade(task, "</think><answer>12</answer>") < 1.0
        assert puzzles.grade(task, "no tags") == 0.0
        assert [m["role"] for m in row["prompt"]] == ["system", "user"]


def test_the_answer_is_the_last_one_after_the_reasoning():
    assert puzzles.answer("<answer>a</answer></think><answer>b</answer>") == "b"
    assert puzzles.answer("<answer>a</answer></think> nothing") == ""


@pytest.mark.parametrize("cls", METHODS)
def test_the_command_line_holds_every_argument(cls, monkeypatch):
    built = []
    monkeypatch.setattr(cls, "learn", lambda self: built.append(self))
    cls.cli(["--steps", "2", "--reasoning", "100", "trainer.save_freq=20"])
    (method,) = built
    assert (method.steps, method.reasoning) == (2, 100)
    assert method.verl == ["trainer.save_freq=20"]


def test_a_saved_setup_gives_defaults_the_command_line_overrides(
    monkeypatch, tmp_path
):
    built = []
    monkeypatch.setattr(CoevolvingJudge, "learn", lambda s: built.append(s))
    setup = tmp_path / "setup.yaml"
    setup.write_text("steps: 7\ngpus: 2\nverl:\n  - trainer.save_freq=5\n")
    CoevolvingJudge.cli(
        ["--config", str(setup), "--steps", "3", "trainer.save_freq=9"]
    )
    (method,) = built
    assert (method.steps, method.gpus) == (3, 2)
    assert method.verl == ["trainer.save_freq=5", "trainer.save_freq=9"]
    assert "config" not in method.kwargs
    setup.write_text("stepz: 7\n")
    with pytest.raises(SystemExit):
        CoevolvingJudge.cli(["--config", str(setup)])


def test_every_saved_setup_is_one_the_methods_take(monkeypatch):
    for setup in sorted((base_method.ROOT / "configs").glob("*.yaml")):
        for cls in METHODS:
            monkeypatch.setattr(cls, "learn", lambda self: None)
            cls.cli(["--config", str(setup)])


def test_a_method_is_rebuilt_from_its_spec(monkeypatch, tmp_path):
    method = CoevolvingJudge(freeze_at=7, data=str(tmp_path))
    monkeypatch.setenv(base_method.METHOD, json.dumps(method.spec()))
    base_method.running.cache_clear()
    again = base_method.running()
    assert type(again) is CoevolvingJudge and again.freeze_at == 7
    assert again.data == tmp_path.resolve()
    base_method.running.cache_clear()


@pytest.mark.parametrize("cls", METHODS)
def test_each_method_composes_a_verl_config(cls, tmp_path):
    (tmp_path / "data").mkdir()
    method = cls(data=str(tmp_path / "data"), runs=str(tmp_path), steps=3)
    config = method.config()
    assert config.trainer.total_training_steps == 3
    assert config.data.max_response_length == method.response_length
    assert config.actor_rollout_ref.rollout.agent.default_agent_loop == "oi"
    for override in cls.verl_overrides:
        key, value = override.split("=")
        node = config
        for part in key.split("."):
            node = node[part]
        assert str(node) == value


def test_the_response_length_fits_two_capped_answers_and_the_request():
    method = BaseMethod(reasoning=2048, answering=256)
    assert method.response_length == 2 * (2048 + 256 + 16) + 256


WORDS = {
    1: "PUZZLE",
    2: "think ",
    3: "</think>",
    6: "<|im_end|>",
    7: "ASK",
    8: "<think>",
}


class Loop:
    """A loop over a fake policy: it thinks then answers right, then is
    still thinking at the cap and answers wrong.
    """

    response_length = 600

    def __init__(self, right: str):
        words = {
            **WORDS,
            4: f"<answer>{right}</answer>",
            5: "<answer>x</answer>",
        }
        index = {v: k for k, v in words.items()}
        self.tokenizer = SimpleNamespace(
            decode=lambda ids: "".join(words[i] for i in ids),
            convert_tokens_to_ids=lambda tokens: [index[t] for t in tokens],
            encode=lambda text, add_special_tokens: [3],
        )
        self.turns = iter([[2, 3], [4, 6], [2, 2], [5, 6]])
        self.versions = {}
        self.asked = []

    async def ct_build_initial_tokens(self, messages):
        return [1]

    async def write(self, ids, sampling_params, tokens):
        stop = sampling_params.get("stop_token_ids")
        self.asked.append((list(ids), tokens, stop))
        return SimpleNamespace(token_ids=next(self.turns), log_probs=None)

    async def told(self, ids, mask, logprobs, messages, text):
        assert text == pairs.SNEAKY
        return ids + [7], mask + [0], logprobs

    answer = MethodLoop.answer
    output = MethodLoop.output


class Judge(BaseMethod):
    """A judge that is sure the honest answer alone is right."""

    async def judge(self, loop, pair):
        self.shown = pair
        return [-9.0, math.log(0.5), -9.0, math.log(0.5)], {"seen": 1.0}


def test_the_game_pairs_two_capped_answers_and_pays_the_policy(
    monkeypatch, tmp_path
):
    log = tmp_path / "episodes.jsonl"
    monkeypatch.setenv(progress.EPISODES, str(log))
    task = puzzles.tasks(puzzles.PUZZLE, 4, seed=0)[0]
    row = puzzles.row(task, 0)
    row["raw_prompt"] = row["prompt"]
    loop = Loop(task["entry"]["answer"])
    method = Judge(reasoning=50, answering=10)
    output = asyncio.run(method.rollout(loop, {}, row))
    assert output.prompt_ids == [1]
    assert output.response_ids == [2, 3, 4, 6, 7, 2, 2, 3, 5, 6]
    assert output.response_mask == [1, 1, 1, 1, 0, 1, 1, 0, 1, 1]
    budgets = [(tokens, stop is not None) for _, tokens, stop in loop.asked]
    assert budgets == [(50, True), (10, False), (50, True), (10, False)]
    pair = method.shown
    assert pair.honest_correct and not pair.sneaky_correct
    assert "ASK" not in pair.honest + pair.sneaky
    assert (pair.honest_ids, pair.sneaky_ids) == ([2, 3, 4, 6], [2, 2, 3, 5, 6])
    assert output.reward_score == pytest.approx(1 + math.log(2))
    (record,) = [json.loads(line) for line in log.read_text().splitlines()]
    assert record["metrics"] == {
        "honest_correct": 1.0,
        "sneaky_correct": 0.0,
        "judge_accuracy": 1.0,
        "judge_logloss": pytest.approx(math.log(2)),
        "seen": 1.0,
    }
    assert (record["task"], record["output_tokens"]) == (0, 8)
    assert list(record["seconds"]) == ["honest", "sneaky", "judge"]
    timing = progress.report({}, [record])["time_and_memory"]
    assert list(timing) == ["honest_seconds", "sneaky_seconds", "judge_seconds"]
    kinds = [event[0] for event in record["after"]]
    assert kinds == ["good", "method", ""]
    assert "→ only honest right: 0.50" in record["after"][1][2]


def test_the_report_groups_pairs_and_renders_a_page(tmp_path):
    records = [
        {
            "task": task,
            "reward": reward,
            "metrics": {"judge_accuracy": accuracy},
            "output_tokens": 10,
            "reasoning_tokens": 4,
            "input_tokens": 20,
            "transcript": "<|im_start|>user\npuzzle<|im_end|>"
            "<|im_start|>assistant\n<think>hm</think>answer<|im_end|>",
            "after": [("good", "Grade", "honest right", "")],
        }
        for task, reward, accuracy in [
            (0, 1.0, 1.0),
            (0, 0.0, 0.0),
            (1, 0.5, 1.0),
        ]
    ]
    groups = progress.report({"actor/kl_loss": 0.1}, records)
    assert list(groups) == ["puzzles", "training"]
    assert groups["puzzles"]["judge_accuracy"] == pytest.approx(2 / 3)
    assert groups["training"] == {"kl": 0.1, "groups_with_signal": 0.5}
    page = progress.pages(records, 3)["puzzles/pair"]
    assert "Puzzle" in page and "Policy" in page and "honest right" in page
    assert progress.lines(3, groups)[0].startswith("step    3  puzzles")


def test_the_question_shuffles_and_unshuffles_back():
    pair = pairs.Pair("task", "HONEST", "SNEAKY", True, False, [], [], [])
    orders = set()
    for _ in range(50):
        text, honest_first = pairs.question(pair)
        assert (text.index("HONEST") < text.index("SNEAKY")) == honest_first
        shown = [0.0, 1.0, 2.0, 3.0]
        back = pairs.unshuffled(shown, honest_first)
        assert back == (shown if honest_first else [0.0, 2.0, 1.0, 3.0])
        orders.add(honest_first)
    assert orders == {True, False}
    assert pairs.ANSWERS[pair.truth] == (1, 0) and pair.only_honest_correct == 1
