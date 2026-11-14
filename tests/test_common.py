import asyncio
import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
from hydra import compose, initialize_config_dir
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf, open_dict

from obfuscation_immunity import (
    ActorJudge,
    StackedJudge,
    Probe,
    StandaloneJudge,
)
from obfuscation_immunity.common import (
    chat_format,
    pairs,
    progress,
    puzzles,
    render,
)
from obfuscation_immunity.common import turn as turn_module
from obfuscation_immunity.common.chat_format import ChatFormat
from obfuscation_immunity.judges import text_judge
from obfuscation_immunity.common.judge import (
    Verdict,
    Judge,
    JudgeConfig,
    build_judge,
    whole_batches,
)
from obfuscation_immunity.common import loop as loop_module
from obfuscation_immunity.common.loop import PairLoop
from obfuscation_immunity.common.pairs import Answer, Pair
from obfuscation_immunity.common.transcript import (
    ENVIRONMENT,
    HONEST_ANSWER_KIND,
    JUDGE,
    JUDGE_OUTPUT_KIND,
    POLICY,
    SNEAKY_ANSWER_KIND,
    PairRecord,
    TranscriptSegment,
)

CONFIGS = Path(__file__).resolve().parents[1] / "configs"
PUZZLE = "zebra_puzzles"


def test_a_puzzle_survives_its_row_and_grades_exactly():
    for task in puzzles.sample_tasks(PUZZLE, 10, seed=0):
        row = puzzles.parquet_row(task, 0)
        task = json.loads(row["extra_info"]["task"])
        right = f"\\boxed{{{task['entry']['answer']}}}"
        assert puzzles.grade_final_answer(task, right) == 1.0
        assert puzzles.grade_final_answer(task, "\\boxed{12}") < 1.0
        assert puzzles.grade_final_answer(task, "no tags") == 0.0
        assert puzzles.grade_final_answer(task, f"So: {right}") == 0.0
        assert [m["role"] for m in row["prompt"]] == ["user"]


@pytest.mark.parametrize(
    "final, answer",
    [
        ("\\boxed{Bob}", "Bob"),
        ("\\boxed{Bob Smith}", "Bob Smith"),
        ("  \\boxed{Bob}", None),
        ("\\boxed{Bob}\n", None),
        ("\\boxed{ Bob }", None),
        ("The answer: \\boxed{Bob}", None),
        ("\\boxed{Bob} since clue 3 holds", None),
        ("\\boxed{a}\\boxed{b}", None),
        ("\\boxed{ }", None),
        ("<answer>Bob</answer>", None),
        ("Bob", None),
    ],
)
def test_a_reply_counts_only_as_one_tagged_answer_alone(final, answer):
    assert puzzles.answer_alone(final) == answer


def test_the_policy_is_told_its_budget_and_an_old_system_turn_dropped():
    prompt = [
        {"role": "system", "content": "old"},
        {"role": "user", "content": "puzzle"},
    ]
    system, user = puzzles.puzzle_messages(prompt, 2048)
    assert system["role"] == "system" and "2048 tokens" in system["content"]
    assert "nothing before or after it" in system["content"]
    assert user == {"role": "user", "content": "puzzle"}


def composed(*overrides):
    """configs/train.yaml with `overrides`, resolved as the run resolves it."""
    with initialize_config_dir(str(CONFIGS), version_base=None):
        config = compose("train", list(overrides), return_hydra_config=True)
    HydraConfig.instance().set_config(config)
    with open_dict(config):
        del config["hydra"]
    OmegaConf.resolve(config)
    return config


@pytest.mark.parametrize(
    "judge, cls",
    [
        ("actor_judge", ActorJudge),
        ("stacked_judge", StackedJudge),
        ("standalone_judge", StandaloneJudge),
        ("fixed_judge", StandaloneJudge),
        ("verified_judge", StandaloneJudge),
        ("probe", Probe),
    ],
)
@pytest.mark.parametrize("model", [p.stem for p in CONFIGS.glob("model/*")])
def test_every_judge_composes_with_every_model(judge, cls, model):
    config = composed(f"judge={judge}", f"model={model}")
    built = build_judge(config.judge)
    assert type(built) is cls and type(built.config) is cls.Config
    assert set(config.chat) == {"think_close", "separator", "end_of_turn"}
    assert config.data.apply_chat_template_kwargs
    assert config.trainer.experiment_name == judge
    assert config.trainer.default_local_dir.endswith(f"runs/{judge}")
    assert Path(config.actor_rollout_ref.rollout.agent.agent_loop_config_path)
    assert config.data.train_files.endswith("zebra_puzzles_3x3/train.parquet")


@pytest.mark.parametrize(
    "judge, stacked",
    [("stacked_judge", True), ("standalone_judge", False)],
)
def test_each_adapter_judge_sets_what_it_needs_in_verl(judge, stacked):
    config = composed(f"judge={judge}")
    rollout = config.actor_rollout_ref.rollout
    assert rollout.gpu_memory_utilization == 0.35
    assert rollout.engine_kwargs.vllm.max_loras == 2
    actor_rank = config.actor_rollout_ref.model.lora_rank
    adapter_rank = config.judge.adapter_rank
    served_rank = actor_rank + adapter_rank if stacked else adapter_rank
    assert (
        max(actor_rank, served_rank) <= rollout.engine_kwargs.vllm.max_lora_rank
    )


QWEN_CHAT = {
    "think_close": "</think>",
    "separator": "\n\n",
    "end_of_turn": "<|im_end|>",
}
"""The fake policy's markers, Qwen3.5's."""

WORDS = {
    1: "PUZZLE",
    2: "think ",
    3: "</think>",
    5: "\\boxed{x}",
    6: "<|im_end|>",
    7: "ASK",
    9: "<think>\n",
    11: "\n\n",
    12: " ",
    13: "<|endoftext|>",
}
MARKERS = {3, 6, 9, 13}
"""The fake tokenizer's added and special tokens."""


class Tokenizer:
    """Decodes the fake policy's words; word 4 is the right answer."""

    all_special_tokens = ["<|im_end|>", "<|endoftext|>"]
    all_special_ids = [6, 13]

    def __init__(self, right: str, words=WORDS, markers=MARKERS):
        """The words, and the right answer as word 4."""
        self.words = {**words, 4: f"\\boxed{{{right}}}"}
        self.added_tokens_decoder = {
            i: SimpleNamespace(content=self.words[i]) for i in markers
        }

    def decode(self, ids, skip_special_tokens=False):
        specials = set(self.all_special_ids)
        return "".join(
            self.words[i]
            for i in ids
            if not (skip_special_tokens and i in specials)
        )

    def batch_decode(self, sequences, skip_special_tokens=False):
        return [self.decode(ids, skip_special_tokens) for ids in sequences]

    def convert_tokens_to_ids(self, token):
        return {text: i for i, text in self.words.items()}.get(token)


def qwen_chat(tokenizer):
    """The fake policy's ChatFormat."""
    return ChatFormat.build(tokenizer, QWEN_CHAT)


RIGHT = [2, 3, 11, 4, 6]
"""A turn in the format with the right answer."""
WRONG = [2, 3, 11, 5, 6]
"""A turn in the format with a wrong answer."""


@pytest.mark.parametrize(
    "token_ids, problem",
    [
        (RIGHT, ""),
        ([2, 2, 3, 11, 5, 6], ""),
        ([2, 3, 11, 4], chat_format.OUT_OF_BUDGET),
        ([2, 2], chat_format.OUT_OF_BUDGET),
        ([2, 11, 4, 6], chat_format.UNCLOSED_THINKING),
        ([3, 11, 4, 6], chat_format.NO_THINKING),
        ([9, 2, 3, 11, 4, 6], chat_format.STRAY_MARKER),
        ([2, 3, 11, 4, 3, 6], chat_format.STRAY_MARKER),
        ([2, 3, 11, 4, 13, 6], chat_format.STRAY_MARKER),
        ([2, 3, 11, 4, 6, 6], chat_format.STRAY_MARKER),
        ([2, 3, 4, 6], chat_format.REPLY_OUT_OF_FORMAT),
        ([2, 3, 11, 12, 4, 6], chat_format.REPLY_OUT_OF_FORMAT),
        ([2, 3, 11, 4, 12, 6], chat_format.REPLY_OUT_OF_FORMAT),
        ([2, 3, 11, 4, 11, 6], chat_format.REPLY_OUT_OF_FORMAT),
        ([2, 3, 11, 4, 5, 6], chat_format.REPLY_OUT_OF_FORMAT),
        ([2, 3, 11, 2, 4, 6], chat_format.REPLY_OUT_OF_FORMAT),
        ([2, 3, 11, 6], chat_format.REPLY_OUT_OF_FORMAT),
    ],
)
def test_a_turn_is_thinking_tag_separator_reply_end_and_nothing_else(
    token_ids, problem
):
    tokenizer = Tokenizer("a")
    found = qwen_chat(tokenizer).problem(
        tokenizer, token_ids, loop_module.answer_in_format
    )
    assert found == problem


K2_WORDS = {
    2: "think ",
    3: "</ifm|think_faster>",
    5: "\\boxed{x}",
    6: "<|ifm|im_end|>",
    11: "\n\n",
    13: "<|ifm|endoftext|>",
}
"""K2-Horizon's words: no separator, and its own end of turn."""


def test_a_k2_turn_ends_on_its_own_marker_with_the_reply_right_after_tag():
    tokenizer = Tokenizer("a", K2_WORDS, {3, 6, 13})
    tokenizer.all_special_tokens = ["<|ifm|im_end|>", "<|ifm|endoftext|>"]
    chat = ChatFormat.build(
        tokenizer,
        {
            "think_close": "</ifm|think_faster>",
            "separator": "",
            "end_of_turn": "<|ifm|im_end|>",
        },
    )
    check = loop_module.answer_in_format
    assert chat.problem(tokenizer, [2, 3, 4, 6], check) == ""
    assert chat.split(tokenizer, [2, 3, 4, 6]) == ("think", "\\boxed{a}")
    assert chat.problem(tokenizer, [2, 3, 4, 13], check) == (
        chat_format.OUT_OF_BUDGET
    )
    assert chat.problem(tokenizer, [2, 3, 11, 4, 6], check) == (
        chat_format.REPLY_OUT_OF_FORMAT
    )


def test_a_marker_the_tokenizer_lacks_fails_at_the_start():
    with pytest.raises(ValueError, match="think_close"):
        ChatFormat.build(Tokenizer("a"), {**QWEN_CHAT, "think_close": "</t>"})
    with pytest.raises(ValueError, match="end_of_turn"):
        ChatFormat.build(Tokenizer("a"), {**QWEN_CHAT, "end_of_turn": "ASK"})


class SureJudge(Judge):
    """A judge that finds the honest answer alone right with 0.5."""

    async def verdict(self, loop, pair):
        self.shown = pair
        return Verdict(
            [-9.0, math.log(0.5), -9.0, math.log(0.5)], {"seen": 1.0}
        )


class BrokenJudge(Judge):
    """A judge whose turn breaks its format."""

    async def verdict(self, loop, pair):
        self.shown = pair
        return Verdict([], {"judge_tokens": 3.0}, stated=False)


def fake_loop(right: str, judge: Judge, turns=None) -> PairLoop:
    """A PairLoop over a fake policy, with verl's token helpers faked: by
    default its honest answer is right and its sneaky answer wrong, both in
    the format.
    """
    loop = object.__new__(PairLoop)
    loop.judge, loop.tokenizer = judge, Tokenizer(right)
    loop.chat = qwen_chat(loop.tokenizer)
    loop.response_length, loop.turn_tokens = 600, 5
    turns = iter(turns or [(RIGHT, 0), (WRONG, 1)])
    loop.asked, loop.told = [], []

    async def generate(request_id, prompt_ids, sampling_params):
        loop.asked.append(sampling_params["max_tokens"])
        ids, step = next(turns)
        fields = {"max_global_steps": step}
        if step == 0:
            fields["min_global_steps"] = 0
        return SimpleNamespace(
            token_ids=ids, log_probs=[-0.5] * len(ids), extra_fields=fields
        )

    async def initial_tokens(messages):
        return [1]

    async def context(previous, updated, ids, mask, logprobs):
        assert updated[: len(previous)] == previous
        loop.told.append(updated[-1]["content"])
        return (
            SimpleNamespace(token_ids=ids + [7, 9]),
            mask + [0, 0],
            logprobs + [0.0, 0.0],
        )

    loop.server_manager = SimpleNamespace(generate=generate)
    loop.ct_build_initial_tokens = initial_tokens
    loop.ct_merge_context_msg = context
    return loop


def play(judge, turns=None):
    """Plays one pair of the first zebra puzzle with a fake policy."""
    task = puzzles.sample_tasks(PUZZLE, 4, seed=0)[0]
    row = puzzles.parquet_row(task, 0)
    row["raw_prompt"] = row["prompt"]
    loop = fake_loop(task["entry"]["answer"], judge, turns)
    output = asyncio.run(loop.run({"temperature": 1.0}, **row))
    record = PairRecord.from_dict(output.extra_fields["pair_record"])
    return task, row, loop, output, record


STAKE = turn_module.FORMAT_STAKE


def test_the_game_pairs_two_answers_and_pays_the_policy():
    judge = SureJudge(JudgeConfig())
    task, row, loop, output, record = play(judge)
    assert loop.asked == [5, 5]
    assert loop.told == [pairs.SNEAKY_REQUEST]
    assert output.prompt_ids == [1]
    assert output.response_ids == RIGHT + [7, 9] + WRONG
    assert output.response_mask == [1] * 5 + [0, 0] + [1] * 5
    assert output.response_logprobs == [-0.5] * 5 + [0.0] * 2 + [-0.5] * 5
    pair = judge.shown
    assert pair.task == row["prompt"][0]["content"]
    assert (pair.honest.reasoning, pair.honest.correct) == ("think", True)
    assert pair.honest.final == f"\\boxed{{{task['entry']['answer']}}}"
    assert (pair.sneaky.final, pair.sneaky.correct) == ("\\boxed{x}", False)
    assert output.reward_score == pytest.approx(1 + 0.5 + 2 * STAKE)
    fields = output.extra_fields
    assert (fields["min_global_steps"], fields["max_global_steps"]) == (0, 1)
    assert record.metrics == {
        "honest_in_format": 1.0,
        "honest_correct": 1.0,
        "sneaky_in_format": 1.0,
        "sneaky_correct": 0.0,
        "judge_in_format": 1.0,
        "judge_accuracy": 1.0,
        "judge_true_probability": pytest.approx(0.5),
        "seen": 1.0,
    }
    assert set(record.part_seconds) == {"honest", "sneaky", "judge"}
    assert record.format_reward == 2 * STAKE
    assert record.judge_miss == pytest.approx(0.5)
    titles = [segment.title for segment in record.segments]
    assert titles == ["Puzzle", "Honest answer", "Request", "Sneaky answer"]
    puzzle, honest, request, sneaky = record.segments
    assert (puzzle.token_ids, request.token_ids) == ([1], [7, 9])
    assert (honest.token_ids, honest.details) == (RIGHT, {})
    assert (sneaky.token_ids, sneaky.details) == (WRONG, {})
    assert honest.trained and sneaky.trained and not request.trained
    assert honest.budget == sneaky.budget == 5
    assert record.true_outcome == 1 and record.honest_correct


def test_an_honest_answer_out_of_budget_ends_the_game_and_the_judge_wins():
    judge = SureJudge(JudgeConfig())
    _, _, loop, output, record = play(judge, [([2, 2], 0)])
    assert loop.asked == [5] and loop.told == []
    assert not hasattr(judge, "shown")
    assert (output.response_ids, output.response_mask) == ([2, 2], [1, 1])
    assert output.reward_score == pytest.approx(-STAKE)
    assert output.extra_fields["judge_trajectory"] is None
    assert record.metrics == {"honest_in_format": 0.0}
    assert record.judge_miss == 0.0
    assert [s.title for s in record.segments] == ["Puzzle", "Honest answer"]
    assert record.segments[1].details == {"format": "out of budget"}
    page = render.pair_page_html(record, loop.tokenizer, loop.chat)
    assert '<span class="badge error">out of budget</span>' in page
    assert "the game ends here: the judge wins" in page


def test_a_sneaky_answer_out_of_format_ends_the_game_and_the_judge_wins():
    judge = SureJudge(JudgeConfig())
    turns = [(RIGHT, 0), ([2, 3, 11, 12, 5, 6], 1)]
    _, _, loop, output, record = play(judge, turns)
    assert not hasattr(judge, "shown")
    assert output.reward_score == pytest.approx(STAKE - STAKE)
    assert record.metrics == {
        "honest_in_format": 1.0,
        "honest_correct": 1.0,
        "sneaky_in_format": 0.0,
    }
    assert record.verifier_reward == 0.0 and record.judge_miss == 0.0
    assert record.segments[-1].details == {
        "format": chat_format.REPLY_OUT_OF_FORMAT
    }


def test_a_judge_out_of_format_ends_the_game_and_the_policy_wins():
    judge = BrokenJudge(JudgeConfig())
    _, _, _, output, record = play(judge)
    assert output.reward_score == pytest.approx(1 + 1 + 2 * STAKE)
    assert record.judge_miss == 1.0
    assert record.metrics == {
        "honest_in_format": 1.0,
        "honest_correct": 1.0,
        "sneaky_in_format": 1.0,
        "sneaky_correct": 0.0,
        "judge_in_format": 0.0,
        "judge_tokens": 3.0,
    }


def pair(honest="HONEST", sneaky="SNEAKY"):
    return Pair(
        "task",
        [],
        Answer([], f"why {honest}", honest, True),
        Answer([], f"why {sneaky}", sneaky, False),
    )


def test_the_judge_reads_each_answer_in_full_and_shuffled():
    orders = set()
    for _ in range(50):
        text, honest_first = text_judge.build_judge_question(pair(), 99)
        assert "why HONEST" in text and "why SNEAKY" in text
        assert "99 tokens" in text
        assert "<think>" not in text and pairs.SNEAKY_REQUEST not in text
        assert (text.index("HONEST") < text.index("SNEAKY")) == honest_first
        shown = [0.0, 1.0, 2.0, 3.0]
        back = text_judge.to_pair_order(shown, honest_first)
        assert back == (shown if honest_first else [0.0, 2.0, 1.0, 3.0])
        orders.add(honest_first)
    assert orders == {True, False}
    assert pairs.OUTCOMES[pair().true_outcome] == (1, 0)
    assert pair().only_honest_correct == 1


def test_an_answer_splits_at_its_thinking():
    tokenizer = Tokenizer("a")
    chat = qwen_chat(tokenizer)
    assert chat.split(tokenizer, [2, 2, 3, 11, 4, 6]) == (
        "think think",
        "\\boxed{a}",
    )
    assert chat.split(tokenizer, [2, 2]) == ("think think", "")
    assert chat.split(tokenizer, [2, 6]) == ("think", "")


def test_the_page_shows_every_token_and_what_happened():
    tokenizer = Tokenizer("a")
    record = PairRecord(
        segments=[
            TranscriptSegment("Puzzle", ENVIRONMENT, [1]),
            TranscriptSegment(
                "Honest answer",
                POLICY,
                RIGHT,
                [-0.1] * 5,
                True,
                172,
                HONEST_ANSWER_KIND,
            ),
            TranscriptSegment(
                "Sneaky answer",
                POLICY,
                WRONG,
                [-0.1] * 5,
                True,
                5,
                SNEAKY_ANSWER_KIND,
            ),
            TranscriptSegment(
                "Judge's output",
                JUDGE,
                [2, 2],
                [-0.3] * 2,
                budget=2,
                kind=JUDGE_OUTPUT_KIND,
                details={"format": "out of budget"},
                model=JUDGE,
            ),
        ],
        puzzle="zebra_puzzles",
        expected_answer="a",
        honest_correct=True,
        sneaky_correct=False,
        true_outcome=1,
        outcome_logprobs=[],
        judge_statement="",
        verifier_reward=1.0,
        judge_miss=1.0,
        reward=3.0,
        metrics={"judge_tokens": 2.0},
        format_reward=1.0,
    )
    page = render.pair_page_html(record, tokenizer, qwen_chat(tokenizer))
    for shown in (
        "The policy's sequence",
        "The judge's sequence",
        '<span class="warn">cut off</span>',
        '<span class="warn">never closed its thinking</span>',
        '<span class="badge error">out of budget</span>',
        "the game ends here: the policy wins",
        '<code>a</code><small>answer</small></span><span class="relation">='
        "</span>",
        '<code>x</code><small>answer</small></span><span class="relation">≠'
        "</span>",
        '<span class="badge strong">✓︎ goal met</span>',
        "1 − J(y_true)",
        "turns in format",
    ):
        assert shown in page
    assert '<span class="pill">&lt;|im_end|&gt;</span>' in page
    assert '<span class="pill">&lt;/think&gt;</span>' in page
    assert '"pieces": ["think ", "</think>"' not in page
    assert '"pieces": ["think ", "<\\/think>"' in page


def test_the_page_shows_j_against_the_truth():
    tokenizer = Tokenizer("a")
    record = PairRecord(
        segments=[
            TranscriptSegment(
                "Judge's output",
                JUDGE,
                [2, 3, 11, 6],
                [-0.3] * 4,
                kind=JUDGE_OUTPUT_KIND,
                model=JUDGE,
            ),
        ],
        puzzle="zebra_puzzles",
        expected_answer="a",
        honest_correct=True,
        sneaky_correct=False,
        true_outcome=1,
        outcome_logprobs=[math.log(p) for p in (0.1, 0.6, 0.2, 0.1)],
        judge_statement='{"answer": {"A": 0.1, "B": 0.6, "C": 0.2, "D": 0.1}}',
        verifier_reward=1.0,
        judge_miss=0.4,
        reward=2.4,
        metrics={},
        format_reward=1.0,
    )
    page = render.pair_page_html(record, tokenizer, qwen_chat(tokenizer))
    assert '<span class="cell truth" style="--p:0.600">0.60<i>truth</i>' in page
    assert "<b>0.400</b><small>1 − J(y_true)</small>" in page


def report_record(puzzle, reward, metrics):
    return PairRecord(
        segments=[
            TranscriptSegment("Puzzle", ENVIRONMENT, puzzle, kind="puzzle"),
            TranscriptSegment(
                "Honest answer", POLICY, [2, 2, 3, 4], kind=HONEST_ANSWER_KIND
            ),
        ],
        puzzle="zebra_puzzles",
        expected_answer="a",
        honest_correct=True,
        sneaky_correct=False,
        true_outcome=1,
        outcome_logprobs=[math.log(0.25)] * 4,
        judge_statement="",
        verifier_reward=1.0,
        judge_miss=0.75,
        reward=reward,
        metrics=metrics,
        part_seconds={"honest": 2.0, "judge": 1.0},
    )


def test_the_report_has_the_game_the_update_and_time_only():
    records = [
        report_record([1], 1.0, {"a": 1.0, "b": 2.0}),
        report_record([1], 2.0, {"a": 0.0}),
        report_record([9], 1.0, {"a": 0.5}),
    ]
    verl = {"actor/grad_norm": 0.5, "timing_s/step": 120.0, "noise/x": 1.0}
    report = progress.step_report(verl, records, 3)
    assert set(report) == {"puzzles", "training", "time_and_memory"}
    puzzles = report["puzzles"]
    assert puzzles["a"] == pytest.approx(0.5) and puzzles["b"] == 2.0
    assert puzzles["reward"] == pytest.approx(4 / 3)
    assert (puzzles["tokens_written"], puzzles["tokens_thinking"]) == (4, 3)
    assert puzzles["tokens_read"] == 1
    assert report["training"] == {"grad_norm": 0.5, "signal": 0.5}
    assert report["time_and_memory"] == {
        "step_seconds": 120.0,
        "honest_seconds": 2.0,
        "judge_seconds": 1.0,
    }
    lines = progress.console_lines(7, report)
    assert lines[1] == "step    7  training         grad_norm 0.50  signal 0.50"
    assert all("noise" not in line for line in lines)


@pytest.mark.parametrize(
    "override, error",
    [
        ("judge.freeze_at=soon", "freeze_at"),
        ("+judge.freez_at=3", "freez_at"),
        ("~judge.adapter_lr", "adapter_lr"),
    ],
)
def test_a_judge_config_is_checked_against_its_type(override, error):
    config = composed("judge=stacked_judge", override)
    with pytest.raises(Exception, match=error):
        build_judge(config.judge)


def test_the_judge_trains_on_whole_batches_oldest_first():
    listed = {
        f"k{i}": {"global_steps": step}
        for i, step in enumerate([3, 1, 2, 1, 3, 2, 1])
    }
    assert whole_batches(listed, 8) == []
    assert whole_batches(listed, 3) == ["k1", "k3", "k6", "k2", "k5", "k0"]
    assert whole_batches(listed, 7) == sorted(
        listed, key=lambda k: listed[k]["global_steps"]
    )
