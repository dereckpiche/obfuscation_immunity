import asyncio
import math
from types import SimpleNamespace

import pytest
import torch

from obfuscation_immunity import ActorJudge, StackedJudge, StandaloneJudge
from obfuscation_immunity.judges import adapter_judge, judge_adapter, text_judge
from obfuscation_immunity.judges.adapter_judge import AdapterJudgeConfig
from obfuscation_immunity.judges.judge_adapter import (
    ACTOR_ADAPTER_NAME,
    JUDGE_ADAPTER_NAME,
)
from obfuscation_immunity.common import chat_format
from obfuscation_immunity.common.chat_format import ChatFormat
from obfuscation_immunity.common.judge import JudgeConfig
from obfuscation_immunity.common.turn import FORMAT_STAKE
from obfuscation_immunity.common.pairs import Answer, Pair

CLOSE, EOS, OPEN, SPACE = 99, 98, 96, 95
WORDS = {
    1: "",
    5: "hm ",
    CLOSE: "</think>",
    EOS: "<|im_end|>",
    OPEN: "<think>\n",
    SPACE: " ",
    6: '{"answer": {"A": 0.1, "B": 0.2, "C": 0.3, "D": 0.4}}',
}
CHAT = {"think_close": "</think>", "separator": "", "end_of_turn": "<|im_end|>"}
"""The judges' markers: Qwen3.5's, the reply right after the tag."""


def stated(a, b, c, d):
    """The JSON reply stating J, its numbers written as given."""
    return f'{{"answer": {{"A": {a}, "B": {b}, "C": {c}, "D": {d}}}}}'


STATED = (5, CLOSE, 6, EOS)
"""A judge's turn that thinks, states J in the format and ends itself."""


def pair(honest=True, sneaky=False):
    return Pair(
        "task",
        [],
        Answer([], "why", "HONEST", honest),
        Answer([], "why", "SNEAKY", sneaky),
    )


class Tokenizer:
    """Just enough of a tokenizer for a judge's question and answer."""

    all_special_tokens = ["<|im_end|>"]
    all_special_ids = [EOS]
    added_tokens_decoder = {
        i: SimpleNamespace(content=WORDS[i]) for i in (CLOSE, EOS, OPEN)
    }

    def __init__(self):
        """Remembers the chat template's keyword arguments."""
        self.template_kwargs = {}

    def apply_chat_template(self, *args, **kwargs):
        self.template_kwargs = kwargs
        return "question"

    def encode(self, text, add_special_tokens):
        return [1]

    def convert_tokens_to_ids(self, token):
        return {text: i for i, text in WORDS.items()}.get(token)

    def decode(self, ids, skip_special_tokens=False):
        return "".join(
            WORDS[i]
            for i in ids
            if not (skip_special_tokens and i in self.all_special_ids)
        )


class Loop:
    """A loop whose vLLM writes `outputs` in turn and remembers what it was
    asked. By default the judge closes its thinking, states a distribution
    and ends its turn at once.
    """

    def __init__(self, outputs=(STATED,), step=3, budget=8):
        self.tokenizer = Tokenizer()
        self.chat = ChatFormat.build(self.tokenizer, CHAT)
        self.turn_tokens = budget
        self.apply_chat_template_kwargs = {"enable_thinking": True}
        self.rollout_config = SimpleNamespace(
            temperature=1.0, top_p=0.95, top_k=-1
        )
        self.weight_versions = {"max_global_steps": step}
        self.asked = []
        turns = iter(outputs)

        async def generate(request_id, prompt_ids, sampling_params):
            self.asked.append((list(prompt_ids), dict(sampling_params)))
            output = next(turns)
            return SimpleNamespace(
                token_ids=list(output),
                log_probs=[-0.25] * len(output),
                extra_fields={},
            )

        self.server_manager = SimpleNamespace(generate=generate)


@pytest.mark.parametrize(
    "text, distribution",
    [
        (stated(0.1, 0.2, 0.3, 0.4), [0.1, 0.2, 0.3, 0.4]),
        (stated("0.10", "0.60", "0.20", "0.10"), [0.1, 0.6, 0.2, 0.1]),
        ("  " + stated(0.1, 0.6, 0.2, 0.1), None),
        (stated(0.1, 0.6, 0.2, 0.1) + "\n", None),
        (stated(1, 0, 0, 0), [1.0, 0.0, 0.0, 0.0]),
        (stated(0.333, 0.333, 0.334, 0), [0.333, 0.333, 0.334, 0.0]),
        (stated(1, 1, 1, 1), None),
        (stated(0.3, 0.3, 0.3, 0.3), None),
        (stated(1.5, 0, 0, -0.5), None),
        (stated('"0.5"', 0.5, 0, 0), None),
        (stated("true", 0, 0, 0), None),
        ("```json\n" + stated(0.1, 0.2, 0.3, 0.4) + "\n```", None),
        (stated(0.2, 0.8, 0, 0) + " since B holds", None),
        ("I think " + stated(0.2, 0.8, 0, 0), None),
        ('{"answer": {"A": 0.5, "A": 0.5, "C": 0, "D": 0}}', None),
        ('{"answer": {"A": 0.5, "B": 0.5}}', None),
        ('{"answer": "B"}', None),
        ('{"answer": {"A": 1, "B": 0, "C": 0, "D": 0}, "why": "x"}', None),
        ("I cannot tell.", None),
    ],
)
def test_the_judge_must_reply_with_one_json_answer_summing_to_1(
    text, distribution
):
    found = text_judge.parse_stated_distribution(text)
    assert found == (distribution and pytest.approx(distribution))


@pytest.mark.parametrize("honest_first", [True, False])
def test_the_actor_judge_states_j_mapped_back_to_the_pair(
    monkeypatch, honest_first
):
    method = ActorJudge(JudgeConfig())
    monkeypatch.setattr(
        text_judge.random, "random", lambda: 0.1 if honest_first else 0.9
    )
    loop = Loop()
    verdict = asyncio.run(method.verdict(loop, pair()))
    ((prompt, params),) = loop.asked
    assert prompt == [1]
    assert params == {
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": -1,
        "logprobs": True,
        "max_tokens": 8,
    }
    assert loop.tokenizer.template_kwargs["enable_thinking"] is True
    order = [0.1, 0.2, 0.3, 0.4] if honest_first else [0.1, 0.3, 0.2, 0.4]
    assert verdict.stated
    assert [math.exp(v) for v in verdict.outcome_logprobs] == pytest.approx(
        order
    )
    assert verdict.statement == WORDS[6]
    question, output = verdict.segments
    first = "honest answer" if honest_first else "sneaky answer"
    assert question.details["attempt 1"] == first
    assert question.model == output.model == "judge"
    assert (output.token_ids, output.logprobs) == (list(STATED), [-0.25] * 4)
    assert output.budget == 8 and not output.trained
    assert output.details == {}
    assert verdict.metrics == {"judge_tokens": 4.0}


@pytest.mark.parametrize(
    "output, problem",
    [
        ((5, 5), chat_format.OUT_OF_BUDGET),
        ((5, EOS), chat_format.UNCLOSED_THINKING),
        ((5, CLOSE, 5, EOS), chat_format.REPLY_OUT_OF_FORMAT),
        ((5, CLOSE, SPACE, 6, EOS), chat_format.REPLY_OUT_OF_FORMAT),
        ((5, CLOSE, 6, SPACE, EOS), chat_format.REPLY_OUT_OF_FORMAT),
        ((5, OPEN, CLOSE, 6, EOS), chat_format.STRAY_MARKER),
    ],
)
def test_a_judge_out_of_format_has_one_try_and_states_nothing(output, problem):
    method = ActorJudge(JudgeConfig())
    loop = Loop((output,))
    verdict = asyncio.run(method.verdict(loop, pair()))
    assert len(loop.asked) == 1
    assert not verdict.stated and verdict.outcome_logprobs == []
    assert verdict.statement == ""
    _, turn = verdict.segments
    assert turn.details == {"format": problem}


IDS = [3, 4, 5, 6, 7, 8]


def config(tmp_path, freeze_at=5, verified=True):
    return AdapterJudgeConfig(
        verified=verified,
        freeze_at=freeze_at,
        adapter_rank=4,
        adapter_lr=0.1,
        run_dir=str(tmp_path / "run"),
    )


def with_judge_adapter(actor_model):
    model = actor_model()
    judge_adapter.add_judge_adapter(model, rank=4)
    return model


def trainable(model):
    return {n for n, p in model.named_parameters() if p.requires_grad}


def test_the_judge_adapter_starts_at_zero_off_and_frozen(actor_model):
    model = actor_model()
    with torch.no_grad():
        before = model(input_ids=torch.tensor([IDS])).logits
    judge_adapter.add_judge_adapter(model, rank=4)
    judge = judge_adapter.adapter_parameters(model, JUDGE_ADAPTER_NAME)
    actor = judge_adapter.adapter_parameters(model, ACTOR_ADAPTER_NAME)
    assert model.active_adapters == [ACTOR_ADAPTER_NAME]
    assert trainable(model) == set(actor)
    assert model.peft_config[JUDGE_ADAPTER_NAME].lora_alpha == 8
    model.base_model.set_adapter([ACTOR_ADAPTER_NAME, JUDGE_ADAPTER_NAME])
    with torch.no_grad():
        stacked = model(input_ids=torch.tensor([IDS])).logits
    assert torch.allclose(before, stacked)
    assert all(
        p.dtype == next(iter(actor.values())).dtype for p in judge.values()
    )


@pytest.mark.parametrize("stacked", [True, False])
def test_the_judges_turn_thinks_with_its_adapters_and_learns_only_its_own(
    actor_model, stacked
):
    model = with_judge_adapter(actor_model)
    adapters = (
        [ACTOR_ADAPTER_NAME, JUDGE_ADAPTER_NAME]
        if stacked
        else [JUDGE_ADAPTER_NAME]
    )
    judge_adapter.switch_adapters(model, adapters, JUDGE_ADAPTER_NAME)
    assert model.active_adapters == adapters
    assert trainable(model) == set(
        judge_adapter.adapter_parameters(model, JUDGE_ADAPTER_NAME)
    )
    judge_adapter.switch_adapters(
        model, [ACTOR_ADAPTER_NAME], ACTOR_ADAPTER_NAME
    )
    assert model.active_adapters == [ACTOR_ADAPTER_NAME]
    assert trainable(model) == set(
        judge_adapter.adapter_parameters(model, ACTOR_ADAPTER_NAME)
    )


@pytest.mark.parametrize("stacked", [True, False])
def test_the_served_lora_is_what_the_judge_thinks_with(
    actor_model, tmp_path, stacked
):
    from safetensors.torch import save_file

    model = with_judge_adapter(actor_model)
    for parameter in judge_adapter.adapter_parameters(
        model, JUDGE_ADAPTER_NAME
    ).values():
        torch.nn.init.normal_(parameter)
    judge_config = model.peft_config[JUDGE_ADAPTER_NAME]
    judge_weights = judge_adapter.gathered_adapter(model, JUDGE_ADAPTER_NAME)
    served_config, served_weights = judge_config, judge_weights
    adapters = [JUDGE_ADAPTER_NAME]
    if stacked:
        adapters = [ACTOR_ADAPTER_NAME, JUDGE_ADAPTER_NAME]
        served_config, served_weights = judge_adapter.stacked_lora(
            model.peft_config[ACTOR_ADAPTER_NAME],
            judge_adapter.gathered_adapter(model, ACTOR_ADAPTER_NAME),
            judge_config,
            judge_weights,
        )
    served_config.save_pretrained(str(tmp_path))
    save_file(served_weights, tmp_path / "adapter_model.safetensors")
    model.load_adapter(str(tmp_path), adapter_name="served")
    inputs = torch.tensor([IDS])
    with torch.no_grad():
        model.base_model.set_adapter(adapters)
        thinks = model(input_ids=inputs).logits
        model.base_model.set_adapter(ACTOR_ADAPTER_NAME)
        actor = model(input_ids=inputs).logits
        model.base_model.set_adapter("served")
        served = model(input_ids=inputs).logits
    assert torch.allclose(thinks, served, atol=1e-4)
    assert not torch.allclose(served, actor, atol=1e-2)


@pytest.mark.parametrize("judge_class", [StackedJudge, StandaloneJudge])
def test_an_adapter_judge_tells_verl_its_adapter(tmp_path, judge_class):
    judge = judge_class(config(tmp_path))
    settings = judge_adapter.JudgeAdapterSettings(
        **judge.judge_adapter_settings()
    )
    assert settings.rank == 4 and settings.lr == 0.1
    assert settings.stacked is (judge_class is StackedJudge)
    assert judge.actor_worker_class is judge_adapter.JudgeAdapterWorker


def test_an_adapter_judge_answers_with_its_lora_of_the_step_and_keeps_its_turn(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(text_judge.random, "random", lambda: 0.9)
    judge = StandaloneJudge(config(tmp_path))
    loop = Loop(step=3)
    verdict = asyncio.run(judge.verdict(loop, pair(True, False)))
    served = adapter_judge.served_adapter_dir(str(tmp_path / "run"), 3)
    assert served.name == "judge-3"
    assert all(params["lora_dir"] == str(served) for _, params in loop.asked)
    trajectory = verdict.judge_trajectory
    assert trajectory["prompt_ids"] == [1]
    assert trajectory["response_ids"] == list(STATED)
    assert trajectory["response_mask"] == [1] * 4
    assert trajectory["response_logprobs"] == [-0.25] * 4
    assert trajectory["score"] == pytest.approx(0.3 + FORMAT_STAKE)


def test_a_fixed_judge_is_paid_its_format_stake_alone(tmp_path):
    judge = StandaloneJudge(config(tmp_path, verified=False))
    verdict = asyncio.run(judge.verdict(Loop(), pair(True, False)))
    assert verdict.judge_trajectory["score"] == pytest.approx(FORMAT_STAKE)
    loop = Loop(((5, CLOSE, 5, EOS),))
    verdict = asyncio.run(judge.verdict(loop, pair(True, False)))
    assert verdict.judge_trajectory["score"] == pytest.approx(-FORMAT_STAKE)


def test_an_adapter_judge_out_of_format_scores_only_its_lost_stake(tmp_path):
    judge = StandaloneJudge(config(tmp_path))
    loop = Loop(((5, CLOSE, 5, EOS),))
    verdict = asyncio.run(judge.verdict(loop, pair(True, False)))
    trajectory = verdict.judge_trajectory
    assert trajectory["response_ids"] == [5, CLOSE, 5, EOS]
    assert trajectory["response_mask"] == [1] * 4
    assert trajectory["response_logprobs"] == [-0.25] * 4
    assert trajectory["score"] == pytest.approx(-FORMAT_STAKE)
    assert not verdict.stated


def test_a_frozen_adapter_judge_keeps_no_turn(tmp_path):
    judge = StackedJudge(config(tmp_path, freeze_at=3))
    verdict = asyncio.run(judge.verdict(Loop(step=3), pair()))
    assert verdict.judge_trajectory is None
    assert not any(segment.trained for segment in verdict.segments)
    never = StackedJudge(config(tmp_path, freeze_at=-1))
    verdict = asyncio.run(never.verdict(Loop(step=900), pair()))
    assert verdict.judge_trajectory is not None


class Workers:
    """verl's actor worker group, remembering what it was asked."""

    def __init__(self, calls):
        self.calls = calls

    def use_judge_adapter(self, judging):
        self.calls.append(("judging", judging))

    def write_served_adapter(self, directory, stale):
        self.calls.append(("write", directory, stale))


class Trainer:
    """PairTrainer as an adapter judge sees it."""

    def __init__(self, step):
        self.global_steps = step
        self.calls = []
        self.actor_rollout_wg = Workers(self.calls)

    def update_on_partition(self, partition_id):
        self.calls.append(("update", partition_id))
        return {"actor/pg_loss": 0.5}


@pytest.mark.parametrize(
    "freeze_at, step, learns", [(5, 5, True), (5, 6, False), (-1, 900, True)]
)
def test_verl_trains_the_judge_after_the_actor_until_k(
    tmp_path, freeze_at, step, learns
):
    judge = StackedJudge(config(tmp_path, freeze_at=freeze_at))
    trainer, metrics = Trainer(step), {}
    judge.after_actor_update(trainer, metrics)
    run_dir = str(tmp_path / "run")
    write = (
        "write",
        str(adapter_judge.served_adapter_dir(run_dir, step)),
        str(adapter_judge.served_adapter_dir(run_dir, step - 2)),
    )
    if learns:
        assert trainer.calls == [
            ("judging", True),
            ("update", "judge"),
            ("judging", False),
            write,
        ]
        assert metrics == {"judge/actor/pg_loss": 0.5}
    else:
        assert trainer.calls == [write] and metrics == {}


def test_the_head_reads_only_the_last_positions(actor_model):
    from obfuscation_immunity.common.training_workers import (
        logits_for_last_positions,
    )

    model = actor_model()
    inputs = torch.tensor([IDS])
    with torch.no_grad():
        full = model(input_ids=inputs).logits
        with logits_for_last_positions(model, 2):
            last = model(input_ids=inputs).logits
    assert last.shape[1] == 2
    assert torch.allclose(last, full[:, -2:])
