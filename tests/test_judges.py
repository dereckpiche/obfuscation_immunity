import asyncio
import math
from types import SimpleNamespace

import pytest
import torch

from obfuscation_immunity import ActorJudge, CoevolvingJudge
from obfuscation_immunity.coevolving_judge import coevolving_judge
from obfuscation_immunity.common import pairs
from obfuscation_immunity.common.pairs import Pair

CLOSE, LETTER_IDS = 99, [10, 11, 12, 13]


def pair(honest=True, sneaky=False):
    return Pair("task", "HONEST", "SNEAKY", honest, sneaky, [], [], [])


class Tokenizer:
    """Just enough of a tokenizer for a judge's question."""

    def apply_chat_template(self, *args, **kwargs):
        return "question"

    def encode(self, text, add_special_tokens):
        return [1] if text == "question" else [7] * text.count("\n")

    def convert_tokens_to_ids(self, tokens):
        return CLOSE if tokens == "</think>" else LETTER_IDS


class Loop:
    """A loop whose vLLM thinks `thought` and remembers what it was asked."""

    def __init__(self, thought=(5, CLOSE)):
        self.tokenizer = Tokenizer()
        self.asked = []

        async def generate(request_id, prompt_ids, sampling_params):
            self.asked.append((list(prompt_ids), dict(sampling_params)))
            return SimpleNamespace(token_ids=list(thought))

        self.server_manager = SimpleNamespace(generate=generate)


def test_the_judge_thinks_then_its_thinking_is_closed():
    loop = Loop()
    closed, length = asyncio.run(pairs.reasoned(loop, [1], 8, adapter="a"))
    assert (closed, length) == ([1, 5, CLOSE, 7, 7], 2)
    ((_, params),) = loop.asked
    assert params["max_tokens"] == 8 and params["adapter"] == "a"
    assert params["stop_token_ids"] == [CLOSE]
    still, length = asyncio.run(pairs.reasoned(Loop((5, 5)), [1], 8))
    assert (still, length) == ([1, 5, 5, 7, 7, 7], 2)


@pytest.mark.parametrize("scale", [1.0, 1e-300])
def test_the_actor_judge_thinks_renormalises_and_maps_back_to_the_pair(
    monkeypatch, scale
):
    method = ActorJudge(judging=8)
    shown = dict(zip(LETTER_IDS, [0.1, 0.2, 0.3, 0.4]))
    after = []

    async def ask(loop, ids, letter):
        after.append(ids)
        return math.log(scale * shown[letter])

    method.ask = ask
    monkeypatch.setattr(pairs.random, "random", lambda: 0.9)
    logp, numbers = asyncio.run(method.judge(Loop(), pair()))
    assert numbers == {
        "judge_answer_mass": pytest.approx(scale),
        "judge_thinking": 2,
    }
    assert [math.exp(v) for v in logp] == pytest.approx([0.1, 0.3, 0.2, 0.4])
    assert after[0] == [1, 5, CLOSE, 7, 7]


IDS, START, LENGTH, LETTERS = [3, 4, 5, 6, 7, 8], 2, 3, [10, 11, 12, 13]


@pytest.fixture
def method(monkeypatch, tmp_path):
    monkeypatch.setenv("OI_RUN_DIR", str(tmp_path / "runs" / "run"))
    monkeypatch.setattr(coevolving_judge, "PSI", {})
    monkeypatch.setattr(
        coevolving_judge.tempfile, "gettempdir", lambda: str(tmp_path / "tmp")
    )
    (tmp_path / "runs" / "run").mkdir(parents=True)
    return CoevolvingJudge(freeze_at=1, psi_rank=4, psi_lr=0.1)


def psi(model):
    return {
        n: p.detach().clone()
        for n, p in coevolving_judge.parameters(model).items()
    }


def read(method, model, truth=1, length=LENGTH):
    return method.read(model, IDS, START, length, LETTERS, truth)


def test_psi_starts_as_the_actor_judging_itself(actor_model, method):
    model = actor_model()
    with torch.no_grad():
        logits = model(input_ids=torch.tensor([IDS])).logits[0, -1, LETTERS]
    alone = torch.log_softmax(logits.float(), dim=-1).tolist()
    assert read(method, model) == pytest.approx(alone, abs=1e-6)


def test_judging_keeps_psi_gradient_aside_and_leaves_pi_as_it_was(
    actor_model, method
):
    model = actor_model()
    pi = {
        n: p.detach().clone()
        for n, p in model.named_parameters()
        if ".default." in n
    }
    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    read(method, model)
    assert model.active_adapters == ["default"]
    assert {
        n for n, p in model.named_parameters() if p.requires_grad
    } == trainable
    assert all(p.grad is None for p in model.parameters())
    assert any(torch.any(g) for g in coevolving_judge.PSI["grads"].values())
    assert coevolving_judge.PSI["pairs"].item() == 1
    now = dict(model.named_parameters())
    assert all(torch.equal(pi[n], now[n]) for n in pi)


def test_the_thinking_learns_once_the_verdict_leaves_the_baseline(
    actor_model, method
):
    model = actor_model()
    first = read(method, model, truth=1)[1]
    assert coevolving_judge.PSI["baseline"] == pytest.approx(first)
    kept = dict(coevolving_judge.PSI)
    grads = {}
    for length in (0, LENGTH):
        coevolving_judge.PSI.update(
            kept,
            grads={n: torch.zeros_like(g) for n, g in kept["grads"].items()},
        )
        second = read(method, model, truth=2, length=length)[2]
        grads[length] = coevolving_judge.PSI["grads"]
    assert any(
        not torch.allclose(grads[0][n], grads[LENGTH][n]) for n in grads[0]
    )
    assert coevolving_judge.PSI["baseline"] == pytest.approx(
        first + 0.1 * (second - first)
    )


def test_psi_steps_until_k_then_freezes_and_is_read_back(
    actor_model, method, tmp_path
):
    model = actor_model()
    method.update(model, 0)
    zero = psi(model)
    read(method, model, truth=2)
    method.update(model, 1)
    moved = psi(model)
    assert any(not torch.equal(moved[n], zero[n]) for n in zero)
    assert (tmp_path / "runs" / "run" / "psi.safetensors").exists()
    assert coevolving_judge.PSI["frozen"]
    read(method, model, truth=2)
    assert coevolving_judge.PSI["pairs"].item() == 0
    method.update(model, 2)
    assert all(torch.equal(moved[n], p) for n, p in psi(model).items())
    adapters = sorted(p.name for p in (tmp_path / "tmp" / "run").iterdir())
    assert adapters == ["judge-1", "judge-2"]
    coevolving_judge.PSI.clear()
    again = actor_model(1)
    method.update(again, 2)
    assert all(torch.equal(moved[n], p) for n, p in psi(again).items())


def test_the_judges_adapter_is_pi_and_psi_stacked(actor_model, method):
    model = actor_model()
    method.update(model, 0)
    read(method, model, truth=2)
    method.update(model, 1)
    path = coevolving_judge.PSI["adapter"]
    assert path.endswith("judge-1")
    model.load_adapter(path, adapter_name="judge")
    inputs = torch.tensor([IDS])
    with torch.no_grad():
        model.base_model.set_adapter(["default", "psi"])
        both = model(input_ids=inputs).logits
        model.base_model.set_adapter("default")
        alone = model(input_ids=inputs).logits
        model.base_model.set_adapter("judge")
        stacked = model(input_ids=inputs).logits
    assert torch.allclose(both, stacked, atol=1e-2)
    assert not torch.allclose(both, alone, atol=1e-2)


def test_the_coevolving_judge_thinks_with_its_adapter_and_reads_in_order(
    method, monkeypatch
):
    seen = {}

    async def on_a_worker(fn, *args):
        if fn is coevolving_judge.current:
            return "/judge-3"
        seen["read"] = args
        return [0.0, 1.0, 2.0, 3.0]

    monkeypatch.setattr(coevolving_judge.workers, "on_a_worker", on_a_worker)
    monkeypatch.setattr(pairs.random, "random", lambda: 0.9)
    loop = Loop()
    logp, numbers = asyncio.run(method.judge(loop, pair(True, False)))
    ((_, params),) = loop.asked
    assert params["adapter"] == "/judge-3"
    assert seen["read"] == ([1, 5, CLOSE, 7, 7], 1, 2, LETTER_IDS, 2)
    assert logp == [0.0, 2.0, 1.0, 3.0]
    assert numbers == {"judge_thinking": 2}


def test_the_head_reads_only_the_last_positions(actor_model):
    from obfuscation_immunity.common.workers import last_logits

    model = actor_model()
    inputs = torch.tensor([IDS])
    with torch.no_grad():
        full = model(input_ids=inputs).logits
        with last_logits(model, 2):
            last = model(input_ids=inputs).logits
    assert last.shape[1] == 2
    assert torch.allclose(last, full[:, -2:])


def test_the_thinking_log_probability_is_exact_in_chunks():
    torch.manual_seed(0)
    logits = torch.randn(300, 50, dtype=torch.bfloat16, requires_grad=True)
    tokens = torch.randint(0, 50, (300,))
    chunked = coevolving_judge.logprob(logits, tokens)
    whole = torch.log_softmax(logits.float(), -1).gather(-1, tokens[:, None])
    assert chunked.item() == pytest.approx(whole.sum().item(), rel=1e-5)
    (grad,) = torch.autograd.grad(chunked, logits)
    (expected,) = torch.autograd.grad(whole.sum(), logits)
    assert torch.allclose(grad.float(), expected.float())
