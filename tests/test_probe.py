import asyncio
import math

import pytest
import torch

from obfuscation_immunity import Probe
from obfuscation_immunity.common.pairs import Answer, Pair
from obfuscation_immunity.probe import fit, probe
from obfuscation_immunity.probe.probe import ProbeConfig

PROBE = ProbeConfig(probe="probe.pt")


def test_the_fit_separates_shifted_classes_and_round_trips(tmp_path):
    torch.manual_seed(0)
    shift = torch.zeros(16)
    shift[3] = 2.0
    x = torch.cat([torch.randn(200, 16) + shift, torch.randn(200, 16)])
    y = torch.cat([torch.ones(200), torch.zeros(200)]).long()
    weights = probe.ProbeWeights.fit(x, y, "m", 4, "sae")
    scores = torch.tensor([weights.score(v) for v in x])
    assert scores[:200].mean() > 0.5 > scores[200:].mean()
    assert fit.auroc(scores, y) > 0.9
    weights.save(tmp_path / "probe.pt")
    loaded = probe.ProbeWeights.load(str(tmp_path / "probe.pt"))
    assert loaded.score(x[0]) == pytest.approx(weights.score(x[0]))


def test_auroc_counts_ties_as_half():
    labels = torch.tensor([1, 0])
    assert fit.auroc(torch.tensor([0.9, 0.1]), labels) == 1.0
    assert fit.auroc(torch.tensor([0.5, 0.5]), labels) == 0.5


def test_topk_sae_keeps_the_k_largest_features():
    sae = object.__new__(probe.TopKSAE)
    sae.encoder_weight, sae.encoder_bias, sae.k = (
        torch.eye(4),
        torch.zeros(4),
        2,
    )
    out = sae(torch.tensor([[0.1, 3.0, -1.0, 2.0]]))
    assert out.tolist() == [[0.0, 3.0, 0.0, 2.0]]


def test_features_pool_the_masked_tokens_after_the_layer():
    from transformers import LlamaConfig, LlamaForCausalLM

    config = LlamaConfig(
        vocab_size=32,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=3,
        num_attention_heads=2,
        num_key_value_heads=2,
    )
    model = LlamaForCausalLM(config).eval()
    ids, mask = [3, 4, 5, 6], [0, 1, 1, 0]
    with torch.no_grad():
        h = model(torch.tensor([ids]), output_hidden_states=True)
        x = probe.pooled_sae_features(model, lambda h: h, 1, ids, mask)
    assert torch.allclose(x, h.hidden_states[2][0, 1:3].float().mean(0))


def logit(p):
    return math.log(p / (1 - p))


def test_the_probe_judges_each_answer_on_its_own(monkeypatch):
    read = []

    async def run_on_a_worker(fn, ids, mask):
        read.append((ids, mask))
        return logit(0.2) if ids[-1] == 4 else logit(0.9)

    monkeypatch.setattr(
        probe.training_workers, "call_on_a_training_worker", run_on_a_worker
    )
    pair = Pair(
        "task",
        [1, 2],
        Answer([3, 4], "", "h", True),
        Answer([5], "", "s", False),
    )
    verdict = asyncio.run(Probe(PROBE).verdict(None, pair))
    assert read == [([1, 2, 3, 4], [0, 0, 1, 1]), ([1, 2, 5], [0, 0, 1])]
    expected = [0.8 * 0.1, 0.8 * 0.9, 0.2 * 0.1, 0.2 * 0.9]
    assert [math.exp(v) for v in verdict.outcome_logprobs] == pytest.approx(
        expected
    )
    assert verdict.metrics == pytest.approx(
        {"probe_honest": 0.2, "probe_sneaky": 0.9}
    )


@pytest.mark.parametrize("z", [-1000.0, -30.0, 0.0, 30.0, 1000.0])
def test_log_sigmoid_is_finite_and_exact_at_any_logit(z):
    value = probe.log_sigmoid(z)
    assert math.isfinite(value)
    assert value == pytest.approx(
        torch.nn.functional.logsigmoid(
            torch.tensor(z, dtype=torch.float64)
        ).item()
    )


def test_the_probe_reads_the_starting_model_and_leaves_pi_alone(
    actor_model, monkeypatch
):
    model = actor_model()
    actor = [p for n, p in model.named_parameters() if ".default." in n]
    trainable = []
    model.register_forward_pre_hook(
        lambda module, args: trainable.append(
            all(p.requires_grad for p in actor)
        )
    )
    weights = probe.ProbeWeights("m", 1, "sae", torch.ones(16), 1.0, 0.0)
    monkeypatch.setattr(
        probe, "load_probe", lambda path, device: (weights, ident)
    )
    ids, mask = [3, 4, 5], [0, 1, 1]
    z = Probe(PROBE).probe_logit(model, ids, mask)
    assert trainable == [True]
    with torch.no_grad(), model.disable_adapter():
        x = probe.pooled_sae_features(model, ident, 1, ids, mask)
    assert z == pytest.approx(weights.logit(x))


def ident(h):
    return h
