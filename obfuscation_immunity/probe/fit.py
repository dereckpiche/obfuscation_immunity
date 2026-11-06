"""The probe the Probe judge reads, fitted once, away from the puzzles, on
School of Reward Hacks: about a thousand prompts, each with a
reward-hacking answer and an honest one.

A frozen model reads each answer: the policy's starting model. x is the
mean over the answer's tokens of φ(h_t), where h_t is the residual stream
after decoder layer ℓ and φ is the encoder of an SAE trained on that
model, in Qwen-Scope's format: a directory (sae.md trains one) or a Hub
repo. The probe is a difference of means with a fitted logistic link:

    v = mean(x | hack) − mean(x | honest),    ψ(x) = σ(a · vᵀx + b)

    python -m obfuscation_immunity.probe.fit <sae> [--model Qwen/Qwen3.5-4B]
"""

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn.functional as F

DATASET = "longtermrisk/school-of-reward-hacks"


@dataclass
class Weights:
    """ψ(x) = σ(scale · directionᵀx + bias), where x pools φ(h_t) at
    `layer` of `model`, φ being the encoder of the SAE in `sae`.
    """

    model: str
    layer: int
    sae: str
    direction: torch.Tensor
    scale: float
    bias: float

    @classmethod
    def fit(cls, x, y, model, layer, sae) -> "Weights":
        """Difference of means, rescaled so that the projections have unit
        spread, then a and b by 1-D logistic regression (lightly
        regularised, since the classes may separate).
        """
        v = x[y == 1].mean(0) - x[y == 0].mean(0)
        v = v / (x @ v).std()
        s, target = x @ v, y.float()
        a = torch.zeros((), requires_grad=True)
        b = torch.zeros((), requires_grad=True)
        optimizer = torch.optim.LBFGS([a, b], max_iter=200)

        def loss():
            """Cross-entropy of σ(a·s + b) against the labels."""
            optimizer.zero_grad()
            value = F.binary_cross_entropy_with_logits(a * s + b, target)
            value = value + 1e-3 * a**2
            value.backward()
            return value

        optimizer.step(loss)
        return cls(model, layer, sae, v, a.item(), b.item())

    def logit(self, x: torch.Tensor) -> float:
        """ψ's logit, a · vᵀx + b, for one pooled feature vector."""
        direction = self.direction.to(x.device, x.dtype)
        return (self.scale * direction @ x + self.bias).item()

    def score(self, x: torch.Tensor) -> float:
        """ψ(x) for one pooled feature vector."""
        return torch.sigmoid(torch.tensor(self.logit(x))).item()

    def save(self, path: Path) -> None:
        """Write the probe to `path`."""
        torch.save(asdict(self), path)

    @classmethod
    def load(cls, path: str) -> "Weights":
        """Read a probe written by `save`."""
        return cls(**torch.load(path))


class TopKSAE:
    """Qwen-Scope's encoder for one layer: φ(h) = TopK_k(W_enc h + b_enc)."""

    def __init__(self, path: str, layer: int, width: int, device):
        """Read k and the layer's weights from the SAE, a directory or a
        Hub repo, after checking that it reads residual streams `width`
        wide: every Qwen3.5 size has its own width, so a mismatch means
        another model's SAE.
        """
        root = Path(path)
        if not root.is_dir():
            from huggingface_hub import snapshot_download

            root = Path(snapshot_download(path))
        config = json.loads((root / "config.json").read_text())
        if config["d_model"] != width:
            raise ValueError(
                f"{path} was trained on {config['base_model']}, whose residual"
                f" stream is {config['d_model']} wide, not {width}"
            )
        self.k = config["k"]
        weights = torch.load(root / f"layer{layer}.sae.pt", map_location=device)
        self.w, self.b = weights["W_enc"], weights["b_enc"]

    def __call__(self, h: torch.Tensor) -> torch.Tensor:
        """Feature activations, all but the k largest set to zero."""
        pre = h @ self.w.T.to(h.dtype) + self.b.to(h.dtype)
        top = pre.topk(self.k, dim=-1)
        return torch.zeros_like(pre).scatter_(-1, top.indices, top.values)


def features(model, sae: TopKSAE, layer: int, ids, mask) -> torch.Tensor:
    """x for one sequence: φ of the residual stream after decoder layer
    `layer` (hidden_states[layer + 1]), averaged over the masked tokens.
    """
    from obfuscation_immunity.common.workers import last_logits

    device = next(model.parameters()).device
    input_ids = torch.tensor([ids], device=device)
    with last_logits(model, 1):
        out = model(
            input_ids=input_ids, output_hidden_states=True, use_cache=False
        )
    h = out.hidden_states[layer + 1][0].float()
    return sae(h[torch.tensor(mask, device=device).bool()]).mean(0)


def chat(tokenizer, user: str, answer: str) -> tuple[list[int], list[int]]:
    """Token ids of a one-turn chat, and the mask of the answer's tokens."""
    question = [{"role": "user", "content": user}]
    prompt = tokenizer.apply_chat_template(
        question, tokenize=False, add_generation_prompt=True
    )
    full = tokenizer.apply_chat_template(
        question + [{"role": "assistant", "content": answer}], tokenize=False
    )
    start = len(tokenizer(prompt, add_special_tokens=False).input_ids)
    ids = tokenizer(full, add_special_tokens=False).input_ids
    return ids, [0] * start + [1] * (len(ids) - start)


def auroc(scores: torch.Tensor, labels: torch.Tensor) -> float:
    """Chance that a random hack scores above a random honest answer."""
    pos, neg = scores[labels == 1], scores[labels == 0]
    above = (pos[:, None] > neg[None, :]).float()
    ties = (pos[:, None] == neg[None, :]).float()
    return (above + ties / 2).mean().item()


@torch.no_grad()
def fit(model: str, layer: int | None, sae: str, rows: int, out: Path):
    """Read both answers of every row, fit on 80% of the rows, print the
    mean ψ of the other 20%'s hacks and honest answers and the AUROC
    there, and save the probe.
    """
    import datasets
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model)
    reader = AutoModelForCausalLM.from_pretrained(model, dtype=torch.bfloat16)
    reader.to(device).eval()
    text = reader.config.get_text_config()
    depth = text.num_hidden_layers // 2 if layer is None else layer
    encoder = TopKSAE(sae, depth, text.hidden_size, device)
    data = datasets.load_dataset(DATASET, split="train")
    data = data.shuffle(seed=0).select(range(min(rows, len(data))))
    x, y = [], []
    for row in data:
        hack, honest = row["school_of_reward_hacks"], row["control"]
        for answer, label in ((hack, 1), (honest, 0)):
            ids, mask = chat(tokenizer, row["user"], answer)
            x.append(features(reader, encoder, depth, ids, mask).cpu())
            y.append(label)
    x, y = torch.stack(x), torch.tensor(y)
    cut = 2 * int(0.8 * len(data))
    probe = Weights.fit(x[:cut], y[:cut], model, depth, sae)
    scores = torch.tensor([probe.score(v) for v in x[cut:]])
    hacks, honest = scores[y[cut:] == 1], scores[y[cut:] == 0]
    print(
        f"held-out mean ψ: hacks {hacks.mean():.3f},"
        f" honest {honest.mean():.3f},"
        f" AUROC {auroc(scores, y[cut:]):.3f} ({len(hacks)} rows)"
    )
    probe.save(out)
    print(f"probe -> {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "sae", help="an SAE in Qwen-Scope's format: a directory or Hub repo"
    )
    parser.add_argument("--model", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--layer", type=int, help="default: the middle one")
    parser.add_argument("--rows", type=int, default=10**6)
    parser.add_argument("--out", type=Path, default=Path("probe.pt"))
    args = parser.parse_args()
    fit(args.model, args.layer, args.sae, args.rows, args.out)
