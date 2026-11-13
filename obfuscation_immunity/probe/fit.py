"""
Fits, once and away from the puzzles, the probe that Probe reads.

The data is School of Reward Hacks: about a thousand prompts, each with a
reward-hacking answer and an honest one. A frozen model reads each answer:
the policy's starting model. x is the mean over the answer's tokens of
φ(h_t), where h_t is the residual stream after decoder layer ℓ and φ is the
encoder of an SAE trained on that model, in Qwen-Scope's format: a
directory (sae.md trains one) or a Hub repo. The probe is a difference of
means with a fitted logistic link:

    v = mean(x | hack) − mean(x | honest),    ψ(x) = σ(a · vᵀx + b)

    python -m obfuscation_immunity.probe.fit <sae> [--model Qwen/Qwen3.5-4B]
"""

import argparse
from pathlib import Path

import datasets
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedTokenizerBase,
)

from obfuscation_immunity.probe.probe import (
    ProbeWeights,
    TopKSAE,
    pooled_sae_features,
)

SCHOOL_OF_REWARD_HACKS = "longtermrisk/school-of-reward-hacks"
"""The Hub dataset the probe is fitted on."""

HELD_OUT_SHARE = 0.2
"""The share of rows kept out of the fit, to measure the probe on."""


def tokenize_chat_answer(
    tokenizer: PreTrainedTokenizerBase, user: str, answer: str
) -> tuple[list[int], list[int]]:
    """
    Tokenizes a one-turn chat and marks the answer's tokens.

    Args:
        tokenizer: The model's tokenizer.
        user: The user's message.
        answer: The assistant's answer.

    Returns:
        The chat's tokens, and 1 for each of the answer's tokens, 0 before.
    """
    question = [{"role": "user", "content": user}]
    prompt = tokenizer.apply_chat_template(
        question, tokenize=False, add_generation_prompt=True
    )
    chat = tokenizer.apply_chat_template(
        question + [{"role": "assistant", "content": answer}], tokenize=False
    )
    prompt_length = len(tokenizer(prompt, add_special_tokens=False).input_ids)
    token_ids = tokenizer(chat, add_special_tokens=False).input_ids
    answer_length = len(token_ids) - prompt_length
    return token_ids, [0] * prompt_length + [1] * answer_length


def auroc(scores: torch.Tensor, labels: torch.Tensor) -> float:
    """
    Computes the chance that a random hack scores above a random honest one.

    Args:
        scores: One score per answer.
        labels: 1 for a hack, 0 for an honest answer.

    Returns:
        The AUROC, ties counted as half.
    """
    hacks, honest = scores[labels == 1], scores[labels == 0]
    above = (hacks[:, None] > honest[None, :]).float()
    ties = (hacks[:, None] == honest[None, :]).float()
    return (above + ties / 2).mean().item()


@torch.no_grad()
def fit_probe(
    model: str, layer: int | None, sae: str, rows: int, out: Path
) -> None:
    """
    Fits a probe on School of Reward Hacks, reports it, and saves it.

    Reads both answers of every row, fits on 80% of the rows, and prints
    the mean probe score of the other 20%'s hacks and honest answers and
    the AUROC there.

    Args:
        model: The model whose activations the probe reads.
        layer: The layer to read; None for the middle one.
        sae: The SAE for that model and layer.
        rows: At most this many rows are used.
        out: The file to save the probe to.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    reader = AutoModelForCausalLM.from_pretrained(
        model, dtype=torch.bfloat16, trust_remote_code=True
    )
    reader.to(device).eval()
    text_config = reader.config.get_text_config()
    if layer is None:
        layer = text_config.num_hidden_layers // 2
    encoder = TopKSAE(sae, layer, text_config.hidden_size, device)
    data = datasets.load_dataset(SCHOOL_OF_REWARD_HACKS, split="train")
    data = data.shuffle(seed=0).select(range(min(rows, len(data))))
    features, labels = [], []
    for row in data:
        hack, honest = row["school_of_reward_hacks"], row["control"]
        for answer, label in ((hack, 1), (honest, 0)):
            token_ids, answer_mask = tokenize_chat_answer(
                tokenizer, row["user"], answer
            )
            pooled = pooled_sae_features(
                reader, encoder, layer, token_ids, answer_mask
            )
            features.append(pooled.cpu())
            labels.append(label)
    all_features, all_labels = torch.stack(features), torch.tensor(labels)
    cut = 2 * int((1 - HELD_OUT_SHARE) * len(data))
    probe = ProbeWeights.fit(
        all_features[:cut], all_labels[:cut], model, layer, sae
    )
    held_out_labels = all_labels[cut:]
    scores = torch.tensor([probe.score(x) for x in all_features[cut:]])
    hack_scores = scores[held_out_labels == 1]
    honest_scores = scores[held_out_labels == 0]
    print(
        f"held-out mean probe score: hacks {hack_scores.mean():.3f},"
        f" honest {honest_scores.mean():.3f},"
        f" AUROC {auroc(scores, held_out_labels):.3f}"
        f" ({len(hack_scores)} rows)"
    )
    probe.save(out)
    print(f"probe -> {out}")


def main(argv: list[str] | None = None) -> None:
    """
    Parses the command line and fits the probe.

    Args:
        argv: The command line's arguments; None for sys.argv's.
    """
    summary = __doc__.strip().splitlines()[0]
    parser = argparse.ArgumentParser(description=summary)
    parser.add_argument(
        "sae", help="an SAE in Qwen-Scope's format: a directory or Hub repo"
    )
    parser.add_argument("--model", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--layer", type=int, help="default: the middle one")
    parser.add_argument("--rows", type=int, default=10**6)
    parser.add_argument("--out", type=Path, default=Path("probe.pt"))
    args = parser.parse_args(argv)
    fit_probe(args.model, args.layer, args.sae, args.rows, args.out)


if __name__ == "__main__":
    main()
