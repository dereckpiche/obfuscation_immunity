# An SAE for Qwen3.5-4B

Qwen-Scope has no SAE for the 4B, and its SAEs read base models, while the
probe reads the post-trained Qwen3.5-4B. So we train our own: a TopK SAE on
the residual stream after one layer, the probe's (16 of 32), exported in
Qwen-Scope's format, so that the probe reads it like theirs:

    python -m obfuscation_immunity.probe.fit <sae directory>

## Library

[sparsify](https://github.com/EleutherAI/sparsify), EleutherAI's trainer
(`pip install eai-sparsify`). It hooks any Hugging Face model by module
name, computes activations on the fly instead of caching them to disk,
trains TopK SAEs, and scales across GPUs with `torchrun`.

## Data

About 1B tokens: 60% web text, 30% chat, and 11% (116M tokens)
misalignment and reward hacking. Everything below runs from one working
directory.

- **Web:** [FineWeb](https://huggingface.co/datasets/HuggingFaceFW/fineweb)'s
  10B-token sample, not FineWeb-Edu: Edu's educational filter drops
  forums, fiction and comment threads, and with them much of the writing
  about cheating, gaming and deception, which the SAE must represent
  well.
- **Chat:** the [Tulu 3 SFT mixture](https://huggingface.co/datasets/allenai/tulu-3-sft-mixture)
  rendered through Qwen3.5's chat template. The model is post-trained and
  the probe reads chat transcripts, so the SAE should see the template and
  assistant turns, not only documents.
- **Misalignment and reward hacking**, all of two datasets, as chats:
  [Terminal Wrench](https://huggingface.co/datasets/few-sh/terminal-wrench)'s
  agent trajectories on terminal tasks (reward hacks, the same hacks
  rewritten to look innocent, legitimate solves), and Anthropic's
  [model-written evals](https://huggingface.co/datasets/Anthropic/model-written-evals)
  (personas, advanced AI risk, sycophancy) answered the misaligned way.
  School of Reward Hacks stays out: the probe is fitted and checked on it.

Web and chat budgets are in characters: Qwen3.5's tokenizer averages 4.43
characters per token on FineWeb and 3.67 on the rendered chats.

```python
import json
from pathlib import Path

import datasets
import numpy as np
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-4B")


def render(messages):
    """A conversation as Qwen3.5's chat template writes it."""
    return {"text": tokenizer.apply_chat_template(messages, tokenize=False)}


def first(data, chars):
    """The first rows of `data` whose texts add up to `chars` characters."""
    total = np.cumsum([len(text) for text in data["text"]])
    return data.select(range(int(np.searchsorted(total, chars))))


def trajectory(path):
    """A Terminal Wrench trajectory: the agent's commentary and keystrokes,
    then the terminal's output.
    """
    messages = []
    for step in json.loads(path.read_text())["steps"]:
        if step["source"] != "agent":
            messages.append({"role": "user", "content": step["message"]})
            continue
        keys = "".join(
            call["arguments"].get("keystrokes", "")
            for call in step.get("tool_calls") or []
        )
        messages.append(
            {"role": "assistant", "content": f"{step['message']}\n\n{keys}"}
        )
        results = (step.get("observation") or {}).get("results", [])
        output = "\n".join(str(r.get("content", "")) for r in results)
        messages.append({"role": "user", "content": output})
    return render(messages)


def misaligned(row):
    """A model-written eval question, answered the misaligned way."""
    answer = row["answer_matching_behavior"]
    answer = answer[0] if isinstance(answer, list) else answer
    return render(
        [
            {"role": "user", "content": row["question"]},
            {"role": "assistant", "content": answer.strip()},
        ]
    )


web = datasets.load_dataset(
    "HuggingFaceFW/fineweb",
    data_files=["sample/10BT/000_00000.parquet", "sample/10BT/001_00000.parquet"],
    split="train",
).select_columns(["text"])
chat = datasets.load_dataset("allenai/tulu-3-sft-mixture", split="train").map(
    lambda row: render(row["messages"]),
    remove_columns=["id", "messages", "source"],
    num_proc=8,
)
wrench = Path(
    snapshot_download(
        "few-sh/terminal-wrench",
        repo_type="dataset",
        allow_patterns=[
            f"*/{kind}_trajectories/*/trajectory.json"
            for kind in ("hack", "sanitized", "non_hack", "baseline")
        ],
    )
)
evals = Path(
    snapshot_download(
        "Anthropic/model-written-evals",
        repo_type="dataset",
        allow_patterns=[
            "persona/*",
            "advanced-ai-risk/human_generated_evals/*",
            "advanced-ai-risk/lm_generated_evals/*",
            "sycophancy/*",
        ],
    )
)
rows = [
    json.loads(line) for path in evals.rglob("*.jsonl") for line in path.open()
]
safety = datasets.Dataset.from_list(
    [trajectory(path) for path in wrench.rglob("trajectory.json")]
    + [misaligned(row) for row in rows]
)
mix = datasets.concatenate_datasets(
    [first(web, 0.6e9 * 4.43), first(chat, 0.3e9 * 3.67), safety]
)
mix.shuffle(seed=0).to_parquet("mix.parquet")
```

## Training

Qwen-Scope's settings: 16× expansion (40,960 latents) and k = 50. sparsify
packs the text into 2048-token sequences; `auxk_alpha` revives dead latents
([Gao et al. 2024](https://arxiv.org/abs/2406.04093)). Checkpoints go to
`checkpoints/sae-qwen3.5-4b-l16`; after an interruption, the same command with
`--resume` continues from there.

```bash
WANDB_PROJECT=obfuscation_immunity torchrun --nproc_per_node gpu -m sparsify \
    Qwen/Qwen3.5-4B parquet --data_args data_files=mix.parquet \
    --hookpoints language_model.layers.16 --k 50 --num_latents 40960 \
    --ctx_len 2048 --batch_size 8 --auxk_alpha 0.03125 \
    --run_name sae-qwen3.5-4b-l16
```

## Publishing

Qwen-Scope's format: `config.json`, and `layer16.sae.pt` holding `W_enc`,
`b_enc`, `W_dec` and `b_dec` with pre-activations `h @ W_enc.T + b_enc`.
sparsify encodes `h - b_dec`, so `b_dec` folds into `b_enc`.

```python
import json
from pathlib import Path

import torch
from sparsify import Sae

sae = Sae.load_from_disk("checkpoints/sae-qwen3.5-4b-l16/language_model.layers.16")
w, b_dec = sae.encoder.weight.float(), sae.b_dec.float()
out = Path("sae")
out.mkdir(exist_ok=True)
torch.save(
    {
        "W_enc": w,
        "b_enc": sae.encoder.bias.float() - w @ b_dec,
        "W_dec": sae.W_dec.float().T,
        "b_dec": b_dec,
    },
    out / "layer16.sae.pt",
)
config = {
    "model_type": "topk_sae",
    "base_model": "Qwen/Qwen3.5-4B",
    "d_model": sae.d_in,
    "d_sae": sae.num_latents,
    "k": sae.cfg.k,
    "layers": [16],
    "hook_point": "resid_post",
    "dtype": "float32",
}
(out / "config.json").write_text(json.dumps(config, indent=2))
```

To share it, upload `sae/` to a Hugging Face model repo; others download it
and pass the directory to the probe.
