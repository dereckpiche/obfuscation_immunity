#!/bin/bash
# Download the models the setups use into the HF cache.
# Run once per cluster on its login node: cluv run <cluster> bash scripts/prefetch.sh
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH" HF_HUB_DISABLE_XET=1
export HF_TOKEN_PATH="$HOME/.config/huggingface/token"

MODELS="Qwen/Qwen3.5-4B Qwen/Qwen3.5-2B IFM/K2-Horizon-3.7B IFM/K2-Horizon-0.9B"

for model in $MODELS; do
  if uvx --quiet --from huggingface_hub hf download "$model" --quiet >/dev/null; then
    echo "model $model ok"
  else
    echo "model $model FAILED"
  fi
done
echo "hf cache $(du -sh "$HOME/.cache/huggingface" | cut -f1)"
