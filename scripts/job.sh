#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --time=3:00:00

# DRAC: FLA compiles Qwen3.5's linear-attention kernels (TileLang) with nvcc; wandb bypasses the proxy.
if [ -n "${CC_CLUSTER:-}" ]; then
  module load cuda/13.2 httpproxy
  export NO_PROXY=api.wandb.ai,.wandb.ai,.googleapis.com no_proxy=api.wandb.ai,.wandb.ai,.googleapis.com
fi
ulimit -u 65536 2>/dev/null || true  # Ray starts many processes
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True RAY_worker_register_timeout_seconds=300
echo "$(hostname)  $(nvidia-smi -L | head -1)  commit $GIT_COMMIT"
srun uv run "$@"
