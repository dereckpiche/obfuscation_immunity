#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --time=3:00:00

# FLA (Qwen3.5's linear attention) and FlashInfer compile kernels with nvcc.
if [ -n "${CC_CLUSTER:-}" ]; then
  # DRAC: its CUDA module; the proxy where compute nodes have no internet, which wandb bypasses.
  module load cuda/13.2
  module is-avail httpproxy && module load httpproxy
  export NO_PROXY=api.wandb.ai,.wandb.ai,.googleapis.com no_proxy=api.wandb.ai,.wandb.ai,.googleapis.com
else
  # Mila's CUDA module's nvcc dies with a bus error: use the env's CUDA 13 toolkit, whose nvcc
  # (13.2) is newer than its runtime headers (13.0), and give it the lib64/ and libcudart.so it lacks.
  export CUDA_HOME=$PWD/.venv/lib/python3.12/site-packages/nvidia/cu13
  export NVCC_APPEND_FLAGS="-DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK" PATH=$CUDA_HOME/bin:$PATH
  [ -e "$CUDA_HOME/lib64" ] || ln -s lib "$CUDA_HOME/lib64"
  ln -sf libcudart.so.13 "$CUDA_HOME/lib/libcudart.so"
fi
ulimit -u 65536 2>/dev/null || true  # Ray starts many processes
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True RAY_worker_register_timeout_seconds=300
echo "$(hostname)  $(nvidia-smi -L | wc -l)x $(nvidia-smi -L | head -1 | cut -d'(' -f1)  commit $GIT_COMMIT"
# The venv cluv synced, used as is: under `uv run`, Ray starts each worker with `uv run` too,
# and each re-syncs the shared venv while the others import from it.
export RAY_ENABLE_UV_RUN_RUNTIME_ENV=0
source .venv/bin/activate
# srun sets ROCR_VISIBLE_DEVICES in its step on DRAC; verl refuses it beside CUDA_VISIBLE_DEVICES.
srun env -u ROCR_VISIBLE_DEVICES "$@"
