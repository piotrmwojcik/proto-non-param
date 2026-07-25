#!/bin/bash
#SBATCH --job-name=pnp_VG
#SBATCH --account=plgtabcfs-gpu-a100
#SBATCH --partition=plgrid-gpu-a100

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=24:00:00

#SBATCH --output=/net/people/plgrid/plgpiotrwojcik/proto-non-param/logs/%x_%j.out
#SBATCH --error=/net/people/plgrid/plgpiotrwojcik/proto-non-param/logs/%x_%j.err

set -eo pipefail

CONDA_SH="${HOME}/miniconda3/etc/profile.d/conda.sh"

if [[ ! -f "${CONDA_SH}" ]]; then
    CONDA_SH="${HOME}/anaconda3/etc/profile.d/conda.sh"
fi

if [[ ! -f "${CONDA_SH}" ]]; then
    echo "ERROR: Conda initialisation script not found." >&2
    exit 1
fi

source "${CONDA_SH}"
conda activate open_clip

# ============================================================

# Paths

# ============================================================

PROJECT_DIR="/net/people/plgrid/plgpiotrwojcik/proto-non-param"
TRAIN_SCRIPT="${PROJECT_DIR}/train.py"

SCRATCH_ROOT="/net/tscratch/people/plgpiotrwojcik"
RUN_ROOT="${SCRATCH_ROOT}/pnp_vg_logs"
WANDB_DIR="${RUN_ROOT}/wandb"
CHECKPOINT_DIR="${RUN_ROOT}/checkpoints"
CACHE_DIR="${SCRATCH_ROOT}/.cache"

mkdir -p "${WANDB_DIR}"
mkdir -p "${CHECKPOINT_DIR}"
mkdir -p "${CACHE_DIR}"

# ============================================================

# Environment

# ============================================================

echo "Job ID: ${SLURM_JOB_ID:-unknown}"
echo "Job name: ${SLURM_JOB_NAME:-unknown}"
echo "Node: $(hostname)"
echo "Start time: $(date --iso-8601=seconds)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

source "${HOME}/.bashrc"

if command -v conda >/dev/null 2>&1; then
eval "$(conda shell.bash hook)"
else
source "${HOME}/miniconda3/etc/profile.d/conda.sh"
fi

conda activate open_clip

cd "${PROJECT_DIR}"

export PYTHONPATH="${PROJECT_DIR}/src:${PYTHONPATH:-}"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export TOKENIZERS_PARALLELISM=false

export TORCH_CUDNN_V8_API_ENABLED=1
export TORCH_HOME="${CACHE_DIR}/torch"
export XDG_CACHE_HOME="${CACHE_DIR}"
export HF_HOME="${CACHE_DIR}/huggingface"
export TRANSFORMERS_CACHE="${CACHE_DIR}/huggingface/transformers"

export WANDB_ENTITY="piotrmwojcik"
export WANDB_PROJECT="proto-non-param"
export WANDB_DIR="${WANDB_DIR}"

# ============================================================

# Validation

# ============================================================

if [[ ! -f "${TRAIN_SCRIPT}" ]]; then
echo "ERROR: Training script not found: ${TRAIN_SCRIPT}" >&2
exit 1
fi

echo "Python executable: $(which python)"
echo "Training script: ${TRAIN_SCRIPT}"

nvidia-smi

python - <<'PY'
import sys
import torch

print("Python:", sys.version)
print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA version:", torch.version.cuda)

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is unavailable inside the SLURM job.")

print("GPU:", torch.cuda.get_device_name(0))
PY

python -m py_compile "${TRAIN_SCRIPT}"

# ============================================================

# Training

# ============================================================

srun --unbuffered python -u "${TRAIN_SCRIPT}"
--backbone dinov2_vitb14
--backbone-module modeling.backbones
--num-splits 1
--batch-size 64
--epochs 20
--num-workers 4
--negatives-per-positive 4
--encode-batch-size 32
--text-dim 4096
--log-dir "${WANDB_DIR}"
--checkpoint-dir "${CHECKPOINT_DIR}"
--visualize-every-steps 10
--visualize-samples 1
--visualize-images-per-batch 4

echo "End time: $(date --iso-8601=seconds)"
