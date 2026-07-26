#!/bin/bash
#SBATCH --job-name=pnp_vg
#SBATCH --account=plgunhype-gpu-a100
#SBATCH --partition=plgrid-gpu-a100

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=24:00:00

#SBATCH --output=/net/people/plgrid/plgpiotrwojcik/proto-non-param/logs/%x_%j.out
#SBATCH --error=/net/people/plgrid/plgpiotrwojcik/proto-non-param/logs/%x_%j.err

set -euo pipefail

###############################################################################
# Paths
###############################################################################

PROJECT_DIR="/net/people/plgrid/plgpiotrwojcik/proto-non-param"
TRAIN_SCRIPT="${PROJECT_DIR}/train.py"

SCRATCH_ROOT="/net/tscratch/people/plgpiotrwojcik"

VG_ROOT="${SCRATCH_ROOT}/vg"

RUN_ROOT="${SCRATCH_ROOT}/pnp_vg"

WANDB_DIR="${RUN_ROOT}/wandb"
CHECKPOINT_DIR="${RUN_ROOT}/checkpoints"

TORCH_CACHE="/net/people/plgrid/plgpiotrwojcik/.cache/torch"
HF_CACHE="${SCRATCH_ROOT}/model_cache"

mkdir -p \
    "${PROJECT_DIR}/logs" \
    "${WANDB_DIR}" \
    "${CHECKPOINT_DIR}" \
    "${HF_CACHE}" \
    "${TORCH_CACHE}"

###############################################################################
# Conda
###############################################################################

if [[ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]]; then
    source "${HOME}/miniconda3/etc/profile.d/conda.sh"
elif [[ -f "${HOME}/anaconda3/etc/profile.d/conda.sh" ]]; then
    source "${HOME}/anaconda3/etc/profile.d/conda.sh"
else
    echo "Cannot locate conda.sh"
    exit 1
fi

conda activate open_clip

###############################################################################
# Environment
###############################################################################

cd "${PROJECT_DIR}"

export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

export TOKENIZERS_PARALLELISM=false

export TORCH_HOME="${TORCH_CACHE}"

export HF_HOME="${HF_CACHE}"
export HUGGINGFACE_HUB_CACHE="${HF_CACHE}/hub"
export TRANSFORMERS_CACHE="${HF_CACHE}/transformers"

export WANDB_ENTITY="piotrmwojcik"
export WANDB_PROJECT="proto-non-param"
export WANDB_DIR="${WANDB_DIR}"

###############################################################################
# Information
###############################################################################

echo "============================================================"
echo "Job ID        : ${SLURM_JOB_ID}"
echo "Node          : $(hostname)"
echo "Date          : $(date)"
echo "Python        : $(command -v python)"
echo "Project       : ${PROJECT_DIR}"
echo "Dataset       : ${VG_ROOT}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "============================================================"

###############################################################################
# Validation
###############################################################################

[[ -f "${TRAIN_SCRIPT}" ]]
[[ -f "${PROJECT_DIR}/modeling/backbone.py" ]]
[[ -f "${PROJECT_DIR}/modeling/pnp.py" ]]
[[ -d "${VG_ROOT}" ]]

python -m py_compile \
    "${TRAIN_SCRIPT}" \
    "${PROJECT_DIR}/modeling/backbone.py" \
    "${PROJECT_DIR}/modeling/pnp.py" \
    "${PROJECT_DIR}/visual_genome_scene_graph_dataset.py"

python <<'PY'
import sys
import torch

print("Python:", sys.version)
print("Torch:", torch.__version__)
print("CUDA:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())

if not torch.cuda.is_available():
    raise RuntimeError("CUDA unavailable")

print("GPU:", torch.cuda.get_device_name(0))

free_mem, total_mem = torch.cuda.mem_get_info()

print(f"GPU memory free : {free_mem/1024**3:.2f} GB")
print(f"GPU memory total: {total_mem/1024**3:.2f} GB")
PY

python <<'PY'
from modeling.backbone import DINOv2Backbone
from modeling.pnp import PNP

print("Backbone import OK")
print("PNP import OK")
PY

nvidia-smi

###############################################################################
# Training
###############################################################################

srun --unbuffered \
python -u "${TRAIN_SCRIPT}" \
    --backbone dinov2_vitb14 \
    --backbone-module modeling.backbone \
    --dataset-root "${VG_ROOT}" \
    --llm-cache-dir "${HF_CACHE}" \
    --num-splits 0 \
    --batch-size 2 \
    --epochs 20 \
    --num-workers 4 \
    --negatives-per-positive 4 \
    --encode-batch-size 4 \
    --text-dim 4096 \
    --checkpoint-dir "${CHECKPOINT_DIR}" \
    --log-dir "${WANDB_DIR}" \
    --visualize-every-steps 100 \
    --visualize-samples 1 \
    --visualize-images-per-batch 2

STATUS=$?

echo
echo "============================================================"
echo "Finished at: $(date)"
echo "Exit code  : ${STATUS}"
echo "============================================================"

exit ${STATUS}