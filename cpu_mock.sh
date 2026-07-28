#!/bin/bash
set -euo pipefail

PROJECT_DIR="/net/people/plgrid/plgpiotrwojcik/proto-non-param"
TRAIN_SCRIPT="${PROJECT_DIR}/train.py"

SCRATCH_ROOT="/net/tscratch/people/plgpiotrwojcik"
VG_ROOT="${SCRATCH_ROOT}/vg"
RUN_ROOT="${SCRATCH_ROOT}/pnp_vg_cpu_mock"

WANDB_DIR="${RUN_ROOT}/wandb"
CHECKPOINT_DIR="${RUN_ROOT}/checkpoints"

mkdir -p "${WANDB_DIR}" "${CHECKPOINT_DIR}"

if [[ ! -f "${TRAIN_SCRIPT}" ]]; then
    echo "ERROR: Training script not found: ${TRAIN_SCRIPT}" >&2
    exit 1
fi

cd "${PROJECT_DIR}"

export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES=""
export WANDB_MODE=disabled
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

echo "Python: $(command -v python)"
echo "Training script: ${TRAIN_SCRIPT}"
echo "Dataset root: ${VG_ROOT}"

python -u "${TRAIN_SCRIPT}" \
    --backbone dinov2_vitb14 \
    --backbone-module modeling.backbone \
    --dataset-root "${VG_ROOT}" \
    --num-splits 0 \
    --batch-size 1 \
    --epochs 1 \
    --num-workers 0 \
    --negatives-per-positive 1 \
    --encode-batch-size 1 \
    --text-dim 4096 \
    --checkpoint-dir "${CHECKPOINT_DIR}" \
    --log-dir "${WANDB_DIR}" \
    --visualize-every-steps 0 \
    --visualize-samples 1 \
    --visualize-images-per-batch 1 \
    --mock-text-embeddings \
    --max-steps 1

STATUS=$?

echo
echo "============================================================"
echo "Finished at: $(date)"
echo "Exit code  : ${STATUS}"
echo "============================================================"

exit "${STATUS}"
