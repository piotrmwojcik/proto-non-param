#!/bin/bash
# KL + frozen residual training on MSCOCO with milestone checkpoints.
#
# Mirrors slurm_train_vg_fullvocab_long.sh but trains on COCO captions
# (coco_clip dataset) with the MSCOCO vocabulary (mscoco_new_cache.pt).
#
# Key differences vs VG training:
#   --dataset coco_clip       COCO images + captions (not VG region descriptions)
#   --target-mode prob        frequency-weighted KL (COCO has 5 captions/image)
#   --vocab-cache-path        mscoco_new_cache.pt (COCO noun vocab)
#
# ~10 min/epoch × 80 epochs ≈ 13 h; wall time set to 16 h.
# Checkpoint dir: ${LOG_DIR}/run_A_kl_frozen_coco_80ep/
#
# After training, find the optimal epoch:
#   CKPT_DIR=$SCRATCH/train_logs/coco/run_A_kl_frozen_coco_80ep \
#   VARIANT=coco_A_80ep \
#   bash ~/proto-VLM/scripts/slurm_eval_collapse_check.sh
#
# Configuration — override via environment variables:
#   SCRATCH         Base scratch path (default: /net/tscratch/people/plgabedychaj)
#   COCO_ROOT       Path to COCO images root (containing train2014/ and val2014/)
#   COCO_ANN_TRAIN  Path to captions_train2014.json
#   COCO_ANN_VAL    Path to captions_val2014.json
#   COCO_CACHE      Path to mscoco_new_cache.pt vocab cache
#   LOG_DIR         Base directory for checkpoints
#   WANDB_ENTITY    W&B entity (default: gmum)
#
# Usage:
#   bash scripts/slurm_train_coco_long.sh

set -e

SCRATCH="${SCRATCH:-/net/tscratch/people/plgabedychaj}"
COCO_ROOT="${COCO_ROOT:-${SCRATCH}/coco}"
COCO_ANN_TRAIN="${COCO_ANN_TRAIN:-${SCRATCH}/coco/annotations/captions_train2014.json}"
COCO_ANN_VAL="${COCO_ANN_VAL:-${SCRATCH}/coco/annotations/captions_val2014.json}"
COCO_CACHE="${COCO_CACHE:-${SCRATCH}/vocab/mscoco_new_cache.pt}"
LOG_DIR="${LOG_DIR:-${SCRATCH}/train_logs/coco}"
WANDB_ENTITY="${WANDB_ENTITY:-gmum}"

PARTITION="plgrid-gpu-a100"
ACCOUNT="plgunhype-gpu-a100"
LOG_SLURM="${SCRATCH}/logs"

mkdir -p "${LOG_SLURM}" "${LOG_DIR}"

if [ ! -f "${COCO_CACHE}" ]; then
    echo "ERROR: COCO vocab cache not found at ${COCO_CACHE}"
    exit 1
fi

if [ ! -f "${COCO_ANN_TRAIN}" ]; then
    echo "ERROR: COCO train annotations not found at ${COCO_ANN_TRAIN}"
    exit 1
fi

echo "=== COCO Long Training (KL + frozen δ, 80 epochs, save every 5) ==="
echo "  SCRATCH      : ${SCRATCH}"
echo "  COCO_ROOT    : ${COCO_ROOT}"
echo "  Annotations  : ${COCO_ANN_TRAIN}"
echo "  Vocab cache  : ${COCO_CACHE}"
echo "  Checkpoint   : ${LOG_DIR}/run_A_kl_frozen_coco_80ep/"
echo ""

WRAP_HEADER="
set -e
export HF_HOME=${SCRATCH}/hf_cache
export TRANSFORMERS_CACHE=${SCRATCH}/hf_cache
export PYTHONPATH=${SCRATCH}/dinov2:\$PYTHONPATH
source ${SCRATCH}/venv/bin/activate
cd ~/proto-VLM/proto-non-param
"

JOB_A=$(sbatch --parsable \
  --job-name=coco-A-kl \
  --partition="${PARTITION}" \
  --account="${ACCOUNT}" \
  --gres=gpu:1 \
  --cpus-per-task=8 \
  --mem=64G \
  --time=16:00:00 \
  --output="${LOG_SLURM}/coco_A_%j.out" \
  --error="${LOG_SLURM}/coco_A_%j.err" \
  --wrap="${WRAP_HEADER}
python train.py \\
  --dataset coco_clip \\
  --coco-root ${COCO_ROOT} \\
  --coco-annotations-train ${COCO_ANN_TRAIN} \\
  --coco-annotations-val ${COCO_ANN_VAL} \\
  --vocab-cache-path ${COCO_CACHE} \\
  --backbone dinov2_vitb14 \\
  --batch-size 64 \\
  --epochs 80 \\
  --lr-schedule cosine \\
  --lr-warmup-epochs 5 \\
  --num-workers 8 \\
  --backbone-lr 1e-5 \\
  --text-proj-lr 1e-4 \\
  --target-mode prob \\
  --kl-coef 1.0 \\
  --loss-type kl \\
  --save-every 5 \\
  --wandb-entity ${WANDB_ENTITY} \\
  --wandb-log-images 16 \\
  --log-dir ${LOG_DIR}/run_A_kl_frozen_coco_80ep
")

echo "Submitted job ${JOB_A}"
echo ""
echo "Monitor with : squeue -u \$USER"
echo "Log          : ${LOG_SLURM}/coco_A_${JOB_A}.out"
echo ""
echo "After training, find the optimal epoch with:"
echo "  CKPT_DIR=${LOG_DIR}/run_A_kl_frozen_coco_80ep \\"
echo "  VARIANT=coco_A_80ep \\"
echo "  bash ~/proto-VLM/scripts/slurm_eval_collapse_check.sh"
echo ""
echo "Then run full RIS eval on the best epoch checkpoint:"
echo "  CKPT=${LOG_DIR}/run_A_kl_frozen_coco_80ep/ckpt_ep<N>.pth \\"
echo "  VARIANT=coco_A_ep<N> \\"
echo "  bash ~/proto-VLM/scripts/slurm_eval_pnp_threshold_sweep.sh"
