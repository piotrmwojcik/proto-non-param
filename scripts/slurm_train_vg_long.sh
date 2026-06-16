#!/bin/bash
# Long training (80 epochs) for the two frozen-residual candidates on Visual Genome.
#
# Submits 2 independent SLURM jobs:
#   A — KL loss  + frozen residual  (baseline candidate)
#   C — JSD loss + frozen residual  (JSD candidate)
#
# These are the top-2 frozen-residual entries from the 20-epoch ablation
# (slurm_train_vg_ablation.sh). Residual is intentionally kept frozen here.
#
# Configuration — override via environment variables:
#   SCRATCH       Base scratch path (default: /net/tscratch/people/plgabedychaj)
#   VG_ROOT       Path to VG image root (containing VG_100K/ and VG_100K_2/)
#   VG_DESC       Path to region_descriptions.json
#   LOG_DIR       Base directory for checkpoints
#   VOCAB_DIR     Directory with vg_cache.pt
#   WANDB_ENTITY  W&B entity (default: gmum)
#
# Usage:
#   bash scripts/slurm_train_vg_long.sh
#   SCRATCH=/my/scratch bash scripts/slurm_train_vg_long.sh

set -e

# ---- Cluster paths ----
SCRATCH="${SCRATCH:-/net/tscratch/people/plgabedychaj}"
VG_ROOT="${VG_ROOT:-${SCRATCH}/vg}"
VG_DESC="${VG_DESC:-${VG_ROOT}/region_descriptions.json}"
LOG_DIR="${LOG_DIR:-${SCRATCH}/train_logs/vg_long}"
VOCAB_DIR="${VOCAB_DIR:-${SCRATCH}/vocab}"
WANDB_ENTITY="${WANDB_ENTITY:-gmum}"

PARTITION="plgrid-gpu-a100"
ACCOUNT="plgunhype-gpu-a100"
LOG_SLURM="${SCRATCH}/logs"

VG_CACHE="${VOCAB_DIR}/vg_cache.pt"

mkdir -p "${LOG_SLURM}" "${LOG_DIR}"

# Pre-flight: vocab cache must exist
if [ ! -f "${VG_CACHE}" ]; then
    echo "ERROR: VG vocab cache not found at ${VG_CACHE}"
    echo "Build it first with: bash scripts/slurm_build_vg_vocab.sh"
    exit 1
fi

echo "=== Visual Genome Long Training (80 epochs, frozen residual) ==="
echo "  SCRATCH   : ${SCRATCH}"
echo "  VG_ROOT   : ${VG_ROOT}"
echo "  LOG_DIR   : ${LOG_DIR}"
echo "  VOCAB     : ${VG_CACHE}"
echo ""

# ---- Shared training args ----
TRAIN_COMMON="--dataset visual_genome \
  --vg-root ${VG_ROOT} \
  --vg-region-descriptions ${VG_DESC} \
  --vocab-cache-path ${VG_CACHE} \
  --backbone dinov2_vitb14 \
  --batch-size 64 \
  --epochs 80 \
  --num-workers 8 \
  --backbone-lr 1e-5 \
  --text-proj-lr 1e-4 \
  --target-mode topk \
  --top-k-concepts 5 \
  --kl-coef 1.0 \
  --wandb-entity ${WANDB_ENTITY} \
  --wandb-log-images 16"

# PLGrid A100 partition hard limit is 3 days; request the maximum.
SLURM_COMMON="--partition=${PARTITION} \
  --account=${ACCOUNT} \
  --gres=gpu:1 \
  --cpus-per-task=8 \
  --mem=64G \
  --time=3-00:00:00"

WRAP_HEADER="
set -e
export HF_HOME=${SCRATCH}/hf_cache
export TRANSFORMERS_CACHE=${SCRATCH}/hf_cache
export PYTHONPATH=${SCRATCH}/dinov2:\$PYTHONPATH
source ${SCRATCH}/venv/bin/activate
cd ~/proto-VLM/proto-non-param
"

# -----------------------------------------------------------------------
# Run A — KL + frozen residual
# -----------------------------------------------------------------------
JOB_A=$(sbatch --parsable \
  ${SLURM_COMMON} \
  --job-name=vg-long-A-kl \
  --output="${LOG_SLURM}/vg_long_A_%j.out" \
  --error="${LOG_SLURM}/vg_long_A_%j.err" \
  --wrap="${WRAP_HEADER}
python train.py \\
  ${TRAIN_COMMON} \\
  --loss-type kl \\
  --log-dir ${LOG_DIR}/run_A_kl_frozen_80ep
")
echo "Submitted A (KL + frozen residual, 80ep): ${JOB_A}"

# -----------------------------------------------------------------------
# Run C — JSD + frozen residual
# -----------------------------------------------------------------------
JOB_C=$(sbatch --parsable \
  ${SLURM_COMMON} \
  --job-name=vg-long-C-jsd \
  --output="${LOG_SLURM}/vg_long_C_%j.out" \
  --error="${LOG_SLURM}/vg_long_C_%j.err" \
  --wrap="${WRAP_HEADER}
python train.py \\
  ${TRAIN_COMMON} \\
  --loss-type jsd \\
  --log-dir ${LOG_DIR}/run_C_jsd_frozen_80ep
")
echo "Submitted C (JSD + frozen residual, 80ep): ${JOB_C}"

echo ""
echo "Long-training runs:"
echo "  A — KL  + frozen residual : ${JOB_A}"
echo "  C — JSD + frozen residual : ${JOB_C}"
echo ""
echo "Monitor with : squeue -u \$USER"
echo "Logs under   : ${LOG_SLURM}/"
echo "Checkpoints  : ${LOG_DIR}/run_{A,C}_*/"
