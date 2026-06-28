#!/bin/bash
# Diagnose prototype utilization entropy for a trained PNP checkpoint.
# Runs on a single GPU; takes ~5 minutes for 50 batches × 64 images.
#
# Usage:
#   bash scripts/slurm_check_prototype_utilization.sh
#
# Override checkpoint via CKPT env var, e.g.:
#   CKPT=.../run_D_.../ckpt.pth bash scripts/slurm_check_prototype_utilization.sh

set -e

SCRATCH="/net/tscratch/people/plgabedychaj"
REPO=~/proto-non-param
VOCAB_DIR="${VOCAB_DIR:-${SCRATCH}/vocab}"
VG_ROOT="${VG_ROOT:-${SCRATCH}/vg}"
VG_DESC="${VG_DESC:-${VG_ROOT}/region_descriptions.json}"

# Default: run B (best single-caption baseline)
CKPT="${CKPT:-${SCRATCH}/train_logs/vg_contrastive/run_B_contrastive10_k1_30ep/ckpt.pth}"

PARTITION="plgrid-gpu-a100"
ACCOUNT="plgunhype-gpu-a100"
LOG_SLURM="${SCRATCH}/logs"
mkdir -p "${LOG_SLURM}"

JOB=$(sbatch --parsable \
  --job-name=pnp-proto-util \
  --partition="${PARTITION}" \
  --account="${ACCOUNT}" \
  --gres=gpu:1 \
  --cpus-per-task=4 \
  --mem=32G \
  --time=00:30:00 \
  --output="${LOG_SLURM}/proto_util_%j.out" \
  --error="${LOG_SLURM}/proto_util_%j.err" \
  --wrap="
set -e
source ${SCRATCH}/venv/bin/activate
export HF_HUB_CACHE=${SCRATCH}/.cache/huggingface/hub
export TRANSFORMERS_CACHE=${SCRATCH}/.cache/huggingface/hub
cd ${REPO}

python scripts/check_prototype_utilization.py \
  --ckpt ${CKPT} \
  --vocab-cache-path ${VOCAB_DIR}/vg_cache.pt \
  --vg-root ${VG_ROOT} \
  --vg-region-descriptions ${VG_DESC} \
  --n-batches 50 \
  --batch-size 64 \
  --num-workers 4
")

echo "Submitted job ${JOB}"
echo "  Checkpoint : ${CKPT}"
echo "  Log        : ${LOG_SLURM}/proto_util_${JOB}.out"
echo ""
echo "Watch with : tail -f ${LOG_SLURM}/proto_util_${JOB}.out"
