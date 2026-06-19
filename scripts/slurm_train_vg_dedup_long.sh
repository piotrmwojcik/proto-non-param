#!/bin/bash
# Longer training for PNP-A-dedup (KL + frozen residual, deduplicated vocab).
#
# The 20-epoch dedup run (slurm_train_vg_dedup.sh) had not fully converged
# at ep19 (test loss still decreasing). This script runs 80 epochs with an
# explicit cosine LR schedule + 5-epoch linear warmup for maximum performance.
#
# ~9 min/epoch × 80 epochs ≈ 12 h total; wall time set to 14 h.
#
# Assumes the dedup vocab cache already exists (built by slurm_train_vg_dedup.sh).
# Checkpoint saved to: ${LOG_DIR}/run_A_kl_frozen_dedup_t${THRESHOLD_STR}_80ep/
#
# After training, evaluate with:
#   EP_SUFFIX=_80ep bash ~/proto-VLM/scripts/slurm_eval_pnp_vg_dedup.sh
#
# Configuration — override via environment variables:
#   SCRATCH       Base scratch path (default: /net/tscratch/people/plgabedychaj)
#   VG_ROOT       Path to VG image root (containing VG_100K/ and VG_100K_2/)
#   VG_DESC       Path to region_descriptions.json
#   LOG_DIR       Base directory for checkpoints
#   VOCAB_DIR     Directory containing vg_cache_dedup_t090.pt
#   THRESHOLD     Cosine-similarity threshold used when building dedup cache (default: 0.90)
#   WANDB_ENTITY  W&B entity (default: gmum)
#
# Usage:
#   bash scripts/slurm_train_vg_dedup_long.sh

set -e

SCRATCH="${SCRATCH:-/net/tscratch/people/plgabedychaj}"
VG_ROOT="${VG_ROOT:-${SCRATCH}/vg}"
VG_DESC="${VG_DESC:-${VG_ROOT}/region_descriptions.json}"
LOG_DIR="${LOG_DIR:-${SCRATCH}/train_logs/vg_dedup}"
VOCAB_DIR="${VOCAB_DIR:-${SCRATCH}/vocab}"
WANDB_ENTITY="${WANDB_ENTITY:-gmum}"
THRESHOLD="${THRESHOLD:-0.90}"

PARTITION="plgrid-gpu-a100"
ACCOUNT="plgunhype-gpu-a100"
LOG_SLURM="${SCRATCH}/logs"

THRESHOLD_STR="${THRESHOLD/./}"          # "0.90" → "090"
DEDUP_CACHE="${VOCAB_DIR}/vg_cache_dedup_t${THRESHOLD_STR}.pt"

mkdir -p "${LOG_SLURM}" "${LOG_DIR}"

if [ ! -f "${DEDUP_CACHE}" ]; then
    echo "ERROR: Dedup vocab cache not found at ${DEDUP_CACHE}"
    echo "Build it first: bash scripts/slurm_train_vg_dedup.sh"
    exit 1
fi

echo "=== PNP-A-dedup Long Training (KL + frozen δ, θ=${THRESHOLD}, 80 epochs) ==="
echo "  SCRATCH      : ${SCRATCH}"
echo "  VG_ROOT      : ${VG_ROOT}"
echo "  LOG_DIR      : ${LOG_DIR}"
echo "  Dedup vocab  : ${DEDUP_CACHE}"
echo "  Checkpoint   : ${LOG_DIR}/run_A_kl_frozen_dedup_t${THRESHOLD_STR}_80ep/"
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
  --job-name=vg-dedup-A-80ep \
  --partition="${PARTITION}" \
  --account="${ACCOUNT}" \
  --gres=gpu:1 \
  --cpus-per-task=8 \
  --mem=64G \
  --time=14:00:00 \
  --output="${LOG_SLURM}/vg_dedup_A_80ep_%j.out" \
  --error="${LOG_SLURM}/vg_dedup_A_80ep_%j.err" \
  --wrap="${WRAP_HEADER}
python train.py \\
  --dataset visual_genome \\
  --vg-root ${VG_ROOT} \\
  --vg-region-descriptions ${VG_DESC} \\
  --vocab-cache-path ${DEDUP_CACHE} \\
  --backbone dinov2_vitb14 \\
  --batch-size 64 \\
  --epochs 80 \\
  --lr-schedule cosine \\
  --lr-warmup-epochs 5 \\
  --num-workers 8 \\
  --backbone-lr 1e-5 \\
  --text-proj-lr 1e-4 \\
  --target-mode topk \\
  --top-k-concepts 5 \\
  --kl-coef 1.0 \\
  --loss-type kl \\
  --save-every 10 \\
  --wandb-entity ${WANDB_ENTITY} \\
  --wandb-log-images 16 \\
  --log-dir ${LOG_DIR}/run_A_kl_frozen_dedup_t${THRESHOLD_STR}_80ep
")

echo "Submitted job ${JOB_A}"
echo ""
echo "Monitor with : squeue -u \$USER"
echo "Log          : ${LOG_SLURM}/vg_dedup_A_80ep_${JOB_A}.out"
echo ""
echo "When complete, evaluate with:"
echo "  EP_SUFFIX=_80ep bash ~/proto-VLM/scripts/slurm_eval_pnp_vg_dedup.sh"
