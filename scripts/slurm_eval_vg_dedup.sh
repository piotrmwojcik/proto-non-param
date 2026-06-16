#!/bin/bash
# Evaluate dedup-vocab ablation checkpoints (runs A and C) with two inference modes:
#   dedup_only  — score against the training (deduplicated) vocabulary
#   full_vocab  — score against the original full vocabulary
#
# Reports P@K and R@K for K ∈ {5, 10} for both modes, logged to W&B.
# This isolates whether the projection head generalises to concepts not seen
# during training (the fine-grained words merged away by deduplication).
#
# Prerequisites: both checkpoints must exist (from slurm_train_vg_dedup.sh).
#
# Configuration — override via environment variables:
#   CKPT_A       Checkpoint A path  (KL  + frozen residual, dedup)
#   CKPT_C       Checkpoint C path  (JSD + frozen residual, dedup)
#   SCRATCH      Base scratch path  (default: /net/tscratch/people/plgabedychaj)
#   VG_ROOT      Path to VG images
#   VG_DESC      Path to region_descriptions.json
#   VOCAB_DIR    Directory with vg_cache.pt and vg_cache_dedup_t*.pt
#   THRESHOLD    Cosine-similarity threshold used in training (default: 0.90)
#   WANDB_ENTITY W&B entity (default: gmum)
#
# Usage:
#   bash scripts/slurm_eval_vg_dedup.sh
#
#   # Override checkpoints explicitly:
#   CKPT_A=/path/to/ckpt.pth CKPT_C=/path/to/ckpt.pth \
#       bash scripts/slurm_eval_vg_dedup.sh

set -e

SCRATCH="${SCRATCH:-/net/tscratch/people/plgabedychaj}"
VG_ROOT="${VG_ROOT:-${SCRATCH}/vg}"
VG_DESC="${VG_DESC:-${VG_ROOT}/region_descriptions.json}"
VOCAB_DIR="${VOCAB_DIR:-${SCRATCH}/vocab}"
WANDB_ENTITY="${WANDB_ENTITY:-gmum}"
THRESHOLD="${THRESHOLD:-0.90}"

THRESHOLD_STR="${THRESHOLD/./}"
DEDUP_BASE="${SCRATCH}/train_logs/vg_dedup"

CKPT_A="${CKPT_A:-${DEDUP_BASE}/run_A_kl_frozen_dedup_t${THRESHOLD_STR}/ckpt.pth}"
CKPT_C="${CKPT_C:-${DEDUP_BASE}/run_C_jsd_frozen_dedup_t${THRESHOLD_STR}/ckpt.pth}"

PARTITION="plgrid-gpu-a100"
ACCOUNT="plgunhype-gpu-a100"
LOG_SLURM="${SCRATCH}/logs"

VG_CACHE_FULL="${VOCAB_DIR}/vg_cache.pt"
VG_CACHE_DEDUP="${VOCAB_DIR}/vg_cache_dedup_t${THRESHOLD_STR}.pt"

mkdir -p "${LOG_SLURM}"

echo "=== VG Dedup-Vocab Ablation Eval (θ=${THRESHOLD}) ==="
echo "  Ckpt A (KL  + frozen, dedup): ${CKPT_A}"
echo "  Ckpt C (JSD + frozen, dedup): ${CKPT_C}"
echo "  Train vocab (dedup): ${VG_CACHE_DEDUP}"
echo "  Eval  vocab (full) : ${VG_CACHE_FULL}"
echo ""

for CKPT in "${CKPT_A}" "${CKPT_C}"; do
    if [ ! -f "${CKPT}" ]; then
        echo "ERROR: checkpoint not found: ${CKPT}"
        echo "Run slurm_train_vg_dedup.sh first."
        exit 1
    fi
done

if [ ! -f "${VG_CACHE_DEDUP}" ]; then
    echo "ERROR: dedup vocab cache not found: ${VG_CACHE_DEDUP}"
    echo "Run slurm_train_vg_dedup.sh first (job 0 builds it)."
    exit 1
fi

WRAP_HEADER="
set -e
export HF_HOME=${SCRATCH}/hf_cache
export TRANSFORMERS_CACHE=${SCRATCH}/hf_cache
export PYTHONPATH=${SCRATCH}/dinov2:\$PYTHONPATH
source ${SCRATCH}/venv/bin/activate
cd ~/proto-VLM/proto-non-param
"

# -----------------------------------------------------------------------
# Helper: submit one eval job
#   $1  label (A / C)
#   $2  checkpoint path
#   $3  W&B run name
# -----------------------------------------------------------------------
submit_eval() {
    local LABEL="$1"
    local CKPT="$2"
    local RUN_NAME="$3"

    local JOB
    JOB=$(sbatch --parsable \
      --job-name="eval-dedup-${LABEL}" \
      --partition="${PARTITION}" \
      --account="${ACCOUNT}" \
      --gres=gpu:1 \
      --cpus-per-task=8 \
      --mem=32G \
      --time=02:00:00 \
      --output="${LOG_SLURM}/eval_dedup_${LABEL}_%j.out" \
      --error="${LOG_SLURM}/eval_dedup_${LABEL}_%j.err" \
      --wrap="${WRAP_HEADER}
python eval_dedup_vocab.py \\
  --ckpt              ${CKPT} \\
  --train-vocab-cache ${VG_CACHE_DEDUP} \\
  --eval-vocab-cache  ${VG_CACHE_FULL} \\
  --vg-root           ${VG_ROOT} \\
  --vg-region-descriptions ${VG_DESC} \\
  --mode both \\
  --target-mode topk \\
  --top-k-concepts 5 \\
  --batch-size 64 \\
  --num-workers 8 \\
  --wandb-entity ${WANDB_ENTITY} \\
  --wandb-run-name ${RUN_NAME}
")
    echo "Submitted eval-dedup-${LABEL}: ${JOB}"
}

submit_eval "A" "${CKPT_A}" "eval-dedup-A-kl-frozen-t${THRESHOLD_STR}"
submit_eval "C" "${CKPT_C}" "eval-dedup-C-jsd-frozen-t${THRESHOLD_STR}"

echo ""
echo "Both eval jobs submitted."
echo "Monitor with : squeue -u \$USER"
echo "Logs under   : ${LOG_SLURM}/eval_dedup_{A,C}_*.{out,err}"
echo "W&B runs     : https://wandb.ai/${WANDB_ENTITY}/proto-non-param"
echo ""
echo "Compare against non-dedup runs A/C from slurm_train_vg_ablation.sh"
echo "using the full_vocab P@K / R@K numbers."
