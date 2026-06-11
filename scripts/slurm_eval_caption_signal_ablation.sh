#!/bin/bash
# Evaluate all three caption-signal ablation checkpoints in separate W&B runs.
#
# Submits 3 independent SLURM jobs:
#   A  eval-sig-A   Word top-5 only checkpoint
#   B  eval-sig-B   Caption-only checkpoint
#   C  eval-sig-C   Combined checkpoint
#
# Each job runs eval_augmented_prototypes.py --mode both so results are logged
# for both word-level and caption-level prototype scoring in W&B.
#
# Configuration — override via environment variables:
#   LOG_DIR          Base directory where training checkpoints live
#   VOCAB_DIR        Directory containing vg_cache.pt and vg_test_caption_prototypes.pt
#   VG_ROOT          Path to VG image root
#   VG_DESC          Path to region_descriptions.json
#   WANDB_ENTITY     W&B entity (team or username)
#
# Usage:
#   bash scripts/slurm_eval_caption_signal_ablation.sh

set -e

SCRATCH="/net/tscratch/people/plgabedychaj"
LOG_DIR="${LOG_DIR:-${SCRATCH}/train_logs/vg_caption_signal_ablation}"
VOCAB_DIR="${VOCAB_DIR:-${SCRATCH}/vocab}"
VG_ROOT="${VG_ROOT:-${SCRATCH}/vg}"
VG_DESC="${VG_DESC:-${VG_ROOT}/region_descriptions.json}"
WANDB_ENTITY="${WANDB_ENTITY:-gmum}"

PARTITION="plgrid-gpu-a100"
ACCOUNT="plgunhype-gpu-a100"
LOG_SLURM="${SCRATCH}/logs"

VG_CACHE="${VOCAB_DIR}/vg_cache.pt"
CAPTION_PROTO="${VOCAB_DIR}/vg_test_caption_prototypes.pt"

mkdir -p "${LOG_SLURM}"

echo "=== Caption Signal Ablation — Evaluation ==="
echo "  Checkpoints : ${LOG_DIR}/run_{A,B,C}*/ckpt.pth"
echo "  Caption proto: ${CAPTION_PROTO}"
echo ""

# -----------------------------------------------------------------------
# Helper: submit one eval job
#   $1  label (A/B/C)
#   $2  run subdir name
#   $3  W&B run name
# -----------------------------------------------------------------------
submit_eval() {
  local LABEL="$1"
  local RUN_DIR="$2"
  local RUN_NAME="$3"
  local CKPT="${LOG_DIR}/${RUN_DIR}/ckpt.pth"

  local JOB
  JOB=$(sbatch --parsable \
    --job-name="eval-sig-${LABEL}" \
    --partition="${PARTITION}" \
    --account="${ACCOUNT}" \
    --gres=gpu:1 \
    --cpus-per-task=8 \
    --mem=32G \
    --time=02:00:00 \
    --output="${LOG_SLURM}/eval_sig_${LABEL}_%j.out" \
    --error="${LOG_SLURM}/eval_sig_${LABEL}_%j.err" \
    --wrap="
set -e
export HF_HOME=${SCRATCH}/hf_cache
export PYTHONPATH=${SCRATCH}/dinov2:\$PYTHONPATH
source ${SCRATCH}/venv/bin/activate
cd ~/proto-VLM/proto-non-param

python eval_augmented_prototypes.py \\
  --ckpt ${CKPT} \\
  --vocab-cache-path ${VG_CACHE} \\
  --caption-prototypes-path ${CAPTION_PROTO} \\
  --source-dataset vg_test \\
  --vg-root ${VG_ROOT} \\
  --vg-region-descriptions ${VG_DESC} \\
  --mode both \\
  --topk 5 \\
  --batch-size 64 \\
  --num-workers 8 \\
  --wandb-entity ${WANDB_ENTITY} \\
  --wandb-run-name ${RUN_NAME}
")
  echo "Submitted eval-sig-${LABEL}: ${JOB}  (ckpt: ${RUN_DIR}/ckpt.pth)"
}

submit_eval "A" "run_A_word_only"    "caption-ablation-A-word-only"
submit_eval "B" "run_B_caption_only" "caption-ablation-B-caption-only"
submit_eval "C" "run_C_combined"     "caption-ablation-C-combined"

echo ""
echo "All 3 eval jobs submitted. Monitor with: squeue -u \$USER"
echo "Logs: ${LOG_SLURM}/eval_sig_{A,B,C}_*.{out,err}"
