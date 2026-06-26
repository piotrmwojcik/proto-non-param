#!/bin/bash
# Zero-shot RIS evaluation for all three caption-signal ablation checkpoints.
#
# Submits 21 SLURM jobs: 3 variants (A/B/C) × 7 dataset/split combos
#   Gref : val
#   unc  : val, testA, testB
#   unc+ : val, testA, testB
#
# Results land in:
#   eval_results/ablation_A/pnp_refer/{dataset}_{split}.json
#   eval_results/ablation_B/pnp_refer/{dataset}_{split}.json
#   eval_results/ablation_C/pnp_refer/{dataset}_{split}.json
#
# After all jobs finish, generate the comparison table with:
#   python scripts/compare_ris_results.py --ablation-dir eval_results
#
# Configuration — override via environment variables:
#   LOG_DIR   Base dir containing run_A_word_only / run_B_caption_only / run_C_combined
#   DATA_ROOT Path to refcoco/ directory
#
# Usage:
#   bash scripts/slurm_eval_pnp_caption_ablation.sh

set -e

SCRATCH="/net/tscratch/people/plgabedychaj"
REPO=~/proto-non-param
LOG_DIR="${LOG_DIR:-${SCRATCH}/train_logs/vg_caption_signal_ablation}"
DATA_ROOT="${DATA_ROOT:-${SCRATCH}/data/refcoco}"

PARTITION="plgrid-gpu-a100"
ACCOUNT="plgunhype-gpu-a100"
LOG_SLURM="${SCRATCH}/logs"

mkdir -p "${LOG_SLURM}"

declare -A CKPTS=(
  [A]="${LOG_DIR}/run_A_word_only/ckpt.pth"
  [B]="${LOG_DIR}/run_B_caption_only/ckpt.pth"
  [C]="${LOG_DIR}/run_C_combined/ckpt.pth"
)

declare -A LABELS=(
  [A]="word-only"
  [B]="caption-only"
  [C]="combined"
)

# dataset -> space-separated splits
declare -A SPLITS=(
  [Gref]="val"
  [unc]="val testA testB"
  [unc+]="val testA testB"
)

echo "=== PNP Caption Signal Ablation — RIS Evaluation ==="
echo "  Checkpoints : ${LOG_DIR}/run_{A,B,C}*/ckpt.pth"
echo "  Data root   : ${DATA_ROOT}"
echo "  Results     : eval_results/ablation_{A,B,C}/pnp_refer/"
echo ""

for VARIANT in A B C; do
  CKPT="${CKPTS[$VARIANT]}"
  OUT_DIR="${REPO}/eval_results/ablation_${VARIANT}"
  echo "-- Variant ${VARIANT} (${LABELS[$VARIANT]}) --"

  for DATASET in Gref unc unc+; do
    for SPLIT in ${SPLITS[$DATASET]}; do
      JOB=$(sbatch --parsable \
        --job-name="pnp-${VARIANT}-${DATASET}-${SPLIT}" \
        --partition="${PARTITION}" \
        --account="${ACCOUNT}" \
        --gres=gpu:1 \
        --cpus-per-task=4 \
        --mem=32G \
        --time=04:00:00 \
        --output="${LOG_SLURM}/pnp_abl_${VARIANT}_${DATASET}_${SPLIT}_%j.out" \
        --error="${LOG_SLURM}/pnp_abl_${VARIANT}_${DATASET}_${SPLIT}_%j.err" \
        --wrap="
set -e
source ${SCRATCH}/venv/bin/activate
export HF_HUB_CACHE=${SCRATCH}/.cache/huggingface/hub
export TRANSFORMERS_CACHE=${SCRATCH}/.cache/huggingface/hub
cd ${REPO}

python scripts/evaluate_pnp_refer.py \
  --ckpt ${CKPT} \
  --dataset ${DATASET} \
  --data_split ${SPLIT} \
  --data_root ${DATA_ROOT} \
  --out_dir ${OUT_DIR}
")
      echo "  ${JOB}  ${DATASET}/${SPLIT}"
    done
  done
  echo ""
done

echo "All 21 jobs submitted. Monitor with: squeue -u \$USER"
echo ""
echo "After completion, generate comparison table with:"
echo "  python scripts/compare_ris_results.py --ablation-dir eval_results"
