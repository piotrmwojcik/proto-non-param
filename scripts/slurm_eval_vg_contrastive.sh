#!/bin/bash
# Zero-shot RIS evaluation for the two contrastive VG training runs (30 epochs).
#
# Submits 14 SLURM jobs: 2 variants (A/B) × 7 dataset/split combos
#   Gref : val
#   unc  : val, testA, testB
#   unc+ : val, testA, testB
#
# Variants:
#   A — uniform distribution + contrastive_coef=0.5  (30 epochs)
#   B — uniform distribution + contrastive_coef=1.0  (30 epochs)
#
# Results land in:
#   eval_results/vg_contrastive/contr_A/pnp_refer/{dataset}_{split}.json
#   eval_results/vg_contrastive/contr_B/pnp_refer/{dataset}_{split}.json
#
# After completion, generate comparison table:
#   python scripts/compare_ris_results.py \
#       --eval_dir eval_results \
#       --ablation-dir eval_results/vg_contrastive \
#       --ablation-type vg_contrastive \
#       --out eval_results/vg_contrastive/comparison.md
#
# Configuration — override via environment variables:
#   CONTR_BASE   Base dir with run_A_* / run_B_* subdirs
#   DATA_ROOT    Path to refcoco/ directory
#
# Usage:
#   bash scripts/slurm_eval_vg_contrastive.sh

set -e

SCRATCH="/net/tscratch/people/plgabedychaj"
REPO=~/proto-non-param
CONTR_BASE="${CONTR_BASE:-${SCRATCH}/train_logs/vg_contrastive}"
DATA_ROOT="${DATA_ROOT:-${SCRATCH}/data/refcoco}"

PARTITION="plgrid-gpu-a100"
ACCOUNT="plgunhype-gpu-a100"
LOG_SLURM="${SCRATCH}/logs"

mkdir -p "${LOG_SLURM}"

declare -A CKPTS=(
  [A]="${CONTR_BASE}/run_A_uniform_contrastive05_30ep/ckpt.pth"
  [B]="${CONTR_BASE}/run_B_uniform_contrastive10_30ep/ckpt.pth"
)

declare -A LABELS=(
  [A]="uniform + contrastive=0.5 (30 epochs)"
  [B]="uniform + contrastive=1.0 (30 epochs)"
)

declare -A SPLITS=(
  [Gref]="val"
  [unc]="val testA testB"
  [unc+]="val testA testB"
)

echo "=== PNP Contrastive VG — Zero-shot RIS Evaluation (30 epochs) ==="
echo "  Ckpt A : ${CKPTS[A]}"
echo "  Ckpt B : ${CKPTS[B]}"
echo "  Data   : ${DATA_ROOT}"
echo "  Results: eval_results/vg_contrastive/contr_{A,B}/pnp_refer/"
echo ""

for VARIANT in A B; do
  CKPT="${CKPTS[$VARIANT]}"
  if [ ! -f "${CKPT}" ]; then
    echo "ERROR: checkpoint not found for variant ${VARIANT}: ${CKPT}"
    exit 1
  fi

  OUT_DIR="${REPO}/eval_results/vg_contrastive/contr_${VARIANT}"
  echo "-- Variant ${VARIANT} (${LABELS[$VARIANT]}) --"

  for DATASET in Gref unc unc+; do
    for SPLIT in ${SPLITS[$DATASET]}; do
      JOB=$(sbatch --parsable \
        --job-name="pnp-contr${VARIANT}-${DATASET}-${SPLIT}" \
        --partition="${PARTITION}" \
        --account="${ACCOUNT}" \
        --gres=gpu:1 \
        --cpus-per-task=4 \
        --mem=32G \
        --time=04:00:00 \
        --output="${LOG_SLURM}/pnp_contr_${VARIANT}_${DATASET}_${SPLIT}_%j.out" \
        --error="${LOG_SLURM}/pnp_contr_${VARIANT}_${DATASET}_${SPLIT}_%j.err" \
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

echo "14 jobs submitted. Monitor with: squeue -u \$USER"
echo ""
echo "After completion, generate comparison table with:"
echo "  python scripts/compare_ris_results.py \\"
echo "      --eval_dir eval_results \\"
echo "      --ablation-dir eval_results/vg_contrastive \\"
echo "      --ablation-type vg_contrastive \\"
echo "      --out eval_results/vg_contrastive/comparison.md"
