#!/bin/bash
# Zero-shot RIS evaluation for the two long-trained VG checkpoints (80 epochs).
#
# Submits 14 SLURM jobs: 2 variants (A/C) × 7 dataset/split combos
#   Gref : val
#   unc  : val, testA, testB
#   unc+ : val, testA, testB
#
# Results land in:
#   eval_results/vg_long/long_A/pnp_refer/{dataset}_{split}.json   (KL  + frozen, 80ep)
#   eval_results/vg_long/long_C/pnp_refer/{dataset}_{split}.json   (JSD + frozen, 80ep)
#
# Compare against the 20-epoch ablation runs A/C with:
#   python scripts/compare_ris_results.py \
#       --eval_dir eval_results \
#       --ablation-dir eval_results/vg_long \
#       --ablation-type vg_long \
#       --out eval_results/vg_long/comparison.md
#
# Configuration — override via environment variables:
#   LONG_BASE    Base dir with run_A_kl_frozen_80ep / run_C_jsd_frozen_80ep
#   DATA_ROOT    Path to refcoco/ directory
#
# Usage:
#   bash scripts/slurm_eval_pnp_vg_long.sh

set -e

SCRATCH="/net/tscratch/people/plgabedychaj"
REPO=~/proto-non-param
LONG_BASE="${LONG_BASE:-${SCRATCH}/train_logs/vg_long}"
DATA_ROOT="${DATA_ROOT:-${SCRATCH}/data/refcoco}"

PARTITION="plgrid-gpu-a100"
ACCOUNT="plgunhype-gpu-a100"
LOG_SLURM="${SCRATCH}/logs"

mkdir -p "${LOG_SLURM}"

declare -A CKPTS=(
  [A]="${LONG_BASE}/run_A_kl_frozen_80ep/ckpt.pth"
  [C]="${LONG_BASE}/run_C_jsd_frozen_80ep/ckpt.pth"
)

declare -A LABELS=(
  [A]="KL + frozen residual (80 epochs)"
  [C]="JSD + frozen residual (80 epochs)"
)

declare -A SPLITS=(
  [Gref]="val"
  [unc]="val testA testB"
  [unc+]="val testA testB"
)

echo "=== PNP Long Training — Zero-shot RIS Evaluation (80 epochs) ==="
echo "  Ckpt A : ${CKPTS[A]}"
echo "  Ckpt C : ${CKPTS[C]}"
echo "  Data   : ${DATA_ROOT}"
echo "  Results: eval_results/vg_long/long_{A,C}/pnp_refer/"
echo ""

for VARIANT in A C; do
  CKPT="${CKPTS[$VARIANT]}"
  OUT_DIR="${REPO}/eval_results/vg_long/long_${VARIANT}"
  echo "-- Variant ${VARIANT} (${LABELS[$VARIANT]}) --"

  for DATASET in Gref unc unc+; do
    for SPLIT in ${SPLITS[$DATASET]}; do
      JOB=$(sbatch --parsable \
        --job-name="pnp-long${VARIANT}-${DATASET}-${SPLIT}" \
        --partition="${PARTITION}" \
        --account="${ACCOUNT}" \
        --gres=gpu:1 \
        --cpus-per-task=4 \
        --mem=32G \
        --time=04:00:00 \
        --output="${LOG_SLURM}/pnp_long_${VARIANT}_${DATASET}_${SPLIT}_%j.out" \
        --error="${LOG_SLURM}/pnp_long_${VARIANT}_${DATASET}_${SPLIT}_%j.err" \
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

echo "All 14 jobs submitted. Monitor with: squeue -u \$USER"
echo ""
echo "After completion, generate comparison table with:"
echo "  python scripts/compare_ris_results.py \\"
echo "      --eval_dir eval_results \\"
echo "      --ablation-dir eval_results/vg_long \\"
echo "      --ablation-type vg_long \\"
echo "      --out eval_results/vg_long/comparison.md"
