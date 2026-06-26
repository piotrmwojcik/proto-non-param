#!/bin/bash
# Zero-shot RIS evaluation for the 2×2 VG ablation checkpoints.
#
# Submits 28 SLURM jobs: 4 variants (A/B/C/D) × 7 dataset/split combos
#   Gref : val
#   unc  : val, testA, testB
#   unc+ : val, testA, testB
#
# Results land in (relative to $REPO):
#   eval_results/vg_ablation/ablation_A/pnp_refer/{dataset}_{split}.json   (KL  + frozen residual)
#   eval_results/vg_ablation/ablation_B/pnp_refer/{dataset}_{split}.json   (KL  + trained residual)
#   eval_results/vg_ablation/ablation_C/pnp_refer/{dataset}_{split}.json   (JSD + frozen residual)
#   eval_results/vg_ablation/ablation_D/pnp_refer/{dataset}_{split}.json   (JSD + trained residual)
#
# After all jobs finish, generate the comparison table with:
#   python scripts/compare_ris_results.py \
#       --eval_dir eval_results \
#       --ablation-dir eval_results/vg_ablation \
#       --ablation-type vg_ablation \
#       --out eval_results/vg_ablation/comparison.md
#
# Configuration — override via environment variables:
#   ABLATION_BASE   Base dir with run_A_kl_frozen / run_B_kl_residual / ... subdirs
#   DATA_ROOT       Path to refcoco/ directory (must contain Gref/, unc/, unc+ .npz batches)
#
# Usage:
#   bash scripts/slurm_eval_pnp_vg_ablation.sh

set -e

SCRATCH="/net/tscratch/people/plgabedychaj"
REPO=~/proto-non-param
ABLATION_BASE="${ABLATION_BASE:-${SCRATCH}/train_logs/vg_ablation}"
DATA_ROOT="${DATA_ROOT:-${SCRATCH}/data/refcoco}"

PARTITION="plgrid-gpu-a100"
ACCOUNT="plgunhype-gpu-a100"
LOG_SLURM="${SCRATCH}/logs"

mkdir -p "${LOG_SLURM}"

declare -A CKPTS=(
  [A]="${ABLATION_BASE}/run_A_kl_frozen/ckpt.pth"
  [B]="${ABLATION_BASE}/run_B_kl_residual/ckpt.pth"
  [C]="${ABLATION_BASE}/run_C_jsd_frozen/ckpt.pth"
  [D]="${ABLATION_BASE}/run_D_jsd_residual/ckpt.pth"
)

declare -A LABELS=(
  [A]="KL + frozen residual"
  [B]="KL + trained residual"
  [C]="JSD + frozen residual"
  [D]="JSD + trained residual"
)

# dataset -> space-separated splits
declare -A SPLITS=(
  [Gref]="val"
  [unc]="val testA testB"
  [unc+]="val testA testB"
)

echo "=== PNP VG Ablation — Zero-shot RIS Evaluation ==="
echo "  Checkpoints : ${ABLATION_BASE}/run_{A,B,C,D}*/ckpt.pth"
echo "  Data root   : ${DATA_ROOT}"
echo "  Results     : eval_results/vg_ablation/ablation_{A,B,C,D}/pnp_refer/"
echo ""

for VARIANT in A B C D; do
  CKPT="${CKPTS[$VARIANT]}"
  OUT_DIR="${REPO}/eval_results/vg_ablation/ablation_${VARIANT}"
  echo "-- Variant ${VARIANT} (${LABELS[$VARIANT]}) --"

  for DATASET in Gref unc unc+; do
    for SPLIT in ${SPLITS[$DATASET]}; do
      JOB=$(sbatch --parsable \
        --job-name="pnp-vg${VARIANT}-${DATASET}-${SPLIT}" \
        --partition="${PARTITION}" \
        --account="${ACCOUNT}" \
        --gres=gpu:1 \
        --cpus-per-task=4 \
        --mem=32G \
        --time=04:00:00 \
        --output="${LOG_SLURM}/pnp_vgabl_${VARIANT}_${DATASET}_${SPLIT}_%j.out" \
        --error="${LOG_SLURM}/pnp_vgabl_${VARIANT}_${DATASET}_${SPLIT}_%j.err" \
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

echo "All 28 jobs submitted. Monitor with: squeue -u \$USER"
echo ""
echo "After completion, generate comparison table with:"
echo "  python scripts/compare_ris_results.py \\"
echo "      --eval_dir eval_results \\"
echo "      --ablation-dir eval_results/vg_ablation \\"
echo "      --ablation-type vg_ablation \\"
echo "      --out eval_results/vg_ablation/comparison.md"
