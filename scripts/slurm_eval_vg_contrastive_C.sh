#!/bin/bash
# Zero-shot RIS evaluation for run C only (contrastive-only, 30 epochs).
# Use this when runs A and B are already evaluated.
#
# Submits 7 SLURM jobs: 1 variant (C) × 7 dataset/split combos
#   Gref : val
#   unc  : val, testA, testB
#   unc+ : val, testA, testB
#
# Results land in:
#   eval_results/vg_contrastive/contr_C/pnp_refer/{dataset}_{split}.json
#
# After completion, generate comparison table (A+B results must also exist):
#   python scripts/compare_ris_results.py \
#       --eval_dir eval_results \
#       --ablation-dir eval_results/vg_contrastive \
#       --ablation-type vg_contrastive \
#       --out eval_results/vg_contrastive/comparison.md
#
# Configuration — override via environment variables:
#   CONTR_BASE   Base dir containing run_C_uniform_contrastive_only_30ep/
#   DATA_ROOT    Path to refcoco/ directory
#
# Usage:
#   bash scripts/slurm_eval_vg_contrastive_C.sh

set -e

SCRATCH="/net/tscratch/people/plgabedychaj"
REPO=~/proto-non-param
CONTR_BASE="${CONTR_BASE:-${SCRATCH}/train_logs/vg_contrastive}"
DATA_ROOT="${DATA_ROOT:-${SCRATCH}/data/refcoco}"

PARTITION="plgrid-gpu-a100"
ACCOUNT="plgunhype-gpu-a100"
LOG_SLURM="${SCRATCH}/logs"

CKPT="${CONTR_BASE}/run_C_uniform_contrastive_only_30ep/ckpt.pth"
OUT_DIR="${REPO}/eval_results/vg_contrastive/contr_C"

mkdir -p "${LOG_SLURM}"

if [ ! -f "${CKPT}" ]; then
    echo "ERROR: checkpoint not found: ${CKPT}"
    exit 1
fi

declare -A SPLITS=(
  [Gref]="val"
  [unc]="val testA testB"
  [unc+]="val testA testB"
)

echo "=== PNP Contrastive-Only (run C) — Zero-shot RIS Evaluation ==="
echo "  Ckpt : ${CKPT}"
echo "  Data : ${DATA_ROOT}"
echo "  Out  : ${OUT_DIR}/pnp_refer/"
echo ""

for DATASET in Gref unc unc+; do
  for SPLIT in ${SPLITS[$DATASET]}; do
    JOB=$(sbatch --parsable \
      --job-name="pnp-contrC-${DATASET}-${SPLIT}" \
      --partition="${PARTITION}" \
      --account="${ACCOUNT}" \
      --gres=gpu:1 \
      --cpus-per-task=4 \
      --mem=32G \
      --time=04:00:00 \
      --output="${LOG_SLURM}/pnp_contr_C_${DATASET}_${SPLIT}_%j.out" \
      --error="${LOG_SLURM}/pnp_contr_C_${DATASET}_${SPLIT}_%j.err" \
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
echo "7 jobs submitted. Monitor with: squeue -u \$USER"
echo ""
echo "After all jobs finish, compare against SaG and CTRL-O:"
echo "  cd ~/proto-non-param"
echo "  python scripts/compare_ris_results.py \\"
echo "      --eval_dir eval_results \\"
echo "      --ablation-dir eval_results/vg_contrastive \\"
echo "      --ablation-type vg_contrastive \\"
echo "      --out eval_results/vg_contrastive/comparison.md"
