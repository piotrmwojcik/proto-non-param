#!/bin/bash
# Zero-shot RIS evaluation for runs D and E only.
# Use this when runs A–C are already evaluated.
#
# Submits 14 SLURM jobs: 2 variants (D/E) × 7 dataset/split combos
#   Gref : val
#   unc  : val, testA, testB
#   unc+ : val, testA, testB
#
# Variants:
#   D — k=5 random-averaged phrases, hard InfoNCE (30 epochs)
#   E — k=5, soft negatives tau_label=0.5 (30 epochs)
#
# Results land in:
#   eval_results/vg_contrastive/contr_D/pnp_refer/{dataset}_{split}.json
#   eval_results/vg_contrastive/contr_E/pnp_refer/{dataset}_{split}.json
#
# After completion, generate comparison table (A/B results must also exist):
#   python scripts/compare_ris_results.py \
#       --eval_dir eval_results \
#       --ablation-dir eval_results/vg_contrastive \
#       --ablation-type vg_contrastive \
#       --out eval_results/vg_contrastive/comparison.md
#
# Configuration — override via environment variables:
#   CONTR_BASE   Base dir containing run_D_* / run_E_* subdirs
#   DATA_ROOT    Path to refcoco/ directory
#
# Usage:
#   bash scripts/slurm_eval_vg_contrastive_DE.sh

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
  [D]="${CONTR_BASE}/run_D_contrastive10_k5_30ep/ckpt.pth"
  [E]="${CONTR_BASE}/run_E_contrastive10_k5_soft05_30ep/ckpt.pth"
)

declare -A LABELS=(
  [D]="k=5, hard InfoNCE (30 epochs)"
  [E]="k=5, soft negatives tau=0.5 (30 epochs)"
)

declare -A SPLITS=(
  [Gref]="val"
  [unc]="val testA testB"
  [unc+]="val testA testB"
)

echo "=== PNP Contrastive VG — Zero-shot RIS Evaluation (runs D/E, 30 epochs) ==="
echo "  Ckpt D : ${CKPTS[D]}"
echo "  Ckpt E : ${CKPTS[E]}"
echo "  Data   : ${DATA_ROOT}"
echo "  Results: eval_results/vg_contrastive/contr_{D,E}/pnp_refer/"
echo ""

for VARIANT in D E; do
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
echo "After all jobs finish, compare against baselines:"
echo "  cd ~/proto-non-param"
echo "  python scripts/compare_ris_results.py \\"
echo "      --eval_dir eval_results \\"
echo "      --ablation-dir eval_results/vg_contrastive \\"
echo "      --ablation-type vg_contrastive \\"
echo "      --out eval_results/vg_contrastive/comparison.md"
