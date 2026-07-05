#!/bin/bash
# Zero-shot RIS evaluation for run K1 (ViT-L/14 + contrastive k=1 + KoLeo).
#
# Submits 7 SLURM jobs: 1 variant (K1) × 7 dataset/split combos
#
# Results land in:
#   eval_results/vg_contrastive/contr_K1/pnp_refer/{dataset}_{split}.json
#
# Usage:
#   bash scripts/slurm_eval_vg_contrastive_K1.sh

set -e

SCRATCH="/net/tscratch/people/plgabedychaj"
REPO=~/proto-non-param
CONTR_BASE="${CONTR_BASE:-${SCRATCH}/train_logs/vg_contrastive}"
DATA_ROOT="${DATA_ROOT:-${SCRATCH}/data/refcoco}"

PARTITION="plgrid-gpu-a100"
ACCOUNT="plgunhype-gpu-a100"
LOG_SLURM="${SCRATCH}/logs"

CKPT="${CONTR_BASE}/run_K1_vitl14_contrastive10_k1_koleo01_30ep/ckpt.pth"

mkdir -p "${LOG_SLURM}"

if [ ! -f "${CKPT}" ]; then
    echo "ERROR: checkpoint not found: ${CKPT}"
    exit 1
fi

OUT_DIR="${REPO}/eval_results/vg_contrastive/contr_K1"

echo "=== PNP Contrastive VG — Zero-shot RIS Evaluation (run K1, ViT-L/14 + KoLeo) ==="
echo "  Ckpt : ${CKPT}"
echo "  Data : ${DATA_ROOT}"
echo ""

declare -A SPLITS=(
  [Gref]="val"
  [unc]="val testA testB"
  [unc+]="val testA testB"
)

for DATASET in Gref unc unc+; do
    for SPLIT in ${SPLITS[$DATASET]}; do
        JOB=$(sbatch --parsable \
            --job-name="pnp-contrK1-${DATASET}-${SPLIT}" \
            --partition="${PARTITION}" \
            --account="${ACCOUNT}" \
            --gres=gpu:1 \
            --cpus-per-task=4 \
            --mem=32G \
            --time=04:00:00 \
            --output="${LOG_SLURM}/pnp_contr_K1_${DATASET}_${SPLIT}_%j.out" \
            --error="${LOG_SLURM}/pnp_contr_K1_${DATASET}_${SPLIT}_%j.err" \
            --wrap="
set -e
source ${SCRATCH}/venv/bin/activate
export HF_HUB_CACHE=${SCRATCH}/.cache/huggingface/hub
export TRANSFORMERS_CACHE=${SCRATCH}/.cache/huggingface/hub
export TORCH_HOME=${SCRATCH}/torch_cache
export PYTHONPATH=${SCRATCH}/dinov2:\$PYTHONPATH
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
