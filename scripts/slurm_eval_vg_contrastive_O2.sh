#!/bin/bash
# Zero-shot RIS evaluation for run O2 (ViT-L/14 + SK λ=0.3 + SigReg λ=0.5, no contrastive).
#
# Usage:
#   bash scripts/slurm_eval_vg_contrastive_O2.sh

set -e

SCRATCH="/net/tscratch/people/plgabedychaj"
REPO=~/proto-non-param
CONTR_BASE="${CONTR_BASE:-${SCRATCH}/train_logs/vg_contrastive}"
DATA_ROOT="${DATA_ROOT:-${SCRATCH}/data/refcoco}"

PARTITION="plgrid-gpu-a100"
ACCOUNT="plgunhype-gpu-a100"
LOG_SLURM="${SCRATCH}/logs"

CKPT="${CONTR_BASE}/run_O2_vitl14_sk03_sigreg05_30ep/ckpt.pth"

mkdir -p "${LOG_SLURM}"

if [ ! -f "${CKPT}" ]; then
    echo "ERROR: checkpoint not found: ${CKPT}"
    exit 1
fi

OUT_DIR="${REPO}/eval_results/vg_contrastive/contr_O2"

echo "=== PNP — Zero-shot RIS Evaluation (run O2, ViT-L/14 + SK λ=0.3 + SigReg λ=0.5, no contrastive) ==="
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
            --job-name="pnp-O2-${DATASET}-${SPLIT}" \
            --partition="${PARTITION}" \
            --account="${ACCOUNT}" \
            --gres=gpu:1 \
            --cpus-per-task=4 \
            --mem=32G \
            --time=04:00:00 \
            --output="${LOG_SLURM}/pnp_O2_${DATASET}_${SPLIT}_%j.out" \
            --error="${LOG_SLURM}/pnp_O2_${DATASET}_${SPLIT}_%j.err" \
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
