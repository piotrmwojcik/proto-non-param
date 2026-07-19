#!/bin/bash
# Tier-0 experiment: evaluate the existing M1 checkpoint at 448px instead of 224px.
#
# 224px → 16×16 patch grid (~14px/patch): sub-patch objects ("white purse",
# "watch on his left hand") are structurally unlocalizable — 7/10 hardest Gref
# failures were this pattern. 448px → 32×32 grid, 4× spatial resolution, same
# checkpoint (DINOv2 interpolates positional embeddings). No retraining.
#
# Usage:
#   bash scripts/slurm_eval_vg_contrastive_M1_res448.sh
#
# Compare with:
#   BASE_DIR=eval_results/vg_contrastive/contr_M1/pnp_refer \
#   SI_DIR=eval_results/vg_contrastive/contr_M1_res448/pnp_refer \
#   bash scripts/compare_single_instance_eval.sh

set -e

SCRATCH="/net/tscratch/people/plgabedychaj"
REPO=~/proto-non-param
CONTR_BASE="${CONTR_BASE:-${SCRATCH}/train_logs/vg_contrastive}"
DATA_ROOT="${DATA_ROOT:-${SCRATCH}/data/refcoco}"

PARTITION="plgrid-gpu-a100"
ACCOUNT="plgunhype-gpu-a100"
LOG_SLURM="${SCRATCH}/logs"

CKPT="${CONTR_BASE}/run_M1_vitl14_sk10_koleo01_30ep/ckpt.pth"

mkdir -p "${LOG_SLURM}"

if [ ! -f "${CKPT}" ]; then
    echo "ERROR: checkpoint not found: ${CKPT}"
    exit 1
fi

OUT_DIR="${REPO}/eval_results/vg_contrastive/contr_M1_res448"

echo "=== PNP — Zero-shot RIS Evaluation (run M1 @ 448px, 32×32 patch grid) ==="
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
        # val splits confirmed +1.2-1.3 mIoU — skip any split already evaluated
        if [ -f "${OUT_DIR}/pnp_refer/${DATASET}_${SPLIT}.json" ]; then
            echo "  skip   ${DATASET}/${SPLIT} (result exists)"
            continue
        fi
        JOB=$(sbatch --parsable \
            --job-name="pnp-M1r448-${DATASET}-${SPLIT}" \
            --partition="${PARTITION}" \
            --account="${ACCOUNT}" \
            --gres=gpu:1 \
            --cpus-per-task=4 \
            --mem=32G \
            --time=04:00:00 \
            --output="${LOG_SLURM}/pnp_M1r448_${DATASET}_${SPLIT}_%j.out" \
            --error="${LOG_SLURM}/pnp_M1r448_${DATASET}_${SPLIT}_%j.err" \
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
  --out_dir ${OUT_DIR} \
  --img-size 448
")
        echo "  ${JOB}  ${DATASET}/${SPLIT}"
    done
done

echo ""
echo "Jobs submitted (existing results skipped). Monitor with: squeue -u \$USER"
