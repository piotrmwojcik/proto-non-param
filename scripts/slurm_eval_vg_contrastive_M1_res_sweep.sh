#!/bin/bash
# Resolution sweep on the existing M1 checkpoint — probe where the 448px gain saturates.
#
# 448px gave +0.85-0.91 oIoU / +1.14-1.35 mIoU uniformly across all 7 splits
# (largest single gain of the series, no retraining). Grid sizes: 224→16×16,
# 448→32×32, 672→48×48, 896→64×64. Attention cost grows ~quadratically with
# patch count but these are batch-1 eval jobs — fine on A100, generous time limit.
#
# Probe on the three val splits only; promote the winner to all 7 splits
# afterwards (like slurm_eval_vg_contrastive_M1_res448.sh).
#
# Usage:
#   bash scripts/slurm_eval_vg_contrastive_M1_res_sweep.sh
#   SIZES="560" bash scripts/slurm_eval_vg_contrastive_M1_res_sweep.sh   # custom sizes
#
# Compare (per size):
#   SI_DIR=eval_results/vg_contrastive/contr_M1_res672/pnp_refer \
#   bash scripts/compare_single_instance_eval.sh

set -e

SCRATCH="/net/tscratch/people/plgabedychaj"
REPO=~/proto-non-param
CONTR_BASE="${CONTR_BASE:-${SCRATCH}/train_logs/vg_contrastive}"
DATA_ROOT="${DATA_ROOT:-${SCRATCH}/data/refcoco}"
SIZES="${SIZES:-672 896}"

PARTITION="plgrid-gpu-a100"
ACCOUNT="plgunhype-gpu-a100"
LOG_SLURM="${SCRATCH}/logs"

CKPT="${CONTR_BASE}/run_M1_vitl14_sk10_koleo01_30ep/ckpt.pth"

mkdir -p "${LOG_SLURM}"

if [ ! -f "${CKPT}" ]; then
    echo "ERROR: checkpoint not found: ${CKPT}"
    exit 1
fi

for SIZE in ${SIZES}; do
    if [ $((SIZE % 14)) -ne 0 ]; then
        echo "ERROR: ${SIZE} is not a multiple of 14 (ViT patch size) — skipping"
        continue
    fi

    OUT_DIR="${REPO}/eval_results/vg_contrastive/contr_M1_res${SIZE}"
    GRID=$((SIZE / 14))

    echo ""
    echo "=== PNP-M1 @ ${SIZE}px (${GRID}×${GRID} patch grid) ==="

    for DATASET in Gref unc unc+; do
        SPLIT="val"
        if [ -f "${OUT_DIR}/pnp_refer/${DATASET}_${SPLIT}.json" ]; then
            echo "  skip   ${DATASET}/${SPLIT} (result exists)"
            continue
        fi
        JOB=$(sbatch --parsable \
            --job-name="pnp-M1r${SIZE}-${DATASET}-${SPLIT}" \
            --partition="${PARTITION}" \
            --account="${ACCOUNT}" \
            --gres=gpu:1 \
            --cpus-per-task=4 \
            --mem=48G \
            --time=08:00:00 \
            --output="${LOG_SLURM}/pnp_M1r${SIZE}_${DATASET}_${SPLIT}_%j.out" \
            --error="${LOG_SLURM}/pnp_M1r${SIZE}_${DATASET}_${SPLIT}_%j.err" \
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
  --img-size ${SIZE}
")
        echo "  ${JOB}  ${DATASET}/${SPLIT}"
    done
done

echo ""
echo "Jobs submitted. Monitor with: squeue -u \$USER"
