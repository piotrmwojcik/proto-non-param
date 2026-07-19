#!/bin/bash
# Fill in the remaining splits for M1 @ 672px (testA/testB of unc and unc+).
#
# The res-sweep probe already produced val results (+1.0-1.1 oIoU, +1.5-1.6 mIoU
# over the 224px baseline — better than 448px on every val split); this completes
# the 7-split row. Skips any split whose result JSON already exists, so it is
# safe to run alongside/after the sweep script.
#
# Usage:
#   bash scripts/slurm_eval_vg_contrastive_M1_res672_full.sh

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

OUT_DIR="${REPO}/eval_results/vg_contrastive/contr_M1_res672"

echo "=== PNP-M1 @ 672px (48×48 patch grid) — remaining splits ==="
echo "  Ckpt : ${CKPT}"
echo ""

declare -A SPLITS=(
  [Gref]="val"
  [unc]="val testA testB"
  [unc+]="val testA testB"
)

for DATASET in Gref unc unc+; do
    for SPLIT in ${SPLITS[$DATASET]}; do
        if [ -f "${OUT_DIR}/pnp_refer/${DATASET}_${SPLIT}.json" ]; then
            echo "  skip   ${DATASET}/${SPLIT} (result exists)"
            continue
        fi
        JOB=$(sbatch --parsable \
            --job-name="pnp-M1r672-${DATASET}-${SPLIT}" \
            --partition="${PARTITION}" \
            --account="${ACCOUNT}" \
            --gres=gpu:1 \
            --cpus-per-task=4 \
            --mem=48G \
            --time=08:00:00 \
            --output="${LOG_SLURM}/pnp_M1r672_${DATASET}_${SPLIT}_%j.out" \
            --error="${LOG_SLURM}/pnp_M1r672_${DATASET}_${SPLIT}_%j.err" \
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
  --img-size 672
")
        echo "  ${JOB}  ${DATASET}/${SPLIT}"
    done
done

echo ""
echo "Jobs submitted (existing results skipped). Monitor with: squeue -u \$USER"
