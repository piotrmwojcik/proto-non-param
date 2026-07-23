#!/bin/bash
# Zero-shot RIS evaluation for run M1-RC (M1's config, but with the KL target
# rebuilt per-step from k=3 randomly-drawn region phrases instead of M1's
# fixed union-of-all-phrases target). Mirrors slurm_eval_vg_contrastive_M1.sh
# exactly (same splits, same default resolution) for a direct, apples-to-apples
# comparison against M1's own numbers.
#
# Usage:
#   bash scripts/slurm_eval_vg_contrastive_M1_RC.sh
#   RANDOM_CAPTION_K=5 bash scripts/slurm_eval_vg_contrastive_M1_RC.sh   # match a different k run

set -e

SCRATCH="/net/tscratch/people/plgabedychaj"
REPO=~/proto-non-param
CONTR_BASE="${CONTR_BASE:-${SCRATCH}/train_logs/vg_contrastive}"
DATA_ROOT="${DATA_ROOT:-${SCRATCH}/data/refcoco}"
RANDOM_CAPTION_K="${RANDOM_CAPTION_K:-3}"

PARTITION="plgrid-gpu-a100"
ACCOUNT="plgunhype-gpu-a100"
LOG_SLURM="${SCRATCH}/logs"

CKPT="${CONTR_BASE}/run_M1-RC_vitl14_sk10_koleo01_randk${RANDOM_CAPTION_K}_30ep/ckpt.pth"

mkdir -p "${LOG_SLURM}"

if [ ! -f "${CKPT}" ]; then
    echo "ERROR: checkpoint not found: ${CKPT}"
    exit 1
fi

OUT_DIR="${REPO}/eval_results/vg_contrastive/contr_M1-RC-k${RANDOM_CAPTION_K}"

echo "=== PNP — Zero-shot RIS Evaluation (run M1-RC, random-caption-target k=${RANDOM_CAPTION_K}) ==="
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
            --job-name="pnp-M1RC-${DATASET}-${SPLIT}" \
            --partition="${PARTITION}" \
            --account="${ACCOUNT}" \
            --gres=gpu:1 \
            --cpus-per-task=4 \
            --mem=32G \
            --time=04:00:00 \
            --output="${LOG_SLURM}/pnp_M1RC_${DATASET}_${SPLIT}_%j.out" \
            --error="${LOG_SLURM}/pnp_M1RC_${DATASET}_${SPLIT}_%j.err" \
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
