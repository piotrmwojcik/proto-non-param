#!/bin/bash
# Deletion faithfulness test (inference-only) -- see
# scripts/eval_deletion_faithfulness.py's module docstring.
#
# Usage:
#   bash scripts/slurm_eval_deletion_faithfulness.sh
#   DATASET=unc SPLIT=testA bash scripts/slurm_eval_deletion_faithfulness.sh
#   N_SAMPLES=500 DELETE_FRAC=0.1 bash scripts/slurm_eval_deletion_faithfulness.sh

set -e

SCRATCH="${SCRATCH:-/net/tscratch/people/plgabedychaj}"
REPO=~/proto-non-param
CONTR_BASE="${CONTR_BASE:-${SCRATCH}/train_logs/vg_contrastive}"
DATA_ROOT="${DATA_ROOT:-${SCRATCH}/data/refcoco}"
DATASET="${DATASET:-Gref}"
SPLIT="${SPLIT:-val}"
IMG_SIZE="${IMG_SIZE:-672}"
N_SAMPLES="${N_SAMPLES:-300}"
DELETE_FRAC="${DELETE_FRAC:-0.2}"

PARTITION="plgrid-gpu-a100"
ACCOUNT="plgunhype-gpu-a100"
LOG_SLURM="${SCRATCH}/logs"

CKPT="${CONTR_BASE}/run_M1_vitl14_sk10_koleo01_30ep/ckpt.pth"

mkdir -p "${LOG_SLURM}"

if [ ! -f "${CKPT}" ]; then
    echo "ERROR: checkpoint not found: ${CKPT}"
    exit 1
fi

OUT_DIR="${REPO}/results/deletion_faithfulness_${DATASET}_${SPLIT}"

JOB=$(sbatch --parsable \
    --job-name="pnp-deletion-faith" \
    --partition="${PARTITION}" \
    --account="${ACCOUNT}" \
    --gres=gpu:1 \
    --cpus-per-task=4 \
    --mem=32G \
    --time=02:00:00 \
    --output="${LOG_SLURM}/pnp_deletion_faith_%j.out" \
    --error="${LOG_SLURM}/pnp_deletion_faith_%j.err" \
    --wrap="
set -e
source ${SCRATCH}/venv/bin/activate
export HF_HUB_CACHE=${SCRATCH}/.cache/huggingface/hub
export TRANSFORMERS_CACHE=${SCRATCH}/.cache/huggingface/hub
export TORCH_HOME=${SCRATCH}/torch_cache
export PYTHONPATH=${SCRATCH}/dinov2:\$PYTHONPATH
cd ${REPO}

python scripts/eval_deletion_faithfulness.py \
  --ckpt ${CKPT} \
  --dataset ${DATASET} \
  --data_split ${SPLIT} \
  --data_root ${DATA_ROOT} \
  --img-size ${IMG_SIZE} \
  --n-samples ${N_SAMPLES} \
  --delete-frac ${DELETE_FRAC} \
  --out-dir ${OUT_DIR}
")
echo "Submitted: ${JOB}"
echo "Output: ${OUT_DIR}/{summary.json,per_example.csv,deletion_faithfulness.png}"
