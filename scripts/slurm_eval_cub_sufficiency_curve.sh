#!/bin/bash
# CLIP-substituted pseudo-intervention curve for a train_cub_joint.py checkpoint.
# See scripts/eval_cub_sufficiency_curve.py's module docstring for the caveat:
# this swaps in CLIP similarity, not ground-truth concept labels.
#
# Usage:
#   bash scripts/slurm_eval_cub_sufficiency_curve.sh

set -e

SCRATCH="${SCRATCH:-/net/tscratch/people/plgabedychaj}"
REPO=~/proto-non-param
CUB_ROOT="${CUB_ROOT:-${SCRATCH}/cub200}"
CUB_ANNOTATIONS="${CUB_ANNOTATIONS:-${SCRATCH}/cub200/annotations}"
LOG_DIR="${LOG_DIR:-${SCRATCH}/train_logs/cub_joint}"

PARTITION="plgrid-gpu-a100"
ACCOUNT="plgunhype-gpu-a100"
LOG_SLURM="${SCRATCH}/logs"

CKPT="${LOG_DIR}/ckpt.pth"

mkdir -p "${LOG_SLURM}"

if [ ! -f "${CKPT}" ]; then
    echo "ERROR: checkpoint not found: ${CKPT} -- run slurm_train_cub_joint.sh first"
    exit 1
fi

OUT_DIR="${REPO}/results/cub_sufficiency_curve"

JOB=$(sbatch --parsable \
    --job-name="pnp-cub-suffcurve" \
    --partition="${PARTITION}" \
    --account="${ACCOUNT}" \
    --gres=gpu:1 \
    --cpus-per-task=4 \
    --mem=32G \
    --time=02:00:00 \
    --output="${LOG_SLURM}/pnp_cub_suffcurve_%j.out" \
    --error="${LOG_SLURM}/pnp_cub_suffcurve_%j.err" \
    --wrap="
set -e
source ${SCRATCH}/venv/bin/activate
export HF_HUB_CACHE=${SCRATCH}/.cache/huggingface/hub
export TRANSFORMERS_CACHE=${SCRATCH}/.cache/huggingface/hub
export TORCH_HOME=${SCRATCH}/torch_cache
export PYTHONPATH=${SCRATCH}/dinov2:\$PYTHONPATH
cd ${REPO}

python scripts/eval_cub_sufficiency_curve.py \
  --ckpt ${CKPT} \
  --cub-root ${CUB_ROOT} \
  --cub-annotations ${CUB_ANNOTATIONS} \
  --out-dir ${OUT_DIR}
")
echo "Submitted: ${JOB}"
echo "Output: ${OUT_DIR}/{sufficiency_curve.json,sufficiency_curve.png}"
