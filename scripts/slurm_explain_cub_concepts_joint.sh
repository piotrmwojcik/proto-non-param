#!/bin/bash
# Per-image / per-class concept interpretability report for the Joint CUB
# checkpoint. See scripts/explain_cub_concepts.py's module docstring.
#
# For the Sequential (Stage 2) checkpoint, run explain_cub_concepts.py
# --mode sequential directly on the login node instead (no GPU needed --
# it only loads a small cached activations file + a small sklearn model).
#
# Usage:
#   bash scripts/slurm_explain_cub_concepts_joint.sh
#   CLASS_NAMES="Black_footed_Albatross Indigo_Bunting" \
#       bash scripts/slurm_explain_cub_concepts_joint.sh
#   IMAGE_INDICES="0 1 2 3 4" bash scripts/slurm_explain_cub_concepts_joint.sh

set -e

SCRATCH="${SCRATCH:-/net/tscratch/people/plgabedychaj}"
REPO=~/proto-non-param
CUB_ROOT="${CUB_ROOT:-${SCRATCH}/cub200}"
CUB_ANNOTATIONS="${CUB_ANNOTATIONS:-${SCRATCH}/cub200/annotations}"
VOCAB_DIR="${VOCAB_DIR:-${SCRATCH}/vocab}"
LOG_DIR="${LOG_DIR:-${SCRATCH}/train_logs/cub_joint}"
IMAGE_INDICES="${IMAGE_INDICES:-0 1 2}"
CLASS_NAMES="${CLASS_NAMES:-}"

PARTITION="plgrid-gpu-a100"
ACCOUNT="plgunhype-gpu-a100"
LOG_SLURM="${SCRATCH}/logs"

CKPT="${LOG_DIR}/ckpt.pth"
SCORES_OUT="${VOCAB_DIR}/cub_clip_scores.pt"

mkdir -p "${LOG_SLURM}"

if [ ! -f "${CKPT}" ]; then
    echo "ERROR: checkpoint not found: ${CKPT} -- run slurm_train_cub_joint.sh first"
    exit 1
fi

OUT_DIR="${REPO}/results/cub_explain_joint"

CLASS_ARG=""
if [ -n "${CLASS_NAMES}" ]; then
    CLASS_ARG="--class-names ${CLASS_NAMES}"
fi

JOB=$(sbatch --parsable \
    --job-name="pnp-cub-explain" \
    --partition="${PARTITION}" \
    --account="${ACCOUNT}" \
    --gres=gpu:1 \
    --cpus-per-task=4 \
    --mem=32G \
    --time=01:00:00 \
    --output="${LOG_SLURM}/pnp_cub_explain_%j.out" \
    --error="${LOG_SLURM}/pnp_cub_explain_%j.err" \
    --wrap="
set -e
source ${SCRATCH}/venv/bin/activate
export HF_HUB_CACHE=${SCRATCH}/.cache/huggingface/hub
export TRANSFORMERS_CACHE=${SCRATCH}/.cache/huggingface/hub
export TORCH_HOME=${SCRATCH}/torch_cache
export PYTHONPATH=${SCRATCH}/dinov2:\$PYTHONPATH
cd ${REPO}

python scripts/explain_cub_concepts.py \
  --mode joint \
  --ckpt ${CKPT} \
  --cub-root ${CUB_ROOT} \
  --cub-annotations ${CUB_ANNOTATIONS} \
  --clip-scores-cub ${SCORES_OUT} \
  --image-indices ${IMAGE_INDICES} \
  --out-dir ${OUT_DIR} \
  ${CLASS_ARG}
")
echo "Submitted: ${JOB}"
echo "Output: ${OUT_DIR}/explain_joint.json"
