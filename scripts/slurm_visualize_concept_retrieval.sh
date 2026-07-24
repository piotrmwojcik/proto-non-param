#!/bin/bash
# Qualitative concept-retrieval figure for M1 over RefCOCOg/val images.
#
# IMG_SIZE must stay 224: PNP.forward() unconditionally also runs the image
# through CLIP ViT-B/32's own image encoder for a diagnostic side-output, and
# that encoder (unlike the DINOv2 backbone) has a fixed 224px positional
# embedding — see visualize_concept_retrieval.py's --img-size help text.
#
# Usage:
#   bash scripts/slurm_visualize_concept_retrieval.sh
#   CONCEPTS="dog red wooden" bash scripts/slurm_visualize_concept_retrieval.sh   # explicit words
#   N_CONCEPTS=30 bash scripts/slurm_visualize_concept_retrieval.sh               # more concepts
#   SEPARATE_FIGURES=0 bash scripts/slurm_visualize_concept_retrieval.sh          # one combined grid instead

set -e

SCRATCH="/net/tscratch/people/plgabedychaj"
REPO=~/proto-non-param
CONTR_BASE="${CONTR_BASE:-${SCRATCH}/train_logs/vg_contrastive}"
DATA_ROOT="${DATA_ROOT:-${SCRATCH}/data/refcoco}"
IMG_SIZE="${IMG_SIZE:-224}"
N_CONCEPTS="${N_CONCEPTS:-20}"
SEPARATE_FIGURES="${SEPARATE_FIGURES:-1}"

PARTITION="plgrid-gpu-a100"
ACCOUNT="plgunhype-gpu-a100"
LOG_SLURM="${SCRATCH}/logs"

CKPT="${CONTR_BASE}/run_M1_vitl14_sk10_koleo01_30ep/ckpt.pth"

mkdir -p "${LOG_SLURM}"

if [ ! -f "${CKPT}" ]; then
    echo "ERROR: checkpoint not found: ${CKPT}"
    exit 1
fi

OUT_DIR="${REPO}/results/concept_retrieval"

CONCEPTS_ARG="--n-concepts ${N_CONCEPTS}"
if [ -n "${CONCEPTS:-}" ]; then
    CONCEPTS_ARG="--concepts ${CONCEPTS}"
fi

SEPARATE_ARG=""
if [ "${SEPARATE_FIGURES}" = "1" ]; then
    SEPARATE_ARG="--separate-figures"
fi

JOB=$(sbatch --parsable \
    --job-name="pnp-concept-retrieval" \
    --partition="${PARTITION}" \
    --account="${ACCOUNT}" \
    --gres=gpu:1 \
    --cpus-per-task=4 \
    --mem=32G \
    --time=02:00:00 \
    --output="${LOG_SLURM}/pnp_concept_retrieval_%j.out" \
    --error="${LOG_SLURM}/pnp_concept_retrieval_%j.err" \
    --wrap="
set -e
source ${SCRATCH}/venv/bin/activate
export HF_HUB_CACHE=${SCRATCH}/.cache/huggingface/hub
export TRANSFORMERS_CACHE=${SCRATCH}/.cache/huggingface/hub
export TORCH_HOME=${SCRATCH}/torch_cache
export PYTHONPATH=${SCRATCH}/dinov2:\$PYTHONPATH
cd ${REPO}

python scripts/visualize_concept_retrieval.py \
  --ckpt ${CKPT} \
  --data-root ${DATA_ROOT} \
  --dataset Gref \
  --split val \
  --img-size ${IMG_SIZE} \
  --out-dir ${OUT_DIR} \
  ${CONCEPTS_ARG} \
  ${SEPARATE_ARG}
")
echo "Submitted: ${JOB}"
if [ "${SEPARATE_FIGURES}" = "1" ]; then
    echo "Output: ${OUT_DIR}/concept_retrieval_<word>.png (one per concept)"
else
    echo "Output: ${OUT_DIR}/concept_retrieval_Gref_val.png"
fi
