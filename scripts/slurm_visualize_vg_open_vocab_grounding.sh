#!/bin/bash
# Zero-shot open-vocabulary grounding demo on VG images -- see
# scripts/visualize_vg_open_vocab_grounding.py's module docstring.
#
# Usage:
#   bash scripts/slurm_visualize_vg_open_vocab_grounding.sh
#   N_IMAGES=10 bash scripts/slurm_visualize_vg_open_vocab_grounding.sh
#
#   # Custom phrases: one per line in a file (safer than shell-quoting
#   # multi-word phrases through an env var):
#   cat > /tmp/phrases.txt <<'EOF'
#   a red car
#   a person wearing a hat
#   a wooden table
#   EOF
#   PHRASES_FILE=/tmp/phrases.txt bash scripts/slurm_visualize_vg_open_vocab_grounding.sh

set -e

SCRATCH="${SCRATCH:-/net/tscratch/people/plgabedychaj}"
REPO=~/proto-non-param
CONTR_BASE="${CONTR_BASE:-${SCRATCH}/train_logs/vg_contrastive}"
VG_ROOT="${VG_ROOT:-${SCRATCH}/vg}"
N_IMAGES="${N_IMAGES:-6}"
IMG_SIZE="${IMG_SIZE:-672}"

PARTITION="plgrid-gpu-a100"
ACCOUNT="plgunhype-gpu-a100"
LOG_SLURM="${SCRATCH}/logs"

CKPT="${CONTR_BASE}/run_M1_vitl14_sk10_koleo01_30ep/ckpt.pth"

mkdir -p "${LOG_SLURM}"

if [ ! -f "${CKPT}" ]; then
    echo "ERROR: checkpoint not found: ${CKPT}"
    exit 1
fi

OUT_DIR="${REPO}/results/vg_open_vocab_grounding"

PHRASES_ARG=""
if [ -n "${PHRASES_FILE:-}" ]; then
    PHRASES_ARG="--phrases-file ${PHRASES_FILE}"
fi

JOB=$(sbatch --parsable \
    --job-name="pnp-vg-grounding" \
    --partition="${PARTITION}" \
    --account="${ACCOUNT}" \
    --gres=gpu:1 \
    --cpus-per-task=4 \
    --mem=32G \
    --time=01:00:00 \
    --output="${LOG_SLURM}/pnp_vg_grounding_%j.out" \
    --error="${LOG_SLURM}/pnp_vg_grounding_%j.err" \
    --wrap="
set -e
source ${SCRATCH}/venv/bin/activate
export HF_HUB_CACHE=${SCRATCH}/.cache/huggingface/hub
export TRANSFORMERS_CACHE=${SCRATCH}/.cache/huggingface/hub
export TORCH_HOME=${SCRATCH}/torch_cache
export PYTHONPATH=${SCRATCH}/dinov2:\$PYTHONPATH
cd ${REPO}

python scripts/visualize_vg_open_vocab_grounding.py \
  --ckpt ${CKPT} \
  --vg-root ${VG_ROOT} \
  --n-images ${N_IMAGES} \
  --img-size ${IMG_SIZE} \
  --out-dir ${OUT_DIR} \
  ${PHRASES_ARG}
")
echo "Submitted: ${JOB}"
echo "Output: ${OUT_DIR}/grounding_<image_id>.png"
