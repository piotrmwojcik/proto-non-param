#!/bin/bash
# Prototype dictionary inspection (nearest-neighbor shift + t-SNE) -- see
# scripts/inspect_prototype_dictionary.py's module docstring.
#
# Usage:
#   bash scripts/slurm_inspect_prototype_dictionary.sh
#   WORDS="dog car red running happy" bash scripts/slurm_inspect_prototype_dictionary.sh
#
#   # Semantic-group mode: one comma-separated group per line in a file
#   cat > /tmp/groups.txt <<'EOF'
#   cat,lion,dog
#   furniture,chair,table
#   EOF
#   GROUPS_FILE=/tmp/groups.txt bash scripts/slurm_inspect_prototype_dictionary.sh

set -e

SCRATCH="${SCRATCH:-/net/tscratch/people/plgabedychaj}"
REPO=~/proto-non-param
CONTR_BASE="${CONTR_BASE:-${SCRATCH}/train_logs/vg_contrastive}"
WORDS="${WORDS:-}"
GROUPS_FILE="${GROUPS_FILE:-}"

PARTITION="plgrid-gpu-a100"
ACCOUNT="plgunhype-gpu-a100"
LOG_SLURM="${SCRATCH}/logs"

CKPT="${CONTR_BASE}/run_M1_vitl14_sk10_koleo01_30ep/ckpt.pth"

mkdir -p "${LOG_SLURM}"

if [ ! -f "${CKPT}" ]; then
    echo "ERROR: checkpoint not found: ${CKPT}"
    exit 1
fi

OUT_DIR="${REPO}/results/prototype_dictionary"

WORDS_ARG=""
if [ -n "${WORDS}" ]; then
    WORDS_ARG="--words ${WORDS}"
fi

GROUPS_ARG=""
if [ -n "${GROUPS_FILE}" ]; then
    GROUPS_ARG="--groups-file ${GROUPS_FILE}"
fi

JOB=$(sbatch --parsable \
    --job-name="pnp-proto-dict" \
    --partition="${PARTITION}" \
    --account="${ACCOUNT}" \
    --gres=gpu:1 \
    --cpus-per-task=4 \
    --mem=32G \
    --time=01:00:00 \
    --output="${LOG_SLURM}/pnp_proto_dict_%j.out" \
    --error="${LOG_SLURM}/pnp_proto_dict_%j.err" \
    --wrap="
set -e
source ${SCRATCH}/venv/bin/activate
export HF_HUB_CACHE=${SCRATCH}/.cache/huggingface/hub
export TRANSFORMERS_CACHE=${SCRATCH}/.cache/huggingface/hub
export TORCH_HOME=${SCRATCH}/torch_cache
export PYTHONPATH=${SCRATCH}/dinov2:\$PYTHONPATH
cd ${REPO}

python scripts/inspect_prototype_dictionary.py \
  --ckpt ${CKPT} \
  --out-dir ${OUT_DIR} \
  ${WORDS_ARG} \
  ${GROUPS_ARG}
")
echo "Submitted: ${JOB}"
echo "Output: ${OUT_DIR}/{nearest_neighbor_shift.json,prototype_tsne.png,group_clustering.json (if GROUPS_FILE given)}"
