#!/bin/bash
# mIoU-vs-expression-length analysis for RefCOCO and RefCOCO+ (Gref/val already
# done, see analyze_miou_vs_length.py) -- see
# scripts/analyze_miou_vs_length_multi.py's module docstring.
#
# CPU-only (no model forward pass, just JSON loading + bootstrap + plotting),
# but run via sbatch anyway rather than the login node for queueing/logging.
#
# Usage:
#   bash scripts/slurm_analyze_miou_vs_length_multi.sh
#   EVAL_DIR=$SCRATCH/eval_results bash scripts/slurm_analyze_miou_vs_length_multi.sh

set -e

SCRATCH="${SCRATCH:-/net/tscratch/people/plgabedychaj}"
REPO=~/proto-non-param
DATA_ROOT="${DATA_ROOT:-${SCRATCH}/data/refcoco}"
EVAL_DIR="${EVAL_DIR:-${SCRATCH}/eval_results}"

PARTITION="plgrid-gpu-a100"
ACCOUNT="plgunhype-gpu-a100"
LOG_SLURM="${SCRATCH}/logs"

OUT_DIR="${REPO}/results/miou_vs_length_multi"

mkdir -p "${LOG_SLURM}"

JOB=$(sbatch --parsable \
    --job-name="pnp-length-multi" \
    --partition="${PARTITION}" \
    --account="${ACCOUNT}" \
    --cpus-per-task=4 \
    --mem=16G \
    --time=00:30:00 \
    --output="${LOG_SLURM}/pnp_length_multi_%j.out" \
    --error="${LOG_SLURM}/pnp_length_multi_%j.err" \
    --wrap="
set -e
source ${SCRATCH}/venv/bin/activate
export HF_HUB_CACHE=${SCRATCH}/.cache/huggingface/hub
export TRANSFORMERS_CACHE=${SCRATCH}/.cache/huggingface/hub
cd ${REPO}

python scripts/analyze_miou_vs_length_multi.py \
  --data-root ${DATA_ROOT} \
  --eval-dir ${EVAL_DIR} \
  --out-dir ${OUT_DIR}
")
echo "Submitted: ${JOB}"
echo "Output: ${OUT_DIR}/{unc,unc+}/{per_example.csv,per_bucket.csv,miou_vs_length.png,table.tex}"
