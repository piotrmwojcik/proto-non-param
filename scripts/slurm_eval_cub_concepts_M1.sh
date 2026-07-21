#!/bin/bash
# Zero-shot PNP concept-bottleneck evaluation on CUB-200 using Label-free-CBM's
# 379 GPT-generated concept phrases, against the current best checkpoint (M1)
# at the current best resolution (672px) plus 224px for a near-zero-cost
# resolution comparison (672 was validated for dense referring segmentation;
# whole-image classification via pooled concept activation hasn't been
# separately checked).
#
# Prerequisites (checked below, not assumed):
#   - CUB-200 downloaded+organized at $SCRATCH/cub200 (scripts/download_cub200.py
#     via scripts/slurm_download_cub200.sh if missing)
#   - Label-free-CBM concept file at $SCRATCH/vocab/cub_filtered_concepts.txt
#     (downloaded below if missing)
#
# Usage:
#   bash scripts/slurm_eval_cub_concepts_M1.sh

set -e

SCRATCH="/net/tscratch/people/plgabedychaj"
REPO=~/proto-non-param
CONTR_BASE="${CONTR_BASE:-${SCRATCH}/train_logs/vg_contrastive}"
CUB_ROOT="${CUB_ROOT:-${SCRATCH}/cub200}"
CONCEPTS_FILE="${CONCEPTS_FILE:-${SCRATCH}/vocab/cub_filtered_concepts.txt}"

PARTITION="plgrid-gpu-a100"
ACCOUNT="plgunhype-gpu-a100"
LOG_SLURM="${SCRATCH}/logs"

CKPT="${CONTR_BASE}/run_M1_vitl14_sk10_koleo01_30ep/ckpt.pth"

mkdir -p "${LOG_SLURM}" "${SCRATCH}/vocab"

if [ ! -f "${CKPT}" ]; then
    echo "ERROR: checkpoint not found: ${CKPT}"
    exit 1
fi

if [ ! -d "${CUB_ROOT}/train" ] || [ ! -d "${CUB_ROOT}/test" ]; then
    echo "ERROR: CUB-200 not found/organized at ${CUB_ROOT}"
    echo "Run first:  bash scripts/slurm_download_cub200.sh"
    exit 1
fi

if [ ! -f "${CONCEPTS_FILE}" ]; then
    echo "Concept file not found — downloading to ${CONCEPTS_FILE} ..."
    wget -O "${CONCEPTS_FILE}" \
      https://raw.githubusercontent.com/Trustworthy-ML-Lab/Label-free-CBM/main/data/concept_sets/cub_filtered.txt
fi

OUT_DIR="${REPO}/eval_results/cub_concepts"

echo "=== PNP — Zero-shot CUB-200 concept-bottleneck eval (run M1) ==="
echo "  Ckpt      : ${CKPT}"
echo "  CUB root  : ${CUB_ROOT}"
echo "  Concepts  : ${CONCEPTS_FILE}"
echo "  Out       : ${OUT_DIR}"
echo ""

for SIZE in 672 224; do
    JOB=$(sbatch --parsable \
        --job-name="pnp-cub-concepts-${SIZE}" \
        --partition="${PARTITION}" \
        --account="${ACCOUNT}" \
        --gres=gpu:1 \
        --cpus-per-task=4 \
        --mem=32G \
        --time=04:00:00 \
        --output="${LOG_SLURM}/pnp_cub_concepts_${SIZE}_%j.out" \
        --error="${LOG_SLURM}/pnp_cub_concepts_${SIZE}_%j.err" \
        --wrap="
set -e
source ${SCRATCH}/venv/bin/activate
export HF_HUB_CACHE=${SCRATCH}/.cache/huggingface/hub
export TRANSFORMERS_CACHE=${SCRATCH}/.cache/huggingface/hub
export TORCH_HOME=${SCRATCH}/torch_cache
export PYTHONPATH=${SCRATCH}/dinov2:\$PYTHONPATH
cd ${REPO}

python scripts/evaluate_pnp_cub_concepts.py \
  --ckpt ${CKPT} \
  --cub-root ${CUB_ROOT} \
  --concepts-file ${CONCEPTS_FILE} \
  --img-size ${SIZE} \
  --out-dir ${OUT_DIR}
")
    echo "  ${JOB}  img-size=${SIZE}"
done

echo ""
echo "2 jobs submitted. Monitor with: squeue -u \$USER"
echo "Results: ${OUT_DIR}/result_img{672,224}.json"
