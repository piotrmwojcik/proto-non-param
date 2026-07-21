#!/bin/bash
# Post-fine-tuning evaluation for the Stage 2 CUB checkpoint: Label-free-CBM's
# remaining two stages, run after slurm_train_cub_labelfreecbm.sh completes.
#   3. Interpretability-cutoff filter (their default 0.45): drop concepts whose
#      *fine-tuned* representation no longer tracks CLIP's own judgment.
#   4. Sparse elastic-net classifier over the surviving concepts' activations
#      (regenerated via evaluate_pnp_cub_concepts.py, unchanged, just pointed
#      at the fine-tuned checkpoint + final concept list).
# Also re-runs Stage 1's script with the fine-tuned checkpoint against the
# ORIGINAL unfiltered concept list, to isolate "did fine-tuning help at all"
# from the filtering/sparse-classifier changes (see plan verification step 4).
#
# Usage:
#   bash scripts/slurm_eval_cub_labelfreecbm_finetuned.sh

set -e

SCRATCH="${SCRATCH:-/net/tscratch/people/plgabedychaj}"
REPO=~/proto-non-param
CUB_ROOT="${CUB_ROOT:-${SCRATCH}/cub200}"
CUB_ANNOTATIONS="${CUB_ANNOTATIONS:-${SCRATCH}/cub200/annotations}"
VOCAB_DIR="${VOCAB_DIR:-${SCRATCH}/vocab}"
LOG_DIR="${LOG_DIR:-${SCRATCH}/train_logs/cub_labelfreecbm}"
IMG_SIZE="${IMG_SIZE:-672}"

PARTITION="plgrid-gpu-a100"
ACCOUNT="plgunhype-gpu-a100"
LOG_SLURM="${SCRATCH}/logs"

CKPT="${LOG_DIR}/ckpt.pth"
FILTERED_CONCEPTS="${VOCAB_DIR}/cub_clip_scores_concepts_filtered.txt"
SCORES_FILE="${VOCAB_DIR}/cub_clip_scores.pt"
ORIGINAL_CONCEPTS="${VOCAB_DIR}/cub_filtered_concepts.txt"

mkdir -p "${LOG_SLURM}"

if [ ! -f "${CKPT}" ]; then
    echo "ERROR: fine-tuned checkpoint not found: ${CKPT} -- run slurm_train_cub_labelfreecbm.sh first"
    exit 1
fi

OUT_STAGE2="${REPO}/eval_results/cub_concepts_stage2"
OUT_BEFORE_AFTER="${REPO}/eval_results/cub_concepts_stage2_vs_stage1"

JOB=$(sbatch --parsable \
    --job-name="pnp-cub-labelfreecbm-eval" \
    --partition="${PARTITION}" \
    --account="${ACCOUNT}" \
    --gres=gpu:1 \
    --cpus-per-task=4 \
    --mem=32G \
    --time=04:00:00 \
    --output="${LOG_SLURM}/pnp_cub_labelfreecbm_eval_%j.out" \
    --error="${LOG_SLURM}/pnp_cub_labelfreecbm_eval_%j.err" \
    --wrap="
set -e
source ${SCRATCH}/venv/bin/activate
export HF_HUB_CACHE=${SCRATCH}/.cache/huggingface/hub
export TRANSFORMERS_CACHE=${SCRATCH}/.cache/huggingface/hub
export TORCH_HOME=${SCRATCH}/torch_cache
export PYTHONPATH=${SCRATCH}/dinov2:\$PYTHONPATH
cd ${REPO}

echo '=== Before/after: fine-tuned ckpt vs original unfiltered concept list (isolates fine-tuning effect) ==='
python scripts/evaluate_pnp_cub_concepts.py \
  --ckpt ${CKPT} \
  --cub-root ${CUB_ROOT} \
  --concepts-file ${ORIGINAL_CONCEPTS} \
  --img-size ${IMG_SIZE} \
  --out-dir ${OUT_BEFORE_AFTER}

echo '=== Step 3: interpretability-cutoff filter ==='
python scripts/filter_cub_concepts_interpretability.py \
  --ckpt ${CKPT} \
  --concepts-file ${FILTERED_CONCEPTS} \
  --clip-scores-file ${SCORES_FILE} \
  --cub-root ${CUB_ROOT} \
  --cub-annotations ${CUB_ANNOTATIONS} \
  --img-size ${IMG_SIZE} \
  --out-concepts-file ${OUT_STAGE2}/concepts_final.txt

echo '=== Regenerate activations with fine-tuned ckpt + final concept list ==='
python scripts/evaluate_pnp_cub_concepts.py \
  --ckpt ${CKPT} \
  --cub-root ${CUB_ROOT} \
  --concepts-file ${OUT_STAGE2}/concepts_final.txt \
  --img-size ${IMG_SIZE} \
  --out-dir ${OUT_STAGE2}

echo '=== Step 4: sparse elastic-net classifier ==='
python scripts/fit_sparse_cub_probe.py \
  --activations-cache ${OUT_STAGE2}/activations_img${IMG_SIZE}.pt \
  --out-dir ${OUT_STAGE2}
")
echo "Submitted: ${JOB}"

echo ""
echo "Monitor with : squeue -u \$USER"
echo "Before/after (fine-tune only): ${OUT_BEFORE_AFTER}/result_img${IMG_SIZE}.json"
echo "Full pipeline (filter+sparse): ${OUT_STAGE2}/sparse_probe_result.json"
