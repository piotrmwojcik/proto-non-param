#!/bin/bash
# Stage 2 fine-tuning: PNP warm-started from M1, adapted to CUB-200 with
# Label-free-CBM's concept methodology (CLIP-cutoff filter -> CLIP-similarity
# soft-label fine-tuning). Runs three steps in one job (steps 1-2 are cheap;
# chaining avoids SLURM job-dependency complexity for a short pipeline):
#   1. Build a CLIP embedding cache for the 370 Label-free-CBM concept phrases.
#   2. Compute CLIP image-vs-concept scores over CUB train+val and apply the
#      CLIP-cutoff filter (their default 0.25) -- drops concepts CLIP itself
#      can't reliably detect, before training ever sees them.
#   3. Fine-tune, warm-started from M1, using the filtered vocab as the
#      prototype pool and the filtered CLIP scores as KL targets
#      (--clip-scores-cub, since these GPT-generated concepts have no human
#      per-image CUB label to train against directly).
#
# --epochs 15 / LR values are a starting point (M1's own VG recipe, shortened
# since this is adapting an already-good representation to a much smaller
# dataset, not training from scratch) -- watch the W&B curves and adjust if
# it under/overfits.
#
# Prerequisites: CUB-200 downloaded (slurm_download_cub200.sh) and the
# Label-free-CBM concept file present ($SCRATCH/vocab/cub_filtered_concepts.txt,
# downloaded by slurm_eval_cub_concepts_M1.sh in Stage 1).
#
# Usage:
#   bash scripts/slurm_train_cub_labelfreecbm.sh

set -e

SCRATCH="${SCRATCH:-/net/tscratch/people/plgabedychaj}"
REPO=~/proto-non-param
CONTR_BASE="${CONTR_BASE:-${SCRATCH}/train_logs/vg_contrastive}"
CUB_ROOT="${CUB_ROOT:-${SCRATCH}/cub200}"
CUB_ANNOTATIONS="${CUB_ANNOTATIONS:-${SCRATCH}/cub200/annotations}"
CONCEPTS_FILE="${CONCEPTS_FILE:-${SCRATCH}/vocab/cub_filtered_concepts.txt}"
VOCAB_DIR="${VOCAB_DIR:-${SCRATCH}/vocab}"
LOG_DIR="${LOG_DIR:-${SCRATCH}/train_logs/cub_labelfreecbm}"
WANDB_ENTITY="${WANDB_ENTITY:-gmum}"
CLIP_CUTOFF="${CLIP_CUTOFF:-0.25}"

PARTITION="plgrid-gpu-a100"
ACCOUNT="plgunhype-gpu-a100"
LOG_SLURM="${SCRATCH}/logs"

INIT_CKPT="${CONTR_BASE}/run_M1_vitl14_sk10_koleo01_30ep/ckpt.pth"

mkdir -p "${LOG_SLURM}" "${VOCAB_DIR}" "${LOG_DIR}"

if [ ! -f "${INIT_CKPT}" ]; then
    echo "ERROR: M1 checkpoint not found: ${INIT_CKPT}"
    exit 1
fi
if [ ! -d "${CUB_ROOT}/train" ]; then
    echo "ERROR: CUB-200 not found/organized at ${CUB_ROOT} -- run slurm_download_cub200.sh first"
    exit 1
fi
if [ ! -f "${CONCEPTS_FILE}" ]; then
    echo "Concept file not found — downloading to ${CONCEPTS_FILE} ..."
    wget -O "${CONCEPTS_FILE}" \
      https://raw.githubusercontent.com/Trustworthy-ML-Lab/Label-free-CBM/main/data/concept_sets/cub_filtered.txt
fi

CONCEPT_CACHE="${VOCAB_DIR}/cub_labelfreecbm_cache.pt"
SCORES_OUT="${VOCAB_DIR}/cub_clip_scores.pt"
FILTERED_CONCEPTS="${VOCAB_DIR}/cub_clip_scores_concepts_filtered.txt"
FILTERED_VOCAB_CACHE="${VOCAB_DIR}/cub_clip_scores_vocab_filtered.pt"

JOB=$(sbatch --parsable \
    --job-name="pnp-cub-labelfreecbm-finetune" \
    --partition="${PARTITION}" \
    --account="${ACCOUNT}" \
    --gres=gpu:1 \
    --cpus-per-task=8 \
    --mem=64G \
    --time=1-00:00:00 \
    --output="${LOG_SLURM}/pnp_cub_labelfreecbm_%j.out" \
    --error="${LOG_SLURM}/pnp_cub_labelfreecbm_%j.err" \
    --wrap="
set -e
source ${SCRATCH}/venv/bin/activate
export HF_HOME=${SCRATCH}/hf_cache
export TRANSFORMERS_CACHE=${SCRATCH}/hf_cache
export HF_HUB_CACHE=${SCRATCH}/.cache/huggingface/hub
export TORCH_HOME=${SCRATCH}/torch_cache
export PYTHONPATH=${SCRATCH}/dinov2:\$PYTHONPATH
cd ${REPO}

echo '=== Step 1: build concept cache ==='
python scripts/build_cub_concept_cache.py \
  --concepts-file ${CONCEPTS_FILE} \
  --cache-out ${CONCEPT_CACHE}

echo '=== Step 2: compute + CLIP-cutoff-filter CUB concept scores ==='
python build_clip_vocab_scores.py \
  --dataset cub \
  --data-root ${CUB_ROOT} \
  --annotations ${CUB_ANNOTATIONS} \
  --vocab-cache ${CONCEPT_CACHE} \
  --output ${SCORES_OUT} \
  --concept-clip-cutoff ${CLIP_CUTOFF}

echo '=== Step 3: fine-tune, warm-started from M1 ==='
python train.py \
  --dataset cub200 \
  --cub-root ${CUB_ROOT} \
  --cub-annotations ${CUB_ANNOTATIONS} \
  --vocab-cache-path ${FILTERED_VOCAB_CACHE} \
  --clip-scores-cub ${SCORES_OUT} \
  --init-ckpt ${INIT_CKPT} \
  --backbone dinov2_vitl14 \
  --batch-size 64 \
  --epochs 15 \
  --lr-schedule cosine \
  --lr-warmup-epochs 2 \
  --num-workers 8 \
  --backbone-lr 1e-5 \
  --text-proj-lr 1e-4 \
  --text-proj-hidden-dim 2048 \
  --target-mode uniform \
  --loss-type kl \
  --kl-coef 1.0 \
  --contrastive-coef 0.0 \
  --sk-coef 0.1 \
  --sk-eps 0.10 \
  --koleo-coef 0.1 \
  --agg-mode topk \
  --save-every 5 \
  --wandb-entity ${WANDB_ENTITY} \
  --wandb-log-images 16 \
  --log-dir ${LOG_DIR}
")
echo "Submitted (build cache -> filtered CLIP scores -> fine-tune from M1): ${JOB}"

echo ""
echo "Monitor with : squeue -u \$USER"
echo "Logs under   : ${LOG_SLURM}/pnp_cub_labelfreecbm_${JOB}.out"
echo "Checkpoint   : ${LOG_DIR}/ckpt.pth"
echo "Filtered concept list: ${FILTERED_CONCEPTS}"
echo "Filtered vocab cache : ${FILTERED_VOCAB_CACHE}"
echo "Filtered CLIP scores : ${SCORES_OUT}"
