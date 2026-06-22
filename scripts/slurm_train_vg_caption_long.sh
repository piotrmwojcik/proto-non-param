#!/bin/bash
# KL + caption-alignment training on full VG vocabulary with milestone checkpoints.
#
# Combines the word-distribution KL loss (kl_coef=1.0) with the caption-level
# CLIP alignment loss (caption_coef=1.0) using per-image VG region phrase
# embeddings built by build_vg_caption_embeddings.py.
#
# Based on the combined variant (run_C_combined) from slurm_train_vg_caption_signal.sh,
# with additions missing from that run:
#   --loss-type kl        explicit KL loss
#   --lr-schedule cosine  cosine decay (already the default, now explicit)
#   --lr-warmup-epochs 5  linear warmup before cosine decay
#   --save-every 5        ckpt_ep005.pth … ckpt_ep030.pth (6 checkpoints)
#   prototype_residual    zero-initialised (auto, residual-lr=0 default)
#
# Pipeline:
#   Job 1 (CPU, ~30 min) — build vg_caption_embs.pt   [skipped if exists]
#   Job 2 (GPU, ~5.5 h)  — training, 30 epochs
#
# Checkpoint dir: ${LOG_DIR}/run_A_kl_caption_frozen_30ep/
#
# After training, find optimal epoch:
#   CKPT_DIR=$SCRATCH/train_logs/vg_caption/run_A_kl_caption_frozen_30ep \
#   VARIANT=caption_A_30ep \
#   bash ~/proto-VLM/scripts/slurm_eval_collapse_check.sh
#
# Configuration — override via environment variables:
#   SCRATCH       Base scratch path (default: /net/tscratch/people/plgabedychaj)
#   VG_ROOT       Path to VG image root (containing VG_100K/ and VG_100K_2/)
#   VG_DESC       Path to region_descriptions.json
#   LOG_DIR       Base directory for checkpoints
#   VOCAB_DIR     Directory containing vg_cache.pt and vg_caption_embs.pt
#   WANDB_ENTITY  W&B entity (default: gmum)
#
# Usage:
#   bash scripts/slurm_train_vg_caption_long.sh

set -e

SCRATCH="${SCRATCH:-/net/tscratch/people/plgabedychaj}"
VG_ROOT="${VG_ROOT:-${SCRATCH}/vg}"
VG_DESC="${VG_DESC:-${VG_ROOT}/region_descriptions.json}"
LOG_DIR="${LOG_DIR:-${SCRATCH}/train_logs/vg_caption}"
VOCAB_DIR="${VOCAB_DIR:-${SCRATCH}/vocab}"
WANDB_ENTITY="${WANDB_ENTITY:-gmum}"

PARTITION="plgrid-gpu-a100"
ACCOUNT="plgunhype-gpu-a100"
LOG_SLURM="${SCRATCH}/logs"

VG_CACHE="${VOCAB_DIR}/vg_cache.pt"
CAPTION_EMBS="${VOCAB_DIR}/vg_caption_embs.pt"

mkdir -p "${LOG_SLURM}" "${LOG_DIR}"

if [ ! -f "${VG_CACHE}" ]; then
    echo "ERROR: VG vocab cache not found at ${VG_CACHE}"
    echo "Build it first: bash scripts/slurm_train_vg_caption_signal.sh"
    exit 1
fi

echo "=== VG Caption Long Training (KL + caption-align, frozen δ, 30 epochs) ==="
echo "  SCRATCH       : ${SCRATCH}"
echo "  VG_ROOT       : ${VG_ROOT}"
echo "  VG vocab      : ${VG_CACHE}"
echo "  Caption embs  : ${CAPTION_EMBS}"
echo "  Checkpoint    : ${LOG_DIR}/run_A_kl_caption_frozen_30ep/"
echo ""

WRAP_HEADER="
set -e
export HF_HOME=${SCRATCH}/hf_cache
export TRANSFORMERS_CACHE=${SCRATCH}/hf_cache
export PYTHONPATH=${SCRATCH}/dinov2:\$PYTHONPATH
source ${SCRATCH}/venv/bin/activate
cd ~/proto-VLM/proto-non-param
"

# -----------------------------------------------------------------------
# Job 1 — Build caption embedding pool (GPU, ~1 h for 108 K images)
# Skipped if the file already exists.
# -----------------------------------------------------------------------
if [ -f "${CAPTION_EMBS}" ]; then
    echo "Caption embeddings already exist — skipping build job."
    CAPTION_DEP=""
else
    JOB1=$(sbatch --parsable \
      --job-name=build-caption-embs \
      --partition="${PARTITION}" \
      --account="${ACCOUNT}" \
      --gres=gpu:1 \
      --cpus-per-task=4 \
      --mem=32G \
      --time=02:00:00 \
      --output="${LOG_SLURM}/caption_embs_%j.out" \
      --error="${LOG_SLURM}/caption_embs_%j.err" \
      --wrap="${WRAP_HEADER}
python vocab/build_vg_caption_embeddings.py \\
  --region-descriptions ${VG_DESC} \\
  --vg-root ${VG_ROOT} \\
  --vocab-cache-path ${VG_CACHE} \\
  --cache-out ${CAPTION_EMBS} \\
  --split both \\
  --pool-size 50 \\
  --clip-model-name ViT-B-32 \\
  --clip-pretrained openai \\
  --batch-size 512 \\
  --device cuda
")
    echo "Submitted job 1 (build-caption-embs): ${JOB1}"
    CAPTION_DEP="--dependency=afterok:${JOB1}"
fi

# -----------------------------------------------------------------------
# Job 2 — Training: KL + caption alignment, 30 epochs
# -----------------------------------------------------------------------
JOB2=$(sbatch --parsable \
  ${CAPTION_DEP} \
  --job-name=vg-caption-A \
  --partition="${PARTITION}" \
  --account="${ACCOUNT}" \
  --gres=gpu:1 \
  --cpus-per-task=8 \
  --mem=64G \
  --time=08:00:00 \
  --output="${LOG_SLURM}/vg_caption_A_%j.out" \
  --error="${LOG_SLURM}/vg_caption_A_%j.err" \
  --wrap="${WRAP_HEADER}
python train.py \\
  --dataset visual_genome \\
  --vg-root ${VG_ROOT} \\
  --vg-region-descriptions ${VG_DESC} \\
  --vocab-cache-path ${VG_CACHE} \\
  --backbone dinov2_vitb14 \\
  --batch-size 64 \\
  --epochs 30 \\
  --lr-schedule cosine \\
  --lr-warmup-epochs 5 \\
  --num-workers 8 \\
  --backbone-lr 1e-5 \\
  --text-proj-lr 1e-4 \\
  --target-mode topk \\
  --top-k-concepts 5 \\
  --loss-type kl \\
  --kl-coef 1.0 \\
  --caption-coef 1.0 \\
  --caption-embeds-path ${CAPTION_EMBS} \\
  --caption-sample-k 5 \\
  --save-every 5 \\
  --wandb-entity ${WANDB_ENTITY} \\
  --wandb-log-images 16 \\
  --log-dir ${LOG_DIR}/run_A_kl_caption_frozen_30ep
")

echo "Submitted job 2 (training): ${JOB2}"
echo ""
echo "Monitor with : squeue -u \$USER"
echo "Log          : ${LOG_SLURM}/vg_caption_A_${JOB2}.out"
echo ""
echo "After training, find the optimal epoch:"
echo "  CKPT_DIR=${LOG_DIR}/run_A_kl_caption_frozen_30ep \\"
echo "  VARIANT=caption_A_30ep \\"
echo "  bash ~/proto-VLM/scripts/slurm_eval_collapse_check.sh"
echo ""
echo "Then run full RIS eval on the best epoch checkpoint:"
echo "  CKPT=${LOG_DIR}/run_A_kl_caption_frozen_30ep/ckpt_ep<N>.pth \\"
echo "  VARIANT=caption_A_ep<N> \\"
echo "  bash ~/proto-VLM/scripts/slurm_eval_pnp_threshold_sweep.sh"
