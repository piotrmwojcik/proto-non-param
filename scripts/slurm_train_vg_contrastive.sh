#!/bin/bash
# Contrastive training on Visual Genome with uniform word distribution.
#
# Training setup:
#   - target-mode uniform : equal weight over all unique vocab words per image
#     (no frequency bias; all region-description words are equally supervising)
#   - contrastive loss     : symmetric InfoNCE between pred_text_embedding and
#     per-image CLIP phrase embeddings (batch[4] from vg_collate_fn)
#   - KL distribution loss : kl_coef=1.0 kept alongside contrastive
#   - caption-sample-k 1  : single randomly drawn phrase per image as positive
#     key → cleaner contrastive signal with no averaging noise
#
# Two runs are submitted:
#   A — KL(uniform) + contrastive_coef=0.5  (balanced)
#   B — KL(uniform) + contrastive_coef=1.0  (contrastive-dominant)
#
# Pipeline:
#   Job 1 (GPU, ~1 h)  — build vg_caption_embs.pt  [skipped if exists]
#   Jobs A,B (GPU)     — training, 30 epochs each   [depend on job 1]
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
#   bash scripts/slurm_train_vg_contrastive.sh

set -e

SCRATCH="${SCRATCH:-/net/tscratch/people/plgabedychaj}"
VG_ROOT="${VG_ROOT:-${SCRATCH}/vg}"
VG_DESC="${VG_DESC:-${VG_ROOT}/region_descriptions.json}"
LOG_DIR="${LOG_DIR:-${SCRATCH}/train_logs/vg_contrastive}"
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
    echo "Build it first: bash scripts/slurm_build_vg_vocab.sh"
    exit 1
fi

echo "=== VG Contrastive Training (uniform distribution + InfoNCE, 30 epochs) ==="
echo "  SCRATCH       : ${SCRATCH}"
echo "  VG_ROOT       : ${VG_ROOT}"
echo "  VG vocab      : ${VG_CACHE}"
echo "  Caption embs  : ${CAPTION_EMBS}"
echo "  Checkpoints   : ${LOG_DIR}/"
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
# Skipped if the file already exists (shared with other VG caption runs).
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

# ---- Shared training args ----
TRAIN_COMMON="--dataset visual_genome \\
  --vg-root ${VG_ROOT} \\
  --vg-region-descriptions ${VG_DESC} \\
  --vocab-cache-path ${VG_CACHE} \\
  --backbone dinov2_vitb14 \\
  --batch-size 128 \\
  --epochs 30 \\
  --lr-schedule cosine \\
  --lr-warmup-epochs 5 \\
  --num-workers 8 \\
  --backbone-lr 1e-5 \\
  --text-proj-lr 1e-4 \\
  --target-mode uniform \\
  --loss-type kl \\
  --kl-coef 1.0 \\
  --caption-embeds-path ${CAPTION_EMBS} \\
  --caption-sample-k 1 \\
  --contrastive-temp 0.07 \\
  --save-every 5 \\
  --wandb-entity ${WANDB_ENTITY} \\
  --wandb-log-images 16"

SLURM_COMMON="${CAPTION_DEP} \
  --partition=${PARTITION} \
  --account=${ACCOUNT} \
  --gres=gpu:1 \
  --cpus-per-task=8 \
  --mem=64G \
  --time=2-00:00:00"

# -----------------------------------------------------------------------
# Run A — contrastive_coef=0.5  (balanced: KL + light contrastive)
# -----------------------------------------------------------------------
JOB_A=$(sbatch --parsable \
  ${SLURM_COMMON} \
  --job-name=vg-contr-A \
  --output="${LOG_SLURM}/vg_contrastive_A_%j.out" \
  --error="${LOG_SLURM}/vg_contrastive_A_%j.err" \
  --wrap="${WRAP_HEADER}
python train.py \\
  ${TRAIN_COMMON} \\
  --contrastive-coef 0.5 \\
  --log-dir ${LOG_DIR}/run_A_uniform_contrastive05_30ep
")
echo "Submitted A (uniform + contrastive=0.5, 30ep): ${JOB_A}"

# -----------------------------------------------------------------------
# Run B — contrastive_coef=1.0  (contrastive-dominant)
# -----------------------------------------------------------------------
JOB_B=$(sbatch --parsable \
  ${SLURM_COMMON} \
  --job-name=vg-contr-B \
  --output="${LOG_SLURM}/vg_contrastive_B_%j.out" \
  --error="${LOG_SLURM}/vg_contrastive_B_%j.err" \
  --wrap="${WRAP_HEADER}
python train.py \\
  ${TRAIN_COMMON} \\
  --contrastive-coef 1.0 \\
  --log-dir ${LOG_DIR}/run_B_uniform_contrastive10_30ep
")
echo "Submitted B (uniform + contrastive=1.0, 30ep): ${JOB_B}"

echo ""
echo "Contrastive runs:"
echo "  A — uniform + contrastive=0.5 : ${JOB_A}"
echo "  B — uniform + contrastive=1.0 : ${JOB_B}"
echo ""
echo "Monitor with : squeue -u \$USER"
echo "Logs under   : ${LOG_SLURM}/"
echo "Checkpoints  : ${LOG_DIR}/run_{A,B}_*/"
echo ""
echo "After training, run the collapse check on each run:"
echo "  CKPT_DIR=${LOG_DIR}/run_A_uniform_contrastive05_30ep \\"
echo "  VARIANT=contrastive_A_30ep \\"
echo "  bash ~/proto-VLM/scripts/slurm_eval_collapse_check.sh"
