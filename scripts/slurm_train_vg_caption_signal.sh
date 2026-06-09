#!/bin/bash
# Ablation: word-level top-5 vs caption-level vs combined training signal.
#
# Submits 5 SLURM jobs:
#   1. build-vg-vocab          Build word-level VG vocabulary cache (CPU) — skip if exists
#   2. build-caption-embs      Build per-image CLIP phrase embedding pool (GPU), both splits
#   3A. train-signal-A         Word top-5 only (baseline, kl_coef=1 caption_coef=0)
#   3B. train-signal-B         Caption-only   (kl_coef=0 caption_coef=1)
#   3C. train-signal-C         Combined       (kl_coef=1 caption_coef=1)
#
# 3A depends on job 1 only; 3B and 3C depend on jobs 1 and 2.
#
# Configuration — override via environment variables:
#   VG_ROOT         Path to VG image root (containing VG_100K/ and VG_100K_2/)
#   VG_DESC         Path to region_descriptions.json
#   LOG_DIR         Base directory for training logs and checkpoints
#   VOCAB_DIR       Directory where vocabulary caches are saved
#   WANDB_ENTITY    W&B entity (team or username)
#
# Usage:
#   bash scripts/slurm_train_vg_caption_signal.sh
#   LOG_DIR=/my/logs bash scripts/slurm_train_vg_caption_signal.sh

set -e

# ---- Cluster paths (Athena/PLGrid defaults) ----
SCRATCH="/net/tscratch/people/plgabedychaj"
VG_ROOT="${VG_ROOT:-${SCRATCH}/vg}"
VG_DESC="${VG_DESC:-${VG_ROOT}/region_descriptions.json}"
LOG_DIR="${LOG_DIR:-${SCRATCH}/train_logs/vg_caption_signal_ablation}"
VOCAB_DIR="${VOCAB_DIR:-${SCRATCH}/vocab}"
WANDB_ENTITY="${WANDB_ENTITY:-gmum}"

PARTITION="plgrid-gpu-a100"
ACCOUNT="plgunhype-gpu-a100"
LOG_SLURM="${SCRATCH}/logs"

mkdir -p "${LOG_SLURM}" "${VOCAB_DIR}" "${LOG_DIR}"

CAPTION_EMBS="${VOCAB_DIR}/vg_caption_embs.pt"
VG_CACHE="${VOCAB_DIR}/vg_cache.pt"

echo "=== VG Caption Signal Ablation ==="
echo "  VG_ROOT         : ${VG_ROOT}"
echo "  CAPTION_EMBS    : ${CAPTION_EMBS}"
echo "  LOG_DIR         : ${LOG_DIR}"
echo ""

# -----------------------------------------------------------------------
# Job 1 — Build VG word vocabulary (CPU, ~30 min)
# -----------------------------------------------------------------------
JOB1=$(sbatch --parsable \
  --job-name=vg-vocab \
  --partition="${PARTITION}" \
  --account="${ACCOUNT}" \
  --cpus-per-task=4 \
  --mem=32G \
  --time=01:00:00 \
  --output="${LOG_SLURM}/vg_vocab_%j.out" \
  --error="${LOG_SLURM}/vg_vocab_%j.err" \
  --wrap="
set -e
export HF_HOME=${SCRATCH}/hf_cache
export PYTHONPATH=${SCRATCH}/dinov2:\$PYTHONPATH
source ${SCRATCH}/venv/bin/activate
cd ~/proto-VLM/proto-non-param

python vocab/build_vg_vocab.py \\
  --region-descriptions ${VG_DESC} \\
  --vocab-out  ${VOCAB_DIR}/vg.txt \\
  --cache-out  ${VG_CACHE} \\
  --clip-model-name ViT-B-32 \\
  --clip-pretrained openai \\
  --min-count 5 \\
  --max-doc-freq 0.5
")
echo "Submitted job 1 (build-vg-vocab): ${JOB1}"

# -----------------------------------------------------------------------
# Job 2 — Build per-image caption embedding pool (GPU, ~1 h for 108K images)
# Depends on job 1 (needs vg_cache.pt to reproduce the split)
# -----------------------------------------------------------------------
JOB2=$(sbatch --parsable \
  --job-name=build-caption-embs \
  --partition="${PARTITION}" \
  --account="${ACCOUNT}" \
  --gres=gpu:1 \
  --cpus-per-task=4 \
  --mem=32G \
  --time=02:00:00 \
  --dependency=afterok:"${JOB1}" \
  --output="${LOG_SLURM}/caption_embs_%j.out" \
  --error="${LOG_SLURM}/caption_embs_%j.err" \
  --wrap="
set -e
export HF_HOME=${SCRATCH}/hf_cache
export PYTHONPATH=${SCRATCH}/dinov2:\$PYTHONPATH
source ${SCRATCH}/venv/bin/activate
cd ~/proto-VLM/proto-non-param

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
echo "Submitted job 2 (build-caption-embs): ${JOB2}"

# -----------------------------------------------------------------------
# Shared training args
# -----------------------------------------------------------------------
TRAIN_COMMON="
  --dataset visual_genome \\
  --vg-root ${VG_ROOT} \\
  --vg-region-descriptions ${VG_DESC} \\
  --vocab-cache-path ${VG_CACHE} \\
  --backbone dinov2_vitb14 \\
  --batch-size 64 \\
  --epochs 20 \\
  --num-workers 8 \\
  --backbone-lr 1e-5 \\
  --text-proj-lr 1e-4 \\
  --target-type topk \\
  --top-k-concepts 5 \\
  --wandb-entity ${WANDB_ENTITY} \\
  --wandb-log-images 16"

# -----------------------------------------------------------------------
# Job 3A — Train: word top-5 only (baseline)
# Depends on job 1 only (no caption embeddings needed)
# -----------------------------------------------------------------------
JOB3A=$(sbatch --parsable \
  --job-name=train-sig-A \
  --partition="${PARTITION}" \
  --account="${ACCOUNT}" \
  --gres=gpu:1 \
  --cpus-per-task=8 \
  --mem=64G \
  --time=2-00:00:00 \
  --dependency=afterok:"${JOB1}" \
  --output="${LOG_SLURM}/train_sig_A_%j.out" \
  --error="${LOG_SLURM}/train_sig_A_%j.err" \
  --wrap="
set -e
export HF_HOME=${SCRATCH}/hf_cache
export TRANSFORMERS_CACHE=${SCRATCH}/hf_cache
export PYTHONPATH=${SCRATCH}/dinov2:\$PYTHONPATH
source ${SCRATCH}/venv/bin/activate
cd ~/proto-VLM/proto-non-param

python train.py \\
  ${TRAIN_COMMON} \\
  --log-dir ${LOG_DIR}/run_A_word_only \\
  --kl-coef 1.0 \\
  --caption-coef 0.0
")
echo "Submitted job 3A (train word-only): ${JOB3A}"

# -----------------------------------------------------------------------
# Job 3B — Train: caption-only (kl_coef=0)
# Depends on jobs 1 and 2
# -----------------------------------------------------------------------
JOB3B=$(sbatch --parsable \
  --job-name=train-sig-B \
  --partition="${PARTITION}" \
  --account="${ACCOUNT}" \
  --gres=gpu:1 \
  --cpus-per-task=8 \
  --mem=64G \
  --time=2-00:00:00 \
  --dependency=afterok:"${JOB1}:${JOB2}" \
  --output="${LOG_SLURM}/train_sig_B_%j.out" \
  --error="${LOG_SLURM}/train_sig_B_%j.err" \
  --wrap="
set -e
export HF_HOME=${SCRATCH}/hf_cache
export TRANSFORMERS_CACHE=${SCRATCH}/hf_cache
export PYTHONPATH=${SCRATCH}/dinov2:\$PYTHONPATH
source ${SCRATCH}/venv/bin/activate
cd ~/proto-VLM/proto-non-param

python train.py \\
  ${TRAIN_COMMON} \\
  --log-dir ${LOG_DIR}/run_B_caption_only \\
  --kl-coef 0.0 \\
  --caption-coef 1.0 \\
  --caption-embeds-path ${CAPTION_EMBS} \\
  --caption-sample-k 5
")
echo "Submitted job 3B (train caption-only): ${JOB3B}"

# -----------------------------------------------------------------------
# Job 3C — Train: combined (kl_coef=1 + caption_coef=1)
# Depends on jobs 1 and 2
# -----------------------------------------------------------------------
JOB3C=$(sbatch --parsable \
  --job-name=train-sig-C \
  --partition="${PARTITION}" \
  --account="${ACCOUNT}" \
  --gres=gpu:1 \
  --cpus-per-task=8 \
  --mem=64G \
  --time=2-00:00:00 \
  --dependency=afterok:"${JOB1}:${JOB2}" \
  --output="${LOG_SLURM}/train_sig_C_%j.out" \
  --error="${LOG_SLURM}/train_sig_C_%j.err" \
  --wrap="
set -e
export HF_HOME=${SCRATCH}/hf_cache
export TRANSFORMERS_CACHE=${SCRATCH}/hf_cache
export PYTHONPATH=${SCRATCH}/dinov2:\$PYTHONPATH
source ${SCRATCH}/venv/bin/activate
cd ~/proto-VLM/proto-non-param

python train.py \\
  ${TRAIN_COMMON} \\
  --log-dir ${LOG_DIR}/run_C_combined \\
  --kl-coef 1.0 \\
  --caption-coef 1.0 \\
  --caption-embeds-path ${CAPTION_EMBS} \\
  --caption-sample-k 5
")
echo "Submitted job 3C (train combined): ${JOB3C}"

echo ""
echo "Pipeline submitted. Job chain:"
echo "  ${JOB1} (vocab) ──────────────────────────────→ ${JOB3A} (A: word-only)"
echo "  ${JOB1} (vocab) → ${JOB2} (caption-embs) ───→ ${JOB3B} (B: caption-only)"
echo "                   ${JOB2} (caption-embs) ───→ ${JOB3C} (C: combined)"
echo ""
echo "Monitor with: squeue -u \$USER"
echo "Logs under  : ${LOG_SLURM}/"
echo "Checkpoints : ${LOG_DIR}/run_{A,B,C}*/"
