#!/bin/bash
# Dedup-vocab ablation: train on deduplicated VG vocabulary, 20 epochs.
#
# Mirrors the frozen-residual candidates from slurm_train_vg_ablation.sh
# (runs A and C only), but replaces the training vocabulary with a
# deduplicated version produced by vocab/deduplicate_vocab.py.
#
# Pipeline
# --------
#   Job 0 (CPU, ~10 min) — build vg_cache_dedup_t${THRESHOLD_STR}.pt
#                          Skipped if the file already exists.
#   Job A (GPU, 20 ep)   — KL  + frozen residual, dedup vocab
#   Job C (GPU, 20 ep)   — JSD + frozen residual, dedup vocab
#
# Compare against the non-dedup ablation in slurm_train_vg_ablation.sh.
# Eval with slurm_eval_vg_dedup.sh which runs both dedup-vocab and
# full-vocab inference and reports P@K / R@K.
#
# Configuration — override via environment variables:
#   SCRATCH       Base scratch path (default: /net/tscratch/people/plgabedychaj)
#   VG_ROOT       Path to VG image root (containing VG_100K/ and VG_100K_2/)
#   VG_DESC       Path to region_descriptions.json
#   LOG_DIR       Base directory for checkpoints
#   VOCAB_DIR     Directory with vg_cache.pt and dedup output
#   THRESHOLD     Cosine-similarity threshold for dedup (default: 0.90)
#   WANDB_ENTITY  W&B entity (default: gmum)
#
# Usage:
#   bash scripts/slurm_train_vg_dedup.sh
#   THRESHOLD=0.85 bash scripts/slurm_train_vg_dedup.sh

set -e

SCRATCH="${SCRATCH:-/net/tscratch/people/plgabedychaj}"
VG_ROOT="${VG_ROOT:-${SCRATCH}/vg}"
VG_DESC="${VG_DESC:-${VG_ROOT}/region_descriptions.json}"
LOG_DIR="${LOG_DIR:-${SCRATCH}/train_logs/vg_dedup}"
VOCAB_DIR="${VOCAB_DIR:-${SCRATCH}/vocab}"
WANDB_ENTITY="${WANDB_ENTITY:-gmum}"
THRESHOLD="${THRESHOLD:-0.90}"

PARTITION="plgrid-gpu-a100"
ACCOUNT="plgunhype-gpu-a100"
LOG_SLURM="${SCRATCH}/logs"

VG_CACHE="${VOCAB_DIR}/vg_cache.pt"
THRESHOLD_STR="${THRESHOLD/./}"          # "0.90" → "090"
DEDUP_CACHE="${VOCAB_DIR}/vg_cache_dedup_t${THRESHOLD_STR}.pt"
DEDUP_MAPPING="${VOCAB_DIR}/vg_dedup_mapping_t${THRESHOLD_STR}.json"

mkdir -p "${LOG_SLURM}" "${LOG_DIR}"

if [ ! -f "${VG_CACHE}" ]; then
    echo "ERROR: VG vocab cache not found at ${VG_CACHE}"
    echo "Build it first: bash scripts/slurm_build_vg_vocab.sh"
    exit 1
fi

echo "=== VG Dedup-Vocab Ablation (θ=${THRESHOLD}) ==="
echo "  SCRATCH      : ${SCRATCH}"
echo "  VG_ROOT      : ${VG_ROOT}"
echo "  LOG_DIR      : ${LOG_DIR}"
echo "  Full vocab   : ${VG_CACHE}"
echo "  Dedup vocab  : ${DEDUP_CACHE}"
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
# Job 0 — build deduplicated vocab cache (CPU-only, fast)
# Skipped if cache already exists.
# -----------------------------------------------------------------------
if [ -f "${DEDUP_CACHE}" ]; then
    echo "Dedup cache already exists — skipping build job."
    DEDUP_DEP=""
else
    JOB0=$(sbatch --parsable \
      --job-name=vg-dedup-vocab \
      --partition="${PARTITION}" \
      --account="${ACCOUNT}" \
      --cpus-per-task=4 \
      --mem=32G \
      --time=00:30:00 \
      --output="${LOG_SLURM}/vg_dedup_vocab_%j.out" \
      --error="${LOG_SLURM}/vg_dedup_vocab_%j.err" \
      --wrap="${WRAP_HEADER}
python vocab/deduplicate_vocab.py \\
  --cache-in    ${VG_CACHE} \\
  --cache-out   ${DEDUP_CACHE} \\
  --mapping-out ${DEDUP_MAPPING} \\
  --threshold   ${THRESHOLD}
")
    echo "Submitted job 0 (build dedup vocab): ${JOB0}"
    DEDUP_DEP="--dependency=afterok:${JOB0}"
fi

# ---- Shared training args ----
TRAIN_COMMON="--dataset visual_genome \
  --vg-root ${VG_ROOT} \
  --vg-region-descriptions ${VG_DESC} \
  --vocab-cache-path ${DEDUP_CACHE} \
  --backbone dinov2_vitb14 \
  --batch-size 64 \
  --epochs 20 \
  --num-workers 8 \
  --backbone-lr 1e-5 \
  --text-proj-lr 1e-4 \
  --target-mode topk \
  --top-k-concepts 5 \
  --kl-coef 1.0 \
  --wandb-entity ${WANDB_ENTITY} \
  --wandb-log-images 16"

SLURM_COMMON="${DEDUP_DEP} \
  --partition=${PARTITION} \
  --account=${ACCOUNT} \
  --gres=gpu:1 \
  --cpus-per-task=8 \
  --mem=64G \
  --time=2-00:00:00"

# -----------------------------------------------------------------------
# Job A — KL + frozen residual, dedup vocab
# -----------------------------------------------------------------------
JOB_A=$(sbatch --parsable \
  ${SLURM_COMMON} \
  --job-name=vg-dedup-A-kl \
  --output="${LOG_SLURM}/vg_dedup_A_%j.out" \
  --error="${LOG_SLURM}/vg_dedup_A_%j.err" \
  --wrap="${WRAP_HEADER}
python train.py \\
  ${TRAIN_COMMON} \\
  --loss-type kl \\
  --log-dir ${LOG_DIR}/run_A_kl_frozen_dedup_t${THRESHOLD_STR}
")
echo "Submitted A (KL + frozen residual, dedup vocab): ${JOB_A}"

# -----------------------------------------------------------------------
# Job C — JSD + frozen residual, dedup vocab
# -----------------------------------------------------------------------
JOB_C=$(sbatch --parsable \
  ${SLURM_COMMON} \
  --job-name=vg-dedup-C-jsd \
  --output="${LOG_SLURM}/vg_dedup_C_%j.out" \
  --error="${LOG_SLURM}/vg_dedup_C_%j.err" \
  --wrap="${WRAP_HEADER}
python train.py \\
  ${TRAIN_COMMON} \\
  --loss-type jsd \\
  --log-dir ${LOG_DIR}/run_C_jsd_frozen_dedup_t${THRESHOLD_STR}
")
echo "Submitted C (JSD + frozen residual, dedup vocab): ${JOB_C}"

echo ""
echo "Dedup ablation (θ=${THRESHOLD}):"
echo "  A — KL  + frozen residual : ${JOB_A}"
echo "  C — JSD + frozen residual : ${JOB_C}"
echo ""
echo "When complete, evaluate with:"
echo "  CKPT_A=${LOG_DIR}/run_A_kl_frozen_dedup_t${THRESHOLD_STR}/ckpt.pth \\"
echo "  CKPT_C=${LOG_DIR}/run_C_jsd_frozen_dedup_t${THRESHOLD_STR}/ckpt.pth \\"
echo "  bash scripts/slurm_eval_vg_dedup.sh"
echo ""
echo "Monitor with : squeue -u \$USER"
echo "Logs under   : ${LOG_SLURM}/"
