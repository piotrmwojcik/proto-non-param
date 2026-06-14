#!/bin/bash
# 2x2 ablation: (KL vs JSD) x (frozen residual vs trained+constrained residual) on Caltech-101.
#
# Submits 4 independent SLURM jobs (no dependency chain; all can run in parallel):
#   A — baseline : KL loss  + frozen residual    (replicates current model behavior)
#   B — residual : KL loss  + trained residual   (isolates residual effect)
#   C — JSD      : JSD loss + frozen residual    (isolates loss effect)
#   D — both     : JSD loss + trained residual   (full ProtoLang proposal)
#
# All runs use --target-mode topk (uniform 1/K distribution).
# topk is required for JSD to have clean zeros providing the negative signal.
#
# Configuration — override via environment variables:
#   SCRATCH       Base scratch path (default: /net/tscratch/people/plgabedychaj)
#   LOG_DIR       Base directory for checkpoints (default: ${SCRATCH}/train_logs/caltech_ablation)
#   WANDB_ENTITY  W&B entity (default: gmum)
#
# Usage:
#   bash scripts/slurm_train_caltech_ablation.sh
#   SCRATCH=/my/scratch bash scripts/slurm_train_caltech_ablation.sh

set -e

# ---- Cluster paths ----
SCRATCH="${SCRATCH:-/net/tscratch/people/plgabedychaj}"
LOG_DIR="${LOG_DIR:-${SCRATCH}/train_logs/caltech_ablation}"
WANDB_ENTITY="${WANDB_ENTITY:-gmum}"

PARTITION="plgrid-gpu-a100"
ACCOUNT="plgbcfg-gpu-a100"
LOG_SLURM="${SCRATCH}/logs"

CALTECH_ROOT="${SCRATCH}/caltech101"
CALTECH_DESC="${SCRATCH}/descriptions/caltech101_descriptions.json"
VOCAB_CACHE="${SCRATCH}/vocab/caltech101_cache.pt"

mkdir -p "${LOG_SLURM}" "${LOG_DIR}"

# Pre-flight: vocab cache must exist
if [ ! -f "${VOCAB_CACHE}" ]; then
    echo "ERROR: vocab cache not found at ${VOCAB_CACHE}"
    echo "Build it first with: python vocab/build_vocab.py"
    exit 1
fi

echo "=== Caltech-101 Residual × Loss Ablation ==="
echo "  SCRATCH   : ${SCRATCH}"
echo "  LOG_DIR   : ${LOG_DIR}"
echo "  VOCAB     : ${VOCAB_CACHE}"
echo ""

# ---- Shared training args ----
TRAIN_COMMON="--dataset caltech101 \
  --caltech-root ${CALTECH_ROOT} \
  --caltech-descriptions ${CALTECH_DESC} \
  --vocab-cache-path ${VOCAB_CACHE} \
  --backbone dinov2_vitb14 \
  --batch-size 64 \
  --epochs 50 \
  --num-workers 8 \
  --backbone-lr 1e-5 \
  --text-proj-lr 1e-4 \
  --target-mode topk \
  --top-k-concepts 10 \
  --kl-coef 1.0 \
  --wandb-entity ${WANDB_ENTITY} \
  --wandb-log-images 16"

SLURM_COMMON="--partition=${PARTITION} \
  --account=${ACCOUNT} \
  --gres=gpu:1 \
  --cpus-per-task=8 \
  --mem=64G \
  --time=2-00:00:00"

WRAP_HEADER="
set -e
export HF_HOME=${SCRATCH}/hf_cache
export TRANSFORMERS_CACHE=${SCRATCH}/hf_cache
export PYTHONPATH=${SCRATCH}/dinov2:\$PYTHONPATH
source ${SCRATCH}/venv/bin/activate
cd ~/proto-VLM/proto-non-param
"

# -----------------------------------------------------------------------
# Run A — baseline: KL + frozen residual (current behavior)
# -----------------------------------------------------------------------
JOB_A=$(sbatch --parsable \
  ${SLURM_COMMON} \
  --job-name=cal-A-kl-base \
  --output="${LOG_SLURM}/caltech_abl_A_%j.out" \
  --error="${LOG_SLURM}/caltech_abl_A_%j.err" \
  --wrap="${WRAP_HEADER}
python train.py \\
  ${TRAIN_COMMON} \\
  --loss-type kl \\
  --log-dir ${LOG_DIR}/run_A_kl_frozen
")
echo "Submitted A (KL + frozen residual): ${JOB_A}"

# -----------------------------------------------------------------------
# Run B — residual: KL + trained + constrained residual
# -----------------------------------------------------------------------
JOB_B=$(sbatch --parsable \
  ${SLURM_COMMON} \
  --job-name=cal-B-kl-res \
  --output="${LOG_SLURM}/caltech_abl_B_%j.out" \
  --error="${LOG_SLURM}/caltech_abl_B_%j.err" \
  --wrap="${WRAP_HEADER}
python train.py \\
  ${TRAIN_COMMON} \\
  --loss-type kl \\
  --residual-lr 1e-4 \\
  --residual-eps 0.1 \\
  --residual-reg-coef 0.01 \\
  --log-dir ${LOG_DIR}/run_B_kl_residual
")
echo "Submitted B (KL + trained residual): ${JOB_B}"

# -----------------------------------------------------------------------
# Run C — JSD: JSD + frozen residual
# -----------------------------------------------------------------------
JOB_C=$(sbatch --parsable \
  ${SLURM_COMMON} \
  --job-name=cal-C-jsd-base \
  --output="${LOG_SLURM}/caltech_abl_C_%j.out" \
  --error="${LOG_SLURM}/caltech_abl_C_%j.err" \
  --wrap="${WRAP_HEADER}
python train.py \\
  ${TRAIN_COMMON} \\
  --loss-type jsd \\
  --log-dir ${LOG_DIR}/run_C_jsd_frozen
")
echo "Submitted C (JSD + frozen residual): ${JOB_C}"

# -----------------------------------------------------------------------
# Run D — both: JSD + trained + constrained residual
# -----------------------------------------------------------------------
JOB_D=$(sbatch --parsable \
  ${SLURM_COMMON} \
  --job-name=cal-D-jsd-res \
  --output="${LOG_SLURM}/caltech_abl_D_%j.out" \
  --error="${LOG_SLURM}/caltech_abl_D_%j.err" \
  --wrap="${WRAP_HEADER}
python train.py \\
  ${TRAIN_COMMON} \\
  --loss-type jsd \\
  --residual-lr 1e-4 \\
  --residual-eps 0.1 \\
  --residual-reg-coef 0.01 \\
  --log-dir ${LOG_DIR}/run_D_jsd_residual
")
echo "Submitted D (JSD + trained residual): ${JOB_D}"

echo ""
echo "Ablation matrix:"
echo "              | Frozen residual | Trained residual"
echo "  KL loss     | ${JOB_A} (A)   | ${JOB_B} (B)"
echo "  JSD loss    | ${JOB_C} (C)   | ${JOB_D} (D)"
echo ""
echo "Monitor with : squeue -u \$USER"
echo "Logs under   : ${LOG_SLURM}/"
echo "Checkpoints  : ${LOG_DIR}/run_{A,B,C,D}_*/"
