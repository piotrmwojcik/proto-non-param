#!/bin/bash
# KL + frozen residual training on full VG vocabulary with milestone checkpoints.
#
# Motivation: the 80-epoch full-vocab run in slurm_train_vg_long.sh lacked
# per-epoch checkpoints, cosine warmup, and zero residual init (all fixed
# after that run). This script re-trains with all fixes so the optimal epoch
# can be identified via slurm_eval_collapse_check.sh (same as dedup analysis).
#
# Key differences from slurm_train_vg_long.sh:
#   --save-every 5        → ckpt_ep005.pth … ckpt_ep080.pth (16 checkpoints)
#   --lr-warmup-epochs 5  → linear warmup then cosine decay to 0
#   prototype_residual    → zero-initialised (auto, residual-lr=0)
#
# ~10.5 min/epoch × 80 epochs ≈ 14 h; wall time set to 16 h.
# Checkpoint dir: ${LOG_DIR}/run_A_kl_frozen_fullvocab_80ep/
#
# After training, find the optimal epoch:
#   CKPT_DIR=$SCRATCH/train_logs/vg_fullvocab/run_A_kl_frozen_fullvocab_80ep \
#   VARIANT=fullvocab_A_80ep \
#   bash ~/proto-VLM/scripts/slurm_eval_collapse_check.sh
#
# Configuration — override via environment variables:
#   SCRATCH       Base scratch path (default: /net/tscratch/people/plgabedychaj)
#   VG_ROOT       Path to VG image root (containing VG_100K/ and VG_100K_2/)
#   VG_DESC       Path to region_descriptions.json
#   LOG_DIR       Base directory for checkpoints
#   VOCAB_DIR     Directory with vg_cache.pt
#   WANDB_ENTITY  W&B entity (default: gmum)
#
# Usage:
#   bash scripts/slurm_train_vg_fullvocab_long.sh

set -e

SCRATCH="${SCRATCH:-/net/tscratch/people/plgabedychaj}"
VG_ROOT="${VG_ROOT:-${SCRATCH}/vg}"
VG_DESC="${VG_DESC:-${VG_ROOT}/region_descriptions.json}"
LOG_DIR="${LOG_DIR:-${SCRATCH}/train_logs/vg_fullvocab}"
VOCAB_DIR="${VOCAB_DIR:-${SCRATCH}/vocab}"
WANDB_ENTITY="${WANDB_ENTITY:-gmum}"

PARTITION="plgrid-gpu-a100"
ACCOUNT="plgunhype-gpu-a100"
LOG_SLURM="${SCRATCH}/logs"

VG_CACHE="${VOCAB_DIR}/vg_cache.pt"

mkdir -p "${LOG_SLURM}" "${LOG_DIR}"

if [ ! -f "${VG_CACHE}" ]; then
    echo "ERROR: VG vocab cache not found at ${VG_CACHE}"
    echo "Build it first with: bash scripts/slurm_build_vg_vocab.sh"
    exit 1
fi

echo "=== VG Full-Vocab Long Training (KL + frozen δ, 80 epochs, save every 5) ==="
echo "  SCRATCH      : ${SCRATCH}"
echo "  VG_ROOT      : ${VG_ROOT}"
echo "  LOG_DIR      : ${LOG_DIR}"
echo "  Vocab        : ${VG_CACHE}"
echo "  Checkpoint   : ${LOG_DIR}/run_A_kl_frozen_fullvocab_80ep/"
echo ""

WRAP_HEADER="
set -e
export HF_HOME=${SCRATCH}/hf_cache
export TRANSFORMERS_CACHE=${SCRATCH}/hf_cache
export PYTHONPATH=${SCRATCH}/dinov2:\$PYTHONPATH
source ${SCRATCH}/venv/bin/activate
cd ~/proto-VLM/proto-non-param
"

JOB_A=$(sbatch --parsable \
  --job-name=vg-full-A-kl \
  --partition="${PARTITION}" \
  --account="${ACCOUNT}" \
  --gres=gpu:1 \
  --cpus-per-task=8 \
  --mem=64G \
  --time=16:00:00 \
  --output="${LOG_SLURM}/vg_fullvocab_A_%j.out" \
  --error="${LOG_SLURM}/vg_fullvocab_A_%j.err" \
  --wrap="${WRAP_HEADER}
python train.py \\
  --dataset visual_genome \\
  --vg-root ${VG_ROOT} \\
  --vg-region-descriptions ${VG_DESC} \\
  --vocab-cache-path ${VG_CACHE} \\
  --backbone dinov2_vitb14 \\
  --batch-size 64 \\
  --epochs 80 \\
  --lr-schedule cosine \\
  --lr-warmup-epochs 5 \\
  --num-workers 8 \\
  --backbone-lr 1e-5 \\
  --text-proj-lr 1e-4 \\
  --target-mode topk \\
  --top-k-concepts 5 \\
  --kl-coef 1.0 \\
  --loss-type kl \\
  --save-every 5 \\
  --wandb-entity ${WANDB_ENTITY} \\
  --wandb-log-images 16 \\
  --log-dir ${LOG_DIR}/run_A_kl_frozen_fullvocab_80ep
")

echo "Submitted job ${JOB_A}"
echo ""
echo "Monitor with : squeue -u \$USER"
echo "Log          : ${LOG_SLURM}/vg_fullvocab_A_${JOB_A}.out"
echo ""
echo "After training, find the optimal epoch with:"
echo "  CKPT_DIR=${LOG_DIR}/run_A_kl_frozen_fullvocab_80ep \\"
echo "  VARIANT=fullvocab_A_80ep \\"
echo "  bash ~/proto-VLM/scripts/slurm_eval_collapse_check.sh"
echo ""
echo "Then run full RIS eval on the best epoch checkpoint:"
echo "  CKPT=${LOG_DIR}/run_A_kl_frozen_fullvocab_80ep/ckpt_ep<N>.pth \\"
echo "  VARIANT=fullvocab_A_ep<N> \\"
echo "  bash ~/proto-VLM/scripts/slurm_eval_pnp_threshold_sweep.sh"
