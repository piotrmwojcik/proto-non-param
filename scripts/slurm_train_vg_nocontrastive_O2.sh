#!/bin/bash
# Run O2 — re-run of O with adjusted loss weights.
#
# O2 = ViT-L + SK + SigReg (no contrastive, no KoLeo)
#      sigreg-coef: 0.02 → 0.5   (was <1% of total loss, now ~6%)
#      sk-coef:     0.1  → 0.3   (was ~3% of total loss, now ~8%)
#
# Ablation purpose:
#   O2 vs O:  does meaningful SigReg gradient signal improve over the dead-loss O run?
#   O2 vs M1: can SK+SigReg match SK+KoLeo with correct loss scaling?
#
# Usage:
#   bash scripts/slurm_train_vg_nocontrastive_O2.sh

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

mkdir -p "${LOG_SLURM}" "${LOG_DIR}"

if [ ! -f "${VG_CACHE}" ]; then
    echo "ERROR: required file not found: ${VG_CACHE}"
    exit 1
fi

WRAP_HEADER="
set -e
export HF_HOME=${SCRATCH}/hf_cache
export TRANSFORMERS_CACHE=${SCRATCH}/hf_cache
export TORCH_HOME=${SCRATCH}/torch_cache
export PYTHONPATH=${SCRATCH}/dinov2:\$PYTHONPATH
source ${SCRATCH}/venv/bin/activate
cd ~/proto-non-param
"

TRAIN_COMMON="--dataset visual_genome \
  --vg-root ${VG_ROOT} \
  --vg-region-descriptions ${VG_DESC} \
  --vocab-cache-path ${VG_CACHE} \
  --backbone dinov2_vitl14 \
  --batch-size 64 \
  --epochs 30 \
  --lr-schedule cosine \
  --lr-warmup-epochs 5 \
  --num-workers 8 \
  --backbone-lr 1e-5 \
  --text-proj-lr 1e-4 \
  --text-proj-hidden-dim 2048 \
  --target-mode uniform \
  --loss-type kl \
  --kl-coef 1.0 \
  --contrastive-coef 0.0 \
  --save-every 5 \
  --wandb-entity ${WANDB_ENTITY} \
  --wandb-log-images 16"

SLURM_COMMON="--partition=${PARTITION} \
  --account=${ACCOUNT} \
  --gres=gpu:1 \
  --cpus-per-task=8 \
  --mem=64G \
  --time=2-00:00:00"

JOB_O2=$(sbatch --parsable \
  ${SLURM_COMMON} \
  --job-name=vg-O2-sk03-sigreg05 \
  --output="${LOG_SLURM}/vg_O2_sk03_sigreg05_%j.out" \
  --error="${LOG_SLURM}/vg_O2_sk03_sigreg05_%j.err" \
  --wrap="${WRAP_HEADER}
python train.py \\
  ${TRAIN_COMMON} \\
  --sk-coef 0.3 \\
  --sk-eps 0.10 \\
  --sigreg-coef 0.5 \\
  --sigreg-sketch-dim 64 \\
  --log-dir ${LOG_DIR}/run_O2_vitl14_sk03_sigreg05_30ep
")
echo "Submitted O2 (ViT-L, SK+SigReg, sk=0.3, sigreg=0.5, no contrastive, 30ep): ${JOB_O2}"

echo ""
echo "Monitor with : squeue -u \$USER"
echo "Logs under   : ${LOG_SLURM}/"
echo "Checkpoint   : ${LOG_DIR}/run_O2_vitl14_sk03_sigreg05_30ep/"
