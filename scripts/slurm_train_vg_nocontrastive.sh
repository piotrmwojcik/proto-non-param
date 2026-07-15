#!/bin/bash
# Runs M1 and M2 — diversity losses only, no contrastive (InfoNCE off).
#
# M1 = ViT-L + SK + KoLeo           (no contrastive)
# M2 = ViT-L + SK + KoLeo + MSN     (no contrastive)
#
# Ablation purpose: isolate whether SK/KoLeo/MSN help without the
# contrastive signal, and whether they can substitute for it.
# Compare against: K2 (same losses + contrastive), H (baseline).
#
# Usage:
#   bash scripts/slurm_train_vg_nocontrastive.sh

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
  --sk-coef 0.1 \
  --sk-eps 0.10 \
  --koleo-coef 0.1 \
  --save-every 5 \
  --wandb-entity ${WANDB_ENTITY} \
  --wandb-log-images 16"

SLURM_COMMON="--partition=${PARTITION} \
  --account=${ACCOUNT} \
  --gres=gpu:1 \
  --cpus-per-task=8 \
  --mem=64G \
  --time=2-00:00:00"

# ---------------------------------------------------------------------------
# M1 — SK + KoLeo, no contrastive
# ---------------------------------------------------------------------------
JOB_M1=$(sbatch --parsable \
  ${SLURM_COMMON} \
  --job-name=vg-M1-sk-koleo \
  --output="${LOG_SLURM}/vg_M1_sk_koleo_%j.out" \
  --error="${LOG_SLURM}/vg_M1_sk_koleo_%j.err" \
  --wrap="${WRAP_HEADER}
python train.py \\
  ${TRAIN_COMMON} \\
  --log-dir ${LOG_DIR}/run_M1_vitl14_sk10_koleo01_30ep
")
echo "Submitted M1 (ViT-L, SK+KoLeo, no contrastive, 30ep): ${JOB_M1}"

# ---------------------------------------------------------------------------
# M2 — SK + KoLeo + MSN, no contrastive
# ---------------------------------------------------------------------------
JOB_M2=$(sbatch --parsable \
  ${SLURM_COMMON} \
  --job-name=vg-M2-sk-koleo-msn \
  --output="${LOG_SLURM}/vg_M2_sk_koleo_msn_%j.out" \
  --error="${LOG_SLURM}/vg_M2_sk_koleo_msn_%j.err" \
  --wrap="${WRAP_HEADER}
python train.py \\
  ${TRAIN_COMMON} \\
  --msn-coef 0.1 \\
  --msn-mask-ratio 0.25 \\
  --log-dir ${LOG_DIR}/run_M2_vitl14_sk10_koleo01_msn01_mask25_30ep
")
echo "Submitted M2 (ViT-L, SK+KoLeo+MSN, no contrastive, 30ep): ${JOB_M2}"

echo ""
echo "Monitor with : squeue -u \$USER"
echo "Logs under   : ${LOG_SLURM}/"
echo "Checkpoints  : ${LOG_DIR}/run_M{1,2}_*/"
