#!/bin/bash
# Submit runs F and G: deduplication + online hard positive mining (top-k by pred_emb sim).
# Use this when runs A–E are already done.
#
# Run F — k=5, hard mining, hard InfoNCE labels
#   --contrastive-hard-mining --caption-sample-k 5
#
# Run G — k=5, hard mining, soft negatives (tau_label=0.5)
#   --contrastive-hard-mining --caption-sample-k 5 --contrastive-label-temp 0.5
#
# Both keep: KL + contrastive_coef=1.0, 30 epochs, uniform target, frozen residual
#
# Configuration — override via environment variables:
#   SCRATCH, VG_ROOT, VG_DESC, LOG_DIR, VOCAB_DIR, WANDB_ENTITY
#
# Usage:
#   bash scripts/slurm_train_vg_contrastive_FG.sh

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

for f in "${VG_CACHE}" "${CAPTION_EMBS}"; do
    if [ ! -f "${f}" ]; then
        echo "ERROR: required file not found: ${f}"
        exit 1
    fi
done

WRAP_HEADER="
set -e
export HF_HOME=${SCRATCH}/hf_cache
export TRANSFORMERS_CACHE=${SCRATCH}/hf_cache
export PYTHONPATH=${SCRATCH}/dinov2:\$PYTHONPATH
source ${SCRATCH}/venv/bin/activate
cd ~/proto-non-param
"

TRAIN_COMMON="--dataset visual_genome \
  --vg-root ${VG_ROOT} \
  --vg-region-descriptions ${VG_DESC} \
  --vocab-cache-path ${VG_CACHE} \
  --backbone dinov2_vitb14 \
  --batch-size 128 \
  --epochs 30 \
  --lr-schedule cosine \
  --lr-warmup-epochs 5 \
  --num-workers 8 \
  --backbone-lr 1e-5 \
  --text-proj-lr 1e-4 \
  --target-mode uniform \
  --loss-type kl \
  --kl-coef 1.0 \
  --caption-embeds-path ${CAPTION_EMBS} \
  --contrastive-temp 0.07 \
  --contrastive-coef 1.0 \
  --caption-sample-k 5 \
  --contrastive-hard-mining \
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
# Run F — hard mining, hard InfoNCE labels
# ---------------------------------------------------------------------------
JOB_F=$(sbatch --parsable \
  ${SLURM_COMMON} \
  --job-name=vg-contr-F \
  --output="${LOG_SLURM}/vg_contrastive_F_%j.out" \
  --error="${LOG_SLURM}/vg_contrastive_F_%j.err" \
  --wrap="${WRAP_HEADER}
python train.py \\
  ${TRAIN_COMMON} \\
  --log-dir ${LOG_DIR}/run_F_contrastive10_k5_hardmining_30ep
")
echo "Submitted F (k=5, hard mining, hard InfoNCE, 30ep): ${JOB_F}"

# ---------------------------------------------------------------------------
# Run G — hard mining + soft negatives (tau_label=0.5)
# ---------------------------------------------------------------------------
JOB_G=$(sbatch --parsable \
  ${SLURM_COMMON} \
  --job-name=vg-contr-G \
  --output="${LOG_SLURM}/vg_contrastive_G_%j.out" \
  --error="${LOG_SLURM}/vg_contrastive_G_%j.err" \
  --wrap="${WRAP_HEADER}
python train.py \\
  ${TRAIN_COMMON} \\
  --contrastive-label-temp 0.5 \\
  --log-dir ${LOG_DIR}/run_G_contrastive10_k5_hardmining_soft05_30ep
")
echo "Submitted G (k=5, hard mining, soft negatives tau=0.5, 30ep): ${JOB_G}"

echo ""
echo "Contrastive runs F and G:"
echo "  F — k=5, dedup, hard mining, hard labels  : ${JOB_F}"
echo "  G — k=5, dedup, hard mining, soft labels  : ${JOB_G}"
echo ""
echo "Monitor with : squeue -u \$USER"
echo "Logs under   : ${LOG_SLURM}/"
echo "Checkpoints  : ${LOG_DIR}/run_{F,G}_*/"
