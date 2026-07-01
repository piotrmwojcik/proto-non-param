#!/bin/bash
# Run I — ViT-L/14 + best contrastive strategy from ViT-B sweep (run G).
#
#   dinov2_vitl14  (dim=1024, 24+1 blocks with n_splits=1)
#   KL + contrastive_coef=1.0, k=5, hard mining, soft labels (tau=0.5)
#   text_proj_hidden_dim=2048  (scaled from 1024 for vitb)
#   batch_size=64  (reduced from 128 to fit ViT-L on one A100)
#
# Configuration — override via environment variables:
#   SCRATCH, VG_ROOT, VG_DESC, LOG_DIR, VOCAB_DIR, WANDB_ENTITY
#
# Usage:
#   bash scripts/slurm_train_vg_contrastive_vitl_best.sh

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
export TORCH_HOME=${SCRATCH}/torch_cache
export PYTHONPATH=${SCRATCH}/dinov2:\$PYTHONPATH
source ${SCRATCH}/venv/bin/activate
cd ~/proto-non-param
"

JOB_I=$(sbatch --parsable \
  --partition="${PARTITION}" \
  --account="${ACCOUNT}" \
  --job-name=vg-contr-I-vitl-best \
  --gres=gpu:1 \
  --cpus-per-task=8 \
  --mem=64G \
  --time=2-00:00:00 \
  --output="${LOG_SLURM}/vg_contrastive_I_vitl_best_%j.out" \
  --error="${LOG_SLURM}/vg_contrastive_I_vitl_best_%j.err" \
  --wrap="${WRAP_HEADER}
python train.py \\
  --dataset visual_genome \\
  --vg-root ${VG_ROOT} \\
  --vg-region-descriptions ${VG_DESC} \\
  --vocab-cache-path ${VG_CACHE} \\
  --backbone dinov2_vitl14 \\
  --batch-size 64 \\
  --epochs 30 \\
  --lr-schedule cosine \\
  --lr-warmup-epochs 5 \\
  --num-workers 8 \\
  --backbone-lr 1e-5 \\
  --text-proj-lr 1e-4 \\
  --text-proj-hidden-dim 2048 \\
  --target-mode uniform \\
  --loss-type kl \\
  --kl-coef 1.0 \\
  --caption-embeds-path ${CAPTION_EMBS} \\
  --contrastive-temp 0.07 \\
  --contrastive-coef 1.0 \\
  --caption-sample-k 5 \\
  --contrastive-hard-mining \\
  --contrastive-label-temp 0.5 \\
  --save-every 5 \\
  --wandb-entity ${WANDB_ENTITY} \\
  --wandb-log-images 16 \\
  --log-dir ${LOG_DIR}/run_I_vitl14_contrastive10_k5_hardmining_soft05_30ep
")

echo "Submitted I (ViT-L/14, k=5, hard mining, soft labels tau=0.5, 30ep): ${JOB_I}"
echo ""
echo "Monitor with : squeue -u \$USER"
echo "Logs under   : ${LOG_SLURM}/"
echo "Checkpoint   : ${LOG_DIR}/run_I_vitl14_contrastive10_k5_hardmining_soft05_30ep/"
