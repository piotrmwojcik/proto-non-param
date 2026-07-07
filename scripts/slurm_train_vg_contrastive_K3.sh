#!/bin/bash
# Run K3 — ViT-L/14, k=5, hard mining + KoLeo.
#
# Tests whether hard mining (select hardest caption among k=5) produces
# sharper pred_text_embeddings that KoLeo can spread more effectively.
# Baseline comparison: K1 (k=1, KoLeo only) and H (k=1, no diversity loss).
#
# Usage:
#   bash scripts/slurm_train_vg_contrastive_K3.sh

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

JOB=$(sbatch --parsable \
  --job-name=vg-contr-K3-hardmine-koleo \
  --partition="${PARTITION}" \
  --account="${ACCOUNT}" \
  --gres=gpu:1 \
  --cpus-per-task=8 \
  --mem=64G \
  --time=2-00:00:00 \
  --output="${LOG_SLURM}/vg_contrastive_K3_hardmine_koleo_%j.out" \
  --error="${LOG_SLURM}/vg_contrastive_K3_hardmine_koleo_%j.err" \
  --wrap="
set -e
export HF_HOME=${SCRATCH}/hf_cache
export TRANSFORMERS_CACHE=${SCRATCH}/hf_cache
export TORCH_HOME=${SCRATCH}/torch_cache
export PYTHONPATH=${SCRATCH}/dinov2:\$PYTHONPATH
source ${SCRATCH}/venv/bin/activate
cd ~/proto-non-param

python train.py \
  --dataset visual_genome \
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
  --caption-embeds-path ${CAPTION_EMBS} \
  --contrastive-temp 0.07 \
  --contrastive-coef 1.0 \
  --caption-sample-k 5 \
  --contrastive-hard-mining \
  --koleo-coef 0.1 \
  --save-every 5 \
  --wandb-entity ${WANDB_ENTITY} \
  --wandb-log-images 16 \
  --log-dir ${LOG_DIR}/run_K3_vitl14_contrastive10_k5_hardmine_koleo01_30ep
")
echo "Submitted K3 (ViT-L, k=5, hard mining + KoLeo λ=0.1, 30ep): ${JOB}"
echo "Monitor with: squeue -u \$USER"
