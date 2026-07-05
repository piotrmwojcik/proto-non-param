#!/bin/bash
# Runs K1 and K2 — KoLeo nearest-neighbour repulsion on pred_text_embedding.
#
# K1 = H (ViT-L, k=1, contrastive) + KoLeo λ=0.1   → isolates KoLeo contribution
# K2 = J (K1 + SK)                 + KoLeo λ=0.1   → tests SK+KoLeo stacking
#
# Configuration — override via environment variables:
#   SCRATCH, VG_ROOT, VG_DESC, LOG_DIR, VOCAB_DIR, WANDB_ENTITY
#
# Usage:
#   bash scripts/slurm_train_vg_contrastive_koleo.sh

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
  --caption-embeds-path ${CAPTION_EMBS} \
  --contrastive-temp 0.07 \
  --contrastive-coef 1.0 \
  --caption-sample-k 1 \
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
# K1 — ViT-L, k=1, contrastive + KoLeo (no SK)
# ---------------------------------------------------------------------------
JOB_K1=$(sbatch --parsable \
  ${SLURM_COMMON} \
  --job-name=vg-contr-K1-koleo \
  --output="${LOG_SLURM}/vg_contrastive_K1_koleo_%j.out" \
  --error="${LOG_SLURM}/vg_contrastive_K1_koleo_%j.err" \
  --wrap="${WRAP_HEADER}
python train.py \\
  ${TRAIN_COMMON} \\
  --log-dir ${LOG_DIR}/run_K1_vitl14_contrastive10_k1_koleo01_30ep
")
echo "Submitted K1 (ViT-L, k=1, KoLeo λ=0.1, no SK, 30ep): ${JOB_K1}"

# ---------------------------------------------------------------------------
# K2 — ViT-L, k=1, contrastive + SK + KoLeo
# ---------------------------------------------------------------------------
JOB_K2=$(sbatch --parsable \
  ${SLURM_COMMON} \
  --job-name=vg-contr-K2-sk-koleo \
  --output="${LOG_SLURM}/vg_contrastive_K2_sk_koleo_%j.out" \
  --error="${LOG_SLURM}/vg_contrastive_K2_sk_koleo_%j.err" \
  --wrap="${WRAP_HEADER}
python train.py \\
  ${TRAIN_COMMON} \\
  --sk-coef 0.1 \\
  --sk-eps 0.10 \\
  --log-dir ${LOG_DIR}/run_K2_vitl14_contrastive10_k1_sk10_koleo01_30ep
")
echo "Submitted K2 (ViT-L, k=1, SK λ=0.1 + KoLeo λ=0.1, 30ep): ${JOB_K2}"

echo ""
echo "Monitor with : squeue -u \$USER"
echo "Logs under   : ${LOG_SLURM}/"
echo "Checkpoints  : ${LOG_DIR}/run_K{1,2}_*/"
