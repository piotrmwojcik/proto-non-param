#!/bin/bash
# Run Q — SK (balanced) + KoLeo, no contrastive, no SigReg, no cross-attn.
#
# Rationale (from W&B review of O2 and P):
#   - O2 (SK λ=0.3 + SigReg λ=0.5): l_sk converges nicely at the higher λ=0.3,
#     but l_sigreg stays flat even when given real gradient weight — SigReg
#     doesn't train regardless of coefficient. Drop it.
#   - P (SK+KoLeo+cross-attn): l_koleo behaves correctly, but cross-attention
#     aggregation underperformed the fixed top-k=5 pooling on eval. Revert to
#     top-k (now the default again in modeling/pnp.py — --agg-mode topk).
#   - Q keeps what worked from each: SK at the higher λ=0.3 (from O2) + KoLeo
#     λ=0.1 (from M1/K2/N) + top-k=5 pooling (proven), SigReg and cross-attn
#     both dropped.
#
# Ablation purpose:
#   Q vs M1/K2: does the higher SK weight (0.3 vs 0.1) improve over the
#   original SK+KoLeo config now that top-k pooling is confirmed the better
#   aggregation?
#
# Usage:
#   bash scripts/slurm_train_vg_nocontrastive_Q.sh

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
  --agg-mode topk \
  --save-every 5 \
  --wandb-entity ${WANDB_ENTITY} \
  --wandb-log-images 16"

SLURM_COMMON="--partition=${PARTITION} \
  --account=${ACCOUNT} \
  --gres=gpu:1 \
  --cpus-per-task=8 \
  --mem=64G \
  --time=2-00:00:00"

JOB_Q=$(sbatch --parsable \
  ${SLURM_COMMON} \
  --job-name=vg-Q-sk03-koleo01-topk \
  --output="${LOG_SLURM}/vg_Q_sk03_koleo01_topk_%j.out" \
  --error="${LOG_SLURM}/vg_Q_sk03_koleo01_topk_%j.err" \
  --wrap="${WRAP_HEADER}
python train.py \\
  ${TRAIN_COMMON} \\
  --sk-coef 0.3 \\
  --sk-eps 0.10 \\
  --koleo-coef 0.1 \\
  --log-dir ${LOG_DIR}/run_Q_vitl14_sk03_koleo01_30ep
")
echo "Submitted Q (ViT-L, SK λ=0.3 + KoLeo λ=0.1, top-k=5 pooling, no contrastive, 30ep): ${JOB_Q}"

echo ""
echo "Monitor with : squeue -u \$USER"
echo "Logs under   : ${LOG_SLURM}/"
echo "Checkpoint   : ${LOG_DIR}/run_Q_vitl14_sk03_koleo01_30ep/"
