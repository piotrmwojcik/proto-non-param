#!/bin/bash
# Run R — SK and KoLeo reverse-annealed (high early, low late), no contrastive,
# no SigReg, no cross-attn.
#
# Rationale:
#   O2/P's W&B curves showed SK converges better at a higher lambda (0.3) and
#   KoLeo trains correctly at its existing lambda (0.1) — both under a constant
#   coefficient for all 30 epochs. R tests whether starting each coefficient
#   even higher and cosine-decaying it down to the proven steady-state value
#   gives the model strong early structuring pressure on the prototype pool /
#   embedding spread, then lets the main KL objective dominate for fine-tuning.
#   This "structure early, relax later" curriculum has precedent in annealed
#   Sinkhorn (optimal transport) and cosine-annealed structural loss weights
#   in local descriptor learning (arXiv:2303.06124).
#
# Ablation purpose:
#   R vs Q (constant SK=0.3+KoLeo=0.1): does annealing from a higher initial
#   value (SK 0.5, KoLeo 0.3) down to the same steady-state (0.1 each) beat
#   holding the coefficient constant throughout training?
#
# Usage:
#   bash scripts/slurm_train_vg_nocontrastive_R.sh

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

JOB_R=$(sbatch --parsable \
  ${SLURM_COMMON} \
  --job-name=vg-R-sk-koleo-anneal \
  --output="${LOG_SLURM}/vg_R_sk_koleo_anneal_%j.out" \
  --error="${LOG_SLURM}/vg_R_sk_koleo_anneal_%j.err" \
  --wrap="${WRAP_HEADER}
python train.py \\
  ${TRAIN_COMMON} \\
  --sk-coef 0.1 \\
  --sk-coef-init 0.5 \\
  --sk-eps 0.10 \\
  --koleo-coef 0.1 \\
  --koleo-coef-init 0.3 \\
  --log-dir ${LOG_DIR}/run_R_vitl14_sk_anneal05to01_koleo_anneal03to01_30ep
")
echo "Submitted R (ViT-L, SK 0.5->0.1 + KoLeo 0.3->0.1 annealed, top-k=5 pooling, no contrastive, 30ep): ${JOB_R}"

echo ""
echo "Monitor with : squeue -u \$USER"
echo "Logs under   : ${LOG_SLURM}/"
echo "Checkpoint   : ${LOG_DIR}/run_R_vitl14_sk_anneal05to01_koleo_anneal03to01_30ep/"
