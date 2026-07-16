#!/bin/bash
# Runs N and O — diversity losses only, no contrastive (InfoNCE off).
#
# N = ViT-L + SK + KoLeo + iBOT        (no contrastive, replaces MSN with iBOT)
# O = ViT-L + SK + SigReg              (no contrastive, SigReg as KoLeo alternative)
#
# Ablation purpose:
#   N vs M2: iBOT (per-patch CE) vs MSN (global CE) — which masking signal is better?
#   O vs M1: SigReg (global ECF) vs KoLeo (local NN repulsion) — which diversity loss?
#
# Usage:
#   bash scripts/slurm_train_vg_nocontrastive_N_O.sh

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
# N — SK + KoLeo + iBOT, no contrastive
# ---------------------------------------------------------------------------
JOB_N=$(sbatch --parsable \
  ${SLURM_COMMON} \
  --job-name=vg-N-sk-koleo-ibot \
  --output="${LOG_SLURM}/vg_N_sk_koleo_ibot_%j.out" \
  --error="${LOG_SLURM}/vg_N_sk_koleo_ibot_%j.err" \
  --wrap="${WRAP_HEADER}
python train.py \\
  ${TRAIN_COMMON} \\
  --koleo-coef 0.1 \\
  --msn-mask-ratio 0.25 \\
  --ibot-coef 0.1 \\
  --log-dir ${LOG_DIR}/run_N_vitl14_sk10_koleo01_ibot01_mask25_30ep
")
echo "Submitted N (ViT-L, SK+KoLeo+iBOT, no contrastive, 30ep): ${JOB_N}"

# ---------------------------------------------------------------------------
# O — SK + SigReg, no contrastive (no KoLeo — tests SigReg as alternative)
# ---------------------------------------------------------------------------
JOB_O=$(sbatch --parsable \
  ${SLURM_COMMON} \
  --job-name=vg-O-sk-sigreg \
  --output="${LOG_SLURM}/vg_O_sk_sigreg_%j.out" \
  --error="${LOG_SLURM}/vg_O_sk_sigreg_%j.err" \
  --wrap="${WRAP_HEADER}
python train.py \\
  ${TRAIN_COMMON} \\
  --sigreg-coef 0.02 \\
  --sigreg-sketch-dim 64 \\
  --log-dir ${LOG_DIR}/run_O_vitl14_sk10_sigreg002_30ep
")
echo "Submitted O (ViT-L, SK+SigReg, no contrastive, 30ep): ${JOB_O}"

echo ""
echo "Monitor with : squeue -u \$USER"
echo "Logs under   : ${LOG_SLURM}/"
echo "Checkpoints  : ${LOG_DIR}/run_N_* ${LOG_DIR}/run_O_*/"
