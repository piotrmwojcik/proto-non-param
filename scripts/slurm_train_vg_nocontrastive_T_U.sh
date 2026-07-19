#!/bin/bash
# Runs T and U — Tier-1 loss ablations on top of the M1 recipe (no contrastive).
#
# T = M1 + PMSN-style Sinkhorn prior (--sk-prior-tau 0.5)
#     SK's uniform target marginal forces equal mass onto 15.8k mostly-rare VG
#     words; VG word frequency is Zipfian. T swaps the uniform marginal for a
#     tempered empirical prior ∝ freq^0.5 (PMSN, Assran et al. 2022).
#
# U = SK + VICReg (var+cov) replacing KoLeo (--vicreg-coef 0.1, koleo 0)
#     VICReg's covariance term fights dimensional collapse, which KoLeo's
#     NN-repulsion doesn't address; unlike SigReg (dead on unit-sphere
#     embeddings — runs O/O2) VICReg has no Gaussianity assumption.
#
# Ablation purpose:
#   T vs M1: does a long-tail-matched prototype-usage prior beat uniform?
#   U vs M1: does VICReg var+cov beat KoLeo as the embedding anti-collapse loss?
#
# Preflight (once, fast, CPU is fine):
#   python scripts/test_sk_prior_vicreg.py
#
# Usage:
#   bash scripts/slurm_train_vg_nocontrastive_T_U.sh

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
# T — M1 + PMSN-style tempered-empirical Sinkhorn prior
# ---------------------------------------------------------------------------
JOB_T=$(sbatch --parsable \
  ${SLURM_COMMON} \
  --job-name=vg-T-sk-priortau05 \
  --output="${LOG_SLURM}/vg_T_sk_priortau05_%j.out" \
  --error="${LOG_SLURM}/vg_T_sk_priortau05_%j.err" \
  --wrap="${WRAP_HEADER}
python train.py \\
  ${TRAIN_COMMON} \\
  --koleo-coef 0.1 \\
  --sk-prior-tau 0.5 \\
  --log-dir ${LOG_DIR}/run_T_vitl14_sk10_koleo01_priortau05_30ep
")
echo "Submitted T (ViT-L, SK+KoLeo + PMSN prior tau=0.5, no contrastive, 30ep): ${JOB_T}"

# ---------------------------------------------------------------------------
# U — SK + VICReg replacing KoLeo
# ---------------------------------------------------------------------------
JOB_U=$(sbatch --parsable \
  ${SLURM_COMMON} \
  --job-name=vg-U-sk-vicreg \
  --output="${LOG_SLURM}/vg_U_sk_vicreg_%j.out" \
  --error="${LOG_SLURM}/vg_U_sk_vicreg_%j.err" \
  --wrap="${WRAP_HEADER}
python train.py \\
  ${TRAIN_COMMON} \\
  --vicreg-coef 0.1 \\
  --log-dir ${LOG_DIR}/run_U_vitl14_sk10_vicreg01_30ep
")
echo "Submitted U (ViT-L, SK+VICReg (no KoLeo), no contrastive, 30ep): ${JOB_U}"

echo ""
echo "Monitor with : squeue -u \$USER"
echo "Logs under   : ${LOG_SLURM}/"
echo "Checkpoints  : ${LOG_DIR}/run_T_* ${LOG_DIR}/run_U_*/"
