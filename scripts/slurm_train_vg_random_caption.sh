#!/bin/bash
# M1-RC — M1's exact loss config (SK + KoLeo, no contrastive), but with the KL
# target built per-step from a random draw of --random-caption-target-k region
# phrases instead of M1's fixed, precomputed union-of-all-phrases distribution.
#
# Ablation purpose: does resampling the target each epoch (vs. one fixed target
# per image for the whole run) change what the model learns? Compare against M1.
#
# Usage:
#   bash scripts/slurm_train_vg_random_caption.sh
#   RANDOM_CAPTION_K=5 bash scripts/slurm_train_vg_random_caption.sh   # override k

set -e

SCRATCH="${SCRATCH:-/net/tscratch/people/plgabedychaj}"
VG_ROOT="${VG_ROOT:-${SCRATCH}/vg}"
VG_DESC="${VG_DESC:-${VG_ROOT}/region_descriptions.json}"
LOG_DIR="${LOG_DIR:-${SCRATCH}/train_logs/vg_contrastive}"
VOCAB_DIR="${VOCAB_DIR:-${SCRATCH}/vocab}"
WANDB_ENTITY="${WANDB_ENTITY:-gmum}"
RANDOM_CAPTION_K="${RANDOM_CAPTION_K:-3}"

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

JOB=$(sbatch --parsable \
  --partition="${PARTITION}" \
  --account="${ACCOUNT}" \
  --gres=gpu:1 \
  --cpus-per-task=8 \
  --mem=64G \
  --time=2-00:00:00 \
  --job-name=vg-M1-RC-random-caption \
  --output="${LOG_SLURM}/vg_M1_RC_random_caption_%j.out" \
  --error="${LOG_SLURM}/vg_M1_RC_random_caption_%j.err" \
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
  --contrastive-coef 0.0 \\
  --sk-coef 0.1 \\
  --sk-eps 0.10 \\
  --koleo-coef 0.1 \\
  --random-caption-target \\
  --random-caption-target-k ${RANDOM_CAPTION_K} \\
  --save-every 5 \\
  --wandb-entity ${WANDB_ENTITY} \\
  --wandb-log-images 16 \\
  --log-dir ${LOG_DIR}/run_M1-RC_vitl14_sk10_koleo01_randk${RANDOM_CAPTION_K}_30ep
")
echo "Submitted M1-RC (ViT-L, SK+KoLeo, no contrastive, random-caption-target k=${RANDOM_CAPTION_K}, 30ep): ${JOB}"

echo ""
echo "Monitor with : squeue -u \$USER"
echo "Logs under   : ${LOG_SLURM}/"
echo "Checkpoint   : ${LOG_DIR}/run_M1-RC_vitl14_sk10_koleo01_randk${RANDOM_CAPTION_K}_30ep/"
