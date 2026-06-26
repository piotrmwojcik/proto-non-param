#!/bin/bash
# Submit only run C: contrastive-only (kl_coef=0, contrastive_coef=1.0, 30 epochs).
# Use this when runs A and B are already done.
#
# Usage:
#   bash scripts/slurm_train_vg_contrastive_C.sh

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

JOB_C=$(sbatch --parsable \
  --partition="${PARTITION}" \
  --account="${ACCOUNT}" \
  --gres=gpu:1 \
  --cpus-per-task=8 \
  --mem=64G \
  --time=2-00:00:00 \
  --job-name=vg-contr-C \
  --output="${LOG_SLURM}/vg_contrastive_C_%j.out" \
  --error="${LOG_SLURM}/vg_contrastive_C_%j.err" \
  --wrap="${WRAP_HEADER}
python train.py \\
  --dataset visual_genome \\
  --vg-root ${VG_ROOT} \\
  --vg-region-descriptions ${VG_DESC} \\
  --vocab-cache-path ${VG_CACHE} \\
  --backbone dinov2_vitb14 \\
  --batch-size 128 \\
  --epochs 30 \\
  --lr-schedule cosine \\
  --lr-warmup-epochs 5 \\
  --num-workers 8 \\
  --backbone-lr 1e-5 \\
  --text-proj-lr 1e-4 \\
  --target-mode uniform \\
  --loss-type kl \\
  --kl-coef 0.0 \\
  --caption-embeds-path ${CAPTION_EMBS} \\
  --caption-sample-k 1 \\
  --contrastive-temp 0.07 \\
  --contrastive-coef 1.0 \\
  --save-every 5 \\
  --wandb-entity ${WANDB_ENTITY} \\
  --wandb-log-images 16 \\
  --log-dir ${LOG_DIR}/run_C_uniform_contrastive_only_30ep
")
echo "Submitted C (contrastive-only=1.0, 30ep): ${JOB_C}"
echo "Monitor : squeue -u \$USER"
echo "Log     : ${LOG_SLURM}/vg_contrastive_C_${JOB_C}.out"
echo "Ckpt    : ${LOG_DIR}/run_C_uniform_contrastive_only_30ep/"
