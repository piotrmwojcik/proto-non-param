#!/bin/bash
# Smoke test for contrastive VG training — 1 epoch, small batch, no W&B run saved.
# Verifies: uniform distribution loads, l_contrastive appears, batch[4] shape is correct.
#
# Usage:
#   bash scripts/slurm_test_vg_contrastive.sh

set -e

SCRATCH="${SCRATCH:-/net/tscratch/people/plgabedychaj}"
VG_ROOT="${VG_ROOT:-${SCRATCH}/vg}"
VG_DESC="${VG_DESC:-${VG_ROOT}/region_descriptions.json}"
VOCAB_DIR="${VOCAB_DIR:-${SCRATCH}/vocab}"
WANDB_ENTITY="${WANDB_ENTITY:-gmum}"

PARTITION="plgrid-gpu-a100"
ACCOUNT="plgunhype-gpu-a100"
LOG_SLURM="${SCRATCH}/logs"
LOG_DIR="${SCRATCH}/train_logs/vg_contrastive_test"

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
  --job-name=pnp-contr-test \
  --partition="${PARTITION}" \
  --account="${ACCOUNT}" \
  --gres=gpu:1 \
  --cpus-per-task=4 \
  --mem=32G \
  --time=02:00:00 \
  --output="${LOG_SLURM}/pnp_contrastive_test_%j.out" \
  --error="${LOG_SLURM}/pnp_contrastive_test_%j.err" \
  --wrap="
set -e
export HF_HOME=${SCRATCH}/hf_cache
export TRANSFORMERS_CACHE=${SCRATCH}/hf_cache
export PYTHONPATH=${SCRATCH}/dinov2:\$PYTHONPATH
source ${SCRATCH}/venv/bin/activate
cd ~/proto-VLM/proto-non-param

python train.py \\
  --dataset visual_genome \\
  --vg-root ${VG_ROOT} \\
  --vg-region-descriptions ${VG_DESC} \\
  --vocab-cache-path ${VG_CACHE} \\
  --caption-embeds-path ${CAPTION_EMBS} \\
  --backbone dinov2_vitb14 \\
  --target-mode uniform \\
  --contrastive-coef 0.5 \\
  --contrastive-temp 0.07 \\
  --caption-sample-k 1 \\
  --kl-coef 1.0 \\
  --loss-type kl \\
  --batch-size 32 \\
  --epochs 1 \\
  --num-workers 4 \\
  --wandb-entity ${WANDB_ENTITY} \\
  --log-dir ${LOG_DIR}
")

echo "Submitted smoke test: ${JOB}"
echo "Log : ${LOG_SLURM}/pnp_contrastive_test_${JOB}.out"
echo "Watch: tail -f ${LOG_SLURM}/pnp_contrastive_test_${JOB}.out"
