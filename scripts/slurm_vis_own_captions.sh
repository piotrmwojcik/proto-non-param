#!/bin/bash
# Per-image own-caption visualisation for the 2×2 VG ablation checkpoints.
#
# Submits 4 SLURM jobs (one per variant A/B/C/D).
# Each job runs visualize_caption_activation.py, sampling --n-random images
# from the VG test set and visualising --n-own-captions region descriptions
# per image as augmented prototypes, logged to W&B under own_captions/.
#
# Results: W&B project proto-non-param, runs vis-own-cap-{A,B,C,D}-*
#
# Configuration — override via environment variables:
#   CKPT_A/B/C/D   Checkpoint paths
#   VOCAB_DIR      Directory containing vg_cache.pt
#   VG_ROOT        Path to VG image root
#   VG_DESC        Path to region_descriptions.json
#   WANDB_ENTITY   W&B entity (team or username)
#   N_IMAGES       Number of random test images per variant (default 30)
#   N_CAPTIONS     Own captions per image (default 5)
#
# Usage:
#   bash scripts/slurm_vis_own_captions.sh

set -e

SCRATCH="${SCRATCH:-/net/tscratch/people/plgabedychaj}"
ABLATION_BASE="${ABLATION_BASE:-${SCRATCH}/train_logs/vg_ablation}"
VOCAB_DIR="${VOCAB_DIR:-${SCRATCH}/vocab}"
VG_ROOT="${VG_ROOT:-${SCRATCH}/vg}"
VG_DESC="${VG_DESC:-${VG_ROOT}/region_descriptions.json}"
WANDB_ENTITY="${WANDB_ENTITY:-gmum}"

CKPT_A="${CKPT_A:-${ABLATION_BASE}/run_A_kl_frozen/ckpt.pth}"
CKPT_B="${CKPT_B:-${ABLATION_BASE}/run_B_kl_residual/ckpt.pth}"
CKPT_C="${CKPT_C:-${ABLATION_BASE}/run_C_jsd_frozen/ckpt.pth}"
CKPT_D="${CKPT_D:-${ABLATION_BASE}/run_D_jsd_residual/ckpt.pth}"

N_IMAGES="${N_IMAGES:-30}"
N_CAPTIONS="${N_CAPTIONS:-5}"

PARTITION="plgrid-gpu-a100"
ACCOUNT="plgunhype-gpu-a100"
LOG_SLURM="${SCRATCH}/logs"
VG_CACHE="${VOCAB_DIR}/vg_cache.pt"

mkdir -p "${LOG_SLURM}"

echo "=== VG Ablation — Per-Image Own-Caption Visualisation ==="
echo "  Images per variant : ${N_IMAGES}"
echo "  Captions per image : ${N_CAPTIONS}"
echo ""

declare -A CKPTS=([A]="${CKPT_A}" [B]="${CKPT_B}" [C]="${CKPT_C}" [D]="${CKPT_D}")
declare -A LABELS=(
  [A]="kl-frozen"
  [B]="kl-residual"
  [C]="jsd-frozen"
  [D]="jsd-residual"
)

for VARIANT in A B C D; do
  CKPT="${CKPTS[$VARIANT]}"
  LABEL="${LABELS[$VARIANT]}"
  RUN_NAME="vis-own-cap-${VARIANT}-${LABEL}"

  JOB=$(sbatch --parsable \
    --job-name="vis-own-${VARIANT}" \
    --partition="${PARTITION}" \
    --account="${ACCOUNT}" \
    --gres=gpu:1 \
    --cpus-per-task=4 \
    --mem=16G \
    --time=01:00:00 \
    --output="${LOG_SLURM}/vis_own_${VARIANT}_%j.out" \
    --error="${LOG_SLURM}/vis_own_${VARIANT}_%j.err" \
    --wrap="
set -e
export HF_HOME=${SCRATCH}/hf_cache
export PYTHONPATH=${SCRATCH}/dinov2:\$PYTHONPATH
source ${SCRATCH}/venv/bin/activate
cd ~/proto-VLM/proto-non-param

python visualize_caption_activation.py \\
  --ckpt ${CKPT} \\
  --vocab-cache-path ${VG_CACHE} \\
  --source-dataset vg_test \\
  --vg-root ${VG_ROOT} \\
  --vg-region-descriptions ${VG_DESC} \\
  --n-random ${N_IMAGES} \\
  --n-own-captions ${N_CAPTIONS} \\
  --top-patches 5 \\
  --wandb-entity ${WANDB_ENTITY} \\
  --wandb-run-name ${RUN_NAME}
")
  echo "Submitted variant ${VARIANT} (${LABEL}): job ${JOB}"
done

echo ""
echo "All 4 jobs submitted. Monitor with: squeue -u \$USER"
echo "Logs: ${LOG_SLURM}/vis_own_{A,B,C,D}_*.{out,err}"
echo "W&B:  https://wandb.ai/${WANDB_ENTITY}/proto-non-param"
