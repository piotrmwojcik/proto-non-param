#!/bin/bash
# Evaluate PNP checkpoints with word-only and caption-augmented prototype modes.
#
# Submits up to 4 SLURM jobs:
#   0  build-caption-proto   Build vg_test_caption_prototypes.pt (GPU, ~10 min)
#                            Skipped if the file already exists.
#   A  eval-aug-A            Word top-5 only checkpoint  (depends on job 0)
#   B  eval-aug-B            Caption-only checkpoint     (depends on job 0)
#   C  eval-aug-C            Combined checkpoint         (depends on job 0)
#
# Each eval job runs eval_augmented_prototypes.py --mode both, logging
# both word-level and caption-level prototype statistics to W&B.
#
# Configuration — override via environment variables:
#   CKPT_A       Checkpoint A path  (default: vg_caption_signal_ablation/run_A_word_only)
#   CKPT_B       Checkpoint B path  (default: vg_caption_signal_ablation/run_B_caption_only)
#   CKPT_C       Checkpoint C path  (default: vg_caption_signal_ablation/run_C_combined)
#   VOCAB_DIR    Directory for vocab caches
#   VG_ROOT      Path to VG image root
#   VG_DESC      Path to region_descriptions.json
#   WANDB_ENTITY W&B entity (team or username)
#
# Usage:
#   bash scripts/slurm_eval_augmented.sh
#
#   # Custom checkpoints:
#   CKPT_A=/path/to/ckpt.pth CKPT_B=/path/to/ckpt.pth CKPT_C=/path/to/ckpt.pth \
#       bash scripts/slurm_eval_augmented.sh

set -e

SCRATCH="/net/tscrax  steastch/people/plgabedychaj"
ABLATION_BASE="${SCRATCH}/train_logs/vg_caption_signal_ablation"
VOCAB_DIR="${VOCAB_DIR:-${SCRATCH}/vocab}"
VG_ROOT="${VG_ROOT:-${SCRATCH}/vg}"
VG_DESC="${VG_DESC:-${VG_ROOT}/region_descriptions.json}"
WANDB_ENTITY="${WANDB_ENTITY:-gmum}"

CKPT_A="${CKPT_A:-${ABLATION_BASE}/run_A_word_only/ckpt.pth}"
CKPT_B="${CKPT_B:-${ABLATION_BASE}/run_B_caption_only/ckpt.pth}"
CKPT_C="${CKPT_C:-${ABLATION_BASE}/run_C_combined/ckpt.pth}"

PARTITION="plgrid-gpu-a100"
ACCOUNT="plgunhype-gpu-a100"
LOG_SLURM="${SCRATCH}/logs"

VG_CACHE="${VOCAB_DIR}/vg_cache.pt"
CAPTION_PROTO="${VOCAB_DIR}/vg_test_caption_prototypes.pt"

mkdir -p "${LOG_SLURM}"

echo "=== Augmented Prototype Evaluation ==="
echo "  Ckpt A : ${CKPT_A}"
echo "  Ckpt B : ${CKPT_B}"
echo "  Ckpt C : ${CKPT_C}"
echo "  Caption proto: ${CAPTION_PROTO}"
echo ""

# -----------------------------------------------------------------------
# Job 0 — Build vg_test_caption_prototypes.pt (GPU, ~10 min)
# Encodes unique VG val-split region phrases as CLIP embeddings.
# Skipped if the file already exists.
# -----------------------------------------------------------------------
if [ -f "${CAPTION_PROTO}" ]; then
  echo "Caption prototype cache already exists — skipping build job."
  PROTO_DEP=""
else
  JOB0=$(sbatch --parsable \
    --job-name=build-caption-proto \
    --partition="${PARTITION}" \
    --account="${ACCOUNT}" \
    --gres=gpu:1 \
    --cpus-per-task=4 \
    --mem=16G \
    --time=00:30:00 \
    --output="${LOG_SLURM}/build_caption_proto_%j.out" \
    --error="${LOG_SLURM}/build_caption_proto_%j.err" \
    --wrap="
set -e
export HF_HOME=${SCRATCH}/hf_cache
export PYTHONPATH=${SCRATCH}/dinov2:\$PYTHONPATH
source ${SCRATCH}/venv/bin/activate
cd ~/proto-non-param

python vocab/build_caption_prototypes.py \\
  --source vg_test \\
  --vg-root ${VG_ROOT} \\
  --region-descriptions ${VG_DESC} \\
  --vocab-cache-path ${VG_CACHE} \\
  --cache-out ${CAPTION_PROTO} \\
  --clip-model-name ViT-B-32 \\
  --clip-pretrained openai \\
  --min-words 5 \\
  --batch-size 512 \\
  --device cuda
")
  echo "Submitted job 0 (build-caption-proto): ${JOB0}"
  PROTO_DEP="--dependency=afterok:${JOB0}"
fi

# -----------------------------------------------------------------------
# Helper: submit one eval job
#   $1  label (A/B/C)
#   $2  checkpoint path
#   $3  W&B run name
# -----------------------------------------------------------------------
submit_eval() {
  local LABEL="$1"
  local CKPT="$2"
  local RUN_NAME="$3"

  local JOB
  JOB=$(sbatch --parsable \
    ${PROTO_DEP} \
    --job-name="eval-aug-${LABEL}" \
    --partition="${PARTITION}" \
    --account="${ACCOUNT}" \
    --gres=gpu:1 \
    --cpus-per-task=8 \
    --mem=32G \
    --time=02:00:00 \
    --output="${LOG_SLURM}/eval_aug_${LABEL}_%j.out" \
    --error="${LOG_SLURM}/eval_aug_${LABEL}_%j.err" \
    --wrap="
set -e
export HF_HOME=${SCRATCH}/hf_cache
export PYTHONPATH=${SCRATCH}/dinov2:\$PYTHONPATH
source ${SCRATCH}/venv/bin/activate
cd ~/proto-non-param

python eval_augmented_prototypes.py \\
  --ckpt ${CKPT} \\
  --vocab-cache-path ${VG_CACHE} \\
  --caption-prototypes-path ${CAPTION_PROTO} \\
  --source-dataset vg_test \\
  --vg-root ${VG_ROOT} \\
  --vg-region-descriptions ${VG_DESC} \\
  --mode both \\
  --topk 5 \\
  --batch-size 64 \\
  --num-workers 8 \\
  --wandb-entity ${WANDB_ENTITY} \\
  --wandb-run-name ${RUN_NAME}
")
  echo "Submitted eval-aug-${LABEL}: ${JOB}"
}

submit_eval "A" "${CKPT_A}" "eval-aug-A-word-only"
submit_eval "B" "${CKPT_B}" "eval-aug-B-caption-only"
submit_eval "C" "${CKPT_C}" "eval-aug-C-combined"

echo ""
echo "All jobs submitted. Monitor with: squeue -u \$USER"
echo "Logs: ${LOG_SLURM}/eval_aug_{A,B,C}_*.{out,err}"
