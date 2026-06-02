#!/bin/bash
# VG-only training + caption-augmented inference ablation — full pipeline.
#
# Submits four SLURM jobs chained with afterok dependencies:
#   1. build-vg-vocab        Build word-level VG vocabulary cache (CPU)
#   2. build-caption-proto   Build phrase-level caption prototype caches (CPU/GPU)
#   3. train-vg-only         Train PNP on VG descriptions only (GPU)
#   4. eval-augmented        Evaluate word-only vs augmented prototypes (GPU)
#
# Configuration — override via environment variables before running:
#   VG_ROOT            Path to VG image root (containing VG_100K/ and VG_100K_2/)
#   VG_DESC            Path to region_descriptions.json
#   COCO_ROOT          Path to COCO image root
#   COCO_ANN_VAL       Path to captions_val2014.json
#   LOG_DIR            Base directory for training logs and checkpoints
#   VOCAB_DIR          Directory where vocabulary caches are saved
#   WANDB_ENTITY       W&B entity (team or username)
#
# Usage:
#   bash scripts/run_vg_caption_ablation.sh
#   # or with overrides:
#   VG_ROOT=/my/vg LOG_DIR=/my/logs bash scripts/run_vg_caption_ablation.sh

set -e

# ---- Cluster paths (defaults matching the Athena/PLGrid setup) ----
SCRATCH="/net/tscratch/people/plgabedychaj"
VG_ROOT="${VG_ROOT:-${SCRATCH}/vg}"
VG_DESC="${VG_DESC:-${VG_ROOT}/region_descriptions.json}"
COCO_ROOT="${COCO_ROOT:-${SCRATCH}/coco_2014}"
COCO_ANN_VAL="${COCO_ANN_VAL:-${COCO_ROOT}/annotations/captions_val2014.json}"
LOG_DIR="${LOG_DIR:-${SCRATCH}/train_logs/vg_caption_ablation}"
VOCAB_DIR="${VOCAB_DIR:-${SCRATCH}/vocab}"
WANDB_ENTITY="${WANDB_ENTITY:-gmum}"

PARTITION="plgrid-gpu-a100"
ACCOUNT="plgunhype-gpu-a100"
LOG_SLURM="${SCRATCH}/logs"

mkdir -p "${LOG_SLURM}" "${VOCAB_DIR}" "${LOG_DIR}"

echo "=== VG Caption Ablation Pipeline ==="
echo "  VG_ROOT     : ${VG_ROOT}"
echo "  VG_DESC     : ${VG_DESC}"
echo "  COCO_ROOT   : ${COCO_ROOT}"
echo "  LOG_DIR     : ${LOG_DIR}"
echo "  VOCAB_DIR   : ${VOCAB_DIR}"
echo ""

# -----------------------------------------------------------------------
# Job 1 — Build VG word vocabulary (CPU-only, short)
# -----------------------------------------------------------------------
JOB1=$(sbatch --parsable \
  --job-name=vg-vocab \
  --partition="${PARTITION}" \
  --account="${ACCOUNT}" \
  --cpus-per-task=4 \
  --mem=32G \
  --time=01:00:00 \
  --output="${LOG_SLURM}/vg_vocab_%j.out" \
  --error="${LOG_SLURM}/vg_vocab_%j.err" \
  --wrap="
set -e
export HF_HOME=${SCRATCH}/hf_cache
export PYTHONPATH=${SCRATCH}/dinov2:\$PYTHONPATH
source ${SCRATCH}/venv/bin/activate
cd ~/proto-VLM/proto-non-param

python vocab/build_vg_vocab.py \\
  --region-descriptions ${VG_DESC} \\
  --vocab-out  ${VOCAB_DIR}/vg.txt \\
  --cache-out  ${VOCAB_DIR}/vg_cache.pt \\
  --clip-model-name ViT-B-32 \\
  --clip-pretrained openai \\
  --min-count 5 \\
  --max-doc-freq 0.5
")
echo "Submitted job 1 (build-vg-vocab): ${JOB1}"

# -----------------------------------------------------------------------
# Job 2 — Build caption prototype caches (GPU for faster CLIP encoding)
# -----------------------------------------------------------------------
JOB2=$(sbatch --parsable \
  --job-name=build-caption-proto \
  --partition="${PARTITION}" \
  --account="${ACCOUNT}" \
  --gres=gpu:1 \
  --cpus-per-task=4 \
  --mem=32G \
  --time=02:00:00 \
  --dependency=afterok:"${JOB1}" \
  --output="${LOG_SLURM}/caption_proto_%j.out" \
  --error="${LOG_SLURM}/caption_proto_%j.err" \
  --wrap="
set -e
export HF_HOME=${SCRATCH}/hf_cache
export PYTHONPATH=${SCRATCH}/dinov2:\$PYTHONPATH
source ${SCRATCH}/venv/bin/activate
cd ~/proto-VLM/proto-non-param

echo '--- Building VG test caption prototypes ---'
python vocab/build_caption_prototypes.py \\
  --source vg_test \\
  --vg-root ${VG_ROOT} \\
  --region-descriptions ${VG_DESC} \\
  --vocab-cache-path ${VOCAB_DIR}/vg_cache.pt \\
  --cache-out ${VOCAB_DIR}/vg_test_caption_prototypes.pt \\
  --clip-model-name ViT-B-32 \\
  --clip-pretrained openai \\
  --seed 42 \\
  --val-ratio 0.1

echo '--- Building COCO val caption prototypes ---'
python vocab/build_caption_prototypes.py \\
  --source coco \\
  --coco-annotations ${COCO_ANN_VAL} \\
  --cache-out ${VOCAB_DIR}/coco_caption_prototypes.pt \\
  --clip-model-name ViT-B-32 \\
  --clip-pretrained openai \\
  --max-captions 50000
")
echo "Submitted job 2 (build-caption-proto): ${JOB2}"

# -----------------------------------------------------------------------
# Job 3 — Train VG-only (GPU, long)
# -----------------------------------------------------------------------
JOB3=$(sbatch --parsable \
  --job-name=train-vg-only \
  --partition="${PARTITION}" \
  --account="${ACCOUNT}" \
  --gres=gpu:1 \
  --cpus-per-task=8 \
  --mem=64G \
  --time=2-00:00:00 \
  --dependency=afterok:"${JOB1}" \
  --output="${LOG_SLURM}/train_vg_only_%j.out" \
  --error="${LOG_SLURM}/train_vg_only_%j.err" \
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
  --vocab-cache-path ${VOCAB_DIR}/vg_cache.pt \\
  --log-dir ${LOG_DIR} \\
  --backbone dinov2_vitb14 \\
  --num-splits 1 \\
  --batch-size 64 \\
  --epochs 20 \\
  --num-workers 8 \\
  --backbone-lr 1e-5 \\
  --text-proj-lr 1e-4 \\
  --kl-coef 1.0 \\
  --wandb-entity ${WANDB_ENTITY} \\
  --wandb-log-images 16
")
echo "Submitted job 3 (train-vg-only): ${JOB3}"

# -----------------------------------------------------------------------
# Job 4 — Evaluate augmented prototypes (GPU)
# Depends on both jobs 2 (caption caches) and 3 (checkpoint)
# -----------------------------------------------------------------------
JOB4=$(sbatch --parsable \
  --job-name=eval-augmented \
  --partition="${PARTITION}" \
  --account="${ACCOUNT}" \
  --gres=gpu:1 \
  --cpus-per-task=8 \
  --mem=64G \
  --time=04:00:00 \
  --dependency=afterok:"${JOB2}:${JOB3}" \
  --output="${LOG_SLURM}/eval_augmented_%j.out" \
  --error="${LOG_SLURM}/eval_augmented_%j.err" \
  --wrap="
set -e
export HF_HOME=${SCRATCH}/hf_cache
export PYTHONPATH=${SCRATCH}/dinov2:\$PYTHONPATH
source ${SCRATCH}/venv/bin/activate
cd ~/proto-VLM/proto-non-param

CKPT=\"${LOG_DIR}/ckpt.pth\"
echo \"Using checkpoint: \$CKPT\"

echo '--- Evaluating with VG test caption prototypes ---'
python eval_augmented_prototypes.py \\
  --ckpt \"\$CKPT\" \\
  --vocab-cache-path ${VOCAB_DIR}/vg_cache.pt \\
  --caption-prototypes-path ${VOCAB_DIR}/vg_test_caption_prototypes.pt \\
  --source-dataset vg_test \\
  --vg-root ${VG_ROOT} \\
  --vg-region-descriptions ${VG_DESC} \\
  --mode both \\
  --topk 5 \\
  --batch-size 64 \\
  --num-workers 8 \\
  --wandb-entity ${WANDB_ENTITY} \\
  --wandb-run-name vg-caption-ablation-vgtest

echo '--- Evaluating with COCO caption prototypes ---'
python eval_augmented_prototypes.py \\
  --ckpt \"\$CKPT\" \\
  --vocab-cache-path ${VOCAB_DIR}/vg_cache.pt \\
  --caption-prototypes-path ${VOCAB_DIR}/coco_caption_prototypes.pt \\
  --source-dataset coco_val \\
  --coco-root ${COCO_ROOT} \\
  --coco-annotations ${COCO_ANN_VAL} \\
  --mode both \\
  --topk 5 \\
  --batch-size 64 \\
  --num-workers 8 \\
  --wandb-entity ${WANDB_ENTITY} \\
  --wandb-run-name vg-caption-ablation-coco
")
echo "Submitted job 4 (eval-augmented): ${JOB4}"

echo ""
echo "Pipeline submitted. Job chain:"
echo "  ${JOB1} (vocab) → ${JOB2} (captions) ┐"
echo "  ${JOB1} (vocab) → ${JOB3} (train)    ┘→ ${JOB4} (eval)"
echo ""
echo "Monitor with: squeue -u \$USER"
