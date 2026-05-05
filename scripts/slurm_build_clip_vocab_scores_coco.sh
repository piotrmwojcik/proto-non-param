#!/bin/bash
#SBATCH --job-name=clip_vocab_coco
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --account=plgunhype-gpu-a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/net/tscratch/people/plgabedychaj/logs/clip_vocab_coco_%j.out
#SBATCH --error=/net/tscratch/people/plgabedychaj/logs/clip_vocab_coco_%j.err

set -e
mkdir -p /net/tscratch/people/plgabedychaj/logs
mkdir -p /net/tscratch/people/plgabedychaj/vocab

export HF_HOME=/net/tscratch/people/plgabedychaj/hf_cache
export PYTHONPATH="/net/tscratch/people/plgabedychaj/dinov2:$PYTHONPATH"
source /net/tscratch/people/plgabedychaj/venv/bin/activate
cd ~/proto-VLM/proto-non-param

# Build CLIP-based vocab scores for COCO train split.
# --caption-stats is optional: remove it to get pure CLIP targets (alpha irrelevant),
# or point it at a cached CocoCLIPDataset .pt to produce mixed_labels.

python build_clip_vocab_scores.py \
  --dataset coco \
  --data-root /net/tscratch/people/plgabedychaj/coco_dataset/raw \
  --annotations /net/tscratch/people/plgabedychaj/coco_dataset/raw/annotations/captions_train2017.json \
  --vocab-cache /net/tscratch/people/plgabedychaj/vocab/vg_cache.pt \
  --clip-model ViT-B-32 \
  --clip-pretrained openai \
  --temperature 1.0 \
  --alpha 1.0 \
  --output /net/tscratch/people/plgabedychaj/vocab/coco_train_clip_scores.pt \
  --batch-size 512 \
  --num-workers 8

# Uncomment below to also build val scores:
# python build_clip_vocab_scores.py \
#   --dataset coco \
#   --data-root /net/tscratch/people/plgabedychaj/coco_dataset/raw \
#   --annotations /net/tscratch/people/plgabedychaj/coco_dataset/raw/annotations/captions_val2017.json \
#   --vocab-cache /net/tscratch/people/plgabedychaj/vocab/vg_cache.pt \
#   --clip-model ViT-B-32 \
#   --clip-pretrained openai \
#   --temperature 0.07 \
#   --alpha 0.5 \
#   --output /net/tscratch/people/plgabedychaj/vocab/coco_val_clip_scores.pt \
#   --batch-size 512 \
#   --num-workers 8
