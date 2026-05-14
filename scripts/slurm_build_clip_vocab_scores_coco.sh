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

# Uses the clean COCO vocab (~4K words, no misspellings) instead of vg_cache.pt (15K noisy).
# Temperature is NOT set here — softmax is applied at load time via --clip-scores-temperature
# in train.py (default 0.07), so raw clip_scores are what matter.
# Prerequisite: vocab/mscoco_new_cache.pt must exist (run `python build_cache.py` if not).

python build_clip_vocab_scores.py \
  --dataset coco \
  --data-root /net/tscratch/people/plgabedychaj/coco_dataset/raw \
  --annotations /net/tscratch/people/plgabedychaj/coco_dataset/raw/annotations/captions_train2017.json \
  --vocab-cache vocab/mscoco_new_cache.pt \
  --clip-model ViT-B-32 \
  --clip-pretrained openai \
  --output /net/tscratch/people/plgabedychaj/vocab/coco_train_clip_scores.pt \
  --batch-size 512 \
  --num-workers 8

python build_clip_vocab_scores.py \
  --dataset coco \
  --data-root /net/tscratch/people/plgabedychaj/coco_dataset/raw \
  --annotations /net/tscratch/people/plgabedychaj/coco_dataset/raw/annotations/captions_val2017.json \
  --vocab-cache vocab/mscoco_new_cache.pt \
  --clip-model ViT-B-32 \
  --clip-pretrained openai \
  --output /net/tscratch/people/plgabedychaj/vocab/coco_val_clip_scores.pt \
  --batch-size 512 \
  --num-workers 8
