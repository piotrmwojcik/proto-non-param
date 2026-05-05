#!/bin/bash
#SBATCH --job-name=clip_vocab_vg
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --account=plgunhype-gpu-a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:30:00
#SBATCH --output=/net/tscratch/people/plgabedychaj/logs/clip_vocab_vg_%j.out
#SBATCH --error=/net/tscratch/people/plgabedychaj/logs/clip_vocab_vg_%j.err

set -e
mkdir -p /net/tscratch/people/plgabedychaj/logs
mkdir -p /net/tscratch/people/plgabedychaj/vocab

export HF_HOME=/net/tscratch/people/plgabedychaj/hf_cache
export PYTHONPATH="/net/tscratch/people/plgabedychaj/dinov2:$PYTHONPATH"
source /net/tscratch/people/plgabedychaj/venv/bin/activate
cd ~/proto-VLM/proto-non-param

# Build CLIP-based vocab scores for Visual Genome (~108K images).
# VG region_descriptions.json is large (~1.5 GB) — mem=48G covers JSON load + tensors.
# --caption-stats is optional: remove it to get pure CLIP targets (alpha irrelevant),
# or point it at a cached VisualGenomeDataset .pt to produce mixed_labels.

python build_clip_vocab_scores.py \
  --dataset vg \
  --data-root /net/tscratch/people/plgabedychaj/vg \
  --annotations /net/tscratch/people/plgabedychaj/vg/region_descriptions.json \
  --vocab-cache /net/tscratch/people/plgabedychaj/vocab/vg_cache.pt \
  --clip-model ViT-B-32 \
  --clip-pretrained openai \
  --temperature 1.0 \
  --alpha 1.0 \
  --output /net/tscratch/people/plgabedychaj/vocab/vg_clip_scores.pt \
  --batch-size 512 \
  --num-workers 8
