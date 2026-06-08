#!/bin/bash
#SBATCH --job-name=vis-captions
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --account=plgunhype-gpu-a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/net/tscratch/people/plgabedychaj/logs/vis_captions_%j.out
#SBATCH --error=/net/tscratch/people/plgabedychaj/logs/vis_captions_%j.err

set -e
mkdir -p /net/tscratch/people/plgabedychaj/logs

export HF_HOME=/net/tscratch/people/plgabedychaj/hf_cache
export PYTHONPATH=/net/tscratch/people/plgabedychaj/dinov2:$PYTHONPATH
source /net/tscratch/people/plgabedychaj/venv/bin/activate
cd ~/proto-VLM/proto-non-param

python visualize_caption_activation.py \
  --ckpt /net/tscratch/people/plgabedychaj/train_logs/vg_baseline/ckpt.pth \
  --vocab-cache-path /net/tscratch/people/plgabedychaj/vocab/vg_cache.pt \
  --caption-prototypes-path /net/tscratch/people/plgabedychaj/vocab/vg_test_caption_prototypes.pt \
  --source-dataset vg_test \
  --vg-root /net/tscratch/people/plgabedychaj/vg \
  --vg-region-descriptions /net/tscratch/people/plgabedychaj/vg/region_descriptions.json \
  --n-random 30 \
  --n-captions 4 \
  --out-dir /net/tscratch/people/plgabedychaj/caption_vis \
  --wandb-entity gmum \
  --wandb-run-name caption-activation-vis
