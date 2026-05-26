#!/bin/bash
#SBATCH --job-name=train_coco_vg_binary
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --account=plgunhype-gpu-a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --output=/net/tscratch/people/plgabedychaj/logs/train_coco_vg_binary_%j.out
#SBATCH --error=/net/tscratch/people/plgabedychaj/logs/train_coco_vg_binary_%j.err

set -e
mkdir -p /net/tscratch/people/plgabedychaj/logs

export HF_HOME=/net/tscratch/people/plgabedychaj/hf_cache
export TRANSFORMERS_CACHE=/net/tscratch/people/plgabedychaj/hf_cache
export PYTHONPATH="/net/tscratch/people/plgabedychaj/dinov2:$PYTHONPATH"

source /net/tscratch/people/plgabedychaj/venv/bin/activate
cd ~/proto-VLM/proto-non-param

# Binary concept targets: each vocab word present in captions/regions gets label=1, absent=0.
# BCE loss with pos_weight=100 compensates for ~800:1 negative:positive ratio.
# If mixture weights collapse to zero in the first epoch, increase --bce-pos-weight.

python train.py \
  --dataset coco_vg \
  --vg-root /net/tscratch/people/plgabedychaj/vg \
  --vg-region-descriptions /net/tscratch/people/plgabedychaj/vg/region_descriptions.json \
  --coco-root /net/tscratch/people/plgabedychaj/coco_dataset/raw \
  --coco-annotations-train /net/tscratch/people/plgabedychaj/coco_dataset/raw/annotations/captions_train2017.json \
  --coco-annotations-val /net/tscratch/people/plgabedychaj/coco_dataset/raw/annotations/captions_val2017.json \
  --vocab-cache-path /net/tscratch/people/plgabedychaj/vocab/vg_cache.pt \
  --log-dir /net/tscratch/people/plgabedychaj/train_logs/coco_vg_binary \
  --target-mode binary \
  --bce-coef 1.0 \
  --bce-pos-weight 100.0 \
  --backbone dinov2_vitb14 \
  --batch-size 64 \
  --epochs 100 \
  --num-workers 8 \
  --backbone-lr 1e-5 \
  --text-proj-lr 1e-4 \
  --wandb-entity gmum \
  --wandb-log-images 16
