#!/bin/bash
#SBATCH --job-name=train_coco_vg_clip
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --account=plgunhype-gpu-a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --output=/net/tscratch/people/plgabedychaj/logs/train_coco_vg_clip_%j.out
#SBATCH --error=/net/tscratch/people/plgabedychaj/logs/train_coco_vg_clip_%j.err

set -e
mkdir -p /net/tscratch/people/plgabedychaj/logs

export HF_HOME=/net/tscratch/people/plgabedychaj/hf_cache
export TRANSFORMERS_CACHE=/net/tscratch/people/plgabedychaj/hf_cache
export PYTHONPATH="/net/tscratch/people/plgabedychaj/dinov2:$PYTHONPATH"

source /net/tscratch/people/plgabedychaj/venv/bin/activate
cd ~/proto-VLM/proto-non-param

# Train with CLIP scalar-product vocabulary targets.
# Raw clip_scores (cosine similarities) are stored in the .pt files; softmax is
# applied at load time at --clip-scores-temperature=0.07 (CLIP's own training temp).
# Scores built by: sbatch scripts/slurm_build_clip_vocab_scores_{coco,vg}.sh

python train.py \
  --dataset coco_vg \
  --vg-root /net/tscratch/people/plgabedychaj/vg \
  --vg-region-descriptions /net/tscratch/people/plgabedychaj/vg/region_descriptions.json \
  --coco-root /net/tscratch/people/plgabedychaj/coco_dataset/raw \
  --coco-annotations-train /net/tscratch/people/plgabedychaj/coco_dataset/raw/annotations/captions_train2017.json \
  --coco-annotations-val /net/tscratch/people/plgabedychaj/coco_dataset/raw/annotations/captions_val2017.json \
  --vocab-cache-path vocab/mscoco_new_cache.pt \
  --clip-scores-vg /net/tscratch/people/plgabedychaj/vocab/vg_clip_scores.pt \
  --clip-scores-coco-train /net/tscratch/people/plgabedychaj/vocab/coco_train_clip_scores.pt \
  --clip-scores-coco-val /net/tscratch/people/plgabedychaj/vocab/coco_val_clip_scores.pt \
  --clip-scores-temperature 0.07 \
  --clip-scores-top-k 50 \
  --log-dir /net/tscratch/people/plgabedychaj/train_logs/coco_vg_clip_scores \
  --backbone dinov2_vitb14 \
  --batch-size 64 \
  --epochs 100 \
  --num-workers 8 \
  --backbone-lr 1e-5 \
  --text-proj-lr 1e-4 \
  --kl-coef 1.0 \
  --wandb-entity gmum \
  --wandb-log-images 16
