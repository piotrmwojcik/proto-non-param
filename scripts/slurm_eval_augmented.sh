#!/bin/bash
#SBATCH --job-name=eval-augmented
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --account=plgunhype-gpu-a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/net/tscratch/people/plgabedychaj/logs/eval_augmented_%j.out
#SBATCH --error=/net/tscratch/people/plgabedychaj/logs/eval_augmented_%j.err

set -e
mkdir -p /net/tscratch/people/plgabedychaj/logs

export HF_HOME=/net/tscratch/people/plgabedychaj/hf_cache
export PYTHONPATH=/net/tscratch/people/plgabedychaj/dinov2:$PYTHONPATH
source /net/tscratch/people/plgabedychaj/venv/bin/activate
cd ~/proto-VLM/proto-non-param

python eval_augmented_prototypes.py \
  --ckpt /net/tscratch/people/plgabedychaj/train_logs/vg_baseline/ckpt.pth \
  --vocab-cache-path /net/tscratch/people/plgabedychaj/vocab/vg_cache.pt \
  --caption-prototypes-path /net/tscratch/people/plgabedychaj/vocab/vg_test_caption_prototypes.pt \
  --source-dataset vg_test \
  --vg-root /net/tscratch/people/plgabedychaj/vg \
  --vg-region-descriptions /net/tscratch/people/plgabedychaj/vg/region_descriptions.json \
  --mode both \
  --topk 5 \
  --batch-size 64 \
  --num-workers 8 \
  --wandb-entity gmum \
  --wandb-run-name vg-caption-ablation-vgtest
