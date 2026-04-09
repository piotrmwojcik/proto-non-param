#!/bin/bash
#SBATCH --job-name=train_cub200
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --account=plgbcfg-gpu-a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --output=/net/tscratch/people/plgabedychaj/logs/train_%j.out
#SBATCH --error=/net/tscratch/people/plgabedychaj/logs/train_%j.err

mkdir -p /net/tscratch/people/plgabedychaj/logs
export HF_HOME=/net/tscratch/people/plgabedychaj/hf_cache
export TRANSFORMERS_CACHE=/net/tscratch/people/plgabedychaj/hf_cache
export PYTHONPATH="/net/tscratch/people/plgabedychaj/dinov2:$PYTHONPATH"

source /net/tscratch/people/plgabedychaj/venv/bin/activate
cd ~/proto-VLM/proto-non-param

python train.py \
  --dataset cub200 \
  --cub-root /net/tscratch/people/plgabedychaj/cub200 \
  --cub-annotations /net/tscratch/people/plgabedychaj/cub200/annotations \
  --vocab-cache-path /net/tscratch/people/plgabedychaj/vocab/cub200_cache.pt \
  --log-dir /net/tscratch/people/plgabedychaj/train_logs/cub200_exp1 \
  --batch-size 64 \
  --epochs 50 \
  --num-workers 8 \
  --wandb-log-images 16
