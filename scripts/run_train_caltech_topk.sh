#!/bin/bash
#SBATCH --job-name=train_caltech_topk
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --account=plgbcfg-gpu-a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --output=/net/tscratch/people/plgabedychaj/logs/train_caltech_topk_%j.out
#SBATCH --error=/net/tscratch/people/plgabedychaj/logs/train_caltech_topk_%j.err

set -e
mkdir -p /net/tscratch/people/plgabedychaj/logs

export HF_HOME=/net/tscratch/people/plgabedychaj/hf_cache
export TRANSFORMERS_CACHE=/net/tscratch/people/plgabedychaj/hf_cache
export PYTHONPATH="/net/tscratch/people/plgabedychaj/dinov2:$PYTHONPATH"

source /net/tscratch/people/plgabedychaj/venv/bin/activate
cd ~/proto-VLM/proto-non-param

# Top-K binarized targets with KL divergence.
# Per image: keep the K most frequent vocabulary concepts, assign uniform 1/K weight.
# All other concepts get 0. KL loss trains against this peaked uniform distribution.

python train.py \
  --dataset caltech101 \
  --caltech-root /net/tscratch/people/plgabedychaj/caltech101 \
  --caltech-descriptions /net/tscratch/people/plgabedychaj/descriptions/caltech101_descriptions.json \
  --vocab-cache-path /net/tscratch/people/plgabedychaj/vocab/caltech101_cache.pt \
  --log-dir /net/tscratch/people/plgabedychaj/train_logs/caltech_topk10 \
  --target-mode topk \
  --top-k-concepts 10 \
  --batch-size 64 \
  --epochs 50 \
  --num-workers 8 \
  --backbone-lr 1e-5 \
  --text-proj-lr 1e-4 \
  --kl-coef 1.0 \
  --wandb-entity gmum \
  --wandb-log-images 16
