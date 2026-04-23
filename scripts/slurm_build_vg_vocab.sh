#!/bin/bash
#SBATCH --job-name=build_vg_vocab
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --account=plgbcfg-gpu-a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/net/tscratch/people/plgabedychaj/logs/vocab_vg_%j.out
#SBATCH --error=/net/tscratch/people/plgabedychaj/logs/vocab_vg_%j.err

set -e
mkdir -p /net/tscratch/people/plgabedychaj/logs
mkdir -p /net/tscratch/people/plgabedychaj/vocab

export PYTHONPATH="/net/tscratch/people/plgabedychaj/dinov2:$PYTHONPATH"
source /net/tscratch/people/plgabedychaj/venv/bin/activate
cd ~/proto-VLM/proto-non-param

python vocab/build_vg_vocab.py \
  --region-descriptions /net/tscratch/people/plgabedychaj/vg/region_descriptions.json \
  --vocab-out  /net/tscratch/people/plgabedychaj/vocab/vg.txt \
  --cache-out  /net/tscratch/people/plgabedychaj/vocab/vg_cache.pt \
  --clip-model-name ViT-L-14 \
  --clip-pretrained openai \
  --min-count 5 \
  --max-doc-freq 0.5
