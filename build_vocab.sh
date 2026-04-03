#!/bin/bash
#SBATCH --job-name=build_vocab
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/net/tscratch/people/plgabedychaj/logs/vocab_%j.out
#SBATCH --error=/net/tscratch/people/plgabedychaj/logs/vocab_%j.err

mkdir -p /net/tscratch/people/plgabedychaj/logs
mkdir -p /net/tscratch/people/plgabedychaj/vocab

export PYTHONPATH="/net/tscratch/people/plgabedychaj/dinov2:$PYTHONPATH"
source /net/tscratch/people/plgabedychaj/venv/bin/activate
cd ~/proto-VLM/proto-non-param

python build_caltech101_vocab.py \
  --descriptions /net/tscratch/people/plgabedychaj/descriptions/caltech101_descriptions.json \
  --vocab-out vocab/caltech101.txt \
  --cache-out /net/tscratch/people/plgabedychaj/vocab/caltech101_cache.pt
