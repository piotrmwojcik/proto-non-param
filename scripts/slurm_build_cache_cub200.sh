#!/bin/bash
#SBATCH --job-name=build_cub200_cache
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --gres=gpu:1
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --output=/net/tscratch/people/plgabedychaj/logs/cache_cub200_%j.out
#SBATCH --error=/net/tscratch/people/plgabedychaj/logs/cache_cub200_%j.err

mkdir -p /net/tscratch/people/plgabedychaj/logs
mkdir -p /net/tscratch/people/plgabedychaj/vocab

source /net/tscratch/people/plgabedychaj/venv/bin/activate
cd ~/proto-VLM/proto-non-param

python scripts/build_cub200_cache.py \
  --annotations-dir /net/tscratch/people/plgabedychaj/cub200/annotations \
  --cache-out /net/tscratch/people/plgabedychaj/vocab/cub200_cache.pt
