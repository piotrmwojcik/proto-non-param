#!/bin/bash
#SBATCH --job-name=download_cub200
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --account=plgbcfg-gpu-a100
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --output=/net/tscratch/people/plgabedychaj/logs/download_cub200_%j.out
#SBATCH --error=/net/tscratch/people/plgabedychaj/logs/download_cub200_%j.err

mkdir -p /net/tscratch/people/plgabedychaj/logs

source /net/tscratch/people/plgabedychaj/venv/bin/activate
cd ~/proto-VLM/proto-non-param

python scripts/download_cub200.py \
  --output-dir /net/tscratch/people/plgabedychaj/cub200 \
  --val-per-class 5 \
  --seed 42
