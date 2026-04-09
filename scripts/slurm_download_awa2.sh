#!/bin/bash
#SBATCH --job-name=download_awa2
#SBATCH --partition=all
#SBATCH --account=plgbcfg-gpu-a100
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --output=/net/tscratch/people/plgabedychaj/logs/download_awa2_%j.out
#SBATCH --error=/net/tscratch/people/plgabedychaj/logs/download_awa2_%j.err

mkdir -p /net/tscratch/people/plgabedychaj/logs

source /net/tscratch/people/plgabedychaj/venv/bin/activate
cd ~/proto-VLM/proto-non-param

python scripts/download_awa2.py \
  --output-dir /net/tscratch/people/plgabedychaj/awa2 \
  --train-ratio 0.7 \
  --val-ratio 0.1 \
  --seed 42
