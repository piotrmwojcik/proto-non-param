#!/bin/bash
#SBATCH --job-name=download_vg
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --account=plgbcfg-gpu-a100
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=/net/tscratch/people/plgabedychaj/logs/download_vg_%j.out
#SBATCH --error=/net/tscratch/people/plgabedychaj/logs/download_vg_%j.err

set -e
mkdir -p /net/tscratch/people/plgabedychaj/logs

source /net/tscratch/people/plgabedychaj/venv/bin/activate
cd ~/proto-VLM/proto-non-param

python scripts/download_vg.py \
  --output-dir /net/tscratch/people/plgabedychaj/vg
