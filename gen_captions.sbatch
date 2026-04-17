#!/bin/bash
#SBATCH --job-name=print-cubs-a100
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH -A plgtabcfs-gpu-a100
#SBATCH --gres=gpu:1
#SBATCH --mem=32GB
#SBATCH --time=24:00:00
#SBATCH --output=logs/job_%j.out
#SBATCH --error=logs/job_%j.err

set -euo pipefail

mkdir -p logs

export HF_HOME=/net/tscratch/people/plgpiotrwojcik/.cache/huggingface
export HUGGINGFACE_HUB_CACHE=/net/tscratch/people/plgpiotrwojcik/.cache/huggingface/hub
export SCRATCH_HOME=/net/tscratch/people/plgpiotrwojcik/

source ~/.bashrc
conda activate unhype

cd /net/people/plgrid/plgpiotrwojcik/proto-non-param

python print_cubs.py /net/tscratch/people/plgpiotrwojcik/CUBS/ \
    --output-dir /net/tscratch/people/plgpiotrwojcik/CUBS/cub_crops \
    --csv-out /net/tscratch/people/plgpiotrwojcik/CUBS/cub_captions.csv \
    --use-llm \
    --llm-backend hf \
    --llm-model Qwen/Qwen2.5-7B-Instruct \
    --num-captions 5 \
    --caption-max-words 28 \
    --llm-max-tokens 96 \
    --keep-raw-if-llm-fails