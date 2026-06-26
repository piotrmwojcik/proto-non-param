#!/bin/bash
#SBATCH --job-name=pnp-eval
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --account=plgunhype-gpu-a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/net/tscratch/people/plgabedychaj/logs/pnp_eval_%j.out
#SBATCH --error=/net/tscratch/people/plgabedychaj/logs/pnp_eval_%j.err

# Evaluate PNP (Proto-Non-Parametric) on a single dataset + split using
# zero-shot referring image segmentation (no fine-tuning on RIS data).
#
# Required env vars:
#   DATASET  — Gref | unc | unc+
#   SPLIT    — val | testA | testB
#   CKPT     — path to PNP .pth checkpoint
#
# Examples:
#   DATASET=Gref SPLIT=val CKPT=$SCRATCH/checkpoints/pnp/best.pth \
#       sbatch scripts/slurm_eval_pnp_refer.sh
#
#   for split in val testA testB; do
#       DATASET=unc SPLIT=$split CKPT=$SCRATCH/checkpoints/pnp/best.pth \
#           sbatch scripts/slurm_eval_pnp_refer.sh
#   done

set -euo pipefail

SCRATCH=/net/tscratch/people/plgabedychaj
REPO=~/proto-non-param

: "${DATASET:?'DATASET env var required (Gref|unc|unc+)'}"
: "${SPLIT:?'SPLIT env var required (val|testA|testB)'}"
: "${CKPT:?'CKPT env var required (path to .pth file)'}"

mkdir -p "$SCRATCH/logs"

echo "==> Job started on $(hostname) at $(date)"
echo "==> Evaluating PNP: dataset=$DATASET  split=$SPLIT"
echo "==> Checkpoint: $CKPT"

module load Python/3.10.4
module load CUDA/12.4.0
module load cuDNN/9.2.1.18-CUDA-12.4.0

source "$SCRATCH/venv/bin/activate"

export HF_HUB_CACHE="$SCRATCH/.cache/huggingface/hub"
export TRANSFORMERS_CACHE="$SCRATCH/.cache/huggingface/hub"

cd "$REPO"

python scripts/evaluate_pnp_refer.py \
    --ckpt "$CKPT" \
    --dataset "$DATASET" \
    --data_split "$SPLIT" \
    --data_root "$SCRATCH/data/refcoco" \
    --out_dir eval_results

# Copy results to scratch for easy access
RESULT_JSON="eval_results/pnp_refer/${DATASET}_${SPLIT}.json"
if [ -f "$RESULT_JSON" ]; then
    mkdir -p "$SCRATCH/eval_results/pnp_refer"
    cp "$RESULT_JSON" "$SCRATCH/eval_results/pnp_refer/"
    echo "==> Results copied to $SCRATCH/eval_results/pnp_refer/"
fi

echo "==> Eval done at $(date)"
