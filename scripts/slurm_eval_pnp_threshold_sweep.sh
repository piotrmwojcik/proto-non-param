#!/bin/bash
# Threshold ablation for PNP zero-shot RIS evaluation.
#
# Runs evaluate_pnp_refer.py with --threshold-sweep over a range of values.
# A single GPU job covers all thresholds in one forward pass (activation is
# computed once per sample; only the binarization step is repeated).
#
# Results land in:
#   eval_results/threshold_sweep/{VARIANT}/pnp_refer/{dataset}_{split}_tXXX.json
# where XXX = threshold × 100 (e.g. t050 for 0.50, t035 for 0.35).
#
# After all jobs complete, print a summary table:
#   python scripts/summarize_threshold_sweep.py \
#       --sweep-dir eval_results/threshold_sweep/{VARIANT}
#
# Configuration — override via environment variables:
#   CKPT         Path to checkpoint (required)
#   VARIANT      Label for output subdir, e.g. "dedup_A_80ep" (default: pnp)
#   DATA_ROOT    Path to refcoco/ directory
#   THRESHOLDS   Space-separated list of thresholds (default: 0.3 0.35 0.4 0.45 0.5 0.55 0.6 0.65 0.7)
#
# Usage:
#   CKPT=$SCRATCH/train_logs/vg_dedup/run_A_kl_frozen_dedup_t090_80ep/ckpt.pth \
#   VARIANT=dedup_A_80ep \
#   bash scripts/slurm_eval_pnp_threshold_sweep.sh

set -e

SCRATCH="/net/tscratch/people/plgabedychaj"
REPO=~/proto-non-param
DATA_ROOT="${DATA_ROOT:-${SCRATCH}/data/refcoco}"
VARIANT="${VARIANT:-pnp}"
THRESHOLDS="${THRESHOLDS:-0.3 0.35 0.4 0.45 0.5 0.55 0.6 0.65 0.7}"

if [ -z "${CKPT}" ]; then
    echo "ERROR: CKPT is not set."
    echo "Usage: CKPT=<path/to/ckpt.pth> VARIANT=<label> bash $0"
    exit 1
fi

if [ ! -f "${CKPT}" ]; then
    echo "ERROR: Checkpoint not found: ${CKPT}"
    exit 1
fi

PARTITION="plgrid-gpu-a100"
ACCOUNT="plgunhype-gpu-a100"
LOG_SLURM="${SCRATCH}/logs"
OUT_BASE="${REPO}/eval_results/threshold_sweep/${VARIANT}"

mkdir -p "${LOG_SLURM}"

declare -A SPLITS=(
  [Gref]="val"
  [unc]="val testA testB"
  [unc+]="val testA testB"
)

echo "=== PNP Threshold Sweep — ${VARIANT} ==="
echo "  Ckpt       : ${CKPT}"
echo "  Thresholds : ${THRESHOLDS}"
echo "  Results    : ${OUT_BASE}/pnp_refer/"
echo ""

for DATASET in Gref unc unc+; do
  for SPLIT in ${SPLITS[$DATASET]}; do
    JOB=$(sbatch --parsable \
      --job-name="pnp-thr-${VARIANT}-${DATASET}-${SPLIT}" \
      --partition="${PARTITION}" \
      --account="${ACCOUNT}" \
      --gres=gpu:1 \
      --cpus-per-task=4 \
      --mem=32G \
      --time=04:00:00 \
      --output="${LOG_SLURM}/pnp_thr_${VARIANT}_${DATASET}_${SPLIT}_%j.out" \
      --error="${LOG_SLURM}/pnp_thr_${VARIANT}_${DATASET}_${SPLIT}_%j.err" \
      --wrap="
set -e
source ${SCRATCH}/venv/bin/activate
export HF_HUB_CACHE=${SCRATCH}/.cache/huggingface/hub
export TRANSFORMERS_CACHE=${SCRATCH}/.cache/huggingface/hub
export TORCH_HOME=${SCRATCH}/torch_cache
export PYTHONPATH=${SCRATCH}/dinov2:\$PYTHONPATH
cd ${REPO}

python scripts/evaluate_pnp_refer.py \
  --ckpt ${CKPT} \
  --dataset ${DATASET} \
  --data_split ${SPLIT} \
  --data_root ${DATA_ROOT} \
  --out_dir ${OUT_BASE} \
  --threshold-sweep ${THRESHOLDS}
")
    echo "  ${JOB}  ${DATASET}/${SPLIT}"
  done
done

echo ""
echo "7 jobs submitted. Monitor with: squeue -u \$USER"
echo ""
echo "After completion, summarize with:"
echo "  python scripts/summarize_threshold_sweep.py \\"
echo "      --sweep-dir ${OUT_BASE} \\"
echo "      --out ${OUT_BASE}/threshold_comparison.md"
