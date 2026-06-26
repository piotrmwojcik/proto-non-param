#!/bin/bash
# Collapse check: evaluate milestone checkpoints from a long training run.
#
# Submits one SLURM job per saved ckpt_epXXX.pth, evaluating on Gref/val
# (fast, ~10 min per checkpoint). Results are written to:
#   eval_results/collapse_check/{VARIANT}/ep{N}/pnp_refer/Gref_val.json
#
# After all jobs finish, print the learning curve:
#   python scripts/summarize_collapse_check.py \
#       --check-dir eval_results/collapse_check/{VARIANT}
#
# Configuration — override via environment variables:
#   CKPT_DIR   Directory containing ckpt_ep010.pth … ckpt_ep080.pth (required)
#   VARIANT    Label for output subdir, e.g. "dedup_A_80ep"
#   DATA_ROOT  Path to refcoco/ directory
#   EPOCHS     Space-separated list of epoch numbers to check (default: 10 20 30 40 50 60 70 80)
#
# Usage:
#   CKPT_DIR=$SCRATCH/train_logs/vg_dedup/run_A_kl_frozen_dedup_t090_80ep \
#   VARIANT=dedup_A_80ep \
#   bash scripts/slurm_eval_collapse_check.sh

set -e

SCRATCH="/net/tscratch/people/plgabedychaj"
REPO=~/proto-non-param
DATA_ROOT="${DATA_ROOT:-${SCRATCH}/data/refcoco}"
EPOCHS="${EPOCHS:-10 20 30 40 50 60 70 80}"

if [ -z "${CKPT_DIR}" ]; then
    echo "ERROR: CKPT_DIR is not set."
    echo "Usage: CKPT_DIR=<path/to/run_dir> VARIANT=<label> bash $0"
    exit 1
fi

if [ -z "${VARIANT}" ]; then
    echo "ERROR: VARIANT is not set (used as output subdir label)."
    echo "Usage: CKPT_DIR=<path> VARIANT=dedup_A_80ep bash $0"
    exit 1
fi

PARTITION="plgrid-gpu-a100"
ACCOUNT="plgunhype-gpu-a100"
LOG_SLURM="${SCRATCH}/logs"
OUT_BASE="${REPO}/eval_results/collapse_check/${VARIANT}"

mkdir -p "${LOG_SLURM}"

echo "=== Collapse Check — ${VARIANT} ==="
echo "  CKPT_DIR : ${CKPT_DIR}"
echo "  Epochs   : ${EPOCHS}"
echo "  Results  : ${OUT_BASE}/ep{N}/pnp_refer/Gref_val.json"
echo ""

MISSING=0
for EP_NUM in ${EPOCHS}; do
    EP=$(printf "%03d" ${EP_NUM})
    CKPT="${CKPT_DIR}/ckpt_ep${EP}.pth"
    if [ ! -f "${CKPT}" ]; then
        echo "  WARNING: checkpoint not found — ${CKPT}"
        MISSING=$((MISSING + 1))
    fi
done

if [ "${MISSING}" -gt 0 ]; then
    echo ""
    echo "  ${MISSING} checkpoint(s) missing. Training may still be in progress,"
    echo "  or --save-every was not set. Only found checkpoints will be evaluated."
    echo ""
fi

SUBMITTED=0
for EP_NUM in ${EPOCHS}; do
    EP=$(printf "%03d" ${EP_NUM})
    CKPT="${CKPT_DIR}/ckpt_ep${EP}.pth"
    [ -f "${CKPT}" ] || continue

    OUT_DIR="${OUT_BASE}/ep${EP}"

    JOB=$(sbatch --parsable \
      --job-name="collapse-${VARIANT}-ep${EP}" \
      --partition="${PARTITION}" \
      --account="${ACCOUNT}" \
      --gres=gpu:1 \
      --cpus-per-task=4 \
      --mem=32G \
      --time=02:00:00 \
      --output="${LOG_SLURM}/collapse_${VARIANT}_ep${EP}_%j.out" \
      --error="${LOG_SLURM}/collapse_${VARIANT}_ep${EP}_%j.err" \
      --wrap="
set -e
source ${SCRATCH}/venv/bin/activate
export HF_HUB_CACHE=${SCRATCH}/.cache/huggingface/hub
export TRANSFORMERS_CACHE=${SCRATCH}/.cache/huggingface/hub
cd ${REPO}

echo '========================================'
echo ' Collapse check: ${VARIANT}  epoch ${EP_NUM}'
echo ' Checkpoint: ${CKPT}'
echo '========================================'

python scripts/evaluate_pnp_refer.py \
  --ckpt ${CKPT} \
  --dataset Gref \
  --data_split val \
  --data_root ${DATA_ROOT} \
  --out_dir ${OUT_DIR}

echo ''
echo '--- Summary ep${EP_NUM} ---'
python -c \"
import json
ep_num = ${EP_NUM}
path = '${OUT_DIR}/pnp_refer/Gref_val.json'
with open(path) as f: r = json.load(f)
s = r['summary']
oiou, miou = s['cIoU'], s['mIoU']
flag = ''
if oiou < 5:
    flag = '  *** COLLAPSE DETECTED ***'
elif oiou < 10:
    flag = '  !! LOW — check carefully'
print(f'  Epoch {ep_num:>3}  oIoU: {oiou:5.1f}%   mIoU: {miou:5.1f}%{flag}')
\"
")

    echo "  Submitted job ${JOB}  →  ep${EP_NUM}"
    SUBMITTED=$((SUBMITTED + 1))
done

echo ""
echo "${SUBMITTED} job(s) submitted. Monitor with: squeue -u \$USER"
echo ""
echo "After all jobs finish, print the full learning curve:"
echo "  python scripts/summarize_collapse_check.py \\"
echo "      --check-dir ${OUT_BASE}"
