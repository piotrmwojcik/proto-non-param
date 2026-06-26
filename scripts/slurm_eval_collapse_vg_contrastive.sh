#!/bin/bash
# Collapse check for the two contrastive VG training runs.
#
# Evaluates milestone checkpoints ep005 … ep030 (saved every 5 epochs)
# on Gref/val only — fast (~10 min per checkpoint).
#
# Results land in:
#   eval_results/collapse_check/contrastive_A_30ep/ep{N}/pnp_refer/Gref_val.json
#   eval_results/collapse_check/contrastive_B_30ep/ep{N}/pnp_refer/Gref_val.json
#
# After all jobs finish, print each learning curve:
#   python scripts/summarize_collapse_check.py \
#       --check-dir eval_results/collapse_check/contrastive_A_30ep
#   python scripts/summarize_collapse_check.py \
#       --check-dir eval_results/collapse_check/contrastive_B_30ep
#
# Configuration — override via environment variables:
#   CONTR_BASE   Base dir with run_A_* / run_B_* subdirs
#   DATA_ROOT    Path to refcoco/ directory
#   EPOCHS       Space-separated milestone epoch numbers (default: 5 10 15 20 25 30)
#
# Usage:
#   bash scripts/slurm_eval_collapse_vg_contrastive.sh

set -e

SCRATCH="/net/tscratch/people/plgabedychaj"
REPO=~/proto-non-param
CONTR_BASE="${CONTR_BASE:-${SCRATCH}/train_logs/vg_contrastive}"
DATA_ROOT="${DATA_ROOT:-${SCRATCH}/data/refcoco}"
EPOCHS="${EPOCHS:-5 10 15 20 25 30}"

PARTITION="plgrid-gpu-a100"
ACCOUNT="plgunhype-gpu-a100"
LOG_SLURM="${SCRATCH}/logs"

mkdir -p "${LOG_SLURM}"

declare -A CKPT_DIRS=(
  [A]="${CONTR_BASE}/run_A_uniform_contrastive05_30ep"
  [B]="${CONTR_BASE}/run_B_uniform_contrastive10_30ep"
)

declare -A VARIANTS=(
  [A]="contrastive_A_30ep"
  [B]="contrastive_B_30ep"
)

echo "=== Collapse Check — Contrastive VG runs A and B (30 epochs) ==="
echo "  Epochs : ${EPOCHS}"
echo ""

TOTAL=0

for RUN in A B; do
  CKPT_DIR="${CKPT_DIRS[$RUN]}"
  VARIANT="${VARIANTS[$RUN]}"
  OUT_BASE="${REPO}/eval_results/collapse_check/${VARIANT}"

  echo "-- Run ${RUN} (${VARIANT}) --"
  echo "   CKPT_DIR : ${CKPT_DIR}"

  SUBMITTED=0
  for EP_NUM in ${EPOCHS}; do
    EP=$(printf "%03d" ${EP_NUM})
    CKPT="${CKPT_DIR}/ckpt_ep${EP}.pth"
    if [ ! -f "${CKPT}" ]; then
      echo "   WARNING: checkpoint not found — ${CKPT}"
      continue
    fi

    OUT_DIR="${OUT_BASE}/ep${EP}"

    JOB=$(sbatch --parsable \
      --job-name="collapse-contr${RUN}-ep${EP}" \
      --partition="${PARTITION}" \
      --account="${ACCOUNT}" \
      --gres=gpu:1 \
      --cpus-per-task=4 \
      --mem=32G \
      --time=02:00:00 \
      --output="${LOG_SLURM}/collapse_contr${RUN}_ep${EP}_%j.out" \
      --error="${LOG_SLURM}/collapse_contr${RUN}_ep${EP}_%j.err" \
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

    echo "   Submitted job ${JOB}  →  ep${EP_NUM}"
    SUBMITTED=$((SUBMITTED + 1))
    TOTAL=$((TOTAL + 1))
  done
  echo "   ${SUBMITTED} job(s) submitted for run ${RUN}"
  echo ""
done

echo "${TOTAL} total job(s) submitted. Monitor with: squeue -u \$USER"
echo ""
echo "After all jobs finish, summarize each run's learning curve:"
echo "  python scripts/summarize_collapse_check.py \\"
echo "      --check-dir ${REPO}/eval_results/collapse_check/contrastive_A_30ep"
echo "  python scripts/summarize_collapse_check.py \\"
echo "      --check-dir ${REPO}/eval_results/collapse_check/contrastive_B_30ep"
