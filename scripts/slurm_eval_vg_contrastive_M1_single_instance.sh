#!/bin/bash
# Zero-shot RIS evaluation for run M1 with --single-instance mask post-processing
# (ViT-L/14 + SK + KoLeo, no contrastive). Same checkpoint as the M1 baseline eval —
# only the mask extraction changes (collapses to the single highest-mean-activation
# connected component), isolating the effect of the multi-instance-disambiguation fix.
#
# Runs only the "val" split of each dataset (Gref/val first is the primary test per
# the plan; unc/val + unc+/val are the cheap spot-check that it doesn't regress
# splits where multi-region masks are rarely an issue). testA/testB are intentionally
# left out of this first pass — rerun with the full M1 eval script's SPLITS if this
# looks promising.
#
# Usage:
#   bash scripts/slurm_eval_vg_contrastive_M1_single_instance.sh

set -e

SCRATCH="/net/tscratch/people/plgabedychaj"
REPO=~/proto-non-param
CONTR_BASE="${CONTR_BASE:-${SCRATCH}/train_logs/vg_contrastive}"
DATA_ROOT="${DATA_ROOT:-${SCRATCH}/data/refcoco}"

PARTITION="plgrid-gpu-a100"
ACCOUNT="plgunhype-gpu-a100"
LOG_SLURM="${SCRATCH}/logs"

CKPT="${CONTR_BASE}/run_M1_vitl14_sk10_koleo01_30ep/ckpt.pth"

mkdir -p "${LOG_SLURM}"

if [ ! -f "${CKPT}" ]; then
    echo "ERROR: checkpoint not found: ${CKPT}"
    exit 1
fi

OUT_DIR="${REPO}/eval_results/vg_contrastive/contr_M1_single_instance"

echo "=== PNP — Zero-shot RIS Evaluation (run M1, --single-instance mask post-processing) ==="
echo "  Ckpt : ${CKPT}"
echo "  Data : ${DATA_ROOT}"
echo ""

declare -A SPLITS=(
  [Gref]="val"
  [unc]="val"
  [unc+]="val"
)

for DATASET in Gref unc unc+; do
    for SPLIT in ${SPLITS[$DATASET]}; do
        JOB=$(sbatch --parsable \
            --job-name="pnp-M1si-${DATASET}-${SPLIT}" \
            --partition="${PARTITION}" \
            --account="${ACCOUNT}" \
            --gres=gpu:1 \
            --cpus-per-task=4 \
            --mem=32G \
            --time=04:00:00 \
            --output="${LOG_SLURM}/pnp_M1si_${DATASET}_${SPLIT}_%j.out" \
            --error="${LOG_SLURM}/pnp_M1si_${DATASET}_${SPLIT}_%j.err" \
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
  --out_dir ${OUT_DIR} \
  --single-instance
")
        echo "  ${JOB}  ${DATASET}/${SPLIT}"
    done
done

echo ""
echo "3 jobs submitted. Monitor with: squeue -u \$USER"
echo ""
echo "Compare against the M1 baseline (no --single-instance):"
echo "  eval_results/vg_contrastive/contr_M1/pnp_refer/{Gref_val,unc_val,unc+_val}.json"
echo "  eval_results/vg_contrastive/contr_M1_single_instance/pnp_refer/{Gref_val,unc_val,unc+_val}.json"
