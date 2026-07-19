#!/bin/bash
# Zero-shot RIS evaluation for runs T (SK+KoLeo + PMSN prior tau=0.5) and
# U (SK+VICReg, no KoLeo). 7 jobs per run (Gref/val + unc,unc+ × val/testA/testB).
#
# Usage:
#   bash scripts/slurm_eval_vg_contrastive_T_U.sh

set -e

SCRATCH="/net/tscratch/people/plgabedychaj"
REPO=~/proto-non-param
CONTR_BASE="${CONTR_BASE:-${SCRATCH}/train_logs/vg_contrastive}"
DATA_ROOT="${DATA_ROOT:-${SCRATCH}/data/refcoco}"

PARTITION="plgrid-gpu-a100"
ACCOUNT="plgunhype-gpu-a100"
LOG_SLURM="${SCRATCH}/logs"

mkdir -p "${LOG_SLURM}"

declare -A CKPTS=(
    [T]="${CONTR_BASE}/run_T_vitl14_sk10_koleo01_priortau05_30ep/ckpt.pth"
    [U]="${CONTR_BASE}/run_U_vitl14_sk10_vicreg01_30ep/ckpt.pth"
)

for MODEL in T U; do
    CKPT="${CKPTS[$MODEL]}"
    OUT_DIR="${REPO}/eval_results/vg_contrastive/contr_${MODEL}"

    echo ""
    echo "=== PNP-${MODEL}: ${CKPT} ==="
    if [ ! -f "${CKPT}" ]; then
        echo "ERROR: checkpoint not found — skipping ${MODEL}"
        continue
    fi

    for DATASET in Gref unc unc+; do
        case $DATASET in
            Gref) SPLITS="val" ;;
            *)    SPLITS="val testA testB" ;;
        esac

        for SPLIT in $SPLITS; do
            JOB=$(sbatch --parsable \
                --job-name="pnp-${MODEL}-${DATASET}-${SPLIT}" \
                --partition="${PARTITION}" \
                --account="${ACCOUNT}" \
                --gres=gpu:1 \
                --cpus-per-task=4 \
                --mem=32G \
                --time=04:00:00 \
                --output="${LOG_SLURM}/pnp_${MODEL}_${DATASET}_${SPLIT}_%j.out" \
                --error="${LOG_SLURM}/pnp_${MODEL}_${DATASET}_${SPLIT}_%j.err" \
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
  --out_dir ${OUT_DIR}
")
            echo "  ${JOB}  ${DATASET}/${SPLIT}"
        done
    done
done

echo ""
echo "Jobs submitted. Monitor with: squeue -u \$USER"
