#!/bin/bash
# Batch evaluation of J and K2 with hardest-sample image dump.
#
# Submits 14 SLURM jobs (7 per model) in parallel.
#
# Usage:
#   bash scripts/slurm_eval_hardest_J_K2.sh
#
# To dump more/fewer samples:
#   DUMP_N=20 bash scripts/slurm_eval_hardest_J_K2.sh
#
# Hardest samples land in:
#   eval_results/vg_contrastive/contr_J/pnp_refer/hardest_{dataset}_{split}/
#   eval_results/vg_contrastive/contr_K2/pnp_refer/hardest_{dataset}_{split}/

set -e

SCRATCH="/net/tscratch/people/plgabedychaj"
REPO=~/proto-non-param
CONTR_BASE="${CONTR_BASE:-${SCRATCH}/train_logs/vg_contrastive}"
DATA_ROOT="${DATA_ROOT:-${SCRATCH}/data/refcoco}"
DUMP_N="${DUMP_N:-10}"

PARTITION="plgrid-gpu-a100"
ACCOUNT="plgunhype-gpu-a100"
LOG_SLURM="${SCRATCH}/logs"

mkdir -p "${LOG_SLURM}"

declare -A CKPTS=(
    [J]="${CONTR_BASE}/run_J_vitl14_contrastive10_k1_sk10_30ep/ckpt.pth"
    [K2]="${CONTR_BASE}/run_K2_vitl14_contrastive10_k1_sk10_koleo01_30ep/ckpt.pth"
)

for MODEL in J K2; do
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
  --out_dir ${OUT_DIR} \
  --dump-hardest ${DUMP_N}
")
            echo "  ${JOB}  ${DATASET}/${SPLIT}"
        done
    done
done

echo ""
echo "14 jobs submitted. Monitor with: squeue -u \$USER"
