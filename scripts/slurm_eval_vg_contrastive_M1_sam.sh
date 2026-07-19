#!/bin/bash
# Tier-0 experiment: SAM proposal-and-rank inference on the existing M1 checkpoint.
#
# Instead of thresholding the dense activation map, SAM generates instance mask
# proposals and the proposal with the highest mean activation wins — the object-
# competition mechanism behind current zero-shot RIS SOTA (CoPatch 44.1 Gref mIoU,
# HybridGL, TAS all use propose-then-rank; CTRL-O gets it from slot attention).
# Directly targets the wrong-blob and no-instance-competition failures in the
# hardest-Gref dump. No retraining.
#
# One-time setup on Athena before first run:
#   pip install segment-anything                       # inside $SCRATCH/venv
#   mkdir -p $SCRATCH/sam && wget -P $SCRATCH/sam \
#     https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
#
# Usage:
#   bash scripts/slurm_eval_vg_contrastive_M1_sam.sh
#
# Compare with:
#   BASE_DIR=eval_results/vg_contrastive/contr_M1/pnp_refer \
#   SI_DIR=eval_results/vg_contrastive/contr_M1_sam/pnp_refer \
#   bash scripts/compare_single_instance_eval.sh

set -e

SCRATCH="/net/tscratch/people/plgabedychaj"
REPO=~/proto-non-param
CONTR_BASE="${CONTR_BASE:-${SCRATCH}/train_logs/vg_contrastive}"
DATA_ROOT="${DATA_ROOT:-${SCRATCH}/data/refcoco}"
SAM_CKPT="${SAM_CKPT:-${SCRATCH}/sam/sam_vit_h_4b8939.pth}"

PARTITION="plgrid-gpu-a100"
ACCOUNT="plgunhype-gpu-a100"
LOG_SLURM="${SCRATCH}/logs"

CKPT="${CONTR_BASE}/run_M1_vitl14_sk10_koleo01_30ep/ckpt.pth"

mkdir -p "${LOG_SLURM}"

if [ ! -f "${CKPT}" ]; then
    echo "ERROR: checkpoint not found: ${CKPT}"
    exit 1
fi

if [ ! -f "${SAM_CKPT}" ]; then
    echo "ERROR: SAM checkpoint not found: ${SAM_CKPT}"
    echo "Download it first:"
    echo "  mkdir -p ${SCRATCH}/sam && wget -P ${SCRATCH}/sam \\"
    echo "    https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
    exit 1
fi

if ! (source "${SCRATCH}/venv/bin/activate" && python -c "import segment_anything" 2>/dev/null); then
    echo "ERROR: segment-anything not installed in ${SCRATCH}/venv"
    echo "Install it first:  source ${SCRATCH}/venv/bin/activate && pip install segment-anything"
    exit 1
fi

OUT_DIR="${REPO}/eval_results/vg_contrastive/contr_M1_sam"

echo "=== PNP — Zero-shot RIS Evaluation (run M1 + SAM proposal-and-rank) ==="
echo "  Ckpt : ${CKPT}"
echo "  SAM  : ${SAM_CKPT}"
echo "  Data : ${DATA_ROOT}"
echo ""

declare -A SPLITS=(
  [Gref]="val"
  [unc]="val"
  [unc+]="val"
)

# SAM automatic mask generation is ~2-5 s/image on A100 → generous time limit.
for DATASET in Gref unc unc+; do
    for SPLIT in ${SPLITS[$DATASET]}; do
        JOB=$(sbatch --parsable \
            --job-name="pnp-M1sam-${DATASET}-${SPLIT}" \
            --partition="${PARTITION}" \
            --account="${ACCOUNT}" \
            --gres=gpu:1 \
            --cpus-per-task=4 \
            --mem=48G \
            --time=12:00:00 \
            --output="${LOG_SLURM}/pnp_M1sam_${DATASET}_${SPLIT}_%j.out" \
            --error="${LOG_SLURM}/pnp_M1sam_${DATASET}_${SPLIT}_%j.err" \
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
  --sam-checkpoint ${SAM_CKPT} \
  --sam-model-type vit_h
")
        echo "  ${JOB}  ${DATASET}/${SPLIT}"
    done
done

echo ""
echo "3 jobs submitted. Monitor with: squeue -u \$USER"
