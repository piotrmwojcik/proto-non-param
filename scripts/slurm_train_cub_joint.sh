#!/bin/bash
# Joint training of PNP's CUB-200 concept encoder + a classifier head, with a
# CLIP-substituted "sufficiency" regularizer (see scripts/train_cub_joint.py's
# module docstring for the full motivation and the CLIP-substitution caveat).
#
# *** CAVEAT: sufficiency_coef uses CLIP similarity as a weak, unverified
# substitute for ground-truth concept labels -- not a reproduction of the
# paper's mechanism (Espinosa Zarlenga et al., ICML 2026), an adapted,
# weaker-guarantee version of it. Report every result from this run with
# that caveat attached. ***
#
# Reuses the concept cache / filtered vocab / CLIP scores already built by
# slurm_train_cub_labelfreecbm.sh (Stage 2, Sequential) -- no rebuild here.
# Warm-starts from the same M1 base checkpoint Stage 2 used (not Stage 2's
# own fine-tuned result), so this is a clean, apples-to-apples comparison
# against Stage 2's 67.41% top-1 / 93.06% top-5 (Sequential training).
#
# Deliberately a separate, standalone pipeline: does not touch train.py,
# modeling/pnp.py, or clip_dataset.py.
#
# Usage:
#   bash scripts/slurm_train_cub_joint.sh

set -e

SCRATCH="${SCRATCH:-/net/tscratch/people/plgabedychaj}"
REPO=~/proto-non-param
CONTR_BASE="${CONTR_BASE:-${SCRATCH}/train_logs/vg_contrastive}"
CUB_ROOT="${CUB_ROOT:-${SCRATCH}/cub200}"
CUB_ANNOTATIONS="${CUB_ANNOTATIONS:-${SCRATCH}/cub200/annotations}"
VOCAB_DIR="${VOCAB_DIR:-${SCRATCH}/vocab}"
LOG_DIR="${LOG_DIR:-${SCRATCH}/train_logs/cub_joint}"
CLS_COEF="${CLS_COEF:-1.0}"
SUFFICIENCY_COEF="${SUFFICIENCY_COEF:-1.0}"

PARTITION="plgrid-gpu-a100"
ACCOUNT="plgunhype-gpu-a100"
LOG_SLURM="${SCRATCH}/logs"

INIT_CKPT="${CONTR_BASE}/run_M1_vitl14_sk10_koleo01_30ep/ckpt.pth"
FILTERED_VOCAB_CACHE="${VOCAB_DIR}/cub_clip_scores_vocab_filtered.pt"
SCORES_OUT="${VOCAB_DIR}/cub_clip_scores.pt"

mkdir -p "${LOG_SLURM}" "${LOG_DIR}"

if [ ! -f "${INIT_CKPT}" ]; then
    echo "ERROR: M1 checkpoint not found: ${INIT_CKPT}"
    exit 1
fi
if [ ! -f "${FILTERED_VOCAB_CACHE}" ]; then
    echo "ERROR: filtered CUB vocab cache not found: ${FILTERED_VOCAB_CACHE} -- run slurm_train_cub_labelfreecbm.sh first"
    exit 1
fi
if [ ! -f "${SCORES_OUT}" ]; then
    echo "ERROR: CUB CLIP scores not found: ${SCORES_OUT} -- run slurm_train_cub_labelfreecbm.sh first"
    exit 1
fi

JOB=$(sbatch --parsable \
    --job-name="pnp-cub-joint" \
    --partition="${PARTITION}" \
    --account="${ACCOUNT}" \
    --gres=gpu:1 \
    --cpus-per-task=8 \
    --mem=64G \
    --time=1-00:00:00 \
    --output="${LOG_SLURM}/pnp_cub_joint_%j.out" \
    --error="${LOG_SLURM}/pnp_cub_joint_%j.err" \
    --wrap="
set -e
source ${SCRATCH}/venv/bin/activate
export HF_HOME=${SCRATCH}/hf_cache
export TRANSFORMERS_CACHE=${SCRATCH}/hf_cache
export HF_HUB_CACHE=${SCRATCH}/.cache/huggingface/hub
export TORCH_HOME=${SCRATCH}/torch_cache
export PYTHONPATH=${SCRATCH}/dinov2:\$PYTHONPATH
cd ${REPO}

python scripts/train_cub_joint.py \
  --init-ckpt ${INIT_CKPT} \
  --vocab-cache-path ${FILTERED_VOCAB_CACHE} \
  --clip-scores-cub ${SCORES_OUT} \
  --cub-root ${CUB_ROOT} \
  --cub-annotations ${CUB_ANNOTATIONS} \
  --cls-coef ${CLS_COEF} \
  --sufficiency-coef ${SUFFICIENCY_COEF} \
  --epochs 15 \
  --log-dir ${LOG_DIR}
")
echo "Submitted (Joint CUB training, cls_coef=${CLS_COEF} sufficiency_coef=${SUFFICIENCY_COEF}): ${JOB}"

echo ""
echo "Monitor with : squeue -u \$USER"
echo "Checkpoint   : ${LOG_DIR}/ckpt.pth"
echo "Result       : ${LOG_DIR}/result.json"
echo ""
echo "Compare against Sequential baseline: 67.41% top-1 / 93.06% top-5"
echo "(eval_results/cub_concepts_stage2/sparse_probe_result.json)"
