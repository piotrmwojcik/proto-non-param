#!/bin/bash
# Zero-shot RIS evaluation for the dedup-vocab ablation checkpoints (A and C).
#
# Submits 14 SLURM jobs: 2 variants (A/C) × 7 dataset/split combos
#   Gref : val
#   unc  : val, testA, testB
#   unc+ : val, testA, testB
#
# Results land in:
#   eval_results/vg_dedup/dedup_A/pnp_refer/{dataset}_{split}.json   (KL  + frozen, dedup vocab)
#   eval_results/vg_dedup/dedup_C/pnp_refer/{dataset}_{split}.json   (JSD + frozen, dedup vocab)
#
# Compare against the non-dedup ablation runs with:
#   python scripts/compare_ris_results.py \
#       --eval_dir eval_results \
#       --ablation-dir eval_results/vg_dedup \
#       --ablation-type vg_dedup \
#       --out eval_results/vg_dedup/comparison.md
#
# NOTE: This script runs the standard RIS eval (evaluate_pnp_refer.py) which
# uses the training vocab from the checkpoint's hparams.vocab_cache_path.
# For the cross-vocab generalisation eval (dedup→full vocab, P@K/R@K), use
# scripts/slurm_eval_vg_dedup.sh instead.
#
# Configuration — override via environment variables:
#   DEDUP_BASE   Base dir with run_A_kl_frozen_dedup_t090 / run_C_jsd_frozen_dedup_t090
#   DATA_ROOT    Path to refcoco/ directory
#   THRESHOLD    Threshold string used in training (default: 090)
#   EP_SUFFIX    Optional epoch suffix appended to checkpoint dirs, e.g. "_40ep"
#                (default: empty → uses the original 20-epoch dirs)
#                Use EP_SUFFIX=_40ep for the longer PNP-A-dedup run.
#
# Usage:
#   bash scripts/slurm_eval_pnp_vg_dedup.sh                      # 20ep run
#   EP_SUFFIX=_40ep bash scripts/slurm_eval_pnp_vg_dedup.sh      # 40ep run (A only)

set -e

SCRATCH="/net/tscratch/people/plgabedychaj"
REPO=~/proto-non-param
DEDUP_BASE="${DEDUP_BASE:-${SCRATCH}/train_logs/vg_dedup}"
DATA_ROOT="${DATA_ROOT:-${SCRATCH}/data/refcoco}"
THRESHOLD="${THRESHOLD:-090}"
EP_SUFFIX="${EP_SUFFIX:-}"

PARTITION="plgrid-gpu-a100"
ACCOUNT="plgunhype-gpu-a100"
LOG_SLURM="${SCRATCH}/logs"

mkdir -p "${LOG_SLURM}"

declare -A CKPTS=(
  [A]="${DEDUP_BASE}/run_A_kl_frozen_dedup_t${THRESHOLD}${EP_SUFFIX}/ckpt.pth"
  [C]="${DEDUP_BASE}/run_C_jsd_frozen_dedup_t${THRESHOLD}${EP_SUFFIX}/ckpt.pth"
)

declare -A LABELS=(
  [A]="KL + frozen residual (dedup vocab θ=0.${THRESHOLD}${EP_SUFFIX})"
  [C]="JSD + frozen residual (dedup vocab θ=0.${THRESHOLD}${EP_SUFFIX})"
)

declare -A SPLITS=(
  [Gref]="val"
  [unc]="val testA testB"
  [unc+]="val testA testB"
)

# When EP_SUFFIX is set, only A has the long checkpoint; skip C if it doesn't exist.
if [ -n "${EP_SUFFIX}" ] && [ ! -f "${CKPTS[C]}" ]; then
    VARIANTS="A"
    echo "Note: C checkpoint not found for suffix '${EP_SUFFIX}' — evaluating A only."
else
    VARIANTS="A C"
fi

echo "=== PNP Dedup-Vocab Ablation — Zero-shot RIS Evaluation (θ=0.${THRESHOLD}${EP_SUFFIX}) ==="
echo "  Ckpt A : ${CKPTS[A]}"
echo "  Ckpt C : ${CKPTS[C]}"
echo "  Data   : ${DATA_ROOT}"
echo "  Results: eval_results/vg_dedup/dedup_{A,C}${EP_SUFFIX}/pnp_refer/"
echo ""

for VARIANT in ${VARIANTS}; do
  CKPT="${CKPTS[$VARIANT]}"
  OUT_DIR="${REPO}/eval_results/vg_dedup/dedup_${VARIANT}${EP_SUFFIX}"
  echo "-- Variant ${VARIANT} (${LABELS[$VARIANT]}) --"

  for DATASET in Gref unc unc+; do
    for SPLIT in ${SPLITS[$DATASET]}; do
      JOB=$(sbatch --parsable \
        --job-name="pnp-ded${VARIANT}${EP_SUFFIX}-${DATASET}-${SPLIT}" \
        --partition="${PARTITION}" \
        --account="${ACCOUNT}" \
        --gres=gpu:1 \
        --cpus-per-task=4 \
        --mem=32G \
        --time=04:00:00 \
        --output="${LOG_SLURM}/pnp_dedup_${VARIANT}${EP_SUFFIX}_${DATASET}_${SPLIT}_%j.out" \
        --error="${LOG_SLURM}/pnp_dedup_${VARIANT}${EP_SUFFIX}_${DATASET}_${SPLIT}_%j.err" \
        --wrap="
set -e
source ${SCRATCH}/venv/bin/activate
export HF_HUB_CACHE=${SCRATCH}/.cache/huggingface/hub
export TRANSFORMERS_CACHE=${SCRATCH}/.cache/huggingface/hub
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
  echo ""
done

echo "Jobs submitted. Monitor with: squeue -u \$USER"
echo ""
echo "After completion, generate comparison table with:"
echo "  python scripts/compare_ris_results.py \\"
echo "      --eval_dir eval_results \\"
echo "      --ablation-dir eval_results/vg_dedup \\"
echo "      --ablation-type vg_dedup \\"
echo "      --out eval_results/vg_dedup/comparison.md"
