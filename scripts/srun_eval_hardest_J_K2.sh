#!/bin/bash
# Interactive (srun) evaluation of J and K2 with hardest-sample image dump.
#
# Runs 14 sequential srun jobs (7 per model). Each blocks until done.
# Estimated wall time: ~6-8 h total on A100.
#
# Usage (from Athena login node):
#   bash scripts/srun_eval_hardest_J_K2.sh
#
# To dump more/fewer samples:
#   DUMP_N=20 bash scripts/srun_eval_hardest_J_K2.sh
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

run_eval() {
    local model="$1" ckpt="$2" dataset="$3" split="$4" out_dir="$5"
    echo "  [${model}] ${dataset}/${split}"
    srun \
        --job-name="pnp-${model}-${dataset}-${split}" \
        --partition="${PARTITION}" \
        --account="${ACCOUNT}" \
        --gres=gpu:1 \
        --ntasks=1 \
        --cpus-per-task=4 \
        --mem=32G \
        --time=04:00:00 \
        bash -c "
source '${SCRATCH}/venv/bin/activate'
export HF_HUB_CACHE='${SCRATCH}/.cache/huggingface/hub'
export TRANSFORMERS_CACHE='${SCRATCH}/.cache/huggingface/hub'
export TORCH_HOME='${SCRATCH}/torch_cache'
export PYTHONPATH='${SCRATCH}/dinov2:\$PYTHONPATH'
cd '${REPO}'
python scripts/evaluate_pnp_refer.py \
  --ckpt '${ckpt}' \
  --dataset '${dataset}' \
  --data_split '${split}' \
  --data_root '${DATA_ROOT}' \
  --out_dir '${out_dir}' \
  --dump-hardest '${DUMP_N}'
"
}

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

    run_eval "${MODEL}" "${CKPT}" Gref  val    "${OUT_DIR}"
    run_eval "${MODEL}" "${CKPT}" unc   val    "${OUT_DIR}"
    run_eval "${MODEL}" "${CKPT}" unc   testA  "${OUT_DIR}"
    run_eval "${MODEL}" "${CKPT}" unc   testB  "${OUT_DIR}"
    run_eval "${MODEL}" "${CKPT}" unc+  val    "${OUT_DIR}"
    run_eval "${MODEL}" "${CKPT}" unc+  testA  "${OUT_DIR}"
    run_eval "${MODEL}" "${CKPT}" unc+  testB  "${OUT_DIR}"
done

echo ""
echo "Done. Results + hardest dumps in:"
echo "  ${REPO}/eval_results/vg_contrastive/contr_J/pnp_refer/"
echo "  ${REPO}/eval_results/vg_contrastive/contr_K2/pnp_refer/"
