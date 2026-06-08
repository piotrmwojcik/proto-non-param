#!/bin/bash
#SBATCH --job-name=vis-captions
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --account=plgunhype-gpu-a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --output=/net/tscratch/people/plgabedychaj/logs/vis_captions_%j.out
#SBATCH --error=/net/tscratch/people/plgabedychaj/logs/vis_captions_%j.err

set -e
mkdir -p /net/tscratch/people/plgabedychaj/logs

export HF_HOME=/net/tscratch/people/plgabedychaj/hf_cache
export PYTHONPATH=/net/tscratch/people/plgabedychaj/dinov2:$PYTHONPATH
source /net/tscratch/people/plgabedychaj/venv/bin/activate
cd ~/proto-VLM/proto-non-param

SCRATCH=/net/tscratch/people/plgabedychaj
CAP_CACHE="${SCRATCH}/vocab/vg_test_caption_prototypes_min5w.pt"

# Step 1 — rebuild caption cache keeping only phrases with ≥5 words
# (removes short single-object labels like "a desk", "sky", "a cup")
echo "=== Rebuilding caption prototype cache (min-words=5) ==="
python vocab/build_caption_prototypes.py \
  --source vg_test \
  --vg-root "${SCRATCH}/vg" \
  --region-descriptions "${SCRATCH}/vg/region_descriptions.json" \
  --vocab-cache-path "${SCRATCH}/vocab/vg_cache.pt" \
  --cache-out "${CAP_CACHE}" \
  --clip-model-name ViT-B-32 \
  --clip-pretrained openai \
  --seed 42 \
  --val-ratio 0.1 \
  --min-words 5

echo "=== Caption cache ready: ${CAP_CACHE} ==="

# Step 2 — visualise per-image caption activations
python visualize_caption_activation.py \
  --ckpt "${SCRATCH}/train_logs/vg_baseline/ckpt.pth" \
  --vocab-cache-path "${SCRATCH}/vocab/vg_cache.pt" \
  --caption-prototypes-path "${CAP_CACHE}" \
  --source-dataset vg_test \
  --vg-root "${SCRATCH}/vg" \
  --vg-region-descriptions "${SCRATCH}/vg/region_descriptions.json" \
  --n-random 30 \
  --n-captions 4 \
  --out-dir "${SCRATCH}/caption_vis_min5w" \
  --wandb-entity gmum \
  --wandb-run-name caption-activation-vis-min5w
