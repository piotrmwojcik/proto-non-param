#!/usr/bin/env python3
"""
Stage 2's third Label-free-CBM step: after fine-tuning, drop concepts whose
*learned* representation no longer tracks CLIP's own judgment well (their
"interpretability cutoff", default 0.45 on their cosine-cubed similarity).

Reuses evaluate_pnp_cub_concepts.py's extraction functions directly (no
duplication) to get the fine-tuned model's own per-image concept-activation
vectors on CUB val, then compares each concept column against the cached
CLIP ground-truth scores from the same build_clip_vocab_scores.py run used
for training, via per-concept Pearson correlation across the val images.

Note: Pearson correlation is a standard, interpretable substitute for their
custom cosine-cubed similarity metric (which is tuned for projection-training
gradient behavior, not needed for a one-off post-hoc filter) -- their 0.45
threshold was calibrated for their own metric's range, not Pearson's [-1, 1],
so this script prints the full correlation distribution; treat --cutoff as a
starting point to sanity-check against those printed stats, not a fixed law.

Usage:
  python scripts/filter_cub_concepts_interpretability.py \
    --ckpt $SCRATCH/train_logs/cub_labelfreecbm/ckpt.pth \
    --concepts-file eval_results/cub_concepts_stage2/concepts_clip_filtered.txt \
    --clip-scores-file $SCRATCH/vocab/cub_clip_scores.pt \
    --cub-root $SCRATCH/cub200 \
    --cub-annotations $SCRATCH/cub200/annotations \
    --out-concepts-file eval_results/cub_concepts_stage2/concepts_interp_filtered.txt
"""

import argparse
import os
import sys

import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, SCRIPT_DIR)

from evaluate_pnp_refer import build_model, build_img_transform          # noqa: E402
from evaluate_pnp_cub_concepts import (build_concept_prototypes,          # noqa: E402
                                        encode_split, list_split, build_class_index)
from clip_dataset import build_cub_path_to_id                            # noqa: E402


def main():
    p = argparse.ArgumentParser(description="Label-free-CBM-style interpretability-cutoff concept filter")
    p.add_argument("--ckpt", required=True, help="Fine-tuned PNP checkpoint")
    p.add_argument("--concepts-file", required=True, help="CLIP-cutoff-filtered concept list (from build_clip_vocab_scores.py)")
    p.add_argument("--clip-scores-file", required=True, help="Filtered CLIP scores .pt (same run as --concepts-file)")
    p.add_argument("--cub-root", required=True)
    p.add_argument("--cub-annotations", required=True)
    p.add_argument("--img-size", type=int, default=672)
    p.add_argument("--interpretability-cutoff", type=float, default=0.45)
    p.add_argument("--out-concepts-file", required=True)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    args = p.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    print(f"Loading fine-tuned PNP model from {args.ckpt} ...")
    net, tokenizer, hparams = build_model(args.ckpt, device)

    with open(args.concepts_file) as f:
        concepts = [line.strip() for line in f if line.strip()]
    print(f"Loaded {len(concepts)} concepts from {args.concepts_file}")

    concept_prototypes = build_concept_prototypes(net, tokenizer, concepts, device)

    class_to_idx = build_class_index(args.cub_root)
    img_transform = build_img_transform(args.img_size)
    val_samples = list_split(args.cub_root, "val", class_to_idx)
    print(f"CUB val: {len(val_samples)} images")

    pnp_activations, _ = encode_split(net, img_transform, val_samples, concept_prototypes,
                                       device, args.batch_size)  # [N_val, C]

    # Match val images -> CUB image_id -> row in the cached CLIP scores file.
    path_to_id = build_cub_path_to_id(args.cub_root, args.cub_annotations)
    val_ids = [path_to_id[p] for p, _ in val_samples]

    scores_data = torch.load(args.clip_scores_file, map_location="cpu")
    scores_vocab = scores_data["vocab"]
    assert scores_vocab == concepts, (
        "--clip-scores-file's vocab order must match --concepts-file exactly "
        "(both should come from the same build_clip_vocab_scores.py run)."
    )
    id_to_row = {img_id: i for i, img_id in enumerate(scores_data["image_ids"])}
    row_idx = [id_to_row[img_id] for img_id in val_ids if img_id in id_to_row]
    matched_val_idx = [i for i, img_id in enumerate(val_ids) if img_id in id_to_row]
    print(f"Matched {len(row_idx)}/{len(val_samples)} val images to cached CLIP scores")

    clip_scores_val = scores_data["clip_scores"][row_idx].float()          # [N_matched, C]
    pnp_activations_val = pnp_activations[matched_val_idx]                  # [N_matched, C]

    # Per-concept Pearson correlation across the matched val images.
    x = pnp_activations_val - pnp_activations_val.mean(dim=0, keepdim=True)
    y = clip_scores_val - clip_scores_val.mean(dim=0, keepdim=True)
    numer = (x * y).sum(dim=0)
    denom = (x.norm(dim=0) * y.norm(dim=0)).clamp(min=1e-8)
    correlation = numer / denom                                            # [C]

    print(f"\nPer-concept correlation stats: "
          f"min={correlation.min():.3f} p25={correlation.quantile(0.25):.3f} "
          f"median={correlation.median():.3f} p75={correlation.quantile(0.75):.3f} "
          f"max={correlation.max():.3f}")

    keep_mask = correlation > args.interpretability_cutoff
    kept = [c for c, keep in zip(concepts, keep_mask.tolist()) if keep]
    print(f"Interpretability cutoff {args.interpretability_cutoff}: "
          f"keeping {len(kept)}/{len(concepts)} concepts")

    if not kept:
        raise RuntimeError(
            f"--interpretability-cutoff {args.interpretability_cutoff} filtered out all "
            f"{len(concepts)} concepts -- check the printed correlation distribution above "
            f"and lower the cutoff."
        )

    os.makedirs(os.path.dirname(args.out_concepts_file) or ".", exist_ok=True)
    with open(args.out_concepts_file, "w", encoding="utf-8") as f:
        f.write("\n".join(kept) + "\n")
    print(f"Saved final concept list -> {args.out_concepts_file}")


if __name__ == "__main__":
    main()
