#!/usr/bin/env python3
"""
Deletion faithfulness test for PNP's zero-shot referring segmentation
(inference-only, no training).

For each (image, referring expression) test example:
  1. Compute the baseline per-patch cosine-similarity map (same mechanism as
     evaluate_pnp_refer.py) and its IoU against ground truth.
  2. DELETE the top-`--delete-frac` fraction of patches by raw activation
     score (zero their patch-token vectors -- cheap: one backbone forward
     pass per image, masking happens post-backbone, no re-encoding needed),
     recompute the map and IoU: "top-k deleted".
  3. DELETE the same number of RANDOMLY chosen patches instead: "random-k
     deleted" -- the control.

If PNP's activation map is faithful (the highlighted patches are actually
what the prediction depends on, not just decorative), deleting the top-k
patches should hurt IoU much more than deleting a random k. The gap
(drop_topk - drop_random), aggregated with a paired bootstrap CI, is the
quantitative faithfulness metric.

Token-level masking (zeroing patch_tokens post-backbone) is a standard,
cheap simplification of "the image never had this content" -- it does NOT
re-run the backbone on a pixel-blanked image (self-attention already mixed
information across patches during the single forward pass), which would be
more faithful but far more expensive. This tests whether the FINAL
similarity/thresholding step depends on these specific patch representations,
which is the thing "faithful" actually needs to be true about.

Usage:
  python scripts/eval_deletion_faithfulness.py \
    --ckpt $SCRATCH/train_logs/vg_contrastive/run_M1_vitl14_sk10_koleo01_30ep/ckpt.pth \
    --dataset Gref --data_split val --data_root $SCRATCH/data/refcoco \
    --img-size 672 --n-samples 300 --out-dir results/deletion_faithfulness
"""

import argparse
import csv
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, SCRIPT_DIR)

from evaluate_pnp_refer import build_model, build_img_transform, mask_IU  # noqa: E402
from evaluation.refer_dataset import ReferDataset  # noqa: E402


def iou_from_patch_tokens(patch_tokens, p_query, gt_mask, patch_grid, threshold):
    """Same scoring/normalize/threshold pipeline as evaluate_pnp_refer.py's
    eval loop, factored out so it can be re-run cheaply on masked variants
    of the same patch_tokens tensor."""
    scores = (patch_tokens * p_query.unsqueeze(1)).sum(dim=-1)  # [1, N]
    H_gt, W_gt = gt_mask.shape[:2]
    spatial = scores.view(1, 1, patch_grid, patch_grid)
    spatial = F.interpolate(spatial, size=(H_gt, W_gt), mode="bilinear", align_corners=False)
    activation = spatial.squeeze().cpu().numpy()
    a_min, a_max = activation.min(), activation.max()
    if a_max > a_min:
        activation = (activation - a_min) / (a_max - a_min + 1e-8)
    pred_mask = (activation >= threshold).astype(np.uint8)
    gt_bin = (gt_mask > 0).astype(np.uint8)
    I, U = mask_IU(pred_mask, gt_bin)
    return float(I) / (float(U) + 1e-8) if U > 0 else 0.0


def bootstrap_ci(values, n_bootstrap=2000, seed=0):
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    n = len(values)
    means = np.array([values[rng.integers(0, n, size=n)].mean() for _ in range(n_bootstrap)])
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(values.mean()), float(lo), float(hi)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--dataset", default="Gref")
    p.add_argument("--data_split", default="val")
    p.add_argument("--data_root", required=True)
    p.add_argument("--img-size", type=int, default=672)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--delete-frac", type=float, default=0.2,
                   help="Fraction of patches to delete (top-k by raw score, or random)")
    p.add_argument("--n-samples", type=int, default=300,
                   help="Randomly subsample this many test examples (keeps cost low)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-bootstrap", type=int, default=2000)
    p.add_argument("--device", default="cuda")
    p.add_argument("--out-dir", required=True)
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    print(f"Loading PNP model from {args.ckpt} ...")
    net, tokenizer, hparams = build_model(args.ckpt, device)
    assert args.img_size % 14 == 0, "--img-size must be a multiple of the ViT patch size (14)"
    patch_grid = args.img_size // 14
    n_patches = patch_grid * patch_grid
    n_delete = max(1, round(args.delete_frac * n_patches))
    img_transform = build_img_transform(args.img_size)

    data_root = os.path.join(args.data_root, args.dataset)
    ds = ReferDataset(root=data_root, splitset=args.data_split)
    print(f"Dataset: {args.dataset}/{args.data_split}  ({len(ds)} samples)")

    rng = np.random.default_rng(args.seed)
    n_samples = min(args.n_samples, len(ds))
    sample_indices = rng.choice(len(ds), size=n_samples, replace=False)
    print(f"Sampling {n_samples} examples (delete_frac={args.delete_frac} -> "
          f"{n_delete}/{n_patches} patches per image)")

    rows = []
    with torch.inference_mode():
        for idx in sample_indices:
            sentence, img_id, pil_img, gt_mask, _ = ds.get_raw_item(int(idx))

            img_t = img_transform(pil_img).unsqueeze(0).to(device)
            tokens = tokenizer([str(sentence)]).to(device)
            e_query = net.clip_model.encode_text(tokens)
            e_query = F.normalize(e_query.float(), dim=-1)
            p_query = net.text_projection_head(e_query)  # [1, D]

            patch_tokens = net.backbone(img_t)[0]
            patch_tokens = F.normalize(patch_tokens, dim=-1)  # [1, N, D]

            baseline_iou = iou_from_patch_tokens(patch_tokens, p_query, gt_mask, patch_grid, args.threshold)

            # Raw per-patch scores (pre-upsample) decide the top-k deletion set --
            # a fixed fraction of patches, independent of the segmentation
            # threshold itself, avoiding circularity with what defines the mask.
            raw_scores = (patch_tokens * p_query.unsqueeze(1)).sum(dim=-1)[0]  # [N]
            topk_idx = torch.topk(raw_scores, k=n_delete).indices
            rand_idx = torch.from_numpy(
                rng.choice(patch_tokens.shape[1], size=n_delete, replace=False)
            ).to(device)

            patch_tokens_topk_deleted = patch_tokens.clone()
            patch_tokens_topk_deleted[0, topk_idx] = 0.0
            topk_iou = iou_from_patch_tokens(patch_tokens_topk_deleted, p_query, gt_mask, patch_grid, args.threshold)

            patch_tokens_rand_deleted = patch_tokens.clone()
            patch_tokens_rand_deleted[0, rand_idx] = 0.0
            rand_iou = iou_from_patch_tokens(patch_tokens_rand_deleted, p_query, gt_mask, patch_grid, args.threshold)

            rows.append({
                "index": int(idx), "img_id": img_id, "sentence": str(sentence),
                "baseline_iou": round(baseline_iou, 6),
                "topk_deleted_iou": round(topk_iou, 6),
                "random_deleted_iou": round(rand_iou, 6),
                "drop_topk": round(baseline_iou - topk_iou, 6),
                "drop_random": round(baseline_iou - rand_iou, 6),
            })

    per_example_path = os.path.join(args.out_dir, "per_example.csv")
    with open(per_example_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"Saved {per_example_path}")

    baseline_mean, baseline_lo, baseline_hi = bootstrap_ci([r["baseline_iou"] for r in rows], args.n_bootstrap, args.seed)
    topk_mean, topk_lo, topk_hi = bootstrap_ci([r["topk_deleted_iou"] for r in rows], args.n_bootstrap, args.seed)
    rand_mean, rand_lo, rand_hi = bootstrap_ci([r["random_deleted_iou"] for r in rows], args.n_bootstrap, args.seed)
    # Paired: faithfulness gap = drop_topk - drop_random, per example, bootstrapped
    # over examples so the pairing (same image, same random draw) is preserved.
    gap_values = [r["drop_topk"] - r["drop_random"] for r in rows]
    gap_mean, gap_lo, gap_hi = bootstrap_ci(gap_values, args.n_bootstrap, args.seed)

    summary = {
        "ckpt": args.ckpt, "dataset": args.dataset, "split": args.data_split,
        "n_samples": n_samples, "delete_frac": args.delete_frac, "n_delete_patches": n_delete,
        "n_total_patches": n_patches, "threshold": args.threshold,
        "baseline_iou": {"mean": round(100 * baseline_mean, 4), "ci_lo": round(100 * baseline_lo, 4), "ci_hi": round(100 * baseline_hi, 4)},
        "topk_deleted_iou": {"mean": round(100 * topk_mean, 4), "ci_lo": round(100 * topk_lo, 4), "ci_hi": round(100 * topk_hi, 4)},
        "random_deleted_iou": {"mean": round(100 * rand_mean, 4), "ci_lo": round(100 * rand_lo, 4), "ci_hi": round(100 * rand_hi, 4)},
        "faithfulness_gap": {
            "mean": round(100 * gap_mean, 4), "ci_lo": round(100 * gap_lo, 4), "ci_hi": round(100 * gap_hi, 4),
            "interpretation": ("positive and CI excludes 0 => deleting top-activating patches hurts IoU "
                               "significantly more than deleting random patches (faithful); "
                               "CI includes 0 => no evidence the activation map is more than randomly informative"),
        },
    }
    summary_path = os.path.join(args.out_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
    print(f"Saved {summary_path}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5, 5), dpi=140)
    labels = ["Baseline", "Top-k deleted", "Random-k deleted"]
    means = [baseline_mean * 100, topk_mean * 100, rand_mean * 100]
    los = [baseline_mean * 100 - baseline_lo * 100, topk_mean * 100 - topk_lo * 100, rand_mean * 100 - rand_lo * 100]
    his = [baseline_hi * 100 - baseline_mean * 100, topk_hi * 100 - topk_mean * 100, rand_hi * 100 - rand_mean * 100]
    ax.bar(labels, means, yerr=[los, his], capsize=5, color=["#1f77b4", "#d62728", "#7f7f7f"])
    ax.set_ylabel("mIoU (%)")
    ax.set_title(f"Deletion faithfulness ({args.dataset}/{args.data_split}, "
                 f"delete top/random {int(100*args.delete_frac)}% of patches)")
    fig.tight_layout()
    fig_path = os.path.join(args.out_dir, "deletion_faithfulness.png")
    fig.savefig(fig_path)
    print(f"Saved {fig_path}")


if __name__ == "__main__":
    main()
