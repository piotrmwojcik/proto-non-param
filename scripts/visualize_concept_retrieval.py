#!/usr/bin/env python3
"""
Qualitative "concept retrieval" figure: for a handful of vocabulary concepts,
retrieve and display the top-k most-activating image regions from a corpus,
based on similarity to that concept's learned prototype vector.

Corpus: RefCOCOg images (deduplicated — a referring-expression dataset has
many expressions per image, retrieval needs unique images), reusing
ReferDataset the same way evaluate_pnp_refer.py does.

Model loading (build_model) and per-patch scoring are the same mechanism
evaluate_pnp_refer.py/PNP.forward() already use — this script is eval-only,
no training. Reuses find_high_activation_crop/draw_rect_on_image/
denorm_to_uint8/overlay_heatmap from eval_retreive_concepts.py as-is.

Vocabulary caveat: the checkpoint's vocab (vocab_cache_path, VG-derived) is
dominated by concrete objects/attributes — no relational phrases (those live
only in VG's separate caption-embedding pool, a different representation).
Concept suggestion below is limited to nouns/adjectives accordingly.

Usage:
  python scripts/visualize_concept_retrieval.py \
    --ckpt $SCRATCH/train_logs/vg_contrastive/run_M1_vitl14_sk10_koleo01_30ep/ckpt.pth \
    --data-root $SCRATCH/data/refcoco --dataset Gref --split val \
    --img-size 672 --out-dir results/concept_retrieval
"""

import argparse
import math
import os
import random
import sys

import nltk
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, SCRIPT_DIR)

from evaluation.refer_dataset import ReferDataset  # noqa: E402
from evaluate_pnp_refer import build_model, build_img_transform  # noqa: E402
from eval_retreive_concepts import (  # noqa: E402
    find_high_activation_crop, draw_rect_on_image, overlay_heatmap,
)


def dedup_images(data_root, dataset, split):
    """One entry per unique img_id from ReferDataset (which is one row per
    referring expression, i.e. many rows per image)."""
    ds = ReferDataset(root=os.path.join(data_root, dataset), splitset=split)
    seen = set()
    images = []
    for i in range(len(ds)):
        _, img_id, pil_img, _, _ = ds.get_raw_item(i)
        if img_id in seen:
            continue
        seen.add(img_id)
        images.append((img_id, pil_img))
    return images


def suggest_concepts(vocab_words, n_nouns=5, n_adjs=3, seed=0):
    """Classify vocab words by single-word POS tag, sample a noun/adjective mix.
    Relational phrases aren't in this vocab (see module docstring) — objects and
    attributes only."""
    tagged = nltk.pos_tag(vocab_words)
    nouns = [w for w, pos in tagged if pos.startswith("NN") and w.isalpha()]
    adjs = [w for w, pos in tagged if pos.startswith("JJ") and w.isalpha()]

    rng = random.Random(seed)
    picked_nouns = rng.sample(nouns, min(n_nouns, len(nouns)))
    picked_adjs = rng.sample(adjs, min(n_adjs, len(adjs)))
    return picked_nouns + picked_adjs


@torch.inference_mode()
def collect_patch_logits(net, images, img_transform, device, batch_size=16):
    """Forward pass over the deduped image corpus. Returns patch_prototype_logits
    [M, N, V] (cpu) and the list of PIL images in the same order (kept for crops)."""
    all_logits = []
    pil_images = [img for _, img in images]

    for start in tqdm(range(0, len(pil_images), batch_size), desc="Corpus forward pass"):
        batch = pil_images[start:start + batch_size]
        img_t = torch.stack([img_transform(im) for im in batch]).to(device)
        outputs = net(img_t)
        all_logits.append(outputs["patch_prototype_logits"].detach().cpu())

    return torch.cat(all_logits, dim=0), pil_images


def make_figure(concepts, concept_word_to_score_and_patch, pil_images, topk, out_path):
    n_concepts = len(concepts)
    fig, axes = plt.subplots(n_concepts, topk, figsize=(3 * topk, 3 * n_concepts), dpi=140)
    if n_concepts == 1:
        axes = axes[None, :]

    for row, concept in enumerate(concepts):
        topk_values, topk_indices, patch_logits_col = concept_word_to_score_and_patch[concept]
        for col, (score, global_idx) in enumerate(zip(topk_values.tolist(), topk_indices.tolist())):
            ax = axes[row, col]
            img_uint8 = _display_uint8(pil_images[global_idx])

            hm = patch_logits_col[global_idx]  # [N]
            N = hm.shape[0]
            H = W = int(math.sqrt(N))
            hm_up = F.interpolate(hm.view(1, 1, H, W), size=img_uint8.shape[:2],
                                   mode="bilinear", align_corners=False)[0, 0]

            bbox = find_high_activation_crop(hm_up.numpy(), percentile=95)
            overlay = overlay_heatmap(img_uint8, hm_up, alpha=0.45)
            overlay_box = draw_rect_on_image(overlay, bbox)

            ax.imshow(overlay_box)
            ax.axis("off")
            ax.set_title(f"score={score:.3f}", fontsize=9)
        # ax.axis("off") hides a normal ylabel, so annotate with text instead
        axes[row, 0].text(-0.1, 0.5, concept, transform=axes[row, 0].transAxes,
                          rotation=90, va="center", ha="center", fontsize=12)

    fig.suptitle("Concept retrieval: top-activating regions per vocabulary word", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path)
    plt.close(fig)


def _display_uint8(pil_img):
    """Plain RGB uint8 array for display — no CLIP normalization involved,
    so no denorm needed (unlike the model's input tensor)."""
    return np.array(pil_img.convert("RGB"))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--data-root", required=True, help="e.g. $SCRATCH/data/refcoco")
    p.add_argument("--dataset", default="Gref")
    p.add_argument("--split", default="val")
    p.add_argument("--img-size", type=int, default=224,
                   help="Must stay 224 (CLIP ViT-B/32's fixed positional-embedding size) "
                        "unless PNP.forward() is changed: forward() unconditionally also "
                        "runs x through self.clip_model.encode_image() for a diagnostic "
                        "side-output (clip_vocab_logits), and that encoder — unlike the "
                        "DINOv2 backbone — doesn't support other resolutions.")
    p.add_argument("--concepts", type=str, nargs="+", default=None,
                   help="Override automatic concept suggestion with explicit vocab words")
    p.add_argument("--n-nouns", type=int, default=5)
    p.add_argument("--n-adjs", type=int, default=3)
    p.add_argument("--topk", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", required=True)
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    print(f"Loading PNP model from {args.ckpt} ...")
    net, _, hparams = build_model(args.ckpt, device)
    assert args.img_size % 14 == 0, "--img-size must be a multiple of the ViT patch size (14)"
    img_transform = build_img_transform(args.img_size)

    vocab_words = net.vocab_words
    vocab_to_idx = {w: i for i, w in enumerate(vocab_words)}
    print(f"Checkpoint vocab: {hparams.vocab_cache_path} ({len(vocab_words)} words)")

    if args.concepts:
        missing = [c for c in args.concepts if c not in vocab_to_idx]
        if missing:
            raise ValueError(f"Requested concepts not in checkpoint vocab: {missing}")
        concepts = args.concepts
    else:
        concepts = suggest_concepts(vocab_words, args.n_nouns, args.n_adjs, args.seed)
        print(f"Auto-suggested concepts (relational phrases not available in this vocab): {concepts}")

    print(f"Building deduplicated {args.dataset}/{args.split} image corpus ...")
    images = dedup_images(args.data_root, args.dataset, args.split)
    print(f"  {len(images)} unique images")

    patch_logits, pil_images = collect_patch_logits(net, images, img_transform, device, args.batch_size)

    concept_indices = [vocab_to_idx[c] for c in concepts]

    # mixture_weights-equivalent per-image score: max patch activation per concept
    # (simplest, standard "does this concept fire anywhere in the image" ranking).
    concept_word_to_score_and_patch = {}
    for concept, col in zip(concepts, concept_indices):
        col_logits = patch_logits[:, :, col]           # [M, N]
        img_scores = col_logits.max(dim=1).values       # [M]
        k = min(args.topk, img_scores.shape[0])
        topk_values, topk_indices = torch.topk(img_scores, k=k)
        concept_word_to_score_and_patch[concept] = (topk_values, topk_indices, col_logits)

    out_path = os.path.join(args.out_dir, f"concept_retrieval_{args.dataset}_{args.split}.png")
    make_figure(concepts, concept_word_to_score_and_patch, pil_images, args.topk, out_path)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
