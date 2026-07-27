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
    """One entry per unique physical image from ReferDataset (which is one row
    per referring expression, i.e. many rows per image).

    Dedup key is im_name, NOT img_id: ReferDataset.get_raw_item's "img_id" is
    parsed from the .npz filename's trailing number (build_batches.py's
    `n_batch`, a global counter incremented once per referring expression,
    unrelated to image identity) -- so it's actually unique per SENTENCE, not
    per image, making a dedup keyed on it a no-op. im_name (COCO_{split}_
    {image_id}, build_batches.py:72) is the real, stable per-image identifier
    -- same value for every sentence about the same photo."""
    ds = ReferDataset(root=os.path.join(data_root, dataset), splitset=split)
    seen = set()
    images = []
    for i in range(len(ds)):
        _, _, pil_img, _, im_name = ds.get_raw_item(i)
        if im_name in seen:
            continue
        seen.add(im_name)
        images.append((im_name, pil_img))
    return images


def suggest_concepts(vocab_words, n_nouns=5, n_adjs=3, seed=0):
    """Classify vocab words by single-word POS tag, sample a noun/adjective mix.
    Relational phrases aren't in this vocab (see module docstring) — objects and
    attributes only.

    The vocab (15k+ words, auto-extracted from raw VG captions via NLTK, no
    manual curation) is Zipfian: a handful of good common words, a long tail
    of typos/rare tokens/extraction artifacts ("jerysey", "mountial",
    "withred", ...). Sampling uniformly over every POS-tagged noun/adjective
    mostly hits that tail. Filtering candidates through NLTK's own English
    word list first (nltk.corpus.words) keeps the sample to real, recognizable
    words — cheap, no new dependency (NLTK is already required throughout
    this codebase for POS tagging/lemmatization)."""
    nltk.download("words", quiet=True)
    from nltk.corpus import words as nltk_words
    real_words = set(nltk_words.words())

    tagged = nltk.pos_tag(vocab_words)
    nouns = [w for w, pos in tagged if pos.startswith("NN") and w.isalpha() and w in real_words]
    adjs = [w for w, pos in tagged if pos.startswith("JJ") and w.isalpha() and w in real_words]

    rng = random.Random(seed)
    picked_nouns = rng.sample(nouns, min(n_nouns, len(nouns)))
    picked_adjs = rng.sample(adjs, min(n_adjs, len(adjs)))
    return picked_nouns + picked_adjs


@torch.inference_mode()
def collect_patch_logits(net, images, img_transform, device, concept_indices, batch_size=16):
    """Forward pass over the deduped image corpus. Returns patch_prototype_logits
    sliced to just concept_indices, [M, N, C] (cpu), and the PIL images in the
    same order (kept for crops). Slicing to C columns (a handful of concepts)
    before accumulating is essential — the full [M, N, V] tensor over the whole
    vocab (V ~ 10-20k VG words) OOMs even at moderate corpus sizes."""
    all_logits = []
    pil_images = [img for _, img in images]

    for start in tqdm(range(0, len(pil_images), batch_size), desc="Corpus forward pass"):
        batch = pil_images[start:start + batch_size]
        img_t = torch.stack([img_transform(im) for im in batch]).to(device)
        outputs = net(img_t)
        all_logits.append(outputs["patch_prototype_logits"][:, :, concept_indices].detach().cpu())

    return torch.cat(all_logits, dim=0), pil_images


def _draw_concept_row(axes_row, concept, topk_values, topk_indices, patch_logits_col, pil_images,
                       show_box=True, label_fontsize=20):
    """Draw one concept's top-k crops into a pre-made row of axes. Shared by both
    the combined-grid and one-file-per-concept output modes below."""
    for col, (score, global_idx) in enumerate(zip(topk_values.tolist(), topk_indices.tolist())):
        ax = axes_row[col]
        img_uint8 = _display_uint8(pil_images[global_idx])

        hm = patch_logits_col[global_idx]  # [N]
        N = hm.shape[0]
        H = W = int(math.sqrt(N))
        hm_up = F.interpolate(hm.view(1, 1, H, W), size=img_uint8.shape[:2],
                               mode="bilinear", align_corners=False)[0, 0]

        overlay = overlay_heatmap(img_uint8, hm_up, alpha=0.45)
        if show_box:
            bbox = find_high_activation_crop(hm_up.numpy(), percentile=95)
            overlay = draw_rect_on_image(overlay, bbox)

        ax.imshow(overlay)
        ax.axis("off")
    # ax.axis("off") hides a normal ylabel, so annotate with text instead
    axes_row[0].text(-0.12, 0.5, concept, transform=axes_row[0].transAxes,
                     rotation=90, va="center", ha="center",
                     fontsize=label_fontsize, fontweight="bold")


def make_figure(concepts, concept_word_to_score_and_patch, pil_images, topk, out_path, show_box=True):
    """Combined grid: one figure, one row per concept."""
    n_concepts = len(concepts)
    fig, axes = plt.subplots(n_concepts, topk, figsize=(3 * topk, 3 * n_concepts), dpi=140)
    if n_concepts == 1:
        axes = axes[None, :]

    for row, concept in enumerate(concepts):
        topk_values, topk_indices, patch_logits_col = concept_word_to_score_and_patch[concept]
        _draw_concept_row(axes[row], concept, topk_values, topk_indices, patch_logits_col, pil_images,
                           show_box=show_box)

    fig.suptitle("Concept retrieval: top-activating regions per vocabulary word", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path)
    plt.close(fig)


def make_separate_figures(concepts, concept_word_to_score_and_patch, pil_images, topk, out_dir, show_box=True):
    """One file per concept: results/concept_retrieval_<word>.png, easier to
    skim/share individually than a single tall grid when requesting many concepts.

    No on-image title: the word is already given by the rotated row label, and
    the "concept retrieval" framing belongs in the paper's own figure caption,
    not duplicated here."""
    paths = []
    for concept in concepts:
        topk_values, topk_indices, patch_logits_col = concept_word_to_score_and_patch[concept]
        fig, axes = plt.subplots(1, topk, figsize=(3 * topk, 3), dpi=140)
        _draw_concept_row(axes, concept, topk_values, topk_indices, patch_logits_col, pil_images,
                           show_box=show_box)
        fig.tight_layout()
        out_path = os.path.join(out_dir, f"concept_retrieval_{concept}.png")
        fig.savefig(out_path)
        plt.close(fig)
        paths.append(out_path)
    return paths


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
    p.add_argument("--n-concepts", type=int, default=None,
                   help="Convenience total count (60/40 noun/adj split), overrides "
                        "--n-nouns/--n-adjs when set and --concepts isn't given. "
                        "E.g. --n-concepts 20")
    p.add_argument("--separate-figures", action="store_true",
                   help="Save one PNG per concept (concept_retrieval_<word>.png) "
                        "instead of a single combined grid — easier to skim/share "
                        "when requesting many concepts.")
    p.add_argument("--no-box", action="store_true",
                   help="Skip drawing the red high-activation bounding box; show "
                        "only the heatmap overlay. Useful for comparing both "
                        "renderings before picking one for the paper.")
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
        n_nouns, n_adjs = args.n_nouns, args.n_adjs
        if args.n_concepts is not None:
            n_nouns = round(args.n_concepts * 0.6)
            n_adjs = args.n_concepts - n_nouns
        concepts = suggest_concepts(vocab_words, n_nouns, n_adjs, args.seed)
        print(f"Auto-suggested concepts (relational phrases not available in this vocab): {concepts}")

    print(f"Building deduplicated {args.dataset}/{args.split} image corpus ...")
    images = dedup_images(args.data_root, args.dataset, args.split)
    print(f"  {len(images)} unique images")

    concept_indices = [vocab_to_idx[c] for c in concepts]
    # patch_logits is pre-sliced to just these concept columns (order matches
    # `concepts`), NOT the full vocab — see collect_patch_logits docstring.
    patch_logits, pil_images = collect_patch_logits(
        net, images, img_transform, device, concept_indices, args.batch_size
    )

    # mixture_weights-equivalent per-image score: max patch activation per concept
    # (simplest, standard "does this concept fire anywhere in the image" ranking).
    concept_word_to_score_and_patch = {}
    for col, concept in enumerate(concepts):
        col_logits = patch_logits[:, :, col]           # [M, N]
        img_scores = col_logits.max(dim=1).values       # [M]
        k = min(args.topk, img_scores.shape[0])
        topk_values, topk_indices = torch.topk(img_scores, k=k)
        concept_word_to_score_and_patch[concept] = (topk_values, topk_indices, col_logits)

    show_box = not args.no_box
    if args.separate_figures:
        paths = make_separate_figures(concepts, concept_word_to_score_and_patch, pil_images,
                                      args.topk, args.out_dir, show_box=show_box)
        print(f"Saved {len(paths)} figures to {args.out_dir}/")
    else:
        out_path = os.path.join(args.out_dir, f"concept_retrieval_{args.dataset}_{args.split}.png")
        make_figure(concepts, concept_word_to_score_and_patch, pil_images, args.topk, out_path,
                    show_box=show_box)
        print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
