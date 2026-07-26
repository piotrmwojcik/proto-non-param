#!/usr/bin/env python3
"""
Prototype dictionary inspection (inference-only, no images needed -- pure
vocabulary-embedding analysis, cheap).

Two views:

1. Nearest-neighbor shift: for a handful of query words, compare their
   k-nearest OTHER vocabulary words (a) in CLIP's own original text-embedding
   space (before the projection head) vs (b) in the learned prototype space
   (after PNP.get_prototypes()'s projection). Does the projection head
   preserve CLIP's semantic neighborhoods, or does training reshape them?
   Reported as both neighbor lists side by side plus their overlap count.

2. t-SNE of the prototype space: scatter plot of a sampled subset of
   prototypes (real-word-filtered via nltk.corpus.words, same filter used in
   visualize_concept_retrieval.py, to avoid the vocab's un-curated typo tail
   dominating the picture), colored by coarse POS (noun/adjective/other),
   with the query words from --words annotated directly on the plot.

Usage:
  python scripts/inspect_prototype_dictionary.py \
    --ckpt $SCRATCH/train_logs/vg_contrastive/run_M1_vitl14_sk10_koleo01_30ep/ckpt.pth \
    --words dog car red running happy \
    --out-dir results/prototype_dictionary
"""

import argparse
import json
import os
import random
import sys

import nltk
import numpy as np
import torch
import torch.nn.functional as F

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, SCRIPT_DIR)

from evaluate_pnp_refer import build_model  # noqa: E402


def nearest_neighbors(query_idx, embeddings, names, k=8):
    """embeddings: [V, D] already L2-normalized. Returns [(name, cosine_sim), ...],
    excluding the query word itself."""
    sims = embeddings @ embeddings[query_idx]
    order = torch.argsort(sims, descending=True)
    out = []
    for i in order.tolist():
        if i == query_idx:
            continue
        out.append((names[i], round(float(sims[i]), 4)))
        if len(out) == k:
            break
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--words", type=str, nargs="*", default=None,
                   help="Query words for the nearest-neighbor comparison and t-SNE "
                        "annotation (default: 8 random real words from the vocab)")
    p.add_argument("--k-neighbors", type=int, default=8)
    p.add_argument("--n-tsne-words", type=int, default=2000,
                   help="Random real-word subsample size for the t-SNE plot "
                        "(full vocab is too slow/cluttered)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--out-dir", required=True)
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    print(f"Loading PNP model from {args.ckpt} ...")
    net, _, hparams = build_model(args.ckpt, device)
    vocab_words = net.vocab_words
    print(f"Checkpoint vocab: {hparams.vocab_cache_path} ({len(vocab_words)} words)")

    with torch.inference_mode():
        clip_embs = F.normalize(net.vocab_clip_embeddings, dim=-1)          # [V, 512] original CLIP space
        prototypes = F.normalize(net.get_prototypes(), dim=-1)               # [V, D] learned projected space

    clip_embs_cpu = clip_embs.cpu()
    prototypes_cpu = prototypes.cpu()

    # --- real-word filter (same pattern as visualize_concept_retrieval.py) ---
    nltk.download("words", quiet=True)
    from nltk.corpus import words as nltk_words
    real_word_set = set(nltk_words.words())
    real_indices = [i for i, w in enumerate(vocab_words) if w.isalpha() and w in real_word_set]
    print(f"{len(real_indices)}/{len(vocab_words)} vocab words pass the real-word filter")

    rng = random.Random(args.seed)
    words = args.words
    if not words:
        words = [vocab_words[i] for i in rng.sample(real_indices, min(8, len(real_indices)))]
        print(f"No --words given, sampled: {words}")

    word_to_idx = {w: i for i, w in enumerate(vocab_words)}
    missing = [w for w in words if w not in word_to_idx]
    if missing:
        raise ValueError(f"Words not in checkpoint vocab: {missing}")

    # --- 1. nearest-neighbor shift ---------------------------------------------
    print("\n=== Nearest-neighbor shift: CLIP text space vs. learned prototype space ===")
    nn_report = []
    for w in words:
        idx = word_to_idx[w]
        clip_nn = nearest_neighbors(idx, clip_embs_cpu, vocab_words, args.k_neighbors)
        proto_nn = nearest_neighbors(idx, prototypes_cpu, vocab_words, args.k_neighbors)
        overlap = len(set(n for n, _ in clip_nn) & set(n for n, _ in proto_nn))
        print(f"  '{w}':")
        print(f"    CLIP text space      : {clip_nn}")
        print(f"    learned prototype sp.: {proto_nn}")
        print(f"    overlap: {overlap}/{args.k_neighbors}")
        nn_report.append({
            "word": w, "clip_space_neighbors": clip_nn, "prototype_space_neighbors": proto_nn,
            "overlap": overlap, "k": args.k_neighbors,
        })

    with open(os.path.join(args.out_dir, "nearest_neighbor_shift.json"), "w") as f:
        json.dump(nn_report, f, indent=2)
    print(f"Saved {os.path.join(args.out_dir, 'nearest_neighbor_shift.json')}")

    # --- 2. t-SNE of the prototype space ----------------------------------------
    print(f"\nRunning t-SNE on {min(args.n_tsne_words, len(real_indices))} real-word prototypes ...")
    tsne_indices = rng.sample(real_indices, min(args.n_tsne_words, len(real_indices)))
    # Always include the query words so they can be annotated even if not sampled.
    for w in words:
        idx = word_to_idx[w]
        if idx not in tsne_indices:
            tsne_indices.append(idx)

    from sklearn.manifold import TSNE
    tsne_embs = prototypes_cpu[tsne_indices].numpy()
    coords = TSNE(n_components=2, random_state=args.seed, init="pca",
                  perplexity=min(30, len(tsne_indices) - 1)).fit_transform(tsne_embs)

    tagged = nltk.pos_tag([vocab_words[i] for i in tsne_indices])
    pos_color = []
    for _, pos in tagged:
        if pos.startswith("NN"):
            pos_color.append("#1f77b4")   # noun
        elif pos.startswith("JJ"):
            pos_color.append("#ff7f0e")   # adjective
        else:
            pos_color.append("#7f7f7f")   # other

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 9), dpi=140)
    ax.scatter(coords[:, 0], coords[:, 1], c=pos_color, s=8, alpha=0.5)
    for w in words:
        idx = word_to_idx[w]
        pos_in_sample = tsne_indices.index(idx)
        x, y = coords[pos_in_sample]
        ax.scatter([x], [y], c="red", s=60, zorder=5, edgecolors="black")
        ax.annotate(w, (x, y), fontsize=11, fontweight="bold", xytext=(5, 5),
                   textcoords="offset points")

    from matplotlib.lines import Line2D
    legend_elems = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#1f77b4", markersize=8, label="noun"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#ff7f0e", markersize=8, label="adjective"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#7f7f7f", markersize=8, label="other"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="red", markersize=10,
              markeredgecolor="black", label="query words"),
    ]
    ax.legend(handles=legend_elems, loc="best")
    ax.set_title(f"t-SNE of learned prototype space ({len(tsne_indices)} real words)")
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()

    fig_path = os.path.join(args.out_dir, "prototype_tsne.png")
    fig.savefig(fig_path)
    print(f"Saved {fig_path}")


if __name__ == "__main__":
    main()
