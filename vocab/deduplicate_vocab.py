#!/usr/bin/env python3
"""Deduplicate a CLIP vocab cache by clustering semantically similar concepts.

Uses agglomerative clustering (UPGMA / average linkage) on CLIP cosine distances
to merge synonyms (e.g. car/vehicle, person/man) into a single representative.
The representative of each cluster is the word whose embedding is closest to the
cluster centroid (Eq. 7 of the ProtoLang draft).

Outputs
-------
  --cache-out      Deduplicated cache (.pt); same format as the input cache —
                   drop-in replacement for --vocab-cache-path in train.py.
  --mapping-out    JSON: representative → [merged words] (for analysis / eval).

Threshold guidance (θ = cosine similarity)
  θ = 0.95  very conservative: near-exact synonyms only (safest default)
  θ = 0.90  moderate: merges most synonyms, keeps fine-grained distinctions
  θ = 0.85  aggressive: broader semantic clusters

Usage
-----
    python vocab/deduplicate_vocab.py \\
        --cache-in  vocab/vg_cache.pt \\
        --cache-out vocab/vg_cache_dedup_t90.pt \\
        --mapping-out vocab/vg_dedup_mapping_t90.json \\
        --threshold 0.90
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Deduplicate a CLIP vocab cache via agglomerative clustering"
    )
    parser.add_argument("--cache-in",    type=str, required=True,
                        help="Input vocab cache (.pt) built by build_vg_vocab.py")
    parser.add_argument("--cache-out",   type=str, required=True,
                        help="Output deduplicated cache (.pt)")
    parser.add_argument("--mapping-out", type=str, default=None,
                        help="Output JSON: representative → [merged words] (optional)")
    parser.add_argument("--threshold",   type=float, default=0.90,
                        help="Cosine-similarity threshold θ; concepts with "
                             "cos_sim ≥ θ are merged (default: 0.90)")
    args = parser.parse_args()

    # ---------------------------------------------------------------------- #
    # 1. Load cache
    # ---------------------------------------------------------------------- #
    print(f"Loading cache: {args.cache_in}")
    cache: dict[str, torch.Tensor] = torch.load(args.cache_in, map_location="cpu")
    words = list(cache.keys())
    embs  = torch.stack([cache[w] for w in words], dim=0)   # [V, D]
    embs  = F.normalize(embs, dim=-1)                        # unit sphere
    V, D  = embs.shape
    print(f"  vocabulary size: {V}  embedding dim: {D}")

    # ---------------------------------------------------------------------- #
    # 2. Agglomerative clustering via UPGMA on cosine distances
    #    cosine_distance = 1 - cosine_similarity  (both ∈ [0, 2] for unit vecs)
    # ---------------------------------------------------------------------- #
    print(f"Computing pairwise cosine distances ({V}×{V} / 2 = {V*(V-1)//2:,} pairs)...")
    embs_np = embs.numpy().astype(np.float32)
    dist_condensed = pdist(embs_np, metric="cosine")   # [V*(V-1)/2]

    print("Running agglomerative clustering (average linkage)...")
    Z = linkage(dist_condensed, method="average")

    # distance threshold = 1 - similarity threshold
    distance_threshold = 1.0 - args.threshold
    labels = fcluster(Z, t=distance_threshold, criterion="distance")   # [V], 1-indexed

    n_clusters = int(labels.max())
    print(f"  similarity threshold θ = {args.threshold:.2f}  "
          f"→  {V} concepts merged into {n_clusters} clusters "
          f"({V - n_clusters:+d} reduction, {100*(V-n_clusters)/V:.1f}%)")

    # ---------------------------------------------------------------------- #
    # 3. Pick representative per cluster (closest to centroid, Eq. 7)
    # ---------------------------------------------------------------------- #
    cluster_to_indices: dict[int, list[int]] = {}
    for i, lbl in enumerate(labels):
        cluster_to_indices.setdefault(int(lbl), []).append(i)

    dedup_cache: dict[str, torch.Tensor] = {}
    mapping: dict[str, list[str]] = {}           # representative → merged words

    for lbl, indices in cluster_to_indices.items():
        if len(indices) == 1:
            w = words[indices[0]]
            dedup_cache[w] = embs[indices[0]]
            mapping[w] = [w]
        else:
            cluster_embs = embs[indices]          # [C, D]
            centroid     = cluster_embs.mean(dim=0)
            centroid     = F.normalize(centroid, dim=0)
            dists        = 1.0 - (cluster_embs @ centroid)   # [C]
            best         = int(dists.argmin())
            rep          = words[indices[best]]
            dedup_cache[rep] = embs[indices[best]]
            mapping[rep]     = [words[i] for i in indices]

    print(f"  deduplicated vocabulary size: {len(dedup_cache)}")

    # Print the 20 largest clusters for inspection
    large = sorted(mapping.items(), key=lambda kv: len(kv[1]), reverse=True)
    print("\nTop-20 largest clusters:")
    for rep, members in large[:20]:
        if len(members) > 1:
            others = [m for m in members if m != rep]
            print(f"  {rep!r:20s} ← {others}")

    # ---------------------------------------------------------------------- #
    # 4. Save outputs
    # ---------------------------------------------------------------------- #
    os.makedirs(os.path.dirname(os.path.abspath(args.cache_out)), exist_ok=True)
    torch.save(dedup_cache, args.cache_out)
    print(f"\nSaved deduplicated cache ({len(dedup_cache)} words) → {args.cache_out}")

    if args.mapping_out:
        os.makedirs(os.path.dirname(os.path.abspath(args.mapping_out)), exist_ok=True)
        with open(args.mapping_out, "w", encoding="utf-8") as f:
            json.dump(mapping, f, indent=2, ensure_ascii=False)
        print(f"Saved cluster mapping → {args.mapping_out}")


if __name__ == "__main__":
    main()
