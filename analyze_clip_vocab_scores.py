#!/usr/bin/env python3
"""
analyze_clip_vocab_scores.py

Diagnose the CLIP-based vocabulary score files produced by build_clip_vocab_scores.py.
Shows entropy, score ranges, top-k words, and effective vocabulary size across
different temperatures — helps pick a good temperature before retraining.

Usage:
    python analyze_clip_vocab_scores.py --scores vocab/vg_clip_scores.pt
    python analyze_clip_vocab_scores.py --scores vocab/coco_train_clip_scores.pt
"""

import argparse
import math
import random
import torch
import torch.nn.functional as F


def entropy_bits(probs: torch.Tensor) -> torch.Tensor:
    """Per-row Shannon entropy in bits."""
    log_p = torch.log2(probs.clamp(min=1e-12))
    return -(probs * log_p).sum(dim=-1)


def effective_vocab_size(probs: torch.Tensor) -> torch.Tensor:
    """Number of words with probability above uniform (1/V)."""
    V = probs.shape[-1]
    return (probs > 1.0 / V).float().sum(dim=-1)


def analyse_at_temperature(raw_scores: torch.Tensor, temperature: float, label: str):
    soft = F.softmax(raw_scores / temperature, dim=-1)        # [N, V]
    ent = entropy_bits(soft)                                   # [N]
    eff = effective_vocab_size(soft)                           # [N]
    max_prob = soft.max(dim=-1).values                         # [N]

    V = raw_scores.shape[1]
    max_ent = math.log2(V)

    print(f"\n  [{label}]  T={temperature}")
    print(f"    entropy   : mean={ent.mean():.2f} bits  (max possible = {max_ent:.2f} bits)")
    print(f"    entropy%  : {100*ent.mean()/max_ent:.1f}% of maximum  "
          f"(100% = uniform = useless, <50% = good)")
    print(f"    eff. vocab: mean={eff.mean():.0f} words above uniform  (out of {V})")
    print(f"    top-1 prob: mean={max_prob.mean()*100:.2f}%  "
          f"(higher = sharper = more signal)")


def show_top_words(raw_scores: torch.Tensor, vocab: list, n_images: int = 5,
                   top_k: int = 10, temperature: float = 0.07, seed: int = 42):
    random.seed(seed)
    N = raw_scores.shape[0]
    indices = random.sample(range(N), min(n_images, N))
    soft = F.softmax(raw_scores / temperature, dim=-1)

    print(f"\n--- Top-{top_k} words per image (T={temperature}) ---")
    for idx in indices:
        vals, idxs = soft[idx].topk(top_k)
        words = [vocab[i] for i in idxs.tolist()]
        probs = [f"{v*100:.2f}%" for v in vals.tolist()]
        pairs = "  ".join(f"{w}({p})" for w, p in zip(words, probs))
        print(f"  img[{idx:6d}]: {pairs}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores", required=True, help="Path to *_clip_scores.pt file")
    parser.add_argument("--n-images", type=int, default=5,
                        help="Number of random images to show top-k words for")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print(f"Loading {args.scores} ...")
    data = torch.load(args.scores, map_location="cpu")

    vocab: list = data["vocab"]
    raw_scores: torch.Tensor = data["clip_scores"].float()    # [N, V]
    N, V = raw_scores.shape

    print(f"\n=== File summary ===")
    print(f"  dataset    : {data.get('dataset', 'unknown')}")
    print(f"  clip_model : {data.get('clip_model', 'unknown')}")
    print(f"  images     : {N}")
    print(f"  vocab size : {V}")
    print(f"  saved with : temperature={data.get('temperature', '?')}  "
          f"alpha={data.get('alpha', '?')}")

    print(f"\n=== Raw score statistics ===")
    print(f"  min={raw_scores.min():.4f}  "
          f"max={raw_scores.max():.4f}  "
          f"mean={raw_scores.mean():.4f}  "
          f"std={raw_scores.std():.4f}")
    print(f"  (CLIP similarities typically in [0.10, 0.35]; "
          f"gap between pos/neg is ~0.05–0.15)")

    print(f"\n=== Entropy & effective vocab at different temperatures ===")
    print(f"  (entropy% closer to 0% = sharper = more training signal)")
    for T in [0.02, 0.05, 0.07, 0.1, 0.2, 0.5, 1.0]:
        analyse_at_temperature(raw_scores, T, f"T={T}")

    show_top_words(raw_scores, vocab, n_images=args.n_images, seed=args.seed)

    print("\n=== Recommendation ===")
    # Find T where mean entropy is ~40-60% of max
    max_ent = math.log2(V)
    for T in [0.02, 0.05, 0.07, 0.1, 0.2]:
        soft = F.softmax(raw_scores / T, dim=-1)
        ent_pct = 100 * entropy_bits(soft).mean().item() / max_ent
        eff = effective_vocab_size(soft).mean().item()
        if 30 <= ent_pct <= 70:
            print(f"  T={T} gives {ent_pct:.0f}% entropy ({eff:.0f} words active) — good range")
            break
    else:
        print(f"  Try T=0.07 (CLIP default). Avoid T=1.0 (near-uniform, no signal).")


if __name__ == "__main__":
    main()
