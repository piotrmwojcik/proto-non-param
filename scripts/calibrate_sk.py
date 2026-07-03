#!/usr/bin/env python3
"""
Calibrate Sinkhorn-Knopp eps and coef for PNP vocab_logits.

Run locally (CPU) before choosing SK hyperparameters:
    python scripts/calibrate_sk.py
    python scripts/calibrate_sk.py --ckpt /path/to/ckpt.pth  # real logit statistics

Prints a table over eps values showing:
  - entropy of SK rows (nats) vs theoretical max log(V)
  - effective support of SK rows (exp(H))
  - column-sum std (how uniform the prototype marginal is)
  - l_sk magnitude (relative to typical l_dist ~ 1-5)
"""

import argparse
import math
import torch
import torch.nn.functional as F


def sinkhorn_knopp(logits: torch.Tensor, eps: float, n_iter: int = 3) -> torch.Tensor:
    """Returns Q [B, V] with rows summing to 1, cols uniform (B/V each)."""
    # Subtract max per row before exp for numerical stability
    logits = logits - logits.max(dim=-1, keepdim=True).values
    Q = torch.exp(logits / eps).T  # [V, B]
    Q /= Q.sum()
    K, B = Q.shape
    for _ in range(n_iter):
        Q /= Q.sum(dim=1, keepdim=True) * K   # uniform over V
        Q /= Q.sum(dim=0, keepdim=True) * B   # uniform over B
    return (Q * B).T  # [B, V], rows sum to 1


def report(logits: torch.Tensor, eps_values, n_iter: int = 3, temp: float = 0.2):
    B, V = logits.shape
    print(f"\nB={B}, V={V}, logit_range=[{logits.min():.3f}, {logits.max():.3f}]")
    print(f"{'eps':>8}  {'H(Q_row)':>10}  {'H_max':>8}  {'H/H_max':>9}  {'eff_supp':>10}  {'col_std':>9}  {'l_sk':>8}  {'l_dist':>8}")
    print("-" * 85)

    log_P = F.log_softmax(logits / temp, dim=-1)
    # compute baseline l_dist magnitude (uniform target KL)
    uniform = torch.full_like(logits, 1.0 / V)
    l_dist_ref = F.kl_div(log_P, uniform, reduction="batchmean").item()
    h_max = math.log(V)

    for eps in eps_values:
        Q = sinkhorn_knopp(logits, eps=eps, n_iter=n_iter)
        # entropy of rows of Q
        h_rows = -(Q * (Q + 1e-30).log()).sum(dim=-1).mean().item()
        # effective support
        eff_supp = math.exp(h_rows)
        # column uniformity: each col should be B/V
        col_sums = Q.sum(dim=0)  # [V]
        col_std = col_sums.std().item()
        # SK loss magnitude
        l_sk = -(Q * log_P).sum(dim=-1).mean().item()
        print(f"{eps:>8.3f}  {h_rows:>10.4f}  {h_max:>8.4f}  {h_rows/h_max:>9.4f}  "
              f"{eff_supp:>10.1f}  {col_std:>9.6f}  {l_sk:>8.4f}  {l_dist_ref:>8.4f}")

    print(f"\n  l_dist_ref (KL vs uniform, temp={temp}): {l_dist_ref:.4f}")
    print("  → choose eps so H/H_max ≈ 0.85–0.95 (moderately spread)")
    print("  → choose sk_coef so sk_coef * l_sk ≈ l_dist_ref * kl_coef")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default=None, help="Optional checkpoint to load real logits from "
                   "(runs one batch of synthetic images through the model; requires GPU).")
    p.add_argument("--B", type=int, default=128, help="Batch size (default: 128)")
    p.add_argument("--V", type=int, default=15858, help="Vocab size (default: 15858)")
    p.add_argument("--temp", type=float, default=0.2, help="PNP temperature for log_P (default: 0.2)")
    p.add_argument("--n-iter", type=int, default=3, help="SK iterations (default: 3)")
    args = p.parse_args()

    eps_values = [0.05, 0.1, 0.2, 0.3, 0.5, 1.0, 2.0]

    if args.ckpt is not None:
        print(f"Loading real logit statistics from checkpoint: {args.ckpt}")
        ckpt = torch.load(args.ckpt, map_location="cpu")
        # Try to load stored logit stats if available, else fall through to synthetic
        if "logit_stats" in ckpt:
            mean, std = ckpt["logit_stats"]["mean"], ckpt["logit_stats"]["std"]
            logits = torch.randn(args.B, args.V) * std + mean
            print(f"  Using stored stats: mean={mean:.4f}, std={std:.4f}")
        else:
            print("  No logit_stats in checkpoint; using synthetic cosine-scale logits.")
            args.ckpt = None

    if args.ckpt is None:
        # Synthetic: cosine sims are bounded [-1,1]; trained model has moderate concentration.
        # Approximate: Normal(0, 0.15) with top-5 pooling (reduces variance slightly).
        logits = torch.randn(args.B, args.V) * 0.15
        # Make a few prototypes dominant (simulate partial collapse, H/H_max=0.777)
        n_active = int(0.116 * args.V)   # 11.6% effective from diagnostic
        hot_idx = torch.randperm(args.V)[:n_active]
        logits[:, hot_idx] += 0.3
        print("Synthetic logits: Normal(0, 0.15) + 0.3 boost on 11.6% of prototypes")

    report(logits, eps_values, n_iter=args.n_iter, temp=args.temp)


if __name__ == "__main__":
    main()
