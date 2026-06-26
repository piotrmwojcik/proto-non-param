#!/usr/bin/env python3
"""
Summarize collapse-check results: print an epoch-by-epoch learning curve.

Reads eval_results/collapse_check/{VARIANT}/ep{N}/pnp_refer/Gref_val.json
and prints a table + a simple ASCII progress bar per epoch.

Usage:
  python scripts/summarize_collapse_check.py \
      --check-dir eval_results/collapse_check/dedup_A_80ep
"""

import argparse
import json
import os
import re


def load_results(check_dir: str):
    results = []
    ep_pattern = re.compile(r"^ep(\d+)$")
    for entry in sorted(os.listdir(check_dir)):
        m = ep_pattern.match(entry)
        if not m:
            continue
        epoch = int(m.group(1))
        json_path = os.path.join(check_dir, entry, "pnp_refer", "Gref_val.json")
        if not os.path.isfile(json_path):
            results.append({"epoch": epoch, "oIoU": None, "mIoU": None})
            continue
        with open(json_path) as f:
            data = json.load(f)
        s = data.get("summary", {})
        results.append({
            "epoch": epoch,
            "oIoU": s.get("cIoU"),
            "mIoU": s.get("mIoU"),
        })
    return sorted(results, key=lambda r: r["epoch"])


def bar(value, max_val=25.0, width=25):
    if value is None:
        return " " * width + "  (pending)"
    filled = int(round(value / max_val * width))
    filled = min(filled, width)
    return "█" * filled + "░" * (width - filled)


def flag(oiou):
    if oiou is None:
        return ""
    if oiou < 5:
        return "  *** COLLAPSE ***"
    if oiou < 10:
        return "  !! LOW"
    return ""


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--check-dir", required=True,
                   help="Directory containing ep010/ ep020/ ... subdirs")
    p.add_argument("--max-oiou", type=float, default=25.0,
                   help="Scale for ASCII bar (default: 25%% oIoU)")
    args = p.parse_args()

    if not os.path.isdir(args.check_dir):
        print(f"ERROR: {args.check_dir} not found")
        raise SystemExit(1)

    results = load_results(args.check_dir)
    if not results:
        print(f"No epoch results found in {args.check_dir}")
        raise SystemExit(1)

    variant = os.path.basename(args.check_dir.rstrip("/\\"))
    print(f"\nCollapse check — {variant}  (Gref / val)")
    print(f"{'Epoch':>6}  {'oIoU':>6}  {'mIoU':>6}  oIoU progress (max={args.max_oiou:.0f}%)")
    print("-" * 70)

    prev_oiou = None
    for r in results:
        ep    = r["epoch"]
        oiou  = r["oIoU"]
        miou  = r["mIoU"]

        # Delta vs previous checkpoint
        if oiou is not None and prev_oiou is not None:
            delta = oiou - prev_oiou
            delta_str = f"({delta:+.1f})"
        else:
            delta_str = "       "

        oiou_str = f"{oiou:5.1f}%" if oiou is not None else "  —   "
        miou_str = f"{miou:5.1f}%" if miou is not None else "  —   "

        print(f"  ep{ep:03d}  {oiou_str}  {miou_str}  "
              f"|{bar(oiou, args.max_oiou)}| {delta_str}{flag(oiou)}")

        if oiou is not None:
            prev_oiou = oiou

    # Best epoch
    valid = [r for r in results if r["oIoU"] is not None]
    if valid:
        best = max(valid, key=lambda r: r["oIoU"])
        print(f"\n  Best: ep{best['epoch']:03d}  oIoU={best['oIoU']:.1f}%  mIoU={best['mIoU']:.1f}%")

    print()


if __name__ == "__main__":
    main()
