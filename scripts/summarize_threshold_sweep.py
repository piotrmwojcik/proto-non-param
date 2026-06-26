#!/usr/bin/env python3
"""
Summarize threshold sweep results from evaluate_pnp_refer.py --threshold-sweep.

Reads all {dataset}_{split}_tXXX.json files in sweep-dir/pnp_refer/ and
prints a Markdown table: rows = thresholds, columns = dataset/split combos.

Usage:
  python scripts/summarize_threshold_sweep.py \
      --sweep-dir eval_results/threshold_sweep/dedup_A_80ep \
      --metric oIoU \
      --out eval_results/threshold_sweep/dedup_A_80ep/threshold_comparison.md
"""

import argparse
import json
import os
import re
from collections import defaultdict


DATASETS = ["Gref", "unc", "unc+"]
SPLITS = {"Gref": ["val"], "unc": ["val", "testA", "testB"], "unc+": ["val", "testA", "testB"]}
ALL_COLS = [(ds, sp) for ds in DATASETS for sp in SPLITS[ds]]  # 7 columns


def load_results(pnp_dir: str):
    """Return {threshold: {(dataset, split): {cIoU, mIoU}}}."""
    pattern = re.compile(r"^(.+)_(.+?)_t(\d{3})\.json$")
    data = defaultdict(dict)

    for fname in os.listdir(pnp_dir):
        m = pattern.match(fname)
        if not m:
            continue
        dataset, split, t_str = m.group(1), m.group(2), m.group(3)
        threshold = int(t_str) / 100.0
        with open(os.path.join(pnp_dir, fname)) as f:
            result = json.load(f)
        s = result.get("summary", {})
        data[threshold][(dataset, split)] = {
            "oIoU": s.get("cIoU"),
            "mIoU": s.get("mIoU"),
        }

    return data


def fmt(v):
    return f"{v:.1f}" if v is not None else "—"


def build_table(data: dict, metric: str) -> str:
    thresholds = sorted(data.keys())
    col_labels = [f"{ds}/{sp}" for ds, sp in ALL_COLS]
    headers = ["Threshold"] + col_labels + ["Mean"]

    rows = []
    for t in thresholds:
        vals = [data[t].get(col, {}).get(metric) for col in ALL_COLS]
        valid = [v for v in vals if v is not None]
        mean = sum(valid) / len(valid) if valid else None
        rows.append([f"{t:.2f}"] + [fmt(v) for v in vals] + [fmt(mean)])

    col_widths = [max(len(h), max(len(r[i]) for r in rows))
                  for i, h in enumerate(headers)]

    def row_str(cells):
        return "| " + " | ".join(c.ljust(w) for c, w in zip(cells, col_widths)) + " |"

    sep = "|-" + "-|-".join("-" * w for w in col_widths) + "-|"

    # Find best threshold by mean
    mean_col = len(ALL_COLS) + 1
    best_idx = max(range(len(rows)),
                   key=lambda i: float(rows[i][mean_col]) if rows[i][mean_col] != "—" else -1)

    lines = [row_str(headers), sep]
    for idx, row in enumerate(rows):
        line = row_str(row)
        if idx == best_idx:
            line += "  ← best mean"
        lines.append(line)

    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sweep-dir", required=True,
                   help="Directory containing pnp_refer/ with _tXXX.json files")
    p.add_argument("--metric", default="oIoU", choices=["oIoU", "mIoU"],
                   help="Metric to tabulate (default: oIoU)")
    p.add_argument("--out", default=None,
                   help="Optional .md output path")
    args = p.parse_args()

    pnp_dir = os.path.join(args.sweep_dir, "pnp_refer")
    if not os.path.isdir(pnp_dir):
        print(f"ERROR: {pnp_dir} not found")
        raise SystemExit(1)

    data = load_results(pnp_dir)
    if not data:
        print(f"No threshold sweep results found in {pnp_dir}")
        raise SystemExit(1)

    table = build_table(data, args.metric)
    header = f"# Threshold Sweep — {args.metric} (%)\n\n"
    print(header + table)

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            f.write(header + table + "\n")
        print(f"\nSaved to {args.out}")


if __name__ == "__main__":
    main()
