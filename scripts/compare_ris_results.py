#!/usr/bin/env python3
"""
Compare RIS evaluation results across SaG, CTRL-O, and PNP.

Reads JSON files produced by each method's eval script and prints a
unified Markdown table normalized to the same metric scale (percentages).

Metric mapping:
  SaG    — eval_results/sag_refseg/{dataset}_{split}.json
             summary.avg_cIoU  (%)  → oIoU
             summary.avg_mIoU  (%)  → mIoU

  PNP    — eval_results/pnp_refer/{dataset}_{split}.json
             summary.cIoU      (%)  → oIoU
             summary.mIoU      (%)  → mIoU

  CTRL-O — eval_results/ctrlo/{dataset}_metrics.json
             summary.{split}.omiou_mask  (0-1) × 100  → oIoU
             summary.{split}.miou_mask   (0-1) × 100  → mIoU
             summary.{split}.acc_mask_0.5 (0-1) × 100 → Acc@0.5

Dataset name cross-reference:
  SaG/PNP name  →  CTRL-O name
  Gref          →  refcocog
  unc           →  refcoco
  unc+          →  refcoco+

Usage:
  python scripts/compare_ris_results.py
  python scripts/compare_ris_results.py --eval_dir eval_results --out eval_results/comparison.md

  # Compare caption-signal ablation checkpoints (A/B/C) against baselines:
  python scripts/compare_ris_results.py --ablation-dir eval_results
"""

import argparse
import json
import os
import sys
from typing import Optional


# ---------------------------------------------------------------------------
# Dataset name mappings
# ---------------------------------------------------------------------------

SAG_TO_CTRLO = {
    "Gref": "refcocog",
    "unc":  "refcoco",
    "unc+": "refcoco+",
}

ALL_DATASETS = ["Gref", "unc", "unc+"]
UNC_SPLITS   = ["val", "testA", "testB"]
GREF_SPLITS  = ["val"]


# ---------------------------------------------------------------------------
# JSON loaders for each method
# ---------------------------------------------------------------------------

def load_sag(eval_dir: str, dataset: str, split: str) -> Optional[dict]:
    path = os.path.join(eval_dir, "sag_refseg", f"{dataset}_{split}.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        data = json.load(f)
    s = data.get("summary", {})
    return {
        "oIoU": s.get("avg_cIoU"),
        "mIoU": s.get("avg_mIoU"),
        "Acc@0.5": None,
        "source": path,
    }


def load_pnp(eval_dir: str, dataset: str, split: str) -> Optional[dict]:
    path = os.path.join(eval_dir, "pnp_refer", f"{dataset}_{split}.json")
    if not os.path.exists(path):
        path = os.path.join(eval_dir, f"{dataset}_{split}.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        data = json.load(f)
    s = data.get("summary", {})
    return {
        "oIoU": s.get("cIoU"),
        "mIoU": s.get("mIoU"),
        "Acc@0.5": None,
        "source": path,
    }


def load_ctrlo(eval_dir: str, dataset: str, split: str) -> Optional[dict]:
    ctrlo_name = SAG_TO_CTRLO.get(dataset, dataset)
    path = os.path.join(eval_dir, "ctrlo", f"{ctrlo_name}_metrics.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        data = json.load(f)
    summary = data.get("summary", {})
    s = summary.get(split)
    if s is None:
        return None
    omiou = s.get("omiou_mask") or s.get("omiou_mask_avg")
    miou  = s.get("miou_mask")  or s.get("miou_mask_avg")
    acc   = s.get("acc_mask_0.5") or s.get("acc_mask_0.5_avg")
    return {
        "oIoU":    round(omiou * 100, 2) if omiou is not None else None,
        "mIoU":    round(miou  * 100, 2) if miou  is not None else None,
        "Acc@0.5": round(acc   * 100, 2) if acc   is not None else None,
        "source": path,
        "_note": "CTRL-O uses REFER API (may differ from SaG/PNP .npz batches)",
    }


# ---------------------------------------------------------------------------
# Table formatting
# ---------------------------------------------------------------------------

def fmt(v: Optional[float]) -> str:
    if v is None:
        return "—"
    return f"{v:.1f}"


ABLATION_VARIANTS = {
    "caption_signal": {
        "A": ("PNP-A (word-only)",    "ablation_A"),
        "B": ("PNP-B (caption-only)", "ablation_B"),
        "C": ("PNP-C (combined)",     "ablation_C"),
    },
    "vg_ablation": {
        "A": ("PNP-A (KL + frozen)",    "ablation_A"),
        "B": ("PNP-B (KL + residual)",  "ablation_B"),
        "C": ("PNP-C (JSD + frozen)",   "ablation_C"),
        "D": ("PNP-D (JSD + residual)", "ablation_D"),
    },
    "vg_long": {
        "A": ("PNP-A-long (KL + frozen, 80ep)",  "long_A"),
        "C": ("PNP-C-long (JSD + frozen, 80ep)", "long_C"),
    },
    "vg_dedup": {
        "A": ("PNP-A-dedup (KL + frozen, dedup)",  "dedup_A"),
        "C": ("PNP-C-dedup (JSD + frozen, dedup)", "dedup_C"),
    },
    "vg_contrastive": {
        "A": ("PNP-A (uniform + contrastive=0.5, 30ep)", "contr_A"),
        "B": ("PNP-B (uniform + contrastive=1.0, 30ep)", "contr_B"),
    },
}


def build_table(eval_dir: str, ablation_dir: Optional[str] = None,
                ablation_type: str = "caption_signal") -> str:
    loaders: dict = {
        "SaG":    load_sag,
        "CTRL-O": load_ctrlo,
    }

    if ablation_dir is not None:
        variants = ABLATION_VARIANTS.get(ablation_type, ABLATION_VARIANTS["caption_signal"])
        for key, (label, subdir) in variants.items():
            variant_dir = os.path.join(ablation_dir, subdir)
            if os.path.isdir(variant_dir):
                loaders[label] = lambda ed, ds, sp, d=variant_dir: load_pnp(d, ds, sp)
            else:
                print(f"Warning: ablation variant {key} not found at {variant_dir}",
                      file=sys.stderr)
    else:
        loaders["PNP"] = load_pnp

    rows = []
    headers = ["Dataset", "Split", "Method", "oIoU (%)", "mIoU (%)", "Acc@0.5 (%)"]

    missing_files = []

    for dataset in ALL_DATASETS:
        splits = GREF_SPLITS if dataset == "Gref" else UNC_SPLITS
        for split in splits:
            for method, loader in loaders.items():
                result = loader(eval_dir, dataset, split)
                if result is None:
                    missing_files.append(f"  {method:8s}  {dataset}/{split}")
                    rows.append([dataset, split, method, None, None, None])
                else:
                    rows.append([
                        dataset, split, method,
                        result["oIoU"], result["mIoU"], result.get("Acc@0.5"),
                    ])

    # Markdown table
    col_widths = [max(len(h), max(len(str(r[i] or "—")) for r in rows))
                  for i, h in enumerate(headers)]
    col_widths = [max(w, len(h)) for w, h in zip(col_widths, headers)]

    def row_str(cells):
        return "| " + " | ".join(str(c or "—").ljust(w)
                                 for c, w in zip(cells, col_widths)) + " |"

    sep = "|-" + "-|-".join("-" * w for w in col_widths) + "-|"

    lines = [row_str(headers), sep]
    prev_key = None
    for row in rows:
        key = (row[0], row[1])
        if prev_key and key != prev_key:
            lines.append(sep)
        prev_key = key
        lines.append(row_str([
            row[0], row[1], row[2],
            fmt(row[3]), fmt(row[4]), fmt(row[5]),
        ]))

    table = "\n".join(lines)

    notes = []
    if missing_files:
        notes.append("\n**Missing result files** (run the corresponding eval scripts):\n"
                     + "\n".join(missing_files))
    notes.append(
        "\n**Notes**:\n"
        "- SaG and PNP use the same .npz batch files → directly comparable.\n"
        "- CTRL-O uses the REFER API directly → slight per-sample coverage difference.\n"
        "- oIoU = overall/cumulative IoU = ΣI / ΣU. mIoU = mean per-sample IoU."
    )

    return table + "\n" + "\n".join(notes)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="Compare RIS results across methods")
    p.add_argument("--eval_dir", default="./eval_results",
                   help="Root eval_results/ directory (for SaG, CTRL-O, and base PNP)")
    p.add_argument("--ablation-dir", default=None,
                   help="Directory containing ablation_A/, ablation_B/, ... subdirs. "
                        "When set, shows per-variant PNP entries instead of a single PNP entry.")
    p.add_argument("--ablation-type", default="caption_signal",
                   choices=list(ABLATION_VARIANTS.keys()),
                   help="Which ablation variant set to use. "
                        "'caption_signal' = A/B/C (word-only/caption-only/combined); "
                        "'vg_ablation' = A/B/C/D (KL×JSD × frozen×residual); "
                        "'vg_long' = A/C (frozen residual, 80 epochs); "
                        "'vg_dedup' = A/C (frozen residual, deduplicated vocab); "
                        "'vg_contrastive' = A/B (uniform + InfoNCE, 30 epochs). "
                        "Default: caption_signal")
    p.add_argument("--out", default=None,
                   help="Optional path to save the table as a .md file")
    args = p.parse_args()

    if not os.path.isdir(args.eval_dir):
        print(f"eval_dir not found: {args.eval_dir}", file=sys.stderr)
        print("Run SaG, CTRL-O, and PNP eval scripts first.", file=sys.stderr)
        sys.exit(1)

    table = build_table(args.eval_dir, ablation_dir=args.ablation_dir,
                        ablation_type=args.ablation_type)
    print(table)

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            f.write("# RIS Evaluation Comparison\n\n")
            f.write(table + "\n")
        print(f"\nSaved to {args.out}")


if __name__ == "__main__":
    main()
