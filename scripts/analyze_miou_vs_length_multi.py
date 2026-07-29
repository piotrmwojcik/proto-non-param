#!/usr/bin/env python3
"""
mIoU vs. referring-expression length on RefCOCO and RefCOCO+ (in addition to
the existing Gref/val analysis from analyze_miou_vs_length.py), PNP vs
CTRL-O vs SaG.

Reuses analyze_miou_vs_length.py's loaders/bucketing/bootstrap unchanged —
this script only adds the loop over datasets/splits and the pooling step,
since RefCOCO and RefCOCO+ each have three splits (val/testA/testB) that
need combining into one length distribution per dataset, the same
granularity as the existing Gref table (tab:length in the appendix).

Follows the same eval-file naming convention as compare_ris_results.py:
  PNP    — {eval_dir}/pnp_refer/{dataset}_{split}.json
  SaG    — {eval_dir}/sag_refseg/{dataset}_{split}.json
  CTRL-O — {eval_dir}/ctrlo/{ctrlo_name}_metrics.json  (per_sentence key,
           already covers all splits for that dataset in one file)

Usage:
  python scripts/analyze_miou_vs_length_multi.py \
    --data-root $SCRATCH/data/refcoco \
    --eval-dir $SCRATCH/eval_results \
    --out-dir results/miou_vs_length_multi

Produces, per dataset (unc=RefCOCO, unc+=RefCOCO+):
  results/miou_vs_length_multi/{dataset}/per_example.csv
  results/miou_vs_length_multi/{dataset}/per_bucket.csv
  results/miou_vs_length_multi/{dataset}/miou_vs_length.png
  results/miou_vs_length_multi/{dataset}/table.tex   (ready to paste into
    the appendix, same column layout as the existing Gref-only tab:length)
"""

import argparse
import csv
import os
import sys

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from analyze_miou_vs_length import (  # noqa: E402
    bootstrap_ci,
    bucket_examples,
    load_ctrlo_examples,
    load_pnp_examples,
    load_sag_examples,
)

# Same dataset/split conventions as scripts/compare_ris_results.py.
SAG_TO_CTRLO = {"unc": "refcoco", "unc+": "refcoco+"}
SPLITS = ["val", "testA", "testB"]


def write_latex_table(dataset_label, per_bucket_rows, models, out_path):
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\caption{mIoU (\%) on " + dataset_label + r", bucketed by "
        r"referring-expression length in words, pooled across all splits. "
        r"Best per row in bold.}",
        r"\label{tab:length_" + dataset_label.lower().replace("+", "plus") + "}",
        r"\begin{tabular}{l" + "c" * len(models) + "}",
        r"\toprule",
        "Length (words) & " + " & ".join(models) + r" \\",
        r"\midrule",
    ]
    buckets = sorted(set(r["bucket"] for r in per_bucket_rows))
    for b in buckets:
        row = {r["model"]: r for r in per_bucket_rows if r["bucket"] == b}
        any_row = next(iter(row.values()))
        low, high = int(any_row["length_low"]), int(any_row["length_high"])
        vals = [row[m]["mean_iou"] if m in row else float("nan") for m in models]
        best = max(vals)
        cells = [
            (r"\textbf{%.2f}" % v) if v == best else ("%.2f" % v)
            for v in vals
        ]
        lines.append(f"{low}--{high} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")


def run_one_dataset(dataset, data_root, eval_dir, out_dir, n_buckets, n_bootstrap, seed):
    pnp_examples, sag_examples, ctrlo_examples = [], [], []
    for split in SPLITS:
        pnp_json = os.path.join(eval_dir, "pnp_refer", f"{dataset}_{split}.json")
        sag_json = os.path.join(eval_dir, "sag_refseg", f"{dataset}_{split}.json")
        if os.path.exists(pnp_json):
            pnp_examples += load_pnp_examples(pnp_json, data_root, dataset, split)
        else:
            print(f"  WARNING: missing {pnp_json}, skipping split {split} for PNP")
        if os.path.exists(sag_json):
            sag_examples += load_sag_examples(sag_json, data_root, dataset, split)
        else:
            print(f"  WARNING: missing {sag_json}, skipping split {split} for SaG")

    ctrlo_name = SAG_TO_CTRLO[dataset]
    ctrlo_json = os.path.join(eval_dir, "ctrlo", f"{ctrlo_name}_metrics.json")
    if os.path.exists(ctrlo_json):
        ctrlo_examples = load_ctrlo_examples(ctrlo_json)
    else:
        print(f"  WARNING: missing {ctrlo_json}, CTRL-O column will be empty")

    if not pnp_examples:
        print(f"  No PNP examples found for {dataset} — skipping.")
        return

    os.makedirs(out_dir, exist_ok=True)

    all_examples = pnp_examples + ctrlo_examples + sag_examples
    with open(os.path.join(out_dir, "per_example.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["model", "sentence", "length", "iou"])
        w.writeheader()
        w.writerows(all_examples)

    pnp_lengths = np.array([e["length"] for e in pnp_examples])
    quantiles = np.linspace(0, 100, n_buckets + 1)
    edges = np.unique(np.percentile(pnp_lengths, quantiles))
    n_buckets_actual = len(edges) - 1

    models = [("PNP", pnp_examples, bucket_examples(pnp_examples, edges))]
    if ctrlo_examples:
        models.append(("CTRL-O", ctrlo_examples, bucket_examples(ctrlo_examples, edges)))
    if sag_examples:
        models.append(("SaG", sag_examples, bucket_examples(sag_examples, edges)))

    per_bucket_rows = []
    for b in range(n_buckets_actual):
        lo, hi = edges[b], edges[b + 1]
        for model, examples, bucket_idx in models:
            ious = [e["iou"] for e, bi in zip(examples, bucket_idx) if bi == b]
            mean_iou, ci_lo, ci_hi = bootstrap_ci(ious, n_bootstrap, seed)
            per_bucket_rows.append({
                "model": model, "bucket": b,
                "length_low": lo, "length_high": hi, "n": len(ious),
                "mean_iou": round(100 * mean_iou, 4),
                "ci_lo": round(100 * ci_lo, 4), "ci_hi": round(100 * ci_hi, 4),
            })

    with open(os.path.join(out_dir, "per_bucket.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(per_bucket_rows[0].keys()))
        w.writeheader()
        w.writerows(per_bucket_rows)

    label = "RefCOCO" if dataset == "unc" else "RefCOCO+"
    write_latex_table(label, per_bucket_rows, [m for m, _, _ in models],
                       os.path.join(out_dir, "table.tex"))
    print(f"  Saved {out_dir}/{{per_example.csv, per_bucket.csv, table.tex}}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 5), dpi=140)
    x = np.arange(n_buckets_actual)
    model_colors = {"PNP": "#1f77b4", "CTRL-O": "#d62728", "SaG": "#2ca02c"}
    for model, _, _ in models:
        rows = [r for r in per_bucket_rows if r["model"] == model]
        y = [r["mean_iou"] for r in rows]
        lo = [r["mean_iou"] - r["ci_lo"] for r in rows]
        hi = [r["ci_hi"] - r["mean_iou"] for r in rows]
        ax.errorbar(x, y, yerr=[lo, hi], marker="o", capsize=4, label=model,
                     color=model_colors[model])
    labels = [f"[{r['length_low']:.0f}-{r['length_high']:.0f}]\n(n={r['n']})"
              for r in per_bucket_rows if r["model"] == "PNP"]
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("Referring expression length (words)")
    ax.set_ylabel("mIoU (%)")
    ax.set_title(f"mIoU vs. expression length — {label} (val+testA+testB pooled)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "miou_vs_length.png"))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-root", required=True, help="e.g. $SCRATCH/data/refcoco")
    p.add_argument("--eval-dir", required=True, help="e.g. $SCRATCH/eval_results")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--n-buckets", type=int, default=4)
    p.add_argument("--n-bootstrap", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    for dataset in ["unc", "unc+"]:
        print(f"=== {dataset} (val+testA+testB pooled) ===")
        run_one_dataset(
            dataset, args.data_root, args.eval_dir,
            os.path.join(args.out_dir, dataset),
            args.n_buckets, args.n_bootstrap, args.seed,
        )


if __name__ == "__main__":
    main()
