#!/usr/bin/env python3
"""
mIoU vs. referring-expression length on RefCOCOg, PNP vs CTRL-O vs SaG.

Turns the known PNP-vs-CTRL-O gap on Gref/val into an insight: bucket
examples by expression length (word count) and show whether the gap
concentrates in longer, more compositional expressions. SaG is included in
the same setting for a third reference point.

Inputs:
  - PNP's per-sample eval JSON (evaluate_pnp_refer.py output), which stores
    per-example IoU + a dataset `index` but not the sentence text. Sentence
    text is recovered post-hoc via ReferDataset.get_raw_item(index) against
    the same data root/split (deterministic since ReferDataset.data_list is
    sorted).
  - CTRL-O's per-sentence eval JSON (inference_refer.py output, `per_sentence`
    key), which already stores the sentence text directly.
  - SaG's per-sample eval JSON (sag_refseg/evaluate.py output), which already
    stores per-example {max,avg,min}_iou + `index` per referring expression —
    no patch needed there, unlike CTRL-O. Uses the `avg` pooling mode's IoU
    (avg_iou), matching the mode compare_ris_results.py already treats as
    SaG's canonical reported number (avg_cIoU/avg_mIoU) elsewhere in this
    repo. Sentence text recovered the same way as PNP's, via
    sag_refseg.data.refer_dataset.ReferDataset.get_raw_item(index) — a
    separate (near-identical) copy of the same .npz-batch-reading class, not
    shared code, so it needed the same sorted-glob determinism fix applied
    separately (sag_refseg/data/refer_dataset.py).

Bucket edges are quantiles of PNP's own length distribution (all three models
are evaluated against the same benchmark split, so this is a fair, shared axis).

Usage:
  python scripts/analyze_miou_vs_length.py \
    --pnp-json eval_results/vg_contrastive/contr_M1_res672/pnp_refer/Gref_val.json \
    --data-root $SCRATCH/data/refcoco --dataset Gref --split val \
    --ctrlo-json $SCRATCH/eval_results/ctrlo/refcocog_metrics.json \
    --sag-json $SCRATCH/eval_results/sag_refseg/Gref_val.json \
    --out-dir results/miou_vs_length

Future extension (not implemented here): a POS-based compositionality metric
(e.g. dependency-parse depth) could supplement raw word count for a sharper
compositionality signal than length alone.
"""

import argparse
import csv
import json
import os
import sys

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)          # proto-non-param/
OUTER_REPO_ROOT = os.path.dirname(REPO_ROOT)      # proto-VLM/ (sag_refseg lives here)
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, OUTER_REPO_ROOT)

from evaluation.refer_dataset import ReferDataset  # noqa: E402
from sag_refseg.data.refer_dataset import ReferDataset as SagReferDataset  # noqa: E402


def load_pnp_examples(pnp_json_path, data_root, dataset, split):
    with open(pnp_json_path) as f:
        data = json.load(f)
    per_sample = data["per_sample"]

    ds = ReferDataset(root=os.path.join(data_root, dataset), splitset=split)
    examples = []
    for row in per_sample:
        sentence, _, _, _, _ = ds.get_raw_item(row["index"])
        sentence = str(sentence)
        examples.append({
            "model": "PNP",
            "sentence": sentence,
            "length": len(sentence.split()),
            "iou": float(row["iou"]),
        })
    return examples


def load_sag_examples(sag_json_path, data_root, dataset, split):
    with open(sag_json_path) as f:
        data = json.load(f)
    per_sample = data["per_sample"]

    ds = SagReferDataset(root=os.path.join(data_root, dataset), splitset=split)
    examples = []
    for row in per_sample:
        sentence, _, _, _, _ = ds.get_raw_item(row["index"])
        sentence = str(sentence)
        examples.append({
            "model": "SaG",
            "sentence": sentence,
            "length": len(sentence.split()),
            "iou": float(row["avg_iou"]),
        })
    return examples


def load_ctrlo_examples(ctrlo_json_path):
    with open(ctrlo_json_path) as f:
        data = json.load(f)
    if "per_sentence" not in data:
        raise ValueError(
            f"{ctrlo_json_path} has no 'per_sentence' key — re-run inference_refer.py "
            "with the per-sentence patch (records ref_id/sentence/mask_iou per expression, "
            "not just the oracle per-reference result)."
        )
    examples = []
    for row in data["per_sentence"]:
        sentence = str(row["sentence"])
        examples.append({
            "model": "CTRL-O",
            "sentence": sentence,
            "length": len(sentence.split()),
            "iou": float(row["mask_iou"]),
        })
    return examples


def bootstrap_ci(values, n_bootstrap=2000, seed=0):
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = np.empty(n_bootstrap)
    n = len(values)
    for i in range(n_bootstrap):
        sample = values[rng.integers(0, n, size=n)]
        means[i] = sample.mean()
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(values.mean()), float(lo), float(hi)


def bucket_examples(examples, edges):
    """Assign each example a bucket index in [0, len(edges)-2] via its length."""
    lengths = np.array([e["length"] for e in examples])
    # np.digitize with right=False bins into [edges[i], edges[i+1])except last bucket
    # is closed on both ends via clipping.
    bucket_idx = np.clip(np.digitize(lengths, edges[1:-1], right=True), 0, len(edges) - 2)
    return bucket_idx


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pnp-json", required=True, help="evaluate_pnp_refer.py per-sample result JSON")
    p.add_argument("--data-root", required=True, help="e.g. $SCRATCH/data/refcoco")
    p.add_argument("--dataset", default="Gref")
    p.add_argument("--split", default="val")
    p.add_argument("--ctrlo-json", required=True, help="inference_refer.py per_sentence result JSON")
    p.add_argument("--sag-json", default=None, help="sag_refseg/evaluate.py per-sample result JSON (optional)")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--n-buckets", type=int, default=4)
    p.add_argument("--n-bootstrap", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print(f"Loading PNP examples from {args.pnp_json} ...")
    pnp_examples = load_pnp_examples(args.pnp_json, args.data_root, args.dataset, args.split)
    print(f"  {len(pnp_examples)} PNP examples")

    print(f"Loading CTRL-O examples from {args.ctrlo_json} ...")
    ctrlo_examples = load_ctrlo_examples(args.ctrlo_json)
    print(f"  {len(ctrlo_examples)} CTRL-O examples")

    sag_examples = []
    if args.sag_json:
        print(f"Loading SaG examples from {args.sag_json} ...")
        sag_examples = load_sag_examples(args.sag_json, args.data_root, args.dataset, args.split)
        print(f"  {len(sag_examples)} SaG examples")

    all_examples = pnp_examples + ctrlo_examples + sag_examples

    # --- per-example CSV -----------------------------------------------------
    per_example_path = os.path.join(args.out_dir, "per_example.csv")
    with open(per_example_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["model", "sentence", "length", "iou"])
        w.writeheader()
        w.writerows(all_examples)
    print(f"Saved {per_example_path}")

    # --- quantile buckets from PNP's own length distribution -----------------
    pnp_lengths = np.array([e["length"] for e in pnp_examples])
    quantiles = np.linspace(0, 100, args.n_buckets + 1)
    edges = np.unique(np.percentile(pnp_lengths, quantiles))
    if len(edges) < 2:
        raise ValueError("Length distribution has no spread — cannot bucket.")
    n_buckets = len(edges) - 1
    if n_buckets < args.n_buckets:
        print(f"Note: only {n_buckets} distinct bucket edges after dedup "
              f"(requested {args.n_buckets}) — length distribution has ties at the tails.")

    # (model name, examples, bucket assignment) for every model actually loaded —
    # SaG is optional, only included when --sag-json is passed.
    models = [
        ("PNP", pnp_examples, bucket_examples(pnp_examples, edges)),
        ("CTRL-O", ctrlo_examples, bucket_examples(ctrlo_examples, edges)),
    ]
    if sag_examples:
        models.append(("SaG", sag_examples, bucket_examples(sag_examples, edges)))

    # --- per-bucket mIoU + bootstrap CI --------------------------------------
    per_bucket_rows = []
    for b in range(n_buckets):
        lo, hi = edges[b], edges[b + 1]
        for model, examples, bucket_idx in models:
            ious = [e["iou"] for e, bi in zip(examples, bucket_idx) if bi == b]
            mean_iou, ci_lo, ci_hi = bootstrap_ci(ious, args.n_bootstrap, args.seed)
            per_bucket_rows.append({
                "model": model,
                "bucket": b,
                "length_low": lo,
                "length_high": hi,
                "n": len(ious),
                "mean_iou": round(100 * mean_iou, 4),
                "ci_lo": round(100 * ci_lo, 4),
                "ci_hi": round(100 * ci_hi, 4),
            })

    per_bucket_path = os.path.join(args.out_dir, "per_bucket.csv")
    with open(per_bucket_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(per_bucket_rows[0].keys()))
        w.writeheader()
        w.writerows(per_bucket_rows)
    print(f"Saved {per_bucket_path}")

    for model, examples, _ in models:
        bucket_n_sum = sum(r["n"] for r in per_bucket_rows if r["model"] == model)
        if bucket_n_sum != len(examples):
            print(f"WARNING: {model} bucket counts sum to {bucket_n_sum}, expected {len(examples)}")

    # --- figure ---------------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 5), dpi=140)
    x = np.arange(n_buckets)
    model_colors = {"PNP": "#1f77b4", "CTRL-O": "#d62728", "SaG": "#2ca02c"}
    for model, _, _ in models:
        color = model_colors[model]
        rows = [r for r in per_bucket_rows if r["model"] == model]
        y = [r["mean_iou"] for r in rows]
        lo = [r["mean_iou"] - r["ci_lo"] for r in rows]
        hi = [r["ci_hi"] - r["mean_iou"] for r in rows]
        ax.errorbar(x, y, yerr=[lo, hi], marker="o", capsize=4, label=model, color=color)

    labels = [
        f"[{r['length_low']:.0f}-{r['length_high']:.0f}]\n(n={r['n']})"
        for r in per_bucket_rows if r["model"] == "PNP"
    ]
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("Referring expression length (words)")
    ax.set_ylabel("mIoU (%)")
    ax.set_title(f"mIoU vs. expression length — {args.dataset}/{args.split}")
    ax.legend()
    fig.tight_layout()

    fig_path = os.path.join(args.out_dir, "miou_vs_length.png")
    fig.savefig(fig_path)
    print(f"Saved {fig_path}")


if __name__ == "__main__":
    main()
