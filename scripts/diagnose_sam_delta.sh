#!/bin/bash
# Diagnose a suspicious eval delta by comparing per-sample IoU distributions,
# not just the summary oIoU/mIoU. Checks: sample-count match (rules out an
# accidentally-shrunk eval subset), zero-IoU rate before/after (a genuine
# "rescue" effect should show a big drop here), and how many samples moved
# up vs down.
#
# Usage:
#   BASE_DIR=eval_results/vg_contrastive/contr_M1/pnp_refer \
#   SI_DIR=eval_results/vg_contrastive/contr_M1_sam/pnp_refer \
#   bash scripts/diagnose_sam_delta.sh Gref val

set -e

BASE_DIR="${BASE_DIR:-eval_results/vg_contrastive/contr_M1/pnp_refer}"
SI_DIR="${SI_DIR:-eval_results/vg_contrastive/contr_M1_sam/pnp_refer}"
DATASET="${1:?usage: bash scripts/diagnose_sam_delta.sh <dataset> <split>}"
SPLIT="${2:?usage: bash scripts/diagnose_sam_delta.sh <dataset> <split>}"

python - "$BASE_DIR" "$SI_DIR" "$DATASET" "$SPLIT" <<'EOF'
import json
import os
import sys

base_dir, si_dir, dataset, split = sys.argv[1:]


def load(d):
    path = os.path.join(d, f"{dataset}_{split}.json")
    with open(path) as f:
        return json.load(f)


base = load(base_dir)
si = load(si_dir)

base_by_idx = {s["index"]: s["iou"] for s in base["per_sample"]}
si_by_idx = {s["index"]: s["iou"] for s in si["per_sample"]}

print(f"n_samples: base={len(base_by_idx)}  si={len(si_by_idx)}"
      f"  {'MATCH' if len(base_by_idx) == len(si_by_idx) else 'MISMATCH — different eval subset!'}")

common = sorted(set(base_by_idx) & set(si_by_idx))
zero_base = sum(1 for i in common if base_by_idx[i] == 0.0)
zero_si = sum(1 for i in common if si_by_idx[i] == 0.0)
print(f"zero-IoU rate: base={zero_base}/{len(common)} ({100*zero_base/len(common):.1f}%)"
      f"  si={zero_si}/{len(common)} ({100*zero_si/len(common):.1f}%)")

up = sum(1 for i in common if si_by_idx[i] > base_by_idx[i] + 1e-6)
down = sum(1 for i in common if si_by_idx[i] < base_by_idx[i] - 1e-6)
same = len(common) - up - down
print(f"per-sample moves: up={up}  down={down}  unchanged={same}")

rescued = sum(1 for i in common if base_by_idx[i] == 0.0 and si_by_idx[i] > 0.0)
broken = sum(1 for i in common if base_by_idx[i] > 0.0 and si_by_idx[i] == 0.0)
print(f"zero -> nonzero (rescued): {rescued}   nonzero -> zero (broken): {broken}")

avg_gain_when_up = sum(si_by_idx[i] - base_by_idx[i] for i in common if si_by_idx[i] > base_by_idx[i]) / max(up, 1)
print(f"avg IoU gain on improved samples: {avg_gain_when_up:.3f}")

# Recompute mIoU from per_sample to cross-check against the summary field
recomputed_miou = 100.0 * sum(si_by_idx[i] for i in common) / len(common)
print(f"\nsummary.mIoU = {si['summary']['mIoU']:.2f}   "
      f"recomputed from per_sample = {recomputed_miou:.2f}"
      f"  {'MATCH' if abs(recomputed_miou - si['summary']['mIoU']) < 0.05 else 'MISMATCH — bug in summary aggregation!'}")
EOF
