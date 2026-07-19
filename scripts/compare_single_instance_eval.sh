#!/bin/bash
# Compare a baseline eval against its --single-instance counterpart, side by side.
#
# Reads {dataset}_{split}.json summary.cIoU / summary.mIoU from two pnp_refer/
# output directories and prints a table with deltas. Defaults match the M1
# baseline vs contr_M1_single_instance produced by
# scripts/slurm_eval_vg_contrastive_M1_single_instance.sh.
#
# Configuration — override via environment variables:
#   BASE_DIR   pnp_refer/ dir for the baseline eval (no --single-instance)
#   SI_DIR     pnp_refer/ dir for the --single-instance eval
#   SPLITS     space-separated "dataset:split" pairs to compare (default below)
#
# Usage:
#   bash scripts/compare_single_instance_eval.sh
#   BASE_DIR=eval_results/vg_contrastive/contr_K2/pnp_refer \
#   SI_DIR=eval_results/vg_contrastive/contr_K2_single_instance/pnp_refer \
#   bash scripts/compare_single_instance_eval.sh

set -e

BASE_DIR="${BASE_DIR:-eval_results/vg_contrastive/contr_M1/pnp_refer}"
SI_DIR="${SI_DIR:-eval_results/vg_contrastive/contr_M1_single_instance/pnp_refer}"
SPLITS="${SPLITS:-Gref:val unc:val unc+:val}"

python - "$BASE_DIR" "$SI_DIR" $SPLITS <<'EOF'
import json
import os
import sys

base_dir, si_dir, *pairs = sys.argv[1:]


def load(d, dataset, split):
    path = os.path.join(d, f"{dataset}_{split}.json")
    if not os.path.isfile(path):
        return None
    with open(path) as f:
        s = json.load(f)["summary"]
    return s.get("cIoU"), s.get("mIoU")


print(f"Baseline : {base_dir}")
print(f"Single-instance : {si_dir}\n")
header = f"{'Dataset':<8} {'Split':<6} {'oIoU base':>10} {'oIoU SI':>9} {'diff oIoU':>10}   {'mIoU base':>10} {'mIoU SI':>9} {'diff mIoU':>10}"
print(header)
print("-" * len(header))

for pair in pairs:
    dataset, split = pair.split(":")
    base = load(base_dir, dataset, split)
    si = load(si_dir, dataset, split)

    if base is None or si is None:
        missing = f"{dataset}_{split}.json"
        print(f"{dataset:<8} {split:<6}  missing: {missing} (base={'ok' if base else 'MISSING'}, si={'ok' if si else 'MISSING'})")
        continue

    o_base, m_base = base
    o_si, m_si = si
    d_o = o_si - o_base
    d_m = m_si - m_base
    print(f"{dataset:<8} {split:<6} {o_base:>10.2f} {o_si:>9.2f} {d_o:>+8.2f}   "
          f"{m_base:>10.2f} {m_si:>9.2f} {d_m:>+8.2f}")
EOF
