"""
Cheap, IO-only, GPU-free listing of the unique images
visualize_concept_retrieval.py's dedup_images() actually needs. Run this ON
ATHENA'S LOGIN NODE (plain python, no sbatch, no GPU, no model load)
against the full Gref/val_batch directory, then scp only the printed
filenames down instead of syncing all ~9536 npz files.

Same dedup logic/key as visualize_concept_retrieval.py:dedup_images()
(im_name, first-seen npz in sorted order wins) but skips decoding the
actual image array -- only the "which npz file represents each unique
image" mapping is needed here, so this reads just the tiny
im_name_batch field out of each npz.

Usage (on Athena, login node):
  python scripts/local/list_gref_dedup_images.py \
    --data-root $SCRATCH/data/refcoco --dataset Gref --split val \
    > $HOME/gref_dedup_files.txt

Then from your LOCAL machine, scp using that file list, e.g.:
  scp $(printf 'athena:$SCRATCH/data/refcoco/Gref/val_batch/%s ' $(cat gref_dedup_files.txt)) \
    local_run/assets/gref_val_dedup/
"""
import argparse
import os
import sys
from glob import glob

import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-root", required=True)
    p.add_argument("--dataset", default="Gref")
    p.add_argument("--split", default="val")
    args = p.parse_args()

    npz_paths = sorted(glob(os.path.join(args.data_root, args.dataset, args.split + "_batch", "*")))
    print(f"{len(npz_paths)} npz files found", file=sys.stderr)

    seen = set()
    representative = []
    for path in npz_paths:
        im_name = str(np.load(path)["im_name_batch"])
        if im_name in seen:
            continue
        seen.add(im_name)
        representative.append(os.path.basename(path))

    print(f"{len(representative)} unique images", file=sys.stderr)
    for fname in representative:
        print(fname)


if __name__ == "__main__":
    main()
