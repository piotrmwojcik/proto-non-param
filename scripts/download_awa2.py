"""
Download and organize Animals with Attributes 2 (AwA2) dataset.

Downloads AwA2-base.zip (metadata, ~32 KB) and AwA2-data.zip (images, ~13 GB),
creates seeded per-class train/val/test splits, copies images into class
subdirectories, and copies the metadata files needed at training time.

Usage:
    python scripts/download_awa2.py \
        --output-dir /net/tscratch/people/plgabedychaj/awa2 \
        --train-ratio 0.7 \
        --val-ratio 0.1 \
        --seed 42
"""
import argparse
import os
import random
import shutil
import zipfile
import urllib.request
from pathlib import Path

AWA2_BASE_URL = "https://cvml.ista.ac.at/AwA2/AwA2-base.zip"
AWA2_DATA_URL = "https://cvml.ista.ac.at/AwA2/AwA2-data.zip"


def download_file(url: str, dest: Path):
    if dest.exists():
        print(f"  Already exists: {dest}")
        return
    print(f"  Downloading {url} → {dest} ...")
    dest.parent.mkdir(parents=True, exist_ok=True)

    def _progress(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            pct = min(100, downloaded * 100 // total_size)
            mb = downloaded // 1024 // 1024
            total_mb = total_size // 1024 // 1024
            print(f"\r  {pct}%  ({mb} MB / {total_mb} MB)", end="", flush=True)

    urllib.request.urlretrieve(url, dest, reporthook=_progress)
    print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True, help="Destination directory")
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--keep-archives", action="store_true", help="Keep downloaded .zip files after extraction")
    args = parser.parse_args()

    assert args.train_ratio + args.val_ratio < 1.0, "train + val ratios must be < 1"

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    base_archive = out_dir / "AwA2-base.zip"
    data_archive = out_dir / "AwA2-data.zip"

    # ── 1. Download ───────────────────────────────────────────────────────────
    print("Step 1: Downloading archives...")
    download_file(AWA2_BASE_URL, base_archive)
    download_file(AWA2_DATA_URL, data_archive)

    # ── 2. Extract metadata (base) ────────────────────────────────────────────
    extract_dir = out_dir / "_extracted"
    base_extract = extract_dir / "base"
    if not base_extract.exists():
        print(f"Step 2a: Extracting {base_archive} ...")
        with zipfile.ZipFile(base_archive) as z:
            z.extractall(base_extract)
    else:
        print(f"Step 2a: Already extracted at {base_extract}")

    # ── 3. Extract images (data) ──────────────────────────────────────────────
    data_extract = extract_dir / "data"
    if not data_extract.exists():
        print(f"Step 2b: Extracting {data_archive} (large, may take a while) ...")
        with zipfile.ZipFile(data_archive) as z:
            total = len(z.namelist())
            for i, name in enumerate(z.namelist(), 1):
                z.extract(name, data_extract)
                if i % 5000 == 0:
                    print(f"\r  {i}/{total} files extracted", end="", flush=True)
        print()
    else:
        print(f"Step 2b: Already extracted at {data_extract}")

    # Locate the JPEGImages folder
    jpeg_images = None
    for candidate in [
        data_extract / "Animals_with_Attributes2" / "JPEGImages",
        data_extract / "JPEGImages",
    ]:
        if candidate.exists():
            jpeg_images = candidate
            break
    if jpeg_images is None:
        raise FileNotFoundError(f"Could not find JPEGImages directory under {data_extract}")

    # Locate metadata directory (look for classes.txt)
    meta_dir = None
    for candidate in [
        base_extract / "Animals_with_Attributes2",
        base_extract,
    ]:
        if (candidate / "classes.txt").exists():
            meta_dir = candidate
            break
    if meta_dir is None:
        raise FileNotFoundError(f"Could not find classes.txt under {base_extract}")

    # ── 4. Read class list ─────────────────────────────────────────────────────
    print("Step 3: Parsing class list...")
    class_names = []
    with open(meta_dir / "classes.txt") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(maxsplit=1)
            class_names.append(parts[1])   # e.g. "antelope"
    print(f"  Found {len(class_names)} classes")

    # ── 5. Create per-class splits and copy images ─────────────────────────────
    print("Step 4: Creating splits and copying images...")
    rng = random.Random(args.seed)
    total_train = total_val = total_test = 0

    for cls_name in class_names:
        src_cls_dir = jpeg_images / cls_name
        if not src_cls_dir.exists():
            print(f"  WARNING: {src_cls_dir} not found, skipping")
            continue

        images = sorted(p.name for p in src_cls_dir.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
        rng.shuffle(images)

        n = len(images)
        n_train = int(n * args.train_ratio)
        n_val = int(n * args.val_ratio)

        splits = {
            "train": images[:n_train],
            "val": images[n_train:n_train + n_val],
            "test": images[n_train + n_val:],
        }

        for split_name, split_images in splits.items():
            dst_cls = out_dir / split_name / cls_name
            dst_cls.mkdir(parents=True, exist_ok=True)
            for img_name in split_images:
                src = src_cls_dir / img_name
                dst = dst_cls / img_name
                if not dst.exists():
                    shutil.copy2(src, dst)

        total_train += len(splits["train"])
        total_val += len(splits["val"])
        total_test += len(splits["test"])

    print(f"  Train: {total_train}, Val: {total_val}, Test: {total_test}")

    # ── 6. Copy metadata files ─────────────────────────────────────────────────
    print("Step 5: Copying annotation files...")
    ann_dir = out_dir / "annotations"
    ann_dir.mkdir(exist_ok=True)

    meta_files = [
        "classes.txt",
        "predicates.txt",
        "predicate-matrix-continuous.txt",
        "predicate-matrix-binary.txt",
        "attributes.txt",    # if present (alias for predicates in some versions)
    ]
    for fname in meta_files:
        src = meta_dir / fname
        if src.exists():
            shutil.copy2(src, ann_dir / fname)
            print(f"  Copied {fname}")
        else:
            print(f"  (not found: {fname})")

    print(f"\nDone! Organized dataset at: {out_dir}")
    print(f"  Annotations at: {ann_dir}")

    # ── 7. Cleanup ────────────────────────────────────────────────────────────
    if not args.keep_archives:
        for archive in [base_archive, data_archive]:
            archive.unlink(missing_ok=True)
        print("  Removed archives")


if __name__ == "__main__":
    main()
