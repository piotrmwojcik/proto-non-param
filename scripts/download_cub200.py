"""
Download and organize CUB-200-2011 dataset.

Downloads CUB_200_2011.tgz, parses annotation files to create train/val/test
splits, copies images into class subdirectories, and copies all annotation
files needed at training time.

Usage:
    python scripts/download_cub200.py \
        --output-dir /net/tscratch/people/plgabedychaj/cub200 \
        --val-per-class 5 \
        --seed 42
"""
import argparse
import os
import random
import shutil
import tarfile
import urllib.request
from collections import defaultdict
from pathlib import Path

CUB_URL = "https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz"
ARCHIVE_NAME = "CUB_200_2011.tgz"


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
            print(f"\r  {pct}%  ({downloaded // 1024 // 1024} MB / {total_size // 1024 // 1024} MB)", end="", flush=True)

    urllib.request.urlretrieve(url, dest, reporthook=_progress)
    print()


def parse_lines(path: Path):
    """Return list of (id, value) pairs from CUB annotation files."""
    entries = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(maxsplit=1)
            entries.append((int(parts[0]), parts[1]))
    return entries


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True, help="Destination directory for organized dataset")
    parser.add_argument("--val-per-class", type=int, default=5, help="Number of val images per class (carved from train)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--keep-archive", action="store_true", help="Keep downloaded .tgz archive after extraction")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    archive_path = out_dir / ARCHIVE_NAME

    # ── 1. Download ──────────────────────────────────────────────────────────
    print("Step 1: Downloading archive...")
    download_file(CUB_URL, archive_path)

    # ── 2. Extract ───────────────────────────────────────────────────────────
    extract_dir = out_dir / "_extracted"
    if not extract_dir.exists():
        print(f"Step 2: Extracting {archive_path} → {extract_dir} ...")
        with tarfile.open(archive_path, "r:gz") as tar:
            tar.extractall(extract_dir)
    else:
        print(f"Step 2: Already extracted at {extract_dir}")

    cub_root = extract_dir / "CUB_200_2011"

    # ── 3. Parse annotation files ─────────────────────────────────────────────
    print("Step 3: Parsing annotations...")

    images = dict(parse_lines(cub_root / "images.txt"))          # id → filename
    train_test = dict(parse_lines(cub_root / "train_test_split.txt"))  # id → "1"=train / "0"=test
    image_labels = dict(parse_lines(cub_root / "image_class_labels.txt"))  # id → class_id
    classes = dict(parse_lines(cub_root / "classes.txt"))         # class_id → class_name

    # Group train image ids per class
    train_ids_per_class = defaultdict(list)
    test_ids = []
    for img_id, filename in images.items():
        is_train = train_test[img_id] == "1"
        if is_train:
            cls_id = image_labels[img_id]
            train_ids_per_class[cls_id].append(img_id)
        else:
            test_ids.append(img_id)

    # Carve val from train
    rng = random.Random(args.seed)
    train_ids = []
    val_ids = []
    for cls_id, ids in sorted(train_ids_per_class.items()):
        ids_sorted = sorted(ids)
        rng.shuffle(ids_sorted)
        val_ids.extend(ids_sorted[: args.val_per_class])
        train_ids.extend(ids_sorted[args.val_per_class :])

    print(f"  Train: {len(train_ids)}, Val: {len(val_ids)}, Test: {len(test_ids)}")

    # ── 4. Copy images ────────────────────────────────────────────────────────
    print("Step 4: Copying images...")

    def copy_split(ids, split_name):
        split_dir = out_dir / split_name
        count = 0
        for img_id in ids:
            rel = images[img_id]          # e.g. "001.Black_footed_Albatross/..."
            cls_folder = rel.split("/")[0]
            # Map "001.Black_footed_Albatross" → "Black_footed_Albatross"
            class_name = ".".join(cls_folder.split(".")[1:])
            dst_dir = split_dir / class_name
            dst_dir.mkdir(parents=True, exist_ok=True)
            src = cub_root / "images" / rel
            dst = dst_dir / Path(rel).name
            if not dst.exists():
                shutil.copy2(src, dst)
            count += 1
        print(f"  {split_name}: {count} images")

    copy_split(train_ids, "train")
    copy_split(val_ids, "val")
    copy_split(test_ids, "test")

    # ── 5. Copy annotation files ───────────────────────────────────────────────
    print("Step 5: Copying annotation files...")
    ann_dir = out_dir / "annotations"
    ann_dir.mkdir(exist_ok=True)

    # Top-level annotation files
    for fname in ["images.txt", "train_test_split.txt", "image_class_labels.txt", "classes.txt"]:
        src = cub_root / fname
        if src.exists():
            shutil.copy2(src, ann_dir / fname)

    # attributes/ subdirectory (312 binary attributes + per-image labels)
    src_attrs = cub_root / "attributes"
    dst_attrs = ann_dir / "attributes"
    if src_attrs.exists() and not dst_attrs.exists():
        shutil.copytree(src_attrs, dst_attrs)
        print(f"  Copied attributes/ → {dst_attrs}")
    elif not src_attrs.exists():
        print("  WARNING: attributes/ subdirectory not found in archive!")

    # ── 6. Save val split file for reference ──────────────────────────────────
    with open(ann_dir / "val_ids.txt", "w") as f:
        for img_id in sorted(val_ids):
            f.write(f"{img_id} {images[img_id]}\n")

    print(f"\nDone! Organized dataset at: {out_dir}")
    print(f"  Annotations at: {ann_dir}")

    # ── 7. Cleanup ────────────────────────────────────────────────────────────
    if not args.keep_archive:
        archive_path.unlink(missing_ok=True)
        print(f"  Removed archive {archive_path}")


if __name__ == "__main__":
    main()
