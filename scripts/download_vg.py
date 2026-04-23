"""
Download and prepare the Visual Genome v1.4 dataset.

Downloads:
  - Images Part 1 (~9.4 GB) → VG_100K/
  - Images Part 2 (~5.3 GB) → VG_100K_2/
  - region_descriptions.json (~700 MB compressed)

The output directory layout expected by VisualGenomeDataset:

    output_dir/
    ├── VG_100K/
    │   ├── 1.jpg
    │   └── ...
    ├── VG_100K_2/
    │   ├── 108250.jpg
    │   └── ...
    └── region_descriptions.json

Usage:
    python scripts/download_vg.py \
        --output-dir /net/tscratch/people/plgabedychaj/vg \
        --keep-archives
"""

import argparse
import os
import sys
import urllib.request
import zipfile
from pathlib import Path

VG_IMAGES_1_URL = "https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip"
VG_IMAGES_2_URL = "https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip"
VG_REGION_DESC_URL = (
    "https://homes.cs.washington.edu/~ranjay/visualgenome/data/dataset/"
    "region_descriptions.json.zip"
)


def download_file(url: str, dest: Path) -> None:
    if dest.exists():
        print(f"  Already exists, skipping: {dest}")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  Downloading {url}")
    print(f"  → {dest}")

    def _progress(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            pct = min(100, downloaded * 100 // total_size)
            mb = downloaded / 1024 / 1024
            total_mb = total_size / 1024 / 1024
            print(f"\r  {pct:3d}%  {mb:7.1f} / {total_mb:.1f} MB", end="", flush=True)

    try:
        urllib.request.urlretrieve(url, dest, reporthook=_progress)
    except Exception as e:
        dest.unlink(missing_ok=True)
        raise RuntimeError(f"Download failed: {e}") from e
    print()


def extract_zip(archive: Path, dest_dir: Path, strip_top: bool = False) -> None:
    """Extract zip. If strip_top=True, drop the single top-level directory."""
    print(f"  Extracting {archive.name} ...")
    with zipfile.ZipFile(archive) as z:
        members = z.namelist()
        total = len(members)
        for i, name in enumerate(members, 1):
            if strip_top:
                # drop the first path component
                parts = name.split("/", 1)
                if len(parts) < 2 or not parts[1]:
                    continue
                target = dest_dir / parts[1]
            else:
                target = dest_dir / name
            if name.endswith("/"):
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                with z.open(name) as src, open(target, "wb") as dst:
                    dst.write(src.read())
            if i % 5000 == 0:
                print(f"\r  {i}/{total} files", end="", flush=True)
    print(f"\r  Done ({total} files extracted)        ")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download Visual Genome v1.4")
    parser.add_argument("--output-dir", required=True,
                        help="Destination directory for images and annotations")
    parser.add_argument("--keep-archives", action="store_true",
                        help="Keep downloaded .zip files after extraction")
    parser.add_argument("--skip-images", action="store_true",
                        help="Skip image download (useful if images already present)")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    archives_dir = out_dir / "_archives"
    archives_dir.mkdir(exist_ok=True)

    # ------------------------------------------------------------------ #
    # 1. Images Part 1  →  VG_100K/
    # ------------------------------------------------------------------ #
    vg1_dir = out_dir / "VG_100K"
    if args.skip_images and vg1_dir.exists():
        print(f"Skipping images part 1 (directory exists: {vg1_dir})")
    else:
        archive1 = archives_dir / "images.zip"
        print("==> Step 1/3: Images Part 1")
        download_file(VG_IMAGES_1_URL, archive1)
        if not vg1_dir.exists() or not any(vg1_dir.iterdir()):
            extract_zip(archive1, out_dir)
        else:
            print(f"  Already extracted at {vg1_dir}")
        n1 = sum(1 for _ in vg1_dir.glob("*.jpg"))
        print(f"  {n1} images in VG_100K/")

    # ------------------------------------------------------------------ #
    # 2. Images Part 2  →  VG_100K_2/
    # ------------------------------------------------------------------ #
    vg2_dir = out_dir / "VG_100K_2"
    if args.skip_images and vg2_dir.exists():
        print(f"Skipping images part 2 (directory exists: {vg2_dir})")
    else:
        archive2 = archives_dir / "images2.zip"
        print("==> Step 2/3: Images Part 2")
        download_file(VG_IMAGES_2_URL, archive2)
        if not vg2_dir.exists() or not any(vg2_dir.iterdir()):
            extract_zip(archive2, out_dir)
        else:
            print(f"  Already extracted at {vg2_dir}")
        n2 = sum(1 for _ in vg2_dir.glob("*.jpg"))
        print(f"  {n2} images in VG_100K_2/")

    # ------------------------------------------------------------------ #
    # 3. region_descriptions.json
    # ------------------------------------------------------------------ #
    region_json = out_dir / "region_descriptions.json"
    print("==> Step 3/3: region_descriptions.json")
    if region_json.exists():
        print(f"  Already exists: {region_json}")
    else:
        archive_rd = archives_dir / "region_descriptions.json.zip"
        download_file(VG_REGION_DESC_URL, archive_rd)
        with zipfile.ZipFile(archive_rd) as z:
            names = [n for n in z.namelist() if n.endswith(".json")]
            if not names:
                print("ERROR: no .json found inside region_descriptions.json.zip",
                      file=sys.stderr)
                sys.exit(1)
            print(f"  Extracting {names[0]} ...")
            with z.open(names[0]) as src, open(region_json, "wb") as dst:
                dst.write(src.read())
        print(f"  Saved → {region_json}")

    # ------------------------------------------------------------------ #
    # 4. Cleanup
    # ------------------------------------------------------------------ #
    if not args.keep_archives:
        import shutil
        shutil.rmtree(archives_dir, ignore_errors=True)
        print("Removed archive directory")

    print(f"\nDone.  Dataset root: {out_dir}")
    print(f"  region_descriptions.json: {region_json}")


if __name__ == "__main__":
    main()
