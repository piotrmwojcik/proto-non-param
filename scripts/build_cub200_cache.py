"""
Build a CLIP embedding cache for CUB-200-2011 attribute vocabulary.

Reads attributes/attributes.txt (312 lines: "id has_category::value"),
parses the value after "::", cleans underscores, and encodes each with CLIP
using prompt "a bird with {value}".

Saves {word: tensor(512)} dict (same format as build_caltech101_vocab.py).

Usage:
    python scripts/build_cub200_cache.py \
        --annotations-dir /net/tscratch/people/plgabedychaj/cub200/annotations \
        --cache-out /net/tscratch/people/plgabedychaj/vocab/cub200_cache.pt
"""
import argparse
from pathlib import Path

import torch
import open_clip


def parse_attribute_values(attributes_txt: Path) -> list:
    """
    Parse CUB attributes.txt and return list of (attr_id, value_str) pairs.

    Format:  "1 has_bill_shape::dagger"  →  value = "dagger"
    Multi-word values like "has_wing_color::blue_green" → "blue green"
    """
    entries = []
    with open(attributes_txt) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(maxsplit=1)
            attr_id = int(parts[0])
            full_name = parts[1]  # e.g. "has_bill_shape::dagger"
            if "::" in full_name:
                value = full_name.split("::", 1)[1]
            else:
                value = full_name
            # Replace underscores with spaces for readability
            value = value.replace("_", " ").strip()
            entries.append((attr_id, value))
    return entries


@torch.no_grad()
def build_cache(attr_entries: list, cache_out: str):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Building CLIP cache on {device} for {len(attr_entries)} attributes...")

    model, _, _ = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    model = model.eval().to(device)

    cache = {}
    batch_size = 256

    for i in range(0, len(attr_entries), batch_size):
        batch = attr_entries[i : i + batch_size]
        prompts = [f"a bird with {value}" for _, value in batch]
        tokens = tokenizer(prompts).to(device)
        feats = model.encode_text(tokens)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        feats = feats.cpu()
        for (attr_id, value), feat in zip(batch, feats):
            cache[value] = feat
        print(f"  {min(i + batch_size, len(attr_entries))}/{len(attr_entries)}")

    Path(cache_out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(cache, cache_out)
    dim = next(iter(cache.values())).shape[0]
    print(f"Saved cache → {cache_out}  ({len(cache)} entries, dim={dim})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations-dir", required=True, help="Path to CUB annotations dir (contains attributes/attributes.txt)")
    parser.add_argument("--cache-out", default="vocab/cub200_cache.pt", help="Output CLIP cache file")
    args = parser.parse_args()

    ann_dir = Path(args.annotations_dir)
    attributes_txt = ann_dir / "attributes" / "attributes.txt"

    if not attributes_txt.exists():
        # Fallback: attributes.txt directly in annotations dir
        attributes_txt = ann_dir / "attributes.txt"

    if not attributes_txt.exists():
        raise FileNotFoundError(
            f"attributes.txt not found at {ann_dir / 'attributes' / 'attributes.txt'} "
            f"or {ann_dir / 'attributes.txt'}"
        )

    print(f"Reading attributes from: {attributes_txt}")
    attr_entries = parse_attribute_values(attributes_txt)
    print(f"Found {len(attr_entries)} attribute entries")

    # Deduplicate values (keep first occurrence order)
    seen = set()
    unique_entries = []
    for entry in attr_entries:
        if entry[1] not in seen:
            seen.add(entry[1])
            unique_entries.append(entry)
    print(f"Unique attribute values: {len(unique_entries)}")

    build_cache(unique_entries, args.cache_out)


if __name__ == "__main__":
    main()
