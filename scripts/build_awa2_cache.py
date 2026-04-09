"""
Build a CLIP embedding cache for AwA2 predicate vocabulary.

Reads annotations/predicates.txt (85 lines: "id  predicate_name") and
encodes each predicate with CLIP using prompt "a photo of a {predicate} animal".

Saves {predicate: tensor(512)} dict (same format as build_caltech101_vocab.py).

Usage:
    python scripts/build_awa2_cache.py \
        --annotations-dir /net/tscratch/people/plgabedychaj/awa2/annotations \
        --cache-out /net/tscratch/people/plgabedychaj/vocab/awa2_cache.pt
"""
import argparse
from pathlib import Path

import torch
import open_clip


def parse_predicates(predicates_txt: Path) -> list:
    """Return list of (predicate_id, predicate_name) from predicates.txt."""
    entries = []
    with open(predicates_txt) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(maxsplit=1)
            pred_id = int(parts[0])
            pred_name = parts[1].replace("+", " ").replace("_", " ").strip()
            entries.append((pred_id, pred_name))
    return entries


@torch.no_grad()
def build_cache(pred_entries: list, cache_out: str):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Building CLIP cache on {device} for {len(pred_entries)} predicates...")

    model, _, _ = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    model = model.eval().to(device)

    cache = {}
    batch_size = 256

    for i in range(0, len(pred_entries), batch_size):
        batch = pred_entries[i : i + batch_size]
        prompts = [f"a photo of a {name} animal" for _, name in batch]
        tokens = tokenizer(prompts).to(device)
        feats = model.encode_text(tokens)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        feats = feats.cpu()
        for (pred_id, name), feat in zip(batch, feats):
            cache[name] = feat
        print(f"  {min(i + batch_size, len(pred_entries))}/{len(pred_entries)}")

    Path(cache_out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(cache, cache_out)
    dim = next(iter(cache.values())).shape[0]
    print(f"Saved cache → {cache_out}  ({len(cache)} entries, dim={dim})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations-dir", required=True, help="Path to AwA2 annotations dir (contains predicates.txt)")
    parser.add_argument("--cache-out", default="vocab/awa2_cache.pt", help="Output CLIP cache file")
    args = parser.parse_args()

    ann_dir = Path(args.annotations_dir)
    predicates_txt = ann_dir / "predicates.txt"

    if not predicates_txt.exists():
        raise FileNotFoundError(f"predicates.txt not found at {predicates_txt}")

    print(f"Reading predicates from: {predicates_txt}")
    pred_entries = parse_predicates(predicates_txt)
    print(f"Found {len(pred_entries)} predicates")

    build_cache(pred_entries, args.cache_out)


if __name__ == "__main__":
    main()
