"""
Build a CLIP embedding cache for Label-free-CBM's CUB concept phrases.

Unlike build_cub200_cache.py (parses CUB's official 312 "id has_category::value"
attributes.txt), this reads a plain one-phrase-per-line file (Trustworthy-ML-Lab/
Label-free-CBM's cub_filtered.txt, e.g. "a black cap and back") and encodes each
phrase with CLIP verbatim -- no template wrapping, matching Label-free-CBM's own
protocol (their phrases are already complete descriptive fragments).

Saves {phrase: tensor(512)} dict (same format as build_cub200_cache.py /
build_vg_caption_embeddings.py), consumed by PNP.__init__'s vocab_cache_path and
by build_clip_vocab_scores.py's --vocab-cache.

Usage:
    python scripts/build_cub_concept_cache.py \
        --concepts-file $SCRATCH/vocab/cub_filtered_concepts.txt \
        --cache-out $SCRATCH/vocab/cub_labelfreecbm_cache.pt
"""
import argparse
from pathlib import Path

import torch
import open_clip


def read_concepts(concepts_file: Path) -> list:
    seen, out = set(), []
    with open(concepts_file, encoding="utf-8") as f:
        for line in f:
            phrase = line.strip()
            if phrase and phrase not in seen:
                seen.add(phrase)
                out.append(phrase)
    return out


@torch.no_grad()
def build_cache(concepts: list, cache_out: str, clip_model_name: str, clip_pretrained: str):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Building CLIP cache on {device} for {len(concepts)} concepts...")

    model, _, _ = open_clip.create_model_and_transforms(clip_model_name, pretrained=clip_pretrained)
    tokenizer = open_clip.get_tokenizer(clip_model_name)
    model = model.eval().to(device)

    cache = {}
    batch_size = 256

    for i in range(0, len(concepts), batch_size):
        batch = concepts[i : i + batch_size]
        tokens = tokenizer(batch).to(device)
        feats = model.encode_text(tokens)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        feats = feats.cpu()
        for phrase, feat in zip(batch, feats):
            cache[phrase] = feat
        print(f"  {min(i + batch_size, len(concepts))}/{len(concepts)}")

    Path(cache_out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(cache, cache_out)
    dim = next(iter(cache.values())).shape[0]
    print(f"Saved cache -> {cache_out}  ({len(cache)} entries, dim={dim})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--concepts-file", required=True, help="Plain text file, one concept phrase per line")
    parser.add_argument("--cache-out", default="vocab/cub_labelfreecbm_cache.pt", help="Output CLIP cache file")
    parser.add_argument("--clip-model", default="ViT-B-32", help="open_clip model name (match the PNP checkpoint's coco_clip_model_name)")
    parser.add_argument("--clip-pretrained", default="openai")
    args = parser.parse_args()

    concepts = read_concepts(Path(args.concepts_file))
    print(f"Read {len(concepts)} unique concepts from {args.concepts_file}")

    build_cache(concepts, args.cache_out, args.clip_model, args.clip_pretrained)


if __name__ == "__main__":
    main()
