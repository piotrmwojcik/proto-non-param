#!/usr/bin/env python3
"""Build a CLIP-embedding cache from full caption/phrase strings (not individual words).

Unlike build_vg_vocab.py (which extracts single words via NLTK), this script encodes
raw caption strings as prototypes — enabling phrase-level prototype augmentation at
inference time without retraining.

Two sources are supported via --source:

  coco    Load unique captions from a COCO annotations JSON (train or val).
          Output: vocab/coco_caption_prototypes.pt

  vg_test Extract unique region phrases from the Visual Genome *validation* split.
          Uses VisualGenomeDataset(train=False) directly so the split is guaranteed
          to be identical to the one used during training (same filtering + shuffle).
          Requires --vocab-cache-path (the VG word cache) to instantiate the dataset.
          Output: vocab/vg_test_caption_prototypes.pt

The output cache has the same format as the word vocab caches:
    dict[str, Tensor[clip_text_dim]]   — {phrase: unit-normalised CLIP embedding}

Usage:
    # VG test phrases (exact same split as training)
    python vocab/build_caption_prototypes.py \\
        --source vg_test \\
        --region-descriptions /data/vg/region_descriptions.json \\
        --vg-root /data/vg \\
        --vocab-cache-path vocab/vg_cache.pt \\
        --cache-out vocab/vg_test_caption_prototypes.pt \\
        --seed 42 --val-ratio 0.1

    # COCO captions
    python vocab/build_caption_prototypes.py \\
        --source coco \\
        --coco-annotations /data/coco/annotations/captions_val2014.json \\
        --cache-out vocab/coco_caption_prototypes.pt \\
        --max-captions 50000
"""

import argparse
import json
import sys
from pathlib import Path

import open_clip
import torch

# allow importing from the parent package
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _load_unique_coco_captions(annotations_path: str, max_captions: int, min_words: int = 0) -> list[str]:
    print(f"Loading COCO annotations from {annotations_path} ...")
    with open(annotations_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    captions = sorted({ann["caption"].strip() for ann in data["annotations"] if ann["caption"].strip()})
    if min_words:
        before = len(captions)
        captions = [c for c in captions if len(c.split()) >= min_words]
        print(f"  {before} unique captions → {len(captions)} after min_words≥{min_words} filter")
    else:
        print(f"  {len(captions)} unique captions found")
    if max_captions and len(captions) > max_captions:
        captions = captions[:max_captions]
        print(f"  capped to {max_captions} captions")
    return captions


def _load_vg_test_phrases(
    vg_root: str,
    region_descriptions_path: str,
    vocab_cache_path: str,
    seed: int,
    val_ratio: float,
    max_captions: int,
    min_words: int = 0,
) -> list[str]:
    """Extract unique phrases from the VG validation split.

    Instantiates VisualGenomeDataset(train=False) with the same parameters used during
    training so the split (filtering + shuffle) is byte-for-byte identical.  The vocab
    cache is needed only to reproduce the filtering step — no word-level logic is applied
    to the output phrases themselves.
    """
    from clip_dataset import VisualGenomeDataset

    print(f"Loading VG word vocab from {vocab_cache_path} (needed to reproduce split) ...")
    vocab_cache = torch.load(vocab_cache_path, map_location="cpu")
    vocab_to_idx = {w: i for i, w in enumerate(vocab_cache.keys())}
    print(f"  vocab size: {len(vocab_to_idx)}")

    print("Instantiating VisualGenomeDataset(train=False) to get exact val split ...")
    dataset = VisualGenomeDataset(
        vg_root=vg_root,
        region_descriptions_json=region_descriptions_path,
        vocab_to_idx=vocab_to_idx,
        train=False,
        val_ratio=val_ratio,
        seed=seed,
    )
    print(f"  val split: {len(dataset)} images")

    # Each sample is (im_path, phrases_list, prob_dist)
    phrases: set[str] = set()
    for im_path, img_phrases, _ in dataset.samples:
        for p in img_phrases:
            p = p.strip()
            if p and (not min_words or len(p.split()) >= min_words):
                phrases.add(p)

    phrases_list = sorted(phrases)
    print(f"  {len(phrases_list)} unique phrases from val split (min_words≥{min_words})")
    if max_captions and len(phrases_list) > max_captions:
        phrases_list = phrases_list[:max_captions]
        print(f"  capped to {max_captions} phrases")
    return phrases_list


def _encode_texts(
    texts: list[str],
    model_name: str,
    pretrained: str,
    batch_size: int,
    device: str,
) -> dict[str, torch.Tensor]:
    device_obj = torch.device(device if torch.cuda.is_available() else "cpu")
    print(f"Loading CLIP model {model_name} ({pretrained}) on {device_obj} ...")
    model, _, _ = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
    model = model.eval().to(device_obj)
    tokenizer = open_clip.get_tokenizer(model_name)

    cache: dict[str, torch.Tensor] = {}
    n = len(texts)
    print(f"Encoding {n} texts in batches of {batch_size} ...")
    with torch.no_grad():
        for start in range(0, n, batch_size):
            if start % (batch_size * 20) == 0:
                print(f"  {start}/{n} ...", end="\r")
            batch = texts[start: start + batch_size]
            tokens = tokenizer(batch).to(device_obj)
            embs = model.encode_text(tokens)
            embs = embs / embs.norm(dim=-1, keepdim=True)
            for text, emb in zip(batch, embs.cpu()):
                cache[text] = emb
    print(f"\nEncoded {len(cache)} texts")
    return cache


def main() -> None:
    parser = argparse.ArgumentParser(description="Build phrase-level CLIP prototype cache")
    parser.add_argument("--source", type=str, required=True, choices=["coco", "vg_test"],
                        help="coco: COCO captions JSON | vg_test: VG val-split region phrases")

    # VG args
    parser.add_argument("--vg-root", type=str, default=None,
                        help="Path to VG image root (required for --source vg_test)")
    parser.add_argument("--region-descriptions", type=str, default=None,
                        help="Path to VG region_descriptions.json (required for --source vg_test)")
    parser.add_argument("--vocab-cache-path", type=str, default=None,
                        help="Path to VG word vocab cache (.pt) — needed to reproduce the "
                             "training split exactly (required for --source vg_test)")
    parser.add_argument("--seed", type=int, default=42,
                        help="RNG seed for VG train/val split — must match training (default: 42)")
    parser.add_argument("--val-ratio", type=float, default=0.1,
                        help="Fraction of VG images in the val split — must match training (default: 0.1)")

    # COCO args
    parser.add_argument("--coco-annotations", type=str, default=None,
                        help="Path to COCO captions JSON (required for --source coco)")

    # Shared
    parser.add_argument("--cache-out", type=str, required=True,
                        help="Output path for the prototype cache (.pt)")
    parser.add_argument("--clip-model-name", type=str, default="ViT-B-32")
    parser.add_argument("--clip-pretrained", type=str, default="openai")
    parser.add_argument("--max-captions", type=int, default=0,
                        help="Cap on number of phrases (0 = no cap)")
    parser.add_argument("--min-words", type=int, default=5,
                        help="Minimum number of words in a phrase (default: 5, removes "
                             "short single-object labels like 'a desk' or 'sky')")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    if args.source == "vg_test":
        if not args.vg_root or not args.region_descriptions or not args.vocab_cache_path:
            parser.error("--vg-root, --region-descriptions, and --vocab-cache-path are all "
                         "required for --source vg_test")
        texts = _load_vg_test_phrases(
            vg_root=args.vg_root,
            region_descriptions_path=args.region_descriptions,
            vocab_cache_path=args.vocab_cache_path,
            seed=args.seed,
            val_ratio=args.val_ratio,
            max_captions=args.max_captions,
            min_words=args.min_words,
        )
    else:
        if args.coco_annotations is None:
            parser.error("--coco-annotations is required for --source coco")
        texts = _load_unique_coco_captions(args.coco_annotations, args.max_captions, args.min_words)

    cache = _encode_texts(texts, args.clip_model_name, args.clip_pretrained, args.batch_size, args.device)

    out_path = Path(args.cache_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cache, out_path)
    print(f"Saved caption prototype cache ({len(cache)} entries) → {out_path}")


if __name__ == "__main__":
    main()
