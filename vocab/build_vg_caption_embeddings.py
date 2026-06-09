#!/usr/bin/env python3
"""Build per-image CLIP phrase embedding pools from Visual Genome region descriptions.

Unlike build_vg_vocab.py (word-level) or build_caption_prototypes.py (inference-time
phrase strings), this script produces a *per-image* tensor of shape [N≤pool_size, 512]:
the deduplicated, CLIP-encoded, L2-normalised phrase embeddings for each training (or val)
image.

At training time VisualGenomeDataset randomly samples sample_k rows from this pool and
mean-pools them into a single [512] caption embedding — giving stochastic augmentation
without re-encoding text every forward pass.

Usage:
    python vocab/build_vg_caption_embeddings.py \\
        --region-descriptions /data/vg/region_descriptions.json \\
        --vg-root /data/vg \\
        --vocab-cache-path vocab/vg_cache.pt \\
        --cache-out vocab/vg_caption_embs.pt \\
        --split both \\
        --pool-size 50
"""

import argparse
import sys
from pathlib import Path

import open_clip
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _encode_batch(texts: list[str], model, tokenizer, device) -> torch.Tensor:
    tokens = tokenizer(texts).to(device)
    with torch.no_grad():
        embs = model.encode_text(tokens)
    embs = F.normalize(embs, dim=-1)
    return embs.cpu()


def build_cache(
    vg_root: str,
    region_descriptions_path: str,
    vocab_cache_path: str,
    split: str,
    pool_size: int,
    seed: int,
    val_ratio: float,
    clip_model_name: str,
    clip_pretrained: str,
    batch_size: int,
    device: str,
    max_images: int,
) -> dict[str, torch.Tensor]:
    from clip_dataset import VisualGenomeDataset

    print(f"Loading VG word vocab from {vocab_cache_path} (needed to reproduce split) ...")
    vocab_cache = torch.load(vocab_cache_path, map_location="cpu")
    vocab_to_idx = {w: i for i, w in enumerate(vocab_cache.keys())}
    print(f"  vocab size: {len(vocab_to_idx)}")

    splits = ["train", "val"] if split == "both" else [split]

    device_obj = torch.device(device if torch.cuda.is_available() else "cpu")
    print(f"Loading CLIP {clip_model_name} ({clip_pretrained}) on {device_obj} ...")
    model, _, _ = open_clip.create_model_and_transforms(clip_model_name, pretrained=clip_pretrained)
    model = model.eval().to(device_obj)
    tokenizer = open_clip.get_tokenizer(clip_model_name)

    result: dict[str, torch.Tensor] = {}

    for s in splits:
        print(f"\n=== Processing split: {s} ===")
        dataset = VisualGenomeDataset(
            vg_root=vg_root,
            region_descriptions_json=region_descriptions_path,
            vocab_to_idx=vocab_to_idx,
            train=(s == "train"),
            val_ratio=val_ratio,
            seed=seed,
        )
        print(f"  {len(dataset)} images in {s} split")

        # Collect all unique phrases globally to encode efficiently
        # Maps phrase → index in global phrase list
        global_phrases: list[str] = []
        phrase_to_idx: dict[str, int] = {}
        per_image: list[tuple[str, list[int]]] = []  # (im_path, [phrase_idx, ...])

        n_images = len(dataset.samples) if not max_images else min(max_images, len(dataset.samples))
        for i in range(n_images):
            im_path, phrases, _ = dataset.samples[i]
            # Deduplicate while preserving first-occurrence order
            seen: dict[str, str] = {}
            for p in phrases:
                key = p.strip().lower()
                if key and key not in seen:
                    seen[key] = p.strip()
            unique = list(seen.values())[:pool_size]

            phrase_idxs = []
            for p in unique:
                if p not in phrase_to_idx:
                    phrase_to_idx[p] = len(global_phrases)
                    global_phrases.append(p)
                phrase_idxs.append(phrase_to_idx[p])
            per_image.append((im_path, phrase_idxs))

        print(f"  {len(global_phrases)} unique phrases across {n_images} images")
        print(f"  Encoding in batches of {batch_size} ...")

        # Encode all phrases
        all_embs = torch.zeros(len(global_phrases), 512)
        for start in range(0, len(global_phrases), batch_size):
            if start % (batch_size * 20) == 0:
                print(f"    {start}/{len(global_phrases)} ...", end="\r")
            batch = global_phrases[start: start + batch_size]
            all_embs[start: start + len(batch)] = _encode_batch(batch, model, tokenizer, device_obj)
        print(f"\n  Done encoding.")

        for im_path, phrase_idxs in per_image:
            if phrase_idxs:
                embs = all_embs[phrase_idxs]  # [N, 512]
                result[im_path] = embs

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Build per-image CLIP phrase embedding pools for VG")
    parser.add_argument("--region-descriptions", required=True,
                        help="Path to VG region_descriptions.json")
    parser.add_argument("--vg-root", required=True,
                        help="Path to VG image root (contains VG_100K/, VG_100K_2/)")
    parser.add_argument("--vocab-cache-path", required=True,
                        help="Path to VG word vocab cache (.pt) — needed to reproduce the split")
    parser.add_argument("--cache-out", required=True,
                        help="Output path for the per-image embedding cache (.pt)")
    parser.add_argument("--split", default="train", choices=["train", "val", "both"],
                        help="Which split(s) to process (default: train)")
    parser.add_argument("--pool-size", type=int, default=50,
                        help="Max unique phrases per image to keep (default: 50)")
    parser.add_argument("--seed", type=int, default=42,
                        help="RNG seed for VG train/val split — must match training (default: 42)")
    parser.add_argument("--val-ratio", type=float, default=0.1,
                        help="Fraction of VG images in the val split — must match training (default: 0.1)")
    parser.add_argument("--clip-model-name", default="ViT-B-32")
    parser.add_argument("--clip-pretrained", default="openai")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-images", type=int, default=0,
                        help="Cap on number of images to process per split (0 = no cap; for debugging)")
    args = parser.parse_args()

    cache = build_cache(
        vg_root=args.vg_root,
        region_descriptions_path=args.region_descriptions,
        vocab_cache_path=args.vocab_cache_path,
        split=args.split,
        pool_size=args.pool_size,
        seed=args.seed,
        val_ratio=args.val_ratio,
        clip_model_name=args.clip_model_name,
        clip_pretrained=args.clip_pretrained,
        batch_size=args.batch_size,
        device=args.device,
        max_images=args.max_images,
    )

    out_path = Path(args.cache_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cache, out_path)
    print(f"Saved per-image caption embedding cache ({len(cache)} images) → {out_path}")


if __name__ == "__main__":
    main()
