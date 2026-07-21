#!/usr/bin/env python3
"""
build_clip_vocab_scores.py

For each image in COCO or Visual Genome, compute per-image soft-label distributions
over the vocabulary using CLIP vision-text dot products:

    scores[i, j] = CLIP_vision(image_i) · CLIP_text(vocab_word_j)
    soft_labels   = softmax(scores / temperature)

Optionally blends the CLIP-derived distribution with pre-computed caption statistics:

    mixed = alpha * clip_soft_labels + (1 - alpha) * caption_stats

Output is a .pt dict with keys:
    image_ids        : List[int]        — COCO/VG image ids
    vocab            : List[str]        — ordered vocabulary words
    clip_scores      : Tensor[N, V]     — raw cosine similarities (float16)
    clip_soft_labels : Tensor[N, V]     — temperature-softmax of clip_scores (float16)
    mixed_labels     : Tensor[N, V]     — blended labels (float16), only if --caption-stats given
    alpha            : float
    temperature      : float
    clip_model       : str
    clip_pretrained  : str
    vocab_cache      : str
    dataset          : str

Usage (COCO):
    python build_clip_vocab_scores.py \\
        --dataset coco \\
        --data-root /path/to/coco \\
        --annotations /path/to/captions_train2017.json \\
        --vocab-cache vocab/mscoco_new_cache.pt \\
        --output vocab/coco_train_clip_scores.pt

Usage (VG):
    python build_clip_vocab_scores.py \\
        --dataset vg \\
        --data-root /path/to/vg \\
        --annotations /path/to/region_descriptions.json \\
        --vocab-cache vocab/vg_cache.pt \\
        --output vocab/vg_clip_scores.pt

Usage (CUB, with Label-free-CBM-style CLIP-cutoff concept filtering):
    python build_clip_vocab_scores.py \\
        --dataset cub \\
        --data-root /path/to/cub200 \\
        --annotations /path/to/cub200/annotations \\
        --vocab-cache vocab/cub_labelfreecbm_cache.pt \\
        --output vocab/cub_clip_scores.pt \\
        --concept-clip-cutoff 0.25
    # With --concept-clip-cutoff set, also writes:
    #   vocab/cub_clip_scores_concepts_filtered.txt  (surviving concept phrases)
    #   vocab/cub_clip_scores_vocab_filtered.pt      (surviving concepts' CLIP cache)
    # and --output itself is rewritten to the filtered [N, V_kept] scores.
"""

import argparse
import json
import os
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
import open_clip
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


class _ImageDataset(Dataset):
    def __init__(self, paths: list[str], ids: list[int], transform):
        self.paths = paths
        self.ids = ids
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx: int):
        img = Image.open(self.paths[idx]).convert("RGB")
        return self.transform(img), self.ids[idx]


def _load_coco_images(data_root: str, annotations_json: str) -> tuple[list, list]:
    with open(annotations_json) as f:
        data = json.load(f)

    id_to_file = {img["id"]: img["file_name"] for img in data["images"]}

    if "annotations" in data:
        image_ids = sorted(set(int(ann["image_id"]) for ann in data["annotations"]))
    else:
        image_ids = sorted(id_to_file.keys())

    paths, valid_ids = [], []
    for image_id in image_ids:
        file_name = id_to_file.get(image_id)
        if file_name is None:
            continue
        for subdir in ("train2017", "val2017", "train2014", "val2014", ""):
            candidate = (
                os.path.join(data_root, subdir, file_name)
                if subdir
                else os.path.join(data_root, file_name)
            )
            if os.path.isfile(candidate):
                paths.append(candidate)
                valid_ids.append(image_id)
                break

    return paths, valid_ids


def _load_vg_images(vg_root: str, region_descriptions_json: str) -> tuple[list, list]:
    with open(region_descriptions_json) as f:
        raw = json.load(f)

    paths, ids = [], []
    for entry in raw:
        image_id = int(entry["id"])
        for shard in ("VG_100K", "VG_100K_2"):
            candidate = os.path.join(vg_root, shard, f"{image_id}.jpg")
            if os.path.isfile(candidate):
                paths.append(candidate)
                ids.append(image_id)
                break

    return paths, ids


def _load_cub_images(cub_root: str, annotations_dir: str) -> tuple[list, list]:
    """train + val splits (both get CLIPScoreDataset-wrapped in train.py's cub200
    branch, mirroring dataset_train/dataset_test symmetry). Reuses the same
    images.txt-keyed path->id map CLIPScoreDataset's CUB id_fn uses, so the ids
    produced here line up exactly with what train.py will look up later."""
    from clip_dataset import build_cub_path_to_id

    path_to_id = build_cub_path_to_id(cub_root, annotations_dir)
    paths = sorted(path_to_id.keys())
    ids = [path_to_id[p] for p in paths]
    return paths, ids


def _load_vocab_embeddings(vocab_cache_path: str) -> tuple[list, torch.Tensor]:
    cache = torch.load(vocab_cache_path, map_location="cpu")
    vocab = list(cache.keys())
    vocab_emb = torch.stack([cache[w] for w in vocab], dim=0)
    vocab_emb = F.normalize(vocab_emb, dim=-1)
    return vocab, vocab_emb


def _load_caption_stats(path: str) -> dict[int, torch.Tensor]:
    """Load pre-computed caption stats into {image_id: prob_dist} mapping.

    Supports two formats:
      1. Direct dict {image_id: tensor}
      2. Cached dataset format {"samples": [(im_path, captions, prob_dist), ...]}
         (produced by CocoCLIPDataset / VisualGenomeDataset with use_cache=True)
    """
    data = torch.load(path, map_location="cpu")

    if isinstance(data, dict) and "samples" in data:
        result = {}
        for im_path, _, prob_dist in data["samples"]:
            stem = Path(im_path).stem
            try:
                result[int(stem)] = prob_dist
            except ValueError:
                pass
        return result

    # Assume it is already {image_id: tensor}
    return {int(k): v for k, v in data.items()}


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(
        description="Build per-image CLIP vocab scores for COCO or Visual Genome"
    )
    parser.add_argument("--dataset", required=True, choices=["coco", "vg", "cub"])
    parser.add_argument("--data-root", required=True, help="Root directory of the dataset")
    parser.add_argument(
        "--annotations",
        default=None,
        help="Captions JSON for COCO, region_descriptions.json path for VG, or the "
             "CUB annotations dir (contains images.txt) for cub. For VG defaults to "
             "<data-root>/region_descriptions.json; for cub defaults to "
             "<data-root>/annotations.",
    )
    parser.add_argument(
        "--concept-clip-cutoff", type=float, default=None,
        help="Label-free-CBM-style concept filter (their default: 0.25): after "
             "computing scores, keep only concepts whose mean top-5 image "
             "activation exceeds this cutoff. When set, also writes a filtered "
             "concept list and a filtered vocab cache alongside --output (see "
             "--filtered-concepts-out / --filtered-vocab-cache-out). Off by default "
             "so existing coco/vg invocations are unaffected.",
    )
    parser.add_argument(
        "--filtered-concepts-out", default=None,
        help="Output path for the surviving concept list (only used with "
             "--concept-clip-cutoff). Default: <output>_concepts_filtered.txt",
    )
    parser.add_argument(
        "--filtered-vocab-cache-out", default=None,
        help="Output path for the surviving concepts' CLIP cache (only used with "
             "--concept-clip-cutoff). Default: <output>_vocab_filtered.pt",
    )
    parser.add_argument("--vocab-cache", required=True, help="Path to vocab .pt cache {word: tensor}")
    parser.add_argument("--clip-model", default="ViT-B-32", help="open_clip model name")
    parser.add_argument("--clip-pretrained", default="openai", help="open_clip pretrained weights tag")
    parser.add_argument(
        "--temperature", type=float, default=0.07,
        help="Softmax temperature τ (lower → sharper distribution)"
    )
    parser.add_argument(
        "--alpha", type=float, default=0.5,
        help="Blend weight for CLIP scores. 1.0 = pure CLIP, 0.0 = pure caption stats. "
             "Only used when --caption-stats is provided."
    )
    parser.add_argument(
        "--caption-stats", default=None,
        help="Optional .pt file with pre-computed caption statistics for blending"
    )
    parser.add_argument("--output", required=True, help="Output .pt path")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    # --- CLIP model ---
    print(f"Loading CLIP {args.clip_model} ({args.clip_pretrained})...")
    clip_model, _, preprocess = open_clip.create_model_and_transforms(
        args.clip_model, pretrained=args.clip_pretrained
    )
    clip_model = clip_model.eval().to(device)
    for p in clip_model.parameters():
        p.requires_grad_(False)

    # --- Vocabulary ---
    print(f"Loading vocab from: {args.vocab_cache}")
    vocab, vocab_emb = _load_vocab_embeddings(args.vocab_cache)
    vocab_emb = vocab_emb.to(device)  # [V, D]
    V = len(vocab)
    D = vocab_emb.shape[1]
    print(f"Vocab size: {V}, embedding dim: {D}")

    # --- Image list ---
    print(f"Building {args.dataset} image list...")
    if args.dataset == "coco":
        if args.annotations is None:
            parser.error("--annotations is required for --dataset coco")
        image_paths, image_ids = _load_coco_images(args.data_root, args.annotations)
    elif args.dataset == "cub":
        ann = args.annotations or os.path.join(args.data_root, "annotations")
        image_paths, image_ids = _load_cub_images(args.data_root, ann)
    else:
        ann = args.annotations or os.path.join(args.data_root, "region_descriptions.json")
        image_paths, image_ids = _load_vg_images(args.data_root, ann)

    N = len(image_paths)
    print(f"Found {N} images")

    # --- Optional caption stats ---
    caption_stats_map: Optional[dict[int, torch.Tensor]] = None
    if args.caption_stats:
        print(f"Loading caption stats from: {args.caption_stats}")
        caption_stats_map = _load_caption_stats(args.caption_stats)
        print(f"  Loaded stats for {len(caption_stats_map)} images")

    # --- DataLoader ---
    dataset = _ImageDataset(image_paths, image_ids, preprocess)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=(str(device) != "cpu"),
        shuffle=False,
    )

    # --- Inference ---
    all_ids: list[int] = []
    clip_scores_chunks: list[torch.Tensor] = []
    clip_soft_chunks: list[torch.Tensor] = []

    print("Computing CLIP scores...")
    for imgs, ids in tqdm(loader, unit="batch"):
        imgs = imgs.to(device)

        img_emb = clip_model.encode_image(imgs)        # [B, D]
        img_emb = F.normalize(img_emb, dim=-1)

        scores = img_emb @ vocab_emb.T                 # [B, V]
        soft = F.softmax(scores / args.temperature, dim=-1)

        all_ids.extend(ids.tolist())
        clip_scores_chunks.append(scores.cpu().to(torch.float16))
        clip_soft_chunks.append(soft.cpu().to(torch.float16))

    clip_scores = torch.cat(clip_scores_chunks, dim=0)       # [N, V]
    clip_soft_labels = torch.cat(clip_soft_chunks, dim=0)    # [N, V]

    print(f"Done. Scores shape: {clip_scores.shape}, dtype: {clip_scores.dtype}")

    # --- Optional concept filtering (Label-free-CBM's CLIP-cutoff, their default 0.25):
    # drop concepts CLIP itself can't reliably detect in any image before they ever
    # reach training. Subsets vocab/vocab_emb/clip_scores/clip_soft_labels together so
    # every downstream artifact stays in lockstep. ---
    if args.concept_clip_cutoff is not None:
        k = min(5, N)
        top5_mean = clip_scores.float().topk(k, dim=0).values.mean(dim=0)  # [V]
        keep_mask = top5_mean > args.concept_clip_cutoff
        n_kept = int(keep_mask.sum().item())
        print(f"Concept CLIP-cutoff {args.concept_clip_cutoff}: keeping {n_kept}/{V} concepts")
        if n_kept == 0:
            raise RuntimeError(
                f"--concept-clip-cutoff {args.concept_clip_cutoff} filtered out all "
                f"{V} concepts -- cutoff too strict for this vocab/dataset."
            )

        vocab = [w for w, keep in zip(vocab, keep_mask.tolist()) if keep]
        vocab_emb = vocab_emb[keep_mask]
        clip_scores = clip_scores[:, keep_mask]
        clip_soft_labels = clip_soft_labels[:, keep_mask]
        V = n_kept

        concepts_out = args.filtered_concepts_out or f"{os.path.splitext(args.output)[0]}_concepts_filtered.txt"
        vocab_cache_out = args.filtered_vocab_cache_out or f"{os.path.splitext(args.output)[0]}_vocab_filtered.pt"

        Path(concepts_out).parent.mkdir(parents=True, exist_ok=True)
        with open(concepts_out, "w", encoding="utf-8") as f:
            f.write("\n".join(vocab) + "\n")
        print(f"Saved filtered concept list -> {concepts_out}")

        Path(vocab_cache_out).parent.mkdir(parents=True, exist_ok=True)
        torch.save({w: vocab_emb[i] for i, w in enumerate(vocab)}, vocab_cache_out)
        print(f"Saved filtered vocab cache -> {vocab_cache_out}")

    # --- Optional blending ---
    mixed_labels: Optional[torch.Tensor] = None
    if caption_stats_map is not None:
        print(f"Blending CLIP and caption stats (alpha={args.alpha})...")
        cap_tensor = torch.zeros(N, V, dtype=torch.float32)
        matched = 0
        for i, image_id in enumerate(all_ids):
            if image_id in caption_stats_map:
                dist = caption_stats_map[image_id]
                # Resize if vocab sizes differ (e.g., stats built with different vocab)
                if dist.shape[0] == V:
                    cap_tensor[i] = dist
                    matched += 1
        print(f"  Matched caption stats for {matched}/{N} images")

        mixed = args.alpha * clip_soft_labels.float() + (1.0 - args.alpha) * cap_tensor
        row_sums = mixed.sum(dim=1, keepdim=True).clamp(min=1e-8)
        mixed_labels = (mixed / row_sums).to(torch.float16)

    # --- Save ---
    output_dict = {
        "image_ids": all_ids,
        "vocab": vocab,
        "clip_scores": clip_scores,
        "clip_soft_labels": clip_soft_labels,
        "alpha": args.alpha,
        "temperature": args.temperature,
        "clip_model": args.clip_model,
        "clip_pretrained": args.clip_pretrained,
        "vocab_cache": str(args.vocab_cache),
        "dataset": args.dataset,
    }
    if mixed_labels is not None:
        output_dict["mixed_labels"] = mixed_labels

    torch.save(output_dict, args.output)

    print(f"\nSaved: {args.output}")
    print(f"  images           : {N}")
    print(f"  vocab            : {V} words")
    size_gb = clip_scores.numel() * 2 / 1e9
    print(f"  clip_scores      : {tuple(clip_scores.shape)}  ({size_gb:.2f} GB float16)")
    if mixed_labels is not None:
        print(f"  mixed_labels     : {tuple(mixed_labels.shape)}  alpha={args.alpha}")


if __name__ == "__main__":
    main()
