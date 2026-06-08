#!/usr/bin/env python3
"""Evaluate trained VG-only model with simple word prototypes and/or caption-augmented prototypes.

This script supports three inference modes (--mode):
  word_only   Use only the trained VG word-level prototype pool (standard inference).
  augmented   Extend the prototype pool with phrase-level CLIP embeddings loaded from
              --caption-prototypes-path (no retraining — they pass through the same
              text_projection_head without residuals).
  both        Run both modes sequentially and compare statistics.

For each mode the script logs to W&B:
  - Per-image top-K prototype activation scores
  - Histogram of what fraction of top-K activations are caption vs. word prototypes
  - Top-5 images for the most-activated caption concept with heatmap overlays

Usage:
    python eval_augmented_prototypes.py \\
        --ckpt /path/to/ckpt.pth \\
        --vocab-cache-path vocab/vg_cache.pt \\
        --caption-prototypes-path vocab/vg_test_caption_prototypes.pt \\
        --source-dataset vg_test \\
        --vg-root /data/vg \\
        --vg-region-descriptions /data/vg/region_descriptions.json \\
        --mode both \\
        --topk 5 \\
        --wandb-project proto-non-param
"""

import argparse
import math
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import open_clip
import torch
import torch.nn.functional as F
import wandb
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from clip_dataset import (
    CocoCLIPDataset,
    VisualGenomeDataset,
    coco_clip_collate_fn,
)
from modeling.backbone import DINOv2Backbone, DINOv2BackboneExpanded, DINOBackboneExpanded
from modeling.pnp import PNP


CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD  = (0.26862954, 0.26130258, 0.27577711)


# ---------------------------------------------------------------------------
# Visualisation helpers
# ---------------------------------------------------------------------------

def denorm_to_uint8(x: torch.Tensor, mean=CLIP_MEAN, std=CLIP_STD) -> np.ndarray:
    x = x.detach().cpu()
    mean_t = torch.tensor(mean)[:, None, None]
    std_t  = torch.tensor(std)[:, None, None]
    x = (x * std_t + mean_t).clamp(0, 1)
    return (x * 255).byte().permute(1, 2, 0).numpy()


def overlay_heatmap(img: np.ndarray, hm: torch.Tensor, alpha: float = 0.45) -> np.ndarray:
    hm = hm.detach().cpu()
    hm = (hm - hm.min()) / (hm.max() - hm.min() + 1e-8)
    hm = hm.numpy()
    hm_rgb = (
        np.stack([hm, np.clip(hm * 0.9 + 0.1, 0, 1), np.clip(1.0 - hm * 0.8, 0, 1)], -1) * 255
    ).astype(np.uint8)
    return (alpha * hm_rgb.astype(np.float32) + (1 - alpha) * img.astype(np.float32)).clip(0, 255).astype(np.uint8)


def find_high_activation_crop(act: np.ndarray, percentile: int = 95):
    thr = np.percentile(act, percentile)
    ys, xs = np.where(act >= thr)
    if len(ys) == 0:
        h, w = act.shape
        return 0, h, 0, w
    return ys.min(), ys.max() + 1, xs.min(), xs.max() + 1


def draw_rect(img: np.ndarray, bbox, color=(255, 0, 0), width=3) -> np.ndarray:
    y0, y1, x0, x1 = bbox
    pil = Image.fromarray(img)
    ImageDraw.Draw(pil).rectangle([x0, y0, x1 - 1, y1 - 1], outline=color, width=width)
    return np.array(pil)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def build_model(ckpt_path: str, device: torch.device) -> tuple[PNP, argparse.Namespace]:
    print(f"Loading checkpoint from {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    hparams = argparse.Namespace(**ckpt["hparams"])
    state_dict = ckpt["state_dict"]

    if "dinov2" in hparams.backbone:
        if getattr(hparams, "num_splits", 0) and hparams.num_splits > 0:
            backbone = DINOv2BackboneExpanded(
                name=hparams.backbone,
                n_splits=hparams.num_splits,
                mode="append",
                freeze_norm_layer=True,
            )
        else:
            backbone = DINOv2Backbone(name=hparams.backbone)
    elif "dino" in hparams.backbone:
        backbone = DINOBackboneExpanded(
            name=hparams.backbone,
            n_splits=getattr(hparams, "num_splits", 1),
            mode="block_expansion",
            freeze_norm_layer=True,
        )
    else:
        raise NotImplementedError(f"Backbone {hparams.backbone} not supported here")

    clip_model, _, _ = open_clip.create_model_and_transforms(
        getattr(hparams, "clip_model_name", "ViT-B-32"),
        pretrained=getattr(hparams, "clip_pretrained", "openai"),
    )
    clip_model = clip_model.eval().to(device)
    for p in clip_model.parameters():
        p.requires_grad = False

    model = PNP(
        backbone=backbone,
        dim=backbone.dim,
        temperature=getattr(hparams, "temperature", 0.2),
        clip_text_dim=getattr(hparams, "clip_text_dim", 512),
        text_proj_hidden_dim=getattr(hparams, "text_proj_hidden_dim", 768),
        vocab_cache_path=hparams.vocab_cache_path,
        prototype_init_noise=getattr(hparams, "prototype_init_noise", 0.01),
        clip_model=clip_model,
    )
    # clip_model is always frozen — drop its keys from the checkpoint so mismatches
    # in model size (e.g. ViT-B-32 vs ViT-L-14) don't block loading the trainable parts.
    state_dict = {k: v for k, v in state_dict.items() if not k.startswith("clip_model.")}
    model.load_state_dict(state_dict, strict=False)
    model = model.eval().to(device)
    print(f"Model loaded — vocab size: {model.vocab_size}")
    return model, hparams


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

def build_dataset(args, vocab_to_idx: dict):
    if args.source_dataset == "vg_test":
        if not args.vg_root or not args.vg_region_descriptions:
            raise ValueError("--vg-root and --vg-region-descriptions required for source_dataset=vg_test")
        return VisualGenomeDataset(
            vg_root=args.vg_root,
            region_descriptions_json=args.vg_region_descriptions,
            vocab_to_idx=vocab_to_idx,
            train=False,
            val_ratio=args.vg_val_ratio,
            seed=args.seed,
        )
    elif args.source_dataset == "coco_val":
        if not args.coco_root or not args.coco_annotations:
            raise ValueError("--coco-root and --coco-annotations required for source_dataset=coco_val")
        return CocoCLIPDataset(
            annotations_json=args.coco_annotations,
            coco_root=args.coco_root,
            vocab_to_idx=vocab_to_idx,
            train=False,
        )
    else:
        raise ValueError(f"Unknown source_dataset: {args.source_dataset}")


# ---------------------------------------------------------------------------
# Inference — first pass (scores only, no patch logits accumulated)
# ---------------------------------------------------------------------------

@torch.inference_mode()
def _vocab_logits_chunked(
    patch_tokens: torch.Tensor,
    prototypes: torch.Tensor,
    top_patches: int = 5,
    chunk_size: int = 8192,
) -> torch.Tensor:
    """Compute per-prototype top-patch scores without materialising [B, N, V] at once.

    Processes prototypes in chunks of `chunk_size` to keep GPU memory bounded.
    Peak memory per chunk: B * N * chunk_size * 4 bytes.
    With B=64, N=256, chunk=8192: ~512 MB — safe on a 40 GB A100.
    """
    B, N, D = patch_tokens.shape
    V = prototypes.shape[0]
    vocab_logits = torch.zeros(B, V, device=patch_tokens.device)

    for start in range(0, V, chunk_size):
        end = min(start + chunk_size, V)
        chunk = prototypes[start:end]                                        # [C, D]
        sim = torch.einsum("bnd,cd->bnc", patch_tokens, chunk)              # [B, N, C]
        vocab_logits[:, start:end] = sim.topk(top_patches, dim=1).values.mean(dim=1)

    return vocab_logits                                                      # [B, V]


@torch.inference_mode()
def run_score_pass(
    model: PNP,
    dataloader: DataLoader,
    device: torch.device,
    prototypes: torch.Tensor,
    mode_label: str,
    n_vocab: int,
    topk: int,
    proto_chunk_size: int = 8192,
):
    """Memory-efficient inference pass — never stores [M, V+C] scores.

    Accumulates only what is needed for analysis:
      - frac_caption per image  [M]       (~40 KB for 10K images)
      - topk prototype indices  [M, topk] (~200 KB)
      - caption mean scores     [C]       (~1.5 MB for 392K captions)
      - top-5 images by max caption activation (5 image tensors + captions)
    """
    n_total = prototypes.shape[0]
    n_caption = n_total - n_vocab

    all_frac_caption: list[torch.Tensor] = []
    all_topk_word_scores: list[torch.Tensor] = []   # [M, topk] clipped to word range
    caption_score_sum = torch.zeros(max(n_caption, 1))  # [C]
    n_images_seen = 0

    # Running top-5 images by their highest caption prototype score
    top5: list = []  # list of [score, img_cpu, caption_str]

    for batch in tqdm(dataloader, desc=f"Score pass [{mode_label}]"):
        images, captions, _, _ = batch
        images = images.to(device, non_blocking=True)

        patch_tokens, _, _ = model.backbone(images)
        patch_tokens = F.normalize(patch_tokens, p=2, dim=-1)   # [B, N, D]

        vocab_logits = _vocab_logits_chunked(
            patch_tokens, prototypes, chunk_size=proto_chunk_size
        )                                                        # [B, V+C]
        scores = F.softmax(vocab_logits / model.temperature, dim=-1)  # [B, V+C]

        # Top-K indices per image (used for frac_caption)
        topk_idx = scores.topk(topk, dim=1).indices             # [B, topk]
        is_cap = (topk_idx >= n_vocab).float().mean(dim=1)      # [B]
        all_frac_caption.append(is_cap.cpu())

        # Top-5 word activations (for reporting top word concepts)
        word_scores = scores[:, :n_vocab]                        # [B, V]
        all_topk_word_scores.append(word_scores.cpu())

        if n_caption > 0:
            cap_scores = scores[:, n_vocab:].cpu()              # [B, C]
            caption_score_sum += cap_scores.sum(dim=0)
            n_images_seen += images.shape[0]

            # Track top-5 images by max caption activation
            max_cap = cap_scores.max(dim=1).values              # [B]
            for i in range(images.shape[0]):
                cap_str = captions[i] if isinstance(captions[i], str) \
                    else (captions[i][0] if captions[i] else "")
                top5.append((max_cap[i].item(), images[i].cpu(), cap_str))
                top5.sort(key=lambda x: x[0], reverse=True)
                top5 = top5[:5]

    frac_caption = torch.cat(all_frac_caption, dim=0)            # [M]
    word_scores_all = torch.cat(all_topk_word_scores, dim=0)     # [M, V]
    caption_mean = (caption_score_sum / max(n_images_seen, 1)) if n_caption > 0 \
        else torch.zeros(0)

    return frac_caption, word_scores_all, caption_mean, top5


@torch.inference_mode()
def compute_heatmap_for_image(
    model: PNP,
    img_tensor: torch.Tensor,
    proto_vec: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """Patch-level similarity between one image and one prototype vector [D]."""
    img = img_tensor.unsqueeze(0).to(device)
    patch_tokens, _, _ = model.backbone(img)
    patch_tokens = F.normalize(patch_tokens, p=2, dim=-1)       # [1, N, D]
    proto = F.normalize(proto_vec.to(device), dim=-1)            # [D]
    return torch.einsum("bnd,d->bn", patch_tokens, proto)[0].cpu()  # [N]


# ---------------------------------------------------------------------------
# Second pass: top-N images per concept (lightweight — only K prototype vecs)
# ---------------------------------------------------------------------------

@torch.inference_mode()
def collect_topk_images_per_concept(
    model: PNP,
    dataloader: DataLoader,
    device: torch.device,
    concept_vecs: torch.Tensor,   # [K, D] — the K prototype vectors to score
    n_images: int = 5,
    top_patches: int = 5,
) -> list:
    """For each of the K concept vectors, return the top-N images by activation.

    Returns a list of K lists, each with up to n_images (score, img_cpu, caption) tuples.
    Memory: K * n_images * (1 image tensor) — at most a few hundred MB for K=10.
    """
    K = concept_vecs.shape[0]
    concept_vecs = F.normalize(concept_vecs.to(device), dim=-1)  # [K, D]
    top_images = [[] for _ in range(K)]

    for batch in tqdm(dataloader, desc="Concept retrieval pass", leave=False):
        images, captions, _, _ = batch
        images = images.to(device, non_blocking=True)

        patch_tokens, _, _ = model.backbone(images)
        patch_tokens = F.normalize(patch_tokens, p=2, dim=-1)      # [B, N, D]

        sims = torch.einsum("bnd,kd->bnk", patch_tokens, concept_vecs)  # [B, N, K]
        scores = sims.topk(top_patches, dim=1).values.mean(dim=1)        # [B, K]

        for b in range(images.shape[0]):
            cap = captions[b] if isinstance(captions[b], str) \
                else (captions[b][0] if captions[b] else "")
            for k in range(K):
                top_images[k].append((scores[b, k].item(), images[b].cpu(), cap))
                top_images[k].sort(key=lambda x: x[0], reverse=True)
                top_images[k] = top_images[k][:n_images]

    return top_images


# ---------------------------------------------------------------------------
# Concept panel helper
# ---------------------------------------------------------------------------

def _concept_panel(
    concept_label: str,
    top_images: list,
    proto_vec: torch.Tensor,
    model: PNP,
    device: torch.device,
    mode_label: str,
) -> "wandb.Image":
    """Render a row of images with heatmap overlays for one concept."""
    n_show = len(top_images)
    fig, axes = plt.subplots(1, n_show, figsize=(4 * n_show, 4), dpi=120)
    if n_show == 1:
        axes = [axes]

    for ax, (score, img_tensor, cap_str) in zip(axes, top_images):
        img_uint8 = denorm_to_uint8(img_tensor)
        hm = compute_heatmap_for_image(model, img_tensor, proto_vec, device)
        H = W = int(math.sqrt(hm.shape[0]))
        hm_up = F.interpolate(
            hm.view(1, 1, H, W), size=img_uint8.shape[:2],
            mode="bilinear", align_corners=False,
        )[0, 0]
        bbox = find_high_activation_crop(hm_up.numpy())
        overlay = draw_rect(overlay_heatmap(img_uint8, hm_up), bbox)
        ax.imshow(overlay)
        ax.axis("off")
        ax.set_title(f"score={score:.3f}\n{str(cap_str)[:45]}", fontsize=7)

    fig.suptitle(f"[{mode_label}] {concept_label}", fontsize=9)
    plt.tight_layout()
    img = wandb.Image(fig)
    plt.close(fig)
    return img


# ---------------------------------------------------------------------------
# Analysis & W&B logging
# ---------------------------------------------------------------------------

def log_mode_results(
    mode_label: str,
    frac_caption: torch.Tensor,
    word_scores_all: torch.Tensor,
    caption_mean: torch.Tensor,
    vocab_words: list,
    caption_words: Optional[list],
    topk: int,
    model: PNP,
    prototypes: torch.Tensor,
    dataloader: DataLoader,
    device: torch.device,
    n_top_concepts: int = 5,
    n_top_images: int = 5,
):
    n_word_proto = len(vocab_words)
    n_caption = len(caption_words) if caption_words else 0
    mean_frac = frac_caption.mean().item()
    pct_any_caption = (frac_caption > 0).float().mean().item()

    print(f"[{mode_label}] mean frac caption in top-{topk}: {mean_frac:.4f} | "
          f"images with ≥1 caption: {pct_any_caption:.4f}")

    # ---- Scalar metrics ----
    wandb.log({
        f"{mode_label}/mean_frac_caption_in_topk":   mean_frac,
        f"{mode_label}/pct_images_with_caption_topk": pct_any_caption,
        f"{mode_label}/frac_caption_histogram":       wandb.Histogram(frac_caption.numpy()),
        f"{mode_label}/n_word_prototypes":            n_word_proto,
        f"{mode_label}/n_caption_prototypes":         n_caption,
    })

    # ---- Top-N word concepts table + panels ----
    mean_word_scores = word_scores_all.mean(dim=0)                 # [V]
    top_word_vals, top_word_idx = mean_word_scores.topk(min(20, n_word_proto))
    word_table = wandb.Table(columns=["rank", "word", "mean_score"])
    for rank, (idx, val) in enumerate(zip(top_word_idx.tolist(), top_word_vals.tolist())):
        word_table.add_data(rank + 1, vocab_words[idx], round(val, 5))
    wandb.log({f"{mode_label}/top_word_concepts_table": word_table})

    top_word_vecs = prototypes[:n_word_proto][top_word_idx[:n_top_concepts]].cpu()  # [K, D]
    word_top_images = collect_topk_images_per_concept(
        model, dataloader, device, top_word_vecs,
        n_images=n_top_images,
    )
    panels = {}
    for rank, (col, images_for_concept) in enumerate(
        zip(top_word_idx[:n_top_concepts].tolist(), word_top_images)
    ):
        label = f"word #{rank+1}: {vocab_words[col]}"
        panels[f"{mode_label}/word_concept_{rank+1:02d}_{vocab_words[col]}"] = _concept_panel(
            label, images_for_concept, top_word_vecs[rank], model, device, mode_label
        )
    wandb.log(panels)

    # ---- Top-N caption concepts table + panels (augmented mode only) ----
    if caption_words and caption_mean.numel() > 0:
        top_cap_vals, top_cap_idx = caption_mean.topk(min(20, n_caption))
        cap_table = wandb.Table(columns=["rank", "caption", "mean_score"])
        for rank, (idx, val) in enumerate(zip(top_cap_idx.tolist(), top_cap_vals.tolist())):
            cap_table.add_data(rank + 1, caption_words[idx][:120], round(val, 5))
        wandb.log({f"{mode_label}/top_caption_concepts_table": cap_table})

        top_cap_vecs = prototypes[n_word_proto:][top_cap_idx[:n_top_concepts]].cpu()  # [K, D]
        cap_top_images = collect_topk_images_per_concept(
            model, dataloader, device, top_cap_vecs,
            n_images=n_top_images,
        )
        panels = {}
        for rank, (col, images_for_concept) in enumerate(
            zip(top_cap_idx[:n_top_concepts].tolist(), cap_top_images)
        ):
            label = f"caption #{rank+1}: {caption_words[col][:60]}"
            panels[f"{mode_label}/caption_concept_{rank+1:02d}"] = _concept_panel(
                label, images_for_concept, top_cap_vecs[rank], model, device, mode_label
            )
        wandb.log(panels)

    top5_words = [(vocab_words[i], round(v, 4))
                  for i, v in zip(top_word_idx[:5].tolist(), top_word_vals[:5].tolist())]
    print(f"[{mode_label}] Top-5 word concepts: {top5_words}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Evaluate augmented vs word-only prototypes")
    parser.add_argument("--ckpt", type=str, required=True,
                        help="Path to checkpoint .pth file saved by train.py")
    parser.add_argument("--vocab-cache-path", type=str, default="vocab/vg_cache.pt")
    parser.add_argument("--caption-prototypes-path", type=str, default=None,
                        help="Path to caption prototype cache (.pt) built by "
                             "vocab/build_caption_prototypes.py")
    parser.add_argument("--mode", type=str, default="both",
                        choices=["word_only", "augmented", "both"])

    parser.add_argument("--source-dataset", type=str, default="vg_test",
                        choices=["vg_test", "coco_val"])
    parser.add_argument("--vg-root", type=str, default=None)
    parser.add_argument("--vg-region-descriptions", type=str, default=None)
    parser.add_argument("--vg-val-ratio", type=float, default=0.1)
    parser.add_argument("--coco-root", type=str, default=None)
    parser.add_argument("--coco-annotations", type=str, default=None)

    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--wandb-project", type=str, default="proto-non-param")
    parser.add_argument("--wandb-run-name", type=str, default="augmented-proto-eval")
    parser.add_argument("--wandb-entity", type=str, default=None)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load word vocabulary
    print(f"Loading vocab cache: {args.vocab_cache_path}")
    vocab_cache = torch.load(args.vocab_cache_path, map_location="cpu")
    vocab_words = list(vocab_cache.keys())
    vocab_to_idx = {w: i for i, w in enumerate(vocab_words)}

    # Load caption prototypes (augmented mode only)
    caption_embeds: Optional[torch.Tensor] = None
    caption_words:  Optional[list] = None
    if args.mode in ("augmented", "both"):
        if args.caption_prototypes_path is None:
            raise ValueError("--caption-prototypes-path is required for augmented mode")
        print(f"Loading caption prototypes: {args.caption_prototypes_path}")
        cap_cache = torch.load(args.caption_prototypes_path, map_location="cpu")
        caption_words  = list(cap_cache.keys())
        caption_embeds = torch.stack([cap_cache[w] for w in caption_words], dim=0)  # [C, 512]
        print(f"  {len(caption_words)} caption prototypes loaded")

    # Build model & dataset
    model, hparams = build_model(args.ckpt, device)
    dataset = build_dataset(args, vocab_to_idx)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=coco_clip_collate_fn,
    )
    print(f"Dataset: {len(dataset)} samples, {len(dataloader)} batches")

    wandb.init(
        project=args.wandb_project,
        name=args.wandb_run_name,
        entity=args.wandb_entity,
        config=vars(args),
    )

    modes_to_run = (
        [("word_only", None), ("augmented", caption_embeds)]
        if args.mode == "both"
        else [("word_only", None)] if args.mode == "word_only"
        else [("augmented", caption_embeds)]
    )

    for mode_label, extra_embeds in modes_to_run:
        # Pre-compute prototype set for this mode
        with torch.no_grad():
            if extra_embeds is not None:
                prototypes = model.get_prototypes_augmented(extra_embeds.to(device))
            else:
                prototypes = model.get_prototypes()
            prototypes = F.normalize(prototypes, p=2, dim=-1)  # [V(+C), D]

        frac_caption, word_scores_all, caption_mean, _ = run_score_pass(
            model, dataloader, device, prototypes, mode_label,
            n_vocab=model.vocab_size, topk=args.topk,
        )
        log_mode_results(
            mode_label=mode_label,
            frac_caption=frac_caption,
            word_scores_all=word_scores_all,
            caption_mean=caption_mean,
            vocab_words=vocab_words,
            caption_words=caption_words if mode_label == "augmented" else None,
            topk=args.topk,
            model=model,
            prototypes=prototypes,
            dataloader=dataloader,
            device=device,
        )

    print("Done.")
    wandb.finish()


if __name__ == "__main__":
    main()
