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
    model.load_state_dict(state_dict, strict=True)
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
def run_score_pass(
    model: PNP,
    dataloader: DataLoader,
    device: torch.device,
    prototypes: torch.Tensor,
    mode_label: str,
) -> tuple[torch.Tensor, list, list, torch.Tensor]:
    """One pass accumulating [M, V+C] mixture scores, images (cpu), captions, dataset indices."""
    all_scores:   list[torch.Tensor] = []
    all_images:   list[torch.Tensor] = []
    all_captions: list = []
    all_indices:  list[torch.Tensor] = []

    for batch in tqdm(dataloader, desc=f"Score pass [{mode_label}]"):
        images, captions, _, indices = batch
        images = images.to(device, non_blocking=True)

        patch_tokens, _, _ = model.backbone(images)
        patch_tokens = F.normalize(patch_tokens, p=2, dim=-1)          # [B, N, D]

        patch_logits = torch.einsum("bnd,vd->bnv", patch_tokens, prototypes)  # [B, N, V+C]
        vocab_logits = patch_logits.topk(5, dim=1).values.mean(dim=1)          # [B, V+C]
        scores = F.softmax(vocab_logits / model.temperature, dim=-1)            # [B, V+C]

        all_scores.append(scores.cpu())
        all_images.extend([im.cpu() for im in images])
        all_captions.extend(list(captions))
        all_indices.append(indices.cpu())

    return (
        torch.cat(all_scores, dim=0),       # [M, V+C]
        all_images,
        all_captions,
        torch.cat(all_indices, dim=0),      # [M]
    )


@torch.inference_mode()
def compute_heatmap_for_image(
    model: PNP,
    img_tensor: torch.Tensor,
    prototypes: torch.Tensor,
    proto_col: int,
    device: torch.device,
) -> torch.Tensor:
    """Return patch-level logit vector for a single prototype column on one image."""
    img = img_tensor.unsqueeze(0).to(device)
    patch_tokens, _, _ = model.backbone(img)
    patch_tokens = F.normalize(patch_tokens, p=2, dim=-1)  # [1, N, D]
    patch_logits = torch.einsum("bnd,vd->bnv", patch_tokens, prototypes)  # [1, N, V+C]
    return patch_logits[0, :, proto_col].cpu()  # [N]


# ---------------------------------------------------------------------------
# Analysis & W&B logging
# ---------------------------------------------------------------------------

def log_mode_results(
    mode_label: str,
    all_scores: torch.Tensor,
    all_images: list,
    all_captions: list,
    vocab_words: list[str],
    caption_words: list[str] | None,
    topk: int,
    model: PNP,
    prototypes: torch.Tensor,
    device: torch.device,
):
    n_word_proto = len(vocab_words)
    n_total = all_scores.shape[1]

    # Per-image top-K prototype indices
    topk_per_image = all_scores.topk(topk, dim=1).indices  # [M, topk]
    is_caption = (topk_per_image >= n_word_proto)           # [M, topk] bool

    frac_caption = is_caption.float().mean(dim=1)           # [M]
    mean_frac = frac_caption.mean().item()
    print(f"[{mode_label}] Mean fraction of top-{topk} from caption prototypes: {mean_frac:.4f}")

    wandb.log({
        f"{mode_label}/mean_frac_caption_in_topk": mean_frac,
        f"{mode_label}/frac_caption_histogram": wandb.Histogram(frac_caption.numpy()),
        f"{mode_label}/n_word_prototypes": n_word_proto,
        f"{mode_label}/n_caption_prototypes": n_total - n_word_proto,
    })

    # Top-5 images for the most-activated caption concept (augmented mode only)
    if caption_words and n_total > n_word_proto:
        caption_scores = all_scores[:, n_word_proto:]       # [M, C]
        mean_caption_scores = caption_scores.mean(dim=0)    # [C]
        top_concept_col = int(mean_caption_scores.argmax().item())
        top_concept_text = caption_words[top_concept_col]
        global_proto_col = n_word_proto + top_concept_col

        top5_vals, top5_img_idx = caption_scores[:, top_concept_col].topk(min(5, len(all_images)))
        n_show = len(top5_img_idx)

        fig, axes = plt.subplots(1, n_show, figsize=(4 * n_show, 4), dpi=120)
        if n_show == 1:
            axes = [axes]

        for ax, score, img_idx in zip(axes, top5_vals.tolist(), top5_img_idx.tolist()):
            img_tensor = all_images[img_idx]
            img_uint8 = denorm_to_uint8(img_tensor)

            # Targeted second pass: one image, one prototype column
            hm = compute_heatmap_for_image(model, img_tensor, prototypes, global_proto_col, device)
            H = W = int(math.sqrt(hm.shape[0]))
            hm_up = F.interpolate(
                hm.view(1, 1, H, W), size=img_uint8.shape[:2], mode="bilinear", align_corners=False
            )[0, 0]
            bbox = find_high_activation_crop(hm_up.numpy())
            overlay = overlay_heatmap(img_uint8, hm_up)
            overlay = draw_rect(overlay, bbox)

            ax.imshow(overlay)
            ax.axis("off")
            caption = all_captions[img_idx]
            if isinstance(caption, list):
                caption = caption[0] if caption else ""
            ax.set_title(f"score={score:.3f}\n{str(caption)[:50]}", fontsize=8)

        fig.suptitle(f"[{mode_label}] Top caption concept: '{top_concept_text[:70]}'", fontsize=9)
        plt.tight_layout()
        wandb.log({f"{mode_label}/top_caption_concept": wandb.Image(fig)})
        plt.close(fig)

    # Top-5 most-activated word prototypes by mean score
    word_scores = all_scores[:, :n_word_proto]
    mean_word_scores = word_scores.mean(dim=0)
    top5_word_vals, top5_word_idx = mean_word_scores.topk(5)
    top5_words = [(vocab_words[i], round(v.item(), 4)) for i, v in zip(top5_word_idx.tolist(), top5_word_vals)]
    print(f"[{mode_label}] Top-5 word prototypes by mean activation: {top5_words}")
    wandb.log({f"{mode_label}/top5_word_concepts": str(top5_words)})


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
    caption_embeds: torch.Tensor | None = None
    caption_words:  list[str]  | None = None
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

        scores, images, captions, indices = run_score_pass(
            model, dataloader, device, prototypes, mode_label
        )
        log_mode_results(
            mode_label=mode_label,
            all_scores=scores,
            all_images=images,
            all_captions=captions,
            vocab_words=vocab_words,
            caption_words=caption_words if mode_label == "augmented" else None,
            topk=args.topk,
            model=model,
            prototypes=prototypes,
            device=device,
        )

    print("Done.")
    wandb.finish()


if __name__ == "__main__":
    main()
