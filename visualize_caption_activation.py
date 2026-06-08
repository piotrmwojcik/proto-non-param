#!/usr/bin/env python3
"""Per-image caption-prototype activation visualiser.

For each selected test image shows the top-K caption prototypes that activate
most strongly, one subplot per caption, with the caption text as the title and
an activation heatmap overlay.  No statistics — pure visual sanity-check.

Usage:
    # 20 random test images, top-4 captions each
    python visualize_caption_activation.py \\
        --ckpt /path/to/ckpt.pth \\
        --vocab-cache-path vocab/vg_cache.pt \\
        --caption-prototypes-path vocab/vg_test_caption_prototypes.pt \\
        --source-dataset vg_test \\
        --vg-root /data/vg \\
        --vg-region-descriptions /data/vg/region_descriptions.json \\
        --n-random 20 --n-captions 4

    # specific indices
    python visualize_caption_activation.py ... --indices 0 7 42 99
"""

import argparse
import math
from pathlib import Path
from typing import List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import open_clip
import torch
import torch.nn.functional as F
import wandb
from tqdm import tqdm

from clip_dataset import CocoCLIPDataset, VisualGenomeDataset
from modeling.backbone import (
    DINOBackboneExpanded,
    DINOv2Backbone,
    DINOv2BackboneExpanded,
)
from modeling.pnp import PNP


CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD  = (0.26862954, 0.26130258, 0.27577711)


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------

def denorm(x: torch.Tensor) -> np.ndarray:
    m = torch.tensor(CLIP_MEAN)[:, None, None]
    s = torch.tensor(CLIP_STD)[:, None, None]
    return ((x.cpu() * s + m).clamp(0, 1) * 255).byte().permute(1, 2, 0).numpy()


def heatmap_overlay(img: np.ndarray, hm: torch.Tensor, alpha: float = 0.50) -> np.ndarray:
    h = hm.detach().cpu().numpy().astype(np.float32)
    h = (h - h.min()) / (h.max() - h.min() + 1e-8)
    # warm colourmap: red=hot, blue=cold
    rgb = np.stack([h, np.clip(h * 0.7 + 0.15, 0, 1), np.clip(1.0 - h * 0.9, 0, 1)], -1)
    rgb = (rgb * 255).astype(np.uint8)
    return (alpha * rgb + (1 - alpha) * img.astype(np.float32)).clip(0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def build_model(ckpt_path: str, device: torch.device) -> PNP:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    hparams = argparse.Namespace(**ckpt["hparams"])
    sd = {k: v for k, v in ckpt["state_dict"].items() if not k.startswith("clip_model.")}

    if "dinov2" in hparams.backbone:
        if getattr(hparams, "num_splits", 0):
            backbone = DINOv2BackboneExpanded(
                hparams.backbone, n_splits=hparams.num_splits,
                mode="append", freeze_norm_layer=True,
            )
        else:
            backbone = DINOv2Backbone(name=hparams.backbone)
    else:
        backbone = DINOBackboneExpanded(
            hparams.backbone,
            n_splits=getattr(hparams, "num_splits", 1),
            mode="block_expansion", freeze_norm_layer=True,
        )

    clip_model, _, _ = open_clip.create_model_and_transforms(
        getattr(hparams, "clip_model_name", "ViT-B-32"),
        pretrained=getattr(hparams, "clip_pretrained", "openai"),
    )
    clip_model.eval().to(device)
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
    model.load_state_dict(sd, strict=False)
    model.eval().to(device)
    print(f"Model loaded — vocab: {model.vocab_size} words")
    return model


# ---------------------------------------------------------------------------
# Scoring & heatmap for a single image
# ---------------------------------------------------------------------------

@torch.inference_mode()
def patch_tokens_for(model: PNP, img: torch.Tensor, device: torch.device) -> torch.Tensor:
    pt, _, _ = model.backbone(img.unsqueeze(0).to(device))
    return F.normalize(pt, p=2, dim=-1)[0]           # [N, D]


@torch.inference_mode()
def score_image_vs_captions(
    patch_tokens: torch.Tensor,          # [N, D]  on CPU
    cap_vecs: torch.Tensor,              # [C, D]  on CPU (projected)
    device: torch.device,
    top_patches: int = 5,
    chunk_size: int = 8192,
) -> torch.Tensor:
    """Return [C] activation scores (top-patch mean similarity)."""
    C = cap_vecs.shape[0]
    pt = patch_tokens.to(device)         # [N, D]
    scores = torch.zeros(C)

    for start in range(0, C, chunk_size):
        end = min(start + chunk_size, C)
        chunk = F.normalize(cap_vecs[start:end].to(device), dim=-1)   # [c, D]
        sim = torch.einsum("nd,cd->nc", pt, chunk)                     # [N, c]
        scores[start:end] = sim.topk(top_patches, dim=0).values.mean(dim=0).cpu()

    return scores                                                        # [C]


def patch_heatmap(
    patch_tokens: torch.Tensor,     # [N, D]  on CPU
    cap_vec: torch.Tensor,          # [D]     on CPU (projected)
    img_hw: tuple,
    device: torch.device,
) -> np.ndarray:
    """Return heatmap upsampled to img_hw (H, W)."""
    pt  = patch_tokens.to(device)
    vec = F.normalize(cap_vec.to(device), dim=-1)
    sim = torch.einsum("nd,d->n", pt, vec).cpu()    # [N]
    side = int(math.sqrt(sim.shape[0]))
    hm_up = F.interpolate(
        sim.view(1, 1, side, side),
        size=img_hw, mode="bilinear", align_corners=False,
    )[0, 0]
    return hm_up


# ---------------------------------------------------------------------------
# Figure rendering
# ---------------------------------------------------------------------------

def render_panel(
    img_tensor: torch.Tensor,
    patch_tokens: torch.Tensor,
    cap_vecs: torch.Tensor,         # [C, D]
    caption_words: List[str],
    top_idx: List[int],
    scores: torch.Tensor,
    image_idx: int,
    device: torch.device,
) -> plt.Figure:
    """One figure with n_captions subplots — each showing the image with a
    different caption's heatmap and the caption text as the subtitle."""
    K = len(top_idx)
    img_uint8 = denorm(img_tensor)
    H, W = img_uint8.shape[:2]

    fig, axes = plt.subplots(1, K, figsize=(4.5 * K, 5.5), dpi=100)
    if K == 1:
        axes = [axes]

    for ax, cap_idx in zip(axes, top_idx):
        hm = patch_heatmap(patch_tokens, cap_vecs[cap_idx], (H, W), device)
        overlay = heatmap_overlay(img_uint8, hm)
        ax.imshow(overlay)
        ax.axis("off")

        score = scores[cap_idx].item()
        text  = caption_words[cap_idx]
        # Wrap long captions manually for readability
        words = text.split()
        lines, line = [], []
        for w in words:
            line.append(w)
            if len(" ".join(line)) > 38:
                lines.append(" ".join(line))
                line = []
        if line:
            lines.append(" ".join(line))
        wrapped = "\n".join(lines[:4])   # at most 4 display lines

        ax.set_title(f"score {score:.3f}\n{wrapped}", fontsize=7, loc="center")

    fig.suptitle(f"Test image {image_idx}", fontsize=10, fontweight="bold", y=1.01)
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Visualise per-image caption activations")
    ap.add_argument("--ckpt",                  required=True)
    ap.add_argument("--vocab-cache-path",       default="vocab/vg_cache.pt")
    ap.add_argument("--caption-prototypes-path", required=True)
    ap.add_argument("--source-dataset",         default="vg_test",
                    choices=["vg_test", "coco_val"])
    ap.add_argument("--vg-root",                default=None)
    ap.add_argument("--vg-region-descriptions", default=None)
    ap.add_argument("--vg-val-ratio",           type=float, default=0.1)
    ap.add_argument("--coco-root",              default=None)
    ap.add_argument("--coco-annotations",       default=None)

    ap.add_argument("--indices",   type=int, nargs="+", default=None,
                    help="Explicit test-set indices to visualise")
    ap.add_argument("--n-random",  type=int, default=20,
                    help="Random images to visualise (ignored if --indices given)")
    ap.add_argument("--n-captions", type=int, default=4,
                    help="Top-K caption prototypes shown per image")
    ap.add_argument("--top-patches", type=int, default=5)

    ap.add_argument("--seed",          type=int, default=42)
    ap.add_argument("--device",        default="cuda")
    ap.add_argument("--out-dir",       default=None,
                    help="Save PNG panels here (in addition to W&B)")
    ap.add_argument("--wandb-project", default="proto-non-param")
    ap.add_argument("--wandb-run-name", default="caption-vis")
    ap.add_argument("--wandb-entity",  default=None)
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # --- Vocab + caption prototypes ---
    vocab_cache   = torch.load(args.vocab_cache_path, map_location="cpu")
    vocab_to_idx  = {w: i for i, w in enumerate(vocab_cache.keys())}

    cap_cache     = torch.load(args.caption_prototypes_path, map_location="cpu")
    caption_words = list(cap_cache.keys())
    cap_embeds    = torch.stack([cap_cache[w] for w in caption_words])  # [C, 512]
    print(f"Caption prototypes: {len(caption_words)}")

    # --- Model + project captions to visual space ---
    model = build_model(args.ckpt, device)
    with torch.no_grad():
        # get_prototypes_augmented returns [V+C, D]; slice off word part
        all_protos  = model.get_prototypes_augmented(cap_embeds.to(device))
        cap_vecs    = all_protos[model.vocab_size:].cpu()   # [C, D]
    print(f"Caption prototype shape in visual space: {tuple(cap_vecs.shape)}")

    # --- Dataset ---
    if args.source_dataset == "vg_test":
        if not (args.vg_root and args.vg_region_descriptions):
            raise ValueError("--vg-root and --vg-region-descriptions required for vg_test")
        dataset = VisualGenomeDataset(
            vg_root=args.vg_root,
            region_descriptions_json=args.vg_region_descriptions,
            vocab_to_idx=vocab_to_idx,
            train=False,
            val_ratio=args.vg_val_ratio,
            seed=args.seed,
        )
    else:
        if not (args.coco_root and args.coco_annotations):
            raise ValueError("--coco-root and --coco-annotations required for coco_val")
        from clip_dataset import coco_clip_collate_fn  # noqa: F401
        dataset = CocoCLIPDataset(
            annotations_json=args.coco_annotations,
            coco_root=args.coco_root,
            vocab_to_idx=vocab_to_idx,
            train=False,
        )
    print(f"Test set size: {len(dataset)}")

    # --- Select indices ---
    if args.indices:
        indices = args.indices
    else:
        rng = torch.Generator().manual_seed(args.seed)
        indices = torch.randperm(len(dataset), generator=rng)[:args.n_random].tolist()
    print(f"Visualising indices: {indices}")

    if args.out_dir:
        Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    wandb.init(
        project=args.wandb_project,
        name=args.wandb_run_name,
        entity=args.wandb_entity,
        config=vars(args),
    )

    panels = {}
    for img_idx in tqdm(indices, desc="Rendering"):
        sample = dataset[img_idx]
        img_tensor = sample[0]              # [3, H, W] normalised

        pt = patch_tokens_for(model, img_tensor, device)   # [N, D]
        scores = score_image_vs_captions(
            pt, cap_vecs, device, top_patches=args.top_patches,
        )
        top_idx = scores.topk(args.n_captions).indices.tolist()

        fig = render_panel(
            img_tensor, pt, cap_vecs, caption_words,
            top_idx, scores, img_idx, device,
        )
        key = f"caption_activation/img_{img_idx:05d}"
        panels[key] = wandb.Image(fig)
        if args.out_dir:
            fig.savefig(
                Path(args.out_dir) / f"img_{img_idx:05d}.png",
                bbox_inches="tight",
            )
        plt.close(fig)

    wandb.log(panels)
    print(f"Logged {len(panels)} panels to W&B.")
    wandb.finish()


if __name__ == "__main__":
    main()
