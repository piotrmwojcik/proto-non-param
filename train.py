#!/usr/bin/env python3
import sys
import logging
from collections import defaultdict
from logging import Logger
from pathlib import Path
import numpy as np
import math
import random
import open_clip
from collections import defaultdict
import argparse
from sklearn.manifold import TSNE
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

import wandb
import lightning as L
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn.functional as F

from clip_dataset import (CocoCLIPDataset, Caltech101CLIPDataset, CUBCLIPDataset,
                           AwA2CLIPDataset, VisualGenomeDataset, coco_clip_collate_fn,
                           vg_collate_fn, CLIPScoreDataset)
from modeling.backbone import DINOv2Backbone, DINOv2BackboneExpanded, DINOBackboneExpanded, CLIPBackbone
from modeling.pnp import PNP, PNPCriterion
from modeling.utils import print_parameters


CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def denorm_to_uint8(
    x: torch.Tensor,
    mean=CLIP_MEAN,
    std=CLIP_STD,
) -> np.ndarray:
    x = x.detach().cpu()
    mean_t = torch.tensor(mean)[:, None, None]
    std_t = torch.tensor(std)[:, None, None]
    x = (x * std_t + mean_t).clamp(0, 1)
    x = (x * 255).byte().permute(1, 2, 0).numpy()
    return x


def overlay_heatmap(img_uint8: np.ndarray, hm: torch.Tensor, alpha: float = 0.45) -> np.ndarray:
    hm = hm.detach().cpu()
    hm = (hm - hm.min()) / (hm.max() - hm.min() + 1e-8)
    hm = hm.numpy()

    r = hm
    g = np.clip(hm * 0.9 + 0.1, 0, 1)
    b = np.clip(1.0 - hm * 0.8, 0, 1)
    hm_rgb = (np.stack([r, g, b], axis=-1) * 255).astype(np.uint8)

    out = alpha * hm_rgb.astype(np.float32) + (1 - alpha) * img_uint8.astype(np.float32)
    return out.clip(0, 255).astype(np.uint8)


def find_high_activation_crop(activation_map, percentile=95):
    threshold = np.percentile(activation_map, percentile)
    mask = activation_map >= threshold

    ys, xs = np.where(mask)
    if len(ys) == 0 or len(xs) == 0:
        h, w = activation_map.shape
        return 0, h, 0, w

    lower_y, upper_y = ys.min(), ys.max() + 1
    lower_x, upper_x = xs.min(), xs.max() + 1
    return lower_y, upper_y, lower_x, upper_x


def draw_rect_on_image(img_uint8, bbox, color=(255, 0, 0), width=3):
    """
    img_uint8: HxWx3 uint8 numpy array
    bbox: (y0, y1, x0, x1)
    """
    y0, y1, x0, x1 = bbox
    img_pil = Image.fromarray(img_uint8)
    draw = ImageDraw.Draw(img_pil)
    draw.rectangle([x0, y0, x1 - 1, y1 - 1], outline=color, width=width)
    return np.array(img_pil)


@torch.no_grad()
def wandb_log_top_proto_heatmaps(
    *,
    model: nn.Module,
    images: torch.Tensor,
    outputs: dict,
    step: int,
    captions=None,
    max_items: int = 48,
    top_k: int = 5,
    mean=CLIP_MEAN,
    std=CLIP_STD,
    log_key: str = "test/top_proto_heatmaps",
    tsne_key: str = "test/proto_tsne",
    tsne_max_points: int = 300,
    log_tsne: bool = False,
    crop_percentile: float = 95,
):
    """
    Logs one grid image per sample:
      - raw image
      - top-k prototype heatmaps with prototype words / scores
      - rectangle around high-activation region on each heatmap
    """
    patch_logits = outputs["patch_prototype_logits"]   # [B, N, V]
    mix_weights = outputs["mixture_weights"]           # [B, V]

    B, N, V = patch_logits.shape
    H = W = int(math.sqrt(N))
    _, _, Hi, Wi = images.shape

    top_vals, top_idx = mix_weights.topk(k=top_k, dim=-1)   # [B, K]

    B_log = min(B, max_items)
    ncols = 1 + top_k  # raw + one column per prototype

    fig, axes = plt.subplots(
        nrows=B_log,
        ncols=ncols,
        figsize=(3 * ncols, 3 * B_log),
        dpi=100,
        squeeze=False,
    )

    for b in range(B_log):
        img_uint8 = denorm_to_uint8(images[b], mean=mean, std=std)
        raw_caption = str(captions[b]) if captions is not None else ""

        # Column 0: raw image
        ax = axes[b][0]
        ax.imshow(img_uint8)
        ax.set_ylabel(raw_caption, fontsize=7, rotation=0, labelpad=60, va="center")
        ax.set_title("raw" if b == 0 else "", fontsize=9)
        ax.axis("off")

        # Columns 1..top_k: prototype heatmaps
        for rank, proto_idx in enumerate(top_idx[b].tolist()):
            hm = patch_logits[b, :, proto_idx].view(1, 1, H, W)
            hm_up = F.interpolate(
                hm, size=(Hi, Wi), mode="bilinear", align_corners=False
            )[0, 0]

            hm_np = hm_up.detach().cpu().numpy()
            bbox = find_high_activation_crop(hm_np, percentile=crop_percentile)
            overlay = overlay_heatmap(img_uint8, hm_up, alpha=0.45)
            overlay_box = draw_rect_on_image(overlay, bbox, color=(255, 0, 0), width=3)

            word = model.vocab_words[proto_idx]
            score = float(top_vals[b, rank].item())

            ax = axes[b][rank + 1]
            ax.imshow(overlay_box)
            ax.set_title(f"top{rank+1}: {word}\n{score:.3f}" if b == 0 else f"{word}\n{score:.3f}", fontsize=8)
            ax.axis("off")

    plt.tight_layout()

    wandb.log({
        "global_step": step,
        log_key: wandb.Image(fig),
    })
    plt.close(fig)


def _annealed_coef(epoch: int, total_epochs: int, init: float, final: float) -> float:
    """Cosine decay from `init` (epoch 0) to `final` (last epoch)."""
    if total_epochs <= 1:
        return final
    t = min(epoch / (total_epochs - 1), 1.0)
    return final + 0.5 * (init - final) * (1 + math.cos(math.pi * t))


def train(
    model: nn.Module,
    criterion: nn.Module,
    dataloader: DataLoader,
    epoch: int,
    optimizer: optim.Optimizer,
    logger: Logger,
    device: torch.device,
    clip_model: nn.Module,
    noun_embeddings: torch.Tensor,
    target_temperature: float = 0.01,
    *,
    vocab_to_idx=None,
    residual_eps: float = 0.0,
):
    model.train()

    running_losses = defaultdict(float)

    for i, batch in enumerate(tqdm(dataloader)):
        images, captions, target_dist, indices, *rest = batch
        images = images.to(device, non_blocking=True)
        target_dist = target_dist.to(device, non_blocking=True)
        words_sim_distribution = target_dist
        caption_embs = rest[0].to(device, non_blocking=True) if rest else None
        pool_lens = rest[1].to(device, non_blocking=True) if len(rest) > 1 else None

        # ---- DEBUG PRINT ----
        if i % 200 == 0:
            b = 0
            topk_vals, topk_idx = words_sim_distribution[b].topk(10)

            words = [model.vocab_words[j] for j in topk_idx.tolist()]
            weights = topk_vals.tolist()

            print("\nAll captions:")
            for c in captions[b]:
                print(" ", c)

            print("Top-10 words:")
            for w, s in zip(words, weights):
                print(f"  {w:15s} {s:.7f}")

        outputs = model(images)
        criterion_batch = (images, words_sim_distribution, indices, captions)
        if caption_embs is not None:
            criterion_batch = criterion_batch + (caption_embs,)
        if pool_lens is not None:
            criterion_batch = criterion_batch + (pool_lens,)
        loss_dict = criterion(outputs, criterion_batch, model)

        loss = sum(v for k, v in loss_dict.items() if not k.startswith("_"))
        if not isinstance(loss, torch.Tensor):
            raise ValueError("Loss is not a tensor")

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        # ℓ₂-ball projection for prototype_residual (PDF Eq. 5): δ ← δ · min(1, ε/‖δ‖)
        # Only runs when residual is being trained (residual_eps > 0 signals it's active)
        if residual_eps > 0 and model.prototype_residual.requires_grad:
            with torch.no_grad():
                norms = model.prototype_residual.norm(dim=-1, keepdim=True)  # [V, 1]
                model.prototype_residual.data.mul_((residual_eps / norms).clamp_(max=1.0))

        log_dict = {}
        for k, v in loss_dict.items():
            running_losses[k] += v.item() * images.size(0)
            log_dict[f"train/{k}"] = v.item()

        global_step = epoch * len(dataloader) + i
        log_dict["train/total_loss"] = loss.item()
        log_dict["train/sk_coef"] = criterion.sk_coef
        log_dict["train/koleo_coef"] = criterion.koleo_coef
        log_dict["epoch"] = epoch
        log_dict["global_step"] = global_step
        wandb.log(log_dict)

    for k, v in running_losses.items():
        loss_avg = v / len(dataloader.dataset)
        logger.info(f"EPOCH {epoch} train {k}: {loss_avg:.4f}")


@torch.inference_mode()
def test(
    model: nn.Module,
    criterion: nn.Module,
    dataloader: DataLoader,
    epoch: int,
    logger: Logger,
    device: torch.device,
    clip_model: nn.Module,
    *,
    train_steps_per_epoch: int,
    log_every: int = 50,
    vocab_to_idx=None,
    wandb_log_images: int = 8,
):
    model.eval()

    running_losses = defaultdict(float)
    num_samples = 0

    for i, batch in enumerate(tqdm(dataloader)):
        images, captions, target_dist, indices, *rest = batch

        global_step = epoch * train_steps_per_epoch + i

        images = images.to(device, non_blocking=True)
        target_dist = target_dist.to(device, non_blocking=True)
        words_sim_distribution = target_dist
        caption_embs = rest[0].to(device, non_blocking=True) if rest else None
        pool_lens = rest[1].to(device, non_blocking=True) if len(rest) > 1 else None

        # --------------------------
        # Model forward
        # --------------------------
        outputs = model(images)
        criterion_batch = (images, words_sim_distribution, indices, captions)
        if caption_embs is not None:
            criterion_batch = criterion_batch + (caption_embs,)
        if pool_lens is not None:
            criterion_batch = criterion_batch + (pool_lens,)
        loss_dict = criterion(outputs, criterion_batch, model)

        bs = images.size(0)
        num_samples += bs

        for k, v in loss_dict.items():
            running_losses[k] += v.item() * bs

        # --------------------------
        # Logging
        # --------------------------
        log_batches = set(
            random.sample(
                range(len(dataloader)),
                k=max(1, len(dataloader) // log_every)
            )
        )

        # choose random image in the batch
        b = random.randrange(images.shape[0])

        for i, batch in enumerate(dataloader):
            if i in log_batches:
                # choose random image in batch
                b = random.randrange(images.size(0))

                log_dict = {
                    "epoch": epoch,
                    "global_step": global_step,
                    "eval/batch_idx": i,
                    "eval/sample_idx": b,
                }
                if "mixture_weights" in outputs:
                    topk_vals, topk_idx = outputs["mixture_weights"].topk(k=7, dim=-1)

                    words = [
                        model.vocab_words[j]
                        for j in topk_idx[b].tolist()
                    ]

                    log_dict["eval/top_words"] = ", ".join(words)

                for k, v in loss_dict.items():
                    log_dict[f"eval/{k}"] = v.item()

                wandb.log(log_dict)

                # --------------------------
                # Visualization logging
                # --------------------------
                n_log = min(wandb_log_images, images.shape[0])
                wandb_log_top_proto_heatmaps(
                    model=model,
                    images=images[:n_log],
                    outputs={k: v[:n_log] if hasattr(v, "__getitem__") and getattr(v, "shape", None) is not None and len(
                        v.shape) > 0 and v.shape[0] == images.shape[0] else v
                             for k, v in outputs.items()},
                    step=global_step,
                    captions=captions[:n_log],
                    log_tsne=False,
                )
        # --------------------------
        # Epoch metrics
        # --------------------------
        avg_losses = {}

        for k, v in running_losses.items():
            avg_losses[k] = v / num_samples
            logger.info(f"EPOCH {epoch} test {k}: {avg_losses[k]:.4f}")

        avg_losses["total_loss"] = sum(
            v for k, v in avg_losses.items() if not k.startswith("_")
        )

        wandb.log({
            "epoch": epoch,
            "global_step": epoch * train_steps_per_epoch + len(dataloader) - 1,
            **{f"test/{k}": v for k, v in avg_losses.items()},
        })

        return avg_losses

def build_backbone(args):
    if "dinov2" in args.backbone:
        if args.num_splits and args.num_splits > 0:
            backbone = DINOv2BackboneExpanded(
                name=args.backbone,
                n_splits=args.num_splits,
                mode="append",
                freeze_norm_layer=True,
            )
        else:
            backbone = DINOv2Backbone(name=args.backbone)

        dim = backbone.dim

    elif "dino" in args.backbone:
        backbone = DINOBackboneExpanded(
            name=args.backbone,
            n_splits=args.num_splits,
            mode="block_expansion",
            freeze_norm_layer=True,
        )
        dim = backbone.dim
    elif "clip" in args.backbone:
        backbone = CLIPBackbone(name=args.backbone)
        dim = backbone.dim
    else:
        raise NotImplementedError(f"Backbone {args.backbone} not implemented.")

    # ---------------------------------------------------
    # Freeze everything first
    # ---------------------------------------------------
    #for p in backbone.parameters():
    #    p.requires_grad = False

    return backbone, dim


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    parser = argparse.ArgumentParser()

    parser.add_argument("--log-dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--dataset", type=str, default="coco_clip",
                        choices=["coco_clip", "caltech101", "cub200", "awa2", "visual_genome", "coco_vg"])
    parser.add_argument("--coco-root", type=str, default="/data/pwojcik/UnGuide/coco30_bck/")
    parser.add_argument("--caltech-root", type=str, default=None, help="Path to caltech101 directory")
    parser.add_argument("--caltech-descriptions", type=str, default=None, help="Path to caltech101_descriptions.json")
    parser.add_argument("--cub-root", type=str, default=None, help="Path to CUB-200-2011 organized directory")
    parser.add_argument("--cub-annotations", type=str, default=None, help="Path to CUB annotations dir (contains attributes/)")
    parser.add_argument("--awa-root", type=str, default=None, help="Path to AwA2 organized directory")
    parser.add_argument("--awa-annotations", type=str, default=None, help="Path to AwA2 annotations dir (contains predicates.txt)")
    parser.add_argument("--vg-root", type=str, default=None,
                        help="Path to VG image root (contains VG_100K/ and VG_100K_2/)")
    parser.add_argument("--vg-region-descriptions", type=str, default=None,
                        help="Path to region_descriptions.json (VG v1.4)")
    parser.add_argument("--vg-val-ratio", type=float, default=0.1,
                        help="Fraction of VG images used for validation (default: 0.1)")
    parser.add_argument("--coco-annotations-train", type=str, default="/data/pwojcik/coco_2014/annotations/captions_train2014.json")
    parser.add_argument("--coco-annotations-val", type=str, default="/data/pwojcik/coco_2014/annotations/captions_val2014.json")
    parser.add_argument("--coco-val-ratio", type=float, default=0.1)
    parser.add_argument("--coco-clip-model-name", type=str, default="ViT-B-32")
    parser.add_argument("--coco-clip-pretrained", type=str, default="openai")
    parser.add_argument("--visual-coef", type=float, default=0.0)
    parser.add_argument("--cover-coef", type=float, default=0.0)
    parser.add_argument("--caption-coef", type=float, default=0.0,
                        help="Weight for caption-level cosine alignment loss (VG only; default: 0.0)")
    parser.add_argument("--caption-embeds-path", type=str, default=None,
                        help="Path to per-image CLIP phrase embedding pool "
                             "built by vocab/build_vg_caption_embeddings.py")
    parser.add_argument("--caption-sample-k", type=int, default=5,
                        help="Phrases randomly sampled per image at training time (default: 5)")
    parser.add_argument("--caption-pool-size", type=int, default=50,
                        help="Max phrases per image in the offline pool (informational; default: 50)")

    parser.add_argument("--clip-scores-coco-train", type=str, default=None,
                        help="Path to CLIP vocab scores .pt for COCO train split "
                             "(built by build_clip_vocab_scores.py). Replaces caption-derived targets.")
    parser.add_argument("--clip-scores-coco-val", type=str, default=None,
                        help="Path to CLIP vocab scores .pt for COCO val split.")
    parser.add_argument("--clip-scores-vg", type=str, default=None,
                        help="Path to CLIP vocab scores .pt for Visual Genome "
                             "(covers all images; works for both train and val splits).")
    parser.add_argument("--clip-scores-temperature", type=float, default=0.07,
                        help="Softmax temperature applied to raw CLIP cosine similarities "
                             "when loading score files. 0.07 matches CLIP's own training "
                             "temperature and gives well-peaked distributions. "
                             "Higher values (e.g. 1.0) approach uniform — avoid.")
    parser.add_argument("--clip-scores-top-k", type=int, default=50,
                        help="Number of top-scoring vocab words to keep per image before "
                             "softmax. With 12K+ vocab words, CLIP similarities are too "
                             "close together to produce peaked distributions without masking. "
                             "top_k=50 gives max entropy log2(50)≈5.6 bits.")
    parser.add_argument("--clip-scores-caption-filter", action="store_true", default=False,
                        help="Before top-k selection, mask out vocab words that do not appear "
                             "in the image's captions. Prevents synonyms (e.g. 'feline') from "
                             "filling top-k slots when the caption uses 'cat'. Falls back to "
                             "unmasked scores for images with fewer than top-k caption words "
                             "in the vocabulary.")


    parser.add_argument(
        "--backbone",
        type=str,
        default="dinov2_vitb14",
        choices=["dinov2_vitb14", "dinov2_vits14", "dinov2_vitl14", "clip_vitb32", "dino_vitb16"],
    )
    parser.add_argument("--clip-model-name", type=str, default="ViT-L-14")
    parser.add_argument("--clip-pretrained", type=str, default="openai")
    parser.add_argument("--clip-patch-size", type=int, default=16)
    parser.add_argument("--freeze-backbone", action="store_true", default=False)
    parser.add_argument("--num-splits", type=int, default=1)
    parser.add_argument(
        "--unfreeze-last-blocks",
        type=int,
        default=0,
        help="Number of last transformer blocks to unfreeze in the backbone",
    )

    parser.add_argument("--vocab-cache-path", type=str, default="vocab/mscoco_new_cache.pt")
    parser.add_argument("--clip-text-dim", type=int, default=512)
    parser.add_argument("--kl-coef", type=float, default=1.0)
    parser.add_argument("--target-mode", type=str, default="prob",
                        choices=["prob", "binary", "topk", "uniform"],
                        help="prob=frequency-weighted KL; binary=0/1 BCE; "
                             "topk=top-K uniform KL; uniform=equal weight over all present words")
    parser.add_argument("--top-k-concepts", type=int, default=10,
                        help="K for --target-mode=topk: keep only top-K concepts per image, uniform 1/K weight")
    parser.add_argument("--bce-coef", type=float, default=1.0,
                        help="Weight for BCE loss (used when --target-mode=binary)")
    parser.add_argument("--bce-pos-weight", type=float, default=100.0,
                        help="pos_weight for BCE to counter class imbalance (used when --target-mode=binary)")
    parser.add_argument("--text-proj-hidden-dim", type=int, default=768)
    parser.add_argument("--prototype-init-noise", type=float, default=0.01)
    parser.add_argument("--save-every", type=int, default=0,
                        help="Save a milestone checkpoint ckpt_ep{N:03d}.pth every N epochs "
                             "(0 = disabled). Useful for collapse detection and post-hoc "
                             "per-epoch evaluation. Does not replace the rolling ckpt.pth.")
    parser.add_argument("--residual-lr", type=float, default=0.0,
                        help="LR for prototype_residual; 0 = keep frozen (current default)")
    parser.add_argument("--residual-eps", type=float, default=0.1,
                        help="ℓ₂-ball radius for residual projection (PDF Eq. 5); only active when --residual-lr > 0")
    parser.add_argument("--residual-reg-coef", type=float, default=0.0,
                        help="Weight for soft residual regularization Lreg = (1/V)Σ‖δ_v‖² (PDF Eq. 14)")
    parser.add_argument("--loss-type", type=str, default="kl", choices=["kl", "jsd"],
                        help="Distribution alignment loss: 'kl' (default) or 'jsd' (Jensen-Shannon; "
                             "pairs best with --target-mode topk for clean zero entries)")
    parser.add_argument("--temperature", type=float, default=0.2)

    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--bin-coef", type=float, default=0.1)
    parser.add_argument("--backbone-lr", type=float, default=1.0e-5)
    parser.add_argument("--text-proj-lr", type=float, default=1.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--lr-schedule", type=str, default="cosine", choices=["none", "cosine"],
                        help="LR schedule: cosine decay to 0 over all epochs, or none (fixed LR)")
    parser.add_argument("--lr-warmup-epochs", type=int, default=0,
                        help="Linear warmup epochs before cosine decay (0 = no warmup)")

    parser.add_argument("--cosine-coef", type=float, default=1.0)
    parser.add_argument("--entropy-coef", type=float, default=0.0)
    parser.add_argument("--contrastive-coef", type=float, default=0.0,
                        help="Weight for in-batch symmetric InfoNCE loss on pred_text_embedding "
                             "vs CLIP phrase embeddings. Requires --caption-embeds-path.")
    parser.add_argument("--contrastive-temp", type=float, default=0.07,
                        help="Temperature for the contrastive similarity matrix (default: 0.07)")
    parser.add_argument("--contrastive-label-temp", type=float, default=0.0,
                        help="Temperature for soft-negative labels built from phrase-phrase "
                             "cosine similarity. 0.0 = standard hard InfoNCE (default). "
                             "Typical value: 0.5.")
    parser.add_argument("--contrastive-hard-mining", action="store_true",
                        help="Select top-k phrases per image by cosine similarity to "
                             "pred_text_embedding (online hard positive mining) instead of "
                             "random sampling. Also deduplicates the phrase pool. "
                             "Requires --caption-embeds-path.")
    parser.add_argument("--sk-coef", type=float, default=0.0,
                        help="Weight for Sinkhorn-Knopp batch diversity loss. "
                             "Calibrated default: 0.1 (l_sk≈9 nats for cosine logits).")
    parser.add_argument("--sk-eps", type=float, default=0.10,
                        help="Sinkhorn temperature. 0.10 gives H/H_max≈0.89 for cosine-scale "
                             "vocab_logits (std≈0.15); SwAV default 0.05 is too small here.")
    parser.add_argument("--sk-n-iter", type=int, default=3,
                        help="Sinkhorn-Knopp normalisation iterations (default: 3).")
    parser.add_argument("--sk-coef-init", type=float, default=None,
                        help="If set, anneal --sk-coef from this value down to --sk-coef "
                             "(steady-state) via cosine decay over training. "
                             "Default: None = constant --sk-coef (previous behavior).")
    parser.add_argument("--sk-prior-tau", type=float, default=0.0,
                        help="PMSN-style non-uniform Sinkhorn prior: target vocab marginal "
                             "∝ empirical_word_freq^tau. 0 = uniform (previous behavior); "
                             "1 = full empirical distribution; 0.5 = tempered. Matches "
                             "long-tailed (Zipfian) VG vocab instead of forcing equal "
                             "mass onto rare words.")
    parser.add_argument("--wandb-entity", type=str, default=None,
                        help="W&B entity (team/org) to log runs under. Defaults to personal account.")
    parser.add_argument("--wandb-log-images", type=int, default=8,
                        help="Number of images to visualize per W&B log step (default: 8)")
    parser.add_argument("--koleo-coef", type=float, default=0.0,
                        help="Weight for KoLeo nearest-neighbour repulsion on "
                             "pred_text_embedding. DINOv3 default: 0.1.")
    parser.add_argument("--koleo-coef-init", type=float, default=None,
                        help="Same as --sk-coef-init but for --koleo-coef.")
    parser.add_argument("--vicreg-coef", type=float, default=0.0,
                        help="Weight for VICReg variance+covariance anti-collapse loss on "
                             "pred_text_embedding (single-view, no invariance term). "
                             "Alternative to KoLeo; the covariance term also fights "
                             "dimensional collapse. Default 0 (off).")
    parser.add_argument("--msn-coef", type=float, default=0.0,
                        help="Weight for MSN masked-prediction loss. Default 0 (off).")
    parser.add_argument("--msn-mask-ratio", type=float, default=0.25,
                        help="Fraction of patches to mask in MSN anchor pass "
                             "(0.25 = 64/256 patches for ViT-L/14@224).")
    parser.add_argument("--ibot-coef", type=float, default=0.0,
                        help="Weight for iBOT per-patch masked CE loss. "
                             "Requires --msn-mask-ratio > 0.")
    parser.add_argument("--sigreg-coef", type=float, default=0.0,
                        help="Weight for SigReg ECF goodness-of-fit loss on pred_text_embedding. "
                             "LeJEPA default: 0.02.")
    parser.add_argument("--sigreg-sketch-dim", type=int, default=64,
                        help="Random projection dimension for SigReg (default: 64).")
    parser.add_argument("--agg-mode", type=str, default="topk", choices=("topk", "cross_attn"),
                        help="Patch->vocab aggregation: 'topk' (default, mean of top-k patch "
                             "similarities per concept) or 'cross_attn' (learnable-temperature "
                             "softmax attention over patches; ablation, underperformed topk).")
    parser.add_argument("--topk-k", type=int, default=5,
                        help="k for --agg-mode topk (default: 5).")
    parser.add_argument("--attn-temp-init", type=float, default=0.1,
                        help="Initial value of the learnable patch attention temperature τ "
                             "(only used when --agg-mode cross_attn). Small = sparse "
                             "attention (≈top-1), large = uniform pooling.")

    args = parser.parse_args()

    if (args.caption_coef != 0 or args.contrastive_coef != 0) and not args.caption_embeds_path:
        parser.error("--caption-coef and --contrastive-coef require --caption-embeds-path "
                     "(build with vocab/build_vg_caption_embeddings.py --split both)")

    wandb.init(
        entity=args.wandb_entity,
        project="proto-non-param",
        config=vars(args),
        dir=args.log_dir,
    )

    wandb.define_metric("global_step")
    wandb.define_metric("train/*", step_metric="global_step")
    wandb.define_metric("test/*", step_metric="global_step")
    wandb.define_metric("eval/*", step_metric="global_step")

    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s][%(name)s][%(levelname)s] - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.FileHandler((log_dir / "train.log").as_posix()),
            logging.StreamHandler(sys.stdout),
        ],
        force=True,
    )
    logger = logging.getLogger(__name__)

    L.seed_everything(args.seed)

    logger.info(f"Train on {args.dataset} dataset")

    cache = torch.load(args.vocab_cache_path, map_location="cpu")
    vocab_words = list(cache.keys())
    vocab_to_idx = {w: i for i, w in enumerate(vocab_words)}

    print('Building datasets')

    if args.dataset == "caltech101":
        if args.caltech_root is None or args.caltech_descriptions is None:
            raise ValueError("--caltech-root and --caltech-descriptions are required for caltech101 dataset")
        dataset_train = Caltech101CLIPDataset(
            descriptions_json=args.caltech_descriptions,
            caltech_root=args.caltech_root,
            vocab_to_idx=vocab_to_idx,
            train=True,
            target_type=args.target_mode,
            top_k_concepts=args.top_k_concepts,
        )
        dataset_test = Caltech101CLIPDataset(
            descriptions_json=args.caltech_descriptions,
            caltech_root=args.caltech_root,
            vocab_to_idx=vocab_to_idx,
            train=False,
            target_type=args.target_mode,
            top_k_concepts=args.top_k_concepts,
        )
    elif args.dataset == "cub200":
        if args.cub_root is None or args.cub_annotations is None:
            raise ValueError("--cub-root and --cub-annotations are required for cub200 dataset")
        dataset_train = CUBCLIPDataset(
            dataset_root=args.cub_root,
            annotations_dir=args.cub_annotations,
            vocab_to_idx=vocab_to_idx,
            train=True,
        )
        dataset_test = CUBCLIPDataset(
            dataset_root=args.cub_root,
            annotations_dir=args.cub_annotations,
            vocab_to_idx=vocab_to_idx,
            train=False,
        )
    elif args.dataset == "awa2":
        if args.awa_root is None or args.awa_annotations is None:
            raise ValueError("--awa-root and --awa-annotations are required for awa2 dataset")
        dataset_train = AwA2CLIPDataset(
            dataset_root=args.awa_root,
            annotations_dir=args.awa_annotations,
            vocab_to_idx=vocab_to_idx,
            train=True,
        )
        dataset_test = AwA2CLIPDataset(
            dataset_root=args.awa_root,
            annotations_dir=args.awa_annotations,
            vocab_to_idx=vocab_to_idx,
            train=False,
        )
    elif args.dataset == "visual_genome":
        if args.vg_root is None or args.vg_region_descriptions is None:
            raise ValueError("--vg-root and --vg-region-descriptions are required for visual_genome dataset")
        dataset_train = VisualGenomeDataset(
            vg_root=args.vg_root,
            region_descriptions_json=args.vg_region_descriptions,
            vocab_to_idx=vocab_to_idx,
            train=True,
            val_ratio=args.vg_val_ratio,
            seed=args.seed,
            target_type=args.target_mode,
            top_k_concepts=args.top_k_concepts,
            caption_embeds_path=args.caption_embeds_path,
            caption_sample_k=args.caption_sample_k,
            hard_mining=args.contrastive_hard_mining,
            caption_pool_size=args.caption_pool_size,
        )
        dataset_test = VisualGenomeDataset(
            vg_root=args.vg_root,
            region_descriptions_json=args.vg_region_descriptions,
            vocab_to_idx=vocab_to_idx,
            train=False,
            val_ratio=args.vg_val_ratio,
            seed=args.seed,
            target_type=args.target_mode,
            top_k_concepts=args.top_k_concepts,
            caption_embeds_path=args.caption_embeds_path,
            caption_sample_k=args.caption_sample_k,
            hard_mining=args.contrastive_hard_mining,
            caption_pool_size=args.caption_pool_size,
        )
        if args.clip_scores_vg:
            dataset_train = CLIPScoreDataset(dataset_train, args.clip_scores_vg, args.clip_scores_temperature, args.clip_scores_top_k, args.clip_scores_caption_filter)
            dataset_test = CLIPScoreDataset(dataset_test, args.clip_scores_vg, args.clip_scores_temperature, args.clip_scores_top_k, args.clip_scores_caption_filter)
    elif args.dataset == "coco_vg":
        if args.vg_root is None or args.vg_region_descriptions is None:
            raise ValueError("--vg-root and --vg-region-descriptions are required for coco_vg dataset")
        from torch.utils.data import ConcatDataset

        _vg_train = VisualGenomeDataset(
            vg_root=args.vg_root,
            region_descriptions_json=args.vg_region_descriptions,
            vocab_to_idx=vocab_to_idx,
            train=True,
            val_ratio=args.vg_val_ratio,
            seed=args.seed,
            target_type=args.target_mode,
            top_k_concepts=args.top_k_concepts,
        )
        _coco_train = CocoCLIPDataset(
            annotations_json=args.coco_annotations_train,
            coco_root=args.coco_root,
            vocab_to_idx=vocab_to_idx,
            train=True,
            target_type=args.target_mode,
            top_k_concepts=args.top_k_concepts,
        )
        if args.clip_scores_vg:
            _vg_train = CLIPScoreDataset(_vg_train, args.clip_scores_vg, args.clip_scores_temperature, args.clip_scores_top_k, args.clip_scores_caption_filter)
        if args.clip_scores_coco_train:
            _coco_train = CLIPScoreDataset(_coco_train, args.clip_scores_coco_train, args.clip_scores_temperature, args.clip_scores_top_k, args.clip_scores_caption_filter)
        dataset_train = ConcatDataset([_vg_train, _coco_train])

        _vg_val = VisualGenomeDataset(
            vg_root=args.vg_root,
            region_descriptions_json=args.vg_region_descriptions,
            vocab_to_idx=vocab_to_idx,
            train=False,
            val_ratio=args.vg_val_ratio,
            seed=args.seed,
            target_type=args.target_mode,
            top_k_concepts=args.top_k_concepts,
        )
        _coco_val = CocoCLIPDataset(
            annotations_json=args.coco_annotations_val,
            coco_root=args.coco_root,
            vocab_to_idx=vocab_to_idx,
            train=False,
            target_type=args.target_mode,
            top_k_concepts=args.top_k_concepts,
        )
        if args.clip_scores_vg:
            _vg_val = CLIPScoreDataset(_vg_val, args.clip_scores_vg, args.clip_scores_temperature, args.clip_scores_top_k, args.clip_scores_caption_filter)
        if args.clip_scores_coco_val:
            _coco_val = CLIPScoreDataset(_coco_val, args.clip_scores_coco_val, args.clip_scores_temperature, args.clip_scores_top_k, args.clip_scores_caption_filter)
        dataset_test = ConcatDataset([_vg_val, _coco_val])
    else:
        dataset_train = CocoCLIPDataset(
            annotations_json=args.coco_annotations_train,
            coco_root=args.coco_root,
            vocab_to_idx=vocab_to_idx,
            train=True,
            target_type=args.target_mode,
            top_k_concepts=args.top_k_concepts,
        )
        dataset_test = CocoCLIPDataset(
            annotations_json=args.coco_annotations_val,
            coco_root=args.coco_root,
            vocab_to_idx=vocab_to_idx,
            train=False,
            target_type=args.target_mode,
            top_k_concepts=args.top_k_concepts,
        )
        if args.clip_scores_coco_train:
            dataset_train = CLIPScoreDataset(dataset_train, args.clip_scores_coco_train, args.clip_scores_temperature, args.clip_scores_top_k, args.clip_scores_caption_filter)
        if args.clip_scores_coco_val:
            dataset_test = CLIPScoreDataset(dataset_test, args.clip_scores_coco_val, args.clip_scores_temperature, args.clip_scores_top_k, args.clip_scores_caption_filter)

    print('Done with datasets')
    print('Train: ', len(dataset_train))
    print('Test: ', len(dataset_test))

    _collate = (
        vg_collate_fn
        if (args.dataset == "visual_genome" and args.caption_embeds_path is not None)
        or args.contrastive_coef != 0
        else coco_clip_collate_fn
    )

    dataloader_train = DataLoader(
        dataset=dataset_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=_collate,
        shuffle=True,
        pin_memory=True,
    )

    dataloader_test = DataLoader(
        dataset=dataset_test,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=_collate,
        shuffle=False,
        pin_memory=True,
    )

    backbone, dim = build_backbone(args)

    clip_model, _, _ = open_clip.create_model_and_transforms(
        args.coco_clip_model_name,
        pretrained=args.coco_clip_pretrained,
    )
    clip_model = clip_model.eval().to(device)

    for p in clip_model.parameters():
        p.requires_grad = False

    net = PNP(
        backbone=backbone,
        dim=dim,
        temperature=args.temperature,
        clip_text_dim=args.clip_text_dim,
        text_proj_hidden_dim=args.text_proj_hidden_dim,
        vocab_cache_path=args.vocab_cache_path,
        prototype_init_noise=0.0 if args.residual_lr == 0 else args.prototype_init_noise,
        clip_model=clip_model,
        msn_mask_ratio=args.msn_mask_ratio,
        agg_mode=args.agg_mode,
        topk_k=args.topk_k,
        attn_temp_init=args.attn_temp_init,
    )
    # freeze backbone first
    #for p in net.backbone.parameters():
    #    p.requires_grad = False

    bb = net.backbone
    print("Backbone class:", type(bb))

    if args.unfreeze_last_blocks > 0:
        print("Backbone child modules:", list(bb._modules.keys()))

        blocks = None

        # common cases
        if hasattr(bb, "model") and hasattr(bb.model, "blocks"):
            blocks = bb.model.blocks
        elif hasattr(bb, "blocks"):
            blocks = bb.blocks
        else:
            # search one level deeper
            for child_name, child in bb.named_children():
                print(f"Inspect child: {child_name} -> {type(child)}")
                if hasattr(child, "blocks"):
                    blocks = child.blocks
                    print(f"Found transformer blocks in bb.{child_name}.blocks")
                    break
                if hasattr(child, "model") and hasattr(child.model, "blocks"):
                    blocks = child.model.blocks
                    print(f"Found transformer blocks in bb.{child_name}.model.blocks")
                    break

        if blocks is None:
            print("All backbone parameter names:")
            for name, _ in bb.named_parameters():
                print(name)
            raise AttributeError("Could not find transformer blocks in net.backbone")

        #n_blocks = len(blocks)
        #start = max(0, n_blocks - args.unfreeze_last_blocks)

        #for block in blocks[start:]:
        #    for p in block.parameters():
        #        p.requires_grad = True

        #print(f"Unfroze last {args.unfreeze_last_blocks} transformer blocks")

        for name, p in bb.named_parameters():
            if p.requires_grad:
                print("TRAINABLE BACKBONE:", name)

    sk_prior = None
    if args.sk_prior_tau > 0:
        if not hasattr(dataset_train, "samples"):
            raise ValueError("--sk-prior-tau requires a dataset with cached .samples "
                             "(word-frequency marginal is computed from them)")
        freqs = torch.zeros(len(dataset_train.samples[0][2]))
        for s in dataset_train.samples:
            freqs += s[2]
        sk_prior = freqs.clamp_min(1e-8).pow(args.sk_prior_tau)
        sk_prior /= sk_prior.sum()
        print(f"SK prior (tau={args.sk_prior_tau}): "
              f"max={sk_prior.max():.2e} min={sk_prior.min():.2e} "
              f"effective vocab={(1.0 / sk_prior.pow(2).sum()).item():.0f}/{len(sk_prior)}")

    criterion = PNPCriterion(
        kl_coef=args.kl_coef,
        entropy_coef=args.entropy_coef,
        visual_coef=args.visual_coef,
        bin_coef=args.bin_coef,
        cover_coef=args.cover_coef,
        temperature=args.temperature,
        use_binary=(args.target_mode == "binary"),
        bce_coef=args.bce_coef,
        pos_weight_val=args.bce_pos_weight,
        caption_coef=args.caption_coef,
        loss_type=args.loss_type,
        residual_reg_coef=args.residual_reg_coef,
        contrastive_coef=args.contrastive_coef,
        contrastive_temp=args.contrastive_temp,
        contrastive_label_temp=args.contrastive_label_temp,
        contrastive_k=args.caption_sample_k,
        sk_coef=args.sk_coef,
        sk_eps=args.sk_eps,
        sk_n_iter=args.sk_n_iter,
        sk_prior=sk_prior,
        koleo_coef=args.koleo_coef,
        vicreg_coef=args.vicreg_coef,
        msn_coef=args.msn_coef,
        ibot_coef=args.ibot_coef,
        sigreg_coef=args.sigreg_coef,
        sigreg_sketch_dim=args.sigreg_sketch_dim,
    )

    net.to(device)
    criterion.to(device)

    param_groups = [
        {"params": net.text_projection_head.parameters(), "lr": args.text_proj_lr},
    ]
    # add backbone as separate group
    backbone_params = [p for p in net.backbone.parameters() if p.requires_grad]
    if backbone_params:
        param_groups.append({
            "params": backbone_params,
            "lr": args.backbone_lr,
        })
    # add prototype_residual if requested (default residual_lr=0 keeps old frozen behavior)
    if args.residual_lr > 0:
        param_groups.append({
            "params": [net.prototype_residual],
            "lr": args.residual_lr,
        })

    optimizer = optim.AdamW(param_groups, weight_decay=args.weight_decay)

    if args.lr_schedule == "cosine":
        if args.lr_warmup_epochs > 0:
            warmup_sched = optim.lr_scheduler.LinearLR(
                optimizer, start_factor=1e-3, total_iters=args.lr_warmup_epochs
            )
            cosine_sched = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=args.epochs - args.lr_warmup_epochs, eta_min=0
            )
            scheduler = optim.lr_scheduler.SequentialLR(
                optimizer, schedulers=[warmup_sched, cosine_sched],
                milestones=[args.lr_warmup_epochs]
            )
        else:
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=args.epochs, eta_min=0
            )
    else:
        scheduler = None

    print_parameters(net=net, logger=logger)

    best_epoch = 0
    best_val_cosine = float("-inf")

    cache = torch.load(args.vocab_cache_path, map_location="cpu")
    vocab_words = list(cache.keys())
    vocab_to_idx = {w: i for i, w in enumerate(vocab_words)}
    noun_embeddings = torch.stack([cache[w] for w in vocab_words], dim=0)
    noun_embeddings = F.normalize(noun_embeddings, dim=-1).to(device)

    for epoch in range(args.epochs):
        if args.sk_coef_init is not None:
            criterion.sk_coef = _annealed_coef(epoch, args.epochs, args.sk_coef_init, args.sk_coef)
        if args.koleo_coef_init is not None:
            criterion.koleo_coef = _annealed_coef(epoch, args.epochs, args.koleo_coef_init, args.koleo_coef)

        train(
            model=net,
            criterion=criterion,
            dataloader=dataloader_train,
            epoch=epoch,
            optimizer=optimizer,
            logger=logger,
            device=device,
            clip_model=clip_model,
            noun_embeddings=noun_embeddings,
            target_temperature=0.01,
            vocab_to_idx=vocab_to_idx,
            residual_eps=args.residual_eps if args.residual_lr > 0 else 0.0,
        )

        epoch_metrics = test(
            model=net,
            criterion=criterion,
            dataloader=dataloader_test,
            epoch=epoch,
            logger=logger,
            device=device,
            clip_model=clip_model,
            train_steps_per_epoch=len(dataloader_train),
            vocab_to_idx=vocab_to_idx,
            wandb_log_images=args.wandb_log_images,
        )

        if scheduler is not None:
            scheduler.step()

        epoch_metric = -sum(
            v for k, v in epoch_metrics.items()
            if k.startswith("test/") and not k.startswith("test/_")
        )
        ckpt_payload = {
            "state_dict": {k: v.detach().cpu() for k, v in net.state_dict().items()},
            "hparams": vars(args),
        }
        torch.save(ckpt_payload, log_dir / "ckpt.pth")
        logger.info("Model saved as ckpt.pth")

        if args.save_every > 0 and (epoch + 1) % args.save_every == 0:
            milestone_path = log_dir / f"ckpt_ep{epoch + 1:03d}.pth"
            torch.save(ckpt_payload, milestone_path)
            logger.info(f"Milestone checkpoint saved: {milestone_path.name}")

        if epoch_metric > best_val_cosine:
            best_val_cosine = epoch_metric
            best_epoch = epoch

    logger.info(f"DONE! Best epoch is epoch {best_epoch} with cosine similarity {best_val_cosine:.4f}.")


if __name__ == "__main__":
    main()