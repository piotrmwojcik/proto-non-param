#!/usr/bin/env python3
from html import parser
import sys
import logging
from collections import defaultdict
from logging import Logger
from pathlib import Path
import numpy as np
import math
import random
import os
import nltk
from nltk.stem import WordNetLemmatizer
import open_clip
from collections import defaultdict
import argparse
from sklearn.manifold import TSNE
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

from vg_dataset import build_webdataset, collate_batch
import wandb
import lightning as L
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn.functional as F

from clip_dataset import CocoCLIPDataset, coco_clip_collate_fn
from modeling.backbone import (
    DINOv2Backbone,
    DINOv2BackboneExpanded,
    DINOBackboneExpanded,
    CLIPBackbone,
)
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


def overlay_heatmap(
    img_uint8: np.ndarray, hm: torch.Tensor, alpha: float = 0.45
) -> np.ndarray:
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
    patch_logits = outputs["patch_prototype_logits"]  # [B, N, V]
    mix_weights = outputs["mixture_weights"]  # [B, V]

    B, N, V = patch_logits.shape
    H = W = int(math.sqrt(N))
    _, _, Hi, Wi = images.shape

    top_vals, top_idx = mix_weights.topk(k=top_k, dim=-1)  # [B, K]

    sample_grids = []
    B_log = min(B, max_items)

    for b in range(B_log):
        img_uint8 = denorm_to_uint8(images[b], mean=mean, std=std)
        raw_caption = str(captions[b]) if captions is not None else ""

        words = [model.vocab_words[j] for j in top_idx[b].tolist()]

        panel_imgs = [img_uint8]
        panel_titles = ["raw"]

        for rank, proto_idx in enumerate(top_idx[b].tolist()):
            hm = patch_logits[b, :, proto_idx].view(1, 1, H, W)
            hm_up = F.interpolate(
                hm, size=(Hi, Wi), mode="bilinear", align_corners=False
            )[0, 0]

            hm_np = hm_up.detach().cpu().numpy()

            # find bounding box on the upsampled heatmap
            bbox = find_high_activation_crop(hm_np, percentile=crop_percentile)

            # draw rectangle on overlay
            overlay = overlay_heatmap(img_uint8, hm_up, alpha=0.45)
            overlay_box = draw_rect_on_image(overlay, bbox, color=(255, 0, 0), width=3)

            word = model.vocab_words[proto_idx]
            score = float(top_vals[b, rank].item())

            panel_imgs.append(overlay_box)
            panel_titles.append(f"top{rank+1}: {word}\n{score:.3f}")

        n_panels = len(panel_imgs)
        ncols = min(3, n_panels)
        nrows = int(math.ceil(n_panels / ncols))

        fig, axes = plt.subplots(
            nrows=nrows,
            ncols=ncols,
            figsize=(4 * ncols, 4 * nrows),
            dpi=120,
        )

        if not isinstance(axes, np.ndarray):
            axes = np.array([axes])
        axes = axes.flatten()

        for ax, im, title in zip(axes, panel_imgs, panel_titles):
            ax.imshow(im)
            ax.set_title(title, fontsize=10)
            ax.axis("off")

        for ax in axes[len(panel_imgs) :]:
            ax.axis("off")

        suptitle = raw_caption
        if words:
            suptitle += f"\nTop words: {', '.join(words)}"

        fig.suptitle(suptitle, fontsize=11)
        plt.tight_layout(rect=[0, 0, 1, 0.92])

        sample_grids.append(wandb.Image(fig, caption=f"sample={b}"))
        plt.close(fig)


        log_dict = {
            "global_step": step,
            log_key: sample_grids,
        }

        print(f"W&B logging {len(sample_grids)} heatmap image(s) to {log_key} at step={step}")
        wandb.log(log_dict, step=step)

def _slice_batch_outputs(outputs: dict, batch_idx: int, batch_size: int) -> dict:
    sliced = {}

    for k, v in outputs.items():
        if torch.is_tensor(v) and v.ndim > 0 and v.shape[0] == batch_size:
            sliced[k] = v[batch_idx : batch_idx + 1]
        else:
            sliced[k] = v

    return sliced

def train(
    model,
    criterion,
    dataloader,
    optimizer,
    logger,
    device,
    max_steps,
    *,
    start_step=0,
    vocab_to_idx=None,
):
    model.train()

    running_losses = defaultdict(float)
    num_samples = 0

    for i, batch in enumerate(tqdm(dataloader)):
        if i >= 10:
            break

        images, captions, target_dist, indices = batch

        bs = images.size(0)
        num_samples += bs

        images = images.to(device, non_blocking=True)
        target_dist = target_dist.to(device, non_blocking=True)

        words_sim_distribution = target_dist.clamp_min(1e-8)

        if i % 200 == 0:
            b = 0

            raw_dist = target_dist[b]
            dist = words_sim_distribution[b]

            topk_vals, topk_idx = raw_dist.topk(10)

            print("\nAll captions:")
            for c in captions[b]:
                print(" ", c)

            print("\nDistribution stats:")
            print(f"  shape:          {tuple(raw_dist.shape)}")
            print(f"  raw sum:        {raw_dist.sum().item():.7f}")
            print(f"  raw min:        {raw_dist.min().item():.7f}")
            print(f"  raw max:        {raw_dist.max().item():.7f}")
            print(f"  raw mean:       {raw_dist.mean().item():.7f}")
            print(f"  raw nonzero:    {(raw_dist > 0).sum().item()} / {raw_dist.numel()}")
            print(f"  clamped min:    {dist.min().item():.7f}")

            print("\nTop-10 words:")
            for idx, val in zip(topk_idx.tolist(), topk_vals.tolist()):
                word = model.vocab_words[idx]
                print(f"  {idx:6d}  {word:20s}  {val:.7f}")

            print("\nFull nonzero raw distribution:")
            for idx, val in enumerate(raw_dist.tolist()):
                if val > 0:
                    word = model.vocab_words[idx]
                    print(f"  {idx:6d}  {word:20s}  {val:.7f}")

        outputs = model(images)

        loss_dict = criterion(
            outputs,
            (images, words_sim_distribution, indices, captions),
            model,
        )

        loss = sum(v for k, v in loss_dict.items() if not k.startswith("_"))

        if not isinstance(loss, torch.Tensor):
            raise ValueError("Loss is not a tensor")

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        global_step = start_step + i

        log_dict = {
            "global_step": global_step,
            "train/total_loss": loss.item(),
        }

        for k, v in loss_dict.items():
            running_losses[k] += v.item() * bs
            log_dict[f"train/{k}"] = v.item()

        wandb.log(log_dict, step=global_step)

    for k, v in running_losses.items():
        loss_avg = v / max(1, num_samples)
        logger.info(f"TRAIN {k}: {loss_avg:.4f}")

@torch.inference_mode()
def test(
    model,
    criterion,
    dataloader,
    logger,
    device,
    *,
    global_step: int,
    log_every: int = 50,
):
    model.eval()

    running_losses = defaultdict(float)
    num_samples = 0

    print("\nSTARTING EVALUATION\n")

    for i, batch in enumerate(tqdm(dataloader)):
        images, captions, target_dist, indices = batch

        images = images.to(device, non_blocking=True)
        target_dist = target_dist.to(device, non_blocking=True)

        words_sim_distribution = target_dist.clamp_min(1e-8)

        outputs = model(images)

        print("--!!! ", i)

        loss_dict = criterion(
            outputs,
            (images, words_sim_distribution, indices, captions),
            model,
        )

        bs = images.size(0)
        num_samples += bs

        for k, v in loss_dict.items():
            running_losses[k] += v.item() * bs

        eval_step = global_step + i

        should_log_images = i == 0 or (log_every is not None and log_every > 0 and i % log_every == 0)

        print("!!! ", i, should_log_images)

        if should_log_images:
            b = random.randrange(bs)

            log_dict = {
                "global_step": eval_step,
                "eval/batch_idx": i,
                "eval/sample_idx": b,
            }

            if "mixture_weights" in outputs:
                topk_vals, topk_idx = outputs["mixture_weights"].topk(k=7, dim=-1)
                words = [model.vocab_words[j] for j in topk_idx[b].tolist()]
                log_dict["eval/top_words"] = ", ".join(words)

            for k, v in loss_dict.items():
                log_dict[f"eval/{k}"] = v.item()

            print("LOGGING IMAGE TO WANDB")
            print("outputs keys:", outputs.keys())
            print("patch logits shape:", outputs["patch_prototype_logits"].shape)
            print("image shape:", images[b : b + 1].shape)
            print("eval_step:", eval_step)

            sample_outputs = _slice_batch_outputs(
                outputs=outputs,
                batch_idx=b,
                batch_size=bs,
            )

            wandb_log_top_proto_heatmaps(
                model=model,
                images=images[b : b + 1],
                outputs=sample_outputs,
                step=eval_step,
                captions=[captions[b]],
                log_key="eval/top_proto_heatmaps",
                log_tsne=False,
            )

            wandb.log(log_dict, step=eval_step)

    avg_losses = {}

    for k, v in running_losses.items():
        avg_losses[k] = v / max(1, num_samples)
        logger.info(f"TEST {k}: {avg_losses[k]:.4f}")

    avg_losses["total_loss"] = sum(
        v for k, v in avg_losses.items() if not k.startswith("_")
    )

    wandb.log(
        {
            "global_step": global_step,
            **{f"test/{k}": v for k, v in avg_losses.items()},
        },
        step=global_step,
    )

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
    # for p in backbone.parameters():
    #    p.requires_grad = False

    return backbone, dim


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    parser = argparse.ArgumentParser()

    parser.add_argument("--log-dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument(
        "--dataset",
        type=str,
        default="coco_clip",
        choices=["coco_clip", "cub_clip"],
    )
    parser.add_argument(
        "--coco-root", type=str, default="/data/pwojcik/UnGuide/coco30_bck/"
    )
    parser.add_argument("--coco-val-ratio", type=float, default=0.1)
    parser.add_argument("--coco-clip-model-name", type=str, default="ViT-B-32")
    parser.add_argument("--coco-clip-pretrained", type=str, default="openai")
    parser.add_argument("--visual-coef", type=float, default=0.0)
    parser.add_argument("--cover-coef", type=float, default=0.0)

    parser.add_argument(
        "--backbone",
        type=str,
        default="dinov2_vitb14",
        choices=["dinov2_vitb14", "dinov2_vits14", "clip_vitb32", "dino_vitb16"],
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

    parser.add_argument(
        "--vocab-cache-path", type=str, default="vocab/mscoco_new_cache.pt"
    )
    parser.add_argument("--clip-text-dim", type=int, default=512)
    parser.add_argument("--kl-coef", type=float, default=1.0)
    parser.add_argument("--text-proj-hidden-dim", type=int, default=768)
    parser.add_argument("--prototype-init-noise", type=float, default=0.01)
    parser.add_argument("--eval-every-steps", type=int, default=1000)
    parser.add_argument("--temperature", type=float, default=0.2)

    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--bin-coef", type=float, default=0.1)
    parser.add_argument("--backbone-lr", type=float, default=1.0e-5)
    parser.add_argument("--text-proj-lr", type=float, default=1.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)

    parser.add_argument("--cosine-coef", type=float, default=1.0)
    parser.add_argument("--entropy-coef", type=float, default=0.0)
    parser.add_argument("--max-steps", type=int, default=1000)

    args = parser.parse_args()

    wandb.init(
        entity=os.environ.get("WANDB_ENTITY"),
        project=os.environ.get("WANDB_PROJECT", "proto-non-param"),
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

    logger.info("Train on COCO CLIP dataset")

    cache = torch.load(args.vocab_cache_path, map_location="cpu")
    vocab_words = list(cache.keys())
    vocab_to_idx = {w: i for i, w in enumerate(vocab_words)}

    print("Building datasets")

    from glob import glob

    shards = sorted(glob("/net/tscratch/people/plgpiotrwojcik/vg_coco/train/shard-*.tar"))

    nltk.download("punkt", quiet=True)
    nltk.download("averaged_perceptron_tagger", quiet=True)
    nltk.download("wordnet", quiet=True)
    nltk.download("omw-1.4", quiet=True)

    # -------------------------
    # Build datasets
    # -------------------------
    shards = sorted(glob("/net/tscratch/people/plgpiotrwojcik/vg_coco/train/shard-*.tar"))
    if len(shards) == 0:
        raise RuntimeError("No train shards found!")
    dataset_train = build_webdataset(
        shards=shards,
        vocab_to_idx=vocab_to_idx,
        train=True,
        shardshuffle=True,
        sample_shuffle=1000,
    ).with_epoch(args.eval_every_steps * args.batch_size)

    dataset_test = CocoCLIPDataset(
        annotations_json="/net/tscratch/people/plgpiotrwojcik/coco_2014/annotations/captions_val2014.json",
        coco_root="/net/tscratch/people/plgpiotrwojcik/coco_2014/val2014",
        vocab_to_idx=vocab_to_idx,
        train=False,
    )

    print("Done with datasets")
    print("Train shards:", len(shards))
    print("Test:", len(dataset_test))

    # -------------------------
    # Build dataloaders
    # -------------------------
    dataloader_train = DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=collate_batch,
        pin_memory=True,
        persistent_workers=(args.num_workers > 0),
        prefetch_factor=2 if args.num_workers > 0 else None,
    )

    dataloader_test = DataLoader(
        dataset_test,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=coco_clip_collate_fn,
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
        prototype_init_noise=args.prototype_init_noise,
        clip_model=clip_model,  # ← added
    )
    # freeze backbone first
    # for p in net.backbone.parameters():
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

        # n_blocks = len(blocks)
        # start = max(0, n_blocks - args.unfreeze_last_blocks)

        # for block in blocks[start:]:
        #    for p in block.parameters():
        #        p.requires_grad = True

        # print(f"Unfroze last {args.unfreeze_last_blocks} transformer blocks")

        for name, p in bb.named_parameters():
            if p.requires_grad:
                print("TRAINABLE BACKBONE:", name)

    criterion = PNPCriterion(
        kl_coef=args.kl_coef,
        entropy_coef=args.entropy_coef,
        visual_coef=args.visual_coef,
        bin_coef=args.bin_coef,
        cover_coef=args.cover_coef,
        temperature=args.temperature,
    )

    net.to(device)

    param_groups = [
        {"params": net.text_projection_head.parameters(), "lr": args.text_proj_lr},
    ]
    # add backbone as separate group
    backbone_params = [p for p in net.backbone.parameters() if p.requires_grad]
    if backbone_params:
        param_groups.append(
            {
                "params": backbone_params,
                "lr": args.backbone_lr,
            }
        )

    optimizer = optim.AdamW(param_groups, weight_decay=args.weight_decay)

    print_parameters(net=net, logger=logger)

    cache = torch.load(args.vocab_cache_path, map_location="cpu")
    vocab_words = list(cache.keys())
    vocab_to_idx = {w: i for i, w in enumerate(vocab_words)}
    noun_embeddings = torch.stack([cache[w] for w in vocab_words], dim=0)
    noun_embeddings = F.normalize(noun_embeddings, dim=-1).to(device)

    global_step = 0
    best_test_loss = float("inf")

    while global_step < args.max_steps:
        steps_this_epoch = min(args.eval_every_steps, args.max_steps - global_step)

        train(
            model=net,
            criterion=criterion,
            dataloader=dataloader_train,
            optimizer=optimizer,
            logger=logger,
            device=device,
            max_steps=steps_this_epoch,
            start_step=global_step,
            vocab_to_idx=vocab_to_idx,
        )

        global_step += steps_this_epoch

        test_metrics = test(
            model=net,
            criterion=criterion,
            dataloader=dataloader_test,
            logger=logger,
            device=device,
            global_step=global_step,
        )

        torch.save(
            {
                "state_dict": {
                    k: v.detach().cpu() for k, v in net.state_dict().items()
                },
                "hparams": vars(args),
            },
            log_dir / "ckpt.pth",
        )

        logger.info(f"Checkpoint saved at step {global_step}")

        if test_metrics["total_loss"] < best_test_loss:
            best_test_loss = test_metrics["total_loss"]
            torch.save(
                {
                    "state_dict": {
                        k: v.detach().cpu() for k, v in net.state_dict().items()
                    },
                    "hparams": vars(args),
                },
                log_dir / "best_ckpt.pth",
            )
            logger.info(f"Best checkpoint saved at step {global_step}")

if __name__ == "__main__":
    main()
