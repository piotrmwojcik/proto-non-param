#!/usr/bin/env python3
import os
import sys
import math
import random
import logging
import argparse
from pathlib import Path
from logging import Logger
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import wandb
import lightning as L
import open_clip

from test_cub_dataset_local import CUBTokenDataset, cub_token_collate_fn
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


def denorm_to_uint8(x: torch.Tensor, mean=CLIP_MEAN, std=CLIP_STD) -> np.ndarray:
    x = x.detach().cpu()
    mean_t = torch.tensor(mean)[:, None, None]
    std_t = torch.tensor(std)[:, None, None]
    x = (x * std_t + mean_t).clamp(0, 1)
    return (x * 255).byte().permute(1, 2, 0).numpy()


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

    return ys.min(), ys.max() + 1, xs.min(), xs.max() + 1


def draw_rect_on_image(img_uint8, bbox, color=(255, 0, 0), width=3):
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
    max_items: int = 8,
    top_k: int = 5,
    mean=CLIP_MEAN,
    std=CLIP_STD,
    log_key: str = "test/top_proto_heatmaps",
    crop_percentile: float = 95,
):
    if "vocab_logits" not in outputs:
        return

    # Prefer the concept maps used by the forward concept branch.
    if "patch_vocab_logits" in outputs:
        patch_logits_vocab = outputs["patch_vocab_logits"]  # [B, N, V]
    elif "patch_prototype_logits" in outputs:
        patch_logits = outputs["patch_prototype_logits"]

        if patch_logits.ndim == 4:
            patch_logits_vocab = patch_logits.max(dim=2).values  # [B, N, V]
        elif patch_logits.ndim == 3:
            patch_logits_vocab = patch_logits  # [B, N, V]
        else:
            return
    else:
        return

    concept_scores = torch.sigmoid(outputs["vocab_logits"])  # [B, V]

    B, N, V = patch_logits_vocab.shape
    H = W = int(math.sqrt(N))
    _, _, Hi, Wi = images.shape

    top_k = min(top_k, V)
    top_vals, top_idx = concept_scores.topk(k=top_k, dim=-1)

    sample_grids = []
    B_log = min(B, max_items)
    for b in range(B_log):
        img_uint8 = denorm_to_uint8(images[b], mean=mean, std=std)
        raw_caption = str(captions[b]) if captions is not None else ""

        panel_imgs = [img_uint8]
        panel_titles = ["raw"]

        words = []

        for rank, proto_idx in enumerate(top_idx[b].tolist()):
            word = model.vocab_words[proto_idx]
            words.append(word)

            hm = patch_logits_vocab[b, :, proto_idx].view(1, 1, H, W)
            hm_up = F.interpolate(
                hm,
                size=(Hi, Wi),
                mode="bilinear",
                align_corners=False,
            )[0, 0]

            hm_np = hm_up.detach().cpu().numpy()
            bbox = find_high_activation_crop(hm_np, percentile=crop_percentile)

            overlay = overlay_heatmap(img_uint8, hm_up, alpha=0.45)
            overlay_box = draw_rect_on_image(overlay, bbox, color=(255, 0, 0), width=3)

            score = float(top_vals[b, rank].item())
            panel_imgs.append(overlay_box)
            panel_titles.append(f"top{rank + 1}: {word}\n{score:.3f}")

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

        for ax in axes[len(panel_imgs):]:
            ax.axis("off")

        suptitle = raw_caption
        if words:
            suptitle += f"\nTop concepts: {', '.join(words)}"

        fig.suptitle(suptitle, fontsize=11)
        plt.tight_layout(rect=[0, 0, 1, 0.92])

        sample_grids.append(wandb.Image(fig, caption=f"sample={b}"))
        plt.close(fig)

    wandb.log({"global_step": step, log_key: sample_grids})


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

    return backbone, dim


def train(
    model: nn.Module,
    criterion: nn.Module,
    dataloader: DataLoader,
    epoch: int,
    optimizer: optim.Optimizer,
    logger: Logger,
    device: torch.device,
):
    model.train()

    running_losses = defaultdict(float)
    num_samples = 0
    num_correct = 0

    for i, batch in enumerate(tqdm(dataloader, desc=f"train epoch {epoch}")):
        images = batch["images"].to(device, non_blocking=True)
        tokens = batch["tokens"]
        class_idxs = batch["class_idxs"].to(device, non_blocking=True)
        image_ids = batch["image_ids"].to(device, non_blocking=True)
        captions = batch["captions"]

        if i % 200 == 0:
            b = 0
            print("\nCaption:")
            print(" ", captions[b])
            print("Target concept tokens:")
            for token_id in tokens[b].tolist():
                print(f"  {token_id:4d}  {model.vocab_words[token_id]}")

        outputs = model(images)

        loss_dict = criterion(
            outputs,
            batch={
                "images": images,
                "tokens": tokens,
                "class_idxs": class_idxs,
                "image_ids": image_ids,
                "captions": captions,
            },
            model=model,
        )

        loss = sum(v for k, v in loss_dict.items() if not k.startswith("_"))

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        bs = images.size(0)
        num_samples += bs

        log_dict = {
            "epoch": epoch,
            "global_step": epoch * len(dataloader) + i,
            "train/total_loss": loss.item(),
        }

        for k, v in loss_dict.items():
            running_losses[k] += v.item() * bs
            log_dict[f"train/{k}"] = v.item()

        if "class_logits" in outputs:
            pred_class = outputs["class_logits"].argmax(dim=-1)
            num_correct += (pred_class == class_idxs).sum().item()
            log_dict["train/acc_batch"] = (pred_class == class_idxs).float().mean().item()

        if "vocab_logits" in outputs:
            concept_scores = torch.sigmoid(outputs["vocab_logits"])
            topk_vals, topk_idx = concept_scores.topk(k=min(7, concept_scores.shape[-1]), dim=-1)
            b = 0
            words = [model.vocab_words[j] for j in topk_idx[b].tolist()]
            log_dict["train/top_concepts"] = ", ".join(words)

        wandb.log(log_dict)

    avg_losses = {}

    for k, v in running_losses.items():
        avg_losses[k] = v / max(1, num_samples)
        logger.info(f"EPOCH {epoch} train {k}: {avg_losses[k]:.4f}")

    avg_losses["total_loss"] = sum(
        v for k, v in avg_losses.items() if not k.startswith("_")
    )
    avg_losses["acc"] = num_correct / max(1, num_samples)

    logger.info(f"EPOCH {epoch} train acc: {avg_losses['acc']:.4f}")

    return avg_losses


@torch.inference_mode()
def test(
    model: nn.Module,
    criterion: nn.Module,
    dataloader: DataLoader,
    epoch: int,
    logger: Logger,
    device: torch.device,
    *,
    train_steps_per_epoch: int,
    log_every: int = 50,
):
    model.eval()

    running_losses = defaultdict(float)
    num_samples = 0
    num_correct = 0

    if len(dataloader) > 0:
        log_batches = set(
            random.sample(
                range(len(dataloader)),
                k=max(1, len(dataloader) // log_every),
            )
        )
    else:
        log_batches = set()

    for i, batch in enumerate(tqdm(dataloader, desc=f"test epoch {epoch}")):
        images = batch["images"].to(device, non_blocking=True)
        tokens = batch["tokens"]
        class_idxs = batch["class_idxs"].to(device, non_blocking=True)
        image_ids = batch["image_ids"].to(device, non_blocking=True)
        captions = batch["captions"]

        global_step = epoch * train_steps_per_epoch + i

        outputs = model(images)

        loss_dict = criterion(
            outputs,
            batch={
                "images": images,
                "tokens": tokens,
                "class_idxs": class_idxs,
                "image_ids": image_ids,
                "captions": captions,
            },
            model=model,
        )

        total_loss = sum(v for k, v in loss_dict.items() if not k.startswith("_"))

        bs = images.size(0)
        num_samples += bs

        for k, v in loss_dict.items():
            running_losses[k] += v.item() * bs

        running_losses["total_loss"] += total_loss.item() * bs

        pred_class = None
        if "class_logits" in outputs:
            pred_class = outputs["class_logits"].argmax(dim=-1)
            num_correct += (pred_class == class_idxs).sum().item()

        if i in log_batches:
            b = random.randrange(images.size(0))

            log_dict = {
                "epoch": epoch,
                "global_step": global_step,
                "test/batch_idx": i,
                "test/sample_idx": b,
                "test/class_idx": class_idxs[b].item(),
                "test/image_id": image_ids[b].item(),
                "test/caption": captions[b],
                "test/total_loss": total_loss.item(),
            }

            if "vocab_logits" in outputs:
                concept_scores = torch.sigmoid(outputs["vocab_logits"])
                topk_vals, topk_idx = concept_scores.topk(k=min(7, concept_scores.shape[-1]), dim=-1)
                words = [model.vocab_words[j] for j in topk_idx[b].tolist()]
                log_dict["test/top_concepts"] = ", ".join(words)

            if pred_class is not None:
                log_dict["test/pred_class"] = pred_class[b].item()
                log_dict["test/correct"] = int(pred_class[b] == class_idxs[b])

            for loss_name, loss_val in loss_dict.items():
                log_dict[f"test/{loss_name}"] = loss_val.item()

            wandb.log(log_dict)

            wandb_log_top_proto_heatmaps(
                model=model,
                images=images[b:b + 1],
                outputs={
                    name: (
                        value[b:b + 1]
                        if torch.is_tensor(value)
                        and value.ndim > 0
                        and value.shape[0] == images.shape[0]
                        else value
                    )
                    for name, value in outputs.items()
                },
                step=global_step,
                captions=[captions[b]],
            )

    avg_losses = {}

    for k, v in running_losses.items():
        avg_losses[k] = v / max(1, num_samples)
        logger.info(f"EPOCH {epoch} test {k}: {avg_losses[k]:.4f}")

    avg_losses["acc"] = num_correct / max(1, num_samples)
    logger.info(f"EPOCH {epoch} test acc: {avg_losses['acc']:.4f}")

    wandb.log(
        {
            "epoch": epoch,
            "global_step": epoch * train_steps_per_epoch + len(dataloader) - 1,
            **{f"test/{k}": v for k, v in avg_losses.items()},
        }
    )

    return avg_losses


def load_clip_model(args, device):
    clip_model, _, _ = open_clip.create_model_and_transforms(
        args.coco_clip_model_name,
        pretrained=None,
    )

    ckpt = torch.load(args.coco_clip_pretrained, map_location="cpu")
    state_dict = ckpt.get("state_dict", ckpt)

    cleaned_state_dict = {}
    for k, v in state_dict.items():
        new_key = k
        for prefix in ("module.", "model.", "_orig_mod."):
            if new_key.startswith(prefix):
                new_key = new_key[len(prefix):]
        cleaned_state_dict[new_key] = v

    missing, unexpected = clip_model.load_state_dict(cleaned_state_dict, strict=False)

    print("Loaded CLIP checkpoint:", args.coco_clip_pretrained)
    print("Missing keys:", len(missing))
    print("Unexpected keys:", len(unexpected))
    if missing:
        print("First missing keys:", missing[:20])
    if unexpected:
        print("First unexpected keys:", unexpected[:20])

    clip_model = clip_model.eval().to(device)

    for p in clip_model.parameters():
        p.requires_grad = False

    return clip_model


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--log-dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--train-csv-path", type=str, default="/net/tscratch/people/plgpiotrwojcik/cub_captions_simple_train.csv")
    parser.add_argument("--test-csv-path", type=str, default="/net/tscratch/people/plgpiotrwojcik/cub_captions_simple_test.csv")

    parser.add_argument("--backbone", type=str, default="dinov2_vitb14",
                        choices=["dinov2_vitb14", "dinov2_vits14", "clip_vitb32", "dino_vitb16"])
    parser.add_argument("--num-splits", type=int, default=1)
    parser.add_argument("--unfreeze-last-blocks", type=int, default=0)

    parser.add_argument("--coco-clip-model-name", type=str, default="ViT-B-32")
    parser.add_argument("--coco-clip-pretrained", type=str, default="openai")

    parser.add_argument("--vocab-cache-path", type=str, default="vocab/mscoco_new_cache.pt")
    parser.add_argument("--clip-text-dim", type=int, default=512)
    parser.add_argument("--text-proj-hidden-dim", type=int, default=768)
    parser.add_argument("--prototype-init-noise", type=float, default=0.01)

    parser.add_argument(
    "--dataset",
    type=str,
    default="cub_clip",
    choices=["cub_clip", "coco_clip"],
)

    parser.add_argument("--visual-coef", type=float, default=0.0)

    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--cls-coef", type=float, default=0.0)
    parser.add_argument("--concept-coef", type=float, default=0.2)

    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--backbone-lr", type=float, default=1.0e-5)
    parser.add_argument("--text-proj-lr", type=float, default=1.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    wandb.init(
        entity=os.environ.get("WANDB_ENTITY", "piotrmwojcik"),
        project=os.environ.get("WANDB_PROJECT", "proto-non-param"),
        config=vars(args),
        dir=args.log_dir,
    )

    wandb.define_metric("global_step")
    wandb.define_metric("train/*", step_metric="global_step")
    wandb.define_metric("test/*", step_metric="global_step")

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

    cache = torch.load(args.vocab_cache_path, map_location="cpu")
    vocab_words = list(cache.keys())
    vocab_to_idx = {w.lower(): i for i, w in enumerate(vocab_words)}

    print(f"Loaded vocab size: {len(vocab_words)}")
    print("Building datasets")

    dataset_train = CUBTokenDataset(
        csv_path=args.train_csv_path,
        vocab_to_idx=vocab_to_idx,
        train=True,
    )

    dataset_test = CUBTokenDataset(
        csv_path=args.test_csv_path,
        vocab_to_idx=vocab_to_idx,
        train=False,
    )

    print("Done with datasets")
    print("Train:", len(dataset_train))
    print("Test:", len(dataset_test))

    dataloader_train = DataLoader(
        dataset=dataset_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=cub_token_collate_fn,
        shuffle=True,
        pin_memory=True,
    )

    dataloader_test = DataLoader(
        dataset=dataset_test,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=cub_token_collate_fn,
        shuffle=False,
        pin_memory=True,
    )

    backbone, dim = build_backbone(args)
    clip_model = load_clip_model(args, device)

    net = PNP(
        backbone=backbone,
        dim=dim,
        temperature=args.temperature,
        clip_text_dim=args.clip_text_dim,
        text_proj_hidden_dim=args.text_proj_hidden_dim,
        vocab_cache_path=args.vocab_cache_path,
        prototype_init_noise=args.prototype_init_noise,
        clip_model=clip_model,
    )

    net.to(device)

    criterion = PNPCriterion(
        l_ppd_coef=0.8,
        cls_coef=args.cls_coef,
        concept_coef=args.concept_coef,
        concept_temperature=args.temperature,
    )

    param_groups = [
        {
            "params": [
                p for name, p in net.named_parameters()
                if p.requires_grad and not name.startswith("backbone.")
            ],
            "lr": args.text_proj_lr,
        }
    ]

    backbone_params = [
        p for name, p in net.named_parameters()
        if p.requires_grad and name.startswith("backbone.")
    ]

    if backbone_params:
        param_groups.append(
            {
                "params": backbone_params,
                "lr": args.backbone_lr,
            }
        )

    optimizer = optim.AdamW(param_groups, weight_decay=args.weight_decay)

    print_parameters(net=net, logger=logger)

    best_epoch = 0
    best_acc = float("-inf")

    for epoch in range(args.epochs):
        train(
            model=net,
            criterion=criterion,
            dataloader=dataloader_train,
            epoch=epoch,
            optimizer=optimizer,
            logger=logger,
            device=device,
        )

        epoch_metrics = test(
            model=net,
            criterion=criterion,
            dataloader=dataloader_test,
            epoch=epoch,
            logger=logger,
            device=device,
            train_steps_per_epoch=len(dataloader_train),
        )

        epoch_acc = epoch_metrics.get("acc", 0.0)

        ckpt = {
            "state_dict": {
                k: v.detach().cpu() for k, v in net.state_dict().items()
            },
            "hparams": vars(args),
            "epoch": epoch,
            "acc": epoch_acc,
        }

        torch.save(ckpt, log_dir / "ckpt.pth")
        logger.info("Model saved as ckpt.pth")

        if epoch_acc > best_acc:
            best_acc = epoch_acc
            best_epoch = epoch
            torch.save(ckpt, log_dir / "best_ckpt.pth")
            logger.info("Model saved as best_ckpt.pth")

    logger.info(f"DONE! Best epoch: {best_epoch}, best acc: {best_acc:.4f}")


if __name__ == "__main__":
    main()