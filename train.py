#!/usr/bin/env python3
import sys
import logging
from collections import defaultdict
from logging import Logger
from pathlib import Path
import argparse

import wandb
import lightning as L
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn.functional as F

from clip_dataset import CocoCLIPDataset
from modeling.backbone import DINOv2Backbone, DINOv2BackboneExpanded, DINOBackboneExpanded
from modeling.pnp import CLIPPatch16Backbone, PNP, PNPCriterion
from modeling.utils import print_parameters


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
    running_cosine = 0.0

    for i, batch in enumerate(tqdm(dataloader)):
        images, target_txt, indices = batch
        images = images.to(device, non_blocking=True)
        target_txt = target_txt.to(device, non_blocking=True)

        outputs = model(images)
        loss_dict = criterion(outputs, (images, target_txt, indices))

        loss = sum(v for k, v in loss_dict.items() if not k.startswith("_"))
        if not isinstance(loss, torch.Tensor):
            raise ValueError("Loss is not a tensor")

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            pred_txt = F.normalize(outputs["pred_text_embedding"], dim=-1)
            tgt_txt = F.normalize(target_txt, dim=-1)
            batch_cosine = F.cosine_similarity(pred_txt, tgt_txt, dim=-1).mean().item()
            running_cosine += batch_cosine * images.size(0)

        log_dict = {}
        for k, v in loss_dict.items():
            running_losses[k] += v.item() * images.size(0)
            log_dict[f"train/{k}"] = v.item()

        global_step = epoch * len(dataloader) + i
        log_dict["train/total_loss"] = loss.item()
        log_dict["train/cosine_similarity"] = batch_cosine
        log_dict["epoch"] = epoch
        log_dict["global_step"] = global_step
        wandb.log(log_dict)

    for k, v in running_losses.items():
        loss_avg = v / len(dataloader.dataset)
        logger.info(f"EPOCH {epoch} train {k}: {loss_avg:.4f}")

    epoch_cosine = running_cosine / len(dataloader.dataset)
    logger.info(f"EPOCH {epoch} train cosine similarity: {epoch_cosine:.4f}")


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
    log_every: int = 20,
):
    model.eval()

    running_losses = defaultdict(float)
    running_cosine = 0.0

    for i, batch in enumerate(tqdm(dataloader)):
        images, target_txt, indices = batch
        images = images.to(device, non_blocking=True)
        target_txt = target_txt.to(device, non_blocking=True)

        outputs = model(images)
        loss_dict = criterion(outputs, (images, target_txt, indices))

        pred_txt = F.normalize(outputs["pred_text_embedding"], dim=-1)
        tgt_txt = F.normalize(target_txt, dim=-1)
        batch_cosine = F.cosine_similarity(pred_txt, tgt_txt, dim=-1).mean().item()
        running_cosine += batch_cosine * images.size(0)

        for k, v in loss_dict.items():
            running_losses[k] += v.item() * images.size(0)

        if i % log_every == 0:
            global_step = epoch * train_steps_per_epoch + i

            topk_vals, topk_idx = outputs["mixture_weights"].topk(k=5, dim=-1)
            preview_words = []
            for b in range(min(4, images.size(0))):
                words = [model.vocab_words[j] for j in topk_idx[b].tolist()]
                preview_words.append(", ".join(words))

            log_dict = {
                "eval/cosine_similarity": batch_cosine,
                "eval/batch_idx": i,
                "epoch": epoch,
                "global_step": global_step,
            }
            if preview_words:
                log_dict["eval/top_words_sample0"] = preview_words[0]
            wandb.log(log_dict)

    avg_losses = {}
    for k, v in running_losses.items():
        avg_losses[k] = v / len(dataloader.dataset)
        logger.info(f"EPOCH {epoch} test {k}: {avg_losses[k]:.4f}")

    epoch_cosine = running_cosine / len(dataloader.dataset)
    logger.info(f"EPOCH {epoch} test cosine similarity: {epoch_cosine:.4f}")

    wandb.log({
        "test/cosine_similarity": epoch_cosine,
        "epoch": epoch,
        "global_step": epoch * train_steps_per_epoch + (len(dataloader) - 1),
        **{f"test/{k}": v for k, v in avg_losses.items()},
    })

    return epoch_cosine


def build_backbone(args):
    if "dinov2" in args.backbone:
        if args.num_splits and args.num_splits > 0:
            backbone = DINOv2BackboneExpanded(
                name=args.backbone,
                n_splits=args.num_splits,
                mode="block_expansion",
                freeze_norm_layer=True,
            )
        else:
            backbone = DINOv2Backbone(name=args.backbone)
        dim = backbone.dim
    elif "clip" in args.backbone:
        backbone = CLIPPatch16Backbone(
            model_name=args.clip_model_name,
            pretrained=args.clip_pretrained,
            patch_size=args.clip_patch_size,
        )
        dim = backbone.dim
    elif "dino" in args.backbone:
        backbone = DINOBackboneExpanded(
            name=args.backbone,
            n_splits=args.num_splits,
            mode="block_expansion",
            freeze_norm_layer=True,
        )
        dim = backbone.dim
    else:
        raise NotImplementedError(f"Backbone {args.backbone} not implemented.")

    return backbone, dim


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    parser = argparse.ArgumentParser()

    parser.add_argument("--log-dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--dataset", type=str, default="coco_clip", choices=["coco_clip"])
    parser.add_argument("--coco-csv-path", type=str, default="assets/coco_30k.csv")
    parser.add_argument("--coco-root", type=str, default="/data/pwojcik/UnGuide/coco30_bck/")
    parser.add_argument("--coco-val-ratio", type=float, default=0.1)
    parser.add_argument("--coco-clip-model-name", type=str, default="ViT-B-32")
    parser.add_argument("--coco-clip-pretrained", type=str, default="openai")

    parser.add_argument(
        "--backbone",
        type=str,
        default="dinov2_vitb14",
        choices=["dinov2_vitb14", "dinov2_vits14", "clip_vit_l_14_patch16", "dino_vitb16"],
    )
    parser.add_argument("--clip-model-name", type=str, default="ViT-L-14")
    parser.add_argument("--clip-pretrained", type=str, default="openai")
    parser.add_argument("--clip-patch-size", type=int, default=16)
    parser.add_argument("--freeze-backbone", action="store_true", default=False)
    parser.add_argument("--num-splits", type=int, default=1)

    parser.add_argument("--vocab-cache-path", type=str, default="vocab/mscoco_nouns_clip_cache.pt")
    parser.add_argument("--clip-text-dim", type=int, default=512)
    parser.add_argument("--text-proj-hidden-dim", type=int, default=768)
    parser.add_argument("--prototype-init-noise", type=float, default=0.01)
    parser.add_argument("--temperature", type=float, default=0.2)

    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--backbone-lr", type=float, default=1.0e-5)
    parser.add_argument("--prototype-lr", type=float, default=1.0e-4)
    parser.add_argument("--text-proj-lr", type=float, default=1.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)

    parser.add_argument("--cosine-coef", type=float, default=1.0)
    parser.add_argument("--mse-coef", type=float, default=0.0)
    parser.add_argument("--entropy-coef", type=float, default=0.0)

    args = parser.parse_args()

    wandb.init(
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

    logger.info("Train on COCO CLIP dataset")

    dataset_train = CocoCLIPDataset(
        csv_path=args.coco_csv_path,
        coco_root=args.coco_root,
        split="train",
        val_ratio=args.coco_val_ratio,
        seed=args.seed,
        model_name=args.coco_clip_model_name,
        pretrained=args.coco_clip_pretrained,
    )

    dataset_test = CocoCLIPDataset(
        csv_path=args.coco_csv_path,
        coco_root=args.coco_root,
        split="val",
        val_ratio=args.coco_val_ratio,
        seed=args.seed,
        model_name=args.coco_clip_model_name,
        pretrained=args.coco_clip_pretrained,
    )

    dataloader_train = DataLoader(
        dataset=dataset_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        pin_memory=True,
    )

    dataloader_test = DataLoader(
        dataset=dataset_test,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=True,
    )

    backbone, dim = build_backbone(args)

    net = PNP(
        backbone=backbone,
        dim=dim,
        temperature=args.temperature,
        clip_text_dim=args.clip_text_dim,
        text_proj_hidden_dim=args.text_proj_hidden_dim,
        vocab_cache_path=args.vocab_cache_path,
        prototype_init_noise=args.prototype_init_noise,
    )
    criterion = PNPCriterion(
        cosine_coef=args.cosine_coef,
        mse_coef=args.mse_coef,
        entropy_coef=args.entropy_coef,
    )

    net.to(device)

    param_groups = [
        {"params": [net.prototypes], "lr": args.prototype_lr},
        {"params": net.prototype_embed.parameters(), "lr": args.prototype_lr},
        {"params": net.text_projection_head.parameters(), "lr": args.text_proj_lr},
    ]

    if not args.freeze_backbone:
        if hasattr(net.backbone, "set_requires_grad"):
            net.backbone.set_requires_grad()
        if hasattr(net.backbone, "learnable_parameters"):
            backbone_params = list(net.backbone.learnable_parameters())
        else:
            backbone_params = [p for p in net.backbone.parameters() if p.requires_grad]
        if backbone_params:
            param_groups.append({"params": backbone_params, "lr": args.backbone_lr})

    optimizer = optim.AdamW(param_groups, weight_decay=args.weight_decay)

    print_parameters(net=net, logger=logger)

    best_epoch = 0
    best_val_cosine = float("-inf")

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

        epoch_metric = test(
            model=net,
            criterion=criterion,
            dataloader=dataloader_test,
            epoch=epoch,
            logger=logger,
            device=device,
            train_steps_per_epoch=len(dataloader_train),
        )

        torch.save(
            {
                "state_dict": {k: v.detach().cpu() for k, v in net.state_dict().items()},
                "hparams": vars(args),
            },
            log_dir / "ckpt.pth",
        )
        logger.info("Model saved as ckpt.pth")

        if epoch_metric > best_val_cosine:
            best_val_cosine = epoch_metric
            best_epoch = epoch

    logger.info(f"DONE! Best epoch is epoch {best_epoch} with cosine similarity {best_val_cosine:.4f}.")


if __name__ == "__main__":
    main()