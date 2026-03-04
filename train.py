#!/usr/bin/env python3
import sys
import logging
from collections import defaultdict
from logging import Logger
from pathlib import Path
import argparse
import wandb
import numpy as np
from modeling.pnp import CLIPPatch16Backbone
import torchvision
import matplotlib.pyplot as plt
import torch.nn.functional as F

import lightning as L
import torch
import torchvision.transforms as T
from torch import nn, optim
from torch.utils.data import DataLoader
from torchmetrics.classification import MulticlassAccuracy
from tqdm import tqdm

from data import CUBDataset, TinyImageNetDataset, train_transforms, test_transforms
from modeling.backbone import DINOv2Backbone, DINOv2BackboneExpanded, DINOBackboneExpanded
from modeling.pnp import PCA, PNP, PNPCriterion
from modeling.utils import print_parameters


def denorm_to_uint8(x: torch.Tensor,
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)) -> np.ndarray:
    """
    x: (3,H,W) normalized tensor
    returns: HxWx3 uint8
    """
    x = x.detach().cpu()
    mean_t = torch.tensor(mean)[:, None, None]
    std_t = torch.tensor(std)[:, None, None]
    x = (x * std_t + mean_t).clamp(0, 1)
    x = (x * 255).byte().permute(1, 2, 0).numpy()
    return x


def overlay_heatmap(img_uint8: np.ndarray, hm: torch.Tensor, alpha: float = 0.45) -> np.ndarray:
    """
    img_uint8: HxWx3 uint8
    hm: (H,W) tensor, any range
    returns: HxWx3 uint8
    """
    hm = hm.detach().cpu()
    hm = (hm - hm.min()) / (hm.max() - hm.min() + 1e-8)
    hm = hm.numpy()  # HxW in [0,1]

    # simple colormap (no cv2): red/yellow-ish
    r = hm
    g = np.clip(hm * 0.9 + 0.1, 0, 1)
    b = np.clip(1.0 - hm * 0.8, 0, 1)
    hm_rgb = (np.stack([r, g, b], axis=-1) * 255).astype(np.uint8)

    out = (alpha * hm_rgb.astype(np.float32) + (1 - alpha) * img_uint8.astype(np.float32))
    return out.clip(0, 255).astype(np.uint8)



def _to_uint8_img(img_chw: torch.Tensor, mean=None, std=None) -> np.ndarray:
    """
    img_chw: (3,H,W) torch tensor
    mean/std optional: if normalized, pass lists/tuples of length 3
    """
    x = img_chw.detach().float().cpu()
    if mean is not None and std is not None:
        mean = torch.tensor(mean)[:, None, None]
        std = torch.tensor(std)[:, None, None]
        x = x * std + mean
    x = x.clamp(0, 1)
    x = (x.permute(1, 2, 0).numpy() * 255).astype(np.uint8)  # HWC
    return x


@torch.no_grad()
def wandb_log_proto_and_fg_from_outputs(
    *,
    images: torch.Tensor,
    outputs: dict,
    step: int,
    mean=(0.48145466, 0.4578275, 0.40821073),
    std=(0.26862954, 0.26130258, 0.27577711),
    max_items: int = 4,
    log_key_heatmaps="eval/proto_heatmaps",
    log_key_fg="eval/pseudo_fg_overlay",
):

    logits = outputs["class_logits"]
    preds = logits.argmax(dim=1)

    ppl = outputs["patch_prototype_logits"]  # [B,N,C,K]
    B, N, C, K = ppl.shape
    H = W = int(np.sqrt(N))

    ppl_maps = ppl.view(B, H, W, C, K).permute(0, 3, 4, 1, 2)

    pred_maps = ppl_maps[torch.arange(B, device=images.device), preds]

    _, _, Hi, Wi = images.shape
    pred_maps_up = F.interpolate(pred_maps, size=(Hi, Wi), mode="bilinear", align_corners=False)

    pseudo_patch_labels = outputs["pseudo_patch_labels"]

    if (pseudo_patch_labels.shape[-2], pseudo_patch_labels.shape[-1]) != (Hi, Wi):
        pseudo_patch_labels = F.interpolate(
            pseudo_patch_labels.unsqueeze(1).float(),
            size=(Hi, Wi),
            mode="nearest"
        ).squeeze(1)

    bg_id = 196
    pseudo_mask = (pseudo_patch_labels != bg_id).float()

    B_log = min(B, max_items)

    heatmap_panels = []
    fg_panels = []

    for b in range(B_log):

        img_uint8 = denorm_to_uint8(images[b], mean=mean, std=std)
        pred_cls = int(preds[b])

        heatmap_panels.append(
            wandb.Image(img_uint8, caption=f"raw | pred={pred_cls}")
        )

        for k in range(K):
            hm = pred_maps_up[b, k]
            overlay = overlay_heatmap(img_uint8, hm, alpha=0.45)

            heatmap_panels.append(
                wandb.Image(overlay, caption=f"class={pred_cls} proto={k}")
            )

        mask = pseudo_mask[b].cpu().numpy()

        fig = plt.figure(figsize=(4,4), dpi=150)
        plt.imshow(img_uint8)
        plt.imshow(mask, alpha=0.45)
        plt.axis("off")
        plt.tight_layout(pad=0)

        fg_panels.append(
            wandb.Image(fig, caption=f"pseudo_fg | pred={pred_cls}")
        )

        plt.close(fig)

    wandb.log({
        "global_step": step,
        log_key_heatmaps: heatmap_panels,
        log_key_fg: fg_panels
    })

def train(model: nn.Module, criterion: nn.Module | None, dataloader: DataLoader, epoch: int,
          optimizer: optim.Optimizer | None, logger: Logger, device: torch.device):
    model.train()
    running_losses = defaultdict(float)
    mca_train = MulticlassAccuracy(num_classes=len(dataloader.dataset.classes), average="micro").to(device)

    for i, batch in enumerate(tqdm(dataloader)):
        batch = tuple(item.to(device) for item in batch)
        images, labels = batch[:2]

        outputs = model(images, labels=labels)

        if criterion is not None and optimizer is not None:
            loss_dict = criterion(outputs, batch)  # type: dict[str, torch.Tensor]
            loss = sum(val for key, val in loss_dict.items() if not key.startswith("_"))

            if not isinstance(loss, torch.Tensor):
                raise ValueError

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            log_dict = {}
            for k, v in loss_dict.items():
                running_losses[k] += v.item() * dataloader.batch_size
                log_dict[f"train/{k}"] = v.item()

            log_dict["train/total_loss"] = loss.item()
            log_dict["epoch"] = epoch
            #log_dict["step"] = epoch * len(dataloader) + i
            global_step = epoch * len(dataloader) + i
            log_dict["global_step"] = global_step
            wandb.log(log_dict)

        mca_train(outputs["class_logits"], labels)

    for k, v in running_losses.items():
        loss_avg = v / len(dataloader.dataset)
        logger.info(f"EPOCH {epoch} train {k}: {loss_avg:.4f}")

    epoch_acc_train = mca_train.compute().item()
    logger.info(f"EPOCH {epoch} train acc: {epoch_acc_train:.4f}")


@torch.inference_mode()
def test(
    model: nn.Module,
    dataloader: DataLoader,
    epoch: int,
    logger: Logger,
    device: torch.device,
    *,
    train_steps_per_epoch: int,
    log_every: int = 20,
):
    model.eval()
    mca_test = MulticlassAccuracy(
        num_classes=len(dataloader.dataset.classes),
        average="micro"
    ).to(device)

    for i, batch in enumerate(tqdm(dataloader)):
        batch = tuple(item.to(device) for item in batch)
        images, labels = batch[:2]

        outputs = model(images)
        mca_test(outputs["class_logits"], labels)

        if i % log_every == 0:
            global_step = epoch * train_steps_per_epoch + i

            with torch.no_grad():
                outputs = model(images)  # inference pass (labels=None)
                preds = outputs["class_logits"].argmax(dim=1)

            # diverse sample selection
            unique_classes = preds.unique()
            selected_indices = []
            for c in unique_classes:
                idxs = (preds == c).nonzero(as_tuple=True)[0]
                rand_idx = idxs[torch.randint(len(idxs), (1,))]
                selected_indices.append(rand_idx.item())

            # if not enough classes, fill with random samples
            if len(selected_indices) < 4:
                remaining = list(set(range(len(images))) - set(selected_indices))
                extra = torch.randperm(len(remaining))[: 4 - len(selected_indices)]
                selected_indices += [remaining[i] for i in extra]

            selected_indices = selected_indices[:4]
            sel = torch.tensor(selected_indices, device=images.device)

            # ---- NEW: compute outputs WITH pseudo labels for selected images only ----
            with torch.no_grad():
                outputs_sel = model(images[sel], labels=preds[sel], use_gumbel=False)
                # outputs_sel now contains "pseudo_patch_labels" (unless disable_clustering=True)

            wandb_log_proto_and_fg_from_outputs(
                images=images[sel],
                outputs=outputs_sel,
                step=global_step,
            )

            # also log a scalar so you can see activity immediately
            wandb.log({"eval/global_step": global_step, "eval/batch_idx": i, "epoch": epoch, "global_step": global_step})

    epoch_acc_test = mca_test.compute().item()
    logger.info(f"EPOCH {epoch} test acc: {epoch_acc_test:.4f}")

    wandb.log({
        "test/accuracy": epoch_acc_test,
        "epoch": epoch,
        "global_step": epoch * train_steps_per_epoch + (len(dataloader) - 1),
    })

    return epoch_acc_test

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    parser = argparse.ArgumentParser()

    parser.add_argument("--log-dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--data-root", type=str, default="./datasets")
    parser.add_argument("--dataset", type=str, default="CUB", choices=["CUB", "tiny_imagenet"])

    parser.add_argument(
        "--backbone",
        type=str,
        default="dinov2_vitb14",
        choices=["dinov2_vitb14", "dinov2_vits14", "clip_vit_l_14_patch16"]
    )
    parser.add_argument("--clip-model-name", type=str, default="ViT-L-14")
    parser.add_argument("--clip-pretrained", type=str, default="openai")
    parser.add_argument("--clip-patch-size", type=int, default=16)
    parser.add_argument("--clip-image-size", type=int, default=224)
    parser.add_argument("--freeze-backbone", action="store_true", default=True)
    parser.add_argument("--num-splits", type=int, default=1)

    # Model related hyperparameters
    parser.add_argument("--num-prototypes", type=int, default=5, help="Number of prototypes per class")
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--sa-initial-value", type=float, default=0.5)

    # Optimization hyperparameters
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--backbone-lr", type=float, default=1.0e-4)
    parser.add_argument("--weight-decay", type=float, default=0)
    parser.add_argument("--classifier-lr", type=float, default=1.0e-6)
    parser.add_argument("--fine-tuning-start-epoch", type=int, default=1)

    args = parser.parse_args()

    wandb.init(
        project="proto-non-param",
        config=vars(args),
        dir=args.log_dir
    )

    wandb.define_metric("global_step")
    wandb.define_metric("train/*", step_metric="global_step")
    wandb.define_metric("test/*", step_metric="global_step")
    wandb.define_metric("eval/*", step_metric="global_step")  # for heatmaps etc.

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

    normalize = T.Normalize(mean=(0.485, 0.456, 0.406,), std=(0.229, 0.224, 0.225,))
    transforms = T.Compose([
        T.Resize((224, 224,)),
        T.ToTensor(),
        normalize
    ])

    if args.dataset == "CUB":
        logger.info("Train on CUB-200-2011")
        n_classes = 200
        dataset_dir = Path(args.data_root) / "cub200_cropped"
        dataset_train = CUBDataset((dataset_dir / "train_cropped_augmented").as_posix(),
                                   transforms=transforms)
        dataloader_train = DataLoader(dataset=dataset_train, batch_size=128, num_workers=8, shuffle=True)

        dataset_test = CUBDataset((dataset_dir / "test_cropped").as_posix(),
                                   transforms=transforms)
        dataloader_test = DataLoader(dataset=dataset_test, batch_size=128, num_workers=8, shuffle=True)
    elif args.dataset == "tiny_imagenet":
        logger.info("Train on Tiny ImageNet")
        n_classes = 196
        #dataset_dir = Path(args.data_root) / "tiny-imagenet-200"  # adjust if your folder name differs
        dataset_dir = Path(args.data_root) / "dataset"


        dataset_train = TinyImageNetDataset(
            root=dataset_dir.as_posix(),
            split="train",
            transforms=train_transforms,
        )

        dataloader_train = DataLoader(
            dataset=dataset_train,
            batch_size=8,
            num_workers=8,
            shuffle=True
        )

        dataset_test = TinyImageNetDataset(
            root=dataset_dir.as_posix(),
            split="val",  # TinyImageNet uses val as the "test" split
            transforms=test_transforms,
        )

        dataloader_test = DataLoader(
            dataset=dataset_test,
            batch_size=32,
            num_workers=8,
            shuffle=False
        )
    else:
        raise NotImplementedError(f"Dataset {args.dataset} is not implemented")

    if "dinov2" in args.backbone:
        if args.num_splits and args.num_splits > 0:
            backbone = DINOv2BackboneExpanded(
                name=args.backbone,
                n_splits=args.num_splits,
                mode="block_expansion",
                freeze_norm_layer=True
            )
        else:
            backbone = DINOv2Backbone(name=args.backbone)
        dim = backbone.dim
    elif "clip" in args.backbone:
        # examples:
        #   --backbone clip_vit_l_14_patch16
        #   --backbone clip_vit_b_32_patch16
        #
        # You can parse model variant from args.backbone if you want; here’s a simple default:
        backbone = CLIPPatch16Backbone(
            model_name="ViT-L-14",
            pretrained="openai",
            patch_size=16,
        )
        dim = backbone.dim
    elif "dino" in args.backbone:
        backbone = DINOBackboneExpanded(
            name=args.backbone,
            n_splits=args.num_splits,
            mode="block_expansion",
            freeze_norm_layer=True
        )
        dim = backbone.dim
    else:
        raise NotImplementedError(f"Backbone {args.backbone} not implemented.")

    # Can be substituted with other off-the-shelf methods
    fg_extractor = PCA(bg_class=n_classes, compare_fn="le", threshold=0.5)

    net = PNP(
        backbone=backbone,
        dim=dim,
        fg_extractor=fg_extractor,
        n_prototypes=args.num_prototypes,
        n_classes=n_classes,
        gamma=args.gamma,
        temperature=args.temperature,
        sa_init=args.sa_initial_value,
        use_sinkhorn=True,
        norm_prototypes=False
    )
    criterion = PNPCriterion(l_ppd_coef=0.8, n_prototypes=args.num_prototypes, num_classes=n_classes)

    net.to(device)

    best_epoch, best_test_epoch = 0, 0.0

    for epoch in range(args.epochs):
        is_fine_tuning = epoch >= args.fine_tuning_start_epoch

        # Stage 2 training
        if is_fine_tuning:
            logger.info("Start fine-tuning backbone...")
            for name, param in net.named_parameters():
                param.requires_grad = ("backbone" not in name) and ("fg_extractor" not in name)

            net.backbone.set_requires_grad()

            param_groups = [{'params': net.backbone.learnable_parameters(),
                             'lr': args.backbone_lr}]
            param_groups += [{'params': net.classifier.parameters(), 'lr': args.classifier_lr}]

            optimizer = optim.Adam(param_groups)

            net.optimizing_prototypes = False
        # Stage 1 training
        else:
            for params in net.parameters():
                params.requires_grad = False
            optimizer = None
            net.optimizing_prototypes = True

        if epoch > 0:
            net.initializing = False

        print_parameters(net=net, logger=logger)
        logger.info(f"net.initializing: {net.initializing}")
        logger.info(f"net.optimizing_prototypes: {net.optimizing_prototypes}")

        train(
            model=net,
            criterion=criterion if is_fine_tuning else None,
            dataloader=dataloader_train,
            epoch=epoch,
            optimizer=optimizer if is_fine_tuning else None,
            logger=logger,
            device=device
        )

        epoch_acc_test = test(
            model=net,
            dataloader=dataloader_test,
            epoch=epoch,
            logger=logger,
            device=device,
            train_steps_per_epoch=len(dataloader_train),
        )

        torch.save(
            dict(
                state_dict={k: v.detach().cpu() for k, v in net.state_dict().items()},
                hparams=vars(args),
            ),
            log_dir / "ckpt.pth"
        )
        logger.info("Model saved as ckpt.pth")

        if epoch_acc_test > best_test_epoch:
            best_val_acc = epoch_acc_test
            best_epoch = epoch

    logger.info(f"DONE! Best epoch is epoch {best_epoch} with accuracy {best_val_acc}.")


if __name__ == '__main__':
    main()
