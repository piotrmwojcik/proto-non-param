#!/usr/bin/env python3
import sys
import logging
from collections import defaultdict
from logging import Logger
from pathlib import Path
import numpy as np
import math
import open_clip
from collections import defaultdict
import argparse
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

import wandb
import lightning as L
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn.functional as F

from clip_dataset import CocoCLIPDataset
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





@torch.no_grad()
def wandb_log_top_proto_heatmaps(
    *,
    model: nn.Module,
    images: torch.Tensor,
    outputs: dict,
    step: int,
    captions=None,
    max_items: int = 20,
    top_k: int = 5,
    mean=CLIP_MEAN,
    std=CLIP_STD,
    log_key: str = "test/top_proto_heatmaps",
    tsne_key: str = "test/proto_tsne",
    tsne_max_points: int = 300,
    log_tsne: bool = False,
):
    """
    Logs, for a few images:
      - raw image with original caption
      - top-k prototype heatmaps with prototype words
      - optional t-SNE of frozen vocab embeddings vs learned prototypes
    """
    patch_logits = outputs["patch_prototype_logits"]   # [B, N, V]
    mix_weights = outputs["mixture_weights"]           # [B, V]

    B, N, V = patch_logits.shape
    H = W = int(math.sqrt(N))
    _, _, Hi, Wi = images.shape

    top_vals, top_idx = mix_weights.topk(k=top_k, dim=-1)   # [B, K]

    panels = []
    B_log = min(B, max_items)

    for b in range(B_log):
        img_uint8 = denorm_to_uint8(images[b], mean=mean, std=std)
        top_words = [model.vocab_words[j] for j in top_idx[b].tolist()]

        raw_caption = ""
        if captions is not None:
            raw_caption = str(captions[b])

        panels.append(
            wandb.Image(
                img_uint8,
                caption=(
                    f"caption: {raw_caption}"
                    + (f" | top: {', '.join(top_words)}" if top_words else "")
                )
            )
        )

        for rank, proto_idx in enumerate(top_idx[b].tolist()):
            hm = patch_logits[b, :, proto_idx].view(1, 1, H, W)
            hm_up = F.interpolate(
                hm, size=(Hi, Wi), mode="bilinear", align_corners=False
            )[0, 0]
            overlay = overlay_heatmap(img_uint8, hm_up, alpha=0.45)

            word = model.vocab_words[proto_idx]
            score = float(top_vals[b, rank].item())
            heatmap_caption = f"top{rank+1} | {word} | weight={score:.3f}"
            if captions is not None:
                heatmap_caption = f"caption: {raw_caption} | " + heatmap_caption

            panels.append(
                wandb.Image(
                    overlay,
                    caption=heatmap_caption
                )
            )

    log_dict = {
        "global_step": step,
        log_key: panels,
    }

    if log_tsne:
        frozen = F.normalize(model.vocab_clip_embeddings, dim=-1).detach().cpu()
        learned = F.normalize(model.get_prototypes(), dim=-1).detach().cpu()

        d_frozen = frozen.shape[1]
        d_learned = learned.shape[1]

        if d_frozen != d_learned:
            common_dim = min(d_frozen, d_learned)
            frozen_for_tsne = frozen[:, :common_dim]
            learned_for_tsne = learned[:, :common_dim]
        else:
            frozen_for_tsne = frozen
            learned_for_tsne = learned

        n_total = frozen_for_tsne.shape[0]
        n_keep = min(tsne_max_points, n_total)

        perm = torch.randperm(n_total)[:n_keep]
        frozen_sel = frozen_for_tsne[perm].numpy()
        learned_sel = learned_for_tsne[perm].numpy()
        words_sel = [model.vocab_words[i] for i in perm.tolist()]

        X = np.concatenate([frozen_sel, learned_sel], axis=0)

        tsne = TSNE(
            n_components=2,
            perplexity=min(30, max(5, n_keep // 10)),
            init="pca",
            learning_rate="auto",
            random_state=42,
        )
        Z = tsne.fit_transform(X)

        Z_frozen = Z[:n_keep]
        Z_learned = Z[n_keep:]

        fig = plt.figure(figsize=(8, 8), dpi=150)
        plt.scatter(Z_frozen[:, 0], Z_frozen[:, 1], s=18, alpha=0.7, label="frozen_vocab")
        plt.scatter(Z_learned[:, 0], Z_learned[:, 1], s=18, alpha=0.7, label="learned_proto")

        if B_log > 0:
            active_idx = top_idx[0].tolist()
            active_words = [model.vocab_words[j] for j in active_idx]
            active_set = set(active_words)

            for i, w in enumerate(words_sel):
                if w in active_set:
                    plt.text(
                        Z_frozen[i, 0], Z_frozen[i, 1], f"F:{w}",
                        fontsize=7, alpha=0.8
                    )
                    plt.text(
                        Z_learned[i, 0], Z_learned[i, 1], f"L:{w}",
                        fontsize=7, alpha=0.8
                    )

        plt.legend()
        plt.title("t-SNE: frozen vocab embeddings vs learned prototypes")
        plt.tight_layout()

        log_dict[tsne_key] = wandb.Image(
            fig, caption="frozen vs learned prototype space"
        )
        plt.close(fig)

    wandb.log(log_dict)

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
):
    model.train()

    running_losses = defaultdict(float)

    for i, batch in enumerate(tqdm(dataloader)):
        images, captions, target_dist, indices = batch
        images = images.to(device, non_blocking=True)
        target_dist = target_dist.to(device, non_blocking=True)

        # avoid exact zeros for KL / log-based losses
        words_sim_distribution = target_dist.clamp_min(1e-8)

        # ---- DEBUG PRINT ----
        if i % 200 == 0:
            b = 0
            topk_vals, topk_idx = words_sim_distribution[b].topk(10)

            words = [model.vocab_words[j] for j in topk_idx.tolist()]
            weights = topk_vals.tolist()

            print("\nCaption:", captions[b])
            print("Top-10 words:")
            for w, s in zip(words, weights):
                print(f"  {w:15s} {s:.7f}")

        outputs = model(images)
        loss_dict = criterion(outputs, (images, words_sim_distribution, indices))

        loss = sum(v for k, v in loss_dict.items() if not k.startswith("_"))
        if not isinstance(loss, torch.Tensor):
            raise ValueError("Loss is not a tensor")

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        log_dict = {}
        for k, v in loss_dict.items():
            running_losses[k] += v.item() * images.size(0)
            log_dict[f"train/{k}"] = v.item()

        global_step = epoch * len(dataloader) + i
        log_dict["train/total_loss"] = loss.item()
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
    noun_embeddings: torch.Tensor,
    target_temperature: float = 0.07,
    *,
    train_steps_per_epoch: int,
    log_every: int = 5,
    vocab_to_idx=None,
):
    model.eval()

    running_losses = defaultdict(float)
    num_samples = 0

    for i, batch in enumerate(tqdm(dataloader)):
        images, captions, indices = batch
        images = images.to(device, non_blocking=True)

        global_step = epoch * train_steps_per_epoch + i

        # --------------------------
        # Caption → noun distribution
        # --------------------------
        img_feat = clip_model.encode_image(images)
        img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)

        B, D = img_feat.shape
        V = noun_embeddings.shape[0]

        images, captions, target_dist, indices = batch
        images = images.to(device, non_blocking=True)
        target_dist = target_dist.to(device, non_blocking=True)
        words_sim_distribution = target_dist.clamp_min(1e-8)

        # --------------------------
        # Model forward
        # --------------------------
        outputs = model(images)
        loss_dict = criterion(outputs, (images, words_sim_distribution, indices))

        bs = images.size(0)
        num_samples += bs

        for k, v in loss_dict.items():
            running_losses[k] += v.item() * bs

        # --------------------------
        # Logging
        # --------------------------
        if i % log_every == 0:

            log_dict = {
                "epoch": epoch,
                "global_step": global_step,
                "eval/batch_idx": i,
            }

            if "mixture_weights" in outputs:
                topk_vals, topk_idx = outputs["mixture_weights"].topk(k=7, dim=-1)

                words = [
                    model.vocab_words[j]
                    for j in topk_idx[0].tolist()
                ]

                log_dict["eval/top_words_sample0"] = ", ".join(words)

            for k, v in loss_dict.items():
                log_dict[f"eval/{k}"] = v.item()

            wandb.log(log_dict)

            # --------------------------
            # Visualization logging
            # --------------------------
            wandb_log_top_proto_heatmaps(
                model=model,
                images=images,
                outputs=outputs,
                step=global_step,
                captions=captions,
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
                mode="block_expansion",
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
    for p in backbone.parameters():
        p.requires_grad = False

    # ---------------------------------------------------
    # Unfreeze last N transformer blocks
    # ---------------------------------------------------

    return backbone, dim


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    parser = argparse.ArgumentParser()

    parser.add_argument("--log-dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--dataset", type=str, default="coco_clip", choices=["coco_clip"])
    parser.add_argument("--coco-root", type=str, default="/data/pwojcik/UnGuide/coco30_bck/")
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

    parser.add_argument("--vocab-cache-path", type=str, default="vocab/mscoco_new_cache.pt")
    parser.add_argument("--clip-text-dim", type=int, default=512)
    parser.add_argument("--kl-coef", type=float, default=1.0)
    parser.add_argument("--text-proj-hidden-dim", type=int, default=768)
    parser.add_argument("--prototype-init-noise", type=float, default=0.01)
    parser.add_argument("--temperature", type=float, default=0.2)

    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--bin-coef", type=float, default=0.1)
    parser.add_argument("--backbone-lr", type=float, default=1.0e-5)
    parser.add_argument("--text-proj-lr", type=float, default=1.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)

    parser.add_argument("--cosine-coef", type=float, default=1.0)
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

    cache = torch.load(args.vocab_cache_path, map_location="cpu")
    vocab_words = list(cache.keys())
    vocab_to_idx = {w: i for i, w in enumerate(vocab_words)}

    dataset_train = CocoCLIPDataset(
        annotations_json="/data/pwojcik/coco_2014/annotations/captions_train2014.json",
        coco_root="/data/pwojcik/coco_2014",
        vocab_to_idx=vocab_to_idx,
        model_name=args.coco_clip_model_name,
        pretrained=args.coco_clip_pretrained,
    )

    dataset_test = CocoCLIPDataset(
        annotations_json="/data/pwojcik/coco_2014/annotations/captions_val2014.json",
        coco_root="/data/pwojcik/coco_2014",
        vocab_to_idx=vocab_to_idx,
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
        param_groups.append({
            "params": backbone_params,
            "lr": args.backbone_lr,
        })


    optimizer = optim.AdamW(param_groups, weight_decay=args.weight_decay)

    print_parameters(net=net, logger=logger)

    best_epoch = 0
    best_val_cosine = float("-inf")

    cache = torch.load(args.vocab_cache_path, map_location="cpu")
    vocab_words = list(cache.keys())
    vocab_to_idx = {w: i for i, w in enumerate(vocab_words)}
    noun_embeddings = torch.stack([cache[w] for w in vocab_words], dim=0)
    noun_embeddings = F.normalize(noun_embeddings, dim=-1).to(device)

    for epoch in range(args.epochs):
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
            vocab_to_idx=vocab_to_idx
        )

        epoch_metrics = test(
            model=net,
            criterion=criterion,
            dataloader=dataloader_test,
            epoch=epoch,
            logger=logger,
            device=device,
            clip_model=clip_model,
            noun_embeddings=noun_embeddings,
            target_temperature=0.01,
            train_steps_per_epoch=len(dataloader_train),
            vocab_to_idx=vocab_to_idx,  # ADD THIS
        )

        epoch_metric = -sum(
            v for k, v in epoch_metrics.items()
            if k.startswith("test/") and not k.startswith("test/_")
        )
        torch.save(
            {
                "state_dict": {k: v.detach().cpu() for k, v in net.state_dict().items()},
                "hparams": vars(args),
            },
            log_dir / "ckpt.pth",
        )
        logger.info("Model saved as ckpt.pth")
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