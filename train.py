from __future__ import annotations

from contextlib import nullcontext
from functools import partial
from pathlib import Path
from typing import Any
import argparse
import importlib
import os
import random

import torch
import torch.nn as nn
import wandb
import torch.nn.functional as F
from llm2vec import LLM2Vec
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from modeling.pnp import PNP, PNPContrastiveCriterion
from visual_genome_scene_graph_dataset import (
    DEFAULT_VG_ROOT,
    VisualGenomeSceneGraphDataset,
    scene_graph_collate_fn,
)
from llm2vec_utils import (
    encode_pair_strings,
    load_llm2vec,
)


def get_images(batch: Any) -> torch.Tensor:
    """
    Extract the image tensor from the Visual Genome collated batch.

    Adjust the tuple index here if scene_graph_collate_fn returns a tuple
    with images in a different position.
    """
    if isinstance(batch, dict):
        for key in ("images", "image", "pixel_values"):
            value = batch.get(key)

            if torch.is_tensor(value):
                return value

        raise KeyError(
            "Could not find images in the batch dictionary. "
            "Expected one of: images, image, pixel_values."
        )

    if isinstance(batch, (tuple, list)):
        for value in batch:
            if (
                torch.is_tensor(value)
                and value.ndim == 4
                and value.shape[1] in (1, 3, 4)
            ):
                return value

        raise ValueError(
            "Could not find a [B, C, H, W] image tensor in the batch."
        )

    raise TypeError(
        f"Unsupported batch type: {type(batch)!r}"
    )


def prepare_embeddings(
    embeddings: torch.Tensor,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """
    Move frozen LLM2Vec embeddings to the PNP device.

    Embeddings remain detached because LLM2Vec is used only as the
    fixed text-embedding generator.
    """
    if not torch.is_tensor(embeddings):
        embeddings = torch.as_tensor(embeddings)

    return embeddings.detach().to(
        device=device,
        dtype=dtype,
        non_blocking=True,
    )


def flatten_negative_embeddings(
    embeddings: torch.Tensor,
    *,
    batch_size: int,
) -> tuple[torch.Tensor, int]:
    """
    Convert negative embeddings to [B * K, D].

    Supports:

        [B, K, D]
        [B * K, D]
        [B, D]
    """
    if embeddings.ndim == 3:
        if embeddings.shape[0] != batch_size:
            raise ValueError(
                "Negative embedding batch dimension does not match "
                f"the image batch: {embeddings.shape[0]} versus "
                f"{batch_size}"
            )

        negatives_per_positive = embeddings.shape[1]
        embeddings = embeddings.reshape(
            batch_size * negatives_per_positive,
            embeddings.shape[-1],
        )

        return embeddings, negatives_per_positive

    if embeddings.ndim == 2:
        if embeddings.shape[0] % batch_size != 0:
            raise ValueError(
                "The number of negative embeddings must be divisible "
                f"by the image batch size. Received "
                f"{embeddings.shape[0]} embeddings for batch size "
                f"{batch_size}."
            )

        negatives_per_positive = (
            embeddings.shape[0] // batch_size
        )

        return embeddings, negatives_per_positive

    raise ValueError(
        "Negative embeddings must have shape [B, K, D] or "
        f"[B * K, D], received {tuple(embeddings.shape)}"
    )


def repeat_images_for_negatives(
    images: torch.Tensor,
    negatives_per_positive: int,
) -> torch.Tensor:
    """
    Match [B * K, D] negative embeddings with repeated images.

    Ordering is:

        image 0 negative 0
        image 0 negative 1
        ...
        image 1 negative 0
        image 1 negative 1
        ...
    """
    return images.repeat_interleave(
        negatives_per_positive,
        dim=0,
    )


def validate_similarity_map(
    similarity_map: torch.Tensor,
    *,
    batch_size: int,
    name: str,
) -> None:
    if not torch.is_tensor(similarity_map):
        raise TypeError(
            f"{name} must be a torch.Tensor, "
            f"but got {type(similarity_map)!r}"
        )

    if similarity_map.ndim != 4:
        raise ValueError(
            f"{name} must be a 4D similarity map with shape "
            "[B, 1, H, W], "
            f"but received shape {tuple(similarity_map.shape)}"
        )

    if similarity_map.shape[0] != batch_size:
        raise ValueError(
            f"{name} batch size mismatch: expected {batch_size}, "
            f"got {similarity_map.shape[0]}"
        )

    if similarity_map.shape[1] != 1:
        raise ValueError(
            f"{name} channel dimension must be 1, "
            f"but received {similarity_map.shape[1]}"
        )

    if similarity_map.shape[2] < 1 or similarity_map.shape[3] < 1:
        raise ValueError(
            f"{name} spatial dimensions must be positive, "
            f"but received {tuple(similarity_map.shape[2:])}"
        )


def train(
    *,
    model: PNP,
    llm2vec: LLM2Vec,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: PNPContrastiveCriterion,
    device: torch.device,
    epochs: int,
    encode_batch_size: int = 32,
    gradient_clip_norm: float | None = 1.0,
    use_amp: bool = True,
    checkpoint_dir: str | Path | None = None,
    visualize_every_steps: int = 0,
    visualize_samples: int = 1,
    visualize_images_per_batch: int = 4,
) -> None:
    model.train()

    amp_enabled = (
        use_amp
        and device.type == "cuda"
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=amp_enabled,
    )

    checkpoint_path = (
        Path(checkpoint_dir)
        if checkpoint_dir is not None
        else None
    )

    if checkpoint_path is not None:
        checkpoint_path.mkdir(
            parents=True,
            exist_ok=True,
        )

    global_step = 0

    for epoch in range(epochs):
        progress = tqdm(
            dataloader,
            desc=f"Epoch {epoch + 1}/{epochs}",
        )

        running_loss = 0.0

        for batch_index, batch in enumerate(progress):
            images = get_images(batch).to(
                device=device,
                non_blocking=True,
            )

            batch_size = images.shape[0]

            # LLM2Vec is a frozen embedding generator. This operation is
            # outside autocast because the encoder may manage its own dtype.
            with torch.no_grad():
                pair_embeddings = encode_pair_strings(
                    llm2vec=llm2vec,
                    batch=batch,
                    encode_batch_size=encode_batch_size,
                )

            positive_anchor_embeddings = prepare_embeddings(
                pair_embeddings["positive_anchor_embeddings"],
                device=device,
                dtype=images.dtype,
            )
            positive_text_embeddings = prepare_embeddings(
                pair_embeddings["positive_text_embeddings"],
                device=device,
                dtype=images.dtype,
            )
            negative_anchor_embeddings = prepare_embeddings(
                pair_embeddings["negative_anchor_embeddings"],
                device=device,
                dtype=images.dtype,
            )
            negative_text_embeddings = prepare_embeddings(
                pair_embeddings["negative_text_embeddings"],
                device=device,
                dtype=images.dtype,
            )

            if positive_anchor_embeddings.ndim != 2:
                raise ValueError(
                    "positive_anchor_embeddings must have shape "
                    f"[B, D], received "
                    f"{tuple(positive_anchor_embeddings.shape)}"
                )

            if positive_text_embeddings.ndim != 2:
                raise ValueError(
                    "positive_text_embeddings must have shape "
                    f"[B, D], received "
                    f"{tuple(positive_text_embeddings.shape)}"
                )

            if positive_anchor_embeddings.shape[0] != batch_size:
                raise ValueError(
                    "Positive anchor embedding count does not match "
                    f"the image batch: "
                    f"{positive_anchor_embeddings.shape[0]} versus "
                    f"{batch_size}"
                )

            if positive_text_embeddings.shape[0] != batch_size:
                raise ValueError(
                    "Positive text embedding count does not match "
                    f"the image batch: "
                    f"{positive_text_embeddings.shape[0]} versus "
                    f"{batch_size}"
                )

            negative_anchor_embeddings, anchor_negative_count = (
                flatten_negative_embeddings(
                    negative_anchor_embeddings,
                    batch_size=batch_size,
                )
            )

            negative_text_embeddings, text_negative_count = (
                flatten_negative_embeddings(
                    negative_text_embeddings,
                    batch_size=batch_size,
                )
            )

            if anchor_negative_count != text_negative_count:
                raise ValueError(
                    "Negative anchor and text counts differ: "
                    f"{anchor_negative_count} versus "
                    f"{text_negative_count}"
                )

            negatives_per_positive = anchor_negative_count

            negative_images = repeat_images_for_negatives(
                images,
                negatives_per_positive,
            )

            optimizer.zero_grad(set_to_none=True)

            autocast_context = (
                torch.autocast(
                    device_type="cuda",
                    dtype=torch.float16,
                )
                if amp_enabled
                else nullcontext()
            )

            with autocast_context:
                # Positive maps: [B, 1, H, W]
                positive_anchor_maps = model(
                    images,
                    positive_anchor_embeddings,
                )
                positive_text_maps = model(
                    images,
                    positive_text_embeddings,
                )

                # Negative maps: [B * K, 1, H, W]
                negative_anchor_maps = model(
                    negative_images,
                    negative_anchor_embeddings,
                )
                negative_text_maps = model(
                    negative_images,
                    negative_text_embeddings,
                )

                loss_dict = criterion(
                    positive_anchor_maps=positive_anchor_maps,
                    positive_text_maps=positive_text_maps,
                    negative_anchor_maps=negative_anchor_maps,
                    negative_text_maps=negative_text_maps,
                    batch_size=batch_size,
                    negatives_per_positive=negatives_per_positive,
                )

                total_loss = sum(
                    value
                    for name, value in loss_dict.items()
                    if name.startswith("l_")
                )

            if not torch.isfinite(total_loss):
                raise FloatingPointError(
                    "Non-finite loss encountered at "
                    f"epoch={epoch}, batch={batch_index}: "
                    f"{total_loss.detach().item()}"
                )

            scaler.scale(total_loss).backward()

            if gradient_clip_norm is not None:
                scaler.unscale_(optimizer)

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=gradient_clip_norm,
                )

            scaler.step(optimizer)
            scaler.update()

            global_step += 1
            running_loss += total_loss.detach().item()

            average_loss = running_loss / (
                batch_index + 1
            )

            progress.set_postfix(
                loss=f"{total_loss.detach().item():.4f}",
                average=f"{average_loss:.4f}",
                positive=(
                    f"{loss_dict['positive_similarity'].item():.3f}"
                ),
                negative=(
                    f"{loss_dict['hardest_negative_similarity'].item():.3f}"
                ),
            )
            wandb_metrics = {
                "train/loss": total_loss.detach().item(),
                "train/loss_avg": average_loss,
                "train/epoch": epoch + 1,
            }
            for name, value in loss_dict.items():
                if torch.is_tensor(value) and value.numel() == 1:
                    wandb_metrics[f"train/{name}"] = value.detach().item()

            if wandb.run is not None:
                wandb.log(wandb_metrics, step=global_step)

            if (
                visualize_every_steps > 0
                and global_step % visualize_every_steps == 0
            ):
                visualize_heatmaps(
                    model=model,
                    llm2vec=llm2vec,
                    dataloader=dataloader,
                    device=device,
                    encode_batch_size=encode_batch_size,
                    num_samples=visualize_samples,
                    images_per_batch=visualize_images_per_batch,
                    global_step=global_step,
                )

        if checkpoint_path is not None:
            torch.save(
                {
                    "epoch": epoch + 1,
                    "global_step": global_step,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scaler": scaler.state_dict(),
                },
                checkpoint_path
                / f"pnp_epoch_{epoch + 1:03d}.pt",
            )

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train dictionary-free PNP on Visual Genome."
    )
    parser.add_argument("--backbone", default="dinov2_vitb14")
    parser.add_argument(
        "--backbone-module",
        default="modeling.backbones",
        help="Python module containing the backbone classes.",
    )
    parser.add_argument("--num-splits", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--negatives-per-positive", type=int, default=4)
    parser.add_argument("--encode-batch-size", type=int, default=32)
    parser.add_argument("--text-dim", type=int, default=4096)
    parser.add_argument("--log-dir", default="wandb")
    parser.add_argument(
        "--checkpoint-dir",
        default="checkpoints/pnp_visual_genome",
    )
    parser.add_argument("--visualize-samples", type=int, default=1)
    parser.add_argument(
        "--visualize-every-steps",
        type=int,
        default=100,
        help="Log heatmaps every N optimiser steps; set to 0 to disable.",
    )
    parser.add_argument(
        "--visualize-images-per-batch",
        type=int,
        default=4,
    )
    return parser.parse_args()


def build_backbone(args: argparse.Namespace):
    try:
        module = importlib.import_module(args.backbone_module)
    except ImportError as exc:
        raise ImportError(
            f"Could not import backbone module {args.backbone_module!r}. "
            "Pass --backbone-module with the module used by your project."
        ) from exc

    if "dinov2" in args.backbone:
        if args.num_splits > 0:
            cls_name = "DINOv2BackboneExpanded"
            kwargs = {
                "name": args.backbone,
                "n_splits": args.num_splits,
                "mode": "append",
                "freeze_norm_layer": True,
            }
        else:
            cls_name = "DINOv2Backbone"
            kwargs = {"name": args.backbone}
    elif "dino" in args.backbone:
        cls_name = "DINOBackboneExpanded"
        kwargs = {
            "name": args.backbone,
            "n_splits": args.num_splits,
            "mode": "block_expansion",
            "freeze_norm_layer": True,
        }
    elif "clip" in args.backbone:
        cls_name = "CLIPBackbone"
        kwargs = {"name": args.backbone}
    else:
        raise NotImplementedError(
            f"Backbone {args.backbone!r} is not supported."
        )

    if not hasattr(module, cls_name):
        raise ImportError(
            f"{cls_name} was not found in {args.backbone_module!r}."
        )

    backbone = getattr(module, cls_name)(**kwargs)
    if not hasattr(backbone, "dim"):
        raise AttributeError(
            f"{cls_name} must expose a 'dim' attribute."
        )
    return backbone, backbone.dim

def main() -> None:
    args = parse_args()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )


    wandb.init(
        entity=os.environ.get("WANDB_ENTITY"),
        project=os.environ.get("WANDB_PROJECT", "proto-non-param"),
        config=vars(args),
        dir=args.log_dir,
    )

    dataset = VisualGenomeSceneGraphDataset(
        root=DEFAULT_VG_ROOT,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
        collate_fn=partial(
            scene_graph_collate_fn,
            negatives_per_positive=args.negatives_per_positive,
        ),
        drop_last=True,
    )

    llm2vec = load_llm2vec(
        cache_dir=(
            "/net/tscratch/people/"
            "plgpiotrwojcik/model_cache"
        )
    )

    # The LLM2Vec encoder supplies fixed target embeddings. Only PNP is
    # optimised by this training loop.
    if hasattr(llm2vec, "model"):
        llm2vec.model.eval()

        for parameter in llm2vec.model.parameters():
            parameter.requires_grad = False


    backbone, dim = build_backbone(args)

    # Replace these values with the actual model construction used by
    # your project.
    model = PNP(
        backbone=backbone,
        visual_dim=backbone.dim,
        text_dim=args.text_dim,
        projection_hidden_dim=1024,
        temperature=0.2,
    ).to(device)

    criterion = PNPContrastiveCriterion(
        infonce_coef=1.0,
        ranking_coef=1.0,
        consistency_coef=0.25,
        entropy_coef=0.01,
        temperature=0.07,
        margin=0.2,
    )

    optimizer = AdamW(
        [
            {
                "params": model.text_projection_head.parameters(),
                "lr": 1e-4,
            },
            {
                "params": [
                    parameter
                    for parameter in model.backbone.parameters()
                    if parameter.requires_grad
                ],
                "lr": 1e-5,
            },
        ],
        weight_decay=1e-4,
    )

    train(
        model=model,
        llm2vec=llm2vec,
        dataloader=dataloader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        epochs=args.epochs,
        encode_batch_size=args.encode_batch_size,
        gradient_clip_norm=1.0,
        use_amp=True,
        checkpoint_dir=args.checkpoint_dir,
        visualize_every_steps=args.visualize_every_steps,
        visualize_samples=args.visualize_samples,
        visualize_images_per_batch=args.visualize_images_per_batch,
    )

    if wandb.run is not None:
        wandb.finish()


def _description_to_text(value: Any) -> str:
    if isinstance(value, str):
        return value

    if isinstance(value, dict):
        for key in ("description", "text", "caption", "sentence"):
            candidate = value.get(key)
            if isinstance(candidate, str) and candidate.strip():
                return candidate

    if isinstance(value, (list, tuple)) and value:
        return _description_to_text(random.choice(value))

    raise ValueError(
        f"Unsupported or empty description value: {type(value)!r}"
    )


def _encode_descriptions(
    *,
    llm2vec: LLM2Vec,
    descriptions: list[str],
    encode_batch_size: int,
) -> torch.Tensor:
    inputs = [["", description] for description in descriptions]
    chunks: list[torch.Tensor] = []

    for start in range(0, len(inputs), encode_batch_size):
        encoded = llm2vec.encode(
            inputs[start : start + encode_batch_size]
        )
        if not torch.is_tensor(encoded):
            encoded = torch.as_tensor(encoded)
        chunks.append(encoded)

    if not chunks:
        raise ValueError("No descriptions were available to encode.")

    return torch.cat(chunks, dim=0)


def visualize_heatmaps(
    *,
    model: PNP,
    llm2vec: LLM2Vec,
    dataloader: DataLoader,
    device: torch.device,
    encode_batch_size: int = 32,
    num_samples: int = 1,
    images_per_batch: int = 4,
    global_step: int | None = None,
) -> None:
    """Encode descriptions, generate PNP heatmaps, and log them to W&B."""
    import matplotlib.pyplot as plt

    if wandb.run is None:
        raise RuntimeError(
            "visualize_heatmaps requires an active wandb run."
        )

    was_training = model.training
    model.eval()
    iterator = iter(dataloader)

    try:
        with torch.no_grad():
            for sample_idx in range(num_samples):
                try:
                    batch = next(iterator)
                except StopIteration:
                    iterator = iter(dataloader)
                    batch = next(iterator)

                if not isinstance(batch, dict) or "descriptions" not in batch:
                    raise KeyError(
                        "scene_graph_collate_fn must return a dictionary "
                        "containing a 'descriptions' entry."
                    )

                images = get_images(batch).to(
                    device=device,
                    non_blocking=True,
                )
                descriptions = [
                    _description_to_text(value)
                    for value in batch["descriptions"]
                ]

                if len(descriptions) != images.shape[0]:
                    raise ValueError(
                        "Description count does not match image batch size: "
                        f"{len(descriptions)} versus {images.shape[0]}."
                    )

                text_embeddings = _encode_descriptions(
                    llm2vec=llm2vec,
                    descriptions=descriptions,
                    encode_batch_size=encode_batch_size,
                )
                text_embeddings = F.normalize(
                    text_embeddings.float(),
                    dim=-1,
                )
                text_embeddings = prepare_embeddings(
                    text_embeddings,
                    device=device,
                    dtype=images.dtype,
                )

                autocast_context = (
                    torch.autocast(
                        device_type="cuda",
                        dtype=torch.float16,
                    )
                    if device.type == "cuda"
                    else nullcontext()
                )
                with autocast_context:
                    feature_maps = model(images, text_embeddings)

                logged_images = []
                count = min(images.shape[0], images_per_batch)

                for img_idx in range(count):
                    img = images[img_idx].detach().float().cpu()
                    heatmap = feature_maps[
                        img_idx, 0
                    ].detach().float().cpu()

                    img = (img - img.min()) / (
                        img.max() - img.min() + 1e-8
                    )
                    heatmap = F.interpolate(
                        heatmap[None, None],
                        size=img.shape[-2:],
                        mode="bilinear",
                        align_corners=False,
                    )[0, 0]
                    heatmap = (heatmap - heatmap.min()) / (
                        heatmap.max() - heatmap.min() + 1e-8
                    )

                    display_image = (
                        img.permute(1, 2, 0)
                        if img.shape[0] == 3
                        else img.squeeze()
                    )

                    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
                    axes[0].imshow(
                        display_image,
                        cmap=None if img.shape[0] == 3 else "gray",
                    )
                    axes[0].set_title("Original image")
                    axes[0].axis("off")

                    axes[1].imshow(
                        display_image,
                        cmap=None if img.shape[0] == 3 else "gray",
                    )
                    axes[1].imshow(heatmap, cmap="jet", alpha=0.5)
                    axes[1].set_title(
                        descriptions[img_idx],
                        wrap=True,
                    )
                    axes[1].axis("off")
                    fig.tight_layout()

                    logged_images.append(
                        wandb.Image(
                            fig,
                            caption=descriptions[img_idx],
                        )
                    )
                    plt.close(fig)

                log_kwargs = {}
                if global_step is not None:
                    log_kwargs["step"] = global_step
                    log_kwargs["commit"] = sample_idx == num_samples - 1

                wandb.log(
                    {
                        f"heatmaps/sample_{sample_idx}": logged_images,
                        "heatmaps/global_step": (
                            global_step
                            if global_step is not None
                            else sample_idx
                        ),
                    },
                    **log_kwargs,
                )
    finally:
        model.train(was_training)


if __name__ == "__main__":
    main()