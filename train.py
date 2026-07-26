import argparse
import importlib
import os
import random
from contextlib import nullcontext
from functools import partial
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from llm2vec import LLM2Vec
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import wandb
from modeling.pnp import PNP
from visual_genome_scene_graph_dataset import (
    DEFAULT_VG_ROOT,
    VisualGenomeSceneGraphDataset,
    scene_graph_collate_fn,
)


def load_llm2vec(
    *,
    model_name: str,
    peft_model_name: str | None,
    cache_dir: str | Path | None,
    device: torch.device,
) -> LLM2Vec:
    """Load and freeze the LLM2Vec text encoder."""
    dtype = torch.float32 if device.type == "cpu" else torch.bfloat16
    kwargs: dict[str, Any] = {
        "cache_dir": str(cache_dir) if cache_dir is not None else None,
        "device_map": str(device),
        "torch_dtype": dtype,
    }
    if peft_model_name:
        kwargs["peft_model_name_or_path"] = peft_model_name

    encoder = LLM2Vec.from_pretrained(model_name, **kwargs)

    if hasattr(encoder, "model"):
        encoder.model.eval()
        for parameter in encoder.model.parameters():
            parameter.requires_grad_(False)

    return encoder


def _encode_texts(
    *,
    llm2vec: LLM2Vec,
    texts: list[str],
    encode_batch_size: int,
) -> torch.Tensor:
    """Encode strings in bounded chunks and return [N, D]."""
    if encode_batch_size <= 0:
        raise ValueError("encode_batch_size must be positive")
    if not texts:
        raise ValueError("Cannot encode an empty text list")

    chunks: list[torch.Tensor] = []
    for start in range(0, len(texts), encode_batch_size):
        inputs = [["", text] for text in texts[start : start + encode_batch_size]]
        encoded = llm2vec.encode(inputs)
        if not torch.is_tensor(encoded):
            encoded = torch.as_tensor(encoded)
        if encoded.ndim != 2:
            raise ValueError(
                "LLM2Vec must return [N, D] embeddings, "
                f"received {tuple(encoded.shape)}"
            )
        chunks.append(encoded)

    return torch.cat(chunks, dim=0)


def encode_pair_strings(
    *,
    llm2vec: LLM2Vec,
    batch: dict[str, Any],
    encode_batch_size: int,
) -> dict[str, torch.Tensor]:
    """Encode all positive and negative relationship strings."""
    key_map = {
        "positive_anchor_embeddings": "positive_anchor_texts",
        "positive_text_embeddings": "positive_texts",
        "negative_anchor_embeddings": "negative_anchor_texts",
        "negative_text_embeddings": "negative_texts",
    }

    result: dict[str, torch.Tensor] = {}
    for output_key, batch_key in key_map.items():
        if batch_key not in batch:
            raise KeyError(
                f"scene_graph_collate_fn did not return {batch_key!r}. "
                f"Available keys: {sorted(batch)}"
            )
        result[output_key] = _encode_texts(
            llm2vec=llm2vec,
            texts=[str(value) for value in batch[batch_key]],
            encode_batch_size=encode_batch_size,
        )

    return result


def get_images(batch: Any) -> torch.Tensor:
    images = batch["image"]

    if not torch.is_tensor(images):
        raise TypeError(
            f"Expected batch['image'] to be a tensor, got {type(images)!r}"
        )

    if images.ndim != 4:
        raise ValueError(
            f"Expected [B, C, H, W], got {tuple(images.shape)}"
        )

    return images


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


class PNPContrastiveCriterion(nn.Module):
    """Contrast positive and negative PNP similarity maps."""

    def __init__(
        self,
        infonce_coef: float = 1.0,
        binary_coef: float = 1.0,
        ranking_coef: float = 1.0,
        temperature: float = 0.07,
        margin: float = 0.2,
    ) -> None:
        super().__init__()
        if temperature <= 0:
            raise ValueError("temperature must be greater than zero")
        if margin < 0:
            raise ValueError("margin must be non-negative")

        self.infonce_coef = infonce_coef
        self.binary_coef = binary_coef
        self.ranking_coef = ranking_coef
        self.temperature = temperature
        self.margin = margin

    @staticmethod
    def _flatten_maps(
        maps: torch.Tensor,
        name: str,
    ) -> torch.Tensor:
        if maps.ndim != 4 or maps.shape[1] != 1:
            raise ValueError(
                f"{name} must have shape [N, 1, H, W], "
                f"received {tuple(maps.shape)}."
            )
        return maps.flatten(start_dim=1).float()

    def forward(
        self,
        *,
        positive_anchor_maps: torch.Tensor,
        positive_text_maps: torch.Tensor,
        negative_anchor_maps: torch.Tensor,
        negative_text_maps: torch.Tensor,
        batch_size: int,
        negatives_per_positive: int,
    ) -> dict[str, torch.Tensor]:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if negatives_per_positive <= 0:
            raise ValueError("negatives_per_positive must be positive")

        expected_negatives = batch_size * negatives_per_positive
        if positive_anchor_maps.shape[0] != batch_size:
            raise ValueError(
                "positive_anchor_maps count does not match batch_size: "
                f"{positive_anchor_maps.shape[0]} versus {batch_size}."
            )
        if positive_text_maps.shape[0] != batch_size:
            raise ValueError(
                "positive_text_maps count does not match batch_size: "
                f"{positive_text_maps.shape[0]} versus {batch_size}."
            )
        if negative_anchor_maps.shape[0] != expected_negatives:
            raise ValueError(
                "negative_anchor_maps count does not match "
                "batch_size * negatives_per_positive: "
                f"{negative_anchor_maps.shape[0]} versus {expected_negatives}."
            )
        if negative_text_maps.shape[0] != expected_negatives:
            raise ValueError(
                "negative_text_maps count does not match "
                "batch_size * negatives_per_positive: "
                f"{negative_text_maps.shape[0]} versus {expected_negatives}."
            )

        positive_anchor = self._flatten_maps(
            positive_anchor_maps,
            "positive_anchor_maps",
        )
        positive_text = self._flatten_maps(
            positive_text_maps,
            "positive_text_maps",
        )
        negative_anchor = self._flatten_maps(
            negative_anchor_maps,
            "negative_anchor_maps",
        )
        negative_text = self._flatten_maps(
            negative_text_maps,
            "negative_text_maps",
        )

        positive_anchor_normalized = F.normalize(positive_anchor, dim=-1)
        positive_text_normalized = F.normalize(positive_text, dim=-1)
        negative_anchor_normalized = F.normalize(negative_anchor, dim=-1)
        negative_text_normalized = F.normalize(negative_text, dim=-1)

        positive_similarity = (
            positive_anchor_normalized * positive_text_normalized
        ).sum(dim=-1)
        negative_similarity = (
            negative_anchor_normalized * negative_text_normalized
        ).sum(dim=-1)
        grouped_negative_similarity = negative_similarity.reshape(
            batch_size,
            negatives_per_positive,
        )
        hardest_negative_similarity = grouped_negative_similarity.max(dim=1).values

        losses: dict[str, torch.Tensor] = {}

        if self.infonce_coef != 0:
            logits = (
                positive_anchor_normalized
                @ positive_text_normalized.transpose(0, 1)
            ) / self.temperature
            labels = torch.arange(batch_size, device=logits.device)
            infonce = 0.5 * (
                F.cross_entropy(logits, labels)
                + F.cross_entropy(logits.transpose(0, 1), labels)
            )
            losses["l_infonce"] = self.infonce_coef * infonce

        if self.binary_coef != 0:
            pair_logits = torch.cat(
                [positive_similarity, negative_similarity],
                dim=0,
            ) / self.temperature
            targets = torch.cat(
                [
                    torch.ones_like(positive_similarity),
                    torch.zeros_like(negative_similarity),
                ],
                dim=0,
            )
            losses["l_binary"] = self.binary_coef * (
                F.binary_cross_entropy_with_logits(pair_logits, targets)
            )

        if self.ranking_coef != 0:
            ranking = F.relu(
                self.margin
                - positive_similarity
                + hardest_negative_similarity
            ).mean()
            losses["l_ranking"] = self.ranking_coef * ranking

        losses["positive_similarity"] = (
            positive_similarity.mean().detach()
        )
        losses["hardest_negative_similarity"] = (
            hardest_negative_similarity.mean().detach()
        )
        return losses


def get_triple_image_id(triple: Any) -> int:
    """Extract the source image ID from a relationship record."""
    if isinstance(triple, dict):
        return int(triple["image_id"])

    if hasattr(triple, "image_id"):
        return int(triple.image_id)

    raise TypeError(
        "Each triple must contain an image_id field. "
        f"Received {type(triple)!r}."
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
    max_steps: int = 0,
) -> None:
    model.train()

    amp_enabled = use_amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    checkpoint_path = (
        Path(checkpoint_dir)
        if checkpoint_dir is not None
        else None
    )
    if checkpoint_path is not None:
        checkpoint_path.mkdir(parents=True, exist_ok=True)

    global_step = 0

    for epoch in range(epochs):
        progress = tqdm(
            dataloader,
            desc=f"Epoch {epoch + 1}/{epochs}",
        )
        running_loss = 0.0
        processed_batches = 0

        for batch_index, batch in enumerate(progress):
            # images contains one tensor per unique image in the loader batch.
            images = get_images(batch).to(
                device=device,
                non_blocking=True,
            )

            positive_triples = batch["positive_triples"]
            negative_triples = batch["negative_triples"]
            num_positive_pairs = len(positive_triples)
            num_negative_pairs = len(negative_triples)

            if num_positive_pairs == 0 or num_negative_pairs == 0:
                continue

            if num_negative_pairs % num_positive_pairs != 0:
                raise ValueError(
                    "The number of negative triples must be divisible by "
                    "the number of positive triples. Received "
                    f"{num_negative_pairs} negatives for "
                    f"{num_positive_pairs} positives."
                )

            negatives_per_positive = (
                num_negative_pairs // num_positive_pairs
            )

            # LLM2Vec is frozen and only generates text embeddings.
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

            # Accept [N, D] or [P, K, D] negatives and convert to [N, D].
            if negative_anchor_embeddings.ndim == 3:
                negative_anchor_embeddings = (
                    negative_anchor_embeddings.flatten(0, 1)
                )
            if negative_text_embeddings.ndim == 3:
                negative_text_embeddings = (
                    negative_text_embeddings.flatten(0, 1)
                )

            embedding_groups = {
                "positive_anchor_embeddings": (
                    positive_anchor_embeddings,
                    num_positive_pairs,
                ),
                "positive_text_embeddings": (
                    positive_text_embeddings,
                    num_positive_pairs,
                ),
                "negative_anchor_embeddings": (
                    negative_anchor_embeddings,
                    num_negative_pairs,
                ),
                "negative_text_embeddings": (
                    negative_text_embeddings,
                    num_negative_pairs,
                ),
            }

            for name, (embeddings, expected_count) in embedding_groups.items():
                if embeddings.ndim != 2:
                    raise ValueError(
                        f"{name} must have shape [N, D], received "
                        f"{tuple(embeddings.shape)}"
                    )
                if embeddings.shape[0] != expected_count:
                    raise ValueError(
                        f"{name} count does not match its relationship "
                        f"records: {embeddings.shape[0]} versus "
                        f"{expected_count}."
                    )

            embedding_dims = {
                positive_anchor_embeddings.shape[1],
                positive_text_embeddings.shape[1],
                negative_anchor_embeddings.shape[1],
                negative_text_embeddings.shape[1],
            }
            if len(embedding_dims) != 1:
                raise ValueError(
                    "All positive and negative embeddings must have the "
                    f"same dimension, received {sorted(embedding_dims)}."
                )

            batch_image_ids = batch["image_id"]
            if len(batch_image_ids) != images.shape[0]:
                raise ValueError(
                    "Image ID count does not match the unique image batch: "
                    f"{len(batch_image_ids)} versus {images.shape[0]}."
                )

            image_id_to_batch_index = {
                int(image_id): index
                for index, image_id in enumerate(batch_image_ids)
            }

            try:
                positive_image_indices = torch.tensor(
                    [
                        image_id_to_batch_index[
                            get_triple_image_id(triple)
                        ]
                        for triple in positive_triples
                    ],
                    dtype=torch.long,
                    device=device,
                )
                negative_image_indices = torch.tensor(
                    [
                        image_id_to_batch_index[
                            get_triple_image_id(triple)
                        ]
                        for triple in negative_triples
                    ],
                    dtype=torch.long,
                    device=device,
                )
            except KeyError as error:
                raise ValueError(
                    "A relationship triple refers to an image that is not "
                    f"present in the current batch: {error.args[0]}."
                ) from error

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
            positive_images = images.index_select(
                dim=0,
                index=positive_image_indices,
            )
            negative_images = images.index_select(
                dim=0,
                index=negative_image_indices,
            )

            positive_anchor_maps = model(
                positive_images,
                positive_anchor_embeddings,
            )
            positive_text_maps = model(
                positive_images,
                positive_text_embeddings,
            )
            negative_anchor_maps = model(
                negative_images,
                negative_anchor_embeddings,
            )
            negative_text_maps = model(
                negative_images,
                negative_text_embeddings,
            )

            validate_similarity_map(
                positive_anchor_maps,
                batch_size=num_positive_pairs,
                name="positive_anchor_maps",
            )
            validate_similarity_map(
                positive_text_maps,
                batch_size=num_positive_pairs,
                name="positive_text_maps",
            )
            validate_similarity_map(
                negative_anchor_maps,
                batch_size=num_negative_pairs,
                name="negative_anchor_maps",
            )
            validate_similarity_map(
                negative_text_maps,
                batch_size=num_negative_pairs,
                name="negative_text_maps",
            )

            loss_dict = criterion(
                positive_anchor_maps=positive_anchor_maps,
                positive_text_maps=positive_text_maps,
                negative_anchor_maps=negative_anchor_maps,
                negative_text_maps=negative_text_maps,
                batch_size=num_positive_pairs,
                negatives_per_positive=negatives_per_positive,
            )

            train_losses = [
                value
                for name, value in loss_dict.items()
                if name.startswith("l_")
            ]

            if not train_losses:
                raise ValueError(
                    "The criterion returned no losses whose names start "
                    "with 'l_'."
                )

            total_loss = torch.stack(
                [loss.reshape(()) for loss in train_losses]
            ).sum()

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
            if max_steps > 0 and global_step >= max_steps:
                print(
                    f"CPU smoke test completed after {global_step} step(s).",
                    flush=True,
                )
                return
            processed_batches += 1
            running_loss += total_loss.detach().item()
            average_loss = running_loss / processed_batches

            postfix = {
                "loss": f"{total_loss.detach().item():.4f}",
                "average": f"{average_loss:.4f}",
            }
            if "positive_similarity" in loss_dict:
                postfix["positive"] = (
                    f"{loss_dict['positive_similarity'].detach().item():.3f}"
                )
            if "hardest_negative_similarity" in loss_dict:
                postfix["negative"] = (
                    f"{loss_dict['hardest_negative_similarity'].detach().item():.3f}"
                )
            progress.set_postfix(**postfix)

            wandb_metrics = {
                "train/loss": total_loss.detach().item(),
                "train/loss_avg": average_loss,
                "train/epoch": epoch + 1,
                "train/positive_pairs": num_positive_pairs,
                "train/negative_pairs": num_negative_pairs,
                "train/negatives_per_positive": negatives_per_positive,
            }
            for name, value in loss_dict.items():
                if torch.is_tensor(value) and value.numel() == 1:
                    wandb_metrics[f"train/{name}"] = (
                        value.detach().item()
                    )

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
                model.train()

        if checkpoint_path is not None:
            torch.save(
                {
                    "epoch": epoch + 1,
                    "global_step": global_step,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scaler": scaler.state_dict(),
                },
                checkpoint_path / f"pnp_epoch_{epoch + 1:03d}.pt",
            )

class MockLLM2Vec:
    def __init__(self, embedding_dim: int = 4096) -> None:
        self.embedding_dim = embedding_dim

    def encode(self, inputs: list[list[str]]) -> torch.Tensor:
        embeddings = []

        for instruction, text in inputs:
            seed = hash((instruction, text)) & 0x7FFFFFFF
            generator = torch.Generator(device="cpu")
            generator.manual_seed(seed)

            embedding = torch.randn(
                self.embedding_dim,
                generator=generator,
                dtype=torch.float32,
            )
            embedding = F.normalize(embedding, dim=0)
            embeddings.append(embedding)

        return torch.stack(embeddings, dim=0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train dictionary-free PNP on Visual Genome."
    )
    parser.add_argument("--backbone", default="dinov2_vitb14")
    parser.add_argument(
        "--backbone-module",
        default="modeling.backbone",
        help="Python module containing the backbone classes.",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path(DEFAULT_VG_ROOT),
        help="Visual Genome dataset root.",
    )
    parser.add_argument(
        "--llm-model",
        default="McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp",
    )
    parser.add_argument(
        "--llm-peft-model",
        default=(
            "McGill-NLP/"
            "LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-unsup-simcse"
        ),
    )
    parser.add_argument(
        "--llm-cache-dir",
        type=Path,
        default=Path(
            "/net/tscratch/people/plgpiotrwojcik/model_cache"
        ),
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
        "--mock-text-embeddings",
        action="store_true",
        help="Use deterministic random text embeddings instead of loading LLM2Vec.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=0,
        help="Stop after N optimiser steps; 0 means no limit.",
    )
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
        root=args.dataset_root,
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

    if args.mock_text_embeddings:
        print(
            "Using mock text embeddings; LLM2Vec will not be loaded.",
            flush=True,
        )
        llm2vec = MockLLM2Vec(embedding_dim=args.text_dim)
    else:
        llm2vec = load_llm2vec(
            cache_dir=args.llm_cache_dir,
            device=device,
            model_name=args.llm_model,
            peft_model_name=args.llm_peft_model,
        )

    # The LLM2Vec encoder supplies fixed target embeddings. Only PNP is
    # optimised by this training loop.
    if hasattr(llm2vec, "model"):
        llm2vec.model.eval()

        for parameter in llm2vec.model.parameters():
            parameter.requires_grad = False


    backbone, _ = build_backbone(args)

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
        binary_coef=1.0,
        ranking_coef=1.0,
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
        max_steps=args.max_steps,
    )

    if wandb.run is not None:
        wandb.finish()


def _description_to_text(value: Any) -> str:
    if isinstance(value, str):
        return value

    if isinstance(value, dict):
        for key in ("phrase", "description", "text", "caption", "sentence"):
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