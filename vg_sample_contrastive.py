#!/usr/bin/env python3
"""
Standalone sanity check for the Visual Genome scene-graph dataset.

The script:

1. Loads the dataset.
2. Fetches one batch.
3. Checks positive and negative triples.
4. Unnormalises each image tensor.
5. Saves the images as PNG files.
6. Prints the corresponding descriptions.

Usage:
    python test_dataset.py

Images are saved to:
    dataset_sanity_output/
"""

from functools import partial
from pathlib import Path
from typing import Any, Sequence

import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from visual_genome_scene_graph_dataset import (
    DEFAULT_VG_ROOT,
    VisualGenomeSceneGraphDataset,
    scene_graph_collate_fn,
)


# Replace these values if your dataset uses different normalisation.
IMAGE_MEAN = (0.485, 0.456, 0.406)
IMAGE_STD = (0.229, 0.224, 0.225)


def get_triple_image_id(triple: Any) -> int:
    """Extract an image ID from a triple object or dictionary."""
    if isinstance(triple, dict):
        return int(triple["image_id"])

    if hasattr(triple, "image_id"):
        return int(triple.image_id)

    raise TypeError(
        f"Cannot extract image_id from triple of type {type(triple)}"
    )


def unnormalise_image(
    image: torch.Tensor,
    mean: Sequence[float],
    std: Sequence[float],
) -> torch.Tensor:
    """
    Undo channel-wise image normalisation.

    Given a normalised tensor:

        normalised = (original - mean) / std

    this function reconstructs:

        original = normalised * std + mean

    Args:
        image:
            Tensor with shape [C, H, W].
        mean:
            Per-channel normalisation mean.
        std:
            Per-channel normalisation standard deviation.

    Returns:
        Float tensor with values clamped to [0, 1].
    """
    if not torch.is_tensor(image):
        raise TypeError(f"Expected torch.Tensor, got {type(image)}")

    if image.ndim != 3:
        raise ValueError(
            "Expected an image tensor with shape [C, H, W], "
            f"got {tuple(image.shape)}"
        )

    image = image.detach().cpu().float()

    channels = image.shape[0]

    if len(mean) != channels:
        raise ValueError(
            f"Mean has {len(mean)} values, but image has {channels} channels"
        )

    if len(std) != channels:
        raise ValueError(
            f"Std has {len(std)} values, but image has {channels} channels"
        )

    mean_tensor = torch.tensor(
        mean,
        dtype=image.dtype,
    ).view(channels, 1, 1)

    std_tensor = torch.tensor(
        std,
        dtype=image.dtype,
    ).view(channels, 1, 1)

    image = image * std_tensor + mean_tensor

    return image.clamp(0.0, 1.0)


def save_unnormalised_image(
    image: Any,
    output_path: Path,
    mean: Sequence[float],
    std: Sequence[float],
) -> None:
    """
    Save a dataset image in a displayable, unnormalised form.

    Supports:
      - torch.Tensor with shape [C, H, W]
      - PIL.Image.Image
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if torch.is_tensor(image):
        unnormalised = unnormalise_image(
            image=image,
            mean=mean,
            std=std,
        )
        save_image(unnormalised, str(output_path))
        return

    if isinstance(image, Image.Image):
        image.convert("RGB").save(output_path)
        return

    raise TypeError(
        f"Unsupported image type: {type(image)}"
    )


def get_batch_item(values: Any, index: int) -> Any:
    """
    Retrieve one item from a tensor, list, tuple, or similar batch container.
    """
    try:
        return values[index]
    except (TypeError, IndexError, KeyError) as exc:
        raise RuntimeError(
            f"Could not retrieve batch item at index {index}"
        ) from exc


def print_description(description: Any) -> None:
    """Print a description in a readable form."""
    if description is None:
        print(" description: unavailable")
        return

    if isinstance(description, str):
        print(" description:", description)
        return

    if isinstance(description, (list, tuple)):
        if not description:
            print(" description: empty")
            return

        print(" descriptions:")

        for description_index, item in enumerate(description):
            print(f"   [{description_index}] {item}")

        return

    print(" description:", description)


def main() -> None:
    batch_size = 8
    negatives_per_positive = 4

    output_dir = Path("dataset_sanity_output")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading dataset...")

    dataset = VisualGenomeSceneGraphDataset(
        root=DEFAULT_VG_ROOT,
    )

    print(f"Dataset size: {len(dataset)}")

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=partial(
            scene_graph_collate_fn,
            negatives_per_positive=negatives_per_positive,
        ),
    )

    print("Fetching first batch...")

    try:
        batch = next(iter(loader))
    except StopIteration as exc:
        raise RuntimeError("The dataset is empty") from exc

    print("\nBatch keys:")

    for key in sorted(batch.keys()):
        print(" ", key)

    required_keys = {
        "image",
        "image_id",
        "positive_triples",
        "negative_triples",
    }

    missing_keys = required_keys.difference(batch.keys())

    if missing_keys:
        raise KeyError(
            f"Batch is missing required keys: {sorted(missing_keys)}"
        )

    images = batch["image"]
    image_ids = batch["image_id"]
    descriptions = batch.get("descriptions")

    print("\nImages")

    if torch.is_tensor(images):
        print(" shape:", tuple(images.shape))
        print(" dtype:", images.dtype)
        print(" normalised min/max:", images.min().item(), images.max().item())

        if images.ndim != 4:
            raise ValueError(
                "Expected batched images with shape [B, C, H, W], "
                f"got {tuple(images.shape)}"
            )
    else:
        print(" type:", type(images))
        print(" count:", len(images))

    print("\nUnique images:", len(image_ids))
    print("Image IDs:", image_ids)

    if len(images) != len(image_ids):
        raise ValueError(
            "Number of images does not match number of image IDs: "
            f"{len(images)} images versus {len(image_ids)} IDs"
        )

    if descriptions is not None and len(descriptions) != len(image_ids):
        print(
            "\nWarning: number of descriptions does not match "
            "number of image IDs."
        )
        print(" descriptions:", len(descriptions))
        print(" image IDs:", len(image_ids))

    print("\nSaving unnormalised images...")

    for index, raw_image_id in enumerate(image_ids):
        image_id = int(raw_image_id)
        image = get_batch_item(images, index)

        output_path = output_dir / f"image_{image_id}.png"

        save_unnormalised_image(
            image=image,
            output_path=output_path,
            mean=IMAGE_MEAN,
            std=IMAGE_STD,
        )

        print(f"\nImage {index}")
        print(" image_id:", image_id)
        print(" saved to:", output_path.resolve())

        if descriptions is not None and index < len(descriptions):
            description = get_batch_item(descriptions, index)
            print_description(description)
        else:
            print_description(None)

    positive = batch["positive_triples"]
    negative = batch["negative_triples"]

    print("\nPositive triples:", len(positive))
    print("Negative triples:", len(negative))

    assert len(positive) > 0, "No positive triples were returned"
    assert len(negative) > 0, "No negative triples were returned"

    assert len(negative) % len(positive) == 0, (
        "The number of negative triples is not divisible by "
        "the number of positive triples"
    )

    actual_negatives_per_positive = len(negative) // len(positive)

    print(
        "Negatives per positive:",
        actual_negatives_per_positive,
    )

    assert actual_negatives_per_positive == negatives_per_positive, (
        f"Expected {negatives_per_positive} negatives per positive, "
        f"got {actual_negatives_per_positive}"
    )

    image_id_set = {int(image_id) for image_id in image_ids}

    print("\nChecking positive triples...")

    for triple_index, triple in enumerate(positive):
        triple_image_id = get_triple_image_id(triple)

        assert triple_image_id in image_id_set, (
            f"Positive triple {triple_index} refers to image "
            f"{triple_image_id}, which is not in the batch"
        )

    print("OK")

    print("Checking negative triples...")

    for triple_index, triple in enumerate(negative):
        triple_image_id = get_triple_image_id(triple)

        assert triple_image_id in image_id_set, (
            f"Negative triple {triple_index} refers to image "
            f"{triple_image_id}, which is not in the batch"
        )

    print("OK")

    if descriptions is not None:
        print("\nNumber of description entries:", len(descriptions))
    else:
        print("\nNo descriptions key was found in the batch.")

    print("\nFirst positive triple:")
    first_positive = positive[0]
    print({
        "image_id": first_positive.image_id,
        "object_id": first_positive.object_id,
        "anchor_text": first_positive.anchor_text,
        "positive_text": first_positive.positive_text,
        "relationship_id": first_positive.relationship_id,
    })

    print("\nFirst negative triple:")
    first_negative = negative[0]
    print({
        "image_id": first_negative.image_id,
        "anchor_object_id": first_negative.anchor_object_id,
        "anchor_text": first_negative.anchor_text,
        "negative_text": first_negative.negative_text,
        "negative_image_id": first_negative.negative_image_id,
        "negative_object_id": first_negative.negative_object_id,
    })

    print("\nSaved image directory:")
    print(output_dir.resolve())

    print("\nDataset sanity check PASSED.")


if __name__ == "__main__":
    main()
