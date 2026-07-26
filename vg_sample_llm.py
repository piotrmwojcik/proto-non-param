#!/usr/bin/env python3
"""
Standalone sanity check for the Visual Genome dataset.

Usage:
    python test_dataset.py
"""

from functools import partial

import torch
from torch.utils.data import DataLoader

from visual_genome_scene_graph_dataset import (
    DEFAULT_VG_ROOT,
    VisualGenomeSceneGraphDataset,
    scene_graph_collate_fn,
)


def get_triple_image_id(triple):
    if isinstance(triple, dict):
        return int(triple["image_id"])
    if hasattr(triple, "image_id"):
        return int(triple.image_id)
    raise TypeError(type(triple))


def main():

    batch_size = 8
    negatives_per_positive = 4

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
    batch = next(iter(loader))

    print("\nBatch keys:")
    for key in sorted(batch.keys()):
        print(" ", key)

    images = batch["image"]

    #assert torch.is_tensor(images)
    #assert images.ndim == 4

    print("\nImages")
    print(" shape:", tuple(images.shape))
    print(" dtype:", images.dtype)
    print(" min/max:", images.min().item(), images.max().item())

    image_ids = batch["image_id"]

    print("\nUnique images:", len(image_ids))
    print("Image IDs:", image_ids)

    positive = batch["positive_triples"]
    negative = batch["negative_triples"]

    print("\nPositive triples:", len(positive))
    print("Negative triples:", len(negative))

    assert len(positive) > 0
    assert len(negative) > 0

    assert len(negative) % len(positive) == 0

    print(
        "Negatives per positive:",
        len(negative) // len(positive),
    )

    image_id_set = set(map(int, image_ids))

    print("\nChecking positive triples...")

    for triple in positive:
        img_id = get_triple_image_id(triple)
        assert img_id in image_id_set

    print("OK")

    print("Checking negative triples...")

    for triple in negative:
        img_id = get_triple_image_id(triple)
        assert img_id in image_id_set

    print("OK")

    if "descriptions" in batch:
        print("\nDescriptions:", len(batch["descriptions"]))
        print(batch["descriptions"][0])

    print("\nFirst positive triple:")
    print(positive[0])

    print("\nFirst negative triple:")
    print(negative[0])

    print("\nDataset sanity check PASSED.")


if __name__ == "__main__":
    main()