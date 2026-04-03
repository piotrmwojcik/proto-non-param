"""
Tests for Caltech101CLIPDataset.

Run with:
    python -m pytest proto-non-param/test_caltech_dataset.py -v
or from inside proto-non-param/:
    python -m pytest test_caltech_dataset.py -v
"""
import json
import os
import tempfile

import pytest
import torch
from PIL import Image

from clip_dataset import Caltech101CLIPDataset, coco_clip_collate_fn


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

VOCAB = {"cat": 0, "dog": 1, "face": 2, "wheel": 3, "background": 4}


def _make_fake_image(path: str):
    """Write a tiny 32x32 RGB JPEG to path."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.new("RGB", (32, 32), color=(128, 64, 32)).save(path)


def _make_descriptions_json(tmp_dir: str, caltech_root: str) -> str:
    """
    Create a small descriptions JSON with 6 entries:
      3 in train (Faces class), 3 in val (Faces class).
    Returns path to the JSON file.
    """
    entries = {}
    for split in ("train", "val"):
        for i in range(1, 4):
            rel = f"{split}/Faces/image_{i:04d}.jpg"
            # Server-style absolute path inside caltech_root
            server_path = f"/net/tscratch/people/plgabedychaj/caltech101/{rel}"
            local_path = os.path.join(caltech_root, rel)
            _make_fake_image(local_path)
            entries[server_path] = [
                "A cat and a dog are visible in this face image.",
                "The background shows a wheel near the face.",
            ]

    json_path = os.path.join(tmp_dir, "descriptions.json")
    with open(json_path, "w") as f:
        json.dump(entries, f)
    return json_path


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_path_normalization():
    """Strip of server prefix leaves only the relative part after caltech101/."""
    norm = "/net/tscratch/people/plgabedychaj/caltech101/train/Faces/image_0001.jpg"
    norm = norm.replace("\\", "/")
    marker = "caltech101/"
    idx = norm.find(marker)
    assert idx != -1
    rel = norm[idx + len(marker):]
    assert rel == "train/Faces/image_0001.jpg"


def test_path_normalization_windows_style():
    """Windows-style backslashes are handled."""
    win_path = r"C:\data\caltech101\train\Faces\image_0001.jpg"
    norm = win_path.replace("\\", "/")
    marker = "caltech101/"
    idx = norm.find(marker)
    assert idx != -1
    rel = norm[idx + len(marker):]
    assert rel == "train/Faces/image_0001.jpg"


def test_split_filtering_train():
    with tempfile.TemporaryDirectory() as tmp:
        caltech_root = os.path.join(tmp, "caltech101")
        json_path = _make_descriptions_json(tmp, caltech_root)
        ds = Caltech101CLIPDataset(
            descriptions_json=json_path,
            caltech_root=caltech_root,
            vocab_to_idx=VOCAB,
            train=True,
            use_cache=False,
        )
        assert len(ds) == 3, f"Expected 3 train samples, got {len(ds)}"
        # All paths should contain /train/
        for im_path, _, _ in ds.samples:
            assert os.sep + "train" + os.sep in im_path or "/train/" in im_path


def test_split_filtering_val():
    with tempfile.TemporaryDirectory() as tmp:
        caltech_root = os.path.join(tmp, "caltech101")
        json_path = _make_descriptions_json(tmp, caltech_root)
        ds = Caltech101CLIPDataset(
            descriptions_json=json_path,
            caltech_root=caltech_root,
            vocab_to_idx=VOCAB,
            train=False,
            use_cache=False,
        )
        assert len(ds) == 3, f"Expected 3 val samples, got {len(ds)}"


def test_prob_dist_valid():
    """Probability distribution has correct shape, is non-negative, and sums to 1."""
    with tempfile.TemporaryDirectory() as tmp:
        caltech_root = os.path.join(tmp, "caltech101")
        json_path = _make_descriptions_json(tmp, caltech_root)
        ds = Caltech101CLIPDataset(
            descriptions_json=json_path,
            caltech_root=caltech_root,
            vocab_to_idx=VOCAB,
            train=True,
            use_cache=False,
        )
        for _, _, prob_dist in ds.samples:
            assert prob_dist.shape == (len(VOCAB),)
            assert (prob_dist >= 0).all()
            total = prob_dist.sum().item()
            assert abs(total - 1.0) < 1e-5 or total == 0.0, f"prob_dist sums to {total}"


def test_getitem_shapes():
    """__getitem__ returns correct tensor shapes."""
    with tempfile.TemporaryDirectory() as tmp:
        caltech_root = os.path.join(tmp, "caltech101")
        json_path = _make_descriptions_json(tmp, caltech_root)
        ds = Caltech101CLIPDataset(
            descriptions_json=json_path,
            caltech_root=caltech_root,
            vocab_to_idx=VOCAB,
            train=True,
            use_cache=False,
        )
        img_tensor, descriptions, prob_dist, idx = ds[0]
        assert img_tensor.shape == (3, 224, 224), f"Unexpected image shape: {img_tensor.shape}"
        assert isinstance(descriptions, list)
        assert prob_dist.shape == (len(VOCAB),)
        assert isinstance(idx, int)


def test_batch_shapes():
    """coco_clip_collate_fn stacks a batch correctly."""
    with tempfile.TemporaryDirectory() as tmp:
        caltech_root = os.path.join(tmp, "caltech101")
        json_path = _make_descriptions_json(tmp, caltech_root)
        ds = Caltech101CLIPDataset(
            descriptions_json=json_path,
            caltech_root=caltech_root,
            vocab_to_idx=VOCAB,
            train=True,
            use_cache=False,
        )
        batch = [ds[i] for i in range(len(ds))]
        images, captions, prob_dists, indices = coco_clip_collate_fn(batch)
        assert images.shape == (3, 3, 224, 224)
        assert prob_dists.shape == (3, len(VOCAB))
        assert indices.shape == (3,)
        assert len(captions) == 3


def test_cache_roundtrip():
    """Dataset saves and loads cache correctly."""
    with tempfile.TemporaryDirectory() as tmp:
        caltech_root = os.path.join(tmp, "caltech101")
        json_path = _make_descriptions_json(tmp, caltech_root)
        cache_dir = os.path.join(tmp, "cache")

        ds1 = Caltech101CLIPDataset(
            descriptions_json=json_path,
            caltech_root=caltech_root,
            vocab_to_idx=VOCAB,
            train=True,
            cache_dir=cache_dir,
            use_cache=True,
        )
        ds2 = Caltech101CLIPDataset(
            descriptions_json=json_path,
            caltech_root=caltech_root,
            vocab_to_idx=VOCAB,
            train=True,
            cache_dir=cache_dir,
            use_cache=True,
        )
        assert len(ds1) == len(ds2)
        for (p1, d1, pd1), (p2, d2, pd2) in zip(ds1.samples, ds2.samples):
            assert p1 == p2
            assert torch.allclose(pd1, pd2)
