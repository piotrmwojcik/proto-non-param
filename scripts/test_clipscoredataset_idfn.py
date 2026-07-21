"""Regression check for CLIPScoreDataset's id_fn addition (Stage 2 CUB fine-tuning).

Confirms (a) omitting id_fn reproduces the old int(stem) lookup exactly
(VG/COCO training must be unaffected) and (b) a provided id_fn correctly
resolves non-integer filenames (the CUB case) and returns None -> fallback
for misses. Local torch is broken on the dev machine -- run on Athena:
    python scripts/test_clipscoredataset_idfn.py
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from clip_dataset import CLIPScoreDataset


class FakeBase:
    """Minimal stand-in for a *CLIPDataset: samples = [(im_path, caption, prob_dist), ...]."""
    def __init__(self, paths):
        V = 4
        self.samples = [(p, "caption", torch.ones(V) / V) for p in paths]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        im_path, caption, prob_dist = self.samples[i]
        return torch.zeros(3, 4, 4), caption, prob_dist, i


def make_scores_file(path, image_ids, V=4):
    torch.save({
        "image_ids": image_ids,
        "vocab": [f"w{i}" for i in range(V)],
        "clip_scores": torch.randn(len(image_ids), V).half(),
    }, path)


with tempfile.TemporaryDirectory() as tmp:
    scores_path = os.path.join(tmp, "scores.pt")

    # (a) default behavior: integer-stem filenames (VG/COCO style) -- must be unchanged
    int_paths = [os.path.join(tmp, f"{i}.jpg") for i in (10, 20, 30)]
    make_scores_file(scores_path, image_ids=[10, 20, 30])
    base = FakeBase(int_paths)
    ds = CLIPScoreDataset(base, scores_path, top_k=4)
    assert all(lbl is not None for lbl in ds._labels), "default int(stem) lookup should match all 3"
    print("default int(stem) lookup: OK (3/3 matched)")

    # (a2) default behavior on a non-numeric filename -> must still fall back to None (old ValueError path)
    base_bad = FakeBase(int_paths + [os.path.join(tmp, "not_a_number.jpg")])
    ds_bad = CLIPScoreDataset(base_bad, scores_path, top_k=4)
    assert ds_bad._labels[-1] is None, "non-numeric stem must fall back to None without id_fn"
    print("default fallback on non-numeric stem: OK")

    # (b) id_fn provided: CUB-style non-integer filenames resolved via a lookup dict
    cub_paths = [
        os.path.join(tmp, "Black_Footed_Albatross_0001_796111.jpg"),
        os.path.join(tmp, "Least_Auklet_0001_795090.jpg"),
        os.path.join(tmp, "unmapped_image.jpg"),  # deliberately not in id_fn's table
    ]
    make_scores_file(scores_path, image_ids=[1, 2])
    id_map = {cub_paths[0]: 1, cub_paths[1]: 2}
    base_cub = FakeBase(cub_paths)
    ds_cub = CLIPScoreDataset(base_cub, scores_path, top_k=4, id_fn=id_map.get)
    assert ds_cub._labels[0] is not None and ds_cub._labels[1] is not None, "id_fn hits should match"
    assert ds_cub._labels[2] is None, "id_fn miss should fall back to None"
    print("id_fn CUB-style lookup: OK (2/2 matched, 1 miss falls back)")

print("\nAll CLIPScoreDataset id_fn regression checks passed.")
