from pathlib import Path
from typing import Callable, Optional

import albumentations as A
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from albumentations.augmentations.crops.functional import crop_keypoint_by_coords
from PIL import Image
from torchvision.datasets import ImageFolder
from torchvision.transforms.functional import to_tensor
import torchvision.transforms as T

import os
from PIL import Image
from torch.utils.data import Dataset
from typing import Callable, Optional, Tuple, List, Dict


train_transforms = T.Compose([
    T.RandomHorizontalFlip(p=0.5),
    T.RandomRotation(15),
    T.RandomAffine(
        degrees=0,
        shear=(-10, 10)   # approximates skew/shear
    ),
    T.RandomPerspective(
        distortion_scale=0.5,
        p=0.5             # approximates random_distortion
    ),
    T.ToTensor(),
])

class TinyImageNetDataset(Dataset):
    """
    Tiny ImageNet dataset loader.

    Supports multiple common layouts:

    Official:
      root/
        train/<wnid>/images/*.JPEG
        val/images/*.JPEG
        val/val_annotations.txt

    Repacked variants:
      root/
        train/<wnid>/*.JPEG
        val/<wnid>/*.JPEG           (or val/<wnid>/images/*.JPEG)
      or
        val/*.JPEG                  (unlabeled; label = -1)

    Returns: image, label_id, index
    """

    def __init__(
        self,
        root: str,
        split: str = "train",  # "train" or "val"
        transforms: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ):
        self.root = root
        self.split = split
        self.transform = transforms
        self.target_transform = target_transform

        train_dir = os.path.join(root, "train")
        val_dir = os.path.join(root, "val")

        if not os.path.isdir(train_dir):
            raise FileNotFoundError(f"Could not find train dir: {train_dir}")
        if not os.path.isdir(val_dir):
            raise FileNotFoundError(f"Could not find val dir: {val_dir}")

        # Build wnid -> class_id mapping from train folder names
        wnids = sorted([d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))])
        if len(wnids) == 0:
            raise ValueError(f"No class folders found under: {train_dir}")

        self.class_to_idx: Dict[str, int] = {wnid: i for i, wnid in enumerate(wnids)}
        self.samples: List[Tuple[str, int]] = []

        self.classes = wnids  # ImageFolder-compatible

        exts = (".jpeg", ".jpg", ".png")

        if split == "train":
            for wnid in wnids:
                label = self.class_to_idx[wnid]
                # Support both:
                # 1) train/<wnid>/images/*.JPEG (official)
                # 2) train/<wnid>/*.JPEG        (repack)
                candidates = [
                    os.path.join(train_dir, wnid, "images"),
                    os.path.join(train_dir, wnid),
                ]
                img_dir = next((d for d in candidates if os.path.isdir(d)), None)
                if img_dir is None:
                    continue

                for fn in os.listdir(img_dir):
                    if fn.lower().endswith(exts):
                        self.samples.append((os.path.join(img_dir, fn), label))

        elif split == "val":
            ann_path = os.path.join(val_dir, "val_annotations.txt")

            # Case 1: official annotations exist
            if os.path.isfile(ann_path):
                img_dir_candidates = [os.path.join(val_dir, "images"), val_dir]
                img_dir = next((d for d in img_dir_candidates if os.path.isdir(d)), None)
                if img_dir is None:
                    raise FileNotFoundError(f"Could not find val image directory in: {img_dir_candidates}")

                with open(ann_path, "r") as f:
                    for line in f:
                        parts = line.strip().split("\t")
                        if len(parts) < 2:
                            continue
                        img_name, wnid = parts[0], parts[1]
                        if wnid not in self.class_to_idx:
                            continue
                        label = self.class_to_idx[wnid]
                        img_path = os.path.join(img_dir, img_name)

                        # extension/case fallback
                        if not os.path.isfile(img_path):
                            base, _ = os.path.splitext(img_name)
                            for cand in (base + ".JPEG", base + ".jpeg", base + ".jpg", base + ".png"):
                                cand_path = os.path.join(img_dir, cand)
                                if os.path.isfile(cand_path):
                                    img_path = cand_path
                                    break

                        if os.path.isfile(img_path):
                            self.samples.append((img_path, label))

            else:
                # Case 2: val is already organized into class folders
                val_wnids = sorted([
                    d for d in os.listdir(val_dir)
                    if os.path.isdir(os.path.join(val_dir, d)) and d in self.class_to_idx
                ])

                if len(val_wnids) > 0:
                    for wnid in val_wnids:
                        label = self.class_to_idx[wnid]
                        candidates = [
                            os.path.join(val_dir, wnid, "images"),
                            os.path.join(val_dir, wnid),
                        ]
                        img_dir = next((d for d in candidates if os.path.isdir(d)), None)
                        if img_dir is None:
                            continue
                        for fn in os.listdir(img_dir):
                            if fn.lower().endswith(exts):
                                self.samples.append((os.path.join(img_dir, fn), label))

                else:
                    # Case 3: val is a flat folder with images only -> unlabeled
                    img_dir_candidates = [os.path.join(val_dir, "images"), val_dir]
                    img_dir = next((d for d in img_dir_candidates if os.path.isdir(d)), None)
                    if img_dir is None:
                        raise FileNotFoundError(f"Could not find val image directory in: {img_dir_candidates}")

                    for fn in os.listdir(img_dir):
                        if fn.lower().endswith(exts):
                            self.samples.append((os.path.join(img_dir, fn), -1))

        else:
            raise ValueError(f"Unknown split: {split}. Use 'train' or 'val'.")

        if len(self.samples) == 0:
            raise ValueError(
                f"TinyImageNetDataset(split='{split}') found 0 samples. "
                f"root={root}, train_dir={train_dir}, val_dir={val_dir}"
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        im_path, label = self.samples[index]
        img = Image.open(im_path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            label = self.target_transform(label)
        return img, label, index


class CUBDataset(ImageFolder):
    def __init__(self,
                 root: str,
                 transforms: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None):
        super().__init__(
            root=root,
            transform=transforms,
            target_transform=target_transform
        )

    def __getitem__(self, index: int):
        im_path, label = self.samples[index]
        im_pt = self.transform(Image.open(im_path).convert("RGB"))
        return im_pt, label, index


class CUBEvalDataset(ImageFolder):
    def __init__(self,
                 images_root: str,
                 annotations_root: str,
                 normalization: bool = True,
                 input_size: int = 224):
        transforms = [A.Resize(width=input_size, height=input_size)]
        transforms += [A.Normalize(mean=(0.485, 0.456, 0.406,), std=(0.229, 0.224, 0.225,))] if normalization else []

        super().__init__(
            root=images_root,
            transform=A.Compose(
                transforms,
                keypoint_params=A.KeypointParams(
                    format='xy',
                    label_fields=None,
                    remove_invisible=True,
                    angle_in_degrees=True
                )
            ),
            target_transform=None
        )
        self.input_size = input_size
        annotations_root = Path("datasets") / "CUB_200_2011"

        path_df = pd.read_csv(annotations_root / "images.txt", header=None, names=["image_id", "image_path"], sep=" ")
        bbox_df = pd.read_csv(annotations_root / "bounding_boxes.txt", header=None,
                              names=["image_id", "x", "y", "w", "h"], sep=" ")
        self.bbox_df = path_df.merge(bbox_df, on="image_id")
        self.part_loc_df = pd.read_csv(annotations_root / "parts" / "part_locs.txt", header=None,
                                       names=["image_id", "part_id", "kp_x", "kp_y", "visible"], sep=" ")

        attributes_np = np.loadtxt(annotations_root / "attributes" / "class_attribute_labels_continuous.txt")
        self.attributes = F.normalize(torch.tensor(attributes_np, dtype=torch.float32), p=2, dim=-1)

    def __getitem__(self, index: int):
        im_path, label = self.samples[index]
        im = np.array(Image.open(im_path).convert("RGB"))

        row = self.bbox_df[self.bbox_df["image_path"] == "/".join(Path(im_path).parts[-2:])].iloc[0]
        image_id = row["image_id"]
        bbox_coords = row[["x", "y", "w", "h"]].values.flatten()

        mask = self.part_loc_df["image_id"] == image_id
        keypoints = self.part_loc_df[mask][["kp_x", "kp_y"]].values

        keypoints_cropped = [crop_keypoint_by_coords(keypoint=tuple(kp) + (None, None,), crop_coords=bbox_coords[:2])
                             for kp in keypoints]
        keypoints_cropped = [(np.clip(x, 0, self.input_size), np.clip(y, 0, self.input_size),) for x, y, _, _ in
                             keypoints_cropped]

        transformed = self.transform(image=im, keypoints=keypoints_cropped)
        transformed_im, transformed_keypoints = transformed["image"], transformed["keypoints"]

        return to_tensor(transformed_im), torch.tensor(transformed_keypoints,
                                                       dtype=torch.float32), label, self.attributes[label, :], index
