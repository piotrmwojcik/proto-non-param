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


import os
from PIL import Image
from torch.utils.data import Dataset
from typing import Callable, Optional, Tuple, List, Dict


class TinyImageNetDataset(Dataset):
    """
    Tiny ImageNet dataset loader.

    Expected structure:
      root/
        train/
          <wnid>/
            images/*.JPEG
        val/
          images/*.JPEG
          val_annotations.txt

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

        # Build wnid -> class_id mapping from train folder names
        wnids = sorted([d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))])
        self.class_to_idx: Dict[str, int] = {wnid: i for i, wnid in enumerate(wnids)}

        self.samples: List[Tuple[str, int]] = []

        if split == "train":
            for wnid in wnids:
                label = self.class_to_idx[wnid]

                # Support both:
                # 1) train/<wnid>/images/*.JPEG  (official)
                # 2) train/<wnid>/*.JPEG         (some repacks)
                candidates = [
                    os.path.join(train_dir, wnid, "images"),
                    os.path.join(train_dir, wnid),
                ]
                img_dir = next((d for d in candidates if os.path.isdir(d)), None)
                if img_dir is None:
                    continue

                for fn in os.listdir(img_dir):
                    if fn.lower().endswith((".jpeg", ".jpg", ".png")):
                        self.samples.append((os.path.join(img_dir, fn), label))

        elif split == "val":
            ann_path = os.path.join(val_dir, "val_annotations.txt")

            # Support both:
            # 1) val/images/*.JPEG  (official)
            # 2) val/*.JPEG         (some repacks)
            img_dir_candidates = [
                os.path.join(val_dir, "images"),
                val_dir,
            ]
            img_dir = next((d for d in img_dir_candidates if os.path.isdir(d)), None)
            if img_dir is None:
                raise FileNotFoundError(f"Could not find val image directory in: {img_dir_candidates}")

            # val_annotations.txt format:
            # <image_filename>\t<wnid>\t<x0>\t<y0>\t<x1>\t<y1>
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

                    # Some packs may have different extension casing; try fallback search.
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
            raise ValueError(f"Unknown split: {split}. Use 'train' or 'val'.")

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
