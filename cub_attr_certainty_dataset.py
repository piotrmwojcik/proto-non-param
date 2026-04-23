#!/usr/bin/env python3
import os
import argparse
from collections import defaultdict

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
from torchvision.transforms import v2


def load_attributes_txt(attributes_txt_path: str):
    """
    attributes.txt format:
      attribute_id attribute_name

    Returns:
      attr_id_to_name: dict[int, str] with 1-based ids from file
      attr_name_to_idx: dict[str, int] with 0-based indices for tensors
      attr_names: list[str] in 0-based tensor order
    """
    attr_id_to_name = {}
    with open(attributes_txt_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            first_space = line.find(" ")
            attr_id = int(line[:first_space])
            attr_name = line[first_space + 1 :]
            attr_id_to_name[attr_id] = attr_name

    max_id = max(attr_id_to_name)
    attr_names = [attr_id_to_name[i] for i in range(1, max_id + 1)]
    attr_name_to_idx = {name: i for i, name in enumerate(attr_names)}
    return attr_id_to_name, attr_name_to_idx, attr_names


def load_images_txt(images_txt_path: str):
    """
    images.txt format:
      image_id relative_path
    """
    image_id_to_relpath = {}
    with open(images_txt_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            first_space = line.find(" ")
            image_id = int(line[:first_space])
            relpath = line[first_space + 1 :]
            image_id_to_relpath[image_id] = relpath
    return image_id_to_relpath


def load_train_test_split(split_txt_path: str):
    """
    train_test_split.txt format:
      image_id is_train
    where is_train is 1 for train, 0 for test
    """
    image_id_to_is_train = {}
    with open(split_txt_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            image_id_str, is_train_str = line.split()
            image_id_to_is_train[int(image_id_str)] = int(is_train_str)
    return image_id_to_is_train


def load_bounding_boxes(bboxes_txt_path: str):
    """
    bounding_boxes.txt format:
      image_id x y width height
    """
    image_id_to_bbox = {}
    with open(bboxes_txt_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            image_id = int(parts[0])
            x, y, w, h = map(float, parts[1:])
            image_id_to_bbox[image_id] = (x, y, w, h)
    return image_id_to_bbox


def load_image_attribute_labels(attr_labels_path: str):
    """
    image_attribute_labels.txt format:
      image_id attribute_id is_present certainty_id time

    Returns:
      image_id_to_attr_rows: dict[int, list[tuple[attr_id, is_present, certainty_id]]]
    """
    image_id_to_attr_rows = defaultdict(list)
    with open(attr_labels_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            image_id_str, attr_id_str, is_present_str, certainty_id_str, _time_str = (
                line.split()
            )
            image_id = int(image_id_str)
            attr_id = int(attr_id_str)
            is_present = int(is_present_str)
            certainty_id = int(certainty_id_str)
            image_id_to_attr_rows[image_id].append((attr_id, is_present, certainty_id))
    return image_id_to_attr_rows


class CUBAttributeCertaintyDataset(Dataset):
    """
    Alternative dataset using official CUB attribute labels with certainty weighting.

    Key behavior:
    - uses official train/test split
    - crops images using bounding boxes
    - keeps only top-2 certainty levels: certainty_id in {3, 4}
    - builds a weighted attribute vector over 312 attributes

    Returned sample:
      image_tensor, attr_prob_dist, attr_binary_mask, image_id
    """

    def __init__(
        self,
        cub_root: str,
        train: bool = True,
        image_size: int = 224,
        use_certainty_weights: bool = True,
        keep_top2_certainties_only: bool = True,
    ):
        self.cub_root = cub_root
        self.train = train
        self.image_size = image_size
        self.use_certainty_weights = use_certainty_weights
        self.keep_top2_certainties_only = keep_top2_certainties_only

        images_txt = os.path.join(cub_root, "images.txt")
        split_txt = os.path.join(cub_root, "train_test_split.txt")
        bboxes_txt = os.path.join(cub_root, "bounding_boxes.txt")
        attr_labels_txt = os.path.join(
            cub_root, "attributes", "image_attribute_labels.txt"
        )
        attributes_txt = os.path.join(cub_root, "attributes", "attributes.txt")
        images_root = os.path.join(cub_root, "images")

        self.images_root = images_root
        self.image_id_to_relpath = load_images_txt(images_txt)
        self.image_id_to_is_train = load_train_test_split(split_txt)
        self.image_id_to_bbox = load_bounding_boxes(bboxes_txt)
        self.image_id_to_attr_rows = load_image_attribute_labels(attr_labels_txt)
        self.attr_id_to_name, self.attr_name_to_idx, self.attr_names = (
            load_attributes_txt(attributes_txt)
        )

        self.num_attributes = len(self.attr_names)

        self.train_transform = v2.Compose(
            [
                v2.RandomResizedCrop(
                    size=image_size,
                    scale=(0.8, 1.0),
                    interpolation=v2.InterpolationMode.BICUBIC,
                ),
                v2.RandomHorizontalFlip(p=0.5),
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                ),
            ]
        )

        self.eval_transform = transforms.Compose(
            [
                transforms.Resize(
                    (image_size, image_size),
                    interpolation=transforms.InterpolationMode.BICUBIC,
                ),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                ),
            ]
        )

        # Certainty mapping using top two certainty levels.
        # CUB certainty_id meanings are typically:
        # 1 = not visible, 2 = guessing, 3 = probably, 4 = definitely
        #
        # We keep only 3 and 4. If weighted:
        #   3 -> 0.5
        #   4 -> 1.0
        # If not weighted:
        #   3/4 -> 1.0
        self.certainty_to_weight = {
            3: 0.5,
            4: 1.0,
        }

        samples = []
        for image_id, relpath in self.image_id_to_relpath.items():
            is_train = self.image_id_to_is_train[image_id] == 1
            if is_train != self.train:
                continue

            img_path = os.path.join(self.images_root, relpath)
            if not os.path.isfile(img_path):
                continue

            if image_id not in self.image_id_to_bbox:
                continue

            if image_id not in self.image_id_to_attr_rows:
                continue

            attr_weight_vec, attr_binary_vec = self._build_attribute_vectors(image_id)

            samples.append(
                {
                    "image_id": image_id,
                    "image_path": img_path,
                    "bbox": self.image_id_to_bbox[image_id],
                    "attr_weight_vec": attr_weight_vec,
                    "attr_binary_vec": attr_binary_vec,
                }
            )

        self.samples = samples

    def _build_attribute_vectors(self, image_id: int):
        """
        Returns:
          attr_prob_dist: FloatTensor [312], normalized weighted present attributes
          attr_binary_vec: FloatTensor [312], 1 for accepted present attrs else 0
        """
        weighted = torch.zeros(self.num_attributes, dtype=torch.float32)
        binary = torch.zeros(self.num_attributes, dtype=torch.float32)

        for attr_id, is_present, certainty_id in self.image_id_to_attr_rows[image_id]:
            if is_present != 1:
                continue

            if self.keep_top2_certainties_only and certainty_id not in (3, 4):
                continue

            idx = attr_id - 1

            if self.use_certainty_weights:
                weight = self.certainty_to_weight.get(certainty_id, 0.0)
            else:
                weight = 1.0

            if weight > 0:
                weighted[idx] = weight
                binary[idx] = 1.0

        total = weighted.sum()
        if total > 0:
            attr_prob_dist = weighted / total
        else:
            attr_prob_dist = weighted.clone()

        return attr_prob_dist, binary

    @staticmethod
    def _crop_with_bbox(img: Image.Image, bbox):
        x, y, w, h = bbox
        left = max(0, int(round(x)))
        top = max(0, int(round(y)))
        right = min(img.width, int(round(x + w)))
        bottom = min(img.height, int(round(y + h)))
        return img.crop((left, top, right, bottom))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]
        img = Image.open(sample["image_path"]).convert("RGB")
        img = self._crop_with_bbox(img, sample["bbox"])

        if self.train:
            img_tensor = self.train_transform(img)
        else:
            img_tensor = self.eval_transform(img)

        return (
            img_tensor,
            sample["attr_weight_vec"],
            sample["attr_binary_vec"],
            sample["image_id"],
        )


def cub_attr_collate_fn(batch):
    images, attr_prob_dists, attr_binary_masks, image_ids = zip(*batch)
    images = torch.stack(images, dim=0)
    attr_prob_dists = torch.stack(attr_prob_dists, dim=0)
    attr_binary_masks = torch.stack(attr_binary_masks, dim=0)
    image_ids = torch.tensor(image_ids, dtype=torch.long)
    return images, attr_prob_dists, attr_binary_masks, image_ids


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cub-root", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--split", type=str, choices=["train", "test"], default="train")
    parser.add_argument("--no-certainty-weights", action="store_true")
    args = parser.parse_args()

    train = args.split == "train"

    dataset = CUBAttributeCertaintyDataset(
        cub_root=args.cub_root,
        train=train,
        image_size=args.image_size,
        use_certainty_weights=not args.no_certainty_weights,
        keep_top2_certainties_only=True,
    )

    print(f"Split: {args.split}")
    print(f"Dataset size: {len(dataset)}")
    print(f"Num attributes: {dataset.num_attributes}")

    if len(dataset) == 0:
        print("Dataset is empty.")
        return

    img, attr_prob_dist, attr_binary_vec, image_id = dataset[0]
    nonzero = torch.nonzero(attr_prob_dist).squeeze(-1).tolist()
    if isinstance(nonzero, int):
        nonzero = [nonzero]

    print("\nSingle sample:")
    print(" image_id:", image_id)
    print(" image shape:", tuple(img.shape))
    print(" attr_prob_dist shape:", tuple(attr_prob_dist.shape))
    print(" attr_binary_vec shape:", tuple(attr_binary_vec.shape))
    print(" nonzero attr ids (0-based):", nonzero[:30])

    if nonzero:
        print(" nonzero attr names:", [dataset.attr_names[i] for i in nonzero[:30]])
        print(" nonzero attr probs:", [float(attr_prob_dist[i]) for i in nonzero[:30]])

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=cub_attr_collate_fn,
    )

    images, attr_prob_dists, attr_binary_masks, image_ids = next(iter(loader))

    print("\nBatch:")
    print(" images shape:", tuple(images.shape))
    print(" attr_prob_dists shape:", tuple(attr_prob_dists.shape))
    print(" attr_binary_masks shape:", tuple(attr_binary_masks.shape))
    print(" image_ids:", image_ids.tolist())


if __name__ == "__main__":
    main()
