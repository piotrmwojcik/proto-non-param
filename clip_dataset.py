import os
import json
import random
import torch
from torch.utils.data import Dataset
from PIL import Image
import open_clip


class CocoCLIPDataset(Dataset):
    def __init__(
        self,
        annotations_json: str,
        coco_root: str,
        vocab_cache_path: str = "vocab/laion_clip_cache.pt",
        seed: int = 42,
        device: str = "cuda",
        model_name: str = "ViT-B-32",
        pretrained: str = "openai",
    ):
        self.annotations_json = annotations_json
        self.coco_root = coco_root
        self.device = torch.device(device)
        self.rng = random.Random(seed)

        with open(annotations_json, "r") as f:
            coco_data = json.load(f)

        # image_id -> file_name
        image_id_to_file = {
            img["id"]: img["file_name"]
            for img in coco_data["images"]
        }

        # image_id -> list of captions
        captions_per_image = {}
        for ann in coco_data["annotations"]:
            image_id = int(ann["image_id"])
            caption = ann["caption"]

            if image_id not in captions_per_image:
                captions_per_image[image_id] = []
            captions_per_image[image_id].append(caption)

        samples = []
        for image_id, captions in captions_per_image.items():
            file_name = image_id_to_file.get(image_id)
            if file_name is None:
                continue

            im_path = self._find_image_path(file_name)
            if im_path is not None and len(captions) > 0:
                samples.append((im_path, captions))

        self.samples = samples

        model, preprocess, _ = open_clip.create_model_and_transforms(
            model_name,
            pretrained=pretrained,
        )

        self.model = model.eval().to(self.device)
        for p in self.model.parameters():
            p.requires_grad = False

        self.preprocess = preprocess
        self.tokenizer = open_clip.get_tokenizer(model_name)

        self.transform = None
        self.target_transform = None

    def _find_image_path(self, file_name: str):
        candidates = [
            os.path.join(self.coco_root, "train2014", file_name),
            os.path.join(self.coco_root, "val2014", file_name),
        ]
        for path in candidates:
            if os.path.isfile(path):
                return path
        return None

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index: int):
        im_path, captions = self.samples[index]

        img = Image.open(im_path).convert("RGB")
        img_tensor = self.transform(img) if self.transform is not None else self.preprocess(img)

        caption = self.rng.choice(captions)

        return img_tensor, caption, index