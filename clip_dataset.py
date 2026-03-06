import os
import random
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
import open_clip


class CocoCLIPDataset(Dataset):
    def __init__(
        self,
        csv_path: str,
        coco_root: str,
        split: str = "train",
        val_ratio: float = 0.1,
        seed: int = 42,
        device: str = "cpu",
    ):
        self.csv_path = csv_path
        self.coco_root = coco_root
        self.device = device

        df = pd.read_csv(csv_path)

        # Build (image_path, caption) pairs
        samples = []
        for _, row in df.iterrows():
            coco_id = int(row["coco_id"])
            caption = row["prompt"]

            im_path = self._find_image_path(coco_id)
            if im_path is not None:
                samples.append((im_path, caption))

        rng = random.Random(seed)
        rng.shuffle(samples)

        split_idx = int(len(samples) * (1 - val_ratio))
        if split == "train":
            self.samples = samples[:split_idx]
        elif split == "val":
            self.samples = samples[split_idx:]
        else:
            raise ValueError("split must be 'train' or 'val'")

        model, preprocess, _ = open_clip.create_model_and_transforms(
            "ViT-B-32",
            pretrained="openai",
        )
        self.model = model.eval().to(device)
        self.preprocess = preprocess
        self.tokenizer = open_clip.get_tokenizer("ViT-B-32")

        self.transform = None
        self.target_transform = None

    def _find_image_path(self, coco_id: int):
        filename = f"{coco_id:012d}.jpg"

        candidates = [
            os.path.join(self.coco_root, "val2014"),
            self.coco_root,
        ]

        for path in candidates:
            if os.path.basename(path) == filename and os.path.isfile(path):
                return path

            candidate_file = os.path.join(path, filename)
            if os.path.isfile(candidate_file):
                return candidate_file

        return None

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        im_path, label = self.samples[index]

        img = Image.open(im_path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            label = self.target_transform(label)

        img_tensor = self.preprocess(img).unsqueeze(0).to(self.device)
        text_tokens = self.tokenizer([label]).to(self.device)

        with torch.no_grad():
            img_feat = self.model.encode_image(img_tensor)
            txt_feat = self.model.encode_text(text_tokens)

            img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
            txt_feat = txt_feat / txt_feat.norm(dim=-1, keepdim=True)

        return img_feat.squeeze(0), txt_feat.squeeze(0), index

train_dataset = CocoCLIPDataset(
    csv_path="assets/coco_30k.csv",
    coco_root="/data/pwojcik/UnGuide/coco30_bck/",
    split="train",
    val_ratio=0.1,
    device="cuda",
)

val_dataset = CocoCLIPDataset(
    csv_path="assets/coco_30k.csv",
    coco_root="/data/pwojcik/UnGuide/coco30_bck/",
    split="val",
    val_ratio=0.1,
    device="cuda",
)

img_emb, txt_emb, idx = train_dataset[0]
print(img_emb.shape)  # usually [512]
print(txt_emb.shape)  # usually [512]
print(idx)