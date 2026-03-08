import os
import random
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
import torch.nn.functional as F
import open_clip


class CocoCLIPDataset(Dataset):
    def __init__(
        self,
        csv_path: str,
        coco_root: str,
        split: str = "train",
        val_ratio: float = 0.1,
        seed: int = 42,
        model_name: str = "ViT-B-32",
        pretrained: str = "openai",
    ):
        self.csv_path = csv_path
        self.coco_root = coco_root

        # Build filename -> full path once
        val2014_dir = os.path.join(self.coco_root, "val2014")
        self.file_index = {
            fname: os.path.join(val2014_dir, fname)
            for fname in os.listdir(val2014_dir)
            if fname.endswith(".jpg")
        }

        df = pd.read_csv(csv_path)

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
            model_name,
            pretrained=pretrained,
        )
        self.model = model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

        self.preprocess = preprocess
        self.tokenizer = open_clip.get_tokenizer(model_name)

        self.transform = None
        self.target_transform = None

    def _find_image_path(self, coco_id: int):
        filename = f"COCO_val2014_{coco_id:012d}.jpg"
        return self.file_index.get(filename)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        im_path, caption = self.samples[index]

        img = Image.open(im_path).convert("RGB")

        # Return image tensor in the format expected by the proto network / CLIP backbone
        if self.transform is not None:
            img_tensor = self.transform(img)
        else:
            img_tensor = self.preprocess(img)   # [3, H, W], CLIP-normalized

        if self.target_transform is not None:
            caption = self.target_transform(caption)

        text_tokens = self.tokenizer([caption])   # [1, context_len]

        with torch.no_grad():
            txt_feat = self.model.encode_text(text_tokens)   # [1, D]
            txt_feat = txt_feat / txt_feat.norm(dim=-1, keepdim=True)

        return img_tensor, txt_feat.squeeze(0), index

    train_dataset = CocoCLIPDataset(
        csv_path="assets/coco_30k.csv",
        coco_root="/data/pwojcik/UnGuide/coco30_bck/",
        split="train",
        val_ratio=0.1,
    )

    val_dataset = CocoCLIPDataset(
        csv_path="assets/coco_30k.csv",
        coco_root="/data/pwojcik/UnGuide/coco30_bck/",
        split="val",
        val_ratio=0.1,
    )

    num_samples = 5

    for i in range(num_samples):
        img_tensor, txt_emb, idx = train_dataset[i]

        print(f"sample {i} (idx={idx})")
        print("image tensor shape:", img_tensor.shape)  # expected [3, 224, 224]
        print("text emb shape:", txt_emb.shape)  # expected [512] for ViT-B-32
        print("image min/max:", img_tensor.min().item(), img_tensor.max().item())
        print()