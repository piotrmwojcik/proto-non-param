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
        vocab_cache_path: str = "vocab/mscoco_nouns_clip_cache.pt",
        split: str = "train",
        val_ratio: float = 0.1,
        seed: int = 42,
        device: str = "cuda",
        model_name: str = "ViT-B-32",
        pretrained: str = "openai",
    ):
        self.csv_path = csv_path
        self.coco_root = coco_root
        self.device = torch.device(device)

        # -------------------------------------------------
        # Build filename → full path
        # -------------------------------------------------
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

        # -------------------------------------------------
        # CLIP model
        # -------------------------------------------------
        model, preprocess, _ = open_clip.create_model_and_transforms(
            model_name,
            pretrained=pretrained,
        )

        self.model = model.eval()

        self.model = self.model.to(self.device)
        for p in self.model.parameters():
            p.requires_grad = False

        self.preprocess = preprocess
        self.tokenizer = open_clip.get_tokenizer(model_name)

        self.transform = None
        self.target_transform = None

        # -------------------------------------------------
        # Load cached noun CLIP embeddings
        # -------------------------------------------------
        cache = torch.load(vocab_cache_path, map_location="cpu")

        self.vocab_words = list(cache.keys())

        noun_embs = torch.stack(
            [cache[w] for w in self.vocab_words],
            dim=0
        )  # [V, 512]

        noun_embs = noun_embs / noun_embs.norm(dim=-1, keepdim=True)

        # frozen buffer
        self.noun_embeddings = noun_embs  # [V, 512]
        self.noun_embeddings = self.noun_embeddings.to(self.device)

    def _find_image_path(self, coco_id: int):
        filename = f"COCO_val2014_{coco_id:012d}.jpg"
        return self.file_index.get(filename)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index: int):
        im_path, caption = self.samples[index]

        img = Image.open(im_path).convert("RGB")

        if self.transform is not None:
            img_tensor = self.transform(img)
        else:
            img_tensor = self.preprocess(img)

        return img_tensor, caption, index