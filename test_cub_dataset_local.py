#!/usr/bin/env python3
import os
import csv
import re
import argparse
import random
from collections import Counter, defaultdict

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
from torchvision.transforms import v2


COLORS = {
    "blue",
    "brown",
    "iridescent",
    "purple",
    "rufous",
    "grey",
    "yellow",
    "olive",
    "green",
    "pink",
    "orange",
    "black",
    "white",
    "red",
    "buff",
}

PATTERNS = {"solid", "spotted", "striped", "multi-colored"}

COLOR_PARTS = {
    "wing",
    "upperparts",
    "underparts",
    "back",
    "breast",
    "throat",
    "eye",
    "forehead",
    "nape",
    "belly",
    "leg",
    "bill",
    "crown",
    "primary",
}

PATTERN_PARTS = {"breast", "head", "back", "tail", "belly", "wing"}


def normalize_caption_text(caption: str) -> str:
    text = caption.lower()

    replacements = {
        "eyes": "eye",
        "legs": "leg",
        "wings": "wing",
        "bills": "bill",
        "beak": "bill",
        "beaks": "bill",
        "tails": "tail",
        "breasts": "breast",
        "throats": "throat",
        "foreheads": "forehead",
        "gray": "grey",
        "multi colored": "multi-colored",
    }

    for src, dst in replacements.items():
        text = re.sub(rf"\b{re.escape(src)}\b", dst, text)

    return text


def extract_caption_words(caption: str, vocab_to_idx: dict[str, int]) -> list[int]:
    """
    Strict CUB attribute extractor.

    Only matches explicit local phrases like:
      - 'red bill' -> bill color red
      - 'black eyes' -> eye color black
      - 'spotted head' -> head pattern spotted
      - 'grey back' -> back color grey

    Does NOT use noun+adjective fallback, which causes false positives across commas.
    """
    text = caption.lower()

    replacements = {
        "eyes": "eye",
        "legs": "leg",
        "wings": "wing",
        "bills": "bill",
        "beak": "bill",
        "beaks": "bill",
        "tails": "tail",
        "breasts": "breast",
        "throats": "throat",
        "foreheads": "forehead",
        "gray": "grey",
        "multi colored": "multi-colored",
    }

    for src, dst in replacements.items():
        text = re.sub(rf"\b{re.escape(src)}\b", dst, text)

    tokens = re.findall(r"[a-zA-Z-]+", text)

    colors = {
        "blue",
        "brown",
        "iridescent",
        "purple",
        "rufous",
        "grey",
        "yellow",
        "olive",
        "green",
        "pink",
        "orange",
        "black",
        "white",
        "red",
        "buff",
    }
    patterns = {"solid", "spotted", "striped", "multi-colored"}

    color_parts = {
        "wing",
        "upperparts",
        "underparts",
        "back",
        "breast",
        "throat",
        "eye",
        "forehead",
        "nape",
        "belly",
        "leg",
        "bill",
        "crown",
        "primary",
    }
    pattern_parts = {"breast", "head", "back", "tail", "belly", "wing"}

    matched_phrases = []

    def add_phrase(phrase: str):
        if phrase in vocab_to_idx and phrase not in matched_phrases:
            matched_phrases.append(phrase)

    for i in range(len(tokens) - 1):
        a, b = tokens[i], tokens[i + 1]

        # adjective + noun only
        if a in colors and b in color_parts:
            add_phrase(f"{b} color {a}")

        if a in patterns and b in pattern_parts:
            add_phrase(f"{b} pattern {a}")

    return [vocab_to_idx[p] for p in matched_phrases]


def load_vocab_cache(vocab_cache_path: str, device: str = "cpu"):
    cache = torch.load(vocab_cache_path, map_location="cpu")

    if not isinstance(cache, dict) or len(cache) == 0:
        raise ValueError(f"Invalid or empty vocab cache: {vocab_cache_path}")

    vocab_words = list(cache.keys())
    vocab_to_idx = {word: idx for idx, word in enumerate(vocab_words)}

    embeddings = []
    expected_shape = None

    for word in vocab_words:
        emb = cache[word]
        if not isinstance(emb, torch.Tensor):
            raise TypeError(f"Cache entry for '{word}' is not a tensor")

        emb = emb.float().cpu()

        if emb.ndim != 1:
            raise ValueError(
                f"Cache entry for '{word}' must be 1D, got shape {tuple(emb.shape)}"
            )

        if expected_shape is None:
            expected_shape = emb.shape
        elif emb.shape != expected_shape:
            raise ValueError(
                f"Inconsistent embedding shape for '{word}': got {tuple(emb.shape)}, "
                f"expected {tuple(expected_shape)}"
            )

        embeddings.append(emb)

    noun_embeddings = torch.stack(embeddings, dim=0)
    noun_embeddings = F.normalize(noun_embeddings, dim=-1).to(device)

    return vocab_words, vocab_to_idx, noun_embeddings


class CUBCLIPDataset(Dataset):
    def __init__(
        self,
        csv_path: str,
        vocab_to_idx: dict[str, int],
        seed: int = 42,
        device: str = "cuda",
        train: bool = True,
    ):
        self.csv_path = csv_path
        self.device = torch.device(device)
        self.rng = random.Random(seed)
        self.vocab_to_idx = vocab_to_idx
        self.vocab_size = len(vocab_to_idx)
        self.train = train

        self.train_transform = v2.Compose(
            [
                v2.RandomResizedCrop(
                    size=224,
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
                    (224, 224), interpolation=transforms.InterpolationMode.BICUBIC
                ),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                ),
            ]
        )

        captions_per_image = defaultdict(list)

        print(f"Reading CSV: {csv_path}")
        with open(csv_path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            expected = {"filepath", "caption"}
            if reader.fieldnames is None:
                raise ValueError(f"CSV has no header: {csv_path}")
            missing = expected - set(reader.fieldnames)
            if missing:
                raise ValueError(
                    f"CSV must contain columns {sorted(expected)}, missing: {sorted(missing)}"
                )

            for row in reader:
                im_path = row["filepath"].strip()
                caption = row["caption"].strip()

                if not im_path or not caption:
                    continue

                captions_per_image[im_path].append(caption)

        print("Building samples")
        samples = []

        for im_path, captions in captions_per_image.items():
            if not os.path.isfile(im_path):
                print(f"Skipping missing image: {im_path}")
                continue

            counts = Counter()
            total_valid_words = 0

            for caption in captions:
                attr_idxs = extract_caption_words(caption, self.vocab_to_idx)
                for idx in attr_idxs:
                    counts[idx] += 1
                total_valid_words += len(attr_idxs)

            prob_dist = torch.zeros(self.vocab_size, dtype=torch.float32)

            if total_valid_words > 0:
                for idx, cnt in counts.items():
                    prob_dist[idx] = cnt / total_valid_words

            samples.append((im_path, captions, prob_dist))

        self.samples = samples
        print(f"Done computing frequency. Total samples: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index: int):
        im_path, captions, prob_dist = self.samples[index]

        img = Image.open(im_path).convert("RGB")
        if self.train:
            img_tensor = self.train_transform(img)
        else:
            img_tensor = self.eval_transform(img)

        return img_tensor, captions, prob_dist, index


def cub_clip_collate_fn(batch):
    images, captions, prob_dists, indices = zip(*batch)
    images = torch.stack(images, dim=0)
    prob_dists = torch.stack(prob_dists, dim=0)
    indices = torch.tensor(indices, dtype=torch.long)
    return images, list(captions), prob_dists, indices


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv-path",
        type=str,
        required=True,
        help="Path to CUB CSV with columns: filepath,caption",
    )
    parser.add_argument(
        "--vocab-cache-path",
        type=str,
        required=True,
        help="Path to vocab cache .pt saved as {word: embedding_tensor}",
    )
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--train", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    vocab_words, vocab_to_idx, noun_embeddings = load_vocab_cache(
        args.vocab_cache_path,
        device=args.device,
    )

    print(f"Loaded vocab cache: {args.vocab_cache_path}")
    print(f"Vocab size: {len(vocab_words)}")
    print(f"Noun embeddings shape: {tuple(noun_embeddings.shape)}")

    dataset = CUBCLIPDataset(
        csv_path=args.csv_path,
        vocab_to_idx=vocab_to_idx,
        train=args.train,
        device=args.device,
        seed=args.seed,
    )

    print(f"Dataset size: {len(dataset)}")

    if len(dataset) == 0:
        print("Dataset is empty. Check CSV paths and image files.")
        return

    sample = dataset[0]
    img_tensor, captions, prob_dist, index = sample

    nonzero = torch.nonzero(prob_dist).squeeze(-1).tolist()
    if isinstance(nonzero, int):
        nonzero = [nonzero]

    print("\nSingle sample:")
    print(" index:", index)
    print(" image shape:", tuple(img_tensor.shape))
    print(" num captions:", len(captions))
    print(" prob_dist shape:", tuple(prob_dist.shape))
    print(" nonzero vocab indices:", nonzero[:50])

    if nonzero:
        print(" nonzero vocab words:", [vocab_words[i] for i in nonzero])
        print(" nonzero probs:", [float(prob_dist[i]) for i in nonzero])

    print("\nAll captions for this sample:")
    for i, cap in enumerate(captions, start=1):
        matched_idxs = extract_caption_words(cap, vocab_to_idx)
        matched_words = [vocab_words[j] for j in matched_idxs]
        print(f" {i}. {cap}")
        print(f"    matched: {matched_words}")

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=cub_clip_collate_fn,
    )

    images, captions_batch, prob_dists, indices = next(iter(loader))

    print("\nBatch:")
    print(" images shape:", tuple(images.shape))
    print(" batch captions len:", len(captions_batch))
    print(" prob_dists shape:", tuple(prob_dists.shape))
    print(" indices:", indices.tolist())

    print("\nDone.")


if __name__ == "__main__":
    main()
