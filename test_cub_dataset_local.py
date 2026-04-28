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

TAIL_SHAPES = {
    "forked": "tail shape forked tail",
    "rounded": "tail shape rounded tail",
    "notched": "tail shape notched tail",
    "fan-shaped": "tail shape fan-shaped tail",
    "pointed": "tail shape pointed tail",
    "squared": "tail shape squared tail",
}

BILL_SHAPES = {
    "curved": "bill shape curved (up or down)",
    "dagger": "bill shape dagger",
    "hooked": "bill shape hooked",
    "needle": "bill shape needle",
    "spatulate": "bill shape spatulate",
    "cone": "bill shape cone",
    "specialized": "bill shape specialized",
}

SIZE_PHRASES = {
    "large": "size large (16 - 32 in)",
    "small": "size small (5 - 9 in)",
    "very large": "size very large (32 - 72 in)",
    "medium": "size medium (9 - 16 in)",
    "very small": "size very small (3 - 5 in)",
}

SHAPE_PHRASES = {
    "upright-perching water-like": "shape upright-perching water-like",
    "chicken-like-marsh": "shape chicken-like-marsh",
    "long-legged-like": "shape long-legged-like",
    "duck-like": "shape duck-like",
    "owl-like": "shape owl-like",
    "gull-like": "shape gull-like",
    "hummingbird-like": "shape hummingbird-like",
    "pigeon-like": "shape pigeon-like",
    "tree-clinging-like": "shape tree-clinging-like",
    "hawk-like": "shape hawk-like",
    "sandpiper-like": "shape sandpiper-like",
    "upland-ground-like": "shape upland-ground-like",
    "swallow-like": "shape swallow-like",
    "perching-like": "shape perching-like",
}

BILL_LENGTH_PHRASES = {
    "about the same as head": "bill length about the same as head",
    "longer than head": "bill length longer than head",
    "shorter than head": "bill length shorter than head",
}


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

    text = re.sub(r"\s+", " ", text).strip()
    return text


def extract_caption_words(caption: str, vocab_to_idx: dict[str, int]) -> list[int]:
    """
    Strict CUB attribute extractor.

    Supports:
    - color + part: "red bill" -> bill color red
    - pattern + part: "spotted head" -> head pattern spotted
    - tail shape: "notched tail" -> tail shape notched tail
    - size: "small (5 - 9 in) size" -> size small (5 - 9 in)
    - shape: "perching-like shape" -> shape perching-like
    - bill shape: "hooked bill" -> bill shape hooked
    - bill length: "bill longer than head" -> bill length longer than head
    """
    text = normalize_caption_text(caption)
    tokens = re.findall(r"[a-zA-Z-]+", text)

    matched_phrases: list[str] = []

    def add_phrase(phrase: str):
        if phrase in vocab_to_idx and phrase not in matched_phrases:
            matched_phrases.append(phrase)

    # 1) local adjective+noun rules only
    for i in range(len(tokens) - 1):
        a, b = tokens[i], tokens[i + 1]

        if a in COLORS and b in COLOR_PARTS:
            add_phrase(f"{b} color {a}")

        if a in PATTERNS and b in PATTERN_PARTS:
            add_phrase(f"{b} pattern {a}")

        if a in TAIL_SHAPES and b == "tail":
            add_phrase(TAIL_SHAPES[a])

        if a in BILL_SHAPES and b == "bill":
            add_phrase(BILL_SHAPES[a])

    # 2) size phrases
    for key, phrase in SIZE_PHRASES.items():
        if re.search(rf"\b{re.escape(key)}\b.*\bsize\b", text):
            add_phrase(phrase)

    # 3) shape phrases
    for key, phrase in SHAPE_PHRASES.items():
        if re.search(rf"\b{re.escape(key)}\b.*\bshape\b", text):
            add_phrase(phrase)

    # 4) bill length phrases
    for key, phrase in BILL_LENGTH_PHRASES.items():
        if re.search(rf"\bbill\b.*\b{re.escape(key)}\b", text):
            add_phrase(phrase)

    # 5) special exact phrases that may appear literally
    special_literal_phrases = [
        "bill shape hooked seabird",
        "bill shape all-purpose",
    ]
    for phrase in special_literal_phrases:
        if phrase in text:
            add_phrase(phrase)

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


import os
import random
import csv
from collections import defaultdict, Counter

import torch
from torch.utils.data import Dataset
from PIL import Image

from torchvision import transforms
from torchvision.transforms import v2


def class_idx_from_path(im_path: str) -> int:
    """
    Example:
    /.../images/001.Black_footed_Albatross/img.jpg -> 0
    /.../cub_part_crops_clip/002.Laysan_Albatross/img.jpg -> 1
    """
    class_dir = os.path.basename(os.path.dirname(im_path))
    class_id = int(class_dir.split(".")[0])
    return class_id - 1


def extract_caption_words(caption: str, vocab_to_idx: dict[str, int]) -> list[int]:
    """
    Simple word extractor.
    Replace this with your existing implementation if you already have one.
    """
    caption = caption.lower()
    tokens = (
        caption.replace(",", " ")
        .replace(".", " ")
        .replace("-", " ")
        .replace("_", " ")
        .split()
    )

    return [vocab_to_idx[t] for t in tokens if t in vocab_to_idx]


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
                    (224, 224),
                    interpolation=transforms.InterpolationMode.BICUBIC,
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
                    f"CSV must contain columns {sorted(expected)}, "
                    f"missing: {sorted(missing)}"
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

            if total_valid_words == 0:
                print(f"Skipping no-attribute sample: {im_path}")
                continue

            prob_dist = torch.zeros(self.vocab_size, dtype=torch.float32)

            for idx, cnt in counts.items():
                prob_dist[idx] = cnt / total_valid_words

            try:
                class_idx = class_idx_from_path(im_path)
            except Exception as e:
                print(f"Skipping bad class path: {im_path} ({e})")
                continue

            samples.append(
                {
                    "filepath": im_path,
                    "captions": captions,
                    "prob_dist": prob_dist,
                    "class_idx": class_idx,
                }
            )

        self.samples = samples

        print(f"Done computing frequency.")
        print(f"Total samples: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]

        im_path = sample["filepath"]
        captions = sample["captions"]
        prob_dist = sample["prob_dist"]
        class_idx = sample["class_idx"]

        img = Image.open(im_path).convert("RGB")

        if self.train:
            img_tensor = self.train_transform(img)
        else:
            img_tensor = self.eval_transform(img)

        return img_tensor, captions, prob_dist, class_idx, index


def cub_clip_collate_fn(batch):
    images, captions, prob_dists, class_idxs, indices = zip(*batch)

    images = torch.stack(images, dim=0)
    prob_dists = torch.stack(prob_dists, dim=0)
    class_idxs = torch.tensor(class_idxs, dtype=torch.long)
    indices = torch.tensor(indices, dtype=torch.long)

    return images, list(captions), prob_dists, class_idxs, indices


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
