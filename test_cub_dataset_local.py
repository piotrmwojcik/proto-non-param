#!/usr/bin/env python3
import csv
import argparse
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms


def class_idx_from_path(image_path: str) -> int:
    class_dir = Path(image_path).parent.name
    class_id = int(class_dir.split(".")[0])
    return class_id - 1


def extract_tokens(caption: str, vocab_to_idx: dict[str, int]) -> list[int]:
    caption_lower = caption.lower().strip()

    if ", with " not in caption_lower:
        return []

    attr_text = caption_lower.split(", with ", 1)[1]
    attr_text = attr_text.rstrip(".")

    token_ids = []

    for part in attr_text.split(","):
        attr = " ".join(part.strip().split())

        if not attr:
            continue

        if attr in vocab_to_idx:
            token_ids.append(vocab_to_idx[attr])
        else:
            print(f"Missing vocab token: {attr}")

    return token_ids


def load_vocab_cache(vocab_cache_path: str) -> tuple[list[str], dict[str, int]]:
    cache = torch.load(vocab_cache_path, map_location="cpu")

    if not isinstance(cache, dict) or len(cache) == 0:
        raise ValueError(f"Invalid vocab cache: {vocab_cache_path}")

    vocab_words = list(cache.keys())
    vocab_to_idx = {word.lower(): i for i, word in enumerate(vocab_words)}

    return vocab_words, vocab_to_idx


def decode_tokens(tokens: torch.Tensor, vocab_words: list[str]) -> list[str]:
    names = []

    for idx in tokens.tolist():
        if 0 <= idx < len(vocab_words):
            names.append(vocab_words[idx])
        else:
            names.append(f"<INVALID:{idx}>")

    return names


class CUBTokenDataset(Dataset):
    def __init__(
        self,
        csv_path: str,
        vocab_to_idx: dict[str, int],
        vocab_words: list[str] | None = None,
        train: bool = True,
    ):
        self.vocab_to_idx = vocab_to_idx
        self.vocab_words = vocab_words

        if train:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                ),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                ),
            ])

        self.samples = []

        with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)

            if reader.fieldnames is None:
                raise ValueError("CSV has no header row")

            reader.fieldnames = [
                name.strip().replace("\ufeff", "")
                for name in reader.fieldnames
            ]

            required = {"filepath", "caption"}
            missing = required - set(reader.fieldnames)
            if missing:
                raise ValueError(
                    f"CSV missing columns: {sorted(missing)}. "
                    f"Found columns: {reader.fieldnames}"
                )

            for image_id, row in enumerate(reader):
                full_path = Path(row["filepath"].strip())
                caption = row["caption"].strip()

                if not full_path.is_file():
                    print(f"Skipping missing image: {full_path}")
                    continue

                token_ids = extract_tokens(caption, vocab_to_idx)
                class_idx = class_idx_from_path(str(full_path))

                token_names = None
                if vocab_words is not None:
                    token_names = [
                        vocab_words[token_id]
                        if 0 <= token_id < len(vocab_words)
                        else f"<INVALID:{token_id}>"
                        for token_id in token_ids
                    ]

                self.samples.append({
                    "image_id": image_id,
                    "image_path": str(full_path),
                    "caption": caption,
                    "tokens": token_ids,
                    "token_names": token_names,
                    "class_idx": class_idx,
                })

        print(f"Loaded samples: {len(self.samples)}")

        if len(self.samples) == 0:
            raise RuntimeError("No valid samples loaded")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]

        image = Image.open(sample["image_path"]).convert("RGB")
        image = self.transform(image)

        tokens = torch.tensor(sample["tokens"], dtype=torch.long)
        class_idx = torch.tensor(sample["class_idx"], dtype=torch.long)
        image_id = torch.tensor(sample["image_id"], dtype=torch.long)

        return (
            image,
            tokens,
            class_idx,
            image_id,
            sample["caption"],
            sample["token_names"],
        )


def cub_token_collate_fn(batch):
    images, tokens, class_idxs, image_ids, captions, token_names = zip(*batch)

    return {
        "images": torch.stack(images, dim=0),
        "tokens": list(tokens),
        "class_idxs": torch.stack(class_idxs, dim=0),
        "image_ids": torch.stack(image_ids, dim=0),
        "captions": list(captions),
        "token_names": list(token_names),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv-path", required=True)
    parser.add_argument("--vocab-cache-path", required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--train", action="store_true")
    args = parser.parse_args()

    vocab_words, vocab_to_idx = load_vocab_cache(args.vocab_cache_path)

    print(f"Loaded vocab size: {len(vocab_words)}")

    dataset = CUBTokenDataset(
        csv_path=args.csv_path,
        vocab_to_idx=vocab_to_idx,
        vocab_words=vocab_words,
        train=args.train,
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=args.train,
        num_workers=args.num_workers,
        collate_fn=cub_token_collate_fn,
    )

    batch = next(iter(loader))

    print("Batch images:", batch["images"].shape)
    print("Batch class idxs:", batch["class_idxs"])
    print("Batch image ids:", batch["image_ids"])
    print("Batch tokens:", batch["tokens"])
    print("Batch captions:", batch["captions"])

    print("\nDecoded token names:")
    for i, token_ids in enumerate(batch["tokens"]):
        token_names = decode_tokens(token_ids, vocab_words)

        print(f"\nSample {i}")
        print("Caption:", batch["captions"][i])
        print("Token ids:", token_ids.tolist())
        print("Token names:", token_names)


if __name__ == "__main__":
    main()