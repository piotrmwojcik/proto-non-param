import os
import json
import random
from collections import Counter

import torch
from torch.utils.data import Dataset
from PIL import Image
import open_clip


import nltk
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

COMMON_VERBS = {
    "be", "is", "am", "are", "was", "were", "been", "being",
    "have", "has", "had", "having",
    "do", "does", "did", "doing"
}


def extract_caption_words(caption: str, vocab_to_idx: dict[str, int]):
    """
    Extract nouns, verbs, adjectives, and adverbs from caption using NLTK
    and map them to vocab indices, excluding very common verbs.
    """

    tokens = nltk.word_tokenize(caption.lower())
    pos_tags = nltk.pos_tag(tokens)

    word_idxs = []
    seen = set()

    for word, pos in pos_tags:

        if pos.startswith(("NN", "VB", "JJ")):

            if pos.startswith("NN"):
                lemma = lemmatizer.lemmatize(word, pos="n")

            elif pos.startswith("VB"):
                lemma = lemmatizer.lemmatize(word, pos="v")

                if lemma in COMMON_VERBS:
                    continue

            else:  # adjectives (JJ)
                lemma = lemmatizer.lemmatize(word, pos="a")

            if lemma in vocab_to_idx and lemma not in seen:
                word_idxs.append(vocab_to_idx[lemma])
                seen.add(lemma)

    return word_idxs


class CocoCLIPDataset(Dataset):
    def __init__(
        self,
        annotations_json: str,
        coco_root: str,
        vocab_to_idx: dict[str, int],   # ← add this
        seed: int = 42,
        device: str = "cuda",
    ):
        self.annotations_json = annotations_json
        self.coco_root = coco_root
        self.device = torch.device(device)
        self.rng = random.Random(seed)
        self.vocab_to_idx = vocab_to_idx
        self.vocab_size = len(vocab_to_idx)

        with open(annotations_json, "r") as f:
            coco_data = json.load(f)

        # image_id -> file_name
        image_id_to_file = {
            img["id"]: img["file_name"]
            for img in coco_data["images"]
        }

        # image_id -> list of captions
        captions_per_image = {}
        print('Gather annotations')
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
            if im_path is None or len(captions) == 0:
                continue

            # -----------------------------------
            # Build probability distribution from all captions
            # -----------------------------------
            counts = Counter()
            total_valid_words = 0

            for caption in captions:
                word_idxs = extract_caption_words(caption, self.vocab_to_idx)

                for idx in word_idxs:
                    counts[idx] += 1
                total_valid_words += len(word_idxs)

            prob_dist = torch.zeros(self.vocab_size, dtype=torch.float32)

            if total_valid_words > 0:
                for idx, cnt in counts.items():
                    prob_dist[idx] = cnt / total_valid_words

            samples.append((im_path, captions, prob_dist))

        self.samples = samples
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
        im_path, captions, prob_dist = self.samples[index]

        img = Image.open(im_path).convert("RGB")
        img_tensor = self.transform(img) if self.transform is not None else self.preprocess(img)

        # keep one random caption for logging/debugging
        caption = self.rng.choice(captions)

        return img_tensor, caption, prob_dist, index