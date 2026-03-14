import os
import re
import json
import random
import hashlib
from collections import Counter, defaultdict

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
import open_clip

import nltk
from nltk.stem import WordNetLemmatizer

# -----------------------------
# NLTK setup
# -----------------------------
nltk.download("punkt")
nltk.download("averaged_perceptron_tagger")
nltk.download("wordnet")
nltk.download("omw-1.4")

lemmatizer = WordNetLemmatizer()

COMMON_VERBS = {
    "be", "is", "am", "are", "was", "were", "been", "being",
    "have", "has", "had", "having",
    "do", "does", "did", "doing"
}


# -----------------------------
# Word extraction
# -----------------------------
def extract_caption_words(caption: str, vocab_to_idx: dict[str, int]):
    """
    Extract nouns, verbs, and adjectives from caption using NLTK
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
            else:
                lemma = lemmatizer.lemmatize(word, pos="a")

            if lemma in vocab_to_idx and lemma not in seen:
                word_idxs.append(vocab_to_idx[lemma])
                seen.add(lemma)

    return word_idxs


def extract_caption_lemmas(caption: str, vocab_to_idx: dict[str, int] | None = None):
    """
    Same logic as extract_caption_words, but returns lemmas instead of indices.
    If vocab_to_idx is provided, only keeps lemmas present in vocab.
    """
    tokens = nltk.word_tokenize(caption.lower())
    pos_tags = nltk.pos_tag(tokens)

    lemmas = []
    seen = set()

    for word, pos in pos_tags:
        if pos.startswith(("NN", "VB", "JJ")):
            if pos.startswith("NN"):
                lemma = lemmatizer.lemmatize(word, pos="n")
            elif pos.startswith("VB"):
                lemma = lemmatizer.lemmatize(word, pos="v")
                if lemma in COMMON_VERBS:
                    continue
            else:
                lemma = lemmatizer.lemmatize(word, pos="a")

            if vocab_to_idx is not None and lemma not in vocab_to_idx:
                continue

            if lemma not in seen:
                lemmas.append(lemma)
                seen.add(lemma)

    return lemmas


# -----------------------------
# Cache utils
# -----------------------------
def _make_cache_path(annotation_file: str, vocab_to_idx: dict, cache_dir: str = None):
    if cache_dir is None:
        cache_dir = os.path.join(os.path.dirname(annotation_file), ".cache")
    os.makedirs(cache_dir, exist_ok=True)

    vocab_hash = hashlib.md5(
        json.dumps(vocab_to_idx, sort_keys=True).encode("utf-8")
    ).hexdigest()[:12]

    ann_base = os.path.splitext(os.path.basename(annotation_file))[0]
    return os.path.join(cache_dir, f"{ann_base}_samples_{vocab_hash}.pt")


# -----------------------------
# Dataset
# -----------------------------
class CocoCLIPDataset(Dataset):
    def __init__(
        self,
        annotations_json: str,
        coco_root: str,
        vocab_to_idx: dict[str, int],
        seed: int = 42,
        device: str = "cuda",
        cache_dir: str = None,
        use_cache: bool = True,
        train: bool = False,
    ):
        self.annotations_json = annotations_json
        self.coco_root = coco_root
        self.device = torch.device(device)
        self.rng = random.Random(seed)
        self.vocab_to_idx = vocab_to_idx
        self.vocab_size = len(vocab_to_idx)
        self.train = train

        self.train_transform = transforms.Compose([
            transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ])

        self.eval_transform = transforms.Compose([
            transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ])

        cache_path = _make_cache_path(annotations_json, vocab_to_idx, cache_dir)

        if use_cache and os.path.exists(cache_path):
            print(f"Loading dataset cache from: {cache_path}")
            cached = torch.load(cache_path, map_location="cpu")
            self.samples = cached["samples"]
            print(f"Loaded {len(self.samples)} cached samples")
            return

        with open(annotations_json, "r") as f:
            coco_data = json.load(f)

        image_id_to_file = {
            img["id"]: img["file_name"]
            for img in coco_data["images"]
        }

        captions_per_image = {}
        print("Gather annotations")
        for ann in coco_data["annotations"]:
            image_id = int(ann["image_id"])
            caption = ann["caption"]
            captions_per_image.setdefault(image_id, []).append(caption)
        print("Done gather annotations")

        samples = []
        print("Building samples")
        for image_id, captions in captions_per_image.items():
            file_name = image_id_to_file.get(image_id)
            if file_name is None:
                continue

            im_path = self._find_image_path(file_name)
            if im_path is None or len(captions) == 0:
                continue

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
        print(f"Done computing frequency. Total samples: {len(self.samples)}")

        if use_cache:
            torch.save({"samples": self.samples}, cache_path)
            print(f"Saved dataset cache to: {cache_path}")

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
        img_tensor = self.train_transform(img) if self.train else self.eval_transform(img)

        return img_tensor, captions, prob_dist, index


# -----------------------------
# CLIP helpers
# -----------------------------
@torch.no_grad()
def encode_image_pil(model, preprocess, image_pil, device):
    image = preprocess(image_pil).unsqueeze(0).to(device)
    image_features = model.encode_image(image)
    image_features = F.normalize(image_features, dim=-1)
    return image_features


@torch.no_grad()
def encode_texts(model, tokenizer, texts, device):
    text_tokens = tokenizer(texts).to(device)
    text_features = model.encode_text(text_tokens)
    text_features = F.normalize(text_features, dim=-1)
    return text_features


@torch.no_grad()
def clip_similarity(model, preprocess, tokenizer, image_pil, texts, device):
    image_features = encode_image_pil(model, preprocess, image_pil, device)
    text_features = encode_texts(model, tokenizer, texts, device)
    sims = image_features @ text_features.T
    return sims.squeeze(0).cpu()


# -----------------------------
# Caption ablation
# -----------------------------
def remove_word_from_caption(caption: str, target_lemma: str):
    """
    Remove the first token in the caption whose lemmatized form matches target_lemma.
    Returns new caption and whether removal happened.
    """
    tokens = nltk.word_tokenize(caption)
    pos_tags = nltk.pos_tag([t.lower() for t in tokens])

    new_tokens = []
    removed = False

    for original_token, (_, pos) in zip(tokens, pos_tags):
        lower = original_token.lower()

        if pos.startswith("NN"):
            lemma = lemmatizer.lemmatize(lower, pos="n")
        elif pos.startswith("VB"):
            lemma = lemmatizer.lemmatize(lower, pos="v")
        elif pos.startswith("JJ"):
            lemma = lemmatizer.lemmatize(lower, pos="a")
        else:
            lemma = lower

        if not removed and lemma == target_lemma:
            removed = True
            continue

        new_tokens.append(original_token)

    # lightweight detokenization
    text = " ".join(new_tokens)
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    return text.strip(), removed


# -----------------------------
# Experiment
# -----------------------------
def run_ablation_experiment(
    annotations_json: str,
    coco_root: str,
    sample_size: int = 50,
    seed: int = 42,
    device: str = "cuda",
    clip_model_name: str = "ViT-B-32",
    clip_pretrained: str = "openai",
):
    rng = random.Random(seed)
    device = torch.device(device if torch.cuda.is_available() else "cpu")

    # Build vocabulary from COCO captions
    print("Building vocabulary...")
    with open(annotations_json, "r") as f:
        coco_data = json.load(f)

    all_words = set()
    for ann in coco_data["annotations"]:
        tokens = word_tokenize(ann["caption"].lower())
        for t in tokens:
            if t.isalpha():
                all_words.add(t)

    word_list = sorted(all_words)
    vocab_to_idx = {w: i for i, w in enumerate(word_list)}
    print(f"Vocabulary size: {len(vocab_to_idx)}")

    # Dataset
    dataset = CocoCLIPDataset(
        annotations_json=annotations_json,
        coco_root=coco_root,
        vocab_to_idx=vocab_to_idx,
        seed=seed,
        device=str(device),
        use_cache=True,
        train=False,
    )

    # CLIP
    print("Loading CLIP...")
    model, preprocess, _ = open_clip.create_model_and_transforms(
        clip_model_name,
        pretrained=clip_pretrained,
    )
    model = model.eval().to(device)
    tokenizer = open_clip.get_tokenizer(clip_model_name)

    # Sample indices
    sample_indices = list(range(len(dataset)))
    rng.shuffle(sample_indices)
    sample_indices = sample_indices[:sample_size]

    print(f"Running ablation on {len(sample_indices)} images...")

    per_word_scores = defaultdict(list)
    per_example_results = []

    for sample_num, idx in enumerate(sample_indices, start=1):
        im_path, captions, prob_dist = dataset.samples[idx]
        image_pil = Image.open(im_path).convert("RGB")

        # compute per-caption ablations
        for caption in captions:
            candidate_words = extract_caption_lemmas(caption, vocab_to_idx=vocab_to_idx)
            if not candidate_words:
                continue

            texts = [caption]
            ablated_captions = []
            ablated_words = []

            for w in candidate_words:
                caption_removed, removed = remove_word_from_caption(caption, w)
                if removed and caption_removed:
                    texts.append(caption_removed)
                    ablated_captions.append(caption_removed)
                    ablated_words.append(w)

            if not ablated_captions:
                continue

            sims = clip_similarity(
                model=model,
                preprocess=preprocess,
                tokenizer=tokenizer,
                image_pil=image_pil,
                texts=texts,
                device=device,
            )

            full_score = sims[0].item()
            ablated_scores = sims[1:].tolist()

            for w, cap_removed, s_removed in zip(ablated_words, ablated_captions, ablated_scores):
                delta = full_score - s_removed
                per_word_scores[w].append(delta)
                per_example_results.append({
                    "image_path": im_path,
                    "original_caption": caption,
                    "removed_word": w,
                    "ablated_caption": cap_removed,
                    "full_score": full_score,
                    "ablated_score": s_removed,
                    "delta": delta,
                })

        if sample_num % 10 == 0:
            print(f"Processed {sample_num}/{len(sample_indices)} images")

    # summarize words
    summary = []
    for w, deltas in per_word_scores.items():
        summary.append({
            "word": w,
            "count": len(deltas),
            "mean_delta": sum(deltas) / len(deltas),
            "max_delta": max(deltas),
            "min_delta": min(deltas),
        })

    summary.sort(key=lambda x: (-x["mean_delta"], -x["count"], x["word"]))

    print("\nTop words by mean ablation drop:")
    for row in summary[:50]:
        print(
            f"{row['word']:15s} "
            f"count={row['count']:4d} "
            f"mean_delta={row['mean_delta']:.6f} "
            f"max={row['max_delta']:.6f}"
        )

    print("\nSample detailed examples:")
    per_example_results.sort(key=lambda x: -x["delta"])
    for ex in per_example_results[:20]:
        print("\n----------------------------------------")
        print("Image:   ", ex["image_path"])
        print("Word:    ", ex["removed_word"])
        print("Delta:   ", f"{ex['delta']:.6f}")
        print("Caption: ", ex["original_caption"])
        print("Ablated: ", ex["ablated_caption"])

    return summary, per_example_results


if __name__ == "__main__":
    annotations_json = "/data/pwojcik/coco_2014/annotations/captions_train2014.json"
    coco_root = "/data/pwojcik/coco_2014"

    summary, details = run_ablation_experiment(
        annotations_json=annotations_json,
        coco_root=coco_root,
        sample_size=50,
        seed=42,
        device="cuda",
        clip_model_name="ViT-B-32",
        clip_pretrained="openai",
    )