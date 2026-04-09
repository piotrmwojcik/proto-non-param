import os
import json
import random
import hashlib
from collections import Counter

import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
from torchvision.transforms import v2

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
            else:  # adjectives
                lemma = lemmatizer.lemmatize(word, pos="a")

            if lemma in vocab_to_idx and lemma not in seen:
                word_idxs.append(vocab_to_idx[lemma])
                seen.add(lemma)

    return word_idxs


def _make_cache_path(annotation_file: str, vocab_to_idx: dict, cache_dir: str = None):
    if cache_dir is None:
        cache_dir = os.path.join(os.path.dirname(annotation_file), ".cache")
    os.makedirs(cache_dir, exist_ok=True)

    vocab_hash = hashlib.md5(
        json.dumps(vocab_to_idx, sort_keys=True).encode("utf-8")
    ).hexdigest()[:12]

    ann_base = os.path.splitext(os.path.basename(annotation_file))[0]
    return os.path.join(cache_dir, f"{ann_base}_samples_{vocab_hash}.pt")


def coco_clip_collate_fn(batch):
    images, captions, prob_dists, indices = zip(*batch)

    images = torch.stack(images, dim=0)
    prob_dists = torch.stack(prob_dists, dim=0)
    indices = torch.tensor(indices, dtype=torch.long)

    # keep variable-length caption lists as-is
    captions = list(captions)

    return images, captions, prob_dists, indices


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
        train: bool = True
    ):
        self.annotations_json = annotations_json
        self.coco_root = coco_root
        self.device = torch.device(device)
        self.rng = random.Random(seed)
        self.vocab_to_idx = vocab_to_idx
        self.vocab_size = len(vocab_to_idx)
        self.train = train

        self.train_transform = v2.Compose([
            v2.RandomResizedCrop(
                size=224,
                scale=(0.8, 1.0),
                interpolation=v2.InterpolationMode.BICUBIC
            ),
            v2.RandomHorizontalFlip(p=0.5),
            v2.ToTensor(),
            v2.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ])

        # validation / test transform
        self.eval_transform = transforms.Compose([
            transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ])

        cache_path = _make_cache_path(annotations_json, vocab_to_idx, cache_dir)

        # Try loading from cache first
        if use_cache and os.path.exists(cache_path):
            print(f"Loading dataset cache from: {cache_path}")
            cached = torch.load(cache_path, map_location="cpu")
            self.samples = cached["samples"]
            print(f"Loaded {len(self.samples)} cached samples")
            return

        # Otherwise build dataset
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

            if image_id not in captions_per_image:
                captions_per_image[image_id] = []
            captions_per_image[image_id].append(caption)
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

        # Save cache
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
        if self.train:
            img_tensor = self.train_transform(img)
        else:
            img_tensor = self.eval_transform(img)

        return img_tensor, captions, prob_dist, index


class Caltech101CLIPDataset(Dataset):
    """Dataset for Caltech-101 images with VLM-generated descriptions.

    Expects a descriptions JSON with format:
        { "/some/path/caltech101/train/ClassName/image_0001.jpg": ["desc1", ...], ... }

    The machine-specific prefix before "caltech101/" is stripped so the
    dataset works on any machine given the correct caltech_root.
    """

    def __init__(
        self,
        descriptions_json: str,
        caltech_root: str,
        vocab_to_idx: dict,
        train: bool = True,
        cache_dir: str = None,
        use_cache: bool = True,
    ):
        self.caltech_root = caltech_root
        self.vocab_to_idx = vocab_to_idx
        self.vocab_size = len(vocab_to_idx)
        self.train = train

        self.train_transform = v2.Compose([
            v2.RandomResizedCrop(
                size=224,
                scale=(0.8, 1.0),
                interpolation=v2.InterpolationMode.BICUBIC,
            ),
            v2.RandomHorizontalFlip(p=0.5),
            v2.ToTensor(),
            v2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])

        self.eval_transform = transforms.Compose([
            transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])

        cache_path = _make_cache_path(descriptions_json, vocab_to_idx, cache_dir)

        if use_cache and os.path.exists(cache_path):
            print(f"Loading dataset cache from: {cache_path}")
            cached = torch.load(cache_path, map_location="cpu")
            self.samples = cached["samples"]
            print(f"Loaded {len(self.samples)} cached samples")
            return

        split_key = "/train/" if train else "/val/"

        with open(descriptions_json, "r", encoding="utf-8") as f:
            raw = json.load(f)

        samples = []
        print(f"Building Caltech101 {'train' if train else 'val'} samples...")
        for json_path, descriptions in raw.items():
            # Normalise separators so the split check works on both OS
            norm = json_path.replace("\\", "/")
            if split_key not in norm:
                continue

            # Extract the relative part after "caltech101/"
            marker = "caltech101/"
            idx = norm.find(marker)
            if idx == -1:
                continue
            rel = norm[idx + len(marker):]   # e.g. "train/Faces/image_0001.jpg"

            im_path = os.path.join(caltech_root, *rel.split("/"))
            if not os.path.isfile(im_path):
                continue

            if not descriptions:
                continue

            counts = Counter()
            total_valid = 0
            for desc in descriptions:
                word_idxs = extract_caption_words(desc, vocab_to_idx)
                for wi in word_idxs:
                    counts[wi] += 1
                total_valid += len(word_idxs)

            prob_dist = torch.zeros(self.vocab_size, dtype=torch.float32)
            if total_valid > 0:
                for wi, cnt in counts.items():
                    prob_dist[wi] = cnt / total_valid

            samples.append((im_path, descriptions, prob_dist))

        self.samples = samples
        print(f"Done. Total samples: {len(self.samples)}")

        if use_cache:
            torch.save({"samples": self.samples}, cache_path)
            print(f"Saved cache to: {cache_path}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index: int):
        im_path, descriptions, prob_dist = self.samples[index]
        img = Image.open(im_path).convert("RGB")
        img_tensor = self.train_transform(img) if self.train else self.eval_transform(img)
        return img_tensor, descriptions, prob_dist, index


class CUBCLIPDataset(Dataset):
    """CUB-200-2011 dataset using attribute annotations as vocabulary.

    Each image has a certainty-weighted probability distribution over the 312
    binary attributes (e.g. "blue wings", "hooked bill").

    Expects:
        dataset_root/train/<ClassName>/<image>.jpg
        dataset_root/val/<ClassName>/<image>.jpg
        annotations_dir/attributes/attributes.txt
        annotations_dir/attributes/image_attribute_labels.txt
        annotations_dir/images.txt
        annotations_dir/image_class_labels.txt
        annotations_dir/train_test_split.txt (for reference only)

    Certainty weighting:  1=not_visible→0,  2=guessing→0.5,
                          3=probably→0.75,  4=definitely→1.0
    """

    CERTAINTY_WEIGHTS = {1: 0.0, 2: 0.5, 3: 0.75, 4: 1.0}

    def __init__(
        self,
        dataset_root: str,
        annotations_dir: str,
        vocab_to_idx: dict,
        train: bool = True,
        cache_dir: str = None,
        use_cache: bool = True,
    ):
        self.dataset_root = dataset_root
        self.annotations_dir = annotations_dir
        self.vocab_to_idx = vocab_to_idx
        self.vocab_size = len(vocab_to_idx)
        self.train = train

        self.train_transform = v2.Compose([
            v2.RandomResizedCrop(size=224, scale=(0.8, 1.0),
                                 interpolation=v2.InterpolationMode.BICUBIC),
            v2.RandomHorizontalFlip(p=0.5),
            v2.ToTensor(),
            v2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])
        self.eval_transform = transforms.Compose([
            transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])

        # Use a stable cache key based on split + annotations dir + vocab
        cache_key_str = f"cub200_{'train' if train else 'val'}_{annotations_dir}"
        cache_path = _make_cache_path(cache_key_str, vocab_to_idx, cache_dir)

        if use_cache and os.path.exists(cache_path):
            print(f"Loading CUB dataset cache from: {cache_path}")
            cached = torch.load(cache_path, map_location="cpu")
            self.samples = cached["samples"]
            self.attribute_values = cached["attribute_values"]
            print(f"Loaded {len(self.samples)} cached samples")
            return

        ann_dir = annotations_dir

        # Parse attribute names: "1 has_bill_shape::dagger" → value = "dagger"
        attr_file = os.path.join(ann_dir, "attributes", "attributes.txt")
        if not os.path.exists(attr_file):
            attr_file = os.path.join(ann_dir, "attributes.txt")
        attribute_values = self._parse_attribute_values(attr_file)
        self.attribute_values = attribute_values  # list of str, indexed by attr_id-1
        n_attrs = len(attribute_values)

        # Parse image_attribute_labels.txt → {image_id: tensor(n_attrs)}
        print("Building CUB per-image attribute distributions...")
        attr_label_file = os.path.join(ann_dir, "attributes", "image_attribute_labels.txt")
        img_attr_scores = self._parse_image_attribute_labels(attr_label_file, n_attrs)

        # Parse images.txt: {image_id: relative_path}
        images_file = os.path.join(ann_dir, "images.txt")
        img_id_to_rel = {}
        with open(images_file) as f:
            for line in f:
                parts = line.strip().split(maxsplit=1)
                img_id_to_rel[int(parts[0])] = parts[1]

        split_dir = "train" if train else "val"
        split_root = os.path.join(dataset_root, split_dir)

        samples = []
        for img_id, rel_path in img_id_to_rel.items():
            # rel_path: "001.Black_footed_Albatross/Black_Footed_Albatross_0001_796111.jpg"
            parts = rel_path.split("/")
            cls_folder = parts[0]
            # Map "001.Black_footed_Albatross" → "Black_footed_Albatross"
            class_name = ".".join(cls_folder.split(".")[1:])
            filename = parts[-1]
            im_path = os.path.join(split_root, class_name, filename)

            if not os.path.isfile(im_path):
                continue

            if img_id not in img_attr_scores:
                continue

            raw_scores = img_attr_scores[img_id]  # tensor(n_attrs)
            # Map attribute values → vocab indices
            attr_words = []
            for attr_idx in range(n_attrs):
                val = attribute_values[attr_idx]
                if val in vocab_to_idx:
                    attr_words.append(val)

            # Build prob_dist over vocab from attribute scores
            prob_dist = torch.zeros(self.vocab_size, dtype=torch.float32)
            for attr_idx in range(n_attrs):
                val = attribute_values[attr_idx]
                if val in vocab_to_idx:
                    prob_dist[vocab_to_idx[val]] += raw_scores[attr_idx]

            total = prob_dist.sum()
            if total > 0:
                prob_dist = prob_dist / total

            samples.append((im_path, attr_words, prob_dist))

        self.samples = samples
        print(f"Done. CUB {split_dir}: {len(self.samples)} samples")

        if use_cache:
            torch.save({"samples": self.samples, "attribute_values": self.attribute_values},
                       cache_path)
            print(f"Saved cache to: {cache_path}")

    def _parse_attribute_values(self, attr_file: str) -> list:
        """Return list of value strings indexed by (attr_id - 1)."""
        entries = {}
        with open(attr_file) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(maxsplit=1)
                attr_id = int(parts[0])
                full_name = parts[1]
                if "::" in full_name:
                    value = full_name.split("::", 1)[1]
                else:
                    value = full_name
                value = value.replace("_", " ").strip()
                entries[attr_id] = value
        max_id = max(entries.keys())
        return [entries.get(i + 1, "") for i in range(max_id)]

    def _parse_image_attribute_labels(self, label_file: str, n_attrs: int) -> dict:
        """Parse image_attribute_labels.txt into {image_id: tensor(n_attrs)}."""
        # Format: image_id  attribute_id  is_present  certainty_id  time
        scores = {}
        with open(label_file) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                img_id = int(parts[0])
                attr_id = int(parts[1])
                is_present = int(parts[2])
                certainty_id = int(parts[3])

                if img_id not in scores:
                    scores[img_id] = torch.zeros(n_attrs, dtype=torch.float32)

                weight = self.CERTAINTY_WEIGHTS.get(certainty_id, 0.0)
                scores[img_id][attr_id - 1] = is_present * weight

        return scores

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index: int):
        im_path, attr_words, prob_dist = self.samples[index]
        img = Image.open(im_path).convert("RGB")
        img_tensor = self.train_transform(img) if self.train else self.eval_transform(img)
        return img_tensor, attr_words, prob_dist, index


class AwA2CLIPDataset(Dataset):
    """Animals with Attributes 2 (AwA2) dataset using predicate annotations.

    Each image uses its class-level predicate distribution from
    predicate-matrix-continuous.txt (50×85 matrix, values in [0,100]).

    Expects:
        dataset_root/train/<classname>/<image>.jpg
        dataset_root/val/<classname>/<image>.jpg
        annotations_dir/predicates.txt
        annotations_dir/predicate-matrix-continuous.txt
        annotations_dir/classes.txt
    """

    def __init__(
        self,
        dataset_root: str,
        annotations_dir: str,
        vocab_to_idx: dict,
        train: bool = True,
        cache_dir: str = None,
        use_cache: bool = True,
    ):
        self.dataset_root = dataset_root
        self.annotations_dir = annotations_dir
        self.vocab_to_idx = vocab_to_idx
        self.vocab_size = len(vocab_to_idx)
        self.train = train

        self.train_transform = v2.Compose([
            v2.RandomResizedCrop(size=224, scale=(0.8, 1.0),
                                 interpolation=v2.InterpolationMode.BICUBIC),
            v2.RandomHorizontalFlip(p=0.5),
            v2.ToTensor(),
            v2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])
        self.eval_transform = transforms.Compose([
            transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])

        cache_key_str = f"awa2_{'train' if train else 'val'}_{annotations_dir}"
        cache_path = _make_cache_path(cache_key_str, vocab_to_idx, cache_dir)

        if use_cache and os.path.exists(cache_path):
            print(f"Loading AwA2 dataset cache from: {cache_path}")
            cached = torch.load(cache_path, map_location="cpu")
            self.samples = cached["samples"]
            print(f"Loaded {len(self.samples)} cached samples")
            return

        ann_dir = annotations_dir

        # Parse predicates.txt → list of predicate names (indexed by pred_id-1)
        predicates = self._parse_predicates(os.path.join(ann_dir, "predicates.txt"))
        n_preds = len(predicates)

        # Parse classes.txt → {class_name: class_idx (0-based)}
        class_name_to_idx = self._parse_classes(os.path.join(ann_dir, "classes.txt"))

        # Parse predicate-matrix-continuous.txt → tensor(n_classes, n_preds)
        pred_matrix = self._parse_pred_matrix(
            os.path.join(ann_dir, "predicate-matrix-continuous.txt"),
            len(class_name_to_idx), n_preds
        )

        # Normalize each row to sum to 1
        row_sums = pred_matrix.sum(dim=1, keepdim=True).clamp(min=1e-8)
        pred_matrix = pred_matrix / row_sums

        split_dir = "train" if train else "val"
        split_root = os.path.join(dataset_root, split_dir)

        if not os.path.isdir(split_root):
            raise FileNotFoundError(f"Split directory not found: {split_root}")

        # Build per-class prob_dist mapped to vocab
        pred_words_per_class = {}
        prob_dist_per_class = {}
        for cls_name, cls_idx in class_name_to_idx.items():
            row = pred_matrix[cls_idx]  # tensor(n_preds)
            attr_words = [p for p in predicates if p in vocab_to_idx]
            prob_dist = torch.zeros(self.vocab_size, dtype=torch.float32)
            for pred_idx, pred_name in enumerate(predicates):
                if pred_name in vocab_to_idx:
                    prob_dist[vocab_to_idx[pred_name]] += row[pred_idx]
            total = prob_dist.sum()
            if total > 0:
                prob_dist = prob_dist / total
            pred_words_per_class[cls_name] = attr_words
            prob_dist_per_class[cls_name] = prob_dist

        print(f"Building AwA2 {'train' if train else 'val'} samples...")
        samples = []
        for cls_name in sorted(os.listdir(split_root)):
            cls_dir = os.path.join(split_root, cls_name)
            if not os.path.isdir(cls_dir):
                continue
            if cls_name not in prob_dist_per_class:
                print(f"  WARNING: class '{cls_name}' not in annotations, skipping")
                continue
            prob_dist = prob_dist_per_class[cls_name]
            attr_words = pred_words_per_class[cls_name]
            for fname in sorted(os.listdir(cls_dir)):
                if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
                    continue
                im_path = os.path.join(cls_dir, fname)
                samples.append((im_path, attr_words, prob_dist))

        self.samples = samples
        print(f"Done. AwA2 {split_dir}: {len(self.samples)} samples")

        if use_cache:
            torch.save({"samples": self.samples}, cache_path)
            print(f"Saved cache to: {cache_path}")

    def _parse_predicates(self, predicates_txt: str) -> list:
        """Return list of predicate name strings (ordered by predicate id)."""
        entries = {}
        with open(predicates_txt) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(maxsplit=1)
                pred_id = int(parts[0])
                name = parts[1].replace("+", " ").replace("_", " ").strip()
                entries[pred_id] = name
        max_id = max(entries.keys())
        return [entries[i + 1] for i in range(max_id)]

    def _parse_classes(self, classes_txt: str) -> dict:
        """Return {class_name: 0-based index}."""
        result = {}
        with open(classes_txt) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(maxsplit=1)
                cls_id = int(parts[0])
                cls_name = parts[1].strip()
                result[cls_name] = cls_id - 1  # 0-based
        return result

    def _parse_pred_matrix(self, matrix_txt: str, n_classes: int, n_preds: int) -> torch.Tensor:
        """Parse predicate-matrix-continuous.txt → tensor(n_classes, n_preds)."""
        rows = []
        with open(matrix_txt) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                vals = [float(v) for v in line.split()]
                rows.append(vals)
        return torch.tensor(rows, dtype=torch.float32)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index: int):
        im_path, attr_words, prob_dist = self.samples[index]
        img = Image.open(im_path).convert("RGB")
        img_tensor = self.train_transform(img) if self.train else self.eval_transform(img)
        return img_tensor, attr_words, prob_dist, index