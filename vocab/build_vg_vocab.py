#!/usr/bin/env python3
"""Build a CLIP-embedding vocabulary cache from Visual Genome region descriptions.

Mirrors the pattern of build_caltech101_vocab.py.  Scans all region phrases,
extracts nouns / adjectives / verbs via NLTK lemmatisation, filters by
frequency and document-frequency, then encodes the retained words with a
frozen CLIP text encoder and saves them as a {word: embedding} .pt cache.

The output cache is in the exact format expected by train.py:
    cache = torch.load("vocab/vg_cache.pt")   # dict[str, Tensor[512]]

Usage:
    python vocab/build_vg_vocab.py \\
        --region-descriptions /data/vg/region_descriptions.json \\
        --vocab-out vocab/vg.txt \\
        --cache-out vocab/vg_cache.pt \\
        --clip-model-name ViT-L-14 \\
        --clip-pretrained openai \\
        --min-count 5 \\
        --max-doc-freq 0.5
"""

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

import open_clip
import torch
import nltk
from nltk.stem import WordNetLemmatizer

# allow importing extract_caption_words from the parent package
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
nltk.download("averaged_perceptron_tagger", quiet=True)
nltk.download("averaged_perceptron_tagger_eng", quiet=True)
nltk.download("wordnet", quiet=True)

_lemmatizer = WordNetLemmatizer()
_COMMON_VERBS = {
    "be", "is", "am", "are", "was", "were", "been", "being",
    "have", "has", "had", "having",
    "do", "does", "did", "doing",
}


def _extract_words(text: str) -> list[str]:
    """Return lemmatised nouns / adjectives / content-verbs from a phrase."""
    tokens = nltk.word_tokenize(text.lower())
    pos_tags = nltk.pos_tag(tokens)
    words: list[str] = []
    seen: set[str] = set()
    for word, pos in pos_tags:
        if pos.startswith(("NN", "VB", "JJ")):
            if pos.startswith("NN"):
                lemma = _lemmatizer.lemmatize(word, pos="n")
            elif pos.startswith("VB"):
                lemma = _lemmatizer.lemmatize(word, pos="v")
                if lemma in _COMMON_VERBS:
                    continue
            else:
                lemma = _lemmatizer.lemmatize(word, pos="a")
            if lemma not in seen:
                words.append(lemma)
                seen.add(lemma)
    return words


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build VG vocabulary + CLIP embedding cache"
    )
    parser.add_argument("--region-descriptions", type=str, required=True,
                        help="Path to region_descriptions.json (VG v1.4)")
    parser.add_argument("--vocab-out",  type=str, default="vocab/vg.txt",
                        help="Output path for plain-text word list")
    parser.add_argument("--cache-out",  type=str, default="vocab/vg_cache.pt",
                        help="Output path for CLIP embedding cache (.pt)")
    parser.add_argument("--clip-model-name",  type=str, default="ViT-B-32")
    parser.add_argument("--clip-pretrained",  type=str, default="openai")
    parser.add_argument("--min-count",    type=int,   default=5,
                        help="Minimum total occurrences across the dataset")
    parser.add_argument("--max-doc-freq", type=float, default=0.5,
                        help="Max fraction of images a word may appear in")
    args = parser.parse_args()

    # ------------------------------------------------------------------ #
    # 1. Load JSON
    # ------------------------------------------------------------------ #
    print(f"Loading region descriptions from {args.region_descriptions} ...")
    with open(args.region_descriptions, "r", encoding="utf-8") as f:
        raw = json.load(f)
    n_images = len(raw)
    print(f"  {n_images} images found")

    # ------------------------------------------------------------------ #
    # 2. Count word occurrences and document frequencies
    # ------------------------------------------------------------------ #
    word_count: Counter = Counter()   # total occurrences across all regions
    doc_count:  Counter = Counter()   # number of images that contain the word

    for i, entry in enumerate(raw):
        if i % 10_000 == 0:
            print(f"  scanning {i}/{n_images} ...", end="\r")
        phrases = [
            r["phrase"] for r in entry.get("regions", [])
            if r.get("phrase", "").strip()
        ]
        img_words: set[str] = set()
        for phrase in phrases:
            for w in _extract_words(phrase):
                word_count[w] += 1
                img_words.add(w)
        for w in img_words:
            doc_count[w] += 1

    print(f"\n  raw vocabulary size: {len(word_count)}")

    # ------------------------------------------------------------------ #
    # 3. Filter
    # ------------------------------------------------------------------ #
    max_docs = int(args.max_doc_freq * n_images)
    vocab_words = sorted(
        w for w, cnt in word_count.items()
        if cnt >= args.min_count and doc_count[w] <= max_docs
    )
    print(f"  vocabulary after filtering (min_count={args.min_count}, "
          f"max_doc_freq={args.max_doc_freq}): {len(vocab_words)} words")

    # ------------------------------------------------------------------ #
    # 4. Save plain-text word list
    # ------------------------------------------------------------------ #
    os.makedirs(os.path.dirname(os.path.abspath(args.vocab_out)), exist_ok=True)
    with open(args.vocab_out, "w", encoding="utf-8") as f:
        f.write("\n".join(vocab_words))
    print(f"Saved word list → {args.vocab_out}")

    # ------------------------------------------------------------------ #
    # 5. Encode with CLIP
    # ------------------------------------------------------------------ #
    print(f"Encoding {len(vocab_words)} words with {args.clip_model_name} ...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, _ = open_clip.create_model_and_transforms(
        args.clip_model_name, pretrained=args.clip_pretrained
    )
    model = model.eval().to(device)
    tokenizer = open_clip.get_tokenizer(args.clip_model_name)

    cache: dict[str, torch.Tensor] = {}
    batch_size = 256
    with torch.no_grad():
        for i in range(0, len(vocab_words), batch_size):
            batch = vocab_words[i: i + batch_size]
            tokens = tokenizer(batch).to(device)
            embeddings = model.encode_text(tokens)
            embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
            for word, emb in zip(batch, embeddings.cpu()):
                cache[word] = emb

    # ------------------------------------------------------------------ #
    # 6. Save cache
    # ------------------------------------------------------------------ #
    os.makedirs(os.path.dirname(os.path.abspath(args.cache_out)), exist_ok=True)
    torch.save(cache, args.cache_out)
    print(f"Saved CLIP cache ({len(cache)} words) → {args.cache_out}")


if __name__ == "__main__":
    main()
