"""
Extract unique content words (nouns, verbs, adjectives) from Caltech101 descriptions
and build a CLIP embedding cache for use in training.

Usage:
    python build_caltech101_vocab.py \
        --descriptions /path/to/caltech101_descriptions.json \
        --vocab-out vocab/caltech101.txt \
        --cache-out vocab/caltech101_cache.pt
"""
import argparse
import json
from collections import Counter
from pathlib import Path

import torch
import open_clip

import nltk
from nltk.stem import WordNetLemmatizer

nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
nltk.download("averaged_perceptron_tagger", quiet=True)
nltk.download("averaged_perceptron_tagger_eng", quiet=True)
nltk.download("wordnet", quiet=True)

lemmatizer = WordNetLemmatizer()

COMMON_VERBS = {
    "be", "is", "am", "are", "was", "were", "been", "being",
    "have", "has", "had", "having",
    "do", "does", "did", "doing",
}


def extract_words(text: str) -> list:
    tokens = nltk.word_tokenize(text.lower())
    pos_tags = nltk.pos_tag(tokens)
    words = []
    for word, pos in pos_tags:
        if pos.startswith("NN"):
            lemma = lemmatizer.lemmatize(word, pos="n")
        elif pos.startswith("VB"):
            lemma = lemmatizer.lemmatize(word, pos="v")
            if lemma in COMMON_VERBS:
                continue
        elif pos.startswith("JJ"):
            lemma = lemmatizer.lemmatize(word, pos="a")
        else:
            continue
        if lemma.isalpha() and len(lemma) >= 3:
            words.append(lemma)
    return words


@torch.no_grad()
def build_cache(words: list, cache_out: str):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Building CLIP cache on {device} for {len(words)} words...")

    model, _, _ = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    model = model.eval().to(device)

    batch_size = 256
    cache = {}

    for i in range(0, len(words), batch_size):
        batch = words[i:i + batch_size]
        prompts = [f"a photo of a {w}" for w in batch]
        tokens = tokenizer(prompts).to(device)
        feats = model.encode_text(tokens)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        feats = feats.cpu()
        for word, feat in zip(batch, feats):
            cache[word] = feat
        print(f"  {min(i + batch_size, len(words))}/{len(words)}")

    Path(cache_out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(cache, cache_out)
    print(f"Saved cache → {cache_out}  (dim={next(iter(cache.values())).shape})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--descriptions", required=True, help="Path to caltech101_descriptions.json")
    parser.add_argument("--vocab-out", default="vocab/caltech101.txt", help="Output vocab text file")
    parser.add_argument("--cache-out", default="vocab/caltech101_cache.pt", help="Output CLIP cache file")
    parser.add_argument("--min-count", type=int, default=2, help="Minimum word frequency to include (default: 2)")
    args = parser.parse_args()

    print(f"Loading descriptions from {args.descriptions}...")
    with open(args.descriptions, "r", encoding="utf-8") as f:
        data = json.load(f)

    counts = Counter()
    for descriptions in data.values():
        for desc in descriptions:
            for word in extract_words(desc):
                counts[word] += 1

    vocab = sorted(w for w, c in counts.items() if c >= args.min_count)
    print(f"Vocabulary size: {len(vocab)} words (min_count={args.min_count})")

    Path(args.vocab_out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.vocab_out, "w", encoding="utf-8") as f:
        f.write("\n".join(vocab))
    print(f"Saved vocab → {args.vocab_out}")

    build_cache(vocab, args.cache_out)


if __name__ == "__main__":
    main()
