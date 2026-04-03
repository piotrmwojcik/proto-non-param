"""
Extract meaningful nouns and adjectives from Caltech101 descriptions
and build a CLIP embedding cache for use in training.

Filters applied:
  - Nouns and adjectives only (no verbs)
  - Minimum word frequency (--min-count)
  - Maximum document frequency (--max-doc-freq): drops words appearing in
    too many images (generic words like "background", "image", "visible")
  - VLM artifact blocklist: description-generation artifacts removed explicitly

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

from nltk.corpus import wordnet as wn

lemmatizer = WordNetLemmatizer()


def is_meaningful_noun_or_adj(word: str) -> bool:
    """Return True only if the word is a genuine noun or adjective per WordNet,
    and not primarily a verb (more verb synsets than noun+adj synsets)."""
    noun_synsets = wn.synsets(word, pos=wn.NOUN)
    adj_synsets = wn.synsets(word, pos=wn.ADJ) + wn.synsets(word, pos=wn.ADJ_SAT)
    verb_synsets = wn.synsets(word, pos=wn.VERB)

    has_noun_or_adj = len(noun_synsets) + len(adj_synsets) > 0
    primarily_verb = len(verb_synsets) > len(noun_synsets) + len(adj_synsets)

    return has_noun_or_adj and not primarily_verb

# Words that are VLM description artifacts or too generic to be meaningful
VLM_BLOCKLIST = {
    # description artifacts
    "image", "images", "photo", "photograph", "picture", "pictures",
    "background", "foreground", "scene", "view", "shot",
    # generic visual verbs turned nouns
    "appear", "show", "display", "depict", "feature", "capture",
    "visible", "visual", "overall", "general",
    # generic spatial/quantity words
    "area", "part", "section", "side", "top", "bottom", "center",
    "left", "right", "number", "variety", "type", "kind", "example",
    # common but meaningless adjectives in descriptions
    "various", "several", "different", "similar", "possible",
    "additional", "certain", "particular", "specific",
}


def extract_words(text: str) -> list:
    """Extract nouns and adjectives only."""
    tokens = nltk.word_tokenize(text.lower())
    pos_tags = nltk.pos_tag(tokens)
    words = []
    for word, pos in pos_tags:
        if pos.startswith("NN"):
            lemma = lemmatizer.lemmatize(word, pos="n")
        elif pos.startswith("JJ"):
            lemma = lemmatizer.lemmatize(word, pos="a")
        else:
            continue  # skip verbs and everything else
        if lemma.isalpha() and len(lemma) >= 3 and lemma not in VLM_BLOCKLIST:
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
    parser.add_argument("--min-count", type=int, default=3,
                        help="Minimum total word occurrences to include (default: 3)")
    parser.add_argument("--max-doc-freq", type=float, default=0.3,
                        help="Exclude words appearing in more than this fraction of images (default: 0.3)")
    args = parser.parse_args()

    print(f"Loading descriptions from {args.descriptions}...")
    with open(args.descriptions, "r", encoding="utf-8") as f:
        data = json.load(f)

    total_images = len(data)
    word_counts = Counter()      # total occurrences
    doc_counts = Counter()       # number of images containing each word

    for descriptions in data.values():
        words_in_image = set()
        for desc in descriptions:
            for word in extract_words(desc):
                word_counts[word] += 1
                words_in_image.add(word)
        for word in words_in_image:
            doc_counts[word] += 1

    max_doc_count = int(args.max_doc_freq * total_images)
    print(f"Total images: {total_images}, max doc count threshold: {max_doc_count} ({args.max_doc_freq*100:.0f}%)")

    vocab = sorted(
        w for w, c in word_counts.items()
        if c >= args.min_count
        and doc_counts[w] <= max_doc_count
        and is_meaningful_noun_or_adj(w)
    )

    print(f"Vocabulary size: {len(vocab)} words "
          f"(min_count={args.min_count}, max_doc_freq={args.max_doc_freq})")

    # Show top filtered-out words for debugging
    filtered_out = [w for w in word_counts if w not in vocab and word_counts[w] >= args.min_count]
    filtered_out.sort(key=lambda w: doc_counts[w], reverse=True)
    print(f"Top words removed by doc-freq filter: {filtered_out[:20]}")

    Path(args.vocab_out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.vocab_out, "w", encoding="utf-8") as f:
        f.write("\n".join(vocab))
    print(f"Saved vocab → {args.vocab_out}")

    build_cache(vocab, args.cache_out)


if __name__ == "__main__":
    main()
