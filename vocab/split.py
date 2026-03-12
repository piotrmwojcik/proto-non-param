import torch
import nltk
from nltk import pos_tag

# make sure these are available
nltk.download("averaged_perceptron_tagger")
nltk.download("punkt")

vocab_cache_path = "vocab/laion_clip_cache.pt"

# -----------------------
# Load concept dictionary
# -----------------------
cache = torch.load(vocab_cache_path, map_location="cpu")

vocab_words = list(cache.keys())   # order matters
V = len(vocab_words)

print("Vocabulary size:", V)

# -----------------------
# POS tagging
# -----------------------
pos_tags = pos_tag(vocab_words)

noun_mask = torch.zeros(V, dtype=torch.bool)
verb_mask = torch.zeros(V, dtype=torch.bool)
adj_mask  = torch.zeros(V, dtype=torch.bool)

for i, (word, tag) in enumerate(pos_tags):

    if tag.startswith("NN"):
        noun_mask[i] = True

    elif tag.startswith("VB"):
        verb_mask[i] = True

    elif tag.startswith("JJ"):
        adj_mask[i] = True


# -----------------------
# Print statistics
# -----------------------
print("Nouns:", noun_mask.sum().item())
print("Verbs:", verb_mask.sum().item())
print("Adjectives:", adj_mask.sum().item())

# -----------------------
# Example usage
# -----------------------
noun_words = [w for w, m in zip(vocab_words, noun_mask) if m]
verb_words = [w for w, m in zip(vocab_words, verb_mask) if m]
adj_words  = [w for w, m in zip(vocab_words, adj_mask) if m]

print("Sample nouns:", noun_words[:10])
print("Sample verbs:", verb_words[:10])
print("Sample adjectives:", adj_words[:10])