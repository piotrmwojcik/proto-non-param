import re
import torch
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet as wn

# -----------------------
# NLTK resources
# -----------------------
nltk.download("wordnet")
nltk.download("omw-1.4")

vocab_cache_path = "vocab/laion_clip_cache.pt"

# -----------------------
# Load concept dictionary
# -----------------------
cache = torch.load(vocab_cache_path, map_location="cpu")
vocab_words = list(cache.keys())   # keep original order
V = len(vocab_words)

print("Vocabulary size:", V)

lemmatizer = WordNetLemmatizer()

# -----------------------
# Helpers
# -----------------------
def normalize_token(word: str) -> str:
    word = word.lower().strip()
    word = re.sub(r"[^a-z]", "", word)   # keep only letters
    return word

def classify_wordnet(word: str):
    """
    Return booleans: is_noun, is_verb, is_adj
    Uses WordNet synsets after lemmatization.
    """
    noun_lemma = lemmatizer.lemmatize(word, pos="n")
    verb_lemma = lemmatizer.lemmatize(word, pos="v")
    adj_lemma  = lemmatizer.lemmatize(word, pos="a")

    noun_synsets = wn.synsets(noun_lemma, pos=wn.NOUN)
    verb_synsets = wn.synsets(verb_lemma, pos=wn.VERB)
    adj_synsets  = wn.synsets(adj_lemma, pos=wn.ADJ) + wn.synsets(adj_lemma, pos=wn.ADJ_SAT)

    return (
        len(noun_synsets) > 0,
        len(verb_synsets) > 0,
        len(adj_synsets) > 0,
        noun_lemma,
        verb_lemma,
        adj_lemma,
    )

# -----------------------
# Masks
# -----------------------
valid_mask = torch.zeros(V, dtype=torch.bool)
noun_mask = torch.zeros(V, dtype=torch.bool)
verb_mask = torch.zeros(V, dtype=torch.bool)
adj_mask  = torch.zeros(V, dtype=torch.bool)

normalized_words = []
kept_words = []

for i, raw_word in enumerate(vocab_words):
    word = normalize_token(raw_word)
    normalized_words.append(word)

    # basic filtering
    if len(word) < 3:
        continue
    if not word.isalpha():
        continue

    is_noun, is_verb, is_adj, noun_lemma, verb_lemma, adj_lemma = classify_wordnet(word)

    # keep only if recognized by WordNet in at least one category
    if not (is_noun or is_verb or is_adj):
        continue

    valid_mask[i] = True
    kept_words.append(word)

    if is_noun:
        noun_mask[i] = True
    if is_verb:
        verb_mask[i] = True
    if is_adj:
        adj_mask[i] = True

# -----------------------
# Optional disjoint masks
# priority: noun > adjective > verb
# -----------------------
noun_only_mask = noun_mask.clone()
adj_only_mask  = adj_mask & ~noun_only_mask
verb_only_mask = verb_mask & ~noun_only_mask & ~adj_only_mask

# -----------------------
# Print statistics
# -----------------------
print("Valid words:", valid_mask.sum().item())
print("Nouns:", noun_mask.sum().item())
print("Verbs:", verb_mask.sum().item())
print("Adjectives:", adj_mask.sum().item())

print("Disjoint nouns:", noun_only_mask.sum().item())
print("Disjoint adjectives:", adj_only_mask.sum().item())
print("Disjoint verbs:", verb_only_mask.sum().item())

# -----------------------
# Example usage
# -----------------------
valid_words = [w for w, m in zip(vocab_words, valid_mask) if m]
noun_words = [w for w, m in zip(vocab_words, noun_mask) if m]
verb_words = [w for w, m in zip(vocab_words, verb_mask) if m]
adj_words  = [w for w, m in zip(vocab_words, adj_mask) if m]

noun_only_words = [w for w, m in zip(vocab_words, noun_only_mask) if m]
adj_only_words  = [w for w, m in zip(vocab_words, adj_only_mask) if m]
verb_only_words = [w for w, m in zip(vocab_words, verb_only_mask) if m]

print("Sample valid words:", valid_words[:20])
print("Sample nouns:", noun_words[:20])
print("Sample verbs:", verb_words[:20])
print("Sample adjectives:", adj_words[:20])

print("Sample disjoint nouns:", noun_only_words[:20])
print("Sample disjoint verbs:", verb_only_words[:20])
print("Sample disjoint adjectives:", adj_only_words[:20])

# -----------------------
# Save masks for reuse
# -----------------------
torch.save(
    {
        "vocab_words": vocab_words,
        "normalized_words": normalized_words,
        "valid_mask": valid_mask,
        "noun_mask": noun_mask,
        "verb_mask": verb_mask,
        "adj_mask": adj_mask,
        "noun_only_mask": noun_only_mask,
        "verb_only_mask": verb_only_mask,
        "adj_only_mask": adj_only_mask,
    },
    "vocab/laion_clip_pos_masks.pt",
)

print("Saved masks to vocab/laion_clip_pos_masks.pt")