import spacy

# load POS tagger
nlp = spacy.load("en_core_web_sm")

input_file = "vocab/mscoco.txt"
output_file = "vocab/mscoco_nouns.txt"

nouns = []

with open(input_file, "r") as f:
    words = [w.strip() for w in f if w.strip()]

# process words as a single doc for efficiency
doc = nlp(" ".join(words))

for token in doc:
    if token.pos_ in {"NOUN", "PROPN"}:
        nouns.append(token.text)

with open(output_file, "w") as f:
    for w in nouns:
        f.write(w + "\n")

print(f"Total words: {len(words)}")
print(f"Nouns found: {len(nouns)}")
print(f"Saved to: {output_file}")