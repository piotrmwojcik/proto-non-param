import nltk

nltk.download('punkt', download_dir='./nltk_data')
nltk.download('punkt_tab', download_dir='./nltk_data')
nltk.download('averaged_perceptron_tagger', download_dir='./nltk_data')

import os
os.environ['NLTK_DATA'] = './nltk_data'

input_file = "vocab/mscoco.txt"
output_file = "vocab/mscoco_nouns.txt"

nouns = []

with open(input_file, "r", encoding="utf-8") as f:
    lines = f.readlines()

for line in lines:
    # tokenize sentence
    tokens = nltk.word_tokenize(line.strip())

    # POS tagging
    tagged = nltk.pos_tag(tokens)

    # extract nouns
    for word, tag in tagged:
        if tag in ["NN", "NNS", "NNP", "NNPS"]:
            nouns.append(word.lower())

# remove duplicates (optional)
nouns = sorted(set(nouns))

with open(output_file, "w", encoding="utf-8") as f:
    for noun in nouns:
        f.write(noun + "\n")

print(f"Saved {len(nouns)} nouns to {output_file}")