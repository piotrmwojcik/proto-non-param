import json
import nltk
from nltk.tokenize import word_tokenize

nltk.download("punkt")

annotations_json = "/data/pwojcik/coco_2014/annotations/captions_train2014.json"

# -----------------------------
# Load COCO captions
# -----------------------------
with open(annotations_json, "r") as f:
    coco_data = json.load(f)

captions_per_image = {}

for ann in coco_data["annotations"]:
    image_id = int(ann["image_id"])
    caption = ann["caption"]

    if image_id not in captions_per_image:
        captions_per_image[image_id] = []

    captions_per_image[image_id].append(caption)

# -----------------------------
# Collect words
# -----------------------------
all_words = set()

import json
import nltk
from nltk.tokenize import word_tokenize

nltk.download("punkt")

annotations_json = "/data/pwojcik/coco_2014/annotations/captions_train2014.json"

# -----------------------------
# Load COCO captions
# -----------------------------
with open(annotations_json, "r") as f:
    coco_data = json.load(f)

captions_per_image = {}

for ann in coco_data["annotations"]:
    image_id = int(ann["image_id"])
    caption = ann["caption"]

    if image_id not in captions_per_image:
        captions_per_image[image_id] = []

    captions_per_image[image_id].append(caption)

# -----------------------------
# Collect words
# -----------------------------
all_words = set()

for captions in captions_per_image.values():
    for caption in captions:

        tokens = word_tokenize(caption.lower())

        for t in tokens:
            if t.isalpha():   # keep only alphabetic tokens
                all_words.add(t)

# -----------------------------
# Convert to sorted list
# -----------------------------
word_list = sorted(list(all_words))

print("Total unique words:", len(word_list))
print("\nSample words:")
print(word_list[:50])

# Print full list if desired
print("\nFull vocabulary:")
for w in word_list:
    print(w)

# -----------------------------
# Convert to sorted list
# -----------------------------
word_list = sorted(list(all_words))

print("Total unique words:", len(word_list))
print("\nSample words:")
print(word_list[:50])

# Print full list if desired
print("\nFull vocabulary:")
for w in word_list:
    print(w)