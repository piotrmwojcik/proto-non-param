# build_clip_noun_cache.py

import torch
import open_clip
from pathlib import Path


VOCAB_PATH = "vocab/mscoco_nouns.txt"
CACHE_PATH = "vocab/mscoco_nouns_clip_cache.pt"


@torch.no_grad()
def main():

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load nouns
    with open(VOCAB_PATH, "r", encoding="utf-8") as f:
        nouns = [line.strip() for line in f if line.strip()]

    print(f"Loaded {len(nouns)} nouns")

    # Load CLIP
    model, _, _ = open_clip.create_model_and_transforms(
        "ViT-B-32",
        pretrained="openai",
    )
    tokenizer = open_clip.get_tokenizer("ViT-B-32")

    model = model.eval().to(device)

    batch_size = 256
    cache = {}

    for i in range(0, len(nouns), batch_size):

        batch = nouns[i:i + batch_size]
        prompts = [f"a photo of a {w}" for w in batch]

        tokens = tokenizer(prompts).to(device)
        text_features = model.encode_text(tokens)

        # normalize like CLIP retrieval
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        text_features = text_features.cpu()

        for word, feat in zip(batch, text_features):
            cache[word] = feat

        print(f"Processed {min(i + batch_size, len(nouns))}/{len(nouns)}")

    # Ensure output directory exists
    Path(CACHE_PATH).parent.mkdir(parents=True, exist_ok=True)

    torch.save(cache, CACHE_PATH)

    print(f"\nSaved CLIP cache to: {CACHE_PATH}")
    print(f"Embedding dim: {next(iter(cache.values())).shape}")


if __name__ == "__main__":
    main()