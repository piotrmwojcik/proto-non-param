import torch
import open_clip
from pathlib import Path


VOCAB_PATH = "vocab/merged_vocab.txt"
CACHE_PATH = "/net/tscratch/people/plgpiotrwojcik/vocab/merged_vocab_openclip_pretrained.pt"

MODEL_NAME = "ViT-B-32"
PRETRAINED = "openai"   # OpenAI CLIP weights for ViT-B/32


@torch.no_grad()
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    with open(VOCAB_PATH, "r", encoding="utf-8") as f:
        nouns = [line.strip() for line in f if line.strip()]

    print(f"Loaded {len(nouns)} nouns")

    model, _, _ = open_clip.create_model_and_transforms(
        MODEL_NAME,
        pretrained=PRETRAINED,
    )

    model = model.eval().to(device)

    print(f"Loaded pretrained OpenCLIP model: {MODEL_NAME}, pretrained={PRETRAINED}")

    tokenizer = open_clip.get_tokenizer(MODEL_NAME)

    batch_size = 256
    cache = {}

    for i in range(0, len(nouns), batch_size):
        batch = nouns[i : i + batch_size]
        prompts = [f"a photo of a {w}" for w in batch]

        tokens = tokenizer(prompts).to(device)

        text_features = model.encode_text(tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        text_features = text_features.cpu()

        for word, feat in zip(batch, text_features):
            cache[word] = feat

        print(f"Processed {min(i + batch_size, len(nouns))}/{len(nouns)}", flush=True)

    Path(CACHE_PATH).parent.mkdir(parents=True, exist_ok=True)
    torch.save(cache, CACHE_PATH)

    print(f"\nSaved CLIP cache to: {CACHE_PATH}")
    print(f"Embedding dim: {next(iter(cache.values())).shape}")


if __name__ == "__main__":
    main()