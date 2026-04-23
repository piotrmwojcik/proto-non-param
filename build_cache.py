import torch
import open_clip
from pathlib import Path


VOCAB_PATH = "vocab/birds.txt"
CACHE_PATH = "/net/tscratch/people/plgpiotrwojcik/vocab/birds_cache.pt"

CKPT_PATH = "/net/tscratch/people/plgpiotrwojcik/open_clip_train_logs/cub_vitb32_openai_train_v2/checkpoints/epoch_latest.pt"


@torch.no_grad()
def main():

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # -------------------------
    # Load vocab
    # -------------------------
    with open(VOCAB_PATH, "r", encoding="utf-8") as f:
        nouns = [line.strip() for line in f if line.strip()]

    print(f"Loaded {len(nouns)} nouns")

    # -------------------------
    # Build CLIP model (same arch!)
    # -------------------------
    model, _, _ = open_clip.create_model_and_transforms(
        "ViT-B-32",
        pretrained=None,
    )

    ckpt = torch.load(CKPT_PATH, map_location="cpu")
    state_dict = ckpt.get("state_dict", ckpt)

    cleaned_state_dict = {}
    for k, v in state_dict.items():
        new_key = k
        for prefix in ("module.", "model.", "_orig_mod."):
            if new_key.startswith(prefix):
                new_key = new_key[len(prefix) :]
        cleaned_state_dict[new_key] = v

    missing, unexpected = model.load_state_dict(cleaned_state_dict, strict=False)

    model = model.eval().to(device)

    print("Loaded checkpoint: ", CKPT_PATH, flush=True)
    print("Missing keys:", len(missing))
    print("Unexpected keys:", len(unexpected))
    if missing:
        print("First missing keys:", missing[:20])
    if unexpected:
        print("First unexpected keys:", unexpected[:20])

    model = model.eval().to(device)

    tokenizer = open_clip.get_tokenizer("ViT-B-32")

    # -------------------------
    # Encode vocab
    # -------------------------
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

        print(f"Processed {min(i + batch_size, len(nouns))}/{len(nouns)}")

    # -------------------------
    # Save cache
    # -------------------------
    Path(CACHE_PATH).parent.mkdir(parents=True, exist_ok=True)
    torch.save(cache, CACHE_PATH)

    print(f"\nSaved CLIP cache to: {CACHE_PATH}")
    print(f"Embedding dim: {next(iter(cache.values())).shape}")


if __name__ == "__main__":
    main()
