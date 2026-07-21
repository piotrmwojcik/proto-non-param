#!/usr/bin/env python3
"""
Zero-shot concept-bottleneck evaluation of PNP on CUB-200 using Label-free-CBM's
GPT-generated concept set (cub_filtered.txt, 379 short phrases e.g. "a black
cap and back").

Strategy: mirrors evaluate_pnp_refer.py's inference-time text->prototype path
(CLIP-encode a phrase -> net.text_projection_head -> visual-space prototype),
just applied to a whole concept vocabulary instead of one referring
expression. The frozen checkpoint's own PNP.forward() is NOT called, since it
is wired to the original training vocabulary buffers -- this script builds a
separate [379, D] concept-prototype matrix and replicates only the relevant
patch-similarity + top-k aggregation slice of PNP.forward().

Per image: patch/concept cosine logits -> same top-k mean pooling the
checkpoint was trained/inferred with (read off net.agg_mode / net.topk_k, not
hardcoded) -> one [379] concept-activation vector. A linear probe
(nn.Linear(379, 200) + CrossEntropyLoss) is then fit on train+val activations
and evaluated on the official CUB test split, matching how Label-free-CBM
itself is scored (accuracy of a linear layer over concept-similarity scores).

Usage:
  python scripts/evaluate_pnp_cub_concepts.py \
    --ckpt $SCRATCH/train_logs/vg_contrastive/run_M1_vitl14_sk10_koleo01_30ep/ckpt.pth \
    --cub-root $SCRATCH/cub200 \
    --concepts-file $SCRATCH/vocab/cub_filtered_concepts.txt \
    --img-size 672 \
    --out-dir eval_results/cub_concepts
"""

import argparse
import json
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, SCRIPT_DIR)

from evaluate_pnp_refer import build_model, build_img_transform  # noqa: E402


# ---------------------------------------------------------------------------
# Data: reuse download_cub200.py's already-organized <split>/<ClassName>/*.jpg
# layout directly -- no need to re-parse CUB's raw annotation files.
# ---------------------------------------------------------------------------

IMG_EXTS = (".jpg", ".jpeg", ".png")


def list_split(cub_root: str, split: str, class_to_idx: dict) -> list:
    split_dir = os.path.join(cub_root, split)
    samples = []
    for cls_name in sorted(os.listdir(split_dir)):
        cls_dir = os.path.join(split_dir, cls_name)
        if not os.path.isdir(cls_dir):
            continue
        label = class_to_idx[cls_name]
        for fname in sorted(os.listdir(cls_dir)):
            if fname.lower().endswith(IMG_EXTS):
                samples.append((os.path.join(cls_dir, fname), label))
    return samples


def build_class_index(cub_root: str) -> dict:
    """Stable 0..199 label indices from sorted class-folder names, shared
    across train/val/test (all three splits share the same class set)."""
    classes = sorted(os.listdir(os.path.join(cub_root, "train")))
    return {name: i for i, name in enumerate(classes)}


# ---------------------------------------------------------------------------
# Concept prototypes: CLIP-encode each concept phrase verbatim (no template
# wrapping -- matches Label-free-CBM's own protocol), project through the
# checkpoint's own text_projection_head.
# ---------------------------------------------------------------------------

@torch.no_grad()
def build_concept_prototypes(net, tokenizer, concepts: list, device) -> torch.Tensor:
    tokens = tokenizer(concepts).to(device)
    text_emb = net.clip_model.encode_text(tokens)
    text_emb = F.normalize(text_emb.float(), dim=-1)
    proto = net.text_projection_head(text_emb)
    proto = F.normalize(proto, dim=-1)
    return proto  # [n_concepts, D]


# ---------------------------------------------------------------------------
# Per-image concept-activation vector: replicates PNP.forward()'s top-k
# aggregation slice against the swapped-in concept_prototypes, bypassing
# forward() itself (which is wired to the original vocab buffers).
# ---------------------------------------------------------------------------

@torch.no_grad()
def encode_split(net, img_transform, samples: list, concept_prototypes: torch.Tensor,
                 device, batch_size: int = 32):
    activations, labels = [], []
    k = net.topk_k if net.agg_mode == "topk" else None

    for start in tqdm(range(0, len(samples), batch_size), desc="Encoding"):
        batch = samples[start:start + batch_size]
        imgs = torch.stack([img_transform(Image.open(p).convert("RGB")) for p, _ in batch]).to(device)
        batch_labels = [lab for _, lab in batch]

        patch_tokens = net.backbone(imgs)[0]                     # [B, N, D]
        patch_tokens = F.normalize(patch_tokens, dim=-1)
        logits = torch.einsum("bnd,cd->bnc", patch_tokens, concept_prototypes)  # [B, N, C]

        if net.agg_mode == "cross_attn":
            attn = F.softmax(logits / net.attn_temp, dim=1)
            concept_vec = (attn * logits).sum(dim=1)              # [B, C]
        else:
            concept_vec = logits.topk(k, dim=1).values.mean(dim=1)  # [B, C]

        activations.append(concept_vec.cpu())
        labels.extend(batch_labels)

    return torch.cat(activations, dim=0), torch.tensor(labels, dtype=torch.long)


# ---------------------------------------------------------------------------
# Linear probe: nn.Linear(n_concepts, n_classes), matches how Label-free-CBM
# itself is scored (a linear layer over concept-similarity scores).
# ---------------------------------------------------------------------------

def train_probe(fit_x: torch.Tensor, fit_y: torch.Tensor, n_classes: int,
                epochs: int, lr: float, device) -> nn.Linear:
    probe = nn.Linear(fit_x.shape[1], n_classes).to(device)
    opt = torch.optim.Adam(probe.parameters(), lr=lr)
    fit_x, fit_y = fit_x.to(device), fit_y.to(device)

    for _ in range(epochs):
        opt.zero_grad()
        loss = F.cross_entropy(probe(fit_x), fit_y)
        loss.backward()
        opt.step()

    return probe


@torch.no_grad()
def eval_probe(probe: nn.Linear, test_x: torch.Tensor, test_y: torch.Tensor, device):
    logits = probe(test_x.to(device))
    top1 = (logits.argmax(dim=-1).cpu() == test_y).float().mean().item()
    top5 = (logits.topk(5, dim=-1).indices.cpu() == test_y.unsqueeze(1)).any(dim=1).float().mean().item()
    return top1, top5


def main():
    p = argparse.ArgumentParser(description="Zero-shot PNP concept-bottleneck eval on CUB-200")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--cub-root", required=True)
    p.add_argument("--concepts-file", required=True)
    p.add_argument("--img-size", type=int, default=672)
    p.add_argument("--out-dir", default="./eval_results/cub_concepts")
    p.add_argument("--probe-epochs", type=int, default=100)
    p.add_argument("--probe-lr", type=float, default=1e-2)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    args = p.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print(f"Loading PNP model from {args.ckpt} ...")
    net, tokenizer, hparams = build_model(args.ckpt, device)

    with open(args.concepts_file) as f:
        concepts = [line.strip() for line in f if line.strip()]
    print(f"Loaded {len(concepts)} concepts from {args.concepts_file}")

    concept_prototypes = build_concept_prototypes(net, tokenizer, concepts, device)
    assert concept_prototypes.shape == (len(concepts), net.dim), (
        f"expected [{len(concepts)}, {net.dim}], got {tuple(concept_prototypes.shape)}"
    )
    norms = concept_prototypes.norm(dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-4), "concept prototypes must be unit-normalized"
    print(f"Concept prototypes: {tuple(concept_prototypes.shape)}, agg_mode={net.agg_mode}"
          + (f", topk_k={net.topk_k}" if net.agg_mode == "topk" else ""))

    class_to_idx = build_class_index(args.cub_root)
    print(f"{len(class_to_idx)} CUB classes")

    img_transform = build_img_transform(args.img_size)

    fit_samples = list_split(args.cub_root, "train", class_to_idx) + \
                  list_split(args.cub_root, "val", class_to_idx)
    test_samples = list_split(args.cub_root, "test", class_to_idx)
    print(f"Fit (train+val): {len(fit_samples)} images   Test: {len(test_samples)} images")

    os.makedirs(args.out_dir, exist_ok=True)
    cache_path = os.path.join(args.out_dir, f"activations_img{args.img_size}.pt")
    if os.path.isfile(cache_path):
        print(f"Loading cached activations from {cache_path}")
        cache = torch.load(cache_path)
        fit_x, fit_y, test_x, test_y = cache["fit_x"], cache["fit_y"], cache["test_x"], cache["test_y"]
    else:
        fit_x, fit_y = encode_split(net, img_transform, fit_samples, concept_prototypes,
                                     device, args.batch_size)
        test_x, test_y = encode_split(net, img_transform, test_samples, concept_prototypes,
                                       device, args.batch_size)
        torch.save({"fit_x": fit_x, "fit_y": fit_y, "test_x": test_x, "test_y": test_y}, cache_path)
        print(f"Saved activations to {cache_path}")

    print(f"Training linear probe ({args.probe_epochs} epochs, lr={args.probe_lr}) ...")
    probe = train_probe(fit_x, fit_y, len(class_to_idx), args.probe_epochs, args.probe_lr, device)
    fit_top1, _ = eval_probe(probe, fit_x, fit_y, device)
    top1, top5 = eval_probe(probe, test_x, test_y, device)

    # Sanity signal: if fit accuracy is also near-chance, the activation
    # pipeline itself is broken (concept swap-in / aggregation bug), not just
    # weak generalization from train to test.
    print(f"\nFit (train+val) top-1: {100*fit_top1:.2f}%")
    print(f"Test top-1: {100*top1:.2f}%   top-5: {100*top5:.2f}%")
    print(f"(random chance: {100/len(class_to_idx):.2f}%; Label-free-CBM's own CUB accuracy: ~74.6%)")

    result = {
        "ckpt": args.ckpt,
        "img_size": args.img_size,
        "n_concepts": len(concepts),
        "n_classes": len(class_to_idx),
        "n_fit": len(fit_samples),
        "n_test": len(test_samples),
        "fit_top1_acc": round(100 * fit_top1, 4),
        "top1_acc": round(100 * top1, 4),
        "top5_acc": round(100 * top5, 4),
    }
    out_path = os.path.join(args.out_dir, f"result_img{args.img_size}.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved result to {out_path}")


if __name__ == "__main__":
    main()
