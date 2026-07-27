#!/usr/bin/env python3
"""
CLIP-substituted pseudo-intervention curve for a train_cub_joint.py checkpoint.

*** CAVEAT: this is NOT the paper's intervention curve. ***
Espinosa Zarlenga et al.'s intervention curves swap concept representations
for GROUND-TRUTH concept labels and check whether task accuracy rises. We
have no ground truth for CUB's GPT-3-generated concepts -- this script
swaps in CLIP's own raw image-vs-concept similarity instead, the same weak
substitute train_cub_joint.py's sufficiency term uses. A rising curve here
is evidence the sufficiency term is doing *something* internally consistent,
not evidence of genuine concept-intervention behavior.

Method: sweep the fraction of concepts whose activation is swapped from the
model's own vocab_logits to CLIP's raw similarity score (same fixed random
subset of concept indices per fraction, for determinism), evaluate top-1/
top-5 on the held-out CUB test split at each fraction, plot the curve.

Test-split CLIP scores aren't in the cached build_clip_vocab_scores.py
output (that cache only covers train+val, since it exists to build training
targets) -- computed fresh here via the checkpoint's own CLIP image encoder
against its own concept vocab embeddings, which is the same computation
build_clip_vocab_scores.py does, just for test images.

Usage:
  python scripts/eval_cub_sufficiency_curve.py \
    --ckpt $SCRATCH/train_logs/cub_joint/ckpt.pth \
    --cub-root $SCRATCH/cub200 --cub-annotations $SCRATCH/cub200/annotations \
    --out-dir results/cub_sufficiency_curve
"""

import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, SCRIPT_DIR)

from evaluate_pnp_cub_concepts import build_class_index, list_split  # noqa: E402
from train_cub_joint import (  # noqa: E402
    build_pnp, standardize, IMAGENET_MEAN, IMAGENET_STD,
)
from torchvision.transforms import v2  # noqa: E402


@torch.no_grad()
def encode_test_split(net, samples, device, batch_size=32):
    """vocab_logits (model's own concept activations) + raw CLIP-vs-concept
    similarity, computed fresh per image (see module docstring: the cached
    build_clip_vocab_scores.py output doesn't cover the test split)."""
    transform = v2.Compose([
        v2.Resize((224, 224), interpolation=v2.InterpolationMode.BICUBIC),
        v2.ToTensor(),
        v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    clip_transform = v2.Compose([
        v2.Resize((224, 224), interpolation=v2.InterpolationMode.BICUBIC),
        v2.ToTensor(),
        v2.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                      std=(0.26862954, 0.26130258, 0.27577711)),
    ])

    vocab_logits_all, raw_score_all, labels_all = [], [], []
    vocab_emb = F.normalize(net.vocab_clip_embeddings, dim=-1)  # [V, 512]

    for start in tqdm(range(0, len(samples), batch_size), desc="Encoding test split"):
        batch = samples[start:start + batch_size]
        paths = [p for p, _ in batch]
        labels = [lab for _, lab in batch]
        pil_imgs = [Image.open(p).convert("RGB") for p in paths]

        imgs = torch.stack([transform(im) for im in pil_imgs]).to(next(net.parameters()).device)
        outputs = net(imgs)
        vocab_logits_all.append(outputs["vocab_logits"].cpu())

        clip_imgs = torch.stack([clip_transform(im) for im in pil_imgs]).to(imgs.device)
        clip_img_emb = F.normalize(net.clip_model.encode_image(clip_imgs).float(), dim=-1)
        raw_score = clip_img_emb @ vocab_emb.T  # [B, V]
        raw_score_all.append(raw_score.cpu())

        labels_all.extend(labels)

    return (torch.cat(vocab_logits_all, dim=0), torch.cat(raw_score_all, dim=0),
            torch.tensor(labels_all, dtype=torch.long))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", required=True, help="train_cub_joint.py checkpoint (ckpt.pth)")
    p.add_argument("--cub-root", required=True)
    p.add_argument("--cub-annotations", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--fractions", type=float, nargs="+",
                   default=[0.0, 0.1, 0.25, 0.5, 0.75, 1.0])
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)

    print("Loading train_cub_joint.py checkpoint ...")
    ckpt = torch.load(args.ckpt, map_location="cpu")
    hparams = ckpt["hparams"]
    net = build_pnp(hparams["vocab_cache_path"], device)
    net.load_state_dict(ckpt["state_dict"])
    net.eval()

    cls_head = nn.Linear(net.vocab_size, hparams["num_classes"]).to(device)
    cls_head.load_state_dict(ckpt["cls_head_state_dict"])
    cls_head.eval()

    class_to_idx = build_class_index(args.cub_root)
    test_samples = list_split(args.cub_root, "test", class_to_idx)
    print(f"Test images: {len(test_samples)}")

    vocab_logits, raw_score, labels = encode_test_split(net, test_samples, device, args.batch_size)
    vocab_logits, raw_score, labels = vocab_logits.to(device), raw_score.to(device), labels.to(device)

    std_vocab = standardize(vocab_logits)
    std_raw = standardize(raw_score)

    rng = torch.Generator().manual_seed(args.seed)
    n_concepts = vocab_logits.shape[1]
    perm = torch.randperm(n_concepts, generator=rng)

    results = []
    for frac in args.fractions:
        n_swap = int(round(frac * n_concepts))
        subset = perm[:n_swap]

        mixed = std_vocab.clone()
        if n_swap > 0:
            mixed[:, subset] = std_raw[:, subset]

        with torch.no_grad():
            logits = cls_head(mixed)
            top5 = logits.topk(5, dim=-1).indices
            top1_acc = (top5[:, 0] == labels).float().mean().item()
            top5_acc = (top5 == labels.unsqueeze(1)).any(dim=1).float().mean().item()

        print(f"  fraction={frac:.2f}  n_swap={n_swap}/{n_concepts}  "
              f"top1={100 * top1_acc:.2f}%  top5={100 * top5_acc:.2f}%")
        results.append({"fraction": frac, "n_swap": n_swap,
                        "top1_acc": round(100 * top1_acc, 4),
                        "top5_acc": round(100 * top5_acc, 4)})

    out_path = os.path.join(args.out_dir, "sufficiency_curve.json")
    with open(out_path, "w") as f:
        json.dump({
            "ckpt": args.ckpt,
            "n_concepts": n_concepts,
            "n_test": len(test_samples),
            "caveat": ("CLIP-substituted pseudo-intervention curve, NOT ground-truth "
                       "concept labels -- see module docstring."),
            "results": results,
        }, f, indent=2)
    print(f"Saved {out_path}")

    fig, ax = plt.subplots(figsize=(6, 5), dpi=140)
    fracs = [r["fraction"] * 100 for r in results]
    top1s = [r["top1_acc"] for r in results]
    ax.plot(fracs, top1s, marker="o")
    ax.set_xlabel("Concepts CLIP-substituted (%)", fontsize=13)
    ax.set_ylabel("Top-1 accuracy (%)", fontsize=13)
    ax.tick_params(axis="both", labelsize=11)
    ax.set_title("CLIP-substituted pseudo-intervention curve\n"
                  "(NOT ground-truth concept labels)", fontsize=13)
    fig.tight_layout()
    fig_path = os.path.join(args.out_dir, "sufficiency_curve.png")
    fig.savefig(fig_path)
    print(f"Saved {fig_path}")


if __name__ == "__main__":
    main()
