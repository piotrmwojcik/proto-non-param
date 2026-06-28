#!/usr/bin/env python3
"""
Diagnostic: measure prototype utilization entropy from a trained PNP checkpoint.

Runs N batches of VG training images through the model and aggregates
mixture_weights [B, V] to compute the marginal distribution over vocabulary
prototypes.  Reports:
  - entropy of the marginal vs maximum possible entropy (log V)
  - effective number of prototypes  exp(H)
  - fraction of prototypes receiving > threshold weight
  - top-20 most-activated prototype words

Usage on Athena:
  python scripts/check_prototype_utilization.py \
      --ckpt /net/tscratch/people/plgabedychaj/train_logs/vg_contrastive/run_B_contrastive10_k1_30ep/ckpt.pth \
      --vocab-cache-path /net/tscratch/people/plgabedychaj/vocab/vg_cache.pt \
      --vg-root /net/tscratch/people/plgabedychaj/vg \
      --vg-region-descriptions /net/tscratch/people/plgabedychaj/vg/region_descriptions.json \
      --n-batches 50 \
      --batch-size 64
"""

import argparse
import math
import sys
import os

import torch
import torch.nn.functional as F
import open_clip

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modeling.backbone import MODEL_DICT, DIM_DICT
from modeling.pnp import PNP
from clip_dataset import VisualGenomeDataset
from torch import nn


class _DinoBackbone(nn.Module):
    """Minimal DINOv2 backbone that avoids torch.hub.load.

    Depth is inferred from the checkpoint state_dict so it works for both
    the standard 12-block vitb14 and any DINOv2BackboneExpanded variant
    (e.g. n_splits=1 → 13 blocks).  Returns 3-tuple to match the training
    backbone interface (PNP only consumes the first element).
    """
    def __init__(self, name: str = "dinov2_vitb14", depth: int = 12):
        super().__init__()
        self.dino = MODEL_DICT[name](depth=depth)
        self.dim = DIM_DICT[name]

    def forward(self, x: torch.Tensor):
        fd = self.dino.forward_features(x)
        patches = fd["x_norm_patchtokens"]   # [B, N, D]  (registers already excluded)
        cls     = fd["x_norm_clstoken"]      # [B, D]
        return patches, patches, cls          # 3-tuple: PNP only uses element 0


def _detect_depth(state_dict: dict) -> int:
    """Infer transformer block count from checkpoint keys."""
    indices = [
        int(k.split(".")[3])
        for k in state_dict
        if k.startswith("backbone.dino.blocks.")
    ]
    return max(indices) + 1 if indices else 12


def build_model(ckpt_path: str, vocab_cache_path: str, device: torch.device) -> PNP:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    hp = ckpt.get("hparams", {})
    state_dict = ckpt.get("state_dict", ckpt)

    backbone_name = hp.get("backbone", "dinov2_vitb14")
    depth = _detect_depth(state_dict)
    print(f"  Backbone : {backbone_name}  depth={depth}")
    backbone = _DinoBackbone(name=backbone_name, depth=depth)
    dim = backbone.dim

    clip_model, _, _ = open_clip.create_model_and_transforms(
        hp.get("coco_clip_model_name", "ViT-B-16"),
        pretrained=hp.get("coco_clip_pretrained", "openai"),
    )
    clip_model = clip_model.eval()
    for p in clip_model.parameters():
        p.requires_grad = False

    net = PNP(
        backbone=backbone,
        dim=dim,
        temperature=hp.get("temperature", 0.2),
        clip_text_dim=hp.get("clip_text_dim", 512),
        text_proj_hidden_dim=hp.get("text_proj_hidden_dim", 1024),
        vocab_cache_path=vocab_cache_path,
        prototype_init_noise=0.0,
        clip_model=clip_model,
    )
    net.load_state_dict(state_dict, strict=True)
    net.eval()
    return net.to(device)


@torch.no_grad()
def collect_weights(net: PNP, loader, n_batches: int, device: torch.device) -> torch.Tensor:
    V = net.vocab_size
    marginal = torch.zeros(V, device=device)
    total = 0
    for i, batch in enumerate(loader):
        if i >= n_batches:
            break
        imgs = batch[0].to(device, non_blocking=True)
        out = net(imgs)
        weights = out["mixture_weights"]   # [B, V], softmax-normalised
        marginal += weights.sum(0)
        total += weights.shape[0]
        if (i + 1) % 10 == 0:
            print(f"  batch {i+1}/{n_batches}  ({total} images)", flush=True)
    marginal /= total
    return marginal


def report(marginal: torch.Tensor, words: list, threshold: float) -> None:
    V = marginal.shape[0]
    p = (marginal + 1e-12)
    p = p / p.sum()
    H = -(p * p.log()).sum().item()
    H_max = math.log(V)
    eff_n = math.exp(H)
    active = (marginal > threshold).sum().item()

    print(f"\n=== Prototype Utilization ===")
    print(f"  Vocabulary size V            : {V}")
    print(f"  Entropy H                    : {H:.3f}  (nats)  {H/math.log(2):.2f} bits")
    print(f"  Max entropy log(V)           : {H_max:.3f}  (nats)  {H_max/math.log(2):.2f} bits")
    print(f"  H / H_max  (uniformity)      : {H/H_max:.4f}  (1.000 = perfectly uniform)")
    print(f"  Effective prototypes exp(H)  : {eff_n:.0f}  ({100*eff_n/V:.1f}% of V)")
    print(f"  Active prototypes (> {threshold}): {active}  ({100*active/V:.1f}% of V)")

    print(f"\n  Top-20 most-activated prototypes:")
    top_vals, top_idx = marginal.topk(20)
    for rank, (idx, val) in enumerate(zip(top_idx.tolist(), top_vals.tolist()), 1):
        word = words[idx] if idx < len(words) else f"[{idx}]"
        print(f"    {rank:>2}. {word:<28} {val:.5f}")

    bot_vals, bot_idx = marginal.topk(5, largest=False)
    print(f"\n  5 least-activated prototypes:")
    for idx, val in zip(bot_idx.tolist(), bot_vals.tolist()):
        word = words[idx] if idx < len(words) else f"[{idx}]"
        print(f"      {word:<28} {val:.2e}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--vocab-cache-path", required=True)
    p.add_argument("--vg-root", required=True)
    p.add_argument("--vg-region-descriptions", required=True)
    p.add_argument("--n-batches", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--threshold", type=float, default=1e-4)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    device = torch.device(args.device)
    print(f"Loading: {args.ckpt}")

    net = build_model(args.ckpt, args.vocab_cache_path, device)
    print(f"V = {net.vocab_size}")

    cache = torch.load(args.vocab_cache_path, map_location="cpu")
    vocab_words = list(cache.keys())
    vocab_to_idx = {w: i for i, w in enumerate(vocab_words)}

    dataset = VisualGenomeDataset(
        vg_root=args.vg_root,
        region_descriptions_json=args.vg_region_descriptions,
        vocab_to_idx=vocab_to_idx,
        train=True,
        caption_embeds_path=None,   # not needed
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(args.device == "cuda"),
    )

    print(f"Running {args.n_batches} batches × {args.batch_size} images …")
    marginal = collect_weights(net, loader, args.n_batches, device)
    report(marginal, vocab_words, args.threshold)


if __name__ == "__main__":
    main()
