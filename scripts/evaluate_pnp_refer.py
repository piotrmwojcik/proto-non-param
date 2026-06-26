#!/usr/bin/env python3
"""
Zero-shot referring image segmentation evaluation for PNP (Proto-Non-Parametric).

Strategy: CLIP-encode the referring expression → project through PNP's
text_projection_head into visual space → cosine similarity with DINOv2 patch
tokens → spatial activation map → threshold → binary mask.

Data: reuses sag_refseg's pre-built .npz batches (same data as SaG eval).

Usage:
  python scripts/evaluate_pnp_refer.py \
    --ckpt $SCRATCH/checkpoints/pnp/best.pth \
    --dataset Gref \
    --data_split val \
    --data_root $SCRATCH/data/refcoco
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Path setup: script lives inside proto-non-param/scripts/
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, REPO_ROOT)

from modeling.pnp import PNP                                          # noqa: E402
from modeling.backbone import (DINOv2Backbone, DINOv2BackboneExpanded,  # noqa: E402
                                DINOBackboneExpanded, CLIPBackbone)
from evaluation.refer_dataset import ReferDataset                     # noqa: E402

import open_clip                                                       # noqa: E402


# ---------------------------------------------------------------------------
# Metric helpers (matches sag_refseg/evaluate.py convention)
# ---------------------------------------------------------------------------

def mask_IU(pred: np.ndarray, gt: np.ndarray):
    """Return (intersection, union) pixel counts."""
    pred_bool = pred.astype(bool)
    gt_bool = gt.astype(bool)
    I = (pred_bool & gt_bool).sum()
    U = (pred_bool | gt_bool).sum()
    return I, U


# ---------------------------------------------------------------------------
# Model construction (mirrors build_backbone in proto-non-param/train.py)
# ---------------------------------------------------------------------------

def build_backbone(hparams):
    backbone_name = hparams.backbone
    num_splits = getattr(hparams, "num_splits", 0)
    if "dinov2" in backbone_name:
        if num_splits and num_splits > 0:
            backbone = DINOv2BackboneExpanded(
                name=backbone_name,
                n_splits=num_splits,
                mode="append",
                freeze_norm_layer=True,
            )
        else:
            backbone = DINOv2Backbone(name=backbone_name)
    elif "dino" in backbone_name:
        backbone = DINOBackboneExpanded(
            name=backbone_name,
            n_splits=num_splits,
            mode="block_expansion",
            freeze_norm_layer=True,
        )
    elif "clip" in backbone_name:
        backbone = CLIPBackbone(name=backbone_name)
    else:
        raise NotImplementedError(f"Backbone {backbone_name} not supported.")
    return backbone, backbone.dim


def build_model(ckpt_path: str, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    assert "hparams" in ckpt and "state_dict" in ckpt, (
        "Checkpoint must contain 'hparams' and 'state_dict' keys."
    )
    hparams = argparse.Namespace(**ckpt["hparams"])

    backbone, dim = build_backbone(hparams)

    clip_model_name = getattr(hparams, "coco_clip_model_name", "ViT-B-32")
    clip_pretrained = getattr(hparams, "coco_clip_pretrained", "openai")
    clip_model, _, _ = open_clip.create_model_and_transforms(
        clip_model_name, pretrained=clip_pretrained
    )
    clip_model = clip_model.eval().to(device)
    for p in clip_model.parameters():
        p.requires_grad = False

    vocab_cache_path = hparams.vocab_cache_path
    if not os.path.isabs(vocab_cache_path):
        vocab_cache_path = os.path.join(REPO_ROOT, vocab_cache_path)

    net = PNP(
        backbone=backbone,
        dim=dim,
        temperature=getattr(hparams, "temperature", 0.2),
        clip_text_dim=getattr(hparams, "clip_text_dim", 512),
        text_proj_hidden_dim=getattr(hparams, "text_proj_hidden_dim", 768),
        vocab_cache_path=vocab_cache_path,
        prototype_init_noise=getattr(hparams, "prototype_init_noise", 0.01),
        clip_model=clip_model,
    )
    net.load_state_dict(ckpt["state_dict"])
    net.eval().to(device)

    tokenizer = open_clip.get_tokenizer(clip_model_name)
    return net, tokenizer, hparams


# ---------------------------------------------------------------------------
# Image preprocessing (CLIP normalization, 224×224)
# ---------------------------------------------------------------------------

CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)

_img_transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
])


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

@torch.inference_mode()
def evaluate(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print(f"Loading PNP model from {args.ckpt} ...")
    net, tokenizer, hparams = build_model(args.ckpt, device)

    # Number of patches: 224 / 14 = 16 → 16×16 = 256 patches
    patch_grid = 224 // 14  # 16

    # ── Dataset ────────────────────────────────────────────────────────────
    data_root = os.path.join(args.data_root, args.dataset)
    ds = ReferDataset(root=data_root, splitset=args.data_split)
    if len(ds) == 0:
        raise RuntimeError(
            f"No batch files found in {data_root}/{args.data_split}_batch/\n"
            f"  Run sag_refseg/scripts/build_batches.py first."
        )
    print(f"Dataset: {args.dataset} / {args.data_split}  ({len(ds)} samples)")

    # ── Threshold list ─────────────────────────────────────────────────────
    # --threshold-sweep overrides --threshold when provided
    thresholds = sorted(set(args.threshold_sweep if args.threshold_sweep else [args.threshold]))
    sweep_mode = len(thresholds) > 1
    if sweep_mode:
        print(f"Threshold sweep: {thresholds}")

    # Per-threshold accumulators
    total_I  = {t: 0 for t in thresholds}
    total_U  = {t: 0 for t in thresholds}
    per_sample = {t: [] for t in thresholds}

    # ── Eval loop ──────────────────────────────────────────────────────────
    for i in tqdm(range(len(ds)), desc=f"Eval {args.dataset}/{args.data_split}"):
        sentence, img_id, pil_img, gt_mask, _ = ds.get_raw_item(i)

        # Image → [1, 3, 224, 224]
        img_t = _img_transform(pil_img).unsqueeze(0).to(device)

        # Expression → CLIP text embedding [1, 512]
        # sentence may be a numpy bytes/str; coerce to Python str
        tokens = tokenizer([str(sentence)]).to(device)
        e_query = net.clip_model.encode_text(tokens)          # [1, 512]
        e_query = F.normalize(e_query.float(), dim=-1)

        # Project to visual space [1, D]
        p_query = net.text_projection_head(e_query)           # [1, D]

        # DINOv2 patch tokens [1, N, D]
        # backbone may return 2 or 3 values depending on variant — take first
        patch_tokens = net.backbone(img_t)[0]
        patch_tokens = F.normalize(patch_tokens, dim=-1)      # [1, N, D]

        # Cosine similarity → spatial scores [1, N]
        scores = (patch_tokens * p_query.unsqueeze(1)).sum(dim=-1)  # [1, N]

        # Reshape to [1, 1, H_p, W_p] and upsample to GT mask resolution
        H_gt, W_gt = gt_mask.shape[:2]
        spatial = scores.view(1, 1, patch_grid, patch_grid)
        spatial = F.interpolate(spatial, size=(H_gt, W_gt), mode="bilinear",
                                align_corners=False)           # [1, 1, H_gt, W_gt]
        activation = spatial.squeeze().cpu().numpy()           # [H_gt, W_gt]

        # Normalize to [0, 1] once; apply all thresholds in one pass
        a_min, a_max = activation.min(), activation.max()
        if a_max > a_min:
            activation = (activation - a_min) / (a_max - a_min + 1e-8)

        gt_bin = (gt_mask > 0).astype(np.uint8)
        for t in thresholds:
            pred_mask = (activation >= t).astype(np.uint8)
            I, U = mask_IU(pred_mask, gt_bin)
            sample_iou = float(I) / (float(U) + 1e-8) if U > 0 else 0.0
            total_I[t] += I
            total_U[t] += U
            per_sample[t].append({
                "img_id": img_id,
                "index": i,
                "iou": round(sample_iou, 6),
            })

    # ── Save JSON (one per threshold) ──────────────────────────────────────
    out_dir = os.path.join(args.out_dir, "pnp_refer")
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n{'Threshold':>10}  {'oIoU (%)':>10}  {'mIoU (%)':>10}  n")
    print("-" * 42)

    for t in thresholds:
        c_iou = 100.0 * total_I[t] / (total_U[t] + 1e-8)
        m_iou = 100.0 * float(np.mean([s["iou"] for s in per_sample[t]]))
        n = len(per_sample[t])
        print(f"{t:>10.2f}  {c_iou:>10.2f}  {m_iou:>10.2f}  {n}")

        # Filename: dataset_split.json for single threshold (backward compat),
        # dataset_split_tXXX.json for sweep (XXX = threshold × 100, zero-padded).
        if sweep_mode:
            t_str = f"{int(round(t * 100)):03d}"
            fname = f"{args.dataset}_{args.data_split}_t{t_str}.json"
        else:
            fname = f"{args.dataset}_{args.data_split}.json"

        result = {
            "dataset": args.dataset,
            "split": args.data_split,
            "ckpt": args.ckpt,
            "threshold": t,
            "summary": {
                "cIoU": round(c_iou, 4),
                "mIoU": round(m_iou, 4),
                "n_samples": n,
            },
            "per_sample": per_sample[t],
        }
        out_path = os.path.join(out_dir, fname)
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)

    print(f"\nResults saved to {out_dir}/")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="PNP zero-shot RIS evaluation")
    p.add_argument("--ckpt", required=True, help="Path to PNP .pth checkpoint")
    p.add_argument("--dataset", required=True, choices=("Gref", "unc", "unc+"),
                   help="Dataset name (must have pre-built .npz batches)")
    p.add_argument("--data_split", default="val", choices=("val", "testA", "testB"),
                   help="Split to evaluate")
    p.add_argument("--data_root", default="./data/refcoco",
                   help="Root containing {Gref,unc,unc+}/ subdirs with *_batch/ dirs")
    p.add_argument("--out_dir", default="./eval_results",
                   help="Output directory (results go to out_dir/pnp_refer/)")
    p.add_argument("--threshold", type=float, default=0.5,
                   help="Threshold on normalized activation map for binary mask "
                        "(single value; use --threshold-sweep for ablation)")
    p.add_argument("--threshold-sweep", type=float, nargs="+", default=None,
                   metavar="T",
                   help="Sweep over multiple thresholds in a single forward pass. "
                        "Overrides --threshold. Saves one JSON per value with _tXXX suffix. "
                        "Example: --threshold-sweep 0.3 0.4 0.5 0.6 0.7")
    p.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    return p.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
