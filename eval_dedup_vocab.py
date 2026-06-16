#!/usr/bin/env python3
"""Evaluate a model trained on a deduplicated vocab against the full vocabulary.

This is the ablation-study counterpart to training with a deduplicated concept
vocabulary (see vocab/deduplicate_vocab.py).

The key hypothesis: the projection head g is shared across all concepts, so at
inference time we can substitute a larger full vocabulary by computing
  p_v = normalize(g(normalize(e_v)))   (δ_v = 0, no residual)
for any concept v whose CLIP embedding e_v is available — even if v was not in
the training vocabulary.

Two evaluation modes are compared
----------------------------------
  dedup_only   Score test images against the training vocabulary (deduplicated).
               Ground truth is also built from the same dedup vocab.
               Measures learning quality on the reduced concept set.

  full_vocab   Score test images against the full original vocabulary.
               Ground truth is rebuilt from the full vocab.
               Measures generalisation of g to unseen (fine-grained) concepts.

Metrics: P@K and R@K for K ∈ {5, 10}.

Usage
-----
    python eval_dedup_vocab.py \\
        --ckpt              /path/to/ckpt.pth \\
        --train-vocab-cache vocab/vg_cache_dedup_t90.pt \\
        --eval-vocab-cache  vocab/vg_cache.pt \\
        --vg-root           /data/vg \\
        --vg-region-descriptions /data/vg/region_descriptions.json \\
        --mode both \\
        --wandb-run-name dedup-ablation-A-kl
"""

import argparse
import math

import torch
import torch.nn.functional as F
import wandb
from torch.utils.data import DataLoader
from tqdm import tqdm

import open_clip

from clip_dataset import VisualGenomeDataset, coco_clip_collate_fn
from modeling.backbone import DINOv2Backbone, DINOv2BackboneExpanded, DINOBackboneExpanded
from modeling.pnp import PNP


# --------------------------------------------------------------------------- #
# Model loading
# --------------------------------------------------------------------------- #

def build_model(ckpt_path: str, device: torch.device) -> tuple[PNP, argparse.Namespace]:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    hparams = argparse.Namespace(**ckpt["hparams"])
    state_dict = ckpt["state_dict"]

    if "dinov2" in hparams.backbone:
        if getattr(hparams, "num_splits", 0) and hparams.num_splits > 0:
            backbone = DINOv2BackboneExpanded(
                name=hparams.backbone,
                n_splits=hparams.num_splits,
                mode="append",
                freeze_norm_layer=True,
            )
        else:
            backbone = DINOv2Backbone(name=hparams.backbone)
    elif "dino" in hparams.backbone:
        backbone = DINOBackboneExpanded(
            name=hparams.backbone,
            n_splits=getattr(hparams, "num_splits", 1),
            mode="block_expansion",
            freeze_norm_layer=True,
        )
    else:
        raise NotImplementedError(f"Backbone {hparams.backbone} not supported")

    clip_model, _, _ = open_clip.create_model_and_transforms(
        getattr(hparams, "coco_clip_model_name", "ViT-B-32"),
        pretrained=getattr(hparams, "coco_clip_pretrained", "openai"),
    )
    clip_model = clip_model.eval().to(device)
    for p in clip_model.parameters():
        p.requires_grad = False

    model = PNP(
        backbone=backbone,
        dim=backbone.dim,
        temperature=getattr(hparams, "temperature", 0.2),
        clip_text_dim=getattr(hparams, "clip_text_dim", 512),
        text_proj_hidden_dim=getattr(hparams, "text_proj_hidden_dim", 768),
        vocab_cache_path=hparams.vocab_cache_path,
        prototype_init_noise=getattr(hparams, "prototype_init_noise", 0.01),
        clip_model=clip_model,
    )
    state_dict = {k: v for k, v in state_dict.items() if not k.startswith("clip_model.")}
    model.load_state_dict(state_dict, strict=False)
    model = model.eval().to(device)
    print(f"Checkpoint loaded — training vocab size: {model.vocab_size}")
    return model, hparams


# --------------------------------------------------------------------------- #
# Prototype construction
# --------------------------------------------------------------------------- #

@torch.no_grad()
def build_prototypes(
    model: PNP,
    vocab_cache: dict,
    device: torch.device,
) -> tuple[list[str], torch.Tensor]:
    """Build visual prototypes for any vocab cache through the trained g (δ=0).

    Works for both the training vocab (uses the stored residuals if frozen=0, they
    don't matter) and a completely new vocab (δ=0 by construction).
    """
    words = list(vocab_cache.keys())
    embs = torch.stack([vocab_cache[w] for w in words], dim=0).to(device)  # [V, 512]
    embs = F.normalize(embs, dim=-1)
    protos = model.text_projection_head(embs)   # [V, D]
    protos = F.normalize(protos, dim=-1)
    return words, protos


# --------------------------------------------------------------------------- #
# Retrieval evaluation
# --------------------------------------------------------------------------- #

@torch.no_grad()
def eval_retrieval(
    model: PNP,
    dataloader: DataLoader,
    prototypes: torch.Tensor,
    device: torch.device,
    Ks: tuple[int, ...] = (5, 10),
    top_patches: int = 5,
) -> dict[str, float]:
    """P@K and R@K for concept retrieval.

    For each test image the model scores against `prototypes`, takes top-K,
    and compares against the ground-truth concept set from the dataloader
    (which must be built with the same vocab as the prototypes).
    """
    all_prec = {k: [] for k in Ks}
    all_rec  = {k: [] for k in Ks}
    K_max = max(Ks)

    for batch in tqdm(dataloader, desc="eval", leave=False):
        images, _captions, target_dist, _indices = batch[:4]
        images      = images.to(device, non_blocking=True)
        target_dist = target_dist.to(device, non_blocking=True)  # [B, V]

        patch_tokens, _, _ = model.backbone(images)
        patch_tokens = F.normalize(patch_tokens, dim=-1)          # [B, N, D]

        # top-patch-mean scoring (same as train.py forward)
        sims = torch.einsum("bnd,vd->bnv", patch_tokens, prototypes)  # [B, N, V]
        vocab_logits = sims.topk(top_patches, dim=1).values.mean(dim=1)  # [B, V]

        pred_topk = vocab_logits.topk(K_max, dim=-1).indices   # [B, K_max]
        gt_binary = (target_dist > 0).float()                  # [B, V]

        for k in Ks:
            pred_k = pred_topk[:, :k]                          # [B, k]
            # build prediction indicator [B, V]
            pred_ind = torch.zeros_like(gt_binary)
            # scatter 1 at predicted positions
            pred_ind.scatter_(1, pred_k, 1.0)

            tp        = (pred_ind * gt_binary).sum(dim=-1)     # [B]
            precision = tp / k
            n_gt      = gt_binary.sum(dim=-1).clamp(min=1)
            recall    = tp / n_gt

            all_prec[k].extend(precision.cpu().tolist())
            all_rec[k].extend(recall.cpu().tolist())

    results: dict[str, float] = {}
    for k in Ks:
        results[f"P@{k}"]  = float(sum(all_prec[k]) / len(all_prec[k]))
        results[f"R@{k}"]  = float(sum(all_rec[k])  / len(all_rec[k]))
    return results


# --------------------------------------------------------------------------- #
# Dataset builder
# --------------------------------------------------------------------------- #

def build_vg_val(args, vocab_to_idx: dict) -> VisualGenomeDataset:
    return VisualGenomeDataset(
        vg_root=args.vg_root,
        region_descriptions_json=args.vg_region_descriptions,
        vocab_to_idx=vocab_to_idx,
        train=False,
        val_ratio=args.vg_val_ratio,
        seed=args.seed,
        target_type=args.target_mode,
        top_k_concepts=args.top_k_concepts,
    )


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser(
        description="Dedup-vocab ablation eval: dedup-trained model vs full vocab at inference"
    )
    parser.add_argument("--ckpt",              type=str, required=True)
    parser.add_argument("--train-vocab-cache", type=str, required=True,
                        help="Deduplicated vocab cache used during training")
    parser.add_argument("--eval-vocab-cache",  type=str, required=True,
                        help="Full vocab cache used for full-vocab inference eval")
    parser.add_argument("--mode", type=str, default="both",
                        choices=["dedup_only", "full_vocab", "both"])

    parser.add_argument("--vg-root",               type=str, required=True)
    parser.add_argument("--vg-region-descriptions", type=str, required=True)
    parser.add_argument("--vg-val-ratio",          type=float, default=0.1)
    parser.add_argument("--target-mode",           type=str,  default="topk",
                        choices=["prob", "binary", "topk"])
    parser.add_argument("--top-k-concepts",        type=int,  default=5)

    parser.add_argument("--batch-size",   type=int, default=64)
    parser.add_argument("--num-workers",  type=int, default=8)
    parser.add_argument("--seed",         type=int, default=42)
    parser.add_argument("--device",       type=str, default="cuda")
    parser.add_argument("--top-patches",  type=int, default=5,
                        help="Patches per concept for scoring (default: 5)")

    parser.add_argument("--wandb-project",  type=str, default="proto-non-param")
    parser.add_argument("--wandb-run-name", type=str, default="dedup-ablation-eval")
    parser.add_argument("--wandb-entity",   type=str, default=None)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ------------------------------------------------------------------ #
    # Load model
    # ------------------------------------------------------------------ #
    model, hparams = build_model(args.ckpt, device)

    # ------------------------------------------------------------------ #
    # Load vocab caches
    # ------------------------------------------------------------------ #
    print(f"Loading train vocab (dedup): {args.train_vocab_cache}")
    train_cache = torch.load(args.train_vocab_cache, map_location="cpu")
    print(f"  {len(train_cache)} concepts")

    print(f"Loading eval vocab (full):   {args.eval_vocab_cache}")
    eval_cache = torch.load(args.eval_vocab_cache, map_location="cpu")
    print(f"  {len(eval_cache)} concepts")

    # ------------------------------------------------------------------ #
    # W&B
    # ------------------------------------------------------------------ #
    wandb.init(
        project=args.wandb_project,
        name=args.wandb_run_name,
        entity=args.wandb_entity,
        config={
            **vars(args),
            "train_vocab_size": len(train_cache),
            "eval_vocab_size":  len(eval_cache),
        },
    )

    results_table = wandb.Table(columns=["mode", "vocab_size", "P@5", "R@5", "P@10", "R@10"])

    modes_to_run = {
        "dedup_only": train_cache,
        "full_vocab":  eval_cache,
    }
    if args.mode == "dedup_only":
        modes_to_run = {"dedup_only": train_cache}
    elif args.mode == "full_vocab":
        modes_to_run = {"full_vocab": eval_cache}

    for mode_label, vocab_cache in modes_to_run.items():
        print(f"\n{'='*60}")
        print(f"Mode: {mode_label}  (vocab size: {len(vocab_cache)})")
        print(f"{'='*60}")

        # Build vocab_to_idx for dataset ground truth
        vocab_to_idx = {w: i for i, w in enumerate(vocab_cache.keys())}

        dataset = build_vg_val(args, vocab_to_idx)
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
            collate_fn=coco_clip_collate_fn,
        )
        print(f"Dataset: {len(dataset)} val images, {len(loader)} batches")

        # Build prototypes for this vocab
        words, prototypes = build_prototypes(model, vocab_cache, device)
        print(f"Prototypes: {prototypes.shape}")

        # Evaluate
        metrics = eval_retrieval(
            model, loader, prototypes, device,
            Ks=(5, 10),
            top_patches=args.top_patches,
        )

        print(f"Results [{mode_label}]:")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")
            wandb.log({f"{mode_label}/{k}": v})

        results_table.add_data(
            mode_label,
            len(vocab_cache),
            metrics["P@5"],
            metrics["R@5"],
            metrics["P@10"],
            metrics["R@10"],
        )

    wandb.log({"results_summary": results_table})

    # Print comparison if both modes ran
    if args.mode == "both":
        print("\n=== Summary ===")
        print(f"{'Mode':<15} {'V size':>8}  {'P@5':>7}  {'R@5':>7}  {'P@10':>7}  {'R@10':>7}")
        for row in results_table.data:
            print(f"{row[0]:<15} {row[1]:>8}  {row[2]:>7.4f}  {row[3]:>7.4f}  {row[4]:>7.4f}  {row[5]:>7.4f}")

    wandb.finish()
    print("\nDone.")


if __name__ == "__main__":
    main()
