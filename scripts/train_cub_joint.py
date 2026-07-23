#!/usr/bin/env python3
"""
Standalone, SEPARATE pipeline: JOINT training of PNP's CUB-200 concept
encoder + a classifier head, with a CLIP-substituted "sufficiency"
regularizer (adapted from Espinosa Zarlenga, "In Defense of Information
Leakage in Concept-based Models", ICML 2026).

*** CAVEAT -- report this alongside any result from this script: ***
The paper's sufficiency regularizer L_int is defined against GROUND-TRUTH
concept labels. We have none -- CUB's GPT-3-generated concept phrases were
never verified by a human, only CLIP-similarity-scored. This script
substitutes CLIP's own raw image-vs-concept similarity for that ground
truth. This is a deliberate WEAKENING of what the paper proves, not a
reproduction of it -- results are an adapted, weaker-guarantee version of
the mechanism.

Unlike the existing Sequential Stage-2 pipeline (train.py --dataset
cub200, fine-tune the encoder only, then scripts/fit_sparse_cub_probe.py
as a separate post-hoc classifier fit), this script trains the concept
encoder (KL + SK + KoLeo, same formulas as Stage 2) and a classifier head
TOGETHER end to end, matching Koh et al.'s definition of "Joint" CBM
training.

Deliberately kept separate from the shared pipeline: does NOT modify
train.py, modeling/pnp.py, or clip_dataset.py. Imports read-only utility
functions from those files, but owns its own dataset, training loop, and
CLI.

Usage:
  python scripts/train_cub_joint.py \
    --init-ckpt $SCRATCH/train_logs/vg_contrastive/run_M1_vitl14_sk10_koleo01_30ep/ckpt.pth \
    --vocab-cache-path $SCRATCH/vocab/cub_clip_scores_vocab_filtered.pt \
    --clip-scores-cub $SCRATCH/vocab/cub_clip_scores.pt \
    --cub-root $SCRATCH/cub200 --cub-annotations $SCRATCH/cub200/annotations \
    --cls-coef 1.0 --sufficiency-coef 1.0 \
    --log-dir $SCRATCH/train_logs/cub_joint
"""

import argparse
import json
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, SCRIPT_DIR)

from modeling.pnp import PNP, sinkhorn_knopp, koleo_loss              # noqa: E402
from evaluate_pnp_refer import build_backbone                          # noqa: E402
from evaluate_pnp_cub_concepts import build_class_index, list_split    # noqa: E402
from clip_dataset import build_cub_path_to_id                          # noqa: E402

import open_clip  # noqa: E402


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# M1's fixed hyperparameters. This script is specifically an M1-warm-start
# tool, not a generic multi-checkpoint one -- M1's own checkpoint hparams
# point at the VG vocab, not the CUB vocab this script needs, so they can't
# be read off the checkpoint the way evaluate_pnp_refer.py's build_model()
# does; hardcoding the known values is simpler than re-deriving them.
M1_BACKBONE = "dinov2_vitl14"
M1_NUM_SPLITS = 1
M1_TEMPERATURE = 0.2
M1_CLIP_TEXT_DIM = 512
M1_TEXT_PROJ_HIDDEN_DIM = 2048
M1_PROTOTYPE_INIT_NOISE = 0.01
M1_AGG_MODE = "topk"
M1_TOPK_K = 5


class CubJointDataset(Dataset):
    """CUB image -> (KL target, raw CLIP score, species label).

    Standalone: does not wrap or subclass CUBCLIPDataset / CLIPScoreDataset,
    per the decision to keep this pipeline fully separate from the shared
    training code. Species labels come free from list_split()'s
    (path, class_index) tuples -- no new label-parsing logic needed.
    """

    def __init__(self, cub_root, cub_annotations, clip_scores_path, class_to_idx,
                 splits, train, top_k=50, temperature=0.07):
        self.train = train
        samples = []
        for split in splits:
            samples.extend(list_split(cub_root, split, class_to_idx))
        self.samples = samples  # list of (path, species_label)

        data = torch.load(clip_scores_path, map_location="cpu")
        image_ids = data["image_ids"]
        raw_scores = data["clip_scores"].float()  # [N, V]
        id_to_row = {img_id: i for i, img_id in enumerate(image_ids)}

        # build_cub_path_to_id only covers train/val (it exists to support
        # CLIPScoreDataset, which only ever wraps training data) -- for a
        # test-split instance every lookup below misses and falls back to a
        # zero vector, which is fine: the eval loop never reads kl_target or
        # raw_score, only vocab_logits from a forward pass.
        path_to_id = build_cub_path_to_id(cub_root, cub_annotations)

        # Precompute the processed (top-k masked, temperature-softmax'd) KL
        # target the same way CLIPScoreDataset does (clip_dataset.py:1046-1055),
        # and keep the raw row too -- that's the sufficiency term's input.
        if top_k < raw_scores.shape[1]:
            topk_vals, topk_idx = raw_scores.topk(top_k, dim=-1)
            masked = torch.full_like(raw_scores, float("-inf"))
            masked.scatter_(-1, topk_idx, topk_vals)
        else:
            masked = raw_scores
        soft_labels = torch.softmax(masked / temperature, dim=-1)

        self.kl_target = []
        self.raw_score = []
        missed = 0
        for path, _ in self.samples:
            row = None
            img_id = path_to_id.get(path)
            if img_id is not None:
                row = id_to_row.get(img_id)
            if row is None:
                missed += 1
                self.kl_target.append(torch.zeros(raw_scores.shape[1]))
                self.raw_score.append(torch.zeros(raw_scores.shape[1]))
            else:
                self.kl_target.append(soft_labels[row])
                self.raw_score.append(raw_scores[row])
        if missed:
            print(f"CubJointDataset({'train' if train else 'eval'}): "
                  f"{missed}/{len(self.samples)} images had no CLIP score match "
                  f"(expected for test-split instances; concept-alignment terms "
                  f"aren't used at eval time)")

        if train:
            self.transform = v2.Compose([
                v2.RandomResizedCrop(size=224, scale=(0.8, 1.0), interpolation=v2.InterpolationMode.BICUBIC),
                v2.RandomHorizontalFlip(p=0.5),
                v2.ToTensor(),
                v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ])
        else:
            self.transform = v2.Compose([
                v2.Resize((224, 224), interpolation=v2.InterpolationMode.BICUBIC),
                v2.ToTensor(),
                v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]
        img = Image.open(path).convert("RGB")
        img_tensor = self.transform(img)
        return img_tensor, self.kl_target[index], self.raw_score[index], label


def build_pnp(vocab_cache_path, device):
    hparams = argparse.Namespace(backbone=M1_BACKBONE, num_splits=M1_NUM_SPLITS)
    backbone, dim = build_backbone(hparams)
    clip_model, _, _ = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
    clip_model = clip_model.eval().to(device)
    for p in clip_model.parameters():
        p.requires_grad = False

    net = PNP(
        backbone=backbone,
        dim=dim,
        temperature=M1_TEMPERATURE,
        clip_text_dim=M1_CLIP_TEXT_DIM,
        text_proj_hidden_dim=M1_TEXT_PROJ_HIDDEN_DIM,
        vocab_cache_path=vocab_cache_path,
        prototype_init_noise=M1_PROTOTYPE_INIT_NOISE,
        clip_model=clip_model,
        agg_mode=M1_AGG_MODE,
        topk_k=M1_TOPK_K,
    )
    return net.to(device)


def warm_start(net, init_ckpt_path):
    """Same shape-filtered load pattern as train.py's --init-ckpt handling
    (train.py:906-923): vocab-sized buffers (prototype_residual,
    vocab_clip_embeddings) don't transfer since the CUB vocab differs in
    size/identity from M1's VG vocab; everything else (backbone, projection
    head) does. prototype_residual is intentionally left out of the
    optimizer below, matching the existing Stage-2 script's convention
    (residual stays frozen at its fresh random init)."""
    init_state = torch.load(init_ckpt_path, map_location="cpu")["state_dict"]
    own_state = net.state_dict()
    compatible = {k: v for k, v in init_state.items()
                  if k in own_state and own_state[k].shape == v.shape}
    skipped = sorted(set(init_state.keys()) - set(compatible.keys()))
    if skipped:
        print(f"Warm-start: skipped shape-mismatched keys (fresh init kept): {skipped}")
    missing, _ = net.load_state_dict(compatible, strict=False)
    if missing:
        print(f"Warm-start: missing keys (fresh init): {missing}")


def standardize(x, eps=1e-6):
    """Per-sample z-score across the concept dimension. Required before feeding
    vocab_logits and raw CLIP scores through the same classifier head -- they
    live in different embedding spaces (patch-cosine-similarity in visual
    space vs. CLIP image-vs-text cosine similarity) and are not otherwise on
    comparable scales; without this, swapping one for the other is a
    distribution-shift crash, not a meaningful "intervention"."""
    mean = x.mean(dim=-1, keepdim=True)
    std = x.std(dim=-1, keepdim=True)
    return (x - mean) / (std + eps)


def compute_losses(net, cls_head, images, kl_target, raw_score, species_label, args):
    outputs = net(images)
    vocab_logits = outputs["vocab_logits"]  # [B, V]

    losses = {}

    # Concept-alignment losses: identical formulas to PNPCriterion
    # (modeling/pnp.py:533-544, 647-652, 655-657), duplicated here rather than
    # depending on PNPCriterion, per the decision to keep this pipeline
    # separate from the shared training code.
    if args.kl_coef != 0:
        target = kl_target.clamp_min(1e-8)
        target = target / (target.sum(dim=-1, keepdim=True) + 1e-8)
        pred_log_probs = F.log_softmax(vocab_logits / net.temperature, dim=-1)
        losses["l_kl"] = args.kl_coef * F.kl_div(pred_log_probs, target, reduction="batchmean")

    if args.sk_coef != 0:
        Q = sinkhorn_knopp(vocab_logits.detach(), eps=args.sk_eps, n_iter=args.sk_n_iter)
        log_P = F.log_softmax(vocab_logits / net.temperature, dim=-1)
        losses["l_sk"] = args.sk_coef * (-(Q * log_P).sum(dim=-1).mean())

    if args.koleo_coef != 0:
        losses["l_koleo"] = args.koleo_coef * koleo_loss(outputs["pred_text_embedding"])

    # Classification loss, from the model's own concept activations.
    if args.cls_coef != 0:
        losses["l_cls"] = args.cls_coef * F.cross_entropy(
            cls_head(standardize(vocab_logits)), species_label
        )

    # Adapted sufficiency term (L_int): ground truth swapped for CLIP's raw
    # similarity score -- see module docstring caveat.
    if args.sufficiency_coef != 0:
        losses["l_sufficiency"] = args.sufficiency_coef * F.cross_entropy(
            cls_head(standardize(raw_score)), species_label
        )

    return losses


@torch.no_grad()
def evaluate(net, cls_head, loader, device):
    net.eval()
    correct1 = correct5 = total = 0
    for images, _, _, labels in loader:
        images, labels = images.to(device), labels.to(device)
        vocab_logits = net(images)["vocab_logits"]
        logits = cls_head(standardize(vocab_logits))
        top5 = logits.topk(5, dim=-1).indices
        correct1 += (top5[:, 0] == labels).sum().item()
        correct5 += (top5 == labels.unsqueeze(1)).any(dim=1).sum().item()
        total += labels.shape[0]
    net.train()
    return correct1 / total, correct5 / total


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--init-ckpt", required=True, help="M1 base checkpoint to warm-start from")
    p.add_argument("--vocab-cache-path", required=True, help="Filtered CUB concept vocab cache")
    p.add_argument("--clip-scores-cub", required=True, help="build_clip_vocab_scores.py output for CUB")
    p.add_argument("--cub-root", required=True)
    p.add_argument("--cub-annotations", required=True)
    p.add_argument("--log-dir", required=True)
    p.add_argument("--num-classes", type=int, default=200)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--backbone-lr", type=float, default=1e-5)
    p.add_argument("--text-proj-lr", type=float, default=1e-4)
    p.add_argument("--cls-lr", type=float, default=1e-3)
    p.add_argument("--kl-coef", type=float, default=1.0)
    p.add_argument("--sk-coef", type=float, default=0.1)
    p.add_argument("--sk-eps", type=float, default=0.10)
    p.add_argument("--sk-n-iter", type=int, default=3)
    p.add_argument("--koleo-coef", type=float, default=0.1)
    p.add_argument("--cls-coef", type=float, default=1.0)
    p.add_argument("--sufficiency-coef", type=float, default=1.0)
    p.add_argument("--clip-scores-top-k", type=int, default=50)
    p.add_argument("--clip-scores-temperature", type=float, default=0.07)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--save-every", type=int, default=5)
    args = p.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    os.makedirs(args.log_dir, exist_ok=True)

    print("*** CAVEAT: this run's sufficiency_coef term substitutes CLIP similarity "
          "for ground-truth concept labels -- see module docstring. ***")

    class_to_idx = build_class_index(args.cub_root)
    print(f"{len(class_to_idx)} CUB classes")

    train_ds = CubJointDataset(args.cub_root, args.cub_annotations, args.clip_scores_cub,
                               class_to_idx, splits=("train", "val"), train=True,
                               top_k=args.clip_scores_top_k, temperature=args.clip_scores_temperature)
    test_ds = CubJointDataset(args.cub_root, args.cub_annotations, args.clip_scores_cub,
                              class_to_idx, splits=("test",), train=False,
                              top_k=args.clip_scores_top_k, temperature=args.clip_scores_temperature)
    print(f"Train: {len(train_ds)}   Test: {len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True, drop_last=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True)

    net = build_pnp(args.vocab_cache_path, device)
    warm_start(net, args.init_ckpt)
    n_concepts = net.vocab_size
    cls_head = nn.Linear(n_concepts, args.num_classes).to(device)
    print(f"Concept vocab: {n_concepts} concepts -> classifier head -> {args.num_classes} classes")

    optimizer = torch.optim.AdamW([
        {"params": net.backbone.parameters(), "lr": args.backbone_lr},
        {"params": net.text_projection_head.parameters(), "lr": args.text_proj_lr},
        {"params": cls_head.parameters(), "lr": args.cls_lr},
    ])

    net.train()
    for epoch in range(args.epochs):
        epoch_losses = {}
        for images, kl_target, raw_score, species_label in tqdm(train_loader, desc=f"Epoch {epoch}"):
            images = images.to(device)
            kl_target = kl_target.to(device)
            raw_score = raw_score.to(device)
            species_label = species_label.to(device)

            losses = compute_losses(net, cls_head, images, kl_target, raw_score, species_label, args)
            loss = sum(losses.values())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            for k, v in losses.items():
                epoch_losses[k] = epoch_losses.get(k, 0.0) + v.item()

        n_batches = len(train_loader)
        loss_str = "  ".join(f"{k}={v / n_batches:.4f}" for k, v in epoch_losses.items())
        print(f"Epoch {epoch}: {loss_str}")

        ckpt_payload = {
            "state_dict": {k: v.detach().cpu() for k, v in net.state_dict().items()},
            "cls_head_state_dict": {k: v.detach().cpu() for k, v in cls_head.state_dict().items()},
            "hparams": vars(args),
        }
        torch.save(ckpt_payload, os.path.join(args.log_dir, "ckpt.pth"))
        if args.save_every > 0 and (epoch + 1) % args.save_every == 0:
            torch.save(ckpt_payload, os.path.join(args.log_dir, f"ckpt_ep{epoch + 1:03d}.pth"))

    top1, top5 = evaluate(net, cls_head, test_loader, device)
    print(f"\nTest top-1: {100 * top1:.2f}%   top-5: {100 * top5:.2f}%")
    print("*** CAVEAT: sufficiency_coef used CLIP similarity as a weak, unverified "
          "substitute for ground-truth concept labels -- not a reproduction of the "
          "paper's mechanism, an adapted, weaker-guarantee version of it. ***")

    result = {
        "init_ckpt": args.init_ckpt,
        "vocab_cache_path": args.vocab_cache_path,
        "n_concepts": n_concepts,
        "n_classes": args.num_classes,
        "n_train": len(train_ds),
        "n_test": len(test_ds),
        "cls_coef": args.cls_coef,
        "sufficiency_coef": args.sufficiency_coef,
        "top1_acc": round(100 * top1, 4),
        "top5_acc": round(100 * top5, 4),
        "caveat": ("sufficiency_coef term uses CLIP image-vs-concept similarity as a weak, "
                   "unverified substitute for the ground-truth concept labels the paper's "
                   "L_int regularizer is defined against -- this is an adapted, "
                   "weaker-guarantee version of that mechanism, not a reproduction."),
    }
    with open(os.path.join(args.log_dir, "result.json"), "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved result to {os.path.join(args.log_dir, 'result.json')}")


if __name__ == "__main__":
    main()
