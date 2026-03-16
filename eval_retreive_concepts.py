#!/usr/bin/env python3
import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import open_clip
import torch
import torch.nn.functional as F
import wandb
from torch.utils.data import DataLoader

from clip_dataset import CocoCLIPDataset, coco_clip_collate_fn
from modeling.backbone import (
    DINOv2Backbone,
    DINOv2BackboneExpanded,
    DINOBackboneExpanded,
    CLIPBackbone,
)
from modeling.pnp import PNP


CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def denorm_to_uint8(x: torch.Tensor, mean=CLIP_MEAN, std=CLIP_STD) -> np.ndarray:
    x = x.detach().cpu()
    mean_t = torch.tensor(mean)[:, None, None]
    std_t = torch.tensor(std)[:, None, None]
    x = (x * std_t + mean_t).clamp(0, 1)
    x = (x * 255).byte().permute(1, 2, 0).numpy()
    return x


def build_backbone(hparams):
    backbone_name = hparams["backbone"]
    num_splits = hparams.get("num_splits", 0)

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
        dim = backbone.dim

    elif "dino" in backbone_name:
        backbone = DINOBackboneExpanded(
            name=backbone_name,
            n_splits=num_splits,
            mode="block_expansion",
            freeze_norm_layer=True,
        )
        dim = backbone.dim

    elif "clip" in backbone_name:
        backbone = CLIPBackbone(name=backbone_name)
        dim = backbone.dim

    else:
        raise NotImplementedError(f"Unsupported backbone: {backbone_name}")

    return backbone, dim


def make_model(ckpt_path: str, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    hparams = ckpt["hparams"]

    backbone, dim = build_backbone(hparams)

    clip_model, _, _ = open_clip.create_model_and_transforms(
        hparams["coco_clip_model_name"],
        pretrained=hparams["coco_clip_pretrained"],
    )
    clip_model = clip_model.eval().to(device)
    for p in clip_model.parameters():
        p.requires_grad = False

    model = PNP(
        backbone=backbone,
        dim=dim,
        temperature=hparams["temperature"],
        clip_text_dim=hparams["clip_text_dim"],
        text_proj_hidden_dim=hparams["text_proj_hidden_dim"],
        vocab_cache_path=hparams["vocab_cache_path"],
        prototype_init_noise=hparams["prototype_init_noise"],
        clip_model=clip_model,
    )

    model.load_state_dict(ckpt["state_dict"], strict=True)
    model = model.to(device).eval()
    return model, hparams


@torch.inference_mode()
def collect_scores(model, dataloader, device, concept_indices):
    """
    Returns:
        concept_to_examples: dict[idx] -> list of dict(score, image, captions, dataset_index)
    """
    concept_to_examples = {idx: [] for idx in concept_indices}

    for batch in dataloader:
        images, captions, _, indices = batch
        images = images.to(device, non_blocking=True)

        outputs = model(images)

        if "mixture_weights" in outputs:
            scores = outputs["mixture_weights"]   # [B, V]
        else:
            scores = outputs["vocab_logits"].softmax(dim=-1)

        scores = scores.detach().cpu()
        images_cpu = images.detach().cpu()
        indices_cpu = indices.detach().cpu()

        B = images_cpu.shape[0]
        for b in range(B):
            for cidx in concept_indices:
                concept_to_examples[cidx].append(
                    {
                        "score": float(scores[b, cidx].item()),
                        "image": images_cpu[b],
                        "captions": captions[b],
                        "dataset_index": int(indices_cpu[b].item()),
                    }
                )

    return concept_to_examples


def log_top5_with_boxes(model, concept_word, concept_idx, examples, device):
    """
    Logs top-5 images for a concept with activation boxes.
    """

    top5 = sorted(examples, key=lambda x: x["score"], reverse=True)[:5]

    fig, axes = plt.subplots(1, 5, figsize=(20, 4), dpi=140)

    for ax, ex in zip(axes, top5):

        img = ex["image"].unsqueeze(0).to(device)

        outputs = model(img)

        patch_logits = outputs["patch_prototype_logits"]  # [1, N, V]
        hm = patch_logits[0, :, concept_idx]

        H = W = int(math.sqrt(hm.shape[0]))
        hm = hm.view(1, 1, H, W)

        hm_up = F.interpolate(
            hm,
            size=(img.shape[-2], img.shape[-1]),
            mode="bilinear",
            align_corners=False,
        )[0, 0]

        hm_np = hm_up.detach().cpu().numpy()

        bbox = find_high_activation_crop(hm_np, percentile=95)

        img_uint8 = denorm_to_uint8(ex["image"])

        overlay = overlay_heatmap(img_uint8, hm_up, alpha=0.45)

        overlay_box = draw_rect_on_image(overlay, bbox)

        ax.imshow(overlay_box)
        ax.axis("off")

        caption = ex["captions"][0] if isinstance(ex["captions"], list) else str(ex["captions"])

        ax.set_title(
            f"{concept_word}\nscore={ex['score']:.3f}\n{caption[:40]}",
            fontsize=9
        )

    plt.tight_layout()

    wandb.log({
        f"retrieval/{concept_word}": wandb.Image(fig)
    })

    plt.close(fig)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--concepts", type=str, nargs="+", required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--wandb-project", type=str, default="proto-non-param")
    parser.add_argument("--wandb-run-name", type=str, default="concept-retrieval")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    model, hparams = make_model(args.ckpt, device)

    # vocab
    cache = torch.load(hparams["vocab_cache_path"], map_location="cpu")
    vocab_words = list(cache.keys())
    vocab_to_idx = {w: i for i, w in enumerate(vocab_words)}

    missing = [c for c in args.concepts if c not in vocab_to_idx]
    if missing:
        raise ValueError(f"Concepts not found in vocab: {missing}")

    concept_indices = [vocab_to_idx[c] for c in args.concepts]

    # training dataset only
    dataset_train = CocoCLIPDataset(
        annotations_json="/data/pwojcik/coco_2014/annotations/captions_train2014.json",
        coco_root="/data/pwojcik/coco_2014",
        vocab_to_idx=vocab_to_idx,
        train=False,
    )

    dataloader = DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=coco_clip_collate_fn,
    )

    wandb.init(
        project=args.wandb_project,
        name=args.wandb_run_name,
        config={
            "ckpt": args.ckpt,
            "concepts": args.concepts,
            "batch_size": args.batch_size,
        },
    )

    concept_to_examples = collect_scores(model, dataloader, device, concept_indices)

    for concept_word in args.concepts:
        log_top5_with_boxes(
            model,
            concept_word,
            vocab_to_idx[concept_word],
            concept_to_examples[vocab_to_idx[concept_word]],
            device,
        )

    wandb.finish()


if __name__ == "__main__":
    main()