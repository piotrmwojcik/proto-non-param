#!/usr/bin/env python3
import argparse
import math

import matplotlib.pyplot as plt
import numpy as np
import open_clip
import torch
import torch.nn.functional as F
import wandb
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader
from tqdm import tqdm

from clip_dataset import CocoCLIPDataset, coco_clip_collate_fn
from modeling.backbone import DINOv2BackboneExpanded
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


def overlay_heatmap(img_uint8: np.ndarray, hm: torch.Tensor, alpha: float = 0.45) -> np.ndarray:
    hm = hm.detach().cpu()
    hm = (hm - hm.min()) / (hm.max() - hm.min() + 1e-8)
    hm = hm.numpy()

    r = hm
    g = np.clip(hm * 0.9 + 0.1, 0, 1)
    b = np.clip(1.0 - hm * 0.8, 0, 1)
    hm_rgb = (np.stack([r, g, b], axis=-1) * 255).astype(np.uint8)

    out = alpha * hm_rgb.astype(np.float32) + (1 - alpha) * img_uint8.astype(np.float32)
    return out.clip(0, 255).astype(np.uint8)


def find_high_activation_crop(activation_map, percentile=95):
    threshold = np.percentile(activation_map, percentile)
    mask = activation_map >= threshold
    ys, xs = np.where(mask)

    if len(ys) == 0 or len(xs) == 0:
        h, w = activation_map.shape
        return 0, h, 0, w

    return ys.min(), ys.max() + 1, xs.min(), xs.max() + 1


def draw_rect_on_image(img_uint8, bbox, color=(255, 0, 0), width=3):
    y0, y1, x0, x1 = bbox
    img_pil = Image.fromarray(img_uint8)
    draw = ImageDraw.Draw(img_pil)
    draw.rectangle([x0, y0, x1 - 1, y1 - 1], outline=color, width=width)
    return np.array(img_pil)


def build_model(device: torch.device, ckpt_path: str):
    print(f"[INFO] Loading checkpoint from: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state_dict = ckpt["state_dict"]

    backbone = DINOv2BackboneExpanded(
        name="dinov2_vitb14",
        n_splits=1,
        mode="append",
        freeze_norm_layer=True,
    )
    dim = backbone.dim

    clip_model, _, _ = open_clip.create_model_and_transforms(
        "ViT-B-32",
        pretrained="openai",
    )
    clip_model = clip_model.eval().to(device)
    for p in clip_model.parameters():
        p.requires_grad = False

    model = PNP(
        backbone=backbone,
        dim=dim,
        temperature=0.07,
        clip_text_dim=512,
        text_proj_hidden_dim=768,
        vocab_cache_path="vocab/laion_clip_cache.pt",
        prototype_init_noise=0.01,
        clip_model=clip_model,
    )

    model.load_state_dict(state_dict, strict=True)
    model = model.to(device).eval()
    print("[INFO] Model ready")
    return model


@torch.inference_mode()
def collect_dataset_outputs(model, dataloader, device, concept_indices):
    """
    Single pass over dataset.

    Returns:
        all_scores:        [M, C]
        all_patch_logits:  [M, N, C]
        all_images:        list[tensor CxHxW]
        all_captions:      list[list[str] or str]
        all_dataset_idx:   [M]
    """
    score_chunks = []
    patch_chunks = []
    idx_chunks = []

    all_images = []
    all_captions = []

    print("[INFO] Running one retrieval pass over dataset")
    for batch in tqdm(dataloader, desc="Dataset pass", total=len(dataloader)):
        images, captions, _, indices = batch
        images = images.to(device, non_blocking=True)

        outputs = model(images)

        if "mixture_weights" in outputs:
            scores = outputs["mixture_weights"][:, concept_indices]   # [B, C]
        else:
            scores = outputs["vocab_logits"].softmax(dim=-1)[:, concept_indices]

        patch_logits = outputs["patch_prototype_logits"][:, :, concept_indices]  # [B, N, C]

        score_chunks.append(scores.detach().cpu())
        patch_chunks.append(patch_logits.detach().cpu())
        idx_chunks.append(indices.detach().cpu())

        all_images.extend([im.detach().cpu() for im in images])
        all_captions.extend(list(captions))

    all_scores = torch.cat(score_chunks, dim=0)          # [M, C]
    all_patch_logits = torch.cat(patch_chunks, dim=0)    # [M, N, C]
    all_dataset_idx = torch.cat(idx_chunks, dim=0)       # [M]

    print(f"[INFO] Finished dataset pass: {all_scores.shape[0]} images")
    return all_scores, all_patch_logits, all_images, all_captions, all_dataset_idx


def log_top5_with_boxes(concept_word, concept_col, topk_values, topk_indices, all_images, all_captions, all_dataset_idx, all_patch_logits):
    k = topk_indices.numel()
    fig, axes = plt.subplots(1, k, figsize=(4 * k, 4), dpi=140)
    if k == 1:
        axes = [axes]

    print(f"[INFO] Logging concept '{concept_word}' with top-{k} scores:",
          [round(float(v), 4) for v in topk_values.tolist()])

    for ax, score, global_idx in zip(axes, topk_values.tolist(), topk_indices.tolist()):
        img = all_images[global_idx]
        img_uint8 = denorm_to_uint8(img)

        hm = all_patch_logits[global_idx, :, concept_col]   # [N]
        N = hm.shape[0]
        H = W = int(math.sqrt(N))
        hm = hm.view(1, 1, H, W)

        Hi, Wi = img_uint8.shape[:2]
        hm_up = F.interpolate(hm, size=(Hi, Wi), mode="bilinear", align_corners=False)[0, 0]

        bbox = find_high_activation_crop(hm_up.numpy(), percentile=95)
        overlay = overlay_heatmap(img_uint8, hm_up, alpha=0.45)
        overlay_box = draw_rect_on_image(overlay, bbox)

        caption = all_captions[global_idx]
        if isinstance(caption, list):
            caption = caption[0] if len(caption) > 0 else ""

        dataset_idx = int(all_dataset_idx[global_idx].item())

        ax.imshow(overlay_box)
        ax.axis("off")
        ax.set_title(
            f"score={score:.3f}\nidx={dataset_idx}\n{str(caption)[:60]}",
            fontsize=9,
        )

    fig.suptitle(f"Top {k} images for concept: {concept_word}", fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.90])
    wandb.log({f"retrieval/{concept_word}": wandb.Image(fig)})
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--concepts", type=str, nargs="+", required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--wandb-project", type=str, default="proto-non-param")
    parser.add_argument("--wandb-run-name", type=str, default="concept-retrieval-boxes")
    parser.add_argument("--annotations-json", type=str, default="/data/pwojcik/coco_2014/annotations/captions_train2014.json")
    parser.add_argument("--coco-root", type=str, default="/data/pwojcik/coco_2014")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    vocab_cache_path = "vocab/laion_clip_cache.pt"
    print(f"[INFO] Loading vocab cache: {vocab_cache_path}")
    cache = torch.load(vocab_cache_path, map_location="cpu")
    vocab_words = list(cache.keys())
    vocab_to_idx = {w: i for i, w in enumerate(vocab_words)}

    missing = [c for c in args.concepts if c not in vocab_to_idx]
    if missing:
        raise ValueError(f"Concepts not found in vocab: {missing}")

    concept_indices = [vocab_to_idx[c] for c in args.concepts]
    print(f"[INFO] Concepts: {args.concepts}")
    print(f"[INFO] Concept indices: {concept_indices}")

    model = build_model(device, args.ckpt)

    print("[INFO] Building dataset")
    dataset = CocoCLIPDataset(
        annotations_json=args.annotations_json,
        coco_root=args.coco_root,
        vocab_to_idx=vocab_to_idx,
        train=False,
    )
    print(f"[INFO] Dataset size: {len(dataset)}")

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=coco_clip_collate_fn,
    )
    print(f"[INFO] Number of batches: {len(dataloader)}")

    wandb.init(
        project=args.wandb_project,
        name=args.wandb_run_name,
        config={
            "ckpt": args.ckpt,
            "concepts": args.concepts,
            "backbone": "dinov2_vitb14",
            "num_splits": 1,
            "vocab_cache_path": vocab_cache_path,
            "temperature": 0.07,
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
            "topk": args.topk,
        },
    )

    all_scores, all_patch_logits, all_images, all_captions, all_dataset_idx = collect_dataset_outputs(
        model=model,
        dataloader=dataloader,
        device=device,
        concept_indices=concept_indices,
    )

    # all_scores is [M, C], so topk per concept is vectorized
    print("[INFO] Computing top-k per concept")
    topk_values, topk_indices = torch.topk(all_scores, k=args.topk, dim=0)  # [K, C], [K, C]

    for concept_col, concept_word in enumerate(tqdm(args.concepts, desc="Logging concepts")):
        log_top5_with_boxes(
            concept_word=concept_word,
            concept_col=concept_col,
            topk_values=topk_values[:, concept_col],
            topk_indices=topk_indices[:, concept_col],
            all_images=all_images,
            all_captions=all_captions,
            all_dataset_idx=all_dataset_idx,
            all_patch_logits=all_patch_logits,
        )

    print("[INFO] Done")
    wandb.finish()


if __name__ == "__main__":
    main()