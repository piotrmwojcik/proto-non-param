#!/usr/bin/env python3
"""
Zero-shot open-vocabulary grounding demo on Visual Genome images (qualitative,
no ground truth, no training -- inference only). Analogous to CTRL-O's
Fig. 3: ground arbitrary free-text phrases, not confined to RefCOCO's
referring-expression style or vocabulary, to show the open-vocabulary
mechanism working directly on the model's own training-distribution images.

Mechanism: identical to evaluate_pnp_refer.py's per-example scoring (CLIP-
encode the phrase -> project through text_projection_head -> cosine
similarity with DINOv2 patch tokens -> heatmap). Unlike
visualize_concept_retrieval.py, this script does NOT call PNP.forward()
(which unconditionally also runs the image through CLIP's own fixed-224px
image encoder) -- it manually calls net.backbone(...) and
net.clip_model.encode_text(...) separately, same as evaluate_pnp_refer.py's
eval loop. So --img-size is free to be anything (e.g. 672px, M1's headline
resolution), not locked to 224.

Rendering: a thresholded SALIENCY MASK, not a bounding box -- a rectangle
implies a precise, confident localization even when the true signal is
diffuse or the concept is absent; a heat-colored mask (only pixels whose
normalized activation clears --mask-threshold, same 0.5 default
evaluate_pnp_refer.py uses for the actual segmentation decision, everything
else dimmed) is more honest about what the model actually computed.

Layout per image: original (no overlay) first, then one masked panel per
phrase in --phrases.

Images: sampled directly from the VG image shards (VG_100K/VG_100K_2) by
globbing filenames -- no region_descriptions.json needed, since this is
purely qualitative and doesn't use any VG annotations at all.

Usage:
  python scripts/visualize_vg_open_vocab_grounding.py \
    --ckpt $SCRATCH/train_logs/vg_contrastive/run_M1_vitl14_sk10_koleo01_30ep/ckpt.pth \
    --vg-root $SCRATCH/vg --n-images 6 --img-size 672 \
    --out-dir results/vg_open_vocab_grounding

  # One specific hand-picked image (e.g. a paper teaser) instead of a
  # random sample -- --vg-root not needed in this mode:
  python scripts/visualize_vg_open_vocab_grounding.py \
    --ckpt ... --image path/to/2365136.jpg --phrases "a person wearing a hat" "the sky" \
    --out-dir results/teaser
"""

import argparse
import math
import os
import re
import random
import sys
from glob import glob

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, SCRIPT_DIR)

from evaluate_pnp_refer import build_model, build_img_transform  # noqa: E402
from eval_retreive_concepts import overlay_heatmap  # noqa: E402

DEFAULT_PHRASES = [
    "a person wearing a hat",
    "a dog",
    "the sky",
    "a smile",
]


def sample_vg_images(vg_root, n_images, seed):
    paths = sorted(glob(os.path.join(vg_root, "VG_100K", "*.jpg")) +
                   glob(os.path.join(vg_root, "VG_100K_2", "*.jpg")))
    if not paths:
        raise RuntimeError(f"No images found under {vg_root}/VG_100K{{,_2}}/*.jpg")
    rng = random.Random(seed)
    return rng.sample(paths, min(n_images, len(paths)))


def saliency_mask_overlay(img_uint8, hm_up, threshold, alpha=0.6, dim_factor=0.35):
    """Heat-colored overlay only on patches whose normalized activation
    clears `threshold` (same min-max-normalize + threshold convention
    evaluate_pnp_refer.py uses for the real segmentation decision) --
    everywhere else is dimmed, not boxed. A bounding box implies a precise,
    confident localization even when the model's true signal is diffuse or
    the queried concept is absent from the image; a soft, thresholded mask
    is a more honest rendering of what was actually computed."""
    activation = hm_up.detach().cpu().numpy()
    a_min, a_max = activation.min(), activation.max()
    if a_max > a_min:
        activation = (activation - a_min) / (a_max - a_min + 1e-8)
    mask = activation >= threshold

    heat = overlay_heatmap(img_uint8, hm_up, alpha=alpha)
    dimmed = (img_uint8.astype(np.float32) * dim_factor).astype(np.uint8)
    return np.where(mask[..., None], heat, dimmed).astype(np.uint8)


def jet_heatmap_overlay(img_uint8, hm_up, alpha=0.5):
    """Classic Grad-CAM-style continuous overlay: full 'jet' colormap
    (blue=low activation, red=high) alpha-blended over the whole image,
    no thresholding or dimming. Punchier/more standard-looking than
    saliency_mask_overlay, at the cost of that style's explicit "below
    threshold, don't trust this" signal -- use for hero/teaser figures,
    not the honesty-focused main experimental ones."""
    activation = hm_up.detach().cpu().numpy()
    a_min, a_max = activation.min(), activation.max()
    if a_max > a_min:
        activation = (activation - a_min) / (a_max - a_min + 1e-8)
    heat_rgb = (plt.get_cmap("jet")(activation)[..., :3] * 255).astype(np.uint8)
    out = alpha * heat_rgb.astype(np.float32) + (1 - alpha) * img_uint8.astype(np.float32)
    return out.clip(0, 255).astype(np.uint8)


def _slug(text):
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def save_single_panel(img_array, title, out_path, fontsize=18):
    """One image, caption only -- no axes/frame, for dropping straight into a
    LaTeX figure with subcaptions instead of only having the combined grid."""
    fig, ax = plt.subplots(figsize=(4.5, 5), dpi=140)
    ax.imshow(img_array)
    ax.axis("off")
    ax.set_title(title, fontsize=fontsize, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--vg-root", default=None, help="Contains VG_100K/ and VG_100K_2/ shards; "
                   "required unless --image is given")
    p.add_argument("--image", type=str, default=None,
                   help="Use this exact image file instead of randomly sampling from "
                        "--vg-root -- for a specific hand-picked figure (e.g. a paper teaser).")
    p.add_argument("--n-images", type=int, default=6)
    p.add_argument("--phrases", type=str, nargs="*", default=DEFAULT_PHRASES,
                   help="Arbitrary free-text phrases, grounded independently on every sampled image")
    p.add_argument("--phrases-file", type=str, default=None,
                   help="One phrase per line; overrides --phrases if given. Safer than "
                        "passing multi-word phrases through shell/env-var quoting.")
    p.add_argument("--mask-threshold", type=float, default=0.5,
                   help="Same normalized-activation threshold evaluate_pnp_refer.py uses "
                        "for the real segmentation decision (--style mask only)")
    p.add_argument("--style", choices=("mask", "jet"), default="mask",
                   help="'mask' (default): thresholded saliency mask, dimmed below "
                        "--mask-threshold -- honest about 'not found', used for the main "
                        "experimental figures. 'jet': continuous Grad-CAM-style jet-colormap "
                        "overlay, no thresholding -- punchier, for hero/teaser figures.")
    p.add_argument("--overlay-alpha", type=float, default=None,
                   help="Overlay blend strength; defaults to 0.6 for mask style, 0.5 for jet")
    p.add_argument("--save-panels", action="store_true",
                   help="Additionally save each panel (original + each phrase) as its own "
                        "file with just a caption -- for dropping individual images into a "
                        "LaTeX figure, alongside the combined grid.")
    p.add_argument("--img-size", type=int, default=672)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--out-dir", required=True)
    args = p.parse_args()

    if args.phrases_file:
        with open(args.phrases_file, encoding="utf-8-sig") as f:
            args.phrases = [line.strip() for line in f if line.strip()]

    if not args.image and not args.vg_root:
        p.error("either --image or --vg-root is required")

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    assert args.img_size % 14 == 0, "--img-size must be a multiple of the ViT patch size (14)"

    print(f"Loading PNP model from {args.ckpt} ...")
    net, tokenizer, _ = build_model(args.ckpt, device)
    img_transform = build_img_transform(args.img_size)

    if args.image:
        image_paths = [args.image]
        print(f"Using single image: {args.image}")
    else:
        image_paths = sample_vg_images(args.vg_root, args.n_images, args.seed)
        print(f"Sampled {len(image_paths)} VG images")

    with torch.inference_mode():
        # Encode every phrase once, reused across all images.
        p_queries = {}
        for phrase in args.phrases:
            tokens = tokenizer([phrase]).to(device)
            e_query = net.clip_model.encode_text(tokens)
            e_query = F.normalize(e_query.float(), dim=-1)
            p_queries[phrase] = net.text_projection_head(e_query)  # [1, D]

        for img_path in image_paths:
            pil_img = Image.open(img_path).convert("RGB")
            img_t = img_transform(pil_img).unsqueeze(0).to(device)

            patch_tokens = net.backbone(img_t)[0]
            patch_tokens = F.normalize(patch_tokens, dim=-1)  # [1, N, D]

            img_display = np.array(pil_img.resize((args.img_size, args.img_size)))
            base = os.path.splitext(os.path.basename(img_path))[0]

            n_panels = 1 + len(args.phrases)
            fig, axes = plt.subplots(1, n_panels, figsize=(4 * n_panels, 4.5), dpi=140)

            axes[0].imshow(img_display)
            axes[0].axis("off")
            axes[0].set_title("Original", fontsize=16, fontweight="bold")
            if args.save_panels:
                save_single_panel(img_display, "Original",
                                  os.path.join(args.out_dir, f"grounding_{base}_original.png"))

            for ax, phrase in zip(axes[1:], args.phrases):
                scores = (patch_tokens * p_queries[phrase].unsqueeze(1)).sum(dim=-1)[0]  # [N]
                H = W = int(math.sqrt(scores.shape[0]))
                hm_up = F.interpolate(scores.view(1, 1, H, W), size=img_display.shape[:2],
                                      mode="bilinear", align_corners=False)[0, 0]

                if args.style == "jet":
                    masked = jet_heatmap_overlay(img_display, hm_up, alpha=args.overlay_alpha or 0.5)
                else:
                    masked = saliency_mask_overlay(img_display, hm_up, args.mask_threshold,
                                                   alpha=args.overlay_alpha or 0.6)

                ax.imshow(masked)
                ax.axis("off")
                ax.set_title(f'"{phrase}"', fontsize=16, fontweight="bold")
                if args.save_panels:
                    save_single_panel(masked, f'"{phrase}"',
                                      os.path.join(args.out_dir, f"grounding_{base}_{_slug(phrase)}.png"))

            # No on-image suptitle: the "zero-shot, no ground truth, heat vs.
            # dimmed" framing already lives in the paper's own figure caption.
            fig.tight_layout()

            out_path = os.path.join(args.out_dir, f"grounding_{base}.png")
            fig.savefig(out_path)
            plt.close(fig)
            print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
