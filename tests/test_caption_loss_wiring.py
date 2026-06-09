"""Synthetic unit tests for the caption-level loss wiring.

Tests:
  1. vg_collate_fn stacks 5-tuple batches correctly
  2. PNPCriterion(caption_coef=1.0) computes a finite scalar l_caption
  3. loss.backward() runs without error (gradient flows)
  4. Two __getitem__ calls on the same index in train mode produce different caption_embs
     when pool size > caption_sample_k  (stochastic sampling is active)
  5. caption_coef=0.0 produces no l_caption key in loss_dict
  6. --caption-coef without --caption-embeds-path triggers argparse error
"""

import sys
from pathlib import Path
import types
import torch
import torch.nn.functional as F

# Make sure the package root is on the path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from clip_dataset import vg_collate_fn, VisualGenomeDataset
from modeling.pnp import PNPCriterion


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _fake_outputs(B: int, V: int) -> dict:
    vocab_logits = torch.randn(B, V, requires_grad=True)
    weights = F.softmax(vocab_logits / 0.2, dim=-1)
    vocab_clip_embs = F.normalize(torch.randn(V, 512), dim=-1)
    pred_text = F.normalize((weights.detach() @ vocab_clip_embs), dim=-1)
    return {
        "vocab_logits": vocab_logits,
        "mixture_weights": weights,
        "patch_tokens": torch.randn(B, 16, 768),
        "patch_prototype_logits": torch.randn(B, 16, V),
        "pred_text_embedding": pred_text,
        "prototypes": F.normalize(torch.randn(V, 768), dim=-1),
    }


def _fake_batch_5tuple(B: int, V: int):
    images = torch.randn(B, 3, 224, 224)
    target_dist = torch.rand(B, V)
    indices = torch.arange(B)
    captions = [f"dummy phrase {i}" for i in range(B)]
    caption_embs = F.normalize(torch.randn(B, 512), dim=-1)
    # criterion receives (images, target_dist, indices, captions, caption_embs)
    return images, target_dist, indices, captions, caption_embs


# ---------------------------------------------------------------------------
# test 1: collate
# ---------------------------------------------------------------------------

def test_vg_collate_fn_5tuple():
    B = 4
    img_h, img_w = 224, 224
    V = 100
    items = [
        (
            torch.randn(3, img_h, img_w),
            f"phrase {i}",
            torch.rand(V),
            i,
            F.normalize(torch.randn(512), dim=-1),
        )
        for i in range(B)
    ]
    images, phrases, prob_dists, indices, caption_embs = vg_collate_fn(items)
    assert images.shape == (B, 3, img_h, img_w), f"images shape {images.shape}"
    assert prob_dists.shape == (B, V), f"prob_dists shape {prob_dists.shape}"
    assert indices.shape == (B,), f"indices shape {indices.shape}"
    assert caption_embs.shape == (B, 512), f"caption_embs shape {caption_embs.shape}"
    assert len(phrases) == B
    print("PASS test_vg_collate_fn_5tuple")


# ---------------------------------------------------------------------------
# test 2 & 3: PNPCriterion with caption_coef=1.0 — scalar + backward
# ---------------------------------------------------------------------------

def test_pnp_criterion_caption_loss_forward_backward():
    B, V = 8, 500
    criterion = PNPCriterion(kl_coef=1.0, caption_coef=1.0)
    outputs = _fake_outputs(B, V)
    batch = _fake_batch_5tuple(B, V)

    loss_dict = criterion(outputs, batch, model=None)
    assert "l_caption" in loss_dict, f"missing l_caption in {list(loss_dict.keys())}"
    total = sum(loss_dict.values())
    assert total.shape == (), f"loss should be scalar, got {total.shape}"
    assert torch.isfinite(total), f"loss is non-finite: {total}"

    total.backward()
    assert outputs["vocab_logits"].grad is not None, "no gradient on vocab_logits"
    print(f"PASS test_pnp_criterion_caption_loss_forward_backward  "
          f"(l_caption={loss_dict['l_caption'].item():.4f})")


# ---------------------------------------------------------------------------
# test 4: caption_coef=0.0 → no l_caption key
# ---------------------------------------------------------------------------

def test_pnp_criterion_no_caption_when_coef_zero():
    B, V = 4, 200
    criterion = PNPCriterion(kl_coef=1.0, caption_coef=0.0)
    outputs = _fake_outputs(B, V)
    batch = _fake_batch_5tuple(B, V)

    loss_dict = criterion(outputs, batch, model=None)
    assert "l_caption" not in loss_dict, \
        f"l_caption should not appear with caption_coef=0.0, got {list(loss_dict.keys())}"
    print("PASS test_pnp_criterion_no_caption_when_coef_zero")


# ---------------------------------------------------------------------------
# test 5: stochastic sampling — same index yields different caption_embs
# ---------------------------------------------------------------------------

def test_getitem_stochastic_sampling():
    """VisualGenomeDataset.__getitem__ must return different caption_embs across
    calls when pool_size > caption_sample_k (i.e. randperm is active)."""
    import tempfile, os
    from PIL import Image

    # Create a tiny dummy dataset structure
    with tempfile.TemporaryDirectory() as tmp:
        # One fake JPEG
        im_path = os.path.join(tmp, "img0.jpg")
        Image.new("RGB", (64, 64), color=(128, 64, 32)).save(im_path)

        # Build a pool of 20 phrase embeddings (>> caption_sample_k=5)
        pool = F.normalize(torch.randn(20, 512), dim=-1)
        caption_embeds = {im_path: pool}

        # Minimal vocab (just need a dict to satisfy __init__)
        vocab_to_idx = {"dog": 0, "cat": 1, "tree": 2}

        # Patch samples directly so we don't need the full VG JSON
        ds = VisualGenomeDataset.__new__(VisualGenomeDataset)
        ds.train = True
        ds.caption_embeds = caption_embeds
        ds.caption_sample_k = 5
        from torchvision import transforms
        t = transforms.Compose([transforms.Resize((64, 64)), transforms.ToTensor()])
        ds.train_transform = t
        ds.eval_transform = t
        # Single sample tuple (im_path, phrases, prob_dist)
        prob_dist = torch.zeros(len(vocab_to_idx))
        prob_dist[0] = 1.0
        ds.samples = [(im_path, ["a dog", "a cat"], prob_dist)]

        # Draw two samples from the same index
        result_a = ds[0]
        result_b = ds[0]
        emb_a, emb_b = result_a[4], result_b[4]

        assert emb_a.shape == (512,), f"wrong shape {emb_a.shape}"
        # With pool=20 and k=5 the probability of drawing the exact same 5 indices
        # twice is 1/C(20,5) ≈ 1/15504 — effectively zero for a test
        assert not torch.allclose(emb_a, emb_b), \
            "caption_embs are identical across two calls — stochastic sampling not active"
        print("PASS test_getitem_stochastic_sampling")


# ---------------------------------------------------------------------------
# test 6: argparse guard
# ---------------------------------------------------------------------------

def test_train_argparse_guard():
    """--caption-coef != 0 without --caption-embeds-path must call parser.error."""
    import subprocess, sys
    train_py = Path(__file__).resolve().parent.parent / "train.py"
    # Pass minimal required args + caption-coef without caption-embeds-path
    result = subprocess.run(
        [sys.executable, str(train_py),
         "--caption-coef", "1.0",
         "--dataset", "visual_genome",
         "--vg-root", "/nonexistent",
         "--region-descriptions-json", "/nonexistent"],
        capture_output=True, text=True,
    )
    combined = result.stdout + result.stderr
    assert result.returncode != 0, "train.py should have exited with error"
    assert "caption-embeds-path" in combined, \
        f"Expected caption-embeds-path in error message, got:\n{combined}"
    print("PASS test_train_argparse_guard")


# ---------------------------------------------------------------------------
# runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    test_vg_collate_fn_5tuple()
    test_pnp_criterion_caption_loss_forward_backward()
    test_pnp_criterion_no_caption_when_coef_zero()
    test_getitem_stochastic_sampling()
    test_train_argparse_guard()
    print("\nAll tests passed.")
