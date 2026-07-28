"""CPU smoke test for modeling.backbone.DINOv2Backbone."""

import os

# Hide GPUs before importing PyTorch.
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import torch

from modeling.backbone import DINOv2Backbone


def main() -> None:
    device = torch.device("cpu")
    model_name = "dinov2_vitb14"
    image_size = 518
    patch_size = 14
    expected_grid_size = image_size // patch_size
    expected_patch_count = expected_grid_size**2
    expected_dim = 768

    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Device: {device}")
    print(f"Loading backbone: {model_name}")

    backbone = DINOv2Backbone(name=model_name).to(device)
    backbone.eval()

    assert backbone.dim == expected_dim, (
        f"Expected backbone.dim={expected_dim}, got {backbone.dim}"
    )

    images = torch.randn(
        1,
        3,
        image_size,
        image_size,
        device=device,
    )

    with torch.inference_mode():
        patch_tokens, cls_token = backbone(images)

    print(f"Patch-token shape: {tuple(patch_tokens.shape)}")
    print(f"CLS-token shape:   {tuple(cls_token.shape)}")

    assert patch_tokens.device.type == "cpu"
    assert cls_token.device.type == "cpu"
    assert patch_tokens.shape == (
        1,
        expected_patch_count,
        expected_dim,
    ), (
        "Unexpected patch-token shape: "
        f"expected {(1, expected_patch_count, expected_dim)}, "
        f"got {tuple(patch_tokens.shape)}"
    )
    assert cls_token.shape == (1, expected_dim), (
        "Unexpected CLS-token shape: "
        f"expected {(1, expected_dim)}, got {tuple(cls_token.shape)}"
    )
    assert torch.isfinite(patch_tokens).all()
    assert torch.isfinite(cls_token).all()

    with torch.inference_mode():
        feature_map, reshaped_cls_token = backbone(
            images,
            reshape=True,
        )

    print(f"Feature-map shape: {tuple(feature_map.shape)}")

    assert feature_map.shape == (
        1,
        expected_dim,
        expected_grid_size,
        expected_grid_size,
    ), (
        "Unexpected reshaped feature-map shape: "
        f"expected "
        f"{(1, expected_dim, expected_grid_size, expected_grid_size)}, "
        f"got {tuple(feature_map.shape)}"
    )
    assert reshaped_cls_token.shape == cls_token.shape
    assert torch.isfinite(feature_map).all()

    print("CPU backbone test passed.")


if __name__ == "__main__":
    main()

