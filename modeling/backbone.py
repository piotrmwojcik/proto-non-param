from functools import partial
from math import sqrt

import open_clip
import torch
from dinov2.layers.block import Block, MemEffAttention
from dinov2.models.vision_transformer import (
    DinoVisionTransformer as Dinov2VisionTransformer,
)
from einops import rearrange
from torch import nn

from .utils import append_blocks, block_expansion_dino


DINOV2_COMMON_KWARGS = {
    "img_size": 518,
    "patch_size": 14,
    "mlp_ratio": 4,
    "init_values": 1.0,
    "ffn_layer": "mlp",
    "block_chunks": 0,
    "num_register_tokens": 4,
    "interpolate_antialias": True,
    "interpolate_offset": 0.0,
    "block_fn": partial(Block, attn_class=MemEffAttention),
}

DINO_COMMON_KWARGS = {
    "num_classes": 0,
    "mlp_ratio": 4,
    "qkv_bias": True,
    "norm_layer": partial(nn.LayerNorm, eps=1e-6),
}

VIT_SMALL_KWARGS = {
    "embed_dim": 384,
    "num_heads": 6,
}

VIT_BASE_KWARGS = {
    "embed_dim": 768,
    "num_heads": 12,
}

MODEL_DICT = {
    "dinov2_vits14": partial(
        Dinov2VisionTransformer,
        **VIT_SMALL_KWARGS,
        **DINOV2_COMMON_KWARGS,
    ),
    "dinov2_vitb14": partial(
        Dinov2VisionTransformer,
        **VIT_BASE_KWARGS,
        **DINOV2_COMMON_KWARGS,
    ),
}

URL_DICT = {
    "dinov2_vits14": (
        "https://dl.fbaipublicfiles.com/dinov2/"
        "dinov2_vits14/dinov2_vits14_reg4_pretrain.pth"
    ),
    "dinov2_vitb14": (
        "https://dl.fbaipublicfiles.com/dinov2/"
        "dinov2_vitb14/dinov2_vitb14_reg4_pretrain.pth"
    ),
}

DIM_DICT = {
    "dinov2_vits14": 384,
    "dinov2_vitb14": 768,
}

CLIP_NAME_MAP = {
    "clip_vitb32": "ViT-B-32",
    "clip_vitb16": "ViT-B-16",
    "clip_vitl14": "ViT-L-14",
}

CLIP_DIM_DICT = {
    "clip_vitb32": 768,
    "clip_vitb16": 768,
    "clip_vitl14": 1024,
}


def _validate_name(
    name: str,
    mapping: dict[str, object],
    kind: str,
) -> None:
    if name not in mapping:
        supported = ", ".join(sorted(mapping))
        raise ValueError(
            f"Unsupported {kind} {name!r}. "
            f"Supported values: {supported}"
        )


def _load_dinov2(name: str) -> nn.Module:
    """
    Construct DINOv2 directly and load register-model pretrained weights.

    This avoids torch.hub.load(), whose hubconf.py can import the wrong
    installed ``dinov2`` package and fail on optional modules such as
    ``dinov2.hub.cell_dino``.
    """
    _validate_name(name, MODEL_DICT, "DINOv2 backbone")

    model = MODEL_DICT[name](depth=12)

    state_dict = torch.hub.load_state_dict_from_url(
        URL_DICT[name],
        map_location="cpu",
        check_hash=False,
    )

    model.load_state_dict(state_dict, strict=True)
    return model


class CLIPBackbone(nn.Module):
    def __init__(
        self,
        name: str = "clip_vitb32",
        pretrained: str = "openai",
    ) -> None:
        super().__init__()
        _validate_name(name, CLIP_NAME_MAP, "CLIP backbone")

        model_name = CLIP_NAME_MAP[name]
        model, _, _ = open_clip.create_model_and_transforms(
            model_name,
            pretrained=pretrained,
        )

        self.clip = model.visual
        self.dim = CLIP_DIM_DICT[name]

    def learnable_parameters(self):
        return self.clip.parameters()

    def set_requires_grad(self) -> None:
        for parameter in self.parameters():
            parameter.requires_grad = True

    def forward(
        self,
        x: torch.Tensor,
        key: str = "x_norm_patchtokens",
        cls_key: str = "x_norm_clstoken",
        reshape: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if key != "x_norm_patchtokens":
            raise ValueError(
                f"Unsupported key for CLIPBackbone: {key}"
            )

        if cls_key != "x_norm_clstoken":
            raise ValueError(
                f"Unsupported cls_key for CLIPBackbone: {cls_key}"
            )

        x = self.clip.conv1(x)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)

        cls = self.clip.class_embedding.to(x.dtype)
        cls = cls + torch.zeros(
            x.shape[0],
            1,
            x.shape[-1],
            dtype=x.dtype,
            device=x.device,
        )
        x = torch.cat([cls, x], dim=1)

        x = x + self.clip.positional_embedding.to(x.dtype)
        x = self.clip.ln_pre(x)

        x = x.permute(1, 0, 2)
        x = self.clip.transformer(x)
        x = x.permute(1, 0, 2)

        x = self.clip.ln_post(x)

        cls_token = x[:, 0, :]
        feature = x[:, 1:, :]

        if reshape:
            _, number_of_patches, _ = feature.shape
            height = width = int(sqrt(number_of_patches))

            if height * width != number_of_patches:
                raise ValueError(
                    "CLIP patch tokens cannot be reshaped into "
                    f"a square feature map: {number_of_patches} patches."
                )

            feature = rearrange(
                feature,
                "b (h w) d -> b d h w",
                h=height,
                w=width,
            )

        # Preserve the three-output interface expected elsewhere.
        return feature, feature, cls_token


class DINOv2Backbone(nn.Module):
    def __init__(
        self,
        name: str = "dinov2_vitb14",
    ) -> None:
        super().__init__()
        _validate_name(name, MODEL_DICT, "DINOv2 backbone")

        self.dino = _load_dinov2(name)
        self.dim = DIM_DICT[name]

    def learnable_parameters(self):
        return self.dino.parameters()

    def set_requires_grad(self) -> None:
        for parameter in self.parameters():
            parameter.requires_grad = True

    def forward(
        self,
        x: torch.Tensor,
        key: str = "x_norm_patchtokens",
        cls_key: str = "x_norm_clstoken",
        reshape: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        feature_dict: dict[str, torch.Tensor] = (
            self.dino.forward_features(x)
        )

        if key not in feature_dict:
            raise KeyError(
                f"Missing DINOv2 output key {key!r}. "
                f"Available keys: {sorted(feature_dict)}"
            )

        if cls_key not in feature_dict:
            raise KeyError(
                f"Missing DINOv2 output key {cls_key!r}. "
                f"Available keys: {sorted(feature_dict)}"
            )

        feature = feature_dict[key]
        cls_token = feature_dict[cls_key]

        if reshape:
            _, number_of_patches, _ = feature.shape
            height = width = int(sqrt(number_of_patches))

            if height * width != number_of_patches:
                raise ValueError(
                    "DINOv2 patch tokens cannot be reshaped into "
                    f"a square feature map: {number_of_patches} patches."
                )

            feature = rearrange(
                feature,
                "b (h w) d -> b d h w",
                h=height,
                w=width,
            )

        return feature, cls_token


class DINOv2BackboneExpanded(nn.Module):
    def __init__(
        self,
        name: str = "dinov2_vitb14",
        n_splits: int = 0,
        mode: str = "block_expansion",
        freeze_norm_layer: bool = True,
    ) -> None:
        super().__init__()
        _validate_name(name, MODEL_DICT, "DINOv2 backbone")

        if mode not in {"block_expansion", "append"}:
            raise ValueError(
                f"Unsupported mode {mode!r}. "
                "Expected 'block_expansion' or 'append'."
            )

        if n_splits < 0:
            raise ValueError("n_splits must be non-negative")

        self.dim = DIM_DICT[name]

        expansion_function = (
            block_expansion_dino
            if mode == "block_expansion"
            else append_blocks
        )

        if n_splits > 0:
            architecture = MODEL_DICT[name]

            state_dict = torch.hub.load_state_dict_from_url(
                URL_DICT[name],
                map_location="cpu",
                check_hash=False,
            )

            (
                expanded_state_dict,
                number_of_blocks,
                learnable_parameter_names,
                _zero_parameter_names,
            ) = expansion_function(
                state_dict=state_dict,
                n_splits=n_splits,
                freeze_layer_norm=freeze_norm_layer,
            )

            self.dino = architecture(depth=number_of_blocks)
            self.dino.load_state_dict(
                expanded_state_dict,
                strict=True,
            )
            self.learnable_param_names = set(
                learnable_parameter_names
            )
        else:
            self.dino = _load_dinov2(name)
            self.learnable_param_names = set()

        self.set_requires_grad()

    def learnable_parameters(self):
        return [
            parameter
            for parameter_name, parameter in self.dino.named_parameters()
            if parameter_name in self.learnable_param_names
        ]

    def set_requires_grad(self) -> None:
        for parameter_name, parameter in self.dino.named_parameters():
            parameter.requires_grad = (
                parameter_name in self.learnable_param_names
            )

    def forward(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.dino.prepare_tokens_with_masks(x)

        original_feature: torch.Tensor | None = None

        for index, block in enumerate(self.dino.blocks):
            x = block(x)

            if index == len(self.dino.blocks) - 2:
                original_feature = self.dino.norm(x)

        x = self.dino.norm(x)

        if original_feature is None:
            raise RuntimeError(
                "Could not capture the penultimate DINOv2 block output."
            )

        patch_start = self.dino.num_register_tokens + 1

        return (
            x[:, patch_start:, :],
            original_feature[:, patch_start:, :],
            x[:, 0, :],
        )


class DINOBackboneExpanded(nn.Module):
    """
    Legacy DINO backbone.

    The current dictionaries contain only DINOv2 models. Add a legacy DINO
    architecture, URL, and dimension before using this class.
    """

    def __init__(
        self,
        name: str = "dino_vitb16",
        n_splits: int = 0,
        mode: str = "block_expansion",
        freeze_norm_layer: bool = True,
    ) -> None:
        super().__init__()

        if (
            name not in MODEL_DICT
            or name not in URL_DICT
            or name not in DIM_DICT
        ):
            raise ValueError(
                f"Legacy DINO backbone {name!r} is not configured. "
                "Add it to MODEL_DICT, URL_DICT, and DIM_DICT first."
            )

        if mode not in {"block_expansion", "append"}:
            raise ValueError(
                f"Unsupported mode {mode!r}. "
                "Expected 'block_expansion' or 'append'."
            )

        if n_splits < 0:
            raise ValueError("n_splits must be non-negative")

        self.dim = DIM_DICT[name]

        expansion_function = (
            block_expansion_dino
            if mode == "block_expansion"
            else append_blocks
        )

        architecture = MODEL_DICT[name]

        state_dict = torch.hub.load_state_dict_from_url(
            URL_DICT[name],
            map_location="cpu",
            check_hash=False,
        )

        if n_splits > 0:
            (
                expanded_state_dict,
                number_of_blocks,
                learnable_parameter_names,
                _zero_parameter_names,
            ) = expansion_function(
                state_dict=state_dict,
                n_splits=n_splits,
                freeze_layer_norm=freeze_norm_layer,
            )

            self.dino = architecture(depth=number_of_blocks)
            self.dino.load_state_dict(
                expanded_state_dict,
                strict=True,
            )
            self.learnable_param_names = set(
                learnable_parameter_names
            )
        else:
            self.dino = architecture(depth=12)
            self.dino.load_state_dict(
                state_dict,
                strict=True,
            )
            self.learnable_param_names = set()

        self.set_requires_grad()

    def learnable_parameters(self):
        return [
            parameter
            for parameter_name, parameter in self.dino.named_parameters()
            if parameter_name in self.learnable_param_names
        ]

    def set_requires_grad(self) -> None:
        for parameter_name, parameter in self.dino.named_parameters():
            parameter.requires_grad = (
                parameter_name in self.learnable_param_names
            )

    def forward_with_original_feature(
        self,
        x: torch.Tensor,
        return_attn: bool = False,
    ):
        x = self.dino.prepare_tokens(x)

        original_feature: torch.Tensor | None = None
        attention_maps = []

        for index, block in enumerate(self.dino.blocks):
            if return_attn:
                x, attention = block(
                    x,
                    return_attn=True,
                )
                attention_maps.append(attention)
            else:
                x = block(x)

            if index == 11:
                original_feature = self.dino.norm(x)

        x = self.dino.norm(x)

        if original_feature is None:
            raise RuntimeError(
                "Could not capture the original DINO feature at block 11."
            )

        if return_attn:
            return (
                x[:, 1:],
                original_feature[:, 1:],
                x[:, 0, :],
                attention_maps,
            )

        return (
            x[:, 1:],
            original_feature[:, 1:],
            x[:, 0, :],
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.dino.prepare_tokens(x)

        for block in self.dino.blocks:
            x = block(x)

        x = self.dino.norm(x)
        return x[:, 1:], x[:, 0, :]