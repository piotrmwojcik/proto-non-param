from math import isqrt
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn


class ProjectionHead(nn.Module):
    """
    Project a word embedding into the visual feature space.

    LayerNorm is used instead of BatchNorm so the module also works reliably
    with a batch size of one.
    """

    def __init__(
        self,
        input_dim: int = 512,
        hidden_dim: int = 1024,
        output_dim: int = 768,
        normalize_output: bool = True,
    ) -> None:
        super().__init__()

        self.normalize_output = normalize_output

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [..., input_dim]

        Returns:
            Projected tensor with shape [..., output_dim].
        """
        x = self.net(x)

        if self.normalize_output:
            x = F.normalize(x, p=2, dim=-1)

        return x


class PNP(nn.Module):
    """
    Image-word patch similarity model.

    The model receives:

        images:
            Image batch with shape [B, 3, H, W].

        word_embedding:
            One embedding per image with shape [B, text_dim], or one shared
            embedding with shape [text_dim].

    It returns a cosine-similarity map between the projected word embedding and
    every spatial image feature.

    No vocabulary, prototype dictionary, or cached word embeddings are used.
    """

    def __init__(
        self,
        backbone: nn.Module,
        *,
        visual_dim: int = 768,
        text_dim: int = 512,
        projection_hidden_dim: int = 1024,
        temperature: float = 1.0,
        remove_cls_token: bool = False,
    ) -> None:
        super().__init__()

        if temperature <= 0:
            raise ValueError("temperature must be greater than zero")

        self.backbone = backbone
        self.visual_dim = visual_dim
        self.text_dim = text_dim
        self.temperature = temperature
        self.remove_cls_token = remove_cls_token

        self.text_projection_head = ProjectionHead(
            input_dim=text_dim,
            hidden_dim=projection_hidden_dim,
            output_dim=visual_dim,
            normalize_output=True,
        )

    def _extract_patch_tokens(
        self,
        images: torch.Tensor,
    ) -> torch.Tensor:
        """
        Extract patch tokens from the visual backbone.

        This supports backbones returning either:

            patch_tokens

        or:

            (patch_tokens, auxiliary_output_1, auxiliary_output_2)
        """
        backbone_output = self.backbone(images)

        if isinstance(backbone_output, torch.Tensor):
            patch_tokens = backbone_output
        elif isinstance(backbone_output, (tuple, list)):
            if not backbone_output:
                raise ValueError("The backbone returned an empty tuple/list")

            patch_tokens = backbone_output[0]
        elif isinstance(backbone_output, dict):
            patch_tokens = None

            for key in (
                "patch_tokens",
                "tokens",
                "features",
                "x",
            ):
                value = backbone_output.get(key)

                if isinstance(value, torch.Tensor):
                    patch_tokens = value
                    break

            if patch_tokens is None:
                raise ValueError(
                    "Could not find patch tokens in the backbone output "
                    "dictionary"
                )
        else:
            raise TypeError(
                "Unsupported backbone output type: "
                f"{type(backbone_output)!r}"
            )

        if patch_tokens.ndim != 3:
            raise ValueError(
                "Expected patch tokens with shape [B, N, D], "
                f"but received {tuple(patch_tokens.shape)}"
            )

        if patch_tokens.shape[-1] != self.visual_dim:
            raise ValueError(
                f"Expected visual feature dimension {self.visual_dim}, "
                f"but received {patch_tokens.shape[-1]}"
            )

        if self.remove_cls_token:
            if patch_tokens.shape[1] < 2:
                raise ValueError(
                    "Cannot remove the CLS token because the backbone returned "
                    "fewer than two tokens"
                )

            patch_tokens = patch_tokens[:, 1:, :]

        return patch_tokens

    @staticmethod
    def _resolve_spatial_size(
        number_of_patches: int,
        spatial_size: Optional[tuple[int, int]],
    ) -> tuple[int, int]:
        """
        Determine the height and width of the patch feature map.
        """
        if spatial_size is not None:
            height, width = spatial_size

            if height <= 0 or width <= 0:
                raise ValueError(
                    "spatial_size values must be positive"
                )

            if height * width != number_of_patches:
                raise ValueError(
                    f"spatial_size={spatial_size} contains "
                    f"{height * width} locations, but the backbone returned "
                    f"{number_of_patches} patch tokens"
                )

            return height, width

        side = isqrt(number_of_patches)

        if side * side != number_of_patches:
            raise ValueError(
                f"The backbone returned {number_of_patches} patch tokens, "
                "which cannot be automatically reshaped into a square map. "
                "Pass spatial_size=(height, width) to forward()."
            )

        return side, side

    @staticmethod
    def _prepare_word_embedding(
        word_embedding: torch.Tensor,
        batch_size: int,
    ) -> torch.Tensor:
        """
        Convert the word embedding to shape [B, text_dim].

        A one-dimensional embedding is shared across the whole image batch.
        A batch containing one embedding is also expanded when needed.
        """
        if word_embedding.ndim == 1:
            word_embedding = word_embedding.unsqueeze(0)

        if word_embedding.ndim != 2:
            raise ValueError(
                "word_embedding must have shape [text_dim] or "
                f"[B, text_dim], but received {tuple(word_embedding.shape)}"
            )

        word_batch_size = word_embedding.shape[0]

        if word_batch_size == 1 and batch_size > 1:
            word_embedding = word_embedding.expand(batch_size, -1)
        elif word_batch_size != batch_size:
            raise ValueError(
                f"Image batch size is {batch_size}, but the word embedding "
                f"batch size is {word_batch_size}"
            )

        return word_embedding

    def forward(
        self,
        images: torch.Tensor,
        word_embedding: torch.Tensor,
        *,
        spatial_size: Optional[tuple[int, int]] = None,
        output_size: Optional[tuple[int, int]] = None,
    ) -> torch.Tensor:
        """
        Compute image-word cosine similarity.

        Args:
            images:
                Image tensor with shape [B, 3, H, W].

            word_embedding:
                Word embedding with shape [text_dim] or [B, text_dim].

            spatial_size:
                Optional patch-grid shape ``(patch_height, patch_width)``.
                This is only required when the number of patch tokens is not
                a perfect square.

            output_size:
                Optional final similarity-map size. When supplied, the patch
                similarity map is bilinearly interpolated to this resolution.
                For example, use ``images.shape[-2:]`` to obtain a map at the
                input image resolution.

        Returns:
            Similarity map with shape [B, 1, patch_height, patch_width], or
            [B, 1, output_height, output_width] when output_size is given.
        """
        if images.ndim != 4:
            raise ValueError(
                "images must have shape [B, 3, H, W], "
                f"but received {tuple(images.shape)}"
            )

        patch_tokens = self._extract_patch_tokens(images)
        batch_size, number_of_patches, _ = patch_tokens.shape

        word_embedding = self._prepare_word_embedding(
            word_embedding,
            batch_size=batch_size,
        )

        if word_embedding.shape[-1] != self.text_dim:
            raise ValueError(
                f"Expected word embedding dimension {self.text_dim}, "
                f"but received {word_embedding.shape[-1]}"
            )

        # Ensure that the word embedding uses the same device and floating-point
        # type as the visual features.
        word_embedding = word_embedding.to(
            device=patch_tokens.device,
            dtype=patch_tokens.dtype,
        )

        # [B, N, D]
        patch_tokens = F.normalize(
            patch_tokens,
            p=2,
            dim=-1,
        )

        # [B, D]
        projected_word = self.text_projection_head(word_embedding)
        projected_word = F.normalize(
            projected_word,
            p=2,
            dim=-1,
        )

        # Cosine similarity between every patch and the corresponding word.
        #
        # [B, N, D] * [B, 1, D] -> [B, N]
        patch_similarity = torch.sum(
            patch_tokens * projected_word.unsqueeze(1),
            dim=-1,
        )

        patch_similarity = patch_similarity / self.temperature

        patch_height, patch_width = self._resolve_spatial_size(
            number_of_patches,
            spatial_size,
        )

        # [B, N] -> [B, 1, patch_height, patch_width]
        similarity_map = patch_similarity.reshape(
            batch_size,
            1,
            patch_height,
            patch_width,
        )

        if output_size is not None:
            similarity_map = F.interpolate(
                similarity_map,
                size=output_size,
                mode="bilinear",
                align_corners=False,
            )

        return similarity_map


class PNPCriterion(nn.Module):
    """
    Matches predicted noun distribution to the target noun distribution from the dataset.
    Also optionally regularizes prototypes to stay visually aligned with image patches.
    """
    def __init__(
        self,
        kl_coef: float = 1.0,
        bin_coef: float = 0.1,
        entropy_coef: float = 0.0,
        visual_coef: float = 0.0,
        cover_coef: float = 0.0,
        temperature: float = 0.07,
    ) -> None:
        super().__init__()
        self.kl_coef = kl_coef
        self.bin_coef = bin_coef
        self.entropy_coef = entropy_coef
        self.visual_coef = visual_coef
        self.cover_coef = cover_coef
        self.temperature = temperature

    def forward(self, outputs: dict[str, torch.Tensor], batch: tuple[torch.Tensor, ...], model):
        vocab_logits = outputs["vocab_logits"]              # [B, V]
        mixture_weights = outputs["mixture_weights"]        # [B, V]
        patch_logits = outputs["patch_prototype_logits"]
        target_dist = batch[1]
        captions = batch[-1]

        loss_dict = {}

        # 1) distribution matching: target noun distribution vs predicted noun distribution
        target_dist = target_dist / (target_dist.sum(dim=-1, keepdim=True) + 1e-8)
        pred_log_probs = F.log_softmax(vocab_logits / self.temperature, dim=-1)

        # b = 0
        #
        # target = target_dist[b]
        # pred = pred_log_probs[b].exp()  # convert log-prob → prob
        #
        # # top tokens in target distribution
        # topk_vals, topk_idx = target.topk(10)
        #
        # print("\n========== SAMPLE DEBUG ==========")
        #
        # print("\nAll captions:")
        # for c in captions[b]:
        #     print(" ", c)
        #
        # print("\nTop target tokens vs prediction:")
        # print(f"{'token':15s} {'target':>10s} {'pred':>10s} {'diff':>10s}")
        #
        # for idx in topk_idx.tolist():
        #     token = model.vocab_words[idx]
        #     t = target[idx].item()
        #     p = pred[idx].item()
        #     diff = p - t
        #
        #     print(f"{token:15s} {t:10.6f} {p:10.6f} {diff:10.6f}")
        #
        # # also show model's top predictions
        # pred_vals, pred_idx = pred.topk(10)
        #
        # print("\nTop predicted tokens:")
        # print(f"{'token':15s} {'pred':>10s} {'target':>10s}")
        #
        # for idx in pred_idx.tolist():
        #     token = model.vocab_words[idx]
        #     p = pred[idx].item()
        #     t = target[idx].item()
        #
        #     print(f"{token:15s} {p:10.6f} {t:10.6f}")
        #
        # print("==================================\n")

        l_kl = F.kl_div(
            pred_log_probs,
            target_dist,
            reduction="batchmean",
        )
        loss_dict["l_dist"] = self.kl_coef * l_kl

        # -----------------------------------
        # binary supervision from target_dist
        # -----------------------------------
        #target_binary = (target_dist > 1e-6).float()

        #l_bin = F.binary_cross_entropy_with_logits(
        #    gate_logits,
        #    target_binary,
        #    reduction="mean",
        #)

        #loss_dict["l_bin"] = self.bin_coef * l_bin

        # 2) optional entropy regularization on predicted distribution
        if self.entropy_coef != 0:
            entropy = -(mixture_weights * torch.log(mixture_weights + 1e-8)).sum(dim=-1).mean()
            loss_dict["l_entropy"] = self.entropy_coef * entropy

        # 3) optional visual similarity: learned prototype mixture should match some patches
        if self.visual_coef != 0:
            patch_tokens = outputs["patch_tokens"]          # [B, N, D]
            prototypes = outputs["prototypes"]              # [V, D]

            proto_mix = F.normalize(mixture_weights @ prototypes, dim=-1)   # [B, D]
            patch_sims = torch.einsum("bd,bnd->bn", proto_mix, patch_tokens)  # [B, N]

            k = min(5, patch_sims.shape[1])
            topk_vals = patch_sims.topk(k=k, dim=1).values
            l_visual = 1.0 - topk_vals.mean()

            loss_dict["l_visual"] = self.visual_coef * l_visual

        # 4) optional coverage: selected prototype mixture should explain at least one patch well
        if self.cover_coef != 0:
            patch_scores = torch.einsum("bnv,bv->bn", patch_logits, mixture_weights)  # [B, N]
            l_cover = -patch_scores.max(dim=1).values.mean()
            loss_dict["l_cover"] = self.cover_coef * l_cover

        return loss_dict