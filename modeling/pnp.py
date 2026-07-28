from math import isqrt
from typing import Optional, Any

import torch
import torch.nn.functional as F
from torch import nn


class ProjectionHead(nn.Module):
    """
    Project a 4096-dimensional LLM2Vec/LLaMA-3-8B embedding into the
    visual feature space.

    LayerNorm is used instead of BatchNorm so the module works reliably
    with small batches, including a batch size of one.
    """

    def __init__(
        self,
        input_dim: int = 4096,
        hidden_dim: int = 256,
        output_dim: int = 768,
        normalize_output: bool = True,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.normalize_output = normalize_output

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:
                LLM2Vec embeddings with shape [..., 4096].

        Returns:
            Projected embeddings with shape [..., output_dim].
        """
        if x.shape[-1] != self.input_dim:
            raise ValueError(
                f"Expected input dimension {self.input_dim}, "
                f"but received tensor with shape {tuple(x.shape)}."
            )

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


class PCPCriterion(nn.Module):
    """
    Contrastive criterion for positive and negative PNP pairs.

    Expected outputs:
        positive_anchor_embeddings: [B, D]
        positive_text_embeddings:   [B, D]
        negative_anchor_embeddings: [B, D]
        negative_text_embeddings:   [B, D]

    The loss contains:

    1. Symmetric InfoNCE over positive anchor-text pairs.
    2. Binary contrastive supervision:
       - positive pair similarities should be high;
       - negative pair similarities should be low.
    3. Optional margin ranking:
       positive similarity should exceed negative similarity by `margin`.

    All embeddings are L2-normalized inside the criterion.
    """

    def __init__(
        self,
        infonce_coef: float = 1.0,
        binary_coef: float = 1.0,
        ranking_coef: float = 1.0,
        temperature: float = 0.07,
        margin: float = 0.2,
    ) -> None:
        super().__init__()

        if temperature <= 0:
            raise ValueError("temperature must be greater than zero.")

        if margin < 0:
            raise ValueError("margin must be non-negative.")

        self.infonce_coef = infonce_coef
        self.binary_coef = binary_coef
        self.ranking_coef = ranking_coef
        self.temperature = temperature
        self.margin = margin

    @staticmethod
    def _normalize_embeddings(
        embeddings: torch.Tensor,
        name: str,
    ) -> torch.Tensor:
        if embeddings.ndim != 2:
            raise ValueError(
                f"{name} must have shape [B, D], "
                f"but received {tuple(embeddings.shape)}."
            )

        return F.normalize(embeddings.float(), dim=-1)

    def forward(
        self,
        outputs: dict[str, torch.Tensor],
        batch: tuple[torch.Tensor, ...] | dict[str, Any] | None = None,
        model: nn.Module | None = None,
    ) -> dict[str, torch.Tensor]:
        del batch, model

        positive_anchor = self._normalize_embeddings(
            outputs["positive_anchor_embeddings"],
            "positive_anchor_embeddings",
        )
        positive_text = self._normalize_embeddings(
            outputs["positive_text_embeddings"],
            "positive_text_embeddings",
        )
        negative_anchor = self._normalize_embeddings(
            outputs["negative_anchor_embeddings"],
            "negative_anchor_embeddings",
        )
        negative_text = self._normalize_embeddings(
            outputs["negative_text_embeddings"],
            "negative_text_embeddings",
        )

        batch_size = positive_anchor.shape[0]

        expected_shape = positive_anchor.shape
        embedding_tensors = {
            "positive_text_embeddings": positive_text,
            "negative_anchor_embeddings": negative_anchor,
            "negative_text_embeddings": negative_text,
        }

        for name, tensor in embedding_tensors.items():
            if tensor.shape != expected_shape:
                raise ValueError(
                    "All pair embeddings must have the same shape. "
                    f"positive_anchor_embeddings has shape {expected_shape}, "
                    f"but {name} has shape {tuple(tensor.shape)}."
                )

        loss_dict: dict[str, torch.Tensor] = {}

        # -------------------------------------------------------------
        # 1) Symmetric InfoNCE over the positive anchor-text pairs.
        #
        # Row i should select positive_text[i].
        # Column i should select positive_anchor[i].
        # -------------------------------------------------------------
        if self.infonce_coef != 0:
            positive_logits = (
                positive_anchor @ positive_text.transpose(0, 1)
            ) / self.temperature

            labels = torch.arange(
                batch_size,
                device=positive_logits.device,
            )

            anchor_to_text_loss = F.cross_entropy(
                positive_logits,
                labels,
            )
            text_to_anchor_loss = F.cross_entropy(
                positive_logits.transpose(0, 1),
                labels,
            )

            infonce_loss = 0.5 * (
                anchor_to_text_loss + text_to_anchor_loss
            )

            loss_dict["l_infonce"] = (
                self.infonce_coef * infonce_loss
            )

        # Pairwise cosine similarities.
        positive_similarity = (
            positive_anchor * positive_text
        ).sum(dim=-1)

        negative_similarity = (
            negative_anchor * negative_text
        ).sum(dim=-1)

        # -------------------------------------------------------------
        # 2) Binary contrastive supervision.
        #
        # Cosine similarity is converted into a logit by temperature.
        # Positive pairs receive target 1 and negative pairs target 0.
        # -------------------------------------------------------------
        if self.binary_coef != 0:
            pair_logits = torch.cat(
                [
                    positive_similarity,
                    negative_similarity,
                ],
                dim=0,
            ) / self.temperature

            pair_targets = torch.cat(
                [
                    torch.ones_like(positive_similarity),
                    torch.zeros_like(negative_similarity),
                ],
                dim=0,
            )

            binary_loss = F.binary_cross_entropy_with_logits(
                pair_logits,
                pair_targets,
            )

            loss_dict["l_binary"] = (
                self.binary_coef * binary_loss
            )

        # -------------------------------------------------------------
        # 3) Pairwise margin ranking.
        #
        # Enforces:
        #     positive_similarity >= negative_similarity + margin
        # -------------------------------------------------------------
        if self.ranking_coef != 0:
            ranking_loss = F.relu(
                self.margin
                - positive_similarity
                + negative_similarity
            ).mean()

            loss_dict["l_ranking"] = (
                self.ranking_coef * ranking_loss
            )

        # Detached statistics are useful for logging, but should not be
        # included when blindly summing every value in loss_dict.
        loss_dict["positive_similarity"] = (
            positive_similarity.mean().detach()
        )
        loss_dict["negative_similarity"] = (
            negative_similarity.mean().detach()
        )

        return loss_dict