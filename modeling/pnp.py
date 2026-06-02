from math import sqrt

import torch
import torch.nn.functional as F
from einops import einsum
from torch import nn
import open_clip


class ProjectionHead(nn.Module):
    """
    SimCLR-style projection head.

    Maps input_dim -> hidden_dim -> output_dim
    with BN + ReLU in between.
    """
    def __init__(
        self,
        input_dim: int = 512,
        hidden_dim: int = 768,
        output_dim: int = 768,
        use_bn: bool = True,
        normalize_output: bool = True,
    ):
        super().__init__()
        self.normalize_output = normalize_output

        layers = [nn.Linear(input_dim, hidden_dim, bias=not use_bn)]
        if use_bn:
            layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(hidden_dim, output_dim, bias=True))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape
        x = x.reshape(-1, orig_shape[-1])
        x = self.net(x)

        if self.normalize_output:
            x = F.normalize(x, dim=-1)

        x = x.reshape(*orig_shape[:-1], -1)
        return x


class NonNegLinear(nn.Module):
    """Applies a linear transformation to the incoming data with non-negative weights`
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(NonNegLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        self.normalization_multiplier = nn.Parameter(torch.ones((1,),requires_grad=True))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)

    def forward(self, input):
        return F.linear(input,torch.relu(self.weight), self.bias)


class PNP(nn.Module):
    """
    Global prototype pool model.

    - One prototype per vocabulary item
    - Prototype pool size == vocab cache size
    - Reconstructs a CLIP text embedding as a soft mixture over vocab embeddings
    """
    def __init__(
        self,
        backbone: nn.Module,
        *,
        dim: int = 768,
        temperature: float = 0.2,
        clip_text_dim: int = 512,
        text_proj_hidden_dim: int = 1024,
        vocab_cache_path: str = "vocab/mscoco_new_cache.pt",
        prototype_init_noise: float = 0.01,
        clip_model = None
    ):
        super().__init__()
        self.backbone = backbone
        self.dim = dim
        self.temperature = temperature
        self.clip_text_dim = clip_text_dim
        self.prototype_init_noise = prototype_init_noise

        # CLIP image model used for visual gating / concept selection
        self.clip_model = clip_model
        if self.clip_model is not None:
            self.clip_model.eval()
            for p in self.clip_model.parameters():
                p.requires_grad = False

        # CLIP text space -> image / ViT feature space
        self.text_projection_head = ProjectionHead(
            input_dim=clip_text_dim,
            hidden_dim=text_proj_hidden_dim,
            output_dim=dim,
            use_bn=True,
            normalize_output=True,
        )

        # Load frozen vocab CLIP embeddings: dict[str, tensor(512)]
        cache = torch.load(vocab_cache_path, map_location="cpu")
        self.vocab_words = list(cache.keys())

        vocab_clip_embs = torch.stack([cache[w] for w in self.vocab_words], dim=0)  # [V, 512]
        vocab_clip_embs = F.normalize(vocab_clip_embs, dim=-1)

        self.register_buffer("vocab_clip_embeddings", vocab_clip_embs)  # [V, 512]
        self.vocab_size = vocab_clip_embs.shape[0]

        self.prototype_residual = nn.Parameter(
            torch.randn(self.vocab_size, self.clip_text_dim) * self.prototype_init_noise
        )

        #self.prototype_classifier = NonNegLinear(
        #    in_features=self.vocab_size,
        #    out_features=self.vocab_size,
        #    bias=True
        #)

    def get_prototypes(self) -> torch.Tensor:
        """
        Compute visual prototypes from frozen CLIP text embeddings plus
        a trainable residual, then project to visual space.
        """
        clip_proto = self.vocab_clip_embeddings + self.prototype_residual  # [V, 512]
        clip_proto = F.normalize(clip_proto, dim=-1)

        proto = self.text_projection_head(clip_proto)  # [V, D]
        proto = F.normalize(proto, dim=-1)
        return proto

    def get_prototypes_augmented(
        self,
        extra_clip_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Word prototypes (with trained residuals) optionally concatenated with
        caption prototypes (no residuals) for inference-time vocabulary augmentation.

        Args:
            extra_clip_embeds: [C, clip_text_dim] CLIP text embeddings of captions/
                phrases to append to the prototype pool. These are projected through
                the same text_projection_head but have no trained residual, since
                they were not part of training. Caller must ensure the model is in
                eval() mode so BatchNorm1d uses its running statistics.

        Returns:
            [V + C, D] or [V, D] (when extra_clip_embeds is None) unit-normalised
            visual prototypes.
        """
        clip_proto = self.vocab_clip_embeddings + self.prototype_residual  # [V, 512]
        clip_proto = F.normalize(clip_proto, dim=-1)

        if extra_clip_embeds is not None:
            extra = F.normalize(extra_clip_embeds.to(clip_proto.device), dim=-1)  # [C, 512]
            clip_proto = torch.cat([clip_proto, extra], dim=0)  # [V+C, 512]

        proto = self.text_projection_head(clip_proto)  # [V+C, D]
        proto = F.normalize(proto, dim=-1)
        return proto

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: [B, 3, H, W]

        Returns:
            dict with:
                patch_tokens: [B, N, D]
                patch_prototype_logits: [B, N, V]
                vocab_logits: [B, V]
                clip_vocab_logits: [B, V]
                mixture_weights: [B, V]
                pred_text_embedding: [B, 512]
                clip_image_embedding: [B, 512]
                prototypes: [V, D]
        """
        # -----------------------------------
        # Backbone patch features
        # -----------------------------------
        patch_tokens, _, _ = self.backbone(x)  # [B, N, D]
        patch_tokens = F.normalize(patch_tokens, p=2, dim=-1)

        prototypes = self.get_prototypes()  # [V, D]
        prototypes = F.normalize(prototypes, p=2, dim=-1)

        patch_prototype_logits = einsum(
            patch_tokens,
            prototypes,
            "B n_patches dim, V dim -> B n_patches V",
        )  # [B, N, V]

        # -----------------------------------
        # Image-level prototype logits
        # -----------------------------------
        k = 5
        topk_vals = patch_prototype_logits.topk(k, dim=1).values
        vocab_logits = topk_vals.mean(dim=1)
        #vocab_logits = self.prototype_classifier(vocab_logits)  # [B, V]

        # -----------------------------------
        # CLIP visual embedding -> vocab diagnostics
        # -----------------------------------
        with torch.no_grad():
            clip_image_embedding = self.clip_model.encode_image(x)  # [B, 512]
            clip_image_embedding = F.normalize(clip_image_embedding, p=2, dim=-1)

        vocab_clip_embeddings = F.normalize(self.vocab_clip_embeddings, p=2, dim=-1)  # [V, 512]

        clip_vocab_logits = einsum(
            clip_image_embedding,
            vocab_clip_embeddings,
            "B dim, V dim -> B V",
        )  # [B, V]

        # -----------------------------------
        # Use prototype logits only
        # -----------------------------------
        weights = F.softmax(vocab_logits / self.temperature, dim=-1)  # [B, V]

        pred_text_embedding = einsum(
            weights,
            vocab_clip_embeddings,
            "B V, V dim -> B dim",
        )  # [B, 512]
        pred_text_embedding = F.normalize(pred_text_embedding, p=2, dim=-1)

        # diagnostics: top CLIP words
        diag_k = 7

        outputs = dict(
            patch_tokens=patch_tokens,
            patch_prototype_logits=patch_prototype_logits,
            vocab_logits=vocab_logits,
            clip_vocab_logits=clip_vocab_logits,
            clip_gate_logits=None,
            mixture_weights=weights,
            pred_text_embedding=pred_text_embedding,
            clip_image_embedding=clip_image_embedding,
            prototypes=prototypes,
        )
        return outputs

    def push_forward(self, x: torch.Tensor):
        """
        Returns a spatial map over the vocab prototype pool.
        """
        patch_tokens, _, _ = self.backbone(x)
        patch_tokens = F.normalize(patch_tokens, p=2, dim=-1)
        prototypes = self.get_prototypes()  # [V, D]

        patch_prototype_logits = einsum(
            patch_tokens,
            prototypes,
            "B n_patches dim, V dim -> B n_patches V",
        )  # [B, N, V]

        _, n_patches, V = patch_prototype_logits.shape
        H = W = int(sqrt(n_patches))

        prototype_logits = patch_prototype_logits.permute(0, 2, 1).reshape(-1, V, H, W)
        pooled = F.avg_pool2d(prototype_logits, kernel_size=(2, 2), stride=2)
        return None, pooled


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
        use_binary: bool = False,
        bce_coef: float = 1.0,
        pos_weight_val: float = 100.0,
    ) -> None:
        super().__init__()
        self.kl_coef = kl_coef
        self.bin_coef = bin_coef
        self.entropy_coef = entropy_coef
        self.visual_coef = visual_coef
        self.cover_coef = cover_coef
        self.temperature = temperature
        self.use_binary = use_binary
        self.bce_coef = bce_coef
        self.pos_weight_val = pos_weight_val

    def forward(self, outputs: dict[str, torch.Tensor], batch: tuple[torch.Tensor, ...], model):
        vocab_logits = outputs["vocab_logits"]              # [B, V]
        mixture_weights = outputs["mixture_weights"]        # [B, V]
        patch_logits = outputs["patch_prototype_logits"]
        target_dist = batch[1]
        captions = batch[-1]

        loss_dict = {}

        # 1) distribution matching: binary BCE or soft KL
        if self.use_binary:
            # target_dist contains 0/1 presence labels; vocab_logits are raw pre-softmax
            pos_weight = torch.full(
                (vocab_logits.shape[-1],),
                fill_value=self.pos_weight_val,
                device=vocab_logits.device,
                dtype=vocab_logits.dtype,
            )
            l_bce = F.binary_cross_entropy_with_logits(
                vocab_logits,
                target_dist,
                pos_weight=pos_weight,
                reduction="mean",
            )
            loss_dict["l_bce"] = self.bce_coef * l_bce
        else:
            # clamp before normalization so log(target) doesn't produce NaN for zero entries
            target_dist = target_dist.clamp_min(1e-8)
            target_dist = target_dist / (target_dist.sum(dim=-1, keepdim=True) + 1e-8)
            pred_log_probs = F.log_softmax(vocab_logits / self.temperature, dim=-1)
            l_kl = F.kl_div(
                pred_log_probs,
                target_dist,
                reduction="batchmean",
            )
            loss_dict["l_dist"] = self.kl_coef * l_kl

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