from math import sqrt
from typing import Optional

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
        clip_model = None,
        msn_mask_ratio: float = 0.0,
        agg_mode: str = "topk",
        topk_k: int = 5,
        attn_temp_init: float = 0.1,
    ):
        super().__init__()
        assert agg_mode in ("topk", "cross_attn")
        self.backbone = backbone
        self.dim = dim
        self.temperature = temperature
        self.agg_mode = agg_mode
        self.topk_k = topk_k
        self.attn_temp = nn.Parameter(torch.tensor(attn_temp_init))
        self.clip_text_dim = clip_text_dim
        self.prototype_init_noise = prototype_init_noise

        # CLIP image model used for visual gating / concept selection
        self.clip_model = clip_model
        self.msn_mask_ratio = msn_mask_ratio
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

    def get_prototypes(self, use_residual: bool = True) -> torch.Tensor:
        """
        Compute visual prototypes from frozen CLIP text embeddings plus
        a trainable residual, then project to visual space.

        Args:
            use_residual: if False, ignore prototype_residual and use only the
                frozen CLIP embeddings. Useful for inference ablations comparing
                with vs. without the learned residual.
        """
        if use_residual:
            clip_proto = self.vocab_clip_embeddings + self.prototype_residual  # [V, 512]
        else:
            clip_proto = self.vocab_clip_embeddings.clone()  # [V, 512]
        clip_proto = F.normalize(clip_proto, dim=-1)

        proto = self.text_projection_head(clip_proto)  # [V, D]
        proto = F.normalize(proto, dim=-1)
        return proto

    def get_prototypes_augmented(
        self,
        extra_clip_embeds: Optional[torch.Tensor] = None,
        use_residual: bool = True,
    ) -> torch.Tensor:
        """Word prototypes (with trained residuals) optionally concatenated with
        caption prototypes (no residuals) for inference-time vocabulary augmentation.

        Args:
            extra_clip_embeds: [C, clip_text_dim] CLIP text embeddings of captions/
                phrases to append to the prototype pool. These are projected through
                the same text_projection_head but have no trained residual, since
                they were not part of training. Caller must ensure the model is in
                eval() mode so BatchNorm1d uses its running statistics.
            use_residual: if False, ignore prototype_residual for the word prototypes.
                Caption prototypes (extra_clip_embeds) never use a residual regardless.

        Returns:
            [V + C, D] or [V, D] (when extra_clip_embeds is None) unit-normalised
            visual prototypes.
        """
        if use_residual:
            clip_proto = self.vocab_clip_embeddings + self.prototype_residual  # [V, 512]
        else:
            clip_proto = self.vocab_clip_embeddings.clone()  # [V, 512]
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
        attn_weights = None
        if self.agg_mode == "cross_attn":
            attn_weights = F.softmax(patch_prototype_logits / self.attn_temp, dim=1)  # [B, N, V]
            vocab_logits = (attn_weights * patch_prototype_logits).sum(dim=1)          # [B, V]
        else:
            topk_vals = patch_prototype_logits.topk(self.topk_k, dim=1).values  # [B, k, V]
            vocab_logits = topk_vals.mean(dim=1)                                # [B, V]

        # MSN: post-backbone masking — zero out a random fraction of patch tokens
        # and recompute vocab_logits from the survivors. No second backbone pass.
        # ponytail: avoids 2x backbone cost; backbone saw the full image but prototype
        # assignment is still forced to work from partial patch evidence.
        vocab_logits_masked = None
        ibot_logits_masked = None
        ibot_logits_full = None
        if self.msn_mask_ratio > 0 and self.training:
            B, N = patch_tokens.shape[:2]
            n_mask = int(N * self.msn_mask_ratio)
            msn_masks = torch.zeros(B, N, dtype=torch.bool, device=x.device)
            for i in range(B):
                msn_masks[i, torch.randperm(N, device=x.device)[:n_mask]] = True
            patch_tokens_m = patch_tokens.clone()
            patch_tokens_m[msn_masks] = 0.0          # zero out masked positions
            ppl_m = einsum(patch_tokens_m, prototypes, "B n_patches dim, V dim -> B n_patches V")
            if self.agg_mode == "cross_attn":
                attn_weights_m = F.softmax(ppl_m / self.attn_temp, dim=1)
                vocab_logits_masked = (attn_weights_m * ppl_m).sum(dim=1)
            else:
                topk_vals_m = ppl_m.topk(self.topk_k, dim=1).values
                vocab_logits_masked = topk_vals_m.mean(dim=1)
            # iBOT: per-patch CE at masked positions (reuses ppl_m, no extra compute)
            ibot_logits_masked = ppl_m[msn_masks]                          # [B*n_mask, V]
            ibot_logits_full = patch_prototype_logits.detach()[msn_masks]  # [B*n_mask, V]

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
            attn_weights=attn_weights,
            vocab_logits=vocab_logits,
            clip_vocab_logits=clip_vocab_logits,
            clip_gate_logits=None,
            mixture_weights=weights,
            pred_text_embedding=pred_text_embedding,
            clip_image_embedding=clip_image_embedding,
            prototypes=prototypes,
            vocab_logits_masked=vocab_logits_masked,
            ibot_logits_masked=ibot_logits_masked,
            ibot_logits_full=ibot_logits_full,
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


@torch.no_grad()
def sinkhorn_knopp(logits: torch.Tensor, eps: float = 0.10, n_iter: int = 3,
                   prior: torch.Tensor = None) -> torch.Tensor:
    """Doubly-stochastic assignment matrix via Sinkhorn-Knopp (DINO/SwAV style).

    Returns Q [B, V] where rows sum to 1 (per-sample distribution) and
    column sums match `prior` (each prototype receives prior[v] of the total
    assignment across the batch). prior=None → uniform (original behavior,
    forces equal prototype usage). A non-uniform prior (PMSN, Assran et al.
    2022) matches long-tailed vocabularies where uniform usage is unrealistic.

    ponytail: eps=0.10 calibrated for cosine-scale vocab_logits (std≈0.15);
    SwAV default eps=0.05 is tuned for larger dot-product logits and yields
    near one-hot Q on cosine inputs (col uniformity fails in 3 iters).
    """
    # Subtract row max for numerical stability before exp
    logits = logits - logits.max(dim=-1, keepdim=True).values
    Q = torch.exp(logits / eps).T  # [V, B]
    Q /= Q.sum()
    K, B = Q.shape
    for _ in range(n_iter):
        if prior is None:
            Q /= Q.sum(dim=1, keepdim=True) * K       # uniform over vocab
        else:
            Q = Q / Q.sum(dim=1, keepdim=True) * prior.unsqueeze(1)  # prior[v] over vocab
        Q /= Q.sum(dim=0, keepdim=True) * B            # uniform over batch
    return (Q / Q.sum(dim=0, keepdim=True)).T  # [B, V], rows sum to exactly 1


def vicreg_loss(z: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    """VICReg variance + covariance regularizer (Bardes et al. 2022), single-view.

    No invariance term (we have one view); this is the anti-collapse half only.
    z is unit-normalised [B, D], so per-dim std is ~1/sqrt(D) — rescale by
    sqrt(D) so the standard std-target of 1 is meaningful. Covariance term is
    down-weighted 1/25 to match VICReg's default var:cov = 25:1 ratio.
    """
    B, D = z.shape
    z = z * sqrt(D)
    z = z - z.mean(dim=0)
    std = (z.var(dim=0) + eps).sqrt()
    var_loss = F.relu(1.0 - std).mean()
    cov = (z.T @ z) / max(B - 1, 1)
    cov_loss = cov.fill_diagonal_(0.0).pow(2).sum() / D
    return var_loss + cov_loss / 25.0


def koleo_loss(z: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """KoLeo nearest-neighbour repulsion (DINOv3, Darcet et al. 2023).
    Maximises -E[log(nn_dist)] to spread L2-normalised embeddings across the batch.
    Applied to pred_text_embedding [B, 512] which is already unit-normalised.
    """
    z = F.normalize(z, dim=-1)                                  # [B, D]
    sim = z @ z.T                                               # [B, B]
    sim.fill_diagonal_(-1.0)                                    # exclude self
    nn_sim = sim.max(dim=-1).values                             # [B]
    nn_dist = (2.0 - 2.0 * nn_sim).clamp(min=eps).sqrt()      # cosine → L2
    return -nn_dist.clamp(min=eps).log().mean()


class _EppsPulley(nn.Module):
    """Univariate Epps-Pulley goodness-of-fit test.
    Adapted from galilai-group/stable-pretraining (LeJEPA, Balestriero & LeCun 2025).
    """
    def __init__(self, t_max: float = 3.0, n_points: int = 17):
        super().__init__()
        assert n_points % 2 == 1
        t = torch.linspace(0, t_max, n_points)
        dt = t_max / (n_points - 1)
        phi = (-0.5 * t ** 2).exp()
        weights = torch.full((n_points,), 2 * dt)
        weights[[0, -1]] = dt                   # trapezoidal quadrature
        self.register_buffer("t", t)
        self.register_buffer("phi", phi)
        self.register_buffer("weights", weights * phi)  # pre-weight by Gaussian CF

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [N, S]. Returns per-slice statistic [S]."""
        N = x.size(0)
        x_t = x.unsqueeze(-1) * self.t         # [N, S, n_points]
        cos_mean = x_t.cos().mean(0)            # [S, n_points]
        sin_mean = x_t.sin().mean(0)
        err = (cos_mean - self.phi).square() + sin_mean.square()
        return (err @ self.weights) * N         # [S]


class SlicedEppsPulley(nn.Module):
    """Sliced Epps-Pulley goodness-of-fit for multivariate normality.
    Uses a step-seeded generator so random projection directions are deterministic per step.
    Adapted from galilai-group/stable-pretraining (LeJEPA, Balestriero & LeCun 2025).
    """
    def __init__(self, num_slices: int = 1024, t_max: float = 3.0, n_points: int = 17):
        super().__init__()
        self.num_slices = num_slices
        self.ep = _EppsPulley(t_max=t_max, n_points=n_points)
        self.register_buffer("global_step", torch.zeros((), dtype=torch.long))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [N, D]. Returns scalar mean EP statistic."""
        with torch.no_grad():
            g = torch.Generator(device=x.device).manual_seed(self.global_step.item())
            A = torch.randn(x.size(-1), self.num_slices, device=x.device, generator=g)
            A = A / A.norm(p=2, dim=0)
            self.global_step.add_(1)
        proj = x @ A                            # [N, num_slices]
        return self.ep(proj).mean()


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
        caption_coef: float = 0.0,
        loss_type: str = "kl",
        residual_reg_coef: float = 0.0,
        contrastive_coef: float = 0.0,
        contrastive_temp: float = 0.07,
        contrastive_label_temp: float = 0.0,
        contrastive_k: int = 1,
        sk_coef: float = 0.0,
        sk_eps: float = 0.10,
        sk_n_iter: int = 3,
        sk_prior: torch.Tensor = None,
        koleo_coef: float = 0.0,
        vicreg_coef: float = 0.0,
        msn_coef: float = 0.0,
        ibot_coef: float = 0.0,
        sigreg_coef: float = 0.0,
        sigreg_sketch_dim: int = 64,
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
        self.caption_coef = caption_coef
        if loss_type not in ("kl", "jsd"):
            raise ValueError(f"loss_type must be 'kl' or 'jsd', got {loss_type!r}")
        self.loss_type = loss_type
        self.residual_reg_coef = residual_reg_coef
        self.contrastive_coef = contrastive_coef
        self.contrastive_temp = contrastive_temp
        self.contrastive_label_temp = contrastive_label_temp
        self.contrastive_k = contrastive_k
        self.sk_coef = sk_coef
        self.sk_eps = sk_eps
        self.sk_n_iter = sk_n_iter
        if sk_prior is not None:
            self.register_buffer("sk_prior", sk_prior)
        else:
            self.sk_prior = None
        self.koleo_coef = koleo_coef
        self.vicreg_coef = vicreg_coef
        self.msn_coef = msn_coef
        self.ibot_coef = ibot_coef
        self.sigreg_coef = sigreg_coef
        self.sigreg = SlicedEppsPulley(num_slices=sigreg_sketch_dim)

    def forward(self, outputs: dict[str, torch.Tensor], batch: tuple[torch.Tensor, ...], model):
        vocab_logits = outputs["vocab_logits"]              # [B, V]
        mixture_weights = outputs["mixture_weights"]        # [B, V]
        patch_logits = outputs["patch_prototype_logits"]
        target_dist = batch[1]
        captions = batch[-1]

        loss_dict = {}

        # 1) distribution matching: binary BCE, soft KL, or JSD
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
        elif self.kl_coef != 0:
            if self.loss_type == "kl":
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
            else:
                # JSD: L_JSD = 0.5 * KL(target || m) + 0.5 * KL(q_hat || m), m = (target + q_hat) / 2
                # Pairs best with --target-mode topk so target has genuine zeros (the negative signal).
                # Do NOT clamp target to eps — zeros are the negative signal JSD is designed to exploit.
                target_dist = target_dist / (target_dist.sum(dim=-1, keepdim=True) + 1e-8)
                q_hat = F.softmax(vocab_logits / self.temperature, dim=-1)  # [B, V], always > 0
                m = 0.5 * (target_dist + q_hat)                             # [B, V], always > 0

                # KL(target || m): zero entries contribute 0 by convention (0 log 0 = 0)
                kl_target_m = torch.where(
                    target_dist > 0,
                    target_dist * (target_dist.clamp(min=1e-8).log() - m.log()),
                    torch.zeros_like(target_dist),
                ).sum(dim=-1).mean()

                # KL(q_hat || m): always well-defined since q_hat > 0 and m > 0
                kl_pred_m = (q_hat * (q_hat.log() - m.log())).sum(dim=-1).mean()

                l_jsd = 0.5 * (kl_target_m + kl_pred_m)
                loss_dict["l_dist"] = self.kl_coef * l_jsd

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

        # 5) optional caption-level alignment: pred_text_embedding vs per-image CLIP caption emb
        if self.caption_coef != 0:
            caption_emb = batch[4].to(vocab_logits.device)        # [B, 512]
            pred_emb = outputs["pred_text_embedding"]              # [B, 512], already normalised
            caption_emb = F.normalize(caption_emb, dim=-1)
            l_caption = 1.0 - (pred_emb * caption_emb).sum(dim=-1).mean()
            loss_dict["l_caption"] = self.caption_coef * l_caption

        # 6) optional residual regularization: (1/V) Σ ‖δ_v‖² (PDF Eq. 14)
        if self.residual_reg_coef != 0:
            l_residual_reg = model.prototype_residual.pow(2).sum(dim=-1).mean()
            loss_dict["l_residual_reg"] = self.residual_reg_coef * l_residual_reg

        # 7) in-batch contrastive: pred_text_embedding (image side) vs CLIP phrase embeddings
        # Symmetric InfoNCE over the [B, B] cosine similarity matrix.
        # Requires caption_emb as batch[4] (provided by vg_collate_fn + caption_embeds_path).
        # When contrastive_label_temp > 0, uses soft negatives: the one-hot target is replaced
        # by a distribution derived from pairwise phrase-phrase cosine similarity, so in-batch
        # images with semantically similar phrases receive partial credit rather than being
        # treated as hard negatives.
        # When batch[4].dim() == 3, hard-mining mode: batch[4] is a padded pool [B, P, 512]
        # and batch[5] is valid lengths [B]; top-k phrases per image are selected online by
        # cosine similarity to pred_text_embedding, then averaged into the positive key.
        if self.contrastive_coef != 0:
            pred_emb = outputs["pred_text_embedding"]          # [B, 512], already normalised

            if batch[4].dim() == 3:  # hard mining: pool [B, P, 512] + lengths [B]
                pools = F.normalize(batch[4].to(vocab_logits.device), dim=-1)  # [B, P, 512]
                pool_lens = batch[5].to(vocab_logits.device)                   # [B]
                sims = (pools * pred_emb.unsqueeze(1)).sum(-1)                 # [B, P]
                pad_mask = torch.arange(pools.shape[1], device=sims.device).unsqueeze(0) >= pool_lens.unsqueeze(1)
                sims = sims.masked_fill(pad_mask, -float('inf'))
                k = min(self.contrastive_k, pools.shape[1])
                top_idx = sims.topk(k, dim=1).indices                          # [B, k]
                selected = pools[torch.arange(pred_emb.shape[0], device=pools.device).unsqueeze(1), top_idx]
                cap_emb = F.normalize(selected.mean(1), dim=-1)                # [B, 512]
            else:
                cap_emb = F.normalize(batch[4].to(vocab_logits.device), dim=-1)  # [B, 512]
            sim = pred_emb @ cap_emb.T / self.contrastive_temp  # [B, B]

            if self.contrastive_label_temp > 0:
                with torch.no_grad():
                    label_sim = cap_emb @ cap_emb.T / self.contrastive_label_temp  # [B, B]
                    soft_labels = F.softmax(label_sim, dim=-1)                      # [B, B]
                l_i2t = -(soft_labels * F.log_softmax(sim,   dim=-1)).sum(-1).mean()
                l_t2i = -(soft_labels * F.log_softmax(sim.T, dim=-1)).sum(-1).mean()
                l_contrastive = (l_i2t + l_t2i) / 2
            else:
                labels = torch.arange(sim.shape[0], device=sim.device)
                l_contrastive = (F.cross_entropy(sim, labels) + F.cross_entropy(sim.T, labels)) / 2

            loss_dict["l_contrastive"] = self.contrastive_coef * l_contrastive

        # 8) Sinkhorn-Knopp batch diversity: enforce uniform prototype usage across the batch.
        # Q is computed @no_grad from vocab_logits; cross-entropy from Q → log_P trains the
        # model to match the doubly-stochastic assignment (more spread prototype use).
        if self.sk_coef != 0:
            Q = sinkhorn_knopp(vocab_logits.detach(), eps=self.sk_eps, n_iter=self.sk_n_iter,
                               prior=self.sk_prior)
            log_P = F.log_softmax(vocab_logits / self.temperature, dim=-1)
            l_sk = -(Q * log_P).sum(dim=-1).mean()
            loss_dict["l_sk"] = self.sk_coef * l_sk

        # 9) KoLeo: push nearest-neighbour pred_text_embeddings apart across the batch.
        if self.koleo_coef != 0:
            l_koleo = koleo_loss(outputs["pred_text_embedding"])
            loss_dict["l_koleo"] = self.koleo_coef * l_koleo

        # 9b) VICReg variance+covariance anti-collapse on pred_text_embedding
        # (alternative to KoLeo; covariance term fights dimensional collapse).
        if self.vicreg_coef != 0:
            l_vicreg = vicreg_loss(outputs["pred_text_embedding"])
            loss_dict["l_vicreg"] = self.vicreg_coef * l_vicreg

        # 10) MSN: predict full-image prototype distribution from masked-patch view.
        if self.msn_coef != 0 and outputs.get("vocab_logits_masked") is not None:
            target   = F.softmax(outputs["vocab_logits"].detach() / self.temperature, dim=-1)
            log_pred = F.log_softmax(outputs["vocab_logits_masked"] / self.temperature, dim=-1)
            l_msn = -(target * log_pred).sum(dim=-1).mean()
            loss_dict["l_msn"] = self.msn_coef * l_msn

        # 11) iBOT: per-patch masked CE at masked positions only (patch-level, not global).
        # Requires --msn-mask-ratio > 0 so the masking block runs and ibot_logits_* are set.
        if self.ibot_coef != 0 and outputs.get("ibot_logits_masked") is not None:
            target   = F.softmax(outputs["ibot_logits_full"] / self.temperature, dim=-1)
            log_pred = F.log_softmax(outputs["ibot_logits_masked"] / self.temperature, dim=-1)
            l_ibot = -(target * log_pred).sum(dim=-1).mean()
            loss_dict["l_ibot"] = self.ibot_coef * l_ibot

        # 12) SigReg: push pred_text_embedding toward isotropic Gaussian via sliced ECF matching.
        if self.sigreg_coef != 0:
            l_sigreg = self.sigreg(outputs["pred_text_embedding"])
            loss_dict["l_sigreg"] = self.sigreg_coef * l_sigreg

        return loss_dict