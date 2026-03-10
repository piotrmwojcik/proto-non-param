from math import sqrt

import torch
import torch.nn.functional as F
from einops import einsum
from torch import nn
import open_clip


class CLIPPatch16Backbone(nn.Module):
    def __init__(self, model_name: str = "ViT-B-14", pretrained: str = "openai", patch_size: int = 16):
        super().__init__()

        model, _, _ = open_clip.create_model_and_transforms(
            model_name,
            pretrained=pretrained,
        )

        self.model = model
        self.visual = model.visual
        self.dim = self.visual.output_dim
        self.patch_size = patch_size

        for p in self.parameters():
            p.requires_grad = False

    def set_requires_grad(self, requires_grad: bool = True):
        for p in self.parameters():
            p.requires_grad = requires_grad
        if not requires_grad:
            self.eval()

    def learnable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]

    @torch.no_grad()
    def forward(self, x: torch.Tensor):
        """
        Args:
            x: [B, 3, H, W], already CLIP-normalized

        Returns:
            patch_tokens: [B, N, D]
            raw_patch_tokens: [B, N, D]
            cls_tokens: [B, D]
        """
        B, C, H, W = x.shape
        p = self.patch_size
        assert H % p == 0 and W % p == 0, f"H,W must be divisible by {p}."

        patches = F.unfold(x, kernel_size=p, stride=p).transpose(1, 2)  # [B, N, C*p*p]
        N = patches.shape[1]
        patches = patches.reshape(B * N, C, p, p)  # [B*N, 3, p, p]

        # OpenAI CLIP checkpoints typically expect 224x224
        patches = F.interpolate(patches, size=(224, 224), mode="bilinear", align_corners=False)

        feats = self.model.encode_image(patches)  # [B*N, D]
        feats = F.normalize(feats, dim=-1)
        feats = feats.reshape(B, N, -1)  # [B, N, D]

        patch_tokens = feats
        raw_patch_tokens = feats
        cls_tokens = feats.mean(dim=1)

        return patch_tokens, raw_patch_tokens, cls_tokens


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
        text_proj_hidden_dim: int = 768,
        vocab_cache_path: str = "vocab/laion_clip_cache.pt",
        prototype_init_noise: float = 0.01,
    ):
        super().__init__()
        self.backbone = backbone
        self.dim = dim
        self.temperature = temperature
        self.clip_text_dim = clip_text_dim
        self.prototype_init_noise = prototype_init_noise

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

    def get_prototypes(self) -> torch.Tensor:
        """
        Compute visual prototypes on the fly from cached CLIP text embeddings.
        """
        proto = self.text_projection_head(self.vocab_clip_embeddings)  # [V, D]
        return proto

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: [B, 3, H, W]

        Returns:
            dict with:
                patch_prototype_logits: [B, N, V]
                vocab_logits: [B, V]
                mixture_weights: [B, V]
                pred_text_embedding: [B, 512]
        """
        patch_tokens, _, _ = self.backbone(x)  # [B, N, D]
        patch_tokens = F.normalize(patch_tokens, p=2, dim=-1)

        prototypes = self.get_prototypes()  # [V, D]

        # Patch-to-vocab prototype logits
        patch_prototype_logits = einsum(
            patch_tokens,
            prototypes,
            "B n_patches dim, V dim -> B n_patches V",
        )  # [B, N, V]

        # Image-level logits over vocab pool
        vocab_logits = patch_prototype_logits.max(dim=1).values  # [B, V]

        weights = F.softmax(vocab_logits / self.temperature, dim=-1)  # [B, V]

        outputs = {
            "patch_prototype_logits": patch_prototype_logits,  # [B, N, V]
            "vocab_logits": vocab_logits,                      # [B, V]
            "mixture_weights": weights,                        # [B, V]
            "patch_tokens": patch_tokens,                      # [B, N, D]
            "prototypes": prototypes,                          # [V, D]
        }
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
        entropy_coef: float = 0.0,
        visual_coef: float = 0.0,
        cover_coef: float = 0.0,
        temperature: float = 0.07,
    ) -> None:
        super().__init__()
        self.kl_coef = kl_coef
        self.entropy_coef = entropy_coef
        self.visual_coef = visual_coef
        self.cover_coef = cover_coef
        self.temperature = temperature

    def forward(self, outputs: dict[str, torch.Tensor], batch: tuple[torch.Tensor, ...]):
        vocab_logits = outputs["vocab_logits"]              # [B, V]
        mixture_weights = outputs["mixture_weights"]        # [B, V]
        patch_logits = outputs["patch_prototype_logits"]    # [B, N, V]
        target_dist = batch[1]                              # [B, V]

        loss_dict = {}

        # 1) distribution matching: target noun distribution vs predicted noun distribution
        target_dist = target_dist / (target_dist.sum(dim=-1, keepdim=True) + 1e-8)
        pred_log_probs = F.log_softmax(vocab_logits / self.temperature, dim=-1)

        l_kl = F.kl_div(
            pred_log_probs,
            target_dist,
            reduction="batchmean",
        )
        loss_dict["l_dist"] = self.kl_coef * l_kl
        loss_dict["_l_dist_unadjusted"] = l_kl

        # 2) optional entropy regularization on predicted distribution
        if self.entropy_coef != 0:
            entropy = -(mixture_weights * torch.log(mixture_weights + 1e-8)).sum(dim=-1).mean()
            loss_dict["l_entropy"] = self.entropy_coef * entropy
            loss_dict["_l_entropy_unadjusted"] = entropy

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
            loss_dict["_l_visual_unadjusted"] = l_visual

        # 4) optional coverage: selected prototype mixture should explain at least one patch well
        if self.cover_coef != 0:
            patch_scores = torch.einsum("bnv,bv->bn", patch_logits, mixture_weights)  # [B, N]
            l_cover = -patch_scores.max(dim=1).values.mean()
            loss_dict["l_cover"] = self.cover_coef * l_cover
            loss_dict["_l_cover_unadjusted"] = l_cover

        return loss_dict