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
        vocab_cache_path: str = "vocab/mscoco_nouns_clip_cache.pt",
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

    def reconstruct_text_embedding(self, vocab_logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            vocab_logits: [B, V]

        Returns:
            weights: [B, V]
            pred_txt: [B, 512]
        """
        weights = F.softmax(vocab_logits / self.temperature, dim=-1)  # [B, V]
        pred_txt = weights @ self.vocab_clip_embeddings               # [B, 512]
        pred_txt = F.normalize(pred_txt, dim=-1)
        return weights, pred_txt

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

        # Reconstruct target text embedding as a mixture over vocab CLIP embeddings
        weights, pred_text_embedding = self.reconstruct_text_embedding(vocab_logits)

        outputs = {
            "patch_prototype_logits": patch_prototype_logits,  # [B, N, V]
            "vocab_logits": vocab_logits,  # [B, V]
            "mixture_weights": weights,  # [B, V]
            "pred_text_embedding": pred_text_embedding,  # [B, 512]
            "patch_tokens": patch_tokens,  # [B, N, D]
            "prototypes": prototypes,  # [V, D]
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
    Matches the predicted mixture embedding to the target CLIP text embedding
    and can also regularize translated prototypes to stay visually aligned
    with the image patch grid.
    """
    def __init__(
        self,
        cosine_coef: float = 1.0,
        mse_coef: float = 0.0,
        entropy_coef: float = 0.0,
        visual_coef: float = 0.0,
        cover_coef: float = 0.0,
    ) -> None:
        super().__init__()
        self.cosine_coef = cosine_coef
        self.mse_coef = mse_coef
        self.entropy_coef = entropy_coef
        self.visual_coef = visual_coef
        self.cover_coef = cover_coef

    def forward(self, outputs: dict[str, torch.Tensor], batch: tuple[torch.Tensor, ...]):
        pred_txt = outputs["pred_text_embedding"]          # [B, 512]
        mixture_weights = outputs["mixture_weights"]       # [B, V]
        patch_logits = outputs["patch_prototype_logits"]   # [B, N, V]
        target_txt = batch[1]                              # [B, 512]

        pred_txt = F.normalize(pred_txt, dim=-1)
        target_txt = F.normalize(target_txt, dim=-1)

        loss_dict = {}

        # 1) text alignment
        l_cosine = 1.0 - F.cosine_similarity(pred_txt, target_txt, dim=-1).mean()
        loss_dict["l_txt"] = self.cosine_coef * l_cosine
        loss_dict["_l_txt_unadjusted"] = l_cosine

        # 2) optional MSE
        if self.mse_coef != 0:
            l_mse = F.mse_loss(pred_txt, target_txt)
            loss_dict["l_mse"] = self.mse_coef * l_mse
            loss_dict["_l_mse_unadjusted"] = l_mse

        # 3) optional entropy reg on mixture weights
        if self.entropy_coef != 0:
            entropy = -(mixture_weights * torch.log(mixture_weights + 1e-8)).sum(dim=-1).mean()
            loss_dict["l_entropy"] = self.entropy_coef * entropy
            loss_dict["_l_entropy_unadjusted"] = entropy

        # 4) visual similarity between translated prototype mixture and patch grid
        if self.visual_coef != 0:
            patch_tokens = outputs["patch_tokens"]  # [B, N, D]
            prototypes = outputs["prototypes"]  # [V, D]
            mixture_weights = outputs["mixture_weights"]  # [B, V]

            proto_mix = F.normalize(mixture_weights @ prototypes, dim=-1)  # [B, D]
            patch_sims = torch.einsum("bd,bnd->bn", proto_mix, patch_tokens)  # [B, N]

            k = min(5, patch_sims.shape[1])
            topk_vals = patch_sims.topk(k=k, dim=1).values
            l_visual = 1.0 - topk_vals.mean()

            loss_dict["l_visual"] = self.visual_coef * l_visual
            loss_dict["_l_visual_unadjusted"] = l_visual
        # 5) optional patch coverage: selected prototype mixture should explain some patches
        if self.cover_coef != 0:
            patch_scores = torch.einsum("bnv,bv->bn", patch_logits, mixture_weights)  # [B, N]
            l_cover = -patch_scores.max(dim=1).values.mean()
            loss_dict["l_cover"] = self.cover_coef * l_cover
            loss_dict["_l_cover_unadjusted"] = l_cover

        return loss_dict