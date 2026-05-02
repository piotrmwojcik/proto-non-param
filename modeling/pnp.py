from math import sqrt

import torch
import torch.nn.functional as F
from einops import einsum, rearrange
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


class ScoreAggregation(nn.Module):
    def __init__(
        self,
        init_val: float = 0.2,
        n_classes: int = 200,
        n_prototypes: int = 5,
    ) -> None:
        super().__init__()
        self.weights = nn.Parameter(
            torch.full((n_classes, n_prototypes), init_val, dtype=torch.float32)
        )

    def forward(self, x: torch.Tensor):
        """
        x: [B, C, K]
        returns: [B, C]
        """
        n_classes, n_prototypes = self.weights.shape

        assert x.ndim == 3, f"Expected [B, C, K], got {x.shape}"
        assert x.shape[1] == n_classes, f"Expected C={n_classes}, got {x.shape[1]}"
        assert x.shape[2] == n_prototypes, f"Expected K={n_prototypes}, got {x.shape[2]}"

        sa_weights = F.softmax(self.weights, dim=-1) * n_prototypes  # [C, K]
        x = x * sa_weights.unsqueeze(0)                              # [B, C, K]
        x = x.sum(dim=-1)                                            # [B, C]
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

        vocab_clip_embs = torch.stack(
            [cache[w] for w in self.vocab_words],
            dim=0,
        ).float()  # [V, D]

        vocab_clip_embs = F.normalize(vocab_clip_embs, dim=-1)

        self.vocab_size = vocab_clip_embs.shape[0]
        self.num_classes = 200

        self.prototype_classifier = ScoreAggregation(
            init_val=0.2,
            n_classes=self.num_classes,
            n_prototypes=self.vocab_size  # important: this is V
        )
                # One copy of vocab CLIP embeddings per class
        vocab_clip_embs = vocab_clip_embs.unsqueeze(0).repeat(
            self.num_classes, 1, 1
        )  # [C, V, D]

        self.register_buffer(
            "vocab_clip_embeddings",
            vocab_clip_embs,
        )  # [200, V, 512]

        # Learnable class-specific residuals
        self.prototype_residual = nn.Parameter(
            torch.randn(
                self.num_classes,
                self.vocab_size,
                self.clip_text_dim,
            ) * self.prototype_init_noise
        )  # [200, V, 512]
                #self.prototype_classifier = NonNegLinear(
                #    in_features=self.vocab_size,
                #    out_features=self.vocab_size,
                #    bias=True
                #)

    def get_prototypes(self) -> torch.Tensor:
        """
        Returns class-specific visual prototypes.

        vocab_clip_embeddings: [C, V, clip_text_dim]
        prototype_residual:    [C, V, clip_text_dim]

        output:
            proto: [C, V, visual_dim]
        """
        clip_proto = self.vocab_clip_embeddings + self.prototype_residual
        clip_proto = F.normalize(clip_proto, dim=-1)  # [C, V, 512]

        C, V, D = clip_proto.shape

        # Flatten so Linear/MLP projection works normally
        clip_proto = clip_proto.reshape(C * V, D)     # [C*V, 512]

        proto = self.text_projection_head(clip_proto) # [C*V, visual_dim]
        proto = F.normalize(proto, dim=-1)

        proto = proto.reshape(C, V, -1)               # [C, V, visual_dim]

        return proto

    def forward(self, x: torch.Tensor):
        patch_tokens, _, _ = self.backbone(x)  # [B, N, D]
        patch_tokens = F.normalize(patch_tokens, p=2, dim=-1)

        prototypes = self.get_prototypes()  # [C, V, D]
        prototypes = F.normalize(prototypes, p=2, dim=-1)

        patch_prototype_logits = einsum(
            patch_tokens,
            prototypes,
            "B n_patches dim, C V dim -> B n_patches C V",
        )  # [B, N, C, V]

        k = 5
        k = min(k, patch_prototype_logits.shape[1])

        # class classification branch
        topk_vals = patch_prototype_logits.topk(k, dim=1).values
        class_vocab_logits = topk_vals.mean(dim=1)  # [B, C, V]

        class_logits = self.prototype_classifier(class_vocab_logits)  # [B, C]

        # class-agnostic concept detection branch
        patch_vocab_logits = patch_prototype_logits.max(dim=2).values  # [B, N, V]

        topk_vocab_vals = patch_vocab_logits.topk(k, dim=1).values
        vocab_logits = topk_vocab_vals.mean(dim=1)  # [B, V]

        outputs = {
            "patch_tokens": patch_tokens,
            "patch_prototype_logits": patch_prototype_logits,
            "patch_vocab_logits": patch_vocab_logits,
            "class_vocab_logits": class_vocab_logits,
            "class_logits": class_logits,
            "vocab_logits": vocab_logits,
            "prototypes": prototypes,
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
    def __init__(
        self,
        cls_coef: float = 1.0,
        concept_coef: float = 0.2,
        l_ppd_coef: float = 0,
        l_ppd_temp: float = 0.1,
        concept_temperature: float = 0.1,
    ) -> None:
        super().__init__()
        self.l_ppd_coef = l_ppd_coef
        self.l_ppd_temp = l_ppd_temp

        self.cls_coef = cls_coef
        self.concept_coef = concept_coef
        self.concept_temperature = concept_temperature

    @staticmethod
    def build_concept_targets(
        tokens: list[torch.Tensor],
        vocab_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        targets = torch.zeros(len(tokens), vocab_size, device=device)

        for i, token_ids in enumerate(tokens):
            if token_ids.numel() == 0:
                continue

            token_ids = token_ids.to(device=device, dtype=torch.long)
            targets[i, token_ids] = 1.0

        return targets

    def forward(self, outputs, batch, model=None):
        class_logits = outputs["class_logits"]          # [B, C]
        class_vocab_logits = outputs["class_vocab_logits"]  # [B, C, V]

        labels = batch["class_idxs"].to(class_logits.device)  # [B]
        tokens = batch["tokens"]

        loss_dict = {}

        # -----------------------------------
        # Class classification loss
        # -----------------------------------
        l_cls = F.cross_entropy(class_logits, labels)
        loss_dict["l_cls"] = self.cls_coef * l_cls

        # -----------------------------------
        # Concept targets
        # -----------------------------------
        B = class_vocab_logits.shape[0]
        V = class_vocab_logits.shape[-1]

        concept_targets = self.build_concept_targets(
            tokens=tokens,
            vocab_size=V,
            device=class_vocab_logits.device,
        )  # [B, V]

        # -----------------------------------
        # Select only the true class slice
        # -----------------------------------
        true_class_vocab_logits = class_vocab_logits[
            torch.arange(B, device=class_vocab_logits.device),
            labels,
            :
        ]  # [B, V]

        pos_mask = concept_targets > 0

        l_concept = F.binary_cross_entropy_with_logits(
            true_class_vocab_logits[pos_mask] / self.concept_temperature,
            concept_targets[pos_mask],
        )

        loss_dict["l_concept"] = self.concept_coef * l_concept

        return loss_dict

    @staticmethod
    def ppd_criterion(patch_prototype_logits: torch.Tensor,
                      patch_prototype_assignments: torch.Tensor,
                      class_weight: torch.Tensor,
                      temperature: float = 0.1):
        patch_prototype_logits = rearrange(patch_prototype_logits, "B N C K -> B (C K) N") / temperature
        loss = F.cross_entropy(patch_prototype_logits, target=patch_prototype_assignments, weight=class_weight)
        return loss