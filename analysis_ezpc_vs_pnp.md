# Deep Dive: EZPC vs. Proto-Non-Param (PNP) — Researcher's Critical Comparison

Paper: "Explaining CLIP Zero-shot Predictions Through Concepts" (Ozdemir et al., CVPR 2026)
ArXiv: 2603.28211v1

---

## 1. What Each Paper Actually Does

### EZPC (Ozdemir et al., CVPR 2026)
> *"Explain CLIP's already-computed decision by rearranging its embeddings into a human-readable concept basis."*

EZPC learns a single linear projection matrix **A ∈ ℝ^{d×m}** (d = CLIP dim, m = # concepts). Both image and class-text embeddings are projected through A, making classification happen in "concept space." The dual training objective is:

- **ℒ_match = ‖A − Φ‖²** : A's columns should stay close to CLIP text embeddings of the concept vocabulary Φ.
- **ℒ_recon = KL(softmax(c_x C_𝒴ᵀ) ‖ softmax(v_x Tᵀ))** : EZPC's predictions in concept space should match CLIP's raw predictions in embedding space.
- **ℒ_total = ℒ_match + λ ℒ_recon** (λ=1 default)

Initialization: A⁰ = Φ (CLIP concept embeddings). Train for 10K steps with Adam lr=1e-2. CLIP is fully frozen — only A is updated.

### Proto-Non-Param (PNP — this work)
> *"Learn a visual prototype pool from CLIP concept vocabulary, grounded in patches, and train on image-caption noun distributions."*

PNP learns a **non-linear projection head** (512→1024→768, BN+ReLU) that maps CLIP text embeddings of vocabulary words into visual feature space. A DINOv2 backbone extracts patch tokens [B, N, D]; each patch is matched against all V concept prototypes to produce spatial activation maps [B, N, V]. Top-K patch pooling gives image-level logits [B, V], which are softmax'd to a mixture → weighted sum of CLIP vocabulary embeddings → reconstructed text embedding [B, 512]. Training signal = KL/JSD between predicted noun distribution and the target noun distribution derived from captions (MSCOCO or VG). Optionally: caption-embedding alignment, coverage loss, residual regularization.

---

## 2. Core Architectural Differences

| Dimension | EZPC | PNP (ours) |
|---|---|---|
| **Backbone** | Frozen CLIP (RN50 / ViT) | DINOv2 (trainable) + frozen CLIP for diagnostics |
| **Learned component** | Linear A ∈ ℝ^{d×m} | MLP projection head (non-linear) + residual δ_v |
| **Projection type** | Linear, global embedding | Non-linear, patch-level visual-to-concept |
| **Spatial grounding** | Post-hoc (patch CLIP features through A) | Intrinsic — patch_prototype_logits [B, N, V] by design |
| **Concept vocabulary** | GPT-3 generated per dataset (LF-CBM) | MSCOCO / VG nouns from caption co-occurrence |
| **Training signal** | Self-supervised (KL against CLIP predictions) | Supervised by caption noun distributions |
| **Classification head** | argmax_k ⟨c_x, c_k⟩ in concept space | pred_text_embedding → cosine to class descriptions |
| **Sparsity** | Dense (~490 concepts active), ranked | Dense softmax, but K=5 patch top-K pooling |
| **Inference overhead** | ~0.1ms (single matmul over CLIP) | Full DINOv2 backbone + projection |
| **Training data** | No supervision needed | Image-caption pairs required |

---

## 3. Problem Framing — a Fundamental Divide

**EZPC is post-hoc explanation.** It answers: *"Given that CLIP classified this image as X, which concepts drove that decision?"* The reconstruction loss directly minimizes divergence from CLIP's prediction, so EZPC cannot and does not claim to understand images independently — it inherits all of CLIP's biases and errors. If CLIP is wrong, EZPC explains the wrong prediction faithfully. Tab. 10 shows 92.9% top-1 agreement with CLIP on ImageNet-100 — this is the design goal, not a side-effect.

**PNP is a visual understanding model.** It answers: *"What concepts are visually present in this image?"* The training signal comes from caption noun distributions — an external, human-derived ground truth — not from CLIP's own predictions. PNP is trying to learn something new, not just rearrange existing CLIP knowledge. If CLIP fails on a concept, PNP could in principle learn it from captions even if CLIP never did.

These papers **are not direct competitors** — they address different phases of the pipeline. EZPC is a CLIP interpreter. PNP is a visual concept detector. But they share the same surface (concept vocabulary, cosine similarity scoring, zero-shot capability) which makes positioning non-trivial.

---

## 4. EZPC's Critical Weaknesses

### 4a. The ℒ_match Loss Is Decorative

Revealed by Table 12 (training objective ablation, ImageNet-100):

| Objective | Unseen Acc | H |
|---|---|---|
| A = Φ (no training) | 0.0 | 0.0 |
| ℒ_match only | 0.0 | 0.0 |
| **ℒ_recon only** | **0.708** | **0.693** |
| Full objective | 0.690 | 0.682 |

**ℒ_match alone cannot classify at all.** The reconstruction loss alone performs *better* than the full objective. The "concept alignment" — the part that makes explanations human-interpretable — is essentially a regularizer that slightly hurts accuracy. The paper frames this as a principled dual objective, but the learned basis is driven entirely by the KL reconstruction. The naming of A columns as "concepts" is a post-hoc interpretation, not something the loss enforces beyond weak L2 pull to Φ.

Compare with PNP: the KL/JSD target is derived from captions, a genuine extrinsic semantic signal. When PNP converges, it has actually learned to predict which nouns humans associate with the image.

### 4b. Linear Projection Is a Severe Constraint

EZPC's "concept space" is a linear rearrangement of CLIP's embedding dimensions — geometrically, a change of basis. This does not add representational power. PNP's non-linear text_projection_head (2-layer MLP with BN+ReLU) can learn genuinely new mappings between text concept space and visual feature space. The residual δ_v additionally fine-tunes each concept's CLIP embedding individually — something fundamentally impossible with a single shared linear A.

### 4c. Spatial Grounding Is Post-Hoc, Not Architectural

EZPC's spatial grounding (CUB pointing accuracy 0.967, Tab. 5) is computed by projecting patch-level CLIP features through A and selecting the maximally activated patch. This is a visualization trick — the model is never trained to spatially localize concepts. PNP computes `patch_prototype_logits [B, N, V]` for every patch during the forward pass. Spatial localization is a side-product of how classification works. The TopK patch pooling (k=5) is an explicit design choice that forces the model to find the most prototypical image regions.

### 4d. Dataset-Conditional Vocabulary

EZPC uses GPT-3 generated concepts per dataset: 4,751 for ImageNet-1k, 892 for CIFAR-100, 370 for CUB-200. These are class-conditional descriptions — not open-vocabulary. Cross-dataset transfer (Tab. 4) still requires the target concept vocabulary. PNP uses MSCOCO or VG noun inventories — actual co-occurrence statistics from naturalistic captioning, dataset-agnostic.

### 4e. Non-Trivial Performance Drop on Hard Benchmarks

ImageNet-1k: CLIP H=0.530 → EZPC H=0.481, a **9.2% relative drop**.
Places365: 0.362 → 0.352 (-2.8%).

The paper claims negligible performance impact; this framing holds only relative to SpLiCE/Z-CBM, not relative to CLIP itself.

---

## 5. PNP Advantages and Risks

### Genuine advantages over EZPC:
- **Patch-level spatial grounding by design** — concept heatmaps are a forward-pass output, not post-hoc
- **Non-linear concept projection** — can adapt CLIP text semantics to DINOv2 visual feature distributions
- **Extrinsic training signal** — learns from what humans say about images, not CLIP's own geometry
- **Per-concept residuals δ_v** — individually adapts each concept's visual alignment; EZPC's A is a shared global transform
- **Compositional text reconstruction** — `pred_text_embedding` is a soft mixture over CLIP text embeddings; generalizes to unseen class descriptions

### Risks in positioning:
1. **EZPC is vastly cheaper to train** (10K Adam steps, no backbone). PNP requires full DINOv2 training. Gains must justify cost.
2. **EZPC has a principled faithfulness argument**: for interpretability of CLIP's own predictions, its design is correct. PNP's goal is different — be explicit about that.
3. **The residual bug** (δ_v was frozen at random init ≈ 0 before 2026-06-14) means ablations before that date understate PNP's performance. Redo with --residual-lr if not already done.
4. **Vocabulary comparability**: EZPC uses fine-grained attribute concepts ("blue feathers", "conical bill"); PNP uses general nouns. These serve different interpretability purposes — EZPC concepts are class-discriminative; PNP concepts are compositional scene descriptions.

---

## 6. The Conceptual Overlap

Both share:
1. Fixed concept vocabulary with CLIP text embeddings
2. Cosine similarity scoring between images and concepts
3. Top-K concepts as "explanation" of the image
4. ZSL / GZSL evaluation

The direction of projection is inverted:
- EZPC: pulls image embeddings **into** text concept space (image-centric, global)
- PNP: projects text concept embeddings **into** visual patch space (patch-centric, spatial)

This matters: EZPC's A is constrained close to CLIP text embeddings so concepts retain linguistic meaning. PNP's prototypes are trained to be visually discriminative in DINOv2 patch feature space, potentially capturing visual texture/structure that CLIP's text encoder doesn't encode.

---

## 7. What EZPC Gets Right — Lessons for PNP

1. **Causal faithfulness metrics (Tab. 7/8)**: Removing top-N concepts causes 16.9% flip rate vs 1.4% for random. PNP needs analogous causal validation — remove top prototype assignments, measure prediction flip. patch_prototype_logits makes this straightforward.

2. **Concept–region alignment (Tab. 5)**: pointing accuracy + IoU on CUB annotations. PNP's patch_prototype_logits can produce this naturally — a free byproduct worth reporting.

3. **Concept space geometry analysis (Tab. 9)**: measuring how representations change after learning. PNP could report this on residuals δ_v — how far do they move, in which semantic directions?

4. **Vocabulary size scaling (Tab. 11)**: agreement saturates ~3K concepts. PNP needs analogous ablations over vocab size.

---

## 8. Positioning Summary

| Question | EZPC | PNP (ours) |
|---|---|---|
| *What do you learn?* | Linear rearrangement of CLIP into concept space | Non-linear visual-to-concept mapping grounded in captions |
| *Training signal?* | Self-supervised (KL against CLIP) | Supervised (caption noun distributions) |
| *Spatial localization?* | Post-hoc visualization | Intrinsic architectural property |
| *Inference cost?* | Negligible (1 matmul) | Full backbone overhead |
| *Concept vocabulary?* | GPT-3 per dataset | Caption co-occurrence (general-purpose) |
| *Generative?* | No | Yes — reconstructs text embedding compositionally |

**How to cite EZPC in a submission**: EZPC is the strongest recent post-hoc interpretability baseline. Frame PNP as learning a *generative visual concept model* rather than an *explanation of CLIP*. Core argument: EZPC explains what CLIP already knew; PNP learns new visual knowledge grounded in spatial patch features and human caption semantics. EZPC cannot produce a prediction that CLIP would disagree with (by construction); PNP can, because it has its own backbone and extrinsic training signal.

**The vulnerability EZPC leaves open**: zero interpretability of CLIP's failures. If CLIP is wrong (spurious correlation, domain shift), EZPC faithfully explains the wrong prediction. PNP, trained on captions, could give a correct concept distribution even when CLIP misclassifies. This is testable and would be a strong empirical differentiator.

---

**Bottom line**: EZPC is a cleanly executed CVPR-quality post-hoc explanation method competing with SpLiCE and Z-CBM, not with PNP. PNP is an architectural approach to visual concept grounding — harder problem, more training overhead, but genuine spatial understanding and extrinsic semantic signal. The δ_v residual (once properly trained) allows per-concept visual alignment without global linear constraints. Core claim: **PNP learns a spatially-grounded visual concept model from scratch; EZPC re-labels CLIP's geometry.**
