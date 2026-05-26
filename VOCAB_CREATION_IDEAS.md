# Vocabulary / Concept Distribution — Alternative Approaches

## Problem

Current pipeline: per-image CLIP similarity scores → top-50 → softmax target distribution.

**Core issue**: top-50 selects by relevance only. Synonyms cluster in CLIP space, so "cat", "feline", "kitty", "kitten" all score near-identically and fill slots instead of covering the breadth of the image (background, texture, action, scene type, etc.).

---

## Implemented

### Caption-constrained CLIP scoring (`--clip-scores-caption-filter`)

Before top-K selection, mask out any vocab word that does not appear (as a lemma) in the image's captions. Synonyms are suppressed because annotators who write "cat" don't also write "feline".

**Fallback**: images with fewer than `top_k` caption words in vocabulary use unmasked scores.

```bash
python train.py \
    --dataset coco_vg \
    --clip-scores-coco-train $SCRATCH/data/coco/coco_train_clip_scores.pt \
    --clip-scores-coco-val   $SCRATCH/data/coco/coco_val_clip_scores.pt \
    --clip-scores-vg         $SCRATCH/data/vg/vg_clip_scores.pt \
    --clip-scores-caption-filter \
    [... other args ...]
```

---

## Candidates for future experiments

### A. Maximum Marginal Relevance (MMR)

Greedy selection that trades off relevance vs. diversity. At each step, pick the concept that maximises `λ · score - (1-λ) · max_cosine_sim(concept, already_selected)`.

- **Pros**: ~20 lines, no new dependencies, single hyperparameter λ
- **Cons**: greedy (not globally optimal)
- **λ = 0.5** is a good default; λ = 1.0 recovers plain top-K
- Can be applied as a post-processing step on existing precomputed CLIP scores

```python
def mmr(scores, embeddings, k, lam=0.5):
    selected = [scores.argmax().item()]
    while len(selected) < k:
        cand_scores = lam * scores
        sim = (embeddings @ embeddings[selected].T).max(dim=-1).values
        cand_scores -= (1 - lam) * sim
        cand_scores[selected] = -torch.inf
        selected.append(cand_scores.argmax().item())
    return selected
```

---

### B. Determinantal Point Process (DPP)

Principled probabilistic selection: jointly maximises relevance × diversity via a kernel matrix `L[i,j] = score_i · cosine_sim(emb_i, emb_j) · score_j`. Exact sampling via eigendecomposition.

- **Pros**: mathematically optimal relevance/diversity trade-off, well-studied
- **Cons**: O(K³) per image for exact sampling; needs `dppy` or custom Cholesky implementation
- **Reference**: Kulesza & Taskar, "Determinantal Point Processes for Machine Learning" (2012)

---

### C. WordNet synset collapsing (vocabulary-level)

Group vocabulary words by WordNet synset before scoring. "cat", "feline", "kitty" → one canonical token (most frequent lemma in corpus). Scoring and target distribution operate over ~4K synsets instead of 12K words.

- **Pros**: one-time preprocessing; fixes synonym problem globally, not per-image
- **Cons**: WordNet coverage is imperfect for visual concepts (colors, textures, actions)
- **Implementation**: `nltk.corpus.wordnet` already available

```python
from nltk.corpus import wordnet as wn
synset_map = {}
for word in vocab_words:
    syns = wn.synsets(word, pos=wn.NOUN)
    key = syns[0].name() if syns else word
    synset_map.setdefault(key, []).append(word)
```

---

### D. Cluster-then-select (K-Means on CLIP embeddings)

Among the top-N candidates (e.g., 200) by CLIP score, cluster into K groups and take the highest-scoring word per cluster.

- **Pros**: simple, fast (scikit-learn), produces exactly K diverse concepts
- **Cons**: KMeans unstable in 512-dim; cluster sizes uneven; ignores score magnitude within clusters
- **Implementation**: `sklearn.cluster.KMeans(n_clusters=top_k).fit(candidate_embeddings)`

---

### E. LLM-generated concept sets

Use an LLM (LLaMA, GPT) at dataset build time to generate diverse, non-overlapping visual concepts per image from its captions.

```
Prompt: "Given captions: {captions}
List 20 distinct visual concepts. No synonyms. Cover objects, attributes, scene, actions."
```

- **Pros**: explicitly diversity-aware; covers scene, object, attribute, action levels
- **Cons**: LLM inference required at dataset build time (slow); output quality varies; hard to normalise distribution
- **When to try**: if caption-filter and MMR still show synonym clustering

---

### F. Two-level coarse + fine vocabulary

Build a 100-concept coarse taxonomy (WordNet hypernyms: animal, vehicle, food, indoor, outdoor…) plus 2K fine-grained concepts. Loss combines both levels.

- **Pros**: naturally captures broad subjects at coarse level; prototype hierarchy is interpretable
- **Cons**: requires a curated or auto-generated taxonomy; more complex training objective (two KL terms)
- **When to try**: after confirming single-level approaches plateau

---

## Recommended experiment order

| Priority | Approach | Effort | Expected gain |
|----------|----------|--------|---------------|
| Done | Caption-constrained filter | Low | Removes annotator-absent synonyms |
| Next | MMR (λ=0.5) | Low | Diversity within caption vocabulary |
| Then | WordNet synset collapse | Medium | Global vocabulary deduplication |
| Later | DPP | Medium | Optimal relevance+diversity |
| If needed | LLM concepts | High | Semantic coverage beyond captions |
