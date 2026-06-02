# Ablation: VG-only Training + Caption-Augmented Inference Prototypes

## Motivation

The standard PNP setup trains on COCO+VG captions and uses a word-level vocabulary (single
nouns/adjectives) as prototypes. This ablation isolates two questions:

1. **VG-only training**: Does restricting supervision to Visual Genome region descriptions alone
   (richer but fewer captions, more spatially grounded) change what the model learns?

2. **Caption-augmented inference**: After training with word prototypes, can we improve prototype
   expressiveness at test time by extending the vocabulary with full phrase-level CLIP embeddings
   — without any retraining?

The hypothesis: phrase-level prototypes (e.g. *"a bird with red wings"*) encode richer visual
concepts than single words (e.g. *"bird"*) and may match more precisely with the visual features
DINOv2 produces for complex scene regions.

---

## Vocabulary Construction

### Step 1 — Word-level vocabulary (training)

`vocab/build_vg_vocab.py` processes Visual Genome `region_descriptions.json`:

1. **Extract words** from each region phrase using NLTK POS tagging and WordNet lemmatisation.
   Retained POS tags: nouns (`NN*`), adjectives (`JJ*`), content verbs (`VB*` minus copula/aux).
2. **Filter** by frequency: keep words with ≥ `min_count` total occurrences (default 5) and
   document frequency ≤ `max_doc_freq` (default 0.5, to remove ubiquitous words like "the").
3. **Encode** each retained word with a frozen CLIP ViT-B/32 text encoder → unit-normalised
   vector in R^512.
4. **Save** as `vocab/vg_cache.pt`: `dict[str, Tensor[512]]`.

Typical vocabulary size: ~5 000–9 000 words.

### Step 2 — Phrase-level prototypes (inference augmentation)

`vocab/build_caption_prototypes.py` skips word extraction entirely and encodes raw caption
strings with the same CLIP model.

Two sources:

| Source | `--source` | Input | Notes |
|--------|-----------|-------|-------|
| VG test split | `vg_test` | `region_descriptions.json` + `vg_cache.pt` | Only phrases from the 10% held-out images. Requires `--vocab-cache-path` to instantiate `VisualGenomeDataset(train=False)` and reproduce the exact same split. |
| COCO val | `coco` | `captions_val2014.json` | All unique caption strings; cap with `--max-captions` |

Output: same `dict[str, Tensor[512]]` format → `vocab/vg_test_caption_prototypes.pt` or
`vocab/coco_caption_prototypes.pt`.

---

## Training (VG only)

Train the PNP model using only Visual Genome region phrases as supervision:

```bash
# 1. Build word vocabulary
python vocab/build_vg_vocab.py \
    --region-descriptions /data/vg/region_descriptions.json \
    --vocab-out  vocab/vg.txt \
    --cache-out  vocab/vg_cache.pt \
    --clip-model-name ViT-B-32 --clip-pretrained openai \
    --min-count 5 --max-doc-freq 0.5

# 2. Train
python train.py \
    --dataset visual_genome \
    --vg-root /data/vg \
    --vg-region-descriptions /data/vg/region_descriptions.json \
    --vocab-cache-path vocab/vg_cache.pt \
    --log-dir logs/vg_only \
    --backbone dinov2_vitb14 --num-splits 1 \
    --batch-size 64 --epochs 20 \
    --backbone-lr 1e-5 --text-proj-lr 1e-4 --kl-coef 1.0
```

Key differences from the default COCO+VG run:
- `--dataset visual_genome` (not `coco_vg`)
- `--vocab-cache-path vocab/vg_cache.pt` (VG words, not COCO words)
- Training images are from VG train split only (90% of VG, seed=42)

---

## Inference — Augmented Prototype Pool

`modeling/pnp.py` exposes two prototype accessors:

| Method | Prototype set | Residuals |
|--------|--------------|-----------|
| `get_prototypes()` | V word prototypes | Yes (trained) |
| `get_prototypes_augmented(extra)` | V word + C caption | Word protos: yes; caption protos: no |

`get_prototypes_augmented()` concatenates the trained word embeddings (with their learned
residuals) with the external caption CLIP embeddings (no residuals — they were never part of
training), then projects the full `[V+C, 512]` matrix through the shared `text_projection_head`
into visual space `[V+C, D]`.

> **Important**: call `model.eval()` before inference. `text_projection_head` contains a
> `BatchNorm1d` layer that uses running statistics in eval mode — this is correct behaviour and
> ensures consistent projections for caption prototypes.

---

## Evaluation

`eval_augmented_prototypes.py` runs both modes and reports to W&B:

- **`word_only`**: standard VG word prototypes — baseline.
- **`augmented`**: word + caption prototypes — ablation.

```bash
python eval_augmented_prototypes.py \
    --ckpt logs/vg_only/ckpt.pth \
    --vocab-cache-path vocab/vg_cache.pt \
    --caption-prototypes-path vocab/vg_test_caption_prototypes.pt \
    --source-dataset vg_test \
    --vg-root /data/vg \
    --vg-region-descriptions /data/vg/region_descriptions.json \
    --mode both --topk 5 \
    --wandb-project proto-non-param
```

**W&B metrics logged:**
- `word_only/mean_frac_caption_in_topk` — always 0 (sanity check)
- `augmented/mean_frac_caption_in_topk` — fraction of top-5 activations from caption protos
- `augmented/frac_caption_histogram` — distribution across images
- `augmented/top_caption_concept` — heatmap panel for the most-activated caption phrase

---

## Reproduce (one command)

```bash
bash scripts/run_vg_caption_ablation.sh
```

This submits four chained SLURM jobs. Override cluster paths via environment variables:

```bash
VG_ROOT=/my/vg COCO_ROOT=/my/coco LOG_DIR=/my/logs \
    bash scripts/run_vg_caption_ablation.sh
```

Job chain:

```
job1 (build VG vocab)  ──────────────────────────────────┐
                         → job3 (train VG-only, GPU, ~20h) ┐
job1                   → job2 (build caption caches, GPU)  ┘→ job4 (eval, GPU, ~2h)
```
