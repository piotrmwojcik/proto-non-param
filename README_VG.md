# Training on Visual Genome

This document covers how to download, prepare, and train the PNP model on the
[Visual Genome](https://homes.cs.washington.edu/~ranjay/visualgenome/index.html) dataset.

---

## Dataset download

Visual Genome v1.4 is available from the official site.  You need:

| File | What it contains |
|------|-----------------|
| Images Part 1 (~9 GB) | ~60 K JPEG images → `VG_100K/` |
| Images Part 2 (~5 GB) | ~48 K JPEG images → `VG_100K_2/` |
| `region_descriptions.json` | 5.4 M region phrases across all images |

Expected directory layout after extraction:

```
/data/vg/
├── VG_100K/
│   ├── 1.jpg
│   ├── 2.jpg
│   └── ...
├── VG_100K_2/
│   ├── 108250.jpg
│   └── ...
└── region_descriptions.json
```

---

## Step 1 — Build the vocabulary cache

Run the vocabulary builder once before training.  It scans all region phrases,
extracts nouns / adjectives / content-verbs via NLTK lemmatisation, filters by
frequency, encodes with CLIP, and saves a `{word: embedding}` `.pt` file.

```bash
python vocab/build_vg_vocab.py \
    --region-descriptions /data/vg/region_descriptions.json \
    --vocab-out  vocab/vg.txt \
    --cache-out  vocab/vg_cache.pt \
    --clip-model-name ViT-L-14 \
    --clip-pretrained openai \
    --min-count 5 \
    --max-doc-freq 0.5
```

Key options:

| Flag | Default | Meaning |
|------|---------|---------|
| `--min-count` | 5 | Discard words appearing in fewer than N regions total |
| `--max-doc-freq` | 0.5 | Discard words present in more than 50 % of images (stop-word filter) |

Expect ~4 000–6 000 vocabulary words with the defaults above.

---

## Step 2 — Train

```bash
python train.py \
    --dataset visual_genome \
    --vg-root /data/vg \
    --vg-region-descriptions /data/vg/region_descriptions.json \
    --vocab-cache-path vocab/vg_cache.pt \
    --log-dir logs/vg_baseline \
    --backbone dinov2_vitb14 \
    --batch-size 64 \
    --num-workers 8 \
    --epochs 20 \
    --backbone-lr 1e-5 \
    --text-proj-lr 1e-4 \
    --kl-coef 1.0
```

The dataset is split 90 / 10 into train / val using a deterministic shuffle.
Pass `--vg-val-ratio` (default `0.1`) and `--seed` to change the split.

### Suggested experiment configurations

| Config | Extra flags | Purpose |
|--------|-------------|---------|
| Baseline | *(none)* | Direct swap of COCO → VG |
| + Patch losses | `--visual-coef 0.1 --cover-coef 0.1` | Exploit VG's richer per-region coverage |
| Shared COCO vocab | `--vocab-cache-path vocab/mscoco_new_cache.pt` | Cross-dataset vocab transfer |

---

## Step 3 — Evaluate

Evaluation is identical to other datasets:

```bash
python evaluate.py --ckpt-path logs/vg_baseline/ckpt.pth
```

---

## Design rationale

**Why region descriptions?**  
Each VG image has ~50 region phrases on average, compared to ~5 captions in COCO.
The PNP training objective is a KL divergence between the model's predicted
vocabulary distribution and a ground-truth word-frequency histogram derived from
text annotations.  Richer text → better-calibrated histograms → stronger supervision
signal per image.  Region phrases are free text, so they plug directly into the
existing `extract_caption_words` (NLTK noun/adjective/verb) pipeline with zero
model or loss changes.

**Why image-level distributions (no bounding boxes)?**  
The model already performs implicit spatial grounding via patch-prototype cosine
similarity.  Adding region-mask supervision would tighten localisation but would
require a new loss term and significantly complicate the data pipeline.  The
image-level objective is sufficient for prototype quality and keeps the
architecture clean.
