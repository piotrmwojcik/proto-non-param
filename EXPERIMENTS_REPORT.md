# PNP Experiment Report

Running log of experiments beyond the core M1 ablation series (tracked separately
in project memory / `scripts/compare_ris_results.py`'s `vg_contrastive` table).
M1 (ViT-L/14, SK=0.1 + KoLeo=0.1, no contrastive) is the baseline every experiment
below compares against unless stated otherwise.

## 1. M1-RC — random-caption-target VG training variant

**Question**: M1's KL target is a fixed distribution per image, built once by pooling
*all* of an image's Visual Genome region-description phrases together. Does resampling
a random subset of `k` phrases per training step (instead of one fixed target for the
whole run) change what the model learns?

**Setup**: Same M1 config (SK=0.1, KoLeo=0.1, no contrastive), but `--random-caption-target
--random-caption-target-k 3` — each `__getitem__` call draws 3 of the image's `n` region
phrases at random (without replacement) and builds the KL target from just those.
Implementation: `clip_dataset.py` (`VisualGenomeDataset.random_caption_target`),
`scripts/slurm_train_vg_random_caption.sh`.

**Status**: Training completed (`run_M1-RC_vitl14_sk10_koleo01_randk3_30ep`). Zero-shot RIS
eval submitted across all 7 splits (`scripts/slurm_eval_vg_contrastive_M1_RC.sh`) —
results land in `eval_results/vg_contrastive/contr_M1-RC-k3/`, registered in
`compare_ris_results.py` under variant key `M1-RC`. **Numbers not yet pulled into this
report** — run `python scripts/compare_ris_results.py --eval_dir eval_results
--ablation-dir eval_results/vg_contrastive --ablation-type vg_contrastive --out
eval_results/vg_contrastive/comparison.md` and update this section.

## 2. mIoU vs. referring-expression length — PNP (M1@672) vs. CTRL-O, Gref/val

**Question**: Is PNP's gap to CTRL-O on Gref/val (per project memory: 22.0 vs 25.3 mIoU)
concentrated in long/compositional expressions, as a "bag-of-concepts can't handle
compositionality" hypothesis would predict?

**Setup**: `scripts/analyze_miou_vs_length.py` buckets both models' per-sentence IoU by
word-count quartiles (PNP's own length distribution sets the bucket edges). PNP's
per-example data reused from the existing M1@672 eval JSON (no re-run needed — sentence
text recovered post-hoc via `ReferDataset.get_raw_item(index)`, which required making
`ReferDataset.data_list`'s ordering deterministic — `evaluation/refer_dataset.py`, now
`sorted(glob(...))`). CTRL-O's per-sentence data required a patch to `inference_refer.py`
(outer `proto-VLM` repo) to persist per-expression records — it previously only kept an
oracle (best-of-sentences-per-reference) summary, not per-sentence granularity.

**Result — hypothesis REFUTED, opposite pattern found**:

| Length (words) | n (PNP/CTRL-O) | PNP mIoU | CTRL-O mIoU | Gap (CTRL-O − PNP) |
|---|---|---|---|---|
| 1–6   | 3347/3415 | 22.06 | 26.43 | **+4.37** |
| 6–8   | 1956/2008 | 21.92 | 22.99 | +1.07 |
| 8–11  | 2359/2368 | 21.72 | 22.79 | +1.07 |
| 11–37 | 1874/1745 | 22.20 | 21.72 | **−0.48** |

The gap is concentrated in the **shortest** expressions, not the longest. 95% bootstrap
CIs: bucket 0 is clearly significant (PNP `[21.47, 22.64]` vs CTRL-O `[25.67, 27.15]`,
no overlap); by the longest bucket the CIs overlap heavily and PNP's point estimate is
nominally higher. **PNP is essentially length-invariant** (flat ~21.7–22.2 mIoU across
all four buckets); **CTRL-O is the one that's length-sensitive**, strong on short/simple
expressions and degrading monotonically as expressions lengthen, converging toward PNP's
flat baseline.

Reframed claim for the paper: CTRL-O's overall edge on Gref/val comes almost entirely
from short, simple referring expressions (plausibly where slot-attention's object-centric
decomposition has an easy single salient object to bind to) — on longer, more
compositional expressions the two methods are statistically tied.

Note: a small residual noise source exists in the length metric itself — PNP's `.npz`
batches store REFER's `sent` field (lowercased/punctuation-stripped), while CTRL-O's
`inference_refer.py` uses REFER's `raw` field (original text) — a handful of sentences
can land in a different length bucket between the two as a result, though totals match
exactly (9536 examples each), so this is bucket-assignment noise, not a data problem.

**Artifacts**: `results/miou_vs_length/{per_example.csv, per_bucket.csv, miou_vs_length.png}`.

## 3. Qualitative concept-retrieval figure

**Question**: Do PNP's word-based prototypes actually localize on the right image region
when queried?

**Setup**: `scripts/visualize_concept_retrieval.py` — for a handful of vocabulary
concepts, retrieves and displays the top-k most-activating RefCOCOg/val image crops per
concept (heatmap overlay + bounding box via percentile-threshold activation). Built from
`eval_retreive_concepts.py`'s reusable crop/heatmap utilities, swapped to the generic
hparams-driven `build_model()` (the original script hardcoded a ViT-B/14 + LAION-vocab
config incompatible with M1). Two bugs found and fixed during this run:
- Must run at CLIP's native 224px, not M1's headline 672px — `PNP.forward()`
  unconditionally also runs the image through CLIP ViT-B/32's own image encoder (fixed
  224px positional embedding, no interpolation support), for a diagnostic side-output
  unrelated to what this script needs.
- Must slice `patch_prototype_logits` down to the requested concept columns *before*
  accumulating across the corpus, not after — the full-vocabulary tensor (`V` ≈ 15,858
  VG words) OOMs even at a 32G allocation over ~600 images; eval_retreive_concepts.py's
  original code already did this correctly, the port initially didn't.
- Concept auto-suggestion needed a real-word filter (`nltk.corpus.words`) — uniform
  sampling over every POS-tagged noun/adjective in the un-curated, NLTK-auto-extracted
  15,858-word vocab mostly surfaced typos/extraction artifacts (`jerysey`, `mountial`,
  `withred`) rather than usable example concepts.

**Status**: Completed successfully after fixes (job `2818325`, using the corrected
concept-suggestion filter). Output: `results/concept_retrieval/concept_retrieval_Gref_val.png`
— not yet visually reviewed in this report; check the auto-suggested concept list in the
job log and pull the PNG down to confirm quality before using it in the paper.

## 4. CUB-200 concept-bottleneck classification (Label-free-CBM-style)

**Stage 1 — zero-shot** (M1, no fine-tuning): swap Label-free-CBM's 370 GPT-3-generated
CUB concept phrases into M1's prototype pool at inference. **12.58% top-1 @672px**
(10.70% @224px) — far below Label-free-CBM's own ~74.6%, but far above the 0.5% chance
floor, confirming the mechanism transfers, just not well, zero-shot from VG's general
vocabulary to fine-grained bird parts.

**Stage 2 — fine-tuning**, mirroring Label-free-CBM's actual 4-stage `train_cbm.py`
method (CLIP-cutoff filter → fine-tune vs. CLIP-similarity soft targets, warm-started
from M1 → interpretability-cutoff filter → sparse elastic-net classifier, standing in for
their unlicensed `glm_saga`):

| Stage | Concepts | Classifier | Top-1 | Top-5 |
|---|---|---|---|---|
| 1. Zero-shot | 370 | dense probe | 12.58% | — |
| 2. Before/after (isolates fine-tuning alone) | 370 | dense probe, fine-tuned ckpt | 60.32% | 90.37% |
| 2. Full pipeline | 248 (post interpretability filter) | sparse elastic-net (`C=1.0, l1_ratio=0.1`) | **67.41%** | **93.06%** |

Fine-tuning alone is the dominant effect (+47.7pt over zero-shot). Concept-filtering
(370→248) + swapping the dense probe for the sparse elastic-net classifier adds another
+7.1pt on top, while inducing genuine sparsity (25.84% of concept→class weights zeroed by
the L1 term) — closes a good chunk of the gap to Label-free-CBM's own ~74.6%, using a
properly-licensed sklearn substitute for their `glm_saga`.

Operational note: the full-pipeline eval job initially timed out mid-`GridSearchCV`
(36 multinomial-SAGA fits over 200 classes not converging within `max_iter=1000` on only
4 CPUs / 4h) — fixed by bumping `slurm_eval_cub_labelfreecbm_finetuned.sh` to
`--cpus-per-task=16 --time=12:00:00` (completed in ~2h40m).

**Supervision comparison** (relevant to team discussion re: comparing against
ProtoPNet-family models): our pipeline never exposes CUB species labels to the concept
representation itself — only CLIP-similarity (self-supervised, no human labels) trains
the concept encoder; species labels are used only to fit the final (Stage 2) or joint
(Stage 3, below) classifier. ProtoPNet-family models train end-to-end on species labels
throughout and report ~79–87% top-1 on CUB.

## 5. Joint training with a CLIP-substituted sufficiency regularizer

**Status: run, including the attribution ablation. Result: joint training helps a lot;
the sufficiency term specifically does not.**

| Run | Concepts | cls_coef | sufficiency_coef | Top-1 | Top-5 |
|---|---|---|---|---|---|
| Stage 2 (Sequential, §4 above) | 248 | — | — | 67.41% | 93.06% |
| Joint, with sufficiency | 339 | 1.0 | 1.0 | 85.71% | 97.69% |
| **Joint, sufficiency ablated off** | 339 | 1.0 | **0.0** | **85.67%** | **97.83%** |

The with/without-sufficiency runs are statistically indistinguishable (the ablated run's
top-5 is even marginally higher). **The ~18pt jump over Stage 2 comes from joint
end-to-end fine-tuning itself (classification loss backpropagating through the whole
concept encoder, not just a frozen-feature probe) — not from the CLIP-substituted
sufficiency regularizer.** The honest conclusion from this experiment is "joint training
beats sequential training on CUB," a real and useful result, but a more modest claim than
"the paper's sufficiency mechanism helps," which is what originally motivated it. Note
also the concept-count difference (339 vs. Stage 2's 248 — this run used the CLIP-cutoff-
filtered vocab, not the further interpretability-cutoff-filtered one), so the comparison
to Stage 2 isn't perfectly apples-to-apples on concept set, on top of the training-regime
difference.

Motivated by Espinosa Zarlenga, *"In Defense of Information Leakage in Concept-based
Models"* (ICML 2026): Sequential/Independent CBM training (what Stage 2 above does) is
provably capped below what's achievable whenever the concept set is incomplete — our
248-concept GPT-3-generated set, with no human verification, is a textbook incomplete
concept set. Their fix is joint training (concept loss + task loss together) plus a
"sufficiency" regularizer `L_int` = task loss when the bottleneck is intervened to
ground-truth concept values.

**We don't have ground-truth concept labels** — this experiment substitutes CLIP's own
raw image-vs-concept similarity scores for that ground truth. **This is a deliberate
weakening of what the paper proves, not a reproduction of it — every result from this
experiment must be reported with that caveat attached** (both new scripts print/save it
automatically).

Standalone pipeline, no changes to the shared `train.py`/`modeling/pnp.py`/
`clip_dataset.py` used by every other experiment:
- `scripts/train_cub_joint.py` — warm-started from the **same M1 base checkpoint**
  Stage 2 used (not Stage 2's already fine-tuned result, for a clean apples-to-apples
  comparison), jointly training the existing KL/SK/KoLeo concept-alignment losses
  (formulas duplicated from `PNPCriterion`, not imported, per the separation decision)
  alongside a fresh classifier head and two new loss terms: `L_cls` (classification loss
  from the model's own concept activations) and `L_suff` (the adapted sufficiency term,
  classification loss from CLIP's raw similarity scores standing in for ground truth).
  Both feed a shared `nn.Linear(n_concepts, 200)` head after per-sample z-score
  standardization (the two signals live in different embedding spaces / scales).
- `scripts/eval_cub_sufficiency_curve.py` — the closest analog to the paper's
  intervention curves available without real ground truth: sweeps the fraction of
  concepts CLIP-substituted (0-100%) and checks whether test-set accuracy rises. CLIP
  scores for the held-out test split are computed fresh (the cached
  `build_clip_vocab_scores.py` output only covers train+val, since it exists to build
  training targets).
- `scripts/slurm_train_cub_joint.sh`, `scripts/slurm_eval_cub_sufficiency_curve.sh` —
  sbatch launchers, reusing the concept cache / filtered vocab / CLIP scores Stage 2
  already built (no rebuild).

Run order: `bash scripts/slurm_train_cub_joint.sh` (defaults: `cls_coef=1.0
sufficiency_coef=1.0`, override via `CLS_COEF=... SUFFICIENCY_COEF=...` env vars), then
`bash scripts/slurm_eval_cub_sufficiency_curve.sh`.

**Next**: the sufficiency curve (`eval_cub_sufficiency_curve.py`) hasn't been checked yet
on either checkpoint — worth running on both the with- and without-sufficiency runs to
see whether the sufficiency term changed anything about the concept representation's
robustness even without changing accuracy, before writing this off entirely.

Full design/context: `C:\Users\preze\.claude\plans\golden-wondering-conway.md`.
