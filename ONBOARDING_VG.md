# Running VG Experiments on Athena (PLGRID)

Prerequisites and step-by-step guide for training and evaluating PNP on Visual Genome.

---

## 1. Cluster access

You need a PLGRID account with access to the `plgunhype-gpu-a100` allocation.
Set your scratch directory:

```bash
export SCRATCH=/net/tscratch/people/<your_plg_id>
```

Add this to your `~/.bashrc` so all scripts pick it up automatically.

---

## 2. Python environment

Create a virtualenv in scratch (not home — home quota is small):

```bash
python3.9 -m venv ${SCRATCH}/venv
source ${SCRATCH}/venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install wandb open_clip_torch scikit-learn
```

Install DINOv2 at the exact commit the code expects:

```bash
cd ${SCRATCH}
git clone https://github.com/facebookresearch/dinov2.git
cd dinov2
git checkout e1277af2ba9496fbadf7aec6eba56e8d882d1e35
pip install --no-deps -e .
```

Verify the install:

```bash
python -c "import dinov2; print('ok')"
```

---

## 3. Environment variables

Every SLURM script requires these — set them or rely on the defaults:

| Variable | Default in scripts | What it points to |
|---|---|---|
| `SCRATCH` | `/net/tscratch/people/plgabedychaj` | Your scratch directory |
| `VG_ROOT` | `${SCRATCH}/vg` | VG images + region_descriptions.json |
| `VOCAB_DIR` | `${SCRATCH}/vocab` | Pre-built cache files |
| `LOG_DIR` | `${SCRATCH}/train_logs/vg_contrastive` | Checkpoint output |
| `WANDB_ENTITY` | `gmum` | Your W&B team/entity |
| `DATA_ROOT` | `${SCRATCH}/data/refcoco` | RefCOCO/RefCOCO+/RefCOCOg for eval |

Override any of these before submitting a job:

```bash
SCRATCH=/net/tscratch/people/<your_plg_id> bash scripts/slurm_train_vg_contrastive_msn.sh
```

Also set these in `~/.bashrc` to avoid re-exporting every session:

```bash
export TORCH_HOME=${SCRATCH}/torch_cache
export PYTHONPATH=${SCRATCH}/dinov2:$PYTHONPATH
export HF_HOME=${SCRATCH}/hf_cache
export TRANSFORMERS_CACHE=${SCRATCH}/hf_cache
```

---

## 4. Data setup

Run these once. Each step is a prerequisite for the next.

### 4a. Download Visual Genome (~12h, CPU node)

```bash
sbatch scripts/slurm_download_vg.sh
```

Produces:
```
${SCRATCH}/vg/
├── VG_100K/              # images part 1 (~57k images)
├── VG_100K_2/            # images part 2 (~51k images)
└── region_descriptions.json
```

### 4b. Build vocab cache (~1h, 1 GPU)

Extracts nouns and adjectives from region descriptions and CLIP-encodes them.

```bash
sbatch scripts/slurm_build_vg_vocab.sh
```

Produces: `${SCRATCH}/vocab/vg_cache.pt` — `dict[word → Tensor[512]]`, 15 858 words.

### 4c. Build caption embeddings (~2h, 1 GPU)

CLIP-encodes full region phrases per image. Required for contrastive training.

```bash
sbatch scripts/slurm_build_vg_vocab.sh   # if not already done
# then:
python vocab/build_vg_caption_embeddings.py \
  --vg-root ${SCRATCH}/vg \
  --vg-region-descriptions ${SCRATCH}/vg/region_descriptions.json \
  --output ${SCRATCH}/vocab/vg_caption_embs.pt
```

Or submit as a SLURM job — see `scripts/slurm_build_vg_caption_embeddings.sh` if it exists.

Produces: `${SCRATCH}/vocab/vg_caption_embs.pt`

### 4d. Download RefCOCO/RefCOCO+/RefCOCOg (for evaluation)

```
${SCRATCH}/data/refcoco/
├── refcoco/   (unc split)
├── refcoco+/  (unc+ split)
└── refcocog/  (Gref split)
```

These are standard REFER benchmark files. Download from the official REFER repo or ask a team member for the path on the shared storage.

---

## 5. Training

All training scripts live in `scripts/slurm_train_vg_*.sh`. They all:
- Activate `${SCRATCH}/venv`
- Set `TORCH_HOME`, `PYTHONPATH`
- Check that required cache files exist
- Submit a single SLURM job to `plgrid-gpu-a100`, 1× A100, 64 GB, 2-day limit

### Current experiment runs

| Script | Run | Config | Notes |
|---|---|---|---|
| `slurm_train_vg_contrastive_vitl_best.sh` | H/I | ViT-L, k=1 or k=5 | Baseline ViT-L |
| `slurm_train_vg_contrastive_sk.sh` | J | ViT-L + SK | Sinkhorn-Knopp diversity |
| `slurm_train_vg_contrastive_koleo.sh` | K1, K2 | ViT-L + KoLeo / SK+KoLeo | KoLeo repulsion |
| `slurm_train_vg_contrastive_K3.sh` | K3 | ViT-L + k=5 hard-mine + KoLeo | |
| `slurm_train_vg_contrastive_msn.sh` | L | ViT-L + MSN | Masked prototype prediction |

Submit any run:

```bash
bash scripts/slurm_train_vg_contrastive_msn.sh
```

Monitor:

```bash
squeue -u $USER
tail -f ${SCRATCH}/logs/vg_contrastive_L_msn_<job_id>.err
```

Checkpoints are saved every 5 epochs to `${LOG_DIR}/run_<name>/ckpt.pth`.

---

## 6. Evaluation (zero-shot RIS)

After training completes, run the 7-job evaluation (one job per dataset/split).

```bash
bash scripts/slurm_eval_vg_contrastive_L.sh
```

This evaluates on:
- RefCOCOg val
- RefCOCO (unc): val, testA, testB
- RefCOCO+ (unc+): val, testA, testB

Results land in `eval_results/vg_contrastive/contr_<run>/pnp_refer/`.

### Compare all runs

```bash
python scripts/compare_ris_results.py \
  --eval_dir eval_results \
  --ablation-dir eval_results/vg_contrastive \
  --ablation-type vg_contrastive
```

### Threshold sweep (find best operating point)

```bash
bash scripts/slurm_eval_pnp_threshold_sweep.sh   # submits sweep jobs
python scripts/summarize_threshold_sweep.py \
  --sweep-dir eval_results/threshold_sweep/<run_name> \
  --metric oIoU
```

---

## 7. Adding a new run

1. Add a new training SLURM script to `scripts/` (copy the closest existing one)
2. Add the new args to `train.py` argparse if needed
3. Add the run key and label to `scripts/compare_ris_results.py` → `ABLATION_VARIANTS["vg_contrastive"]`
4. Add an eval SLURM script (`slurm_eval_vg_contrastive_<name>.sh`)
5. Commit and push

---

## 8. Key loss terms (reference)

| Flag | Loss | Applied to | Purpose |
|---|---|---|---|
| `--kl-coef` | KL divergence | vocab_logits vs. region word distribution | Primary supervised loss |
| `--contrastive-coef` | InfoNCE | pred_text_embedding vs. caption CLIP | Image-text alignment |
| `--sk-coef` | Sinkhorn-Knopp | vocab_logits | Batch-level prototype diversity |
| `--koleo-coef` | KoLeo repulsion | pred_text_embedding | Push batch embeddings apart |
| `--msn-coef` | MSN masked CE | vocab_logits_masked vs. full | Prototype prediction from partial patches |

---

## 9. Common issues

**`TORCH_HOME` not set → DINOv2 re-downloads weights every job**
Set `export TORCH_HOME=${SCRATCH}/torch_cache` before submitting or it will hit network on each run.

**`PYTHONPATH` missing dinov2 → `ModuleNotFoundError: dinov2`**
Add `export PYTHONPATH=${SCRATCH}/dinov2:$PYTHONPATH` to the job.

**`vg_cache.pt` not found → training exits immediately**
Run step 4b first.

**ViT-L OOM with batch > 64**
Use `--batch-size 64` for `dinov2_vitl14`. ViT-B/14 fits 128.

**`git pull` hangs on Athena**
SSH key passphrase blocks without a TTY. Either load the key into ssh-agent
(`eval $(ssh-agent -s) && ssh-add ~/.ssh/id_ed25519`) or use scp/rsync from
your local machine to copy changed files directly.
