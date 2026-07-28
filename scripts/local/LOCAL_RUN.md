# Running the 6 stuck-on-Athena visualizations locally

Athena's `plgrid-gpu-a100` partition is short on capacity right now (see
`hpc-grants`/`squeue`/`sinfo` — grant allocation is fine, it's queue
congestion). This machine has a usable GPU, so run these locally instead of
waiting. `athena` below is a placeholder SSH alias — adjust to whatever's in
your `~/.ssh/config`.

All commands assume the venv python and repo root:
```
PY = C:\Users\preze\PycharmProjects\proto-VLM\venv\Scripts\python.exe
cd C:\Users\preze\PycharmProjects\proto-VLM\proto-non-param
```

## 0. One-time setup

```
& $PY scripts\local\setup_local_env.py
```

Then set `PYTHONPATH` in every shell you run these from (setup prints the
exact line; PowerShell shown, bash equivalent in the script's own output):
```
$env:PYTHONPATH = "C:\Users\preze\PycharmProjects\proto-VLM\local_run\deps\dinov2;$env:PYTHONPATH"
```

## 1. Checkpoints + vocab caches (shared across all 6 experiments)

Check sizes first, then pull:
```
ssh athena "ls -lh \$SCRATCH/train_logs/vg_contrastive/run_M1_vitl14_sk10_koleo01_30ep/ckpt.pth \$SCRATCH/train_logs/cub_joint/ckpt.pth \$SCRATCH/vocab/vg_cache.pt \$SCRATCH/vocab/cub_clip_scores.pt \$SCRATCH/vocab/cub_clip_scores_vocab_filtered.pt"

scp athena:'$SCRATCH/train_logs/vg_contrastive/run_M1_vitl14_sk10_koleo01_30ep/ckpt.pth' local_run\assets\ckpt_vg_m1.pth
scp athena:'$SCRATCH/train_logs/cub_joint/ckpt.pth' local_run\assets\ckpt_cub_joint.pth
scp athena:'$SCRATCH/vocab/vg_cache.pt' local_run\assets\vg_cache.pt
scp athena:'$SCRATCH/vocab/cub_clip_scores.pt' local_run\assets\cub_clip_scores.pt
scp athena:'$SCRATCH/vocab/cub_clip_scores_vocab_filtered.pt' local_run\assets\cub_clip_scores_vocab_filtered.pt
```

Patch both checkpoints' baked-in Athena vocab path so they load locally:
```
& $PY scripts\local\patch_checkpoint_paths.py --ckpt local_run\assets\ckpt_vg_m1.pth --vocab-cache-path local_run\assets\vg_cache.pt
& $PY scripts\local\patch_checkpoint_paths.py --ckpt local_run\assets\ckpt_cub_joint.pth --vocab-cache-path local_run\assets\cub_clip_scores_vocab_filtered.pt
```
This writes `ckpt_vg_m1_local.pth` / `ckpt_cub_joint_local.pth` — use
*these* patched files as `--ckpt` below, not the raw downloads.

---

## 2. Prototype dictionary inspection — cheapest, zero images

```
& $PY scripts\inspect_prototype_dictionary.py `
  --ckpt local_run\assets\ckpt_vg_m1_local.pth `
  --groups-file <your groups file> `
  --out-dir results\prototype_dictionary
```

## 3. VG open-vocab grounding — needs ~10 arbitrary VG images

```
ssh athena "ls \$SCRATCH/vg/VG_100K | shuf -n 10"
```
scp those 10 filenames into `local_run\assets\vg_images\` (any VG images
work — this is qualitative, exact files don't matter):
```
scp athena:'$SCRATCH/vg/VG_100K/<file1>' athena:'$SCRATCH/vg/VG_100K/<file2>' ... local_run\assets\vg_images\
```
The script expects `VG_100K/` and/or `VG_100K_2/` subfolders under
`--vg-root`, so put the downloaded jpgs in
`local_run\assets\vg_images\VG_100K\`.

```
& $PY scripts\visualize_vg_open_vocab_grounding.py `
  --ckpt local_run\assets\ckpt_vg_m1_local.pth `
  --vg-root local_run\assets\vg_images `
  --n-images 6 `
  --out-dir results\vg_open_vocab_grounding
```

## 4. CUB explain (joint) — CUB comes from the public archive, not Athena

```
& $PY scripts\download_cub200.py --output-dir local_run\assets\cub200 --seed 42
```
(~1.1GB one-time download from Caltech's public dataset host — unrelated to
Athena, no scp needed. Builds `train/val/test/<class>/*.jpg` automatically.)

```
& $PY scripts\explain_cub_concepts.py `
  --mode joint `
  --ckpt local_run\assets\ckpt_cub_joint_local.pth `
  --cub-root local_run\assets\cub200 `
  --cub-annotations local_run\assets\cub200\annotations `
  --clip-scores-cub local_run\assets\cub_clip_scores.pt `
  --n-random-classes 3 `
  --out-dir results\cub_explain_joint
```

## 5. Deletion faithfulness — needs a small random subset of Gref/val_batch

```
ssh athena "ls \$SCRATCH/data/refcoco/Gref/val_batch | shuf -n 200 > \$HOME/gref_subset.txt"
scp athena:'$HOME/gref_subset.txt' local_run\assets\gref_subset.txt
```
Then scp just those 200 files (loop locally over the downloaded list —
e.g. in git-bash: `while read f; do scp "athena:\$SCRATCH/data/refcoco/Gref/val_batch/$f" local_run/assets/refcoco_local/Gref/val_batch/; done < local_run/assets/gref_subset.txt`).

```
& $PY scripts\eval_deletion_faithfulness.py `
  --ckpt local_run\assets\ckpt_vg_m1_local.pth `
  --data_root local_run\assets\refcoco_local `
  --n-samples 200 `
  --out-dir results\deletion_faithfulness_Gref_val
```
(`--n-samples 200` matches however many you actually downloaded — doesn't
need to match Athena's exact seeded 300, any random subset gives a valid
faithfulness-gap figure.)

## 6. CUB sufficiency curve — heaviest, but compute-bound not download-bound

Uses the same `local_run\assets\cub200\` from step 4 (full test split, no
extra download). Watch VRAM (only ~4GB free per `nvidia-smi` last checked —
close other GPU apps first); drop `--batch-size` if it OOMs.

```
& $PY scripts\eval_cub_sufficiency_curve.py `
  --ckpt local_run\assets\ckpt_cub_joint_local.pth `
  --cub-root local_run\assets\cub200 `
  --cub-annotations local_run\assets\cub200\annotations `
  --batch-size 8 `
  --out-dir results\cub_sufficiency_curve
```

## 7. Concept retrieval — needs the ~600 deduped images, not all 9536

First, on Athena's **login node** (no sbatch, no GPU, no queue wait — pure
IO/CPU, takes a couple minutes over the full `val_batch/` dir):
```
ssh athena "cd ~/proto-non-param && python scripts/local/list_gref_dedup_images.py --data-root \$SCRATCH/data/refcoco --dataset Gref --split val > \$HOME/gref_dedup_files.txt"
scp athena:'$HOME/gref_dedup_files.txt' local_run\assets\gref_dedup_files.txt
```
(`list_gref_dedup_images.py` needs to exist on Athena too — either `git
push` this branch and `git pull` there, or `scp` just that one file up
first.)

Then scp the listed files (same loop pattern as step 5, but pull the
target list from `gref_dedup_files.txt`) into the **same**
`local_run\assets\refcoco_local\Gref\val_batch\` folder used in step 5 —
`ReferDataset` just globs whatever's present, no harm in the two file sets
overlapping/coexisting there.

```
& $PY scripts\visualize_concept_retrieval.py `
  --ckpt local_run\assets\ckpt_vg_m1_local.pth `
  --data-root local_run\assets\refcoco_local `
  --concepts <your new concept words> `
  --out-dir results\concept_retrieval
```
