# Running the remaining visualizations locally

Athena's `plgrid-gpu-a100` partition is short on capacity right now (grant
allocation is fine per `hpc-grants` — it's queue congestion, see
`squeue`/`sinfo`). This machine has a usable GPU, so run these locally
instead of waiting.

**All commands below are PowerShell**, matching the actual working shell
(not git-bash) — two gotchas that bit the first attempt, fixed here for
good:
1. PowerShell doesn't understand bash's `\$VAR` escaping inside double
   quotes — `"...\$SCRATCH..."` sends a literal empty string, not `$SCRATCH`.
   Use **single quotes** for any remote command that needs `$SCRATCH`
   expanded *by the remote shell* (`ssh $ATHENA 'cmd with $SCRATCH'`).
2. Modern `scp` defaults to the SFTP protocol, which does **not** invoke a
   remote shell at all — `$SCRATCH` in an `scp` remote path never expands,
   correct quoting or not. Every `scp` below uses the resolved literal path
   instead.

Set these once per shell session:
```powershell
$ATHENA = "plgabedychaj@login01.athena.cyfronet.pl"
$SCRATCH_PATH = "/net/tscratch/people/plgabedychaj"   # literal value of Athena's $SCRATCH
$PY = "C:\Users\preze\PycharmProjects\proto-VLM\venv\Scripts\python.exe"
cd C:\Users\preze\PycharmProjects\proto-VLM\proto-non-param
```

## 0. One-time setup

```powershell
& $PY scripts\local\setup_local_env.py
```

Then (setup prints this too):
```powershell
$env:PYTHONPATH = "C:\Users\preze\PycharmProjects\proto-VLM\local_run\deps\dinov2;$env:PYTHONPATH"
```

## 1. Checkpoints + vocab caches (shared across all remaining experiments)

Check sizes first:
```powershell
ssh $ATHENA 'ls -lh $SCRATCH/train_logs/vg_contrastive/run_M1_vitl14_sk10_koleo01_30ep/ckpt.pth $SCRATCH/train_logs/cub_joint/ckpt.pth $SCRATCH/vocab/vg_cache.pt $SCRATCH/vocab/cub_clip_scores.pt $SCRATCH/vocab/cub_clip_scores_vocab_filtered.pt'
```

Pull them (`New-Item` first so the destination folder exists):
```powershell
New-Item -ItemType Directory -Force local_run\assets | Out-Null

scp "${ATHENA}:${SCRATCH_PATH}/train_logs/vg_contrastive/run_M1_vitl14_sk10_koleo01_30ep/ckpt.pth" local_run\assets\ckpt_vg_m1.pth
scp "${ATHENA}:${SCRATCH_PATH}/train_logs/cub_joint/ckpt.pth" local_run\assets\ckpt_cub_joint.pth
scp "${ATHENA}:${SCRATCH_PATH}/vocab/vg_cache.pt" local_run\assets\vg_cache.pt
scp "${ATHENA}:${SCRATCH_PATH}/vocab/cub_clip_scores.pt" local_run\assets\cub_clip_scores.pt
scp "${ATHENA}:${SCRATCH_PATH}/vocab/cub_clip_scores_vocab_filtered.pt" local_run\assets\cub_clip_scores_vocab_filtered.pt
```
(`${ATHENA}:` with braces, not `$ATHENA:` — PowerShell parses a bare
`$var:` as a scope-qualified variable reference and won't expand it the way
you'd expect.)

Patch both checkpoints' baked-in Athena vocab path so they load locally:
```powershell
& $PY scripts\local\patch_checkpoint_paths.py --ckpt local_run\assets\ckpt_vg_m1.pth --vocab-cache-path local_run\assets\vg_cache.pt
& $PY scripts\local\patch_checkpoint_paths.py --ckpt local_run\assets\ckpt_cub_joint.pth --vocab-cache-path local_run\assets\cub_clip_scores_vocab_filtered.pt
```
This writes `ckpt_vg_m1_local.pth` / `ckpt_cub_joint_local.pth` — use
*these* patched files as `--ckpt` below, not the raw downloads.

---

## 2. Prototype dictionary inspection — cheapest, zero images

Create a groups file (edit the words, one comma-separated group per line —
`<`/`>` placeholders break PowerShell, so this uses a real example instead):
```powershell
@"
cat,lion,dog
furniture,chair,table
"@ | Set-Content -Encoding utf8 local_run\assets\groups.txt
```

```powershell
& $PY scripts\inspect_prototype_dictionary.py `
  --ckpt local_run\assets\ckpt_vg_m1_local.pth `
  --groups-file local_run\assets\groups.txt `
  --out-dir results\prototype_dictionary
```

## 3. VG open-vocab grounding — needs ~10 arbitrary VG images

```powershell
$files = ssh $ATHENA 'ls $SCRATCH/vg/VG_100K | shuf -n 10'
New-Item -ItemType Directory -Force local_run\assets\vg_images\VG_100K | Out-Null
foreach ($f in $files) {
  scp "${ATHENA}:${SCRATCH_PATH}/vg/VG_100K/$f" local_run\assets\vg_images\VG_100K\
}
```
Any VG images work — this is qualitative, exact files don't matter. The
script expects `VG_100K/` and/or `VG_100K_2/` subfolders under `--vg-root`.

```powershell
& $PY scripts\visualize_vg_open_vocab_grounding.py `
  --ckpt local_run\assets\ckpt_vg_m1_local.pth `
  --vg-root local_run\assets\vg_images `
  --n-images 6 `
  --out-dir results\vg_open_vocab_grounding
```

## 4. CUB explain (joint) — CUB comes from the public archive, not Athena

```powershell
& $PY scripts\download_cub200.py --output-dir local_run\assets\cub200 --seed 42
```
(~1.1GB one-time download from Caltech's public dataset host — unrelated to
Athena, no scp needed. Builds `train/val/test/<class>/*.jpg` automatically.)

```powershell
& $PY scripts\explain_cub_concepts.py `
  --mode joint `
  --ckpt local_run\assets\ckpt_cub_joint_local.pth `
  --cub-root local_run\assets\cub200 `
  --cub-annotations local_run\assets\cub200\annotations `
  --clip-scores-cub local_run\assets\cub_clip_scores.pt `
  --n-random-classes 3 `
  --out-dir results\cub_explain_joint
```

## 5. Deletion faithfulness — SKIP, already done on Athena

Job `2825594` completed on Athena before local setup was finished:
`faithfulness_gap = -8.41` (CI excludes 0, but negative — the opposite of
what "faithful" needs). Dropped from the paper per
`EXPERIMENTS_REPORT.md` §7 — no local run needed for this one.

## 6. CUB sufficiency curve — heaviest, but compute-bound not download-bound

Uses the same `local_run\assets\cub200\` from step 4 (full test split, no
extra download). Watch VRAM (only ~4GB free per `nvidia-smi` last checked —
close other GPU apps first); drop `--batch-size` if it OOMs.

```powershell
& $PY scripts\eval_cub_sufficiency_curve.py `
  --ckpt local_run\assets\ckpt_cub_joint_local.pth `
  --cub-root local_run\assets\cub200 `
  --cub-annotations local_run\assets\cub200\annotations `
  --batch-size 8 `
  --out-dir results\cub_sufficiency_curve
```

## 7. Concept retrieval — needs the ~600 deduped images, not all 9536

`list_gref_dedup_images.py` needs to exist on Athena too first — either
push this branch and pull it there:
```powershell
git push origin local/conference-viz
ssh $ATHENA 'cd ~/proto-non-param && git fetch origin && git checkout local/conference-viz && git pull'
```
or `scp` just that one file up if you'd rather not touch Athena's checkout:
```powershell
scp scripts\local\list_gref_dedup_images.py "${ATHENA}:${SCRATCH_PATH}/list_gref_dedup_images.py"
```
(adjust the `cd`/script path below accordingly if you use this route).

Then, on Athena's **login node** (no sbatch, no GPU, no queue wait — pure
IO/CPU, takes a couple minutes over the full `val_batch/` dir):
```powershell
$dedupFiles = ssh $ATHENA 'cd ~/proto-non-param && python scripts/local/list_gref_dedup_images.py --data-root $SCRATCH/data/refcoco --dataset Gref --split val'
```
(progress/counts print to stderr and show in the console; `$dedupFiles`
only captures the filename list from stdout.)

```powershell
New-Item -ItemType Directory -Force local_run\assets\refcoco_local\Gref\val_batch | Out-Null
foreach ($f in $dedupFiles) {
  scp "${ATHENA}:${SCRATCH_PATH}/data/refcoco/Gref/val_batch/$f" local_run\assets\refcoco_local\Gref\val_batch\
}
```

```powershell
& $PY scripts\visualize_concept_retrieval.py `
  --ckpt local_run\assets\ckpt_vg_m1_local.pth `
  --data-root local_run\assets\refcoco_local `
  --concepts umbrella bicycle striped wooden `
  --out-dir results\concept_retrieval
```
(swap `umbrella bicycle striped wooden` for the words you actually want —
space-separated, `--concepts` takes `nargs="+"`; no `<`/`>` around them.)
