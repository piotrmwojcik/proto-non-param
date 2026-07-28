"""
Patch a downloaded PNP checkpoint's baked-in hparams paths so it loads
locally instead of on Athena.

Every checkpoint stores its hparams dict (ckpt["hparams"]) with an
absolute Athena path, e.g. hparams["vocab_cache_path"] =
"/net/tscratch/people/plgabedychaj/vocab/vg_cache.pt". evaluate_pnp_refer.
build_model() only prepends REPO_ROOT when the path is NOT absolute
(os.path.isabs) -- and on Windows, os.path.isabs() treats a leading "/"
as absolute too, so it's used as-is and fails to resolve. This rewrites
the path(s) in-place and saves a new checkpoint file, leaving the
original download untouched.

Usage:
  python scripts/local/patch_checkpoint_paths.py \
    --ckpt local_run/assets/ckpt_vg_m1.pth \
    --vocab-cache-path local_run/assets/vg_cache.pt

  # Any other absolute /net/tscratch/... string in hparams is reported
  # (not rewritten) so you notice if a script needs one you haven't set:
  python scripts/local/patch_checkpoint_paths.py --ckpt local_run/assets/ckpt_cub_joint.pth \
    --vocab-cache-path local_run/assets/cub_clip_scores_vocab_filtered.pt
"""
import argparse
import os

import torch


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", required=True, help="Downloaded checkpoint to patch")
    p.add_argument("--vocab-cache-path", required=True, help="Local path to the vocab cache .pt file")
    p.add_argument("--out", default=None, help="Output path (default: <ckpt>_local.pth)")
    args = p.parse_args()

    ckpt = torch.load(args.ckpt, map_location="cpu")
    assert "hparams" in ckpt, "Checkpoint has no 'hparams' key -- not a PNP training checkpoint?"
    hparams = ckpt["hparams"]

    old = hparams.get("vocab_cache_path")
    new = os.path.abspath(args.vocab_cache_path)
    print(f"vocab_cache_path: {old!r} -> {new!r}")
    hparams["vocab_cache_path"] = new

    for k, v in hparams.items():
        if k != "vocab_cache_path" and isinstance(v, str) and v.startswith("/net/"):
            print(f"WARNING: hparams[{k!r}] = {v!r} is still an Athena-only path, unpatched "
                  f"-- only relevant if the script you're running actually reads it.")

    out_path = args.out or os.path.splitext(args.ckpt)[0] + "_local.pth"
    torch.save(ckpt, out_path)
    print(f"Saved patched checkpoint: {out_path}")


if __name__ == "__main__":
    main()
