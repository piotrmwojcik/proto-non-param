"""
One-time local environment setup for running PNP eval/visualization
scripts on this machine instead of on Athena.

Usage (run with the SAME python you'll use for everything else, e.g. the
venv's python.exe):
  python scripts/local/setup_local_env.py
"""
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]          # .../proto-non-param
PROTO_VLM_ROOT = REPO_ROOT.parent                          # .../proto-VLM
LOCAL_RUN_DIR = PROTO_VLM_ROOT / "local_run"
DINOV2_DIR = LOCAL_RUN_DIR / "deps" / "dinov2"
ASSETS_DIR = LOCAL_RUN_DIR / "assets"

# dinov2's HEAD (commit 12592a9, "dino.txt inference code (#528)") added
# `float | None`-style union-type annotations to dinov2/layers/attention.py,
# which Python evaluates eagerly at class-definition time and which raises
# `TypeError: unsupported operand type(s) for |: 'type' and 'NoneType'` on
# Python <3.10 (this venv is 3.9, matching Athena's).
#
# The obvious fix -- pin to 81b2b64, the commit right before the license
# update that predates 12592a9 -- turned out to be too far back: it also
# predates 9c7e324 ("Add new backbones trained with registers (#282)"),
# and this checkpoint's config needs num_register_tokens (it loads
# dinov2_vitl14_reg4_pretrain.pth). 112792f is the last commit before the
# float | None-breaking pair (b48308a/12592a9) that still has register
# support and the positional-embedding interpolation fix (e1277af,
# relevant since we run at non-224 --img-size).
DINOV2_PIN_COMMIT = "112792f"


def main():
    print("== Installing open_clip_torch ==")
    subprocess.run([sys.executable, "-m", "pip", "install", "open_clip_torch"], check=True)

    print("== Cloning dinov2 (needed on PYTHONPATH, not pip-installable) ==")
    if DINOV2_DIR.exists():
        print(f"  Already present at {DINOV2_DIR}")
    else:
        DINOV2_DIR.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "https://github.com/facebookresearch/dinov2", str(DINOV2_DIR)], check=True)

    print(f"  Pinning to {DINOV2_PIN_COMMIT} (Python 3.9-compatible)")
    subprocess.run(["git", "-C", str(DINOV2_DIR), "checkout", DINOV2_PIN_COMMIT], check=True)

    ASSETS_DIR.mkdir(parents=True, exist_ok=True)

    print("== Smoke-testing imports (with dinov2 on PYTHONPATH) ==")
    sys.path.insert(0, str(DINOV2_DIR))
    import torch
    print("torch", torch.__version__, "cuda:", torch.cuda.is_available())
    import open_clip
    print("open_clip OK")
    from dinov2.models.vision_transformer import DinoVisionTransformer  # noqa: F401
    print("dinov2 OK")

    print()
    print("Setup done. Every local run needs this PYTHONPATH set in the shell first:")
    print(f'  PowerShell:  $env:PYTHONPATH = "{DINOV2_DIR};$env:PYTHONPATH"')
    print(f'  bash:        export PYTHONPATH="{DINOV2_DIR}:$PYTHONPATH"')
    print("Local assets (checkpoints, data) go under:")
    print(f"  {ASSETS_DIR}")


if __name__ == "__main__":
    main()
