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


def main():
    print("== Installing open_clip_torch ==")
    subprocess.run([sys.executable, "-m", "pip", "install", "open_clip_torch"], check=True)

    print("== Cloning dinov2 (needed on PYTHONPATH, not pip-installable) ==")
    if DINOV2_DIR.exists():
        print(f"  Already present at {DINOV2_DIR}")
    else:
        DINOV2_DIR.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "https://github.com/facebookresearch/dinov2", str(DINOV2_DIR)], check=True)

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
