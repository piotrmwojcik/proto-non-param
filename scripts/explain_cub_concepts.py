#!/usr/bin/env python3
"""
Concept-bottleneck interpretability report for CUB: two directions.

1. Per-image: for a given test image, the top-5 concepts whose activation
   (raw, unstandardized vocab_logits -- the model's own patch/prototype
   similarity, not classifier-internal preprocessing) is highest. "What did
   the model see in this photo."

2. Per-class, two views (see Espinosa Zarlenga et al.'s Sec 6.2 "top-m weight
   overlap" for the same technique on the weight side):
   - Classifier weight: sort that class's row in the classifier's learned
     weight matrix, top-5 highest (most positive) values. "What the
     classifier learned to rely on" for that class.
   - Average activation: average the raw concept-activation vector across
     every test image of that class, top-5. "What's visually common" for
     that species -- a different, activation-based (not classifier-based)
     view. The two don't have to agree; where they diverge is itself
     informative (e.g. shortcut concepts the classifier weights heavily but
     that aren't actually visually distinctive, or vice versa).

Two modes, since the two CUB pipelines' classifiers are structurally
different (dense nn.Linear vs. sklearn sparse elastic-net):

  --mode joint       train_cub_joint.py's checkpoint (nn.Linear cls_head)
  --mode sequential   Stage 2's activations cache + fit_sparse_cub_probe.py's
                      fitted sklearn model (coef_)

Usage:
  # Joint checkpoint
  python scripts/explain_cub_concepts.py --mode joint \
    --ckpt $SCRATCH/train_logs/cub_joint/ckpt.pth \
    --cub-root $SCRATCH/cub200 --cub-annotations $SCRATCH/cub200/annotations \
    --image-indices 0 1 2 --class-names Black_footed_Albatross Indigo_Bunting \
    --out-dir results/cub_explain_joint

  # Sequential (Stage 2) checkpoint
  python scripts/explain_cub_concepts.py --mode sequential \
    --activations-cache eval_results/cub_concepts_stage2/activations_img672.pt \
    --sklearn-model eval_results/cub_concepts_stage2/sparse_probe_model.joblib \
    --concepts-file eval_results/cub_concepts_stage2/concepts_final.txt \
    --cub-root $SCRATCH/cub200 \
    --image-indices 0 1 2 --class-names Black_footed_Albatross Indigo_Bunting \
    --out-dir results/cub_explain_sequential
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, SCRIPT_DIR)

from evaluate_pnp_cub_concepts import build_class_index, list_split  # noqa: E402


def top5(values, names, k=5):
    """values: 1D array-like. Returns [(name, value), ...] sorted descending."""
    values = np.asarray(values)
    idx = np.argsort(-values)[:k]
    return [(names[i], round(float(values[i]), 4)) for i in idx]


# ---------------------------------------------------------------------------
# Joint mode
# ---------------------------------------------------------------------------

def run_joint(args):
    from train_cub_joint import build_pnp, CubJointDataset
    import torch.nn as nn

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.ckpt, map_location="cpu")
    hparams = ckpt["hparams"]

    net = build_pnp(hparams["vocab_cache_path"], device)
    net.load_state_dict(ckpt["state_dict"])
    net.eval()

    cls_head = nn.Linear(net.vocab_size, hparams["num_classes"]).to(device)
    cls_head.load_state_dict(ckpt["cls_head_state_dict"])
    cls_head.eval()

    concept_names = net.vocab_words
    class_to_idx = build_class_index(args.cub_root)
    idx_to_class = {i: name for name, i in class_to_idx.items()}

    test_ds = CubJointDataset(args.cub_root, args.cub_annotations, args.clip_scores_cub,
                              class_to_idx, splits=("test",), train=False)

    def image_activation(index):
        img_tensor, _, _, _ = test_ds[index]
        with torch.no_grad():
            outputs = net(img_tensor.unsqueeze(0).to(device))
        return outputs["vocab_logits"][0].cpu().numpy()

    def class_weight_row(class_idx):
        return cls_head.weight[class_idx].detach().cpu().numpy()

    def class_avg_activation(class_idx):
        rows = [i for i, (_, lab) in enumerate(test_ds.samples) if lab == class_idx]
        if not rows:
            return None
        acts = np.stack([image_activation(i) for i in rows])
        return acts.mean(axis=0)

    return concept_names, class_to_idx, idx_to_class, image_activation, class_weight_row, class_avg_activation


# ---------------------------------------------------------------------------
# Sequential mode
# ---------------------------------------------------------------------------

def run_sequential(args):
    import joblib

    cache = torch.load(args.activations_cache, map_location="cpu")
    test_x = cache["test_x"].numpy()
    test_y = cache["test_y"].numpy()

    bundle = joblib.load(args.sklearn_model)
    model = bundle["model"]

    with open(args.concepts_file) as f:
        concept_names = [line.strip() for line in f if line.strip()]
    assert len(concept_names) == test_x.shape[1], (
        f"{args.concepts_file} has {len(concept_names)} concepts but the activations "
        f"cache has {test_x.shape[1]} columns -- wrong concepts file for this cache?"
    )

    class_to_idx = build_class_index(args.cub_root)
    idx_to_class = {i: name for name, i in class_to_idx.items()}

    def image_activation(index):
        return test_x[index]

    def class_weight_row(class_idx):
        return model.coef_[class_idx]

    def class_avg_activation(class_idx):
        rows = test_x[test_y == class_idx]
        if len(rows) == 0:
            return None
        return rows.mean(axis=0)

    return concept_names, class_to_idx, idx_to_class, image_activation, class_weight_row, class_avg_activation


# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", required=True, choices=("joint", "sequential"))
    p.add_argument("--cub-root", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--image-indices", type=int, nargs="*", default=[0, 1, 2],
                   help="Indices into the test split to explain (default: first 3)")
    p.add_argument("--class-names", type=str, nargs="*", default=None,
                   help="CUB class folder names to explain (default: 3 random classes)")
    p.add_argument("--seed", type=int, default=0)
    # joint mode
    p.add_argument("--ckpt", help="[joint] train_cub_joint.py checkpoint")
    p.add_argument("--cub-annotations", help="[joint]")
    p.add_argument("--clip-scores-cub", help="[joint] needed to construct CubJointDataset "
                                              "(unused for activation/weight computation itself)")
    p.add_argument("--device", default="cuda")
    # sequential mode
    p.add_argument("--activations-cache", help="[sequential] evaluate_pnp_cub_concepts.py cache")
    p.add_argument("--sklearn-model", help="[sequential] fit_sparse_cub_probe.py's sparse_probe_model.joblib")
    p.add_argument("--concepts-file", help="[sequential] MUST be the same concepts file used to "
                                            "build --activations-cache, or columns misalign")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    if args.mode == "joint":
        for name in ("ckpt", "cub_annotations", "clip_scores_cub"):
            if getattr(args, name) is None:
                p.error(f"--{name.replace('_', '-')} is required for --mode joint")
        (concept_names, class_to_idx, idx_to_class,
         image_activation, class_weight_row, class_avg_activation) = run_joint(args)
    else:
        for name in ("activations_cache", "sklearn_model", "concepts_file"):
            if getattr(args, name) is None:
                p.error(f"--{name.replace('_', '-')} is required for --mode sequential")
        (concept_names, class_to_idx, idx_to_class,
         image_activation, class_weight_row, class_avg_activation) = run_sequential(args)

    class_names = args.class_names
    if not class_names:
        rng = np.random.default_rng(args.seed)
        class_names = list(rng.choice(sorted(class_to_idx.keys()), size=3, replace=False))
        print(f"No --class-names given, sampled: {class_names}")

    report = {"mode": args.mode, "per_image": [], "per_class": []}

    print("\n=== Per-image: top-5 activating concepts ===")
    for idx in args.image_indices:
        act = image_activation(idx)
        top = top5(act, concept_names)
        print(f"  image[{idx}]: {top}")
        report["per_image"].append({"image_index": idx, "top5_concepts": top})

    print("\n=== Per-class: top-5 by classifier weight vs. top-5 by avg activation ===")
    for cname in class_names:
        if cname not in class_to_idx:
            print(f"  WARNING: class '{cname}' not found, skipping "
                  f"(available e.g.: {sorted(class_to_idx.keys())[:3]}...)")
            continue
        cidx = class_to_idx[cname]

        weight_top = top5(class_weight_row(cidx), concept_names)
        avg_act = class_avg_activation(cidx)
        avg_top = top5(avg_act, concept_names) if avg_act is not None else None

        print(f"  {cname} (class {cidx}):")
        print(f"    by classifier weight : {weight_top}")
        print(f"    by avg activation    : {avg_top}")
        report["per_class"].append({
            "class_name": cname,
            "class_index": cidx,
            "top5_by_weight": weight_top,
            "top5_by_avg_activation": avg_top,
        })

    out_path = os.path.join(args.out_dir, f"explain_{args.mode}.json")
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
