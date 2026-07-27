#!/usr/bin/env python3
"""
Concept-bottleneck interpretability report for CUB: two directions.

1. Per-image, two views:
   - Top activating concepts: raw, unstandardized vocab_logits -- the
     model's own patch/prototype similarity, not classifier-internal
     preprocessing. "What did the model see in this photo."
   - Top CONTRIBUTING concepts (matches Label-free-CBM's own
     evaluate_cbm.ipynb/plots.py figure -- verified against their actual
     code, not assumed): standardized_activation[concept] * weight[predicted
     class, concept], ranked by |contribution| (their plots.py bar() sorts by
     np.argsort(np.abs(values))). This decomposes the model's own predicted-
     class logit into per-concept summands (sum of contributions + bias ==
     that logit) -- "why did the model predict what it predicted for THIS
     photo," as opposed to the class-level views below.

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

Per-class results are also saved as figures (plots/<class>_by_weight.png,
plots/<class>_by_avg_activation.png): an example image of the class next to
a horizontal bar chart of its top concepts, one figure per view. --topk
controls how many concepts appear (default 5).

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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

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


def topk_by_abs(values, names, k=5):
    """Same as top5 but ranked by |value|, matching Label-free-CBM's own
    plots.py bar() (np.argsort(np.abs(values))[::-1]) -- a large negative
    contribution (concept present, but arguing AGAINST the predicted class)
    is as notable as a large positive one."""
    values = np.asarray(values)
    idx = np.argsort(-np.abs(values))[:k]
    return [(names[i], round(float(values[i]), 4)) for i in idx]


def plot_class_concepts(class_name, image_path, concept_scores, title_suffix, out_path,
                        signed=False):
    """Example image (left) + horizontal bar chart of concept scores (right),
    highest |score| at top -- mirrors the standard Label-free-CBM-style figure.
    signed=True colors negative bars red (contributions arguing against the
    class) instead of uniform green."""
    names = [n for n, _ in concept_scores]
    values = [v for _, v in concept_scores]
    colors = ["#4C9F70" if v >= 0 else "#C0392B" for v in values] if signed else "#4C9F70"

    fig, (ax_img, ax_bar) = plt.subplots(
        1, 2, figsize=(11, 0.45 * len(names) + 1.5),
        gridspec_kw={"width_ratios": [1, 1.6]}, dpi=140,
    )

    if image_path is not None:
        ax_img.imshow(Image.open(image_path).convert("RGB"))
    ax_img.axis("off")
    ax_img.set_title(f"Class: {class_name.replace('_', ' ')}", fontsize=17,
                     fontweight="bold", loc="left")

    y = np.arange(len(names))
    ax_bar.barh(y, values, color=colors)
    ax_bar.set_yticks(y)
    ax_bar.set_yticklabels(names, fontsize=13)
    ax_bar.invert_yaxis()  # highest |score| at top
    for yi, v in zip(y, values):
        ha = "right" if v >= 0 else "left"
        ax_bar.text(v * 0.98, yi, f"{v:.3f}", va="center", ha=ha,
                    color="white", fontweight="bold", fontsize=12)
    if signed:
        ax_bar.axvline(0, color="black", linewidth=0.8)
    ax_bar.set_xlabel("Concept Score", fontsize=13)
    ax_bar.tick_params(axis="x", labelsize=11)
    ax_bar.set_title(title_suffix, fontsize=16, fontweight="bold")
    for spine in ("top", "right"):
        ax_bar.spines[spine].set_visible(False)

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Joint mode
# ---------------------------------------------------------------------------

def run_joint(args):
    from train_cub_joint import build_pnp, CubJointDataset, standardize
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

    def class_example_path(class_idx):
        for path, lab in test_ds.samples:
            if lab == class_idx:
                return path
        return None

    def image_path(index):
        return test_ds.samples[index][0]

    def image_true_label(index):
        return test_ds.samples[index][1]

    def image_contribution(index):
        """standardized_activation[concept] * weight[predicted_class, concept],
        for THIS image's own model prediction -- matches Label-free-CBM's
        evaluate_cbm.ipynb figure. Returns (pred_class_idx, contributions,
        bias, pred_logit); sum(contributions) + bias == pred_logit by
        construction (same standardization the classifier was trained on)."""
        act = torch.from_numpy(image_activation(index)).float().unsqueeze(0).to(device)
        std_act = standardize(act)
        with torch.no_grad():
            logits = cls_head(std_act)[0]
        pred_idx = int(logits.argmax().item())
        weight_row = cls_head.weight[pred_idx].detach().cpu().numpy()
        bias = float(cls_head.bias[pred_idx].detach().cpu().item())
        contributions = std_act[0].detach().cpu().numpy() * weight_row
        return pred_idx, contributions, bias, float(logits[pred_idx].item())

    return {
        "concept_names": concept_names, "class_to_idx": class_to_idx, "idx_to_class": idx_to_class,
        "image_activation": image_activation, "class_weight_row": class_weight_row,
        "class_avg_activation": class_avg_activation, "class_example_path": class_example_path,
        "image_path": image_path, "image_true_label": image_true_label,
        "image_contribution": image_contribution,
    }


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
    scaler = bundle["scaler"]

    with open(args.concepts_file) as f:
        concept_names = [line.strip() for line in f if line.strip()]
    assert len(concept_names) == test_x.shape[1], (
        f"{args.concepts_file} has {len(concept_names)} concepts but the activations "
        f"cache has {test_x.shape[1]} columns -- wrong concepts file for this cache?"
    )

    class_to_idx = build_class_index(args.cub_root)
    idx_to_class = {i: name for name, i in class_to_idx.items()}

    # list_split iterates the same sorted (class folder, filename) order
    # encode_split used to build test_x/test_y, so row i here == row i there.
    test_samples = list_split(args.cub_root, "test", class_to_idx)
    assert len(test_samples) == test_x.shape[0], (
        f"{args.cub_root}/test has {len(test_samples)} images but the activations "
        f"cache has {test_x.shape[0]} rows -- mismatched --cub-root for this cache?"
    )

    def image_activation(index):
        return test_x[index]

    def class_weight_row(class_idx):
        return model.coef_[class_idx]

    def class_avg_activation(class_idx):
        rows = test_x[test_y == class_idx]
        if len(rows) == 0:
            return None
        return rows.mean(axis=0)

    def class_example_path(class_idx):
        for path, lab in test_samples:
            if lab == class_idx:
                return path
        return None

    def image_path(index):
        return test_samples[index][0]

    def image_true_label(index):
        return int(test_y[index])

    def image_contribution(index):
        """standardized_activation[concept] * weight[predicted_class, concept],
        for THIS image's own model prediction -- matches Label-free-CBM's
        evaluate_cbm.ipynb figure. Returns (pred_class_idx, contributions,
        bias, pred_logit); sum(contributions) + bias == pred_logit by
        construction (same StandardScaler the classifier was trained on)."""
        std_act = scaler.transform(test_x[index].reshape(1, -1))[0]
        decision = model.decision_function(std_act.reshape(1, -1))[0]  # [n_classes]
        pred_idx = int(np.argmax(decision))
        weight_row = model.coef_[pred_idx]
        bias = float(model.intercept_[pred_idx])
        contributions = std_act * weight_row
        return pred_idx, contributions, bias, float(decision[pred_idx])

    return {
        "concept_names": concept_names, "class_to_idx": class_to_idx, "idx_to_class": idx_to_class,
        "image_activation": image_activation, "class_weight_row": class_weight_row,
        "class_avg_activation": class_avg_activation, "class_example_path": class_example_path,
        "image_path": image_path, "image_true_label": image_true_label,
        "image_contribution": image_contribution,
    }


# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", required=True, choices=("joint", "sequential"))
    p.add_argument("--cub-root", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--image-indices", type=int, nargs="*", default=[0, 1, 2],
                   help="Indices into the test split to explain (default: first 3)")
    p.add_argument("--class-names", type=str, nargs="*", default=None,
                   help="CUB class folder names to explain (default: --n-random-classes random ones)")
    p.add_argument("--n-random-classes", type=int, default=3,
                   help="How many random classes to sample when --class-names isn't given")
    p.add_argument("--seed", type=int, default=0,
                   help="Seed for random class sampling -- change this to get different examples")
    p.add_argument("--topk", type=int, default=5, help="Concepts per plot/report (default: 5)")
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
        ctx = run_joint(args)
    else:
        for name in ("activations_cache", "sklearn_model", "concepts_file"):
            if getattr(args, name) is None:
                p.error(f"--{name.replace('_', '-')} is required for --mode sequential")
        ctx = run_sequential(args)

    concept_names = ctx["concept_names"]
    class_to_idx = ctx["class_to_idx"]
    idx_to_class = ctx["idx_to_class"]
    image_activation = ctx["image_activation"]
    class_weight_row = ctx["class_weight_row"]
    class_avg_activation = ctx["class_avg_activation"]
    class_example_path = ctx["class_example_path"]
    image_path = ctx["image_path"]
    image_true_label = ctx["image_true_label"]
    image_contribution = ctx["image_contribution"]

    class_names = args.class_names
    if not class_names:
        rng = np.random.default_rng(args.seed)
        class_names = list(rng.choice(sorted(class_to_idx.keys()),
                                      size=args.n_random_classes, replace=False))
        print(f"No --class-names given, sampled (seed={args.seed}): {class_names}")

    plot_dir = os.path.join(args.out_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    report = {"mode": args.mode, "per_image": [], "per_class": []}

    print(f"\n=== Per-image: top-{args.topk} activating concepts, and top-{args.topk} "
          f"CONTRIBUTING concepts (Label-free-CBM-style) ===")
    for idx in args.image_indices:
        act = image_activation(idx)
        top = top5(act, concept_names, k=args.topk)

        pred_idx, contributions, bias, pred_logit = image_contribution(idx)
        contrib_top = topk_by_abs(contributions, concept_names, k=args.topk)
        true_idx = image_true_label(idx)
        pred_name = idx_to_class[pred_idx]
        true_name = idx_to_class[true_idx]

        # Sanity check: contributions + bias must reconstruct the predicted
        # logit exactly (same standardization the classifier was trained on).
        recon_err = abs(float(contributions.sum()) + bias - pred_logit)
        if recon_err > 1e-2:
            print(f"  WARNING: image[{idx}] contribution reconstruction error "
                  f"{recon_err:.4f} (expected ~0) -- check standardization")

        print(f"  image[{idx}]  true={true_name}  predicted={pred_name}"
              f"{' (correct)' if pred_idx == true_idx else ' (WRONG)'}")
        print(f"    top activating   : {top}")
        print(f"    top contributing : {contrib_top}")

        img_path = image_path(idx)
        contrib_plot_path = os.path.join(plot_dir, f"image{idx}_contributions.png")
        plot_class_concepts(
            pred_name, img_path, contrib_top,
            f"Most Strongly Contributing Concepts\n(predicted: {pred_name.replace('_', ' ')})",
            contrib_plot_path, signed=True,
        )

        report["per_image"].append({
            "image_index": idx,
            "true_class": true_name,
            "predicted_class": pred_name,
            "correct": pred_idx == true_idx,
            "top5_activating": top,
            "top5_contributing": contrib_top,
            "reconstruction_error": round(recon_err, 6),
        })

    print(f"\n=== Per-class: top-{args.topk} by classifier weight vs. top-{args.topk} by avg activation ===")
    for cname in class_names:
        if cname not in class_to_idx:
            print(f"  WARNING: class '{cname}' not found, skipping "
                  f"(available e.g.: {sorted(class_to_idx.keys())[:3]}...)")
            continue
        cidx = class_to_idx[cname]
        example_path = class_example_path(cidx)

        weight_top = top5(class_weight_row(cidx), concept_names, k=args.topk)
        avg_act = class_avg_activation(cidx)
        avg_top = top5(avg_act, concept_names, k=args.topk) if avg_act is not None else None

        print(f"  {cname} (class {cidx}):")
        print(f"    by classifier weight : {weight_top}")
        print(f"    by avg activation    : {avg_top}")

        weight_plot_path = os.path.join(plot_dir, f"{cname}_by_weight.png")
        plot_class_concepts(cname, example_path, weight_top,
                            "Most Strongly Contributing Concepts\n(by classifier weight)",
                            weight_plot_path)
        if avg_top is not None:
            avg_plot_path = os.path.join(plot_dir, f"{cname}_by_avg_activation.png")
            plot_class_concepts(cname, example_path, avg_top,
                                "Most Strongly Contributing Concepts\n(by average activation)",
                                avg_plot_path)

        report["per_class"].append({
            "class_name": cname,
            "class_index": cidx,
            "example_image": example_path,
            "top5_by_weight": weight_top,
            "top5_by_avg_activation": avg_top,
        })

    out_path = os.path.join(args.out_dir, f"explain_{args.mode}.json")
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
