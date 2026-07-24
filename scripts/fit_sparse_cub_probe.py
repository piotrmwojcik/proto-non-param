#!/usr/bin/env python3
"""
Stage 2's fourth Label-free-CBM step: a sparse elastic-net-regularized final
classifier over concept activations, in place of Stage 1's plain dense probe.

Label-free-CBM trains this via their own glm_saga (GPU-batched proximal SAGA,
computing a full regularization path) -- their repo has no license, so it
can't be vendored. At our scale (~7K images x <=370 concepts x 200 classes)
their path-solver machinery solves a much bigger problem than we have;
sklearn's LogisticRegression(penalty="elasticnet", solver="saga") is the
properly-licensed standard equivalent, with a small (C, l1_ratio) grid
standing in for their path. Hyperparameters are chosen by cross-validation
on the fit set only (GridSearchCV) -- the test set is touched exactly once,
after the grid search, to avoid leaking test performance into model choice.

Loads the activations_img{size}.pt cache produced by evaluate_pnp_cub_concepts.py
(unchanged -- just re-run it against the fine-tuned checkpoint and the final,
stage-3-filtered concept list to regenerate this cache).

Usage:
  python scripts/fit_sparse_cub_probe.py \
    --activations-cache eval_results/cub_concepts_stage2/activations_img672.pt \
    --out-dir eval_results/cub_concepts_stage2
"""

import argparse
import json
import os

import joblib
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler


def main():
    p = argparse.ArgumentParser(description="Fit a sparse elastic-net CUB concept-bottleneck classifier")
    p.add_argument("--activations-cache", required=True,
                   help="activations_img{size}.pt from evaluate_pnp_cub_concepts.py")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--Cs", type=float, nargs="+", default=[0.01, 0.1, 1.0, 10.0],
                   help="Inverse regularization strength grid (smaller = sparser).")
    p.add_argument("--l1-ratios", type=float, nargs="+", default=[0.1, 0.5, 0.9],
                   help="Elastic-net mixing grid (1.0 = pure L1/lasso, 0.0 = pure L2/ridge).")
    p.add_argument("--cv-folds", type=int, default=3)
    p.add_argument("--max-iter", type=int, default=1000)
    args = p.parse_args()

    cache = torch.load(args.activations_cache, map_location="cpu")
    fit_x = cache["fit_x"].numpy()
    fit_y = cache["fit_y"].numpy()
    test_x = cache["test_x"].numpy()
    test_y = cache["test_y"].numpy()
    n_classes = int(max(fit_y.max(), test_y.max())) + 1
    print(f"Fit: {fit_x.shape}   Test: {test_x.shape}   Classes: {n_classes}")

    # Label-free-CBM normalizes concept activations before the final layer.
    scaler = StandardScaler().fit(fit_x)
    fit_x_norm = scaler.transform(fit_x)
    test_x_norm = scaler.transform(test_x)

    print(f"Grid search: Cs={args.Cs} l1_ratios={args.l1_ratios} "
          f"({args.cv_folds}-fold CV on fit set only)...")
    base = LogisticRegression(
        penalty="elasticnet", solver="saga", max_iter=args.max_iter,
        multi_class="multinomial",
    )
    search = GridSearchCV(
        base, {"C": args.Cs, "l1_ratio": args.l1_ratios},
        cv=args.cv_folds, n_jobs=-1,
    )
    search.fit(fit_x_norm, fit_y)
    best = search.best_estimator_
    print(f"Best params: {search.best_params_}  (CV accuracy: {100*search.best_score_:.2f}%)")

    test_top1 = float(best.score(test_x_norm, test_y))
    proba = best.predict_proba(test_x_norm)
    top5_pred = np.argsort(-proba, axis=1)[:, :5]
    test_top5 = float(np.mean([y in row for y, row in zip(test_y, top5_pred)]))

    n_nonzero = int(np.count_nonzero(best.coef_))
    sparsity = 1.0 - n_nonzero / best.coef_.size

    print(f"\nTest top-1: {100*test_top1:.2f}%   top-5: {100*test_top5:.2f}%")
    print(f"Sparsity: {100*sparsity:.1f}% zero weights "
          f"({n_nonzero}/{best.coef_.size} nonzero, {best.coef_.shape[1]} concepts x {n_classes} classes)")

    os.makedirs(args.out_dir, exist_ok=True)
    result = {
        "activations_cache": args.activations_cache,
        "n_concepts": int(fit_x.shape[1]),
        "n_classes": n_classes,
        "n_fit": int(fit_x.shape[0]),
        "n_test": int(test_x.shape[0]),
        "best_params": search.best_params_,
        "cv_accuracy": round(100 * search.best_score_, 4),
        "top1_acc": round(100 * test_top1, 4),
        "top5_acc": round(100 * test_top5, 4),
        "sparsity_pct": round(100 * sparsity, 4),
    }
    out_path = os.path.join(args.out_dir, "sparse_probe_result.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved result to {out_path}")

    # Fitted model + scaler weren't persisted before -- needed by
    # explain_cub_concepts.py to read per-class weights (coef_) directly
    # instead of refitting.
    model_path = os.path.join(args.out_dir, "sparse_probe_model.joblib")
    joblib.dump({"model": best, "scaler": scaler}, model_path)
    print(f"Saved fitted model to {model_path}")


if __name__ == "__main__":
    main()
