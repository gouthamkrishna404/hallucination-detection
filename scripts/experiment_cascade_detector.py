"""Capstone detector: a two-stage cascade motivated directly by
experiment_confident_subgroup_classifier.py's finding that a
17-feature classifier can distinguish confidently-wrong from
confidently-correct answers (AUROC 0.64-0.69 for 2/3 models) even though
raw confidence alone cannot.

Stage 1: route each question to "confident" or "uncertain" by its
mean_logprob relative to a threshold.
Stage 2: within EACH branch, apply a dedicated all-17-rich-feature
classifier (fit separately per branch).

Both the routing threshold AND both branch classifiers are fit ONLY on
each fold's training data (5-fold CV over the full 200 questions) and
applied to that fold's held-out test questions -- so the median-split
threshold itself cannot leak test-set information, unlike a naive
global-median split computed once on all 200 questions.

Usage:
    python scripts/experiment_cascade_detector.py --model qwen2.5-1.5b-instruct --dataset truthfulqa
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from src.config import CFG, set_all_seeds
from src.features_v2 import ALL_RICH_FEATURE_NAMES, ORIGINAL_THREE_FEATURES
from src.metrics_utils import save_json
from src.registry import get_paths
from scripts.experiment_best_candidate import bootstrap_auroc_ci


def cascade_oof(df: pd.DataFrame, feature_cols: list[str], n_splits: int = 5, seed: int = CFG.seed):
    X = df[feature_cols].values
    y = df["label"].values
    mean_lp = df["mean_logprob"].values
    oof_prob = np.zeros(len(df))

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for train_idx, test_idx in skf.split(X, y):
        threshold = np.median(mean_lp[train_idx])  # fit on TRAIN fold only

        train_confident_mask = mean_lp[train_idx] >= threshold
        test_confident_mask = mean_lp[test_idx] >= threshold

        for mask_name, train_mask, test_mask in [("confident", train_confident_mask, test_confident_mask),
                                                    ("uncertain", ~train_confident_mask, ~test_confident_mask)]:
            branch_train_idx = train_idx[train_mask]
            branch_test_idx = test_idx[test_mask]
            if len(branch_test_idx) == 0:
                continue
            if len(np.unique(y[branch_train_idx])) < 2 or len(branch_train_idx) < 10:
                # not enough data/classes in this branch this fold -- fall back to the
                # full training fold instead of a branch-specific model
                clf = LogisticRegression(random_state=seed, max_iter=1000)
                clf.fit(X[train_idx], y[train_idx])
            else:
                clf = LogisticRegression(random_state=seed, max_iter=1000)
                clf.fit(X[branch_train_idx], y[branch_train_idx])
            oof_prob[branch_test_idx] = clf.predict_proba(X[branch_test_idx])[:, 1]

    return y, oof_prob


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--dataset", default="truthfulqa")
    args = ap.parse_args()

    set_all_seeds()
    paths = get_paths(args.model, args.dataset)
    rich = pd.read_csv(paths.features / "single_pass_features_rich.csv")

    y, prob_cascade = cascade_oof(rich, ALL_RICH_FEATURE_NAMES)
    ci_cascade = bootstrap_auroc_ci(y, prob_cascade)

    # Reference: a single flat all-17-feature classifier over the whole dataset, same CV scheme
    from scripts.experiment_best_candidate import out_of_fold_predictions
    y_flat, prob_flat = out_of_fold_predictions(rich, ALL_RICH_FEATURE_NAMES, n_splits=5)
    ci_flat = bootstrap_auroc_ci(y_flat, prob_flat)

    y_orig, prob_orig = out_of_fold_predictions(rich, ORIGINAL_THREE_FEATURES, n_splits=5)
    ci_orig = bootstrap_auroc_ci(y_orig, prob_orig)

    print(f"[{args.model}/{args.dataset}]  (all via 5-fold OOF across full n=200)")
    for name, ci in [("original_3feat_flat", ci_orig), ("all_17_flat", ci_flat), ("two_stage_cascade_17feat", ci_cascade)]:
        flag = "EXCLUDES chance" if ci["excludes_chance"] else "does not exclude chance"
        print(f"  {name:28s} AUROC={ci['point_estimate']:.3f}  95% CI=[{ci['ci_lower_2.5pct']:.3f}, {ci['ci_upper_97.5pct']:.3f}]  ({flag})")

    result = {"original_3feat_flat": ci_orig, "all_17_flat": ci_flat, "two_stage_cascade_17feat": ci_cascade}
    out_path = paths.analysis / "cascade_detector.json"
    save_json(result, out_path)
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()
