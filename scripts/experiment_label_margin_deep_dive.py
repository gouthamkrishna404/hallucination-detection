"""Deep dive on the label-margin finding: is the apparent AUROC rise at
higher label-confidence thresholds (scripts/experiment_label_margin_audit.py)
a real, explainable effect, or a small-sample artifact?

The original audit used a single fixed 60-question eval split, which at
margin>=0.2 leaves as few as 8-14 eval questions -- nowhere near enough to
trust a point estimate. This script applies the same higher-power
out-of-fold + bootstrap-CI procedure used everywhere else in the final
verification pass: every question in the margin-restricted subset gets a
prediction from a fold that never trained on it (5-fold here, since
n is already small), then bootstraps a 95% CI over ALL of those
out-of-fold predictions -- the best-powered honest answer available from
the existing 200-question dataset.

Usage:
    python scripts/experiment_label_margin_deep_dive.py --model qwen2.5-1.5b-instruct --dataset truthfulqa
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from src.config import CFG, set_all_seeds
from src.features_v2 import ORIGINAL_THREE_FEATURES
from src.metrics_utils import save_json
from src.registry import get_paths
from scripts.experiment_best_candidate import CANDIDATE_FEATURES, bootstrap_auroc_ci


def oof_predictions_small_n(df, feature_cols, n_splits, seed=CFG.seed):
    X = df[feature_cols].values
    y = df["label"].values
    oof_prob = np.full(len(df), np.nan)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for train_idx, test_idx in skf.split(X, y):
        clf = LogisticRegression(random_state=seed, max_iter=1000)
        clf.fit(X[train_idx], y[train_idx])
        oof_prob[test_idx] = clf.predict_proba(X[test_idx])[:, 1]
    return y, oof_prob


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--dataset", default="truthfulqa")
    args = ap.parse_args()

    set_all_seeds()
    paths = get_paths(args.model, args.dataset)
    rich = pd.read_csv(paths.features / "single_pass_features_rich.csv")
    rich["label_margin"] = (rich["correct_sim"] - rich["incorrect_sim"]).abs()

    print(f"[{args.model}/{args.dataset}]")
    results = {}
    for thresh in [0.0, 0.1, 0.2]:
        confident = rich[rich["label_margin"] >= thresh].reset_index(drop=True)
        n = len(confident)
        n_pos = int(confident["label"].sum())
        n_splits = 5 if min(n_pos, n - n_pos) >= 5 else 3
        if min(n_pos, n - n_pos) < n_splits:
            print(f"  margin>={thresh}: n={n} (n_pos={n_pos}) -- too few of one class for stratified CV, skipping")
            continue

        row = {"threshold": thresh, "n": n, "n_pos": n_pos, "n_splits": n_splits}
        for name, feats in [("original_3feat", ORIGINAL_THREE_FEATURES), ("candidate_4feat", CANDIDATE_FEATURES)]:
            y, prob = oof_predictions_small_n(confident, feats, n_splits)
            ci = bootstrap_auroc_ci(y, prob, n_boot=2000)
            row[name] = ci
            flag = "EXCLUDES chance" if ci["excludes_chance"] else "does not exclude chance"
            print(f"  margin>={thresh}  n={n:3d} (n_pos={n_pos})  {name:16s} "
                  f"{n_splits}-fold-OOF AUROC={ci['point_estimate']:.3f}  "
                  f"95% CI=[{ci['ci_lower_2.5pct']:.3f}, {ci['ci_upper_97.5pct']:.3f}]  ({flag})")
        results[f"margin_gte_{thresh}"] = row

    out_path = paths.analysis / "label_margin_deep_dive.json"
    save_json(results, out_path)
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
