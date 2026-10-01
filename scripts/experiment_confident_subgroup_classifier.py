"""The sharpest possible version of "can we detect confidently-wrong
hallucinations": restrict to ONLY the confident bucket (above-median
mean_logprob) and ask whether any feature set can separate
wrong-but-confident from correct-but-confident answers -- i.e., can we
tell the two apart WHEN THE MODEL ITSELF GIVES NO HINT via low
confidence? This is the single hardest and most relevant test for the
"confidently-wrong hallucination" problem the whole investigation is
about.

AUDIT NOTE (fixed after the initial version): the median-split threshold
that decides subgroup membership must be fit on training data only, per
fold -- exactly as scripts/experiment_cascade_detector.py already does.
The first version of this script computed the median over the full
200-question set BEFORE cross-validating within the resulting subgroup, a
mild covariate-leakage inconsistency (the routing decision for a
fold's test questions was influenced by feature values of other
questions that later land in that same fold's test set). This version
fits the threshold, and every classifier, using ONLY the current fold's
training data -- 5-fold CV over the FULL 200 questions, with confident-
subgroup membership decided per fold, matching the cascade's rigor
exactly. (In practice the median is a stable statistic and the fix moved
every reported AUROC by <=0.02 -- but the corrected, leakage-free numbers
are the ones reported everywhere else in this project, not the original
ones.)

Usage:
    python scripts/experiment_confident_subgroup_classifier.py --model qwen2.5-1.5b-instruct --dataset truthfulqa
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
from scripts.experiment_best_candidate import CANDIDATE_FEATURES, bootstrap_auroc_ci


def confident_subgroup_oof(df: pd.DataFrame, feature_cols: list[str], n_splits: int = 5, seed: int = CFG.seed):
    """Out-of-fold predictions restricted to the confident subgroup, with
    the subgroup threshold fit PER FOLD on that fold's training data only
    -- a question's fold membership and subgroup routing never depend on
    any information from its own fold's test split."""
    X = df[feature_cols].values
    y = df["label"].values
    mean_lp = df["mean_logprob"].values

    oof_prob, oof_mask = np.full(len(df), np.nan), np.zeros(len(df), dtype=bool)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for train_idx, test_idx in skf.split(X, y):
        threshold = np.median(mean_lp[train_idx])  # train-fold only
        train_confident = train_idx[mean_lp[train_idx] >= threshold]
        test_confident = test_idx[mean_lp[test_idx] >= threshold]
        if len(test_confident) == 0:
            continue
        if len(np.unique(y[train_confident])) < 2 or len(train_confident) < 10:
            continue  # this fold's confident-training-branch is degenerate; skip its confident test questions
        clf = LogisticRegression(random_state=seed, max_iter=1000)
        clf.fit(X[train_confident], y[train_confident])
        oof_prob[test_confident] = clf.predict_proba(X[test_confident])[:, 1]
        oof_mask[test_confident] = True

    return y[oof_mask], oof_prob[oof_mask]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--dataset", default="truthfulqa")
    ap.add_argument("--features", default="rich", choices=["rich", "clean"])
    args = ap.parse_args()

    set_all_seeds()
    paths = get_paths(args.model, args.dataset)
    rich = pd.read_csv(paths.features / f"single_pass_features_{args.features}.csv")

    # Reported purely for context (how big is "the confident half" typically) -- NOT used for any
    # classification decision; every actual train/test split below refits its own per-fold threshold.
    global_median = rich["mean_logprob"].median()
    n_confident_global = int((rich["mean_logprob"] >= global_median).sum())
    n_wrong_confident_global = int(rich[rich["mean_logprob"] >= global_median]["label"].sum())
    print(f"[{args.model}/{args.dataset}] for reference, global-median confident subgroup: "
          f"n={n_confident_global}, n_wrong={n_wrong_confident_global} (not used in any fold below)")

    results = {"global_median_reference": {"n": n_confident_global, "n_wrong": n_wrong_confident_global}}
    for name, feats in [("original_3feat", ORIGINAL_THREE_FEATURES), ("candidate_4feat", CANDIDATE_FEATURES),
                          ("all_17_rich", ALL_RICH_FEATURE_NAMES)]:
        y, prob = confident_subgroup_oof(rich, feats, n_splits=5)
        ci = bootstrap_auroc_ci(y, prob, n_boot=2000)
        ci["n_evaluated"] = len(y)
        results[name] = ci
        flag = "EXCLUDES chance" if ci["excludes_chance"] else "does not exclude chance"
        print(f"  {name:16s} n_eval={len(y):3d}  5-fold-OOF AUROC={ci['point_estimate']:.3f}  "
              f"95% CI=[{ci['ci_lower_2.5pct']:.3f}, {ci['ci_upper_97.5pct']:.3f}]  ({flag})")

    out_path = paths.analysis / f"confident_subgroup_classifier_{args.features}.json"
    save_json(results, out_path)
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()
