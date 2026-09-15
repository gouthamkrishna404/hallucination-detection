"""Classifier comparison: does a more expressive model (Random Forest,
XGBoost, calibrated variants) extract more signal than plain logistic
regression from the same features?

Run against BOTH the original 3-feature set and the full 17-feature rich
set, so "a better classifier" and "more features" are tested as
independent axes rather than conflated. Every classifier uses the SAME
train/eval split as every other result in this project.

With only 140 training examples, high-capacity models (Random Forest,
XGBoost) are at real risk of overfitting -- that's not a caveat to bury,
it's part of what this comparison is FOR: if CV and held-out AUROC
diverge sharply for a complex model but stay close for logistic
regression, that itself is the finding (small-n favors simple models),
and is reported as such rather than picking whichever number looks best.

Usage:
    python scripts/experiment_classifier_comparison.py --model qwen2.5-1.5b-instruct --dataset truthfulqa
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score

from src.config import CFG, set_all_seeds
from src.features_v2 import ALL_RICH_FEATURE_NAMES, ORIGINAL_THREE_FEATURES
from src.metrics_utils import compute_metrics, save_json
from src.registry import get_paths
from scripts.experiment_train_evaluate import get_qid_split

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False


def build_classifiers():
    clfs = {
        "logistic_regression": LogisticRegression(random_state=CFG.seed, max_iter=1000),
        "logistic_regression_platt": CalibratedClassifierCV(
            LogisticRegression(random_state=CFG.seed, max_iter=1000), method="sigmoid", cv=5),
        "logistic_regression_isotonic": CalibratedClassifierCV(
            LogisticRegression(random_state=CFG.seed, max_iter=1000), method="isotonic", cv=5),
        "random_forest": RandomForestClassifier(
            n_estimators=200, max_depth=4, min_samples_leaf=5, random_state=CFG.seed),
    }
    if HAS_XGBOOST:
        clfs["xgboost"] = XGBClassifier(
            n_estimators=100, max_depth=3, learning_rate=0.1, reg_lambda=1.0,
            eval_metric="logloss", random_state=CFG.seed)
    return clfs


def evaluate(name, clf, train_df, eval_df, feature_cols) -> dict:
    X_train, y_train = train_df[feature_cols].values, train_df["label"].values
    X_eval, y_eval = eval_df[feature_cols].values, eval_df["label"].values

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=CFG.seed)
    cv_scores = cross_val_score(clf, X_train, y_train, cv=skf, scoring="roc_auc")

    clf.fit(X_train, y_train)
    y_prob = clf.predict_proba(X_eval)[:, 1]
    held_out = compute_metrics(y_eval, y_prob)

    overfit_gap = float(cv_scores.mean() - held_out["auroc"])
    result = {
        "cv_auroc_mean": float(cv_scores.mean()),
        "cv_auroc_std": float(cv_scores.std()),
        "held_out_auroc": held_out["auroc"],
        "held_out_accuracy": held_out["accuracy"],
        "held_out_f1": held_out["f1"],
        "cv_minus_heldout_gap": overfit_gap,
    }
    print(f"  {name:32s} cv={cv_scores.mean():.3f}±{cv_scores.std():.3f}  held_out={held_out['auroc']:.3f}  gap={overfit_gap:+.3f}")
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--dataset", default="truthfulqa")
    args = ap.parse_args()

    if not HAS_XGBOOST:
        print("NOTE: xgboost not installed -- skipping that classifier (others still run).")

    set_all_seeds()
    paths = get_paths(args.model, args.dataset)
    rich = pd.read_csv(paths.features / "single_pass_features_rich.csv")
    sp = pd.read_csv(paths.features / "single_pass_features.csv")
    train_qids, eval_qids = get_qid_split(sp)
    train_df = rich[rich.qid.isin(train_qids)].reset_index(drop=True)
    eval_df = rich[rich.qid.isin(eval_qids)].reset_index(drop=True)

    all_results = {}
    for feature_set_name, feature_cols in [("original_3feat", ORIGINAL_THREE_FEATURES),
                                             ("all_17_rich_features", ALL_RICH_FEATURE_NAMES)]:
        print(f"\n--- Feature set: {feature_set_name} ({len(feature_cols)} features) ---")
        all_results[feature_set_name] = {}
        for clf_name, clf in build_classifiers().items():
            all_results[feature_set_name][clf_name] = evaluate(clf_name, clf, train_df, eval_df, feature_cols)

    out_path = paths.analysis / "classifier_comparison.json"
    save_json(all_results, out_path)
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
