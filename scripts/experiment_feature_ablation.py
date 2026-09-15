"""Feature-engineering ablation: does any of the expanded feature set (17
candidates from src/features_v2.py, vs. the original 3) carry more signal
for TruthfulQA-style hallucination detection?

For every feature, individually and in several hand-picked and automatic
combinations, reports BOTH:
  - 5-fold cross-validated AUROC on the 140-question TRAIN split only
    (the more reliable number when n is this small -- a single 60-question
    held-out AUROC has wide variance, see "Known limitations")
  - the actual held-out eval AUROC (what every other result in this
    project reports, for apples-to-apples comparability)

A feature/combo is only worth taking seriously if BOTH numbers clear
chance by a real margin -- CV-only or eval-only outperformance on n=140/60
is exactly the kind of thing that regresses to nothing on more data, and
this project's rule is not to report what looks good, but what's real.

Usage:
    python scripts/experiment_feature_ablation.py --model qwen2.5-1.5b-instruct --dataset truthfulqa
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score

from src.config import CFG, set_all_seeds
from src.features_v2 import ALL_RICH_FEATURE_NAMES, ORIGINAL_THREE_FEATURES
from src.metrics_utils import compute_metrics, save_json
from src.registry import get_paths
from src.train_eval import fit_and_predict
from scripts.experiment_train_evaluate import get_qid_split


def cv_auroc(train_df: pd.DataFrame, feature_cols: list[str], label_col: str = "label", n_splits: int = 5) -> dict:
    X = train_df[feature_cols].values
    y = train_df[label_col].values
    clf = LogisticRegression(random_state=CFG.seed, max_iter=1000)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=CFG.seed)
    scores = cross_val_score(clf, X, y, cv=skf, scoring="roc_auc")
    return {"cv_mean": float(scores.mean()), "cv_std": float(scores.std()), "cv_folds": scores.tolist()}


def evaluate_feature_set(name: str, train_df, eval_df, feature_cols: list[str]) -> dict:
    cv = cv_auroc(train_df, feature_cols)
    _, y_eval, y_prob = fit_and_predict(train_df, eval_df, feature_cols)
    held_out = compute_metrics(y_eval, y_prob)
    result = {
        "features": feature_cols,
        "n_features": len(feature_cols),
        "cv_auroc_mean": cv["cv_mean"],
        "cv_auroc_std": cv["cv_std"],
        "held_out_auroc": held_out["auroc"],
        "held_out_accuracy": held_out["accuracy"],
        "held_out_f1": held_out["f1"],
    }
    print(f"  {name:45s} cv_auroc={cv['cv_mean']:.3f}±{cv['cv_std']:.3f}  held_out_auroc={held_out['auroc']:.3f}  (n_feat={len(feature_cols)})")
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--dataset", default="truthfulqa")
    args = ap.parse_args()

    set_all_seeds()
    paths = get_paths(args.model, args.dataset)
    rich = pd.read_csv(paths.features / "single_pass_features_rich.csv")

    # Reuse the EXACT SAME train/eval split as every other result for this model/dataset,
    # derived from single_pass_features.csv's label column (identical qids, identical order).
    sp = pd.read_csv(paths.features / "single_pass_features.csv")
    train_qids, eval_qids = get_qid_split(sp)
    train_df = rich[rich.qid.isin(train_qids)].reset_index(drop=True)
    eval_df = rich[rich.qid.isin(eval_qids)].reset_index(drop=True)
    print(f"[{args.model}/{args.dataset}] train={len(train_df)} eval={len(eval_df)}\n")

    results = {}

    print("--- Individual features ---")
    for feat in ALL_RICH_FEATURE_NAMES:
        results[f"single__{feat}"] = evaluate_feature_set(feat, train_df, eval_df, [feat])

    print("\n--- Original baseline + hand-picked combinations ---")
    combos = {
        "original_3feat": ORIGINAL_THREE_FEATURES,
        "location_spread": ["mean_logprob", "median_logprob", "min_logprob", "logprob_p10", "logprob_p25", "std_logprob"],
        "entropy_group": ["mean_entropy", "max_entropy", "std_entropy"],
        "margin_group": ["mean_margin", "min_margin"],
        "positional_group": ["first_half_mean_logprob", "second_half_mean_logprob", "logprob_delta_second_minus_first", "logprob_trend_slope"],
        "coverage_plus_mean": ["mean_logprob", "frac_low_confidence"],
        "all_17_features": ALL_RICH_FEATURE_NAMES,
    }
    for name, feats in combos.items():
        results[f"combo__{name}"] = evaluate_feature_set(name, train_df, eval_df, feats)

    # Automatic: top-5 individual features by CV AUROC, combined
    individual_cv = {k.replace("single__", ""): v["cv_auroc_mean"] for k, v in results.items() if k.startswith("single__")}
    top5 = sorted(individual_cv, key=individual_cv.get, reverse=True)[:5]
    print(f"\n--- Top-5 individual features by CV AUROC: {top5} ---")
    results["combo__top5_by_cv"] = evaluate_feature_set("top5_by_cv", train_df, eval_df, top5)

    out_path = paths.analysis / "feature_ablation.json"
    save_json(results, out_path)
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
