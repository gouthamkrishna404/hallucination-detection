"""The single most important test in this investigation: is there ANY
feature combination that beats the original 3-feature detector on
TruthfulQA, robustly, across multiple models, with a confidence interval
that actually excludes chance?

Candidate feature set (chosen from scripts/experiment_feature_ablation.py's
cross-model consistency table, NOT the single best-looking number for any
one model -- see README "Feature engineering" for the selection
reasoning): median_logprob, logprob_p25, logprob_trend_slope,
logprob_delta_second_minus_first. Deliberately excludes answer_length
(shown to be a labeling-proxy artifact, not a real signal -- see README
"Label audit") and margin/entropy features (inconsistent or at chance
across models in the ablation).

For both the original 3-feature detector and this candidate, on every
model x dataset combination, reports:
    - held-out AUROC (point estimate)
    - a bootstrap 95% CI on that AUROC (2000 resamples of the eval set)
    - whether the CI excludes 0.5 (the only way to call a result "real"
      rather than noise, given n=60 eval questions)

Usage:
    python scripts/experiment_best_candidate.py --dataset truthfulqa
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
from src.train_eval import fit_and_predict
from scripts.experiment_train_evaluate import get_qid_split

CANDIDATE_FEATURES = ["median_logprob", "logprob_p25", "logprob_trend_slope", "logprob_delta_second_minus_first"]


def bootstrap_auroc_ci(y_true, y_prob, n_boot=2000, seed=CFG.seed):
    rng = np.random.RandomState(seed)
    y_true, y_prob = np.asarray(y_true), np.asarray(y_prob)
    n = len(y_true)
    boot_aurocs = []
    for _ in range(n_boot):
        idx = rng.randint(0, n, n)
        yt, yp = y_true[idx], y_prob[idx]
        if len(np.unique(yt)) < 2:
            continue
        boot_aurocs.append(roc_auc_score(yt, yp))
    boot_aurocs = np.array(boot_aurocs)
    return {
        "point_estimate": float(roc_auc_score(y_true, y_prob)),
        "ci_lower_2.5pct": float(np.percentile(boot_aurocs, 2.5)),
        "ci_upper_97.5pct": float(np.percentile(boot_aurocs, 97.5)),
        "n_bootstrap_reps_used": len(boot_aurocs),
        "excludes_chance": bool(np.percentile(boot_aurocs, 2.5) > 0.5),
    }


def out_of_fold_predictions(df: pd.DataFrame, feature_cols: list[str], n_splits: int = 10, seed: int = CFG.seed):
    """10-fold stratified out-of-fold predictions across the FULL dataset
    (not just the 60-question held-out split): every question is scored by
    a model that never saw it during training, so this is still a valid,
    leakage-free evaluation -- but with n=200 instead of n=60, it has far
    more statistical power to detect a real (if modest) effect, which is
    exactly what a single fixed 60-question split lacks."""
    X = df[feature_cols].values
    y = df["label"].values
    oof_prob = np.zeros(len(df))
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for train_idx, test_idx in skf.split(X, y):
        clf = LogisticRegression(random_state=seed, max_iter=1000)
        clf.fit(X[train_idx], y[train_idx])
        oof_prob[test_idx] = clf.predict_proba(X[test_idx])[:, 1]
    return y, oof_prob


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="truthfulqa")
    ap.add_argument("--models", nargs="+", default=["qwen2.5-1.5b-instruct", "llama-3.2-1b-instruct", "smollm2-1.7b-instruct"])
    args = ap.parse_args()

    set_all_seeds()
    results = {}
    print(f"Dataset: {args.dataset}\n")
    for model in args.models:
        paths = get_paths(model, args.dataset)
        rich = pd.read_csv(paths.features / "single_pass_features_rich.csv")
        sp = pd.read_csv(paths.features / "single_pass_features.csv")
        train_qids, eval_qids = get_qid_split(sp)
        train_df = rich[rich.qid.isin(train_qids)].reset_index(drop=True)
        eval_df = rich[rich.qid.isin(eval_qids)].reset_index(drop=True)

        model_results = {}
        for name, feats in [("original_3feat", ORIGINAL_THREE_FEATURES), ("candidate_4feat", CANDIDATE_FEATURES)]:
            # (a) the fixed 70/30 split used by every other result in this project (n=60 eval)
            _, y_eval, y_prob = fit_and_predict(train_df, eval_df, feats)
            ci_fixed_split = bootstrap_auroc_ci(y_eval, y_prob)

            # (b) 10-fold out-of-fold CV across the FULL 200 questions (n=200, more power)
            y_oof, prob_oof = out_of_fold_predictions(rich, feats)
            ci_oof = bootstrap_auroc_ci(y_oof, prob_oof)

            model_results[name] = {
                "features": feats,
                "fixed_split_n60": ci_fixed_split,
                "out_of_fold_n200": ci_oof,
            }
            excl_fixed = "EXCLUDES chance" if ci_fixed_split["excludes_chance"] else "does NOT exclude chance"
            excl_oof = "EXCLUDES chance" if ci_oof["excludes_chance"] else "does NOT exclude chance"
            print(f"  [{model:25s}] {name:18s}")
            print(f"      fixed split (n=60):  AUROC={ci_fixed_split['point_estimate']:.3f}  "
                  f"95% CI=[{ci_fixed_split['ci_lower_2.5pct']:.3f}, {ci_fixed_split['ci_upper_97.5pct']:.3f}]  ({excl_fixed})")
            print(f"      out-of-fold (n=200): AUROC={ci_oof['point_estimate']:.3f}  "
                  f"95% CI=[{ci_oof['ci_lower_2.5pct']:.3f}, {ci_oof['ci_upper_97.5pct']:.3f}]  ({excl_oof})")
        results[model] = model_results
        print()

    out_path = Path("comparison") / f"{args.dataset}_best_candidate_bootstrap.json"
    save_json(results, out_path)
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()
