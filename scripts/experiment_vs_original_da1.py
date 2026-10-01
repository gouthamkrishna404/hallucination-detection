"""Direct, literal comparison against the original DA1 detector: same
model (Qwen2.5-1.5B-Instruct), same 200-question TruthfulQA subset, same
70/30 seeded split (140 train / 60 eval -- the exact qids used to produce
the frozen 0.482 AUROC in outputs/), same classifier family. The only
thing that changes is the feature set and (for the cascade) the addition
of per-fold-style routing collapsed to a single train/test split.

This is deliberately NOT the higher-power out-of-fold evaluation used
elsewhere in this investigation -- it trades statistical power for being
the most literal, unambiguous "same test questions as the original DA1
result" comparison possible, exactly as requested.

Usage:
    python scripts/experiment_vs_original_da1.py --model qwen2.5-1.5b-instruct --dataset truthfulqa
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from src.config import CFG, set_all_seeds
from src.features_v2 import ALL_RICH_FEATURE_NAMES, ORIGINAL_THREE_FEATURES
from src.metrics_utils import save_json
from src.registry import get_paths
from scripts.experiment_best_candidate import bootstrap_auroc_ci
from scripts.experiment_train_evaluate import get_qid_split


def fixed_split_cascade(train_df, eval_df, feature_cols):
    """Same cascade design as experiment_cascade_detector.py's cascade_oof,
    collapsed to a single train (140) / eval (60) split -- threshold and
    both branch classifiers fit on the 140 TRAIN questions only."""
    X_train, y_train = train_df[feature_cols].values, train_df["label"].values
    X_eval = eval_df[feature_cols].values
    mean_lp_train = train_df["mean_logprob"].values
    mean_lp_eval = eval_df["mean_logprob"].values

    threshold = np.median(mean_lp_train)
    train_confident = mean_lp_train >= threshold
    eval_confident = mean_lp_eval >= threshold

    prob = np.zeros(len(eval_df))
    for branch_name, train_mask, eval_mask in [("confident", train_confident, eval_confident),
                                                  ("uncertain", ~train_confident, ~eval_confident)]:
        if eval_mask.sum() == 0:
            continue
        if len(np.unique(y_train[train_mask])) < 2 or train_mask.sum() < 10:
            clf = LogisticRegression(random_state=CFG.seed, max_iter=1000)
            clf.fit(X_train, y_train)
        else:
            clf = LogisticRegression(random_state=CFG.seed, max_iter=1000)
            clf.fit(X_train[train_mask], y_train[train_mask])
        prob[eval_mask] = clf.predict_proba(X_eval[eval_mask])[:, 1]
    return prob


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--dataset", default="truthfulqa")
    ap.add_argument("--features", default="rich", choices=["rich", "clean"])
    args = ap.parse_args()

    set_all_seeds()
    paths = get_paths(args.model, args.dataset)
    rich = pd.read_csv(paths.features / f"single_pass_features_{args.features}.csv")
    sp = pd.read_csv(paths.features / "single_pass_features.csv")
    train_qids, eval_qids = get_qid_split(sp)  # the EXACT split used to produce the original result

    train_df = rich[rich.qid.isin(train_qids)].reset_index(drop=True)
    eval_df = rich[rich.qid.isin(eval_qids)].reset_index(drop=True)
    print(f"[{args.model}/{args.dataset}]  same fixed split as original DA1 result: "
          f"{len(train_df)} train / {len(eval_df)} eval questions\n")

    results = {}

    # (a) original 3-feature detector, literally the same computation as the DA1/matrix result
    clf = LogisticRegression(random_state=CFG.seed, max_iter=1000)
    clf.fit(train_df[ORIGINAL_THREE_FEATURES].values, train_df["label"].values)
    prob_orig = clf.predict_proba(eval_df[ORIGINAL_THREE_FEATURES].values)[:, 1]
    ci_orig = bootstrap_auroc_ci(eval_df["label"].values, prob_orig)
    results["original_3feat_same_split"] = ci_orig

    # (b) flat 17-feature detector, same split
    clf17 = LogisticRegression(random_state=CFG.seed, max_iter=1000)
    clf17.fit(train_df[ALL_RICH_FEATURE_NAMES].values, train_df["label"].values)
    prob_flat = clf17.predict_proba(eval_df[ALL_RICH_FEATURE_NAMES].values)[:, 1]
    ci_flat = bootstrap_auroc_ci(eval_df["label"].values, prob_flat)
    results["all_17_flat_same_split"] = ci_flat

    # (c) cascade, same split (threshold + branch classifiers fit on the 140 train questions only)
    prob_cascade = fixed_split_cascade(train_df, eval_df, ALL_RICH_FEATURE_NAMES)
    ci_cascade = bootstrap_auroc_ci(eval_df["label"].values, prob_cascade)
    results["cascade_17feat_same_split"] = ci_cascade

    for name, ci in results.items():
        flag = "EXCLUDES chance" if ci["excludes_chance"] else "does not exclude chance"
        print(f"  {name:32s} n_eval={ci is not None and len(eval_df)}  AUROC={ci['point_estimate']:.3f}  "
              f"95% CI=[{ci['ci_lower_2.5pct']:.3f}, {ci['ci_upper_97.5pct']:.3f}]  ({flag})")

    out_path = paths.analysis / f"vs_original_da1_same_split_{args.features}.json"
    save_json(results, out_path)
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
