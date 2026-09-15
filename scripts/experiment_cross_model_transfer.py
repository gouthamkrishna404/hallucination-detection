"""Cross-model transfer: train a detector on one model's features/labels,
test it on another model's features/labels (same dataset, same question
set, only the generating model differs). Tests whether the detector learns
something general about "what hallucination looks like in logprob space"
or merely a model-specific confidence fingerprint.

Two variants, since raw logprob scales can differ across models (different
vocab sizes, different typical confidence levels) for reasons that have
nothing to do with hallucination:
    raw          -- features used exactly as computed
    standardized -- each feature z-scored using the TRAINING model's own
                    train-split mean/std before fitting, and the TEST
                    model's features z-scored the same way (its own
                    mean/std, computed only from ITS train split -- no
                    leakage) before scoring. This separates "the pattern
                    doesn't transfer" from "the pattern transfers but the
                    scale doesn't."

Usage:
    python scripts/experiment_cross_model_transfer.py --dataset truthfulqa
"""
import argparse
import sys
from itertools import permutations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from sklearn.linear_model import LogisticRegression

from src.config import CFG, set_all_seeds
from src.features_v2 import ORIGINAL_THREE_FEATURES
from src.metrics_utils import compute_metrics, save_json
from src.registry import get_paths
from scripts.experiment_train_evaluate import get_qid_split


def load_train_eval(model, dataset, feature_cols):
    paths = get_paths(model, dataset)
    rich = pd.read_csv(paths.features / "single_pass_features_rich.csv")
    sp = pd.read_csv(paths.features / "single_pass_features.csv")
    train_qids, eval_qids = get_qid_split(sp)
    train_df = rich[rich.qid.isin(train_qids)].reset_index(drop=True)
    eval_df = rich[rich.qid.isin(eval_qids)].reset_index(drop=True)
    return train_df, eval_df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="truthfulqa")
    ap.add_argument("--models", nargs="+", default=["qwen2.5-1.5b-instruct", "llama-3.2-1b-instruct", "smollm2-1.7b-instruct"])
    args = ap.parse_args()

    set_all_seeds()
    feature_cols = ORIGINAL_THREE_FEATURES
    data = {m: load_train_eval(m, args.dataset, feature_cols) for m in args.models}

    results = {}
    print(f"Dataset: {args.dataset}  (features: {feature_cols})\n")
    print(f"{'train model':>25s} -> {'test model':<25s} {'raw AUROC':>10s} {'standardized AUROC':>20s}   (within-model AUROC for reference)")
    for train_model, test_model in permutations(args.models, 2):
        train_df, _ = data[train_model]
        _, test_eval_df = data[test_model]

        X_train = train_df[feature_cols].values
        y_train = train_df["label"].values
        X_test = test_eval_df[feature_cols].values
        y_test = test_eval_df["label"].values

        # raw
        clf = LogisticRegression(random_state=CFG.seed, max_iter=1000)
        clf.fit(X_train, y_train)
        y_prob_raw = clf.predict_proba(X_test)[:, 1]
        m_raw = compute_metrics(y_test, y_prob_raw)

        # standardized: z-score each model's features using ITS OWN train-split mean/std
        train_mean, train_std = X_train.mean(axis=0), X_train.std(axis=0) + 1e-8
        test_train_df, _ = data[test_model]
        test_train_mean = test_train_df[feature_cols].values.mean(axis=0)
        test_train_std = test_train_df[feature_cols].values.std(axis=0) + 1e-8

        X_train_z = (X_train - train_mean) / train_std
        X_test_z = (X_test - test_train_mean) / test_train_std
        clf_z = LogisticRegression(random_state=CFG.seed, max_iter=1000)
        clf_z.fit(X_train_z, y_train)
        y_prob_z = clf_z.predict_proba(X_test_z)[:, 1]
        m_z = compute_metrics(y_test, y_prob_z)

        key = f"{train_model}__to__{test_model}"
        results[key] = {"raw_auroc": m_raw["auroc"], "standardized_auroc": m_z["auroc"], "n_eval": m_raw["n"]}
        print(f"{train_model:>25s} -> {test_model:<25s} {m_raw['auroc']:>10.3f} {m_z['auroc']:>20.3f}")

    # Within-model (train==test model, same as every other result in this project) for reference
    print()
    for m in args.models:
        train_df, eval_df = data[m]
        clf = LogisticRegression(random_state=CFG.seed, max_iter=1000)
        clf.fit(train_df[feature_cols].values, train_df["label"].values)
        y_prob = clf.predict_proba(eval_df[feature_cols].values)[:, 1]
        within = compute_metrics(eval_df["label"].values, y_prob)
        results[f"{m}__within_model_reference"] = {"auroc": within["auroc"]}
        print(f"  within-model reference [{m}]: AUROC={within['auroc']:.3f}")

    out_path = Path("comparison") / f"{args.dataset}_cross_model_transfer.json"
    save_json(results, out_path)
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
