"""Generalized training/evaluation for any (model, dataset) pair: trains the
proposed 3-feature detector, the mean-logprob-only baseline, and the
self-consistency baseline, runs the same ablations and subset-size
sensitivity check as the DA1 pipeline, and additionally persists the fitted
classifiers (joblib) so the interactive demo can load them without
retraining.

Usage:
    python scripts/experiment_train_evaluate.py --model llama-3.2-1b-instruct --dataset truthfulqa
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import CFG, set_all_seeds
from src.metrics_utils import (
    compute_metrics,
    plot_calls_vs_metric,
    plot_confusion_matrix,
    plot_feature_distributions,
    plot_roc_curves,
    save_json,
)
from src.registry import get_paths
from src.run_logger import log_event
from src.train_eval import best_threshold_classifier, fit_and_predict


def get_qid_split(single_pass_df: pd.DataFrame):
    qids = single_pass_df["qid"].to_numpy(dtype=object)
    labels = single_pass_df["label"].to_numpy()
    stratify = labels if len(np.unique(labels)) > 1 else None
    train_qids, eval_qids = train_test_split(
        qids, train_size=CFG.train_fraction, random_state=CFG.seed, stratify=stratify,
    )
    return set(train_qids), set(eval_qids)


def evaluate_method(all_metrics, paths, name, train_df, eval_df, feature_cols, calls_per_question, label_col="label"):
    clf, y_eval, y_prob = fit_and_predict(train_df, eval_df, feature_cols, label_col)
    metrics = compute_metrics(y_eval, y_prob)
    metrics["calls_per_question"] = calls_per_question
    metrics["feature_cols"] = feature_cols
    metrics["coefficients"] = dict(zip(feature_cols, clf.coef_[0].tolist()))
    metrics["intercept"] = float(clf.intercept_[0])
    all_metrics[name] = metrics

    preds = eval_df[["qid", "question", label_col]].copy()
    preds["hallucination_probability"] = y_prob
    preds["predicted_label"] = (y_prob >= 0.5).astype(int)
    preds.to_csv(paths.predictions / f"{name}_predictions.csv", index=False)

    joblib.dump(clf, paths.models / f"{name}.joblib")

    plot_confusion_matrix(np.array(metrics["confusion_matrix"]), f"{name}", paths.plots / f"confusion_{name}.png")
    print(f"[{name}] n_eval={metrics['n']} acc={metrics['accuracy']:.3f} "
          f"auroc={metrics['auroc']:.3f} f1={metrics['f1']:.3f} calls/q={calls_per_question}")
    return y_eval, y_prob


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--dataset", default="truthfulqa")
    args = ap.parse_args()

    set_all_seeds()
    paths = get_paths(args.model, args.dataset)
    all_metrics = {}

    sp = pd.read_csv(paths.features / "single_pass_features.csv")
    sc = pd.read_csv(paths.features / "self_consistency_features.csv")

    train_qids, eval_qids = get_qid_split(sp)
    sp_train, sp_eval = sp[sp.qid.isin(train_qids)].reset_index(drop=True), sp[sp.qid.isin(eval_qids)].reset_index(drop=True)
    sc_train, sc_eval = sc[sc.qid.isin(train_qids)].reset_index(drop=True), sc[sc.qid.isin(eval_qids)].reset_index(drop=True)

    print(f"[{args.model}/{args.dataset}] Split: {len(sp_train)} train / {len(sp_eval)} eval "
          f"({CFG.train_fraction:.0%}/{1-CFG.train_fraction:.0%})")

    roc_curves = {}

    y_eval, y_prob = evaluate_method(all_metrics, paths, "proposed_all_three", sp_train, sp_eval,
                                      ["mean_logprob", "min_logprob", "mean_entropy"], calls_per_question=1)
    roc_curves["Proposed (1 call, all 3 features)"] = (y_eval, y_prob)

    y_eval, y_prob = evaluate_method(all_metrics, paths, "baseline_mean_logprob_only", sp_train, sp_eval,
                                      ["mean_logprob"], calls_per_question=1)
    roc_curves["Mean-logprob-only (1 call)"] = (y_eval, y_prob)

    y_eval, y_prob = evaluate_method(all_metrics, paths, "baseline_self_consistency", sc_train, sc_eval,
                                      ["agreement_score"], calls_per_question=5)
    roc_curves["Self-consistency (5 calls)"] = (y_eval, y_prob)

    plot_roc_curves(roc_curves, paths.plots / "roc_comparison.png")
    plot_feature_distributions(sp, ["mean_logprob", "min_logprob", "mean_entropy"], "label", paths.plots / "feature_distributions.png")

    ablation_results = {}
    for feat_name, feats in [
        ("mean_only", ["mean_logprob"]), ("min_only", ["min_logprob"]),
        ("entropy_only", ["mean_entropy"]), ("all_three", ["mean_logprob", "min_logprob", "mean_entropy"]),
    ]:
        _, y_eval_a, y_prob_a = fit_and_predict(sp_train, sp_eval, feats)
        m = compute_metrics(y_eval_a, y_prob_a)
        ablation_results[feat_name] = m
        print(f"[ablation:{feat_name}] acc={m['accuracy']:.3f} auroc={m['auroc']:.3f} f1={m['f1']:.3f}")
    all_metrics["ablation_feature_sets"] = ablation_results

    y_eval_t, y_pred_t, y_prob_t, best_t = best_threshold_classifier(sp_train, sp_eval, "mean_logprob")
    thresh_metrics = compute_metrics(y_eval_t, y_prob_t)
    thresh_metrics["chosen_threshold"] = float(best_t)
    all_metrics["ablation_threshold_vs_logreg"] = {
        "threshold_classifier_mean_logprob": thresh_metrics,
        "logistic_regression_mean_logprob": ablation_results["mean_only"],
    }
    print(f"[ablation:threshold@{best_t:.2f}] acc={thresh_metrics['accuracy']:.3f} "
          f"auroc={thresh_metrics['auroc']:.3f} f1={thresh_metrics['f1']:.3f}")

    subset_sensitivity = {}
    rng = np.random.RandomState(CFG.seed)
    full_n = len(sp_eval)
    for frac in [1.0, 0.75, 0.5, 0.25]:
        n = max(10, int(full_n * frac))
        accs, aurocs = [], []
        for rep in range(50):
            idx = rng.choice(full_n, size=n, replace=True)
            sub = sp_eval.iloc[idx]
            _, y_e, y_p = fit_and_predict(sp_train, sub, ["mean_logprob", "min_logprob", "mean_entropy"])
            m = compute_metrics(y_e, y_p)
            accs.append(m["accuracy"])
            if not np.isnan(m["auroc"]):
                aurocs.append(m["auroc"])
        subset_sensitivity[f"n={n}"] = {
            "accuracy_mean": float(np.mean(accs)), "accuracy_std": float(np.std(accs)),
            "auroc_mean": float(np.mean(aurocs)) if aurocs else None,
            "auroc_std": float(np.std(aurocs)) if aurocs else None,
        }
    all_metrics["subset_size_sensitivity"] = subset_sensitivity

    calls_summary = {
        "Proposed (3 feat)": {"calls_per_question": 1, "accuracy": all_metrics["proposed_all_three"]["accuracy"],
                               "auroc": all_metrics["proposed_all_three"]["auroc"]},
        "Mean-logprob-only": {"calls_per_question": 1, "accuracy": all_metrics["baseline_mean_logprob_only"]["accuracy"],
                               "auroc": all_metrics["baseline_mean_logprob_only"]["auroc"]},
        "Self-consistency": {"calls_per_question": 5, "accuracy": all_metrics["baseline_self_consistency"]["accuracy"],
                              "auroc": all_metrics["baseline_self_consistency"]["auroc"]},
    }
    plot_calls_vs_metric(calls_summary, "accuracy", paths.plots / "calls_vs_accuracy.png")
    plot_calls_vs_metric(calls_summary, "auroc", paths.plots / "calls_vs_auroc.png")
    all_metrics["calls_vs_metric_summary"] = calls_summary

    n_questions = len(sp)
    all_metrics["efficiency_summary"] = {
        "n_questions": n_questions,
        "proposed_total_llm_calls": n_questions * 1,
        "self_consistency_total_llm_calls": n_questions * CFG.self_consistency_n_samples,
        "call_reduction_pct": 100 * (1 - 1 / CFG.self_consistency_n_samples),
    }
    all_metrics["run_config"] = {
        "model": args.model, "dataset": args.dataset, "seed": CFG.seed,
        "subset_size": CFG.subset_size, "train_fraction": CFG.train_fraction,
    }

    save_json(all_metrics, paths.metrics / "all_metrics.json")
    print(f"\n[{args.model}/{args.dataset}] All metrics saved -> {paths.metrics / 'all_metrics.json'}")
    log_event("experiment_train_evaluate", args.model, args.dataset, "completed",
              proposed_auroc=all_metrics["proposed_all_three"]["auroc"],
              proposed_accuracy=all_metrics["proposed_all_three"]["accuracy"],
              mean_logprob_baseline_auroc=all_metrics["baseline_mean_logprob_only"]["auroc"],
              self_consistency_auroc=all_metrics["baseline_self_consistency"]["auroc"])


if __name__ == "__main__":
    main()
