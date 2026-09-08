"""Step 3: train logistic-regression detectors, evaluate, ablate, and compare
against the self-consistency and mean-logprob-only baselines.

Produces (all under outputs/):
  predictions/*.csv    - per-question predictions for each method
  metrics/*.json        - accuracy/AUROC/precision/recall/F1/confusion matrix
  plots/*.png            - confusion matrices, ROC curves, feature distributions,
                            calls-vs-accuracy/AUROC trade-off, subset-size sensitivity

The train/eval question split (70/30) is computed ONCE on the fixed
question set and reused identically for every method, so the proposed
detector, the mean-logprob-only baseline, and the self-consistency
baseline are all evaluated on the exact same held-out questions
(report Section 4.4: "The baseline will use the same model and question
set").
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import CFG, FEATURES_DIR, METRICS_DIR, PREDICTIONS_DIR, set_all_seeds
from src.metrics_utils import (
    compute_metrics,
    plot_calls_vs_metric,
    plot_confusion_matrix,
    plot_feature_distributions,
    plot_roc_curves,
    save_json,
)
from src.train_eval import best_threshold_classifier, fit_and_predict

ALL_METRICS = {}


def get_qid_split(single_pass_df: pd.DataFrame):
    # pandas 3.0 defaults string columns to a PyArrow-backed dtype, whose
    # .values is an ArrowExtensionArray that sklearn's array indexing can't
    # slice with a plain integer index array -- force plain numpy/object arrays.
    qids = single_pass_df["qid"].to_numpy(dtype=object)
    labels = single_pass_df["label"].to_numpy()
    stratify = labels if len(np.unique(labels)) > 1 else None
    train_qids, eval_qids = train_test_split(
        qids,
        train_size=CFG.train_fraction,
        random_state=CFG.seed,
        stratify=stratify,
    )
    return set(train_qids), set(eval_qids)


def evaluate_method(name, train_df, eval_df, feature_cols, calls_per_question, label_col="label"):
    clf, y_eval, y_prob = fit_and_predict(train_df, eval_df, feature_cols, label_col)
    metrics = compute_metrics(y_eval, y_prob)
    metrics["calls_per_question"] = calls_per_question
    metrics["feature_cols"] = feature_cols
    metrics["coefficients"] = dict(zip(feature_cols, clf.coef_[0].tolist()))
    metrics["intercept"] = float(clf.intercept_[0])
    ALL_METRICS[name] = metrics

    preds = eval_df[["qid", "question", label_col]].copy()
    preds["hallucination_probability"] = y_prob
    preds["predicted_label"] = (y_prob >= 0.5).astype(int)
    preds.to_csv(PREDICTIONS_DIR / f"{name}_predictions.csv", index=False)

    plot_confusion_matrix(np.array(metrics["confusion_matrix"]), f"{name}", f"confusion_{name}.png")
    print(f"[{name}] n_eval={metrics['n']} acc={metrics['accuracy']:.3f} "
          f"auroc={metrics['auroc']:.3f} f1={metrics['f1']:.3f} calls/q={calls_per_question}")
    return y_eval, y_prob


def main():
    set_all_seeds()

    sp = pd.read_csv(FEATURES_DIR / "single_pass_features.csv")
    sc = pd.read_csv(FEATURES_DIR / "self_consistency_features.csv")

    train_qids, eval_qids = get_qid_split(sp)
    sp_train, sp_eval = sp[sp.qid.isin(train_qids)].reset_index(drop=True), sp[sp.qid.isin(eval_qids)].reset_index(drop=True)
    sc_train, sc_eval = sc[sc.qid.isin(train_qids)].reset_index(drop=True), sc[sc.qid.isin(eval_qids)].reset_index(drop=True)

    print(f"Split: {len(sp_train)} train / {len(sp_eval)} eval questions "
          f"({CFG.train_fraction:.0%}/{1-CFG.train_fraction:.0%})")

    roc_curves = {}

    # --- Proposed detector: all three features, 1 LLM call/question ---
    y_eval, y_prob = evaluate_method(
        "proposed_all_three", sp_train, sp_eval,
        ["mean_logprob", "min_logprob", "mean_entropy"], calls_per_question=1)
    roc_curves["Proposed (1 call, all 3 features)"] = (y_eval, y_prob)

    # --- Mean-logprob-only baseline (secondary baseline, report 4.4), 1 call/question ---
    y_eval, y_prob = evaluate_method(
        "baseline_mean_logprob_only", sp_train, sp_eval,
        ["mean_logprob"], calls_per_question=1)
    roc_curves["Mean-logprob-only (1 call)"] = (y_eval, y_prob)

    # --- Self-consistency baseline, 5 calls/question ---
    y_eval, y_prob = evaluate_method(
        "baseline_self_consistency", sc_train, sc_eval,
        ["agreement_score"], calls_per_question=5)
    roc_curves["Self-consistency (5 calls)"] = (y_eval, y_prob)

    plot_roc_curves(roc_curves, "roc_comparison.png")
    plot_feature_distributions(sp, ["mean_logprob", "min_logprob", "mean_entropy"], "label", "feature_distributions.png")

    # --- Ablation: mean-only, min-only, entropy-only, all-three ---
    ablation_results = {}
    for feat_name, feats in [
        ("mean_only", ["mean_logprob"]),
        ("min_only", ["min_logprob"]),
        ("entropy_only", ["mean_entropy"]),
        ("all_three", ["mean_logprob", "min_logprob", "mean_entropy"]),
    ]:
        _, y_eval_a, y_prob_a = fit_and_predict(sp_train, sp_eval, feats)
        m = compute_metrics(y_eval_a, y_prob_a)
        ablation_results[feat_name] = m
        print(f"[ablation:{feat_name}] acc={m['accuracy']:.3f} auroc={m['auroc']:.3f} f1={m['f1']:.3f}")
    ALL_METRICS["ablation_feature_sets"] = ablation_results

    # --- Ablation: threshold classifier vs. logistic regression (on mean_logprob) ---
    y_eval_t, y_pred_t, y_prob_t, best_t = best_threshold_classifier(sp_train, sp_eval, "mean_logprob")
    thresh_metrics = compute_metrics(y_eval_t, y_prob_t)
    thresh_metrics["chosen_threshold"] = float(best_t)
    ALL_METRICS["ablation_threshold_vs_logreg"] = {
        "threshold_classifier_mean_logprob": thresh_metrics,
        "logistic_regression_mean_logprob": ablation_results["mean_only"],
    }
    print(f"[ablation:threshold@{best_t:.2f}] acc={thresh_metrics['accuracy']:.3f} "
          f"auroc={thresh_metrics['auroc']:.3f} f1={thresh_metrics['f1']:.3f}")

    # --- Subset-size sensitivity: bootstrap resample the eval set at shrinking sizes ---
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
    ALL_METRICS["subset_size_sensitivity"] = subset_sensitivity
    print("[ablation:subset-size]", json.dumps(subset_sensitivity, indent=2))

    # --- Calls-vs-metric trade-off plot ---
    calls_summary = {
        "Proposed (3 feat)": {"calls_per_question": 1, "accuracy": ALL_METRICS["proposed_all_three"]["accuracy"],
                               "auroc": ALL_METRICS["proposed_all_three"]["auroc"]},
        "Mean-logprob-only": {"calls_per_question": 1, "accuracy": ALL_METRICS["baseline_mean_logprob_only"]["accuracy"],
                               "auroc": ALL_METRICS["baseline_mean_logprob_only"]["auroc"]},
        "Self-consistency": {"calls_per_question": 5, "accuracy": ALL_METRICS["baseline_self_consistency"]["accuracy"],
                              "auroc": ALL_METRICS["baseline_self_consistency"]["auroc"]},
    }
    plot_calls_vs_metric(calls_summary, "accuracy", "calls_vs_accuracy.png")
    plot_calls_vs_metric(calls_summary, "auroc", "calls_vs_auroc.png")
    ALL_METRICS["calls_vs_metric_summary"] = calls_summary

    # --- Efficiency summary ---
    n_questions = len(sp)
    ALL_METRICS["efficiency_summary"] = {
        "n_questions": n_questions,
        "proposed_total_llm_calls": n_questions * 1,
        "self_consistency_total_llm_calls": n_questions * CFG.self_consistency_n_samples,
        "call_reduction_pct": 100 * (1 - 1 / CFG.self_consistency_n_samples),
    }

    save_json(ALL_METRICS, METRICS_DIR / "all_metrics.json")
    print(f"\nAll metrics saved -> {METRICS_DIR / 'all_metrics.json'}")
    print(f"Predictions saved -> {PREDICTIONS_DIR}")
    print(f"Plots saved -> outputs/plots/")


if __name__ == "__main__":
    main()
