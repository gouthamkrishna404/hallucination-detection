"""Metric computation and plotting helpers shared by all evaluation scripts."""
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    auc,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from .config import PLOTS_DIR


def _resolve(path_or_name):
    """Accept either a bare filename (saved under the default DA1 PLOTS_DIR,
    for backward compatibility with scripts/01-03) or a full Path (used by
    the generalized multi-model/multi-dataset scripts to target
    results/<model>/<dataset>/plots/)."""
    from pathlib import Path

    p = Path(path_or_name)
    return p if p.is_absolute() else PLOTS_DIR / p


def compute_metrics(y_true, y_prob, threshold: float = 0.5) -> dict:
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    y_pred = (y_prob >= threshold).astype(int)

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    if len(np.unique(y_true)) > 1:
        metrics["auroc"] = float(roc_auc_score(y_true, y_prob))
    else:
        metrics["auroc"] = float("nan")

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    metrics["confusion_matrix"] = cm.tolist()  # [[TN, FP], [FN, TP]]
    metrics["n"] = int(len(y_true))
    metrics["n_positive"] = int(y_true.sum())
    return metrics


def plot_confusion_matrix(cm, title: str, filename: str):
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Grounded", "Hallucinated"])
    ax.set_yticklabels(["Grounded", "Hallucinated"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i][j]), ha="center", va="center",
                     color="white" if cm[i][j] > np.max(cm) / 2 else "black", fontsize=14)
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(_resolve(filename), dpi=150)
    plt.close(fig)


def plot_roc_curves(curves: dict[str, tuple], filename: str):
    """curves: name -> (y_true, y_prob)"""
    fig, ax = plt.subplots(figsize=(6, 6))
    for name, (y_true, y_prob) in curves.items():
        if len(np.unique(y_true)) < 2:
            continue
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc_val = auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f"{name} (AUROC={auc_val:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Chance")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves: Detection Methods")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(_resolve(filename), dpi=150)
    plt.close(fig)


def plot_feature_distributions(df, feature_cols, label_col, filename):
    fig, axes = plt.subplots(1, len(feature_cols), figsize=(5 * len(feature_cols), 4))
    if len(feature_cols) == 1:
        axes = [axes]
    for ax, col in zip(axes, feature_cols):
        for label, color, name in [(0, "tab:blue", "Grounded"), (1, "tab:red", "Hallucinated")]:
            vals = df[df[label_col] == label][col]
            ax.hist(vals, bins=20, alpha=0.6, color=color, label=name, density=True)
        ax.set_title(col)
        ax.set_xlabel(col)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(_resolve(filename), dpi=150)
    plt.close(fig)


def plot_calls_vs_metric(results: dict, metric_name: str, filename: str):
    """results: method_name -> {"calls_per_question": int, metric_name: value}"""
    fig, ax = plt.subplots(figsize=(6, 5))
    for name, r in results.items():
        ax.scatter(r["calls_per_question"], r[metric_name], s=120, label=name)
        ax.annotate(name, (r["calls_per_question"], r[metric_name]),
                     textcoords="offset points", xytext=(6, 6), fontsize=9)
    ax.set_xlabel("LLM calls per question")
    ax.set_ylabel(metric_name.upper())
    ax.set_title(f"{metric_name.upper()} vs. LLM Call Budget")
    ax.set_xlim(0, 6)
    fig.tight_layout()
    fig.savefig(_resolve(filename), dpi=150)
    plt.close(fig)


def plot_pr_curves(curves: dict[str, tuple], filename: str):
    """curves: name -> (y_true, y_prob). Precision-recall is the more
    informative curve than ROC under class imbalance (report Section 4.4
    doesn't require it, but AUROC alone can look better than it deserves
    when the positive rate is far from 50%, so it's included as a check)."""
    fig, ax = plt.subplots(figsize=(6, 6))
    for name, (y_true, y_prob) in curves.items():
        if len(np.unique(y_true)) < 2:
            continue
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        ap = average_precision_score(y_true, y_prob)
        ax.plot(recall, precision, label=f"{name} (AP={ap:.3f})")
    base_rate = None
    for y_true, _ in curves.values():
        base_rate = np.mean(y_true)
        break
    if base_rate is not None:
        ax.axhline(base_rate, linestyle="--", color="gray", label=f"Chance (base rate={base_rate:.2f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curves: Detection Methods")
    ax.legend(loc="lower left", fontsize=8)
    fig.tight_layout()
    fig.savefig(_resolve(filename), dpi=150)
    plt.close(fig)


def compute_calibration(y_true, y_prob, n_bins: int = 10) -> dict:
    """Expected Calibration Error + per-bin reliability data + Brier score."""
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_ids = np.clip(np.digitize(y_prob, bin_edges) - 1, 0, n_bins - 1)

    bins = []
    ece = 0.0
    n = len(y_true)
    for b in range(n_bins):
        mask = bin_ids == b
        count = int(mask.sum())
        if count == 0:
            bins.append({"bin": b, "count": 0, "mean_predicted": None, "empirical_rate": None})
            continue
        mean_pred = float(y_prob[mask].mean())
        emp_rate = float(y_true[mask].mean())
        bins.append({"bin": b, "count": count, "mean_predicted": mean_pred, "empirical_rate": emp_rate})
        ece += (count / n) * abs(mean_pred - emp_rate)

    return {
        "n_bins": n_bins,
        "bins": bins,
        "ece": float(ece),
        "brier_score": float(brier_score_loss(y_true, y_prob)),
    }


def plot_calibration_curve(y_true, y_prob, filename: str, title: str = "Calibration"):
    cal = compute_calibration(y_true, y_prob, n_bins=10)
    xs = [b["mean_predicted"] for b in cal["bins"] if b["count"] > 0]
    ys = [b["empirical_rate"] for b in cal["bins"] if b["count"] > 0]

    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect calibration")
    ax.plot(xs, ys, marker="o", color="tab:red", label=f"{title} (ECE={cal['ece']:.3f}, Brier={cal['brier_score']:.3f})")
    ax.set_xlabel("Mean predicted hallucination probability")
    ax.set_ylabel("Empirical hallucination rate")
    ax.set_title(f"Calibration: {title}")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(_resolve(filename), dpi=150)
    plt.close(fig)
    return cal


def plot_threshold_sensitivity(y_true, y_prob, filename: str, title: str = ""):
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    thresholds = np.linspace(0.01, 0.99, 99)
    accs, precs, recs, f1s = [], [], [], []
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        accs.append(accuracy_score(y_true, y_pred))
        precs.append(precision_score(y_true, y_pred, zero_division=0))
        recs.append(recall_score(y_true, y_pred, zero_division=0))
        f1s.append(f1_score(y_true, y_pred, zero_division=0))

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(thresholds, accs, label="Accuracy")
    ax.plot(thresholds, precs, label="Precision")
    ax.plot(thresholds, recs, label="Recall")
    ax.plot(thresholds, f1s, label="F1")
    ax.axvline(0.5, linestyle=":", color="gray", label="Default threshold (0.5)")
    ax.set_xlabel("Decision threshold on hallucination probability")
    ax.set_ylabel("Metric value")
    ax.set_title(f"Threshold Sensitivity{': ' + title if title else ''}")
    ax.legend(loc="lower left", fontsize=8)
    fig.tight_layout()
    fig.savefig(_resolve(filename), dpi=150)
    plt.close(fig)


def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
