"""Cross-model comparison for a fixed dataset: builds a side-by-side table
and grouped bar chart of every metric across all models that have been run
on that dataset, so the "is the near-chance result model-specific?"
question has a direct, visual answer.

Usage:
    python scripts/compare_models.py --dataset truthfulqa --models qwen2.5-1.5b-instruct llama-3.2-1b-instruct
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.registry import COMPARISON_DIR, get_paths, resolve_model


def load_metrics(model_key: str, dataset_key: str) -> dict:
    paths = get_paths(model_key, dataset_key)
    with open(paths.metrics / "all_metrics.json", encoding="utf-8") as f:
        return json.load(f)


def build_table(models: list[str], dataset_key: str) -> pd.DataFrame:
    rows = []
    for model_key in models:
        m = load_metrics(model_key, dataset_key)
        display = resolve_model(model_key).get("display_name", model_key)
        ablation = m["ablation_feature_sets"]
        row = {
            "model": display,
            "model_key": model_key,
            "mean_logprob_only_auroc": ablation["mean_only"]["auroc"],
            "min_logprob_only_auroc": ablation["min_only"]["auroc"],
            "entropy_only_auroc": ablation["entropy_only"]["auroc"],
            "proposed_3feat_auroc": m["proposed_all_three"]["auroc"],
            "proposed_3feat_accuracy": m["proposed_all_three"]["accuracy"],
            "proposed_3feat_f1": m["proposed_all_three"]["f1"],
            "mean_logprob_baseline_auroc": m["baseline_mean_logprob_only"]["auroc"],
            "self_consistency_auroc": m["baseline_self_consistency"]["auroc"],
            "self_consistency_accuracy": m["baseline_self_consistency"]["accuracy"],
            "self_consistency_f1": m["baseline_self_consistency"]["f1"],
            "semantic_self_consistency_auroc": m.get("semantic_self_consistency", {}).get("auroc"),
            "calls_proposed": m["efficiency_summary"]["proposed_total_llm_calls"],
            "calls_self_consistency": m["efficiency_summary"]["self_consistency_total_llm_calls"],
        }
        rows.append(row)
    return pd.DataFrame(rows)


def plot_comparison(df: pd.DataFrame, dataset_key: str, out_path: Path):
    metrics = ["mean_logprob_only_auroc", "min_logprob_only_auroc", "entropy_only_auroc",
               "proposed_3feat_auroc", "self_consistency_auroc", "semantic_self_consistency_auroc"]
    labels = ["mean-logprob\nonly", "min-logprob\nonly", "entropy\nonly",
              "proposed\n(3-feat)", "lexical self-\nconsistency (5c)", "semantic self-\nconsistency (5c)"]

    n_models = len(df)
    n_metrics = len(metrics)
    x = np.arange(n_metrics)
    width = 0.8 / n_models

    fig, ax = plt.subplots(figsize=(10, 6))
    for i, (_, row) in enumerate(df.iterrows()):
        vals = [row[m] if row[m] is not None else np.nan for m in metrics]
        ax.bar(x + i * width, vals, width, label=row["model"])

    ax.axhline(0.5, linestyle="--", color="gray", label="Chance (AUROC=0.5)")
    ax.set_xticks(x + width * (n_models - 1) / 2)
    ax.set_xticklabels(labels)
    ax.set_ylabel("AUROC")
    ax.set_title(f"Model Comparison: AUROC by Method ({dataset_key})")
    ax.legend(fontsize=9)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="truthfulqa")
    ap.add_argument("--models", nargs="+", required=True)
    args = ap.parse_args()

    df = build_table(args.models, args.dataset)
    csv_path = COMPARISON_DIR / f"{args.dataset}_model_comparison.csv"
    json_path = COMPARISON_DIR / f"{args.dataset}_model_comparison.json"
    plot_path = COMPARISON_DIR / f"{args.dataset}_model_comparison.png"

    df.to_csv(csv_path, index=False)
    df.to_json(json_path, orient="records", indent=2)
    plot_comparison(df, args.dataset, plot_path)

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 20)
    print(df.to_string(index=False))
    print(f"\nSaved -> {csv_path}\nSaved -> {json_path}\nSaved -> {plot_path}")


if __name__ == "__main__":
    main()
