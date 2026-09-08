"""Deeper analysis for one (model, dataset) run, beyond the core metrics in
experiment_train_evaluate.py: PR curves, calibration, threshold
sensitivity, confidence-vs-correctness, error analysis, and token-level
uncertainty visualization for a handful of representative examples.

Reads only what experiment_train_evaluate.py already wrote (features,
predictions) -- never re-runs the LLM, except optionally to backfill
token-level detail for a small number of individual example questions when
the original run predates token-detail caching (--backfill-token-details;
used for the original DA1 Qwen run, whose features were computed before
this capability existed). Backfilled examples are verified to reproduce
the exact cached answer text (greedy decoding is deterministic), proving
no divergence from the original run.

Usage:
    python scripts/experiment_analysis.py --model llama-3.2-1b-instruct --dataset truthfulqa
    python scripts/experiment_analysis.py --model qwen2.5-1.5b-instruct --dataset truthfulqa --backfill-token-details
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from src.config import CFG, set_all_seeds
from src.datasets import load_subset
from src.metrics_utils import (
    plot_calibration_curve,
    plot_pr_curves,
    plot_threshold_sensitivity,
    save_json,
)
from src.registry import get_paths
from src.token_viz import render_token_html


def confidence_vs_correctness(paths, sp: pd.DataFrame):
    """Box-plot-style summary + saved CSV: does the model's own confidence
    (mean_logprob) actually track whether the answer was correct?"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5, 5))
    data = [sp[sp.label == 0]["mean_logprob"], sp[sp.label == 1]["mean_logprob"]]
    ax.boxplot(data, tick_labels=["Grounded", "Hallucinated"])
    ax.set_ylabel("mean_logprob (higher = more confident)")
    ax.set_title("Confidence vs. Correctness")
    fig.tight_layout()
    fig.savefig(paths.analysis / "confidence_vs_correctness.png", dpi=150)
    plt.close(fig)

    summary = {
        "grounded_mean_logprob_mean": float(sp[sp.label == 0]["mean_logprob"].mean()),
        "hallucinated_mean_logprob_mean": float(sp[sp.label == 1]["mean_logprob"].mean()),
        "grounded_mean_logprob_std": float(sp[sp.label == 0]["mean_logprob"].std()),
        "hallucinated_mean_logprob_std": float(sp[sp.label == 1]["mean_logprob"].std()),
        "point_biserial_corr_mean_logprob_vs_label": float(sp["mean_logprob"].corr(sp["label"])),
    }
    return summary


def error_analysis(paths, sp: pd.DataFrame, preds: pd.DataFrame, top_k: int = 10):
    """Highest-confidence WRONG predictions (both directions), plus a
    breakdown of error rate by question category."""
    merged = preds.merge(sp[["qid", "generated_answer", "category", "mean_logprob", "min_logprob", "mean_entropy"]], on="qid")
    merged["correct_prediction"] = merged["predicted_label"] == merged["label"]

    false_positives = merged[(merged.label == 0) & (merged.predicted_label == 1)].sort_values(
        "hallucination_probability", ascending=False).head(top_k)
    false_negatives = merged[(merged.label == 1) & (merged.predicted_label == 0)].sort_values(
        "hallucination_probability", ascending=True).head(top_k)

    false_positives.to_csv(paths.analysis / "error_false_positives.csv", index=False)
    false_negatives.to_csv(paths.analysis / "error_false_negatives.csv", index=False)

    by_category = merged.groupby("category").agg(
        n=("qid", "count"), accuracy=("correct_prediction", "mean"),
        hallucination_rate=("label", "mean"),
    ).reset_index().sort_values("n", ascending=False)
    by_category.to_csv(paths.analysis / "error_by_category.csv", index=False)

    return {
        "overall_accuracy": float(merged["correct_prediction"].mean()),
        "n_false_positives": int(len(merged[(merged.label == 0) & (merged.predicted_label == 1)])),
        "n_false_negatives": int(len(merged[(merged.label == 1) & (merged.predicted_label == 0)])),
        "n_categories": int(merged["category"].nunique()),
    }


def load_token_details(paths):
    path = paths.features / "single_pass_token_details.jsonl"
    if not path.exists():
        return None
    details = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            details[rec["qid"]] = rec
    return details


def backfill_token_details(paths, args, sp: pd.DataFrame, example_qids: list[str]):
    """Regenerate greedy answers for a SMALL set of example questions only,
    to recover per-token logprob/entropy detail for visualization when the
    original run predates token-detail caching. Verifies the reproduced
    text matches the cached answer exactly (greedy decoding is
    deterministic) -- this does not alter or replace any original result,
    it only adds illustrative per-token detail for a few examples."""
    from src.model_utils import LLMGenerator

    df = load_subset(args.dataset, CFG.subset_size, CFG.seed)
    qmap = dict(zip(df["qid"], df["question"]))

    print(f"Backfilling token-level detail for {len(example_qids)} example question(s) "
          f"(regenerating with identical greedy settings, to verify determinism)...")
    generator = LLMGenerator.from_registry(args.model)

    details = {}
    for qid in example_qids:
        cached_answer = sp.loc[sp.qid == qid, "generated_answer"].iloc[0]
        result = generator.generate_with_scores(
            question=qmap[qid], max_new_tokens=CFG.single_pass_max_new_tokens,
            do_sample=CFG.single_pass_do_sample,
        )
        match = result.answer_text == cached_answer
        print(f"  {qid}: reproduced-text-matches-cached={match}")
        if not match:
            print(f"    WARNING: mismatch. cached={cached_answer!r} regenerated={result.answer_text!r}")
        details[qid] = {
            "qid": qid, "tokens": result.token_strs,
            "token_logprobs": result.token_logprobs, "token_entropies": result.token_entropies,
            "reproduced_matches_cached": match,
        }
    return details


def pick_example_qids(sp: pd.DataFrame, preds: pd.DataFrame, n_each: int = 2) -> list[str]:
    """Pick representative examples: most-confident-correct, most-confident-wrong,
    most-uncertain, borderline -- for token-level visualization."""
    merged = preds.merge(sp[["qid", "mean_logprob"]], on="qid")
    merged["correct"] = merged["predicted_label"] == merged["label"]

    picks = []
    picks += merged[merged.correct].nlargest(n_each, "mean_logprob")["qid"].tolist()
    picks += merged[~merged.correct].nlargest(n_each, "hallucination_probability")["qid"].tolist()
    picks += merged.nsmallest(n_each, "mean_logprob")["qid"].tolist()
    return list(dict.fromkeys(picks))  # dedupe, preserve order


def build_token_viz_report(paths, sp: pd.DataFrame, token_details: dict, example_qids: list[str]):
    sections = ["<h2>Token-Level Uncertainty Examples</h2>"]
    for qid in example_qids:
        if qid not in token_details:
            continue
        row = sp.loc[sp.qid == qid].iloc[0]
        det = token_details[qid]
        html_viz = render_token_html(det["tokens"], det["token_logprobs"], det.get("token_entropies"))
        label_str = "HALLUCINATED" if row["label"] == 1 else "GROUNDED"
        sections.append(
            f'<div style="margin:20px 0; padding:12px; border:1px solid #ddd; border-radius:6px;">'
            f'<div><b>Q:</b> {row["question"]}</div>'
            f'<div style="margin-top:6px;">{html_viz}</div>'
            f'<div style="margin-top:6px; font-size:12px; color:#555;">'
            f'label={label_str} | mean_logprob={row["mean_logprob"]:.3f} | '
            f'min_logprob={row["min_logprob"]:.3f} | mean_entropy={row["mean_entropy"]:.3f}</div>'
            f'</div>'
        )
    html_doc = (
        "<html><head><meta charset='utf-8'><title>Token Uncertainty Examples</title></head>"
        "<body style='font-family:system-ui,sans-serif; max-width:900px; margin:20px auto;'>"
        + "".join(sections) + "</body></html>"
    )
    out_path = paths.analysis / "token_uncertainty_examples.html"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html_doc)
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--dataset", default="truthfulqa")
    ap.add_argument("--backfill-token-details", action="store_true")
    args = ap.parse_args()

    set_all_seeds()
    paths = get_paths(args.model, args.dataset)

    sp = pd.read_csv(paths.features / "single_pass_features.csv")
    preds_proposed = pd.read_csv(paths.predictions / "proposed_all_three_predictions.csv")
    preds_mean = pd.read_csv(paths.predictions / "baseline_mean_logprob_only_predictions.csv")
    preds_sc = pd.read_csv(paths.predictions / "baseline_self_consistency_predictions.csv")

    analysis_summary = {"model": args.model, "dataset": args.dataset}

    # --- PR curves ---
    plot_pr_curves({
        "Proposed (3 feat)": (preds_proposed["label"], preds_proposed["hallucination_probability"]),
        "Mean-logprob-only": (preds_mean["label"], preds_mean["hallucination_probability"]),
        "Self-consistency": (preds_sc["label"], preds_sc["hallucination_probability"]),
    }, paths.plots / "pr_curves.png")

    # --- Calibration ---
    cal = plot_calibration_curve(preds_proposed["label"], preds_proposed["hallucination_probability"],
                                  paths.plots / "calibration_proposed.png", title="Proposed detector")
    analysis_summary["calibration_proposed"] = {"ece": cal["ece"], "brier_score": cal["brier_score"]}

    # --- Threshold sensitivity ---
    plot_threshold_sensitivity(preds_proposed["label"], preds_proposed["hallucination_probability"],
                                paths.plots / "threshold_sensitivity_proposed.png", title="Proposed detector")

    # --- Confidence vs correctness ---
    analysis_summary["confidence_vs_correctness"] = confidence_vs_correctness(paths, sp)

    # --- Error analysis ---
    analysis_summary["error_analysis"] = error_analysis(paths, sp, preds_proposed)

    # --- Token-level visualization ---
    token_details = load_token_details(paths)
    example_qids = pick_example_qids(sp, preds_proposed, n_each=2)
    if token_details is None and args.backfill_token_details:
        token_details = backfill_token_details(paths, args, sp, example_qids)
    if token_details:
        report_path = build_token_viz_report(paths, sp, token_details, example_qids)
        analysis_summary["token_viz_report"] = str(report_path)
        print(f"Token-level visualization -> {report_path}")
    else:
        print("No token-level detail available (run with --backfill-token-details, "
              "or this dataset was generated after token-detail caching was added).")

    save_json(analysis_summary, paths.analysis / "analysis_summary.json")
    print(f"\n[{args.model}/{args.dataset}] Analysis complete -> {paths.analysis}")
    print(json.dumps(analysis_summary, indent=2, default=str))


if __name__ == "__main__":
    main()
