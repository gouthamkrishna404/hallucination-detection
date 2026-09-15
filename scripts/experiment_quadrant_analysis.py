"""The core diagnostic for "why doesn't this work on TruthfulQA": split
every answer into the 2x2 of {correct, wrong} x {confident, uncertain}
and look for measurable differences between the quadrants -- especially
whether "wrong + confident" (the textbook hallucination-that-evades-logprob-
detection case) is systematically different from "wrong + uncertain" in
ways a detector could exploit, or whether it's genuinely indistinguishable
from "correct + confident".

Confidence split: median mean_logprob within the full 200-question set
(not just eval, for a less noisy split point) -- "confident" = above
median (less negative), "uncertain" = below median. Correctness is the
proxy label (0=grounded/correct, 1=hallucinated/wrong).

Usage:
    python scripts/experiment_quadrant_analysis.py --model qwen2.5-1.5b-instruct --dataset truthfulqa
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from src.features_v2 import ALL_RICH_FEATURE_NAMES
from src.metrics_utils import save_json
from src.registry import get_paths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--dataset", default="truthfulqa")
    args = ap.parse_args()

    paths = get_paths(args.model, args.dataset)
    rich = pd.read_csv(paths.features / "single_pass_features_rich.csv")

    median_logprob = rich["mean_logprob"].median()
    rich["is_correct"] = rich["label"] == 0
    rich["is_confident"] = rich["mean_logprob"] >= median_logprob

    quadrants = {
        "correct_confident": rich[rich.is_correct & rich.is_confident],
        "correct_uncertain": rich[rich.is_correct & ~rich.is_confident],
        "wrong_confident": rich[~rich.is_correct & rich.is_confident],   # <- the hard case
        "wrong_uncertain": rich[~rich.is_correct & ~rich.is_confident],   # <- the easy case
    }

    print(f"[{args.model}/{args.dataset}] median mean_logprob split = {median_logprob:.3f}\n")
    print(f"{'Quadrant':22s} {'n':>4s} {'%':>6s}")
    for name, df in quadrants.items():
        print(f"{name:22s} {len(df):4d} {100*len(df)/len(rich):5.1f}%")

    print(f"\n{'Quadrant':22s} " + " ".join(f"{f[:14]:>15s}" for f in ALL_RICH_FEATURE_NAMES))
    quadrant_stats = {}
    for name, df in quadrants.items():
        means = {f: float(df[f].mean()) if len(df) else None for f in ALL_RICH_FEATURE_NAMES}
        quadrant_stats[name] = {"n": len(df), "pct": 100 * len(df) / len(rich), "feature_means": means}
        print(f"{name:22s} " + " ".join(f"{means[f]:15.3f}" if means[f] is not None else f"{'--':>15s}" for f in ALL_RICH_FEATURE_NAMES))

    # The critical comparison: wrong_confident vs correct_confident. If a detector could
    # separate these two, the "confident" bucket wouldn't collapse to one indistinguishable
    # blob -- that's the whole ballgame for TruthfulQA-style hallucination detection.
    wc, cc = quadrants["wrong_confident"], quadrants["correct_confident"]
    separation = {}
    if len(wc) > 2 and len(cc) > 2:
        for f in ALL_RICH_FEATURE_NAMES:
            wc_vals, cc_vals = wc[f].dropna(), cc[f].dropna()
            if len(wc_vals) < 2 or len(cc_vals) < 2:
                continue
            pooled_std = np.sqrt((wc_vals.var() + cc_vals.var()) / 2)
            cohens_d = float((wc_vals.mean() - cc_vals.mean()) / pooled_std) if pooled_std > 0 else 0.0
            separation[f] = {"wrong_confident_mean": float(wc_vals.mean()), "correct_confident_mean": float(cc_vals.mean()), "cohens_d": cohens_d}

    print(f"\n--- wrong_confident vs. correct_confident: the hard case, by |Cohen's d| ---")
    for f, s in sorted(separation.items(), key=lambda kv: -abs(kv[1]["cohens_d"])):
        print(f"  {f:32s} d={s['cohens_d']:+.3f}  (wrong_conf={s['wrong_confident_mean']:.3f} vs correct_conf={s['correct_confident_mean']:.3f})")

    # A few qualitative examples per quadrant for manual inspection.
    examples = {}
    for name, df in quadrants.items():
        sample = df.sample(min(3, len(df)), random_state=42) if len(df) else df
        examples[name] = sample[["question", "generated_answer", "mean_logprob", "mean_entropy"]].to_dict("records")

    result = {
        "median_logprob_split": float(median_logprob),
        "quadrant_counts": {k: {"n": v.shape[0], "pct": 100 * v.shape[0] / len(rich)} for k, v in quadrants.items()},
        "quadrant_feature_means": quadrant_stats,
        "wrong_confident_vs_correct_confident_separation": separation,
        "example_answers_per_quadrant": examples,
    }
    out_path = paths.analysis / "quadrant_analysis.json"
    save_json(result, out_path)
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
