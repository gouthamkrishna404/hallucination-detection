"""The sharpest possible version of "can we detect confidently-wrong
hallucinations": restrict to ONLY the confident bucket (above-median
mean_logprob) and ask whether any feature set can separate
wrong-but-confident from correct-but-confident answers -- i.e., can we
tell the two apart WHEN THE MODEL ITSELF GIVES NO HINT via low
confidence? This is the single hardest and most relevant test for the
"confidently-wrong hallucination" problem the whole investigation is
about.

Evaluated with 5-fold out-of-fold predictions + bootstrap 95% CI within
the confident subgroup (n is roughly half of 200, still enough for 5-fold
CV), for both the original 3 features and the candidate 4 features.

Usage:
    python scripts/experiment_confident_subgroup_classifier.py --model qwen2.5-1.5b-instruct --dataset truthfulqa
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src.config import set_all_seeds
from src.features_v2 import ALL_RICH_FEATURE_NAMES, ORIGINAL_THREE_FEATURES
from src.metrics_utils import save_json
from src.registry import get_paths
from scripts.experiment_best_candidate import CANDIDATE_FEATURES, bootstrap_auroc_ci
from scripts.experiment_label_margin_deep_dive import oof_predictions_small_n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--dataset", default="truthfulqa")
    args = ap.parse_args()

    set_all_seeds()
    paths = get_paths(args.model, args.dataset)
    rich = pd.read_csv(paths.features / "single_pass_features_rich.csv")

    median_logprob = rich["mean_logprob"].median()
    confident = rich[rich["mean_logprob"] >= median_logprob].reset_index(drop=True)
    n, n_pos = len(confident), int(confident["label"].sum())
    print(f"[{args.model}/{args.dataset}] confident subgroup: n={n}, n_wrong(confidently_wrong)={n_pos}, "
          f"n_correct(confidently_correct)={n - n_pos}")

    results = {"n": n, "n_wrong_confident": n_pos, "n_correct_confident": n - n_pos}
    for name, feats in [("original_3feat", ORIGINAL_THREE_FEATURES), ("candidate_4feat", CANDIDATE_FEATURES),
                          ("all_17_rich", ALL_RICH_FEATURE_NAMES)]:
        y, prob = oof_predictions_small_n(confident, feats, n_splits=5)
        ci = bootstrap_auroc_ci(y, prob, n_boot=2000)
        results[name] = ci
        flag = "EXCLUDES chance" if ci["excludes_chance"] else "does not exclude chance"
        print(f"  {name:16s} 5-fold-OOF AUROC={ci['point_estimate']:.3f}  "
              f"95% CI=[{ci['ci_lower_2.5pct']:.3f}, {ci['ci_upper_97.5pct']:.3f}]  ({flag})")

    out_path = paths.analysis / "confident_subgroup_classifier.json"
    save_json(results, out_path)
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()
