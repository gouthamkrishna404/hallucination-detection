"""Validity check on the headline 17-feature TruthfulQA results: does the
improvement survive (a) removing answer_length, a feature shown to be
confounded with the token-F1 proxy labeler (longer answers are more often
labeled "grounded", p=0.004), and (b) scoring against the embedding-based
labeler, which has no measurable length bias (p=0.81)?

Same 5-fold out-of-fold protocol and bootstrap CIs as
experiment_cascade_detector.py; same seed.

Usage:
    python scripts/experiment_length_confound_check.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src.config import set_all_seeds
from src.features_v2 import ALL_RICH_FEATURE_NAMES
from src.metrics_utils import save_json
from src.registry import get_paths
from scripts.experiment_best_candidate import bootstrap_auroc_ci, out_of_fold_predictions
from scripts.experiment_cascade_detector import cascade_oof

MODELS = ["qwen2.5-1.5b-instruct", "llama-3.2-1b-instruct", "smollm2-1.7b-instruct"]
NO_LENGTH = [f for f in ALL_RICH_FEATURE_NAMES if f != "answer_length"]


def main():
    set_all_seeds()
    results = {}
    for model in MODELS:
        paths = get_paths(model, "truthfulqa")
        rich = pd.read_csv(paths.features / "single_pass_features_rich.csv")
        sem = pd.read_csv(paths.features / "single_pass_features_semantic_labels.csv")[["qid", "semantic_label"]]
        rich_sem = rich.merge(sem, on="qid").drop(columns=["label"]).rename(columns={"semantic_label": "label"})

        runs = {
            "flat_17_tokenf1": out_of_fold_predictions(rich, ALL_RICH_FEATURE_NAMES, n_splits=5),
            "flat_16_no_length_tokenf1": out_of_fold_predictions(rich, NO_LENGTH, n_splits=5),
            "cascade_17_tokenf1": cascade_oof(rich, ALL_RICH_FEATURE_NAMES),
            "cascade_16_no_length_tokenf1": cascade_oof(rich, NO_LENGTH),
            "flat_16_no_length_semantic_label": out_of_fold_predictions(rich_sem, NO_LENGTH, n_splits=5),
            "cascade_16_no_length_semantic_label": cascade_oof(rich_sem, NO_LENGTH),
        }
        results[model] = {}
        print(f"\n[{model}]  5-fold OOF, n=200")
        for name, (y, p) in runs.items():
            ci = bootstrap_auroc_ci(y, p)
            results[model][name] = ci
            flag = "EXCLUDES chance" if ci["excludes_chance"] else "does not exclude chance"
            print(f"  {name:38s} AUROC={ci['point_estimate']:.3f}  "
                  f"[{ci['ci_lower_2.5pct']:.3f}, {ci['ci_upper_97.5pct']:.3f}]  ({flag})")

    out = Path("comparison") / "truthfulqa_length_confound_check.json"
    save_json(results, out)
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
