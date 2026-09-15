"""The definitive comparison: every detector this investigation produced,
head-to-head, against the SAME target (the single-pass greedy answer's
proxy label) so the comparison is apples-to-apples -- earlier standalone
baseline numbers for lexical/semantic self-consistency evaluated against
their OWN medoid-derived label, which is the right choice for judging
those methods as complete standalone systems, but the wrong choice for
directly comparing five detectors abreast, since a different target
partially explains a different AUROC on its own. This script fixes the
target across all five:

    1. original_3feat        (1 call)  -- mean/min logprob + mean entropy
    2. candidate_4feat        (1 call)  -- median/p25/trend/positional-delta logprob
    3. lexical_self_consistency (5 calls) -- agreement_score
    4. semantic_self_consistency (5 calls) -- semantic_entropy
    5. hybrid                  (6 calls) -- all 17 rich + agreement_score + semantic_entropy

Every method is scored with 10-fold out-of-fold AUROC across the full
200-question set (leakage-free: each question scored by a fold that
never trained on it) plus a bootstrap 95% CI on those out-of-fold
predictions -- the same higher-power procedure validated in
experiment_best_candidate.py (which showed it correctly detects the
known-real SciQ signal, so it isn't simply too conservative to find an
effect when one exists).

Usage:
    python scripts/experiment_final_headtohead.py --dataset truthfulqa
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from src.config import CFG, set_all_seeds
from src.features_v2 import ALL_RICH_FEATURE_NAMES, ORIGINAL_THREE_FEATURES
from src.metrics_utils import save_json
from src.registry import get_paths
from scripts.experiment_best_candidate import CANDIDATE_FEATURES, bootstrap_auroc_ci, out_of_fold_predictions


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="truthfulqa")
    ap.add_argument("--models", nargs="+", default=["qwen2.5-1.5b-instruct", "llama-3.2-1b-instruct", "smollm2-1.7b-instruct"])
    args = ap.parse_args()

    set_all_seeds()
    all_results = {}

    for model in args.models:
        paths = get_paths(model, args.dataset)
        rich = pd.read_csv(paths.features / "single_pass_features_rich.csv")
        sc = pd.read_csv(paths.features / "self_consistency_features.csv")[["qid", "agreement_score"]]
        sem_path = paths.features / "semantic_self_consistency_features.csv"
        sem = pd.read_csv(sem_path)[["qid", "semantic_entropy"]] if sem_path.exists() else None

        # Merge everything onto the single-pass label -- the one fixed target for this comparison.
        merged = rich.merge(sc, on="qid", how="inner")
        if sem is not None:
            merged = merged.merge(sem, on="qid", how="inner")

        methods = [
            ("1call_original_3feat", ORIGINAL_THREE_FEATURES, 1),
            ("1call_candidate_4feat", CANDIDATE_FEATURES, 1),
            ("5call_lexical_self_consistency", ["agreement_score"], 5),
        ]
        if sem is not None:
            methods.append(("5call_semantic_self_consistency", ["semantic_entropy"], 5))
            methods.append(("6call_hybrid", ALL_RICH_FEATURE_NAMES + ["agreement_score", "semantic_entropy"], 6))

        print(f"\n[{model} / {args.dataset}]")
        model_results = {}
        for name, feats in [(n, f) for n, f, _ in methods]:
            calls = next(c for n, f, c in methods if n == name)
            y_oof, prob_oof = out_of_fold_predictions(merged, feats)
            ci = bootstrap_auroc_ci(y_oof, prob_oof)
            model_results[name] = {"features": feats, "calls_per_question": calls, **ci}
            flag = "EXCLUDES chance" if ci["excludes_chance"] else "does not exclude chance"
            print(f"  {name:32s} calls={calls}  AUROC={ci['point_estimate']:.3f}  "
                  f"95% CI=[{ci['ci_lower_2.5pct']:.3f}, {ci['ci_upper_97.5pct']:.3f}]  ({flag})")
        all_results[model] = model_results

    out_path = Path("comparison") / f"{args.dataset}_final_headtohead.json"
    save_json(all_results, out_path)
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
