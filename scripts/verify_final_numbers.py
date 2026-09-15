"""Recompute every headline number reported in the README directly from
saved outputs on disk -- no numbers are taken from memory or from
intermediate console output. Run this before finalizing any results
write-up; if a printed number here doesn't match the README, the README
is wrong and must be fixed, not the other way around.

Usage:
    python scripts/verify_final_numbers.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

MODELS = ["qwen2.5-1.5b-instruct", "llama-3.2-1b-instruct", "smollm2-1.7b-instruct"]
DATASETS = ["truthfulqa", "sciq"]


def load(model, dataset, rel_path):
    p = Path("results") / model / dataset / rel_path
    if not p.exists():
        return None
    if p.suffix == ".json":
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    return pd.read_csv(p)


def main():
    print("=" * 100)
    print("MAIN MATRIX: all_metrics.json (proposed_all_three, baseline_mean_logprob_only, baseline_self_consistency)")
    print("=" * 100)
    for model in MODELS:
        for dataset in DATASETS:
            m = load(model, dataset, "metrics/all_metrics.json")
            if m is None:
                print(f"  MISSING: {model}/{dataset}")
                continue
            p3 = m["proposed_all_three"]
            mlo = m["baseline_mean_logprob_only"]
            sc = m["baseline_self_consistency"]
            sem_sc = m.get("semantic_self_consistency", {})
            print(f"{model:25s} {dataset:12s} proposed3={p3['auroc']:.3f} mean_lp={mlo['auroc']:.3f} "
                  f"lex_sc={sc['auroc']:.3f} sem_sc={sem_sc.get('auroc', float('nan')):.3f}")

    print()
    print("=" * 100)
    print("FEATURE ABLATION: best individual features (held-out AUROC), TruthfulQA")
    print("=" * 100)
    for model in MODELS:
        d = load(model, "truthfulqa", "analysis/feature_ablation.json")
        if d is None:
            continue
        for feat in ["mean_logprob", "median_logprob", "logprob_p25", "logprob_trend_slope",
                     "logprob_delta_second_minus_first", "answer_length"]:
            key = f"single__{feat}"
            if key in d:
                print(f"  {model:25s} {feat:35s} cv={d[key]['cv_auroc_mean']:.3f} held_out={d[key]['held_out_auroc']:.3f}")

    print()
    print("=" * 100)
    print("QUADRANT ANALYSIS: fraction of wrong answers that are confidently wrong")
    print("=" * 100)
    for model in MODELS:
        for dataset in DATASETS:
            d = load(model, dataset, "analysis/quadrant_analysis.json")
            if d is None:
                continue
            qc = d["quadrant_counts"]
            wc, wu = qc["wrong_confident"]["n"], qc["wrong_uncertain"]["n"]
            frac = wc / (wc + wu) if (wc + wu) else float("nan")
            print(f"  {model:25s} {dataset:12s} wrong_confident={wc:3d} wrong_uncertain={wu:3d}  frac_confidently_wrong={frac:.1%}")

    print()
    print("=" * 100)
    print("BEST CANDIDATE BOOTSTRAP (comparison/*_best_candidate_bootstrap.json)")
    print("=" * 100)
    for dataset in DATASETS:
        p = Path("comparison") / f"{dataset}_best_candidate_bootstrap.json"
        if not p.exists():
            print(f"  MISSING: {p}")
            continue
        with open(p, encoding="utf-8") as f:
            d = json.load(f)
        for model, methods in d.items():
            for method, res in methods.items():
                fs = res["fixed_split_n60"]
                oof = res["out_of_fold_n200"]
                print(f"  {model:25s} {method:18s} fixed_n60={fs['point_estimate']:.3f}[{fs['ci_lower_2.5pct']:.3f},{fs['ci_upper_97.5pct']:.3f}]"
                      f"  oof_n200={oof['point_estimate']:.3f}[{oof['ci_lower_2.5pct']:.3f},{oof['ci_upper_97.5pct']:.3f}]"
                      f"  excludes_chance={oof['excludes_chance']}")

    print()
    print("=" * 100)
    print("CROSS-MODEL TRANSFER (comparison/*_cross_model_transfer.json)")
    print("=" * 100)
    for dataset in DATASETS:
        p = Path("comparison") / f"{dataset}_cross_model_transfer.json"
        if not p.exists():
            continue
        with open(p, encoding="utf-8") as f:
            d = json.load(f)
        for k, v in d.items():
            if "raw_auroc" in v:
                print(f"  {dataset:12s} {k:60s} raw={v['raw_auroc']:.3f} std={v['standardized_auroc']:.3f}")
        for k, v in d.items():
            if "within_model_reference" in k:
                print(f"  {dataset:12s} {k:60s} auroc={v['auroc']:.3f}")

    print()
    print("=" * 100)
    print("HYBRID DETECTOR (analysis/hybrid_detector.json)")
    print("=" * 100)
    for model in MODELS:
        for dataset in DATASETS:
            d = load(model, dataset, "analysis/hybrid_detector.json")
            if d is None:
                continue
            for k, v in d.items():
                print(f"  {model:25s} {dataset:10s} {k:38s} calls={v.get('calls_per_question','?')} auroc={v.get('auroc', float('nan')):.3f}")

    print()
    print("=" * 100)
    print("LABEL ROBUSTNESS (analysis/label_robustness_check.json)")
    print("=" * 100)
    for model in MODELS:
        for dataset in DATASETS:
            d = load(model, dataset, "analysis/label_robustness_check.json")
            if d is None:
                continue
            print(f"  {model:25s} {dataset:12s} agreement={d['label_agreement_rate']:.3f} kappa={d['cohen_kappa']:.3f} "
                  f"auroc_tok={d['proposed_detector_under_token_f1_labels']['auroc']:.3f} "
                  f"auroc_sem={d['proposed_detector_under_semantic_labels']['auroc']:.3f}")


if __name__ == "__main__":
    main()
