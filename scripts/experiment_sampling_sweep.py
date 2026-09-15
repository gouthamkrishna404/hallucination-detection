"""Is 5 samples enough, too many, or too few for the self-consistency
baseline? Sweeps the self-consistency sample budget N in {3, 5, 10} (the
existing 5-sample cache plus 5 newly generated extra samples, see
scripts/extend_self_consistency_samples.py), recomputing lexical agreement
and semantic entropy from the first N samples at each budget and
re-evaluating identically.

N=1 is deliberately NOT part of this sweep as "1 stochastic sample" --
"agreement" is undefined for a single draw, and the project's actual
1-call detector (the greedy single-pass baseline, reported everywhere
else) is a different, more meaningful, deterministic reference point. It
is included in the printed table for context, pulled from the existing
metrics.json rather than redefined here.

Usage:
    python scripts/experiment_sampling_sweep.py --model qwen2.5-1.5b-instruct --dataset truthfulqa
"""
import argparse
import ast
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src.config import set_all_seeds
from src.metrics_utils import compute_metrics, save_json
from src.registry import get_paths
from src.self_consistency import agreement_score
from src.semantic_similarity import semantic_entropy
from src.train_eval import fit_and_predict
from scripts.experiment_train_evaluate import get_qid_split


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--dataset", default="truthfulqa")
    args = ap.parse_args()

    set_all_seeds()
    paths = get_paths(args.model, args.dataset)

    sc = pd.read_csv(paths.features / "self_consistency_features.csv")
    extra = pd.read_csv(paths.features / "self_consistency_extra_samples.csv")
    extra_map = extra.set_index("qid")["extra_answers"].apply(ast.literal_eval).to_dict()

    sp = pd.read_csv(paths.features / "single_pass_features.csv")
    train_qids, eval_qids = get_qid_split(sp)

    results = {}
    for n in [3, 5, 10]:
        records = []
        for _, row in sc.iterrows():
            qid = row["qid"]
            samples = ast.literal_eval(row["sampled_answers"])[:min(n, 5)]
            if n > 5:
                samples = samples + extra_map[qid][: n - 5]
            records.append({
                "qid": qid,
                "agreement_score": agreement_score(samples),
                "semantic_entropy": semantic_entropy(samples),
                "label": row["label"],  # same proxy-label procedure as the n=5 baseline, applied consistently
            })
        df = pd.DataFrame.from_records(records)
        train_df = df[df.qid.isin(train_qids)].reset_index(drop=True)
        eval_df = df[df.qid.isin(eval_qids)].reset_index(drop=True)

        row_result = {"n_samples": n, "total_llm_calls": n * len(sc)}
        for feat_name, feat in [("lexical_agreement", "agreement_score"), ("semantic_entropy", "semantic_entropy")]:
            _, y_eval, y_prob = fit_and_predict(train_df, eval_df, [feat])
            m = compute_metrics(y_eval, y_prob)
            row_result[f"{feat_name}_auroc"] = m["auroc"]
        results[f"n={n}"] = row_result
        print(f"  n_samples={n:2d}  total_calls={row_result['total_llm_calls']:4d}  "
              f"lexical_auroc={row_result['lexical_agreement_auroc']:.3f}  "
              f"semantic_auroc={row_result['semantic_entropy_auroc']:.3f}")

    # Reference: the actual 1-call greedy detector, from the existing metrics file
    metrics_path = paths.metrics / "all_metrics.json"
    if metrics_path.exists():
        with open(metrics_path, encoding="utf-8") as f:
            existing = json.load(f)
        ref_auroc = existing.get("proposed_all_three", {}).get("auroc")
        if ref_auroc is not None:
            results["n=1_greedy_reference"] = {"n_samples": 1, "total_llm_calls": len(sc), "proposed_3feat_auroc": ref_auroc}
            print(f"  n_samples= 1 (greedy, reference)  total_calls={len(sc):4d}  proposed_3feat_auroc={ref_auroc:.3f}")

    out_path = paths.analysis / "sampling_budget_sweep.json"
    save_json(results, out_path)
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
