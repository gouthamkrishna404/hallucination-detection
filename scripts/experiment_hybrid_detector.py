"""Hybrid detector: does combining single-call token-uncertainty features
with multi-call self-consistency signal (lexical agreement, semantic
entropy) substantially beat either alone?

Call-budget honesty: this hybrid needs the 1 greedy generation (for its
token-level features) PLUS the 5 temperature-sampled self-consistency
generations (for agreement_score / semantic_entropy) = 6 LLM calls per
question, MORE than pure self-consistency (5). It is reported with that
true cost, not rounded down to "5 calls" -- if it doesn't clearly beat
self-consistency alone, it isn't a good trade even before considering
whether it beats the 1-call detector.

Compares, on the identical train/eval split:
    1-call, original 3 features           (existing baseline)
    1-call, all 17 rich features            (this investigation's best single-call candidate)
    5-call, self-consistency alone          (existing baseline)
    6-call, rich features + agreement_score + semantic_entropy  (hybrid)

Usage:
    python scripts/experiment_hybrid_detector.py --model qwen2.5-1.5b-instruct --dataset truthfulqa
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src.config import set_all_seeds
from src.features_v2 import ALL_RICH_FEATURE_NAMES, ORIGINAL_THREE_FEATURES
from src.metrics_utils import compute_metrics, save_json
from src.registry import get_paths
from src.train_eval import fit_and_predict
from scripts.experiment_train_evaluate import get_qid_split


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--dataset", default="truthfulqa")
    args = ap.parse_args()

    set_all_seeds()
    paths = get_paths(args.model, args.dataset)

    rich = pd.read_csv(paths.features / "single_pass_features_rich.csv")
    sc = pd.read_csv(paths.features / "self_consistency_features.csv")[["qid", "agreement_score"]]
    sem_path = paths.features / "semantic_self_consistency_features.csv"
    sem = pd.read_csv(sem_path)[["qid", "semantic_entropy"]] if sem_path.exists() else None

    hybrid_df = rich.merge(sc, on="qid", how="inner")
    if sem is not None:
        hybrid_df = hybrid_df.merge(sem, on="qid", how="inner")
    hybrid_features = ALL_RICH_FEATURE_NAMES + ["agreement_score"] + (["semantic_entropy"] if sem is not None else [])

    sp = pd.read_csv(paths.features / "single_pass_features.csv")
    train_qids, eval_qids = get_qid_split(sp)

    configs = [
        ("1call_original_3feat", rich, ORIGINAL_THREE_FEATURES, 1),
        ("1call_all_17_rich", rich, ALL_RICH_FEATURE_NAMES, 1),
        ("hybrid_rich_plus_selfconsistency", hybrid_df, hybrid_features, 6),
    ]

    results = {}
    print(f"[{args.model}/{args.dataset}]")
    for name, df, feats, calls in configs:
        train_df = df[df.qid.isin(train_qids)].reset_index(drop=True)
        eval_df = df[df.qid.isin(eval_qids)].reset_index(drop=True)
        _, y_eval, y_prob = fit_and_predict(train_df, eval_df, feats)
        m = compute_metrics(y_eval, y_prob)
        m["calls_per_question"] = calls
        m["n_features"] = len(feats)
        results[name] = m
        print(f"  {name:38s} calls={calls}  n_feat={len(feats):2d}  auroc={m['auroc']:.3f}  acc={m['accuracy']:.3f}  n_eval={m['n']}")

    # For reference: pull the already-computed pure self-consistency (lexical) baseline AUROC
    import json
    metrics_path = paths.metrics / "all_metrics.json"
    if metrics_path.exists():
        with open(metrics_path, encoding="utf-8") as f:
            existing = json.load(f)
        sc_auroc = existing.get("baseline_self_consistency", {}).get("auroc")
        if sc_auroc is not None:
            print(f"  {'5call_self_consistency_lexical (ref)':38s} calls=5  n_feat= 1  auroc={sc_auroc:.3f}")
            results["reference_5call_self_consistency_lexical"] = {"auroc": sc_auroc, "calls_per_question": 5}

    out_path = paths.analysis / "hybrid_detector.json"
    save_json(results, out_path)
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
