"""Combine the frozen single_pass_features.csv (qid, label, generated_answer,
category, ...) with the per-token cache (logprobs/entropies/margins) into
single_pass_features_rich.csv -- the expanded feature table used by every
downstream investigation script (ablation, classifier comparison, quadrant
analysis, hybrid detector, cross-model transfer).

Never modifies single_pass_features.csv. Requires
single_pass_token_details_v2.jsonl to exist first (see
scripts/backfill_rich_token_details.py).

Usage:
    python scripts/build_rich_features.py --model qwen2.5-1.5b-instruct --dataset truthfulqa
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src.features_v2 import extract_rich_features
from src.registry import get_paths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--dataset", default="truthfulqa")
    args = ap.parse_args()

    paths = get_paths(args.model, args.dataset)
    sp = pd.read_csv(paths.features / "single_pass_features.csv")

    # Prefer the v2 cache (has token_margins); fall back to the original cache
    # (logprobs + entropies only -- no new LLM calls needed either way) if v2
    # hasn't been backfilled for this model/dataset yet. Margin features are
    # simply 0.0/unavailable in that case (extract_rich_features handles it).
    v2_path = paths.features / "single_pass_token_details_v2.jsonl"
    v1_path = paths.features / "single_pass_token_details.jsonl"
    detail_path = v2_path if v2_path.exists() else v1_path
    has_margins = detail_path == v2_path
    print(f"Using token-detail cache: {detail_path.name} (margins {'available' if has_margins else 'NOT available -- backfill with scripts/backfill_rich_token_details.py for margin features'})")

    token_details = {}
    with open(detail_path, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            token_details[d["qid"]] = d

    records = []
    n_missing = 0
    for _, row in sp.iterrows():
        qid = row["qid"]
        if qid not in token_details:
            n_missing += 1
            continue
        d = token_details[qid]
        feats = extract_rich_features(d["token_logprobs"], d["token_entropies"], d.get("token_margins"))
        records.append({
            "qid": qid,
            "question": row["question"],
            "category": row.get("category"),
            "generated_answer": row["generated_answer"],
            "label": row["label"],
            "correct_sim": row.get("correct_sim"),
            "incorrect_sim": row.get("incorrect_sim"),
            **feats,
        })

    if n_missing:
        print(f"WARNING: {n_missing} qid(s) in single_pass_features.csv have no cached token detail "
              f"(likely a determinism mismatch during backfill) and were dropped.")

    out_df = pd.DataFrame.from_records(records)
    out_path = paths.features / "single_pass_features_rich.csv"
    out_df.to_csv(out_path, index=False)
    print(f"Saved {len(out_df)} rows, {len([c for c in out_df.columns if c not in ('qid','question','category','generated_answer','label','correct_sim','incorrect_sim')])} rich features -> {out_path}")


if __name__ == "__main__":
    main()
