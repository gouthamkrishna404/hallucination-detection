"""Generalized single-pass generation: 1 greedy LLM call/question -> logprob
features -> proxy label, for any (model, dataset) pair in the registries.

This is the multi-model/multi-dataset generalization of
scripts/01_generate_single_pass.py -- same feature extraction, same
labeling, same decoding parameters (CFG.single_pass_*), writing to
results/<model>/<dataset>/ instead of the frozen outputs/ tree so DA1
artifacts are never touched.

Usage:
    python scripts/experiment_generate_single_pass.py --model llama-3.2-1b-instruct --dataset truthfulqa
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from tqdm import tqdm

from src.config import CFG, set_all_seeds
from src.datasets import load_subset
from src.features import extract_features
from src.labeling import label_answer
from src.model_utils import LLMGenerator
from src.registry import get_paths
from src.run_logger import log_event


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="key from configs/models.yaml")
    ap.add_argument("--dataset", default="truthfulqa", help="key from configs/datasets.yaml")
    ap.add_argument("--subset-size", type=int, default=CFG.subset_size)
    ap.add_argument("--force", action="store_true", help="regenerate even if cached output exists")
    args = ap.parse_args()

    paths = get_paths(args.model, args.dataset)
    out_path = paths.features / "single_pass_features.csv"
    if out_path.exists() and not args.force:
        print(f"Already exists, skipping (use --force to regenerate): {out_path}")
        return

    log_event("experiment_generate_single_pass", args.model, args.dataset, "started", subset_size=args.subset_size)
    set_all_seeds()
    df = load_subset(args.dataset, args.subset_size, CFG.seed)
    print(f"[{args.model}/{args.dataset}] Loaded {len(df)} questions.")

    generator = LLMGenerator.from_registry(args.model)
    print(f"Model loaded on device: {generator.device}")

    records = []
    token_details = []  # per-question per-token logprob/entropy, for token-level viz + demo
    t0 = time.time()
    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Single-pass [{args.model}/{args.dataset}]"):
        result = generator.generate_with_scores(
            question=row["question"],
            max_new_tokens=CFG.single_pass_max_new_tokens,
            do_sample=CFG.single_pass_do_sample,
        )
        feats = extract_features(result.token_logprobs, result.token_entropies)
        lab = label_answer(result.answer_text, row["best_answer"], row["correct_answers"], row["incorrect_answers"])

        records.append({
            "qid": row["qid"],
            "question": row["question"],
            "category": row["category"],
            "generated_answer": result.answer_text,
            "num_new_tokens": result.num_new_tokens,
            **feats,
            **lab,
            "llm_calls": 1,
        })
        token_details.append({
            "qid": row["qid"],
            "tokens": result.token_strs,
            "token_logprobs": result.token_logprobs,
            "token_entropies": result.token_entropies,
        })

    elapsed = time.time() - t0
    out_df = pd.DataFrame.from_records(records)
    out_df.to_csv(out_path, index=False)

    token_details_path = paths.features / "single_pass_token_details.jsonl"
    with open(token_details_path, "w", encoding="utf-8") as f:
        for rec in token_details:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"Saved {len(out_df)} rows -> {out_path}")
    print(f"Saved token-level detail -> {token_details_path}")
    log_event("experiment_generate_single_pass", args.model, args.dataset, "completed",
              n_questions=len(out_df), elapsed_sec=round(elapsed, 1),
              label_balance=out_df["label"].value_counts().to_dict())
    print(f"Elapsed: {elapsed:.1f}s ({elapsed/len(out_df):.2f}s/question)")
    print(f"Label balance: {out_df['label'].value_counts().to_dict()} (1=hallucinated, 0=grounded)")


if __name__ == "__main__":
    main()
