"""Generalized 5-call self-consistency baseline for any (model, dataset)
pair in the registries. Multi-model/multi-dataset generalization of
scripts/02_generate_self_consistency.py -- identical agreement-score
logic and decoding parameters (CFG.self_consistency_*), writing to
results/<model>/<dataset>/ instead of the frozen outputs/ tree.

Usage:
    python scripts/experiment_generate_self_consistency.py --model llama-3.2-1b-instruct --dataset truthfulqa
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from tqdm import tqdm

from src.config import CFG, set_all_seeds
from src.datasets import load_subset
from src.labeling import label_answer
from src.model_utils import LLMGenerator
from src.registry import get_paths
from src.run_logger import log_event
from src.self_consistency import agreement_score, medoid_answer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--dataset", default="truthfulqa")
    ap.add_argument("--subset-size", type=int, default=CFG.subset_size)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    paths = get_paths(args.model, args.dataset)
    out_path = paths.features / "self_consistency_features.csv"
    if out_path.exists() and not args.force:
        print(f"Already exists, skipping (use --force to regenerate): {out_path}")
        return

    log_event("experiment_generate_self_consistency", args.model, args.dataset, "started", subset_size=args.subset_size)
    set_all_seeds()
    df = load_subset(args.dataset, args.subset_size, CFG.seed)
    print(f"[{args.model}/{args.dataset}] Loaded {len(df)} questions.")

    generator = LLMGenerator.from_registry(args.model)
    print(f"Model loaded on device: {generator.device}")

    records = []
    t0 = time.time()
    for q_idx, row in tqdm(df.iterrows(), total=len(df), desc=f"Self-consistency [{args.model}/{args.dataset}]"):
        answers = []
        for k in range(CFG.self_consistency_n_samples):
            result = generator.generate_with_scores(
                question=row["question"],
                max_new_tokens=CFG.self_consistency_max_new_tokens,
                do_sample=True,
                temperature=CFG.self_consistency_temperature,
                top_p=CFG.self_consistency_top_p,
                seed=CFG.seed * 1000 + q_idx * CFG.self_consistency_n_samples + k,
            )
            answers.append(result.answer_text)

        score = agreement_score(answers)
        rep_answer = medoid_answer(answers)
        lab = label_answer(rep_answer, row["best_answer"], row["correct_answers"], row["incorrect_answers"])

        records.append({
            "qid": row["qid"],
            "question": row["question"],
            "category": row["category"],
            "sampled_answers": answers,
            "representative_answer": rep_answer,
            "agreement_score": score,
            **lab,
            "llm_calls": CFG.self_consistency_n_samples,
        })

    elapsed = time.time() - t0
    out_df = pd.DataFrame.from_records(records)
    out_df.to_csv(out_path, index=False)
    print(f"Saved {len(out_df)} rows -> {out_path}")
    print(f"Elapsed: {elapsed:.1f}s ({elapsed/len(out_df):.2f}s/question, "
          f"{elapsed/(len(out_df)*CFG.self_consistency_n_samples):.2f}s/call)")
    print(f"Label balance: {out_df['label'].value_counts().to_dict()} (1=hallucinated, 0=grounded)")
    log_event("experiment_generate_self_consistency", args.model, args.dataset, "completed",
              n_questions=len(out_df), elapsed_sec=round(elapsed, 1),
              label_balance=out_df["label"].value_counts().to_dict())


if __name__ == "__main__":
    main()
