"""Step 1: single generation per question (greedy) -> logprob features -> proxy label.

This is the proposed detector's ONLY LLM interaction: one call per question.
Outputs are cached to disk so re-running training/eval does not require
regenerating from the model.
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from tqdm import tqdm

from src.config import CFG, FEATURES_DIR, CACHE_DIR, set_all_seeds
from src.data import build_or_load_subset
from src.features import extract_features
from src.labeling import label_answer
from src.model_utils import QwenGenerator

OUT_PATH = FEATURES_DIR / "single_pass_features.csv"


def main():
    set_all_seeds()
    df = build_or_load_subset()
    print(f"Loaded {len(df)} questions.")

    generator = QwenGenerator()
    print(f"Model loaded on device: {generator.device}")

    records = []
    t0 = time.time()
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Single-pass generation"):
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

    elapsed = time.time() - t0
    out_df = pd.DataFrame.from_records(records)
    out_df.to_csv(OUT_PATH, index=False)
    print(f"Saved {len(out_df)} rows -> {OUT_PATH}")
    print(f"Elapsed: {elapsed:.1f}s ({elapsed/len(out_df):.2f}s/question)")
    print(f"Label balance: {out_df['label'].value_counts().to_dict()} (1=hallucinated, 0=grounded)")


if __name__ == "__main__":
    main()
