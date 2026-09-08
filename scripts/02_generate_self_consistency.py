"""Step 2: 5-call self-consistency baseline.

For each question, draws 5 independent temperature-sampled generations
(diversity is required for the agreement signal to be meaningful -- greedy
decoding would make all 5 identical). Computes the mean pairwise
token-F1 agreement score and labels the medoid ("majority representative")
answer against the TruthfulQA references using the same proxy judge as
the single-pass detector.

This script alone accounts for 5x the LLM calls of the proposed detector
(report Section 4.4: "Self-consistency baseline: five independently
sampled generations per question").
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from tqdm import tqdm

from src.config import CFG, FEATURES_DIR, set_all_seeds
from src.data import build_or_load_subset
from src.labeling import label_answer
from src.model_utils import QwenGenerator
from src.self_consistency import agreement_score, medoid_answer

OUT_PATH = FEATURES_DIR / "self_consistency_features.csv"


def main():
    set_all_seeds()
    df = build_or_load_subset()
    print(f"Loaded {len(df)} questions.")

    generator = QwenGenerator()
    print(f"Model loaded on device: {generator.device}")

    records = []
    t0 = time.time()
    for q_idx, row in tqdm(df.iterrows(), total=len(df), desc="Self-consistency generation (5 calls/question)"):
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
    out_df.to_csv(OUT_PATH, index=False)
    print(f"Saved {len(out_df)} rows -> {OUT_PATH}")
    print(f"Elapsed: {elapsed:.1f}s ({elapsed/len(out_df):.2f}s/question, "
          f"{elapsed/(len(out_df)*CFG.self_consistency_n_samples):.2f}s/call)")
    print(f"Label balance: {out_df['label'].value_counts().to_dict()} (1=hallucinated, 0=grounded)")


if __name__ == "__main__":
    main()
