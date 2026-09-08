"""Load the fixed TruthfulQA subset used throughout the project.

Report Section 4.4: "Dataset: TruthfulQA, using a fixed subset of
approximately 200 questions selected before evaluation." The subset is
chosen once with a fixed seed and cached to disk so every script
(single-pass generation, self-consistency generation, training) operates
on exactly the same questions in exactly the same order.
"""
import json

import pandas as pd
from datasets import load_dataset

from .config import CFG, DATA_DIR


SUBSET_PATH = DATA_DIR / f"truthfulqa_subset_{CFG.subset_size}.jsonl"


def build_or_load_subset() -> pd.DataFrame:
    """Return the fixed TruthfulQA subset as a DataFrame, building it once."""
    if SUBSET_PATH.exists():
        return pd.read_json(SUBSET_PATH, lines=True)

    ds = load_dataset(CFG.dataset_name, CFG.dataset_config, split="validation")
    df = ds.to_pandas()

    # Deterministic shuffle + fixed-size subset (report: "selected before evaluation").
    df = df.sample(n=min(CFG.subset_size, len(df)), random_state=CFG.seed).reset_index(drop=True)

    df["qid"] = [f"tqa_{i:04d}" for i in range(len(df))]
    keep_cols = ["qid", "question", "best_answer", "correct_answers", "incorrect_answers", "category"]
    df = df[keep_cols]

    with open(SUBSET_PATH, "w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            record = row.to_dict()
            record["correct_answers"] = list(record["correct_answers"])
            record["incorrect_answers"] = list(record["incorrect_answers"])
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    return df


if __name__ == "__main__":
    d = build_or_load_subset()
    print(f"Loaded {len(d)} questions -> {SUBSET_PATH}")
    print(d.iloc[0])
