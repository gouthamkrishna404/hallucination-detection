"""TruthfulQA loader -> unified schema. Same logic as the original DA1
src/data.py, factored out here so it can be selected via the dataset
registry alongside other datasets.
"""
import pandas as pd
from datasets import load_dataset

from ..registry import resolve_dataset


def build_subset(subset_size: int, seed: int) -> pd.DataFrame:
    cfg = resolve_dataset("truthfulqa")
    ds = load_dataset(cfg["hf_name"], cfg["hf_config"], split=cfg["hf_split"])
    df = ds.to_pandas()

    df = df.sample(n=min(subset_size, len(df)), random_state=seed).reset_index(drop=True)
    df["qid"] = [f"tqa_{i:04d}" for i in range(len(df))]

    keep_cols = ["qid", "question", "best_answer", "correct_answers", "incorrect_answers", "category"]
    return df[keep_cols]
