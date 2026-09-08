"""SciQ loader -> unified schema.

SciQ ships one correct_answer plus three wrong-but-plausible distractors
per science question -- these map directly onto the correct_answers /
incorrect_answers columns the rest of the pipeline expects, with no
special-casing needed anywhere else (labeling, features, training all stay
dataset-agnostic).
"""
import pandas as pd
from datasets import load_dataset

from ..registry import resolve_dataset


def build_subset(subset_size: int, seed: int) -> pd.DataFrame:
    cfg = resolve_dataset("sciq")
    if cfg.get("hf_config"):
        ds = load_dataset(cfg["hf_name"], cfg["hf_config"], split=cfg["hf_split"])
    else:
        ds = load_dataset(cfg["hf_name"], split=cfg["hf_split"])
    df = ds.to_pandas()

    df = df.sample(n=min(subset_size, len(df)), random_state=seed).reset_index(drop=True)
    df["qid"] = [f"sciq_{i:04d}" for i in range(len(df))]
    df["best_answer"] = df["correct_answer"]
    df["correct_answers"] = df["correct_answer"].apply(lambda a: [a])
    df["incorrect_answers"] = df.apply(
        lambda r: [r["distractor1"], r["distractor2"], r["distractor3"]], axis=1
    )
    df["category"] = "science"

    keep_cols = ["qid", "question", "best_answer", "correct_answers", "incorrect_answers", "category"]
    return df[keep_cols]
