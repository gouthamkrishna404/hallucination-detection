"""Dataset loaders. Every loader returns a DataFrame with the unified schema:

    qid, question, best_answer, correct_answers (list[str]),
    incorrect_answers (list[str]), category

which src/labeling.py, src/features.py-consuming scripts, and
src/train_eval.py all depend on -- this is what lets the exact same
generation/labeling/feature/classifier code run over any dataset in the
registry (configs/datasets.yaml) unmodified.

Subsets are selected once (fixed seed, fixed size) and cached to
data/<dataset_key>/subset_<n>.jsonl so every script and every model run
against the identical question set.
"""
import json

import pandas as pd

from ..registry import get_paths, resolve_dataset


def _cache_path(dataset_key: str, subset_size: int):
    from ..config import PROJECT_ROOT
    d = PROJECT_ROOT / "data" / dataset_key
    d.mkdir(parents=True, exist_ok=True)
    return d / f"subset_{subset_size}.jsonl"


def _save_subset(df: pd.DataFrame, path):
    with open(path, "w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            record = row.to_dict()
            record["correct_answers"] = list(record["correct_answers"])
            record["incorrect_answers"] = list(record["incorrect_answers"])
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_subset(dataset_key: str, subset_size: int, seed: int) -> pd.DataFrame:
    """Load (or build+cache) the fixed subset for `dataset_key`."""
    path = _cache_path(dataset_key, subset_size)
    if path.exists():
        return pd.read_json(path, lines=True)

    resolve_dataset(dataset_key)  # validates key
    if dataset_key == "truthfulqa":
        from .truthfulqa import build_subset
    elif dataset_key == "sciq":
        from .sciq import build_subset
    else:
        raise ValueError(f"No loader implemented for dataset '{dataset_key}'")

    df = build_subset(subset_size=subset_size, seed=seed)
    _save_subset(df, path)
    return df
