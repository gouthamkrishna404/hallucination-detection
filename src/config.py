"""Central configuration and reproducibility settings for the whole pipeline.

Every script imports from here so that the model checkpoint, dataset
subset size, split ratio, decoding parameters and random seeds are
frozen in exactly one place (see report Section 4.4, "Reproducibility").
"""
import random
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
CACHE_DIR = OUTPUTS_DIR / "cache"
FEATURES_DIR = OUTPUTS_DIR / "features"
PREDICTIONS_DIR = OUTPUTS_DIR / "predictions"
METRICS_DIR = OUTPUTS_DIR / "metrics"
PLOTS_DIR = OUTPUTS_DIR / "plots"

for d in (DATA_DIR, OUTPUTS_DIR, CACHE_DIR, FEATURES_DIR, PREDICTIONS_DIR, METRICS_DIR, PLOTS_DIR):
    d.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class Config:
    # --- Reproducibility ---
    seed: int = 42

    # --- Model ---
    model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"
    dtype: str = "float16"  # used on GPU; falls back to float32 on CPU

    # --- Dataset ---
    dataset_name: str = "truthfulqa/truthful_qa"  # HF renamed from "truthful_qa"
    dataset_config: str = "generation"
    subset_size: int = 200  # fixed TruthfulQA subset, report Section 4.4

    # --- Generation: single-pass proposed detector (deterministic / greedy) ---
    single_pass_max_new_tokens: int = 64
    single_pass_do_sample: bool = False  # greedy -> reproducible single pass

    # --- Generation: 5-call self-consistency baseline (stochastic) ---
    self_consistency_n_samples: int = 5
    self_consistency_max_new_tokens: int = 64
    self_consistency_temperature: float = 0.7
    self_consistency_top_p: float = 0.9

    # --- Train/eval split ---
    train_fraction: float = 0.70  # report Section 4.4: 70/30 split

    # --- Automatic proxy labeling (reference-similarity judge) ---
    # See README "Ground-truth labeling" for justification: TruthfulQA does not
    # label free-form generations, and no GPT-judge / paid API is available.
    label_margin: float = 0.0  # min (correct_sim - incorrect_sim) to call "grounded"

    # --- Prompt template ---
    system_prompt: str = (
        "You are a careful factual question-answering assistant. "
        "Answer the question directly and concisely in one or two sentences. "
        "If you are not sure, give your best specific answer anyway."
    )


CFG = Config()


def set_all_seeds(seed: int = CFG.seed) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
