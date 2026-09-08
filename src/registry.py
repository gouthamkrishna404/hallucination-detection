"""Model/dataset registry and per-(model,dataset) output path resolution.

Everything the expanded, multi-model/multi-dataset project writes to disk
lives under results/<model_key>/<dataset_key>/... so that runs for
different models or datasets can never collide or overwrite each other.
The original DA1 single-model/single-dataset pipeline (scripts/01-03,
outputs/) is untouched by any of this -- see README "Project layout".
"""
from dataclasses import dataclass
from pathlib import Path

import yaml

from .config import PROJECT_ROOT

CONFIGS_DIR = PROJECT_ROOT / "configs"
RESULTS_ROOT = PROJECT_ROOT / "results"
COMPARISON_DIR = PROJECT_ROOT / "comparison"
LOGS_DIR = PROJECT_ROOT / "logs"


def _load_yaml(name: str) -> dict:
    with open(CONFIGS_DIR / name, encoding="utf-8") as f:
        return yaml.safe_load(f)


MODEL_REGISTRY: dict = _load_yaml("models.yaml")["models"]
DATASET_REGISTRY: dict = _load_yaml("datasets.yaml")["datasets"]


def resolve_model(key: str) -> dict:
    if key not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model key '{key}'. Available: {sorted(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[key]


def resolve_dataset(key: str) -> dict:
    if key not in DATASET_REGISTRY:
        raise ValueError(f"Unknown dataset key '{key}'. Available: {sorted(DATASET_REGISTRY)}")
    return DATASET_REGISTRY[key]


@dataclass(frozen=True)
class RunPaths:
    model_key: str
    dataset_key: str

    @property
    def root(self) -> Path:
        return RESULTS_ROOT / self.model_key / self.dataset_key

    @property
    def features(self) -> Path:
        return self.root / "features"

    @property
    def predictions(self) -> Path:
        return self.root / "predictions"

    @property
    def metrics(self) -> Path:
        return self.root / "metrics"

    @property
    def plots(self) -> Path:
        return self.root / "plots"

    @property
    def models(self) -> Path:
        return self.root / "models"

    @property
    def analysis(self) -> Path:
        return self.root / "analysis"

    @property
    def data_cache(self) -> Path:
        return PROJECT_ROOT / "data" / self.dataset_key

    def ensure(self) -> "RunPaths":
        for d in (self.features, self.predictions, self.metrics, self.plots, self.models, self.analysis, self.data_cache):
            d.mkdir(parents=True, exist_ok=True)
        return self


def get_paths(model_key: str, dataset_key: str) -> RunPaths:
    resolve_model(model_key)  # validates key, raises helpfully otherwise
    resolve_dataset(dataset_key)
    return RunPaths(model_key=model_key, dataset_key=dataset_key).ensure()


for d in (RESULTS_ROOT, COMPARISON_DIR, LOGS_DIR):
    d.mkdir(parents=True, exist_ok=True)
