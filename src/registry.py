"""Model and dataset registry: configs/models.yaml and configs/datasets.yaml."""
import yaml

from .config import PROJECT_ROOT

CONFIGS_DIR = PROJECT_ROOT / "configs"


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
