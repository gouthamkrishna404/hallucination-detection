"""Append-only experiment ledger: every generation/train/eval run appends
one JSON line to logs/experiment_log.jsonl with what was run, when, and
its headline result -- a plain-text audit trail across every
model/dataset/script combination, independent of the structured per-run
metrics.json files.
"""
import json
import platform
from datetime import datetime, timezone

from .registry import LOGS_DIR

LOG_PATH = LOGS_DIR / "experiment_log.jsonl"


def log_event(script: str, model: str, dataset: str, status: str, **extra):
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "script": script,
        "model": model,
        "dataset": dataset,
        "status": status,  # "started" | "completed" | "failed"
        "host": platform.node(),
        **extra,
    }
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, default=str) + "\n")
    return record
