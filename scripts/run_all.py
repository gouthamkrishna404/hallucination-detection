"""Run the complete pipeline end-to-end: single-pass generation,
self-consistency generation, then training/evaluation.

Each stage is idempotent-ish (features are cached to CSV), so re-running
after an interruption just re-does the missing stage if you delete its
output file. To force a full re-run, delete outputs/features/*.csv first.
"""
import subprocess
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
PY = sys.executable


def run(script_name):
    print(f"\n{'='*70}\nRunning {script_name}\n{'='*70}")
    result = subprocess.run([PY, str(SCRIPTS_DIR / script_name)])
    if result.returncode != 0:
        print(f"FAILED: {script_name}")
        sys.exit(result.returncode)


if __name__ == "__main__":
    run("01_generate_single_pass.py")
    run("02_generate_self_consistency.py")
    run("03_train_and_evaluate.py")
    print("\nPipeline complete. See outputs/metrics/all_metrics.json for full results.")
