"""Run one full (model, dataset) experiment end-to-end: single-pass
generation, self-consistency generation, training/evaluation, then
analysis. Skips a stage if its output already exists (pass --force to any
individual script directly if you need to regenerate one stage only).

Usage:
    python scripts/run_experiment.py --model llama-3.2-1b-instruct --dataset truthfulqa
"""
import argparse
import subprocess
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
PY = sys.executable


def run(script_name, extra_args):
    print(f"\n{'='*70}\nRunning {script_name} {' '.join(extra_args)}\n{'='*70}")
    result = subprocess.run([PY, str(SCRIPTS_DIR / script_name), *extra_args])
    if result.returncode != 0:
        print(f"FAILED: {script_name}")
        sys.exit(result.returncode)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--dataset", default="truthfulqa")
    ap.add_argument("--skip-analysis", action="store_true")
    args = ap.parse_args()

    common = ["--model", args.model, "--dataset", args.dataset]
    run("experiment_generate_single_pass.py", common)
    run("experiment_generate_self_consistency.py", common)
    run("experiment_train_evaluate.py", common)
    if not args.skip_analysis:
        run("experiment_analysis.py", common)

    print(f"\nExperiment complete: {args.model} / {args.dataset}")
    print(f"Results -> results/{args.model}/{args.dataset}/")


if __name__ == "__main__":
    main()
