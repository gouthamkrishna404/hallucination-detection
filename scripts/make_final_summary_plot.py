"""Final presentation plot: original DA1 detector vs. the validated
improved detector (10-fold cross-validated AUROC, with bootstrap 95% CI
error bars) on TruthfulQA, for all 3 models -- makes the Qwen/Llama
improvement visually obvious while keeping the SmolLM2 null result
equally visible (same axes, same chance line, no cropping or rescaling
that would minimize it).

Usage:
    python scripts/make_final_summary_plot.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Values transcribed directly from comparison/truthfulqa_final_headtohead.json,
# results/*/truthfulqa/analysis/cascade_detector.json (verified in RESULTS.md Section 3).
MODELS = ["Qwen2.5-1.5B\n(cascade, 17-feat)", "Llama-3.2-1B\n(flat, 17-feat)", "SmolLM2-1.7B\n(best variant)"]
ORIGINAL_OOF = [0.398, 0.564, 0.438]
ORIGINAL_OOF_ERR = [[0.398 - 0.325, 0.564 - 0.482, 0.438 - 0.355], [0.476 - 0.398, 0.641 - 0.564, 0.519 - 0.438]]
IMPROVED_OOF = [0.598, 0.653, 0.485]
IMPROVED_OOF_ERR = [[0.598 - 0.516, 0.653 - 0.576, 0.485 - 0.408], [0.678 - 0.598, 0.727 - 0.653, 0.566 - 0.485]]
SIGNIFICANT = [True, True, False]


def main():
    x = np.arange(len(MODELS))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 6))
    b1 = ax.bar(x - width / 2, ORIGINAL_OOF, width, yerr=ORIGINAL_OOF_ERR, capsize=5,
                label="Original DA1 detector (3-feat)", color="#8c8c8c")
    b2 = ax.bar(x + width / 2, IMPROVED_OOF, width, yerr=IMPROVED_OOF_ERR, capsize=5,
                label="Improved detector (17-feat, validated)", color="#2b6cb0")

    for i, sig in enumerate(SIGNIFICANT):
        if sig:
            ax.annotate("significant\n(CI excludes\nchance)", xy=(x[i] + width / 2, IMPROVED_OOF[i] + IMPROVED_OOF_ERR[1][i] + 0.02),
                        ha="center", fontsize=8, color="#2b6cb0", fontweight="bold")
        else:
            ax.annotate("not\nsignificant", xy=(x[i] + width / 2, IMPROVED_OOF[i] + IMPROVED_OOF_ERR[1][i] + 0.02),
                        ha="center", fontsize=8, color="#8c1c1c", fontweight="bold")

    ax.axhline(0.5, linestyle="--", color="gray", linewidth=1, label="Chance (AUROC = 0.5)")
    ax.set_xticks(x)
    ax.set_xticklabels(MODELS)
    ax.set_ylabel("AUROC (10-fold out-of-fold, 95% bootstrap CI)")
    ax.set_title("TruthfulQA: Original vs. Improved Detector, All 3 Models\n(SmolLM2's null result shown on the same axis, not hidden)")
    ax.set_ylim(0.2, 0.85)
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()

    out_path = Path("comparison") / "truthfulqa_final_summary.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()
