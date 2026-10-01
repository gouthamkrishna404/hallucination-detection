"""Figures for the IEEE paper (paper/figures/), sized for one 3.38-inch column.

Every plotted value is read from saved outputs (results/, comparison/) or
recomputed from the saved feature tables with the same out-of-fold procedure
used in the results -- nothing is hard-coded.

Feature-source policy matches scripts/build_final_results_table.py: Qwen uses
the decontaminated features, Llama and SmolLM2 use their originals.

Usage:
    python scripts/make_paper_figures.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from sklearn.metrics import roc_curve

from src.features_v2 import ORIGINAL_THREE_FEATURES
from scripts.experiment_clean_validation import oof_flat

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "paper" / "figures"
MODELS = [("qwen2.5-1.5b-instruct", "Qwen2.5-1.5B", "clean"),
          ("llama-3.2-1b-instruct", "Llama-3.2-1B", "rich"),
          ("smollm2-1.7b-instruct", "SmolLM2-1.7B", "rich")]
COLORS = ["#1f4e79", "#c55a11", "#548235"]
W = 3.38
REF_SEED = 42

plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
                     "font.size": 7, "mathtext.fontset": "stix", "axes.linewidth": 0.6,
                     "xtick.major.width": 0.6, "ytick.major.width": 0.6, "legend.frameon": False})


def features(key, ds, src):
    return pd.read_csv(ROOT / f"results/{key}/{ds}/features/single_pass_features_{src}.csv")


def load(path):
    p = ROOT / path
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def fig_pipeline():
    fig, ax = plt.subplots(figsize=(W, 2.15))
    ax.set_xlim(0, 10); ax.set_ylim(0, 6.3); ax.axis("off")
    bw, bh, xs = 2.8, 1.45, [0.1, 3.6, 7.1]

    def box(col, y, text, fc):
        ax.add_patch(FancyBboxPatch((xs[col], y), bw, bh, boxstyle="round,pad=0.03,rounding_size=0.15",
                                    fc=fc, ec="black", lw=0.6))
        ax.text(xs[col] + bw / 2, y + bh / 2, text, ha="center", va="center", fontsize=6.0, linespacing=1.2)

    def arrow(x1, y1, x2, y2):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=6, lw=0.6))

    y1, y2, y3 = 4.7, 2.55, 0.25
    box(0, y1, "Factual question\n(TruthfulQA / SciQ)", "#f2f2f2")
    box(1, y1, "LLM: 1 greedy call\n($\\leq$64 new tokens)", "#dae3f3")
    box(2, y1, "Token log-prob $\\ell_t$\nand entropy $H_t$", "#dae3f3")
    box(2, y2, "Features\n(3 original / 17 rich)", "#dae3f3")
    box(1, y2, "Logistic regression\n(flat or cascade)", "#e2f0d9")
    box(0, y2, "$P$(hallucinated)\nPASS/WARN/FLAG", "#dae3f3")
    box(0, y3, "Baseline: 5 samples\n($T$=0.7, top-$p$=0.9)", "#fbe5d6")
    box(1, y3, "Lexical agreement or\nsemantic entropy", "#fbe5d6")
    box(2, y3, "Logistic regression\n$P$(hallucinated)", "#fbe5d6")
    mid = bh / 2
    arrow(xs[0] + bw, y1 + mid, xs[1], y1 + mid)
    arrow(xs[1] + bw, y1 + mid, xs[2], y1 + mid)
    arrow(xs[2] + bw / 2, y1, xs[2] + bw / 2, y2 + bh)
    arrow(xs[2], y2 + mid, xs[1] + bw, y2 + mid)
    arrow(xs[1], y2 + mid, xs[0] + bw, y2 + mid)
    arrow(xs[0] + bw, y3 + mid, xs[1], y3 + mid)
    arrow(xs[1] + bw, y3 + mid, xs[2], y3 + mid)
    fig.savefig(OUT / "fig1_pipeline.png", dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def fig_roc():
    """Out-of-fold ROC of the original 3-feature detector (reference CV seed)."""
    fig, axes = plt.subplots(1, 2, figsize=(W, 1.75), sharey=True)
    for ax, (ds, title) in zip(axes, [("truthfulqa", "TruthfulQA"), ("sciq", "SciQ")]):
        for (key, name, src), c in zip(MODELS, COLORS):
            df = features(key, ds, src)
            y, p = oof_flat(df, ORIGINAL_THREE_FEATURES, seed=REF_SEED)
            fpr, tpr, _ = roc_curve(y, p)
            from sklearn.metrics import roc_auc_score
            ax.plot(fpr, tpr, color=c, lw=0.9, label=f"{name.split('-')[0]} ({roc_auc_score(y, p):.2f})")
        ax.plot([0, 1], [0, 1], "--", color="gray", lw=0.6)
        ax.set_title(title, fontsize=7.5)
        ax.set_xlabel("False positive rate")
        ax.legend(fontsize=5.8, loc="lower right", handlelength=1.2)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    axes[0].set_ylabel("True positive rate")
    fig.tight_layout(pad=0.3, w_pad=0.6)
    fig.savefig(OUT / "fig2_roc_truthfulqa_vs_sciq.png", dpi=300)
    plt.close(fig)


def fig_confidently_wrong():
    fig, ax = plt.subplots(figsize=(W, 1.6))
    x = range(len(MODELS))
    for i, (ds, label, fc) in enumerate([("truthfulqa", "TruthfulQA", "#1f4e79"), ("sciq", "SciQ", "#bdd7ee")]):
        vals = []
        for key, _, src in MODELS:
            q = load(f"results/{key}/{ds}/analysis/quadrant_analysis_{src}.json") or \
                load(f"results/{key}/{ds}/analysis/quadrant_analysis.json")
            c = q["quadrant_counts"]
            wc, wu = c["wrong_confident"]["n"], c["wrong_uncertain"]["n"]
            vals.append(100 * wc / (wc + wu))
        bars = ax.bar([xi + (i - 0.5) * 0.36 for xi in x], vals, 0.36, label=label,
                      color=fc, edgecolor="black", lw=0.5)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 1, f"{v:.1f}", ha="center", fontsize=6)
    ax.set_xticks(list(x)); ax.set_xticklabels([n for _, n, _ in MODELS])
    ax.set_ylabel("Wrong answers that are\nconfident (%)")
    ax.set_ylim(0, 65)
    ax.legend(fontsize=6.5, loc="upper left", ncol=2)
    fig.tight_layout(pad=0.3)
    fig.savefig(OUT / "fig3_confidently_wrong.png", dpi=300)
    plt.close(fig)


def fig_final_validation():
    """TruthfulQA: original 3-feature vs best engineered detector, 5-fold OOF,
    under both proxy labelers. Stars mark Bonferroni-corrected significance."""
    res = load("comparison/final_results.json")
    alpha = 0.05 / 8
    series = [("detectors_tokenf1", "flat__original_3feat", "Original 3-feat (token-F1)", "#d9d9d9"),
              ("detectors_tokenf1", "flat__all_17feat", "17-feat (token-F1)", "#9dc3e6"),
              ("detectors_semantic", "flat__original_3feat", "Original 3-feat (embedding)", "#7f7f7f"),
              ("detectors_semantic", "flat__all_17feat", "17-feat (embedding)", "#1f4e79")]

    fig, ax = plt.subplots(figsize=(W, 2.2))
    width = 0.19
    for gi, (key, name, _src) in enumerate(MODELS):
        cell = res["cells"][f"{key}|truthfulqa"]
        for si, (group, det, label, color) in enumerate(series):
            r = cell.get(group, {}).get(det)
            if r is None:
                continue
            x = gi + (si - 1.5) * width
            pe = r["auroc_reference_seed"]
            lo, hi = r["bootstrap_ci"]
            ax.bar(x, pe, width, color=color, edgecolor="black", lw=0.5,
                   label=label if gi == 0 else None)
            ax.errorbar(x, pe, yerr=[[pe - lo], [hi - pe]], color="black", lw=0.6, capsize=1.5)
            if r["permutation_p_value"] < alpha:
                ax.text(x, hi + 0.012, "*", ha="center", fontsize=8)
    ax.axhline(0.5, ls="--", color="gray", lw=0.6)
    ax.set_xticks(range(len(MODELS)))
    ax.set_xticklabels([n for _, n, _ in MODELS])
    ax.set_ylabel("AUROC (5-fold OOF, 95% CI)")
    ax.set_ylim(0.3, 0.85)
    ax.legend(fontsize=5.4, loc="upper left", ncol=2, handlelength=1.0, columnspacing=0.8)
    fig.tight_layout(pad=0.3)
    fig.savefig(OUT / "fig4_final_validation.png", dpi=300)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    fig_pipeline()
    fig_roc()
    fig_confidently_wrong()
    fig_final_validation()
    print(f"Saved 4 figures -> {OUT}")


if __name__ == "__main__":
    main()
