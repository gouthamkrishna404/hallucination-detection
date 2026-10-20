"""v3 paper figures, drawn only from saved analysis JSON (no recomputation).

fig_v3_signals.png   imitative vs other TruthfulQA errors, AUROC of four signal families
fig_v3_robust.png    imitative-minus-other gap of D0 under every robustness variant (95% CI)
fig_v3_coverage.png  selective prediction: error rate of the answers kept vs coverage (D0)

Same palette, markers and type as scripts/v2_figures.py (which still draws the D0/D2 AUROC
and question-difficulty figures).
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from scripts.v2_figures import DS, GRID, INK, MODELS, MUTED, OUT, chance, plt

ROOT = Path(__file__).resolve().parent.parent
J = lambda p: json.loads((ROOT / p).read_text()) if (ROOT / p).exists() else {}
conf, audit, ext, extra = J("analysis_v2/confirm.json"), J("analysis_v3/audit.json"), \
    J("analysis_v3/extensions.json"), J("analysis_v3/extra.json")


def fig_signals():
    sig = [("D0", "−mean log-prob"), ("D6", "P(True)"), ("D7", "NLI sem. entropy"), ("D2", "3-feature LR")]
    fig, axes = plt.subplots(1, 3, figsize=(3.5, 2.3), sharey=True)
    for ax, m in zip(axes, MODELS):
        name, col, mk = MODELS[m]
        e = ext[f"{m}|truthfulqa"]["error_type"]
        for i, (k, _) in enumerate(sig):
            im, ot = e[k]["auroc_imitative"], e[k]["auroc_other"]
            ax.plot([im, ot], [i, i], color=GRID, lw=2, zorder=1)
            ax.plot(im, i, mk, mfc="white", mec=col, mew=1.4, ms=5.5, zorder=2)
            ax.plot(ot, i, mk, color=col, ms=5.5, zorder=2)
        chance(ax)
        ax.set_title(name, fontsize=8, color=INK)
        ax.set_xlim(0.3, 0.87)
        ax.set_xticks([0.4, 0.6, 0.8])
        ax.tick_params(labelsize=7)
    axes[0].set_yticks(range(len(sig)), [s for _, s in sig], fontsize=7.5)
    axes[0].invert_yaxis()
    fig.supxlabel("AUROC vs correct answers (TruthfulQA confirm)", fontsize=8, y=0.15)
    axes[1].plot([], [], "o", mfc="white", mec=MUTED, label="imitative errors")
    axes[1].plot([], [], "o", color=MUTED, label="other errors")
    fig.legend(fontsize=7, frameon=False, loc="lower center", ncol=2)
    fig.tight_layout(rect=(0, 0.12, 1, 1), w_pad=0.4)
    fig.savefig(OUT / "fig_v3_signals.png", dpi=300)


def variants(m):
    k = f"{m}|truthfulqa"
    v = [("Preregistered", conf[k]["H2_error_type"]),
         ("High-confidence labels", conf[k]["sensitivity_high_conf_only"]["H2_error_type"]),
         ("Length-matched", audit[k]["h2_controls"]["length_quintile"]),
         ("Affirmative answers", ext[k]["form"]["affirmative"]["error_type"]["D0"]),
         ("NLI error types", ext["_independent_h2"][m]["claude_labels_nli_types"])]
    ih = ext["_independent_h2"][m].get("truthjudge_labels_nli_types")
    if ih:
        v.append(("Truth-judge labels + NLI types", ih))
    wq = ext.get("_within_tqa", {}).get(m)
    if wq:
        v.append(("Within question (independent)", {"delta": wq["delta"], "ci": wq["delta_ci"]}))
    p2 = extra.get("prompt2", {}).get(m, {}).get("H2")
    if p2:
        v.append(("Neutral prompt (independent)", p2))
    return v


def fig_robust():
    rows = [n for n, _ in variants(next(iter(MODELS)))]
    fig, ax = plt.subplots(figsize=(3.5, 0.42 * len(rows) + 0.9))
    for j, m in enumerate(MODELS):
        name, col, mk = MODELS[m]
        for i, (_, r) in enumerate(variants(m)):
            y = i + (j - 1) * 0.22
            ax.errorbar(r["delta"], y, xerr=[[r["delta"] - r["ci"][0]], [r["ci"][1] - r["delta"]]], fmt=mk,
                        color=col, ms=4, lw=1.1, capsize=0)
    ax.axvline(0, color=MUTED, lw=1, ls="--", zorder=0)
    ax.grid(axis="x", color=GRID, lw=0.6, zorder=0)
    ax.set_yticks(range(len(rows)), rows, fontsize=7.5)
    ax.invert_yaxis()
    ax.set_xlabel("Δ AUROC, imitative − other errors (95% CI)", fontsize=8)
    for m in MODELS:
        ax.plot([], [], MODELS[m][2], color=MODELS[m][1], label=MODELS[m][0], ms=4)
    fig.legend(fontsize=7, frameon=False, loc="lower center", ncol=3, handletextpad=0.2, columnspacing=0.8)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(OUT / "fig_v3_robust.png", dpi=300)


def fig_coverage():
    fig, axes = plt.subplots(1, 2, figsize=(3.5, 2.3), sharey=False)
    for ax, d in zip(axes, DS):
        for m in MODELS:
            name, col, mk = MODELS[m]
            df = pd.read_csv(ROOT / "results_v2" / m / d / "confirm" / "features.csv").dropna(subset=["judge_label"])
            y = df["judge_label"].to_numpy()[np.argsort(-df["mean_logprob"].to_numpy(), kind="stable")]
            cov = np.arange(1, len(y) + 1) / len(y)
            risk = np.cumsum(y) / np.arange(1, len(y) + 1)
            keep = cov >= 0.2
            ax.plot(cov[keep], risk[keep], color=col, lw=1.3)
            ax.plot(1.0, y.mean(), mk, color=col, ms=4)
        ax.set_title(DS[d], fontsize=8, color=INK)
        ax.set_xlim(0.2, 1.02)
        ax.set_xticks([0.2, 0.6, 1.0])
        ax.grid(color=GRID, lw=0.6, zorder=0)
        ax.tick_params(labelsize=7)
    axes[0].set_ylabel("error rate of kept answers", fontsize=8)
    fig.supxlabel("coverage (most confident first, by −mean log-prob)", fontsize=8, y=0.15)
    for m in MODELS:
        axes[1].plot([], [], color=MODELS[m][1], label=MODELS[m][0])
    fig.legend(fontsize=7, frameon=False, loc="lower center", ncol=3, handletextpad=0.3, columnspacing=0.8)
    fig.tight_layout(rect=(0, 0.12, 1, 1), w_pad=0.6)
    fig.savefig(OUT / "fig_v3_coverage.png", dpi=300)


SCALE = [("llama-3.2-1b-instruct", "Llama-3.2-1B", "#eb6834", "s"), ("qwen2.5-1.5b-instruct", "Qwen2.5-1.5B", "#2a78d6", "o"),
         ("smollm2-1.7b-instruct", "SmolLM2-1.7B", "#1baf7a", "^"), ("qwen2.5-3b-instruct", "Qwen2.5-3B", "#2a78d6", "o"),
         ("qwen2.5-7b-instruct-bnb4", "Qwen2.5-7B (4-bit)", "#2a78d6", "o"),
         ("llama-3.1-8b-instruct-bnb4", "Llama-3.1-8B (4-bit)", "#eb6834", "s")]


def fig_scale():
    """Independent labels and types only: AUROC of D0 for imitative vs other errors, by model size."""
    rows = [(n, c, mk, extra.get("base", {}).get(k) or extra.get("scale", {}).get(k)) for k, n, c, mk in SCALE]
    rows = [r for r in rows if r[3]]
    fig, ax = plt.subplots(figsize=(3.5, 0.3 * len(rows) + 0.9))
    for i, (n, col, mk, r) in enumerate(rows):
        im, ot = r["H2"]["auroc_imitative"], r["H2"]["auroc_other"]
        ax.plot([im, ot], [i, i], color=GRID, lw=2, zorder=1)
        ax.plot(im, i, mk, mfc="white", mec=col, mew=1.4, ms=5.5, zorder=2)
        ax.plot(ot, i, mk, color=col, ms=5.5, zorder=2)
    chance(ax)
    ax.set_yticks(range(len(rows)), [r[0] for r in rows], fontsize=7.5)
    ax.invert_yaxis()
    ax.set_xlim(0.3, 0.75)
    ax.set_xlabel("AUROC of −mean log-prob vs correct answers\n(TruthfulQA confirm, truth-judge labels, NLI types)", fontsize=8)
    ax.plot([], [], "o", mfc="white", mec=MUTED, label="imitative errors")
    ax.plot([], [], "o", color=MUTED, label="other errors")
    fig.legend(fontsize=7, frameon=False, loc="lower center", ncol=2)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(OUT / "fig_v3_scale.png", dpi=300)


if __name__ == "__main__":
    fig_signals()
    fig_robust()
    fig_coverage()
    if extra.get("scale"):
        fig_scale()
    print("wrote v3 figures to", OUT)
