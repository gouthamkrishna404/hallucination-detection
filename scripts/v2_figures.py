"""Paper figures from analysis_v2/{dev,confirm}.json (no recomputation).

fig_v2_auroc.png       D0 and D2 AUROC with 95% CI, every confirm cell
fig_v2_error_type.png  D0 AUROC on imitative vs other errors, dev and confirm
fig_v2_difficulty.png  own D0 vs other models' D0 vs others-wrong count (confirm)

Colors: first three slots of the dataviz reference palette (validated all-pairs),
one per model, plus distinct markers so the figures survive greyscale printing.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
A = ROOT / "analysis_v2"
OUT = ROOT / "paper" / "figures"
MODELS = {"qwen2.5-1.5b-instruct": ("Qwen2.5-1.5B", "#2a78d6", "o"),
          "llama-3.2-1b-instruct": ("Llama-3.2-1B", "#eb6834", "s"),
          "smollm2-1.7b-instruct": ("SmolLM2-1.7B", "#1baf7a", "^")}
DS = {"truthfulqa": "TruthfulQA", "sciq": "SciQ"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#d9d8d4"

plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"], "font.size": 9,
                     "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.spines.top": False, "axes.spines.right": False})
conf = json.loads((A / "confirm.json").read_text())
dev = json.loads((A / "dev.json").read_text())


def chance(ax):
    ax.axvline(0.5, color=MUTED, lw=1, ls="--", zorder=0)
    ax.grid(axis="x", color=GRID, lw=0.6, zorder=0)


def fig_auroc():
    fig, ax = plt.subplots(figsize=(3.5, 3.0))
    cells = [(d, m) for d in DS for m in MODELS]
    for i, (d, m) in enumerate(cells):
        name, col, mk = MODELS[m]
        for k, off, filled in [("D0", -0.15, True), ("D2", 0.15, False)]:
            r = conf[f"{m}|{d}"]["detectors"][k]
            ax.errorbar(r["auroc"], i + off, xerr=[[r["auroc"] - r["ci"][0]], [r["ci"][1] - r["auroc"]]], fmt=mk,
                        color=col, mfc=col if filled else "white", mec=col, mew=1.3, ms=5, lw=1.3, capsize=0)
    chance(ax)
    ax.axhline(2.5, color=GRID, lw=0.8)
    ax.set_xlim(0.38, 0.8)
    ax.set_yticks(range(6), [f"{DS[d]} · {MODELS[m][0]}" for d, m in cells], fontsize=7.5)
    ax.invert_yaxis()
    ax.set_xlabel("AUROC on confirm split (95% CI)")
    ax.plot([], [], "o", color=MUTED, label="D0: −mean log-prob")
    ax.plot([], [], "o", mfc="white", mec=MUTED, label="D2: 3-feature LR, trained on dev")
    fig.legend(fontsize=7, frameon=False, loc="lower center", ncol=2, handletextpad=0.2, columnspacing=0.8)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(OUT / "fig_v2_auroc.png", dpi=300)


def fig_error_type():
    fig, axes = plt.subplots(1, 2, figsize=(3.5, 2.7), sharey=True)
    for ax, (split, src) in zip(axes, [("dev", dev), ("confirm", conf)]):
        for i, m in enumerate(MODELS):
            name, col, mk = MODELS[m]
            r = src[f"{m}|truthfulqa"]
            if split == "dev":
                e = r["by_error_type"]
                im, ot = e["imitative"]["mean_logprob"]["auroc"], e["other"]["mean_logprob"]["auroc"]
            else:
                im, ot = r["H2_error_type"]["auroc_imitative"], r["H2_error_type"]["auroc_other"]
            ax.plot([im, ot], [i, i], color=GRID, lw=2, zorder=1)
            ax.plot(im, i, mk, mfc="white", mec=col, mew=1.5, ms=6, zorder=2)
            ax.plot(ot, i, mk, color=col, ms=6, zorder=2)
        chance(ax)
        ax.set_title(f"TruthfulQA {split}", fontsize=9, color=INK)
        ax.set_xlim(0.3, 0.8)
    fig.supxlabel("AUROC of −mean log-prob", fontsize=9, y=0.16)
    axes[0].set_yticks(range(3), [MODELS[m][0] for m in MODELS], fontsize=8)
    axes[0].invert_yaxis()
    axes[1].plot([], [], "o", mfc="white", mec=MUTED, label="imitative errors")
    axes[1].plot([], [], "o", color=MUTED, label="other errors")
    fig.legend(fontsize=7, frameon=False, loc="lower center", ncol=2)
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    fig.savefig(OUT / "fig_v2_error_type.png", dpi=300)


def fig_difficulty():
    fig, axes = plt.subplots(1, 2, figsize=(3.5, 2.7), sharey=True)
    for ax, d in zip(axes, DS):
        h = conf["_H5_own_vs_other"][d]
        for i, m in enumerate(MODELS):
            name, col, mk = MODELS[m]
            others = [v["other_auroc"] for k, v in h.items() if k.startswith(f"{m}<-") and "others" not in k]
            own = conf[f"{m}|{d}"]["detectors"]["D0"]["auroc"]
            proxy = h[f"{m}<-others_wrong_count"]["auroc"]
            for o in others:
                ax.plot(o, i, mk, mfc="white", mec=col, mew=1.2, ms=5)
            ax.plot(own, i, mk, color=col, ms=6)
            ax.plot(proxy, i, "D", color=INK, ms=4)
        chance(ax)
        ax.set_title(DS[d], fontsize=9, color=INK)
        ax.set_xlim(0.38, 0.88)
    fig.supxlabel("AUROC on this model's errors", fontsize=9, y=0.16)
    axes[0].set_yticks(range(3), [MODELS[m][0] for m in MODELS], fontsize=8)
    axes[0].invert_yaxis()
    axes[1].plot([], [], "o", color=MUTED, label="own −mean log-prob")
    axes[1].plot([], [], "o", mfc="white", mec=MUTED, label="other model's")
    axes[1].plot([], [], "D", color=INK, ms=4, label="# other models wrong")
    fig.legend(fontsize=6.5, frameon=False, loc="lower center", ncol=3, handletextpad=0.2, columnspacing=0.8)
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    fig.savefig(OUT / "fig_v2_difficulty.png", dpi=300)


if __name__ == "__main__":
    fig_auroc()
    fig_error_type()
    fig_difficulty()
    print("wrote figures to", OUT)
