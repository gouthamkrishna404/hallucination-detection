"""Exploratory (not preregistered): what does the dev-trained D2 detector pick up?

On TruthfulQA its standardized weights are positive on mean log-prob AND on mean
entropy and negative on min log-prob in every cell, i.e. it does not score "low
confidence" but a contrast between features that normally move together. Checks, on
the confirm split, with every direction fixed on dev:

  weights     D2 standardized weights and dev feature correlations
  contrasts   AUROC of the two parameter-free contrasts the weights suggest:
              mean_logprob + mean_entropy  (chosen token likely, distribution spread)
              mean_logprob - min_logprob   (fluent overall, one improbable token)
  error_type  D0, D2 and the contrasts on imitative-vs-correct and other-vs-correct
  shape       error rate by D0 quintile (is risk monotone in confidence?)
  H2_gap      the preregistered imitative-minus-other gap, for D2 and mean_minus_min
  length      the contrasts within answer-length quintiles (are they length proxies?)

Usage: python scripts/v3_d2_mechanism.py  -> analysis_v3/d2_mechanism.json
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from src.features_v2 import ORIGINAL_THREE_FEATURES
from src.v2_stats import auroc, boot_ci, clf
from scripts.v2_analysis import load_cell
from scripts.v2_confirm import error_type_gap
from scripts.v2_labels import MODELS
from scripts.v3_audit import quintiles, stratified_auroc

ROOT = Path(__file__).resolve().parent.parent
CONTRASTS = {"mean_plus_entropy": lambda d: d["mean_logprob"] + d["mean_entropy"],
             "mean_minus_min": lambda d: d["mean_logprob"] - d["min_logprob"]}


def oriented(dev, conf, f):
    """Score oriented on dev (larger = riskier there), applied unchanged to confirm."""
    sign = 1.0 if auroc(dev["label"].to_numpy(), f(dev).to_numpy(float)) >= 0.5 else -1.0
    return sign, sign * f(conf).to_numpy(float)


def by_type(conf, s):
    t = conf["judge_type"].where(conf["label"] == 1, "correct").fillna("other").to_numpy()
    out = {}
    for k in ["imitative", "other"]:
        j = np.isin(t, ["correct", k])
        yk = (t[j] == k).astype(int)
        out[k] = {"auroc": auroc(yk, s[j]), "ci": boot_ci(yk, s[j]), "n": int(yk.sum())} if yk.sum() >= 10 else None
    return out


def cell(m, d):
    dev, conf = load_cell(m, d, "dev"), load_cell(m, d, "confirm")
    y = conf["label"].to_numpy()
    model = clf().fit(dev[ORIGINAL_THREE_FEATURES].to_numpy(float), dev["label"])
    scores = {"D0": -conf["mean_logprob"].to_numpy(float),
              "D2": model.predict_proba(conf[ORIGINAL_THREE_FEATURES].to_numpy(float))[:, 1]}
    signs = {}
    for k, f in CONTRASTS.items():
        signs[k], scores[k] = oriented(dev, conf, f)
    lenq = quintiles(conf["answer_length"].to_numpy())
    d0q = quintiles(conf["mean_logprob"].to_numpy())  # 0 = least confident fifth
    return {
        "weights": dict(zip(ORIGINAL_THREE_FEATURES, model[-1].coef_[0].round(3).tolist())),
        "dev_spearman": dev[ORIGINAL_THREE_FEATURES].corr("spearman").round(3).to_dict(),
        "contrast_sign_from_dev": signs,
        "auroc": {k: {"auroc": auroc(y, s), "ci": boot_ci(y, s)} for k, s in scores.items()},
        "error_type": {k: by_type(conf, s) for k, s in scores.items()} if "judge_type" in conf else None,
        "H2_gap": {k: error_type_gap(conf, scores[k]) for k in ["D2", "mean_minus_min"]} if d == "truthfulqa" else None,
        "error_rate_by_D0_quintile": [float(y[d0q == q].mean()) for q in range(5)],
        "length_quintile_auroc": {k: stratified_auroc(y, s, lenq) for k, s in scores.items()},
        "spearman_with_length": {k: float(pd.Series(s).corr(conf["answer_length"], method="spearman"))
                                 for k, s in scores.items()},
    }


def main():
    res = {f"{m}|{d}": cell(m, d) for d in ["truthfulqa", "sciq"] for m in MODELS}
    out = ROOT / "analysis_v3" / "d2_mechanism.json"
    out.write_text(json.dumps(res, indent=1))
    for k, r in res.items():
        a = {s: round(v["auroc"], 3) for s, v in r["auroc"].items()}
        print(k, a, "| by D0 quintile:", [round(x, 2) for x in r["error_rate_by_D0_quintile"]])
        if r["error_type"]:
            for s, t in r["error_type"].items():
                print(f"   {s:18s}", {k: None if v is None else (round(v["auroc"], 3), [round(c, 3) for c in v["ci"]])
                                        for k, v in t.items()})
        if r["H2_gap"]:
            print("   H2 gap", {k: (round(v["delta"], 3), [round(c, 3) for c in v["ci"]], v["p_two_sided"]) for k, v in r["H2_gap"].items()})
        print("   signs", r["contrast_sign_from_dev"], "len-strat", {k: round(v, 3) for k, v in r["length_quintile_auroc"].items()})
    print("wrote", out)


if __name__ == "__main__":
    main()
