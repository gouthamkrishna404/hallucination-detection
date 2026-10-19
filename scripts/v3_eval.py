"""Exploratory (post-preregistration) evaluation of the v3 extensions on the confirm split.

Inputs are the frozen v2 cells plus whichever v3 score files exist:
  results_v3/<m>/<d>/<s>/ptrue.csv        P(True) self-evaluation   -> D6
  results_v3/<m>/<d>/<s>/nli_sc.csv       NLI semantic entropy      -> D7 discrete, D8 weighted,
                                                                       D9 1 - NLI greedy support
  results_v3/<m>/<d>/<s>/truthjudge.csv   independent truth-judge labels

Sections (analysis_v3/extensions.json):
  detectors    AUROC/CI/permutation p of D0, D2 and the new detectors per cell, Holm
               across the 6 cells per detector; paired delta vs D0
  error_type   imitative-vs-correct and other-vs-correct AUROC for each detector
               (TruthfulQA; judge types)
  combo        dev-trained LR on D2 features + P(True), paired vs D2
  form         TruthfulQA: detectors and the imitative gap within affirmative answers
               (no negation cue, not a refusal), since correct answers are often negations
  transfer     D2 trained on one cell's dev split, scored on another cell's confirm
               split (cross-dataset and cross-model)
  labels       truth-judge vs Claude-judge agreement, and the main endpoints recomputed
               (a) with truth-judge labels and (b) on items where both judges agree

Usage:
    python scripts/v3_eval.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

from src.features_v2 import ORIGINAL_THREE_FEATURES
from src.v2_stats import auroc, clf, holm, paired_delta
from scripts.v2_analysis import load_cell
from scripts.v2_confirm import N_BOOT, confident_wrong, error_type_gap, scores, summary
from scripts.v2_labels import MODELS
from scripts.v3_audit import OUT
from scripts.v3_ptrue import V3_ROOT

DATASETS = ["truthfulqa", "sciq"]
EXTRA = {"D6": ("ptrue.csv", "ptrue_risk", +1), "D7": ("nli_sc.csv", "nli_se_discrete", +1),
         "D8": ("nli_sc.csv", "nli_se_weighted", +1), "D9": ("nli_sc.csv", "nli_greedy_support", -1)}
NAMES = {"D0": "-mean log-prob", "D2": "3-feature LR (dev)", "D6": "P(True) self-evaluation",
         "D7": "NLI semantic entropy (discrete)", "D8": "NLI semantic entropy (weighted)",
         "D9": "1 - NLI greedy support"}


def load(m, d, s, label="judge"):
    """v2 cell + any v3 columns. label='judge' keeps v2 labels; 'tj' swaps in the truth-judge."""
    df = load_cell(m, d, s)
    for f in {"ptrue.csv", "nli_sc.csv", "truthjudge.csv"}:
        p = V3_ROOT / m / d / s / f
        if p.exists():
            df = df.merge(pd.read_csv(p), on="qid", how="left")
    if label == "tj":
        df = df.assign(judge_label=df["label"], label=df["tj_label"])
    return df


def all_scores(dev, conf):
    s = {k: v for k, v in scores(dev, conf).items() if k in ("D0", "D2")}
    for k, (_, col, sign) in EXTRA.items():
        if col in conf:
            s[k] = sign * conf[col].to_numpy(float)
    return s


def detectors(res):
    for d in DATASETS:
        for m in MODELS:
            dev, conf = load(m, d, "dev"), load(m, d, "confirm")
            y = conf["label"].to_numpy()
            s = all_scores(dev, conf)
            r = res.setdefault(f"{m}|{d}", {})
            r["detectors"] = {k: summary(y, v) for k, v in s.items()}
            r["paired_vs_D0"] = {k: paired_delta(y, v, s["D0"], N_BOOT) for k, v in s.items() if k != "D0"}
            if d == "truthfulqa":
                r["error_type"] = {k: error_type_gap(conf, v) for k, v in s.items()}
            if "ptrue_risk" in conf:
                f = ORIGINAL_THREE_FEATURES + ["ptrue_risk"]
                p = clf().fit(dev[f].to_numpy(float), dev["label"]).predict_proba(conf[f].to_numpy(float))[:, 1]
                r["combo_D2_plus_ptrue"] = {**summary(y, p), "vs_D2": paired_delta(y, p, s["D2"], N_BOOT)}
    for k in NAMES:
        for path in (["detectors", k], ["paired_vs_D0", k]):
            got = []
            for c in res:
                x = res[c].get(path[0], {}).get(path[1])
                if x and "p_two_sided" in x:
                    got.append(x)
            for x, p in zip(got, holm([x["p_two_sided"] for x in got])):
                x["p_holm"] = p


# Answer form. Correct TruthfulQA answers are often debunkings ("No, ...", "There is no ..."),
# imitative errors are mostly affirmative. A score can separate them by form alone, so H2 and
# P(True) are re-tested within affirmative answers (no negation cue, not a refusal).
NEG = r"\b(?:no|not|never|nothing|none|nobody|neither|nor|cannot|myth|misconception|false|untrue)\b|n't\b"


def form(res):
    for m in MODELS:
        dev, conf = load(m, "truthfulqa", "dev"), load(m, "truthfulqa", "confirm")
        s = all_scores(dev, conf)
        neg = conf["answer"].fillna("").str.contains(NEG, case=False, regex=True)
        refusal = conf["judge_conf"].eq("refusal")
        t = conf["judge_type"].where(conf["label"] == 1, "correct").fillna("other")
        aff = (~neg & ~refusal).to_numpy()
        sub = conf[aff].reset_index(drop=True)
        res[f"{m}|truthfulqa"]["form"] = {
            "negated_share_by_type": neg.groupby(t).mean().to_dict(), "n_refusal": int(refusal.sum()),
            "affirmative": {"n": int(aff.sum()),
                            "detectors": {k: summary(sub["label"].to_numpy(), v[aff]) for k, v in s.items()},
                            "error_type": {k: error_type_gap(sub, v[aff]) for k, v in s.items()}},
            "negated_auroc": {k: auroc(conf["label"][neg & ~refusal], v[(neg & ~refusal).to_numpy()])
                              for k, v in s.items()}}


def transfer():
    """D2 fitted on a source cell's dev split, scored on a target cell's confirm split."""
    out = {}
    cells = {(m, d): (load(m, d, "dev"), load(m, d, "confirm")) for m in MODELS for d in DATASETS}
    f = ORIGINAL_THREE_FEATURES
    for (ms, ds), (dev, _) in cells.items():
        model = clf().fit(dev[f].to_numpy(float), dev["label"])
        for (mt, dt), (_, conf) in cells.items():
            out[f"{ms}|{ds} -> {mt}|{dt}"] = auroc(conf["label"], model.predict_proba(conf[f].to_numpy(float))[:, 1])
    return out


def labels():
    out = {}
    for d in DATASETS:
        for m in MODELS:
            if not (V3_ROOT / m / d / "confirm" / "truthjudge.csv").exists():
                continue
            key = f"{m}|{d}"
            r = out.setdefault(key, {})
            for s in ["dev", "confirm"]:
                df = load(m, d, s)
                a, b = df["label"].to_numpy(), df["tj_label"].to_numpy()
                r[f"agreement_{s}"] = {"n": int(len(a)), "agree": float((a == b).mean()),
                                       "kappa": float(cohen_kappa_score(a, b)), "rate_claude": float(a.mean()),
                                       "rate_truthjudge": float(b.mean()),
                                       "auroc_tj_p_false_vs_claude": auroc(a, 1 - df["tj_p_true"].to_numpy())}
                if "containment_label" in df:
                    c = df[~df["containment_ambiguous"].astype(bool)]
                    r[f"agreement_{s}"]["agree_tj_vs_containment"] = float(
                        (c["tj_label"] == c["containment_label"]).mean())
            if d != "truthfulqa":
                continue  # the truth-judge is out of distribution on SciQ: agreement only
            for name, (dev, conf) in {
                    "truthjudge_labels": (load(m, d, "dev", "tj"), load(m, d, "confirm", "tj")),
                    "consensus_items": tuple(x[x["label"] == x["tj_label"]].reset_index(drop=True)
                                             for x in (load(m, d, "dev"), load(m, d, "confirm")))}.items():
                y = conf["label"].to_numpy()
                s = all_scores(dev, conf)
                e = {"n": int(len(y)), "base_rate": float(y.mean()),
                     "detectors": {k: summary(y, v) for k, v in s.items()},
                     "D2_minus_D0": paired_delta(y, s["D2"], s["D0"], N_BOOT),
                     "confident_wrong": confident_wrong(y, s["D0"])}
                # error types come from the Claude judge; under truth-judge labels an item keeps its
                # type only if the Claude judge also called it wrong, otherwise it is dropped
                t = conf["judge_type"].where((conf["label"] == 1) & (conf["judge_label"] == 1)
                                             if "judge_label" in conf else conf["label"] == 1)
                keep = (conf["label"] == 0) | t.notna()
                e["H2_error_type"] = error_type_gap(conf[keep].reset_index(drop=True), s["D0"][keep.to_numpy()])
                r[name] = e
    for name in ["truthjudge_labels", "consensus_items"]:
        got = [out[c][name]["H2_error_type"] for c in out if name in out[c]
               and "p_two_sided" in out[c][name]["H2_error_type"]]
        for x, p in zip(got, holm([x["p_two_sided"] for x in got])):
            x["p_holm"] = p
        got = [out[c][name]["detectors"]["D0"] for c in out if name in out[c]]
        for x, p in zip(got, holm([x["p_two_sided"] for x in got])):
            x["p_holm"] = p
    return out


def main():
    res = {}
    detectors(res)
    form(res)
    res["_transfer_D2"] = transfer()
    res["_labels"] = labels()
    OUT.mkdir(exist_ok=True)
    (OUT / "extensions.json").write_text(json.dumps(res, indent=1, default=float))
    print("wrote", OUT / "extensions.json")


if __name__ == "__main__":
    main()
