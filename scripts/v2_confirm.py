"""Confirmatory evaluation on the held-out confirm split, exactly as fixed in
PREREGISTRATION.md. Nothing here is tuned: every detector is either a fixed score
with a pre-specified sign or a logistic regression trained on the dev split of the
same model x dataset cell and frozen before it sees a confirm item.

Endpoints (see PREREGISTRATION.md for the hypotheses they test):
  primary     D0 = -mean_logprob, two-sided permutation test of AUROC = 0.5 per
              cell, Holm across the 6 cells
  secondary   D1 = -min_logprob, D2 = 3-feature LR (dev-trained), D3 = 17-feature
              LR (dev-trained), D4 = -sc_greedy_support, D5 = sc_semantic_entropy;
              paired deltas D_k - D0; Holm across cells within each family
  H2          AUROC of D0 for imitative-vs-correct minus other-vs-correct
  H3          share of wrong answers at least as confident (D0) as the median
              correct answer
  H5          own vs other model's D0 on the same questions and labels (difficulty test)
  controls    question-only and length-only scores, and whether D0 adds to them
  sensitivity the same endpoints with judge conf = low / refusal items dropped
  within_q    SciQ sampled answers, containment labels, within-question AUROC

Usage:
    python scripts/v2_confirm.py
    python scripts/v2_confirm.py --dry-run   # code check only: tests on dev (in-sample, meaningless numbers)
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.features_v2 import ALL_RICH_FEATURE_NAMES, ORIGINAL_THREE_FEATURES
from src.v2_stats import SEED, auroc, boot_ci, clf, holm, oof, paired_delta
from scripts.v2_analysis import OUT, RISK_SIGN, load_cell, within_question
from scripts.v2_labels import MODELS

DATASETS = ["truthfulqa", "sciq"]
FIXED = {"D0": "mean_logprob", "D1": "min_logprob", "D4": "sc_greedy_support", "D5": "sc_semantic_entropy"}
FITTED = {"D2": ORIGINAL_THREE_FEATURES, "D3": ALL_RICH_FEATURE_NAMES}
N_PERM, N_BOOT = 10000, 2000


def perm_p_two_sided(y, s, n=N_PERM, seed=SEED):
    y, s = np.asarray(y), np.asarray(s)
    obs = abs(roc_auc_score(y, s) - 0.5)
    rng = np.random.RandomState(seed)
    null = np.array([abs(roc_auc_score(rng.permutation(y), s) - 0.5) for _ in range(n)])
    return float((np.sum(null >= obs) + 1) / (n + 1))


def summary(y, s):
    return {"auroc": auroc(y, s), "ci": boot_ci(y, s, N_BOOT), "p_two_sided": perm_p_two_sided(y, s),
            "n": int(len(y)), "n_pos": int(np.sum(y))}


def scores(dev, conf):
    """Risk score of every detector on the confirm items; fitted ones trained on dev only."""
    s = {k: RISK_SIGN[f] * conf[f].to_numpy(float) for k, f in FIXED.items()}
    for k, feats in FITTED.items():
        m = clf().fit(dev[feats].to_numpy(float), dev["label"].to_numpy())
        s[k] = m.predict_proba(conf[feats].to_numpy(float))[:, 1]
    return s


def error_type_gap(df, s, n=N_BOOT, seed=SEED):
    """AUROC(imitative vs correct) - AUROC(other vs correct), stratified bootstrap."""
    t = df["judge_type"].where(df["label"] == 1, "correct").fillna("other").to_numpy()
    idx = {k: np.where(t == k)[0] for k in ["correct", "imitative", "other"]}
    if min(len(idx["imitative"]), len(idx["other"])) < 10:
        return {"n_imitative": int(len(idx["imitative"])), "n_other": int(len(idx["other"]))}

    def a(c, w):
        return roc_auc_score(np.r_[np.zeros(len(c)), np.ones(len(w))], np.r_[s[c], s[w]])

    obs_i, obs_o = a(idx["correct"], idx["imitative"]), a(idx["correct"], idx["other"])
    rng = np.random.RandomState(seed)
    d = np.array([a(*(rng.choice(idx[k], len(idx[k])) for k in ["correct", "imitative"])) -
                  a(*(rng.choice(idx[k], len(idx[k])) for k in ["correct", "other"])) for _ in range(n)])
    return {"n_imitative": int(len(idx["imitative"])), "n_other": int(len(idx["other"])),
            "auroc_imitative": obs_i, "auroc_other": obs_o, "delta": obs_i - obs_o,
            "ci": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
            "p_two_sided": float(min(1.0, max(2 * min((d <= 0).mean(), (d >= 0).mean()), 1 / n)))}


def confident_wrong(y, s, n=N_BOOT, seed=SEED):
    """Share of wrong answers whose risk is at or below the median correct answer's."""
    thr = np.median(s[y == 0])
    w = s[y == 1] <= thr
    rng = np.random.RandomState(seed)
    bs = [np.mean(rng.choice(w, len(w))) for _ in range(n)]
    return {"share": float(w.mean()), "ci": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
            "null": 0.5, "n_wrong": int(len(w))}


def controls(conf, s0):
    y = conf["label"].to_numpy()
    q = conf[["question_mean_logprob", "question_n_tokens"]].to_numpy(float)
    ql = np.c_[q, conf["answer_length"].to_numpy(float)]
    p_q = oof(ql, y)
    p_qd = oof(np.c_[ql, conf["mean_logprob"].to_numpy(float)], y)
    return {"question_mean_logprob": summary(y, RISK_SIGN["question_mean_logprob"] * conf["question_mean_logprob"]),
            "answer_length": summary(y, conf["answer_length"].to_numpy(float)),
            "controls_oof_auroc": auroc(y, p_q), "controls_plus_D0_oof_auroc": auroc(y, p_qd),
            "D0_added_to_controls": paired_delta(y, p_qd, p_q, N_BOOT),
            "D0_vs_question_only": paired_delta(y, s0, RISK_SIGN["question_mean_logprob"] *
                                                conf["question_mean_logprob"].to_numpy(float), N_BOOT)}


def evaluate(dev, conf):
    y = conf["label"].to_numpy()
    s = scores(dev, conf)
    r = {"n": int(len(y)), "n_pos": int(y.sum()), "base_rate": float(y.mean()),
         "detectors": {k: summary(y, v) for k, v in s.items()},
         "paired_vs_D0": {k: paired_delta(y, v, s["D0"], N_BOOT) for k, v in s.items() if k != "D0"},
         "D3_oof_within_confirm": auroc(y, oof(conf[ALL_RICH_FEATURE_NAMES].to_numpy(float), y)),
         "H3_confident_wrong": confident_wrong(y, s["D0"])}
    if "judge_type" in conf:
        r["H2_error_type"] = error_type_gap(conf, s["D0"])
    r["controls"] = controls(conf, s["D0"])
    return r


def own_vs_other(dataset, split):
    """H5: does model B's own D0 predict B's errors better than model A's D0 on the
    same questions? If not, D0 is measuring question difficulty, not the answer."""
    cells = {m: load_cell(m, dataset, split).set_index("qid") for m in MODELS}
    out = {}
    for b, db in cells.items():
        for a, da in cells.items():
            if a == b:
                continue
            q = db.index.intersection(da.index)
            y = db.loc[q, "label"].to_numpy()
            out[f"{b}<-{a}"] = {"own_auroc": auroc(y, -db.loc[q, "mean_logprob"].to_numpy()),
                                "other_auroc": auroc(y, -da.loc[q, "mean_logprob"].to_numpy()),
                                **paired_delta(y, -db.loc[q, "mean_logprob"].to_numpy(),
                                               -da.loc[q, "mean_logprob"].to_numpy(), N_BOOT)}
    for k, p in zip(out, holm([v["p_two_sided"] for v in out.values()])):
        out[k]["p_holm"] = p
    # a difficulty proxy that uses no uncertainty at all: how many other models got it wrong
    for b, db in cells.items():
        q = db.index
        others = sum(cells[a].reindex(q)["label"].fillna(0) for a in cells if a != b)
        out[f"{b}<-others_wrong_count"] = summary(db["label"].to_numpy(), others.to_numpy(float))
    return out


def add_holm(res, path, key="p_two_sided"):
    """Holm across cells for one endpoint; path walks into each cell's result dict."""
    cells = [c for c in res if not c.startswith("_")]
    got = []
    for c in cells:
        d = res[c]
        for k in path:
            d = d.get(k, {}) if isinstance(d, dict) else {}
        if key in d:
            got.append((c, d))
    for (c, d), p in zip(got, holm([d[key] for _, d in got])):
        d["p_holm"] = p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    test_split = "dev" if ap.parse_args().dry_run else "confirm"
    res = {}
    for d in DATASETS:
        for m in MODELS:
            dev, conf = load_cell(m, d, "dev"), load_cell(m, d, test_split)
            key = f"{m}|{d}"
            print(f"== {key}: dev n={len(dev)} confirm n={len(conf)} pos={int(conf.label.sum())}", flush=True)
            res[key] = evaluate(dev, conf)
            keep = lambda x: x[~x["judge_conf"].isin(["low", "refusal"])].reset_index(drop=True)
            res[key]["sensitivity_high_conf_only"] = evaluate(keep(dev), keep(conf))
            res[key]["within_question"] = within_question(m, d, test_split)
    for k in ["D0", "D1", "D2", "D3", "D4", "D5"]:
        add_holm(res, ["detectors", k])
    for k in ["D1", "D2", "D3", "D4", "D5"]:
        add_holm(res, ["paired_vs_D0", k])
    add_holm(res, ["H2_error_type"])
    res["_H5_own_vs_other"] = {d: own_vs_other(d, test_split) for d in DATASETS}
    OUT.mkdir(exist_ok=True)
    out = OUT / ("confirm_dryrun.json" if test_split == "dev" else "confirm.json")
    out.write_text(json.dumps(res, indent=1))
    print("wrote", out)


if __name__ == "__main__":
    main()
