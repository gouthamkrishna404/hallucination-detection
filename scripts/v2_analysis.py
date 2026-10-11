"""Exploratory analysis of the clean-protocol (v2) cells.

Run on the development split (the original 200 questions), this is where every
modelling decision is made; the confirmation split is evaluated only by
scripts/v2_confirm.py under PREREGISTRATION.md. Everything here is OOF / CV
within one cell, so no number from this script is a held-out result.

Primary labels: judge_label (labels/JUDGE_RUBRIC.md) for both datasets. SciQ
containment is kept as a secondary label: it never called a wrong answer correct
but marked 23/100 correct paraphrases as hallucinated in the judge audit.

Sections (keys of the output JSON, one per cell):
  baselines     fixed single-score detectors (no fitting), with pre-specified risk
                direction, AUROC + bootstrap CI + permutation p
  fitted        logistic regression on feature groups: OOF AUROC, CI, permutation
                p (refit under shuffled labels), 20x repeated CV
  nested        honest estimate of "pick the best feature group by inner CV"
  redundancy    correlation structure / effective dimensionality of the 17 features
  controls      length-only, question-only, TF-IDF of question / answer text, and
                whether uncertainty adds to length+question controls
  by_error_type detection of imitative vs other errors (TruthfulQA judge types,
                SciQ distractor vs other wrong answer)
  quadrants     how many wrong answers are as confident as a typical correct one
  calibration   ECE / Brier of the OOF 3-feature model vs the base-rate Brier
  operating     recall and precision at fixed false-positive rates
Cross-cell sections (in _cross): cross-model difficulty transfer, within-question
discrimination on sampled answers.

Usage:
    python scripts/v2_analysis.py --split dev
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_curve
from sklearn.model_selection import StratifiedKFold

from src.config import PROJECT_ROOT
from src.features_v2 import ALL_RICH_FEATURE_NAMES, ORIGINAL_THREE_FEATURES, extract_rich_features
from src.v2_stats import (auroc, boot_ci, clf, oof, paired_delta, perm_p_fitted, perm_p_fixed, repeated_cv)
from scripts.v2_generate import V2_ROOT
from scripts.v2_labels import MODELS, read_jsonl

OUT = PROJECT_ROOT / "analysis_v2"
# SciQ: judge, not containment -- containment marked 23/100 correct paraphrases as hallucinated (judge audit).
PRIMARY = {"truthfulqa": "judge_label", "sciq": "judge_label"}

# Risk direction fixed a priori: +1 means "larger value -> more likely hallucinated".
RISK_SIGN = {
    "mean_logprob": -1, "median_logprob": -1, "min_logprob": -1, "logprob_p10": -1, "logprob_p25": -1,
    "std_logprob": +1, "frac_low_confidence": +1, "mean_entropy": +1, "max_entropy": +1, "std_entropy": +1,
    "mean_margin": -1, "min_margin": -1, "first_half_mean_logprob": -1, "second_half_mean_logprob": -1,
    "logprob_delta_second_minus_first": -1, "logprob_trend_slope": -1, "answer_length": +1,
    "question_mean_logprob": -1, "question_n_tokens": +1,
    "sc_lexical_agreement": -1, "sc_semantic_entropy": +1, "sc_greedy_support": -1,
}
GROUPS = {
    "mean_logprob_only": ["mean_logprob"],
    "min_logprob_only": ["min_logprob"],
    "mean_entropy_only": ["mean_entropy"],
    "original_3feat": ORIGINAL_THREE_FEATURES,
    "location": ["mean_logprob", "median_logprob", "logprob_p25"],
    "tail": ["min_logprob", "logprob_p10", "frac_low_confidence"],
    "entropy": ["mean_entropy", "max_entropy", "std_entropy"],
    "margin": ["mean_margin", "min_margin"],
    "positional": ["first_half_mean_logprob", "second_half_mean_logprob",
                   "logprob_delta_second_minus_first", "logprob_trend_slope"],
    "all_17feat": ALL_RICH_FEATURE_NAMES,
    "all_16feat_no_length": [f for f in ALL_RICH_FEATURE_NAMES if f != "answer_length"],
}
CONTROLS = {
    "length_only": ["answer_length"],
    "question_only": ["question_mean_logprob", "question_n_tokens"],
    "length_plus_question": ["answer_length", "question_mean_logprob", "question_n_tokens"],
}
SC = {"sc_greedy_support": ["sc_greedy_support"], "sc_semantic_entropy": ["sc_semantic_entropy"],
      "sc_lexical_agreement": ["sc_lexical_agreement"]}


def load_cell(model, dataset, split):
    cell = V2_ROOT / model / dataset / split
    df = pd.read_csv(cell / "features.csv")
    df["label"] = df[PRIMARY[dataset]]
    return df[df["label"].notna()].reset_index(drop=True).assign(label=lambda d: d["label"].astype(int))


def fixed(df, f):
    s = RISK_SIGN[f] * df[f].to_numpy()
    y = df["label"].to_numpy()
    return {"auroc": auroc(y, s), "ci": boot_ci(y, s, 1000), "p_one_sided": perm_p_fixed(y, s, 2000)}


def fitted(df, feats, n_perm=200):
    X, y = df[feats].to_numpy(float), df["label"].to_numpy()
    p = oof(X, y)
    return {"auroc": auroc(y, p), "ci": boot_ci(y, p, 1000), "p": perm_p_fitted(X, y, n_perm),
            "repcv": repeated_cv(X, y)}


def nested(df, candidates, seed=0):
    """Outer 5-fold; inside each training fold, pick the feature group with the best
    inner-CV AUROC, refit it, score the outer test fold."""
    y = df["label"].to_numpy()
    p = np.zeros(len(y))
    picks = []
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(df, y):
        inner = df.iloc[tr].reset_index(drop=True)
        best = max(candidates, key=lambda g: auroc(inner["label"], oof(inner[candidates[g]].to_numpy(float),
                                                                       inner["label"].to_numpy(), seed=seed)))
        picks.append(best)
        m = clf().fit(df.iloc[tr][candidates[best]].to_numpy(float), y[tr])
        p[te] = m.predict_proba(df.iloc[te][candidates[best]].to_numpy(float))[:, 1]
    return {"auroc": auroc(y, p), "ci": boot_ci(y, p, 1000), "picks": picks}


def redundancy(df):
    X = df[ALL_RICH_FEATURE_NAMES].to_numpy(float)
    rho = spearmanr(X).correlation
    varying = X.std(0) > 1e-12
    ev = np.linalg.eigvalsh(np.corrcoef(X[:, varying].T))[::-1]
    ev = np.clip(ev, 0, None)
    share = np.cumsum(ev) / ev.sum()
    high = [(a, b, round(float(rho[i, j]), 3)) for i, a in enumerate(ALL_RICH_FEATURE_NAMES)
            for j, b in enumerate(ALL_RICH_FEATURE_NAMES) if i < j and abs(rho[i, j]) >= 0.9]
    return {"constant_features": [f for f, v in zip(ALL_RICH_FEATURE_NAMES, varying) if not v],
            "n_components_90pct": int(np.searchsorted(share, 0.90) + 1),
            "n_components_95pct": int(np.searchsorted(share, 0.95) + 1),
            "participation_ratio": float(ev.sum() ** 2 / (ev ** 2).sum()),
            "pairs_abs_spearman_ge_0.9": high,
            "n_unique": {f: int(df[f].nunique()) for f in ALL_RICH_FEATURE_NAMES},
            "median": {f: float(df[f].median()) for f in ALL_RICH_FEATURE_NAMES}}


def tfidf_oof(texts, y, seed=42):
    p = np.zeros(len(y))
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(texts, y):
        v = TfidfVectorizer(min_df=2, ngram_range=(1, 2))
        Xtr = v.fit_transform([texts[i] for i in tr])
        m = LogisticRegression(C=1.0, max_iter=2000).fit(Xtr, y[tr])
        p[te] = m.predict_proba(v.transform([texts[i] for i in te]))[:, 1]
    return p


def controls(df, questions):
    y = df["label"].to_numpy()
    out = {k: fitted(df, v, n_perm=200) for k, v in CONTROLS.items()}
    qt = [questions[q] for q in df["qid"]]
    pq = tfidf_oof(qt, y)
    out["question_tfidf"] = {"auroc": auroc(y, pq), "ci": boot_ci(y, pq, 1000)}
    pa = tfidf_oof(df["answer"].fillna("").tolist(), y)
    out["answer_tfidf"] = {"auroc": auroc(y, pa), "ci": boot_ci(y, pa, 1000)}
    base = CONTROLS["length_plus_question"]
    p0 = oof(df[base].to_numpy(float), y)
    p1 = oof(df[base + ORIGINAL_THREE_FEATURES].to_numpy(float), y)
    out["uncertainty_added_to_controls"] = {"controls_auroc": auroc(y, p0), "controls_plus_3feat_auroc": auroc(y, p1),
                                            **paired_delta(y, p1, p0)}
    return out


def by_error_type(df, dataset):
    """Each error type against all correct answers, fixed scores only."""
    out = {}
    # judge rubric: imitative = listed misconception (TruthfulQA) or listed distractor (SciQ)
    if "judge_type" not in df:
        return out
    types = df["judge_type"].where(df["label"] == 1, "correct").fillna("other")
    c = df[types == "correct"]
    for t in ["imitative", "other"]:
        w = df[types == t]
        if len(w) < 5:
            out[t] = {"n": int(len(w))}
            continue
        sub = pd.concat([c, w])
        out[t] = {"n": int(len(w)), **{f: fixed(sub, f) for f in
                  ["mean_logprob", "min_logprob", "mean_entropy", "answer_length", "sc_greedy_support"]}}
    return out


def quadrants(df):
    """Share of wrong answers at least as confident as the median correct answer.
    Under no signal this is 0.5; a perfect detector gives 0."""
    out = {}
    for f in ["mean_logprob", "min_logprob"]:
        r = RISK_SIGN[f] * df[f]
        thr = r[df.label == 0].median()
        wrong = r[df.label == 1]
        out[f] = {"frac_wrong_as_confident_as_median_correct": float((wrong <= thr).mean()),
                  "n_wrong": int(len(wrong))}
    return out


def calibration(df):
    y = df["label"].to_numpy()
    p = oof(df[ORIGINAL_THREE_FEATURES].to_numpy(float), y)
    bins = np.clip((p * 10).astype(int), 0, 9)
    ece = sum(abs(p[bins == b].mean() - y[bins == b].mean()) * (bins == b).mean() for b in range(10) if (bins == b).any())
    return {"ece": float(ece), "brier": float(brier_score_loss(y, p)),
            "brier_base_rate": float(brier_score_loss(y, np.full(len(y), y.mean())))}


def operating(df):
    y = df["label"].to_numpy()
    p = oof(df[ORIGINAL_THREE_FEATURES].to_numpy(float), y)
    fpr, tpr, thr = roc_curve(y, p)
    out = {}
    for target in [0.1, 0.2]:
        i = np.searchsorted(fpr, target, side="right") - 1
        pred = p >= thr[i]
        prec = float(y[pred].mean()) if pred.any() else float("nan")
        out[f"fpr_{target}"] = {"recall": float(tpr[i]), "precision": prec, "base_rate": float(y.mean())}
    return out


def within_question(model, dataset, split):
    """Do sampled answers to the SAME question that are wrong have lower confidence
    than those that are right? Pools wrong-vs-right pairs within questions only, so
    question difficulty cannot contribute. SciQ only (containment label is automatic)."""
    cell = V2_ROOT / model / dataset / split
    if dataset != "sciq" or not (cell / "labels_auto_samples.csv").exists():
        return None
    recs = read_jsonl(cell / "samples.jsonl")
    lab = pd.read_csv(cell / "labels_auto_samples.csv")
    s = pd.DataFrame([{"qid": r["qid"], "k": r["k"], **extract_rich_features(r["logprobs"], r["entropies"],
                                                                             r["margins"])} for r in recs])
    s = s.merge(lab[["qid", "k", "containment_label"]], on=["qid", "k"])
    out = {}
    for f in ["mean_logprob", "min_logprob", "mean_entropy", "answer_length"]:
        conc, per_q = [], {}
        for q, g in s.groupby("qid"):
            r = RISK_SIGN[f] * g[f].to_numpy()
            yl = g["containment_label"].to_numpy()
            if 0 < yl.sum() < len(yl):
                w, c = r[yl == 1], r[yl == 0]
                per_q[q] = (np.sum(w[:, None] > c[None, :]) + 0.5 * np.sum(w[:, None] == c[None, :])) / (len(w) * len(c))
        vals = np.array(list(per_q.values()))
        rng = np.random.RandomState(0)
        bs = [rng.choice(vals, len(vals)).mean() for _ in range(2000)]
        # sign-flip test: under H0 each question's concordance is symmetric around 0.5
        null = [np.mean(0.5 + (vals - 0.5) * rng.choice([-1, 1], len(vals))) for _ in range(5000)]
        out[f] = {"n_mixed_questions": int(len(vals)), "within_question_auroc": float(vals.mean()),
                  "ci": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                  "p_one_sided": float((np.sum(np.array(null) >= vals.mean()) + 1) / 5001)}
    return out


def cross_model(dataset, split):
    """Model A's uncertainty predicting model B's correctness on the same question.
    If this matches A predicting its own errors, the score tracks question
    difficulty, not the specific answer."""
    cells = {m: load_cell(m, dataset, split).set_index("qid") for m in MODELS
             if (V2_ROOT / m / dataset / split / "features.csv").exists()}
    out = {}
    for a, da in cells.items():
        for b, db in cells.items():
            common = da.index.intersection(db.index)
            out[f"{a}->{b}"] = auroc(db.loc[common, "label"], -da.loc[common, "mean_logprob"])
    # other models' error count as a pure question-difficulty predictor
    for b, db in cells.items():
        others = [m for m in cells if m != b]
        if not others:
            continue
        common = db.index
        for m in others:
            common = common.intersection(cells[m].index)
        diff = sum(cells[m].loc[common, "label"] for m in others)
        out[f"others_wrong_count->{b}"] = auroc(db.loc[common, "label"], diff)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="dev")
    a = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    res = {"_cross": {}}
    for d in ["truthfulqa", "sciq"]:
        qs = {r["qid"]: r["question"] for r in
              pd.read_json(PROJECT_ROOT / "data" / d / ("subset_200.jsonl" if a.split == "dev" else "confirm.jsonl"),
                           lines=True).to_dict("records")}
        for m in MODELS:
            if not (V2_ROOT / m / d / a.split / "features.csv").exists():
                continue
            df = load_cell(m, d, a.split)
            key = f"{m}|{d}"
            print(f"== {key}: n={len(df)} pos={int(df.label.sum())}", flush=True)
            r = {"n": int(len(df)), "n_pos": int(df.label.sum())}
            r["baselines"] = {f: fixed(df, f) for f in RISK_SIGN if f in df}
            r["fitted"] = {g: fitted(df, fs) for g, fs in {**GROUPS, **SC}.items()}
            r["nested"] = nested(df, {k: GROUPS[k] for k in ["mean_logprob_only", "min_logprob_only",
                                                            "original_3feat", "location", "tail", "entropy",
                                                            "positional", "all_17feat"]})
            r["redundancy"] = redundancy(df)
            r["controls"] = controls(df, qs)
            r["by_error_type"] = by_error_type(df, d)
            r["quadrants"] = quadrants(df)
            r["calibration"] = calibration(df)
            r["operating"] = operating(df)
            r["within_question"] = within_question(m, d, a.split)
            r["paired"] = {
                "all17_minus_3feat": paired_delta(df.label, oof(df[ALL_RICH_FEATURE_NAMES].to_numpy(float), df.label),
                                                  oof(df[ORIGINAL_THREE_FEATURES].to_numpy(float), df.label)),
                "3feat_minus_neg_mean_logprob": paired_delta(df.label, oof(df[ORIGINAL_THREE_FEATURES].to_numpy(float),
                                                                           df.label), -df["mean_logprob"]),
                "sc_support_minus_neg_mean_logprob": paired_delta(df.label, -df["sc_greedy_support"],
                                                                  -df["mean_logprob"]),
            }
            res[key] = r
            b = r["baselines"]
            print(f"   -mean_lp {b['mean_logprob']['auroc']:.3f}  -min_lp {b['min_logprob']['auroc']:.3f}  "
                  f"ent {b['mean_entropy']['auroc']:.3f}  len {b['answer_length']['auroc']:.3f}  "
                  f"SC {b['sc_greedy_support']['auroc']:.3f}  3feat {r['fitted']['original_3feat']['auroc']:.3f}  "
                  f"17feat {r['fitted']['all_17feat']['auroc']:.3f}  nested {r['nested']['auroc']:.3f}", flush=True)
        res["_cross"][d] = cross_model(d, a.split)
    (OUT / f"{a.split}.json").write_text(json.dumps(res, indent=1, default=float))
    print("saved", OUT / f"{a.split}.json")


if __name__ == "__main__":
    main()
