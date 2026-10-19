"""Post-preregistration audit (v3) of the frozen v2 data. CPU only, no new generations.

Only `containment` was preregistered (PREREGISTRATION.md section 5); v2_confirm.py
never ran it, so it is run here with the unchanged v2_confirm.evaluate. Every
other section is exploratory: chosen after the confirm results were known, and
reported as such.

Sections (keys of analysis_v3/audit.json):
  containment  SciQ, containment label instead of the judge label (preregistered)
  split        exact / near-duplicate question overlap between dev and confirm
  length       D0/D1/D2 AUROC within answer-length quintiles; does D2 add to a
               dev-trained length model?
  d2           standardized D2 weights; dev-trained ablations scored on confirm
  h2_controls  imitative-vs-other gap within length quintiles, and a regression of
               D0 on error type with length, truncation, question and difficulty
               controls (TruthfulQA category fixed effects)
  confident    who the confidently-wrong answers are; risk-coverage of D0 and D2

Usage:
    python scripts/v3_audit.py
"""
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import roc_auc_score

from src.config import PROJECT_ROOT
from src.features_v2 import ORIGINAL_THREE_FEATURES
from src.v2_stats import SEED, auroc, clf, holm, paired_delta
from scripts.v2_analysis import V2_ROOT, load_cell
from scripts.v2_confirm import N_BOOT, evaluate, scores, summary
from scripts.v2_labels import MODELS

OUT = PROJECT_ROOT / "analysis_v3"
DATASETS = ["truthfulqa", "sciq"]


def cell(m, d, split):
    """Judge-labelled cell plus the number of OTHER models that got the question wrong."""
    df = load_cell(m, d, split)
    others = [load_cell(o, d, split).set_index("qid")["label"] for o in MODELS if o != m]
    df["others_wrong"] = sum(o.reindex(df["qid"]).fillna(0).to_numpy() for o in others)
    q = pd.read_json(PROJECT_ROOT / "data" / d / ("subset_200.jsonl" if split == "dev" else "confirm.jsonl"),
                     lines=True)
    return df.merge(q[["qid", "category"]], on="qid", how="left")


# ---------------------------------------------------------------- containment
def containment():
    def load(m, split):
        df = pd.read_csv(V2_ROOT / m / "sciq" / split / "features.csv")
        df = df[~df["containment_ambiguous"].astype(bool)].reset_index(drop=True)
        # error types belong to the judge label; H2 is not defined for this label
        return df.assign(label=df["containment_label"].astype(int)).drop(columns=["judge_type"])

    res = {m: evaluate(load(m, "dev"), load(m, "confirm")) for m in MODELS}
    for k in ["D0", "D1", "D2", "D3", "D4", "D5"]:
        got = [res[m]["detectors"][k] for m in MODELS]
        for g, p in zip(got, holm([g["p_two_sided"] for g in got])):
            g["p_holm"] = p
    for k in ["D1", "D2", "D3", "D4", "D5"]:
        got = [res[m]["paired_vs_D0"][k] for m in MODELS]
        for g, p in zip(got, holm([g["p_two_sided"] for g in got])):
            g["p_holm"] = p
    return res


# ---------------------------------------------------------------------- split
def norm(q):
    return re.sub(r"[^a-z0-9 ]", "", q.lower()).strip()


def split_integrity():
    out = {}
    for d in DATASETS:
        dev = pd.read_json(PROJECT_ROOT / "data" / d / "subset_200.jsonl", lines=True)
        conf = pd.read_json(PROJECT_ROOT / "data" / d / "confirm.jsonl", lines=True)
        v = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5)).fit(pd.concat([dev.question, conf.question]))
        sim = (v.transform(dev.question) @ v.transform(conf.question).T).toarray()
        near = [(dev.question[i], conf.question[j], round(float(sim[i, j]), 3))
                for i, j in zip(*np.where(sim >= 0.8))]
        within = conf.question.map(norm)
        out[d] = {"n_dev": len(dev), "n_confirm": len(conf),
                  "exact_dev_in_confirm": int(dev.question.map(norm).isin(set(within)).sum()),
                  "duplicate_questions_within_confirm": int(within.duplicated().sum()),
                  "near_duplicates_char_tfidf_ge_0.8": near,
                  "max_dev_confirm_similarity": float(sim.max())}
    return out


# --------------------------------------------------------------------- length
def stratified_auroc(y, s, strata):
    """AUROC over within-stratum (wrong, correct) pairs only: pairs are weighted by
    stratum, so a score that only tracks the stratum variable gets 0.5."""
    num = den = 0.0
    for g in np.unique(strata):
        i = strata == g
        npos, nneg = y[i].sum(), (1 - y[i]).sum()
        if npos and nneg:
            num += roc_auc_score(y[i], s[i]) * npos * nneg
            den += npos * nneg
    return num / den if den else float("nan")


def strat_summary(y, s, strata, n=N_BOOT, seed=SEED):
    rng = np.random.RandomState(seed)
    bs = []
    for _ in range(n):
        i = rng.randint(0, len(y), len(y))
        bs.append(stratified_auroc(y[i], s[i], strata[i]))
    return {"auroc": stratified_auroc(y, s, strata), "ci": [float(np.nanpercentile(bs, 2.5)),
                                                            float(np.nanpercentile(bs, 97.5))]}


def quintiles(x):
    return pd.qcut(pd.Series(x).rank(method="first"), 5, labels=False).to_numpy()


LEN_FEATS = ["answer_length", "log_length", "truncated"]


def with_len(df):
    return df.assign(log_length=np.log1p(df["answer_length"]))


def length(dev, conf, s):
    y = conf["label"].to_numpy()
    q = quintiles(conf["answer_length"].to_numpy())
    dev, conf = with_len(dev), with_len(conf)
    p_len = clf().fit(dev[LEN_FEATS].to_numpy(float), dev["label"]).predict_proba(conf[LEN_FEATS].to_numpy(float))[:, 1]
    f = LEN_FEATS + ORIGINAL_THREE_FEATURES
    p_both = clf().fit(dev[f].to_numpy(float), dev["label"]).predict_proba(conf[f].to_numpy(float))[:, 1]
    return {"length_quintile_auroc": {k: strat_summary(y, s[k], q) for k in ["D0", "D1", "D2"]},
            "length_model_dev_trained": summary(y, p_len),
            "D2_plus_length_minus_length": paired_delta(y, p_both, p_len, N_BOOT),
            "D2_minus_length_model": paired_delta(y, s["D2"], p_len, N_BOOT)}


# ------------------------------------------------------------------------- D2
def d2(dev, conf):
    y = conf["label"].to_numpy()
    m = clf().fit(dev[ORIGINAL_THREE_FEATURES].to_numpy(float), dev["label"])
    out = {"standardized_weights": dict(zip(ORIGINAL_THREE_FEATURES, m[-1].coef_[0].round(3).tolist()))}
    subsets = {"mean+min": ["mean_logprob", "min_logprob"], "mean+entropy": ["mean_logprob", "mean_entropy"],
               "min+entropy": ["min_logprob", "mean_entropy"], "entropy": ["mean_entropy"]}
    for k, f in subsets.items():
        p = clf().fit(dev[f].to_numpy(float), dev["label"]).predict_proba(conf[f].to_numpy(float))[:, 1]
        out[k] = auroc(y, p)
    # is the fitted contrast "fluent overall, one improbable token"? sign check on dev
    out["dev_spearman_mean_vs_min"] = float(dev[["mean_logprob", "min_logprob"]].corr("spearman").iloc[0, 1])
    return out


# ---------------------------------------------------------------- H2 controls
def h2_controls(conf, s0, n=N_BOOT, seed=SEED):
    t = conf["judge_type"].where(conf["label"] == 1, "correct").fillna("other").to_numpy()
    q = quintiles(conf["answer_length"].to_numpy())
    idx = {k: np.where(t == k)[0] for k in ["correct", "imitative", "other"]}

    def gap(ix):
        a = {}
        for k in ["imitative", "other"]:
            j = np.r_[ix["correct"], ix[k]]
            a[k] = stratified_auroc((t[j] == k).astype(int), s0[j], q[j])
        return a["imitative"], a["other"]

    obs = gap(idx)
    rng = np.random.RandomState(seed)
    d = [np.subtract(*gap({k: rng.choice(v, len(v)) for k, v in idx.items()})) for _ in range(n)]
    out = {"length_quintile": {"auroc_imitative": obs[0], "auroc_other": obs[1], "delta": obs[0] - obs[1],
                               "ci": [float(np.nanpercentile(d, 2.5)), float(np.nanpercentile(d, 97.5))]}}

    # OLS: z(D0) ~ imitative + other + log length + truncated + z(question log-prob) + log question
    # length + others_wrong + category FE. Coefficient difference imitative - other, bootstrap CI.
    z = lambda x: (x - x.mean()) / x.std()
    X = pd.DataFrame({"imitative": (t == "imitative").astype(float), "other": (t == "other").astype(float),
                      "log_len": np.log1p(conf["answer_length"]), "truncated": conf["truncated"].astype(float),
                      "q_lp": z(conf["question_mean_logprob"]), "q_len": np.log(conf["question_n_tokens"]),
                      "others_wrong": conf["others_wrong"].astype(float)})
    X = pd.concat([X, pd.get_dummies(conf["category"], drop_first=True, dtype=float)], axis=1)
    X.insert(0, "const", 1.0)
    X, yv = X.to_numpy(float), z(s0)

    def coefs(i):
        b = np.linalg.lstsq(X[i], yv[i], rcond=None)[0]
        return b[1], b[2]

    b_i, b_o = coefs(np.arange(len(yv)))
    bs = np.array([coefs(rng.randint(0, len(yv), len(yv))) for _ in range(n)])
    diff = bs[:, 0] - bs[:, 1]
    out["regression"] = {"coef_imitative": float(b_i), "coef_other": float(b_o), "diff": float(b_i - b_o),
                         "ci_diff": [float(np.percentile(diff, 2.5)), float(np.percentile(diff, 97.5))],
                         "ci_imitative": [float(np.percentile(bs[:, 0], 2.5)), float(np.percentile(bs[:, 0], 97.5))],
                         "ci_other": [float(np.percentile(bs[:, 1], 2.5)), float(np.percentile(bs[:, 1], 97.5))],
                         "note": "outcome z(-mean log-prob): positive = less confident than a correct answer "
                                 "with the same controls"}
    out["mean_answer_length"] = {k: float(conf["answer_length"].to_numpy()[v].mean()) for k, v in idx.items()}
    return out


# ------------------------------------------------------------------ confident
def risk_coverage(y, s, covs=(1.0, 0.9, 0.8, 0.7, 0.5)):
    order = np.argsort(s, kind="stable")  # most confident first
    ys = y[order]
    risk = np.cumsum(ys) / np.arange(1, len(ys) + 1)
    return {"aurc": float(risk.mean()), **{f"risk@{c}": float(risk[int(np.ceil(c * len(ys))) - 1]) for c in covs}}


def confident(conf, s):
    y = conf["label"].to_numpy()
    t = conf["judge_type"].where(conf["label"] == 1, "correct").fillna("other").to_numpy() \
        if "judge_type" in conf else None
    out = {"base_rate": float(y.mean())}
    s0 = s["D0"]
    thr = np.median(s0[y == 0])
    cw = (y == 1) & (s0 <= thr)
    if t is not None:
        out["share_imitative_all_wrong"] = float((t[y == 1] == "imitative").mean())
        out["share_imitative_confident_wrong"] = float((t[cw] == "imitative").mean())
        top = s0 <= np.quantile(s0, 0.2)
        out["top20pct_confident"] = {"error_rate": float(y[top].mean()), "n": int(top.sum()),
                                     "share_imitative_of_errors": float((t[top & (y == 1)] == "imitative").mean())}
    rng = np.random.RandomState(SEED)
    for k in ["D0", "D2"]:
        rc = risk_coverage(y, s[k])
        bs = [risk_coverage(y[i], s[k][i])["risk@0.8"] - y[i].mean()
              for i in (rng.randint(0, len(y), len(y)) for _ in range(N_BOOT))]
        rc["risk@0.8_minus_base_ci"] = [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
        out[f"risk_coverage_{k}"] = rc
    out["oracle_aurc"] = risk_coverage(y, y.astype(float))["aurc"]
    return out


def main():
    res = {"containment": containment(), "split": split_integrity()}
    for d in DATASETS:
        for m in MODELS:
            key = f"{m}|{d}"
            print("==", key, flush=True)
            dev, conf = cell(m, d, "dev"), cell(m, d, "confirm")
            s = scores(dev, conf)
            r = res.setdefault(key, {})
            r["length"] = length(dev, conf, s)
            r["d2"] = d2(dev, conf)
            r["confident"] = confident(conf, s)
            if d == "truthfulqa":
                r["h2_controls"] = h2_controls(conf, s["D0"])
    OUT.mkdir(exist_ok=True)
    (OUT / "audit.json").write_text(json.dumps(res, indent=1, default=float))
    print("wrote", OUT / "audit.json")


if __name__ == "__main__":
    main()
