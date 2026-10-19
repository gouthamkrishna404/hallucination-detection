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
  operating    PR-AUC, D2 calibration, precision/recall/F1 at thresholds fixed on dev
  labels       truth-judge vs Claude-judge agreement, and the main endpoints recomputed
               (a) with truth-judge labels and (b) on items where both judges agree
  independent_h2  H2 with automatic NLI error types (results_v3/.../nli_type.csv), with Claude
               labels and with truth-judge labels (the latter uses no Claude input)
  within_tqa   H2 with the question held fixed: sampled answers of one TruthfulQA question
               (truth-judge labels, NLI types)
  examples     illustrative confidently wrong answers, fixed selection rule

Usage:
    python scripts/v3_eval.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, cohen_kappa_score, roc_auc_score

from src.features_v2 import ORIGINAL_THREE_FEATURES
from src.v2_stats import SEED, auroc, clf, holm, paired_delta
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
    for f in ["ptrue.csv", "nli_sc.csv", "truthjudge.csv", "nli_type.csv"]:
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


def operating(dev, conf, s):
    """PR-AUC (random baseline = base rate), D2 calibration, and precision/recall/F1/specificity at
    thresholds fixed on dev: D0 at the dev median risk, D2 at p = 0.5 and at the dev threshold with
    20% false-positive rate. Nothing here is tuned on confirm."""
    y = conf["label"].to_numpy()
    out = {"base_rate": float(y.mean()), "pr_auc": {k: float(average_precision_score(y, v)) for k, v in s.items()}}
    p = s["D2"]
    bins = np.clip((p * 10).astype(int), 0, 9)
    out["D2_calibration"] = {
        "ece_10bin": float(sum(abs(p[bins == b].mean() - y[bins == b].mean()) * (bins == b).mean()
                               for b in range(10) if (bins == b).any())),
        "brier": float(brier_score_loss(y, p)), "brier_base_rate": float(brier_score_loss(y, np.full(len(y), y.mean()))),
        "mean_predicted": float(p.mean())}
    yd = dev["label"].to_numpy()
    d0_dev = -dev["mean_logprob"].to_numpy()
    d2_dev = clf().fit(dev[ORIGINAL_THREE_FEATURES].to_numpy(float), yd).predict_proba(
        dev[ORIGINAL_THREE_FEATURES].to_numpy(float))[:, 1]
    thr = {"D0@dev_median": ("D0", np.median(d0_dev)), "D2@0.5": ("D2", 0.5),
           "D2@dev_fpr0.2": ("D2", np.quantile(d2_dev[yd == 0], 0.8))}
    for name, (k, t) in thr.items():
        pred = s[k] > t
        tp, fp = int((pred & (y == 1)).sum()), int((pred & (y == 0)).sum())
        fn, tn = int((~pred & (y == 1)).sum()), int((~pred & (y == 0)).sum())
        prec = tp / (tp + fp) if tp + fp else float("nan")
        rec = tp / (tp + fn)
        out[name] = {"precision": prec, "recall": rec, "specificity": tn / (tn + fp),
                     "f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0, "flagged": float(pred.mean()),
                     "accuracy": (tp + tn) / len(y), "majority_accuracy": float(max(y.mean(), 1 - y.mean()))}
    return out


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
            r["operating"] = operating(dev, conf, s)
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


def nli_types(conf):
    """Same frame with the Claude error type replaced by the automatic NLI type."""
    return conf.assign(judge_type=np.where(conf["nli_imitative"] == 1, "imitative", "other"))


def independent_h2():
    """H2 with error types that do not come from the Claude judge: (a) Claude labels + NLI types,
    (b) truth-judge labels + NLI types (no Claude input at all). Also NLI-vs-Claude type agreement."""
    out = {}
    for m in MODELS:
        if not (V3_ROOT / m / "truthfulqa" / "confirm" / "nli_type.csv").exists():
            continue
        r = out.setdefault(m, {})
        dev, conf = load(m, "truthfulqa", "dev"), load(m, "truthfulqa", "confirm")
        w = conf[conf["label"] == 1]
        ct = (w["judge_type"] == "imitative").astype(int)
        r["type_agreement_on_claude_errors"] = {"n": int(len(w)), "agree": float((ct == w["nli_imitative"]).mean()),
                                                "kappa": float(cohen_kappa_score(ct, w["nli_imitative"])),
                                                "nli_imitative_share": float(w["nli_imitative"].mean()),
                                                "claude_imitative_share": float(ct.mean())}
        s0 = all_scores(dev, conf)["D0"]
        r["claude_labels_nli_types"] = error_type_gap(nli_types(conf), s0)
        if "tj_label" in conf:
            tconf = load(m, "truthfulqa", "confirm", "tj")
            r["truthjudge_labels_nli_types"] = error_type_gap(nli_types(tconf), -tconf["mean_logprob"].to_numpy())
    for name in ["claude_labels_nli_types", "truthjudge_labels_nli_types"]:
        got = [out[m][name] for m in out if name in out[m] and "p_two_sided" in out[m][name]]
        for x, p in zip(got, holm([x["p_two_sided"] for x in got])):
            x["p_holm"] = p
    return out


def within_tqa(n=2000, seed=0):
    """H2 with the question held fixed. Among the K=5 sampled answers to one TruthfulQA question
    (truth-judge labels, NLI types: no input from the primary judge), is an imitative-wrong sample
    scored riskier by D0 than a correct sample of the same question? And an other-wrong sample?
    Per-question concordance, averaged over questions; sign-flip test against 0.5; bootstrap CI
    over questions; difference by independent bootstrap of the two question sets."""
    from src.features_v2 import extract_rich_features
    from scripts.v2_labels import read_jsonl
    from scripts.v2_generate import V2_ROOT
    out = {}
    for m in MODELS:
        d = V3_ROOT / m / "truthfulqa" / "confirm"
        if not ((d / "truthjudge_samples.csv").exists() and (d / "nli_type_samples.csv").exists()):
            continue
        s = pd.DataFrame([{"qid": r["qid"], "k": r["k"], "mean_logprob": extract_rich_features(
            r["logprobs"], r["entropies"], r["margins"])["mean_logprob"]}
            for r in read_jsonl(V2_ROOT / m / "truthfulqa" / "confirm" / "samples.jsonl")])
        s = s.merge(pd.read_csv(d / "truthjudge_samples.csv"), on=["qid", "k"]).merge(
            pd.read_csv(d / "nli_type_samples.csv"), on=["qid", "k"])
        s["t"] = np.where(s["tj_label"] == 0, "correct", np.where(s["nli_imitative"] == 1, "imitative", "other"))
        rng = np.random.RandomState(seed)
        r = {"n_samples": int(len(s)), "type_share": s["t"].value_counts(normalize=True).to_dict()}
        per = {}
        for t in ["imitative", "other"]:
            v = []
            for _, g in s.groupby("qid"):
                w, c = -g.loc[g["t"] == t, "mean_logprob"].to_numpy(), -g.loc[g["t"] == "correct", "mean_logprob"].to_numpy()
                if len(w) and len(c):
                    v.append((np.sum(w[:, None] > c[None, :]) + 0.5 * np.sum(w[:, None] == c[None, :])) / (len(w) * len(c)))
            v = np.array(v)
            per[t] = v
            null = [np.mean(0.5 + (v - 0.5) * rng.choice([-1, 1], len(v))) for _ in range(5000)]
            bs = [rng.choice(v, len(v)).mean() for _ in range(n)]
            r[t] = {"n_questions": int(len(v)), "within_auroc": float(v.mean()),
                    "ci": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                    "p_two_sided_vs_0.5": float(min(1.0, 2 * min((np.sum(np.array(null) >= v.mean()) + 1) / 5001,
                                                                  (np.sum(np.array(null) <= v.mean()) + 1) / 5001)))}
        diff = [rng.choice(per["imitative"], len(per["imitative"])).mean() - rng.choice(per["other"], len(per["other"])).mean()
                for _ in range(n)]
        r["delta"] = float(per["imitative"].mean() - per["other"].mean())
        r["delta_ci"] = [float(np.percentile(diff, 2.5)), float(np.percentile(diff, 97.5))]
        out[m] = r
    return out


def pooled_h2(n=2000, seed=SEED):
    """Mean over the three models of the preregistered H2 gap (D0, judge labels and types), with a
    bootstrap that resamples QUESTIONS jointly for all models, since the models answer the same
    questions and their gaps are not independent."""
    cells = {m: load(m, "truthfulqa", "confirm").set_index("qid") for m in MODELS}
    qids = np.array(sorted(set.intersection(*(set(c.index) for c in cells.values()))))

    def gap(df):
        t = df["judge_type"].where(df["label"] == 1, "correct").fillna("other").to_numpy()
        s = -df["mean_logprob"].to_numpy()
        a = {}
        for k in ["imitative", "other"]:
            i = (t == "correct") | (t == k)
            a[k] = roc_auc_score((t[i] == k).astype(int), s[i])
        return a["imitative"] - a["other"]

    obs = np.mean([gap(c.loc[qids]) for c in cells.values()])
    rng = np.random.RandomState(seed)
    bs = []
    for _ in range(n):
        q = rng.choice(qids, len(qids))
        bs.append(np.mean([gap(c.loc[q]) for c in cells.values()]))
    return {"mean_delta": float(obs), "ci": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
            "p_two_sided": float(min(1.0, max(2 * min(np.mean(np.array(bs) >= 0), np.mean(np.array(bs) <= 0)), 1 / n)))}


def examples(k=4):
    """Illustrative confidently wrong answers, chosen by a fixed rule: TruthfulQA questions on which
    all three models give a high-confidence imitative error in their own top 30% by mean log-prob,
    ranked by the mean confidence percentile."""
    frames = []
    for m in MODELS:
        df = load(m, "truthfulqa", "confirm").assign(model=m)
        df["conf_pct"] = df["mean_logprob"].rank(pct=True)
        frames.append(df)
    a = pd.concat(frames)
    imi = a[(a["label"] == 1) & (a["judge_type"] == "imitative") & (a["judge_conf"] == "high") & (a["conf_pct"] > 0.7)]
    g = imi.groupby("qid").agg(n=("model", "nunique"), mean_pct=("conf_pct", "mean"))
    q = pd.read_json(OUT.parent / "data" / "truthfulqa" / "confirm.jsonl", lines=True).set_index("qid")
    out = []
    qual = g[g["n"] == 3].sort_values("mean_pct", ascending=False)
    for qid in qual.index[:k]:
        rows = imi[imi["qid"] == qid].set_index("model")
        out.append({"qid": qid, "question": q.loc[qid, "question"], "best_answer": q.loc[qid, "best_answer"],
                    "answers": {m: rows.loc[m, "answer"] for m in MODELS},
                    "conf_pct": {m: float(rows.loc[m, "conf_pct"]) for m in MODELS}})
    return {"n_qualifying": int(len(qual)), "items": out}


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
    res["_independent_h2"] = independent_h2()
    res["_examples"] = examples()
    res["_within_tqa"] = within_tqa()
    res["_pooled_h2"] = pooled_h2()
    OUT.mkdir(exist_ok=True)
    (OUT / "extensions.json").write_text(json.dumps(res, indent=1, default=float))
    print("wrote", OUT / "extensions.json")


if __name__ == "__main__":
    main()
