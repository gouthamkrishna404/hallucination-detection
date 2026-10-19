"""Within-question test on SciQ with a better sample label (exploratory, post-preregistration).

The preregistered within-question analysis labels sampled answers with the containment
check, which never accepted a wrong answer but rejected 23/100 correct paraphrases in the
judge audit. If paraphrases also have lower log-probability, that label noise alone could
produce the within-question signal. Hybrid label, fixed before use:

  containment says correct                          -> correct
  otherwise, NLI-equivalent (bidirectional, question-prefixed) to a greedy answer
  the Claude judge labelled correct for the same question (any model)  -> correct
  otherwise                                         -> wrong

Validation: the same rule applied to each model's GREEDY answers, using only the other two
models' greedy answers as anchors, compared with that model's judge labels.

Output: analysis_v3/within.json

Usage:
    python scripts/v3_within.py --nli C:/Stuff/hf_models/deberta-large-mnli
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

from src.features_v2 import extract_rich_features
from scripts.v2_generate import V2_ROOT, load_split
from scripts.v2_labels import MODELS, containment, read_jsonl
from scripts.v3_audit import OUT
from scripts.v3_nli import NLI

SPLIT = "confirm"


def anchors():
    """qid -> {model: greedy answer} for greedy answers the judge labelled correct."""
    out = {}
    for m in MODELS:
        j = pd.read_csv(V2_ROOT / m / "sciq" / SPLIT / "labels_judge_greedy.csv").set_index("qid")["judge_label"]
        for r in read_jsonl(V2_ROOT / m / "sciq" / SPLIT / "greedy.jsonl"):
            if j.get(r["qid"]) == 0:
                out.setdefault(r["qid"], {})[m] = r["answer"]
    return out


def hybrid(nli, q, answers, qrow, anchor_texts):
    """Labels for a list of answers to one question (1 = wrong), plus containment-ambiguous flags."""
    labs, amb = [], []
    for a in answers:
        c, ambiguous = containment(a, qrow)
        amb.append(ambiguous)
        if c == 0 or not anchor_texts:
            labs.append(c)
            continue
        ta = f"{q} {a}"
        pairs = [p for x in anchor_texts for p in ((ta, f"{q} {x}"), (f"{q} {x}", ta))]
        e = nli.entails(pairs)
        labs.append(0 if any(e[i] and e[i + 1] for i in range(0, len(e), 2)) else 1)
    return labs, amb


def within_auroc(df, f, seed=0):
    per_q = []
    for _, g in df.groupby("qid"):
        r, yl = -g[f].to_numpy(), g["label"].to_numpy()
        if 0 < yl.sum() < len(yl):
            w, c = r[yl == 1], r[yl == 0]
            per_q.append((np.sum(w[:, None] > c[None, :]) + 0.5 * np.sum(w[:, None] == c[None, :])) / (len(w) * len(c)))
    v = np.array(per_q)
    rng = np.random.RandomState(seed)
    bs = [rng.choice(v, len(v)).mean() for _ in range(2000)]
    null = [np.mean(0.5 + (v - 0.5) * rng.choice([-1, 1], len(v))) for _ in range(5000)]
    return {"n_mixed_questions": int(len(v)), "within_question_auroc": float(v.mean()),
            "ci": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
            "p_one_sided": float((np.sum(np.array(null) >= v.mean()) + 1) / 5001)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nli", required=True)
    nli = NLI(ap.parse_args().nli)
    qs = load_split("sciq", SPLIT).set_index("qid")
    anc = anchors()
    res = {}
    for m in MODELS:
        # validation on greedy answers, anchors from the other two models only
        j = pd.read_csv(V2_ROOT / m / "sciq" / SPLIT / "labels_judge_greedy.csv").set_index("qid")["judge_label"]
        rows = []
        for r in read_jsonl(V2_ROOT / m / "sciq" / SPLIT / "greedy.jsonl"):
            q = qs.loc[r["qid"]]
            others = [a for k, a in anc.get(r["qid"], {}).items() if k != m]
            (h,), (amb,) = hybrid(nli, q["question"], [r["answer"]], q, others)
            c, _ = containment(r["answer"], q)
            rows.append({"judge": j.get(r["qid"]), "hybrid": h, "containment": c, "amb": amb})
        v = pd.DataFrame(rows).dropna()
        v = v[~v["amb"]]
        val = {lab: {"agree": float((v[lab] == v["judge"]).mean()), "kappa": float(cohen_kappa_score(v["judge"], v[lab])),
                     "false_wrong": float(((v[lab] == 1) & (v["judge"] == 0)).sum() / (v["judge"] == 0).sum()),
                     "false_correct": float(((v[lab] == 0) & (v["judge"] == 1)).sum() / (v["judge"] == 1).sum())}
               for lab in ["containment", "hybrid"]}

        # sampled answers: anchors from all three models' judged-correct greedy answers
        by_q = {}
        for r in read_jsonl(V2_ROOT / m / "sciq" / SPLIT / "samples.jsonl"):
            by_q.setdefault(r["qid"], []).append(r)
        srows = []
        for qid, smp in by_q.items():
            q = qs.loc[qid]
            texts = [s["answer"] for s in smp]
            labs, amb = hybrid(nli, q["question"], texts, q, list(anc.get(qid, {}).values()))
            # anchors from the other two models only: a sample cannot inherit "correct" merely by
            # matching its own model's greedy answer, which is high-probability by construction
            labs_o, _ = hybrid(nli, q["question"], texts, q, [a for k, a in anc.get(qid, {}).items() if k != m])
            cont = [containment(s["answer"], q)[0] for s in smp]
            for s, lab, lo, a, c in zip(smp, labs, labs_o, amb, cont):
                if not a:
                    f = extract_rich_features(s["logprobs"], s["entropies"], s["margins"])
                    srows.append({"qid": qid, "hybrid": lab, "hybrid_other_anchors": lo, "containment": c,
                                  "mean_logprob": f["mean_logprob"], "min_logprob": f["min_logprob"]})
        sd = pd.DataFrame(srows)
        res[m] = {"validation_on_greedy": val, "n_samples": int(len(sd)),
                  "sample_labels_changed": float((sd["hybrid"] != sd["containment"]).mean())}
        for lab in ["containment", "hybrid", "hybrid_other_anchors"]:
            res[m][lab] = {f: within_auroc(sd.assign(label=sd[lab]), f) for f in ["mean_logprob", "min_logprob"]}
        # conservative: only samples both rules label the same way (the relabelled ones are dropped)
        agree = sd[sd["hybrid"] == sd["containment"]]
        res[m]["agreed_only"] = {f: within_auroc(agree.assign(label=agree["containment"]), f)
                                 for f in ["mean_logprob", "min_logprob"]}
        print(m, json.dumps({k: res[m][k] for k in ["validation_on_greedy", "sample_labels_changed"]}), flush=True)
    OUT.mkdir(exist_ok=True)
    (OUT / "within.json").write_text(json.dumps(res, indent=1))
    print("wrote", OUT / "within.json")


if __name__ == "__main__":
    main()
