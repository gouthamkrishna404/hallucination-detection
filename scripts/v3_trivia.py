"""Second factual-recall dataset with judge-free labels (exploratory, post-preregistration).

Does the SciQ picture (a weak single-pass signal that is mostly question difficulty) hold on a
second recall dataset whose labels need no LLM judge? TriviaQA (rc.nocontext validation, 1,000
questions sampled with seed 42: 200 dev / 800 confirm), answers generated with the frozen v2
protocol (scripts/v2_generate.py --dataset triviaqa --root results_v3/triviaqa).

Labels (1 = hallucinated):
  alias   the answer does not contain any of TriviaQA's answer aliases (normalised token match)
  strict  the answer does not contain the canonical answer string (aliases ignored)

Per model, confirm split: D0, D1 (fixed scores, permutation p, Holm across models), D2 (3-feature LR
trained on the 200 dev answers), D2 - D0, question-only and length-only controls, the others-wrong
difficulty proxy, own D0 vs the other models' D0 on the same questions, selective risk at 80% coverage.

Output: analysis_v3/triviaqa.json

Usage:
    python scripts/v3_trivia.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from src.features_v2 import ORIGINAL_THREE_FEATURES, extract_rich_features
from src.labeling import _normalize
from src.v2_stats import clf, holm, paired_delta
from scripts.v2_confirm import N_BOOT, confident_wrong, summary
from scripts.v2_labels import MODELS, contains, read_jsonl
from scripts.v3_audit import OUT, risk_coverage
from scripts.v3_ptrue import V3_ROOT

ROOT = V3_ROOT / "triviaqa"
DATA = Path(__file__).resolve().parent.parent / "data" / "triviaqa"


def cell(m, split):
    q = pd.read_json(DATA / ("subset_200.jsonl" if split == "dev" else "confirm.jsonl"), lines=True).set_index("qid")
    d = ROOT / m / "triviaqa" / split
    rows = []
    for r in read_jsonl(d / "greedy.jsonl"):
        toks = _normalize(r["answer"])
        ref = q.loc[r["qid"]]
        rows.append({"qid": r["qid"], "answer": r["answer"], "truncated": int(r["truncated"]),
                     "alias": int(not any(contains(toks, a) for a in [ref["best_answer"], *ref["correct_answers"]])),
                     "strict": int(not contains(toks, ref["best_answer"])),
                     **extract_rich_features(r["logprobs"], r["entropies"], r["margins"])})
    return pd.DataFrame(rows).merge(pd.read_json(d / "questions.jsonl", lines=True), on="qid")


def evaluate(cells, label):
    res = {}
    for m in MODELS:
        dev, conf = cells[m]["dev"], cells[m]["confirm"]
        y = conf[label].to_numpy()
        s0 = -conf["mean_logprob"].to_numpy()
        f = ORIGINAL_THREE_FEATURES
        p2 = clf().fit(dev[f].to_numpy(float), dev[label]).predict_proba(conf[f].to_numpy(float))[:, 1]
        others = sum(cells[o]["confirm"].set_index("qid")[label].reindex(conf["qid"]).to_numpy()
                     for o in MODELS if o != m)
        other_d0 = {o: paired_delta(y, s0, -cells[o]["confirm"].set_index("qid")["mean_logprob"]
                                    .reindex(conf["qid"]).to_numpy(), N_BOOT) for o in MODELS if o != m}
        rc = risk_coverage(y, s0)
        res[m] = {"n": int(len(y)), "base_rate": float(y.mean()),
                  "D0": summary(y, s0), "D1": summary(y, -conf["min_logprob"].to_numpy()),
                  "D2": summary(y, p2), "D2_minus_D0": paired_delta(y, p2, s0, N_BOOT),
                  "question_logprob": summary(y, -conf["question_mean_logprob"].to_numpy()),
                  "answer_length": summary(y, conf["answer_length"].to_numpy(float)),
                  "others_wrong_proxy": summary(y, others.astype(float)),
                  "own_minus_other_D0": {o: v["delta"] for o, v in other_d0.items()},
                  "risk@0.8": rc["risk@0.8"], "confident_wrong": confident_wrong(y, s0),
                  "truncated_share": float(conf["truncated"].mean())}
    for k in ["D0", "D1", "D2"]:
        for m, p in zip(MODELS, holm([res[m][k]["p_two_sided"] for m in MODELS])):
            res[m][k]["p_holm"] = p
    return res


def main():
    cells = {m: {s: cell(m, s) for s in ["dev", "confirm"]} for m in MODELS}
    out = {"alias": evaluate(cells, "alias"), "strict": evaluate(cells, "strict"),
           "label_agreement": {m: float((cells[m]["confirm"]["alias"] == cells[m]["confirm"]["strict"]).mean())
                               for m in MODELS}}
    OUT.mkdir(exist_ok=True)
    (OUT / "triviaqa.json").write_text(json.dumps(out, indent=1))
    for m, r in out["alias"].items():
        print(f"{m:24s} rate={r['base_rate']:.3f} D0={r['D0']['auroc']:.3f} {np.round(r['D0']['ci'], 3)} "
              f"D2={r['D2']['auroc']:.3f} proxy={r['others_wrong_proxy']['auroc']:.3f} "
              f"own-other={np.round(list(r['own_minus_other_D0'].values()), 3)} risk@0.8={r['risk@0.8']:.3f}")


if __name__ == "__main__":
    main()
