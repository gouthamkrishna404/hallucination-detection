"""Label-independent extra cells (exploratory): a second prompt and a larger model, TruthfulQA only.

Every label here comes from the AllenAI truth-judge and every error type from the NLI rule
(scripts/v3_nli.py --types-only); the Claude judge is not involved. The same quantities are
computed for the original prompt / original models under the same independent labels, so
the comparison is like for like.

  prompt2  results_v3/prompt2/<model>/truthfulqa/confirm   neutral prompt that allows "I don't know"
  scale    results_v3/scale/qwen2.5-3b-instruct/truthfulqa/{dev,confirm}   bf16, otherwise v2 protocol
  base     the v2 cells (results_v2 + results_v3 label/type files)

Output: analysis_v3/extra.json

Usage:
    python scripts/v3_extra.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from src.features_v2 import ORIGINAL_THREE_FEATURES, extract_rich_features
from src.v2_stats import clf, holm
from scripts.v2_confirm import confident_wrong, error_type_gap, summary
from scripts.v2_generate import V2_ROOT
from scripts.v2_labels import MODELS, read_jsonl
from scripts.v3_audit import OUT
from scripts.v3_ptrue import V3_ROOT

REFUSAL = r"\b(?:i don't know|i do not know|not sure|no way to know|i'm not certain|cannot be determined|unknown)\b"


def cell(answers_dir, labels_dir):
    g = read_jsonl(answers_dir / "greedy.jsonl")
    df = pd.DataFrame([{"qid": r["qid"], "answer": r["answer"], "truncated": int(r["truncated"]),
                        **extract_rich_features(r["logprobs"], r["entropies"], r["margins"])} for r in g])
    for f in ["truthjudge.csv", "nli_type.csv"]:
        df = df.merge(pd.read_csv(labels_dir / f), on="qid")
    return df.assign(label=df["tj_label"].astype(int))


def evaluate(conf, dev=None):
    y = conf["label"].to_numpy()
    s0 = -conf["mean_logprob"].to_numpy()
    typed = conf.assign(judge_type=np.where(conf["nli_imitative"] == 1, "imitative", "other"))
    r = {"n": int(len(y)), "base_rate": float(y.mean()),
         "refusal_share": float(conf["answer"].str.contains(REFUSAL, case=False, regex=True).mean()),
         "truncated_share": float(conf["truncated"].mean()), "mean_answer_tokens": float(conf["answer_length"].mean()),
         "D0": summary(y, s0), "H2": error_type_gap(typed, s0), "confident_wrong": confident_wrong(y, s0)}
    if dev is not None:
        f = ORIGINAL_THREE_FEATURES
        p = clf().fit(dev[f].to_numpy(float), dev["label"]).predict_proba(conf[f].to_numpy(float))[:, 1]
        r["D2"] = summary(y, p)
    return r


def main():
    res = {"base": {}, "prompt2": {}, "scale": {}}
    for m in MODELS:
        lab = lambda s: V3_ROOT / m / "truthfulqa" / s
        if (lab("confirm") / "truthjudge.csv").exists():
            dev = cell(V2_ROOT / m / "truthfulqa" / "dev", lab("dev"))
            res["base"][m] = evaluate(cell(V2_ROOT / m / "truthfulqa" / "confirm", lab("confirm")), dev)
        p2 = V3_ROOT / "prompt2" / m / "truthfulqa" / "confirm"
        if (p2 / "truthjudge.csv").exists() and (p2 / "nli_type.csv").exists():
            res["prompt2"][m] = evaluate(cell(p2, p2))
    sc = V3_ROOT / "scale" / "qwen2.5-3b-instruct" / "truthfulqa"
    if (sc / "confirm" / "truthjudge.csv").exists():
        res["scale"]["qwen2.5-3b-instruct"] = evaluate(cell(sc / "confirm", sc / "confirm"), cell(sc / "dev", sc / "dev"))
    for group in res.values():
        for key in ["D0", "H2"]:
            got = [v[key] for v in group.values() if "p_two_sided" in v[key]]
            for x, p in zip(got, holm([x["p_two_sided"] for x in got])):
                x["p_holm"] = p
    OUT.mkdir(exist_ok=True)
    (OUT / "extra.json").write_text(json.dumps(res, indent=1))
    for g, v in res.items():
        for m, r in v.items():
            print(f"{g:8s} {m:24s} n={r['n']} rate={r['base_rate']:.3f} refusal={r['refusal_share']:.3f} "
                  f"D0={r['D0']['auroc']:.3f} H2 delta={r['H2'].get('delta', float('nan')):.3f}"
                  + (f" D2={r['D2']['auroc']:.3f}" if "D2" in r else ""))


if __name__ == "__main__":
    main()
