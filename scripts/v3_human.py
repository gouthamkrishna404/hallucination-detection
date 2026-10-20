"""Human validation kit for the two automatic judges (TruthfulQA confirm).

  make   writes labels/human/annotation_sheet.csv (blind: question, references, answer, empty
         human_label column) and labels/human/key.csv (item -> model, qid, both judge labels).
         120 items: 30 per model where the primary (Claude) judge and the truth-judge disagree,
         10 per model where they agree; shuffled with a fixed seed.
  score  after annotation: agreement (and kappa) of the human labels with each judge, overall and
         on the disputed items, and which judge humans side with when the judges disagree.

Usage:
    python scripts/v3_human.py make
    python scripts/v3_human.py score
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

from src.config import PROJECT_ROOT
from scripts.v2_labels import MODELS
from scripts.v3_eval import load

DIR = PROJECT_ROOT / "labels" / "human"
N_DIS, N_AGR, SEED = 30, 10, 7


def make():
    q = pd.read_json(PROJECT_ROOT / "data" / "truthfulqa" / "confirm.jsonl", lines=True).set_index("qid")
    rng = np.random.RandomState(SEED)
    items = []
    for m in MODELS:
        df = load(m, "truthfulqa", "confirm")
        dis, agr = df[df["label"] != df["tj_label"]], df[df["label"] == df["tj_label"]]
        for part, n, kind in [(dis, N_DIS, "disagree"), (agr, N_AGR, "agree")]:
            for _, r in part.iloc[rng.choice(len(part), n, replace=False)].iterrows():
                items.append({"model": m, "qid": r["qid"], "kind": kind, "answer": r["answer"],
                              "claude_label": int(r["label"]), "truthjudge_label": int(r["tj_label"])})
    items = [items[i] for i in rng.permutation(len(items))]
    DIR.mkdir(parents=True, exist_ok=True)
    sheet, key = [], []
    for i, it in enumerate(items):
        ref = q.loc[it["qid"]]
        sheet.append({"item": i, "question": ref["question"],
                      "correct_references": " | ".join(dict.fromkeys([ref["best_answer"], *ref["correct_answers"]])),
                      "incorrect_references": " | ".join(ref["incorrect_answers"]), "answer": it["answer"],
                      "human_label": "", "notes": ""})
        key.append({"item": i, **{k: it[k] for k in ["model", "qid", "kind", "claude_label", "truthjudge_label"]}})
    pd.DataFrame(sheet).to_csv(DIR / "annotation_sheet.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(key).to_csv(DIR / "key.csv", index=False)
    print(f"wrote {len(sheet)} items to {DIR / 'annotation_sheet.csv'} (key in key.csv)")


def score():
    s = pd.read_csv(DIR / "annotation_sheet.csv").merge(pd.read_csv(DIR / "key.csv"), on="item")
    s = s[s["human_label"].astype(str).str.strip().isin(["0", "1", "0.0", "1.0"])]
    s["human_label"] = s["human_label"].astype(float).astype(int)
    out = {"n_annotated": int(len(s))}
    for name, sub in [("all", s), ("disputed", s[s["kind"] == "disagree"]), ("agreed", s[s["kind"] == "agree"])]:
        if len(sub) == 0:
            continue
        r = {"n": int(len(sub))}
        for j in ["claude_label", "truthjudge_label"]:
            r[j] = {"agree": float((sub[j] == sub["human_label"]).mean()),
                    "kappa": float(cohen_kappa_score(sub[j], sub["human_label"])) if sub[j].nunique() > 1 else None}
        out[name] = r
    d = s[s["kind"] == "disagree"]
    out["disputed_by_model"] = {m: {"n": int((d["model"] == m).sum()),
                                    "human_sides_with_claude": float((d.loc[d["model"] == m, "human_label"] ==
                                                                      d.loc[d["model"] == m, "claude_label"]).mean())}
                                for m in MODELS if (d["model"] == m).any()}
    (PROJECT_ROOT / "analysis_v3" / "human.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    {"make": make, "score": score}[sys.argv[1]]()
