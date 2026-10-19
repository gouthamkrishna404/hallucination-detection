"""Constants quoted in the paper that are not in the main analysis JSONs, recomputed from source
files, plus an integrity check of the built paper.

  python scripts/v3_constants.py            -> analysis_v3/constants.json
  python scripts/v3_constants.py --check    -> also check the built DOCX: no placeholder text,
                                               citations numbered 1..N in order of first use,
                                               every reference cited
"""
import argparse
import json
import re
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

from src.config import PROJECT_ROOT
from src.v2_stats import auroc, boot_ci
from scripts.v2_analysis import V2_ROOT, load_cell

ROOT = PROJECT_ROOT
PAPER = ROOT / "paper" / "Learned_Falsehoods_Uncertainty_Paper.docx"
MODELS = ["qwen2.5-1.5b-instruct", "llama-3.2-1b-instruct", "smollm2-1.7b-instruct"]


def rejudge():
    out = {}
    for ds, tag in [("truthfulqa", "tqa"), ("sciq", "sciq")]:
        orig = pd.concat([pd.read_csv(p, dtype={"item": str}) for p in (ROOT / "labels/judge" / f"{ds}_dev").glob("labels_*.csv")])
        orig = orig.drop_duplicates("item", keep="last").set_index("item")
        re_ = pd.read_csv(ROOT / "labels/judge/rejudge" / f"labels_{tag}.csv", dtype={"item": str}).set_index("item")
        j = re_.join(orig, lsuffix="_re", how="inner")
        out[ds] = {"n": int(len(j)), "agree": float((j["label_re"] == j["label"]).mean()),
                   "kappa": float(cohen_kappa_score(j["label_re"], j["label"]))}
    return out


def paraphrase_rejection():
    """Among SciQ dev answers the judge called correct, share the containment label calls wrong."""
    rows = [pd.read_csv(V2_ROOT / m / "sciq" / "dev" / "features.csv") for m in
            ["qwen2.5-1.5b-instruct", "llama-3.2-1b-instruct", "smollm2-1.7b-instruct"]]
    df = pd.concat(rows).dropna(subset=["judge_label"])
    df = df[~df["containment_ambiguous"].astype(bool)].drop_duplicates(["answer"])
    ok = df[df["judge_label"] == 0]
    bad = df[df["judge_label"] == 1]
    return {"n_judge_correct": int(len(ok)), "containment_rejects_correct": float((ok["containment_label"] == 1).mean()),
            "containment_accepts_wrong": float((bad["containment_label"] == 0).mean())}


def contamination():
    d = ROOT / "results/qwen2.5-1.5b-instruct/truthfulqa/features"
    a = pd.read_csv(d / "single_pass_features.csv").set_index("qid")
    b = pd.read_csv(d / "single_pass_features_clean.csv").set_index("qid")
    q = a.index.intersection(b.index)
    shift = (b.loc[q, "mean_logprob"] - a.loc[q, "mean_logprob"]).abs()
    y = a.loc[q, "label"]
    sep = abs(a.loc[q][y == 1]["mean_logprob"].mean() - a.loc[q][y == 0]["mean_logprob"].mean())
    return {"n": int(len(q)), "mean_abs_shift": float(shift.mean()), "max_abs_shift": float(shift.max()),
            "class_separation_original": float(sep), "ratio": float(shift.mean() / sep)}


def da1():
    m = json.loads((ROOT / "outputs/metrics/all_metrics.json").read_text())["proposed_all_three"]
    return {"auroc": m["auroc"], "n": m["n"]}


def llama_untruncated():
    df = load_cell("llama-3.2-1b-instruct", "truthfulqa", "confirm")
    d = df[df["truncated"] == 0]
    y, s = d["label"].to_numpy(), -d["mean_logprob"].to_numpy()
    return {"n": int(len(d)), "truncated_share": float(df["truncated"].mean()), "auroc": auroc(y, s), "ci": boot_ci(y, s)}


def judge_disagreement():
    """TruthfulQA confirm: answer length and confidence by (primary judge, truth-judge) label pair."""
    out = {}
    for m in MODELS:
        df = load_cell(m, "truthfulqa", "confirm").merge(
            pd.read_csv(ROOT / "results_v3" / m / "truthfulqa" / "confirm" / "truthjudge.csv"), on="qid")
        g = df.groupby(df["label"].astype(str) + df["tj_label"].astype(str))
        out[m] = {k: {"n": int(len(v)), "answer_tokens": float(v["answer_length"].mean()),
                      "mean_logprob": float(v["mean_logprob"].mean()),
                      "primary_low_conf": float((v["judge_conf"] == "low").mean())} for k, v in g}
    return out


def prompt2():
    out = {}
    for m in MODELS:
        a = [json.loads(l)["answer"] for l in open(ROOT / "results_v3/prompt2" / m / "truthfulqa/confirm/greedy.jsonl",
                                                   encoding="utf-8")]
        b = [json.loads(l)["answer"] for l in open(V2_ROOT / m / "truthfulqa/confirm/greedy.jsonl", encoding="utf-8")]
        out[m] = {"identical_share": float(np.mean([x == y for x, y in zip(a, b)]))}
    return out


def check_docx():
    xml = zipfile.ZipFile(PAPER).read("word/document.xml").decode("utf-8")
    text = re.sub(r"<[^>]+>", "", re.sub(r"</w:p>", "\n", xml))
    assert all(x not in text for x in ["PLACEHOLDER", "undefined", "NaN", "${", "[@"]), "unfilled value in paper"
    body, refs = text.split("References\n", 1) if "References\n" in text else (text, "")
    cited = [int(n) for g in re.findall(r"\[(\d+(?:\], \[\d+)*)\]", body) for n in re.findall(r"\d+", g)]
    first = list(dict.fromkeys(cited))
    n_refs = len(re.findall(r"^\[\d+\]", refs, flags=re.M))
    assert first == list(range(1, len(first) + 1)), f"citations not in order of first use: {first}"
    assert len(first) == n_refs, f"{n_refs} references but {len(first)} cited"
    return {"n_references": n_refs, "ok": True}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    k = {"rejudge": rejudge(), "paraphrase": paraphrase_rejection(), "contamination": contamination(),
         "da1_original": da1(), "llama_untruncated": llama_untruncated(),
         "judge_disagreement": judge_disagreement(), "prompt2": prompt2()}
    (ROOT / "analysis_v3" / "constants.json").write_text(json.dumps(k, indent=1))
    print(json.dumps(k, indent=1))
    if a.check:
        print("paper check:", check_docx())
