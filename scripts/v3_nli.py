"""NLI-based self-consistency for every frozen v2 cell (exploratory, post-preregistration).

The preregistered D4/D5 cluster samples by MiniLM cosine >= 0.75. Semantic entropy as
published (Kuhn et al., 2023; Farquhar et al., 2024) clusters by bidirectional
entailment with an NLI model. This script implements that version on the same K=5
samples, so "self-consistency is worse than D0" can be checked against the
standard implementation rather than a cheap one.

  NLI          microsoft/deberta-large-mnli; premise/hypothesis = question + " " + answer
  equivalence  strict bidirectional entailment (argmax = ENTAILMENT both ways)
  clustering   greedy, each answer compared with the first member of each cluster
  scores       nli_se_discrete    entropy of cluster frequencies over the 5 samples
               nli_se_weighted    entropy of cluster masses, sample weight = exp(mean
                                  token log-prob) (length-normalised likelihood)
               nli_greedy_support fraction of samples equivalent to the greedy answer

`--types-only` instead writes an automatic TruthfulQA error type (does the answer entail a
listed incorrect / correct answer?), an error typing independent of the Claude judge.

Output: results_v3/<model>/<dataset>/<split>/nli_sc.csv, results_v3/<m>/truthfulqa/<s>/nli_type.csv

Usage:
    python scripts/v3_nli.py --nli C:/Stuff/hf_models/deberta-large-mnli
    python scripts/v3_nli.py --nli C:/Stuff/hf_models/deberta-large-mnli --types-only
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from scripts.v2_generate import V2_ROOT, load_split
from scripts.v2_labels import MODELS
from scripts.v3_ptrue import V3_ROOT


class NLI:
    def __init__(self, path):
        self.tok = AutoTokenizer.from_pretrained(path)
        self.model = AutoModelForSequenceClassification.from_pretrained(path, dtype=torch.float32).cuda().eval()
        self.ent = {v.upper(): k for k, v in self.model.config.id2label.items()}["ENTAILMENT"]
        self.cache = {}
        # the classifier head must have loaded: paraphrase entails, contradiction does not
        assert self.entails([("Water boils at 100 degrees Celsius.", "At 100 degrees Celsius, water boils."),
                             ("Paris is the capital of France.", "Paris is not the capital of France.")]) == [True, False]
        self.cache = {}

    @torch.no_grad()
    def entails(self, pairs, bs=64):
        todo = [p for p in dict.fromkeys(pairs) if p not in self.cache]
        for i in range(0, len(todo), bs):
            b = todo[i:i + bs]
            enc = self.tok([p for p, _ in b], [h for _, h in b], return_tensors="pt", padding=True,
                           truncation=True, max_length=256).to("cuda")
            pred = self.model(**enc).logits.argmax(-1).tolist()
            self.cache.update({p: x == self.ent for p, x in zip(b, pred)})
        return [self.cache[p] for p in pairs]


def equivalent_matrix(nli, texts):
    n = len(texts)
    pairs = [(texts[i], texts[j]) for i in range(n) for j in range(n) if i != j]
    e = dict(zip(pairs, nli.entails(pairs)))
    return np.array([[i == j or (e[(texts[i], texts[j])] and e[(texts[j], texts[i])]) for j in range(n)]
                     for i in range(n)])


def clusters(eq):
    ids = [-1] * len(eq)
    reps = []
    for i in range(len(eq)):
        for c, r in enumerate(reps):
            if eq[i, r]:
                ids[i] = c
                break
        else:
            ids[i] = len(reps)
            reps.append(i)
    return np.array(ids)


def entropy(p):
    p = p[p > 0] / p.sum()
    return float(-(p * np.log(p)).sum())


def run_cell(nli, m, d, s):
    cell = V2_ROOT / m / d / s
    q = load_split(d, s).set_index("qid")["question"]
    greedy = [json.loads(l) for l in open(cell / "greedy.jsonl", encoding="utf-8")]
    samples = {}
    for l in open(cell / "samples.jsonl", encoding="utf-8"):
        r = json.loads(l)
        samples.setdefault(r["qid"], []).append(r)
    # one batched NLI pass over every pair of the cell first, then cheap lookups
    allpairs = []
    texts = {}
    for g in greedy:
        t = [f"{q[g['qid']]} {x['answer']}" for x in [g] + samples[g["qid"]]]
        texts[g["qid"]] = t
        allpairs += [(t[i], t[j]) for i in range(len(t)) for j in range(len(t)) if i != j]
    nli.entails(allpairs)
    rows = []
    for g in greedy:
        eq = equivalent_matrix(nli, texts[g["qid"]])
        smp = samples[g["qid"]]
        c = clusters(eq[1:, 1:])
        w = np.array([np.exp(np.mean(x["logprobs"])) if x["logprobs"] else 0.0 for x in smp])
        rows.append({"qid": g["qid"],
                     "nli_se_discrete": entropy(np.bincount(c).astype(float)),
                     "nli_se_weighted": entropy(np.bincount(c, weights=w)) if w.sum() > 0 else 0.0,
                     "nli_greedy_support": float(eq[0, 1:].mean()),
                     "nli_n_clusters": int(c.max() + 1)})
    out = V3_ROOT / m / d / s
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / "nli_sc.csv", index=False)
    print(f"{m}/{d}/{s}: n={len(rows)} mean clusters={np.mean([r['nli_n_clusters'] for r in rows]):.2f}", flush=True)


def type_cell(nli, m, s, root=V2_ROOT, out_root=V3_ROOT, kind="greedy"):
    """Automatic error type for TruthfulQA greedy answers, independent of the Claude judge:
    nli_imitative = the answer entails one of the dataset's listed incorrect answers;
    nli_entails_correct = it entails one of the listed correct answers (both question-prefixed,
    one direction: answer => reference)."""
    q = load_split("truthfulqa", s).set_index("qid")
    rows = []
    for g in (json.loads(l) for l in open(Path(root) / m / "truthfulqa" / s / f"{kind}.jsonl", encoding="utf-8")):
        r = q.loc[g["qid"]]
        prem = f"{r['question']} {g['answer']}"
        inc = nli.entails([(prem, f"{r['question']} {x}") for x in r["incorrect_answers"]])
        cor = nli.entails([(prem, f"{r['question']} {x}") for x in dict.fromkeys([r["best_answer"], *r["correct_answers"]])])
        rows.append({"qid": g["qid"], **({"k": g["k"]} if "k" in g else {}),
                     "nli_imitative": int(any(inc)), "nli_entails_correct": int(any(cor))})
    out = Path(out_root) / m / "truthfulqa" / s
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / ("nli_type.csv" if kind == "greedy" else "nli_type_samples.csv"), index=False)
    print(f"{m}/truthfulqa/{s}: imitative share {np.mean([r['nli_imitative'] for r in rows]):.3f}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nli", required=True)
    ap.add_argument("--types-only", action="store_true")
    ap.add_argument("--root", default=str(V2_ROOT), help="--types-only: where greedy.jsonl files live")
    ap.add_argument("--out", default=str(V3_ROOT), help="--types-only: output root")
    ap.add_argument("--splits", nargs="+", default=["dev", "confirm"])
    ap.add_argument("--samples", action="store_true", help="--types-only: type the K sampled answers instead")
    ap.add_argument("--models", nargs="+", default=MODELS)
    a = ap.parse_args()
    nli = NLI(a.nli)
    if a.types_only:
        for m in a.models:
            for s in a.splits:
                kind = "samples" if a.samples else "greedy"
                if (Path(a.root) / m / "truthfulqa" / s / f"{kind}.jsonl").exists():
                    type_cell(nli, m, s, a.root, a.out, kind)
        return
    for m in MODELS:
        for d in ["truthfulqa", "sciq"]:
            for s in ["dev", "confirm"]:
                run_cell(nli, m, d, s)


if __name__ == "__main__":
    main()
