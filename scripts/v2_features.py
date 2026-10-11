"""Feature table for each clean-protocol (v2) cell.

One row per question, describing the frozen greedy answer:
  * the 17 single-pass uncertainty features (src/features_v2.py), computed from
    the fp32 teacher-forced log-probabilities, entropies and top-1/top-2 margins;
  * controls: truncated (hit the 64-token cap), question_mean_logprob and
    question_n_tokens (properties of the question, not the answer);
  * self-consistency scores from the K=5 samples (5 extra LLM calls):
      sc_lexical_agreement  mean pairwise token-F1 among the samples (the
                            original DA1 baseline score)
      sc_semantic_entropy   entropy of cosine-clustered samples
      sc_greedy_support     fraction of samples semantically matching the greedy
                            answer (cosine >= 0.75) -- the self-consistency
                            score aimed at the answer actually being judged
  * every available label column.

Usage:
    python scripts/v2_features.py
"""
import json
import sys
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from src.features_v2 import extract_rich_features
from src.labeling import token_f1
from src.semantic_similarity import _CLUSTER_SIMILARITY_THRESHOLD as TAU, embed
from scripts.v2_generate import V2_ROOT
from scripts.v2_labels import MODELS, read_jsonl


def clusters(sims):
    ids = [-1] * len(sims)
    nxt = 0
    for i in range(len(sims)):
        if ids[i] == -1:
            ids[i] = nxt
            for j in range(i + 1, len(sims)):
                if ids[j] == -1 and sims[i, j] >= TAU:
                    ids[j] = nxt
            nxt += 1
    return ids


def sc_features(greedy, samples):
    by_q = {}
    for s in samples:
        by_q.setdefault(s["qid"], []).append(s["answer"])
    texts = [g["answer"] or " " for g in greedy] + [a or " " for g in greedy for a in by_q[g["qid"]]]
    e = embed(texts)
    n, rows = len(greedy), []
    for i, g in enumerate(greedy):
        answers = by_q[g["qid"]]
        k = len(answers)
        se = e[n + i * k:n + (i + 1) * k]
        counts = np.bincount(clusters(se @ se.T))
        p = counts / counts.sum()
        rows.append({
            "qid": g["qid"],
            "sc_lexical_agreement": float(np.mean([token_f1(answers[a], answers[b])
                                                   for a, b in combinations(range(k), 2)])),
            "sc_semantic_entropy": float(-(p * np.log(p)).sum()),
            "sc_greedy_support": float(((se @ e[i]) >= TAU).mean()),
        })
    return pd.DataFrame(rows)


def build(cell: Path):
    greedy = read_jsonl(cell / "greedy.jsonl")
    rows = []
    for g in greedy:
        f = extract_rich_features(g["logprobs"], g["entropies"], g["margins"])
        rows.append({"qid": g["qid"], "answer": g["answer"], "truncated": int(g["truncated"]), **f})
    df = pd.DataFrame(rows)
    df = df.merge(pd.read_json(cell / "questions.jsonl", lines=True), on="qid")
    if (cell / "samples.jsonl").exists():
        df = df.merge(sc_features(greedy, read_jsonl(cell / "samples.jsonl")), on="qid")
    for name in ["labels_auto_greedy.csv", "labels_judge_greedy.csv"]:
        if (cell / name).exists():
            df = df.merge(pd.read_csv(cell / name), on="qid", how="left")
    df.to_csv(cell / "features.csv", index=False)
    return df


if __name__ == "__main__":
    for m in MODELS:
        for d in ["truthfulqa", "sciq"]:
            for s in ["dev", "confirm"]:
                cell = V2_ROOT / m / d / s
                if (cell / "meta.json").exists():
                    df = build(cell)
                    print(f"{m}/{d}/{s}: {len(df)} rows, {df.shape[1]} cols")
