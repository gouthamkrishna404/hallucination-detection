"""Labels for the clean-protocol (v2) answers.

  auto     token-F1 and embedding proxy labels for every greedy answer and every
           sample; on SciQ also the containment label (correct short answer
           present as a token subsequence, no distractor present).
  batches  blind judge worksheets: the unique (question, answer) pairs of one
           dataset/split, pooled over the three models, shuffled, model names
           hidden, no detector features shown.
  merge    join the completed judge labels back onto (model, qid).

Judge rubric: labels/JUDGE_RUBRIC.md. Labels: 1 = hallucinated, 0 = correct.

Usage:
    python scripts/v2_labels.py auto
    python scripts/v2_labels.py batches --dataset truthfulqa --split dev
    python scripts/v2_labels.py merge --dataset truthfulqa --split dev
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from src.config import PROJECT_ROOT
from src.labeling import _normalize, label_answer
from scripts.v2_generate import V2_ROOT, load_split

MODELS = ["qwen2.5-1.5b-instruct", "llama-3.2-1b-instruct", "smollm2-1.7b-instruct"]
LABELS = PROJECT_ROOT / "labels"
BATCH = 40


def contains(tokens, ref):
    r = _normalize(ref)
    return bool(r) and any(tokens[i:i + len(r)] == r for i in range(len(tokens) - len(r) + 1))


def containment(answer, q):
    toks = _normalize(answer)
    has_c = any(contains(toks, r) for r in [q["best_answer"], *q["correct_answers"]])
    has_d = any(contains(toks, r) for r in q["incorrect_answers"])
    return int(not (has_c and not has_d)), bool(has_c and has_d)


def embed_labels(answers, qrows):
    """Embedding-proxy label, batched: max cosine to correct refs vs incorrect refs."""
    from src.semantic_similarity import embed
    a = embed(answers)
    out = []
    cache = {}
    for i, q in enumerate(qrows):
        if q["qid"] not in cache:
            cache[q["qid"]] = (embed(list(dict.fromkeys([q["best_answer"], *q["correct_answers"]]))),
                               embed(list(q["incorrect_answers"])))
        c, w = cache[q["qid"]]
        cs, ws = float((c @ a[i]).max()), float((w @ a[i]).max())
        out.append((int(not (cs - ws > 0)), cs, ws))
    return out


def label_frame(recs, qmap, dataset):
    qrows = [qmap[r["qid"]] for r in recs]
    rows = []
    emb = embed_labels([r["answer"] or " " for r in recs], qrows)
    for r, q, e in zip(recs, qrows, emb):
        f1 = label_answer(r["answer"], q["best_answer"], q["correct_answers"], q["incorrect_answers"])
        row = {"qid": r["qid"], "tokenf1_label": f1["label"], "f1_correct_sim": f1["correct_sim"],
               "f1_incorrect_sim": f1["incorrect_sim"], "embed_label": e[0], "embed_correct_sim": e[1],
               "embed_incorrect_sim": e[2]}
        if "k" in r:
            row["k"] = r["k"]
        if dataset == "sciq":
            row["containment_label"], row["containment_ambiguous"] = containment(r["answer"], q)
        rows.append(row)
    return pd.DataFrame(rows)


def read_jsonl(p):
    with open(p, encoding="utf-8") as f:
        return [json.loads(l) for l in f]


def cmd_auto(_):
    for d in ["truthfulqa", "sciq"]:
        for s in ["dev", "confirm"]:
            qmap = {r["qid"]: r for r in load_split(d, s).to_dict("records")}
            for m in MODELS:
                cell = V2_ROOT / m / d / s
                if not (cell / "meta.json").exists():
                    continue
                for kind in ["greedy", "samples"]:
                    out = cell / f"labels_auto_{kind}.csv"
                    if out.exists() or not (cell / f"{kind}.jsonl").exists():
                        continue
                    label_frame(read_jsonl(cell / f"{kind}.jsonl"), qmap, d).to_csv(out, index=False)
                    print("wrote", out)


def item_id(question, answer):
    return hashlib.sha1(f"{question}\x00{answer}".encode()).hexdigest()[:10]


def cmd_batches(a):
    qmap = {r["qid"]: r for r in load_split(a.dataset, a.split).to_dict("records")}
    d = LABELS / "judge" / f"{a.dataset}_{a.split}"
    d.mkdir(parents=True, exist_ok=True)
    # incremental: items already written to a worksheet are never re-issued
    seen = {l[4:].strip() for b in d.glob("batch_*.txt") for l in b.read_text(encoding="utf-8").splitlines()
            if l.startswith("### ")}
    items = {}
    for m in MODELS:
        path = V2_ROOT / m / a.dataset / a.split / "greedy.jsonl"
        if not path.exists():
            continue
        recs = read_jsonl(path)
        if a.sample and a.dataset == "sciq":
            rng = np.random.RandomState(7)
            recs = [recs[i] for i in rng.choice(len(recs), a.sample, replace=False)]
        for r in recs:
            q = qmap[r["qid"]]
            items.setdefault(item_id(q["question"], r["answer"]), (q, r["answer"]))
    keys = sorted(set(items) - seen)
    np.random.RandomState(len(seen)).shuffle(keys)
    start = len(list(d.glob("batch_*.txt")))
    for b in range(0, len(keys), BATCH):
        lines = []
        for k in keys[b:b + BATCH]:
            q, ans = items[k]
            correct = list(dict.fromkeys([q["best_answer"], *q["correct_answers"]]))
            lines.append(f"### {k}\nQ: {q['question']}\nCORRECT: {' | '.join(correct)}\n"
                         f"INCORRECT: {' | '.join(q['incorrect_answers'])}\nANSWER: {ans}\n")
        (d / f"batch_{start + b // BATCH:03d}.txt").write_text("\n".join(lines), encoding="utf-8")
    print(f"{len(keys)} unique items -> {(len(keys) + BATCH - 1) // BATCH} batches in {d}")


def cmd_merge(a):
    d = LABELS / "judge" / f"{a.dataset}_{a.split}"
    lab = pd.concat([pd.read_csv(p, dtype={"item": str, "type": str}) for p in sorted(d.glob("labels_*.csv"))])
    lab = lab.drop_duplicates("item", keep="last").set_index("item")
    qmap = {r["qid"]: r for r in load_split(a.dataset, a.split).to_dict("records")}
    for m in MODELS:
        if not (V2_ROOT / m / a.dataset / a.split / "greedy.jsonl").exists():
            continue
        recs = read_jsonl(V2_ROOT / m / a.dataset / a.split / "greedy.jsonl")
        rows = []
        for r in recs:
            k = item_id(qmap[r["qid"]]["question"], r["answer"])
            if k in lab.index:
                rows.append({"qid": r["qid"], "judge_label": int(lab.loc[k, "label"]),
                             "judge_conf": lab.loc[k, "conf"], "judge_type": lab.loc[k, "type"]})
        out = V2_ROOT / m / a.dataset / a.split / "labels_judge_greedy.csv"
        pd.DataFrame(rows).to_csv(out, index=False)
        print(f"{m}: {len(rows)}/{len(recs)} judged -> {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("auto")
    for name in ["batches", "merge"]:
        p = sub.add_parser(name)
        p.add_argument("--dataset", required=True)
        p.add_argument("--split", required=True)
        p.add_argument("--sample", type=int, default=0, help="SciQ only: judge a random subset per model")
    a = ap.parse_args()
    {"auto": cmd_auto, "batches": cmd_batches, "merge": cmd_merge}[a.cmd](a)
