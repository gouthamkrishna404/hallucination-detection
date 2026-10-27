"""Run the whole method live on a few held-out questions (for a review / viva).

Uses the experiment's own functions (scripts/v2_generate.py), so this is the real
pipeline, not a re-implementation:

  1. load held-out (confirm) questions
  2. load the model in fp32
  3. raw greedy decoding (no repetition penalty, no sampling)
  4. teacher-forced fp32 re-scoring -> per-token log-probabilities and entropies
  5. detector D0 = -mean token log-prob (higher = more likely hallucinated)
  6. compare with the recorded experiment: same answer? judge label?
  7. AUROC of D0 on these questions

Nothing is written to disk.

Usage:
    python scripts/live_run.py                                   # Qwen, SciQ, 200 questions, ~1 min
    python scripts/live_run.py --dataset truthfulqa
    python scripts/live_run.py --model llama-3.2-1b-instruct
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import torch

from scripts.v2_generate import V2_ROOT, generate, load_split, record
from src.config import set_all_seeds
from src.model_utils import LLMGenerator
from src.v2_stats import auroc, boot_ci


def step(n, text):
    print(f"\n[{n}] {text}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen2.5-1.5b-instruct",
                    choices=["qwen2.5-1.5b-instruct", "llama-3.2-1b-instruct", "smollm2-1.7b-instruct"])
    ap.add_argument("--dataset", default="sciq", choices=["sciq", "truthfulqa"])
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--show", type=int, default=8, help="rows shown from each end of the ranking")
    ap.add_argument("--bs", type=int, default=8)
    a = ap.parse_args()
    set_all_seeds()
    t0 = time.time()

    step(1, f"Load {a.n} held-out {a.dataset} questions (confirm split, never used for design decisions)")
    df = load_split(a.dataset, "confirm").head(a.n)
    print(f"    e.g. {df['question'].iloc[0]!r}")

    step(2, f"Load {a.model} in fp32 (src/model_utils.py: LLMGenerator)")
    gen = LLMGenerator.from_registry(a.model, dtype=torch.float32)
    prompts = [gen._build_prompt(q) for q in df["question"]]
    prompt_ids = [gen.tokenizer(p, add_special_tokens=False)["input_ids"] for p in prompts]

    step(3, "Raw greedy decoding, repetition_penalty=1.0 (scripts/v2_generate.py: generate)")
    raw = generate(gen, prompts, do_sample=False, bs=a.bs)

    step(4, "Teacher-forced fp32 re-scoring of each frozen answer (scripts/v2_generate.py: record)")
    recs = [record(gen, qid, pid, r) for qid, pid, r in zip(df["qid"], prompt_ids, raw)]
    ranks = [x for r in recs for x in r["ranks"]]
    print(f"    {len(ranks)} answer tokens; generated token == most probable token for "
          f"{100 * np.mean(np.array(ranks) == 0):.1f}% (must be 100%: proves the probabilities are the model's own)")

    step(5, "Detector D0 = -mean token log-prob")
    d0 = np.array([-np.mean(r["logprobs"]) if r["logprobs"] else 0.0 for r in recs])

    step(6, "Compare with the recorded experiment (results_v2/)")
    cell = V2_ROOT / a.model / a.dataset / "confirm"
    old = {json.loads(l)["qid"]: json.loads(l)["answer"] for l in open(cell / "greedy.jsonl", encoding="utf-8")}
    lab = pd.read_csv(cell / "labels_judge_greedy.csv").set_index("qid")
    rows = []
    for (_, q), r, s in zip(df.iterrows(), recs, d0):
        jl = lab.loc[r["qid"]]
        rows.append({"question": q["question"][:55], "answer": r["answer"][:60], "D0": round(s, 3),
                     "label": "WRONG" if jl["judge_label"] == 1 else "ok",
                     "type": jl["judge_type"] if jl["judge_label"] == 1 else "",
                     "same_as_recorded": r["answer"] == old[r["qid"]]})
    t = pd.DataFrame(rows).sort_values("D0", ascending=False)
    shown = t if len(t) <= 2 * a.show else pd.concat([t.head(a.show), t.tail(a.show)])
    print(f"    least confident (top) and most confident (bottom) {a.show} answers, ranked by D0:")
    with pd.option_context("display.max_colwidth", 60, "display.width", 250):
        print(shown.drop(columns="same_as_recorded").to_string(index=False))
    print(f"    live answer identical to the recorded experiment for {t['same_as_recorded'].sum()}/{len(t)} questions")
    if not t["same_as_recorded"].all():
        print("    WARNING: some answers differ from the recorded run; the AUROC below uses unchanged answers only")

    step(7, "How well does D0 separate wrong from correct answers here?")
    # the recorded label belongs to the recorded answer, so only unchanged answers can be scored
    t = t[t["same_as_recorded"]]
    y = (t["label"] == "WRONG").to_numpy()
    if 0 < y.sum() < len(y):
        lo, hi = boot_ci(y, t["D0"])
        print(f"    {y.sum()}/{len(y)} answers wrong; AUROC = {auroc(y, t['D0']):.3f} [95% CI {lo:.2f}, {hi:.2f}] "
              f"(0.5 = chance; the paper uses all 617 TruthfulQA / 800 SciQ held-out questions)")
    else:
        print("    all answers have the same label; AUROC undefined for this sample")
    print(f"\nDone in {time.time() - t0:.0f} s.")


if __name__ == "__main__":
    main()
