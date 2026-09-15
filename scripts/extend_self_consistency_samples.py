"""Generate 5 MORE self-consistency samples per question (indices 5-9),
continuing the exact same seeded sampling scheme as
experiment_generate_self_consistency.py, to support the 1/3/5/10-sample
budget sweep (scripts/experiment_sampling_sweep.py) without invalidating
the existing 5-sample cache -- the first 5 samples are untouched, these
are purely additive.

Usage:
    python scripts/extend_self_consistency_samples.py --model qwen2.5-1.5b-instruct --dataset truthfulqa
"""
import argparse
import ast
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from tqdm import tqdm

from src.config import CFG, set_all_seeds
from src.datasets import load_subset
from src.model_utils import LLMGenerator
from src.registry import get_paths
from src.run_logger import log_event

N_EXTRA = 5  # samples 5..9, extending the existing 0..4


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--dataset", default="truthfulqa")
    ap.add_argument("--subset-size", type=int, default=200)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    set_all_seeds()
    paths = get_paths(args.model, args.dataset)
    out_path = paths.features / "self_consistency_extra_samples.csv"

    if out_path.exists() and not args.force:
        print(f"Already exists, skipping (use --force to regenerate): {out_path}")
        return

    sc = pd.read_csv(paths.features / "self_consistency_features.csv")
    df = load_subset(args.dataset, args.subset_size, CFG.seed)
    question_map = df.set_index("qid")["question"].to_dict()
    qid_to_index = {qid: i for i, qid in enumerate(df["qid"])}

    log_event("extend_self_consistency_samples", args.model, args.dataset, "started")
    generator = LLMGenerator.from_registry(args.model)
    print(f"Model loaded on device: {generator.device}")

    records = []
    t0 = time.time()
    for _, row in tqdm(sc.iterrows(), total=len(sc), desc=f"Extra samples [{args.model}/{args.dataset}]"):
        qid = row["qid"]
        q_idx = qid_to_index[qid]
        question = question_map[qid]
        extra_answers = []
        for k in range(N_EXTRA):
            sample_idx = CFG.self_consistency_n_samples + k  # continues 0..4 with 5..9
            result = generator.generate_with_scores(
                question=question,
                max_new_tokens=CFG.self_consistency_max_new_tokens,
                do_sample=True,
                temperature=CFG.self_consistency_temperature,
                top_p=CFG.self_consistency_top_p,
                seed=CFG.seed * 1000 + q_idx * 10 + sample_idx,  # 10 = max total samples planned
            )
            extra_answers.append(result.answer_text)
        records.append({"qid": qid, "extra_answers": extra_answers})

    elapsed = time.time() - t0
    out_df = pd.DataFrame.from_records(records)
    out_df.to_csv(out_path, index=False)
    print(f"Saved {len(out_df)} rows -> {out_path}")
    print(f"Elapsed: {elapsed:.1f}s ({elapsed/len(out_df)/N_EXTRA:.2f}s/call, {N_EXTRA*len(out_df)} total calls)")
    log_event("extend_self_consistency_samples", args.model, args.dataset, "completed", n_questions=len(out_df))


if __name__ == "__main__":
    main()
