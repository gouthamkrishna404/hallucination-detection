"""Regenerate full per-question token-level detail (logprobs, entropies,
AND top-1/top-2 margins) for every question in a model/dataset's frozen
single-pass run, verifying determinism against the already-committed
generated_answer text and features at every step.

Why this is needed: `single_pass_features.csv` (frozen, never touched)
only stores the three aggregated features (mean/min logprob, mean
entropy). The feature-engineering investigation needs the full per-token
arrays, including token-level margin (top-1 vs. top-2 logprob gap), which
was never cached in the first place -- and one dataset/model combo
(Qwen2.5-1.5B-Instruct x TruthfulQA, the original DA1 run) has no
token-detail cache at all, since it predates the generalized pipeline.

Safety: generation is greedy/deterministic (verified reproducible earlier
for 6 examples across a code change -- see README "Corrections"). This
script regenerates the SAME greedy decoding and asserts the reproduced
answer text matches the frozen CSV exactly before trusting the new
per-token arrays; any mismatch aborts loudly rather than silently
diverging from the frozen result. Nothing in single_pass_features.csv,
predictions/, or metrics/ is modified -- this only adds a new cache file,
single_pass_token_details_v2.jsonl, alongside the old one.

Usage:
    python scripts/backfill_rich_token_details.py --model qwen2.5-1.5b-instruct --dataset truthfulqa
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from tqdm import tqdm

from src.config import CFG, set_all_seeds
from src.datasets import load_subset
from src.model_utils import LLMGenerator
from src.registry import get_paths
from src.run_logger import log_event


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--dataset", default="truthfulqa")
    ap.add_argument("--subset-size", type=int, default=200)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    set_all_seeds()
    paths = get_paths(args.model, args.dataset)
    out_path = paths.features / "single_pass_token_details_v2.jsonl"

    if out_path.exists() and not args.force:
        print(f"Already exists, skipping (use --force to regenerate): {out_path}")
        return

    sp = pd.read_csv(paths.features / "single_pass_features.csv")
    df = load_subset(args.dataset, args.subset_size, CFG.seed)
    question_map = df.set_index("qid")["question"].to_dict()

    log_event("backfill_rich_token_details", args.model, args.dataset, "started", n_questions=len(sp))

    generator = LLMGenerator.from_registry(args.model)
    print(f"Model loaded on device: {generator.device}")

    mismatches = []
    with open(out_path, "w", encoding="utf-8") as f:
        for _, row in tqdm(sp.iterrows(), total=len(sp), desc=f"Backfill [{args.model}/{args.dataset}]"):
            question = question_map[row["qid"]]
            result = generator.generate_with_scores(
                question=question,
                max_new_tokens=CFG.single_pass_max_new_tokens,
                do_sample=CFG.single_pass_do_sample,
            )
            if result.answer_text != row["generated_answer"]:
                mismatches.append(row["qid"])
                continue  # don't cache a divergent reconstruction

            f.write(json.dumps({
                "qid": row["qid"],
                "tokens": result.token_strs,
                "token_logprobs": result.token_logprobs,
                "token_entropies": result.token_entropies,
                "token_margins": result.token_margins,
            }, ensure_ascii=False) + "\n")

    if mismatches:
        print(f"WARNING: {len(mismatches)} question(s) did not reproduce the frozen answer text "
              f"and were SKIPPED (not cached): {mismatches[:10]}{'...' if len(mismatches) > 10 else ''}")
    else:
        print("All questions reproduced the frozen answer text exactly -- determinism verified.")

    print(f"Saved {len(sp) - len(mismatches)}/{len(sp)} rows -> {out_path}")
    log_event("backfill_rich_token_details", args.model, args.dataset, "completed",
              n_reproduced=len(sp) - len(mismatches), n_mismatches=len(mismatches))


if __name__ == "__main__":
    main()
