"""Recompute token-level uncertainty features from the model's RAW conditional
distribution p(y_t | x, y_<t), instead of from generate()'s returned scores.

Why this exists: HuggingFace applies logits processors during generation even
under greedy decoding. Qwen2.5-1.5B-Instruct ships repetition_penalty = 1.1 in
its generation_config, so the scores returned by generate(output_scores=True)
-- and therefore every log-probability and entropy cached by
backfill_rich_token_details.py -- describe the penalty-adjusted decoding
distribution, not the model's conditional distribution. Differences reach ~1.9
nats on individual tokens and ~0.26 on an answer's mean log-probability, which
is larger than the between-class separation of that feature. Llama-3.2 and
SmolLM2 set no repetition penalty and are unaffected (verified with
scripts/verify_token_alignment.py).

This script does NOT regenerate answers or relabel anything. It reproduces the
frozen greedy answer (asserting the text matches), then runs one teacher-forcing
forward pass over [prompt || answer] and recomputes log-probabilities, entropies
and top-1/top-2 margins from the unmodified logits.

Outputs, alongside the originals (which are left untouched):
    features/single_pass_token_details_clean.jsonl
    features/single_pass_features_clean.csv

Usage:
    python scripts/recompute_clean_features.py --model qwen2.5-1.5b-instruct --dataset truthfulqa
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm

from src.config import CFG, set_all_seeds
from src.datasets import load_subset
from src.features_v2 import extract_rich_features
from src.model_utils import LLMGenerator
from src.registry import get_paths
from src.run_logger import log_event


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--dataset", default="truthfulqa")
    ap.add_argument("--subset-size", type=int, default=200)
    args = ap.parse_args()

    set_all_seeds()
    paths = get_paths(args.model, args.dataset)
    sp = pd.read_csv(paths.features / "single_pass_features.csv")
    df = load_subset(args.dataset, args.subset_size, CFG.seed)
    qmap = df.set_index("qid")["question"].to_dict()

    log_event("recompute_clean_features", args.model, args.dataset, "started", n=len(sp))
    gen = LLMGenerator.from_registry(args.model)
    tok, model = gen.tokenizer, gen.model
    print(f"Model on {gen.device}; recomputing features from raw conditional distribution")

    details_path = paths.features / "single_pass_token_details_clean.jsonl"
    records, mismatches = [], []
    with torch.no_grad(), open(details_path, "w", encoding="utf-8") as fh:
        for _, row in tqdm(sp.iterrows(), total=len(sp), desc=f"Clean features [{args.model}/{args.dataset}]"):
            prompt = gen._build_prompt(qmap[row["qid"]])
            inputs = tok(prompt, return_tensors="pt").to(gen.device)
            plen = inputs["input_ids"].shape[1]

            out = model.generate(**inputs, max_new_tokens=CFG.single_pass_max_new_tokens,
                                 do_sample=False, pad_token_id=tok.pad_token_id,
                                 return_dict_in_generate=True)
            new_ids = out.sequences[0][plen:]
            keep = []
            for tid in new_ids.tolist():
                if tid in gen.stop_ids:
                    break
                keep.append(tid)
            if not keep:
                mismatches.append(row["qid"])
                continue
            if tok.decode(keep, skip_special_tokens=True).strip() != row["generated_answer"]:
                mismatches.append(row["qid"])
                continue

            full = torch.cat([inputs["input_ids"][0], torch.tensor(keep, device=gen.device)]).unsqueeze(0)
            logits = model(full).logits[0].float()
            lps, ents, margins = [], [], []
            for t, tid in enumerate(keep):
                lp = F.log_softmax(logits[plen - 1 + t], dim=-1)
                lps.append(lp[tid].item())
                p = lp.exp()
                ents.append(-(p * lp).sum().item())
                top2 = torch.topk(lp, k=2)
                margins.append((top2.values[0] - top2.values[1]).item())

            fh.write(json.dumps({"qid": row["qid"], "token_logprobs": lps,
                                 "token_entropies": ents, "token_margins": margins}) + "\n")
            feats = extract_rich_features(lps, ents, margins)
            records.append({"qid": row["qid"], "question": row["question"], "category": row.get("category"),
                            "generated_answer": row["generated_answer"], "label": row["label"],
                            "correct_sim": row.get("correct_sim"), "incorrect_sim": row.get("incorrect_sim"),
                            **feats})

    out_df = pd.DataFrame.from_records(records)
    out_path = paths.features / "single_pass_features_clean.csv"
    out_df.to_csv(out_path, index=False)
    if mismatches:
        print(f"WARNING: {len(mismatches)} question(s) did not reproduce the frozen answer and were skipped.")
    print(f"Saved {len(out_df)}/{len(sp)} rows -> {out_path}")

    old = pd.read_csv(paths.features / "single_pass_features_rich.csv").set_index("qid")
    new = out_df.set_index("qid")
    common = old.index.intersection(new.index)
    print("\nFeature shift vs. generate()-derived features (mean |delta|):")
    for f in ["mean_logprob", "min_logprob", "mean_entropy", "median_logprob", "logprob_trend_slope"]:
        d = (old.loc[common, f] - new.loc[common, f]).abs()
        print(f"  {f:26s} mean={d.mean():.4f}  max={d.max():.4f}")
    log_event("recompute_clean_features", args.model, args.dataset, "completed",
              n_ok=len(out_df), n_mismatch=len(mismatches))


if __name__ == "__main__":
    main()
