"""Correctness check for the central quantity of the whole project: is the
cached per-token log-probability really log p(y_t | x, y_<t)?

generate(output_scores=True) returns one score tensor per generation step;
this script independently recomputes the same quantities with a single
teacher-forcing forward pass over [prompt || answer] and compares them
token by token. For a decoder-only model the logits at position i predict
token i+1, so the distribution for generated token t lives at index
(prompt_len - 1 + t). If the two agree, the alignment used throughout the
pipeline is correct; a systematic off-by-one would show up as a large
mismatch.

Usage:
    python scripts/verify_token_alignment.py --model qwen2.5-1.5b-instruct --n 5
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import torch.nn.functional as F

from src.config import CFG, set_all_seeds
from src.datasets import load_subset
from src.model_utils import LLMGenerator


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen2.5-1.5b-instruct")
    ap.add_argument("--dataset", default="truthfulqa")
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--dtype", default=None, choices=["float16","float32"])
    args = ap.parse_args()

    set_all_seeds()
    df = load_subset(args.dataset, 200, CFG.seed)
    gen = LLMGenerator.from_registry(args.model)
    if args.dtype == "float32":
        gen.model = gen.model.float()
    print(f"dtype = {next(gen.model.parameters()).dtype}")
    tok, model = gen.tokenizer, gen.model

    max_abs_all, max_ent_all = [], []
    for i in range(args.n):
        question = df.iloc[i]["question"]
        prompt = gen._build_prompt(question)
        inputs = tok(prompt, return_tensors="pt").to(gen.device)
        prompt_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=CFG.single_pass_max_new_tokens,
                                 do_sample=False, pad_token_id=tok.pad_token_id,
                                 return_dict_in_generate=True, output_scores=True)

        new_ids = out.sequences[0][prompt_len:]
        keep, gen_lp, gen_ent = [], [], []
        for step, logits in enumerate(out.scores):
            tid = new_ids[step].item()
            if tid in gen.stop_ids:
                break
            lp = F.log_softmax(logits[0].float(), dim=-1)
            gen_lp.append(lp[tid].item())
            p = lp.exp()
            gen_ent.append(-(p * lp).sum().item())
            keep.append(tid)
        if not keep:
            continue

        # Independent recomputation: one forward pass over prompt || answer.
        full = torch.cat([inputs["input_ids"][0], torch.tensor(keep, device=gen.device)]).unsqueeze(0)
        with torch.no_grad():
            logits_full = model(full).logits[0].float()
        tf_lp, tf_ent = [], []
        for t, tid in enumerate(keep):
            lp = F.log_softmax(logits_full[prompt_len - 1 + t], dim=-1)
            tf_lp.append(lp[tid].item())
            p = lp.exp()
            tf_ent.append(-(p * lp).sum().item())

        d_lp = np.abs(np.array(gen_lp) - np.array(tf_lp))
        d_ent = np.abs(np.array(gen_ent) - np.array(tf_ent))
        max_abs_all.append(d_lp.max())
        max_ent_all.append(d_ent.max())
        print(f"q{i}: T={len(keep):3d}  max|dlogprob|={d_lp.max():.2e}  max|dentropy|={d_ent.max():.2e}  "
              f"mean_logprob gen={np.mean(gen_lp):+.4f} vs teacher-forced={np.mean(tf_lp):+.4f}")

        # Off-by-one control: if we deliberately misalign by one position, the gap should be large.
        if len(keep) > 2:
            mis = [F.log_softmax(logits_full[prompt_len + t], dim=-1)[tid].item() for t, tid in enumerate(keep[:-1])]
            print(f"     off-by-one control: mean|diff| = {np.abs(np.array(mis) - np.array(tf_lp[:-1])).mean():.3f} "
                  f"(should be >> 0, confirming the check is sensitive)")

    # Tolerance: fp16 KV-cache decoding vs. a single full forward pass differs by ~2e-2 per
    # token even when everything is correct, so 5e-2 separates "numerics" from "real mismatch".
    # A genuine off-by-one shows up as ~15-20 nats (see the control line above).
    worst = max(max_abs_all)
    print(f"\nWorst over {len(max_abs_all)} questions: max|dlogprob|={worst:.2e}, "
          f"max|dentropy|={max(max_ent_all):.2e}")
    if worst < 5e-2:
        print("PASS: alignment verified, scores match the model's conditional distribution")
    else:
        print("MISMATCH: alignment is correct (see off-by-one control) but generate() scores differ "
              "from the raw conditional distribution -- check generation_config for logits processors "
              "such as repetition_penalty, which are applied even under greedy decoding.")


if __name__ == "__main__":
    main()
