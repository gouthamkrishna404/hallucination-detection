"""Clean-protocol (v2) generation and scoring, used for every corrected result.

Protocol, identical for every model and dataset:

  decoding   greedy with every logits processor disabled (repetition_penalty
             forced to 1.0, no temperature/top-k/top-p), fp32, <=64 new tokens;
             plus K=5 samples at T=0.7, top-p=0.9 for the self-consistency
             baseline and the within-question analysis.
  scoring    answers are frozen as token ids, then re-scored with ONE fp32
             teacher-forced forward pass over [prompt || answer]. Every
             log-probability is log p(y_t | x, y_<t) under the unmodified model
             -- no generate() scores are used anywhere. For the greedy answer we
             also record whether the chosen token is the arg-max of the scored
             distribution, which verifies the decoding really was raw greedy.
  question   mean log-probability of the bare question text under the model
             (no chat template), used as a question-familiarity control.

Splits:
  dev      the original fixed 200-question subsets (data/<ds>/subset_200.jsonl).
           All exploratory analysis in this project was done on these.
  confirm  every remaining question of the same split (TruthfulQA 617, SciQ 800),
           never used before. Evaluated only under the pre-registered protocol
           in PREREGISTRATION.md.

Outputs: results_v2/<model>/<dataset>/<split>/{greedy,samples}.jsonl, questions.jsonl

Usage:
    python scripts/v2_generate.py --model llama-3.2-1b-instruct --dataset sciq --split dev
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm

from src.config import CFG, PROJECT_ROOT, set_all_seeds
from src.datasets import load_subset
from src.model_utils import LLMGenerator

V2_ROOT = PROJECT_ROOT / "results_v2"
MAX_NEW = 64
K_SAMPLES = 5
T_SAMPLE, TOP_P = 0.7, 0.9


def load_split(dataset: str, split: str) -> pd.DataFrame:
    dev = load_subset(dataset, 200, CFG.seed)
    if split == "dev":
        return dev
    path = PROJECT_ROOT / "data" / dataset / "confirm.jsonl"
    if path.exists():
        return pd.read_json(path, lines=True)
    from datasets import load_dataset
    if dataset == "truthfulqa":
        full = load_dataset("truthfulqa/truthful_qa", "generation", split="validation").to_pandas()
    else:
        full = load_dataset("allenai/sciq", split="test").to_pandas()
        full["best_answer"] = full["correct_answer"]
        full["correct_answers"] = full["correct_answer"].apply(lambda a: [a])
        full["incorrect_answers"] = full.apply(lambda r: [r.distractor1, r.distractor2, r.distractor3], axis=1)
        full["category"] = "science"
    full = full[~full["question"].isin(set(dev["question"]))].reset_index(drop=True)
    prefix = "tqa_c" if dataset == "truthfulqa" else "sciq_c"
    full["qid"] = [f"{prefix}_{i:04d}" for i in range(len(full))]
    full = full[["qid", "question", "best_answer", "correct_answers", "incorrect_answers", "category"]]
    with open(path, "w", encoding="utf-8") as f:
        for r in full.to_dict("records"):
            r["correct_answers"], r["incorrect_answers"] = list(r["correct_answers"]), list(r["incorrect_answers"])
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return full


def strip_stop(ids, stop_ids):
    out = []
    for t in ids:
        if t in stop_ids:
            return out, False
        out.append(t)
    return out, True  # no stop token within the budget -> truncated


@torch.no_grad()
def generate(gen, prompts, do_sample, n_return=1, bs=16):
    tok, model = gen.tokenizer, gen.model
    tok.padding_side = "left"
    outs = []
    for i in range(0, len(prompts), bs):
        enc = tok(prompts[i:i + bs], return_tensors="pt", padding=True, add_special_tokens=False).to(gen.device)
        kw = dict(max_new_tokens=MAX_NEW, pad_token_id=tok.pad_token_id, repetition_penalty=1.0,
                  num_return_sequences=n_return)
        if do_sample:
            kw.update(do_sample=True, temperature=T_SAMPLE, top_p=TOP_P, top_k=0)
        else:
            kw.update(do_sample=False, temperature=None, top_p=None, top_k=None)
        seq = model.generate(**enc, **kw)
        outs.extend(seq[:, enc["input_ids"].shape[1]:].tolist())
    return outs


@torch.no_grad()
def teacher_force(gen, prompt_ids, answer_ids):
    """fp32 forward over [prompt || answer]; per-token stats of the raw conditional."""
    if not answer_ids:
        return dict(logprobs=[], entropies=[], margins=[], ranks=[], top1_logprobs=[])
    full = torch.tensor([prompt_ids + answer_ids], device=gen.device)
    logits = gen.model(full).logits[0, len(prompt_ids) - 1:len(prompt_ids) - 1 + len(answer_ids)].float()
    lp = F.log_softmax(logits, dim=-1)
    ids = torch.tensor(answer_ids, device=gen.device)
    chosen = lp.gather(1, ids[:, None])[:, 0]
    ent = -(lp.exp() * lp).sum(-1)
    top2 = lp.topk(2, dim=-1).values
    ranks = (lp > chosen[:, None]).sum(-1)
    return dict(logprobs=chosen.tolist(), entropies=ent.tolist(), margins=(top2[:, 0] - top2[:, 1]).tolist(),
                ranks=ranks.tolist(), top1_logprobs=top2[:, 0].tolist())


@torch.no_grad()
def question_logprob(gen, question):
    tok = gen.tokenizer
    ids = tok(question, add_special_tokens=False)["input_ids"]
    bos = tok.bos_token_id if tok.bos_token_id is not None else tok.eos_token_id
    full = torch.tensor([[bos] + ids], device=gen.device)
    lp = F.log_softmax(gen.model(full).logits[0, :-1].float(), dim=-1)
    tl = lp.gather(1, full[0, 1:, None])[:, 0]
    return float(tl.mean()), len(ids)


def record(gen, qid, prompt_ids, raw_ids, k=None):
    ans_ids, truncated = strip_stop(raw_ids, gen.stop_ids)
    tf = teacher_force(gen, prompt_ids, ans_ids)
    rec = {"qid": qid, "answer": gen.tokenizer.decode(ans_ids, skip_special_tokens=True).strip(),
           "token_ids": ans_ids, "tokens": gen._token_pieces(ans_ids), "truncated": truncated, **tf}
    if k is not None:
        rec["k"] = k
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--dataset", required=True, choices=["truthfulqa", "sciq", "triviaqa"])
    ap.add_argument("--split", required=True, choices=["dev", "confirm"])
    ap.add_argument("--no-samples", action="store_true")
    # 8 GB GPUs cannot hold SmolLM2-1.7B in fp32 plus a sampling KV cache. The checkpoints are stored in
    # bf16, so fp32 weights are an exact upcast and the bf16 round trip is lossless; only sampling compute
    # runs in bf16, all scoring stays fp32.
    ap.add_argument("--sample-bf16", action="store_true")
    ap.add_argument("--reuse-greedy", action="store_true", help="keep an existing fp32 greedy.jsonl")
    # v3 prompt-sensitivity check only; the defaults reproduce the frozen v2 protocol exactly
    ap.add_argument("--system-prompt", default=None, help="override CFG.system_prompt")
    ap.add_argument("--root", default=str(V2_ROOT), help="output root")
    # v3 scale check only (Qwen2.5-3B does not fit an 8 GB GPU in fp32)
    ap.add_argument("--dtype", default="fp32", choices=["fp32", "bf16"])
    ap.add_argument("--checkpoint", default=None, help="local path instead of the registry checkpoint")
    ap.add_argument("--bs", type=int, default=16, help="greedy batch size (8 fits fp32 SmolLM2 in 8 GB)")
    args = ap.parse_args()

    set_all_seeds()
    df = load_split(args.dataset, args.split)
    out = Path(args.root) / args.model / args.dataset / args.split
    out.mkdir(parents=True, exist_ok=True)
    dtype = torch.float32 if args.dtype == "fp32" else torch.bfloat16
    gen = (LLMGenerator(args.checkpoint, dtype=dtype) if args.checkpoint
           else LLMGenerator.from_registry(args.model, dtype=dtype))
    tok = gen.tokenizer
    prompts = [gen._build_prompt(q, args.system_prompt) for q in df["question"]]
    prompt_ids = [tok(p, add_special_tokens=False)["input_ids"] for p in prompts]
    print(f"[{args.model}/{args.dataset}/{args.split}] n={len(df)} device={gen.device} dtype={args.dtype}")

    t0 = time.time()
    n_trunc, n_argmax, n_tok = 0, 0, 0
    reuse = args.reuse_greedy and (out / "greedy.jsonl").exists()
    if reuse:
        recs = [json.loads(l) for l in open(out / "greedy.jsonl", encoding="utf-8")]
        assert [r["qid"] for r in recs] == list(df["qid"]), "stale greedy.jsonl"
    else:
        raw = generate(gen, prompts, do_sample=False, bs=args.bs)
        recs = [record(gen, qid, pid, r) for qid, pid, r in
                tqdm(zip(df["qid"], prompt_ids, raw), total=len(df), desc="score greedy")]
        with open(out / "greedy.jsonl", "w", encoding="utf-8") as f:
            f.writelines(json.dumps(rec, ensure_ascii=False) + "\n" for rec in recs)
    for rec in recs:
        n_trunc += rec["truncated"]
        n_argmax += sum(x == 0 for x in rec["ranks"])
        n_tok += len(rec["ranks"])
    print(f"greedy done {time.time() - t0:.0f}s; truncated={n_trunc}/{len(df)}; "
          f"chosen==argmax for {n_argmax}/{n_tok} tokens ({100 * n_argmax / max(n_tok, 1):.3f}%)")

    with open(out / "questions.jsonl", "w", encoding="utf-8") as f:
        for qid, q in zip(df["qid"], df["question"]):
            m, n = question_logprob(gen, q)
            f.write(json.dumps({"qid": qid, "question_mean_logprob": m, "question_n_tokens": n}) + "\n")

    if not args.no_samples:
        torch.manual_seed(CFG.seed)
        if args.sample_bf16:
            gen.model.to(torch.bfloat16)
        raw_s = generate(gen, prompts, do_sample=True, n_return=K_SAMPLES, bs=8)
        gen.model.float()
        with open(out / "samples.jsonl", "w", encoding="utf-8") as f:
            for i, r in enumerate(tqdm(raw_s, desc="score samples")):
                q = i // K_SAMPLES
                f.write(json.dumps(record(gen, df["qid"].iloc[q], prompt_ids[q], r, k=i % K_SAMPLES),
                                   ensure_ascii=False) + "\n")
    meta = {"model": args.model, "dataset": args.dataset, "split": args.split, "n": len(df),
            "checkpoint": gen.model_name, "dtype": args.dtype, "max_new_tokens": MAX_NEW,
            "greedy": {"repetition_penalty": 1.0, "do_sample": False},
            "system_prompt": args.system_prompt or CFG.system_prompt,
            "samples": {"k": K_SAMPLES, "temperature": T_SAMPLE, "top_p": TOP_P, "seed": CFG.seed,
                        "compute_dtype": "bfloat16" if args.sample_bf16 else "float32"},
            "n_truncated": n_trunc, "argmax_agreement": n_argmax / max(n_tok, 1),
            "torch": torch.__version__, "elapsed_sec": round(time.time() - t0, 1)}
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"done in {time.time() - t0:.0f}s -> {out}")


if __name__ == "__main__":
    main()
