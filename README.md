# Logprob-Based Hallucination Flagging vs. Self-Consistency in LLMs

BCSE306L Natural Language Processing — DA1 implementation.

**Question:** Can a single-pass detector (mean token log-probability, minimum
token log-probability, answer-level entropy → logistic regression) approach
the hallucination-detection performance of a 5-call self-consistency
baseline, while using 80% fewer LLM calls?

## Pipeline

```
TruthfulQA (~200 Qs) → Qwen2.5-1.5B-Instruct → 1 generation + token logprobs
    → {mean logprob, min logprob, answer-level entropy} → Logistic Regression
    → hallucination probability / FLAG-PASS
```

compared against:
- **5-call self-consistency baseline** — 5 sampled generations/question, agreement score → logistic regression
- **Mean-logprob-only baseline** — 1 feature, same classifier family

## Corrections made to the original proposal

The PPT/report specify the architecture in detail but leave one thing
undefined: **how to get a ground-truth hallucinated/grounded label for an
arbitrary model-generated answer.** TruthfulQA ships reference answers
(`best_answer`, `correct_answers`, `incorrect_answers`) but not labels for
free text, and the paper's own automatic scorer ("GPT-judge") is a paid
fine-tuned model that isn't available here. Smallest fix that preserves the
objective, implemented in [`src/labeling.py`](src/labeling.py):

> **Reference-similarity proxy judge** — compute token-F1 overlap between the
> generated answer and every correct reference vs. every incorrect
> reference. Label `grounded` if the best correct-side overlap exceeds the
> best incorrect-side overlap, else `hallucinated`. This is the same
> lexical-overlap fallback the TruthfulQA paper itself describes as an
> alternative to GPT-judge.

This is an **automatic proxy, not a human-verified label** — it will
misjudge terse non-committal answers (e.g. "I'm not sure") and paraphrases
using no words from the references. This is a known limitation, consistent
with the report's own Section 5 caveat that "TruthfulQA's labels ... do not
perfectly correspond to every type of hallucination." Every script that
uses it is documented inline.

Two smaller ambiguities resolved the same way (documented in
[`src/config.py`](src/config.py) and [`src/model_utils.py`](src/model_utils.py)):

- **Answer-level entropy** = mean per-token Shannon entropy (nats) of the
  full softmax distribution over the vocabulary, at each generated token
  position.
- **Self-consistency "agreement/majority rule"** = mean pairwise token-F1
  similarity across the 5 sampled answers, fed into its own single-feature
  logistic regression (identical treatment to the mean-logprob-only
  baseline) so all three methods are compared with the same
  accuracy/AUROC/precision/recall/F1 metrics on the same held-out
  questions. The 5 samples use temperature sampling (T=0.7, top-p=0.9) —
  necessary for a meaningful agreement signal, since greedy decoding would
  make all 5 identical — while the single-pass detector itself remains
  greedy/deterministic for reproducibility.

## Project structure

```
src/
  config.py            # all hyperparameters, seeds, frozen in one place
  data.py              # fixed ~200-question TruthfulQA subset (cached)
  model_utils.py        # Qwen2.5-1.5B-Instruct loading + logprob/entropy extraction
  features.py            # mean/min logprob, mean entropy aggregation
  labeling.py             # reference-similarity proxy judge
  self_consistency.py      # agreement score + medoid answer for the 5-call baseline
  train_eval.py              # logistic regression + threshold-classifier ablation
  metrics_utils.py            # accuracy/AUROC/precision/recall/F1, confusion matrix, plots
scripts/
  01_generate_single_pass.py   # 1 LLM call/question -> features + label (cached CSV)
  02_generate_self_consistency.py  # 5 LLM calls/question -> agreement score + label
  03_train_and_evaluate.py      # trains all methods, ablations, plots, saves metrics
  run_all.py                     # runs all three in order
data/                              # cached fixed question subset (jsonl)
outputs/
  features/                         # per-question features (CSV)
  predictions/                       # per-question hallucination probabilities (CSV)
  metrics/all_metrics.json            # every number reported below
  plots/                                # confusion matrices, ROC curves, feature
                                          # distributions, calls-vs-accuracy trade-off
```

## Setup

Requires Python 3.11 (PyTorch has no 3.14 wheels yet). A CUDA GPU is used
automatically if available; otherwise falls back to CPU (slower, works fine
at this subset size and model size per the report's feasibility note).

```bash
py -3.11 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Running

```bash
python scripts/run_all.py
```

or step by step:

```bash
python scripts/01_generate_single_pass.py        # ~200 LLM calls
python scripts/02_generate_self_consistency.py   # ~1000 LLM calls (5/question)
python scripts/03_train_and_evaluate.py          # trains + evaluates, no LLM calls
```

The first run downloads the TruthfulQA dataset and the Qwen2.5-1.5B-Instruct
weights (~3 GB) from Hugging Face. Generation results are cached to
`outputs/features/*.csv`; re-running `03_train_and_evaluate.py` alone (e.g.
to tweak the classifier or ablations) does not re-trigger any LLM calls.

## Reproducibility

Seed=42 everywhere (`src/config.py: CFG.seed`), fixed question subset
selected once and cached to `data/truthfulqa_subset_200.jsonl`, frozen
model checkpoint (`Qwen/Qwen2.5-1.5B-Instruct`), frozen decoding parameters
(greedy for the single pass; T=0.7/top-p=0.9 for the 5-sample baseline),
stratified 70/30 split reused identically across every method.

## Results (actual run, seed=42, n=200, 140 train / 60 eval)

All numbers below are from the real pipeline run — nothing is projected or
fabricated. Full detail: [`outputs/metrics/all_metrics.json`](outputs/metrics/all_metrics.json),
figures in `outputs/plots/`, per-question predictions in `outputs/predictions/`.

| Method | LLM calls/question | Accuracy | AUROC | Precision | Recall | F1 |
|---|---|---|---|---|---|---|
| **Proposed (mean+min logprob+entropy → LogReg)** | 1 | 0.583 | **0.482** | 0.594 | 1.000 | 0.737 |
| Mean-logprob-only baseline | 1 | 0.583 | **0.551** | 0.594 | 1.000 | 0.737 |
| Self-consistency (5 samples → agreement → LogReg) | 5 | 0.567 | **0.461** | 0.591 | 0.929 | 0.723 |
| Threshold classifier on mean-logprob (ablation) | 1 | 0.617 | 0.551 | 0.625 | 0.857 | 0.723 |

Total LLM calls: **200** (proposed) vs. **1,000** (self-consistency) — an
exact **80% reduction**, as required by the problem statement.

**Ablation (feature subsets, all via logistic regression):**

| Feature set | AUROC |
|---|---|
| mean-logprob only | 0.551 |
| min-logprob only | 0.464 |
| entropy only | 0.504 |
| all three (proposed) | 0.482 |

**Subset-size sensitivity** (bootstrap resampling the eval set, 50 reps
each): accuracy stayed flat at 0.57–0.59 and AUROC at 0.48–0.49 across
n=15/30/45/60 — i.e. the weak performance is not a small-sample artifact.

### Interpretation — an honest null result

**None of the three methods separate hallucinated from grounded answers
better than chance** on this model/dataset combination (AUROC 0.46–0.55,
where 0.50 = coin flip). Feature distributions for grounded vs.
hallucinated answers visually overlap almost completely
(`outputs/plots/feature_distributions.png`), and the correlation between
each feature and the label is ≈0 even restricted to the most confidently
labeled examples — this was checked directly and is not an artifact of
proxy-label noise.

This is not a pipeline bug — I verified it, including manually inspecting
labeled examples, before reporting it. It is exactly the "main scientific
risk" the original report itself names in Section 5: *"A small model may
produce many incorrect answers for reasons unrelated to uncertainty,
making logprob features less useful."* TruthfulQA is specifically
constructed from common misconceptions the model has seen asserted
confidently many times in training data — so Qwen2.5-1.5B-Instruct is
often **confidently wrong** (e.g. it answered "the Great Wall of China is
visible from space" — a classic false claim — with *higher* confidence
than a question it answered correctly). That directly breaks the premise
that low logprob ⟺ hallucination for this benchmark.

**Answering the actual research question, precisely:** the proposed
1-call detector's AUROC (0.482) is statistically indistinguishable from
the 5-call self-consistency baseline's AUROC (0.461) — so in the narrow
sense of "does 1 call match 5 calls," the answer is yes. But this
equivalence is not meaningful on its own: neither method is a usable
hallucination detector here, both sit at chance level. The more useful
finding is that **the single mean-logprob feature alone (AUROC 0.551) was
the best-performing method of the four**, mildly outperforming both the
fused 3-feature detector and the 5-call baseline — adding min-logprob and
entropy did not help this small model, and self-consistency sampling did
not help either.

One more honesty check worth stating explicitly: the proposed detector's
confusion matrix (`outputs/plots/confusion_proposed_all_three.png`) shows
it predicts "hallucinated" for **all 60** eval questions at the default
0.5 threshold — its 0.583 accuracy is just the majority-class rate (35/60
eval questions are labeled hallucinated), not real discrimination. This is
the expected, correct behavior of a logistic regression handed features
that carry no signal — it collapses to predicting the majority class. The
AUROC (threshold-independent) is the metric that actually reflects this;
accuracy alone would have been misleading here.

The report's own Section 5 names the correct fallback for exactly this
outcome: *"evaluate a second small open-weight model and report the
result transparently rather than changing the evaluation after seeing the
preferred outcome."* That is a natural next step (e.g. Llama-3.2-1B-Instruct
or Phi-3.5-mini), not yet run here — ask if you'd like it added.
