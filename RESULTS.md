# Final Results Summary

Every number below was recomputed from saved outputs on disk immediately
before this document was written (`scripts/verify_final_numbers.py`), not
taken from memory. For full narrative, methodology, and every caveat, see
[README.md](README.md) — this file is the skimmable results reference.

## 1. Main matrix (original 3-feature detector, all 6 model×dataset combinations)

| Model | Dataset | Proposed (3-feat) | Mean-logprob only | Lexical self-consistency | Semantic self-consistency |
|---|---|---|---|---|---|
| Qwen2.5-1.5B | TruthfulQA | 0.482 | 0.551 | 0.461 | 0.470 |
| Llama-3.2-1B | TruthfulQA | 0.481 | 0.541 | 0.477 | 0.528 |
| SmolLM2-1.7B | TruthfulQA | 0.382 | 0.545 | 0.560 | 0.577 |
| Qwen2.5-1.5B | SciQ | 0.756 | 0.780 | 0.676 | 0.571 |
| Llama-3.2-1B | SciQ | 0.616 | 0.634 | 0.595 | 0.532 |
| SmolLM2-1.7B | SciQ | 0.717 | 0.767 | 0.758 | 0.544 |

All AUROC. Calls/question: 1 (proposed, mean-logprob) / 5 (self-consistency, either flavor).

## 2. Deep investigation: strongest validated detector per model (TruthfulQA)

Every number below is a **10-fold out-of-fold AUROC with bootstrap 95%
CI**, evaluated across the full 200-question set (leakage-free: each
question is scored only by a model that never trained on it).

| Model | Strongest detector found | AUROC | 95% CI | Excludes chance? |
|---|---|---|---|---|
| **Qwen2.5-1.5B** | Two-stage cascade, 17 rich features | **0.598** | [0.516, 0.678] | **Yes** |
| **Llama-3.2-1B** | Flat logistic regression, 17 rich features | **0.653** | [0.576, 0.727] | **Yes** |
| **SmolLM2-1.7B** | *(none — original 3-feat baseline)* | 0.438 | [0.355, 0.519] | No |

For reference, the original 3-feature detector under the same rigorous
out-of-fold procedure: Qwen 0.398 [0.325, 0.476], Llama 0.564 [0.482,
0.641], SmolLM2 0.438 [0.355, 0.519] — none significant.

## 3. Five-way head-to-head (fixed to one target label, TruthfulQA)

| Method (calls/question) | Qwen | Llama | SmolLM2 |
|---|---|---|---|
| Original 3-feat (1) | 0.398 | 0.564 | 0.418 |
| Candidate 4-feat: median/p25/trend/positional-delta logprob (1) | **0.591**✓ | 0.574 | 0.458 |
| Lexical self-consistency (5) | 0.485 | 0.489 | 0.571 |
| Semantic self-consistency (5) | 0.497 | 0.507 | 0.417 |
| Hybrid: 17 feat + both self-consistency signals (6) | 0.580 | **0.652**✓ | 0.519 |

✓ = 95% CI excludes chance (0.5). No entry for SmolLM2 excludes chance.
No single method wins for every model.

## 4. The core diagnostic: how much of the failure is structural?

Fraction of *wrong* answers made with above-median confidence (i.e.,
indistinguishable from correct answers on confidence grounds alone):

| Model | TruthfulQA | SciQ |
|---|---|---|
| Qwen2.5-1.5B | 49.2% | 32.6% |
| Llama-3.2-1B | 51.4% | 40.0% |
| SmolLM2-1.7B | 52.4% | 40.0% |

Roughly half of every model's TruthfulQA mistakes are made as confidently
as its correct answers — a hard ceiling on any purely confidence-based
detector, and the reason SciQ (33–40%) is so much easier.

Within *only* the confident subgroup (the hardest possible test — can we
still tell wrong from correct when raw confidence gives zero signal by
definition?), the full 17-feature set achieves AUROC **0.688** (Qwen,
significant) and **0.640** (Llama, significant); nothing works for
SmolLM2 (0.504, not significant).

## 5. What was ruled out (tried, verified, did not hold up)

| Investigated | Result |
|---|---|
| Random Forest / XGBoost / calibrated LR | No consistent gain over plain logistic regression; higher overfitting risk at n=140 |
| Label-margin restriction (is it just label noise?) | **Ruled out** — no threshold produces a significant result; Qwen's point estimate actively reverses (0.611→0.377) as data is discarded |
| `answer_length` as a feature | Labeling artifact (token-F1 precision bias, p=0.004), not a real signal — excluded from all candidate sets |
| Self-consistency sampling budget beyond N=5 | Attempted (N=10), generation failed twice on environment interruptions (not a code issue), not completed — documented as an open question, not fabricated |
| Cross-model transfer on TruthfulQA | Stays at chance in every direction (0.42–0.62) — nothing to transfer |
| Cross-model transfer on SciQ | **Strong positive** — transfers as well as or better than within-model (e.g. Llama→Qwen: 0.786–0.795 vs. Qwen's own 0.756), confirming the SciQ signal is general, not per-model |

## 6. Bottom line

**TruthfulQA detection was genuinely improved, not just re-explained.**
For Qwen and Llama, validated 1-call detectors (0.60–0.65 AUROC,
bootstrap CIs excluding chance) replace the original near-chance
3-feature detector — found through real feature engineering (positional/
percentile logprob statistics) and a two-stage cascade motivated by a
specific, confirmed diagnostic (the confidently-wrong subgroup carries a
real, separable fingerprint). For SmolLM2, every approach tried failed to
clear chance — a clean negative result, not a gap in the search. Nothing
found here approaches SciQ's 0.75–0.78, and the ~50% "confidently wrong"
rate on TruthfulQA sets a ceiling no confidence-only method can cross —
closing that gap needs a signal external to the model's own token
probabilities.

## Reproducing these numbers

```bash
python scripts/verify_final_numbers.py          # recomputes every number above from disk
```

Individual analyses (per model/dataset, after `run_experiment.py` has
been run): `experiment_feature_ablation.py`, `experiment_classifier_
comparison.py`, `experiment_quadrant_analysis.py`, `experiment_hybrid_
detector.py`, `experiment_best_candidate.py`, `experiment_label_margin_
deep_dive.py`, `experiment_cross_model_transfer.py`, `experiment_
confident_subgroup_classifier.py`, `experiment_cascade_detector.py`,
`experiment_final_headtohead.py`. All under `scripts/`.
