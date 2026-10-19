# Results

> **v3 audit (2026-10-19).** The preregistered results below were audited and extended; the audit
> report is [AUDIT_REPORT.md](AUDIT_REPORT.md) and every headline number with its source file is in
> [RESULTS_REGISTRY.json](RESULTS_REGISTRY.json). Statements in Part A corrected by v3:
> (i) the preregistered SciQ containment sensitivity analysis had not been run (now run: H1 holds);
> (ii) the containment label accepts 2–4 % of wrong answers and rejects 28–30 % of correct ones
> (not "never accepted a wrong answer… 23 % of paraphrases"); (iii) "self-consistency is never better
> than D0" holds for the cosine version only; the standard NLI semantic entropy ties D0 on SciQ;
> (iv) D2 is mean log-prob + min log-prob + mean entropy (PREREGISTRATION §2's "std" is an erratum).
> New, exploratory: the imitative-error blind spot is shared by P(True) and NLI semantic entropy and
> survives length, question, topic, answer-form and NLI-typing controls.

**Read this first.** This file has two parts.

* **Part A — v2, the current results.** A clean protocol rebuilt from scratch
  (raw greedy decoding, fp32 teacher-forced scoring, blind judge labels),
  developed on the original 200-question subsets ("dev") and then tested once,
  as preregistered in [`PREREGISTRATION.md`](PREREGISTRATION.md), on 617
  TruthfulQA and 800 SciQ questions that were never used for any decision
  ("confirm"). Every number in Part A is reproduced by `scripts/v2_confirm.py`
  (`analysis_v2/confirm.json`) or `scripts/v2_analysis.py`
  (`analysis_v2/dev.json`).
* **Part B — v1, superseded.** The earlier audited results on the 200-question
  subsets with automatic proxy labels. Kept for the record; do not cite them as
  current. Section A6 lists which v1 claims changed.

Conventions: AUROC with 2 000-resample bootstrap 95 % CI; p = two-sided label
permutation (10 000) unless stated; Holm correction across the 6 model × dataset
cells within each family. Effect-size words are fixed in the preregistration:
< 0.60 negligible, 0.60–0.70 weak, 0.70–0.80 moderate.

---

# Part A — v2 (preregistered)

## A1. Protocol

| Item | v2 |
|---|---|
| Models | Qwen2.5-1.5B-Instruct, Llama-3.2-1B-Instruct, SmolLM2-1.7B-Instruct |
| Decoding | greedy with every logits processor disabled (no repetition penalty), fp32, ≤ 64 new tokens; 100 % of chosen tokens equal the argmax of the re-scored distribution in every cell |
| Scoring | each frozen answer re-scored by one fp32 teacher-forced pass: per-token log-prob, entropy, top-1/top-2 margin |
| Samples | K = 5 per question (T = 0.7, top-p 0.9) for self-consistency and semantic entropy (SmolLM2 sampled in bf16, disclosed) |
| Labels | Claude-judge with a fixed rubric (`labels/JUDGE_RUBRIC.md`), blind to model, features and log-probs; label + confidence + error type (imitative = repeats a listed misconception / distractor) |
| Judge consistency | blind re-judge of 100 items: 98.3 % (κ 0.97) TruthfulQA, 92.5 % (κ 0.78) SciQ; 100 % on items both times marked high-confidence |
| Splits | dev = original 200 + 200 questions; confirm = 617 TruthfulQA + 800 SciQ, disjoint by question text |

Base rates (confirm, judge): TruthfulQA 0.57 / 0.53 / 0.62, SciQ 0.28 / 0.38 / 0.23
(Qwen / Llama / SmolLM2). Llama hits the 64-token cap on 33 % of TruthfulQA
answers (Qwen 0 %, SmolLM2 1 %); its detection result is unchanged on
non-truncated answers (0.646 [0.590, 0.700], exploratory check).

## A2. H1 (primary): does −mean log-prob (D0) detect hallucinations? — confirm

| Model | TruthfulQA | SciQ |
|---|---|---|
| Qwen2.5-1.5B | 0.466 [0.421, 0.510], Holm p = 0.30 — **no** | **0.688** [0.651, 0.725], Holm p < 0.001 |
| Llama-3.2-1B | **0.653** [0.611, 0.697], Holm p < 0.001 | **0.658** [0.620, 0.695], Holm p < 0.001 |
| SmolLM2-1.7B | 0.500 [0.456, 0.545], Holm p = 0.99 — **no** | **0.702** [0.662, 0.745], Holm p < 0.001 |

Detection in 4 of 6 cells, all **weak** (0.65–0.70). On TruthfulQA two of
three models are at chance.

## A3. H3/H4: does anything beat the single-feature baseline? — confirm

| Detector | TQA Qwen | TQA Llama | TQA SmolLM2 | SciQ Qwen | SciQ Llama | SciQ SmolLM2 |
|---|---|---|---|---|---|---|
| D0 −mean log-prob | 0.466 | 0.653 | 0.500 | 0.688 | 0.658 | 0.702 |
| D1 −min log-prob | 0.539 | **0.713** | 0.560 | 0.708 | 0.671 | 0.706 |
| D2 3-feature LR (trained on dev) | **0.600** | **0.726** | **0.611** | **0.728** | **0.685** | **0.727** |
| D3 17-feature LR (trained on dev) | **0.626** | 0.706 | 0.597 | 0.688 | 0.638 | 0.719 |
| D4 self-consistency (5 samples) | 0.517 | 0.511 | 0.502 | 0.566 | 0.603 | 0.553 |
| D5 semantic entropy (5 samples) | 0.536 | 0.496 | 0.519 | 0.560 | 0.597 | 0.538 |

Paired ΔAUROC versus D0 (paired bootstrap, Holm across cells):

* **D2 − D0** is positive and significant in all 6 cells: +0.13 / +0.07 / +0.11
  on TruthfulQA, +0.04 / +0.03 / +0.03 on SciQ.
* **D3 − D0**: significant on TruthfulQA (+0.16 / +0.05 / +0.10), null on SciQ
  (−0.001 / −0.02 / +0.02). The 17 features add nothing over 3 on SciQ.
* **D1 − D0**: significant on TruthfulQA (+0.06–0.07), null on SciQ.
* **Self-consistency is worse than D0** wherever D0 works (D4 − D0 = −0.14
  Llama/TQA, −0.12 / −0.06 / −0.15 SciQ; all Holm p < 0.05) and never
  significantly better. With K = 5 and cosine-similarity clustering this is a
  statement about this cheap sampling budget, not about sampling methods in
  general.
* Dev → confirm transfer is lossless: frozen D3 on confirm (0.597–0.719) matches
  D3 refit out-of-fold *within* confirm (0.587–0.717).

The fitted gain on TruthfulQA is a learned *contrast*, not plain confidence:
the frozen D2 weights are positive on mean log-prob and entropy and negative on
min log-prob in every cell (e.g. Qwen/TQA standardized +0.72, −0.42, +0.29).
Its mechanism is not established here.

## A4. H2: which errors does uncertainty miss? — confirm

AUROC of D0 for each error type against all correct answers (TruthfulQA):

| Model | imitative errors (n) | other errors (n) | Δ [95 % CI] | Holm p |
|---|---|---|---|---|
| Qwen2.5-1.5B | 0.423 (255) | 0.580 (97) | −0.158 [−0.237, −0.075] | 0.005 |
| Llama-3.2-1B | 0.624 (217) | 0.710 (110) | −0.086 [−0.162, −0.013] | 0.096 |
| SmolLM2-1.7B | 0.451 (277) | 0.627 (106) | −0.176 [−0.262, −0.089] | 0.003 |

Errors that repeat a popular misconception are produced **with as much or more
confidence than correct answers** for Qwen and SmolLM2; non-imitative errors
(fabrications, wrong specifics) are detectable. The sign matches the dev
prediction in all three models; for Llama the gap does not survive Holm (and
shrinks to −0.04, n.s., on high-confidence labels only). On SciQ, where
imitative = a listed distractor (33–43 per cell), the gap is −0.07 / −0.10 /
+0.00, none significant.

Confidently wrong (share of wrong answers at least as confident as the median
correct answer; 0.5 = no signal): TruthfulQA **59 % / 28 % / 54 %**, SciQ
22 % / 29 % / 23 %.

## A5. H5: is it the answer or the question? — confirm

| | TQA Qwen | TQA Llama | TQA SmolLM2 | SciQ Qwen | SciQ Llama | SciQ SmolLM2 |
|---|---|---|---|---|---|---|
| own D0 | 0.466 | 0.653 | 0.500 | 0.688 | 0.658 | 0.702 |
| other models' D0 (range) | 0.44–0.52 | 0.50–0.52 | 0.51–0.55 | 0.63–0.67 | 0.62–0.64 | 0.64–0.67 |
| # other models wrong (no uncertainty) | **0.718** | **0.681** | **0.725** | **0.818** | **0.768** | **0.815** |
| question-only log-prob | 0.502 | 0.496 | 0.538 | 0.541 | 0.553 | 0.586 |

* On SciQ, another model's uncertainty predicts this model's errors almost as
  well as its own (own − other = +0.02 to +0.06; 1 of 6 pairs significant after
  Holm). Most of the SciQ signal is **shared question difficulty**.
* A difficulty proxy that uses no uncertainty at all — how many of the other two
  models got the question wrong — beats every uncertainty detector in every cell.
  (It needs other models' labels, so it is a diagnostic, not a usable detector.)
* There is nonetheless a real **answer-specific** component: among sampled
  answers to the *same* SciQ question, wrong samples have lower log-prob
  (within-question AUROC 0.567 / 0.558 / 0.621 over 231 / 308 / 245 mixed
  questions, sign-flip p ≤ 0.005; containment labels).
* D0 adds to question and length controls on SciQ (+0.08 to +0.11 OOF AUROC)
  and for Llama/TQA (+0.09); it adds nothing for Qwen/TQA or SmolLM2/TQA.
* Llama's TruthfulQA signal is its own (own 0.653 vs others 0.50–0.52, Holm
  p = 0.003), the one case where the answer-specific component is large.

## A6. Corrections to earlier claims

| Earlier claim (v1 / original DA1) | Status |
|---|---|
| Qwen DA1 headline (0.482 on a 60-question holdout) | Invalidated as evidence: decoding was contaminated by a default repetition penalty and labels were token-F1 proxies. Clean D2 (same 3 features) on 617 held-out questions: 0.600 [0.555, 0.644]. |
| SciQ containment label is "near-exact" | Wrong. Against the judge it never calls a wrong answer correct, but marks 23/100 correct paraphrases as hallucinated (κ 0.55). |
| SciQ detection is "strong / robust" | Overstated. It replicates (0.66–0.70) but is weak by the fixed scale and mostly question difficulty (A5). |
| "Roughly half of TruthfulQA errors are confident" (all models) | Model-dependent: 59 % Qwen, 54 % SmolLM2, 28 % Llama. |
| Only Llama shows a TruthfulQA signal | Confirmed for D0 (single feature). False for fitted detectors: dev-trained D2 detects for all three models (0.60–0.73). |
| Original self-consistency baseline | Was scored against the medoid sample's label, not the greedy answer's; v2 scores every detector against the same greedy label. |
| 60-question holdout as "external" test | It was a split of the same 200 questions and badly underpowered; replaced by the 617/800-question confirm split. Still in-distribution, not external. |

## A7. Internal assessment

**Definitely established (preregistered, held-out, Holm-corrected).**
* On SciQ, token uncertainty from one greedy answer separates right from wrong
  answers for all three models (D0 0.66–0.70; D2 0.69–0.73).
* On TruthfulQA, raw mean log-prob is at chance for Qwen and SmolLM2.
* Uncertainty misses imitative falsehoods and catches other errors (Qwen,
  SmolLM2): Δ −0.16 and −0.18.
* Five-sample self-consistency / semantic entropy is never better than D0 and
  usually worse.

**Plausible but not established.**
* The same imitative-error gap for Llama (direction replicated, not significant
  after Holm).
* That the fitted TruthfulQA gain (D2/D3) reflects a real signal mechanism rather
  than a dataset-specific contrast; it transfers from dev to confirm, but nothing
  here tests other datasets.

**Not supported.** Token uncertainty as a reliable, model-agnostic
hallucination detector; any AUROC ≥ 0.80; generalization beyond these two
datasets, this prompt and these three 1–2 B models.

**Invalidated.** The original DA1 Qwen result as evidence about Qwen's
uncertainty (decoding contamination); "near-exact" containment labels.

**Strongest result.** H2 — a preregistered, replicated dissociation: the
uncertainty signal is present for fabricated or idiosyncratic errors and absent
or inverted for errors that repeat a common misconception.

**Biggest limitation.** Labels come from one LLM judge (Claude) — consistent
on re-judging, but not validated against human annotators; and everything is
in-distribution (same two datasets).

**Real contribution.** A clean, preregistered re-test that (i) shows most of the
apparent SciQ signal is question difficulty, (ii) localizes TruthfulQA's failure
to imitative errors, and (iii) documents a decoding-contamination pitfall and a
label-quality pitfall that each changed earlier conclusions.

---

# Part B — v1 results (superseded; kept for the record)

## v1 Final Results Summary (superseded)

All numbers are produced by `scripts/build_final_results_table.py` from saved
experiment outputs and mirrored in [`comparison/final_results.md`](comparison/final_results.md).
If a number here disagrees with that file, that file is right.

**Read this first.** An audit (Section 1) found a decoding-time contamination
defect that invalidated the previously reported Qwen headline. Every number
below is post-correction. Earlier versions of this file reported
Qwen 0.598 / Llama 0.653 as "the main finding"; that framing did not survive
and has been replaced.

**Finding categories used throughout:**
🟢 **VALIDATED** — survives permutation testing, repeated CV, and leakage checks.
🔵 **EXPLORATORY** — real but with a caveat (selection on the same data, or does not survive multiple-testing correction).
🔴 **NEGATIVE** — investigated and found absent.
⚠️ **LIMITATION** — a boundary on trust.

**Statistical conventions.** AUROC is the primary metric. `perm p` is a
permutation test: labels are shuffled and the entire out-of-fold (OOF)
procedure re-run 500 times, so the p-value absorbs optimism in the fitting
procedure itself (minimum resolvable value 1/501 ≈ 0.002). `repCV` is the mean
± sd of 5-fold OOF AUROC over 20 different CV seeds. Eight detector variants
(2 architectures × 4 feature sets) are tested per model/dataset/labeler cell,
so the Bonferroni-corrected threshold is **α = 0.05/8 = 0.00625**; results with
0.00625 < p < 0.05 are marked exploratory, not validated.

## 1. Audit findings that changed the results 🟢 VALIDATED

| # | Defect | Evidence | Consequence |
|---|---|---|---|
| 1 | **Decoding-time contamination (Qwen only)** | Qwen2.5 ships `repetition_penalty=1.1` in `generation_config`; HF applies repetition penalty as a *logits processor* even under greedy decoding. Teacher-forcing recomputation shifts `mean_logprob` by 0.125 on average (max 0.595) — ~8× the between-class separation (0.016). Llama/SmolLM2 configure no penalty (shift ≤ 0.006). | All Qwen features recomputed from raw logits (`scripts/recompute_clean_features.py`, 200/200 answers reproduced exactly). Qwen's previous headline (cascade, 17-feat: 0.598, p=0.026) falls to **0.554, p=0.174 — not significant**. |
| 2 | **Token alignment is correct** | `scripts/verify_token_alignment.py` compares `generate()` scores against an independent teacher-forcing pass; an off-by-one control differs by 15–20 nats, the actual data by ≤0.02 for unpenalised models. | Eq. for ℓ_t is sound; the contamination in (1) is a distribution issue, not an indexing issue. |
| 3 | **Llama greedy decoding is not reproducible here** | 83/200 TruthfulQA and 29/200 SciQ answers failed to regenerate identically. Non-reproduced answers are longer (54.1 vs 43.5 tokens) and hit the 64-token cap far more (0.52 vs 0.28) — fp16 non-determinism compounding over long generations. | Llama's recomputed subset is length-biased and is **not** used. Llama/SmolLM2 keep their original (uncontaminated) features. |
| 4 | **`answer_length` is confounded with the token-F1 labeler** | Token-F1 precision falls with answer length, and faster against incorrect references (r = −0.49) than correct ones (r = −0.33). Qwen/TruthfulQA: grounded answers average 23.7 tokens vs 20.0 (t = 2.90, p = 0.004). | Every result is reported with and without `answer_length`. The bias is **not universal**: Llama/TruthfulQA shows it under *both* labelers (p = 0.004 / 0.019), SmolLM2 under neither. |
| 5 | **Features are unstandardized** | `fit_and_predict` feeds raw features (scales ~0.5 to ~50) to L2 logistic regression, which penalizes small-scale features disproportionately. | Tested with a fold-internal `StandardScaler` (`--scale`): conclusions unchanged. Qwen 17-feat improves to 0.590 (p=0.044, still fails correction); Llama 17-feat 0.636 (p=0.006, still the only robust cell). |
| 6 | **"17 features" is 15 in 5 of 6 cells** | Top-1/top-2 margin features were cached only for Qwen/TruthfulQA; elsewhere they are constant. | Corrected: the clean recomputation now provides margins for all cells. |
| 7 | **SciQ labeler asymmetry** | SciQ has 1 distinct correct reference vs ~3 distractors, both entering a `max`, inflating `incorrect_sim`. | SciQ hallucination base rate swings from 0.45–0.55 (token-F1) to 0.24–0.28 (embedding). Base rates are reported for both. |

## 2. Dataset and labeler characteristics

| Model | Dataset | n | Halluc. rate (token-F1) | Halluc. rate (embedding) | Labeler agreement | κ | Length-bias p |
|---|---|---|---|---|---|---|---|
| Qwen2.5-1.5B | TruthfulQA | 200 | 0.59 | 0.55 | 0.730 | 0.450 | 0.004 |
| Qwen2.5-1.5B | SciQ | 200 | 0.46 | 0.26 | 0.735 | 0.448 | 0.203 |
| Llama-3.2-1B | TruthfulQA | 200 | 0.53 | 0.52 | 0.730 | 0.459 | 0.004 |
| Llama-3.2-1B | SciQ | 200 | 0.55 | 0.28 | 0.655 | 0.338 | 0.463 |
| SmolLM2-1.7B | TruthfulQA | 200 | 0.62 | 0.54 | 0.785 | 0.563 | 0.918 |
| SmolLM2-1.7B | SciQ | 200 | 0.45 | 0.24 | 0.730 | 0.430 | 0.038 |

⚠️ The two labelers agree on only 65.5–78.5% of answers (κ 0.34–0.56), and on
SciQ they disagree about the base rate by roughly a factor of two.

### 2a. Which labeler is right? (SciQ can be settled) 🟢 VALIDATED

SciQ reference answers are short noun phrases, so a near-exact criterion is
available: does the generation contain the correct answer and no distractor?
`scripts/experiment_labeler_adjudication.py` builds that **containment label**
(3–5 ambiguous cases per cell excluded) and scores both proxies against it:

| Model | True halluc. rate (containment) | token-F1 agreement (κ) | embedding agreement (κ) |
|---|---|---|---|
| Qwen2.5-1.5B | 0.52 | **0.939 (0.878)** | 0.685 (0.378) |
| Llama-3.2-1B | 0.64 | **0.913 (0.822)** | 0.605 (0.288) |
| SmolLM2-1.7B | 0.52 | **0.934 (0.868)** | 0.670 (0.345) |

**On SciQ the token-F1 labeler is accurate and the embedding labeler is not.**
The embedding labeler marks only 24–28% of answers hallucinated against a true
rate of 52–64%, i.e. it misses roughly half of the errors — unsurprising, since
cosine similarity between a full sentence and a one-or-two-word reference is
high almost regardless of correctness. This explains why SciQ results appear to
"collapse" under embedding labels (Qwen 0.716 → 0.520): that collapse reflects
labeler failure, not a fragile detector. **Under the containment label the
detector holds for all three models: 0.673–0.704 (Section 4).**

⚠️ No equivalent adjudication is possible on TruthfulQA, whose references are
full sentences with many valid paraphrases. There, labeler disagreement remains
a genuine open uncertainty, and results that flip between labelers are reported
as flipping.

## 3. Headline: TruthfulQA, 5-fold OOF (n = 200)

Qwen uses decontaminated features; Llama and SmolLM2 use their original
(uncontaminated) features. `*` = p < 0.05; `**` = survives Bonferroni (p < 0.00625).

| Model | Detector | token-F1 labels | embedding labels |
|---|---|---|---|
| Qwen2.5-1.5B | original 3-feat | 0.434 (p=0.844) | 0.540 (p=0.186) |
| Qwen2.5-1.5B | all 17-feat, flat | 0.553 (p=0.170) | 0.451 (p=0.798) |
| Qwen2.5-1.5B | all 17-feat, cascade | 0.554 (p=0.174) | 0.496 (p=0.535) |
| Qwen2.5-1.5B | 16-feat (no length) | 0.505 (p=0.481) | 0.501 (p=0.475) |
| Qwen2.5-1.5B | 4-feat subset, cascade 🔵 | 0.605* (p=0.024) | 0.592* (p=0.028) |
| **Llama-3.2-1B** | original 3-feat | 0.566 (p=0.090) | **0.615** (p=0.002)** |
| **Llama-3.2-1B** | **all 17-feat, flat** | **0.653** (p=0.002)** | **0.663** (p=0.002)** |
| **Llama-3.2-1B** | all 17-feat, cascade | **0.630** (p=0.002)** | **0.661** (p=0.002)** |
| **Llama-3.2-1B** | 16-feat (no length) | 0.578 (p=0.072) | **0.655** (p=0.002)** |
| SmolLM2-1.7B | original 3-feat | 0.418 (p=0.904) | 0.401 (p=0.948) |
| SmolLM2-1.7B | all 17-feat, flat | 0.485 (p=0.585) | 0.586* (p=0.042) 🔵 |
| SmolLM2-1.7B | 16-feat (no length) | 0.494 (p=0.507) | 0.581* (p=0.044) 🔵 |

**What this supports.** 🟢 **Only Llama-3.2-1B shows a replicable TruthfulQA
signal.** It is significant under both labelers, under both detector
architectures, stable across 20 CV seeds (0.649 ± 0.011), and survives
Bonferroni correction. Under the embedding labeler it also survives removal of
`answer_length` (0.655, p ≤ 0.002), so it is not merely the length artifact.

🔴 **Qwen shows no validated TruthfulQA signal after decontamination.** The one
variant below α (4-feature cascade, p = 0.024) uses a feature subset selected
by inspecting these same questions and does not survive correction for the 8
variants tested. 🔴 **SmolLM2 is a null**; its two p ≈ 0.04 cells under the
embedding labeler likewise fail correction.

⚠️ **Nothing is significant on the original fixed 60-question holdout** — every
CI includes 0.5 (Qwen cascade 0.583 [0.433, 0.735]; Llama flat 17-feat 0.598
[0.446, 0.738]; SmolLM2 0.503 [0.352, 0.657]). The OOF results use all 200
questions and therefore have more power; they are *not* a second, independent
test set.

## 4. SciQ, 5-fold OOF (n = 200) 🟢 VALIDATED

| Model | original 3-feat | all 17-feat | 16-feat (no length) |
|---|---|---|---|
| Qwen2.5-1.5B | **0.716** (p=0.002)** | 0.686 (p=0.002)** | 0.694 (p=0.002)** |
| Llama-3.2-1B | 0.634 (p=0.004)** | 0.638 (p=0.004)** | 0.638 (p=0.002)** |
| SmolLM2-1.7B | 0.649 (p=0.002)** | 0.641 (p=0.004)** | 0.645 (p=0.002)** |

Every model, every feature set, every detector architecture is significant
after correction, seed-stable (sd ≤ 0.02), and **unaffected by removing
`answer_length`**. Notably the **original 3-feature detector is the best or
near-best variant on SciQ** — the engineered features add nothing here.

**Confirmed against the near-exact containment label** (Section 2a), which does
not depend on either proxy:

| Model | original 3-feat | all 17-feat | 16-feat (no length) |
|---|---|---|---|
| Qwen2.5-1.5B | 0.682 [0.609, 0.753] | 0.679 [0.603, 0.752] | 0.684 [0.611, 0.756] |
| Llama-3.2-1B | 0.676 [0.598, 0.749] | 0.703 [0.620, 0.777] | 0.704 [0.623, 0.778] |
| SmolLM2-1.7B | 0.682 [0.608, 0.753] | 0.673 [0.600, 0.744] | 0.676 [0.602, 0.747] |

Under the containment label every one of these nine variants reaches the
minimum resolvable permutation p ≈ 0.002, with repeated-CV spread ≤ 0.011
(e.g. SmolLM2 3-feat: 0.688 ± 0.003). This is the project's most robust
result: it holds for all three models under the most reliable label
available, and it is the plainest contrast with TruthfulQA.

## 5. The confidently-wrong diagnostic 🟢 VALIDATED

Share of *wrong* answers produced with above-median mean log-probability:

| Model | TruthfulQA | SciQ |
|---|---|---|
| Qwen2.5-1.5B | 50.8% | 32.6% |
| Llama-3.2-1B | 51.4% | 40.0% |
| SmolLM2-1.7B | 52.4% | 40.0% |

On TruthfulQA a wrong answer is about as likely to be confident as uncertain.
Any detector thresholding mean confidence at the median necessarily places
these answers with the confident correct ones. Within the confident subgroup
the association also **reverses direction**: for Qwen, confidently *wrong*
answers have *higher* second-half log-probability than confidently correct ones
(Cohen's d = +0.73) and are shorter (d = −0.71). This motivated the cascade —
which nonetheless failed to produce a validated gain for Qwen.

## 6. What did not work 🔴 NEGATIVE

| Investigated | Outcome |
|---|---|
| Random forest, XGBoost, calibrated LR | No consistent gain over logistic regression; several overfit (Qwen 3-feat XGBoost: CV 0.585 → holdout 0.414) |
| Self-consistency (5 calls) and semantic entropy (5 calls) | Never significant on TruthfulQA for any model; on SciQ matched or trailed the 1-call detector |
| Hybrid (6 calls) | Never clearly earned its extra cost |
| Label-margin restriction | Ruled out — no threshold significant; Qwen's estimate reverses (0.611 → 0.377) as data is discarded |
| 10-sample self-consistency sweep | Not completed (environment interruptions at 62% and 33%); remains open |
| Qwen after decontamination | No validated detector |
| SmolLM2 on TruthfulQA | Null under every feature set, architecture, and labeler |

## 7. Limitations ⚠️

- One 200-question subset per dataset; the 60-question holdout is badly underpowered.
- All labels are automatic proxies that agree only 65.5–78.5% of the time; several conclusions depend on which is used. No human-adjudicated labels exist.
- Three models of 1–2 B parameters, two short-answer datasets; no claim beyond them.
- Llama generations are not bit-reproducible here, and ~38% of its TruthfulQA answers hit the 64-token cap (vs ~0% for Qwen/SmolLM2), weakening cross-model comparability.
- The 4-feature subset was chosen by inspecting these same questions.
- Features are unstandardized in the primary pipeline; the semantic-entropy baseline uses cosine clustering rather than entailment.
- No retrieval or external verifier, so errors stated with full confidence remain out of reach.

## 8. Bottom line

Token-level uncertainty from a single generation is a **strong, robust signal on
ordinary factual recall (SciQ: 0.63–0.72 under the token-F1 label, and
0.67–0.70 under a near-exact containment label, for all three models)** and
**largely absent on misconception-oriented questions (TruthfulQA)**, where
roughly half of all wrong answers are produced as confidently as correct ones.
Finer-grained features recover a replicable signal for exactly **one of three
models (Llama-3.2-1B: 0.653, p ≤ 0.002)**; for Qwen the apparent gain was an
artifact of decoding-time repetition penalty, and for SmolLM2 there is nothing
to recover. The method is therefore useful but strongly model- and
dataset-dependent, and it cannot resolve confidently wrong factual claims
without evidence external to the generating model.

## Reproducing

```bash
python scripts/verify_token_alignment.py --model qwen2.5-1.5b-instruct   # audit finding 1-2
python scripts/recompute_clean_features.py --model qwen2.5-1.5b-instruct --dataset truthfulqa
python scripts/experiment_clean_validation.py --model <key> --dataset <ds> --features <clean|rich> [--labels semantic] [--scale]
python scripts/build_final_results_table.py                              # regenerates every table above
```
