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
> survives length, question, topic, answer-form and NLI-typing controls; with independent truth-judge
> labels it replicates for Qwen and SmolLM2, under a neutral prompt and in Qwen2.5-3B, but not for
> Llama, whose TruthfulQA D0 detection (A2: 0.653) is 0.522 under those labels.

**Part A — v2, the current results.** A clean protocol rebuilt from scratch
(raw greedy decoding, fp32 teacher-forced scoring, blind judge labels),
developed on the original 200-question subsets ("dev") and then tested once,
as preregistered in [`PREREGISTRATION.md`](PREREGISTRATION.md), on 617
TruthfulQA and 800 SciQ questions that were never used for any decision
("confirm"). Every number in Part A is reproduced by `scripts/v2_confirm.py`
(`analysis_v2/confirm.json`) or `scripts/v2_analysis.py`
(`analysis_v2/dev.json`). The superseded v1 results (Part B) are in the git tag
`full-history-2026-10-28`; Section A6 lists which v1 claims changed.

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

*Part B (the superseded v1 results) is kept in the git tag `full-history-2026-10-28`.*
