# Preregistration: confirmatory test of token-uncertainty hallucination detection

Written and committed **before any detector score is compared with a confirm-split
label**. The commit that adds this file is the registration timestamp; the
analysis that implements it is `scripts/v2_confirm.py` in the same commit, and it
is run unchanged afterwards. Anything not listed here and reported later is
exploratory and is labelled as such.

## 1. State of the data at registration

| Item | Status |
|---|---|
| Models | Qwen2.5-1.5B-Instruct, Llama-3.2-1B-Instruct, SmolLM2-1.7B-Instruct |
| Dev split | the original 200 TruthfulQA + 200 SciQ questions; every modelling decision was made here (`scripts/v2_analysis.py`) |
| Confirm split | all remaining TruthfulQA questions (617) and the SciQ test split (800); disjoint from dev by question text (`data/*/confirm.jsonl`) |
| Generations | frozen (`results_v2/`): raw greedy decoding (no logits processors, fp32, 64 new tokens max) + K=5 samples (T=0.7, top-p 0.9, seed 42), every answer re-scored by one fp32 teacher-forced pass |
| Labels | Claude-judge labels (`labels/JUDGE_RUBRIC.md`) for every greedy answer of every cell, produced blind (no model identity, no features, no log-probabilities) |
| Confirm features | `features.csv` exists for confirm cells; **no AUROC, correlation or other comparison with confirm labels has been computed** |

Disclosed deviations and corrections made before registration:
* SmolLM2-1.7B does not fit an 8 GB GPU in fp32 with a sampling KV cache; its K=5
  *samples* were generated in bf16 compute. Its greedy answers and all scoring are
  fp32 (the checkpoint is stored in bf16, so the fp32 weights are an exact upcast).
* One dev label was corrected under the rubric's "reference plainly wrong" clause
  (TruthfulQA claims cousin marriage is illegal in Australia; it is legal).
* Judge consistency, measured before registration by blind re-judging 100 dev
  items: 98.3 % agreement on TruthfulQA (kappa 0.97), 92.5 % on SciQ (kappa
  0.78); every disagreement was on an item marked low-confidence both times.

## 2. Frozen detectors

All scores are oriented so that larger = more likely hallucinated.

| ID | Detector | Fitting |
|---|---|---|
| D0 | −mean token log-prob of the greedy answer | none |
| D1 | −min token log-prob | none |
| D2 | logistic regression on the original 3 features (mean, min, std log-prob) | trained on the dev split of the same model × dataset, then frozen |
| D3 | logistic regression on all 17 token features | as D2 |
| D4 | −(fraction of the 5 samples semantically matching the greedy answer, cosine ≥ 0.75) | none |
| D5 | semantic entropy over the 5 samples | none |

Classifier: `StandardScaler` + L2 logistic regression, C = 1 (no tuning), both
fitted on dev only.

## 3. Hypotheses and tests

Unit of analysis: one model × dataset cell (6 cells). Primary label: judge label.
Every p-value is two-sided. Multiple comparisons: Holm across the 6 cells within
each family.

**H1 (primary).** D0 separates hallucinated from correct answers: AUROC ≠ 0.5,
label-permutation test (10 000 permutations), Holm across 6 cells, α = 0.05.
Reported with a 2 000-resample bootstrap 95 % CI.
*Expectation from dev (not a test):* SciQ cells clearly above 0.5 (dev 0.63–0.68);
TruthfulQA cells weak or null (dev 0.45–0.60). No directional prediction for
TruthfulQA.

**H2 (error type).** On TruthfulQA, uncertainty detects non-imitative errors
better than imitative ones (errors that repeat a listed misconception):
Δ = AUROC(D0; imitative vs correct) − AUROC(D0; other vs correct) ≠ 0,
stratified bootstrap, Holm across the cells where both error types have ≥ 10
items (expected: the 3 TruthfulQA cells; SciQ has too few imitative errors).
*Expectation from dev:* Δ < 0 in all three TruthfulQA cells
(−0.28, −0.09, −0.17). Also reported: each of the two AUROCs with CIs.

**H3 (does anything beat the simple baseline?).** Paired ΔAUROC of D1, D2, D3,
D4, D5 versus D0 on the same items, paired bootstrap, Holm across cells per
detector. No directional prediction; the 17-feature detector D3 did not
reliably beat D0 on dev.

**H4 (detectors on their own).** D1–D5 AUROC ≠ 0.5, as H1, one Holm family per
detector.

**H5 (answer or question?).** If D0 tracks the specific answer rather than the
question's difficulty, a model's own D0 should predict its own errors better than
another model's D0 does on the same questions and the same labels. Test: paired
ΔAUROC(own D0 − other model's D0) on model B's labels, for every ordered model
pair within a dataset (6 per dataset), paired bootstrap, Holm across the 12
comparisons. *Expectation from dev:* small deltas (−0.03 to +0.08), none
significant, i.e. on these datasets D0 is largely a question-difficulty signal.
Reported alongside: the AUROC of a difficulty proxy that uses no uncertainty
(the number of other models that answered the same question wrongly).

## 4. Descriptive endpoints (no hypothesis test)

* Confidently wrong: share of wrong answers whose D0 risk is at or below the
  median correct answer's (0.5 = no signal), with bootstrap CI.
* Controls: AUROC of question-only log-prob and of answer length; OOF AUROC of
  [question log-prob, question length, answer length] with and without D0
  (paired Δ), i.e. whether D0 adds to difficulty and length proxies.
* D3 out-of-fold within confirm (5-fold), to compare dev→confirm transfer with
  within-split fitting.
* Within-question AUROC on SciQ sampled answers (containment labels, sign-flip
  test): does lower confidence mark the wrong samples *of the same question*?

## 5. Sensitivity analyses

* All endpoints recomputed after dropping items the judge marked `low` or
  `refusal` (training and test).
* SciQ: containment label instead of the judge label (secondary label).

## 6. Interpretation rules, fixed now

* A cell "shows detection" only if its Holm-adjusted p < 0.05 **and** the 95 % CI
  excludes 0.5.
* Effect size words: AUROC < 0.60 negligible, 0.60–0.70 weak, 0.70–0.80
  moderate, ≥ 0.80 strong. No result is called "conclusive" or "reliable" on the
  basis of AUROC > 0.5 alone.
* Confirm results are **within-distribution** held-out results (same datasets,
  same prompt, same models). They are not evidence of generalisation to other
  datasets, domains, prompts or models, and will not be described as such.
* If H1 fails in a cell, that is reported as a negative result for that cell; no
  alternative detector is promoted to "primary" after the fact.
* Results contradicting the dev expectations above are reported as such.

---

## Erratum (added 2026-10-19, after registration; nothing above was changed)

* §2 describes D2 as "mean, min, std log-prob". The code registered in the same commit
  (`ORIGINAL_THREE_FEATURES` in `src/features_v2.py`) uses **mean log-prob, min log-prob and mean
  entropy**, and that is what `scripts/v2_confirm.py` ran. Every reported D2 number refers to the
  code; the description above is wrong.
* §5's SciQ containment-label sensitivity analysis was not executed by `scripts/v2_confirm.py`.
  It was run later, unchanged, by `scripts/v3_audit.py` (`analysis_v3/audit.json`, key
  `containment`); see AUDIT_REPORT.md.
