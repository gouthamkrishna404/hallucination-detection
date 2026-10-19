# v3 Research Audit Report

Hostile audit of the preregistered v2 study, the experiments it motivated, and the rebuilt paper.
Every number below is read from a file listed in [`RESULTS_REGISTRY.json`](RESULTS_REGISTRY.json);
the exact commands are in [`RESEARCH_LOG.md`](RESEARCH_LOG.md).

**Paper:** `paper/Learned_Falsehoods_Uncertainty_Paper.docx` / `.pdf` (built by `paper/build_paper.js`,
every number read from the analysis JSONs at build time).

---

## 1. What was recovered

The interrupted session had in fact **finished** its work: the v2 clean protocol, the preregistration
(`7713ee0`), the confirm labels (`2303030`) and the confirm evaluation + paper (`2aa8292`) were committed
and pushed. No half-written artifacts existed in the repository or in any session scratch directory.
This round therefore started from `2aa8292` (tagged `pre-v3-audit`) on a new branch `v3-audit`, and
re-ran `scripts/v2_confirm.py` from the frozen data: **`analysis_v2/confirm.json` is reproduced byte for
byte.**

## 2. Corrections to earlier claims

| # | Earlier statement (v2 paper / RESULTS / PREREGISTRATION) | What the audit found | Status |
|---|---|---|---|
| C1 | Preregistration §5 promised a SciQ containment-label sensitivity analysis | `v2_confirm.py` never ran it. Run now with the unchanged code path: D0 0.678 / 0.651 / 0.692, Holm p < 0.001 — H1 on SciQ holds | omission fixed |
| C2 | PREREGISTRATION §2: D2 = "mean, min, std log-prob" | The registered code and every reported D2 number use **mean log-prob, min log-prob, mean entropy**. The code is what ran; the document text is wrong | documentation error, no number changes |
| C3 | "Containment never accepted a wrong answer but rejected 23 % of correct paraphrases" | On the confirm split it accepts **2–4 %** of wrong answers and rejects **28–30 %** of correct ones (the "23" was 23 of 100 audited *items*) | corrected |
| C4 | "Five-sample self-consistency is never better than D0 and usually worse" | True for the cosine-clustering version only. The standard NLI-entailment semantic entropy **ties D0 on SciQ** (paired Δ −0.03 to +0.03, n.s.). Cosine D5 is *significantly better* than D0 on Qwen/TruthfulQA (+0.07, Holm p = 0.028), but only because D0 is slightly inverted there | corrected |
| C5 | Primary labels described as "blind" | Blind to model identity, features and scores, but written by the same agent that designed the hypotheses | disclosed; independent labels added (§4) |
| C6 | P(True) / "the model knows" framing absent | P(True) on TruthfulQA is *inverted* for Qwen and Llama — largely a **negation bias**: small models call negated statements "False" regardless of truth (AUROC 0.31–0.38 within negated answers), and 26–49 % of correct TruthfulQA answers are negations | new pitfall, documented |
| C7 | Within-question SciQ AUROC 0.56–0.62 (containment labels) | Label noise *attenuated* it: with conservative corrected labels 0.56–0.66 (all p ≤ 0.02). A naive correction (inherit own greedy label) inflates it to 0.61–0.69 and is not used | refined |

## 3. Hostile checks on the preregistered findings (all exploratory)

| Threat | Check | Result |
|---|---|---|
| D2's gain is answer length | length-quintile-matched AUROC; dev-trained length model; D2 added to length | D2 within 0.01 of raw inside length quintiles; length model 0.50–0.57; D2 adds +0.04 to +0.22 — **not length** |
| D2 learns a TruthfulQA-specific contrast | train on SciQ dev, test on TruthfulQA confirm (and cross-model) | SciQ-trained D2 reaches only 0.53–0.72 on TruthfulQA — **partly dataset-specific** |
| H2 gap is answer length | gap within length quintiles | −0.13 / −0.09 / −0.17, all CIs exclude 0 (Llama included) |
| H2 gap is question familiarity, difficulty or topic | OLS of z(D0) on type + length + truncation + question log-prob + question length + others-wrong + 38 category FE | imitative − other = −0.45 / −0.39 / −0.65 SD, CIs exclude 0 |
| H2 gap is answer form (negation) | restrict to affirmative answers | D0 gap −0.17 / −0.06 / −0.16 (CIs exclude 0 for Qwen, SmolLM2; not Llama) |
| H2 gap is the judge's typing | NLI typing (answer entails a listed incorrect answer) | agrees with judge types 81–86 % (κ 0.60–0.69); gap −0.20 / −0.12 / −0.21, Holm p ≤ 0.002, **Llama now significant** |
| H2 is specific to token probability | P(True), NLI semantic entropy, D2 by error type | **every signal family misses imitative errors** (P(True) 0.36–0.44 vs 0.43–0.59; NLI SE 0.45–0.54 vs 0.60–0.61) |
| H2 gap is the primary judge's labels | truth-judge labels (+ NLI types, no primary-judge input) | _filled in §4_ |
| H2 gap is question-level, not answer-level | within-question test on sampled answers | _filled in §4_ |
| H2 depends on the prompt ("give your best specific answer anyway") | neutral prompt allowing "I don't know", independent labels | _filled in §4_ |
| H2 only holds for 1–2 B models | Qwen2.5-3B, independent labels | _filled in §4_ |
| Dev/confirm leakage | exact and char-TF-IDF ≥ 0.8 duplicates | 0 exact; 8 TruthfulQA template variants, 3 SciQ paraphrases; only the low-capacity D2/D3 touch dev |

## 4. Independent-label, within-question, prompt and scale checks

_Completed after the truth-judge runs; see the end of this file._

## 5. What a detector buys in practice

* Abstaining on the 20 % least-confident answers (D0) lowers SciQ error 0.275 → 0.244, 0.379 → 0.316,
  0.233 → 0.192, but leaves TruthfulQA unchanged for Qwen (0.571 → 0.575) and SmolLM2 (0.621 → 0.603).
* The 20 % most confident TruthfulQA answers are wrong 60 % (Qwen) and 61 % (SmolLM2) of the time;
  78–87 % of those errors are imitative.
* D2 is calibrated (ECE 0.03–0.07) but improves Brier over the base rate by only 0.007–0.032; at a
  dev-fixed 20 % FPR threshold it flags 27–33 % of TruthfulQA answers with precision 0.70–0.76 and recall
  0.34–0.48. For Qwen and SmolLM2 on TruthfulQA, the best dev-fixed threshold is at most 0.006 more accurate
  than always predicting the majority class ("hallucinated").

## 6. Strongest findings, negative findings, unresolved

_Final version at the end of this file._

## 7. Reviewer-style assessment

_Final version at the end of this file._

## 8. Title

_Final version at the end of this file._
