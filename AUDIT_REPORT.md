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
| H2 gap is the primary judge's labels | truth-judge labels + NLI types (no primary-judge input) | Qwen −0.20, SmolLM2 −0.17 (Holm p = 0.002); **Llama +0.01 — does not replicate** |
| H2 gap is question-level, not answer-level | within-question test on 3,085 sampled answers per model (independent labels/types) | imitative samples *more* confident than correct samples of the same question: Llama 0.40 (p = 0.002), SmolLM2 0.42 (p = 0.018), Qwen 0.45 (n.s.); other wrong samples 0.45–0.56, none significant |
| H2 depends on the prompt ("give your best specific answer anyway") | neutral prompt allowing "I don't know", independent labels/types | answers change (16–34 % identical); gap −0.19 / −0.06 / −0.14 (Qwen, SmolLM2 significant) |
| H2 only holds for 1–2 B models | Qwen2.5-3B, independent labels and types | gap −0.13 [−0.22, −0.04], p = 0.006; D0 inverted (0.444) — **persists** |
| Dev/confirm leakage | exact and char-TF-IDF ≥ 0.8 duplicates | 0 exact; 8 TruthfulQA template variants, 3 SciQ paraphrases; only the low-capacity D2/D3 touch dev |

## 4. Independent labels change one result

The AllenAI truth-judge (Llama-2-7B fine-tuned on TruthfulQA's human truthfulness ratings, held-out
accuracy 0.945 reported by its authors; run in 4-bit NF4; smoke-tested on the model card's example) was
applied to every greedy, sampled and prompt-variant TruthfulQA answer.

* **Agreement** with the primary judge on TruthfulQA confirm: 87 / 79 / 84 %, κ 0.74 / 0.59 / 0.67. The
  truth-judge marks fewer answers hallucinated (0.52 / 0.37 / 0.52 vs 0.57 / 0.53 / 0.62). On SciQ it is
  out of distribution (κ ≈ 0.2) and is not used.
* **Where they disagree:** mostly answers the primary judge flags and the truth-judge accepts. These are
  longer (Llama 55 vs 46 tokens) and less confident, half carry the primary judge's low-confidence flag,
  and many make a plausible main claim plus a false secondary one (e.g. "the ISS experiences a weak
  gravitational force… not strong enough to cause significant effects", truth-judge P(true) = 1.0). The
  truth-judge was trained on short answers. We do not adjudicate; human labels would.
* **Consequence for H1:** Llama's TruthfulQA D0 detection (0.653, preregistered) is **0.522 under
  truth-judge labels** and 0.607 on the 487 answers both judges agree on. Under truth-judge labels, D0 is
  significantly *inverted* for Qwen (0.443) and SmolLM2 (0.413): their most confident answers are the
  most often wrong. D2 still detects for all three (0.58 / 0.56 / 0.61).
* **Consequence for H2:** replicates for Qwen and SmolLM2 under every label source, typing, prompt and
  control except Qwen's within-question test (where *neither* error type is less confident than correct
  answers). For Llama it holds with the primary labels and within question, but not with truth-judge
  labels on greedy answers or on consensus items.
* **Pooled** over models with a joint question bootstrap: −0.140 [−0.186, −0.096].

**Scale (Qwen2.5-3B-Instruct, bf16, TruthfulQA, independent labels and types).** Fewer errors than the
1.5B model (0.331 vs 0.517) but no better uncertainty: D0 0.444 [0.396, 0.493] (inverted, p = 0.023),
imitative 0.403 vs other 0.534, gap −0.131 [−0.223, −0.040], p = 0.006. bf16 scoring matches the greedy
argmax on 99.1 % of tokens (100 % in fp32), disclosed.

**Second recall dataset (TriviaQA, 1,000 questions, judge-free alias labels; 7B/8B check dropped for
bandwidth and replaced by this).** D0 0.753 / 0.694 / 0.730 (Holm p < 0.001; 0.749 / 0.706 / 0.741 with a
strict canonical-answer label), D2 0.81 / 0.70 / 0.80, difficulty proxy 0.81–0.86, own − other D0
−0.00 to +0.10. The SciQ conclusion generalises: a weak-to-moderate, largely question-level signal on
recall errors. Detectability orders TriviaQA > SciQ > TruthfulQA.

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

**Established (preregistered, held out, Holm-corrected; robust to the audit).**
1. On SciQ, single-pass D0 detects errors for all three models, weakly (0.66–0.70); also under the
   containment label. Most of it is shared question difficulty (others-wrong proxy 0.77–0.82).
2. On TruthfulQA, D0 is at chance for Qwen and SmolLM2 (and significantly inverted under independent
   labels).
3. For Qwen and SmolLM2, misconception-repeating errors are scored like (or more confident than) correct
   answers while other errors are scored as riskier: gap −0.13 to −0.21 under every label source, error
   typing, prompt, and length/question/topic/form control (the one exception: Qwen's within-question
   test, where neither error type is less confident than correct answers).

**Supported but uncertain (exploratory).**
* The blind spot is shared by P(True) and NLI semantic entropy (all three models, primary labels).
* Within a question the misconception is often the model's more probable answer (Llama, SmolLM2).
* A small answer-specific component on SciQ (within-question 0.55–0.66).

**Negative results.** No detector reaches AUROC 0.80 anywhere. P(True) is no better than D0 and is
inverted on TruthfulQA by a negation bias. Five-sample self-consistency (cosine or NLI) never beats D0.
D3 (17 features) never clearly beats D2 (3 features). Abstention on the least-confident 20 % does not lower
Qwen's or SmolLM2's TruthfulQA error rate.

**Weakened / invalidated.** Llama's TruthfulQA detection (label-dependent). "Self-consistency is worse
than D0" (only for the cosine version). "Containment never accepts a wrong answer". The original DA1 Qwen
result (contaminated decoding, proxy labels) — invalidated in v2, still invalid.

**Unresolved.** Which judge is right on long multi-claim answers (needs human labels — a blind
120-item annotation kit is ready in `labels/human/`, scored by `scripts/v3_human.py score`); why Llama differs
(training data, alignment, length); whether the blind spot persists in models much larger than 3 B; why D2's contrast
(fluent answer, one improbable token; `analysis_v3/d2_mechanism.json`, exploratory) transfers poorly across
datasets.

## 7. Reviewer-style assessment

*Summary.* A careful, transparently reported negative/diagnostic study. Its main claim — that uncertainty
read from the generating model is blind to learned misconceptions — is supported for two of three small
models under an unusually wide set of controls, including labels from an independent judge.

*Strengths.* Preregistration with a held-out split and Holm correction; decoding contamination found and
fixed; byte-for-byte reproducible analysis; every number generated from saved artifacts; four signal
families compared; a within-question test that removes question-level confounds; honest reporting of the
cases where the claim fails (Llama under independent labels).

*Weaknesses a reviewer would raise.*
1. Small models only; the effect may change with scale (calibration is known to improve with size).
2. No human labels. Two automatic judges disagree on 13–21 % of TruthfulQA answers, and the disagreement
   changes one preregistered conclusion.
3. All robustness analyses are post hoc. They were designed to break the result and are reported in full,
   but they are not confirmatory.
4. Effect sizes are modest (gaps ≈ 0.15 AUROC, CIs ≈ ±0.08), and TruthfulQA's adversarial construction
   makes imitative errors unusually common; their base rate in natural use is unknown.
5. Error types rely on TruthfulQA's listed incorrect answers (a bias towards *smaller* gaps).
6. The trained D2 helps, but through a dataset-specific contrast; no new method is proposed.

*Verdict.* Suitable as a workshop or short paper on evaluation methodology and negative results; a
main-track paper would need human labels on a subset, at least one model of 7 B or more, and a second
misconception-oriented dataset.

## 8. Title

Candidates considered after the evidence was in:

| Title | Accuracy | Note |
|---|---|---|
| Confidently Wrong: A Preregistered Re-test of Token-Level Uncertainty… (v2) | partial | implies most errors are confident; false for Llama and for SciQ |
| When Confidence Fails: Token Uncertainty Misses the Misconceptions Small LMs Learned | good | too narrow: the blind spot is not specific to token probability |
| Blind to Misconceptions: Uncertainty Signals and Imitative Falsehoods in Small LMs | good | accurate, less memorable |
| **Learned Falsehoods Look Like Knowledge: A Preregistered Test of Uncertainty-Based Hallucination Detection in Small Language Models** | good | chosen |

Why the chosen title matches the evidence: "learned falsehoods" are the imitative errors the study
isolates; "look like knowledge" states the measured result — to token probability, P(True) and semantic
entropy these errors are scored like correct answers, and within a question they are often preferred to
the truth. The subtitle bounds the claim: uncertainty-based detection, small models, a preregistered test.
It does not claim the blind spot holds for every model; the paper reports that it does not for Llama under
independent labels.
