# Research log (v3 audit round)

Durable progress log for the post-preregistration audit. Newest entries last.
Git state at start: `master` = `2aa8292` (tagged `pre-v3-audit`), clean, synced
with origin. Work happens on branch `v3-audit`.

## 2026-10-19 — recovery

* The previous session (transcript `a33ebb07…`) had **finished**: v2 protocol,
  preregistration (`7713ee0`), confirm labels (`2303030`), confirm evaluation +
  paper rebuild (`2aa8292`), pushed. No uncommitted or half-written artifacts
  were found in the repo or in any session scratch directory. Nothing to resume;
  this round is a new hostile audit on top of v2.
* `analysis_v2/confirm.json` reproduces every number in RESULTS.md Part A.

## Audit findings so far

| # | Finding | Severity | Action |
|---|---|---|---|
| F1 | Preregistered sensitivity analysis "SciQ: containment label instead of judge label" (PREREGISTRATION §5) was never computed; `confirm.json` has no containment results. | protocol omission | compute now with the unchanged `v2_confirm.evaluate`, report as the preregistered sensitivity it was |
| F2 | PREREGISTRATION §2 describes D2 as "mean, min, std log-prob"; the registered code (`ORIGINAL_THREE_FEATURES`) and every reported D2 number use **mean log-prob, min log-prob, mean entropy**. The code is what ran; the document text is wrong. | documentation error | disclose; no number changes |
| F3 | Judge labels were written in-context by the same agent that designed the hypotheses (not blind to the hypotheses, only to model identity/features), and never checked against an independent labeler. | main validity threat | independent labeler: AllenAI TruthfulQA truth-judge (Llama-2-7B fine-tuned on TruthfulQA human labels) |
| F4 | D4/D5 cluster samples by MiniLM cosine ≥ 0.75, not the NLI bidirectional-entailment clustering of Kuhn et al.; on SciQ cosine is known to be lenient for short answers. The "self-consistency is worse" claim may be a weak-implementation artifact. | fairness of baseline | add NLI (deberta-large-mnli) semantic entropy / greedy support |

| F5 | Correct TruthfulQA answers are often negations/debunkings (26–49 % vs 8–18 % of imitative errors). Small models' P(True) judges negated statements "False" regardless of truth (AUROC 0.31–0.38 within negated answers), so P(True)'s overall inversion on TruthfulQA is largely a form artifact. | confound | every TruthfulQA error-type result re-tested within affirmative answers |

## Results of the audit so far (all exploratory except F1)

* F1 done: containment sensitivity D0 0.678 / 0.651 / 0.692, Holm p < 0.001; D2 − D0 small (+0.011 to +0.018).
* Length: length-quintile-matched AUROC within ±0.03 of raw for D0/D1/D2; length-only model 0.50–0.57; D2 adds +0.04 to +0.22 over a dev-trained length model. D2 is not a length proxy.
* H2 controls (TruthfulQA confirm): length-matched Δ −0.133 / −0.088 / −0.171, all CIs exclude 0 (Llama included); OLS with length, truncation, question log-prob, question length, others-wrong, category FE: imitative − other = −0.45 / −0.39 / −0.65 SD, CIs exclude 0.
* Within affirmative answers: D0 imitative 0.44 / 0.65 / 0.43 vs other 0.61 / 0.71 / 0.59; P(True) imitative 0.46 / 0.44 / 0.51 vs other 0.64 / 0.50 / 0.67.
* Confidently wrong: the 20 % most confident TruthfulQA answers are wrong 60 % / 32 % / 61 % of the time (base 57 / 53 / 62 %); 78–87 % of those errors are imitative. Abstaining on the 20 % riskiest by D0 leaves Qwen's TruthfulQA error rate unchanged (0.571 → 0.575).
* D2 transfer: SciQ-trained D2 on TruthfulQA 0.53–0.72 (vs TruthfulQA-trained 0.60–0.73): the TruthfulQA gain is partly dataset-specific. D2 partly detects imitative errors (0.56–0.68) where D0 cannot.
* Split integrity: no exact dev/confirm duplicates; 8 TruthfulQA template-variant pairs (e.g. "In France they speak French… In Austria/Australia…") and 3 SciQ paraphrase pairs at char-TF-IDF ≥ 0.8. Only the low-capacity D2/D3 were fitted on dev, so the leakage risk is negligible; reported.

## Infrastructure notes

* Network ~0.5–0.9 MB/s with resets; HF Xet stalls here. Models are fetched with a resumable
  parallel range downloader into `C:/Stuff/hf_models/` (outside the repo); checksums verified.
* `deberta-large-mnli` must run in fp32 under transformers 5.x (fp16 raises a dtype error).

## Checklist

- [x] F1 containment sensitivity
- [x] length / question controls for D2 and for H2 (imitative gap)
- [x] split integrity (exact + near-duplicate questions dev vs confirm)
- [x] confident-wrong composition; selective-prediction (risk–coverage) view
- [ ] independent truth-judge labels (TruthfulQA), agreement + robustness of H1–H3
- [ ] NLI semantic entropy baseline (running)
- [ ] answer-form (negation) control for H2 and P(True)  → in v3_eval `form`
- [ ] within-question test with hybrid NLI sample labels (v3_within.py)
- [ ] calibration, PR-AUC, dev-fixed operating points (v3_eval `operating`)
- [x] P(True) self-evaluation baseline (does the blind spot extend beyond token probability?)
- [ ] scale check (Qwen2.5-3B) — only if the above leaves time; labels from truth-judge
- [ ] results registry, paper rebuild, render QC, audit report
