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

## 2026-10-19 — later findings

* **Reproducibility:** `python scripts/v2_confirm.py` re-run from frozen data reproduces
  `analysis_v2/confirm.json` byte for byte (git diff empty).
* **NLI semantic entropy** (D7–D9): ties D0 on SciQ (n.s.), at chance on TruthfulQA for Qwen/SmolLM2,
  below D0 for Llama. Corrects the v2 claim "self-consistency is worse than D0" → true only for the
  cosine variant. Also misses imitative errors.
* **D5 on Qwen/TruthfulQA is significantly better than D0** (+0.07, Holm p = 0.028) — only because D0
  is slightly inverted there; D5 itself is at chance. v2 text "never better" was imprecise.
* **Containment label** (F6): on the confirm split it accepts 2–4 % of wrong answers and rejects
  28–30 % of correct ones (v2 said "never accepted a wrong answer… rejected 23 % of paraphrases";
  the 23 was 23 of 100 audited items). Corrected in the paper from `within.json`.
* **Within-question SciQ test**: containment 0.549–0.631; other-model-anchored hybrid 0.583–0.655;
  agreed-only 0.559–0.658, all p ≤ 0.02. The own-greedy-anchored hybrid (0.605–0.687) is biased upward
  by construction and is not used.
* **NLI error typing** agrees with Claude typing 81–86 % (κ 0.60–0.69); H2 with NLI types:
  −0.20 / −0.12 / −0.21, Holm p ≤ 0.002 for all three models.
* **GPU contention lesson:** two GPU jobs on the 8 GB card silently spill into system memory (100 %
  util at ~22 W) and run 5–10× slower. Run GPU jobs strictly one at a time.

## Commands (v3)

```bash
python scripts/v3_audit.py                       # containment sensitivity, length/H2 controls, split, coverage
python scripts/v3_ptrue.py --model <key>         # P(True), per model
python scripts/v3_nli.py --nli <deberta-dir>     # NLI semantic entropy
python scripts/v3_nli.py --nli <deberta-dir> --types-only     # automatic error types
python scripts/v3_within.py --nli <deberta-dir>  # within-question test, corrected sample labels
python scripts/v3_truthjudge.py --judge <judge-dir>           # independent labels
python scripts/v2_generate.py --model <key> --dataset truthfulqa --split confirm --no-samples     --system-prompt "<neutral prompt>" --root results_v3/prompt2                # prompt variant
python scripts/v3_eval.py && python scripts/v3_extra.py && python scripts/v3_constants.py
python scripts/v3_figures.py && (cd paper && node build_paper.js)
```

## Infrastructure notes

* Network ~0.5–0.9 MB/s with resets; HF Xet stalls here. Models are fetched with a resumable
  parallel range downloader into `C:/Stuff/hf_models/` (outside the repo); checksums verified.
* `deberta-large-mnli` must run in fp32 under transformers 5.x (fp16 raises a dtype error).

## Checklist

- [x] F1 containment sensitivity
- [x] length / question controls for D2 and for H2 (imitative gap)
- [x] split integrity (exact + near-duplicate questions dev vs confirm)
- [x] confident-wrong composition; selective-prediction (risk–coverage) view
- [x] independent truth-judge labels (TruthfulQA), agreement + robustness of H1–H3
- [x] NLI semantic entropy baseline
- [x] answer-form (negation) control for H2 and P(True)  → in v3_eval `form`
- [x] within-question test with hybrid NLI sample labels (v3_within.py)
- [x] calibration, PR-AUC, dev-fixed operating points (v3_eval `operating`)
- [x] P(True) self-evaluation baseline (does the blind spot extend beyond token probability?)
- [x] scale check (Qwen2.5-3B) — only if the above leaves time; labels from truth-judge
- [x] results registry, paper rebuild, render QC, audit report

## 2026-10-19 — final

* Truth-judge (4-bit, checksums verified, smoke-tested) labels for all greedy, sampled and prompt-variant
  answers. TruthfulQA agreement with the primary judge κ 0.74 / 0.59 / 0.67; SciQ κ ≈ 0.2 (not used).
* Independent H2 (truth-judge + NLI types): −0.20 / +0.01 / −0.17; within question 0.45 / 0.40 / 0.42
  (imitative) vs 0.45–0.56 (other); neutral prompt −0.19 / −0.06 / −0.14; Qwen2.5-3B −0.13 [−0.22, −0.04].
* Llama's preregistered TruthfulQA D0 detection (0.653) is 0.522 under truth-judge labels (0.607 on
  consensus items) — reported as label-dependent.
* Paper: `paper/Learned_Falsehoods_Uncertainty_Paper.{docx,pdf}` (6 pages), all numbers from JSON,
  `scripts/v3_constants.py --check` passes; v2 paper archived in `paper/archive_v2/`.
* Not done: human labels; models ≥ 7 B (bandwidth ~0.5–0.9 MB/s made a 15 GB download impractical);
  a second misconception dataset.

## 2026-10-20 — scale check dropped, TriviaQA added, human-label kit

* 4-bit Qwen2.5-7B / Llama-3.1-8B check abandoned at the user's request (≈ 6–7 h of download left at
  ~0.35 MB/s); partial files deleted. `src/model_utils.py` keeps a guard so pre-quantized models load.
* TriviaQA (rc.nocontext validation, 1,000 questions, seed 42; 200 dev / 800 confirm), frozen v2 protocol,
  judge-free alias labels: D0 0.753 / 0.694 / 0.730, D2 0.81 / 0.70 / 0.80, difficulty proxy 0.81–0.86.
  `python scripts/v3_trivia.py` → `analysis_v3/triviaqa.json`. Added to the paper as §IV-G.
* Human validation kit: `labels/human/annotation_sheet.csv` (120 blind items: 90 judge-disagreement,
  30 agreement), `key.csv`, `README.md`; score with `python scripts/v3_human.py score`.

## 2026-10-25 — D2 mechanism, demo, reproducibility re-run (no downloads)

* **D2 mechanism (exploratory, `scripts/v3_d2_mechanism.py` → `analysis_v3/d2_mechanism.json`).** D2's
  weights are positive on mean log-prob *and* mean entropy and negative on min log-prob, so it scores a
  contrast, not low confidence. The parameter-free contrast mean − min log-prob ("fluent overall, one
  improbable token") detects non-imitative TruthfulQA errors at 0.74 / 0.76 / 0.70 (D0: 0.58 / 0.71 /
  0.63); on imitative ones it is at chance for Qwen and SmolLM2 (0.49, 0.52) and weaker for Llama (0.67). For Qwen, mean + entropy gets a *negative* sign on
  dev and then partly catches imitative errors (0.55 [0.50, 0.60]) while inverting on the others (0.36).
  D2's own imitative-minus-other gap stays negative: −0.08 [−0.17, 0.00] Qwen (p = 0.05), −0.14 Llama,
  −0.18 SmolLM2 (p < 0.001; unadjusted). Error rate by D0 quintile is flat on TruthfulQA for Qwen and
  SmolLM2 (0.47–0.69) and monotone on SciQ and Llama/TruthfulQA. Contrasts are not length proxies
  (length-stratified AUROCs within 0.03). One sentence added to the paper (§IV-C), marked post hoc.
* **Demo rebuilt** (`scripts/demo_app.py`, `DEMO.md`) on the v2 protocol: live raw-greedy answer with
  fp32 re-scoring, token map, D0/D2 against the confirm-split distribution, recorded-answer browser
  (no GPU), findings tab. The v1 demo (proxy-label classifier, PASS/WARN/FLAG) is retired. All five
  example questions reproduce their recorded confirm-split answers exactly.
* `src/token_viz.py`: explicit dark text on the coloured token backgrounds (unreadable in dark mode).
* Reproducibility re-run from frozen data (2026-10-25/26): `v2_analysis`, `v2_confirm`, `v3_audit`,
  `v3_constants`, `v3_registry` reproduce their committed outputs byte for byte. `v3_eval` was stopped twice by
  the system for low memory (≈ 2 GB free on the 15 GB laptop), so `v3_eval`, `v3_extra`, `v3_trivia` were not
  re-checked.
* DA3 presentation: `presentation/build_deck.py` builds `DA3_Hallucination_Detection.pptx` (16 slides on the
  DA1 template, numbers from the analysis JSON, script in speaker notes) and the presenter script HTML; Word
  exports `DA3_Presenter_Script.{docx,pdf}`.

## 2026-10-28 — public repository cleanup and live run

* Tagged the previous state as `full-history-2026-10-28`, then removed the DA1/v1 material the final
  work replaced from the tree: 37 scripts (`01_*`–`03_*`, `experiment_*` and helpers), `results/`,
  `comparison/`, `outputs/`, `paper/archive_v2/`, the v1 paper figures and six unused `src/` modules.
  Nothing was lost: it is all in the tag and in git history.
* The three DA1 files the audit cites (penalised and clean Qwen features, DA1 metrics) moved to
  `evidence_da1/`; `scripts/v3_constants.py` reads them there and reproduces `analysis_v3/constants.json`
  byte for byte. `src/config.py` and `src/registry.py` no longer create the old output folders on import.
* Added `scripts/live_run.py`: the whole method on 200 held-out questions in about a minute, using the
  experiment's own functions. Checked: Qwen SciQ 200/200 answers identical to the recorded run, 100 %
  argmax agreement, AUROC 0.650; TruthfulQA 200/200 identical, AUROC 0.487 [0.41, 0.57].
* README rewritten as a short overview of the final project; RESULTS.md Part B (v1) removed.
* Demo example fix: Qwen answers the Twinkle Twinkle question with "Sir Arthur Sullivan" (checked twice);
  the script said "Lewis Carroll". Corrected in DEMO.md and the presenter script.
