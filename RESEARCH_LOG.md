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

## Checklist

- [ ] F1 containment sensitivity
- [ ] length / question controls for D2 and for H2 (imitative gap)
- [ ] split integrity (exact + near-duplicate questions dev vs confirm)
- [ ] confident-wrong composition; selective-prediction (risk–coverage) view
- [ ] independent truth-judge labels (TruthfulQA), agreement + robustness of H1–H3
- [ ] NLI semantic entropy baseline
- [ ] P(True) self-evaluation baseline (does the blind spot extend beyond token probability?)
- [ ] scale check (Qwen2.5-3B) — only if the above leaves time; labels from truth-judge
- [ ] results registry, paper rebuild, render QC, audit report
