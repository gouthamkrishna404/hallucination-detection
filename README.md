# Learned Falsehoods Look Like Knowledge

**Uncertainty-based hallucination detection in small language models**
BCSE306L Natural Language Processing (DA3) · Goutham Krishna Jayakrishnan (24BCE1910), Shubham Khadilkar (24BCE5275)

**Question.** A language model gives each word it generates a probability, at no extra cost. Can these
token-level uncertainties, read from a single answer, detect hallucinations? And are they as good as
more expensive methods that sample the model several times or ask it to check itself?

**Answer.** These signals detect hallucinations only weakly, and they fail in one specific way.

| | Qwen2.5-1.5B | Llama-3.2-1B | SmolLM2-1.7B |
|---|---|---|---|
| SciQ (school science), AUROC of −mean log-prob | 0.69 | 0.66 | 0.70 |
| TruthfulQA (misconceptions), AUROC of −mean log-prob | 0.47 (chance) | 0.65* | 0.50 (chance) |

Results are preregistered and use 617 TruthfulQA and 800 SciQ held-out questions. *Llama's TruthfulQA
result depends on the labels: it falls to 0.52 with an independent judge's labels.

* **Misconceptions are invisible.** Answers that repeat a popular misconception ("SOS stands for Save Our
  Souls") score like correct answers. P(True) self-evaluation and NLI semantic entropy share this blind
  spot. For Qwen and SmolLM2 it holds under every control we tried and in Qwen2.5-3B.
* **One call is as good as six.** Five-sample self-consistency never beats the one-call score.
* **The signal mostly tracks difficulty.** On SciQ, the uncertainty that works mainly reflects how hard
  the question is.
* **Catching learned falsehoods needs outside evidence**, such as retrieval or an independent verifier.

**Paper:** [`paper/Learned_Falsehoods_Uncertainty_Paper.pdf`](paper/Learned_Falsehoods_Uncertainty_Paper.pdf)
(IEEE, 6 pages). **Audit and reviewer assessment:** [AUDIT_REPORT.md](AUDIT_REPORT.md).
**Every headline number with its source file:** [RESULTS_REGISTRY.json](RESULTS_REGISTRY.json).

## Method

```
question ──► raw greedy decoding (fp32, no repetition penalty, no sampling)
         ──► answer frozen, re-scored in one teacher-forced fp32 pass
             (check: 100 % of generated tokens = the model's most probable token)
         ──► per-token log-prob / entropy / margin ──► 17 features ──► detectors D0–D9
         ──► compared with blind judge labels: AUROC, bootstrap CI, permutation test, Holm
```

| Detector | Model calls | Signal |
|---|---|---|
| D0 | 1 | −mean token log-prob |
| D1 | 1 | −min token log-prob |
| D2 | 1 | logistic regression on mean log-prob, min log-prob and mean entropy (DA1 proposal) |
| D3 | 1 | logistic regression on 17 features |
| D4, D5 | 6 | self-consistency over 5 samples: cosine agreement, cosine semantic entropy |
| D6 | 2 | P(True): the model judges its own answer |
| D7–D9 | 6 | NLI semantic entropy and NLI support (DeBERTa-large-MNLI) |

The hypotheses, detectors and tests were fixed in [PREREGISTRATION.md](PREREGISTRATION.md) before the
held-out (confirm) split was evaluated.

## Run it

```bash
python -m venv .venv && .venv\Scripts\activate      # Windows; source .venv/bin/activate elsewhere
pip install -r requirements.txt

python scripts/live_run.py                          # whole method live on 200 held-out SciQ questions, ~1 min
python scripts/live_run.py --dataset truthfulqa     # same on TruthfulQA: the misconception blind spot
python scripts/demo_app.py                          # interactive demo, http://127.0.0.1:7860
python scripts/demo_app.py --no-model               # recorded answers and findings, no GPU needed
```

`live_run.py` uses the experiment's own functions. It checks that every live answer matches the
recorded one, that every generated token is the model's argmax, and prints the AUROC. The demo
walkthrough is in [DEMO.md](DEMO.md).

Re-running the evaluation from the frozen generations and labels needs no GPU:

```bash
python scripts/v2_confirm.py      # preregistered test        -> analysis_v2/confirm.json
python scripts/v3_audit.py        # controls, length, D2       -> analysis_v3/audit.json
python scripts/v3_constants.py    # audit constants            -> analysis_v3/constants.json
python scripts/v3_registry.py     # headline-number registry  -> RESULTS_REGISTRY.json
cd paper && npm install && node build_paper.js   # paper DOCX, every number read from the JSON files
```

These outputs reproduce byte for byte. The full command history is in [RESEARCH_LOG.md](RESEARCH_LOG.md).

## Repository

| Path | Contents |
|---|---|
| `src/` | model wrapper and per-token scoring (`model_utils.py`), features, statistics, dataset loaders |
| `scripts/v2_*.py` | the preregistered pipeline: generate → features → labels → analysis → confirm |
| `scripts/v3_*.py` | post-hoc audit: P(True), NLI, independent truth-judge, controls, TriviaQA, figures |
| `scripts/live_run.py`, `scripts/demo_app.py` | live run and interactive demo |
| `configs/` | model and dataset registry |
| `data/` | fixed dev (200) and confirm question sets for TruthfulQA, SciQ and TriviaQA |
| `results_v2/`, `results_v3/` | frozen generations (tokens, log-probs, entropies) and features for every model and dataset |
| `labels/` | judge rubric, judge labels, and a blind 120-item human annotation kit (`labels/human/`) |
| `analysis_v2/`, `analysis_v3/` | every computed result (JSON) |
| `paper/` | IEEE paper (DOCX and PDF), its builder and figures |
| `presentation/` | DA3 review deck and presenter script |
| `evidence_da1/` | the DA1 files behind the audit's correction (the hidden repetition penalty) |

Hardware: one RTX 4060 laptop GPU (8 GB). Models: Qwen2.5-1.5B-Instruct, Llama-3.2-1B-Instruct,
SmolLM2-1.7B-Instruct; Qwen2.5-3B-Instruct for the scale check.

The DA1-era scripts and results that this work replaced are kept in the git tag
`full-history-2026-10-28`.

**Limitations.** No human labels yet; the annotation kit is ready. The models are small (3B parameters at
most). The robustness checks were designed after the preregistered test.
