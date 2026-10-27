# Implementation demo — DA3 review

Presentation: `presentation/DA3_Hallucination_Detection.pptx` (16 slides, script in the speaker notes) and
the full presenter script `presentation/DA3_Presenter_Script.{md,pdf,docx}` (timing, slide-by-slide words,
demo clicks, plain-language account of the work, review questions). Rebuild with
`python presentation/build_deck.py`.

Paper (hard copy): `paper/Learned_Falsehoods_Uncertainty_Paper.pdf` — A4, 6 pages, IEEE two-column.
Editable source: `paper/Learned_Falsehoods_Uncertainty_Paper.docx`.

## Before the review

```bash
cd C:\Stuff\Hallucination
.venv\Scripts\python scripts\demo_app.py              # live model + recorded tabs, http://127.0.0.1:7860
.venv\Scripts\python scripts\demo_app.py --no-model   # fallback: no GPU needed
```

* Nothing is downloaded: the models are in the Hugging Face cache and the app runs with
  `HF_HUB_OFFLINE=1`. Start-up takes about a minute (recorded data ≈ 15 s, model ≈ 15 s); each live
  question takes about 2 s on the RTX 4060.
* Close every other GPU program first (an 8 GB card that spills into system memory gets 5–10× slower).
* `--model llama-3.2-1b-instruct` or `--model smollm2-1.7b-instruct` switches the live model
  (SmolLM2 in fp32 is tight on 8 GB).
* If the GPU or the model fails, the app still starts and the *Recorded* and *Findings* tabs work.

## Running the implementation live (about 1 minute)

```bash
.venv\Scripts\python scripts\live_run.py                       # Qwen, 200 held-out SciQ questions
.venv\Scripts\python scripts\live_run.py --dataset truthfulqa  # the misconception blind spot
```

The script prints each pipeline step with the function that does it: load questions → load the model in
fp32 → raw greedy decoding → teacher-forced re-scoring → D0 → comparison with the recorded experiment →
AUROC with a 95 % CI. What to point out:

* "generated token == most probable token for 100.0 %": the probabilities are the model's own (the
  bug found in DA1 broke exactly this).
* "live answer identical to the recorded experiment for 200/200": the pipeline is deterministic, so
  the paper's results can be regenerated.
* SciQ AUROC ≈ 0.65 (weak signal). On TruthfulQA, AUROC ≈ 0.49, and the *most confident* rows at the
  bottom of the table are wrong misconceptions (Armstrong's "one small step for man", "rugby → New
  Zealand"): the blind spot, live.

Close the demo app first; running both at once exceeds the 8 GB of GPU memory.

## Live coding in front of the reviewer (about 5 minutes, tested)

Each change is one line, then one command. Undo each change before the next one.

**1. Run the method as it is.**
`.venv\Scripts\python scripts\live_run.py` → 200/200 answers identical to the recorded run, 100.0 % of
tokens = argmax, AUROC ≈ 0.650.

**2. Change the detector.** In `scripts/live_run.py`, step 5, replace `-np.mean(r["logprobs"])` with
`-np.min(r["logprobs"])` (D1, the least confident token instead of the average). Run again → AUROC
≈ 0.674 [0.60, 0.75]. Say: "The weakest token carries a bit more signal than the average, but the
intervals overlap; the paper tests this properly on 800 questions." Change it back.

**3. Re-create the DA1 bug.** In `scripts/v2_generate.py`, function `generate`, change
`repetition_penalty=1.0` to `repetition_penalty=1.1` (what Qwen's default config silently applied in
DA1). Run `.venv\Scripts\python scripts\live_run.py --n 100` → argmax agreement falls to **90.1 %**,
and only **29/100** answers match the recorded run. Say: "This is the bug we found in our own DA1 work.
The penalty changes which words the model picks, so the probabilities we read were no longer the
model's own. Our pipeline now checks this on every run." Change it back (or `git checkout
scripts/v2_generate.py`).

**4. Show the blind spot.** `.venv\Scripts\python scripts\live_run.py --dataset truthfulqa` → AUROC ≈ 0.49;
the most confident answers at the bottom of the table are wrong misconceptions.

**5. Let the reviewer ask a question** in the demo app (`scripts\demo_app.py`; close it before running
`live_run.py`, both together do not fit in 8 GB).

## Five-minute walkthrough

1. **The question (30 s).** An LLM's own token probabilities are free with every answer. Can they tell
   us when the answer is a hallucination? We test it on three small open models, preregistered, on
   1,417 held-out questions.
2. **Live, a misconception — *Ask the model* tab, example "What did SOS originally stand for?"**
   Qwen answers "Save Our Souls". The token map is almost all green: the model is more confident than in
   87 % of its held-out answers. The answer is wrong (SOS stands for nothing; the recorded judge label
   appears below). This is the main finding: errors that repeat a popular misconception look like knowledge.
3. **Live, a fabrication — "Who composed the tune of 'Twinkle, Twinkle, Little Star'?"**
   Qwen invents "Sir Arthur Sullivan" (the tune is a French folk melody). Overall confidence is high, but the 3-feature detector (D2) puts it in
   the top risk band, where 74 % of held-out answers were wrong. Fabricated specifics *are* partly
   detectable.
4. **Live, SciQ — "What is the end product of glycolysis?"** (correct, low risk) and
   **"What do scientists think to be the oldest eukaryotes?"** (wrong, high risk). On school science the
   signal works modestly (AUROC ≈ 0.7).
5. **Recorded tab — Qwen2.5-1.5B, TruthfulQA, "Confidently wrong".** Of the model's most confident fifth
   of answers, 60 % are wrong and 86 % of those errors repeat a misconception. Click a few times.
6. **Findings tab.** The preregistered table: weak signal everywhere, chance on TruthfulQA for two of
   three models, and the misconception / other-error split that explains it.

Every live answer is generated with the exact experiment protocol; for the example questions it
reproduces the recorded answer word for word, which shows the pipeline is deterministic.

## What runs where (implementation map)

| Step | Code | Output |
|---|---|---|
| Questions, dev/confirm split | `scripts/v2_generate.py` (`load_split`) | `data/<dataset>/` |
| Raw greedy decoding + fp32 teacher-forced re-scoring | `scripts/v2_generate.py` | `results_v2/<model>/<dataset>/<split>/greedy.jsonl` |
| Token features (17, incl. mean/min log-prob, entropy) | `src/features_v2.py`, `scripts/v2_features.py` | `features.csv` |
| Blind judge labels (rubric) | `scripts/v2_labels.py`, `labels/JUDGE_RUBRIC.md` | `labels/judge/` |
| Preregistered evaluation (H1–H5, Holm) | `scripts/v2_confirm.py` | `analysis_v2/confirm.json` |
| Audit: controls, P(True), NLI semantic entropy, independent judge, TriviaQA, 3B model | `scripts/v3_*.py` | `analysis_v3/` |
| Paper (all numbers read from the JSON files) | `paper/build_paper.js` | `paper/*.docx`, `*.pdf` |
| Live run of the whole method | `scripts/live_run.py` | printed |
| Demo | `scripts/demo_app.py`, `src/token_viz.py` | — |

## Questions a reviewer is likely to ask

**Does the method detect hallucinations?** Weakly. AUROC 0.66–0.70 on SciQ, 0.69–0.75 on TriviaQA, and
chance (0.47, 0.50) on TruthfulQA for Qwen and SmolLM2. No AUROC reaches 0.80 on the preregistered
datasets. It is not a reliable stand-alone detector.

**Why does it fail on TruthfulQA?** Because most of the errors there repeat a popular misconception that
the model learned during training. The model is as sure of these as of correct answers (AUROC 0.42 and
0.45 for those errors), while other errors are detectable (0.58 and 0.63). The same blind spot appears in
P(True) self-evaluation and in NLI semantic entropy, so it is not specific to token probability.

**Is the SciQ signal real?** Partly. Most of it is question difficulty: how many *other* models got the
question wrong predicts errors better (0.77–0.82) than the model's own confidence. Within a single
question, wrong sampled answers still have lower probability (0.55–0.66), so a small answer-specific
part exists.

**How were answers labelled?** By an LLM judge with a fixed rubric, blind to the model and its scores
(re-judging agreement 98 % TruthfulQA, 93 % SciQ). Main results were re-checked with an independent judge
trained on human ratings (AllenAI truth-judge). There are no human labels yet; a 120-item blind annotation
kit is ready in `labels/human/`.

**Why preregister?** Earlier versions of this project produced numbers that changed with each analysis
choice (and one was contaminated by a default repetition penalty in decoding). Fixing the hypotheses and
tests before looking at the held-out split means the confirm results cannot have been tuned.

**Why does the 3-feature detector (D2) beat mean log-prob?** (Exploratory analysis,
`scripts/v3_d2_mechanism.py`.) Its weights reward a *contrast*: an answer that is fluent overall but
contains one improbable token. That contrast (mean − min log-prob) detects the non-misconception errors
on TruthfulQA well (AUROC 0.70–0.76, versus 0.58–0.71 for mean log-prob), because averaging hides a single
invented detail. It does not fix the blind spot: D2 still scores misconception errors below other errors
(gap −0.14 Llama, −0.18 SmolLM2, both p < 0.001; −0.08 for Qwen, borderline, p = 0.05).

**Why such small models?** An 8 GB laptop GPU running fp32. One larger model (Qwen2.5-3B) shows the same
blind spot; models of 7 B and up are untested.

**What is the contribution?** A clean, preregistered negative result with a specific explanation: confidence
read from the model cannot catch falsehoods the model has learned, so catching them needs information from
outside the model (retrieval, an external verifier).
