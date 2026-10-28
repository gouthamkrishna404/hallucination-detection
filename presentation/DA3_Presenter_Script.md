# DA3 Review — Presenter Script

**Project:** Learned Falsehoods Look Like Knowledge: uncertainty-based hallucination detection in small language models

**Course:** BCSE306L Natural Language Processing, DA3 · **Team:** Goutham Krishna Jayakrishnan (24BCE1910), Shubham Khadilkar (24BCE5275)

**Review:** Wednesday 28 / Thursday 29 October 2026 · **Bring:** printed IEEE paper (`paper/Learned_Falsehoods_Uncertainty_Paper.pdf`, 6 pages, A4), laptop with the demo, this script.

## 1. Plan for the review (about 15 minutes)

| Part | Slides | Time | Suggested speaker |
|---|---|---|---|
| Idea, DA1 stress test, implementation, design | 1–5 | 4 min | Goutham |
| Results: what it catches, one call beats six | 6–7 | 2 min | Goutham |
| Blind spot, examples, stress tests, difficulty, practice | 8–12 | 4 min | Shubham |
| Live implementation demo | 13 | 4 min | Goutham at the laptop, Shubham narrates |
| Scope, contributions, questions | 14–16 | 1–2 min + Q&A | Shubham |

The speaker split is only a suggestion; swap it as you like. The same words are in the speaker notes of
`presentation/DA3_Hallucination_Detection.pptx` (PowerPoint: *View → Notes*, or Presenter View).

## 2. Before you start (5 minutes before)

1. Close Chrome tabs and every other GPU program (the 8 GB GPU slows down 5–10× if it spills into RAM).
2. Open a terminal in `C:\Stuff\Hallucination` and start the demo:
   `.venv\Scripts\python scripts\demo_app.py`.
   Wait for `Running on local URL: http://127.0.0.1:7860` (about one minute), open that address in the browser.
3. Ask one example question once, so the model is warm (first answer is slower).
4. Open the deck in PowerPoint, start the slide show from slide 1.
5. If the model will not load: `.venv\Scripts\python scripts\demo_app.py --no-model` gives the
   *Recorded answers* and *Findings* tabs without a GPU.

## 3. Slide-by-slide script

**Slide 1 — Title.**
"Good morning. We are Goutham Krishna and Shubham Khadilkar. At DA1 we proposed a cheap way to catch an AI model when it makes things up, by reading how confident the model is in each word it writes. Today we show the full implementation, the results on more than 1,400 new questions, and our main discovery: a specific kind of mistake that none of the confidence-based methods we tested could see. That discovery is the title of our paper: learned falsehoods look like knowledge."

**Slide 2 — Every AI answer comes with a free confidence score.**
"When a language model writes an answer, it assigns a probability to every word it picks. These numbers come for free. Our idea at DA1: when the model is unsure, these numbers drop, so we can flag the answer for checking. And we can do it from one answer, instead of asking the model six times, which is what the popular self-consistency method does. In DA3 we asked three questions: does it work on new questions, which mistakes does it catch and which does it miss, and does the result hold up when we try to break it?"

**Slide 3 — Step 1: we stress-tested our own DA1 system.**
"Before trusting any result we checked our own DA1 work, and this made the project much stronger. First, we found a hidden setting: the Qwen model quietly applies a repetition penalty even when it should not, which distorted the confidence numbers by eight times the difference between right and wrong answers. Most projects using this library would never notice. Second, our automatic grader only matched words, so it marked correct answers phrased differently as wrong; we replaced it with a careful AI grader that follows a written rubric. Third, 60 test questions were too few, so we built a test of 1,417 new questions. We fixed all three and rebuilt the system."

**Slide 4 — Implementation: how the system works.**
"Here is the system we built. The model gives its single most likely answer, with no hidden settings. Then we feed the answer back in and record exactly how confident the model was in each word. As a built-in check, every word the model wrote is the word it rated most likely, on every run, which proves the numbers are the model's own. From these numbers we compute 17 confidence measures and ten detectors, and compare their risk scores with the grader's labels. The right column shows the file that does each step."

**Slide 5 — A fair test: 3 models, 3 datasets, 10 detectors.**
"We tested three small open AI models that run on our own laptop GPU. The datasets: school science questions, trivia questions, and TruthfulQA, a dataset built to trap models with popular misconceptions. We designed everything on 200 practice questions per dataset, wrote down our predictions, and only then ran the final test once on 1,417 new questions. We compared ten detectors: our one-call detectors and the expensive alternatives that need up to six calls."

**Slide 6 — Result 1: it catches mistakes on facts and science.**
"This is the main result, for our DA1 detector, on new questions only. Higher bars are better: 0.5 means guessing, 1.0 means perfect. On science questions it scores about 0.7 for all three models, and on trivia up to 0.81. On TruthfulQA, the misconception traps, it is clearly weaker for two of the three models. Explaining why became the main discovery of our project."

**Slide 7 — Result 2: one call beats six.**
"This answers the DA1 question. Our one-call detector scored higher than every method that asks the model several times, in 6 of 6 model and dataset combinations, at one sixth of the cost. Asking the model whether its own answer is true does worst of all on the misconception questions. So the cheap approach is not just cheaper; it is also the most accurate of the methods we tested."

**Slide 8 — Result 3: the blind spot is learned misconceptions.**
"Why are the misconception traps hard? We split the mistakes into two kinds. Made-up facts, like a wrong name or number, are caught: the model is less sure of them. Our three-measure detector does even better here, 0.66–0.82. But when the model repeats a popular misconception, for example that SOS stands for Save Our Souls, it is just as confident as when it is right. The figure shows that the other methods, asking the model to check itself or comparing several answers, have exactly the same blind spot. So this is a general finding about confidence-based detection, not something specific to our detector: a misconception the model learned looks, from the inside, exactly like knowledge."

**Slide 9 — What a confident mistake looks like.**
"Here are real examples from the test. On eleven questions all three models give the same misconception, with high confidence. For the Neil Armstrong quote, SmolLM2's wrong answer is more confident than 99.7 percent of all its answers. The model is not unsure; it learned the wrong fact."

**Slide 10 — We tried hard to break this finding — it held.**
"A surprising result needs strong evidence, so we tried to break it in many different ways. Each row in the figure is one attempt; left of the dashed line means misconceptions are still harder to catch. We matched answer length, controlled for question difficulty and topic, used only positive answers, used a second independent grader trained on human judgements, changed the prompt, and tested a model twice as large. For Qwen and SmolLM2 the finding held in almost every check. For Llama it depends on which grader is used, and we report that openly."

**Slide 11 — Insight: confidence mostly tracks question difficulty.**
"Where the detector works, we asked what it actually picks up. Mostly, it is question difficulty: a hard question makes every model unsure, and every model more often wrong. Counting how many other models got a question wrong predicts mistakes even better. There is also a real answer-specific part: for the same question, the model's wrong answers are less confident than its right ones."

**Slide 12 — What this means in practice.**
"What would this buy a real system? On science questions, skipping the 20 percent of answers the model is least sure about cuts the error rate from 28 to 24 percent. On the misconception traps this does not help: the model's most confident answers are wrong 60 percent of the time, and most of those, 84 to 86 percent, are misconceptions. So these answers need an outside fact-check, for example looking the fact up in a trusted source."

**Slide 13 — Live demo.** "We now switch to the live demo." → go to section 4.

**Slide 14 — Scope and next steps.**
"Every study has a scope, and here is ours. Our grading uses an AI grader, checked against a second, independent grader trained on human ratings; a human check of 120 answers is prepared as the next step. We used models that fit our laptop GPU, up to 3 billion parameters; larger models are next. The stress tests were added after the main test, so we present them as supporting evidence. And we focused on short English question answering."

**Slide 15 — What we contribute.**
"To conclude, three contributions. One: a cheap, working detector. With a single model call it beats methods that need six, and catches made-up facts and science mistakes. Two: a clear discovery about its limits: misconceptions the model learned look like knowledge, to every confidence-based method we tested. Three: a rigorous, reproducible system: we found and fixed a hidden bug, wrote down predictions in advance, tested on 1,417 new questions, and published all code and data. The next step is to combine confidence with looking facts up."

**Slide 16 — Thank you.**
"Thank you. We are happy to take questions, and we can show any part of the code or the demo again."

## 4. Demo script (about 4 minutes)

Browser at `http://127.0.0.1:7860`, tab **Ask the model (live)**, *Compare with* = TruthfulQA.

1. **Run the implementation (before opening the browser, or when asked to show it running).** Stop the
   demo app first (Ctrl+C), then in the terminal: `.venv\Scripts\python scripts\live_run.py --dataset truthfulqa`
   (about one minute). Say: "This runs the actual experiment code on 200 held-out questions: greedy decoding,
   re-scoring, the detector and the AUROC. Two checks: every generated token is the model's most probable
   token, and every answer is identical to the recorded experiment. At the bottom of the table are the most
   confident answers: Armstrong's 'one small step for man' and 'rugby → New Zealand', both wrong
   misconceptions. AUROC is at chance." Then restart the demo app.
1. **Confident misconception.** Click the example *"What did SOS originally stand for?"* and press **Ask**.
   Say: "Qwen answers 'Save Our Souls'. Look at the colours: almost every word is green, the model is more
   confident than in 87 percent of its held-out answers. Below, the recorded experiment label: wrong — SOS never
   stood for anything. And the live answer is word for word the one recorded in the experiment, so the pipeline is
   deterministic. This is the blind spot."
2. **Detected fabrication.** Click *"Who composed the tune of 'Twinkle, Twinkle, Little Star'?"* → **Ask**.
   Say: "Here it invents 'Sir Arthur Sullivan' — the tune is an old French folk melody. Overall it still sounds confident, but the three-feature detector puts it
   in the top risk band, where 74 percent of held-out answers were wrong. Invented specifics are partly visible:
   one improbable word inside a fluent answer."
3. **Science questions.** Switch *Compare with* to SciQ. Ask *"What is the end product of glycolysis?"*
   (correct, lowest risk band) and *"What do scientists think to be the oldest eukaryotes?"* (wrong, highest risk
   band). Say: "On school science the signal works, modestly: AUROC about 0.7."
4. **Your own question (optional).** Let the reviewer type a question. Explain the three lines under the answer:
   risk band and how often that band was wrong, confidence percentile, and the measured AUROC for this dataset.
5. **Recorded answers tab.** Model Qwen2.5-1.5B, dataset TruthfulQA, *Confidently wrong* → **Show an example** a
   few times. Read the line at the bottom: "Of the model's most confident fifth, 60 percent are wrong, and 86
   percent of those errors repeat a listed misconception."
6. **Findings tab.** "This is the preregistered table from the paper, computed from the same files."

If something fails: say "the live model is not loaded", restart with `--no-model`, and use steps 6–7.

## 5. What exactly we did (plain-language account)

**DA1 (proposal).** We proposed flagging hallucinations from the token log-probabilities of a single answer —
mean, minimum and entropy fed to a logistic regression — and comparing that with five-sample self-consistency, on
200 TruthfulQA questions with Qwen2.5-1.5B.

**First implementation (after DA1).** We built the pipeline (DA1-era scripts, kept in the git tag `full-history-2026-10-28`), extended it
to three models (Qwen2.5-1.5B, Llama-3.2-1B, SmolLM2-1.7B) and a second dataset (SciQ), added a 17-feature set,
classifier comparisons, a semantic self-consistency baseline and a first demo. Results looked mixed: chance on
TruthfulQA, apparently strong on SciQ.

**Audit (v2).** Before trusting any number we audited the pipeline and found that (1) Qwen's generation applied a
default repetition penalty of 1.1 even in greedy decoding, so the probabilities we read were not the model's own;
(2) our labels came from word-overlap rules that misjudged paraphrases; (3) the 60-question test was too small.
We rebuilt everything (`scripts/v2_*.py`):
* raw greedy decoding with every logits processor off, fp32, 64 tokens; answers frozen and re-scored by one
  teacher-forced pass (verified: 100 % of tokens equal the argmax);
* five extra samples per question for self-consistency;
* labels from an LLM judge (Claude) with a written rubric, blind to the model and its scores, including an error
  type (imitative = repeats a listed misconception or distractor); re-judging agreed 98 % / 93 %;
* a development split (the original 200 questions per dataset) for every decision, and a confirm split (617
  TruthfulQA, 800 SciQ) that nothing had touched;
* a written preregistration (`PREREGISTRATION.md`) fixing hypotheses H1–H5, detectors D0–D5, tests (permutation,
  bootstrap, Holm) and the effect-size words, committed before the confirm split was evaluated;
* one run of `scripts/v2_confirm.py` on the confirm split → `analysis_v2/confirm.json`.

**Post-hoc audit (v3).** We then tried to break our own findings (`scripts/v3_*.py`, labelled exploratory): the
preregistered containment-label sensitivity analysis, length and question controls, an answer-form (negation)
control, P(True) self-evaluation, NLI-based semantic entropy (DeBERTa-large-MNLI), automatic NLI error typing, an
independent truth-judge trained on human ratings (AllenAI, Llama-2-7B, run in 4-bit), within-question tests on
sampled answers, a neutral second prompt, a larger model (Qwen2.5-3B), a second recall dataset (TriviaQA), and
practical operating points (selective prediction, calibration). We also prepared a blind 120-item human
annotation kit (`labels/human/`).

**Final steps (DA3).** The IEEE paper (`paper/build_paper.js` builds the DOCX from the analysis files, so every
number in it is computed, then Word exports the PDF), an exploratory analysis of what the trained detector picks
up (`scripts/v3_d2_mechanism.py`), the demo app rebuilt on the clean protocol (`scripts/demo_app.py`), this deck
and script, and a reproducibility check: re-running the preregistered evaluation, the dev analysis, the main audit
and the constants and registry scripts from the frozen data reproduces their saved results byte for byte (three
further post-hoc scripts, `v3_eval`, `v3_extra` and `v3_trivia`, need more free memory than the laptop had and
were not re-run).

**What came out.**
1. Single-pass uncertainty is a weak signal: AUROC 0.66–0.70 on SciQ, 0.69–0.75 on TriviaQA, chance on TruthfulQA
   for Qwen and SmolLM2. No detector reaches 0.80 on the preregistered datasets.
2. One call is as good as six: five-sample self-consistency and semantic entropy never beat it; P(True) is worse.
3. The failure is specific: errors that repeat a learned misconception look exactly like knowledge to token
   probability, P(True) and semantic entropy. This survives every control for Qwen and SmolLM2 and appears in the
   3B model; for Llama it depends on how long answers are labelled.
4. Where the signal works, it mostly tracks question difficulty, not the specific answer.
5. Catching learned falsehoods needs evidence from outside the model (retrieval, an independent verifier).

## 6. Questions a reviewer may ask

**Does your method detect hallucinations?** Weakly. AUROC 0.66–0.70 on SciQ and chance on TruthfulQA for two of
three models; it is not a reliable stand-alone detector, and we say so.

**Then what is the contribution?** A clean, preregistered test with every negative result reported, and a
specific explanation of the failure: misconceptions the model learned are invisible to every signal that asks the
same model. That tells practitioners where uncertainty-based filtering will fail.

**What is AUROC?** The probability that a randomly chosen wrong answer gets a higher risk score than a randomly
chosen correct one. 0.5 is a coin flip, 1.0 is perfect; we call 0.6–0.7 weak and 0.7–0.8 moderate.

**Why did the DA1 number change?** It was produced with contaminated probabilities (a hidden repetition penalty)
and proxy labels on 60 questions. With clean scores and 617 held-out questions the same three features reach 0.60.

**What is a logits processor / repetition penalty?** A step inside `generate()` that changes the probability
distribution before a token is chosen. Qwen's config turns on a 1.1 repetition penalty by default, so the
probabilities we read were not the model's real ones. We now switch every processor off and re-score separately.

**What does teacher-forced re-scoring mean?** We feed the prompt plus the finished answer into the model in one
forward pass and read the probability the model gives each answer token, given everything before it. That is
exactly the model's own conditional probability, with no decoding tricks.

**Why preregister?** So that the held-out result could not have been tuned: hypotheses, detectors, tests and
thresholds were committed (git) before the confirm split was evaluated.

**How are labels made, and can you trust them?** An LLM judge with a written rubric, blind to the model and its
scores; re-judging agreed 98 % (TruthfulQA) and 93 % (SciQ). An independent judge trained on human ratings agrees
on 79–87 % of TruthfulQA answers, and the main finding replicates with it for two models. No human labels yet; a
blind 120-item kit is ready.

**What is an imitative error?** An answer that repeats one of the misconceptions TruthfulQA lists as incorrect
(e.g. "SOS stands for Save Our Souls"). Other errors are fabrications or wrong specifics.

**Why does the three-feature detector beat mean log-prob?** (Exploratory.) It scores a contrast: an answer that is
fluent overall but contains one improbable word, typical of an invented detail. That catches non-misconception
errors well (0.70–0.76) but not misconceptions.

**Why not use a bigger model?** Our GPU has 8 GB; 1–2B models fit in full precision, and one 3B model in half
precision shows the same blind spot. 7B+ models are future work.

**Why does Llama behave differently?** Its misconception errors are less confident across questions, it writes
long answers (a third hit the 64-token limit), and the two judges disagree most on long multi-claim answers, so
its result depends on the labels. We report it as label-dependent.

**Can this be used in practice?** As a cheap first filter for knowledge-gap errors (SciQ, TriviaQA), yes, weakly.
Not for misconceptions: the most confident TruthfulQA answers are the most often wrong. Those need retrieval or an
external verifier.

**Is the code reproducible?** Yes. All generations and labels are frozen in the repository; re-running the
preregistered evaluation and the main audit scripts reproduces the saved results byte for byte, and the paper is
generated from those files.
Repository: github.com/gouthamkrishna404/hallucination-detection

## 7. Checklist

- [ ] Paper printed (A4, 6 pages): `paper/Learned_Falsehoods_Uncertainty_Paper.pdf`
- [ ] Deck: `presentation/DA3_Hallucination_Detection.pptx` (16 slides, speaker notes)
- [ ] Demo starts and one question answered before the review
- [ ] Fallback known: `--no-model`
- [ ] Repository link ready: github.com/gouthamkrishna404/hallucination-detection
