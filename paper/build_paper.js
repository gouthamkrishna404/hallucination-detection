// Builds paper/Token_Uncertainty_Hallucination_Detection.docx (IEEE-style, two-column, A4).
// Layout follows the ParkPulse reference: A4, Times New Roman, 0.63" side margins,
// 0.25" column gap, small-caps section heads, 8 pt captions, three-rule tables.
// Every number is taken from RESULTS.md / comparison/final_results.json, which are
// generated from saved experiment outputs.
// Usage: cd paper && npm install && node build_paper.js
const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, ImageRun, Table, TableRow, TableCell, AlignmentType,
  BorderStyle, WidthType, SectionType, ColumnBreak, TabStopType, LevelFormat,
  Math: OMath, MathRun, MathSubScript, MathSuperScript, MathSum, MathFraction,
} = require("docx");

const COL = 4866; // twips: (11906 - 2*907 - 360) / 2
const FIG = (f) => path.join(__dirname, "figures", f);

// ---------- inline markup: *italic*, **bold**, _{sub}, ^{sup} ----------
function runs(text, base = {}) {
  const out = [];
  const re = /(\*\*[^*]+\*\*|\*[^*]+\*|_\{[^}]+\}|\^\{[^}]+\})/g;
  let last = 0, m;
  while ((m = re.exec(text))) {
    if (m.index > last) out.push(new TextRun({ text: text.slice(last, m.index), ...base }));
    const t = m[0];
    if (t.startsWith("**")) out.push(new TextRun({ text: t.slice(2, -2), ...base, bold: true }));
    else if (t.startsWith("*")) out.push(new TextRun({ text: t.slice(1, -1), ...base, italics: true }));
    else if (t.startsWith("_")) out.push(new TextRun({ text: t.slice(2, -1), ...base, subScript: true }));
    else out.push(new TextRun({ text: t.slice(2, -1), ...base, superScript: true }));
    last = m.index + t.length;
  }
  if (last < text.length) out.push(new TextRun({ text: text.slice(last), ...base }));
  return out;
}

const body = (text) => new Paragraph({
  children: runs(text), alignment: AlignmentType.JUSTIFIED,
  indent: { firstLine: 288 }, spacing: { after: 100, line: 228 },
});
const h1 = (text) => new Paragraph({
  children: [new TextRun({ text, smallCaps: true })], alignment: AlignmentType.CENTER,
  keepNext: true, spacing: { before: 160, after: 80 },
});
const h2 = (text) => new Paragraph({
  children: [new TextRun({ text, italics: true })], keepNext: true, spacing: { before: 100, after: 60 },
});
const item = (text) => new Paragraph({
  children: runs(text), numbering: { reference: "contrib", level: 0 }, alignment: AlignmentType.JUSTIFIED,
  spacing: { after: 40, line: 228 },
});
const bullet = (text) => new Paragraph({
  children: runs(text), numbering: { reference: "bullets", level: 0 }, alignment: AlignmentType.JUSTIFIED,
  spacing: { after: 40, line: 228 },
});

// ---------- equations ----------
const M = (t) => new MathRun(t);
const sub = (b, s) => new MathSubScript({ children: [M(b)], subScript: [M(s)] });
function eq(num, children) {
  return new Paragraph({
    tabStops: [{ type: TabStopType.CENTER, position: COL / 2 }, { type: TabStopType.RIGHT, position: COL }],
    spacing: { before: 60, after: 60 },
    children: [new TextRun("\t"), new OMath({ children }), new TextRun(`\t(${num})`)],
  });
}

// ---------- tables ----------
const RULE = { style: BorderStyle.SINGLE, size: 4, color: "000000" };
const THIN = { style: BorderStyle.SINGLE, size: 2, color: "000000" };
const NONE = { style: BorderStyle.NONE, size: 0, color: "FFFFFF" };
function cell(text, width, { bold = false, align = AlignmentType.CENTER, size = 15 } = {}) {
  return new TableCell({
    width: { size: width, type: WidthType.DXA },
    margins: { top: 30, bottom: 30, left: 50, right: 50 },
    children: String(text).split("\n").map((l) => new Paragraph({
      alignment: align, spacing: { after: 0, line: 216 }, children: runs(l, { size, bold }),
    })),
  });
}
function table(widths, header, rows, size = 15) {
  const mk = (cells, bold) => new TableRow({
    cantSplit: true, tableHeader: bold,
    children: cells.map((c, i) => cell(c, widths[i], {
      bold, size, align: i === 0 ? AlignmentType.LEFT : AlignmentType.CENTER })),
  });
  return new Table({
    width: { size: widths.reduce((a, b) => a + b, 0), type: WidthType.DXA }, columnWidths: widths,
    borders: { top: RULE, bottom: RULE, insideHorizontal: THIN, left: NONE, right: NONE, insideVertical: NONE },
    rows: [mk(header, true), ...rows.map((r) => mk(r, false))],
  });
}
const tcap = (num, text) => new Paragraph({
  children: [new TextRun({ text: `TABLE ${num}.  ${text}`, smallCaps: true, size: 16 })],
  alignment: AlignmentType.CENTER, keepNext: true, spacing: { before: 200, after: 100 },
});
const tnote = (text) => new Paragraph({
  children: runs(text, { size: 14 }), alignment: AlignmentType.JUSTIFIED, spacing: { before: 40, after: 160 },
});

// ---------- figures ----------
function figure(file, pxW, pxH, num, caption) {
  const w = 318, h = Math.round((pxH / pxW) * w);
  return [
    new Paragraph({
      alignment: AlignmentType.CENTER, keepNext: true, spacing: { before: 120, after: 0 },
      children: [new ImageRun({ type: "png", data: fs.readFileSync(FIG(file)), transformation: { width: w, height: h } })],
    }),
    new Paragraph({
      children: runs(`Fig. ${num}.  ${caption}`, { size: 16 }), alignment: AlignmentType.JUSTIFIED,
      spacing: { before: 60, after: 160 },
    }),
  ];
}

// ---------- front matter ----------
const TITLE = "When Does Token-Level Uncertainty Detect Factual Hallucinations? An Audited Study Across Three Small Language Models and Two Datasets";
function author(name, reg, breakFirst = false) {
  const nameRuns = breakFirst ? [new ColumnBreak(), new TextRun({ text: name, bold: true, size: 19 })]
                              : [new TextRun({ text: name, bold: true, size: 19 })];
  return [
    new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 40 }, children: nameRuns }),
    new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 40 }, children: [new TextRun({ text: reg, size: 17 })] }),
    new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ text: "School of Computer Science and Engineering (SCOPE)", italics: true, size: 16 })] }),
    new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ text: "Vellore Institute of Technology (VIT), Chennai", italics: true, size: 16 })] }),
    new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 200 }, children: [new TextRun({ text: "Chennai, India", size: 16 })] }),
  ];
}
const abstractPara = (label, text, before = 0) => new Paragraph({
  alignment: AlignmentType.JUSTIFIED, indent: { firstLine: 272 }, spacing: { before, after: 160 },
  children: [new TextRun({ text: `${label}—`, bold: true, italics: true, size: 18 }),
    new TextRun({ text, bold: true, italics: true, size: 18 })],
});

const ABSTRACT = "Large language models produce fluent answers that are sometimes factually wrong, and detecting such hallucinations cheaply remains unsolved. Token-level uncertainty from a single generation is attractive because it requires no extra inference, unlike sampling-based self-consistency. We investigate when this signal works and when it fails, using Qwen2.5-1.5B-Instruct, Llama-3.2-1B-Instruct and SmolLM2-1.7B-Instruct on fixed 200-question subsets of TruthfulQA and SciQ, with a logistic-regression detector over mean and minimum token log-probability and mean predictive entropy, benchmarked against a five-sample self-consistency baseline. An audit of the pipeline revealed that Qwen2.5 applies a repetition penalty during greedy decoding, so log-probabilities returned by the generation API describe the penalty-adjusted decoding distribution rather than the model's conditional distribution; the resulting feature shift (0.125 in mean log-probability) exceeds the between-class separation, and correcting it removes a previously apparent improvement. After correction, and with permutation tests, repeated cross-validation and multiple-testing correction, the findings are: on SciQ the detector is reliable for all three models (AUROC 0.67-0.70 against a near-exact containment label); on TruthfulQA it is at chance for two of three models, with a replicable but modest signal only for Llama-3.2-1B (0.653, p<=0.002). Roughly half of all wrong TruthfulQA answers are produced with above-median confidence. We also find that measured detectability depends strongly on which automatic labeler is used, and quantify labeler reliability where an exact criterion exists. Token-level uncertainty is therefore useful but strongly model-, dataset- and labeler-dependent.";
const KEYWORDS = "Hallucination detection, large language models, uncertainty estimation, token log-probability, predictive entropy, self-consistency, TruthfulQA, reproducibility";

// ---------- body ----------
const content = [];
const add = (...xs) => xs.flat().forEach((x) => content.push(x));

add(h1("I. Introduction"));
add(body("Large language models generate fluent text that can be incorrect, unsupported or fabricated. On TruthfulQA, a benchmark assembled from common misconceptions, the best model evaluated by its authors was truthful on 58% of questions against 94% for humans [1]. Wherever generated answers feed question answering, search assistance or decision support, unreliable outputs should be flagged, withheld or routed to a stronger verifier."));
add(body("Many detectors sample the model repeatedly and measure agreement between samples, including SelfCheckGPT [2], self-consistency [3] and semantic entropy [4]. These can be effective, but each additional sample multiplies inference cost and latency. A cheaper alternative uses the uncertainty a model already exposes while producing a single answer: the probability assigned to each generated token and the entropy of each next-token distribution [5]. For a detector that runs on every request, avoiding extra generations is a decisive practical advantage."));
add(body("Confidence, however, is not truth. A model can assign high probability to a claim it has seen asserted repeatedly during training even when the claim is false, and TruthfulQA is deliberately built around such imitative falsehoods [1]. Whether single-pass uncertainty detects hallucination should therefore depend on the kind of error being made. This motivates our question: *when can token-level uncertainty from a single generation detect factual hallucinations, and when does it fail?*"));
add(body("The study began as a coursework baseline asking whether a one-call detector could approach a five-call self-consistency baseline at 80% fewer model calls. That baseline performed at chance on TruthfulQA. Rather than stopping there we extended the work to further models and a contrasting dataset, diagnosed the failure, engineered finer-grained features, and then audited the whole pipeline. The audit changed the results: a decoding-time artifact had inflated one model's apparent improvement. We report the corrected findings, including the negative ones. Our contributions are:"));
add(item("A reproducible single-pass hallucination-detection pipeline based on token log-probabilities and predictive entropy, with a controlled comparison against five-sample self-consistency under identical questions, prompts, splits and seeds."));
add(item("An evaluation across three small instruction-tuned models and two datasets with very different factual characteristics."));
add(item("A diagnosis of why sequence-level confidence fails on TruthfulQA, quantifying the share of errors that are produced confidently."));
add(item("A methodological finding: generation-API scores can silently reflect decoding-time logits processors rather than the model's conditional distribution, which materially distorts uncertainty features."));
add(item("A statistical validation protocol (permutation tests, repeated cross-validation, multiple-testing correction) and an adjudication of automatic labelers where an exact criterion exists, showing that measured detectability is strongly labeler-dependent."));

add(h1("II. Related Work"));
add(body("TruthfulQA [1] measures whether models reproduce human misconceptions and is our primary benchmark. FActScore [6] evaluates long-form factual precision by decomposing text into atomic facts and verifying each against a knowledge source, targeting evaluation rather than low-cost online detection."));
add(body("Sampling-based detectors assume a model that knows a fact answers consistently. SelfCheckGPT [2] compares a response against additional stochastic samples and self-consistency [3] aggregates multiple sampled answers. Semantic entropy [4] clusters samples by meaning rather than wording and computes entropy over clusters, reported to outperform naive entropy baselines. All require several generations per input."));
add(body("Single-sample and white-box approaches avoid this cost. Claim-conditioned probability [5] refines token-level uncertainty for fact-checking. MIND [7] detects hallucination from internal states, HaloScope [8] learns from unlabeled generations and reported 82.41% AUROC on TruthfulQA with OPT-13B, and simple factuality probes [10] were reported competitive with multi-sample methods from a single sample at up to 100x fewer FLOPs. InterrogateLLM [9] re-queries the model instead. A broad evaluation of 19 uncertainty-estimation methods [11] identified threshold sensitivity and adversarial prompts as deployment risks. We do not propose a new uncertainty signal; we measure where the simplest single-pass signals succeed and fail on small open-weight models, and we audit the measurement itself."));

add(h1("III. Methodology"));
add(figure("fig1_pipeline.png", 797, 508, 1, "Pipeline. The single-pass detector (top two rows) uses one greedy generation; the self-consistency baseline (bottom row) uses five sampled generations. Reference answers are used only to produce labels for training and evaluation, never as detector inputs."));
add(h2("A. Problem Formulation"));
add(body("Given a question *q*, a model generates *y* = (*y*_{1}, …, *y*_{T}). The detector estimates the probability that *y* is hallucinated (*h* = 1) using only signals produced during generation. Because neither dataset labels free-form generations and no trained judge was available, labels come from automatic proxies. The primary proxy computes token-level F1 overlap between *y* and each reference, labelling *y* grounded when its best overlap with a correct reference exceeds its best overlap with an incorrect one. Section III-G describes a second proxy and an adjudication of both."));
add(h2("B. Single-Pass Log-Probability Detector"));
add(body("Answers are generated greedily. For a decoder-only model the logits produced at step *t* are conditioned on the prompt *x* and the previously generated tokens *y*_{<t}, so they score exactly the token emitted at that step; equivalently, in a single forward pass over the concatenation of prompt and answer, the distribution for *y*_{t} is at index (|*x*| − 1 + *t*). Only generated positions are scored, and generation is truncated at the first end-of-turn token, which is excluded. The token log-probability is"));
add(eq(1, [sub("ℓ", "t"), M("= log "), sub("p", "θ"), M("(" ), sub("y", "t"), M(" ∣ x, "), sub("y", "<t"), M(")")]));
add(body("and the answer-level summaries are the mean and the minimum:"));
add(eq(2, [sub("μ", "ℓ"), M(" = "), new MathFraction({ numerator: [M("1")], denominator: [M("T")] }),
  new MathSum({ children: [sub("ℓ", "t")], subScript: [M("t=1")], superScript: [M("T")] }),
  M(",   "), sub("ℓ", "min"), M(" = "), sub("min", "t"), M(" "), sub("ℓ", "t")]));
add(body("The log-probability describes only the chosen token. Predictive entropy instead describes the entire next-token distribution over the vocabulary *V*:"));
add(eq(3, [sub("H", "t"), M(" = −"), new MathSum({ children: [sub("p", "t"), M("(v) log "), sub("p", "t"), M("(v)")], subScript: [M("v∈V")] })]));
add(body("A token may be emitted with high probability while the remaining mass is spread across many alternatives, or the reverse, so the two quantities are complementary. The answer-level entropy *μ*_{H} is the mean of *H*_{t}. The baseline detector uses **z** = (*μ*_{ℓ}, *ℓ*_{min}, *μ*_{H}) and logistic regression"));
add(eq(4, [M("P(h = 1 ∣ z) = σ("), new MathSuperScript({ children: [M("w")], superScript: [M("⊤")] }), M("z + b)")]));
add(body("whose output is the hallucination probability, mapped to PASS, WARN and FLAG decisions at 0.4 and 0.7 in the accompanying demonstration interface."));
add(h2("C. Decoding-Time Contamination of the Scores"));
add(body("Equation (1) holds only if the scores returned by the generation interface are the model's conditional distribution. They need not be. HuggingFace applies logits processors during generation even under greedy decoding, and Qwen2.5-1.5B-Instruct ships a repetition penalty of 1.1 in its generation configuration. Its returned scores therefore describe a penalty-adjusted decoding distribution. We detect this by recomputing every quantity with an independent teacher-forcing pass and comparing token by token; a deliberate off-by-one control differs by 15–20 nats, so the check distinguishes a genuine misalignment from a distributional difference. Llama-3.2 and SmolLM2 configure no penalty and agree with the recomputation to within floating-point noise. For Qwen we recompute all features from unmodified logits over the frozen answers, which leaves the generated text and labels untouched."));
add(h2("D. Token-Level Uncertainty Features"));
add(body("To test whether three summary statistics discard useful information, 17 features are computed from the same per-token arrays, in seven families: central tendency (mean, median), percentile statistics (minimum, 10th and 25th percentile), dispersion (standard deviation), low-confidence proportion (fraction of tokens with log-probability below −3), entropy statistics (mean, maximum, standard deviation of *H*_{t}), positional statistics (mean log-probability of each half of the answer and their difference), and confidence trend (least-squares slope of *ℓ*_{t} against position). Answer length and the top-1/top-2 log-probability margin complete the set. All features are functions of the model's own generation and never of the references."));
add(h2("E. Self-Consistency Baseline"));
add(body("The baseline draws five answers at temperature 0.7 and top-*p* 0.9. Lexical agreement is the mean pairwise token-F1 similarity among them. A semantic variant approximates semantic entropy [4]: samples are embedded with a Sentence-BERT encoder [13], greedily clustered at cosine similarity 0.75, and scored by the entropy of the cluster proportions,"));
add(eq(5, [M("SE = −"), new MathSum({ children: [new MathFraction({ numerator: [sub("n", "k")], denominator: [M("K")] }), M(" log "), new MathFraction({ numerator: [sub("n", "k")], denominator: [M("K")] })], subScript: [M("k")] })]));
add(body("where *n*_{k} is the size of cluster *k* and *K* = 5. This is a cosine-clustering approximation; the original method clusters by bidirectional entailment. Each score feeds a one-feature logistic regression."));
add(h2("F. Feature-Engineered Detector and Two-Stage Cascade"));
add(body("The engineered detector applies logistic regression to all 17 features. A single linear boundary assumes that higher confidence always means lower hallucination risk; Section V-D shows this relation reverses among confidently generated answers. A cascade therefore routes each answer into a confident branch (*b* = c) or an uncertain one (*b* = u) by its mean log-probability, and applies a branch-specific classifier,"));
add(eq(6, [M("P(h = 1 ∣ z) = σ("), new MathSuperScript({ children: [sub("w", "b")], superScript: [M("⊤")] }), M("z + "), sub("b", "b"), M(")")]));
add(body("where *b* = c when *μ*_{ℓ} ≥ τ and *b* = u otherwise. The threshold τ is the median mean log-probability of the training fold, and each branch classifier is fit only on that branch's training examples, so no threshold or model sees evaluation labels."));
add(h2("G. Labels and Their Adjudication"));
add(body("Every label is a proxy, so we quantify their reliability rather than assume it. Alongside the token-F1 proxy we compute an embedding proxy that replaces lexical overlap with cosine similarity to the same references. On SciQ, where references are short noun phrases, a near-exact criterion is additionally available: an answer is grounded if it contains the correct answer and no distractor. We use this containment label to measure which proxy is trustworthy on that dataset. No equivalent criterion exists for TruthfulQA, whose references are full sentences with many valid paraphrases, so there we report results under both proxies and treat disagreement as genuine uncertainty."));
add(h2("H. Statistical Evaluation"));
add(body("The primary metric is AUROC, which is threshold-independent. Uncertainty is quantified with 95% percentile bootstrap intervals over 2,000 resamples. Two evaluation protocols are used: the original fixed stratified 140/60 split, and *k*-fold out-of-fold (OOF) evaluation over all 200 questions, in which every question is scored by a model that never trained on it. Because a single cross-validation partition can be fortunate, we repeat 5-fold OOF evaluation over 20 different partition seeds and report the spread. We additionally run permutation tests: labels are shuffled and the entire OOF procedure re-run 500 times, giving an empirical null that absorbs optimism in the fitting procedure itself (minimum resolvable *p* ≈ 0.002). Eight detector variants are evaluated per model, dataset and labeler, so we treat α = 0.05/8 = 0.00625 as the corrected significance threshold and label results between 0.00625 and 0.05 exploratory."));

add(h1("IV. Experimental Setup"));
add(h2("A. Models"));
add(body("Three instruction-tuned models were used through their Hugging Face checkpoints: Qwen/Qwen2.5-1.5B-Instruct, Llama-3.2-1B-Instruct (via an ungated mirror of the official weights) and HuggingFaceTB/SmolLM2-1.7B-Instruct. Each received the same chat-template prompt with a system instruction asking for a direct answer of one or two sentences. Inference ran in 16-bit floating point on a single NVIDIA GeForce RTX 4060 GPU with 8 GB of memory."));
add(h2("B. Datasets"));
add(body("TruthfulQA [1] (generation configuration, 817 questions over 38 categories) supplies correct and incorrect reference answers for questions built around misconceptions. SciQ [12] contains crowdsourced science examination questions with one correct answer and three distractors, which serve as incorrect references. A fixed 200-question subset of each was drawn once with a fixed seed and shared by every model."));
add(h2("C. Data Splitting and Configuration"));
add(body("The original experiment used a stratified 140/60 split with seed 42, reused identically for every model and method; later analyses add stratified 5-fold OOF evaluation. Features are computed per answer, and all thresholds, scalings and classifiers are fit on training folds only. Table I summarises the configuration."));
add(tcap("I", "Experimental Configuration"));
add(table([1650, 3216], ["Parameter", "Value"], [
  ["Models", "Qwen2.5-1.5B-Instruct, Llama-3.2-1B-Instruct, SmolLM2-1.7B-Instruct"],
  ["Datasets", "TruthfulQA (generation), SciQ"],
  ["Questions", "200 per dataset (fixed subset, seed 42)"],
  ["Single-pass decoding", "Greedy, ≤64 new tokens"],
  ["Self-consistency", "5 samples, T = 0.7, top-p = 0.9"],
  ["Classifier", "Logistic regression (scikit-learn defaults)"],
  ["Protocols", "Fixed 140/60 split; stratified 5-fold OOF"],
  ["Repeated CV", "20 partition seeds"],
  ["Permutation test", "500 label shuffles"],
  ["Bootstrap", "2,000 resamples, 95% percentile CI"],
  ["Hardware", "NVIDIA RTX 4060 (8 GB), FP16"],
]));

add(h1("V. Results and Analysis"));
add(h2("A. Original Baseline"));
add(body("Table II reproduces the original experiment: Qwen2.5-1.5B-Instruct on TruthfulQA, evaluated on the 60-question holdout of which 35 answers carry the hallucinated label. The three-feature detector reaches an AUROC of 0.482; its accuracy of 0.583 equals the majority-class rate because it predicts every answer hallucinated. No single feature separates the classes and the five-call self-consistency baseline is no better. The detector used 200 LLM calls against 1,000, an 80% reduction, but equivalence between two chance-level detectors is not a useful result. These numbers predate the correction of Section V-B and are shown as the historical starting point."));
add(tcap("II", "Original Baseline (Qwen, TruthfulQA, n = 60)"));
add(table([1866, 600, 800, 800, 800], ["Method", "Calls", "AUROC", "Acc.", "F1"], [
  ["3-feature LR", "1", "0.482", "0.583", "0.737"],
  ["Mean log-prob only", "1", "0.551", "0.583", "0.737"],
  ["Min log-prob only", "1", "0.464", "0.583", "0.737"],
  ["Mean entropy only", "1", "0.504", "0.583", "0.737"],
  ["Self-consistency", "5", "0.461", "0.567", "0.723"],
]));
add(h2("B. A Decoding-Time Artifact in the Features"));
add(body("Recomputing the per-token quantities from unmodified logits (Section III-C) shows that Qwen's cached features were materially distorted, while the other two models were unaffected (Table III). The mean absolute shift in mean log-probability for Qwen, 0.125, is roughly eight times the separation between grounded and hallucinated answers in that feature (0.016), so the contamination is larger than the signal it was meant to carry. The consequence is not cosmetic: the apparently significant Qwen result on TruthfulQA (cascade over 17 features, AUROC 0.598, permutation *p* = 0.026) becomes 0.554 with *p* = 0.174 once the features are corrected. We therefore use corrected features for Qwen throughout, and original features for Llama and SmolLM2, which are already uncontaminated."));
add(tcap("III", "Effect of Recomputing Features From Raw Logits"));
add(table([1300, 900, 900, 900], ["Model", "mean |Δ| μℓ", "max |Δ| μℓ", "mean |Δ| μH"], [
  ["Qwen2.5-1.5B", "0.125", "0.595", "0.202"],
  ["Llama-3.2-1B", "0.006", "0.026", "0.014"],
  ["SmolLM2-1.7B", "0.0006", "0.002", "0.0008"],
]));
add(tnote("Qwen2.5 configures repetition_penalty = 1.1, applied even under greedy decoding; the other two configure none."));
add(body("A second reproducibility issue appeared during this recomputation. Llama's greedy generations did not reproduce exactly: 83 of 200 TruthfulQA answers and 29 of 200 on SciQ differed on a second run under identical settings. Non-reproduced answers are systematically longer (54.1 against 43.5 tokens) and far more often reach the 64-token cap (0.52 against 0.28), consistent with floating-point non-determinism compounding over long generations. Qwen and SmolLM2, whose answers are much shorter, reproduced exactly. We therefore do not use Llama's recomputed subset, which is length-biased, and we report this as a limitation on cross-model comparability."));
add(h2("C. Labeler Reliability"));
add(body("The two automatic proxies agree on only 65.5–78.5% of answers (Cohen's κ 0.34–0.56), and on SciQ they differ about the hallucination rate by roughly a factor of two. On SciQ the question is settleable: against the containment criterion of Section III-G, the token-F1 proxy agrees on 91–94% of answers (κ 0.82–0.88) whereas the embedding proxy agrees on only 61–69% (κ 0.29–0.38), marking 24–28% of answers hallucinated against a true rate of 52–64%. The embedding proxy therefore misses roughly half of the errors on short-answer data, which is unsurprising because cosine similarity between a full sentence and a one-word reference is high almost regardless of correctness. Table IV reports the characteristics used throughout."));
add(tcap("IV", "Dataset and Labeler Characteristics"));
add(table([1150, 620, 500, 640, 660, 650, 646], ["Model", "Data", "n", "Rate (F1)", "Rate (emb.)", "Agree.", "κ"], [
  ["Qwen2.5-1.5B", "TQA", "200", "0.59", "0.55", "0.730", "0.450"],
  ["Llama-3.2-1B", "TQA", "200", "0.53", "0.52", "0.730", "0.459"],
  ["SmolLM2-1.7B", "TQA", "200", "0.62", "0.54", "0.785", "0.563"],
  ["Qwen2.5-1.5B", "SciQ", "200", "0.46", "0.26", "0.735", "0.448"],
  ["Llama-3.2-1B", "SciQ", "200", "0.55", "0.28", "0.655", "0.338"],
  ["SmolLM2-1.7B", "SciQ", "200", "0.45", "0.24", "0.730", "0.430"],
]));
add(tnote("Rate = proportion labelled hallucinated. On SciQ the containment criterion gives 0.52, 0.64 and 0.52 respectively, closely matching the token-F1 proxy."));
add(h2("D. Confidently Wrong Answers"));
add(body("Each answer was assigned to one of four groups by correctness and by whether its mean log-probability fell above or below the model's median. Fig. 3 reports the share of wrong answers that were confident: 50.8%, 51.4% and 52.4% on TruthfulQA against 32.6%, 40.0% and 40.0% on SciQ. On TruthfulQA a wrong answer is therefore about as likely to be confident as uncertain. A detector that thresholds mean confidence at the median necessarily places these answers alongside confident correct ones, which bounds that family of detectors, though not every possible detector."));
add(figure("fig3_confidently_wrong.png", 1014, 480, 3, "Share of wrong answers generated with above-median mean log-probability, by model and dataset."));
add(body("Within the confident group the association also reverses. For Qwen, confidently wrong answers have a *higher* second-half mean log-probability than confidently correct ones (−0.265 against −0.371, Cohen's *d* = +0.73) and a rising rather than falling within-answer trend, and they are shorter (*d* = −0.71). Llama shows the same direction more weakly (*d* = 0.39–0.41) and SmolLM2 shows none (|*d*| ≤ 0.10). Relative to the global assumption that higher confidence implies greater reliability, the relation inside the confident bucket is inverted, which motivates the routed cascade of Section III-F."));
add(h2("E. TruthfulQA After Correction"));
add(body("Table V gives 5-fold OOF results under both proxies with permutation *p*-values. Only Llama-3.2-1B shows a replicable signal: 0.653 under token-F1 labels and 0.663 under embedding labels, both with *p* ≤ 0.002, stable across 20 partition seeds (0.649 ± 0.011), surviving the corrected threshold, and still significant without the answer-length feature under the embedding labeler (0.655). Qwen has no validated signal after decontamination; its only variant below 0.05 uses a four-feature subset chosen by inspecting these same questions and does not survive correction. SmolLM2 is a null throughout. Fig. 4 shows the same comparison graphically."));
add(tcap("V", "TruthfulQA, 5-Fold OOF AUROC (permutation p)"));
add(table([1266, 900, 1350, 1350], ["Model", "Detector", "token-F1 labels", "embedding labels"], [
  ["Qwen", "3-feat", "0.434 (0.844)", "0.540 (0.186)"],
  ["Qwen", "17-feat", "0.553 (0.170)", "0.451 (0.798)"],
  ["Qwen", "casc. 17", "0.554 (0.174)", "0.496 (0.535)"],
  ["Qwen", "16-feat†", "0.505 (0.481)", "0.501 (0.475)"],
  ["Llama", "3-feat", "0.566 (0.090)", "**0.615 (0.002)**"],
  ["Llama", "17-feat", "**0.653 (0.002)**", "**0.663 (0.002)**"],
  ["Llama", "casc. 17", "**0.630 (0.002)**", "**0.661 (0.002)**"],
  ["Llama", "16-feat†", "0.578 (0.072)", "**0.655 (0.002)**"],
  ["SmolLM2", "3-feat", "0.418 (0.904)", "0.401 (0.948)"],
  ["SmolLM2", "17-feat", "0.485 (0.585)", "0.586 (0.042)"],
  ["SmolLM2", "16-feat†", "0.494 (0.507)", "0.581 (0.044)"],
]));
add(tnote("Bold = survives the corrected threshold α = 0.00625. † 17 features without answer length. Minimum resolvable p ≈ 0.002."));
add(figure("fig4_final_validation.png", 1014, 630, 4, "TruthfulQA AUROC under 5-fold out-of-fold evaluation with 95% bootstrap intervals, for the original and 17-feature detectors under both labelers. Stars mark results surviving multiple-testing correction."));
add(h2("F. SciQ"));
add(body("SciQ behaves very differently (Table VI). Under the token-F1 proxy every model, feature set and architecture is significant after correction and stable across seeds, and removing answer length changes nothing. Under the containment label, which depends on neither proxy, the detector holds for all three models at 0.673–0.704 with intervals comfortably excluding 0.5. Results under the embedding proxy are much weaker, but Section V-C shows this reflects that proxy's unreliability on short answers rather than a fragile detector. Notably the original three-feature detector is the best or near-best variant on SciQ: the engineered features add nothing where the simple signal already works. Fig. 2 shows out-of-fold ROC curves for both datasets."));
add(tcap("VI", "SciQ, 5-Fold OOF AUROC"));
add(table([1266, 1200, 1200, 1200], ["Model", "3-feat (F1)", "3-feat (cont.)", "16-feat (cont.)"], [
  ["Qwen2.5-1.5B", "**0.716**", "**0.682**", "**0.684**"],
  ["Llama-3.2-1B", "**0.634**", "**0.676**", "**0.704**"],
  ["SmolLM2-1.7B", "**0.649**", "**0.682**", "**0.676**"],
]));
add(tnote("cont. = near-exact containment label. Every entry has permutation p ≤ 0.004 and a 95% interval excluding 0.5; under the containment label all nine variants reach the minimum resolvable p ≈ 0.002 with repeated-CV spread ≤ 0.011."));
add(figure("fig2_roc_truthfulqa_vs_sciq.png", 1014, 525, 2, "Out-of-fold ROC curves of the original three-feature detector on TruthfulQA (left) and SciQ (right), with AUROC in parentheses."));
add(h2("G. Statistical Validation and Protocol Dependence"));
add(body("Results depend on the evaluation protocol, and we report both. On the original fixed 60-question holdout no detector is significant for any model: the best Llama variant reaches 0.598 with an interval of [0.446, 0.738] and the best Qwen variant 0.583 [0.433, 0.735]. The out-of-fold protocol uses all 200 questions and is correspondingly more powerful; for the Qwen cascade the interval narrows from width 0.306 to 0.162, a ratio of 1.89 against the 1.83 expected from sample size alone. The out-of-fold estimates are not an independent second test set, and we do not treat them as evidence of generalization beyond these questions. Repeated cross-validation further shows that single-partition estimates are unstable: for Qwen's three-feature detector the reference partition gives 0.434 while the mean over 20 partitions is 0.515."));
add(body("Several earlier conclusions did not survive this scrutiny and were corrected rather than retained: an apparent gain from restricting evaluation to confidently labelled answers reversed under out-of-fold evaluation, a confident-subgroup analysis that computed its routing threshold over all questions was refit per fold, and the Qwen improvement described above was traced to the decoding artifact. We also note that features are supplied unstandardised to an L2-regularised classifier, which favours large-scale features such as answer length, and that answer length is itself correlated with the token-F1 proxy on two of three models."));
add(h2("H. Comparison With Multi-Call Approaches"));
add(body("Neither lexical self-consistency nor the semantic-entropy variant reached corrected significance on TruthfulQA for any model, at five calls per question against one. On SciQ both were significant but matched or trailed the single-call detector. A planned extension to ten samples was attempted twice and interrupted by the host environment at 62% and 33% completion, so whether larger sampling budgets help is unresolved and we make no claim about it."));

add(h1("VI. Discussion"));
add(body("Token-level uncertainty detects hallucination when the model's errors arise from not knowing. On SciQ, where questions ask for ordinary recalled facts, a single generation supports reliable detection for all three models under the most trustworthy label available, and five-call self-consistency buys nothing over it. On TruthfulQA the same signal is largely absent, and the diagnosis is concrete: about half of all wrong answers are produced with above-median confidence. These are the imitative falsehoods the benchmark was designed to elicit [1]. We avoid saying the model “knows” or “does not know” anything; the measurable statement is that the association between the confidence signal and correctness is strong in one setting and near zero in the other."));
add(body("Finer-grained features recover a modest signal for one model of three. The pattern they exploit is visible in the confident subgroup, where wrong answers sustain or increase confidence toward the end of a short answer while correct ones decay. That this holds for Llama, weakly for Qwen and not at all for SmolLM2 indicates the effect is model-specific rather than a general property of instruction-tuned models at this scale. The SmolLM2 null is informative and we retain it: it bounds how general any such claim can be."));
add(body("Two methodological findings matter beyond this study. First, generation-interface scores may not be the model's conditional distribution; a repetition penalty configured by default silently shifted one model's features by more than the class separation, and any study reading log-probabilities from a generation API should verify this. Second, measured detectability depends substantially on the automatic labeler, and where an exact criterion exists the two common proxies differ sharply in reliability. Reporting a single proxy without adjudication can therefore overstate or understate detectability."));
add(body("For deployment, the practical implication is that confidence-only screening is appropriate for knowledge-gap errors and inadequate for confidently stated falsehoods. Addressing the latter requires evidence external to the generating model, such as retrieval against a trusted source or an independent verifier, which is a different and larger system than the one studied here."));

add(h1("VII. Limitations"));
add(bullet("Each dataset is a single 200-question subset and the 60-question holdout is badly underpowered; intervals are wide throughout."));
add(bullet("All TruthfulQA labels are automatic proxies that agree on only 65.5–78.5% of answers, and no exact criterion exists there; several conclusions depend on which proxy is used."));
add(bullet("Three models of 1–2 billion parameters and two short-answer datasets do not support claims about larger models, long-form generation or other hallucination types."));
add(bullet("Llama's generations are not bit-reproducible in this environment and roughly 38% of its TruthfulQA answers reach the token cap, weakening cross-model comparability."));
add(bullet("The four-feature subset was selected by inspecting the same questions, and the embedding-label analyses were added after the main results; both are reported as exploratory."));
add(bullet("Features are unstandardised in the primary pipeline, and the semantic-entropy baseline uses cosine clustering rather than entailment."));
add(bullet("The ten-sample self-consistency sweep was not completed, and the detector uses no retrieval or external verifier."));

add(h1("VIII. Conclusion and Future Work"));
add(body("We investigated single-pass token-level uncertainty as a low-cost hallucination detector against five-sample self-consistency, across three small instruction-tuned models and two datasets, and audited the pipeline before drawing conclusions. The signal is reliable for ordinary factual recall, where one call matches or exceeds five, and largely absent for misconception-oriented questions, where about half of all errors are made confidently. Finer-grained features recover a modest, replicable signal for one model of three. Along the way we found that a default repetition penalty had contaminated one model's uncertainty features by more than the class separation, and that the choice of automatic labeler can change the measured conclusion. Future work should confirm these effects on a larger, independently held-out question set, obtain human-adjudicated labels for TruthfulQA, complete the sampling-budget sweep, and combine token-level uncertainty with retrieval or a verifier model to address confidently wrong claims."));

add(h1("References"));
const REFS = [
  "S. Lin, J. Hilton, and O. Evans, “TruthfulQA: Measuring how models mimic human falsehoods,” in *Proc. ACL*, 2022, pp. 3214–3252, doi: 10.18653/v1/2022.acl-long.229.",
  "P. Manakul, A. Liusie, and M. J. F. Gales, “SelfCheckGPT: Zero-resource black-box hallucination detection for generative large language models,” in *Proc. EMNLP*, 2023, pp. 9004–9017, doi: 10.18653/v1/2023.emnlp-main.557.",
  "X. Wang *et al.*, “Self-consistency improves chain of thought reasoning in language models,” in *Proc. ICLR*, 2023.",
  "S. Farquhar, J. Kossen, L. Kuhn, and Y. Gal, “Detecting hallucinations in large language models using semantic entropy,” *Nature*, vol. 630, pp. 625–630, 2024, doi: 10.1038/s41586-024-07421-0.",
  "E. Fadeeva *et al.*, “Fact-checking the output of large language models via token-level uncertainty quantification,” in *Findings ACL*, 2024, pp. 9367–9385, doi: 10.18653/v1/2024.findings-acl.558.",
  "S. Min *et al.*, “FActScore: Fine-grained atomic evaluation of factual precision in long form text generation,” in *Proc. EMNLP*, 2023, pp. 12076–12100, doi: 10.18653/v1/2023.emnlp-main.741.",
  "W. Su *et al.*, “Unsupervised real-time hallucination detection based on the internal states of large language models,” in *Findings ACL*, 2024, pp. 14379–14391, doi: 10.18653/v1/2024.findings-acl.854.",
  "X. Du, C. Xiao, and Y. Li, “HaloScope: Harnessing unlabeled LLM generations for hallucination detection,” in *Adv. Neural Inf. Process. Syst.*, vol. 37, 2024, doi: 10.52202/079017-3270.",
  "Y. Yehuda *et al.*, “InterrogateLLM: Zero-resource hallucination detection in LLM-generated answers,” in *Proc. ACL*, 2024, pp. 9333–9347, doi: 10.18653/v1/2024.acl-long.506.",
  "J. Han *et al.*, “Simple factuality probes detect hallucinations in long-form natural language generation,” in *Findings EMNLP*, 2025.",
  "Y. F. Bakman *et al.*, “Reconsidering LLM uncertainty estimation methods in the wild,” in *Proc. ACL*, 2025, pp. 29531–29556, doi: 10.18653/v1/2025.acl-long.1429.",
  "J. Welbl, N. F. Liu, and M. Gardner, “Crowdsourcing multiple choice science questions,” in *Proc. 3rd Workshop Noisy User-Generated Text (W-NUT)*, 2017.",
  "N. Reimers and I. Gurevych, “Sentence-BERT: Sentence embeddings using Siamese BERT-networks,” in *Proc. EMNLP-IJCNLP*, 2019.",
];
REFS.forEach((r, i) => add(new Paragraph({
  alignment: AlignmentType.JUSTIFIED, indent: { left: 400, hanging: 400 }, spacing: { after: 40 },
  tabStops: [{ type: TabStopType.LEFT, position: 400 }],
  children: [new TextRun({ text: `[${i + 1}]\t`, size: 16 }), ...runs(r, { size: 16 })],
})));

// ---------- document ----------
const page = { size: { width: 11906, height: 16838 }, margin: { top: 720, right: 907, bottom: 1080, left: 907 } };
const doc = new Document({
  styles: { default: { document: { run: { font: "Times New Roman", size: 20 } } } },
  numbering: { config: [
    { reference: "contrib", levels: [{ level: 0, format: LevelFormat.DECIMAL, text: "%1)", alignment: AlignmentType.LEFT,
      style: { paragraph: { indent: { left: 360, hanging: 260 } } } }] },
    { reference: "bullets", levels: [{ level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT,
      style: { paragraph: { indent: { left: 300, hanging: 180 } } } }] },
  ] },
  sections: [
    { properties: { page }, children: [
      new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 300 },
        children: [new TextRun({ text: TITLE.toUpperCase(), bold: true, italics: true, size: 36 })] }),
    ] },
    { properties: { page, type: SectionType.CONTINUOUS, column: { count: 2, space: 360 } }, children: [
      ...author("Goutham Krishna Jayakrishnan", "24BCE1910"),
      ...author("Shubham Khadilkar", "24BCE5275", true),
    ] },
    { properties: { page, type: SectionType.CONTINUOUS, column: { count: 2, space: 360 } }, children: [
      abstractPara("Abstract", ABSTRACT),
      abstractPara("Index Terms", KEYWORDS),
      ...content,
    ] },
  ],
});

Packer.toBuffer(doc).then((buf) => {
  const out = path.join(__dirname, "Token_Uncertainty_Hallucination_Detection.docx");
  fs.writeFileSync(out, buf);
  console.log("Wrote", out);
});
