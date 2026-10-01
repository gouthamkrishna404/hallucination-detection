// Builds paper/Token_Uncertainty_Hallucination_Detection.docx (IEEE-style, two-column, A4).
// Layout mirrors the ParkPulse reference: A4, Times New Roman, 0.63" side margins,
// 0.25" column gap, small-caps section heads, 8 pt captions, three-rule tables.
// Every number below is taken from saved outputs (see RESULTS.md / results/ / comparison/).
// Usage: cd paper && npm install && node build_paper.js
const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, ImageRun, Table, TableRow, TableCell, AlignmentType,
  BorderStyle, WidthType, SectionType, ColumnBreak, TabStopType, LevelFormat,
  Math: OMath, MathRun, MathSubScript, MathSuperScript, MathSubSuperScript, MathSum, MathFraction,
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
function cell(text, width, { bold = false, align = AlignmentType.CENTER } = {}) {
  const lines = String(text).split("\n");
  return new TableCell({
    width: { size: width, type: WidthType.DXA },
    margins: { top: 30, bottom: 30, left: 50, right: 50 },
    children: lines.map((l) => new Paragraph({
      alignment: align, spacing: { after: 0, line: 216 },
      children: runs(l, { size: 15, bold }),
    })),
  });
}
function table(widths, header, rows) {
  const total = widths.reduce((a, b) => a + b, 0);
  const mk = (cells, bold) => new TableRow({
    cantSplit: true, tableHeader: bold,
    children: cells.map((c, i) => cell(c, widths[i], { bold, align: i === 0 ? AlignmentType.LEFT : AlignmentType.CENTER })),
  });
  return new Table({
    width: { size: total, type: WidthType.DXA }, columnWidths: widths,
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
      indent: { left: 0 }, spacing: { before: 60, after: 160 },
    }),
  ];
}

// ---------- front matter ----------
const TITLE = "When Confidence Is Not Truth: Investigating Single-Pass Token-Level Uncertainty for Factual Hallucination Detection";
function author(name, reg) {
  return [
    new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 40 }, children: [new TextRun({ text: name, bold: true, size: 19 })] }),
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

const ABSTRACT = "Large language models (LLMs) produce fluent answers that can be factually wrong, and detecting such hallucinations cheaply remains an open problem. Token-level uncertainty from a single generation is an attractive signal because it requires no additional inference, whereas sampling-based self-consistency multiplies the inference cost. This paper investigates when that signal works and when it fails. A single-pass detector built on the mean and minimum token log-probability and the mean predictive entropy, classified by logistic regression, is compared against a five-sample self-consistency baseline on fixed 200-question subsets of TruthfulQA and SciQ, using Qwen2.5-1.5B-Instruct, Llama-3.2-1B-Instruct, and SmolLM2-1.7B-Instruct. On TruthfulQA the original detector performs at chance for all three models (AUROC 0.38–0.48), and self-consistency does no better (0.46–0.56); on SciQ the same detector reaches 0.62–0.76. A diagnostic analysis shows that 49–52% of wrong TruthfulQA answers are generated with above-median confidence, compared with 33–40% on SciQ. A 17-feature extension with percentile, dispersion, and positional statistics, combined with a confidence-routed two-stage classifier, raises five-fold out-of-fold AUROC to 0.598 (Qwen) and 0.653 (Llama). These gains are not individually significant on the original 60-question holdout, are partly attributable to a length bias in the proxy labeler, and do not appear for SmolLM2. Token-level uncertainty is therefore a useful but dataset- and model-dependent signal that cannot by itself resolve confidently wrong factual claims.";
const KEYWORDS = "Hallucination detection, large language models, uncertainty estimation, token log-probability, predictive entropy, self-consistency, TruthfulQA";

// ---------- body content ----------
const content = [];
const add = (...xs) => xs.flat().forEach((x) => content.push(x));

add(h1("I. Introduction"));
add(body("Large language models generate fluent text that can nevertheless be incorrect, unsupported, or fabricated. On TruthfulQA, a benchmark built from common misconceptions, the best model evaluated by its authors was truthful on 58% of questions compared with 94% for humans [1]. When generated answers feed question answering, search assistance, or decision support, unreliable outputs should be flagged, withheld, or routed to a stronger verifier before they reach a user."));
add(body("Several detection strategies sample the model repeatedly and measure agreement between the samples, including SelfCheckGPT [2], self-consistency [3], and semantic entropy [4]. These methods can be effective, but every additional sample adds inference cost and latency. A cheaper alternative reads the uncertainty the model already exposes while generating a single answer: the probability assigned to each generated token and the entropy of each next-token distribution [5]. For a detector that runs on every request, avoiding extra generations is a substantial practical advantage."));
add(body("Confidence, however, is not truth. A model can assign high probability to a claim it has frequently seen asserted in its training data, even when that claim is false, and TruthfulQA is deliberately constructed around such imitative falsehoods [1]. Whether single-pass uncertainty detects hallucinations therefore depends on the kind of error the model makes, which motivates the central question of this work: *when can token-level uncertainty from a single LLM generation detect factual hallucinations, and when does it fail?*"));
add(body("The study began as a course baseline asking whether a one-call detector could approach a five-call self-consistency baseline while using 80% fewer model calls. That baseline performed at chance on TruthfulQA. Rather than stopping there, the investigation was extended to additional models and a contrasting dataset, a diagnosis of the failure, richer features, and a statistically audited attempt at improvement. The contributions are:"));
add(item("A reproducible single-pass hallucination-detection pipeline based on token-level log-probabilities and predictive entropy."));
add(item("A controlled comparison against five-sample self-consistency and an embedding-based semantic-entropy variant under identical questions, prompts, splits, and seeds."));
add(item("An evaluation across three small instruction-tuned models and two datasets with very different factual characteristics."));
add(item("An analysis of confidently wrong answers and of finer-grained token-level features, including a confidence-routed two-stage classifier."));
add(item("An explicit account of limitations, including labeler bias, evaluation-protocol dependence, and a model for which no approach succeeded."));

add(h1("II. Related Work"));
add(body("TruthfulQA [1] measures whether models reproduce human misconceptions and is the primary benchmark here. FActScore [6] evaluates long-form factual precision by decomposing text into atomic facts and verifying each against a knowledge source; it targets evaluation rather than low-cost online detection."));
add(body("Sampling-based detectors assume that a model that knows a fact answers consistently. SelfCheckGPT [2] compares a response against additional stochastic samples, and self-consistency [3] aggregates multiple sampled answers. Semantic entropy [4] clusters samples by meaning rather than wording and computes entropy over the clusters, and it was reported to outperform naive entropy baselines. All three require several generations per input."));
add(body("Single-sample and white-box approaches avoid this cost. Claim-conditioned probability [5] refines token-level uncertainty for fact-checking model outputs. MIND [7] detects hallucinations in real time from internal states, HaloScope [8] learns from unlabeled generations and reported 82.41% AUROC on TruthfulQA with OPT-13B, and simple factuality probes [10] were reported to be competitive with multi-sample methods from a single sample at up to 100× fewer FLOPs. InterrogateLLM [9] instead re-queries the model. A large evaluation of 19 uncertainty-estimation methods [11] identified threshold sensitivity and adversarial prompts as deployment risks. The present work does not propose a new uncertainty signal; it measures, under a fixed model-call budget, where the simplest single-pass signals succeed and fail on small open-weight models, and it reports negative findings alongside positive ones."));

add(h1("III. Methodology"));
add(figure("fig1_pipeline.png", 797, 508, 1, "Pipeline. The single-pass detector (top two rows) uses one greedy generation; the self-consistency baseline (bottom row) uses five sampled generations. Reference answers are used only to produce proxy labels for training and evaluation, never as detector inputs."));
add(h2("A. Problem Formulation"));
add(body("Given a question *q*, a model generates an answer *y* = (*y*_{1}, …, *y*_{T}). The detector estimates the probability that *y* is hallucinated (*h* = 1) using only signals produced during generation. Because neither dataset labels free-form generations and no trained judge model was available, labels come from a transparent proxy: token-level F1 overlap between *y* and each reference answer. An answer is labeled grounded (*h* = 0) if its best overlap with a correct reference strictly exceeds its best overlap with an incorrect reference, and hallucinated otherwise. The labels are a measurable approximation of truthfulness, and their biases are examined in Section V."));
add(h2("B. Single-Pass Logprob Detector"));
add(body("The answer is generated greedily. For a decoder-only model, the logits *z*_{t} produced at generation step *t* are conditioned on the prompt *x* and the previously generated tokens *y*_{<t}, so they score exactly the token *y*_{t} emitted at that step. Only generated positions are scored; prompt tokens are excluded, and generation is truncated at the first end-of-turn token, which is itself excluded. The token log-probability is"));
add(eq(1, [sub("ℓ", "t"), M("=log softmax"), new MathSubScript({ children: [M("(z")], subScript: [M("t")] }), M(")"), new MathSubScript({ children: [M("")], subScript: [M("")] }), new MathSubScript({ children: [M("[y")], subScript: [M("t")] }), M("]")]));
add(body("and the answer-level summaries are the mean and the minimum:"));
add(eq(2, [new MathSuperScript({ children: [M("ℓ")], superScript: [M("")] }), M("̅ ="), new MathFraction({ numerator: [M("1")], denominator: [M("T")] }), new MathSum({ children: [sub("ℓ", "t")], subScript: [M("t=1")], superScript: [M("T")] }), M(",   "), sub("ℓ", "min"), M("="), sub("min", "t"), M(" "), sub("ℓ", "t")]));
add(body("The log-probability describes only the token that was chosen. Predictive entropy instead describes the whole next-token distribution *p*_{t} over the vocabulary *V*:"));
add(eq(3, [sub("H", "t"), M("=−"), new MathSum({ children: [sub("p", "t"), M("(v) log "), sub("p", "t"), M("(v)")], subScript: [M("v∈V")] })]));
add(body("A token can be emitted with a high probability while the remaining probability mass is spread over many alternatives, or the reverse, so the two quantities are complementary. The answer-level entropy is the mean of *H*_{t} over the generated tokens. The original detector uses the feature vector **z** = (mean log-probability, minimum log-probability, mean entropy) and a logistic regression"));
add(eq(4, [M("P(h=1∣z)=σ("), new MathSuperScript({ children: [M("w")], superScript: [M("⊤")] }), M("z+b)")]));
add(body("whose output is the hallucination probability, mapped to PASS, WARN, or FLAG decisions in the accompanying demonstration interface."));
add(h2("C. Token-Level Uncertainty Features"));
add(body("To test whether the three summary statistics discard useful information, 17 features were computed from the same per-token arrays. They fall into seven families: central tendency (mean and median log-probability), percentile statistics (minimum, 10th and 25th percentiles), dispersion (standard deviation of log-probability), low-confidence proportion (fraction of tokens with log-probability below −3), entropy statistics (mean, maximum, and standard deviation of *H*_{t}), positional statistics (mean log-probability of the first and second halves of the answer and their difference), and confidence trend (least-squares slope of log-probability against token position). Answer length and the top-1/top-2 log-probability margin complete the set. The margin requires the full sorted distribution and was cached only for Qwen; for the other two models the margin features are constant and carry no information. All features are functions of the model’s own generation and never of the reference answers."));
add(h2("D. Self-Consistency Baseline"));
add(body("The baseline draws five answers with temperature 0.7 and top-*p* 0.9. Lexical agreement is the mean pairwise token-F1 similarity between the five samples. A semantic variant approximates semantic entropy [4]: the samples are embedded with a Sentence-BERT encoder [13], greedily clustered at a cosine similarity of 0.75, and scored by the entropy of the cluster proportions,"));
add(eq(5, [M("SE=−"), new MathSum({ children: [new MathFraction({ numerator: [sub("n", "k")], denominator: [M("K")] }), M(" log "), new MathFraction({ numerator: [sub("n", "k")], denominator: [M("K")] })], subScript: [M("k")] })]));
add(body("where *n*_{k} is the size of cluster *k* and *K* = 5. This is a cosine-clustering approximation; the original method clusters by bidirectional entailment. Each score is passed to a one-feature logistic regression."));
add(h2("E. Feature-Engineered Detector"));
add(body("The feature-engineered detector applies logistic regression with scikit-learn default regularization to all 17 features. Random forest, XGBoost, and Platt- and isotonic-calibrated logistic regression were compared on the same splits. A four-feature subset (median log-probability, 25th percentile, trend slope, and half-to-half difference) was also selected for cross-model consistency; because that selection used the same questions, it is reported as exploratory."));
add(h2("F. Two-Stage Cascade"));
add(body("A single linear boundary assumes that higher confidence always means lower hallucination risk. Section V shows that this relation can reverse among confidently generated answers, so a cascade first routes each answer by its mean log-probability and then applies a branch-specific logistic regression over the 17 features:"));
add(eq(6, [M("P(h=1∣z)={"), M("σ("), new MathSuperScript({ children: [sub("w", "c")], superScript: [M("⊤")] }), M("z+"), sub("b", "c"), M("), ℓ̅≥τ;  σ("), new MathSuperScript({ children: [sub("w", "u")], superScript: [M("⊤")] }), M("z+"), sub("b", "u"), M("), otherwise")]));
add(body("The threshold τ is the median mean log-probability of the training fold, and each branch classifier is fit only on that branch’s training examples. If a branch has fewer than ten training examples or a single class, the model trained on the full training fold is used instead. No threshold or model ever sees evaluation labels."));
add(h2("G. Calibration and Statistical Evaluation"));
add(body("The primary metric is the area under the ROC curve (AUROC), which does not depend on a decision threshold. Uncertainty is quantified with 95% percentile bootstrap confidence intervals over 2,000 resamples of the evaluation predictions; a result is called significant when the lower bound exceeds 0.5. Two protocols are used. The fixed holdout reproduces the original baseline exactly: a stratified 140/60 split. Out-of-fold (OOF) evaluation uses stratified *k*-fold cross-validation over all 200 questions, so every question receives a prediction from a model that never trained on it. Calibration was assessed with expected calibration error and the Brier score; Platt and isotonic recalibration are monotone and therefore cannot change the ranking on which AUROC depends."));

add(h1("IV. Experimental Setup"));
add(h2("A. Models"));
add(body("Three instruction-tuned models were used through their Hugging Face checkpoints: Qwen/Qwen2.5-1.5B-Instruct, Llama-3.2-1B-Instruct (via the ungated unsloth mirror of the official weights), and HuggingFaceTB/SmolLM2-1.7B-Instruct. Each model received the same chat-template prompt, a system instruction asking for a direct answer of one or two sentences. Inference ran in 16-bit floating point on a single NVIDIA GeForce RTX 4060 GPU with 8 GB of memory."));
add(h2("B. Datasets"));
add(body("TruthfulQA [1] (generation configuration, 817 questions in 38 categories) supplies best, correct, and incorrect reference answers for questions built around misconceptions. SciQ [12] contains crowdsourced science examination questions with one correct answer and three distractors; the distractors serve as incorrect references. A fixed 200-question subset of each dataset was drawn once with a fixed seed and shared by every model. Proxy hallucination rates on TruthfulQA were 59.0% (Qwen), 52.5% (Llama), and 62.0% (SmolLM2), and on SciQ 46.0%, 55.0%, and 45.0%."));
add(h2("C. Data Splitting"));
add(body("The original experiment used a stratified 70/30 split (140 development and 60 evaluation questions, seed 42), reused identically for every model and method. The later analyses added stratified 10-fold OOF evaluation for the multi-method comparison and 5-fold OOF evaluation for the cascade, subgroup, and confound analyses. In every protocol, features were computed per answer and all thresholds and classifiers were fit on training folds only."));
add(h2("D. Evaluation Metrics"));
add(body("Reported metrics are AUROC with bootstrap intervals, accuracy, precision, recall, and F1 at a 0.5 threshold, expected calibration error, the Brier score, and the number of LLM calls per question."));
add(h2("E. Experimental Configurations"));
add(body("Table I summarizes the fixed configuration shared by all experiments."));
add(tcap("I", "Experimental Configuration"));
add(table([1650, 3216], ["Parameter", "Value"], [
  ["Models", "Qwen2.5-1.5B-Instruct, Llama-3.2-1B-Instruct, SmolLM2-1.7B-Instruct"],
  ["Datasets", "TruthfulQA (generation), SciQ"],
  ["Questions", "200 per dataset (fixed subset)"],
  ["Single-pass decoding", "Greedy, ≤64 new tokens"],
  ["Self-consistency", "5 samples, T = 0.7, top-p = 0.9, ≤64 tokens"],
  ["Classifier", "Logistic regression (scikit-learn defaults)"],
  ["Holdout split", "Stratified 140/60"],
  ["OOF protocols", "Stratified 5-fold and 10-fold"],
  ["Bootstrap", "2,000 resamples, 95% percentile CI"],
  ["Seed", "42"],
  ["Hardware", "NVIDIA RTX 4060 (8 GB), FP16"],
]));

add(h1("V. Results and Analysis"));
add(h2("A. Original DA1 Baseline"));
add(body("Table II reproduces the original experiment: Qwen2.5-1.5B-Instruct on TruthfulQA, evaluated on the 60-question holdout, of which 35 answers carry the hallucinated label. The three-feature detector reaches an AUROC of 0.482. Its accuracy of 0.583 equals the majority-class rate, because it predicts every answer as hallucinated. No single feature separates the classes, and the five-call self-consistency baseline is no better (0.461). The detector used 200 LLM calls against 1,000 for self-consistency, an 80% reduction, but equivalence between two chance-level detectors carries no practical meaning."));
add(tcap("II", "Original Baseline (Qwen2.5-1.5B, TruthfulQA, n = 60)"));
add(table([1866, 600, 800, 800, 800], ["Method", "Calls", "AUROC", "Acc.", "F1"], [
  ["3-feature LR", "1", "0.482", "0.583", "0.737"],
  ["Mean log-prob only", "1", "0.551", "0.583", "0.737"],
  ["Min log-prob only", "1", "0.464", "0.583", "0.737"],
  ["Mean entropy only", "1", "0.504", "0.583", "0.737"],
  ["Self-consistency", "5", "0.461", "0.567", "0.723"],
]));
add(h2("B. Three-Model, Two-Dataset Evaluation"));
add(body("Table III repeats the protocol for all three models on both datasets. On TruthfulQA the original detector ranges from 0.382 to 0.482 and neither self-consistency variant exceeds 0.577. On SciQ the same detector reaches 0.616 to 0.756, and mean log-probability alone reaches 0.634 to 0.780, the highest value for every model. Fig. 2 shows the corresponding ROC curves. The dataset effect is larger than the model effect: changing the model moves AUROC by up to 0.10 within a dataset, whereas moving the same model from TruthfulQA to SciQ raises it by 0.14 to 0.34."));
add(tcap("III", "AUROC on the 60-Question Holdout, All Models and Datasets"));
add(table([1010, 700, 790, 790, 788, 788], ["Model", "Data", "3-feat", "Mean LP", "Lex. SC", "Sem. SC"], [
  ["Qwen2.5-1.5B", "TQA", "0.482", "0.551", "0.461", "0.470"],
  ["Llama-3.2-1B", "TQA", "0.481", "0.541", "0.477", "0.528"],
  ["SmolLM2-1.7B", "TQA", "0.382", "0.545", "0.560", "0.577"],
  ["Qwen2.5-1.5B", "SciQ", "0.756", "0.780", "0.676", "0.571"],
  ["Llama-3.2-1B", "SciQ", "0.616", "0.634", "0.595", "0.532"],
  ["SmolLM2-1.7B", "SciQ", "0.717", "0.767", "0.758", "0.544"],
]));
add(tnote("TQA = TruthfulQA; LP = log-probability; SC = self-consistency (5 calls, scored against the label of its medoid sample)."));
add(figure("fig2_roc_truthfulqa_vs_sciq.png", 1014, 525, 2, "ROC curves of the original three-feature detector on the 60-question holdout for TruthfulQA (left) and SciQ (right). AUROC in parentheses."));
add(body("Because every result rests on a proxy label, all generations were relabeled with an embedding-based judge that replaces token-F1 with cosine similarity to the same references. The two labelers agreed on 65.5% to 78.5% of answers (Cohen’s κ 0.34 to 0.56). The qualitative pattern held in five of the six model–dataset cells; the exception was Llama on TruthfulQA, whose AUROC rose from 0.481 to 0.594 under the embedding labels."));
add(h2("C. Feature Ablation"));
add(body("Table IV lists held-out AUROC for individual features on TruthfulQA. The median log-probability (0.555 to 0.578) is more stable across models than the mean (0.541 to 0.551), and the minimum is consistently weakest. Using all 17 features gives 0.528, 0.598, and 0.503. Answer length appears competitive for two models, but it is confounded with the labeler: token-F1 precision falls as answers grow longer, and it falls faster against incorrect references (correlation −0.49) than against correct ones (−0.33). For Qwen, answers labeled grounded averaged 23.7 tokens against 20.0 for hallucinated answers (*t* = 2.90, *p* = 0.004). The embedding labeler shows no such difference (21.7 versus 21.4 tokens, *p* = 0.81)."));
add(tcap("IV", "Held-Out AUROC of Individual Features on TruthfulQA"));
add(table([2166, 900, 900, 900], ["Feature", "Qwen", "Llama", "SmolLM2"], [
  ["Mean log-prob", "0.551", "0.541", "0.545"],
  ["Median log-prob", "0.555", "0.578", "0.571"],
  ["Minimum log-prob", "0.464", "0.493", "0.392"],
  ["25th percentile log-prob", "0.541", "0.545", "0.550"],
  ["Mean entropy", "0.504", "0.549", "0.442"],
  ["Trend slope", "0.550", "0.574", "0.514"],
  ["Second−first half difference", "0.581", "0.526", "0.555"],
  ["Answer length†", "0.541", "0.578", "0.344"],
  ["Original 3 features (LR)", "0.482", "0.481", "0.382"],
  ["All 17 features (LR)", "0.528", "0.598", "0.503"],
]));
add(tnote("† Confounded with the token-F1 labeler (Section V-C)."));
add(body("More flexible classifiers did not help. Across models and feature sets, random forest, XGBoost, and calibrated variants gave no consistent gain over logistic regression, and several overfit the 140 training questions: for Qwen with three features, XGBoost reached a cross-validated AUROC of 0.585 but only 0.414 on the holdout."));
add(h2("D. Confidently-Wrong Analysis"));
add(body("Each answer was assigned to one of four groups by correctness and by whether its mean log-probability fell above or below the median for that model. Fig. 3 reports the share of wrong answers that were confident. On TruthfulQA the share was 49.2% (58 of 118) for Qwen, 51.4% (54 of 105) for Llama, and 52.4% (65 of 124) for SmolLM2; on SciQ it was 32.6%, 40.0%, and 40.0%. On TruthfulQA a wrong answer is therefore about as likely to be confident as uncertain. Any detector that thresholds mean confidence at the median places these answers on the same side as confident correct answers, which is consistent with the chance-level results above."));
add(figure("fig3_confidently_wrong.png", 1014, 480, 3, "Share of wrong answers generated with above-median mean log-probability, by model and dataset."));
add(body("Within the confident group, the direction of the effect reverses. For Qwen, confidently wrong answers had a higher second-half mean log-probability than confidently correct ones (−0.265 versus −0.371, Cohen’s *d* = 0.73): confidence in correct answers tended to decline toward the end of the answer (half-to-half difference −0.092), whereas it rose in wrong ones (+0.050). The wrong answers were also shorter (16.9 versus 21.7 tokens, *d* = −0.71). Llama showed the same direction with smaller effects (*d* = 0.39 to 0.41), and SmolLM2 showed none (|*d*| ≤ 0.10). A representative confidently wrong Qwen answer is “Stephen Hawking is an example of a scientist in Britain who won the Physics Nobel Prize,” generated with a mean log-probability of −0.161."));
add(body("Restricted to the confident group, with the routing threshold fit inside each training fold, the 17-feature classifier separated wrong from correct answers with 5-fold OOF AUROC 0.656 [0.544, 0.758] for Qwen and 0.642 [0.524, 0.749] for Llama, compared with 0.518 and 0.424 for the original three features; SmolLM2 remained at 0.443 [0.331, 0.557]. The shorter length of confidently wrong answers overlaps with the labeler’s length bias, which Section V-E examines directly."));
add(h2("E. Improved Detector"));
add(body("The cascade of Section III-F applies this finding to all answers. Table V reports it for Qwen and the flat 17-feature classifier for Llama and SmolLM2, since the cascade did not help the latter two (Llama 0.630, SmolLM2 0.417 under 5-fold OOF). Under 5-fold OOF, the Qwen cascade reached 0.598 [0.516, 0.678] and the Llama classifier 0.653 [0.576, 0.727], both significant; the original detector reached 0.399 and 0.566 under the same protocol. On the identical 60-question holdout of the original experiment, the same detectors reached 0.560 [0.404, 0.710] and 0.598 [0.446, 0.738], which move in the same direction but are not individually significant. SmolLM2 was not significant under either protocol."));
add(tcap("V", "TruthfulQA AUROC [95% CI] Under Both Protocols"));
add(table([850, 1216, 1400, 1400], ["Model", "Detector", "Holdout (n = 60)", "5-fold OOF (n = 200)"], [
  ["Qwen", "Original 3-feat", "0.482\n[0.333, 0.629]", "0.399\n[0.324, 0.475]"],
  ["Qwen", "Cascade, 17-feat", "0.560\n[0.404, 0.710]", "**0.598***\n[0.516, 0.678]"],
  ["Llama", "Original 3-feat", "0.481\n[0.324, 0.631]", "0.566\n[0.486, 0.644]"],
  ["Llama", "Flat LR, 17-feat", "0.598\n[0.446, 0.738]", "**0.653***\n[0.576, 0.727]"],
  ["SmolLM2", "Original 3-feat", "0.382\n[0.242, 0.536]", "0.418\n[0.336, 0.498]"],
  ["SmolLM2", "Flat LR, 17-feat", "0.503\n[0.352, 0.657]", "0.485\n[0.408, 0.566]"],
]));
add(tnote("* 95% bootstrap CI excludes 0.5."));
add(body("Because answer length is confounded with the token-F1 labeler, the analysis was repeated without it (Table VI). Against the token-F1 labels, removing length reduced the Qwen cascade to 0.539, no longer significant, and the Llama classifier to 0.578, which remained just significant. Against the embedding labels, which show no length bias, the length-free detectors were significant for Qwen (cascade 0.648, flat 0.592) and Llama (flat 0.655, cascade 0.650), and not for SmolLM2 (0.581). Part of the headline gain therefore reflected the labeler rather than hallucination. A smaller signal from the remaining token-level features survives for Qwen and Llama, although the embedding-label evaluation was added after the main results and should be read as a robustness check rather than a confirmatory test."));
add(tcap("VI", "Answer-Length Confound Check, 5-Fold OOF AUROC"));
add(table([966, 1300, 1300, 1300], ["Model", "17 feat.\nF1 labels", "16 feat. (no length)\nF1 labels", "16 feat. (no length)\nEmbedding labels"], [
  ["Qwen‡", "0.598*\n[0.516, 0.678]", "0.539\n[0.460, 0.619]", "0.648*\n[0.567, 0.722]"],
  ["Llama", "0.653*\n[0.576, 0.727]", "0.578*\n[0.501, 0.654]", "0.655*\n[0.576, 0.732]"],
  ["SmolLM2", "0.485\n[0.408, 0.566]", "0.494\n[0.413, 0.574]", "0.581\n[0.496, 0.655]"],
]));
add(tnote("‡ Two-stage cascade; other rows use the flat classifier. * CI excludes 0.5."));
add(h2("F. Cross-Model Findings"));
add(body("To test whether a detector learns a general confidence pattern or a model-specific one, the original three-feature detector was trained on one model’s answers and evaluated on another’s for the same questions. On SciQ, transfer AUROC ranged from 0.637 to 0.786, and a detector trained on Llama scored 0.786 on Qwen’s answers (0.795 after per-model standardization), above Qwen’s own within-model 0.756. On TruthfulQA, transfer stayed between 0.427 and 0.504 without standardization. This indicates that the association between low confidence and wrong answers on SciQ is shared across these three models; it does not establish generalization beyond them."));
add(h2("G. Statistical Validation"));
add(body("Fig. 4 contrasts the two protocols. The out-of-fold intervals are narrower because every one of the 200 questions contributes an evaluation prediction, rather than 60; for the Qwen cascade the interval width falls from 0.306 to 0.162, a ratio of 1.89, close to the √(200/60) ≈ 1.83 expected from sample size alone. The holdout remains important because it is the only comparison against the original baseline on literally the same questions, and on it the improvement is directional but not significant. The original three-feature detector for Qwen scored 0.399 [0.324, 0.475] under OOF, an interval entirely below 0.5: the relation it learns on training folds reverses on held-out folds, consistent with the sign reversal described in Section V-D."));
add(figure("fig4_final_validation.png", 1014, 630, 4, "TruthfulQA AUROC of the original and improved detectors on the 60-question holdout and under 5-fold OOF evaluation. Error bars are 95% bootstrap CIs; * marks intervals that exclude 0.5. Improved = cascade (Qwen) or flat 17-feature classifier (Llama, SmolLM2)."));
add(body("Several claims did not survive checking and were corrected. An earlier version of the confident-subgroup analysis computed its routing threshold on all 200 questions; fitting it inside each training fold changed AUROC by at most 0.03. An apparent gain from restricting evaluation to answers with a large label margin reversed under OOF evaluation (Qwen: 0.611 at margin ≥ 0.1, 0.377 at margin ≥ 0.2 with *n* = 32) and was attributed to small samples. Earlier summaries also described the 0.598 and 0.653 results without the holdout and length caveats reported here."));
add(h2("H. Comparison with Self-Consistency and Semantic Entropy"));
add(body("Table VII compares all methods against one fixed target, the label of the single greedy answer, under 10-fold OOF evaluation. On TruthfulQA, neither lexical self-consistency nor semantic entropy was significant for any model. The four-feature subset was significant for Qwen (0.591) and the six-call hybrid for Llama (0.652), but the subset was selected on the same data and the hybrid costs six calls. On SciQ, the original single-call detector was significant for every model (0.632 to 0.742), lexical self-consistency matched or trailed it (0.598 to 0.666), and the cosine-clustered semantic entropy was not significant for any model. A planned sweep to ten samples was attempted twice and interrupted by the host environment at 62% and 33% completion, so whether larger sample budgets help remains open."));
add(tcap("VII", "10-Fold OOF AUROC Against the Single-Pass Label"));
add(table([1320, 591, 591, 591, 591, 591, 591], ["Method (calls)", "TQA\nQwen", "TQA\nLlama", "TQA\nSmol", "SciQ\nQwen", "SciQ\nLlama", "SciQ\nSmol"], [
  ["Original 3-feat (1)", "0.398", "0.564", "0.438", "0.742*", "0.632*", "0.651*"],
  ["4-feat subset§ (1)", "0.591*", "0.574", "0.458", "0.709*", "0.615*", "0.653*"],
  ["Lexical SC (5)", "0.485", "0.489", "0.571", "0.639*", "0.598*", "0.666*"],
  ["Semantic entropy (5)", "0.497", "0.507", "0.417", "0.561", "0.511", "0.496"],
  ["Hybrid (6)", "0.580", "0.652*", "0.519", "0.736*", "0.624*", "0.647*"],
]));
add(tnote("* CI excludes 0.5. § Selected on the same questions; exploratory."));

add(h1("VI. Discussion"));
add(body("The research question asked when token-level uncertainty from a single generation can detect factual hallucinations. The evidence gives a conditional answer. On ordinary factual questions such as SciQ, it works: a single call is significant for all three models, matches or exceeds five-call self-consistency, and transfers between models. On TruthfulQA, raw confidence fails, because about half of the wrong answers are generated as confidently as correct ones. These are the imitative falsehoods the benchmark was designed to elicit [1], and a model that has absorbed a misconception has no reason to be uncertain about it."));
add(body("Finer-grained statistics recover part of the signal for Qwen and Llama. Among confident answers, the wrong ones are shorter and maintain or increase their confidence toward the end, whereas correct ones tend to lose confidence late in the answer. A routed classifier can exploit this reversal where a single linear boundary cannot. The improvement is modest, however (AUROC 0.54 to 0.66 depending on labeler and features), it is significant only under out-of-fold evaluation, and part of it is explained by answer length interacting with the proxy labeler. SmolLM2 is an informative negative result: no feature set, classifier, cascade, or sampling method separated its answers on TruthfulQA."));
add(body("Taken together, confidence-only detection can screen for hallucinations when the model is uncertain because it lacks knowledge, but it cannot by itself resolve confidently wrong claims. Detecting those claims likely requires evidence from outside the generating model’s own probabilities, such as retrieval against a trusted source or an independent verifier model."));

add(h1("VII. Limitations"));
add(bullet("Each dataset is a single 200-question subset, and each result uses a single seed; intervals are wide, and the 60-question holdout has limited statistical power."));
add(bullet("All labels are automatic proxies. The token-F1 labeler has a measurable length bias, the two labelers agree on only 65.5–78.5% of answers, and results for some cells depend on which is used."));
add(bullet("Results depend on the model and the dataset; three models of 1–2 billion parameters and two short-answer datasets do not support claims about larger models, other hallucination types, or long-form generation."));
add(bullet("The four-feature subset was selected on the evaluation data, and the embedding-label confound check was added after the main results."));
add(bullet("Margin features were available only for Qwen, and the semantic-entropy baseline uses cosine clustering rather than entailment."));
add(bullet("The ten-sample self-consistency sweep was not completed because of environment interruptions."));
add(bullet("The detector uses no retrieval or external verifier, so it cannot catch errors the model makes with full confidence."));

add(h1("VIII. Conclusion and Future Work"));
add(body("This work investigated single-pass token-level uncertainty as a low-cost hallucination detector against five-sample self-consistency, across three small instruction-tuned models and two datasets. The signal is useful for ordinary factual questions, where one call matches or exceeds five, and weak for misconception-oriented questions, where about half of all errors are made confidently. Finer-grained positional and percentile features with a confidence-routed classifier recovered a modest, cross-validated signal for two of three models, but not on the original holdout alone, not independently of a labeler length bias for Qwen, and not for SmolLM2. Future work should confirm these effects on a larger, independently held-out question set, use entailment-based or human labels, complete the sampling-budget sweep, and combine token-level uncertainty with retrieval or a verifier model to address confidently wrong claims."));

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
        children: [new TextRun({ text: TITLE.toUpperCase(), bold: true, italics: true, size: 40 })] }),
    ] },
    { properties: { page, type: SectionType.CONTINUOUS, column: { count: 2, space: 360 } }, children: [
      ...author("Goutham Krishna Jayakrishnan", "24BCE1910"),
      new Paragraph({ children: [new ColumnBreak()] }),
      ...author("Shubham Khadilkar", "24BCE5275"),
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
