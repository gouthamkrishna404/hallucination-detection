// Builds the v3 paper (IEEE-style, two-column, A4) from the saved analysis files.
// Layout follows the ParkPulse reference: A4, Times New Roman, 0.63" side margins,
// 0.25" column gap, small-caps section heads, 8 pt captions, three-rule tables.
//
// Every number in tables and prose is read at build time from
//   analysis_v2/confirm.json   preregistered confirmatory results   (scripts/v2_confirm.py)
//   analysis_v3/audit.json     post-hoc audit                      (scripts/v3_audit.py)
//   analysis_v3/extensions.json new detectors, labels, form controls (scripts/v3_eval.py)
//   analysis_v3/within.json    within-question test                (scripts/v3_within.py)
//   analysis_v3/extra.json     second prompt, larger model         (scripts/v3_extra.py)
// so the text cannot drift from the results. scripts/v3_check_paper.py cross-checks the output.
// Usage: cd paper && npm install && node build_paper.js
const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, ImageRun, Table, TableRow, TableCell, AlignmentType,
  BorderStyle, WidthType, SectionType, ColumnBreak, TabStopType, LevelFormat,
  Math: OMath, MathRun, MathSubScript, MathSum, MathFraction,
} = require("docx");

const ROOT = path.join(__dirname, "..");
const COL = 4866; // twips: (11906 - 2*907 - 360) / 2
const FIG = (f) => path.join(__dirname, "figures", f);
const OUT = path.join(__dirname, "Learned_Falsehoods_Uncertainty_Paper.docx");

// ---------- data ----------
const J = (p) => JSON.parse(fs.readFileSync(path.join(ROOT, p), "utf8"));
const C = J("analysis_v2/confirm.json"), A = J("analysis_v3/audit.json"), E = J("analysis_v3/extensions.json");
const W = J("analysis_v3/within.json"), X = J("analysis_v3/extra.json"), K = J("analysis_v3/constants.json");
const MS = ["qwen2.5-1.5b-instruct", "llama-3.2-1b-instruct", "smollm2-1.7b-instruct"];
const SHORT = { "qwen2.5-1.5b-instruct": "Qwen", "llama-3.2-1b-instruct": "Llama", "smollm2-1.7b-instruct": "SmolLM2" };
const TQ = (m) => `${m}|truthfulqa`, SQ = (m) => `${m}|sciq`;
const num = (x, d = 3) => (x < 0 ? "−" : "") + Math.abs(x).toFixed(d);
const sg = (x, d = 2) => (x < 0 ? "−" : "+") + Math.abs(x).toFixed(d);
const ci = (c, d = 2) => `[${num(c[0], d)}, ${num(c[1], d)}]`;
const rng = (xs, d = 2) => `${num(Math.min(...xs), d)}–${num(Math.max(...xs), d)}`;
const list3 = (xs, d = 3) => xs.map((x) => num(x, d)).join(", ");
const pct = (x) => `${Math.round(100 * x)}%`;
const P = (p) => (p < 0.001 ? "< 0.001" : `= ${p.toFixed(3)}`);
const sig = (r) => r.p_holm < 0.05 && (r.ci[0] > 0.5 || r.ci[1] < 0.5);
const by = (f) => MS.map(f);

// ---------- references: cited as [@key], numbered in order of first citation ----------
const REFS = {
  lin2022: "S. Lin, J. Hilton, and O. Evans, “TruthfulQA: Measuring how models mimic human falsehoods,” in *Proc. ACL*, 2022, pp. 3214–3252, doi: 10.18653/v1/2022.acl-long.229.",
  manakul2023: "P. Manakul, A. Liusie, and M. J. F. Gales, “SelfCheckGPT: Zero-resource black-box hallucination detection for generative large language models,” in *Proc. EMNLP*, 2023, pp. 9004–9017, doi: 10.18653/v1/2023.emnlp-main.557.",
  wang2023: "X. Wang *et al.*, “Self-consistency improves chain of thought reasoning in language models,” in *Proc. ICLR*, 2023.",
  farquhar2024: "S. Farquhar, J. Kossen, L. Kuhn, and Y. Gal, “Detecting hallucinations in large language models using semantic entropy,” *Nature*, vol. 630, pp. 625–630, 2024, doi: 10.1038/s41586-024-07421-0.",
  fadeeva2024: "E. Fadeeva *et al.*, “Fact-checking the output of large language models via token-level uncertainty quantification,” in *Findings ACL*, 2024, pp. 9367–9385, doi: 10.18653/v1/2024.findings-acl.558.",
  kadavath2022: "S. Kadavath *et al.*, “Language models (mostly) know what they know,” arXiv:2207.05221, 2022.",
  su2024: "W. Su *et al.*, “Unsupervised real-time hallucination detection based on the internal states of large language models,” in *Findings ACL*, 2024, pp. 14379–14391, doi: 10.18653/v1/2024.findings-acl.854.",
  du2024: "X. Du, C. Xiao, and Y. Li, “HaloScope: Harnessing unlabeled LLM generations for hallucination detection,” in *Adv. Neural Inf. Process. Syst.*, vol. 37, 2024, pp. 102948–102972, doi: 10.52202/079017-3270.",
  yehuda2024: "Y. Yehuda *et al.*, “InterrogateLLM: Zero-resource hallucination detection in LLM-generated answers,” in *Proc. ACL*, 2024, pp. 9333–9347, doi: 10.18653/v1/2024.acl-long.506.",
  han2025: "J. Han *et al.*, “Simple factuality probes detect hallucinations in long-form natural language generation,” in *Findings EMNLP*, 2025, doi: 10.18653/v1/2025.findings-emnlp.880.",
  bakman2025: "Y. F. Bakman *et al.*, “Reconsidering LLM uncertainty estimation methods in the wild,” in *Proc. ACL*, 2025, pp. 29531–29556, doi: 10.18653/v1/2025.acl-long.1429.",
  welbl2017: "J. Welbl, N. F. Liu, and M. Gardner, “Crowdsourcing multiple choice science questions,” in *Proc. 3rd Workshop Noisy User-Generated Text (W-NUT)*, 2017.",
  reimers2019: "N. Reimers and I. Gurevych, “Sentence-BERT: Sentence embeddings using Siamese BERT-networks,” in *Proc. EMNLP-IJCNLP*, 2019.",
  kuhn2023: "L. Kuhn, Y. Gal, and S. Farquhar, “Semantic uncertainty: Linguistic invariances for uncertainty estimation in natural language generation,” in *Proc. ICLR*, 2023.",
  zheng2023: "L. Zheng *et al.*, “Judging LLM-as-a-judge with MT-Bench and Chatbot Arena,” in *Adv. Neural Inf. Process. Syst.*, vol. 36, 2023.",
  holm1979: "S. Holm, “A simple sequentially rejective multiple test procedure,” *Scand. J. Statist.*, vol. 6, no. 2, pp. 65–70, 1979.",
  orgad2024: "H. Orgad *et al.*, “LLMs know more than they show: On the intrinsic representation of LLM hallucinations,” arXiv:2410.02707, 2024.",
  kalai2025: "A. T. Kalai *et al.*, “Why language models hallucinate,” arXiv:2509.04664, 2025.",
  geifman2017: "Y. Geifman and R. El-Yaniv, “Selective classification for deep neural networks,” in *Adv. Neural Inf. Process. Syst.*, vol. 30, 2017.",
  touvron2023: "H. Touvron *et al.*, “Llama 2: Open foundation and fine-tuned chat models,” arXiv:2307.09288, 2023.",
  truthjudge: "Allen Institute for AI, “truthfulqa-truth-judge-llama2-7B,” Hugging Face model, 2024. [Online]. Available: https://huggingface.co/allenai/truthfulqa-truth-judge-llama2-7B (training code: https://github.com/yizhongw/truthfulqa_reeval).",
  he2021: "P. He, X. Liu, J. Gao, and W. Chen, “DeBERTa: Decoding-enhanced BERT with disentangled attention,” in *Proc. ICLR*, 2021.",
  williams2018: "A. Williams, N. Nangia, and S. R. Bowman, “A broad-coverage challenge corpus for sentence understanding through inference,” in *Proc. NAACL-HLT*, 2018, pp. 1112–1122, doi: 10.18653/v1/N18-1101.",
  qwen2024: "Qwen Team (A. Yang *et al.*), “Qwen2.5 technical report,” arXiv:2412.15115, 2024.",
  llama2024: "A. Grattafiori *et al.*, “The Llama 3 herd of models,” arXiv:2407.21783, 2024.",
  allal2025: "L. Ben Allal *et al.*, “SmolLM2: When smol goes big — data-centric training of a small language model,” arXiv:2502.02737, 2025.",
};
const CITED = [];
const cite = (text) => text.replace(/\[@(\w+)\]/g, (_, k) => {
  if (!REFS[k]) throw new Error(`unknown reference ${k}`);
  if (!CITED.includes(k)) CITED.push(k);
  return `[${CITED.indexOf(k) + 1}]`;
});

// ---------- inline markup: *italic*, **bold**, _{sub}, ^{sup} ----------
function runs(text, base = {}) {
  text = cite(text);
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
function eq(n, children) {
  return new Paragraph({
    tabStops: [{ type: TabStopType.CENTER, position: COL / 2 }, { type: TabStopType.RIGHT, position: COL }],
    spacing: { before: 60, after: 60 },
    children: [new TextRun("\t"), new OMath({ children }), new TextRun(`\t(${n})`)],
  });
}

// ---------- tables ----------
const RULE = { style: BorderStyle.SINGLE, size: 4, color: "000000" };
const THIN = { style: BorderStyle.SINGLE, size: 2, color: "000000" };
const NONE = { style: BorderStyle.NONE, size: 0, color: "FFFFFF" };
function cell(text, width, { bold = false, align = AlignmentType.CENTER, size = 15 } = {}) {
  return new TableCell({
    width: { size: width, type: WidthType.DXA },
    margins: { top: 25, bottom: 25, left: 45, right: 45 },
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
const tcap = (n, text) => new Paragraph({
  children: [new TextRun({ text: `TABLE ${n}.  ${text}`, smallCaps: true, size: 16 })],
  alignment: AlignmentType.CENTER, keepNext: true, spacing: { before: 200, after: 100 },
});
const tnote = (text) => new Paragraph({
  children: runs(text, { size: 14 }), alignment: AlignmentType.JUSTIFIED, spacing: { before: 40, after: 160 },
});

// ---------- figures ----------
function figure(file, num, caption) {
  const png = fs.readFileSync(FIG(file));
  const pxW = png.readUInt32BE(16), pxH = png.readUInt32BE(20);
  const w = 318, h = Math.round((pxH / pxW) * w);
  return [
    new Paragraph({
      alignment: AlignmentType.CENTER, keepNext: true, spacing: { before: 120, after: 0 },
      children: [new ImageRun({ type: "png", data: png, transformation: { width: w, height: h } })],
    }),
    new Paragraph({
      children: runs(`Fig. ${num}.  ${caption}`, { size: 16 }), alignment: AlignmentType.JUSTIFIED,
      spacing: { before: 60, after: 160 },
    }),
  ];
}

// ---------- derived numbers used in several places ----------
const d0 = (c) => C[c].detectors.D0;
const gap = (m) => C[TQ(m)].H2_error_type;

// ======================================================================= CONTENT
const content = [];
const add = (...xs) => xs.flat().forEach((x) => content.push(x));
const tq = (f) => MS.map((m) => f(TQ(m), m)), sq = (f) => MS.map((m) => f(SQ(m), m));
const cw = (c) => A[c].confident;
const aff = (m, k) => E[TQ(m)].form.affirmative.error_type[k];
const et = (m, k) => E[TQ(m)].error_type[k];
const ih = (m) => E._independent_h2[m];
const tjl = (m) => E._labels[TQ(m)];

// ---------------------------------------------------------------- I. Introduction
add(h1("I. Introduction"));
add(body("Language models produce fluent answers that are sometimes false. If such answers could be flagged cheaply, they could be withheld or routed to a stronger verifier. Sampling-based detectors such as SelfCheckGPT [@manakul2023], self-consistency [@wang2023] and semantic entropy [@farquhar2024] compare several generations and multiply inference cost. The cheapest alternative reads what the model exposes while it produces a single answer: the probability of each generated token and the entropy of each next-token distribution [@fadeeva2024], [@kadavath2022]."));
add(body("Every such signal measures the model's own belief, and belief is not truth. A model that has absorbed a popular misconception should assert it fluently, rate it as true when asked, and repeat it consistently across samples. TruthfulQA [@lin2022] was built around exactly these *imitative falsehoods*. If the argument is right, uncertainty-based detectors should be blind to one specific kind of error regardless of how the uncertainty is read out, while still seeing errors the model makes from not knowing."));
add(body("We test this with three small instruction-tuned models on TruthfulQA and SciQ. An audit of our earlier study found that one model's scores were contaminated by a decoding-time repetition penalty and that our automatic labels were unreliable, so the pipeline was rebuilt, every modelling decision was made on a development split, and detectors and hypotheses were preregistered before a single evaluation on 1,417 held-out questions. We then tried to break the preregistered findings with a post-hoc audit, reported separately as exploratory. We ask: (RQ1) does single-pass token uncertainty detect wrong answers on held-out questions, and how much of the signal concerns the question rather than the answer; (RQ2) which errors does it miss, and is that blind spot specific to token probabilities; and (RQ3) does the answer survive changes of label source, error typing, prompt and model size? Our contributions are:"));
add(item("A preregistered held-out evaluation of single-pass token uncertainty for three small open models on two datasets, with every negative result reported (RQ1)."));
add(item("Evidence that errors repeating a popular misconception are scored like correct answers, not only by token probability but also by the model's own true/false judgement and by NLI-based semantic entropy. For two of three models the gap to other errors survives controls for length, question familiarity and difficulty, topic and answer form, independent labels and error types, and a second prompt; within a question, the misconception is often the model's more probable answer (RQ2–RQ3)."));
add(item("A decomposition showing that most of the SciQ signal is shared question difficulty, with a smaller answer-specific component measured within questions."));
add(item("The practical consequence: abstaining on the least confident answers does not reduce the TruthfulQA error rate for two of three models, plus four documented pitfalls that each changed a conclusion."));

// ---------------------------------------------------------------- II. Related work
add(h1("II. Related Work"));
add(body("TruthfulQA [@lin2022] measures whether models reproduce human misconceptions; its authors call these imitative falsehoods and found that larger models were less truthful on it. SciQ [@welbl2017] contains science examination questions with one correct answer and three distractors. Larger models are reported to be reasonably calibrated about their own answers, and their probability that a proposed answer is true, P(True), separates right from wrong answers [@kadavath2022]; claim-conditioned probability refines token uncertainty for fact-checking [@fadeeva2024]."));
add(body("Sampling-based methods assume that a model which knows a fact answers consistently: SelfCheckGPT [@manakul2023] compares a response with further samples, self-consistency [@wang2023] aggregates sampled answers, and semantic entropy [@farquhar2024], [@kuhn2023] measures uncertainty over clusters of samples that entail each other. White-box methods use internal states: MIND [@su2024], HaloScope [@du2024] and linear factuality probes [@han2025]; Orgad *et al.* [@orgad2024] show that internal representations encode truthfulness and can predict the type of error a model will make. InterrogateLLM [@yehuda2024] re-queries the model. A broad comparison of uncertainty methods [@bakman2025] warns that results depend strongly on the evaluation setup, and Kalai *et al.* [@kalai2025] argue that training and evaluation reward confident guessing. Selective prediction [@geifman2017] turns a confidence score into an abstention rule; we use it to state what a detector would buy in practice."));
add(body("Our primary labels come from an LLM judge applying a written rubric. LLM judges are useful but imperfect [@zheng2023], so we also label every TruthfulQA answer with an independent judge, Llama-2-7B [@touvron2023] fine-tuned on TruthfulQA's human truthfulness ratings as an open replacement for the original GPT-judge [@truthjudge], and type errors with an NLI model [@he2021], [@williams2018]."));

// ---------------------------------------------------------------- III. Method
add(h1("III. Method"));
add(h2("A. Models, Data and Splits"));
add(body("We used Qwen2.5-1.5B-Instruct [@qwen2024], Llama-3.2-1B-Instruct [@llama2024] and SmolLM2-1.7B-Instruct [@allal2025] with one shared system prompt asking for a direct one- or two-sentence answer and, if unsure, a best specific answer. The *dev* split is the original 200-question subset of each dataset; every exploratory analysis and modelling decision before preregistration was made there. The *confirm* split is all remaining TruthfulQA questions (617) and the remaining SciQ test questions (800), disjoint from dev by question text and evaluated only after the preregistration was committed. No question appears in both splits; eight TruthfulQA template variants (e.g. the same sentence frame with a different country) and three SciQ paraphrases have character-level similarity ≥ 0.8 across splits, which matters only for the two small classifiers fitted on dev."));
add(h2("B. Clean Decoding and Scoring"));
add(body(`Our earlier pipeline read scores from the generation interface. Qwen2.5 ships a repetition penalty of 1.1 that HuggingFace applies as a logits processor even under greedy decoding, so those scores described a penalty-adjusted distribution; the shift in mean log-probability (${K.contamination.mean_abs_shift.toFixed(3)} on average) was ${Math.round(K.contamination.ratio)} times the separation between correct and incorrect answers. Answers are now generated greedily with every logits processor disabled, in fp32, up to 64 new tokens, and frozen. Each answer is re-scored by one fp32 teacher-forced pass, so the score for token *y*_{t} is exactly the model's conditional`));
add(eq(1, [sub("ℓ", "t"), M(" = log "), sub("p", "θ"), M("("), sub("y", "t"), M(" ∣ x, "), sub("y", "<t"), M(")")]));
add(body("where *x* is the prompt. In every cell 100% of generated tokens equal the argmax of the re-scored distribution, which verifies both alignment and the absence of decoding modifications. The same pass gives the next-token entropy and the top-1/top-2 margin. For self-consistency we draw *K* = 5 samples per question at temperature 0.7 and top-*p* 0.9, scored the same way (SmolLM2's samples were generated in bf16 to fit an 8 GB GPU; all scoring is fp32)."));
add(h2("C. Labels and Error Types"));
add(body(`*Primary labels.* Every greedy answer was labelled by an LLM judge (Claude) following a fixed written rubric: an answer is hallucinated if it asserts something false that bears on the question; for SciQ, if its final answer is not the correct one. References are evidence, not the definition; an explicit non-answer is labelled correct and flagged. The judge also records an error type: *imitative* if the answer repeats a listed misconception (TruthfulQA) or distractor (SciQ), *other* otherwise. Items were deduplicated across models, shuffled, and shown without model identity, features or probabilities, but the judge was not blind to the study's hypotheses. A blind re-judge of 100 dev items agreed on ${(100 * K.rejudge.truthfulqa.agree).toFixed(1)}% of TruthfulQA (κ = ${K.rejudge.truthfulqa.kappa.toFixed(2)}) and ${(100 * K.rejudge.sciq.agree).toFixed(1)}% of SciQ items (κ = ${K.rejudge.sciq.kappa.toFixed(2)}).`));
add(body(`*Independent checks.* Because the primary judge knew the hypotheses, two checks use no input from it. Labels: every TruthfulQA answer was also scored by the AllenAI truth-judge [@truthjudge], Llama-2-7B fine-tuned on TruthfulQA's human ratings (reported held-out accuracy 0.945), run in 4-bit NF4 with the prompt format it was trained on; an answer is hallucinated if P(yes) < 0.5. Error types: an answer is *imitative* if DeBERTa-large-MNLI [@he2021], [@williams2018] finds that it entails one of the dataset's listed incorrect answers (both prefixed with the question). On SciQ the containment label of our earlier study (does the answer contain the correct term and no distractor) is the preregistered secondary label; against the judge on the confirm split it accepts ${rng(MS.map((m) => 100 * W[m].validation_on_greedy.containment.false_correct), 0)}% of wrong answers but rejects ${rng(MS.map((m) => 100 * W[m].validation_on_greedy.containment.false_wrong), 0)}% of correct ones, mostly paraphrases.`));
add(h2("D. Detectors"));
add(body("Every detector outputs a risk score; larger means more likely hallucinated. D0 is the negative mean token log-probability,"));
add(eq(2, [M("D0 = −"), new MathFraction({ numerator: [M("1")], denominator: [M("T")] }),
  new MathSum({ children: [sub("ℓ", "t")], subScript: [M("t=1")], superScript: [M("T")] })]));
add(body("and D1 the negative minimum log-probability. D2 and D3 are logistic regressions on standardized features, D2 on three (mean and minimum log-probability, mean entropy) and D3 on all 17 token features; both are trained on the dev split of the same model and dataset and frozen. D4 is the negated fraction of the five samples matching the greedy answer (sentence-embedding cosine ≥ 0.75 [@reimers2019]) and D5 the entropy of cosine clusters of the samples. Post hoc we added the standard versions of the sampling and self-evaluation baselines: D6 = 1 − P(True), the model's own probability that its answer is true when asked in a fresh turn [@kadavath2022]; D7 and D8, semantic entropy over the five samples clustered by bidirectional entailment [@farquhar2024], [@kuhn2023] with frequency and likelihood weights; and D9, the negated fraction of samples that are bidirectionally equivalent to the greedy answer."));
add(h2("E. Preregistered Tests and Post-hoc Audit"));
add(body("The preregistration fixed, before any confirm label was compared with a score: H1 (primary), D0 has AUROC ≠ 0.5 in each cell; H2, on TruthfulQA the AUROC of D0 for imitative errors versus correct answers differs from that for other errors; H3–H4, paired AUROC differences of D1–D5 versus D0 and their individual AUROCs; H5, a model's own D0 predicts its errors better than another model's D0 on the same questions. Fixed scores are tested by label permutation (10,000 permutations), differences by paired or stratified bootstrap (2,000 resamples), all two-sided, with Holm correction [@holm1979] across the six cells within each family. AUROC below 0.60 is called negligible, 0.60–0.70 weak and 0.70–0.80 moderate."));
add(body("Everything else is a post-hoc audit, designed after the confirm results were known in order to find reasons the preregistered findings might be artifacts: the preregistered SciQ containment sensitivity analysis that our confirmatory script had omitted, length- and question-controlled versions of H2, an answer-form control, the new detectors, independent labels and error types, a second prompt and a larger model. These results are labelled exploratory; where we report Holm-adjusted *p*-values, the family is the cells of one analysis."));

// ---------------------------------------------------------------- IV. Results
add(h1("IV. Results"));
add(h2("A. Base Rates"));
add(body(`Judge-labelled hallucination rates on the confirm split are ${list3(tq((c) => C[c].base_rate), 2)} on TruthfulQA and ${list3(sq((c) => C[c].base_rate), 2)} on SciQ for Qwen, Llama and SmolLM2 (truth-judge, TruthfulQA: ${list3(MS.map((m) => tjl(m).agreement_confirm.rate_truthjudge), 2)}). Llama reaches the 64-token limit on a third of its TruthfulQA answers; its D0 result is unchanged on non-truncated answers (${num(K.llama_untruncated.auroc)} ${ci(K.llama_untruncated.ci)}).`));

add(h2("B. H1: Does Mean Log-Probability Detect Errors?"));
add(body(`Table I gives every detector on the confirm split. D0 shows detection in four of six cells: all three models on SciQ (${rng(sq((c) => d0(c).auroc))}) and Llama on TruthfulQA (${num(d0(TQ(MS[1])).auroc)}). For Qwen (${num(d0(TQ(MS[0])).auroc)}) and SmolLM2 (${num(d0(TQ(MS[2])).auroc)}) on TruthfulQA it is at chance. Every detected effect is weak by the preregistered scale, and the Llama/TruthfulQA one depends on the labels (Section IV-E). The preregistered SciQ sensitivity analysis with the containment label gives the same answer (D0 ${list3(MS.map((m) => A.containment[m].detectors.D0.auroc))}, all Holm *p* ${P(Math.max(...MS.map((m) => A.containment[m].detectors.D0.p_holm)))}).`));
const detRow = (name, k, src) => [name, ...["truthfulqa", "sciq"].flatMap((d) => MS.map((m) => {
  const r = (src === "C" ? C : E)[`${m}|${d}`].detectors[k];
  return !sig(r) ? num(r.auroc) : r.auroc > 0.5 ? `**${num(r.auroc)}**` : `*${num(r.auroc)}*`;
}))];
add(tcap("I", "AUROC on the Confirm Split (617 TruthfulQA / 800 SciQ Questions)"));
add(table([1600, 545, 545, 545, 545, 545, 545], ["Detector", "TQA\nQwen", "TQA\nLlama", "TQA\nSmol", "SciQ\nQwen", "SciQ\nLlama", "SciQ\nSmol"], [
  detRow("D0 −mean log-prob", "D0", "C"),
  detRow("D1 −min log-prob", "D1", "C"),
  detRow("D2 3-feature LR", "D2", "C"),
  detRow("D3 17-feature LR", "D3", "C"),
  detRow("D4 cosine support (5)", "D4", "C"),
  detRow("D5 cosine entropy (5)", "D5", "C"),
  detRow("D6 1 − P(True)†", "D6", "E"),
  detRow("D7 NLI entropy (5)†", "D7", "E"),
  detRow("D9 NLI support (5)†", "D9", "E"),
]));
add(tnote(`Bold: Holm-adjusted two-sided permutation *p* < 0.05 and 95% bootstrap CI above 0.5; italic: significantly *below* 0.5 (inverted). CIs are about ±0.04. D0–D5 preregistered; † post hoc. D2 and D3 are trained on the 200-question dev split and frozen. (5): uses five extra samples. D8 (likelihood-weighted NLI entropy) is within ${Math.max(...MS.flatMap((m) => [TQ(m), SQ(m)]).map((c) => Math.abs(E[c].detectors.D8.auroc - E[c].detectors.D7.auroc))).toFixed(2)} of D7 everywhere.`));

add(h2("C. What Beats the Single Feature?"));
const dd = (c, k) => C[c].paired_vs_D0[k].delta;
add(body(`The dev-trained detector D2 improves on D0 in every cell (paired ΔAUROC ${sg(Math.min(...tq((c) => dd(c, "D2"))))} to ${sg(Math.max(...tq((c) => dd(c, "D2"))))} on TruthfulQA, ${sg(Math.min(...sq((c) => dd(c, "D2"))))} to ${sg(Math.max(...sq((c) => dd(c, "D2"))))} on SciQ; all Holm *p* ≤ ${Math.max(...MS.flatMap((m) => [TQ(m), SQ(m)]).map((c) => C[c].paired_vs_D0.D2.p_holm)).toFixed(3)}) and detects errors for all three models on TruthfulQA (${rng(tq((c) => C[c].detectors.D2.auroc))}). It is not an answer-length proxy: a dev-trained length model scores ${rng(MS.flatMap((m) => [TQ(m), SQ(m)]).map((c) => A[c].length.length_model_dev_trained.auroc))}, D2 adds ${sg(Math.min(...MS.flatMap((m) => [TQ(m), SQ(m)]).map((c) => A[c].length.D2_plus_length_minus_length.delta)))} to ${sg(Math.max(...MS.flatMap((m) => [TQ(m), SQ(m)]).map((c) => A[c].length.D2_plus_length_minus_length.delta)))} to it, and D2's AUROC within answer-length quintiles is within ${Math.max(...MS.flatMap((m) => [TQ(m), SQ(m)]).map((c) => Math.abs(A[c].length.length_quintile_auroc.D2.auroc - C[c].detectors.D2.auroc))).toFixed(2)} of its raw value. The 17-feature D3 is never clearly better than D2. Part of D2's TruthfulQA gain is specific to that dataset: the same three features trained on SciQ dev reach only ${rng(MS.map((m) => E._transfer_D2[`${m}|sciq -> ${m}|truthfulqa`]))} on TruthfulQA confirm.`));
add(body(`More samples do not help. Five-sample self-consistency with cosine matching (D4, D5) is significantly worse than D0 wherever D0 works, and at chance where D0 fails (its one significant gain, D5 on Qwen/TruthfulQA, is over a slightly inverted D0). The standard entailment-based semantic entropy (D7) is better than the cosine version but still only ties D0 on SciQ (paired Δ ${rng(sq((c) => E[c].paired_vs_D0.D7.delta))}, none significant) and is at chance on TruthfulQA except for Llama (${num(E[TQ(MS[1])].detectors.D7.auroc)}, below D0). The model's own true/false judgement D6 is no better: on SciQ it trails or ties D0, and on TruthfulQA it is *inverted* for Qwen and Llama (${num(E[TQ(MS[0])].detectors.D6.auroc)} and ${num(E[TQ(MS[1])].detectors.D6.auroc)}), rating its wrong answers as more likely true than its correct ones. Section IV-E shows that much of this inversion is caused by answer form.`));

add(h2("D. H2: Which Errors Are Missed?"));
add(body(`Table II splits the TruthfulQA errors by type. For Qwen and SmolLM2, errors that repeat a listed misconception are scored no riskier than correct answers (AUROC ${num(gap(MS[0]).auroc_imitative, 2)} and ${num(gap(MS[2]).auroc_imitative, 2)}), while their other errors are scored as riskier (${num(gap(MS[0]).auroc_other, 2)} and ${num(gap(MS[2]).auroc_other, 2)}); both gaps are significant after correction. Llama shows the same direction (${num(gap(MS[1]).delta, 2)}) but the preregistered test does not survive Holm correction (*p* ${P(gap(MS[1]).p_holm)}). The direction had been predicted from the dev split for all three models. Imitative errors are the majority of TruthfulQA errors (${rng(MS.map((m) => cw(TQ(m)).share_imitative_all_wrong).map((x) => 100 * x), 0)}%), which is why D0 fails on the dataset as a whole.`));
add(tcap("II", "D0 AUROC by Error Type, TruthfulQA Confirm (Preregistered)"));
add(table([1180, 860, 860, 1420, 546], ["Model", "Imitative\n(n)", "Other\n(n)", "Δ [95% CI]", "Holm p"],
  MS.map((m) => { const g = gap(m); return [SHORT[m] === "Qwen" ? "Qwen2.5-1.5B" : SHORT[m] === "Llama" ? "Llama-3.2-1B" : "SmolLM2-1.7B",
    `${num(g.auroc_imitative)} (${g.n_imitative})`, `${num(g.auroc_other)} (${g.n_other})`, `${num(g.delta, 2)} ${ci(g.ci)}`, g.p_holm.toFixed(3)]; })));
add(tnote("Each error type is compared against all correct answers of the same model. AUROC < 0.5: the errors are *more* confident than correct answers."));
add(body(`The blind spot is not specific to token probability (Fig. 1). Asked whether its own answer is true (D6), each model again separates imitative errors less well than other errors (${rng(MS.map((m) => et(m, "D6").auroc_imitative))} versus ${rng(MS.map((m) => et(m, "D6").auroc_other))}); entailment-based semantic entropy over five samples (D7) does the same (${rng(MS.map((m) => et(m, "D7").auroc_imitative))} versus ${rng(MS.map((m) => et(m, "D7").auroc_other))}), meaning the model repeats a learned misconception consistently. Because P(True) and entailment-based semantic entropy do not depend on how typical an answer's wording is, the blind spot is not merely that misconceptions are phrased in common, high-probability words. The trained detector D2 is the only signal that recovers part of the imitative errors (${rng(MS.map((m) => et(m, "D2").auroc_imitative))}), and it still separates them less well than other errors.`));
// Table III: examples chosen by the fixed rule in scripts/v3_eval.py::examples; the short answer and
// correction are editorial summaries, and the build fails unless the summary occurs in every model's answer.
const EXSUM = {
  tqa_c_0022: ["one small step for man", "“…for *a* man”"],
  tqa_c_0307: ["Save Our Souls", "not an acronym"],
  tqa_c_0097: ["Vatican", "Nauru"],
  tqa_c_0343: ["United States", "Argentina"],
};
const exRows = E._examples.items.map((e) => {
  const [key, fix] = EXSUM[e.qid];
  MS.forEach((m) => { if (!e.answers[m].includes(key)) throw new Error(`${e.qid}: "${key}" not in ${m} answer`); });
  return [e.question.replace(/\?.*$/, "?"), `“${key}”`, fix, MS.map((m) => Math.round(100 * e.conf_pct[m])).join(" / ")];
});
add(body(`Table III shows what these errors look like. On ${E._examples.n_qualifying} TruthfulQA confirm questions, all three models give a high-confidence imitative error that is among their own 30% most confident answers; the four shown rank highest. Each model states the same misconception in nearly the same words.`));
add(tcap("III", "Confidently Wrong: Imitative Errors Shared by All Three Models"));
add(table([1700, 1100, 1000, 1066], ["Question", "All three say", "Correct", "Conf. pct\nQ / L / S"], exRows, 14));
add(tnote("Conf. pct: percentile of the answer's mean log-probability among the same model's 617 confirm answers (100 = most confident). Selection rule fixed in scripts/v3_eval.py; answers verbatim in analysis_v3/extensions.json."));
add(figure("fig_v3_signals.png", 1, "TruthfulQA confirm: AUROC of four signal families for imitative errors (hollow) and other errors (filled), each against all correct answers of the same model."));

add(h2("E. Is the Blind Spot an Artifact?"));
add(body(`We tried to explain the imitative gap away (Fig. 2). *Length:* imitative errors are on average ${rng(MS.map((m) => A[TQ(m)].h2_controls.mean_answer_length.other - A[TQ(m)].h2_controls.mean_answer_length.imitative), 1)} tokens shorter than other errors, but within answer-length quintiles the gap is ${list3(MS.map((m) => A[TQ(m)].h2_controls.length_quintile.delta), 2)} (Qwen, Llama, SmolLM2), every CI excluding zero, Llama's included. *Question familiarity, difficulty and topic:* in a regression of standardized D0 on error type with answer length, truncation, the question's own log-probability, question length, the number of other models that got the question wrong, and TruthfulQA category fixed effects, imitative errors remain ${list3(MS.map((m) => -A[TQ(m)].h2_controls.regression.diff), 2)} standard deviations more confident than other errors, CIs excluding zero. *Label confidence:* dropping items the judge marked low-confidence or refusal leaves the Qwen and SmolLM2 gaps intact and shrinks Llama's to ${num(C[TQ(MS[1])].sensitivity_high_conf_only.H2_error_type.delta, 2)} (n.s.).`));
add(body(`*Answer form.* Correct TruthfulQA answers are often debunkings ("No, …", "There is no evidence …"): ${rng(MS.map((m) => 100 * E[TQ(m)].form.negated_share_by_type.correct), 0)}% of correct answers contain a negation cue, against ${rng(MS.map((m) => 100 * E[TQ(m)].form.negated_share_by_type.imitative), 0)}% of imitative errors. This explains P(True)'s inversion: asked to judge a negated statement, these models answer "False" regardless of its truth (D6 AUROC within negated answers ${rng(MS.map((m) => E[TQ(m)].form.negated_auroc.D6))}). Restricted to affirmative answers, D6 is at most ${num(Math.max(...MS.map((m) => E[TQ(m)].form.affirmative.detectors.D6.auroc)))} overall, but the imitative gap remains for D0 (${list3(MS.map((m) => aff(m, "D0").delta), 2)}) and for D6 (${list3(MS.map((m) => aff(m, "D6").delta), 2)}), with CIs excluding zero for Qwen and SmolLM2 but not for Llama.`));
add(body(`*Error typing.* The automatic NLI typing agrees with the judge's error types on ${rng(MS.map((m) => 100 * ih(m).type_agreement_on_claude_errors.agree), 0)}% of errors (κ ${rng(MS.map((m) => ih(m).type_agreement_on_claude_errors.kappa))}). With NLI types the gap is ${list3(MS.map((m) => ih(m).claude_labels_nli_types.delta), 2)}, Holm *p* ≤ ${Math.max(...MS.map((m) => ih(m).claude_labels_nli_types.p_holm)).toFixed(3)} in all three models, Llama included.`));
const tjD0 = (m) => tjl(m).truthjudge_labels.detectors.D0;
const dis = (m, k) => K.judge_disagreement[m][k];
const WQ = (m) => E._within_tqa[m];
const p2 = (m) => X.prompt2[m];
add(body(`*Independent labels.* The truth-judge agrees with the primary judge on ${rng(MS.map((m) => 100 * tjl(m).agreement_confirm.agree), 0)}% of TruthfulQA confirm answers (κ ${list3(MS.map((m) => tjl(m).agreement_confirm.kappa), 2)}) and marks fewer answers hallucinated (${list3(MS.map((m) => tjl(m).agreement_confirm.rate_truthjudge), 2)} versus ${list3(MS.map((m) => tjl(m).agreement_confirm.rate_claude), 2)}). The disagreements are mostly answers the primary judge flags and the truth-judge accepts; they are longer and less confident than other answers (Llama: ${Math.round(dis(MS[1], "10").answer_tokens)} versus ${Math.round(dis(MS[1], "11").answer_tokens)} tokens), about half carry the primary judge's own low-confidence flag, and many state a plausible main claim with a false secondary one, which the rubric counts as hallucinated and which the truth-judge, trained on short answers, tends to accept. Under truth-judge labels D0 on TruthfulQA is ${list3(MS.map((m) => tjD0(m).auroc))}: Llama's preregistered detection does not survive (on the ${tjl(MS[1]).consensus_items.n} answers both judges agree on it is ${num(tjl(MS[1]).consensus_items.detectors.D0.auroc)}), and for Qwen and SmolLM2 D0 is significantly *inverted*: their most confident answers are the most often wrong. With truth-judge labels and NLI types, i.e. with no input from the primary judge, the imitative gap is ${list3(MS.map((m) => ih(m).truthjudge_labels_nli_types.delta), 2)}: it replicates for Qwen and SmolLM2 (Holm *p* ${P(Math.max(ih(MS[0]).truthjudge_labels_nli_types.p_holm, ih(MS[2]).truthjudge_labels_nli_types.p_holm))}) and vanishes for Llama, whose other errors are not detectable under these labels either.`));
add(body(`*Holding the question fixed.* Every comparison so far is across questions, so imitative errors could simply occur on questions that every answer is confident about. Among the five sampled answers to one TruthfulQA question (truth-judge labels, NLI types), imitative wrong samples are scored as *less* risky than correct samples of the same question for Llama (within-question AUROC ${num(WQ(MS[1]).imitative.within_auroc, 2)}, *p* ${P(WQ(MS[1]).imitative["p_two_sided_vs_0.5"])}) and SmolLM2 (${num(WQ(MS[2]).imitative.within_auroc, 2)}, *p* ${P(WQ(MS[2]).imitative["p_two_sided_vs_0.5"])}), and no differently for Qwen (${num(WQ(MS[0]).imitative.within_auroc, 2)}): when a model wavers between a misconception and the truth, the misconception tends to be its more probable answer. Other wrong samples are not reliably less confident than correct samples of the same question either (${rng(MS.map((m) => WQ(m).other.within_auroc))}, none significant), so most of the partial detectability of other errors seen across questions is a question-level effect, as on SciQ (Section IV-F).`));
add(body(`*Prompt.* Our prompt asks for a best specific answer when unsure, which might manufacture confident errors. Under a neutral prompt that allows "I don't know", most answers change (only ${rng(MS.map((m) => 100 * K.prompt2[m].identical_share), 0)}% are identical) and truth-judge hallucination rates fall to ${list3(MS.map((m) => p2(m).base_rate), 2)}, but the pattern does not change: D0 is ${list3(MS.map((m) => p2(m).D0.auroc))} and the imitative gap ${list3(MS.map((m) => p2(m).H2.delta), 2)}, significant for Qwen and SmolLM2 (independent labels and types).`));
const SC = X.scale["qwen2.5-3b-instruct"], SB = X.base[MS[0]];
add(body(`*Scale.* Qwen2.5-3B-Instruct (bf16, otherwise the same protocol; ${(100 * J("results_v3/scale/qwen2.5-3b-instruct/truthfulqa/confirm/meta.json").argmax_agreement).toFixed(1)}% of tokens match the re-scored argmax, against 100% in fp32) makes fewer errors than its 1.5B sibling under the truth-judge (${num(SC.base_rate, 2)} versus ${num(SB.base_rate, 2)}), but its uncertainty is no better: D0 is again significantly inverted (${num(SC.D0.auroc)} ${ci(SC.D0.ci)}) and imitative errors are again scored like correct answers while other errors are not (${num(SC.H2.auroc_imitative, 2)} versus ${num(SC.H2.auroc_other, 2)}, gap ${num(SC.H2.delta, 2)} ${ci(SC.H2.ci)}, independent labels and types). Doubling the size removed errors, not the blind spot.`));
add(body(`Pooled over the three models with a bootstrap that resamples questions jointly, the preregistered gap is ${num(E._pooled_h2.mean_delta, 2)} ${ci(E._pooled_h2.ci)}. For Qwen and SmolLM2 it appears in every analysis in Fig. 2 except Qwen's within-question test, where neither error type is less confident than correct answers; for Llama it depends on the labels.`));
const SCALE_ABSTRACT = ", and it persists in a model twice the size";
add(figure("fig_v3_robust.png", 2, "Imitative-minus-other AUROC gap of D0 on TruthfulQA confirm (95% CI) under the preregistered analysis and each post-hoc control; \"independent\" rows use truth-judge labels and NLI error types only. The within-question row compares sampled answers to the same question. Negative: imitative errors are harder to detect."));

add(h2("F. H5: Answer or Question?"));
const H5 = (d) => C._H5_own_vs_other[d];
const ownOther = (d) => Object.entries(H5(d)).filter(([k]) => !k.includes("others_wrong"));
const wq = (lab) => MS.map((m) => W[m][lab].mean_logprob.within_question_auroc);
add(body(`If D0 reflected the specific answer, a model's own D0 should predict its errors better than another model's D0 on the same questions. On SciQ it barely does (Fig. 3): own minus other is ${sg(Math.min(...ownOther("sciq").map(([, v]) => v.delta)))} to ${sg(Math.max(...ownOther("sciq").map(([, v]) => v.delta)))}, and ${ownOther("sciq").filter(([, v]) => v.p_holm < 0.05).length} of six pairs is significant after correction. A difficulty proxy that uses no uncertainty, the number of the other two models that answered wrongly, reaches ${rng(MS.map((m) => H5("sciq")[`${m}<-others_wrong_count`].auroc))} on SciQ and ${rng(MS.map((m) => H5("truthfulqa")[`${m}<-others_wrong_count`].auroc))} on TruthfulQA, above every uncertainty detector. It needs other models' labels, so it is a diagnostic rather than a detector, but it bounds how much of the measured signal is the question. The surface form of the question is not what drives it: the question's own log-probability scores ${rng(MS.flatMap((m) => [TQ(m), SQ(m)]).map((c) => C[c].controls.question_mean_logprob.auroc))}.`));
add(body(`There is nevertheless an answer-specific component. Among the five sampled answers to the same SciQ question, wrong samples have lower mean log-probability than right ones: within-question AUROC ${list3(wq("containment"))} with the containment label (preregistered), and ${rng([...wq("hybrid_other_anchors"), ...wq("agreed_only")])} under two corrected sample labels that repair containment's rejection of correct paraphrases (every sign-flip *p* ≤ ${Math.max(...MS.flatMap((m) => ["containment", "hybrid_other_anchors", "agreed_only"].map((l) => W[m][l].mean_logprob.p_one_sided))).toFixed(3)}). A third correction, which lets a sample inherit the label of its own model's greedy answer, gives higher values (${rng(wq("hybrid"))}) but is biased upward by construction, because samples that match the greedy answer are high-probability; we do not use it. D0 also adds ${rng(sq((c) => C[c].controls.D0_added_to_controls.delta))} AUROC to question and length controls on SciQ. Llama on TruthfulQA is the one case where the model's own signal clearly exceeds another's (${num(d0(TQ(MS[1])).auroc)} versus ${rng(ownOther("truthfulqa").filter(([k]) => k.startsWith(MS[1])).map(([, v]) => v.other_auroc))}, Holm *p* ${P(Math.max(...ownOther("truthfulqa").filter(([k]) => k.startsWith(MS[1])).map(([, v]) => v.p_holm)))}).`));
add(figure("fig_v2_difficulty.png", 3, "Predicting each model's confirm errors: its own D0 (filled), other models' D0 on the same questions (hollow), and the number of other models that answered wrongly (diamond)."));

add(h2("G. What Would a Detector Buy in Practice?"));
const top = (c) => cw(c).top20pct_confident;
const r80 = (c, k) => cw(c)[`risk_coverage_${k}`]["risk@0.8"];
add(body(`Fig. 4 shows selective prediction with D0: answers are kept from most to least confident and the error rate of the kept answers is tracked. On SciQ, abstaining on the 20% least confident answers lowers the error rate from ${list3(sq((c) => C[c].base_rate), 2)} to ${list3(sq((c) => r80(c, "D0")), 2)}. On TruthfulQA it does almost nothing for Qwen (${num(C[TQ(MS[0])].base_rate)} to ${num(r80(TQ(MS[0]), "D0"))}) and SmolLM2 (${num(C[TQ(MS[2])].base_rate)} to ${num(r80(TQ(MS[2]), "D0"))}): their most confident fifth of answers is wrong ${pct(top(TQ(MS[0])).error_rate)} and ${pct(top(TQ(MS[2])).error_rate)} of the time, and ${pct(top(TQ(MS[0])).share_imitative_of_errors)} and ${pct(top(TQ(MS[2])).share_imitative_of_errors)} of those errors are imitative. The dev-trained D2 does better (${list3(tq((c) => r80(c, "D2")), 2)} at 80% coverage on TruthfulQA) and is well calibrated (expected calibration error ${rng(MS.flatMap((m) => [TQ(m), SQ(m)]).map((c) => E[c].operating.D2_calibration.ece_10bin))}), but it improves the Brier score over always predicting the base rate by only ${rng(MS.flatMap((m) => [TQ(m), SQ(m)]).map((c) => E[c].operating.D2_calibration.brier_base_rate - E[c].operating.D2_calibration.brier), 3)}, and at a threshold fixed on dev to give a 20% false-positive rate it flags ${rng(tq((c) => 100 * E[c].operating["D2@dev_fpr0.2"].flagged), 0)}% of TruthfulQA answers with precision ${rng(tq((c) => E[c].operating["D2@dev_fpr0.2"].precision))} and recall ${rng(tq((c) => E[c].operating["D2@dev_fpr0.2"].recall))}.`));
add(figure("fig_v3_coverage.png", 4, "Selective prediction with D0 on the confirm split: error rate of the kept answers as the least confident are withheld (coverage 1.0 = keep all; marker = base rate)."));

// ---------------------------------------------------------------- V. Discussion
add(h1("V. Discussion"));
add(body(`*Why the two datasets differ.* SciQ errors are mostly gaps in knowledge: only ${rng(MS.map((m) => 100 * cw(SQ(m)).share_imitative_all_wrong), 0)}% of them pick a listed distractor, the model does not know the term, and its uncertainty shows it, mainly at the level of how hard the question is. TruthfulQA errors are mostly learned falsehoods. The detectability of a dataset is then largely the mix of its error types, which is why the same feature is at chance for two models and weak for the third.`));
add(body("*Why every signal shares the blind spot.* Token probability measures how expected an answer is under the model, P(True) asks the same model to rate it, and sampling consistency measures how much of the model's probability mass agrees with it. A misconception the model learned is its expected, endorsed and consistent answer, so all three read it as knowledge. This is what one would predict if hallucinations are partly learned statistics of the training data rather than noise [@kalai2025], and it limits any detector that consults only the generating model. The only signal that recovered part of the imitative errors, the trained D2, did so through a dataset-specific contrast: with minimum log-probability and entropy held fixed, a *higher* mean log-probability raises its predicted risk, which flags unusually fluent answers on TruthfulQA and transfers poorly from SciQ."));
add(body(`*Why the models differ.* Llama's imitative errors are less confident than Qwen's and SmolLM2's across questions (D0 ${num(gap(MS[1]).auroc_imitative, 2)} versus ${num(gap(MS[0]).auroc_imitative, 2)} and ${num(gap(MS[2]).auroc_imitative, 2)}), yet within a question it prefers the misconception most clearly of the three. It writes the longest answers and truncates a third of them, and the two judges disagree most on long answers, so Llama's across-question signal mixes uncertainty with how long, multi-claim answers are labelled. Three models cannot separate training data, alignment and size as causes; the observation is reported, not explained.`));
add(body(`*What changed from our earlier study.* The original Qwen result (AUROC ${K.da1_original.auroc.toFixed(2)} on a ${K.da1_original.n}-question holdout) was produced by contaminated scores and proxy labels; clean scores with the same three features reach ${num(C[TQ(MS[0])].detectors.D2.auroc, 2)} on 617 held-out questions. The SciQ signal, earlier called strong, replicates but is weak and largely question-level, and Llama's TruthfulQA detection, the one positive TruthfulQA result of the preregistered test, depends on how long answers are labelled. The claim that roughly half of TruthfulQA errors are confident holds for two models, not for Llama. That self-consistency adds nothing held for the cheap cosine version; the standard entailment-based version ties the single-pass score on SciQ instead of losing to it.`));

// ---------------------------------------------------------------- VI. Threats
add(h1("VI. Threats to Validity"));
add(bullet("*Labels.* The primary judge is an LLM that knew the hypotheses, though not the model identity or any score. The independent truth-judge, trained on human ratings, and the NLI error typing remove that dependence, but no human annotated our answers."));
add(bullet("*Error types* depend on TruthfulQA's listed incorrect answers: an answer repeating an unlisted misconception counts as *other*, which would shrink, not inflate, the measured gap."));
add(bullet("*Post-hoc analyses* were chosen after the confirm results were known, specifically to find artifacts. We report every one we ran, including those that weaken a claim, and none is promoted to a confirmatory result."));
add(bullet("*Generality.* Three 1–2B models plus one 3B check (bf16, TruthfulQA only), short-answer English QA, greedy decoding with a 64-token limit, and two prompts; larger models, long-form generation and other domains may behave differently. The dev split was examined extensively, so dev numbers are exploratory."));
add(bullet("*Precision.* With 617–800 questions per cell, AUROC CIs are about ±0.04, so differences below about 0.05 between detectors are not resolved. Holm correction is applied within each family (the six cells, or the three TruthfulQA cells), not across families."));

// ---------------------------------------------------------------- VII. Conclusion
add(h1("VII. Conclusion"));
add(body("Single-pass token uncertainty is a weak hallucination signal for small language models. On science questions it works modestly, mostly by tracking how hard the question is; on TruthfulQA it fails for two of three models because most errors there are misconceptions that the model produces with full confidence. That blind spot is shared by the model's own true/false judgement and by entailment-based semantic entropy; for two of the three models it survived every control we could devise, including independent labels and a second prompt, and for the third its size depends on how long answers are labelled. Detectors that read only the generating model measure what it has learned, not whether it is true; catching confident, learned falsehoods needs evidence from outside the model, such as retrieval or an independent verifier."));

add(h1("Reproducibility"));
add(body("Code, frozen generations, labels, analysis outputs and the scripts that build every table, figure and number in this paper are in the project repository listed below. The preregistration is the commit that adds PREREGISTRATION.md; confirmatory results are produced by scripts/v2_confirm.py and the post-hoc audit by scripts/v3_*.py. Models are the public checkpoints named above; all randomness uses seed 42."));
add(new Paragraph({ alignment: AlignmentType.LEFT, spacing: { after: 100 }, children: runs("Repository: https://github.com/gouthamkrishna404/hallucination-detection", { size: 18 }) }));

// ---------------------------------------------------------------- References
add(h1("References"));
[...CITED].forEach((k, i) => add(new Paragraph({
  alignment: AlignmentType.JUSTIFIED, indent: { left: 400, hanging: 400 }, spacing: { after: 40 },
  tabStops: [{ type: TabStopType.LEFT, position: 400 }],
  children: [new TextRun({ text: `[${i + 1}]\t`, size: 16 }), ...runs(REFS[k], { size: 16 })],
})));
const unused = Object.keys(REFS).filter((k) => !CITED.includes(k));
if (unused.length) console.warn("uncited references:", unused.join(", "));

// ---------------------------------------------------------------- front matter + document
const TITLE = "Learned Falsehoods Look Like Knowledge: A Preregistered Test of Uncertainty-Based Hallucination Detection in Small Language Models";
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
const abstractPara = (label, text) => new Paragraph({
  alignment: AlignmentType.JUSTIFIED, indent: { firstLine: 272 }, spacing: { after: 160 },
  children: [new TextRun({ text: `${label}—`, bold: true, italics: true, size: 18 }),
    new TextRun({ text, bold: true, italics: true, size: 18 })],
});
const KEYWORDS = "Hallucination detection, language models, uncertainty estimation, token log-probability, semantic entropy, imitative falsehoods, preregistration, TruthfulQA, SciQ";
const ABSTRACT = `Token-level uncertainty from a single generation is the cheapest hallucination signal available. We test whether it detects the wrong answers of three small instruction-tuned models (Qwen2.5-1.5B, Llama-3.2-1B, SmolLM2-1.7B) on TruthfulQA and SciQ, with raw greedy decoding, fp32 teacher-forced scoring, and an evaluation preregistered on a development split and run once on 1,417 held-out questions. Mean token log-probability detects errors weakly on SciQ (AUROC ${rng(MS.map((m) => d0(SQ(m)).auroc))}), largely through question difficulty, and is at chance on TruthfulQA for two models. The failure is specific: answers that repeat a popular misconception are scored like correct answers, while other errors are partly detectable (preregistered gap ${num(gap(MS[0]).delta, 2)} and ${num(gap(MS[2]).delta, 2)} for two models; ${num(gap(MS[1]).delta, 2)}, not significant, for the third). Post hoc, the same blind spot appears in the model's own true/false judgement and in entailment-based semantic entropy. For two models the gap survives controls for length, question familiarity, difficulty, topic and answer form, labels from an independent judge trained on human ratings, automatic error typing and a second prompt${SCALE_ABSTRACT}, and within a question the misconception is often the model's more probable answer. Abstaining on the least confident fifth of answers barely changes the TruthfulQA error rate for two models. Uncertainty read from the generating model measures what it has learned, not whether it is true.`;

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
        children: [new TextRun({ text: TITLE.toUpperCase(), bold: true, italics: true, size: 34 })] }),
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
Packer.toBuffer(doc).then((buf) => { fs.writeFileSync(OUT, buf); console.log("Wrote", OUT); });
