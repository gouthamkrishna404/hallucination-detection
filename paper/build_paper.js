// Builds paper/Token_Uncertainty_Hallucination_Detection.docx (IEEE-style, two-column, A4).
// Layout follows the ParkPulse reference: A4, Times New Roman, 0.63" side margins,
// 0.25" column gap, small-caps section heads, 8 pt captions, three-rule tables.
// Every number is taken from analysis_v2/confirm.json and analysis_v2/dev.json (RESULTS.md Part A),
// produced by scripts/v2_confirm.py and scripts/v2_analysis.py.
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
const TITLE = "Confidently Wrong: A Preregistered Re-test of Token-Level Uncertainty as a Hallucination Detector in Small Language Models";
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

const ABSTRACT = "Token-level uncertainty from a single generation is the cheapest available hallucination signal, and it is often assumed to work. We tested that assumption, and tried to break it, for three small instruction-tuned models (Qwen2.5-1.5B, Llama-3.2-1B, SmolLM2-1.7B) on TruthfulQA and SciQ. An audit of our earlier study found that one model's scores were distorted by a default decoding-time repetition penalty and that our automatic labels were unreliable, so we rebuilt the protocol: raw greedy decoding, fp32 teacher-forced scoring, and blind labels from a rubric-based LLM judge with measured re-judging consistency (κ 0.97 and 0.78). All modelling choices were made on the original 200-question subsets; detectors, hypotheses and tests were then preregistered and evaluated once on 617 TruthfulQA and 800 SciQ questions never used before. Mean token log-probability detects errors on SciQ for all three models, but only weakly (AUROC 0.66–0.70), and much of that signal is question difficulty: another model's uncertainty predicts a model's errors almost as well as its own, and a difficulty proxy that uses no uncertainty performs better (0.77–0.82). On TruthfulQA it is at chance for two of three models. The errors it misses are specific: for two of the three models, falsehoods that repeat a popular misconception are produced as confidently as correct answers (AUROC 0.42–0.45), while other errors are partly detectable (0.58–0.63). Small logistic-regression detectors trained on the development split transfer to the held-out questions (0.60–0.73); five-sample self-consistency never beats the single-pass score. Token uncertainty is a weak, largely question-level signal that is blind to confidently learned misconceptions.";
const KEYWORDS = "Hallucination detection, large language models, uncertainty estimation, token log-probability, imitative falsehoods, preregistration, TruthfulQA, SciQ";

// ---------- body ----------
const content = [];
const add = (...xs) => xs.flat().forEach((x) => content.push(x));

add(h1("I. Introduction"));
add(body("Language models produce fluent answers that are sometimes false. If every answer could be flagged cheaply when it is likely wrong, unreliable outputs could be withheld or sent to a stronger verifier. Sampling-based detectors such as SelfCheckGPT [2], self-consistency [3] and semantic entropy [4] compare several generations, which multiplies inference cost. The cheapest alternative uses what the model already exposes while producing one answer: the probability of each generated token and the entropy of each next-token distribution [5], [14]."));
add(body("Confidence, however, is not truth. A model can assign high probability to a claim it saw asserted many times during training even when the claim is false, and TruthfulQA is built around exactly such imitative falsehoods [1]. Whether single-pass uncertainty detects hallucination should therefore depend on what kind of error is made, and on whether the uncertainty describes the answer or merely the question."));
add(body("This paper is a re-test. Our earlier study of the same question reported that the signal worked on SciQ and mostly failed on TruthfulQA. An audit then found that one model's scores were contaminated by a decoding-time repetition penalty, that our automatic labels disagreed with each other on up to a third of answers, and that every test reused the same 200 questions. Rather than patch those results, we rebuilt the pipeline, made every modelling decision on the original questions, preregistered the detectors and hypotheses, and evaluated them once on 1,417 new questions. We actively looked for reasons the signal might be an artifact. Our contributions are:"));
add(item("A clean, preregistered held-out evaluation of single-pass token uncertainty for three small open models on two datasets, with every negative result reported."));
add(item("Evidence that the errors uncertainty misses are imitative ones: for two of three models, misconception-repeating answers are as confident as correct answers, while other errors are partly detectable."));
add(item("A falsification test showing that most of the SciQ signal is question difficulty shared across models, with a smaller answer-specific component measured within questions."));
add(item("Two documented pitfalls that each changed earlier conclusions: generation-API scores that silently include logits processors, and proxy labels that misjudge paraphrased answers."));

add(h1("II. Related Work"));
add(body("TruthfulQA [1] measures whether models reproduce human misconceptions; SciQ [12] contains science examination questions with one correct answer and three distractors. Larger models have been reported to be reasonably calibrated about their own answers [14], which motivates using token probabilities as a confidence signal. FActScore [6] instead evaluates long-form factual precision against a knowledge source, an evaluation tool rather than a low-cost online detector."));
add(body("Sampling-based methods assume that a model which knows a fact answers consistently: SelfCheckGPT [2] compares a response with further samples, self-consistency [3] aggregates sampled answers, and semantic entropy [4], [15] measures uncertainty over meaning-level clusters of samples. Single-sample and white-box methods avoid repeated generation: claim-conditioned probability refines token uncertainty for fact-checking [5], MIND [7] and HaloScope [8] use internal states, factuality probes report competitive accuracy from one sample [10], and InterrogateLLM [9] re-queries the model. A broad study of uncertainty-estimation methods [11] warns that results depend strongly on evaluation setup. We do not propose a new signal; we measure how far the simplest ones go on small models, and why they fail."));
add(body("Our labels come from an LLM judge, as TruthfulQA's own evaluation uses a fine-tuned GPT judge [1]; LLM judges are known to be useful but imperfect [16], so we report judge consistency and treat labels as a limitation."));

add(h1("III. Method"));
add(h2("A. Models, Data and Splits"));
add(body("We used Qwen2.5-1.5B-Instruct, Llama-3.2-1B-Instruct and SmolLM2-1.7B-Instruct with one shared chat prompt asking for a direct one- or two-sentence answer. The *dev* split is the original 200-question subset of each dataset; every exploratory analysis and modelling decision was made there. The *confirm* split is all remaining TruthfulQA questions (617) and the SciQ test set (800), disjoint from dev by question text and evaluated only after the preregistration was committed."));
add(h2("B. Clean Decoding and Scoring"));
add(body("Our earlier pipeline read scores from the generation interface. Qwen2.5 ships a repetition penalty of 1.1 that HuggingFace applies as a logits processor even under greedy decoding, so those scores described a penalty-adjusted distribution; the shift in mean log-probability (0.125) was about eight times the separation between correct and incorrect answers. In v2, answers are generated greedily with every logits processor disabled, in fp32, up to 64 new tokens, and frozen. Each answer is then re-scored by one fp32 teacher-forced pass, so the scored distribution for token *y*_{t} is exactly the model's conditional"));
add(eq(1, [sub("ℓ", "t"), M(" = log "), sub("p", "θ"), M("("), sub("y", "t"), M(" ∣ x, "), sub("y", "<t"), M(")")]));
add(body("where *x* is the prompt. In every cell 100% of generated tokens equal the argmax of the re-scored distribution, which verifies both alignment and the absence of decoding modifications. From the same pass we keep the next-token entropy *H*_{t} = −Σ_{v} *p*_{t}(*v*) log *p*_{t}(*v*) and the top-1/top-2 margin. We also fixed two prompt defects in the Llama chat template (an injected current date and a duplicated BOS token). For self-consistency we draw *K* = 5 samples per question at temperature 0.7 and top-*p* 0.9, scored the same way (SmolLM2's samples were generated in bf16 to fit an 8 GB GPU; its greedy answers and all scoring are fp32)."));
add(h2("C. Labels"));
add(body("Every greedy answer was labelled by an LLM judge (Claude) following a fixed written rubric: an answer is hallucinated if it asserts something false that bears on the question; for SciQ, if its final answer is not the correct one. References are evidence, not the definition: where a reference is plainly wrong the judge applies established fact and marks the item low-confidence. The judge also records an error type: *imitative* if the answer repeats a listed misconception (TruthfulQA) or distractor (SciQ), *other* otherwise. Items were deduplicated across models, shuffled, and shown without model identity, features or probabilities. A blind re-judge of 100 dev items agreed on 98.3% of TruthfulQA (κ = 0.97) and 92.5% of SciQ items (κ = 0.78); all disagreements were on items marked low-confidence both times. The judge exposed a flaw in our earlier SciQ label, which checked whether the answer contained the reference string: it never accepted a wrong answer, but rejected 23% of correct paraphrases."));
add(h2("D. Detectors"));
add(body("All detectors output a risk score where larger means more likely hallucinated. D0 is the negative mean token log-probability,"));
add(eq(2, [M("D0 = −"), new MathFraction({ numerator: [M("1")], denominator: [M("T")] }),
  new MathSum({ children: [sub("ℓ", "t")], subScript: [M("t=1")], superScript: [M("T")] })]));
add(body("and D1 the negative minimum log-probability. D2 and D3 are logistic regressions *σ*(**w**^{⊤}**z** + *b*) on standardized features: D2 uses the original three (mean and minimum log-probability, mean entropy) and D3 all 17 token features (location, tail, dispersion, entropy, margin, positional and length statistics). Both are trained on the dev split of the same model and dataset and then frozen. D4 is the fraction of the five samples that match the greedy answer (cosine similarity ≥ 0.75 between sentence embeddings [13]), negated; D5 is semantic entropy over greedy cosine clusters of the samples, an approximation of the entailment-based original [4]."));
add(h2("E. Preregistered Tests"));
add(body("The preregistration fixed, before any confirm label was compared with a score: H1 (primary), D0 has AUROC ≠ 0.5 in each cell; H2, on TruthfulQA the AUROC of D0 for imitative errors versus correct answers differs from that for other errors; H3–H4, paired AUROC differences of D1–D5 versus D0 and their individual AUROCs; H5, a model's own D0 predicts its errors better than another model's D0 on the same questions. Fixed scores are tested by label permutation (10,000 permutations), differences by paired or stratified bootstrap (2,000 resamples), all two-sided, with Holm correction [17] across the six cells within each family. The preregistration also fixed interpretation: AUROC below 0.60 is called negligible, 0.60–0.70 weak and 0.70–0.80 moderate, and confirm results are described as in-distribution held-out results, not as generalization."));

add(h1("IV. Results"));
add(h2("A. Base Rates"));
add(body("Judge-labelled hallucination rates on the confirm split are 0.57, 0.53 and 0.62 on TruthfulQA and 0.28, 0.38 and 0.23 on SciQ for Qwen, Llama and SmolLM2. Llama reaches the 64-token limit on a third of its TruthfulQA answers; its results below are unchanged on non-truncated answers (D0 0.646, 95% CI 0.590–0.700)."));
add(h2("B. H1: Does Mean Log-Probability Detect Errors?"));
add(body("Table I and Fig. 1 give every detector on the confirm split. D0 shows detection in four of six cells: all three models on SciQ (0.66–0.70) and Llama on TruthfulQA (0.653). For Qwen (0.466) and SmolLM2 (0.500) on TruthfulQA it is at chance. Every detected effect is weak by the preregistered scale."));
add(tcap("I", "AUROC on the Confirm Split (617 TruthfulQA / 800 SciQ Questions)"));
add(table([1530, 556, 556, 556, 556, 556, 556], ["Detector", "TQA\nQwen", "TQA\nLlama", "TQA\nSmol", "SciQ\nQwen", "SciQ\nLlama", "SciQ\nSmol"], [
  ["D0 −mean log-prob", "0.466", "**0.653**", "0.500", "**0.688**", "**0.658**", "**0.702**"],
  ["D1 −min log-prob", "0.539", "**0.713**", "**0.560**", "**0.708**", "**0.671**", "**0.706**"],
  ["D2 3-feature LR", "**0.600**", "**0.726**", "**0.611**", "**0.728**", "**0.685**", "**0.727**"],
  ["D3 17-feature LR", "**0.626**", "**0.706**", "**0.597**", "**0.688**", "**0.638**", "**0.719**"],
  ["D4 self-consist. (5)", "0.517", "0.511", "0.502", "**0.566**", "**0.603**", "**0.553**"],
  ["D5 sem. entropy (5)", "0.536", "0.496", "0.519", "**0.560**", "**0.597**", "**0.538**"],
]));
add(tnote("Bold: Holm-adjusted two-sided permutation *p* < 0.05 and 95% bootstrap CI excluding 0.5. D2 and D3 are trained on the 200-question dev split and frozen; CIs are about ±0.04."));
add(figure("fig_v2_auroc.png", 1050, 900, 1, "Confirm-split AUROC with 95% CI for the single-feature score D0 (filled) and the dev-trained 3-feature detector D2 (hollow). Dashed line: chance."));
add(h2("C. H3–H4: What Beats the Single Feature?"));
add(body("The dev-trained three-feature detector D2 improves on D0 in every cell (paired ΔAUROC +0.07 to +0.13 on TruthfulQA, +0.03 to +0.04 on SciQ; all Holm *p* < 0.05) and detects errors for all three models on TruthfulQA (0.60–0.73). The 17-feature D3 adds nothing over D2 on SciQ. Both transfer without loss: frozen D3 on confirm (0.597–0.719) matches D3 refit out-of-fold within confirm (0.587–0.717). The gain is a learned contrast rather than monotone confidence: in every cell the frozen weights are positive on mean log-probability and entropy and negative on minimum log-probability, so given the entropy, a *higher* mean log-probability raises the predicted risk. We do not have a mechanistic explanation for this contrast. Five-sample self-consistency and semantic entropy are never better than D0 and are significantly worse wherever D0 works (−0.06 to −0.16)."));
add(h2("D. H2: Which Errors Does Uncertainty Miss?"));
add(body("Table II splits the TruthfulQA errors by type. For Qwen and SmolLM2, errors that repeat a listed misconception are scored no riskier than correct answers (AUROC 0.42 and 0.45), while their other errors are scored as riskier (0.58 and 0.63); the gaps (−0.16 and −0.18) are significant after correction. Llama shows the same direction (−0.09) but the gap does not survive correction. The pattern was predicted from the dev split for all three models (Fig. 2). As a consequence, 59% (Qwen) and 54% (SmolLM2) of wrong TruthfulQA answers are at least as confident as the median correct answer, against 28% for Llama and 22–29% on SciQ. On SciQ, where imitative errors are the 33–43 answers that give a listed distractor, the gap is −0.07 to +0.00 and not significant."));
add(tcap("II", "D0 AUROC by Error Type, TruthfulQA Confirm"));
add(table([1350, 900, 900, 1000, 716], ["Model", "Imitative\n(n)", "Other\n(n)", "Δ [95% CI]", "Holm p"], [
  ["Qwen2.5-1.5B", "0.423 (255)", "0.580 (97)", "−0.16 [−0.24, −0.08]", "0.005"],
  ["Llama-3.2-1B", "0.624 (217)", "0.710 (110)", "−0.09 [−0.16, −0.01]", "0.096"],
  ["SmolLM2-1.7B", "0.451 (277)", "0.627 (106)", "−0.18 [−0.26, −0.09]", "0.003"],
]));
add(tnote("Each error type is compared against all correct answers of the same model. Imitative: the judge found the answer repeats a listed misconception."));
add(figure("fig_v2_error_type.png", 1050, 810, 2, "D0 AUROC for imitative (hollow) and other (filled) TruthfulQA errors, on the dev split where the hypothesis was formed and on the held-out confirm split where it was tested."));
add(h2("E. H5: Answer or Question?"));
add(body("If D0 reflected the specific answer, a model's own D0 should predict its errors better than another model's D0 on the same questions. On SciQ it barely does (Fig. 3): own minus other is +0.02 to +0.06, and only one of six pairs is significant after correction. A difficulty proxy that uses no uncertainty, the number of the other two models that answered wrongly, reaches 0.77–0.82 on SciQ and 0.68–0.73 on TruthfulQA, above every uncertainty detector. It requires other models' labels, so it is a diagnostic rather than a usable detector, but it shows how much of the measured signal is the question. The surface form of the question is not what drives it: the question's own log-probability is near chance (0.50–0.59) and a TF-IDF model of the question text scored 0.42–0.57 on dev."));
add(body("There is nevertheless an answer-specific component. Among the sampled answers to the same SciQ question, wrong samples have lower mean log-probability (within-question AUROC 0.567, 0.558 and 0.621 over 231, 308 and 245 questions with mixed outcomes; sign-flip *p* ≤ 0.005; sample labels from the automatic containment check, which never accepted a wrong answer in the judge audit), and D0 adds 0.08–0.11 AUROC to question and length controls. Llama on TruthfulQA is the one case where the model's own signal clearly exceeds another's (0.653 versus 0.50–0.52, Holm *p* = 0.003)."));
add(figure("fig_v2_difficulty.png", 1050, 810, 3, "Predicting each model's confirm errors: its own D0 (filled), other models' D0 on the same questions (hollow), and the number of other models that answered wrongly (diamond)."));
add(h2("F. Robustness"));
add(body("Dropping the items the judge marked low-confidence or refusal (a quarter to a third of TruthfulQA items) leaves every conclusion unchanged except that Llama's error-type gap shrinks to −0.04 (not significant). D0 is not a length proxy: answer length alone scores 0.48–0.57. Llama's TruthfulQA result holds on answers that were not truncated (Section IV-A)."));

add(h1("V. Discussion"));
add(body("*Why TruthfulQA and SciQ differ.* SciQ errors are mostly gaps in knowledge: the model does not know the term and its uncertainty shows it, although mainly at the level of how hard the question is. TruthfulQA errors are mostly learned falsehoods. A model that has absorbed a misconception produces it fluently, so its token probabilities are high for the wrong reason. Measured detectability then depends on the mix of error types, which explains why the same feature is at chance for two models and weak for a third."));
add(body("*Why models differ.* Llama's imitative errors are less confident than Qwen's and SmolLM2's (0.62 versus 0.42–0.45), and its own uncertainty carries information other models' does not. We cannot say from three models whether this reflects training data, alignment, or size; the observation is reported, not explained."));
add(body("*What changed from the earlier study.* Three earlier conclusions did not survive. The original Qwen result (AUROC 0.48 on a 60-question holdout) was produced by contaminated scores and proxy labels; with clean scores the same three-feature detector reaches 0.60 on 617 held-out questions. The SciQ signal, earlier described as strong, replicates but is weak and largely question-level. The claim that roughly half of TruthfulQA errors are confident holds for two models but not for Llama. The earlier conclusion that self-consistency adds nothing held up."));
add(body("*What can be claimed.* Within these two datasets, this prompt and these three models, single-pass uncertainty is a weak hallucination signal that does not see confidently learned misconceptions, and small trained combinations of token features are a better use of the same single pass than five extra samples. Nothing here shows reliable or model-agnostic detection, and the confirm split, although held out, is in-distribution."));

add(h1("VI. Limitations"));
add(bullet("Labels come from one LLM judge with a written rubric. Re-judging consistency is high, but the judge was not validated against human annotators and may share systematic biases with the models it judges."));
add(bullet("Three 1–2B models, one prompt, greedy decoding with a 64-token limit, and two short-answer datasets; results may not transfer to larger models, long-form generation or other domains."));
add(bullet("The dev split was examined extensively before preregistration, so the dev numbers are exploratory; only the confirm numbers are confirmatory."));
add(bullet("Self-consistency used five samples and cosine clustering, not entailment-based clustering or larger budgets; SmolLM2's samples were generated in bf16."));
add(bullet("The difficulty proxy needs other models' labels and is not a deployable detector; it is used only to bound how much of the signal is question-level."));

add(h1("VII. Conclusion"));
add(body("We re-tested single-pass token uncertainty as a hallucination detector with a clean protocol and a preregistered held-out evaluation. It detects errors weakly on science questions, mostly by tracking how hard the question is, and fails on TruthfulQA where errors repeat popular misconceptions that the model produces with full confidence. Detectors that learn a small contrast between token features from a development split transfer to new questions and outperform both the single feature and five-sample self-consistency, but remain weak. Detecting confident, learned falsehoods will need evidence from outside the generating model, such as retrieval or verification. Future work should validate the judge against human labels, test larger models and other prompts, and explain the feature contrast the detectors learn."));

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
  "S. Kadavath *et al.*, “Language models (mostly) know what they know,” arXiv:2207.05221, 2022.",
  "L. Kuhn, Y. Gal, and S. Farquhar, “Semantic uncertainty: Linguistic invariances for uncertainty estimation in natural language generation,” in *Proc. ICLR*, 2023.",
  "L. Zheng *et al.*, “Judging LLM-as-a-judge with MT-Bench and Chatbot Arena,” in *Adv. Neural Inf. Process. Syst.*, vol. 36, 2023.",
  "S. Holm, “A simple sequentially rejective multiple test procedure,” *Scand. J. Statist.*, vol. 6, no. 2, pp. 65–70, 1979.",
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
