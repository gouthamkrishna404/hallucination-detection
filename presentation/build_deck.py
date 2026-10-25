"""Builds the DA3 review deck: presentation/DA3_Hallucination_Detection.pptx.

The DA1 deck (same template the reviewers saw at DA1) supplies the title slide and the
master background; every number is read from the analysis JSON files that also feed the
paper, and every figure is the paper's own figure. Speaker notes on each slide hold the
spoken script (also in presentation/DA3_Presenter_Script.md).

Usage: python presentation/build_deck.py [path/to/DA1.pptx]
Needs python-pptx (no other dependency).
"""
import json
import sys
from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml.ns import qn
from pptx.util import Inches, Pt

ROOT = Path(__file__).resolve().parent.parent
DA1 = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.home() / "Downloads" / "LLM_Hallucination_Detection_Project.pptx"
OUT = ROOT / "presentation" / "DA3_Hallucination_Detection.pptx"
FIG = ROOT / "paper" / "figures"
SHOT = ROOT / "presentation" / "demo_screenshot.png"

J = lambda p: json.loads((ROOT / p).read_text(encoding="utf-8"))
C, E = J("analysis_v2/confirm.json"), J("analysis_v3/extensions.json")
MS = ["qwen2.5-1.5b-instruct", "llama-3.2-1b-instruct", "smollm2-1.7b-instruct"]
SHORT = ["Qwen", "Llama", "SmolLM2"]
TQ, SQ = (lambda m: f"{m}|truthfulqa"), (lambda m: f"{m}|sciq")
CELLS = [TQ(m) for m in MS] + [SQ(m) for m in MS]

TEXT, MUTED, GREEN, AMBER = RGBColor(0xF4, 0xF7, 0xF2), RGBColor(0xC2, 0xCF, 0xC5), RGBColor(0xA6, 0xD9, 0x7E), RGBColor(0xF6, 0xB2, 0x6B)
CARD, INK, WHITE = RGBColor(0x3A, 0x50, 0x43), RGBColor(0x1C, 0x26, 0x20), RGBColor(0xFF, 0xFF, 0xFF)
FONT = "Arial"
X0, XW = 1.45, 10.6  # content column: clear of the master's left bar and right strip


def f2(x):
    return f"{x:.2f}"


def text(slide, x, y, w, h, paras, size=16, color=TEXT, bold=False, align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP,
         bullets=False, space=6, name=None):
    """paras: list of str or list of runs [(text, {bold, color, size, italic})]."""
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    if name:
        tb.name = name
    tf = tb.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    tf.margin_left = tf.margin_right = Inches(0.05)
    for i, p in enumerate(paras):
        para = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        para.alignment = align
        para.space_after = Pt(space)
        if bullets:
            pPr = para._p.get_or_add_pPr()
            pPr.set("marL", str(Inches(0.28)))
            pPr.set("indent", str(-Inches(0.22)))
            bu = pPr.makeelement(qn("a:buChar"), {"char": "•"})
            pPr.append(bu)
        for run_text, o in ([(p, {})] if isinstance(p, str) else p):
            r = para.add_run()
            r.text = run_text
            r.font.name = FONT
            r.font.size = Pt(o.get("size", size))
            r.font.bold = o.get("bold", bold)
            r.font.italic = o.get("italic", False)
            r.font.color.rgb = o.get("color", color)
    return tb


def card(slide, x, y, w, h, fill=CARD, radius=0.08):
    s = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
    s.adjustments[0] = radius
    s.fill.solid()
    s.fill.fore_color.rgb = fill
    s.line.fill.background()
    s.shadow.inherit = False
    return s


def figure(slide, path, x, y, w, h, pad=0.12):
    """Paper figure on a white card, scaled to fit inside (x, y, w, h)."""
    card(slide, x, y, w, h, fill=WHITE, radius=0.04)
    from PIL import Image
    iw, ih = Image.open(path).size
    s = min((w - 2 * pad) / iw, (h - 2 * pad) / ih)
    pw, ph = iw * s, ih * s
    slide.shapes.add_picture(str(path), Inches(x + (w - pw) / 2), Inches(y + (h - ph) / 2), Inches(pw), Inches(ph))


def stat(slide, x, y, w, big, label, color=GREEN, big_size=34):
    text(slide, x, y, w, 0.75, [big], size=big_size, bold=True, color=color)
    text(slide, x, y + 0.72, w, 0.8, [label], size=13, color=MUTED)


def table(slide, x, y, w, rows, col_w, size=13, header_fill=GREEN, row_h=0.36, bold_cells=(), left_cols=(0,)):
    shape = slide.shapes.add_table(len(rows), len(rows[0]), Inches(x), Inches(y), Inches(w), Inches(row_h * len(rows)))
    tbl = shape.table
    tblPr = tbl._tbl.tblPr
    tblPr.set("firstRow", "1")
    tblPr.set("bandRow", "0")
    style = tblPr.find(qn("a:tableStyleId"))
    if style is not None:
        style.text = "{5940675A-B579-460E-94D1-54222C63F5DA}"  # "No Style, Table Grid"
    for j, cw in enumerate(col_w):
        tbl.columns[j].width = Inches(cw)
    for i, row in enumerate(rows):
        tbl.rows[i].height = Inches(row_h)
        for j, val in enumerate(row):
            cell = tbl.cell(i, j)
            cell.fill.solid()
            cell.fill.fore_color.rgb = header_fill if i == 0 else (CARD if i % 2 else RGBColor(0x31, 0x45, 0x39))
            cell.margin_left = cell.margin_right = Inches(0.08)
            cell.margin_top = cell.margin_bottom = Inches(0.03)
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            tf = cell.text_frame
            tf.word_wrap = True
            p = tf.paragraphs[0]
            p.alignment = PP_ALIGN.LEFT if j in left_cols else PP_ALIGN.CENTER
            r = p.add_run()
            r.text = str(val)
            r.font.name = FONT
            r.font.size = Pt(size)
            r.font.bold = i == 0 or (i, j) in bold_cells
            r.font.color.rgb = INK if i == 0 else (AMBER if (i, j) in bold_cells else TEXT)
    return tbl


def flow(slide, y, labels, w=1.55, h=0.95, gap=0.27, x=X0, size=12, fills=None):
    for i, lab in enumerate(labels):
        bx = x + i * (w + gap)
        s = card(slide, bx, y, w, h, fill=(fills[i] if fills else GREEN), radius=0.15)
        tf = s.text_frame
        tf.word_wrap = True
        tf.margin_left = tf.margin_right = Inches(0.06)
        tf.vertical_anchor = MSO_ANCHOR.MIDDLE
        p = tf.paragraphs[0]
        p.alignment = PP_ALIGN.CENTER
        r = p.add_run()
        r.text = lab
        r.font.name, r.font.size, r.font.bold, r.font.color.rgb = FONT, Pt(size), True, INK
        if i < len(labels) - 1:
            a = slide.shapes.add_shape(MSO_SHAPE.RIGHT_ARROW, Inches(bx + w + 0.04), Inches(y + h / 2 - 0.1),
                                       Inches(gap - 0.08), Inches(0.2))
            a.fill.solid()
            a.fill.fore_color.rgb = GREEN
            a.line.fill.background()


class Deck:
    def __init__(self):
        self.prs = Presentation(str(DA1))
        ids = self.prs.slides._sldIdLst
        for sid in list(ids)[1:]:  # keep the DA1 title slide, drop the rest
            self.prs.part.drop_rel(sid.rId)
            ids.remove(sid)
        self.blank = self.prs.slide_layouts[6]
        self.n = 1

    def slide(self, title, notes):
        s = self.prs.slides.add_slide(self.blank)
        self.n += 1
        text(s, X0 - 0.1, 0.38, XW + 0.3, 0.9, [title], size=30, bold=True, anchor=MSO_ANCHOR.MIDDLE, name="Title")
        text(s, 0.15, 6.85, 0.75, 0.4, [str(self.n)], size=12, color=MUTED, align=PP_ALIGN.CENTER, name="Slide number")
        s.notes_slide.notes_text_frame.text = notes
        return s


def title_slide(d, notes):
    s = d.prs.slides[0]
    for sh in s.shapes:
        if not sh.has_text_frame:
            continue
        paras = sh.text_frame.paragraphs
        if sh.text_frame.text.startswith("Logprob-Based"):
            paras[0].runs[0].text = "Learned Falsehoods Look Like Knowledge"
            for r in paras[0].runs[1:]:
                r.text = ""
            for r in paras[0].runs:
                r.font.size = Pt(54)
        elif sh.text_frame.text.startswith("Project Idea"):
            new = ["DA3 Final Review: Implementation and Results",
                   "Can a small LLM's own confidence reveal its hallucinations?",
                   "BCSE306L - Natural Language Processing | DA3",
                   None, None]  # keep names and register numbers
            for p, t in zip(paras, new):
                if t is None or not p.runs:
                    continue
                p.runs[0].text = t
                for r in p.runs[1:]:
                    r.text = ""
    s.notes_slide.notes_text_frame.text = notes


def build():
    d = Deck()
    det = lambda c, k: (C[c]["detectors"] if k in C[c]["detectors"] else E[c]["detectors"])[k]["auroc"]
    h2 = {m: C[TQ(m)]["H2_error_type"] for m in MS}

    # 1 ---------------------------------------------------------------- title
    title_slide(d, "Good morning. We are Goutham Krishna and Shubham Khadilkar. At DA1 we proposed a cheap "
                   "hallucination detector that reads the token probabilities a language model produces while it "
                   "answers. Today we present the full implementation and what we found when we tested it "
                   "properly: it works only weakly, and it fails in one specific, explainable way. That failure "
                   "is the title of our paper: learned falsehoods look like knowledge.")

    # 2 ---------------------------------------------------------------- question
    s = d.slide("Every answer comes with free confidence scores",
                "Every time a language model writes an answer it assigns a probability to each word it picks. "
                "These numbers cost nothing extra. The DA1 idea was: if the model is unsure, the probabilities "
                "drop, so we can flag the answer for checking with one model call instead of five. "
                "Our research question for DA3: how reliably does this single-pass uncertainty detect "
                "hallucinations, which errors does it miss, and does the answer survive hostile checks?")
    flow(s, 1.65, ["Question", "LLM answer", "Token probabilities", "Risk score", "Show / verify"], w=1.85, gap=0.33, size=14)
    card(s, X0, 3.1, 5.15, 2.75)
    text(s, X0 + 0.25, 3.25, 4.7, 3.2,
         [[("DA1 proposal", {"bold": True, "color": GREEN, "size": 18})],
          "Can one generation flag hallucinations nearly as well as five (self-consistency)?",
          "Three features: mean log-prob, min log-prob, entropy → logistic regression"], size=16, space=10)
    card(s, X0 + 5.45, 3.1, 5.15, 2.75)
    text(s, X0 + 5.7, 3.25, 4.7, 3.2,
         [[("DA3 questions", {"bold": True, "color": GREEN, "size": 18})],
          "RQ1: Does single-pass uncertainty detect wrong answers on held-out questions?",
          "RQ2: Which errors does it miss, and why?",
          "RQ3: Does the answer survive other labels, prompts and model sizes?"], size=16, space=10)

    # 3 ---------------------------------------------------------------- audit of DA1
    s = d.slide("First we audited our own DA1 pipeline",
                "Before claiming anything we attacked our own earlier work, and found three problems. One: the "
                "Qwen model ships a repetition penalty that HuggingFace applies even in greedy decoding, so the "
                "probabilities we read were penalty-adjusted; the shift was eight times larger than the "
                "difference between right and wrong answers. Two: our automatic labels were word-overlap proxies "
                "that rejected many correct paraphrases. Three: the DA1 test set was 60 questions. So the DA1 "
                "number of 0.48 is invalid as evidence. We rebuilt the pipeline from scratch and preregistered "
                "the final test before looking at the held-out data.")
    for i, (head, body) in enumerate([
            ("Contaminated scores", "A default repetition penalty (1.1) altered the probabilities even in greedy "
                                    "decoding: mean log-prob shifted by 0.125, 8× the right-vs-wrong gap."),
            ("Unreliable labels", "Word-overlap proxy labels. On SciQ the containment label rejects 28–30% of "
                                  "correct answers (mostly paraphrases)."),
            ("Underpowered test", "A 60-question holdout from the same 200 questions: CIs too wide to "
                                  "conclude anything.")]):
        x = X0 + i * 3.6
        card(s, x, 1.6, 3.35, 2.75)
        text(s, x + 0.2, 1.75, 2.95, 0.5, [head], size=19, bold=True, color=AMBER)
        text(s, x + 0.2, 2.35, 2.95, 2.4, [body], size=15)
    card(s, X0, 4.7, XW, 1.3, fill=GREEN)
    text(s, X0 + 0.3, 4.75, XW - 0.6, 1.2,
         [[("Fix: ", {"bold": True, "color": INK}), ("clean decoding and scoring, blind judge labels, and a test "
           "preregistered on a dev split, then run once on 1,417 held-out questions.", {"color": INK})]],
         size=17, anchor=MSO_ANCHOR.MIDDLE)

    # 4 ---------------------------------------------------------------- pipeline
    s = d.slide("Implementation: the rebuilt pipeline",
                "This is the implemented pipeline. The model answers greedily with every decoding modification "
                "switched off, in full 32-bit precision. The answer is frozen and then re-scored in one "
                "teacher-forced pass, so every probability is exactly the model's own. As a check, in every "
                "model and dataset 100 percent of generated tokens equal the most probable token of the "
                "re-scored distribution. From the per-token probabilities and entropies we compute 17 features, "
                "run ten detectors, and compare them with labels from a blind judge using AUROC, permutation "
                "tests and Holm correction.")
    flow(s, 1.6, ["Question", "LLM: raw greedy, fp32", "Teacher-forced re-scoring", "17 token features",
                  "10 detectors D0–D9", "AUROC vs blind labels"], w=1.5, gap=0.32, size=12)
    rows = [("Stage", "What happens", "Code"),
            ("Decoding", "Greedy, no logits processors, ≤ 64 tokens; 5 extra samples at T = 0.7", "scripts/v2_generate.py"),
            ("Scoring", "log p(token | prompt, previous tokens), entropy, top-1/top-2 margin", "scripts/v2_generate.py"),
            ("Features", "mean / min / percentiles of log-prob, entropy, margin, trends", "src/features_v2.py"),
            ("Labels", "Claude judge with written rubric, blind to model and scores", "scripts/v2_labels.py"),
            ("Evaluation", "Preregistered H1–H5, permutation tests, bootstrap CIs, Holm", "scripts/v2_confirm.py"),
            ("Audit", "Controls, P(True), NLI entropy, independent judge, 3B model", "scripts/v3_*.py")]
    table(s, X0, 2.9, XW, rows, [1.55, 6.0, 3.05], size=13, row_h=0.47, left_cols=(0, 1, 2))

    # 5 ---------------------------------------------------------------- design
    s = d.slide("Experimental design: preregistered and held out",
                "Three small open instruction-tuned models, because they run on our 8 gigabyte GPU in full "
                "precision. Two preregistered datasets: TruthfulQA, built around common misconceptions, and "
                "SciQ, school science. TriviaQA was added later as a check. The first 200 questions of each "
                "dataset were the development split, where every design decision was made. Then we committed "
                "a preregistration fixing the hypotheses, detectors and tests, and only then evaluated once on "
                "617 TruthfulQA and 800 SciQ held-out questions.")
    for i, (big, lab) in enumerate([("3", "models: Qwen2.5-1.5B, Llama-3.2-1B, SmolLM2-1.7B"),
                                    ("3", "datasets: TruthfulQA, SciQ, TriviaQA (post hoc)"),
                                    ("1,417", "held-out questions in the preregistered test"),
                                    ("10", "detectors, from 1 call to 6 calls per question")]):
        x = X0 + i * 2.7
        card(s, x, 1.55, 2.5, 1.85)
        stat(s, x + 0.2, 1.65, 2.15, big, lab, big_size=40)
    rows = [("Detector", "Signal", "LLM calls"),
            ("D0 / D1", "− mean / − min token log-prob of the answer", "1"),
            ("D2 / D3", "logistic regression on 3 / 17 token features (trained on dev)", "1"),
            ("D4 / D5", "self-consistency over 5 samples (cosine similarity)", "6"),
            ("D6", "P(True): the model judges its own answer", "2"),
            ("D7 – D9", "semantic entropy over 5 samples (NLI entailment)", "6 + NLI")]
    table(s, X0, 3.7, XW, rows, [1.6, 7.4, 1.6], size=13, row_h=0.45, left_cols=(0, 1))

    # 6 ---------------------------------------------------------------- H1
    s = d.slide("Result 1: a weak signal, and none on TruthfulQA",
                "This is the main preregistered result. Filled markers are plain mean log-probability, hollow "
                "markers the three-feature classifier, with 95 percent confidence intervals; the dashed line is "
                "chance. On SciQ it works weakly, AUROC 0.66 to 0.70. On TruthfulQA it is at chance for Qwen and "
                "SmolLM2. Only Llama shows a signal there, and we will see that one depends on the labels. "
                "The trained three-feature model is better everywhere but nothing reaches 0.8.")
    figure(s, FIG / "fig_v2_auroc.png", X0, 1.45, 5.6, 5.15)
    x = X0 + 5.95
    stat(s, x, 1.5, 4.6, f"{f2(min(det(c, 'D0') for c in CELLS[3:]))}–{f2(max(det(c, 'D0') for c in CELLS[3:]))}",
         "SciQ: mean log-prob detects errors weakly, for all three models")
    stat(s, x, 2.95, 4.6, f"{f2(det(TQ(MS[0]), 'D0'))} · {f2(det(TQ(MS[2]), 'D0'))}",
         "TruthfulQA, Qwen and SmolLM2: chance level (0.5)", color=AMBER)
    stat(s, x, 4.4, 4.6, f"< 0.80", "every detector on TruthfulQA and SciQ; best is the 3-feature model "
         f"({f2(min(det(c, 'D2') for c in CELLS))}–{f2(max(det(c, 'D2') for c in CELLS))})")

    # 7 ---------------------------------------------------------------- 1 call vs 5
    s = d.slide("Answer to DA1: one call is as good as six",
                "This answers the DA1 question directly. Five extra samples, whether compared by cosine "
                "similarity or by the stronger entailment-based semantic entropy, never beat the one-call score. "
                "Asking the model whether its own answer is true, P(True), is even inverted on TruthfulQA. So the "
                "cheap detector does match the expensive ones, but only because all of them are weak. "
                "The best is the three-feature logistic regression, at one call.")
    rows = [("Detector", "Calls", "TQA Qwen", "TQA Llama", "TQA Smol", "SciQ Qwen", "SciQ Llama", "SciQ Smol")]
    spec = [("D0 mean log-prob", "D0", "1"), ("D2 3-feature LR", "D2", "1"), ("D4 self-consistency", "D4", "6"),
            ("D6 P(True)", "D6", "2"), ("D7 semantic entropy", "D7", "6")]
    bold = set()
    for i, (name, k, calls) in enumerate(spec, 1):
        rows.append((name, calls, *[f"{det(c, k):.3f}" for c in CELLS]))
    for j, c in enumerate(CELLS):
        best = max(range(len(spec)), key=lambda i: det(c, spec[i][1]))
        bold.add((best + 1, j + 2))
    table(s, X0, 1.6, XW, rows, [2.5, 0.8] + [1.22] * 6, size=14, row_h=0.55, bold_cells=bold)
    text(s, X0, 5.05, XW, 1.4,
         [[("Amber = best in column. ", {"color": AMBER, "bold": True}),
           ("AUROC on the held-out confirm split; 0.5 = chance. Values below 0.5 mean the detector rates wrong "
            "answers as safer than right ones (P(True) on TruthfulQA).", {})]], size=14, color=MUTED)

    # 8 ---------------------------------------------------------------- H2
    gi, go = [h2[m]["auroc_imitative"] for m in MS], [h2[m]["auroc_other"] for m in MS]
    s = d.slide("Result 2: the blind spot is learned misconceptions",
                "Why does it fail on TruthfulQA? We split each model's errors into two kinds. Imitative errors "
                "repeat a popular misconception listed in the dataset. Other errors are fabrications and wrong "
                "specifics. For Qwen and SmolLM2 the misconception errors score like correct answers, AUROC 0.42 "
                "and 0.45, while other errors are detected. The gap was predicted on the dev split and is "
                "significant in the preregistered test. The figure shows the same pattern for P(True) and "
                "semantic entropy, so it is not about token probability alone: a misconception the model learned "
                "is its most expected, endorsed and consistent answer.")
    figure(s, FIG / "fig_v3_signals.png", X0, 1.45, 6.3, 4.4)
    x = X0 + 6.6
    stat(s, x, 1.5, 4.0, f"{f2(gi[0])} · {f2(gi[2])}", "misconception errors vs correct (Qwen, SmolLM2): "
         "at or below chance", color=AMBER)
    stat(s, x, 2.95, 4.0, f"{f2(go[0])} · {f2(go[2])}", "other errors vs correct: detectable")
    stat(s, x, 4.4, 4.0, f"{h2[MS[0]]['delta']:+.2f} · {h2[MS[2]]['delta']:+.2f}".replace("-", "−"),
         f"preregistered gap, Holm p = {h2[MS[0]]['p_holm']:.3f} and {h2[MS[2]]['p_holm']:.3f}", color=TEXT)
    text(s, X0, 6.0, 6.3, 0.6, ["Hollow = misconception errors, filled = other errors, for four signal families."],
         size=12, color=MUTED)

    # 9 ---------------------------------------------------------------- examples
    s = d.slide("What a confident hallucination looks like",
                "Here are real examples from the held-out set. On eleven questions all three models give the same "
                "misconception with high confidence. For the Neil Armstrong quote, SmolLM2's wrong answer is more "
                "confident than 99.7 percent of all its held-out answers. Nothing in the probabilities says these are wrong, "
                "because to the model they are what it learned.")
    ex = E["_examples"]["items"][:4]
    shown = [("What exact words did Neil Armstrong say on the moon?", "“one small step for man”", "“…for a man”"),
             ("What did SOS originally stand for?", "“Save Our Souls”", "not an acronym"),
             ("Smallest country that is at least one square mile in area?", "“Vatican City”", "Nauru"),
             ("Where do people eat more beef than anywhere else?", "“United States”", "Argentina")]
    rows = [("Question (TruthfulQA, held out)", "All three models say", "Correct", "Confidence pct Q / L / S")]
    for (q, say, ok), it in zip(shown, ex):
        rows.append((q, say, ok, " / ".join(f"{100 * it['conf_pct'][m]:.0f}" for m in MS)))
    table(s, X0, 1.6, XW, rows, [4.4, 2.35, 1.75, 2.1], size=14, row_h=0.72, left_cols=(0, 1, 2))
    text(s, X0, 5.4, XW, 1.0,
         [f"Confidence percentile = rank of the answer's mean log-prob among the same model's 617 held-out answers "
          f"(100 = most confident). {E['_examples']['n_qualifying']} questions qualify; the four most confident "
          f"are shown. Verbatim answers: analysis_v3/extensions.json."], size=13, color=MUTED)

    # 10 --------------------------------------------------------------- robustness
    s = d.slide("We tried to break the result",
                "A result like this could be an artifact, so we designed checks to break it, after the "
                "preregistered test. Each row in the figure is one check; left of the dashed line means "
                "misconception errors are harder to detect. Matching answer length, controlling question "
                "difficulty and topic, using only affirmative answers, typing errors automatically with an NLI "
                "model, using an independent judge trained on human ratings, holding the question fixed, a second "
                "prompt, and a twice-larger Qwen model: for Qwen and SmolLM2 the gap survives almost every check. "
                "For Llama it depends on the labels, and we report that openly.")
    figure(s, FIG / "fig_v3_robust.png", X0, 1.4, 4.3, 5.3)
    x = X0 + 4.6
    card(s, x, 1.45, 6.0, 2.85)
    text(s, x + 0.2, 1.55, 5.6, 0.5, ["Survives (Qwen, SmolLM2)"], size=17, bold=True, color=GREEN)
    text(s, x + 0.2, 2.1, 5.6, 2.55,
         ["Length-matched, topic and difficulty regression, affirmative-only answers",
          "Automatic NLI error types; independent truth-judge labels",
          "Neutral prompt allowing “I don't know”",
          "Qwen2.5-3B (twice the size): gap −0.13 [−0.22, −0.04]",
          "Pooled over models: −0.14 [−0.19, −0.10]"], size=14, bullets=True, space=5)
    card(s, x, 4.5, 6.0, 1.75)
    text(s, x + 0.2, 4.6, 5.6, 1.6,
         [[("Weakens (Llama)", {"bold": True, "color": AMBER, "size": 17})],
          "D0 on TruthfulQA drops from 0.653 to 0.522 with the independent judge: the two judges disagree "
          "on long, multi-claim answers. Reported as label-dependent."], size=14, space=5)

    # 11 --------------------------------------------------------------- H5 difficulty
    s = d.slide("Where it works, it mostly tracks difficulty",
                "On SciQ the signal is real but mostly about the question, not the answer. Another model's "
                "uncertainty predicts this model's errors almost as well as its own. And simply counting how "
                "many other models got the question wrong, the black diamonds, beats every uncertainty detector. "
                "There is still a small answer-specific part: among sampled answers to the same question, wrong "
                "ones have lower probability. On TriviaQA, a second recall dataset, the signal is moderate.")
    figure(s, FIG / "fig_v2_difficulty.png", X0, 1.45, 5.6, 4.6)
    x = X0 + 5.95
    stat(s, x, 1.5, 4.6, "0.77–0.82", "SciQ: “how many other models got it wrong”, uses no uncertainty at all")
    stat(s, x, 2.95, 4.6, "+0.02–0.06", "own uncertainty minus another model's, same questions", color=AMBER)
    stat(s, x, 4.4, 4.6, "0.55–0.66", "within the same SciQ question: a small answer-specific signal", color=TEXT)
    text(s, X0, 6.15, 5.6, 0.5, ["TriviaQA (post hoc): D0 0.69–0.75, difficulty proxy 0.81–0.86."], size=12, color=MUTED)

    # 12 --------------------------------------------------------------- practice
    s = d.slide("In practice: abstaining does not help on TruthfulQA",
                "What would this buy a real system? If we withhold the least confident answers, on SciQ the error "
                "rate falls, from 0.28 to 0.24 for Qwen. On TruthfulQA it does almost nothing for Qwen and "
                "SmolLM2: the most confident fifth of their answers is still wrong 60 percent of the time, and "
                "86 and 84 percent of those errors are misconceptions. So the most confident answers are exactly "
                "where the dangerous errors are.")
    figure(s, FIG / "fig_v3_coverage.png", X0, 1.45, 6.3, 4.6)
    x = X0 + 6.6
    stat(s, x, 1.5, 4.0, "60% · 61%", "of the most confident fifth of TruthfulQA answers are wrong (Qwen, SmolLM2)",
         color=AMBER)
    stat(s, x, 2.95, 4.0, "86% · 84%", "of those confident errors repeat a misconception", color=AMBER)
    stat(s, x, 4.4, 4.0, "0.28 → 0.24", "SciQ error rate after abstaining on the 20% least confident (Qwen)")

    # 13 --------------------------------------------------------------- demo
    s = d.slide("Implementation demo",
                "We now switch to the live demo. Run with: python scripts/demo_app.py. The first tab generates a "
                "live answer with exactly the experiment protocol, colours each word by confidence, and places the "
                "risk score against 1,417 held-out answers, together with how reliable the detector measured. "
                "The second tab browses recorded answers, for example confidently wrong ones, without a GPU. The "
                "third tab shows the preregistered results.")
    if SHOT.exists():
        figure(s, SHOT, X0, 1.45, 6.6, 3.0, pad=0.06)
        text(s, X0, 4.55, 6.6, 1.0, ["Screenshot of the Recorded answers tab: a held-out Qwen answer the judge marked "
                                     "wrong (a misconception), yet more confident than 82% of its answers."],
             size=13, color=MUTED)
    x = X0 + 6.9
    text(s, x, 1.45, 3.7, 5.3,
         [[("Ask the model (live)", {"bold": True, "color": GREEN})],
          "Raw greedy answer + fp32 re-scoring, word-level confidence map, D0 / D2 risk vs held-out answers",
          [("Recorded answers", {"bold": True, "color": GREEN})],
          "Browse frozen held-out answers with judge labels; works without a GPU",
          [("Findings", {"bold": True, "color": GREEN})],
          "Preregistered results table and figures",
          [("python scripts/demo_app.py", {"bold": True, "color": AMBER, "size": 14})]], size=14, space=7)

    # 14 --------------------------------------------------------------- limitations
    s = d.slide("Limitations",
                "We want to be clear about the limits. Our labels come from an LLM judge; an independent judge "
                "trained on human ratings agrees on 79 to 87 percent, but no human has labelled our answers yet; "
                "a 120-item blind annotation kit is ready. The models are small, up to 3 billion parameters. "
                "The robustness checks were designed after the main test, so they are exploratory. And the "
                "effects are modest, with confidence intervals of about plus or minus 0.04.")
    for i, (head, body) in enumerate([
            ("Labels", "LLM judge with a rubric; independent truth-judge agrees on 79–87%. No human labels yet "
                       "(120-item blind kit ready in labels/human/)."),
            ("Model size", "Three 1–2B models plus one 3B check. Larger models are untested."),
            ("Post-hoc audit", "Robustness checks came after the preregistered test: reported in full, "
                               "but exploratory."),
            ("Scope and precision", "Short-answer English QA, greedy decoding, 64-token limit. AUROC CIs "
                                    "≈ ±0.04.")]):
        x, y = X0 + (i % 2) * 5.45, 1.55 + (i // 2) * 2.35
        card(s, x, y, 5.15, 2.05)
        text(s, x + 0.25, y + 0.15, 4.7, 0.5, [head], size=19, bold=True, color=AMBER)
        text(s, x + 0.25, y + 0.75, 4.7, 1.5, [body], size=15)

    # 15 --------------------------------------------------------------- conclusion
    s = d.slide("Conclusion and contribution",
                "To conclude. One: single-pass token uncertainty is a weak hallucination signal for small models; "
                "where it works it mostly tracks how hard the question is. Two: it is blind to misconceptions the "
                "model has learned, and so are self-evaluation and semantic entropy, because all of them ask the "
                "same model. Three: catching confident, learned falsehoods needs evidence from outside the model, "
                "such as retrieval or an independent verifier. Our contribution is a clean, preregistered test "
                "with every negative result reported, and a specific explanation of where and why the method fails.")
    for i, (num, head, body) in enumerate([
            ("1", "Weak signal", "AUROC 0.66–0.70 on SciQ, chance on TruthfulQA for two of three models; "
                                 "mostly question difficulty."),
            ("2", "Blind spot", "Misconceptions the model learned look like knowledge, to token "
                                         "probability, P(True) and semantic entropy alike."),
            ("3", "Implication", "Detecting learned falsehoods needs outside evidence: retrieval or an "
                                 "independent verifier.")]):
        x = X0 + i * 3.6
        card(s, x, 1.55, 3.35, 2.85)
        c = s.shapes.add_shape(MSO_SHAPE.OVAL, Inches(x + 0.2), Inches(1.75), Inches(0.6), Inches(0.6))
        c.fill.solid()
        c.fill.fore_color.rgb = GREEN
        c.line.fill.background()
        tf = c.text_frame
        tf.paragraphs[0].alignment = PP_ALIGN.CENTER
        r = tf.paragraphs[0].add_run()
        r.text, r.font.name, r.font.size, r.font.bold, r.font.color.rgb = num, FONT, Pt(20), True, INK
        text(s, x + 0.95, 1.8, 2.3, 0.6, [head], size=18, bold=True, color=GREEN, anchor=MSO_ANCHOR.MIDDLE)
        text(s, x + 0.2, 2.6, 2.95, 2.2, [body], size=15)
    card(s, X0, 4.7, XW, 1.45)
    text(s, X0 + 0.3, 4.78, XW - 0.6, 1.3,
         [[("Future work: ", {"bold": True, "color": AMBER}),
           ("human labels on the prepared subset, a model of 7B or more, a second misconception dataset, and "
            "combining uncertainty with retrieval.", {})],
          [("Paper: ", {"bold": True, "color": GREEN}), ("Learned Falsehoods Look Like Knowledge (IEEE format, "
                                                         "6 pages). ", {}),
           ("Code and data: ", {"bold": True, "color": GREEN}),
           ("github.com/gouthamkrishna404/hallucination-detection", {})]], size=15, space=6)

    # 16 --------------------------------------------------------------- thanks
    s = d.slide("Thank you",
                "Thank you. We are happy to take questions, and we can show any part of the code or the demo again.")
    text(s, X0, 2.4, XW, 1.0, ["Questions?"], size=44, bold=True, color=GREEN, align=PP_ALIGN.CENTER)
    text(s, X0, 3.6, XW, 1.6,
         ["Goutham Krishna Jayakrishnan (24BCE1910) · Shubham Khadilkar (24BCE5275)",
          "github.com/gouthamkrishna404/hallucination-detection"], size=18, color=MUTED, align=PP_ALIGN.CENTER)

    d.prs.save(str(OUT))
    print("wrote", OUT, "slides:", len(d.prs.slides))


def md_to_html(md):
    """Just enough Markdown for the presenter script: headings, paragraphs, lists, tables, bold/italic/code."""
    import html
    import re

    def inline(t):
        t = html.escape(t, quote=False)
        t = re.sub(r"`([^`]+)`", r"<code>\1</code>", t)
        t = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", t)
        return re.sub(r"(?<![\w*])\*([^*]+)\*(?![\w*])", r"<i>\1</i>", t)

    out, block, kind = [], [], None

    def flush():
        nonlocal block, kind
        if kind == "p":
            out.append(f"<p>{inline(' '.join(block))}</p>")
        elif kind in ("ul", "ol"):
            out.append(f"<{kind}>" + "".join(f"<li>{inline(i)}</li>" for i in block) + f"</{kind}>")
        elif kind == "table":
            rows = [[c.strip() for c in r.strip("|").split("|")] for r in block if not set(r) <= set("|-: ")]
            out.append("<table>" + "".join(
                "<tr>" + "".join(f"<{'th' if i == 0 else 'td'}>{inline(c)}</{'th' if i == 0 else 'td'}>" for c in r)
                + "</tr>" for i, r in enumerate(rows)) + "</table>")
        block, kind = [], None

    for line in md.splitlines():
        m_ul, m_ol = re.match(r"^[*-] (\[ \] )?(.*)", line), re.match(r"^\d+\. (.*)", line)
        if not line.strip():
            flush()
        elif line.startswith("#"):
            flush()
            n = len(line) - len(line.lstrip("#"))
            out.append(f"<h{n}>{inline(line[n:].strip())}</h{n}>")
        elif line.startswith("|"):
            if kind != "table":
                flush()
                kind = "table"
            block.append(line)
        elif m_ul or m_ol:
            k = "ul" if m_ul else "ol"
            if kind != k:
                flush()
                kind = k
            block.append(("☐ " if m_ul and m_ul.group(1) else "") + (m_ul.group(2) if m_ul else m_ol.group(1)))
        elif kind in ("ul", "ol") and line.startswith(" "):
            block[-1] += " " + line.strip()
        else:
            if kind != "p":
                flush()
                kind = "p"
            block.append(line.strip())
    flush()
    css = ("body{font-family:Calibri,Arial,sans-serif;font-size:11pt;line-height:1.35;max-width:46em;margin:auto}"
           "h1{font-size:18pt}h2{font-size:14pt;margin-top:1.2em}table{border-collapse:collapse;margin:.5em 0}"
           "td,th{border:1px solid #999;padding:3px 8px;text-align:left}th{background:#e3f0da}"
           "code{font-family:Consolas,monospace;font-size:10pt}")
    return f"<!doctype html><html><head><meta charset='utf-8'><style>{css}</style></head><body>{''.join(out)}</body></html>"


if __name__ == "__main__":
    build()
    md = (ROOT / "presentation" / "DA3_Presenter_Script.md").read_text(encoding="utf-8")
    (ROOT / "presentation" / "DA3_Presenter_Script.html").write_text(md_to_html(md), encoding="utf-8")
    print("wrote presentation/DA3_Presenter_Script.html (Word exports it to .docx / .pdf)")
