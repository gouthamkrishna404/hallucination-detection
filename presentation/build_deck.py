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
                   "Can an AI model's own confidence reveal when it is making things up?",
                   "BCSE306L - Natural Language Processing | DA3",
                   None, None]  # keep names and register numbers
            for p, t in zip(paras, new):
                if t is None or not p.runs:
                    continue
                p.runs[0].text = t
                for r in p.runs[1:]:
                    r.text = ""
    s.notes_slide.notes_text_frame.text = notes


T = J("analysis_v3/triviaqa.json")
M2 = J("analysis_v3/d2_mechanism.json")
CHART = ROOT / "presentation" / "fig_detection_scores.png"


def rng(vals, d=2):
    lo, hi = min(vals), max(vals)
    return f"{lo:.{d}f}–{hi:.{d}f}"


def score_chart(d2):
    """Plain-language main result: best one-call detector (D2) per dataset and model."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    groups = [("School science\n(SciQ)", SQ), ("Trivia\n(TriviaQA, added later)", None),
              ("Misconception traps\n(TruthfulQA)", TQ)]
    colors = ["#2a78d6", "#eb6834", "#1baf7a"]  # validated categorical slots 1-3, light surface
    fig, ax = plt.subplots(figsize=(6.4, 4.6), dpi=200)
    w = 0.26
    for j, (m, name, col) in enumerate(zip(MS, SHORT, colors)):
        xs, vals = [], []
        for i, (_, f) in enumerate(groups):
            v = d2(f(m)) if f else T["alias"][m]["D2"]["auroc"]
            xs.append(i + (j - 1) * (w + 0.02))
            vals.append(v)
        ax.bar(xs, [v - 0.5 for v in vals], w, bottom=0.5, color=col, label=name, edgecolor="white", linewidth=2)
        for x, v in zip(xs, vals):
            ax.text(x, v + 0.006, f"{v:.2f}", ha="center", va="bottom", fontsize=9, color="#1a1a19")
    ax.axhline(0.5, color="#6b6a64", lw=1.2, ls="--")
    ax.set_xlim(-0.55, 2.9)
    ax.text(2.45, 0.505, "guessing", ha="left", va="bottom", fontsize=9, color="#6b6a64")
    ax.set_xticks(range(3), [g for g, _ in groups], fontsize=10, color="#1a1a19")
    ax.set_ylim(0.45, 0.88)
    ax.set_ylabel("Detection score  (0.5 = guessing, 1.0 = perfect)", fontsize=10, color="#3d3c38")
    ax.yaxis.grid(True, color="#e6e5e0", lw=0.8)
    ax.set_axisbelow(True)
    for sp in ["top", "right", "left"]:
        ax.spines[sp].set_visible(False)
    ax.spines["bottom"].set_color("#b9b8b0")
    ax.tick_params(colors="#3d3c38", length=0)
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.1), fontsize=10)
    fig.tight_layout()
    fig.savefig(CHART, facecolor="white")
    plt.close(fig)


def build():
    d = Deck()
    det = lambda c, k: (C[c]["detectors"] if k in C[c]["detectors"] else E[c]["detectors"])[k]["auroc"]
    h2 = {m: C[TQ(m)]["H2_error_type"] for m in MS}
    d2 = lambda c: det(c, "D2")
    score_chart(d2)
    sciq_d2 = [d2(SQ(m)) for m in MS]
    triv_d2 = [T["alias"][m]["D2"]["auroc"] for m in MS]
    made_up = [M2[TQ(m)]["H2_gap"]["D2"]["auroc_other"] for m in MS]
    multi = ["D4", "D5", "D7", "D8", "D9"]
    wins = sum(d2(c) > max(det(c, k) for k in multi + ["D6"]) for c in CELLS)

    # 1 ---------------------------------------------------------------- title
    title_slide(d, "Good morning. We are Goutham Krishna and Shubham Khadilkar. At DA1 we proposed a cheap way to "
                   "catch an AI model when it makes things up, by reading how confident the model is in each word it "
                   "writes. Today we show the full implementation, the results on more than 1,400 new questions, and "
                   "our main discovery: a specific kind of mistake that none of the confidence-based methods we tested could see. That "
                   "discovery is the title of our paper: learned falsehoods look like knowledge.")

    # 2 ---------------------------------------------------------------- idea
    s = d.slide("Every AI answer comes with a free confidence score",
                "When a language model writes an answer, it assigns a probability to every word it picks. These "
                "numbers come for free. Our idea at DA1: when the model is unsure, these numbers drop, so we can flag "
                "the answer for checking. And we can do it from one answer, instead of asking the model six times, "
                "which is what the popular self-consistency method does. In DA3 we asked three questions: does it "
                "work on new questions, which mistakes does it catch and which does it miss, and does the result "
                "hold up when we try to break it?")
    flow(s, 1.65, ["Question", "AI answers", "Confidence of each word", "Risk score", "Trust or double-check"],
         w=1.85, gap=0.33, size=14)
    card(s, X0, 3.1, 5.15, 2.75)
    text(s, X0 + 0.25, 3.25, 4.7, 3.2,
         [[("Our DA1 proposal", {"bold": True, "color": GREEN, "size": 18})],
          "Spot made-up answers from a single run of the model, using three confidence measures.",
          "Six times cheaper than asking the model six times and comparing the answers."], size=16, space=10)
    card(s, X0 + 5.45, 3.1, 5.15, 2.75)
    text(s, X0 + 5.7, 3.25, 4.7, 3.2,
         [[("DA3 questions", {"bold": True, "color": GREEN, "size": 18})],
          "1. Does it work on new, unseen questions?",
          "2. Which mistakes does it catch, and which slip through? Why?",
          "3. Does the result survive our attempts to break it?"], size=16, space=10)

    # 3 ---------------------------------------------------------------- audit of DA1
    s = d.slide("Step 1: we stress-tested our own DA1 system",
                "Before trusting any result we checked our own DA1 work, and this made the project much stronger. "
                "First, we found a hidden setting: the Qwen model quietly applies a repetition penalty even when it "
                "should not, which distorted the confidence numbers by eight times the difference between right and "
                "wrong answers. Most projects using this library would never notice. Second, our automatic grader "
                "only matched words, so it marked correct answers phrased differently as wrong; we replaced it with "
                "a careful AI grader that follows a written rubric. Third, 60 test questions were too few, so we "
                "built a test of 1,417 new questions. We fixed all three and rebuilt the system.")
    for i, (head, body) in enumerate([
            ("Found a hidden bug", "A default setting in the Qwen model quietly changed its confidence scores, "
                                   "by 8× the right-vs-wrong difference. Found, measured and fixed."),
            ("Fairer grading", "Word matching marked correct paraphrases wrong. Replaced by a rubric-based grader "
                               "that never sees the scores (93–98% consistent on re-grading)."),
            ("A much bigger test", "From 60 test questions to 1,417 new questions the system had never seen.")]):
        x = X0 + i * 3.6
        card(s, x, 1.6, 3.35, 2.75)
        text(s, x + 0.2, 1.75, 2.95, 0.5, [head], size=19, bold=True, color=AMBER)
        text(s, x + 0.2, 2.35, 2.95, 2.4, [body], size=15)
    card(s, X0, 4.7, XW, 1.3, fill=GREEN)
    text(s, X0 + 0.3, 4.75, XW - 0.6, 1.2,
         [[("Result: ", {"bold": True, "color": INK}), ("a clean, rebuilt system. We wrote down our predictions "
           "before the final test, so the results could not be tuned.", {"color": INK})]],
         size=17, anchor=MSO_ANCHOR.MIDDLE)

    # 4 ---------------------------------------------------------------- pipeline
    s = d.slide("Implementation: how the system works",
                "Here is the system we built. The model gives its single most likely answer, with no hidden "
                "settings. Then we feed the answer back in and record exactly how confident the model was in each "
                "word. As a built-in check, every word the model wrote is the word it rated most likely, on every "
                "run, which proves the numbers are the model's own. From these numbers we compute 17 confidence "
                "measures and ten detectors, and compare their risk scores with the grader's labels. The right "
                "column shows the file that does each step.")
    flow(s, 1.6, ["Question", "Model's best answer", "Re-read: confidence per word", "17 confidence measures",
                  "10 detectors", "Compare with graded answers"], w=1.5, gap=0.32, size=12)
    rows = [("Step", "What it does", "Code"),
            ("Answer", "The model's most likely answer, no hidden settings, up to 64 words; 5 extra varied answers",
             "scripts/v2_generate.py"),
            ("Confidence", "How sure the model was of each word, and how close the runner-up word was",
             "scripts/v2_generate.py"),
            ("Measures", "Average, lowest and spread of word confidence, and more (17 in total)", "src/features_v2.py"),
            ("Grading", "Judge with written rubric, blind to model and scores", "scripts/v2_labels.py"),
            ("Testing", "Predictions written down first; statistical tests on 1,417 new questions",
             "scripts/v2_confirm.py"),
            ("Stress tests", "Second grader, other prompts, bigger model, other methods", "scripts/v3_*.py")]
    table(s, X0, 2.9, XW, rows, [1.55, 6.0, 3.05], size=13, row_h=0.47, left_cols=(0, 1, 2))

    # 5 ---------------------------------------------------------------- design
    s = d.slide("A fair test: 3 models, 3 datasets, 10 detectors",
                "We tested three small open AI models that run on our own laptop GPU. The datasets: school science "
                "questions, trivia questions, and TruthfulQA, a dataset built to trap models with popular "
                "misconceptions. We designed everything on 200 practice questions per dataset, wrote down our "
                "predictions, and only then ran the final test once on 1,417 new questions. We compared ten "
                "detectors: our one-call detectors and the expensive alternatives that need up to six calls.")
    for i, (big, lab) in enumerate([("3", "AI models: Qwen2.5, Llama-3.2, SmolLM2 (+ a 3B model)"),
                                    ("3", "datasets: science, trivia, misconception traps"),
                                    ("1,417", "new test questions, never used for design"),
                                    ("10", "detectors compared, from 1 to 6 model calls")]):
        x = X0 + i * 2.7
        card(s, x, 1.55, 2.5, 1.85)
        stat(s, x + 0.2, 1.65, 2.15, big, lab, big_size=40)
    rows = [("Detector", "How it decides", "Model calls"),
            ("Average / weakest word", "the model's average, or lowest, confidence across the answer", "1"),
            ("Ours (DA1)", "a small trained model combining 3 (or 17) confidence measures", "1"),
            ("Self-consistency", "ask 5 more times; do the answers agree?", "6"),
            ("Self-check", "ask the model: “Is this answer true?”", "2"),
            ("Meaning check", "ask 5 more times; do the answers mean the same thing?", "6")]
    table(s, X0, 3.7, XW, rows, [2.6, 6.4, 1.6], size=13, row_h=0.45, left_cols=(0, 1))

    # 6 ---------------------------------------------------------------- main result
    s = d.slide("Result 1: it catches mistakes on facts and science",
                "This is the main result, for our DA1 detector, on new questions only. Higher bars are better: "
                "0.5 means guessing, 1.0 means perfect. On science questions it scores about 0.7 for all three "
                "models, and on trivia up to 0.81. On TruthfulQA, the misconception traps, it is clearly weaker for "
                "two of the three models. Explaining why became the main discovery of our project.")
    figure(s, CHART, X0, 1.45, 5.9, 4.9)
    x = X0 + 6.2
    stat(s, x, 1.5, 4.4, f"up to {max(triv_d2):.2f}", f"on trivia questions ({rng(triv_d2)} across the 3 models)")
    stat(s, x, 2.95, 4.4, rng(sciq_d2), "on school science, consistent across all 3 models")
    stat(s, x, 4.4, 4.4, f"{d2(TQ(MS[0])):.2f} · {d2(TQ(MS[2])):.2f}",
         "on misconception traps for Qwen and SmolLM2: the puzzle we solved next", color=AMBER)

    # 7 ---------------------------------------------------------------- 1 call vs 6
    s = d.slide("Result 2: one call beats six",
                f"This answers the DA1 question. Our one-call detector scored higher than every method that asks the "
                f"model several times, in {wins} of 6 model and dataset combinations, at one sixth of the cost. "
                f"Asking the model whether its own answer is true does worst of all on the misconception questions. "
                f"So the cheap approach is not just cheaper; it is also the most accurate of the methods we tested.")
    rows = [("Detector", "Calls", "Trap Qwen", "Trap Llama", "Trap Smol", "Sci Qwen", "Sci Llama", "Sci Smol")]
    spec = [("Average word confidence", "D0", "1"), ("Ours (DA1, 3 measures)", "D2", "1"),
            ("Self-consistency", "D4", "6"), ("Self-check “is it true?”", "D6", "2"), ("Meaning check", "D7", "6")]
    bold = set()
    for i, (name, k, calls) in enumerate(spec, 1):
        rows.append((name, calls, *[f"{det(c, k):.2f}" for c in CELLS]))
    for j, c in enumerate(CELLS):
        best = max(range(len(spec)), key=lambda i: det(c, spec[i][1]))
        bold.add((best + 1, j + 2))
    table(s, X0, 1.6, XW, rows, [2.9, 0.8] + [1.15] * 6, size=14, row_h=0.55, bold_cells=bold)
    card(s, X0, 5.05, XW, 1.1, fill=GREEN)
    text(s, X0 + 0.3, 5.1, XW - 0.6, 1.0,
         [[("Our 1-call detector beat every multi-call method in ", {"color": INK, "bold": True}),
           (f"{wins} of 6", {"color": INK, "bold": True}),
           (" settings, at 1/6 of the cost. Orange = best in column; 0.5 = guessing.", {"color": INK})]],
         size=16, anchor=MSO_ANCHOR.MIDDLE)

    # 8 ---------------------------------------------------------------- blind spot
    gi = [h2[m]["auroc_imitative"] for m in MS]
    s = d.slide("Result 3: the blind spot is learned misconceptions",
                "Why are the misconception traps hard? We split the mistakes into two kinds. Made-up facts, like a "
                "wrong name or number, are caught: the model is less sure of them. Our three-measure detector does "
                f"even better here, {rng(made_up)}. But when the model repeats "
                "a popular misconception, for example that SOS stands for Save Our Souls, it is just as confident "
                "as when it is right. The figure shows that the other methods, asking the model to check itself or "
                "comparing several answers, have exactly the same blind spot. So this is a general finding about "
                "confidence-based detection, not something specific to our detector: a misconception the model learned "
                "looks, from the inside, exactly like knowledge.")
    figure(s, FIG / "fig_v3_signals.png", X0, 1.45, 6.3, 4.4)
    x = X0 + 6.6
    go = [h2[m]["auroc_other"] for m in MS]
    stat(s, x, 1.5, 4.0, rng(go), "made-up facts (wrong names, numbers): caught by the model's confidence")
    stat(s, x, 2.95, 4.0, f"{gi[0]:.2f} · {gi[2]:.2f}", "repeated misconceptions (Qwen, SmolLM2): as confident "
         "as correct answers", color=AMBER)
    stat(s, x, 4.4, 4.0, "All methods", "share this blind spot: self-check and meaning check too", color=TEXT,
         big_size=28)
    text(s, X0, 6.0, 6.3, 0.6, ["Hollow = misconception mistakes, filled = other mistakes, for four kinds of "
                                "detector. Left of 0.5 = invisible."], size=12, color=MUTED)

    # 9 ---------------------------------------------------------------- examples
    s = d.slide("What a confident mistake looks like",
                "Here are real examples from the test. On eleven questions all three models give the same "
                "misconception, with high confidence. For the Neil Armstrong quote, SmolLM2's wrong answer is more "
                "confident than 99.7 percent of all its answers. The model is not unsure; it learned the wrong fact.")
    ex = E["_examples"]["items"][:4]
    shown = [("What exact words did Neil Armstrong say on the moon?", "“one small step for man”", "“…for a man”"),
             ("What did SOS originally stand for?", "“Save Our Souls”", "nothing"),
             ("Smallest country that is at least one square mile in area?", "“Vatican City”", "Nauru"),
             ("Where do people eat more beef than anywhere else?", "“United States”", "Argentina")]
    rows = [("Question", "All three models say", "Correct", "More confident than … % of its answers (Q / L / S)")]
    for (q, say, ok), it in zip(shown, ex):
        rows.append((q, say, ok, " / ".join(f"{100 * it['conf_pct'][m]:.0f}" for m in MS)))
    table(s, X0, 1.6, XW, rows, [4.1, 2.35, 1.55, 2.6], size=14, row_h=0.72, left_cols=(0, 1, 2))
    text(s, X0, 5.4, XW, 1.0,
         [f"{E['_examples']['n_qualifying']} test questions where all three models confidently repeat the same "
          f"misconception; the four most confident are shown."], size=13, color=MUTED)

    # 10 --------------------------------------------------------------- robustness
    s = d.slide("We tried hard to break this finding — it held",
                "A surprising result needs strong evidence, so we tried to break it in many different ways. Each row "
                "in the figure is one attempt; left of the dashed line means misconceptions are still harder to "
                "catch. We matched answer length, controlled for question difficulty and topic, used only positive "
                "answers, used a second independent grader trained on human judgements, changed the prompt, and "
                "tested a model twice as large. For Qwen and SmolLM2 the finding held in almost every check. For Llama it "
                "depends on which grader is used, and we report that openly.")
    figure(s, FIG / "fig_v3_robust.png", X0, 1.4, 4.3, 5.3)
    x = X0 + 4.6
    card(s, x, 1.45, 6.0, 2.85)
    text(s, x + 0.2, 1.55, 5.6, 0.5, ["Held up (Qwen and SmolLM2)"], size=17, bold=True, color=GREEN)
    text(s, x + 0.2, 2.1, 5.6, 2.55,
         ["Same answer length, same difficulty, same topic",
          "Positive statements only (no “no” or “not”)",
          "A second, independent grader trained on human ratings",
          "A different prompt that allows “I don't know”",
          "A model twice as large (Qwen2.5-3B)"], size=14, bullets=True, space=5)
    card(s, x, 4.5, 6.0, 1.75)
    text(s, x + 0.2, 4.6, 5.6, 1.6,
         [[("Reported openly (Llama)", {"bold": True, "color": AMBER, "size": 17})],
          "Llama's result depends on the grader: the two graders disagree on its long answers. "
          "A 120-question human check is prepared to settle it."], size=14, space=5)

    # 11 --------------------------------------------------------------- difficulty
    s = d.slide("Insight: confidence mostly tracks question difficulty",
                "Where the detector works, we asked what it actually picks up. Mostly, it is question difficulty: a "
                "hard question makes every model unsure, and every model more often wrong. Counting how many other "
                "models got a question wrong predicts mistakes even better. There is also a real answer-specific "
                "part: for the same question, the model's wrong answers are less confident than its right ones.")
    figure(s, FIG / "fig_v2_difficulty.png", X0, 1.45, 5.6, 4.6)
    x = X0 + 5.95
    stat(s, x, 1.5, 4.6, "0.77–0.82", "“how many other models got it wrong”: a strong difficulty signal")
    stat(s, x, 2.95, 4.6, "0.55–0.66", "same question, different answers: wrong ones are less confident",
         color=TEXT)
    stat(s, x, 4.4, 4.6, "Lesson", "pair confidence with a difficulty estimate for the best results", color=AMBER,
         big_size=28)

    # 12 --------------------------------------------------------------- practice
    s = d.slide("What this means in practice",
                "What would this buy a real system? On science questions, skipping the 20 percent of answers the "
                "model is least sure about cuts the error rate from 28 to 24 percent. On the misconception traps "
                "this does not help: the model's most confident answers are wrong 60 percent of the time, and "
                "most of those, 84 to 86 percent, are misconceptions. So these answers need an outside fact-check, for example "
                "looking the fact up in a trusted source.")
    figure(s, FIG / "fig_v3_coverage.png", X0, 1.45, 6.3, 4.6)
    x = X0 + 6.6
    stat(s, x, 1.5, 4.0, "28% → 24%", "science errors after skipping the 20% least confident answers (Qwen)")
    stat(s, x, 2.95, 4.0, "60% · 61%", "of the most confident misconception-trap answers are wrong (Qwen, SmolLM2)",
         color=AMBER)
    stat(s, x, 4.4, 4.0, "→ fact-check", "confident misconceptions need an outside source, not more confidence",
         color=TEXT, big_size=28)

    # 13 --------------------------------------------------------------- demo
    s = d.slide("Live demo",
                "We now switch to the live demo. First we run the whole system on 200 new questions in about a "
                "minute. Then the interactive app: the model answers a question live, each word is coloured by "
                "confidence, and the risk score is compared with the test results.")
    if SHOT.exists():
        figure(s, SHOT, X0, 1.45, 6.6, 3.0, pad=0.06)
        text(s, X0, 4.55, 6.6, 1.0, ["The demo app: a real test answer the grader marked wrong (a misconception), "
                                     "yet more confident than 82% of the model's answers."], size=13, color=MUTED)
    x = X0 + 6.9
    text(s, x, 1.45, 3.7, 5.3,
         [[("1. Run the system", {"bold": True, "color": GREEN})],
          "All steps on 200 new questions, about 1 minute",
          [("2. Ask the model", {"bold": True, "color": GREEN})],
          "Live answer, word-by-word confidence colours, risk score",
          [("3. Recorded answers", {"bold": True, "color": GREEN})],
          "Browse graded test answers, e.g. confident mistakes",
          [("python scripts/live_run.py", {"bold": True, "color": AMBER, "size": 13})],
          [("python scripts/demo_app.py", {"bold": True, "color": AMBER, "size": 13})]], size=14, space=7)

    # 14 --------------------------------------------------------------- scope
    s = d.slide("Scope and next steps",
                "Every study has a scope, and here is ours. Our grading uses an AI grader, checked against a second, "
                "independent grader trained on human ratings; a human check of 120 answers is prepared as the next "
                "step. We used models that fit our laptop GPU, up to 3 billion parameters; larger models are next. "
                "The stress tests were added after the main test, so we present them as supporting evidence. And we "
                "focused on short English question answering.")
    for i, (head, body) in enumerate([
            ("Grading", "AI grader, cross-checked by an independent grader (79–87% agreement). Next: a human check "
                        "of 120 answers, already prepared."),
            ("Model size", "Three 1–2B models plus one 3B model, sized for a laptop GPU. Next: 7B and larger."),
            ("Stress tests", "Added after the main test: supporting evidence, reported in full."),
            ("Task", "Short English question answering. Next: longer answers, retrieval-augmented systems.")]):
        x, y = X0 + (i % 2) * 5.45, 1.55 + (i // 2) * 2.35
        card(s, x, y, 5.15, 2.05)
        text(s, x + 0.25, y + 0.15, 4.7, 0.5, [head], size=19, bold=True, color=AMBER)
        text(s, x + 0.25, y + 0.75, 4.7, 1.5, [body], size=15)

    # 15 --------------------------------------------------------------- conclusion
    s = d.slide("What we contribute",
                "To conclude, three contributions. One: a cheap, working detector. With a single model call it "
                "beats methods that need six, and catches made-up facts and science mistakes. Two: a clear "
                "discovery about its limits: misconceptions the model learned look like knowledge, to every "
                "confidence-based method we tested. Three: a rigorous, reproducible system: we found and fixed a "
                "hidden bug, wrote down predictions in advance, tested on 1,417 new questions, and published all "
                "code and data. The next step is to combine confidence with looking facts up.")
    for i, (num, head, body) in enumerate([
            ("1", "Cheap detector", f"One call beats six; catches mistakes on trivia (up to {max(triv_d2):.2f}) "
                                    "and science (about 0.7)."),
            ("2", "Key discovery", "Learned misconceptions look like knowledge, to every confidence method."),
            ("3", "Rigorous system", "Hidden bug fixed, predictions written first, 1,417 new questions, all "
                                     "code public.")]):
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
         [[("Next: ", {"bold": True, "color": AMBER}),
           ("combine confidence with fact look-up (retrieval), human grading, and larger models.", {})],
          [("Paper: ", {"bold": True, "color": GREEN}), ("Learned Falsehoods Look Like Knowledge (IEEE format, "
                                                         "6 pages). ", {}),
           ("Code and data: ", {"bold": True, "color": GREEN}),
           ("github.com/gouthamkrishna404/hallucination-detection", {})]], size=15, space=6)

    # 16 --------------------------------------------------------------- thanks (layout as edited in PowerPoint)
    s = d.prs.slides.add_slide(d.blank)
    d.n += 1
    text(s, 0, 0, 13.333, 7.5, ["Thank you"], size=30, bold=True, align=PP_ALIGN.CENTER,
         anchor=MSO_ANCHOR.MIDDLE, name="Title")
    text(s, 0.15, 6.85, 0.75, 0.4, [str(d.n)], size=12, color=MUTED, align=PP_ALIGN.CENTER, name="Slide number")
    s.notes_slide.notes_text_frame.text = ("Thank you. We are happy to take questions, and we can show any part of "
                                           "the code or the demo again.")

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
