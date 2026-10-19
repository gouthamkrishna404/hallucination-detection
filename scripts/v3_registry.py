"""Results registry: every headline claim of the v3 paper with its value, source file, JSON path,
producing script and evidential status. Values are read from the analysis files, never typed.

Status vocabulary:
  preregistered  fixed in PREREGISTRATION.md before the confirm labels were compared with any score
  exploratory    post-hoc (v3 audit); chosen after the confirm results were known
  corrected      a v1/v2 statement that this round found to be wrong; the entry gives the right value

Output: RESULTS_REGISTRY.json (repo root)

Usage:
    python scripts/v3_registry.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import PROJECT_ROOT

MS = ["qwen2.5-1.5b-instruct", "llama-3.2-1b-instruct", "smollm2-1.7b-instruct"]
SRC = {"confirm": ("analysis_v2/confirm.json", "scripts/v2_confirm.py"),
       "audit": ("analysis_v3/audit.json", "scripts/v3_audit.py"),
       "ext": ("analysis_v3/extensions.json", "scripts/v3_eval.py"),
       "within": ("analysis_v3/within.json", "scripts/v3_within.py"),
       "extra": ("analysis_v3/extra.json", "scripts/v3_extra.py"),
       "const": ("analysis_v3/constants.json", "scripts/v3_constants.py")}
DATA = {k: json.loads((PROJECT_ROOT / f).read_text()) for k, (f, _) in SRC.items()}


def get(src, *path):
    d = DATA[src]
    for p in path:
        d = d[p]
    return d


def entry(cid, claim, status, src, paths):
    """paths: {label: json path tuple}; missing paths are recorded as pending, not dropped."""
    vals = {}
    for label, path in paths.items():
        try:
            v = get(src, *path)
            vals[label] = round(v, 4) if isinstance(v, float) else v
        except (KeyError, TypeError):
            vals[label] = "pending"
    f, script = SRC[src]
    return {"id": cid, "claim": claim, "status": status, "values": vals, "source": f,
            "paths": {k: "/".join(map(str, p)) for k, p in paths.items()}, "script": script}


def per_model(dataset, *tail):
    return {m: (f"{m}|{dataset}", *tail) for m in MS}


R = [
    entry("H1.sciq", "D0 (-mean token log-prob) detects SciQ errors in all three models (weak).", "preregistered",
          "confirm", {**{f"{m} auroc": p for m, p in per_model("sciq", "detectors", "D0", "auroc").items()},
                      **{f"{m} p_holm": p for m, p in per_model("sciq", "detectors", "D0", "p_holm").items()}}),
    entry("H1.tqa", "D0 on TruthfulQA: chance for Qwen and SmolLM2, weak for Llama.", "preregistered",
          "confirm", {**{f"{m} auroc": p for m, p in per_model("truthfulqa", "detectors", "D0", "auroc").items()},
                      **{f"{m} p_holm": p for m, p in per_model("truthfulqa", "detectors", "D0", "p_holm").items()}}),
    entry("H1.containment", "SciQ H1 holds under the containment label (preregistered sensitivity, first run in v3).",
          "preregistered", "audit", {m: ("containment", m, "detectors", "D0", "auroc") for m in MS}),
    entry("H2.prereg", "Imitative TruthfulQA errors are scored like correct answers; other errors partly detectable.",
          "preregistered", "confirm", {**{f"{m} delta": (f"{m}|truthfulqa", "H2_error_type", "delta") for m in MS},
                                       **{f"{m} p_holm": (f"{m}|truthfulqa", "H2_error_type", "p_holm") for m in MS}}),
    entry("H2.length", "Imitative gap within answer-length quintiles.", "exploratory", "audit",
          {m: (f"{m}|truthfulqa", "h2_controls", "length_quintile", "delta") for m in MS}),
    entry("H2.regression", "Imitative minus other, SD of D0, controlling length, truncation, question log-prob and "
          "length, others-wrong, category FE (negative = imitative more confident).", "exploratory", "audit",
          {m: (f"{m}|truthfulqa", "h2_controls", "regression", "diff") for m in MS}),
    entry("H2.affirmative", "Imitative gap of D0 within affirmative (non-negated, non-refusal) answers.", "exploratory",
          "ext", {m: (f"{m}|truthfulqa", "form", "affirmative", "error_type", "D0", "delta") for m in MS}),
    entry("H2.nli_types", "Imitative gap with NLI error types instead of judge types.", "exploratory", "ext",
          {**{f"{m} delta": ("_independent_h2", m, "claude_labels_nli_types", "delta") for m in MS},
           **{f"{m} p_holm": ("_independent_h2", m, "claude_labels_nli_types", "p_holm") for m in MS}}),
    entry("H2.independent", "Imitative gap with truth-judge labels and NLI types (no input from the primary judge).",
          "exploratory", "ext", {m: ("_independent_h2", m, "truthjudge_labels_nli_types", "delta") for m in MS}),
    entry("H2.prompt2", "Imitative gap under a neutral prompt, independent labels and types.", "exploratory", "extra",
          {m: ("prompt2", m, "H2", "delta") for m in MS}),
    entry("H2.scale", "Imitative gap for Qwen2.5-3B (bf16), independent labels and types.", "exploratory", "extra",
          {"qwen2.5-3b-instruct": ("scale", "qwen2.5-3b-instruct", "H2", "delta")}),
    entry("signals.ptrue", "P(True) misses imitative errors too (AUROC imitative vs other, TruthfulQA).", "exploratory",
          "ext", {**{f"{m} imitative": (f"{m}|truthfulqa", "error_type", "D6", "auroc_imitative") for m in MS},
                  **{f"{m} other": (f"{m}|truthfulqa", "error_type", "D6", "auroc_other") for m in MS}}),
    entry("signals.nli_se", "NLI semantic entropy misses imitative errors too.", "exploratory", "ext",
          {**{f"{m} imitative": (f"{m}|truthfulqa", "error_type", "D7", "auroc_imitative") for m in MS},
           **{f"{m} other": (f"{m}|truthfulqa", "error_type", "D7", "auroc_other") for m in MS}}),
    entry("ptrue.negation", "P(True) within negated TruthfulQA answers (negation bias).", "exploratory", "ext",
          {m: (f"{m}|truthfulqa", "form", "negated_auroc", "D6") for m in MS}),
    entry("D2.gain", "Dev-trained 3-feature LR minus D0, paired.", "preregistered", "confirm",
          {f"{m}|{d}": (f"{m}|{d}", "paired_vs_D0", "D2", "delta") for m in MS for d in ["truthfulqa", "sciq"]}),
    entry("D2.length", "D2 added to a dev-trained length model (paired delta).", "exploratory", "audit",
          {f"{m}|{d}": (f"{m}|{d}", "length", "D2_plus_length_minus_length", "delta") for m in MS
           for d in ["truthfulqa", "sciq"]}),
    entry("SE.nli_vs_D0", "NLI semantic entropy (D7) minus D0, paired; ties on SciQ (corrects v2 'SC worse').",
          "corrected", "ext", {f"{m}|{d}": (f"{m}|{d}", "paired_vs_D0", "D7", "delta") for m in MS
                               for d in ["truthfulqa", "sciq"]}),
    entry("H5.proxy", "Others-wrong difficulty proxy AUROC (no uncertainty used).", "preregistered", "confirm",
          {f"{m}|{d}": ("_H5_own_vs_other", d, f"{m}<-others_wrong_count", "auroc") for m in MS
           for d in ["truthfulqa", "sciq"]}),
    entry("within.sciq", "Within-question AUROC of mean log-prob on SciQ samples (conservative corrected labels).",
          "exploratory", "within", {**{f"{m} containment": (m, "containment", "mean_logprob", "within_question_auroc")
                                       for m in MS},
                                    **{f"{m} other_anchors": (m, "hybrid_other_anchors", "mean_logprob",
                                                              "within_question_auroc") for m in MS}}),
    entry("practice.top20", "Error rate among the 20% most confident TruthfulQA answers (D0).", "exploratory", "audit",
          {m: (f"{m}|truthfulqa", "confident", "top20pct_confident", "error_rate") for m in MS}),
    entry("practice.risk80", "Error rate at 80% coverage (D0) vs base rate, TruthfulQA.", "exploratory", "audit",
          {**{f"{m} risk@0.8": (f"{m}|truthfulqa", "confident", "risk_coverage_D0", "risk@0.8") for m in MS},
           **{f"{m} base": (f"{m}|truthfulqa", "confident", "base_rate") for m in MS}}),
    entry("labels.truthjudge", "Agreement of the truth-judge with the primary judge (TruthfulQA confirm).",
          "exploratory", "ext", {f"{m} kappa": ("_labels", f"{m}|truthfulqa", "agreement_confirm", "kappa") for m in MS}),
    entry("labels.containment", "Containment label vs judge on SciQ confirm: false-correct and false-wrong rates "
          "(corrects v2 'never accepted a wrong answer').", "corrected", "within",
          {**{f"{m} accepts_wrong": (m, "validation_on_greedy", "containment", "false_correct") for m in MS},
           **{f"{m} rejects_correct": (m, "validation_on_greedy", "containment", "false_wrong") for m in MS}}),
    entry("contamination", "Qwen repetition-penalty contamination of v1 scores (mean |shift| and ratio to class "
          "separation).", "corrected", "const", {"shift": ("contamination", "mean_abs_shift"),
                                                 "ratio": ("contamination", "ratio")}),
]

if __name__ == "__main__":
    out = PROJECT_ROOT / "RESULTS_REGISTRY.json"
    out.write_text(json.dumps(R, indent=1, ensure_ascii=False))
    pending = [r["id"] for r in R if "pending" in r["values"].values()]
    print(f"wrote {out}: {len(R)} entries; pending: {pending}")
