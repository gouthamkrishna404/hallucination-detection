"""Interactive demo: type a factual question, get a live single LLM
generation, its uncertainty features, a hallucination probability from the
trained classifier, and a token-level confidence visualization.

    Question -> Generated Answer -> Hallucination Probability -> PASS/WARN/FLAG

Loads a real model (from configs/models.yaml) and a real trained
classifier (results/<model>/<dataset>/models/proposed_all_three.joblib,
produced by experiment_train_evaluate.py) -- nothing here is mocked. If no
trained classifier exists yet for the selected model/dataset, run
experiment_generate_single_pass.py -> experiment_generate_self_consistency.py
-> experiment_train_evaluate.py for that pair first.

Usage:
    python scripts/demo_app.py --model qwen2.5-1.5b-instruct --dataset truthfulqa
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import gradio as gr
import joblib
import numpy as np
import pandas as pd

from src.config import CFG, set_all_seeds
from src.features import extract_features
from src.model_utils import LLMGenerator
from src.registry import get_paths, resolve_model
from src.token_viz import render_token_html, suspicious_tokens

PASS_THRESHOLD = 0.4
WARN_THRESHOLD = 0.7


def load_threshold_reference(paths):
    """Load the eval-set hallucination-probability distribution so we can
    describe a new prediction relative to training data, not just in the
    abstract."""
    pred_path = paths.predictions / "proposed_all_three_predictions.csv"
    if not pred_path.exists():
        return None
    return pd.read_csv(pred_path)


def build_explanation(feats: dict, prob: float, decision: str, susp_tokens: list[dict], ref_df) -> str:
    lines = [f"**Decision: {decision}** (hallucination probability = {prob:.3f})", ""]

    lines.append(
        f"- mean logprob = `{feats['mean_logprob']:.3f}`, "
        f"min logprob = `{feats['min_logprob']:.3f}`, "
        f"mean entropy = `{feats['mean_entropy']:.3f}` nats"
    )

    if ref_df is not None and len(ref_df) > 0:
        pct = float((ref_df["hallucination_probability"] < prob).mean() * 100)
        lines.append(f"- this probability is higher than {pct:.0f}% of the {len(ref_df)} held-out eval questions")

    if susp_tokens:
        tok_str = ", ".join(f"`{t['token'].strip()}` (P={t['probability']:.2f})" for t in susp_tokens[:5])
        lines.append(f"- suspicious low-confidence token(s): {tok_str}")
    else:
        lines.append("- no individual token fell below the low-confidence threshold")

    lines.append("")
    if decision == "FLAG":
        lines.append("Flagged: the aggregate uncertainty features place this answer in the "
                      "high-probability-of-hallucination region learned from the training split.")
    elif decision == "WARN":
        lines.append("Borderline: uncertainty features are ambiguous -- worth a second look, "
                      "not confident enough either way to auto-pass or auto-flag.")
    else:
        lines.append("Passed: uncertainty features resemble the training split's grounded examples.")

    lines.append("")
    lines.append(
        "_Caveat (see README): the classifier's ground-truth labels during training were produced by an "
        "automatic reference-similarity proxy, not human review, and on TruthfulQA the underlying features "
        "were found to carry only chance-level signal for the models tested. Treat this probability as a "
        "demonstration of the pipeline, not a validated hallucination detector._"
    )
    return "\n".join(lines)


def make_predict_fn(generator: LLMGenerator, clf, ref_df):
    def predict(question: str):
        if not question or not question.strip():
            return "_Enter a question above._", "", "", None

        set_all_seeds()
        result = generator.generate_with_scores(
            question=question.strip(),
            max_new_tokens=CFG.single_pass_max_new_tokens,
            do_sample=CFG.single_pass_do_sample,
        )
        feats = extract_features(result.token_logprobs, result.token_entropies)
        X = np.array([[feats["mean_logprob"], feats["min_logprob"], feats["mean_entropy"]]])
        prob = float(clf.predict_proba(X)[0, 1])

        if prob < PASS_THRESHOLD:
            decision = "PASS"
        elif prob < WARN_THRESHOLD:
            decision = "WARN"
        else:
            decision = "FLAG"

        susp = suspicious_tokens(result.token_strs, result.token_logprobs)
        token_html = render_token_html(result.token_strs, result.token_logprobs, result.token_entropies)
        explanation = build_explanation(feats, prob, decision, susp, ref_df)

        feature_table = (
            f"| feature | value |\n|---|---|\n"
            f"| mean logprob | {feats['mean_logprob']:.4f} |\n"
            f"| min logprob | {feats['min_logprob']:.4f} |\n"
            f"| mean entropy | {feats['mean_entropy']:.4f} nats |\n"
            f"| hallucination probability | {prob:.4f} |\n"
        )

        answer_md = f"**Generated answer:** {result.answer_text}"
        return answer_md, feature_table, explanation, token_html

    return predict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen2.5-1.5b-instruct")
    ap.add_argument("--dataset", default="truthfulqa")
    ap.add_argument("--share", action="store_true")
    args = ap.parse_args()

    paths = get_paths(args.model, args.dataset)
    clf_path = paths.models / "proposed_all_three.joblib"
    if not clf_path.exists():
        raise SystemExit(
            f"No trained classifier at {clf_path}.\n"
            f"Run first:\n"
            f"  python scripts/experiment_generate_single_pass.py --model {args.model} --dataset {args.dataset}\n"
            f"  python scripts/experiment_generate_self_consistency.py --model {args.model} --dataset {args.dataset}\n"
            f"  python scripts/experiment_train_evaluate.py --model {args.model} --dataset {args.dataset}"
        )

    print(f"Loading model {resolve_model(args.model)['checkpoint']} ...")
    generator = LLMGenerator.from_registry(args.model)
    clf = joblib.load(clf_path)
    ref_df = load_threshold_reference(paths)
    predict = make_predict_fn(generator, clf, ref_df)

    display_name = resolve_model(args.model).get("display_name", args.model)

    with gr.Blocks(title="Hallucination Flagging Demo") as demo:
        gr.Markdown(
            f"# Logprob-Based Hallucination Flagging\n"
            f"Model: **{display_name}** | Dataset trained on: **{args.dataset}** | "
            f"1 LLM call per question, no resampling.\n\n"
            f"Question → generated answer → mean/min logprob + entropy → logistic regression → "
            f"hallucination probability → PASS / WARN / FLAG."
        )
        with gr.Row():
            question_box = gr.Textbox(label="Factual question", placeholder="e.g. Is the Great Wall of China visible from space?", scale=4)
            submit_btn = gr.Button("Ask", variant="primary", scale=1)

        answer_out = gr.Markdown(label="Answer")
        with gr.Row():
            with gr.Column(scale=1):
                features_out = gr.Markdown(label="Features")
            with gr.Column(scale=2):
                explanation_out = gr.Markdown(label="Explanation")

        gr.Markdown("### Token-level uncertainty")
        token_viz_out = gr.HTML()

        submit_btn.click(predict, inputs=question_box, outputs=[answer_out, features_out, explanation_out, token_viz_out])
        question_box.submit(predict, inputs=question_box, outputs=[answer_out, features_out, explanation_out, token_viz_out])

        gr.Examples(
            examples=[
                "Is the Great Wall of China visible from space?",
                "What happens if you swallow gum?",
                "What percentage of the brain does a human typically use?",
                "Who wrote the play 'Romeo and Juliet'?",
                "What is the most popular sport in Japan?",
            ],
            inputs=question_box,
        )

    demo.launch(share=args.share)


if __name__ == "__main__":
    main()
