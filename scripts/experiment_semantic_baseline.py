"""Semantic-entropy self-consistency baseline: a stronger, literature-grounded
alternative to the lexical (token-F1) agreement-score baseline, computed
from the SAME 5 cached samples in self_consistency_features.csv -- no new
LLM calls.

Rationale (see src/semantic_similarity.py and README "Semantic-entropy
baseline"): the lexical agreement score used in
experiment_generate_self_consistency.py treats two paraphrases of the same
answer as disagreeing if they share few words. Farquhar et al. (2024,
Nature, cited in the DA1 literature review) show that clustering samples
by *meaning* rather than wording produces a stronger uncertainty signal
("semantic entropy"). This script computes a cosine-similarity-clustering
approximation of that idea over the already-generated 5-sample sets, and
evaluates it exactly like every other baseline (logistic regression, same
train/eval split, same metrics) for a direct comparison.

Usage:
    python scripts/experiment_semantic_baseline.py --model qwen2.5-1.5b-instruct --dataset truthfulqa
"""
import argparse
import ast
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from tqdm import tqdm

from src.config import CFG, set_all_seeds
from src.metrics_utils import compute_metrics, plot_confusion_matrix, plot_roc_curves, save_json
from src.registry import get_paths
from src.run_logger import log_event
from src.semantic_similarity import semantic_entropy, semantic_majority_answer
from src.train_eval import fit_and_predict
from scripts.experiment_train_evaluate import get_qid_split


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--dataset", default="truthfulqa")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    set_all_seeds()
    paths = get_paths(args.model, args.dataset)
    out_path = paths.features / "semantic_self_consistency_features.csv"

    if out_path.exists() and not args.force:
        print(f"Already exists, skipping (use --force to regenerate): {out_path}")
    else:
        log_event("experiment_semantic_baseline", args.model, args.dataset, "started")
        sc = pd.read_csv(paths.features / "self_consistency_features.csv")

        records = []
        for _, row in tqdm(sc.iterrows(), total=len(sc), desc=f"Semantic entropy [{args.model}/{args.dataset}]"):
            answers = ast.literal_eval(row["sampled_answers"])
            sem_entropy = semantic_entropy(answers)
            sem_majority = semantic_majority_answer(answers)
            records.append({
                "qid": row["qid"],
                "semantic_entropy": sem_entropy,
                "semantic_majority_answer": sem_majority,
                # reuse the SAME label as the lexical self-consistency run (same medoid-based
                # proxy judge, same reference set) so this is a pure feature-quality comparison,
                # not a relabeling -- relabeling is handled separately by experiment_relabel_semantic.py
                "label": row["label"],
            })

        out_df = pd.DataFrame.from_records(records)
        out_df.to_csv(out_path, index=False)
        print(f"Saved {len(out_df)} rows -> {out_path}")
        log_event("experiment_semantic_baseline", args.model, args.dataset, "completed", n_questions=len(out_df))

    # --- Evaluate: same train/eval split as every other method (based on single_pass labels) ---
    sp = pd.read_csv(paths.features / "single_pass_features.csv")
    sem = pd.read_csv(out_path)
    train_qids, eval_qids = get_qid_split(sp)
    sem_train = sem[sem.qid.isin(train_qids)].reset_index(drop=True)
    sem_eval = sem[sem.qid.isin(eval_qids)].reset_index(drop=True)

    clf, y_eval, y_prob = fit_and_predict(sem_train, sem_eval, ["semantic_entropy"])
    metrics = compute_metrics(y_eval, y_prob)
    metrics["calls_per_question"] = CFG.self_consistency_n_samples
    metrics["feature_cols"] = ["semantic_entropy"]

    plot_confusion_matrix(
        __import__("numpy").array(metrics["confusion_matrix"]),
        "semantic_self_consistency", paths.plots / "confusion_semantic_self_consistency.png",
    )

    # Compare against the lexical self-consistency baseline already on disk
    lexical_metrics_path = paths.metrics / "all_metrics.json"
    comparison = {"semantic_self_consistency": metrics}
    if lexical_metrics_path.exists():
        import json
        with open(lexical_metrics_path, encoding="utf-8") as f:
            existing = json.load(f)
        comparison["lexical_self_consistency"] = existing.get("baseline_self_consistency")
        existing["semantic_self_consistency"] = metrics
        save_json(existing, lexical_metrics_path)

        # Build ROC comparison using actual predicted probabilities from both methods
        sc_eval_lexical = pd.read_csv(paths.features / "self_consistency_features.csv")
        sc_eval_lexical = sc_eval_lexical[sc_eval_lexical.qid.isin(eval_qids)].reset_index(drop=True)
        _, y_eval_lex, y_prob_lex = fit_and_predict(
            pd.read_csv(paths.features / "self_consistency_features.csv").pipe(lambda d: d[d.qid.isin(train_qids)]).reset_index(drop=True),
            sc_eval_lexical, ["agreement_score"],
        )
        plot_roc_curves({
            "Lexical agreement (5 calls)": (y_eval_lex, y_prob_lex),
            "Semantic entropy (5 calls)": (y_eval, y_prob),
        }, paths.plots / "roc_semantic_vs_lexical.png")

    print(f"[semantic_self_consistency] n_eval={metrics['n']} acc={metrics['accuracy']:.3f} "
          f"auroc={metrics['auroc']:.3f} f1={metrics['f1']:.3f}")
    if "lexical_self_consistency" in comparison and comparison["lexical_self_consistency"]:
        lex_auroc = comparison["lexical_self_consistency"]["auroc"]
        print(f"[lexical_self_consistency]  auroc={lex_auroc:.3f}  (for comparison, same eval set)")

    save_json(comparison, paths.analysis / "semantic_vs_lexical_comparison.json")
    print(f"Saved comparison -> {paths.analysis / 'semantic_vs_lexical_comparison.json'}")


if __name__ == "__main__":
    main()
