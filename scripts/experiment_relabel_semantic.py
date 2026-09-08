"""Robustness check: does the DA1/Llama/SciQ finding depend on the specific
choice of proxy labeling method?

Re-labels every single-pass generation using embedding cosine similarity
(src/semantic_similarity.py:semantic_label) instead of token-F1 lexical
overlap (src/labeling.py:label_answer), holding everything else fixed --
same generated answers (no new LLM calls), same features, same classifier,
same train/eval split -- and re-evaluates. If the AUROC conclusion is
unchanged under a completely different labeling method, that's direct
evidence the original finding is a property of the features/model/dataset,
not an artifact of the lexical proxy judge.

Usage:
    python scripts/experiment_relabel_semantic.py --model qwen2.5-1.5b-instruct --dataset truthfulqa
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from sklearn.metrics import cohen_kappa_score
from tqdm import tqdm

from src.config import set_all_seeds
from src.datasets import load_subset
from src.metrics_utils import compute_metrics, save_json
from src.registry import get_paths
from src.run_logger import log_event
from src.semantic_similarity import semantic_label
from src.train_eval import fit_and_predict
from scripts.experiment_train_evaluate import get_qid_split


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--dataset", default="truthfulqa")
    ap.add_argument("--subset-size", type=int, default=200)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    set_all_seeds()
    paths = get_paths(args.model, args.dataset)
    out_path = paths.features / "single_pass_features_semantic_labels.csv"

    sp = pd.read_csv(paths.features / "single_pass_features.csv")

    if out_path.exists() and not args.force:
        print(f"Already exists, skipping relabel (use --force to regenerate): {out_path}")
        relabeled = pd.read_csv(out_path)
    else:
        log_event("experiment_relabel_semantic", args.model, args.dataset, "started")
        from src.config import CFG

        df = load_subset(args.dataset, args.subset_size, CFG.seed)
        ref_map = df.set_index("qid")[["best_answer", "correct_answers", "incorrect_answers"]].to_dict("index")

        records = []
        for _, row in tqdm(sp.iterrows(), total=len(sp), desc=f"Semantic relabel [{args.model}/{args.dataset}]"):
            refs = ref_map[row["qid"]]
            lab = semantic_label(row["generated_answer"], refs["best_answer"], refs["correct_answers"], refs["incorrect_answers"])
            records.append({"qid": row["qid"], **lab})

        lab_df = pd.DataFrame.from_records(records)
        relabeled = sp.merge(lab_df, on="qid")
        relabeled.to_csv(out_path, index=False)
        print(f"Saved {len(relabeled)} rows -> {out_path}")

    # --- Agreement between the two labeling methods ---
    agreement_rate = float((relabeled["label"] == relabeled["semantic_label"]).mean())
    kappa = float(cohen_kappa_score(relabeled["label"], relabeled["semantic_label"]))
    print(f"Label agreement (token-F1 vs semantic): {agreement_rate:.3f}  (Cohen's kappa={kappa:.3f})")
    print(f"Token-F1 label balance:  {relabeled['label'].value_counts().to_dict()}")
    print(f"Semantic label balance:  {relabeled['semantic_label'].value_counts().to_dict()}")

    # --- Re-evaluate the SAME features/split under the semantic label ---
    train_qids, eval_qids = get_qid_split(sp)  # split fixed by original token-F1 labels, held constant
    train_df = relabeled[relabeled.qid.isin(train_qids)].reset_index(drop=True)
    eval_df = relabeled[relabeled.qid.isin(eval_qids)].reset_index(drop=True)

    _, y_eval_tok, y_prob_tok = fit_and_predict(train_df, eval_df, ["mean_logprob", "min_logprob", "mean_entropy"], label_col="label")
    metrics_token = compute_metrics(y_eval_tok, y_prob_tok)

    _, y_eval_sem, y_prob_sem = fit_and_predict(train_df, eval_df, ["mean_logprob", "min_logprob", "mean_entropy"], label_col="semantic_label")
    metrics_semantic = compute_metrics(y_eval_sem, y_prob_sem)

    print(f"\n[proposed_all_three, token-F1 labels]    acc={metrics_token['accuracy']:.3f}  auroc={metrics_token['auroc']:.3f}")
    print(f"[proposed_all_three, semantic labels]     acc={metrics_semantic['accuracy']:.3f}  auroc={metrics_semantic['auroc']:.3f}")

    result = {
        "label_agreement_rate": agreement_rate,
        "cohen_kappa": kappa,
        "token_f1_label_balance": relabeled["label"].value_counts().to_dict(),
        "semantic_label_balance": relabeled["semantic_label"].value_counts().to_dict(),
        "proposed_detector_under_token_f1_labels": metrics_token,
        "proposed_detector_under_semantic_labels": metrics_semantic,
    }
    save_json(result, paths.analysis / "label_robustness_check.json")
    print(f"\nSaved -> {paths.analysis / 'label_robustness_check.json'}")
    log_event("experiment_relabel_semantic", args.model, args.dataset, "completed",
              agreement_rate=agreement_rate, kappa=kappa,
              auroc_token=metrics_token["auroc"], auroc_semantic=metrics_semantic["auroc"])


if __name__ == "__main__":
    main()
