"""Label audit, part 2: how much of the apparent detection failure is
explained by AMBIGUOUS labels specifically (borderline token-F1 margin
between the correct- and incorrect-reference similarity), as opposed to a
genuine absence of signal?

For every question, the token-F1 labeler's confidence in its own decision
is |correct_sim - incorrect_sim| -- a small margin means the label itself
is a coin flip between "closer to a correct paraphrase" and "closer to an
incorrect one." This script buckets questions by that margin and
re-evaluates the best-available detector on progressively more
confidently-labeled subsets only.

If AUROC rises sharply as ambiguous labels are excluded, the original
"near chance" result is partly a labeling-noise artifact. If it stays
flat, the detector genuinely isn't separating the classes even where the
label itself is reliable -- which is what src/labeling.py's "margin>=0.4"
spot-check already suggested during the original DA1 analysis, now
redone properly with the expanded feature set and reported per bucket.

Usage:
    python scripts/experiment_label_margin_audit.py --model qwen2.5-1.5b-instruct --dataset truthfulqa
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from src.config import set_all_seeds
from src.features_v2 import ORIGINAL_THREE_FEATURES
from src.metrics_utils import compute_metrics, save_json
from src.registry import get_paths
from src.train_eval import fit_and_predict
from scripts.experiment_best_candidate import CANDIDATE_FEATURES
from scripts.experiment_train_evaluate import get_qid_split


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--dataset", default="truthfulqa")
    args = ap.parse_args()

    set_all_seeds()
    paths = get_paths(args.model, args.dataset)
    rich = pd.read_csv(paths.features / "single_pass_features_rich.csv")
    rich["label_margin"] = (rich["correct_sim"] - rich["incorrect_sim"]).abs()

    sp = pd.read_csv(paths.features / "single_pass_features.csv")
    train_qids, eval_qids = get_qid_split(sp)

    print(f"[{args.model}/{args.dataset}] label margin distribution:")
    print(rich["label_margin"].describe())

    thresholds = [0.0, 0.1, 0.2, 0.3, 0.4]
    results = {}
    for thresh in thresholds:
        confident = rich[rich["label_margin"] >= thresh]
        train_df = confident[confident.qid.isin(train_qids)].reset_index(drop=True)
        eval_df = confident[confident.qid.isin(eval_qids)].reset_index(drop=True)

        row = {"threshold": thresh, "n_total": len(confident), "n_train": len(train_df), "n_eval": len(eval_df),
               "pct_of_full_dataset": 100 * len(confident) / len(rich)}

        for name, feats in [("original_3feat", ORIGINAL_THREE_FEATURES), ("candidate_4feat", CANDIDATE_FEATURES)]:
            if len(train_df) < 20 or len(eval_df) < 10 or eval_df["label"].nunique() < 2:
                row[f"{name}_auroc"] = None
                continue
            _, y_eval, y_prob = fit_and_predict(train_df, eval_df, feats)
            m = compute_metrics(y_eval, y_prob)
            row[f"{name}_auroc"] = m["auroc"]

        results[f"margin_gte_{thresh}"] = row
        o = row.get("original_3feat_auroc")
        c = row.get("candidate_4feat_auroc")
        o_str = f"{o:.3f}" if o is not None else "n/a (too few eval questions)"
        c_str = f"{c:.3f}" if c is not None else "n/a (too few eval questions)"
        print(f"  margin>={thresh:.1f}: n={len(confident):3d} ({row['pct_of_full_dataset']:.0f}% of data)  "
              f"original_3feat_auroc={o_str}  candidate_4feat_auroc={c_str}")

    out_path = paths.analysis / "label_margin_audit.json"
    save_json(results, out_path)
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
