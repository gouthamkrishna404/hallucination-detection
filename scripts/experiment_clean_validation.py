"""Statistical validation of the detectors, run on either the original
generate()-derived features or the recomputed clean features.

Adds three things the earlier analyses lacked:

1. Repeated cross-validation. Every headline number so far came from a
   single CV seed. This repeats stratified 5-fold OOF evaluation over R
   different CV seeds and reports the spread, which separates a real
   effect from a lucky partition.
2. A permutation test. Labels are shuffled and the whole OOF procedure is
   re-run n_perm times, giving an empirical null distribution for AUROC
   and a p-value that makes no normality assumption. This is a stricter
   check than "the bootstrap CI excludes 0.5", because it also absorbs
   any optimism baked into the fitting procedure itself.
3. Operating points and calibration on the OOF predictions: precision,
   recall, F1, FPR at a training-selected threshold, plus ECE and Brier.

Usage:
    python scripts/experiment_clean_validation.py --model qwen2.5-1.5b-instruct --dataset truthfulqa --features clean
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, f1_score, precision_score, recall_score, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.config import CFG, set_all_seeds
from src.features_v2 import ALL_RICH_FEATURE_NAMES, ORIGINAL_THREE_FEATURES
from src.metrics_utils import compute_calibration, save_json
from src.registry import get_paths
from scripts.experiment_best_candidate import CANDIDATE_FEATURES, bootstrap_auroc_ci

NO_LENGTH = [f for f in ALL_RICH_FEATURE_NAMES if f != "answer_length"]
FEATURE_SETS = {
    "original_3feat": ORIGINAL_THREE_FEATURES,
    "candidate_4feat": CANDIDATE_FEATURES,
    "all_17feat": ALL_RICH_FEATURE_NAMES,
    "all_16feat_no_length": NO_LENGTH,
}


SCALE = False  # set by --scale; standardization is fit inside each training fold only


def _clf():
    """L2 logistic regression, optionally preceded by standardization. Without
    scaling, L2 penalizes the coefficients of small-scale features (log-probs,
    entropies) far more than large-scale ones (answer length in tokens), which
    biases the fit toward whichever feature happens to have the largest units."""
    lr = LogisticRegression(random_state=CFG.seed, max_iter=1000)
    return make_pipeline(StandardScaler(), lr) if SCALE else lr


def oof_flat(df, feats, seed, n_splits=5):
    X, y = df[feats].values, df["label"].values
    prob = np.zeros(len(df))
    for tr, te in StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed).split(X, y):
        clf = _clf().fit(X[tr], y[tr])
        prob[te] = clf.predict_proba(X[te])[:, 1]
    return y, prob


def oof_cascade(df, feats, seed, n_splits=5):
    X, y = df[feats].values, df["label"].values
    lp = df["mean_logprob"].values
    prob = np.zeros(len(df))
    for tr, te in StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed).split(X, y):
        tau = np.median(lp[tr])
        for mask_tr, mask_te in [(lp[tr] >= tau, lp[te] >= tau), (lp[tr] < tau, lp[te] < tau)]:
            idx_tr, idx_te = tr[mask_tr], te[mask_te]
            if len(idx_te) == 0:
                continue
            fit_idx = idx_tr if (len(idx_tr) >= 10 and len(np.unique(y[idx_tr])) > 1) else tr
            clf = _clf().fit(X[fit_idx], y[fit_idx])
            prob[idx_te] = clf.predict_proba(X[idx_te])[:, 1]
    return y, prob


DETECTORS = {"flat": oof_flat, "cascade": oof_cascade}


def operating_point(y, prob):
    """Threshold maximising Youden's J on the same OOF predictions; reported as a
    descriptive operating point, not a tuned held-out result."""
    fpr, tpr, thr = roc_curve(y, prob)
    j = int(np.argmax(tpr - fpr))
    t = thr[j]
    pred = (prob >= t).astype(int)
    tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    return {
        "threshold": float(t),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "fpr": float(fp / (fp + tn)) if (fp + tn) else float("nan"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--dataset", default="truthfulqa")
    ap.add_argument("--features", default="clean", choices=["clean", "rich"])
    ap.add_argument("--repeats", type=int, default=20)
    ap.add_argument("--permutations", type=int, default=500)
    ap.add_argument("--scale", action="store_true")
    ap.add_argument("--labels", default="tokenf1", choices=["tokenf1", "semantic", "containment"])
    args = ap.parse_args()

    global SCALE
    SCALE = args.scale
    set_all_seeds()
    paths = get_paths(args.model, args.dataset)
    fname = "single_pass_features_clean.csv" if args.features == "clean" else "single_pass_features_rich.csv"
    df = pd.read_csv(paths.features / fname)
    if args.labels == "containment":
        # Near-exact reference label for SciQ: the correct short answer appears in the
        # generation and no distractor does. Validated against both proxies in
        # scripts/experiment_labeler_adjudication.py.
        con = pd.read_csv(paths.features / "single_pass_features_containment_labels.csv")[["qid", "containment_label"]]
        df = df.drop(columns=["label"]).merge(con, on="qid").rename(columns={"containment_label": "label"})
    elif args.labels == "semantic":
        # Swap the target for the embedding-based labeler, which (unlike token-F1)
        # shows no measurable answer-length bias. Features are untouched.
        sem = pd.read_csv(paths.features / "single_pass_features_semantic_labels.csv")[["qid", "semantic_label"]]
        df = df.drop(columns=["label"]).merge(sem, on="qid").rename(columns={"semantic_label": "label"})
    print(f"[{args.model}/{args.dataset}] features={args.features} scaled={args.scale} labels={args.labels} n={len(df)} "
          f"hallucinated={int(df.label.sum())}")

    results = {"features": args.features, "n": len(df), "n_positive": int(df.label.sum()),
               "repeats": args.repeats, "permutations": args.permutations, "detectors": {}}

    for det_name, det in DETECTORS.items():
        for fs_name, feats in FEATURE_SETS.items():
            key = f"{det_name}__{fs_name}"
            # repeated CV across CV seeds
            aurocs = []
            for r in range(args.repeats):
                y, p = det(df, feats, seed=1000 + r)
                aurocs.append(roc_auc_score(y, p))
            aurocs = np.array(aurocs)

            # reference partition (the seed used by the earlier analyses)
            y, p = det(df, feats, seed=CFG.seed)
            ci = bootstrap_auroc_ci(y, p)
            obs = ci["point_estimate"]

            # permutation test on the reference partition
            rng = np.random.RandomState(CFG.seed)
            null = np.empty(args.permutations)
            perm_df = df.copy()
            for i in range(args.permutations):
                perm_df["label"] = rng.permutation(df["label"].values)
                yp, pp = det(perm_df, feats, seed=CFG.seed)
                null[i] = roc_auc_score(yp, pp)
            p_value = float((np.sum(null >= obs) + 1) / (args.permutations + 1))

            cal = compute_calibration(y, p, n_bins=10)
            results["detectors"][key] = {
                "auroc_reference_seed": obs,
                "bootstrap_ci": [ci["ci_lower_2.5pct"], ci["ci_upper_97.5pct"]],
                "bootstrap_excludes_chance": ci["excludes_chance"],
                "repeated_cv_mean": float(aurocs.mean()),
                "repeated_cv_std": float(aurocs.std()),
                "repeated_cv_min": float(aurocs.min()),
                "repeated_cv_max": float(aurocs.max()),
                "repeated_cv_frac_above_0.5": float((aurocs > 0.5).mean()),
                "permutation_p_value": p_value,
                "permutation_null_mean": float(null.mean()),
                "permutation_null_95pct": float(np.percentile(null, 95)),
                "operating_point": operating_point(y, p),
                "ece": cal["ece"], "brier": cal["brier_score"],
            }
            r = results["detectors"][key]
            print(f"  {key:34s} AUROC={obs:.3f} CI=[{ci['ci_lower_2.5pct']:.3f},{ci['ci_upper_97.5pct']:.3f}] "
                  f"repCV={aurocs.mean():.3f}±{aurocs.std():.3f} [{aurocs.min():.3f},{aurocs.max():.3f}] "
                  f"perm_p={p_value:.3f}")

    suffix = ("_scaled" if args.scale else "") + {"semantic": "_semlabels", "containment": "_containment"}.get(args.labels, "")
    out = paths.analysis / f"clean_validation_{args.features}{suffix}.json"
    save_json(results, out)
    print(f"Saved -> {out}")


if __name__ == "__main__":
    main()
