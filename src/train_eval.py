"""Logistic-regression training/evaluation and ablation utilities.

Used identically for the proposed 3-feature detector, the mean-logprob-only
baseline, and the self-consistency-agreement baseline: same split, same
classifier family (report Section 4.2: "logistic regression is preferred
because the feature vector has only three dimensions").
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from .config import CFG


def split_train_eval(df: pd.DataFrame, label_col: str = "label"):
    stratify = df[label_col] if df[label_col].nunique() > 1 else None
    train_df, eval_df = train_test_split(
        df,
        train_size=CFG.train_fraction,
        random_state=CFG.seed,
        stratify=stratify,
    )
    return train_df.reset_index(drop=True), eval_df.reset_index(drop=True)


def fit_and_predict(train_df, eval_df, feature_cols: list[str], label_col: str = "label"):
    X_train = train_df[feature_cols].values
    y_train = train_df[label_col].values
    X_eval = eval_df[feature_cols].values
    y_eval = eval_df[label_col].values

    clf = LogisticRegression(random_state=CFG.seed, max_iter=1000)
    clf.fit(X_train, y_train)
    y_prob = clf.predict_proba(X_eval)[:, 1]

    return clf, y_eval, y_prob


def best_threshold_classifier(train_df, eval_df, feature_col: str, label_col: str = "label"):
    """Simple 1-D threshold classifier: pick the split point on the training
    set that maximizes training accuracy, used for the 'threshold vs.
    logistic regression' ablation (report Section 4.4)."""
    x_train = train_df[feature_col].values
    y_train = train_df[label_col].values
    x_eval = eval_df[feature_col].values
    y_eval = eval_df[label_col].values

    candidates = np.unique(x_train)
    best_acc, best_t, best_dir = -1, candidates[0], 1
    for t in candidates:
        for direction in (1, -1):  # direction=1: predict 1 if x>=t ; -1: predict 1 if x<=t
            pred = (x_train >= t).astype(int) if direction == 1 else (x_train <= t).astype(int)
            acc = (pred == y_train).mean()
            if acc > best_acc:
                best_acc, best_t, best_dir = acc, t, direction

    if best_dir == 1:
        y_pred = (x_eval >= best_t).astype(int)
        y_prob = (x_eval - x_eval.min()) / (x_eval.max() - x_eval.min() + 1e-9)
    else:
        y_pred = (x_eval <= best_t).astype(int)
        y_prob = 1 - (x_eval - x_eval.min()) / (x_eval.max() - x_eval.min() + 1e-9)

    return y_eval, y_pred, y_prob, best_t
