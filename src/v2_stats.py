"""Statistics shared by every v2 analysis.

Two kinds of detector are evaluated, and they need different null tests:

  fixed score   a single feature used directly with a pre-specified sign (e.g.
                -mean_logprob). Nothing is fitted, so the label-permutation null
                is just the Mann-Whitney null of the AUROC.
  fitted        a classifier trained on data. Its OOF AUROC is tested by refitting
                the whole OOF procedure under permuted labels, which also absorbs
                any optimism of the fitting procedure.

Every classifier is standardisation + L2 logistic regression (C=1), with the
scaler fit inside each training fold only.
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SEED = 42


def clf():
    return make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000))


def auroc(y, s):
    y = np.asarray(y)
    return float(roc_auc_score(y, s)) if 0 < y.sum() < len(y) else float("nan")


def boot_ci(y, s, n=2000, seed=SEED):
    y, s = np.asarray(y), np.asarray(s)
    rng = np.random.RandomState(seed)
    vals = []
    for _ in range(n):
        i = rng.randint(0, len(y), len(y))
        if 0 < y[i].sum() < len(i):
            vals.append(roc_auc_score(y[i], s[i]))
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def perm_p_fixed(y, s, n=5000, seed=SEED):
    """One-sided p for AUROC > 0.5 of a fixed score (label permutation)."""
    y, s = np.asarray(y), np.asarray(s)
    obs = roc_auc_score(y, s)
    rng = np.random.RandomState(seed)
    null = np.array([roc_auc_score(rng.permutation(y), s) for _ in range(n)])
    return float((np.sum(null >= obs) + 1) / (n + 1))


def oof(X, y, seed=SEED, k=5):
    X, y = np.asarray(X, dtype=float), np.asarray(y)
    p = np.zeros(len(y))
    for tr, te in StratifiedKFold(k, shuffle=True, random_state=seed).split(X, y):
        p[te] = clf().fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
    return p


def repeated_cv(X, y, r=20):
    a = np.array([auroc(y, oof(X, y, seed=1000 + i)) for i in range(r)])
    return {"mean": float(a.mean()), "sd": float(a.std()), "min": float(a.min()), "max": float(a.max())}


def perm_p_fitted(X, y, n=500, seed=SEED):
    obs = auroc(y, oof(X, y))
    rng = np.random.RandomState(seed)
    null = np.array([auroc(yp, oof(X, yp)) for yp in (rng.permutation(y) for _ in range(n))])
    return float((np.sum(null >= obs) + 1) / (n + 1))


def paired_delta(y, s1, s2, n=2000, seed=SEED):
    """AUROC(s1) - AUROC(s2) on the same items: point, 95% bootstrap CI, two-sided p."""
    y, s1, s2 = map(np.asarray, (y, s1, s2))
    rng = np.random.RandomState(seed)
    d = []
    for _ in range(n):
        i = rng.randint(0, len(y), len(y))
        if 0 < y[i].sum() < len(i):
            d.append(roc_auc_score(y[i], s1[i]) - roc_auc_score(y[i], s2[i]))
    d = np.array(d)
    p = 2 * min((d <= 0).mean(), (d >= 0).mean())
    return {"delta": auroc(y, s1) - auroc(y, s2), "ci": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
            "p_two_sided": float(min(1.0, max(p, 1 / n)))}


def summary_fixed(y, s):
    return {"auroc": auroc(y, s), "ci": boot_ci(y, s), "p": perm_p_fixed(y, s), "n": int(len(y)),
            "n_pos": int(np.sum(y))}


def summary_fitted(X, y, n_perm=500):
    p = oof(X, y)
    return {"auroc": auroc(y, p), "ci": boot_ci(y, p), "p": perm_p_fitted(X, y, n_perm),
            "repcv": repeated_cv(X, y), "n": int(len(y)), "n_pos": int(np.sum(y))}


def holm(pvals):
    """Holm-Bonferroni adjusted p-values (same order as input)."""
    p = np.asarray(pvals, dtype=float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(p) - rank) * p[i]))
        adj[i] = running
    return adj.tolist()


if __name__ == "__main__":
    rng = np.random.RandomState(0)
    y = rng.randint(0, 2, 300)
    assert abs(auroc(y, y + rng.normal(0, 1e-3, 300)) - 1) < 1e-9
    assert perm_p_fixed(y, rng.normal(size=300), n=500) > 0.01
    assert holm([0.01, 0.04, 0.03]) == [0.03, 0.06, 0.06]
    X = np.c_[y + rng.normal(0, 1, 300), rng.normal(size=300)]
    assert auroc(y, oof(X, y)) > 0.6
    print("v2_stats self-check ok")
