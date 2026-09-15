"""Expanded uncertainty feature set, built on top of the per-token
logprob/entropy/margin arrays cached by
scripts/backfill_rich_token_details.py.

This is an ADDITIVE investigation, not a replacement: the original
3-feature detector (mean logprob, min logprob, mean entropy) stays
exactly as reported in the DA1/matrix results. Every feature here is
computed from data already available after ONE greedy generation --
nothing here needs extra LLM calls (multi-call, self-consistency-derived
features live in src/self_consistency.py / src/semantic_similarity.py and
are kept separate on purpose, per the investigation's own requirement to
distinguish 1-call from N-call signals).

Feature groups:
    location   -- mean/median/min, percentiles, spread (std)
    coverage   -- fraction of tokens below a fixed "suspicious" threshold
    entropy    -- mean/max/std of the per-token predictive entropy
    margin     -- top-1 vs. top-2 confidence gap (mean/min)
    positional -- does uncertainty concentrate at the start or end of the
                  answer, and is there a directional trend across it
    length     -- raw token count (a possible confound, included so it can
                  be checked for rather than silently baked into the
                  other features)
"""
import numpy as np

_SUSPICIOUS_LOGPROB_THRESHOLD = -3.0  # matches the "suspicious token" highlight rule in token_viz.py
_EMPTY_FALLBACK = -20.0


def extract_rich_features(token_logprobs: list[float], token_entropies: list[float],
                           token_margins: list[float] | None = None) -> dict:
    n = len(token_logprobs)
    if n == 0:
        return {
            "mean_logprob": _EMPTY_FALLBACK, "median_logprob": _EMPTY_FALLBACK, "min_logprob": _EMPTY_FALLBACK,
            "logprob_p10": _EMPTY_FALLBACK, "logprob_p25": _EMPTY_FALLBACK, "std_logprob": 0.0,
            "frac_low_confidence": 1.0,
            "mean_entropy": 0.0, "max_entropy": 0.0, "std_entropy": 0.0,
            "mean_margin": 0.0, "min_margin": 0.0,
            "first_half_mean_logprob": _EMPTY_FALLBACK, "second_half_mean_logprob": _EMPTY_FALLBACK,
            "logprob_delta_second_minus_first": 0.0, "logprob_trend_slope": 0.0,
            "answer_length": 0,
        }

    lp = np.array(token_logprobs, dtype=np.float64)
    ent = np.array(token_entropies, dtype=np.float64)

    feats = {
        "mean_logprob": float(lp.mean()),
        "median_logprob": float(np.median(lp)),
        "min_logprob": float(lp.min()),
        "logprob_p10": float(np.percentile(lp, 10)),
        "logprob_p25": float(np.percentile(lp, 25)),
        "std_logprob": float(lp.std()),
        "frac_low_confidence": float((lp < _SUSPICIOUS_LOGPROB_THRESHOLD).mean()),
        "mean_entropy": float(ent.mean()),
        "max_entropy": float(ent.max()),
        "std_entropy": float(ent.std()),
        "answer_length": n,
    }

    if token_margins:
        m = np.array(token_margins, dtype=np.float64)
        feats["mean_margin"] = float(m.mean())
        feats["min_margin"] = float(m.min())
    else:
        feats["mean_margin"] = 0.0
        feats["min_margin"] = 0.0

    # Positional: split the answer in half, compare mean logprob in each half.
    # A more negative "delta" means confidence DROPS across the answer (model
    # gets less sure as it goes -- a plausible fingerprint of confabulation
    # once it commits to a specific, unsupported claim early on).
    mid = max(1, n // 2)
    first_half = lp[:mid]
    second_half = lp[mid:] if n > mid else lp[:mid]
    feats["first_half_mean_logprob"] = float(first_half.mean())
    feats["second_half_mean_logprob"] = float(second_half.mean())
    feats["logprob_delta_second_minus_first"] = float(second_half.mean() - first_half.mean())

    # Linear trend of logprob across token position (least-squares slope).
    if n >= 2:
        positions = np.arange(n, dtype=np.float64)
        slope = float(np.polyfit(positions, lp, 1)[0])
    else:
        slope = 0.0
    feats["logprob_trend_slope"] = slope

    return feats


ALL_RICH_FEATURE_NAMES = [
    "mean_logprob", "median_logprob", "min_logprob", "logprob_p10", "logprob_p25", "std_logprob",
    "frac_low_confidence", "mean_entropy", "max_entropy", "std_entropy", "mean_margin", "min_margin",
    "first_half_mean_logprob", "second_half_mean_logprob", "logprob_delta_second_minus_first",
    "logprob_trend_slope", "answer_length",
]

# The three features used by the original DA1/matrix detector -- kept as a named
# constant so every downstream script can refer to "the original baseline" by
# name instead of re-typing the list (and risking it silently drifting).
ORIGINAL_THREE_FEATURES = ["mean_logprob", "min_logprob", "mean_entropy"]
