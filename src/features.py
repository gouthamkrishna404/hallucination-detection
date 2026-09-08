"""Feature extraction: turn per-token logprobs/entropies into the three
aggregated uncertainty features described in report Section 4.2.

    mean_logprob    - arithmetic mean of generated-token log-probabilities
    min_logprob     - lowest single token log-probability in the answer
    mean_entropy    - mean per-token full-vocabulary predictive entropy (nats)
"""
import numpy as np

# Fallback for the degenerate case of an empty generation (immediate EOS).
_EMPTY_LOGPROB_FALLBACK = -20.0
_EMPTY_ENTROPY_FALLBACK = 0.0


def extract_features(token_logprobs: list[float], token_entropies: list[float]) -> dict:
    if len(token_logprobs) == 0:
        return {
            "mean_logprob": _EMPTY_LOGPROB_FALLBACK,
            "min_logprob": _EMPTY_LOGPROB_FALLBACK,
            "mean_entropy": _EMPTY_ENTROPY_FALLBACK,
        }
    lp = np.array(token_logprobs, dtype=np.float64)
    ent = np.array(token_entropies, dtype=np.float64)
    return {
        "mean_logprob": float(lp.mean()),
        "min_logprob": float(lp.min()),
        "mean_entropy": float(ent.mean()),
    }
