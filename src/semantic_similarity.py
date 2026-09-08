"""Embedding-based semantic similarity: a stronger self-consistency signal
and an alternative ground-truth labeler, both replacing token-F1 lexical
overlap with sentence-embedding cosine similarity.

Motivation (see README "Semantic-entropy baseline" and literature review
ref. [4], Farquhar et al. 2024): plain lexical agreement between sampled
answers misses paraphrases -- "Paris is the capital" and "The capital of
France is Paris" score low on token-F1 despite being the same claim. A
sentence embedding captures that they mean the same thing. This module
provides:

  1. semantic_entropy(answers)  -- cluster the self-consistency samples by
     meaning (not wording) and compute entropy over cluster sizes, the
     lightweight embedding-based analogue of Farquhar et al.'s semantic
     entropy (which they compute over NLI-clustered samples; here it's
     cosine-similarity clustering, cheaper and CPU-friendly, applied to
     the SAME 5 cached samples already generated for the lexical
     self-consistency baseline -- no additional LLM calls).

  2. semantic_label(...)  -- the same grounded/hallucinated decision as
     src/labeling.py's token-F1 judge, but via embedding cosine similarity
     to the correct/incorrect references, as a robustness check on
     whether the DA1/Llama/SciQ findings depend on the specific proxy
     labeling method.

Uses a small (~90MB), CPU-fast sentence-transformers model so it never
competes with the main LLM for GPU memory during a concurrent generation
run.
"""
from functools import lru_cache

import numpy as np

_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
_CLUSTER_SIMILARITY_THRESHOLD = 0.75  # cosine sim above which two answers are "the same meaning"


@lru_cache(maxsize=1)
def _get_model():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(_MODEL_NAME, device="cpu")


def embed(texts: list[str]) -> np.ndarray:
    model = _get_model()
    return model.encode(texts, normalize_embeddings=True, show_progress_bar=False)


def _cosine_sim_matrix(embeddings: np.ndarray) -> np.ndarray:
    return embeddings @ embeddings.T  # embeddings are L2-normalized -> dot product = cosine sim


def semantic_clusters(answers: list[str], threshold: float = _CLUSTER_SIMILARITY_THRESHOLD) -> list[int]:
    """Greedy single-linkage clustering by cosine similarity. Returns a
    cluster id per answer (same length/order as `answers`)."""
    if len(answers) <= 1:
        return [0] * len(answers)

    embeddings = embed(answers)
    sims = _cosine_sim_matrix(embeddings)

    cluster_ids = [-1] * len(answers)
    next_id = 0
    for i in range(len(answers)):
        if cluster_ids[i] != -1:
            continue
        cluster_ids[i] = next_id
        for j in range(i + 1, len(answers)):
            if cluster_ids[j] == -1 and sims[i, j] >= threshold:
                cluster_ids[j] = next_id
        next_id += 1
    return cluster_ids


def semantic_entropy(answers: list[str], threshold: float = _CLUSTER_SIMILARITY_THRESHOLD) -> float:
    """Shannon entropy (nats) over the distribution of cluster sizes among
    the sampled answers. 0 = all samples mean the same thing (fully
    self-consistent); higher = the model said several different things
    across samples (semantically inconsistent -> more likely
    hallucinating)."""
    if len(answers) <= 1:
        return 0.0
    cluster_ids = semantic_clusters(answers, threshold)
    counts = np.bincount(cluster_ids)
    probs = counts / counts.sum()
    probs = probs[probs > 0]
    return float(-(probs * np.log(probs)).sum())


def semantic_majority_answer(answers: list[str], threshold: float = _CLUSTER_SIMILARITY_THRESHOLD) -> str:
    """The answer belonging to the largest semantic cluster (ties broken by
    first occurrence) -- the semantic analogue of self_consistency.medoid_answer."""
    if len(answers) == 1:
        return answers[0]
    cluster_ids = semantic_clusters(answers, threshold)
    counts = np.bincount(cluster_ids)
    best_cluster = int(np.argmax(counts))
    idx = cluster_ids.index(best_cluster)
    return answers[idx]


def semantic_best_similarity(pred: str, references: list[str]) -> float:
    refs = [r for r in references if r and r.strip()]
    if not refs or not pred.strip():
        return 0.0
    pred_emb = embed([pred])[0]
    ref_embs = embed(refs)
    sims = ref_embs @ pred_emb
    return float(sims.max())


def semantic_label(generated_answer: str, best_answer: str, correct_answers, incorrect_answers, margin: float = 0.0) -> dict:
    """Embedding-similarity analogue of src/labeling.py:label_answer.
    Grounded iff cosine similarity to the closest correct reference
    exceeds similarity to the closest incorrect reference by `margin`."""
    correct_refs = list(correct_answers) + [best_answer]
    incorrect_refs = list(incorrect_answers)

    correct_sim = semantic_best_similarity(generated_answer, correct_refs)
    incorrect_sim = semantic_best_similarity(generated_answer, incorrect_refs)

    is_grounded = (correct_sim - incorrect_sim) > margin
    return {
        "semantic_correct_sim": correct_sim,
        "semantic_incorrect_sim": incorrect_sim,
        "semantic_label": int(not is_grounded),  # 1 = hallucinated, 0 = grounded
    }
