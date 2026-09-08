"""5-call self-consistency baseline: agreement scoring over sampled answers.

Report Section 4.4 / PPT slide 7: "generate five independent answers ...
use an agreement/majority rule to estimate whether the answer is
reliable." We operationalize this as the mean pairwise token-F1 similarity
across the 5 sampled answers (high agreement -> answers mutually
paraphrase each other -> more likely grounded; low agreement -> the model
is confabulating a different fact each time -> more likely hallucinated).

The "representative answer" for ground-truth labeling purposes is the
medoid: the sampled answer with the highest total similarity to the other
four, i.e. the one the model would most likely settle on under majority
vote.
"""
from itertools import combinations

import numpy as np

from .labeling import token_f1


def agreement_score(answers: list[str]) -> float:
    """Mean pairwise token-F1 similarity across the sampled answers."""
    if len(answers) < 2:
        return 1.0
    pairs = list(combinations(range(len(answers)), 2))
    sims = [token_f1(answers[i], answers[j]) for i, j in pairs]
    return float(np.mean(sims))


def medoid_answer(answers: list[str]) -> str:
    """Return the answer with highest total similarity to the rest (majority representative)."""
    if len(answers) == 1:
        return answers[0]
    n = len(answers)
    totals = np.zeros(n)
    for i in range(n):
        for j in range(n):
            if i != j:
                totals[i] += token_f1(answers[i], answers[j])
    return answers[int(np.argmax(totals))]
