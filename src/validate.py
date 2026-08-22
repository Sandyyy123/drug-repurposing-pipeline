"""Retrospective validation harness - the rigor core.

Before trusting a scoring funnel on true unknowns, prove it can separate KNOWN
actives from property-matched decoys. This module computes the standard virtual-
screening validation metrics on a labelled benchmark set, using the SAME consensus
score the funnel produces:

* **Enrichment Factor (EF@x%)** - how many more actives you find in the top x% of
  the ranked list than random selection would give.
      EF@x% = (actives_in_top_k / k) / (total_actives / N),   k = ceil(x * N)
* **ROC-AUC** - implemented exactly via the Mann-Whitney U relationship to the AUC,
  with average-rank tie handling. Perfect ranker -> 1.0; random -> ~0.5.
* **Permutation / null control** - shuffle the labels many times, recompute AUC,
  and report the empirical p-value that the observed AUC beats the null. This is
  the control that answers the client's core concern: is the enrichment real, or
  could a random ranker have produced it?

All maths here is exact and unit-tested (see ``tests/test_validate.py``). Higher
score = more likely active is the convention throughout.
"""

from __future__ import annotations

import math
from typing import Dict, List, Sequence

import numpy as np


def _average_ranks(values: np.ndarray) -> np.ndarray:
    """Average ranks (1-based), smallest value -> rank 1, ties averaged."""
    values = np.asarray(values, dtype=float)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    sorted_vals = values[order]
    i = 0
    n = len(values)
    while i < n:
        j = i
        while j + 1 < n and sorted_vals[j + 1] == sorted_vals[i]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def roc_auc(labels: Sequence[int], scores: Sequence[float]) -> float:
    """Exact ROC-AUC via the Mann-Whitney U statistic (higher score = positive).

    Returns 0.5 when either class is empty (undefined, reported as no-signal).
    """
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    n_pos = int((labels == 1).sum())
    n_neg = int((labels == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return 0.5
    ranks = _average_ranks(scores)
    sum_ranks_pos = ranks[labels == 1].sum()
    auc = (sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def enrichment_factor(labels: Sequence[int], scores: Sequence[float],
                      fraction: float) -> float:
    """Enrichment factor at the top ``fraction`` of the ranked list.

    ``fraction`` in (0, 1]; e.g. 0.01 for EF@1%. Compounds are ranked by score
    descending (best first). Returns 0.0 when there are no actives.
    """
    if not 0.0 < fraction <= 1.0:
        raise ValueError("fraction must be in (0, 1].")
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    n = len(labels)
    total_actives = int((labels == 1).sum())
    if n == 0 or total_actives == 0:
        return 0.0

    k = max(1, math.ceil(fraction * n))
    # Rank by score descending; stable sort keeps input order for ties.
    order = np.argsort(-scores, kind="mergesort")
    top_labels = labels[order][:k]
    actives_in_top = int((top_labels == 1).sum())

    active_rate_top = actives_in_top / k
    active_rate_all = total_actives / n
    return float(active_rate_top / active_rate_all)


def permutation_test(labels: Sequence[int], scores: Sequence[float],
                     n_permutations: int = 1000, seed: int = 42) -> Dict[str, object]:
    """Label-permutation null control for the observed AUC.

    Shuffles the labels ``n_permutations`` times, recomputes AUC each time, and
    reports the empirical one-sided p-value P(null_auc >= observed_auc) with the
    standard ``(count + 1) / (n + 1)`` correction so p is never exactly 0.
    """
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    observed = roc_auc(labels, scores)

    rng = np.random.default_rng(seed)
    null = np.empty(n_permutations, dtype=float)
    permuted = labels.copy()
    for i in range(n_permutations):
        rng.shuffle(permuted)
        null[i] = roc_auc(permuted, scores)

    count_ge = int((null >= observed).sum())
    p_value = (count_ge + 1) / (n_permutations + 1)
    return {
        "n_permutations": n_permutations,
        "observed_auc": round(observed, 4),
        "null_auc_mean": round(float(null.mean()), 4),
        "null_auc_std": round(float(null.std()), 4),
        "p_value": round(p_value, 5),
        "significant_at_0.05": bool(p_value < 0.05),
    }


def validate(labels: Sequence[int], scores: Sequence[float],
             ef_fractions: Sequence[float] = (0.01, 0.05),
             n_permutations: int = 1000, seed: int = 42) -> Dict[str, object]:
    """Run the full retrospective validation and return a report dict."""
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    n = len(labels)
    n_actives = int((labels == 1).sum())

    ef = {}
    for frac in ef_fractions:
        key = f"EF@{frac * 100:g}%"
        ef[key] = round(enrichment_factor(labels, scores, frac), 4)

    report = {
        "n_compounds": int(n),
        "n_actives": n_actives,
        "n_decoys": int(n - n_actives),
        "roc_auc": round(roc_auc(labels, scores), 4),
        "enrichment_factor": ef,
        "permutation_control": permutation_test(
            labels, scores, n_permutations=n_permutations, seed=seed
        ),
        "seed": int(seed),
        "score_convention": "higher score = more likely active",
        "note": ("Demo scores are synthetic (see src/dock.py). These numbers "
                 "demonstrate what the harness COMPUTES; they are not a benchmark "
                 "result. The harness maths is unit-tested in tests/test_validate.py."),
    }
    return report
