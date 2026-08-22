"""Consensus rank-aggregation across docking engines.

This is the heart of the "avoid single-tool bias" story. Instead of trusting one
score, we combine several tools by:

1. **Rank aggregation (Borda / mean rank).** Convert each tool's raw scores to
   ranks where rank 1 = best (orienting by each tool's ``direction``), then take a
   weighted mean of ranks across tools. Lower mean rank = stronger consensus.
2. **Z-score consensus.** Orient each tool's scores so higher = better, standardise
   them (z-score across the library), then take a weighted mean. Higher = better.
   This complements rank aggregation because it keeps score magnitude information
   that pure ranks discard.

Both aggregations use the same per-tool weights (from ``config.yaml``). Ties are
handled with average ranks, which is the statistically correct convention and is
what makes the enrichment / AUC maths downstream well behaved.

All functions here are exact and unit-tested; there is no demo shortcut.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np


def average_rank(values: np.ndarray, higher_is_better: bool) -> np.ndarray:
    """Rank ``values`` with rank 1 = best; ties get the average of their ranks.

    Parameters
    ----------
    values : array of raw scores.
    higher_is_better : if True the largest value is best (rank 1); if False the
        smallest value is best.
    """
    values = np.asarray(values, dtype=float)
    oriented = -values if higher_is_better else values  # smallest oriented = best
    order = np.argsort(oriented, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    sorted_vals = oriented[order]

    i = 0
    n = len(values)
    while i < n:
        j = i
        while j + 1 < n and sorted_vals[j + 1] == sorted_vals[i]:
            j += 1
        # positions i..j are tied; assign them the average 1-based rank
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def _zscore(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    mean = values.mean()
    std = values.std()
    if std == 0.0:
        return np.zeros_like(values)
    return (values - mean) / std


def consensus_ranking(scores_by_tool: Dict[str, List[float]],
                      directions: Dict[str, str],
                      weights: Dict[str, float] = None) -> Dict[str, np.ndarray]:
    """Aggregate per-tool scores into consensus arrays.

    Parameters
    ----------
    scores_by_tool : ``{tool: [score per ligand]}`` (all lists the same length).
    directions     : ``{tool: "higher"|"lower"}`` - which way is better.
    weights        : ``{tool: weight}``. Missing tools default to weight 1.0.
        Weights are normalised to sum to 1.

    Returns
    -------
    dict with, each aligned to the input ligand order:
      ``mean_rank``     : weighted mean rank (lower = better consensus)
      ``z_consensus``   : weighted mean oriented z-score (higher = better)
      ``per_tool_rank`` : ``{tool: ranks}``
      ``order``         : indices sorted best-to-worst by mean_rank (ties broken
                          by z_consensus)
    """
    tools = list(scores_by_tool.keys())
    if not tools:
        raise ValueError("scores_by_tool is empty; nothing to aggregate.")

    lengths = {len(v) for v in scores_by_tool.values()}
    if len(lengths) != 1:
        raise ValueError("All tools must score the same number of ligands.")
    n = lengths.pop()

    if weights is None:
        weights = {}
    w = np.array([float(weights.get(t, 1.0)) for t in tools], dtype=float)
    if w.sum() <= 0:
        raise ValueError("Sum of tool weights must be positive.")
    w = w / w.sum()

    per_tool_rank: Dict[str, np.ndarray] = {}
    rank_matrix = np.zeros((len(tools), n), dtype=float)
    z_matrix = np.zeros((len(tools), n), dtype=float)

    for ti, tool in enumerate(tools):
        raw = np.asarray(scores_by_tool[tool], dtype=float)
        higher = directions.get(tool, "higher") == "higher"
        ranks = average_rank(raw, higher_is_better=higher)
        per_tool_rank[tool] = ranks
        rank_matrix[ti] = ranks
        oriented = raw if higher else -raw
        z_matrix[ti] = _zscore(oriented)

    mean_rank = np.average(rank_matrix, axis=0, weights=w)
    z_consensus = np.average(z_matrix, axis=0, weights=w)

    # Best-to-worst: lowest mean rank first; break ties by higher z_consensus.
    order = np.lexsort((-z_consensus, mean_rank))

    return {
        "mean_rank": mean_rank,
        "z_consensus": z_consensus,
        "per_tool_rank": per_tool_rank,
        "order": order,
    }
