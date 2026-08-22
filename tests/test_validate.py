"""Unit tests proving the validation and consensus maths are correct.

Run with:  pytest -q
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.validate import roc_auc, enrichment_factor, permutation_test, validate
from src.consensus import average_rank, consensus_ranking
from src.dock import TOOL_DIRECTION


# --------------------------------------------------------------------------- #
# ROC-AUC
# --------------------------------------------------------------------------- #
def test_auc_perfect_ranker_is_one():
    # All actives score above all decoys -> AUC exactly 1.0.
    labels = [1, 1, 1, 0, 0, 0]
    scores = [0.9, 0.8, 0.7, 0.3, 0.2, 0.1]
    assert roc_auc(labels, scores) == 1.0


def test_auc_worst_ranker_is_zero():
    # All actives score below all decoys -> AUC exactly 0.0.
    labels = [1, 1, 1, 0, 0, 0]
    scores = [0.1, 0.2, 0.3, 0.7, 0.8, 0.9]
    assert roc_auc(labels, scores) == 0.0


def test_auc_tie_gives_half():
    # One active, one decoy, identical scores -> AUC 0.5 (tie handled as 0.5).
    assert roc_auc([1, 0], [0.5, 0.5]) == 0.5


def test_auc_random_is_near_half():
    rng = np.random.default_rng(0)
    n = 4000
    labels = np.array([1] * (n // 2) + [0] * (n // 2))
    scores = rng.random(n)  # scores independent of labels
    auc = roc_auc(labels, scores)
    assert abs(auc - 0.5) < 0.05


def test_auc_matches_sklearn_when_available():
    rng = np.random.default_rng(3)
    labels = rng.integers(0, 2, size=200)
    if labels.sum() in (0, len(labels)):  # guard against degenerate draw
        labels[0] = 1 - labels[0]
    scores = rng.random(200) + 0.3 * labels  # inject mild signal
    ours = roc_auc(labels, scores)
    try:
        from sklearn.metrics import roc_auc_score
        assert abs(ours - roc_auc_score(labels, scores)) < 1e-9
    except ImportError:  # pragma: no cover
        pytest.skip("scikit-learn not installed")


# --------------------------------------------------------------------------- #
# Enrichment factor (hand-checked toy case)
# --------------------------------------------------------------------------- #
def test_enrichment_factor_hand_checked():
    # 20 compounds, 4 actives. Scores strictly decreasing with index so the
    # ranking is exactly the list order. Actives at positions 0,1,2 and 5.
    labels = [1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    scores = list(range(20, 0, -1))  # 20,19,...,1  (index 0 = best)
    # Top 25% -> k = ceil(0.25 * 20) = 5. Actives in top 5 = positions 0,1,2 = 3.
    # EF = (3/5) / (4/20) = 0.6 / 0.2 = 3.0
    assert enrichment_factor(labels, scores, 0.25) == pytest.approx(3.0)


def test_enrichment_factor_top_is_active():
    # 100 compounds, 10 actives, single best compound is active -> EF@1% = 10.
    # k = ceil(0.01 * 100) = 1; actives_in_top = 1; EF = (1/1)/(10/100) = 10.
    labels = [1] + [0] * 90 + [1] * 9   # 100 entries, 10 actives, best is active
    scores = list(range(100, 0, -1))    # 100 strictly-decreasing scores
    assert len(labels) == len(scores) == 100
    assert enrichment_factor(labels, scores, 0.01) == pytest.approx(10.0)


def test_enrichment_factor_no_actives_is_zero():
    assert enrichment_factor([0, 0, 0], [3, 2, 1], 0.5) == 0.0


def test_enrichment_factor_bad_fraction_raises():
    with pytest.raises(ValueError):
        enrichment_factor([1, 0], [1, 0], 0.0)
    with pytest.raises(ValueError):
        enrichment_factor([1, 0], [1, 0], 1.5)


# --------------------------------------------------------------------------- #
# Permutation control
# --------------------------------------------------------------------------- #
def test_permutation_perfect_ranker_significant():
    # Use a comfortably sized set so the label-shuffle null essentially never
    # reproduces the perfect separation by chance: P(null AUC == 1.0) = 1/C(16,8).
    labels = [1] * 8 + [0] * 8
    scores = list(range(16, 0, -1))  # perfect separation, strictly decreasing
    result = permutation_test(labels, scores, n_permutations=500, seed=1)
    assert result["observed_auc"] == 1.0
    assert result["significant_at_0.05"] is True
    assert result["p_value"] < 0.05


def test_permutation_is_deterministic_with_seed():
    labels = [1, 0, 1, 0, 1, 0, 1, 0]
    scores = [0.6, 0.1, 0.9, 0.2, 0.55, 0.3, 0.8, 0.05]
    a = permutation_test(labels, scores, n_permutations=300, seed=7)
    b = permutation_test(labels, scores, n_permutations=300, seed=7)
    assert a == b


def test_permutation_random_not_significant():
    rng = np.random.default_rng(11)
    labels = np.array([1] * 25 + [0] * 25)
    scores = rng.random(50)  # no relationship to labels
    result = permutation_test(labels, scores, n_permutations=1000, seed=5)
    assert result["p_value"] > 0.05


# --------------------------------------------------------------------------- #
# Consensus rank aggregation (hand-checked toy case)
# --------------------------------------------------------------------------- #
def test_average_rank_ties():
    # values 5,3,3,1 with higher-is-better: 5->1, the two 3s share ranks 2&3 ->2.5, 1->4
    ranks = average_rank(np.array([5.0, 3.0, 3.0, 1.0]), higher_is_better=True)
    assert list(ranks) == [1.0, 2.5, 2.5, 4.0]


def test_consensus_hand_checked():
    # Two tools, three ligands.
    # diffdock (higher better): lig1=0.9, lig2=0.5, lig3=0.1 -> ranks 1,2,3
    # vina    (lower  better): lig1=-5,  lig2=-9,  lig3=-7  -> ranks 3,1,2
    # equal weights -> mean ranks: lig1=2.0, lig2=1.5, lig3=2.5
    scores = {"diffdock": [0.9, 0.5, 0.1], "vina": [-5.0, -9.0, -7.0]}
    out = consensus_ranking(scores, TOOL_DIRECTION, weights={"diffdock": 1.0, "vina": 1.0})
    assert list(out["per_tool_rank"]["diffdock"]) == [1.0, 2.0, 3.0]
    assert list(out["per_tool_rank"]["vina"]) == [3.0, 1.0, 2.0]
    assert list(np.round(out["mean_rank"], 3)) == [2.0, 1.5, 2.5]
    # Best-to-worst order: lig2 (idx1), lig1 (idx0), lig3 (idx2)
    assert list(out["order"]) == [1, 0, 2]


def test_consensus_weighting_shifts_order():
    # Heavily weighting vina should pull its favourite (lig3) toward the top.
    scores = {"diffdock": [0.9, 0.8, 0.1], "vina": [-5.0, -6.0, -12.0]}
    out = consensus_ranking(scores, TOOL_DIRECTION, weights={"diffdock": 0.1, "vina": 0.9})
    # vina strongly prefers lig3 (idx2); with 0.9 weight it should rank first.
    assert int(out["order"][0]) == 2


def test_consensus_length_mismatch_raises():
    with pytest.raises(ValueError):
        consensus_ranking({"diffdock": [0.1, 0.2], "vina": [0.1]}, TOOL_DIRECTION)


# --------------------------------------------------------------------------- #
# Full report smoke test
# --------------------------------------------------------------------------- #
def test_validate_report_structure():
    labels = [1, 1, 0, 0, 1, 0]
    scores = [0.9, 0.8, 0.2, 0.1, 0.7, 0.3]
    report = validate(labels, scores, ef_fractions=(0.5,), n_permutations=200, seed=2)
    assert report["n_compounds"] == 6
    assert report["n_actives"] == 3
    assert report["n_decoys"] == 3
    assert 0.0 <= report["roc_auc"] <= 1.0
    assert "EF@50%" in report["enrichment_factor"]
    assert "p_value" in report["permutation_control"]
