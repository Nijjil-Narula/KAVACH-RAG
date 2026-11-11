"""
tests.test_metrics — Hand-computed unit tests for kavach.retrieval.metrics.

Run with:
    .venv\\Scripts\\python.exe -m pytest -q tests/test_metrics.py
"""
from __future__ import annotations

import math

import pytest

from kavach.retrieval.metrics import (
    dedupe_to_docs,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)


def test_recall_at_k_hand_computed() -> None:
    """Verify Recall@k with partial, complete, and cutoff rankings."""
    retrieved = ["d1", "d2", "d3", "d4"]
    ground_truth = ["d2", "d4", "d5"]

    assert recall_at_k(retrieved, ground_truth, k=1) == pytest.approx(0.0)
    assert recall_at_k(retrieved, ground_truth, k=2) == pytest.approx(1.0 / 3.0)
    assert recall_at_k(retrieved, ground_truth, k=3) == pytest.approx(1.0 / 3.0)
    assert recall_at_k(retrieved, ground_truth, k=4) == pytest.approx(2.0 / 3.0)


def test_recall_at_k_empty_ground_truth() -> None:
    """Empty ground truth returns 0.0 safely without ZeroDivisionError."""
    assert recall_at_k(["d1", "d2"], [], k=3) == 0.0
    assert recall_at_k([], [], k=3) == 0.0


def test_precision_at_k_hand_computed() -> None:
    """Verify Precision@k including edge cases k=0, empty, and short retrieved."""
    retrieved = ["d1", "d2", "d3", "d4"]
    ground_truth = ["d1", "d3"]

    assert precision_at_k(retrieved, ground_truth, k=1) == pytest.approx(1.0)
    assert precision_at_k(retrieved, ground_truth, k=2) == pytest.approx(0.5)
    assert precision_at_k(retrieved, ground_truth, k=3) == pytest.approx(2.0 / 3.0)
    assert precision_at_k(retrieved, ground_truth, k=4) == pytest.approx(0.5)
    assert precision_at_k(retrieved, ground_truth, k=0) == 0.0
    assert precision_at_k([], ground_truth, k=3) == 0.0
    assert precision_at_k(["d1"], ["d1"], k=3) == pytest.approx(1.0 / 3.0)
    assert precision_at_k(retrieved, [], k=3) == 0.0


def test_reciprocal_rank_hand_computed() -> None:
    """Verify 1/rank of first relevant hit and 0.0 on miss."""
    gt = ["d2", "d4"]
    assert reciprocal_rank(["d2", "d1", "d4"], gt) == pytest.approx(1.0)
    assert reciprocal_rank(["d1", "d2", "d4"], gt) == pytest.approx(0.5)
    assert reciprocal_rank(["d1", "d3", "d4"], gt) == pytest.approx(1.0 / 3.0)
    assert reciprocal_rank(["d1", "d3", "d5"], gt) == 0.0
    assert reciprocal_rank([], gt) == 0.0
    assert reciprocal_rank(["d1"], []) == 0.0


def test_ndcg_at_k_hand_computed() -> None:
    """Verify exact DCG/IDCG calculation at k=4 and empty/zero cases."""
    retrieved = ["x", "g1", "y", "g2"]
    ground_truth = ["g1", "g2"]

    expected_dcg = (1.0 / math.log2(3.0)) + (1.0 / math.log2(5.0))
    expected_idcg = (1.0 / math.log2(2.0)) + (1.0 / math.log2(3.0))
    expected_ndcg = expected_dcg / expected_idcg

    assert ndcg_at_k(retrieved, ground_truth, k=4) == pytest.approx(expected_ndcg)
    assert ndcg_at_k(["g1", "g2", "x"], ground_truth, k=3) == pytest.approx(1.0)
    assert ndcg_at_k(["x", "y"], ground_truth, k=2) == 0.0
    assert ndcg_at_k(retrieved, [], k=3) == 0.0
    assert ndcg_at_k(retrieved, ground_truth, k=0) == 0.0


def test_ndcg_at_k_perfect_ranking_when_gt_exceeds_k() -> None:
    """Perfect top-k ranking must yield nDCG@k == 1.0 even when |G| > k."""
    gt = ["g1", "g2", "g3", "g4", "g5"]
    retrieved = ["g4", "g1", "g5", "distractor"]

    assert ndcg_at_k(retrieved, gt, k=3) == pytest.approx(1.0)
    assert ndcg_at_k(["g2", "g3"], gt, k=2) == pytest.approx(1.0)


def test_dedupe_to_docs_order_and_fallback() -> None:
    """Verify first-occurrence order preservation and unmapped fallback."""
    chunk_ids = ["A_1", "B_1", "A_2", "C_1", "B_2", "UNMAPPED"]
    chunk_to_doc = {
        "A_1": "DOC_A",
        "A_2": "DOC_A",
        "B_1": "DOC_B",
        "B_2": "DOC_B",
        "C_1": "DOC_C",
    }

    assert dedupe_to_docs(chunk_ids, chunk_to_doc) == [
        "DOC_A",
        "DOC_B",
        "DOC_C",
        "UNMAPPED",
    ]
    assert dedupe_to_docs([], chunk_to_doc) == []
