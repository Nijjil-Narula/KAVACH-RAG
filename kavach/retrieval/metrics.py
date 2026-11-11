"""
kavach.retrieval.metrics — Disentangled offline retrieval ranking metrics.

All functions operate purely on lists of chunk or document ids and require no
model weights. This module has no side effects and is fully deterministic.

nDCG normalization note:
  The IDCG denominator uses min(|G|, k), NOT |G|. This ensures that a perfect
  ranking (all ground-truth items retrieved in the top positions) achieves
  nDCG@k = 1.0 regardless of whether |G| > k. The original spec formula using
  |G| would score a perfect top-k ranking below 1.0 when |G| > k, which is
  mathematically incorrect for a normalized metric. This deviation is intentional
  and documented here and in the ablation report.
"""
from __future__ import annotations

import math


def recall_at_k(
    retrieved: list[str],
    ground_truth: list[str],
    k: int,
) -> float:
    """
    Fraction of ground-truth items found in the top-k retrieved items.

    recall@k = |R_k ∩ G| / |G|

    Returns 0.0 if ground_truth is empty (no relevant items to retrieve).

    Args:
        retrieved: Ordered list of retrieved chunk or doc ids.
        ground_truth: Set of relevant chunk or doc ids.
        k: Cutoff depth.

    Returns:
        Float in [0.0, 1.0].
    """
    if not ground_truth:
        return 0.0
    top_k = set(retrieved[:k])
    relevant = set(ground_truth)
    return len(top_k & relevant) / len(relevant)


def precision_at_k(
    retrieved: list[str],
    ground_truth: list[str],
    k: int,
) -> float:
    """
    Fraction of top-k retrieved items that are relevant.

    precision@k = |R_k ∩ G| / k

    Returns 0.0 if k == 0 or retrieved is empty.

    Args:
        retrieved: Ordered list of retrieved chunk or doc ids.
        ground_truth: Set of relevant chunk or doc ids.
        k: Cutoff depth.

    Returns:
        Float in [0.0, 1.0].
    """
    if k == 0 or not retrieved:
        return 0.0
    top_k = set(retrieved[:k])
    relevant = set(ground_truth)
    return len(top_k & relevant) / k


def reciprocal_rank(
    retrieved: list[str],
    ground_truth: list[str],
) -> float:
    """
    Reciprocal of the rank of the first relevant item.

    MRR contribution = 1 / rank_i, where rank_i is 1-based.
    Returns 0.0 if no relevant item appears in retrieved.

    Args:
        retrieved: Ordered list of retrieved chunk or doc ids.
        ground_truth: Set of relevant chunk or doc ids.

    Returns:
        Float in [0.0, 1.0].
    """
    relevant = set(ground_truth)
    for rank, item in enumerate(retrieved, start=1):
        if item in relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(
    retrieved: list[str],
    ground_truth: list[str],
    k: int,
) -> float:
    """
    Normalized Discounted Cumulative Gain at depth k (binary relevance).

    DCG@k  = Σ_{i=1..k} rel_i / log2(i + 1)
    IDCG@k = Σ_{i=1..min(|G|,k)} 1 / log2(i + 1)
    nDCG@k = DCG@k / IDCG@k

    IDCG uses min(|G|, k) so that a perfect top-k ranking always scores 1.0.

    Returns 0.0 if ground_truth is empty or IDCG is 0.

    Args:
        retrieved: Ordered list of retrieved chunk or doc ids.
        ground_truth: Set of relevant chunk or doc ids.
        k: Cutoff depth.

    Returns:
        Float in [0.0, 1.0].
    """
    if not ground_truth or k == 0:
        return 0.0

    relevant = set(ground_truth)
    # DCG
    dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank, item in enumerate(retrieved[:k], start=1)
        if item in relevant
    )
    # IDCG: best possible = all min(|G|, k) relevant items at top ranks
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))

    if idcg == 0.0:
        return 0.0
    return dcg / idcg


def dedupe_to_docs(
    chunk_ids: list[str],
    chunk_to_doc: dict[str, str],
) -> list[str]:
    """
    Map chunk ids to doc ids, preserving rank order and deduplicating.

    The first occurrence of each doc_id in the ranked list is kept.
    This is used to convert chunk-level rankings to document-level rankings.

    Args:
        chunk_ids: Ordered list of retrieved chunk ids.
        chunk_to_doc: Mapping from chunk_id to doc_id.

    Returns:
        Deduplicated ordered list of doc_ids.
    """
    seen: set[str] = set()
    result: list[str] = []
    for chunk_id in chunk_ids:
        doc_id = chunk_to_doc.get(chunk_id, chunk_id)
        if doc_id not in seen:
            seen.add(doc_id)
            result.append(doc_id)
    return result
