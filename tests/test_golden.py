"""
tests.test_golden — Validate data/golden/golden_150.jsonl benchmark suite.

Skips gracefully when golden_150.jsonl (or corpus_chunks.parquet) has not yet
been generated.

Run with:
    .venv\\Scripts\\python.exe -m pytest -q tests/test_golden.py
"""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

import pandas as pd
import pytest

from kavach.retrieval.router import (
    ROUTE_FAST_PATH,
    ROUTE_REFUSAL,
    ROUTE_SEMANTIC,
    normalize_query,
)
from kavach.retrieval.schema import GoldenItem, QuestionType

_ROOT = Path(__file__).parent.parent
_GOLDEN_PATH = _ROOT / "data" / "golden" / "golden_150.jsonl"
_PARQUET_PATH = _ROOT / "data" / "processed" / "corpus_chunks.parquet"

_SKIP_GOLDEN = pytest.mark.skipif(
    not _GOLDEN_PATH.exists(),
    reason="golden_150.jsonl not yet generated",
)
_SKIP_BOTH = pytest.mark.skipif(
    not (_GOLDEN_PATH.exists() and _PARQUET_PATH.exists()),
    reason="golden_150.jsonl or corpus_chunks.parquet not yet generated",
)

_EXPECTED_OOS: list[tuple[str, str]] = [
    ("How do I bake a sourdough loaf?", ROUTE_REFUSAL),
    ("Who won the 2022 FIFA World Cup final?", ROUTE_REFUSAL),
    ("What is the best way to train for a marathon in six weeks?", ROUTE_REFUSAL),
    ("Recommend a good sci-fi novel for a long flight.", ROUTE_REFUSAL),
    ("What's the weather forecast for Delhi this weekend?", ROUTE_REFUSAL),
    ("How do I fix a leaking kitchen faucet?", ROUTE_REFUSAL),
    ("Explain how compound interest works on a savings account.", ROUTE_REFUSAL),
    ("Translate 'good morning' into Japanese and French.", ROUTE_REFUSAL),
    (
        "What are the internal firewall rules of Acme Corp's production network?",
        ROUTE_SEMANTIC,
    ),
    ("What is the password policy of my company's HR portal?", ROUTE_SEMANTIC),
]


def _load_golden_items() -> list[GoldenItem]:
    items: list[GoldenItem] = []
    for line in _GOLDEN_PATH.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        items.append(GoldenItem.model_validate(json.loads(stripped)))
    return items


@_SKIP_GOLDEN
def test_golden_schema_and_total_count() -> None:
    """golden_150.jsonl must contain exactly 150 valid GoldenItem records."""
    items = _load_golden_items()
    assert len(items) == 150, f"Expected 150 golden items, got {len(items)}"


@_SKIP_GOLDEN
def test_golden_type_distribution() -> None:
    """Distribution must be exactly 60 / 45 / 15 / 10 / 10 / 10."""
    items = _load_golden_items()
    counts = Counter(item.question_type for item in items)
    expected = {
        QuestionType.FACTUAL: 60,
        QuestionType.MULTI_HOP: 45,
        QuestionType.CONFLICT_TRAP: 15,
        QuestionType.OUT_OF_SCOPE: 10,
        QuestionType.MALFORMED: 10,
        QuestionType.CONTEXT_NOISE: 10,
    }
    assert dict(counts) == expected, f"Unexpected distribution: {dict(counts)}"


@_SKIP_GOLDEN
def test_golden_test_ids_unique_and_sequential() -> None:
    """test_id values must be unique TEST_001..TEST_150."""
    items = _load_golden_items()
    ids = [item.test_id for item in items]
    assert len(set(ids)) == 150, "Duplicate test_id values found"
    assert ids == [f"TEST_{i:03d}" for i in range(1, 151)]


@_SKIP_GOLDEN
def test_golden_normalized_queries_unique() -> None:
    """Normalized queries must be unique across all 150 golden items."""
    items = _load_golden_items()
    normed = [" ".join(normalize_query(it.query).lower().split()) for it in items]
    dupes = [q for q, cnt in Counter(normed).items() if cnt > 1]
    assert not dupes, f"Duplicate normalized queries: {dupes[:5]}"


@_SKIP_GOLDEN
def test_golden_out_of_scope_exact() -> None:
    """OUT_OF_SCOPE items must match exact queries, routes, and empty expected IDs."""
    items = _load_golden_items()
    oos = [it for it in items if it.question_type == QuestionType.OUT_OF_SCOPE]
    assert len(oos) == 10
    actual_pairs = [(it.query, it.expected_route) for it in oos]
    assert actual_pairs == _EXPECTED_OOS
    for it in oos:
        assert it.expected_source_chunk_ids == []
        assert it.expected_doc_ids == []
        assert it.negative_constraint_test is True
        assert it.ground_truth_answer == "INSUFFICIENT_CONTEXT"


@_SKIP_GOLDEN
def test_golden_factual_route_split() -> None:
    """FACTUAL must have 6 FAST_PATH_EXACT_LOOKUP and 54 SEMANTIC_RAG items."""
    items = _load_golden_items()
    factual = [it for it in items if it.question_type == QuestionType.FACTUAL]
    routes = Counter(it.expected_route for it in factual)
    assert routes[ROUTE_FAST_PATH] == 6
    assert routes[ROUTE_SEMANTIC] == 54


@_SKIP_GOLDEN
def test_golden_conflict_traps_spread_and_versions() -> None:
    """CONFLICT_TRAP must cover LLM03x5, LLM05x5, LLM02x2, LLM04x1, LLM06x1, LLM07x1."""
    items = _load_golden_items()
    conflicts = [it for it in items if it.question_type == QuestionType.CONFLICT_TRAP]
    assert len(conflicts) == 15
    llm_counts: Counter[str] = Counter()
    for it in conflicts:
        assert it.conflict_test is True
        assert it.negative_constraint_test is True
        assert it.expected_route == ROUTE_SEMANTIC
        assert not re.search(r"\bv?(1\.1|2\.0)\b", it.query, re.IGNORECASE), (
            f"{it.test_id}: version leaked into CONFLICT_TRAP query: {it.query!r}"
        )
        has_v11 = any("_v1.1_" in d for d in it.expected_doc_ids)
        has_v20 = any("_v2.0_" in d for d in it.expected_doc_ids)
        assert has_v11 and has_v20, f"{it.test_id}: missing v1.1 or v2.0 doc_id"
        m = re.search(r"_(LLM\d{2})$", it.expected_doc_ids[0])
        assert m is not None
        llm_counts[m.group(1)] += 1

    assert dict(llm_counts) == {
        "LLM03": 5,
        "LLM05": 5,
        "LLM02": 2,
        "LLM04": 1,
        "LLM06": 1,
        "LLM07": 1,
    }


@_SKIP_BOTH
def test_golden_chunks_and_entities_in_corpus() -> None:
    """All expected chunk IDs, doc IDs, and required entities must exist in corpus."""
    items = _load_golden_items()
    df = pd.read_parquet(_PARQUET_PATH)
    chunk_to_doc = dict(zip(df["chunk_id"].astype(str), df["doc_id"].astype(str)))
    chunk_to_content = dict(zip(df["chunk_id"].astype(str), df["content"].astype(str)))

    for it in items:
        if it.question_type == QuestionType.OUT_OF_SCOPE:
            continue
        assert it.expected_source_chunk_ids, f"{it.test_id}: empty expected_source_chunk_ids"
        for cid in it.expected_source_chunk_ids:
            assert cid in chunk_to_doc, f"{it.test_id}: missing chunk_id {cid!r}"
        derived_docs = {chunk_to_doc[cid] for cid in it.expected_source_chunk_ids}
        assert set(it.expected_doc_ids) == derived_docs, (
            f"{it.test_id}: expected_doc_ids {it.expected_doc_ids} != {sorted(derived_docs)}"
        )
        concat = "\n".join(chunk_to_content[cid] for cid in it.expected_source_chunk_ids).lower()
        for ent in it.required_entities:
            assert ent.lower() in concat, (
                f"{it.test_id}: entity {ent!r} missing from expected chunks"
            )
