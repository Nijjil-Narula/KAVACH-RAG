"""
tests.test_reranker — CPU-only unit & integration tests for kavach.retrieval.reranker.

Verifies:
  - ``logits_to_probs_and_raw`` / ``logits_to_probs`` on 1-logit and 2-logit heads.
  - Descending sorting strictly by ``rerank_raw_logit``, ``return_all`` & ``top_k`` cap,
    and zero mutation of caller candidate dicts.
  - D2 passage-first pair construction at 576 tokens (query-only truncation with
    warning + truncation_stats, ``RetrievalError`` on passage > 512 tokens, and
    Contingency L 768 relaxation on queries > 96 tokens).
  - Real cross-encoder scoring on CPU (run during Stage 7 ``pytest -q``).
"""
from __future__ import annotations

import logging
import math
from typing import Any

import pytest
import torch

import kavach  # noqa: F401
from kavach.config import Settings
from kavach.exceptions import RetrievalError
from kavach.retrieval.reranker import (
    Reranker,
    _detect_pair_template,
    logits_to_probs,
    logits_to_probs_and_raw,
)


class _DummyWordTokenizer:
    """Minimal deterministic whitespace tokenizer mimicking HF TokenizersBackend."""

    pad_token_id = 0
    padding_side = "right"

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        ids = [1000 + (abs(hash(tok)) % 5000) for tok in text.split() if tok]
        if add_special_tokens:
            return [101, *ids, 102]
        return ids

    def __call__(
        self,
        query: str,
        passage: str | None = None,
        *,
        add_special_tokens: bool = True,
    ) -> dict[str, list[int]]:
        q_ids = self.encode(query, add_special_tokens=False)
        if passage is None:
            ids = [101, *q_ids, 102] if add_special_tokens else q_ids
            return {"input_ids": ids, "attention_mask": [1] * len(ids)}
        p_ids = self.encode(passage, add_special_tokens=False)
        ids = [101, *q_ids, 102, *p_ids, 102] if add_special_tokens else [*q_ids, *p_ids]
        return {"input_ids": ids, "attention_mask": [1] * len(ids)}


def test_logits_to_probs_and_raw_single_and_two_logit() -> None:
    """Single-logit (sigmoid) and two-logit (softmax/log-odds) heads must align."""
    t1 = torch.tensor([0.0, 2.0, -2.0], dtype=torch.float32)
    probs1, raw1 = logits_to_probs_and_raw(t1)
    assert probs1 == pytest.approx(logits_to_probs(t1))
    assert probs1[0] == pytest.approx(0.5, abs=1e-5)
    assert probs1[1] > 0.85
    assert probs1[2] < 0.15
    assert raw1 == pytest.approx([0.0, 2.0, -2.0], abs=1e-6)

    t1_col = t1.unsqueeze(-1)
    probs1c, raw1c = logits_to_probs_and_raw(t1_col)
    assert probs1c == pytest.approx(probs1, abs=1e-6)
    assert raw1c == pytest.approx(raw1, abs=1e-6)

    t2 = torch.tensor([[-2.0, 2.0], [1.0, 1.0], [3.0, -1.0]], dtype=torch.float32)
    probs2, raw2 = logits_to_probs_and_raw(t2)
    assert probs2[0] > 0.98
    assert probs2[1] == pytest.approx(0.5, abs=1e-5)
    assert probs2[2] < 0.02
    assert raw2 == pytest.approx([4.0, 0.0, -4.0], abs=1e-6)


def test_reranker_logit_sorting_topk_and_no_mutation_unit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reranker.rerank() must sort by rerank_raw_logit descending without mutating inputs."""
    settings = Settings(rerank_top_k_max=5)
    rr = Reranker.__new__(Reranker)
    rr._settings = settings
    rr._tokenizer = object()
    rr._model = object()
    rr._pair_template = None
    rr.model_id = "unit-dummy"
    rr.precision = "fp32"
    rr.is_fallback = False
    rr._truncation_stats = {
        "pairs_scored": 0,
        "queries_truncated": 0,
        "passages_truncated": 0,
        "contingency_l_pairs": 0,
    }

    fake_scores = [
        (0.01, -4.595),
        (0.95, 2.944),
        (0.20, -1.386),
        (0.85, 1.735),
        (0.60, 0.405),
        (0.99, 4.595),
    ]
    monkeypatch.setattr(rr, "_score_pairs", lambda q, texts: fake_scores[: len(texts)])

    candidates = [
        {"chunk_id": f"c{i}", "content": f"passage {i}"} for i in range(6)
    ]
    all_scored = rr.rerank("test query", candidates, top_k=2, return_all=True)
    assert [c["chunk_id"] for c in all_scored] == ["c5", "c1", "c3", "c4", "c2", "c0"]
    assert all_scored[0]["rerank_raw_logit"] == pytest.approx(4.595)
    assert all_scored[-1]["rerank_raw_logit"] == pytest.approx(-4.595)

    top2 = rr.rerank("test query", candidates, top_k=2, return_all=False)
    assert [c["chunk_id"] for c in top2] == ["c5", "c1"]

    capped = rr.rerank("test query", candidates, top_k=20, return_all=False)
    assert len(capped) == 5
    assert [c["chunk_id"] for c in capped] == ["c5", "c1", "c3", "c4", "c2"]

    for orig in candidates:
        assert "rerank_score" not in orig
        assert "rerank_raw_logit" not in orig


def test_reranker_d2_passage_first_construction_unit(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Verify D2 passage-first pair assembly, query truncation, overflow error, and Contingency L."""
    tok = _DummyWordTokenizer()
    rr = Reranker.__new__(Reranker)
    rr._settings = Settings(rerank_max_length=576)
    rr._tokenizer = tok
    rr._model = object()
    rr._pair_template = _detect_pair_template(tok)
    rr.model_id = "unit-dummy"
    rr.precision = "fp32"
    rr.is_fallback = False
    rr._truncation_stats = {
        "pairs_scored": 0,
        "queries_truncated": 0,
        "passages_truncated": 0,
        "contingency_l_pairs": 0,
    }

    # 1. Untruncated pair matches tokenizer(query, passage)
    q_short = " ".join(f"q{i}" for i in range(12))
    p_normal = " ".join(f"p{i}" for i in range(40))
    built = rr.build_pair_inputs(q_short, p_normal)
    expected = tok(q_short, p_normal, add_special_tokens=True)["input_ids"]
    assert built["input_ids"] == expected
    assert rr.truncation_stats["queries_truncated"] == 0
    assert rr.truncation_stats["passages_truncated"] == 0

    # 2. Long query + 500-token passage at 576 -> query truncated to 73 tokens, passage untouched
    q_long = " ".join(f"q{i}" for i in range(100))
    p_500 = " ".join(f"p{i}" for i in range(500))
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        trunc_pair = rr.build_pair_inputs(q_long, p_500)
    assert len(trunc_pair["input_ids"]) == 576
    p_expected_ids = tok.encode(p_500, add_special_tokens=False)
    # Tail of [CLS] q_trunc [SEP] p_500 [SEP] must retain every passage token in full
    assert trunc_pair["input_ids"][-(len(p_expected_ids) + 1) : -1] == p_expected_ids
    assert rr.truncation_stats["queries_truncated"] == 1
    assert rr.truncation_stats["passages_truncated"] == 0
    assert any("Reranker D2 query truncation" in rec.message for rec in caplog.records)

    # 3. Contingency L raises long-query pair budget to 768 when query > 96 tokens
    rr._settings = Settings(
        rerank_max_length=576,
        rerank_long_query_max_length=768,
        rerank_long_query_token_threshold=96,
        contingency_l_applied=True,
    )
    rr.reset_truncation_stats()
    cont_pair = rr.build_pair_inputs(q_long, p_500)
    assert len(cont_pair["input_ids"]) == 603
    assert rr.truncation_stats["queries_truncated"] == 0
    assert rr.truncation_stats["contingency_l_pairs"] == 1

    # 4. Passage > 512 tokens must raise RetrievalError loudly
    p_513 = " ".join(f"p{i}" for i in range(513))
    with pytest.raises(RetrievalError, match="Passage token count 513"):
        rr.build_pair_inputs(q_short, p_513)


def test_reranker_real_model() -> None:
    """Load real cross-encoder on CPU and verify logit ordering & sigmoid consistency."""
    settings = Settings(rerank_precision="fp32")
    reranker = Reranker(settings)
    assert reranker.is_loaded is True

    query = "How can CVE-2024-3094 in XZ Utils liblzma be mitigated?"
    candidates: list[dict[str, Any]] = [
        {
            "chunk_id": "off_topic_sourdough",
            "content": "Ferment flour and water starter at room temperature to bake crusty sourdough.",
        },
        {
            "chunk_id": "relevant_xz",
            "content": (
                "CVE-2024-3094 affects XZ Utils 5.6.0 and 5.6.1 liblzma SSHD builds. "
                "Mitigation: immediately downgrade xz and liblzma to uncompromised 5.4.x packages."
            ),
        },
        {
            "chunk_id": "semi_related",
            "content": "Package repositories provide cryptographic signatures for Linux packages.",
        },
        {
            "chunk_id": "off_topic_network",
            "content": "Configure DHCP lease durations on office wireless access points.",
        },
    ]

    ranked = reranker.rerank(query, candidates, top_k=3, return_all=False)
    assert len(ranked) == 3
    assert ranked[0]["chunk_id"] == "relevant_xz"
    logits = [float(c["rerank_raw_logit"]) for c in ranked]
    assert logits == sorted(logits, reverse=True)
    for item in ranked:
        raw_val = float(item["rerank_raw_logit"])
        prob_val = float(item["rerank_score"])
        assert math.isfinite(raw_val)
        expected_prob = 1.0 / (1.0 + math.exp(-raw_val))
        assert prob_val == pytest.approx(expected_prob, abs=1e-5)
