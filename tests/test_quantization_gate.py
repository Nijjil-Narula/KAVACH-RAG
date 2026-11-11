"""
tests.test_quantization_gate — Verification of INT8 quantization safety & precision lock.

Validates:
  1. ``check_quantize_dynamic_safety()`` succeeds cleanly on CPU with supported
     engine and FP32-vs-INT8 cosine similarity > 0.99.
  2. ``quantize_linear_layers(model, exclude_names=...)`` quantizes non-excluded
     backbone Linear layers while preserving excluded heads/embeddings in FP32.
  3. When ``models/precision_lock.json`` and ``reports/phase15_quality_gates.json``
     exist (Stage 3+), they are mutually consistent and satisfy all quality gates.
"""
from __future__ import annotations

import json

import pytest
import torch
import torch.nn as nn


class _SyntheticCrossEncoderLike(nn.Module):
    """Minimal network mirroring ModernBERT backbone + excluded dense1/norm/dense2 head."""

    def __init__(self) -> None:
        super().__init__()
        self.embeddings = nn.ModuleDict({"tok_embeddings": nn.Embedding(32, 16)})
        self.backbone = nn.Sequential(
            nn.Linear(16, 32),
            nn.GELU(),
            nn.Linear(32, 16),
        )
        self.dense1 = nn.Linear(16, 16)
        self.norm = nn.LayerNorm(16)
        self.dense2 = nn.Linear(16, 1)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        emb = self.embeddings["tok_embeddings"](token_ids).mean(dim=1)
        hidden = self.backbone(emb)
        hidden = torch.nn.functional.gelu(self.dense1(hidden))
        hidden = self.norm(hidden)
        return self.dense2(hidden)


class _SyntheticRouterLike(nn.Module):
    """Minimal network mirroring DeBERTa encoder Linear layers + pooler/classifier."""

    def __init__(self) -> None:
        super().__init__()
        self.deberta = nn.Module()
        self.deberta.embeddings = nn.Embedding(32, 16)
        self.deberta.encoder = nn.Module()
        self.deberta.encoder.intermediate_dense = nn.Linear(16, 32)
        self.deberta.encoder.output_dense = nn.Linear(32, 16)
        self.pooler = nn.Module()
        self.pooler.dense = nn.Linear(16, 16)
        self.classifier = nn.Linear(16, 3)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        x = self.deberta.embeddings(token_ids).mean(dim=1)
        h = torch.relu(self.deberta.encoder.intermediate_dense(x))
        h = self.deberta.encoder.output_dense(h)
        pooled = torch.tanh(self.pooler.dense(h))
        return self.classifier(pooled)


def _is_quantized_linear(mod: nn.Module) -> bool:
    return "quantized" in type(mod).__module__ and type(mod).__name__ == "Linear"


def test_quantize_dynamic_safety_smoke() -> None:
    """check_quantize_dynamic_safety() must report smoke_ok=True and cosine > 0.99."""
    import kavach  # noqa: F401
    from kavach.retrieval.quantization import check_quantize_dynamic_safety

    res = check_quantize_dynamic_safety()
    assert res["smoke_ok"] is True, f"Quantization safety smoke failed: {res}"
    assert res["cosine_similarity"] > 0.99
    assert res["quantized_linear_count"] == 2
    assert res["engine"] in res["supported_engines"]
    assert res["import_path"].endswith("quantize_dynamic")


def test_quantize_linear_layers_excludes_heads_and_embeddings() -> None:
    """Excluded head/pooler/embedding layers must stay FP32 torch.nn.Linear."""
    import kavach  # noqa: F401
    from kavach.retrieval.quantization import quantize_linear_layers

    torch.manual_seed(7)
    sample_tokens = torch.tensor([[1, 3, 5, 7], [2, 4, 6, 8]], dtype=torch.long)

    ce = _SyntheticCrossEncoderLike().eval()
    q_ce = quantize_linear_layers(
        ce, exclude_names=("dense1", "dense2", "norm", "classifier", "embeddings")
    )
    assert isinstance(ce.backbone[0], nn.Linear)
    assert _is_quantized_linear(q_ce.backbone[0])
    assert _is_quantized_linear(q_ce.backbone[2])
    assert type(q_ce.dense1) is nn.Linear
    assert q_ce.dense1.weight.dtype == torch.float32
    assert type(q_ce.dense2) is nn.Linear
    assert q_ce.dense2.weight.dtype == torch.float32

    with torch.inference_mode():
        ce_out = ce(sample_tokens)
        q_ce_out = q_ce(sample_tokens)
    assert q_ce_out.shape == ce_out.shape
    assert torch.isfinite(q_ce_out).all()

    router = _SyntheticRouterLike().eval()
    q_router = quantize_linear_layers(
        router, exclude_names=("embeddings", "classifier", "pooler")
    )
    assert _is_quantized_linear(q_router.deberta.encoder.intermediate_dense)
    assert _is_quantized_linear(q_router.deberta.encoder.output_dense)
    assert type(q_router.pooler.dense) is nn.Linear
    assert q_router.pooler.dense.weight.dtype == torch.float32
    assert type(q_router.classifier) is nn.Linear
    assert q_router.classifier.weight.dtype == torch.float32

    with torch.inference_mode():
        r_out = router(sample_tokens)
        q_r_out = q_router(sample_tokens)
    assert q_r_out.shape == r_out.shape
    assert torch.isfinite(q_r_out).all()


def test_precision_lock_matches_quality_gates_if_present() -> None:
    """When Stage 3 lock and report exist, assert gate pass and lock consistency."""
    from kavach.config import Settings

    settings = Settings()
    lock_path = settings.models_dir / "precision_lock.json"
    gates_path = settings.reports_dir / "phase15_quality_gates.json"

    if not lock_path.exists() and not gates_path.exists():
        pytest.skip("Stage 3 precision_lock.json and phase15_quality_gates.json not yet written")

    assert lock_path.exists(), f"Missing {lock_path} when {gates_path} exists"
    assert gates_path.exists(), f"Missing {gates_path} when {lock_path} exists"

    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    gates = json.loads(gates_path.read_text(encoding="utf-8"))

    assert gates["gate_passed"] is True
    assert gates["rss_ceiling_mb"] == 4096.0
    assert lock["rss_ceiling_mb"] == 4096.0
    assert lock["resident_rss_mb"] <= 4096.0

    required_lock_keys = (
        "embed_precision",
        "rerank_precision",
        "router_precision",
        "rerank_max_length",
        "cpu_affinity_mode",
        "quantized_engine",
        "torch_version",
        "locked_at",
        "gate_report_path",
        "resident_rss_mb",
        "rss_ceiling_mb",
        "contingency_l_applied",
        "rerank_long_query_max_length",
        "rerank_long_query_token_threshold",
        "downgrades_applied",
        "vram_probe",
    )
    for key in required_lock_keys:
        assert key in lock, f"Missing required Section 3 key {key!r} in precision_lock.json"

    assert lock["quantized_engine"] in ("onednn", "fbgemm", "x86")
    assert isinstance(lock["torch_version"], str) and lock["torch_version"]
    assert isinstance(lock["locked_at"], str) and lock["locked_at"]
    assert lock["gate_report_path"] == "reports/phase15_quality_gates.json"

    passing = gates["passing_candidates"]
    assert lock["embed_precision"] in passing["embedder"]
    assert lock["rerank_precision"] in passing["reranker"]
    assert lock["router_precision"] in passing["router"]

    locked_in_report = gates["precision_lock"]
    for key in (
        "embed_precision",
        "rerank_precision",
        "router_precision",
        "rerank_max_length",
        "cpu_affinity_mode",
        "quantized_engine",
        "torch_version",
        "locked_at",
        "gate_report_path",
        "contingency_l_applied",
    ):
        assert lock[key] == locked_in_report[key]

    assert lock["vram_probe"]["cuda_available"] is False
    assert gates["vram_probe"]["cuda_available"] is False


def test_reranker_gate_thresholds_strict_098_and_matrix_ranking() -> None:
    """Verify strict >=0.98 top3_overlap threshold, matrix speed ranking, and 40-query strata."""
    from kavach.config import Settings
    from kavach.retrieval.profiling import (
        _comparison_passes,
        _load_golden_items,
        _rank_passing_by_matrix_latency,
        _select_matrix_indices,
    )
    from kavach.retrieval.quantization import get_quantized_engine_info

    info = get_quantized_engine_info()
    assert info["engine"] in info["supported_engines"]

    assert _comparison_passes(
        {
            "doc_mrr_delta": -0.009,
            "chunk_mrr_delta": 0.01,
            "top3_overlap": 0.98,
            "logit_corr": 0.99,
        }
    ) is True
    # 0.95/0.979 must fail under strict >= 0.98 spec
    assert _comparison_passes(
        {
            "doc_mrr_delta": 0.0,
            "chunk_mrr_delta": 0.0,
            "top3_overlap": 0.9799,
            "logit_corr": 0.999,
        }
    ) is False

    fake_matrix = {
        "embedder": {
            "fp32": {"latency_ms": {"mean": 180.0}},
            "int8": {"latency_ms": {"mean": 110.0}},
            "bf16": {"latency_ms": {"mean": 220.0}},
        },
        "reranker": {
            "fp32@640": {"latency_ms": {"mean": 950.0}},
            "int8@640": {"latency_ms": {"mean": 620.0}},
        },
        "router": {
            "fp32": {"latency_ms": {"mean": 42.0}},
            "int8": {"latency_ms": {"mean": 28.0}},
        },
    }
    assert _rank_passing_by_matrix_latency("embedder", ["fp32", "int8"], fake_matrix) == [
        "int8",
        "fp32",
    ]
    assert _rank_passing_by_matrix_latency("reranker", ["fp32", "int8"], fake_matrix) == [
        "int8",
        "fp32",
    ]
    assert _rank_passing_by_matrix_latency("router", ["fp32", "int8"], fake_matrix) == [
        "int8",
        "fp32",
    ]
    with pytest.raises(RuntimeError, match="Missing Stage 2 --matrix"):
        _rank_passing_by_matrix_latency("embedder", ["fp32"], {})

    items = _load_golden_items(Settings())
    indices, counts = _select_matrix_indices(items)
    assert len(indices) == 40
    assert len(set(indices)) == 40
    assert counts == {
        "FACTUAL": 10,
        "MULTI_HOP": 10,
        "CONTEXT_NOISE": 10,
        "CONFLICT_TRAP": 5,
        "MALFORMED": 3,
        "OUT_OF_SCOPE": 2,
    }
