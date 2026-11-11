"""
kavach.retrieval.pipeline — Unified CPU Vector Retrieval Pipeline (Tiers 1–4).

Orchestrates permanently resident CPU models:
  Tier 1: AgentIntentRouter (fast-path regex + DeBERTa intent classifier)
  Tier 2: Embedder (F2LLM-v2-0.6B with MRL-512 slicing)
  Tier 3: ChunkStore + FAISS SQ8Index
  Tier 4: Reranker (cross-encoder relevance scoring + logit abstention gate)

Eagerly loads and warms up all neural components in ``__init__``, verifies
calibrated abstention metadata against active precision and sequence-length
settings, and monitors RSS against ``Settings.rss_ceiling_mb`` (4096.0 MB).
"""
from __future__ import annotations

import os

_PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("HF_HOME", os.path.join(_PROJECT_ROOT, "models", "hf_cache"))

from kavach.runtime_guard import apply_hardware_guard, rss_mb  # noqa: E402

import dataclasses
import json
import logging
import time
from pathlib import Path
from typing import Any, Optional

import torch  # noqa: E402, F401 — imported before faiss via vector_store

from kavach.config import Settings, get_settings
from kavach.exceptions import ConfigError
from kavach.retrieval.embedder import Embedder
from kavach.retrieval.reranker import Reranker
from kavach.retrieval.router import (
    ROUTE_FAST_PATH,
    ROUTE_REFUSAL,
    AgentIntentRouter,
    RouteDecision,
)
from kavach.retrieval.vector_store import ChunkStore, SQ8Index

log = logging.getLogger(__name__)


@dataclasses.dataclass
class PipelineResult:
    """Structured output returned by :meth:`VectorRetrievalPipeline.retrieve`."""

    route: RouteDecision
    candidates: list[dict[str, Any]]
    contexts: list[dict[str, Any]]
    refused: bool
    latency_ms: dict[str, float]
    abstained: bool = False
    abstain_reason: str | None = None
    top1_logit: float | None = None
    contexts_if_not_gated: list[dict[str, Any]] = dataclasses.field(
        default_factory=list
    )


class VectorRetrievalPipeline:
    """End-to-end CPU-resident retrieval pipeline combining Tiers 1 through 4."""

    def __init__(
        self,
        settings: Optional[Settings] = None,
        *,
        parquet_path: Optional[Path | str] = None,
        index_path: Optional[Path | str] = None,
        load_reranker: bool = True,
    ) -> None:
        apply_hardware_guard()
        self._settings = settings or get_settings()
        self._abstain_threshold: float | None = self._validate_abstention_contract()

        pq_path = (
            Path(parquet_path)
            if parquet_path is not None
            else self._settings.data_processed / "corpus_chunks.parquet"
        )
        idx_path = (
            Path(index_path)
            if index_path is not None
            else self._settings.models_dir / "index_sq8_mrl512.faiss"
        )

        self._store = ChunkStore(pq_path)
        self._index = SQ8Index.load(idx_path)

        self._router = AgentIntentRouter(self._settings)
        self._router.load()

        self._embedder = Embedder(self._settings)
        self._reranker: Optional[Reranker] = (
            Reranker(self._settings) if load_reranker else None
        )

        # Warm up all neural components to prime oneDNN caches
        self._router.warmup()
        self._embedder.warmup()
        if self._reranker is not None:
            self._reranker.warmup()

        self.rss_after_load_mb: float = round(rss_mb(), 2)
        self.within_rss_ceiling: bool = (
            self.rss_after_load_mb <= float(self._settings.rss_ceiling_mb)
        )
        if self.within_rss_ceiling:
            log.info(
                "VectorRetrievalPipeline resident and warmed up. "
                "RSS=%.1f MB (ceiling=%.1f MB)",
                self.rss_after_load_mb,
                self._settings.rss_ceiling_mb,
            )
        else:
            log.warning(
                "VectorRetrievalPipeline RSS=%.1f MB exceeds ceiling=%.1f MB.",
                self.rss_after_load_mb,
                self._settings.rss_ceiling_mb,
            )

    def _validate_abstention_contract(self) -> float | None:
        """Verify calibrated abstention JSON exists and matches active settings."""
        if not self._settings.abstention_enabled:
            return None

        cal_path = self._settings.models_dir / "abstain_calibration.json"
        if not cal_path.exists():
            raise ConfigError(
                f"abstention_enabled=True requires calibration file at {cal_path}.",
                details={"abstain_calibration_path": str(cal_path)},
            )

        try:
            cal = json.loads(cal_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ConfigError(
                f"Failed to parse {cal_path}: {exc}",
                details={"abstain_calibration_path": str(cal_path), "error": str(exc)},
            ) from exc

        if not isinstance(cal, dict):
            raise ConfigError(
                f"Invalid JSON object in {cal_path}.",
                details={"abstain_calibration_path": str(cal_path)},
            )

        cal_prec = cal.get("reranker_precision")
        if cal_prec != self._settings.rerank_precision:
            raise ConfigError(
                f"Stale abstention calibration: reranker_precision={cal_prec!r} "
                f"!= settings.rerank_precision={self._settings.rerank_precision!r}.",
                details={
                    "calibration_reranker_precision": cal_prec,
                    "settings_rerank_precision": self._settings.rerank_precision,
                },
            )

        cal_max_len = cal.get("rerank_max_length")
        if cal_max_len != self._settings.rerank_max_length:
            raise ConfigError(
                f"Stale abstention calibration: rerank_max_length={cal_max_len!r} "
                f"!= settings.rerank_max_length={self._settings.rerank_max_length!r}.",
                details={
                    "calibration_rerank_max_length": cal_max_len,
                    "settings_rerank_max_length": self._settings.rerank_max_length,
                },
            )

        if "contingency_l_applied" in cal or self._settings.contingency_l_applied:
            cal_cont_l = bool(cal.get("contingency_l_applied", False))
            if cal_cont_l != bool(self._settings.contingency_l_applied):
                raise ConfigError(
                    "Stale abstention calibration: contingency_l_applied mismatch.",
                    details={
                        "calibration_contingency_l_applied": cal_cont_l,
                        "settings_contingency_l_applied": self._settings.contingency_l_applied,
                    },
                )
            if self._settings.contingency_l_applied:
                cal_long_len = cal.get("rerank_long_query_max_length")
                if cal_long_len != self._settings.rerank_long_query_max_length:
                    raise ConfigError(
                        "Stale abstention calibration: rerank_long_query_max_length mismatch.",
                        details={
                            "calibration_rerank_long_query_max_length": cal_long_len,
                            "settings_rerank_long_query_max_length": self._settings.rerank_long_query_max_length,
                        },
                    )

        threshold_val = self._settings.abstain_logit_threshold
        if threshold_val is None:
            raw_t = cal.get("threshold_T")
            if isinstance(raw_t, (int, float)) and not isinstance(raw_t, bool):
                threshold_val = float(raw_t)
        if threshold_val is None:
            raise ConfigError(
                f"No valid threshold_T found in {cal_path} or Settings.abstain_logit_threshold.",
                details={"abstain_calibration_path": str(cal_path)},
            )
        return float(threshold_val)

    @property
    def store(self) -> ChunkStore:
        """Expose underlying ChunkStore for inspection or doc_id lookups."""
        return self._store

    def retrieve(
        self,
        query: str,
        rerank: bool = True,
        top_k: int = 3,
        candidate_k: Optional[int] = None,
    ) -> PipelineResult:
        """Run routing, retrieval, reranking, and abstention gating for *query*."""
        t_total_start = time.perf_counter()
        latency_ms: dict[str, float] = {}
        eff_candidate_k = (
            int(candidate_k)
            if candidate_k is not None
            else int(self._settings.candidate_k)
        )

        # ── Tier 1: Intent Routing ───────────────────────────────────────────
        decision = self._router.route(query)
        latency_ms["router_ms"] = round(decision.latency_ms, 3)

        # ── Route B: Hard Refusal ────────────────────────────────────────────
        if decision.route == ROUTE_REFUSAL:
            latency_ms["total_ms"] = round(
                (time.perf_counter() - t_total_start) * 1000.0, 3
            )
            return PipelineResult(
                route=decision,
                candidates=[],
                contexts=[],
                refused=True,
                latency_ms=latency_ms,
                abstained=False,
                abstain_reason=None,
                top1_logit=None,
                contexts_if_not_gated=[],
            )

        # ── Route A: Fast-Path Exact Identifier Lookup ───────────────────────
        if decision.route == ROUTE_FAST_PATH and decision.matched_identifier:
            t_lookup = time.perf_counter()
            matches = self._store.lookup_identifier(decision.matched_identifier)
            latency_ms["lookup_ms"] = round(
                (time.perf_counter() - t_lookup) * 1000.0, 3
            )
            if matches:
                cap = min(max(int(top_k), 1), self._settings.rerank_top_k_max)
                capped = [dict(m) for m in matches[:cap]]
                latency_ms["total_ms"] = round(
                    (time.perf_counter() - t_total_start) * 1000.0, 3
                )
                return PipelineResult(
                    route=decision,
                    candidates=capped,
                    contexts=capped,
                    refused=False,
                    latency_ms=latency_ms,
                    abstained=False,
                    abstain_reason=None,
                    top1_logit=None,
                    contexts_if_not_gated=[],
                )
            log.info(
                "Identifier %r absent from ChunkStore hash index; falling back to semantic search.",
                decision.matched_identifier,
            )

        # ── Route C: Semantic RAG (Tier 2 -> Tier 3 -> Tier 4) ───────────────
        search_text = decision.normalized_query or query

        t_embed = time.perf_counter()
        emb_batch = self._embedder.embed([search_text], kind="query", batch_size=1)
        qvec = emb_batch.mrl512[0]
        self._embedder.clear_cache()
        latency_ms["embed_ms"] = round((time.perf_counter() - t_embed) * 1000.0, 3)

        t_search = time.perf_counter()
        hits = self._index.search(qvec, k=eff_candidate_k)
        candidates: list[dict[str, Any]] = []
        for chunk_id, score in hits:
            rec = self._store.get(chunk_id)
            rec["vector_score"] = float(score)
            candidates.append(rec)
        latency_ms["search_ms"] = round((time.perf_counter() - t_search) * 1000.0, 3)

        out_k = max(0, min(int(top_k), self._settings.rerank_top_k_max))
        abstained = False
        abstain_reason: str | None = None
        top1_logit: float | None = None

        if rerank and candidates:
            if self._reranker is None:
                self._reranker = Reranker(self._settings)
                self._reranker.warmup()
            t_rerank = time.perf_counter()
            all_scored = self._reranker.rerank(
                search_text,
                candidates,
                top_k=out_k,
                return_all=True,
            )
            latency_ms["rerank_ms"] = round(
                (time.perf_counter() - t_rerank) * 1000.0, 3
            )
            top1_logit = (
                float(all_scored[0]["rerank_raw_logit"]) if all_scored else None
            )
            topk_results = all_scored[:out_k]

            if self._settings.abstention_enabled:
                assert self._abstain_threshold is not None
                if top1_logit is None or top1_logit < self._abstain_threshold:
                    abstained = True
                    abstain_reason = "LOW_RETRIEVAL_CONFIDENCE"
                    contexts: list[dict[str, Any]] = []
                else:
                    contexts = topk_results
            else:
                contexts = topk_results
            contexts_if_not_gated = topk_results
        else:
            topk_results = [dict(c) for c in candidates[:out_k]]
            if self._settings.abstention_enabled and not candidates:
                abstained = True
                abstain_reason = "LOW_RETRIEVAL_CONFIDENCE"
                contexts = []
            else:
                contexts = topk_results
            contexts_if_not_gated = topk_results

        latency_ms["total_ms"] = round(
            (time.perf_counter() - t_total_start) * 1000.0, 3
        )
        return PipelineResult(
            route=decision,
            candidates=candidates,
            contexts=contexts,
            refused=False,
            latency_ms=latency_ms,
            abstained=abstained,
            abstain_reason=abstain_reason,
            top1_logit=top1_logit,
            contexts_if_not_gated=contexts_if_not_gated,
        )
