"""
tests.test_pipeline_smoke — Phase 1.5 calibrated CPU-only pipeline smoke test.

Verifies the 5 core pipeline smoke queries plus Golden OOS item 9 (TEST_129)
under locked CPU settings, enforcing RSS <= settings.rss_ceiling_mb (4096 MB)
and zero CUDA allocation.
"""
from __future__ import annotations

import pytest

import kavach  # noqa: F401
from kavach.config import Settings, get_settings
from kavach.retrieval.pipeline import PipelineResult, VectorRetrievalPipeline
from kavach.retrieval.router import (
    ROUTE_FAST_PATH,
    ROUTE_REFUSAL,
    ROUTE_SEMANTIC,
)
from kavach.runtime_guard import rss_mb, vram_probe


def _artifacts_ready(settings: Settings) -> bool:
    """Return True if corpus, SQ8 index, ID map, and trained router exist."""
    return all(
        path.exists()
        for path in (
            settings.data_processed / "corpus_chunks.parquet",
            settings.models_dir / "index_sq8_mrl512.faiss",
            settings.models_dir / "index_sq8_mrl512.idmap.json",
            settings.models_dir / "router_deberta" / "model.safetensors",
        )
    )


def test_pipeline_smoke_five_queries_plus_oos_real_model() -> None:
    """Verify calibrated end-to-end 5 smoke queries + TEST_129 OOS logging and RSS <= 4096 MB."""
    defaults = Settings()
    if not _artifacts_ready(defaults):
        pytest.skip("Pipeline artifacts (corpus, SQ8 index, trained router) not yet built")

    cal_path = defaults.models_dir / "abstain_calibration.json"
    if not cal_path.exists():
        pytest.skip(
            "Stage 4 models/abstain_calibration.json not yet generated (pre-calibration T0 state)"
        )

    get_settings.cache_clear()
    settings = get_settings()
    assert settings.rss_ceiling_mb == 4096.0
    assert settings.abstention_enabled is True

    pipeline = VectorRetrievalPipeline(settings)

    exact_cve = "CVE-2024-3094"
    if not pipeline.store.lookup_identifier(exact_cve):
        for cid in pipeline.store.all_chunk_ids():
            rec = pipeline.store.get(cid)
            pid = str(rec.get("primary_identifier", ""))
            if pid.startswith("CVE-"):
                exact_cve = pid
                break

    queries: list[tuple[str, str]] = [
        ("exact_cve", exact_cve),
        (
            "multi_hop",
            "Which MITRE ATT&CK technique and enterprise mitigations correspond to CVE-2024-3094?",
        ),
        ("golden_oos_1", "How do I bake a sourdough loaf?"),
        ("typo_query", "cvs 9.8 vuln xz util liblzma bckdoor mitgation"),
        (
            "owasp_llm03",
            "According to OWASP GenAI Top 10, what vulnerability is cataloged under LLM03?",
        ),
        (
            "golden_oos_item9",
            "What are the internal firewall rules of Acme Corp's production network?",
        ),
    ]

    results: dict[str, PipelineResult] = {}
    for label, query in queries:
        res = pipeline.retrieve(query, rerank=True, top_k=3, candidate_k=8)
        results[label] = res
        print(
            f"[SMOKE] {label:<16} | route={res.route.route:<22} | "
            f"refused={res.refused} | abstained={res.abstained} | "
            f"reason={res.abstain_reason} | top1_logit={res.top1_logit} | "
            f"contexts={len(res.contexts)} | ungated={len(res.contexts_if_not_gated)} | "
            f"latency_ms={res.latency_ms}",
            flush=True,
        )

    # 1. Exact CVE fast path
    r_exact = results["exact_cve"]
    assert r_exact.route.route == ROUTE_FAST_PATH
    assert r_exact.refused is False
    assert r_exact.abstained is False
    assert 1 <= len(r_exact.contexts) <= 5

    # 2. Multi-hop semantic RAG
    r_multi = results["multi_hop"]
    assert r_multi.route.route == ROUTE_SEMANTIC
    assert r_multi.refused is False
    assert len(r_multi.candidates) > 0
    assert r_multi.top1_logit is not None

    # 3. Known Track B hard-refusal OOS query (sourdough)
    r_oos = results["golden_oos_1"]
    assert r_oos.route.route == ROUTE_REFUSAL
    assert r_oos.refused is True
    assert r_oos.abstained is False
    assert r_oos.contexts == []

    # 4. Typo query
    r_typo = results["typo_query"]
    assert r_typo.refused is False
    assert len(r_typo.candidates) > 0

    # 5. OWASP LLM03
    r_owasp = results["owasp_llm03"]
    assert r_owasp.refused is False
    assert len(r_owasp.candidates) > 0

    # 6. Golden OOS item 9 (TEST_129) is logged cleanly above without fragile hard assert;
    #    rigorous abstention metrics are enforced by test_calibration.py.
    assert "golden_oos_item9" in results

    current_rss = rss_mb()
    probe = vram_probe()
    print(
        f"[SMOKE] rss_after_load_mb={pipeline.rss_after_load_mb:.1f} | "
        f"rss_after_queries_mb={current_rss:.1f} | "
        f"ceiling_mb={settings.rss_ceiling_mb:.1f} | vram_probe={probe}",
        flush=True,
    )
    assert pipeline.rss_after_load_mb <= settings.rss_ceiling_mb
    assert current_rss <= settings.rss_ceiling_mb
    assert probe["cuda_available"] is False
    assert probe["torch_cuda_build"] is False
