"""
tests.test_router — Unit and integration tests for Tier 1 AgentIntentRouter.

Run with:
    .venv\\Scripts\\python.exe -m pytest -q tests/test_router.py
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from kavach.config import Settings, get_settings
from kavach.retrieval.router import (
    ROUTE_FAST_PATH,
    ROUTE_REFUSAL,
    ROUTE_SEMANTIC,
    AgentIntentRouter,
    RouteDecision,
    _RE_CVE_EXACT,
    _RE_CWE_EXACT,
    _RE_MITRE_EXACT,
    normalize_query,
)


@pytest.mark.parametrize(
    "ident,pattern",
    [
        ("CVE-2024-3094", _RE_CVE_EXACT),
        ("CVE-2021-44228", _RE_CVE_EXACT),
        ("CVE-2024-1234567", _RE_CVE_EXACT),
        ("CWE-89", _RE_CWE_EXACT),
        ("CWE-787", _RE_CWE_EXACT),
        ("CWE-12345", _RE_CWE_EXACT),
        ("T1190", _RE_MITRE_EXACT),
        ("T1059.001", _RE_MITRE_EXACT),
        ("TA0001", _RE_MITRE_EXACT),
        ("M1051", _RE_MITRE_EXACT),
    ],
)
def test_exact_regex_positives(ident: str, pattern) -> None:
    """Canonical identifiers must match the anchored Track A regexes."""
    assert pattern.match(ident) is not None


@pytest.mark.parametrize(
    "text,pattern",
    [
        ("CVE-2024-123", _RE_CVE_EXACT),
        ("What is CVE-2024-3094?", _RE_CVE_EXACT),
        ("CWE-123456", _RE_CWE_EXACT),
        ("T119", _RE_MITRE_EXACT),
        ("TA001", _RE_MITRE_EXACT),
        ("M105", _RE_MITRE_EXACT),
        ("T1190 Exploit Public-Facing Application", _RE_MITRE_EXACT),
    ],
)
def test_exact_regex_negatives(text: str, pattern) -> None:
    """Truncated IDs and full prose sentences must NOT match Track A."""
    assert pattern.match(text) is None


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("cve2024-3094", "CVE-2024-3094"),
        ("cve 2024 3094", "CVE-2024-3094"),
        ("CVE2021 44228", "CVE-2021-44228"),
        ("cwe 89", "CWE-89"),
        ("cwe22", "CWE-22"),
        ("t 1190", "T1190"),
        ("t-1190", "T1190"),
        ("t1059 001", "T1059.001"),
        ("ta 0001", "TA0001"),
        ("m 1051", "M1051"),
        ("  \x00cve2024-3094\n  ", "CVE-2024-3094"),
        ("ＣＶＥ-２０２４-３０９４", "CVE-2024-3094"),
    ],
)
def test_normalize_query_repairs(raw: str, expected: str) -> None:
    """Broken CVE, CWE, MITRE identifiers and control chars normalize cleanly."""
    assert normalize_query(raw) == expected


@pytest.mark.parametrize(
    "prose",
    [
        "port 8080",
        "from 2024",
        "system 1024",
        "limit 1000",
        "at 2024",
        "RAM 2048",
        "t 100",
        "m 500",
        "Explain port 8080 exposure from 2024 logs.",
    ],
)
def test_normalize_query_no_false_repair_of_prose(prose: str) -> None:
    """Ordinary prose ending in t/m/ta before 3-4 digits must NOT be rewritten."""
    assert normalize_query(prose) == prose


def test_fast_path_routing_without_classifier() -> None:
    """Exact identifiers route via Track A with confidence=1.0 without Track B."""
    router = AgentIntentRouter(get_settings())
    for raw, expected_id in [
        ("cve2024-3094", "CVE-2024-3094"),
        ("CWE-89", "CWE-89"),
        ("t1059 001", "T1059.001"),
        ("TA0001", "TA0001"),
        ("m 1051", "M1051"),
    ]:
        decision = router.route(raw)
        assert isinstance(decision, RouteDecision)
        assert decision.route == ROUTE_FAST_PATH
        assert decision.confidence == pytest.approx(1.0)
        assert decision.matched_identifier == expected_id
        assert decision.normalized_query == expected_id
        assert decision.is_malformed is False
        assert decision.latency_ms >= 0.0
    # Track B model should never have been loaded for Track A queries
    assert router.is_loaded is False


def test_low_confidence_defaults_to_semantic() -> None:
    """Confidence < router_conf_threshold must safely default to SEMANTIC_RAG."""
    settings = Settings(router_conf_threshold=0.65)
    router = AgentIntentRouter(settings)

    with (
        patch.object(router, "_ensure_model_loaded", return_value=None),
        patch.object(
            router,
            "_classify",
            return_value=np.array([0.25, 0.50, 0.25], dtype=np.float32),
        ),
    ):
        decision = router.route("Ambiguous cybersecurity question")
        assert decision.route == ROUTE_SEMANTIC
        assert decision.confidence == pytest.approx(0.50)
        assert decision.is_malformed is False
        assert decision.matched_identifier is None


def test_classifier_refusal_and_malformed_mapping() -> None:
    """High-confidence HARD_REFUSAL and MALFORMED_INPUT map per contract."""
    settings = Settings(router_conf_threshold=0.65)
    router = AgentIntentRouter(settings)

    with (
        patch.object(router, "_ensure_model_loaded", return_value=None),
        patch.object(
            router,
            "_classify",
            return_value=np.array([0.05, 0.90, 0.05], dtype=np.float32),
        ),
    ):
        refusal = router.route("How do I bake sourdough bread?")
        assert refusal.route == ROUTE_REFUSAL
        assert refusal.is_malformed is False

    with (
        patch.object(router, "_ensure_model_loaded", return_value=None),
        patch.object(
            router,
            "_classify",
            return_value=np.array([0.10, 0.10, 0.80], dtype=np.float32),
        ),
    ):
        malformed = router.route("cvs 9.8 vuln xz util")
        assert malformed.route == ROUTE_SEMANTIC
        assert malformed.is_malformed is True


def test_trained_router_smoke_if_present() -> None:
    """Run real trained DeBERTa router if models/router_deberta exists."""
    settings = get_settings()
    trained_dir = settings.models_dir / "router_deberta"
    if not trained_dir.exists() or not any(trained_dir.iterdir()):
        pytest.skip("Trained router not yet present at models/router_deberta")

    router = AgentIntentRouter(settings)
    d_sem = router.route("What is the CVSS score and mitigation for CVE-2024-3094?")
    assert d_sem.route == ROUTE_SEMANTIC
    assert d_sem.matched_identifier is None

    d_oos = router.route("How do I bake a sourdough loaf?")
    assert d_oos.route == ROUTE_REFUSAL
