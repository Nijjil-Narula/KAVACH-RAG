"""Compile KAVACH-RAG Phase 3 Locust benchmark CSVs and /health snapshots into JSON and Markdown reports.

Reads Locust output artifacts from ``reports/locust/``:
  - ``<ep>_<mode>_u<N>_stats.csv``
  - ``<ep>_<mode>_u<N>_failures.csv``
  - ``<ep>_<mode>_u<N>_health.json``
  - ``<ep>_<mode>_u<N>_meta.json`` (companion metadata including loaded_idle_mib and 1 Hz queue maxima)

Aggregates per-query-class rows (``cve-id``, ``oos``, ``semantic``) as well as legacy untagged
``POST /rag/*`` rows into lists so combined request counts, RPS, and CDF-mixture percentiles cover
all traffic while keeping ``GET /health probe`` and ``SSE /rag/async first-token`` separate.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import tempfile
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOCUST_DIR = ROOT / "reports" / "locust"
DEFAULT_SUMMARY_OUT = ROOT / "reports" / "phase3_load_test_summary.json"
DEFAULT_REPORT_OUT = ROOT / "reports" / "phase3_load_test_report.md"
DEFAULT_RETRIEVAL_BASELINE = ROOT / "reports" / "phase15_latency_baseline.json"

VRAM_SOFT_TARGET_MIB = 4710.0
VRAM_HARD_LIMIT_MIB = 5478.0
LOADED_IDLE_DRIFT_LIMIT_MIB = 150.0
SEMAPHORE_LIMIT = 20
CLIENT_TIMEOUT_S = 120.0
QUERY_CLASSES: tuple[str, ...] = ("cve-id", "oos", "semantic")

_STATS_FILE_RE = re.compile(r"^(?P<endpoint>sync|async)_(?P<mode>cold|warm)_u(?P<users>\d+)_stats\.csv$")
_QUERY_CLASS_RE = re.compile(r"\[(?:cold|warm):(?P<qclass>[a-z0-9-]+)\]")
_PROFILE_ORDER = {("sync", "cold"): 0, ("async", "cold"): 1, ("async", "warm"): 2}
_QUANTILE_COLUMNS: tuple[tuple[float, str], ...] = (
    (0.0, "Min Response Time"),
    (0.50, "50%"),
    (0.66, "66%"),
    (0.75, "75%"),
    (0.80, "80%"),
    (0.90, "90%"),
    (0.95, "95%"),
    (0.98, "98%"),
    (0.99, "99%"),
    (0.999, "99.9%"),
    (0.9999, "99.99%"),
    (1.0, "Max Response Time"),
)

FAILURE_CATEGORIES: tuple[str, ...] = (
    "503_status_temporarily_degraded",
    "503_vram_hard_limit_exceeded",
    "503_other",
    "502_bad_gateway",
    "504_gateway_timeout",
    "500_internal_error",
    "429_rate_limited",
    "client_timeout_120s",
    "empty_answer",
    "sse_stream_error",
    "invalid_json",
    "connection_error",
    "other",
)


@dataclass
class StatRow:
    """Normalized representation of a single row in Locust ``*_stats.csv``."""

    row_type: str
    name: str
    request_count: int
    failure_count: int
    failure_rate_pct: float
    min_ms: float | None
    median_ms: float | None
    avg_ms: float | None
    p50_ms: float | None
    p66_ms: float | None
    p75_ms: float | None
    p80_ms: float | None
    p90_ms: float | None
    p95_ms: float | None
    p98_ms: float | None
    p99_ms: float | None
    p99_9_ms: float | None
    p99_99_ms: float | None
    max_ms: float | None
    avg_content_size: float | None
    rps: float
    failures_per_sec: float
    quantiles_ms: list[tuple[float, float]]


def _parse_int(value: str | None) -> int:
    if value is None:
        return 0
    cleaned = value.strip()
    if not cleaned or cleaned.upper() == "N/A":
        return 0
    return int(float(cleaned))


def _parse_float(value: str | None) -> float | None:
    if value is None:
        return None
    cleaned = value.strip()
    if not cleaned or cleaned.upper() == "N/A":
        return None
    parsed = float(cleaned)
    if math.isnan(parsed) or math.isinf(parsed):
        return None
    return parsed


def _parse_stat_row(raw: dict[str, str]) -> StatRow:
    req_count = _parse_int(raw.get("Request Count"))
    fail_count = _parse_int(raw.get("Failure Count"))
    fail_pct = round((fail_count / req_count) * 100.0, 2) if req_count > 0 else 0.0

    quantiles: list[tuple[float, float]] = []
    for prob, col_name in _QUANTILE_COLUMNS:
        val = _parse_float(raw.get(col_name))
        if val is not None:
            quantiles.append((prob, val))

    return StatRow(
        row_type=(raw.get("Type") or "").strip(),
        name=(raw.get("Name") or "").strip(),
        request_count=req_count,
        failure_count=fail_count,
        failure_rate_pct=fail_pct,
        min_ms=_parse_float(raw.get("Min Response Time")),
        median_ms=_parse_float(raw.get("Median Response Time")),
        avg_ms=_parse_float(raw.get("Average Response Time")),
        p50_ms=_parse_float(raw.get("50%")),
        p66_ms=_parse_float(raw.get("66%")),
        p75_ms=_parse_float(raw.get("75%")),
        p80_ms=_parse_float(raw.get("80%")),
        p90_ms=_parse_float(raw.get("90%")),
        p95_ms=_parse_float(raw.get("95%")),
        p98_ms=_parse_float(raw.get("98%")),
        p99_ms=_parse_float(raw.get("99%")),
        p99_9_ms=_parse_float(raw.get("99.9%")),
        p99_99_ms=_parse_float(raw.get("99.99%")),
        max_ms=_parse_float(raw.get("Max Response Time")),
        avg_content_size=_parse_float(raw.get("Average Content Size")),
        rps=_parse_float(raw.get("Requests/s")) or 0.0,
        failures_per_sec=_parse_float(raw.get("Failures/s")) or 0.0,
        quantiles_ms=quantiles,
    )


def _stat_row_summary(row: StatRow | None) -> dict[str, Any] | None:
    if row is None:
        return None
    data = asdict(row)
    data.pop("quantiles_ms", None)
    for key in (
        "min_ms",
        "median_ms",
        "avg_ms",
        "p50_ms",
        "p66_ms",
        "p75_ms",
        "p80_ms",
        "p90_ms",
        "p95_ms",
        "p98_ms",
        "p99_ms",
        "p99_9_ms",
        "p99_99_ms",
        "max_ms",
        "avg_content_size",
        "rps",
        "failures_per_sec",
    ):
        if data[key] is not None:
            data[key] = round(float(data[key]), 2 if key.endswith("_ms") else 4)
    return data


def _extract_query_class(row_name: str) -> str | None:
    match = _QUERY_CLASS_RE.search(row_name)
    if not match:
        return None
    return match.group("qclass")


def _cdf_at_x(quantiles: list[tuple[float, float]], x: float) -> float:
    """Evaluate piecewise-linear empirical CDF F(x) from Locust quantile anchors."""
    if not quantiles:
        return 0.0
    if x < quantiles[0][1]:
        return 0.0
    if x >= quantiles[-1][1]:
        return 1.0

    for idx in range(len(quantiles) - 1, 0, -1):
        p_hi, v_hi = quantiles[idx]
        p_lo, v_lo = quantiles[idx - 1]
        if x >= v_lo:
            if v_hi <= v_lo:
                return p_hi
            frac = (x - v_lo) / (v_hi - v_lo)
            return p_lo + frac * (p_hi - p_lo)
    return quantiles[0][0]


def _mixture_percentile(rows: list[StatRow], target_q: float) -> float | None:
    active = [r for r in rows if r.request_count > 0 and r.quantiles_ms]
    if not active:
        return None
    if len(active) == 1:
        row = active[0]
        mapping = {0.50: row.p50_ms, 0.90: row.p90_ms, 0.95: row.p95_ms, 0.99: row.p99_ms}
        if target_q in mapping and mapping[target_q] is not None:
            return round(float(mapping[target_q]), 2)

    total_n = sum(r.request_count for r in active)
    lo = min(r.quantiles_ms[0][1] for r in active)
    hi = max(r.quantiles_ms[-1][1] for r in active)
    if hi <= lo:
        return round(float(lo), 2)

    for _ in range(60):
        mid = 0.5 * (lo + hi)
        cdf = sum(r.request_count * _cdf_at_x(r.quantiles_ms, mid) for r in active) / total_n
        if cdf < target_q:
            lo = mid
        else:
            hi = mid
    return round(0.5 * (lo + hi), 2)


def _combine_rag_rows(rag_rows: list[StatRow]) -> dict[str, Any]:
    """Sum request counts, failures, and RPS across a list of StatRows cleanly."""
    active = [r for r in rag_rows if r.request_count > 0]
    total_reqs = sum(r.request_count for r in rag_rows)
    total_fails = sum(r.failure_count for r in rag_rows)
    total_rps = round(sum(r.rps for r in rag_rows), 4)
    total_fail_ps = round(sum(r.failures_per_sec for r in rag_rows), 4)
    error_rate_pct = round((total_fails / total_reqs) * 100.0, 2) if total_reqs > 0 else 0.0

    if not active:
        return {
            "row_count": len(rag_rows),
            "request_count": total_reqs,
            "failure_count": total_fails,
            "success_count": max(0, total_reqs - total_fails),
            "error_rate_pct": error_rate_pct,
            "failure_rate_pct": error_rate_pct,
            "rps": total_rps,
            "failures_per_sec": total_fail_ps,
            "min_ms": None,
            "avg_ms": None,
            "p50_ms": None,
            "p90_ms": None,
            "p95_ms": None,
            "p99_ms": None,
            "max_ms": None,
            "percentile_method": "no_requests",
        }

    weighted_avg_terms = [
        (r.avg_ms * r.request_count) for r in active if r.avg_ms is not None
    ]
    avg_denom = sum(r.request_count for r in active if r.avg_ms is not None)
    avg_ms = round(sum(weighted_avg_terms) / avg_denom, 2) if weighted_avg_terms and avg_denom > 0 else None
    mins = [r.min_ms for r in active if r.min_ms is not None]
    maxs = [r.max_ms for r in active if r.max_ms is not None]

    if len(active) == 1:
        only = active[0]
        p50 = round(only.p50_ms, 2) if only.p50_ms is not None else None
        p90 = round(only.p90_ms, 2) if only.p90_ms is not None else None
        p95 = round(only.p95_ms, 2) if only.p95_ms is not None else None
        p99 = round(only.p99_ms, 2) if only.p99_ms is not None else None
        method = "exact_single_row"
    else:
        p50 = _mixture_percentile(active, 0.50)
        p90 = _mixture_percentile(active, 0.90)
        p95 = _mixture_percentile(active, 0.95)
        p99 = _mixture_percentile(active, 0.99)
        method = "cdf_mixture_12pt"

    return {
        "row_count": len(rag_rows),
        "request_count": total_reqs,
        "failure_count": total_fails,
        "success_count": max(0, total_reqs - total_fails),
        "error_rate_pct": error_rate_pct,
        "failure_rate_pct": error_rate_pct,
        "rps": total_rps,
        "failures_per_sec": total_fail_ps,
        "min_ms": round(min(mins), 2) if mins else None,
        "avg_ms": avg_ms,
        "p50_ms": p50,
        "p90_ms": p90,
        "p95_ms": p95,
        "p99_ms": p99,
        "max_ms": round(max(maxs), 2) if maxs else None,
        "percentile_method": method,
    }


def _classify_failure(error_text: str) -> str:
    err = error_text.strip()
    upper = err.upper()
    lower = err.lower()

    if "503" in err and "STATUS_TEMPORARILY_DEGRADED" in upper:
        return "503_status_temporarily_degraded"
    if "503" in err and "VRAM_HARD_LIMIT_EXCEEDED" in upper:
        return "503_vram_hard_limit_exceeded"
    if "HTTP 503" in upper or "503 SERVICE UNAVAILABLE" in upper:
        return "503_other"
    if "HTTP 502" in upper or "502 BAD GATEWAY" in upper:
        return "502_bad_gateway"
    if "HTTP 504" in upper or "504 GATEWAY TIMEOUT" in upper:
        return "504_gateway_timeout"
    if "HTTP 500" in upper or "500 INTERNAL SERVER ERROR" in upper:
        return "500_internal_error"
    if "HTTP 429" in upper or "429 TOO MANY REQUESTS" in upper:
        return "429_rate_limited"
    if "empty answer" in lower:
        return "empty_answer"
    if "sse error event" in lower or "stream ended without done event" in lower:
        return "sse_stream_error"
    if "response is not json" in lower:
        return "invalid_json"
    if any(tok in lower for tok in ("readtimeout", "connecttimeout", "timeouterror", "timed out", "timeout")):
        return "client_timeout_120s"
    if any(
        tok in lower
        for tok in (
            "connectionerror",
            "remotedisconnected",
            "connectionrefused",
            "protocolerror",
            "chunkedencodingerror",
            "connectionreseterror",
            "connection aborted",
        )
    ):
        return "connection_error"
    return "other"


def _parse_failures_csv(path: Path) -> dict[str, Any]:
    counts: dict[str, int] = {cat: 0 for cat in FAILURE_CATEGORIES}
    health_probe_failures = 0
    raw_rag_errors: list[dict[str, Any]] = []
    raw_probe_errors: list[dict[str, Any]] = []

    if not path.exists():
        return {
            "available": False,
            "total_rag_failures": 0,
            "health_probe_failures": 0,
            "categories": counts,
            "raw_rag_errors": raw_rag_errors,
            "raw_probe_errors": raw_probe_errors,
        }

    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for raw in reader:
            method = (raw.get("Method") or "").strip()
            name = (raw.get("Name") or "").strip()
            error = (raw.get("Error") or "").strip()
            occurrences = _parse_int(raw.get("Occurrences"))
            first_seen = (raw.get("First Seen") or "").strip() or None
            last_seen = (raw.get("Last Seen") or "").strip() or None

            if name == "/health probe":
                health_probe_failures += occurrences
                raw_probe_errors.append(
                    {
                        "method": method,
                        "name": name,
                        "error": error,
                        "occurrences": occurrences,
                        "first_seen": first_seen,
                        "last_seen": last_seen,
                    }
                )
            else:
                category = _classify_failure(error)
                counts[category] += occurrences
                raw_rag_errors.append(
                    {
                        "method": method,
                        "name": name,
                        "category": category,
                        "error": error,
                        "occurrences": occurrences,
                        "first_seen": first_seen,
                        "last_seen": last_seen,
                    }
                )

    return {
        "available": True,
        "total_rag_failures": sum(counts.values()),
        "health_probe_failures": health_probe_failures,
        "categories": counts,
        "raw_rag_errors": raw_rag_errors,
        "raw_probe_errors": raw_probe_errors,
    }


def _lookup_path(payload: dict[str, Any], dotted_key: str) -> Any:
    cur: Any = payload
    for part in dotted_key.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return None
    return cur


def _first_present(payload: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        val = _lookup_path(payload, key)
        if val is not None:
            return val
    return None


def _as_optional_float(val: Any) -> float | None:
    if val is None or isinstance(val, bool):
        return None
    try:
        parsed = float(val)
    except (TypeError, ValueError):
        return None
    if math.isnan(parsed) or math.isinf(parsed):
        return None
    return parsed


def _as_optional_int(val: Any) -> int | None:
    num = _as_optional_float(val)
    if num is None:
        return None
    return int(round(num))


def _as_optional_bool(val: Any) -> bool | None:
    if isinstance(val, bool):
        return val
    if isinstance(val, str):
        low = val.strip().lower()
        if low in {"true", "1", "yes"}:
            return True
        if low in {"false", "0", "no"}:
            return False
    return None


def _parse_health_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"available": False, "parse_error": "file_missing", "raw": None}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {"available": False, "parse_error": f"{type(exc).__name__}: {exc}", "raw": None}

    if not isinstance(raw, dict):
        return {"available": False, "parse_error": "not_a_json_object", "raw": None}

    current_vram = _as_optional_float(
        _first_present(
            raw,
            (
                "vram_mib",
                "vram_mb",
                "vram_used_mib",
                "vram_used_mb",
                "current_vram_mib",
                "current_vram_mb",
                "vram.used_mib",
                "vram.used_mb",
                "gpu.vram_mib",
                "gpu.vram_mb",
            ),
        )
    )
    peak_vram = _as_optional_float(
        _first_present(
            raw,
            (
                "peak_vram_mib",
                "peak_vram_mb",
                "vram_peak_mib",
                "vram_peak_mb",
                "gpu_peak_vram_mib",
                "gpu_peak_vram_mb",
                "vram.peak_mib",
                "vram.peak_mb",
                "gpu.peak_mib",
                "gpu.peak_mb",
                "vram_guard.peak_vram_mib",
                "vram_guard.peak_vram_mb",
            ),
        )
    )
    if peak_vram is None:
        peak_vram = current_vram

    vram_breached = _as_optional_bool(
        _first_present(raw, ("vram_breached", "vram.breached", "gpu.vram_breached", "vram_guard.breached"))
    )

    cache_hits = _as_optional_int(_first_present(raw, ("cache_hits", "cache.hits", "query_cache.hits")))
    cache_misses = _as_optional_int(_first_present(raw, ("cache_misses", "cache.misses", "query_cache.misses")))
    cache_hit_rate = _as_optional_float(
        _first_present(raw, ("cache_hit_rate", "cache.hit_rate", "query_cache.hit_rate"))
    )
    if cache_hit_rate is None and cache_hits is not None and cache_misses is not None:
        denom = cache_hits + cache_misses
        cache_hit_rate = round(cache_hits / denom, 4) if denom > 0 else 0.0

    circuit_state = _first_present(raw, ("circuit_state", "circuit.state", "circuit_breaker.state"))
    circuit_open = _as_optional_bool(
        _first_present(
            raw,
            ("circuit_open", "breaker_open", "circuit_breaker.open", "circuit_breaker.is_open", "circuit_breaker.circuit_open"),
        )
    )
    if circuit_open is None and isinstance(circuit_state, str):
        circuit_open = circuit_state.strip().lower() == "open"

    return {
        "available": True,
        "parse_error": None,
        "status": _first_present(raw, ("status", "health_status", "state")),
        "vram_breached": vram_breached,
        "peak_vram_mib": peak_vram,
        "current_vram_mib": current_vram,
        "vram_soft_limit_mib": _as_optional_float(
            _first_present(
                raw,
                ("vram_soft_limit_mib", "vram_soft_limit_mb", "vram_target_mib", "vram_target_mb", "vram.soft_limit_mib"),
            )
        ),
        "vram_hard_limit_mib": _as_optional_float(
            _first_present(
                raw,
                ("vram_hard_limit_mib", "vram_hard_limit_mb", "vram_limit_mib", "vram_limit_mb", "vram.hard_limit_mib"),
            )
        ),
        "gpu_waiting": _as_optional_int(
            _first_present(
                raw,
                (
                    "gpu_waiting",
                    "gpu_semaphore_waiting",
                    "semaphore.gpu_waiting",
                    "semaphore.waiting",
                    "concurrency.gpu_waiting",
                    "concurrency.waiting",
                ),
            )
        ),
        "max_gpu_waiting": _as_optional_int(
            _first_present(
                raw,
                (
                    "max_gpu_waiting",
                    "gpu_waiting_peak",
                    "peak_gpu_waiting",
                    "semaphore.max_waiting",
                    "concurrency.max_gpu_waiting",
                ),
            )
        ),
        "gpu_active": _as_optional_int(
            _first_present(raw, ("gpu_inflight", "gpu_active", "gpu_in_flight", "semaphore.active", "concurrency.gpu_active"))
        ),
        "gpu_semaphore_limit": _as_optional_int(
            _first_present(
                raw,
                ("gpu_semaphore_limit", "semaphore_limit", "semaphore.limit", "concurrency.gpu_semaphore_limit"),
            )
        ),
        "retrieval_waiting": _as_optional_int(
            _first_present(raw, ("retrieval_waiting", "retrieval_lock_waiting", "concurrency.retrieval_waiting"))
        ),
        "cache_hits": cache_hits,
        "cache_misses": cache_misses,
        "cache_coalesced": _as_optional_int(
            _first_present(
                raw,
                (
                    "cache_coalesced",
                    "cache_stampedes_prevented",
                    "inflight_coalesced",
                    "cache.coalesced",
                    "cache.inflight_coalesced",
                    "query_cache.coalesced",
                ),
            )
        ),
        "cache_lock_timeouts": _as_optional_int(_first_present(raw, ("cache.lock_timeouts", "cache_lock_timeouts"))),
        "cache_size": _as_optional_int(
            _first_present(raw, ("cache_size", "cache_entries", "cache.size", "cache.entries", "query_cache.size"))
        ),
        "cache_hit_rate": cache_hit_rate,
        "circuit_open": circuit_open,
        "circuit_state": circuit_state,
        "circuit_trips": _as_optional_int(
            _first_present(raw, ("circuit_trips", "circuit.trips", "circuit_breaker.trips", "circuit_breaker.open_count"))
        ),
        "generator_calls": _as_optional_int(_first_present(raw, ("generator_calls",))),
        "rss_mb": _as_optional_float(
            _first_present(raw, ("rss_mb", "process_rss_mb", "memory_rss_mb", "process.rss_mb"))
        ),
        "raw": raw,
    }


def _parse_meta_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"available": False}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {"available": False, "parse_error": f"{type(exc).__name__}: {exc}"}
    if not isinstance(raw, dict):
        return {"available": False, "parse_error": "not_a_json_object"}
    return {"available": True, **raw}


def _parse_run(stats_path: Path) -> dict[str, Any]:
    match = _STATS_FILE_RE.match(stats_path.name)
    if match:
        endpoint = match.group("endpoint")
        mode = match.group("mode")
        users = int(match.group("users"))
        slug = f"{endpoint}_{mode}_u{users}"
    else:
        slug = stats_path.name.removesuffix("_stats.csv")
        parts = slug.split("_")
        endpoint = parts[0] if parts else "unknown"
        mode = parts[1] if len(parts) > 1 else "unknown"
        users = int(parts[2].lstrip("u")) if len(parts) > 2 and parts[2].startswith("u") and parts[2][1:].isdigit() else 0

    label = "sync" if endpoint == "sync" else f"async-{mode}"

    rag_non_stream_rows: list[StatRow] = []
    rag_stream_rows: list[StatRow] = []
    sse_ttft_rows: list[StatRow] = []
    health_probe_rows: list[StatRow] = []
    locust_aggregated: StatRow | None = None
    other_rows: list[StatRow] = []

    class_non_stream_rows: dict[str, list[StatRow]] = {}
    class_stream_rows: dict[str, list[StatRow]] = {}
    class_rag_rows: dict[str, list[StatRow]] = {}
    class_ttft_rows: dict[str, list[StatRow]] = {}

    with stats_path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for raw in reader:
            row = _parse_stat_row(raw)
            q_class = _extract_query_class(row.name)
            if row.row_type == "" and row.name == "Aggregated":
                locust_aggregated = row
            elif row.row_type == "GET" and row.name == "/health probe":
                health_probe_rows.append(row)
            elif row.row_type == "SSE" or "first-token" in row.name:
                sse_ttft_rows.append(row)
                if q_class:
                    class_ttft_rows.setdefault(q_class, []).append(row)
            elif row.row_type == "POST" and row.name.startswith("/rag/"):
                if "stream" in row.name:
                    rag_stream_rows.append(row)
                    if q_class:
                        class_stream_rows.setdefault(q_class, []).append(row)
                else:
                    rag_non_stream_rows.append(row)
                    if q_class:
                        class_non_stream_rows.setdefault(q_class, []).append(row)
                if q_class:
                    class_rag_rows.setdefault(q_class, []).append(row)
            else:
                other_rows.append(row)

    all_rag_rows = [*rag_non_stream_rows, *rag_stream_rows]
    rag_only = _combine_rag_rows(all_rag_rows)
    rag_non_stream = _combine_rag_rows(rag_non_stream_rows) if rag_non_stream_rows else None
    rag_stream = _combine_rag_rows(rag_stream_rows) if rag_stream_rows else None
    sse_ttft = _combine_rag_rows(sse_ttft_rows) if sse_ttft_rows else None
    health_probe = _combine_rag_rows(health_probe_rows) if health_probe_rows else None

    observed_classes = sorted(
        set(class_rag_rows) | set(class_ttft_rows),
        key=lambda c: (QUERY_CLASSES.index(c) if c in QUERY_CLASSES else 99, c),
    )
    by_query_class: dict[str, Any] = {}
    for q_class in QUERY_CLASSES:
        c_rows = class_rag_rows.get(q_class, [])
        c_ns = class_non_stream_rows.get(q_class, [])
        c_st = class_stream_rows.get(q_class, [])
        c_ttft = class_ttft_rows.get(q_class, [])
        by_query_class[q_class] = {
            "combined": _combine_rag_rows(c_rows),
            "non_stream": _combine_rag_rows(c_ns) if c_ns else None,
            "stream": _combine_rag_rows(c_st) if c_st else None,
            "sse_ttft": _combine_rag_rows(c_ttft) if c_ttft else None,
        }
    for q_class in observed_classes:
        if q_class not in by_query_class:
            c_rows = class_rag_rows.get(q_class, [])
            c_ns = class_non_stream_rows.get(q_class, [])
            c_st = class_stream_rows.get(q_class, [])
            c_ttft = class_ttft_rows.get(q_class, [])
            by_query_class[q_class] = {
                "combined": _combine_rag_rows(c_rows),
                "non_stream": _combine_rag_rows(c_ns) if c_ns else None,
                "stream": _combine_rag_rows(c_st) if c_st else None,
                "sse_ttft": _combine_rag_rows(c_ttft) if c_ttft else None,
            }

    completed_reqs = rag_only["request_count"]
    completed_to_users = round(completed_reqs / users, 4) if users > 0 else None
    most_users_uncompleted = bool(users > 0 and completed_reqs < users)

    failures = _parse_failures_csv(stats_path.with_name(f"{slug}_failures.csv"))
    health = _parse_health_json(stats_path.with_name(f"{slug}_health.json"))
    meta = _parse_meta_json(stats_path.with_name(f"{slug}_meta.json"))

    max_gpu_waiting_1hz = _as_optional_int(meta.get("max_gpu_waiting"))
    max_gpu_active_1hz = _as_optional_int(meta.get("max_gpu_active"))
    max_retrieval_waiting_1hz = _as_optional_int(meta.get("max_retrieval_waiting"))
    queue_sampling = {
        "bound_note": str(meta.get("queue_maxima_bound") or "1 Hz lower bound sampled via GET /health"),
        "sample_rate_hz": _as_optional_float(meta.get("health_sample_rate_hz")) or 1.0,
        "sample_count": _as_optional_int(meta.get("health_sample_count")),
        "max_gpu_waiting_1hz_lower_bound": max_gpu_waiting_1hz,
        "max_gpu_active_1hz_lower_bound": max_gpu_active_1hz,
        "max_retrieval_waiting_1hz_lower_bound": max_retrieval_waiting_1hz,
    }

    loaded_idle_mib = _as_optional_float(meta.get("loaded_idle_mib"))

    return {
        "slug": slug,
        "label": label,
        "endpoint": endpoint,
        "mode": mode,
        "users": users,
        "files": {
            "stats_csv": stats_path.name,
            "failures_csv": f"{slug}_failures.csv" if failures["available"] else None,
            "health_json": f"{slug}_health.json" if health["available"] else None,
            "meta_json": f"{slug}_meta.json" if meta.get("available") else None,
        },
        "completion": {
            "completed_requests": completed_reqs,
            "users": users,
            "completed_to_users_ratio": completed_to_users,
            "min_uncompleted_users": max(0, users - completed_reqs) if users > 0 else 0,
            "most_users_uncompleted": most_users_uncompleted,
        },
        "has_query_class_tags": bool(observed_classes),
        "rag_only": rag_only,
        "rag_non_stream": rag_non_stream,
        "rag_stream": rag_stream,
        "sse_ttft": sse_ttft,
        "by_query_class": by_query_class,
        "health_probe": health_probe,
        "locust_aggregated": _stat_row_summary(locust_aggregated),
        "other_rows": [_stat_row_summary(r) for r in other_rows],
        "queue_sampling": queue_sampling,
        "vram_drift": {
            "loaded_idle_mib": loaded_idle_mib,
            "delta_vs_first_mib": None,
            "exceeds_150_mib": None,
        },
        "failures": failures,
        "health": health,
        "meta": meta,
    }


def _annotate_loaded_idle_drift(runs: list[dict[str, Any]]) -> dict[str, Any]:
    """Compare each run's post-restart loaded_idle_mib against the chronologically first run."""
    with_idle: list[tuple[int, dict[str, Any]]] = [
        (idx, r)
        for idx, r in enumerate(runs)
        if r["vram_drift"]["loaded_idle_mib"] is not None
    ]
    if not with_idle:
        return {
            "available": False,
            "baseline_slug": None,
            "baseline_loaded_idle_mib": None,
            "threshold_delta_mib": LOADED_IDLE_DRIFT_LIMIT_MIB,
            "any_flagged": False,
            "flagged_runs": [],
        }

    def _chrono_key(item: tuple[int, dict[str, Any]]) -> tuple[int, str, int]:
        idx, run = item
        started = run["meta"].get("started_at_utc")
        if isinstance(started, str) and started.strip():
            return (0, started.strip(), idx)
        return (1, "", idx)

    _, baseline_run = min(with_idle, key=_chrono_key)
    baseline_mib = float(baseline_run["vram_drift"]["loaded_idle_mib"])
    flagged: list[dict[str, Any]] = []

    for r in runs:
        idle = r["vram_drift"]["loaded_idle_mib"]
        if idle is None:
            continue
        delta = round(float(idle) - baseline_mib, 2)
        exceeds = delta > LOADED_IDLE_DRIFT_LIMIT_MIB
        r["vram_drift"]["delta_vs_first_mib"] = delta
        r["vram_drift"]["exceeds_150_mib"] = exceeds
        if exceeds:
            flagged.append(
                {
                    "slug": r["slug"],
                    "loaded_idle_mib": float(idle),
                    "delta_vs_first_mib": delta,
                }
            )

    return {
        "available": True,
        "baseline_slug": baseline_run["slug"],
        "baseline_loaded_idle_mib": round(baseline_mib, 2),
        "threshold_delta_mib": LOADED_IDLE_DRIFT_LIMIT_MIB,
        "any_flagged": bool(flagged),
        "flagged_runs": flagged,
    }


def _load_retrieval_baseline(baseline_path: Path | None) -> dict[str, Any] | None:
    if baseline_path is None or not baseline_path.exists():
        return None
    try:
        data = json.loads(baseline_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    overall = data.get("overall_total_latency_ms")
    if not isinstance(overall, dict):
        return None
    median_ms = _as_optional_float(overall.get("median") or overall.get("p50"))
    mean_ms = _as_optional_float(overall.get("mean"))
    if median_ms is None:
        return None
    return {
        "source": str(baseline_path),
        "retrieval_median_ms": round(median_ms, 2),
        "retrieval_mean_ms": round(mean_ms, 2) if mean_ms is not None else None,
        "serialized_retrieval_cap_rps": round(1000.0 / median_ms, 4) if median_ms > 0 else None,
    }


def _load_baseline_gpu(locust_dir: Path) -> dict[str, Any] | None:
    path = locust_dir / "baseline_gpu.json"
    if not path.exists():
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return raw if isinstance(raw, dict) else None


def _evaluate_megaprompt_gates(
    runs: list[dict[str, Any]],
    retrieval_baseline: dict[str, Any] | None,
    loaded_idle_drift: dict[str, Any],
) -> dict[str, Any]:
    """Evaluate strictly the 5 Phase 3 megaprompt gates: E6 and P1-P4."""
    sync_runs = [r for r in runs if r["label"] == "sync"]
    cold_runs = [r for r in runs if r["label"] == "async-cold"]
    warm_runs = [r for r in runs if r["label"] == "async-warm"]

    # ── E6: GPU VRAM Hard Cap & Guardrail Flag ────────────────────────────────
    peak_vrams = [
        (r["slug"], float(r["health"]["peak_vram_mib"]))
        for r in runs
        if r["health"].get("available") and r["health"].get("peak_vram_mib") is not None
    ]
    vram_breach_flags = [
        (r["slug"], bool(r["health"]["vram_breached"]))
        for r in runs
        if r["health"].get("available") and r["health"].get("vram_breached") is not None
    ]
    max_peak_vram = max((v for _, v in peak_vrams), default=None)
    any_vram_breached = any(flag for _, flag in vram_breach_flags)
    breached_slugs = [slug for slug, flag in vram_breach_flags if flag]
    over_cap_slugs = [slug for slug, v in peak_vrams if v > VRAM_HARD_LIMIT_MIB]

    if not peak_vrams and not vram_breach_flags:
        e6_status = "INSUFFICIENT_DATA"
    elif any_vram_breached or (max_peak_vram is not None and max_peak_vram > VRAM_HARD_LIMIT_MIB):
        e6_status = "FAIL"
    else:
        e6_status = "PASS"

    e6 = {
        "id": "E6",
        "name": "GPU VRAM Hard Cap (peak <= 5,478 MiB, vram_breached == false)",
        "status": e6_status,
        "observed": {
            "max_peak_vram_mib": round(max_peak_vram, 2) if max_peak_vram is not None else None,
            "hard_limit_mib": VRAM_HARD_LIMIT_MIB,
            "soft_target_mib": VRAM_SOFT_TARGET_MIB,
            "any_vram_breached": any_vram_breached,
            "breached_runs": breached_slugs,
            "over_hard_limit_runs": over_cap_slugs,
            "snapshot_count": max(len(peak_vrams), len(vram_breach_flags)),
        },
        "target": "peak GPU VRAM <= 5,478 MiB and vram_breached == false across every snapshot",
    }

    # ── P1: async-warm u100 Failure Rate <= 1% ────────────────────────────────
    warm_u100 = next((r for r in warm_runs if r["users"] == 100), None)
    if warm_u100 is None or warm_u100["rag_only"]["request_count"] == 0:
        p1_status = "INSUFFICIENT_DATA"
        p1_obs: dict[str, Any] = {"run_found": warm_u100 is not None, "users": 100, "request_count": 0}
    else:
        err_pct = float(warm_u100["rag_only"]["error_rate_pct"])
        p1_status = "PASS" if err_pct <= 1.0 else "FAIL"
        p1_obs = {
            "slug": warm_u100["slug"],
            "users": 100,
            "request_count": warm_u100["rag_only"]["request_count"],
            "failure_count": warm_u100["rag_only"]["failure_count"],
            "error_rate_pct": err_pct,
        }

    p1 = {
        "id": "P1",
        "name": "async-warm u100 Failure Rate <= 1%",
        "status": p1_status,
        "observed": p1_obs,
        "target": "async-warm at 100 users: failure rate <= 1.0%",
    }

    # ── P2: async-warm u100 P95 Latency <= 250 ms ─────────────────────────────
    if warm_u100 is None or warm_u100["rag_only"]["p95_ms"] is None:
        p2_status = "INSUFFICIENT_DATA"
        p2_obs: dict[str, Any] = {"run_found": warm_u100 is not None, "users": 100, "p95_ms": None}
    else:
        p95 = float(warm_u100["rag_only"]["p95_ms"])
        p2_status = "PASS" if p95 <= 250.0 else "FAIL"
        p2_obs = {
            "slug": warm_u100["slug"],
            "users": 100,
            "request_count": warm_u100["rag_only"]["request_count"],
            "p50_ms": warm_u100["rag_only"]["p50_ms"],
            "p95_ms": p95,
            "max_ms": warm_u100["rag_only"]["max_ms"],
            "rps": warm_u100["rag_only"]["rps"],
            "non_stream_p95_ms": warm_u100["rag_non_stream"]["p95_ms"] if warm_u100.get("rag_non_stream") else None,
            "stream_p95_ms": warm_u100["rag_stream"]["p95_ms"] if warm_u100.get("rag_stream") else None,
            "sse_ttft_p95_ms": warm_u100["sse_ttft"]["p95_ms"] if warm_u100.get("sse_ttft") else None,
        }

    p2 = {
        "id": "P2",
        "name": "async-warm u100 P95 Latency <= 250 ms",
        "status": p2_status,
        "observed": p2_obs,
        "target": "async-warm at 100 users: end-to-end P95 latency <= 250 ms",
    }

    # ── P3: async-cold Failure Rate <= sync Failure Rate at Every User Count ─
    sync_by_u = {r["users"]: r for r in sync_runs}
    cold_by_u = {r["users"]: r for r in cold_runs}
    all_users = sorted(set(sync_by_u) | set(cold_by_u))
    per_tier: dict[str, Any] = {}
    all_tiers_present = True
    all_tiers_ok = True
    mostly_uncompleted_tiers: list[str] = []

    for u in all_users:
        sr = sync_by_u.get(u)
        cr = cold_by_u.get(u)
        if sr is None or cr is None or sr["rag_only"]["request_count"] == 0 or cr["rag_only"]["request_count"] == 0:
            all_tiers_present = False
            per_tier[str(u)] = {
                "users": u,
                "sync_error_rate_pct": sr["rag_only"]["error_rate_pct"] if sr else None,
                "async_cold_error_rate_pct": cr["rag_only"]["error_rate_pct"] if cr else None,
                "sync_completed": sr["rag_only"]["request_count"] if sr else 0,
                "async_cold_completed": cr["rag_only"]["request_count"] if cr else 0,
                "cold_not_worse_than_sync": None,
            }
            continue
        s_err = float(sr["rag_only"]["error_rate_pct"])
        c_err = float(cr["rag_only"]["error_rate_pct"])
        tier_ok = c_err <= s_err
        if not tier_ok:
            all_tiers_ok = False
        if sr["completion"]["most_users_uncompleted"] or cr["completion"]["most_users_uncompleted"]:
            mostly_uncompleted_tiers.append(f"u{u}")
        per_tier[str(u)] = {
            "users": u,
            "sync_error_rate_pct": s_err,
            "async_cold_error_rate_pct": c_err,
            "sync_completed": sr["rag_only"]["request_count"],
            "async_cold_completed": cr["rag_only"]["request_count"],
            "cold_not_worse_than_sync": tier_ok,
        }

    if not all_users or not all_tiers_present:
        p3_status = "INSUFFICIENT_DATA"
    elif all_tiers_ok:
        p3_status = "PASS"
    else:
        p3_status = "FAIL"

    p3 = {
        "id": "P3",
        "name": "async-cold Failure Rate <= sync Failure Rate at Every User Count",
        "status": p3_status,
        "observed": {
            "user_counts": all_users,
            "per_user_count": per_tier,
            "mostly_uncompleted_tiers": mostly_uncompleted_tiers,
            "reliability_caveat": (
                "Failure rates on saturated runs where completed_requests < users reflect only the "
                "subset of requests that drained before --stop-timeout and are NOT reliability evidence."
            ),
        },
        "target": "At every tested user count, async-cold failure rate <= sync failure rate",
    }

    # ── P4: /health Probe P95 < 50 ms During async-cold u100 ──────────────────
    cold_u100 = cold_by_u.get(100)
    if cold_u100 is None or not cold_u100.get("health_probe") or cold_u100["health_probe"].get("p95_ms") is None:
        p4_status = "INSUFFICIENT_DATA"
        p4_obs: dict[str, Any] = {"run_found": cold_u100 is not None, "users": 100, "probe_p95_ms": None}
    else:
        hp = cold_u100["health_probe"]
        probe_p95 = float(hp["p95_ms"])
        probe_fails = int(hp["failure_count"])
        p4_status = "PASS" if (probe_p95 < 50.0 and probe_fails == 0) else "FAIL"
        p4_obs = {
            "slug": cold_u100["slug"],
            "users": 100,
            "probe_count": hp["request_count"],
            "probe_failures": probe_fails,
            "probe_p50_ms": hp["p50_ms"],
            "probe_p95_ms": probe_p95,
            "probe_max_ms": hp["max_ms"],
            "max_retrieval_waiting_1hz_lower_bound": cold_u100["queue_sampling"]["max_retrieval_waiting_1hz_lower_bound"],
            "max_gpu_waiting_1hz_lower_bound": cold_u100["queue_sampling"]["max_gpu_waiting_1hz_lower_bound"],
        }

    p4 = {
        "id": "P4",
        "name": "/health Probe P95 < 50 ms During async-cold u100",
        "status": p4_status,
        "observed": p4_obs,
        "target": "GET /health probe P95 < 50.0 ms with 0 probe failures during async-cold u100",
    }

    # ── Aggregates & Cold Head-of-Line Summary ────────────────────────────────
    total_failure_categories: dict[str, int] = {cat: 0 for cat in FAILURE_CATEGORIES}
    for r in runs:
        for cat, cnt in r["failures"]["categories"].items():
            total_failure_categories[cat] += cnt

    cold_u1 = min(cold_runs, key=lambda r: r["users"]) if cold_runs else None
    cold_u1_avg_ms = cold_u1["rag_only"]["avg_ms"] if cold_u1 else None
    cold_serial_e2e_cap_rps = (
        round(1000.0 / cold_u1_avg_ms, 4) if cold_u1_avg_ms and cold_u1_avg_ms > 0 else None
    )
    peak_cold_rps = max((r["rag_only"]["rps"] for r in cold_runs), default=None)
    peak_warm_rps = max((r["rag_only"]["rps"] for r in warm_runs), default=None)
    peak_sync_rps = max((r["rag_only"]["rps"] for r in sync_runs), default=None)

    cold_hol_rows: list[dict[str, Any]] = []
    for r in [*sync_runs, *cold_runs]:
        per_cls = {
            cls: {
                "request_count": r["by_query_class"][cls]["combined"]["request_count"],
                "failure_count": r["by_query_class"][cls]["combined"]["failure_count"],
                "p50_ms": r["by_query_class"][cls]["combined"]["p50_ms"],
                "p95_ms": r["by_query_class"][cls]["combined"]["p95_ms"],
                "max_ms": r["by_query_class"][cls]["combined"]["max_ms"],
            }
            for cls in QUERY_CLASSES
            if cls in r["by_query_class"]
        }
        cold_hol_rows.append(
            {
                "slug": r["slug"],
                "label": r["label"],
                "users": r["users"],
                "has_query_class_tags": r["has_query_class_tags"],
                "by_query_class": per_cls,
                "max_retrieval_waiting_1hz_lower_bound": r["queue_sampling"]["max_retrieval_waiting_1hz_lower_bound"],
                "max_gpu_waiting_1hz_lower_bound": r["queue_sampling"]["max_gpu_waiting_1hz_lower_bound"],
            }
        )

    mostly_uncompleted_runs = [
        {
            "slug": r["slug"],
            "completed_requests": r["completion"]["completed_requests"],
            "users": r["completion"]["users"],
            "completed_to_users_ratio": r["completion"]["completed_to_users_ratio"],
        }
        for r in runs
        if r["completion"]["most_users_uncompleted"]
    ]

    return {
        "engineering_gates": {"E6": e6},
        "performance_targets": {"P1": p1, "P2": p2, "P3": p3, "P4": p4},
        "aggregates": {
            "max_peak_vram_mib": round(max_peak_vram, 2) if max_peak_vram is not None else None,
            "any_vram_breached": any_vram_breached,
            "loaded_idle_drift": loaded_idle_drift,
            "cold_serial_e2e_cap_rps": cold_serial_e2e_cap_rps,
            "peak_measured_async_cold_rps": peak_cold_rps,
            "peak_measured_async_warm_rps": peak_warm_rps,
            "peak_measured_sync_rps": peak_sync_rps,
            "retrieval_baseline": retrieval_baseline,
            "cold_per_class_head_of_line": cold_hol_rows,
            "mostly_uncompleted_runs": mostly_uncompleted_runs,
            "reliability_note": (
                "Failure rates on runs where completed_requests < users (most spawned virtual users never "
                "completed a request before --stop-timeout) are not reliability evidence."
            ),
            "queue_sampling_note": "Queue maxima (gpu_waiting, gpu_active, retrieval_waiting) are 1 Hz lower bounds sampled via GET /health.",
            "total_failure_categories": total_failure_categories,
        },
    }


def _fmt_num(val: float | int | None, digits: int = 2, suffix: str = "") -> str:
    if val is None:
        return "N/A"
    if isinstance(val, int):
        return f"{val}{suffix}"
    return f"{val:.{digits}f}{suffix}"


def _fmt_class_cell(cls_entry: dict[str, Any] | None, has_tags: bool) -> str:
    if not has_tags or cls_entry is None:
        return "N/A"
    c = cls_entry["combined"]
    if c["request_count"] == 0:
        return "0 / N/A / N/A / N/A"
    return f"{c['request_count']} / {_fmt_num(c['p50_ms'], 1)} / {_fmt_num(c['p95_ms'], 1)} / {_fmt_num(c['max_ms'], 1)}"


def _build_markdown_report(summary: dict[str, Any]) -> str:
    runs: list[dict[str, Any]] = summary["runs"]
    eval_data = summary["evaluation"]
    perf = eval_data["performance_targets"]
    gates = eval_data["engineering_gates"]
    aggs = eval_data["aggregates"]
    drift = aggs["loaded_idle_drift"]
    baseline_gpu = summary.get("baseline_gpu")

    lines: list[str] = [
        "# KAVACH-RAG Phase 3 Load Test & Concurrency Benchmark Report",
        "",
        f"- **Generated at (UTC):** {summary['generated_at_utc']}",
        f"- **Source Locust directory:** `{summary['locust_dir']}`",
        f"- **Total matrix runs compiled:** {len(runs)}",
        f"- **Client request timeout:** {CLIENT_TIMEOUT_S:.0f} s (`benchmarks/locustfile.py`)",
        f"- **GPU concurrency semaphore:** `Semaphore({SEMAPHORE_LIMIT})`",
        f"- **VRAM limits:** Hard limit `<= {VRAM_HARD_LIMIT_MIB:.0f} MiB` (soft reference `<= {VRAM_SOFT_TARGET_MIB:.0f} MiB`)",
    ]
    if baseline_gpu is not None:
        lines.append(
            f"- **Pre-run host GPU baseline (`baseline_gpu.json`):** `{baseline_gpu.get('memory_used_mib', 'N/A')} MiB` "
            f"(threshold `<= {baseline_gpu.get('threshold_mib', 600)} MiB`, passed=`{baseline_gpu.get('passed')}`)"
        )

    lines.extend(
        [
            "",
            "## 1. Executive Summary & Test Matrix",
            "",
            "Each matrix cell started a clean FastAPI server instance (`start_server.ps1 -Restart`), waited for GPU utilization `<= 10%` across 3 consecutive 1s samples (60s cap), recorded post-restart `loaded_idle_mib`, ran headless Locust over an 8-query pool (`2` `cve-id`, `4` `semantic` from `inscope_dev_100.jsonl`, `2` `oos` from `oos_dev_60.jsonl`; dev sets disjoint from `golden_150.jsonl`), sampled `/health` at `1 Hz`, and captured a post-run `/health` JSON snapshot.",
            "",
            "| Run Slug | Profile | Users | Completed / Users (Ratio) | Duration (s) | Locust Exit | RAG RPS | RAG Error % |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )

    for r in runs:
        meta = r["meta"]
        rag = r["rag_only"]
        comp = r["completion"]
        ratio_str = _fmt_num(comp["completed_to_users_ratio"], 2) if comp["completed_to_users_ratio"] is not None else "N/A"
        lines.append(
            "| `{slug}` | `{label}` | {users} | {comp_n} / {users} ({ratio}) | {dur} | {exit_code} | {rps} | {err}% |".format(
                slug=r["slug"],
                label=r["label"],
                users=r["users"],
                comp_n=comp["completed_requests"],
                ratio=ratio_str,
                dur=_fmt_num(meta.get("duration_s"), 1),
                exit_code=meta.get("locust_exit_code", "N/A"),
                rps=_fmt_num(rag["rps"], 2),
                err=_fmt_num(rag["error_rate_pct"], 2),
            )
        )

    lines.extend(
        [
            "",
            "### Scorecard Summary (Megaprompt Gates E6 & P1–P4)",
            "",
            "| Gate / Target | Description | Status |",
            "|---|---|---|",
        ]
    )
    for item in [gates["E6"], perf["P1"], perf["P2"], perf["P3"], perf["P4"]]:
        lines.append(f"| **{item['id']}** | {item['name']} | **{item['status']}** |")

    # ── Section 2 ─────────────────────────────────────────────────────────────
    lines.extend(
        [
            "",
            "## 2. Throughput, Completion Coverage & Scaling (RAG-Only vs Aggregated)",
            "",
            "> **Methodology note:** `benchmarks/locustfile.py` fires auxiliary `GET /health probe` (1 req/s) and synthetic `SSE /rag/async first-token[mode:class]` events alongside tagged `POST /rag/*[mode:class]` HTTP calls. **RAG-Only** sums request counts and RPS across all query classes (`cve-id`, `oos`, `semantic`, or legacy untagged rows) and computes percentiles across combined classes.",
            ">",
            "> **Reliability caveat:** Locust counts only requests that completed before `--run-time` + `--stop-timeout`. **Failure rates on runs where `completed_requests < users` (most spawned virtual users never completed a request before shutdown) are NOT reliability evidence** — they reflect only the early or fast-path/`oos` requests that drained before teardown.",
            "",
            "### 2a. Combined RAG-Only Scaling & Completion Table",
            "",
            "| Profile | Users | Completed / Users (Ratio) | Most Users Uncompleted | RAG Fails | RAG Error % | RAG RPS | P50 (ms) | P95 (ms) | Max (ms) | Aggregated RPS |",
            "|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )

    for r in runs:
        rag = r["rag_only"]
        comp = r["completion"]
        agg = r.get("locust_aggregated")
        ratio_str = _fmt_num(comp["completed_to_users_ratio"], 2) if comp["completed_to_users_ratio"] is not None else "N/A"
        lines.append(
            "| `{label}` | {users} | {comp_n} / {users} ({ratio}) | {uncomp} | {fails} | {err}% | {rps} | {p50} | {p95} | {mx} | {agg_rps} |".format(
                label=r["label"],
                users=r["users"],
                comp_n=comp["completed_requests"],
                ratio=ratio_str,
                uncomp=comp["most_users_uncompleted"],
                fails=rag["failure_count"],
                err=_fmt_num(rag["error_rate_pct"], 2),
                rps=_fmt_num(rag["rps"], 2),
                p50=_fmt_num(rag["p50_ms"], 1),
                p95=_fmt_num(rag["p95_ms"], 1),
                mx=_fmt_num(rag["max_ms"], 1),
                agg_rps=_fmt_num(agg["rps"] if agg else None, 2),
            )
        )

    lines.extend(
        [
            "",
            "### 2b. Per-Route Breakdown (Non-Stream, SSE Stream & SSE First-Token TTFT)",
            "",
            "| Run Slug | Non-Stream Reqs / RPS / P50 / P95 (ms) | Stream Reqs / RPS / P50 / P95 (ms) | SSE TTFT Count / P50 / P95 / P99 (ms) |",
            "|---|---|---|---|",
        ]
    )

    for r in runs:
        ns = r.get("rag_non_stream")
        st = r.get("rag_stream")
        tt = r.get("sse_ttft")
        ns_str = (
            f"{ns['request_count']} / {_fmt_num(ns['rps'], 2)} / {_fmt_num(ns['p50_ms'], 1)} / {_fmt_num(ns['p95_ms'], 1)}"
            if ns
            else "N/A"
        )
        st_str = (
            f"{st['request_count']} / {_fmt_num(st['rps'], 2)} / {_fmt_num(st['p50_ms'], 1)} / {_fmt_num(st['p95_ms'], 1)}"
            if st
            else "N/A"
        )
        tt_str = (
            f"{tt['request_count']} / {_fmt_num(tt['p50_ms'], 1)} / {_fmt_num(tt['p95_ms'], 1)} / {_fmt_num(tt['p99_ms'], 1)}"
            if tt
            else "N/A"
        )
        lines.append(f"| `{r['slug']}` | {ns_str} | {st_str} | {tt_str} |")

    # ── Section 3: Cold Per-Query-Class Latency (Head-of-Line Blocking) ───────
    cold_sync_runs = [r for r in runs if r["label"] in {"sync", "async-cold"}]
    lines.extend(
        [
            "",
            "## 3. Cold-Load Per-Query-Class Latency (Head-of-Line Blocking)",
            "",
            "Per-class breakdown under cold load isolates D2 head-of-line blocking: fast `oos` refusals (~53 ms in isolation) and `cve-id` lookups (~3.7 s) queue behind serialized CPU semantic retrievals (~9.2 s/query) when `u >= 10`.",
            "",
            "| Run Slug | Profile | Users | `cve-id` (N / P50 / P95 / Max ms) | `oos` (N / P50 / P95 / Max ms) | `semantic` (N / P50 / P95 / Max ms) | Max Retrieval Waiting (1 Hz Lower Bound) |",
            "|---|---|---:|---|---|---|---:|",
        ]
    )
    for r in cold_sync_runs:
        bqc = r["by_query_class"]
        has_tags = r["has_query_class_tags"]
        qs = r["queue_sampling"]
        lines.append(
            "| `{slug}` | `{label}` | {users} | {cve} | {oos} | {sem} | {ret_w} |".format(
                slug=r["slug"],
                label=r["label"],
                users=r["users"],
                cve=_fmt_class_cell(bqc.get("cve-id"), has_tags),
                oos=_fmt_class_cell(bqc.get("oos"), has_tags),
                sem=_fmt_class_cell(bqc.get("semantic"), has_tags),
                ret_w=qs["max_retrieval_waiting_1hz_lower_bound"] if qs["max_retrieval_waiting_1hz_lower_bound"] is not None else "N/A",
            )
        )

    # ── Section 4: Event-Loop Probes ──────────────────────────────────────────
    lines.extend(
        [
            "",
            "## 4. Event-Loop Responsiveness (`/health probe`)",
            "",
            "`_probe_loop` issues `GET /health` at `1 Hz` (`5.0 s` timeout) throughout each run to monitor ASGI event-loop latency.",
            "",
            "| Run Slug | Profile | Users | Probe Reqs | Probe Fails | Probe P50 (ms) | Probe P95 (ms) | Probe P99 (ms) | Probe Max (ms) |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )

    for r in runs:
        hp = r.get("health_probe")
        if hp and hp["request_count"] > 0:
            lines.append(
                f"| `{r['slug']}` | `{r['label']}` | {r['users']} | {hp['request_count']} | {hp['failure_count']} | "
                f"{_fmt_num(hp['p50_ms'], 1)} | {_fmt_num(hp['p95_ms'], 1)} | {_fmt_num(hp['p99_ms'], 1)} | {_fmt_num(hp['max_ms'], 1)} |"
            )
        else:
            lines.append(f"| `{r['slug']}` | `{r['label']}` | {r['users']} | 0 | 0 | N/A | N/A | N/A | N/A |")

    # ── Section 5: VRAM, Loaded Idle Drift, 1 Hz Queue Bounds & Cache ─────────
    lines.extend(
        [
            "",
            "## 5. VRAM, Loaded-Idle Drift, 1 Hz Queue Lower Bounds & Cache Telemetry",
            "",
            "> **Queue sampling note:** `Max GPU Waiting`, `Max GPU Active`, and `Max Retrieval Waiting` sampled by `_probe_loop` are **`1 Hz lower bounds`** on instantaneous concurrency peaks (`GET /health` polled once per second).",
            "",
            "| Run Slug | Loaded Idle (MiB / Δ vs Run 1) | Peak VRAM (MiB) | VRAM Breached | GPU Waiting (End / 1 Hz Lower Bound Max) | GPU Active (End / 1 Hz Lower Bound Max) | Retrieval Waiting (End / 1 Hz Lower Bound Max) | Cache Hits / Misses |",
            "|---|---|---|---|---|---|---|---|",
        ]
    )

    for r in runs:
        h = r["health"]
        qs = r["queue_sampling"]
        vd = r["vram_drift"]
        if vd["loaded_idle_mib"] is not None:
            flag_str = " **DRIFT > 150 MiB**" if vd["exceeds_150_mib"] else ""
            delta_str = f"{vd['delta_vs_first_mib']:+.0f}" if vd["delta_vs_first_mib"] is not None else "N/A"
            idle_str = f"{vd['loaded_idle_mib']:.0f} ({delta_str}){flag_str}"
        else:
            idle_str = "N/A"

        gw_end = h.get("gpu_waiting") if h.get("available") else None
        ga_end = h.get("gpu_active") if h.get("available") else None
        rw_end = h.get("retrieval_waiting") if h.get("available") else None
        gw_max = qs["max_gpu_waiting_1hz_lower_bound"]
        ga_max = qs["max_gpu_active_1hz_lower_bound"]
        rw_max = qs["max_retrieval_waiting_1hz_lower_bound"]

        ch = h.get("cache_hits") if h.get("available") else None
        cm = h.get("cache_misses") if h.get("available") else None
        cache_str = f"{ch if ch is not None else 'N/A'} / {cm if cm is not None else 'N/A'}"

        lines.append(
            "| `{slug}` | {idle} | {vram} | {breached} | {gw} | {ga} | {rw} | {cache} |".format(
                slug=r["slug"],
                idle=idle_str,
                vram=_fmt_num(h.get("peak_vram_mib") if h.get("available") else None, 1),
                breached=h.get("vram_breached") if h.get("available") and h.get("vram_breached") is not None else "N/A",
                gw=f"{gw_end if gw_end is not None else 'N/A'} / {gw_max if gw_max is not None else 'N/A'}",
                ga=f"{ga_end if ga_end is not None else 'N/A'} / {ga_max if ga_max is not None else 'N/A'}",
                rw=f"{rw_end if rw_end is not None else 'N/A'} / {rw_max if rw_max is not None else 'N/A'}",
                cache=cache_str,
            )
        )

    lines.extend(
        [
            "",
            f"- **Peak Observed VRAM:** `{_fmt_num(aggs['max_peak_vram_mib'], 1)} MiB` (Hard limit `<= {VRAM_HARD_LIMIT_MIB:.0f} MiB`, `any_vram_breached={aggs['any_vram_breached']}`)",
            f"- **Loaded Idle Drift Check (`> {LOADED_IDLE_DRIFT_LIMIT_MIB:.0f} MiB` vs Run 1):** available=`{drift['available']}`, baseline=`{drift['baseline_slug']}` (`{_fmt_num(drift['baseline_loaded_idle_mib'], 1)} MiB`), flagged=`{len(drift['flagged_runs'])}` run(s)",
        ]
    )
    if drift["flagged_runs"]:
        flagged_desc = ", ".join(f"`{x['slug']}` ({x['loaded_idle_mib']:.0f} MiB, +{x['delta_vs_first_mib']:.0f} MiB)" for x in drift["flagged_runs"])
        lines.append(f"- **Flagged Loaded-Idle Drift Runs:** {flagged_desc}")
    ret_base = aggs.get("retrieval_baseline")
    if ret_base:
        lines.append(
            f"- **Phase 1.5 Retrieval Baseline Serialization Cap (`{Path(ret_base['source']).name}`):** median `{ret_base['retrieval_median_ms']} ms` -> `{_fmt_num(ret_base['serialized_retrieval_cap_rps'], 4)} RPS`."
        )

    # ── Section 6: Failures & Gate Scorecard ──────────────────────────────────
    lines.extend(
        [
            "",
            "## 6. Failures & Megaprompt Gate Evaluation (`E6`, `P1–P4`)",
            "",
            "| Gate / Target | Status | Target / Contract | Key Observed Evidence |",
            "|---|---|---|---|",
        ]
    )
    for item in [gates["E6"], perf["P1"], perf["P2"], perf["P3"], perf["P4"]]:
        obs_json = json.dumps(item["observed"], ensure_ascii=False)
        lines.append(f"| **{item['id']}: {item['name']}** | **{item['status']}** | {item['target']} | `{obs_json}` |")

    lines.extend(["", "### 6a. Observed Raw Failure Signatures", ""])
    any_raw = False
    for r in runs:
        for err_row in r["failures"]["raw_rag_errors"]:
            any_raw = True
            lines.append(
                f"- `{r['slug']}` [{err_row['category']}] ({err_row['occurrences']}x) `{err_row['method']} {err_row['name']}`: `{err_row['error']}`"
            )
    if not any_raw:
        lines.append("- Zero RAG request failures recorded across all runs.")

    # ── Section 7: Limits of This Test ────────────────────────────────────────
    lines.extend(
        [
            "",
            "## 7. Limits of This Test",
            "",
            "1. **Incomplete-User Saturation Window:** Each matrix cell runs for `--run-time 60s` with `--stop-timeout 10` while `TIMEOUT_S = 120.0`. Whenever `completed_requests < users`, most virtual users remained queued and never completed a request before shutdown; failure rates on those runs are **not** reliability evidence.",
            "2. **1 Hz Queue Lower Bounds:** `max_gpu_waiting`, `max_gpu_active`, and `max_retrieval_waiting` are polled at `1 Hz` (`GET /health`) and represent `1 Hz` lower bounds rather than true sub-second maxima.",
            "3. **Single-Host Co-Location & Desktop VRAM:** Locust, FastAPI/Uvicorn, CPU retrieval (`torch.set_num_threads(2)`), Ollama (`OLLAMA_NUM_PARALLEL=1`), and Windows desktop compositing share the host RTX 4050 `6,141 MiB` GPU and CPU cores; `loaded_idle_mib` drift above `+150 MiB` indicates external VRAM growth across the sweep.",
            "4. **Small 8-Query Dev Pool (`GOLDEN_FREE_SEED = 20260`):** `POOL` uses `2` `cve-id`, `4` `semantic` (`inscope_dev_100`), and `2` `oos` (`oos_dev_60`) queries (never `golden_150.jsonl`).",
            "",
        ]
    )
    return "\n".join(lines)


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(content)
            fh.flush()
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def compile_reports(
    locust_dir: Path,
    summary_out: Path,
    report_out: Path,
    retrieval_baseline_path: Path | None = DEFAULT_RETRIEVAL_BASELINE,
) -> dict[str, Any]:
    """Read Locust CSV/JSON artifacts from *locust_dir* and write summary JSON + Markdown report."""
    if not locust_dir.exists() or not locust_dir.is_dir():
        raise FileNotFoundError(f"Locust artifact directory does not exist: {locust_dir}")

    stats_files = sorted(locust_dir.glob("*_stats.csv"))
    if not stats_files:
        raise FileNotFoundError(f"No *_stats.csv files found in {locust_dir}")

    runs = [_parse_run(path) for path in stats_files]
    runs.sort(key=lambda r: (_PROFILE_ORDER.get((r["endpoint"], r["mode"]), 99), r["users"], r["slug"]))

    loaded_idle_drift = _annotate_loaded_idle_drift(runs)
    retrieval_baseline = _load_retrieval_baseline(retrieval_baseline_path)
    baseline_gpu = _load_baseline_gpu(locust_dir)
    evaluation = _evaluate_megaprompt_gates(runs, retrieval_baseline, loaded_idle_drift)

    summary = {
        "schema_version": "2.0",
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "locust_dir": str(locust_dir),
        "run_count": len(runs),
        "constants": {
            "vram_soft_target_mib": VRAM_SOFT_TARGET_MIB,
            "vram_hard_limit_mib": VRAM_HARD_LIMIT_MIB,
            "loaded_idle_drift_limit_mib": LOADED_IDLE_DRIFT_LIMIT_MIB,
            "gpu_semaphore_limit": SEMAPHORE_LIMIT,
            "client_timeout_s": CLIENT_TIMEOUT_S,
        },
        "baseline_gpu": baseline_gpu,
        "evaluation": evaluation,
        "runs": runs,
    }

    report_md = _build_markdown_report(summary)
    _atomic_write_text(summary_out, json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    _atomic_write_text(report_out, report_md)
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compile Phase 3 Locust load-test results into JSON summary and Markdown report."
    )
    parser.add_argument(
        "--locust-dir",
        type=Path,
        default=DEFAULT_LOCUST_DIR,
        help="Directory containing *_stats.csv, *_failures.csv, *_health.json, and *_meta.json files.",
    )
    parser.add_argument(
        "--summary-out",
        type=Path,
        default=DEFAULT_SUMMARY_OUT,
        help="Destination path for phase3_load_test_summary.json.",
    )
    parser.add_argument(
        "--report-out",
        type=Path,
        default=DEFAULT_REPORT_OUT,
        help="Destination path for phase3_load_test_report.md.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    summary = compile_reports(
        locust_dir=args.locust_dir,
        summary_out=args.summary_out,
        report_out=args.report_out,
    )
    print(
        f"Compiled {summary['run_count']} Locust run(s) -> "
        f"{args.summary_out} and {args.report_out}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
