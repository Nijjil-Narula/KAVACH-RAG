"""
kavach.retrieval.profiling — Phase 1.5 CPU profiling, quality gates, and baseline CLI.

Modes (all protected by BenchLock):
  --audit     Hardware topology, quantization safety, and token/truncation audit
              -> reports/phase15_profile.json["audit"]
  --affinity  CPU P-core affinity benchmark (off vs p_core_distinct vs p_core_smt)
              -> reports/phase15_profile.json["affinity"]
  --matrix    Precision x Latency x RSS matrix across Tiers 1, 2, and 4
              -> reports/phase15_profile.json["matrix"]
  --gates     Stage 3 accuracy/drift quality gates + RSS selection + precision lock
              -> reports/phase15_quality_gates.json & models/precision_lock.json
  --baseline  Stage 5 resident warm-pipeline latency & RSS baseline
              -> reports/phase15_latency_baseline.json
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

from kavach.runtime_guard import (  # noqa: E402
    apply_hardware_guard,
    detect_core_topology,
    rss_mb,
    vram_probe,
)

import argparse
import gc
import json
import logging
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import psutil
import torch
from transformers import AutoTokenizer

from kavach.config import Settings, get_settings
from kavach.retrieval.benchlock import BenchLock
from kavach.retrieval.embedder import Embedder
from kavach.retrieval.metrics import dedupe_to_docs, ndcg_at_k, recall_at_k, reciprocal_rank
from kavach.retrieval.pipeline import VectorRetrievalPipeline
from kavach.retrieval.quantization import (
    check_quantize_dynamic_safety,
    get_quantized_engine_info,
)
from kavach.retrieval.reranker import Reranker, _detect_pair_template
from kavach.retrieval.router import AgentIntentRouter
from kavach.retrieval.schema import GoldenItem, QuestionType
from kavach.retrieval.train_router import evaluate_router_precision
from kavach.retrieval.vector_store import ChunkStore, SQ8Index

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("kavach.profiling")


# ─────────────────────────────────────────────────────────────────────────────
# Shared Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _atomic_write_json(path: Path, payload: Any) -> None:
    """Atomically write JSON to *path* using a same-directory temporary file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)
    log.info("Written: %s", path)


def _load_json(path: Path) -> dict[str, Any]:
    """Load JSON dict from *path* or return an empty dict if absent."""
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _update_profile_section(settings: Settings, key: str, section: dict[str, Any]) -> Path:
    """Merge *section* under *key* into reports/phase15_profile.json atomically."""
    profile_path = settings.reports_dir / "phase15_profile.json"
    data = _load_json(profile_path)
    data["updated_at"] = datetime.now(tz=timezone.utc).isoformat()
    data[key] = section
    _atomic_write_json(profile_path, data)
    return profile_path


def _wait_for_cpu_idle(
    max_cpu_pct: float = 15.0,
    stable_s: float = 2.0,
    timeout_s: float = 60.0,
) -> dict[str, Any]:
    """Wait up to *timeout_s* for system CPU usage to settle below *max_cpu_pct*."""
    psutil.cpu_percent(interval=None)
    t0 = time.monotonic()
    calm_since: float | None = None
    last_cpu = 0.0
    while True:
        time.sleep(0.5)
        last_cpu = float(psutil.cpu_percent(interval=None))
        now = time.monotonic()
        if last_cpu <= max_cpu_pct:
            if calm_since is None:
                calm_since = now
            if (now - calm_since) >= stable_s:
                return {
                    "settled": True,
                    "idle_wait_s": round(now - t0, 2),
                    "observed_cpu_pct": round(last_cpu, 1),
                    "threshold_pct": max_cpu_pct,
                }
        else:
            calm_since = None
        if (now - t0) >= timeout_s:
            return {
                "settled": False,
                "idle_wait_s": round(now - t0, 2),
                "observed_cpu_pct": round(last_cpu, 1),
                "threshold_pct": max_cpu_pct,
            }


def _stats_ms(values: list[float]) -> dict[str, Any]:
    """Compute min/mean/median/p50/p95/max summary statistics in milliseconds."""
    if not values:
        return {
            "count": 0,
            "min": None,
            "mean": None,
            "median": None,
            "p50": None,
            "p95": None,
            "max": None,
        }
    arr = np.asarray(values, dtype=np.float64)
    p50 = round(float(np.percentile(arr, 50)), 3)
    return {
        "count": len(values),
        "min": round(float(np.min(arr)), 3),
        "mean": round(float(np.mean(arr)), 3),
        "median": p50,
        "p50": p50,
        "p95": round(float(np.percentile(arr, 95)), 3),
        "max": round(float(np.max(arr)), 3),
    }


def _load_golden_items(settings: Settings) -> list[GoldenItem]:
    """Load and validate golden_150.jsonl."""
    path = settings.data_golden / "golden_150.jsonl"
    items = [
        GoldenItem.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return items


def _load_rerank_checkpoint_rows(settings: Settings) -> list[dict[str, Any]]:
    """Load stored 8-candidate sets from data/processed/ablation_rerank_ckpt.jsonl."""
    ckpt_path = settings.data_processed / "ablation_rerank_ckpt.jsonl"
    rows: list[dict[str, Any]] = []
    for line in ckpt_path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if s:
            rows.append(json.loads(s))
    rows.sort(key=lambda r: int(r["idx"]))
    return rows


def _run_subprocess_json(args: list[str], extra_env: dict[str, str] | None = None) -> dict[str, Any]:
    """Execute an isolated CPU worker subprocess and parse its JSON stdout payload."""
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["OMP_NUM_THREADS"] = "2"
    env["MKL_NUM_THREADS"] = "2"
    env["OPENBLAS_NUM_THREADS"] = "2"
    env["TOKENIZERS_PARALLELISM"] = "false"
    env["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    env["HF_HOME"] = str(Path(_PROJECT_ROOT) / "models" / "hf_cache")
    if extra_env:
        env.update(extra_env)

    cmd = [sys.executable, "-m", "kavach.retrieval.profiling", *args]
    proc = subprocess.run(
        cmd,
        cwd=_PROJECT_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"Worker {' '.join(args)} failed (exit={proc.returncode}):\n"
            f"STDERR:\n{proc.stderr}\nSTDOUT:\n{proc.stdout}"
        )
    for line in reversed(proc.stdout.splitlines()):
        stripped = line.strip()
        if stripped.startswith("{") and stripped.endswith("}"):
            return json.loads(stripped)
    raise RuntimeError(f"Worker {' '.join(args)} emitted no JSON line:\n{proc.stdout}")


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2A: --audit
# ─────────────────────────────────────────────────────────────────────────────


def _collect_os_preflight(wait_idle: bool = True) -> dict[str, Any]:
    """Collect non-invasive Windows OS power, CPU load, and competing process telemetry."""
    battery_info: dict[str, Any] = {"available": False, "power_plugged": None, "percent": None}
    try:
        bat = psutil.sensors_battery()
        if bat is not None:
            battery_info = {
                "available": True,
                "power_plugged": bool(bat.power_plugged) if bat.power_plugged is not None else None,
                "percent": round(float(bat.percent), 1),
            }
    except Exception as exc:
        battery_info["error"] = str(exc)

    power_scheme = ""
    if os.name == "nt":
        try:
            res = subprocess.run(
                ["powercfg", "/getactivescheme"],
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
            power_scheme = (res.stdout or res.stderr or "").strip()
        except Exception as exc:
            power_scheme = f"error: {exc}"

    cpu_pct_1s = round(float(psutil.cpu_percent(interval=1.0)), 1)

    self_pid = os.getpid()
    parent_pid = os.getppid()
    competing_python: list[dict[str, Any]] = []
    competing_project: list[dict[str, Any]] = []
    for proc in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            pid = int(proc.info["pid"])
            if pid in (self_pid, parent_pid):
                continue
            name = str(proc.info.get("name") or "")
            cmdline_list = proc.info.get("cmdline") or []
            cmdline_str = " ".join(str(part) for part in cmdline_list)
            is_python = "python" in name.lower()
            is_project = "kavach" in cmdline_str.lower()
            if is_python or is_project:
                entry = {"pid": pid, "name": name, "cmdline": cmdline_str[:300]}
                if is_python:
                    competing_python.append(entry)
                if is_project:
                    competing_project.append(entry)
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            continue

    idle_result = (
        _wait_for_cpu_idle(max_cpu_pct=15.0, stable_s=2.0, timeout_s=60.0)
        if wait_idle
        else None
    )
    if idle_result is not None and not idle_result.get("settled", True):
        log.warning(
            "CPU remained noisy (>=15%%) after %.1fs wait (observed=%.1f%%); recording noise honestly.",
            idle_result.get("idle_wait_s", 0.0),
            idle_result.get("observed_cpu_pct", 0.0),
        )

    return {
        "battery": battery_info,
        "powercfg_active_scheme": power_scheme,
        "initial_cpu_percent_1s": cpu_pct_1s,
        "competing_python_count": len(competing_python),
        "competing_python_processes": competing_python,
        "competing_project_count": len(competing_project),
        "competing_project_processes": competing_project,
        "cpu_idle_wait": idle_result,
    }


def _audit_model_worker(tier: str) -> dict[str, Any]:
    """Isolated worker measuring load, first-call (cold JIT), and warm-call latency in FP32."""
    apply_hardware_guard()
    cfg = Settings(
        embed_precision="fp32",
        rerank_precision="fp32",
        router_precision="fp32",
        rerank_max_length=576,
        cpu_affinity_mode="off",
        abstention_enabled=False,
    )
    items = _load_golden_items(cfg)
    sem_indices = [
        i for i, it in enumerate(items) if it.expected_route == "SEMANTIC_RAG"
    ][:5]
    if not sem_indices:
        raise RuntimeError("No SEMANTIC_RAG queries available in golden_150.jsonl")

    rss_before = round(rss_mb(), 2)
    warm_latencies: list[float] = []

    if tier == "router":
        t_load0 = time.perf_counter()
        router = AgentIntentRouter(cfg)
        router.load()
        load_ms = (time.perf_counter() - t_load0) * 1000.0
        rss_load = round(rss_mb(), 2)

        neural_indices = [
            i for i, it in enumerate(items)
            if it.expected_route == "SEMANTIC_RAG" and router.needs_model_for(it.query)
        ]
        if len(neural_indices) < 5:
            neural_indices = sem_indices
        first_q = items[neural_indices[0]].query
        warm_queries = [items[idx].query for idx in neural_indices[:5]]

        t_first0 = time.perf_counter()
        _ = router.route(first_q)
        first_ms = (time.perf_counter() - t_first0) * 1000.0

        t_warm0 = time.perf_counter()
        router.warmup(passes=3)
        warmup_ms = (time.perf_counter() - t_warm0) * 1000.0
        rss_warm = round(rss_mb(), 2)

        for q in warm_queries:
            t0 = time.perf_counter()
            _ = router.route(q)
            warm_latencies.append((time.perf_counter() - t0) * 1000.0)
    elif tier == "embedder":
        t_load0 = time.perf_counter()
        embedder = Embedder(cfg)
        load_ms = (time.perf_counter() - t_load0) * 1000.0
        rss_load = round(rss_mb(), 2)

        first_q = items[sem_indices[0]].query
        t_first0 = time.perf_counter()
        _ = embedder.embed([first_q], kind="query", batch_size=1)
        embedder.clear_cache()
        first_ms = (time.perf_counter() - t_first0) * 1000.0

        t_warm0 = time.perf_counter()
        embedder.warmup(passes=3)
        warmup_ms = (time.perf_counter() - t_warm0) * 1000.0
        rss_warm = round(rss_mb(), 2)

        for idx in sem_indices:
            q = items[idx].query
            t0 = time.perf_counter()
            _ = embedder.embed([q], kind="query", batch_size=1)
            embedder.clear_cache()
            warm_latencies.append((time.perf_counter() - t0) * 1000.0)
    elif tier == "reranker":
        store = ChunkStore(cfg.data_processed / "corpus_chunks.parquet")
        ckpt_rows = _load_rerank_checkpoint_rows(cfg)

        t_load0 = time.perf_counter()
        reranker = Reranker(cfg)
        load_ms = (time.perf_counter() - t_load0) * 1000.0
        rss_load = round(rss_mb(), 2)

        first_idx = sem_indices[0]
        first_cands = [store.get(str(c["chunk_id"])) for c in ckpt_rows[first_idx]["scored_8"]]
        t_first0 = time.perf_counter()
        _ = reranker.rerank(items[first_idx].query, first_cands, top_k=3, return_all=True)
        first_ms = (time.perf_counter() - t_first0) * 1000.0

        t_warm0 = time.perf_counter()
        reranker.warmup(passes=3)
        warmup_ms = (time.perf_counter() - t_warm0) * 1000.0
        rss_warm = round(rss_mb(), 2)

        for idx in sem_indices:
            cands = [store.get(str(c["chunk_id"])) for c in ckpt_rows[idx]["scored_8"]]
            t0 = time.perf_counter()
            _ = reranker.rerank(items[idx].query, cands, top_k=3, return_all=True)
            warm_latencies.append((time.perf_counter() - t0) * 1000.0)
    else:
        raise ValueError(f"Unknown audit model tier: {tier!r}")

    warm_stats = _stats_ms(warm_latencies)
    warm_mean = float(warm_stats["mean"] or 0.0)
    return {
        "tier": tier,
        "precision": "fp32",
        "rss_before_mb": rss_before,
        "rss_after_load_mb": rss_load,
        "rss_after_warm_mb": rss_warm,
        "load_time_ms": round(load_ms, 3),
        "first_call_ms": round(first_ms, 3),
        "warmup_3_passes_ms": round(warmup_ms, 3),
        "warm_call_ms": warm_mean,
        "warm_calls_ms": warm_stats,
        "first_to_warm_ratio": round(first_ms / warm_mean, 2) if warm_mean > 0 else None,
        "vram_probe": dict(vram_probe()),
    }


def run_audit(settings: Settings) -> dict[str, Any]:
    """Audit OS preflight, cold-vs-warm latency, quantization safety, and token budgets."""
    apply_hardware_guard()
    log.info("=== Stage 2A: Hardware, OS Preflight, Cold-vs-Warm & Token Budget Audit ===")

    os_preflight = _collect_os_preflight(wait_idle=True)
    topo = detect_core_topology()
    quant_safety = check_quantize_dynamic_safety()
    items = _load_golden_items(settings)
    store = ChunkStore(settings.data_processed / "corpus_chunks.parquet")
    ckpt_rows = _load_rerank_checkpoint_rows(settings)

    rerank_tok = AutoTokenizer.from_pretrained(settings.rerank_model_id, local_files_only=True)
    pair_template = _detect_pair_template(rerank_tok)
    special_count = pair_template.special_token_count

    q_tok_lens = [
        len(rerank_tok.encode(it.query, add_special_tokens=False))
        for it in items
    ]
    p_tok_lens = [
        len(rerank_tok.encode(store.get(cid)["content"], add_special_tokens=False))
        for cid in store.all_chunk_ids()
    ]

    pair_lens: list[int] = []
    pairs_over_576 = 0
    pairs_over_768 = 0
    pairs_over_1024 = 0
    truncated_query_ids_at_576: list[str] = []

    for row in ckpt_rows:
        q_len = len(rerank_tok.encode(items[int(row["idx"])].query, add_special_tokens=False))
        any_trunc = False
        for cand in row["scored_8"]:
            p_len = len(
                rerank_tok.encode(store.get(str(cand["chunk_id"]))["content"], add_special_tokens=False)
            )
            tot = q_len + p_len + special_count
            pair_lens.append(tot)
            if tot > 576:
                pairs_over_576 += 1
                any_trunc = True
            if tot > 768:
                pairs_over_768 += 1
            if tot > 1024:
                pairs_over_1024 += 1
        if any_trunc:
            truncated_query_ids_at_576.append(str(row["test_id"]))

    cold_vs_warm: dict[str, Any] = {}
    for tier in ("router", "embedder", "reranker"):
        log.info("Auditing cold-vs-warm latency for tier=%s ...", tier)
        cold_vs_warm[tier] = _run_subprocess_json(
            ["--_audit-model-worker", "--tier", tier],
            extra_env={"KAVACH_CPU_AFFINITY_MODE": "off"},
        )

    sum_load_ms = round(sum(cold_vs_warm[t]["load_time_ms"] for t in ("router", "embedder", "reranker")), 3)
    sum_first_ms = round(sum(cold_vs_warm[t]["first_call_ms"] for t in ("router", "embedder", "reranker")), 3)
    sum_warm_ms = round(sum(cold_vs_warm[t]["warm_call_ms"] for t in ("router", "embedder", "reranker")), 3)

    section = {
        "audited_at": datetime.now(tz=timezone.utc).isoformat(),
        "os_preflight": os_preflight,
        "torch_version": torch.__version__,
        "torch_num_threads": torch.get_num_threads(),
        "torch_interop_threads": torch.get_num_interop_threads(),
        "venv_gpu_absent": not (Path(_PROJECT_ROOT) / ".venv_gpu").exists(),
        "vram_probe": dict(vram_probe()),
        "rss_mb": round(rss_mb(), 2),
        "rss_ceiling_mb": float(settings.rss_ceiling_mb),
        "core_topology": topo,
        "quantization_safety": quant_safety,
        "cold_vs_warm": {
            "tiers": cold_vs_warm,
            "summary": {
                "sum_load_time_ms": sum_load_ms,
                "sum_first_call_ms": sum_first_ms,
                "sum_warm_call_ms": sum_warm_ms,
                "cold_load_plus_first_ms": round(sum_load_ms + sum_first_ms, 3),
                "cold_vs_warm_delta_ms": round(sum_load_ms + sum_first_ms - sum_warm_ms, 3),
            },
        },
        "token_audit": {
            "rerank_special_tokens": special_count,
            "golden_queries_over_96_tokens": sum(1 for x in q_tok_lens if x > 96),
            "golden_query_tokens": _stats_ms([float(x) for x in q_tok_lens]),
            "corpus_passages_over_512_tokens": sum(1 for x in p_tok_lens if x > 512),
            "corpus_passage_tokens": _stats_ms([float(x) for x in p_tok_lens]),
            "ckpt_pairs_total": len(pair_lens),
            "ckpt_pairs_over_576": pairs_over_576,
            "ckpt_pairs_over_768": pairs_over_768,
            "ckpt_pairs_over_1024": pairs_over_1024,
            "queries_with_truncation_at_576": truncated_query_ids_at_576,
            "ckpt_pair_tokens": _stats_ms([float(x) for x in pair_lens]),
        },
    }
    _update_profile_section(settings, "audit", section)
    return section


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2B: --affinity
# ─────────────────────────────────────────────────────────────────────────────


def _affinity_worker(mode: str) -> dict[str, Any]:
    """Isolated worker measuring latency under a fresh process CPU affinity mask."""
    apply_hardware_guard()
    settings = Settings(
        embed_precision="fp32",
        rerank_precision="fp32",
        router_precision="fp32",
        rerank_max_length=576,
        cpu_affinity_mode=mode,  # type: ignore[arg-type]
        abstention_enabled=False,
    )
    items = _load_golden_items(settings)
    store = ChunkStore(settings.data_processed / "corpus_chunks.parquet")
    ckpt_rows = _load_rerank_checkpoint_rows(settings)

    router = AgentIntentRouter(settings)
    router.load()
    router.warmup()

    embedder = Embedder(settings)
    embedder.warmup()

    reranker = Reranker(settings)
    reranker.warmup()

    # Benchmark 5 representative semantic queries across all 3 tiers
    sample_indices = [0, 10, 60, 105, 130]
    router_ms: list[float] = []
    embed_ms: list[float] = []
    rerank_ms: list[float] = []
    total_ms: list[float] = []

    for idx in sample_indices:
        query = items[idx].query
        cands = [store.get(str(c["chunk_id"])) for c in ckpt_rows[idx]["scored_8"]]

        t0 = time.perf_counter()
        _ = router.route(query)
        t_r = (time.perf_counter() - t0) * 1000.0

        t1 = time.perf_counter()
        _ = embedder.embed([query], kind="query", batch_size=1)
        embedder.clear_cache()
        t_e = (time.perf_counter() - t1) * 1000.0

        t2 = time.perf_counter()
        _ = reranker.rerank(query, cands, top_k=3, return_all=True)
        t_rr = (time.perf_counter() - t2) * 1000.0

        router_ms.append(t_r)
        embed_ms.append(t_e)
        rerank_ms.append(t_rr)
        total_ms.append(t_r + t_e + t_rr)

    return {
        "mode": mode,
        "cpu_affinity": sorted(psutil.Process().cpu_affinity()),
        "rss_mb": round(rss_mb(), 2),
        "vram_probe": dict(vram_probe()),
        "router_ms": _stats_ms(router_ms),
        "embed_ms": _stats_ms(embed_ms),
        "rerank_8_ms": _stats_ms(rerank_ms),
        "total_neural_ms": _stats_ms(total_ms),
    }


def run_affinity(settings: Settings) -> dict[str, Any]:
    """Benchmark off vs p_core_distinct vs p_core_smt in fresh subprocesses."""
    log.info("=== Stage 2B: CPU Core Affinity Benchmark ===")
    topo = detect_core_topology()
    if topo.get("affinity_unavailable"):
        section = {
            "measured_at": datetime.now(tz=timezone.utc).isoformat(),
            "affinity_unavailable": True,
            "reason": topo.get("reason"),
            "recommended_affinity_mode": "off",
            "modes": {},
        }
        _update_profile_section(settings, "affinity", section)
        return section

    modes = ("off", "p_core_distinct", "p_core_smt")
    results: dict[str, Any] = {}
    for mode in modes:
        idle = _wait_for_cpu_idle()
        log.info("Benchmarking affinity mode=%s (idle=%s)...", mode, idle)
        res = _run_subprocess_json(
            ["--_affinity-worker", "--mode", mode],
            extra_env={"KAVACH_CPU_AFFINITY_MODE": mode},
        )
        res["idle_check"] = idle
        results[mode] = res

    off_mean = float(results["off"]["total_neural_ms"]["mean"])
    best_mode = min(
        modes, key=lambda m: float(results[m]["total_neural_ms"]["mean"])
    )
    best_mean = float(results[best_mode]["total_neural_ms"]["mean"])
    improvement_pct = ((off_mean - best_mean) / off_mean * 100.0) if off_mean > 0 else 0.0
    # Keep "off" if improvement is within 2% noise band; otherwise select best_mode
    recommended = best_mode if (best_mode != "off" and improvement_pct >= 2.0) else "off"

    section = {
        "measured_at": datetime.now(tz=timezone.utc).isoformat(),
        "affinity_unavailable": False,
        "modes": results,
        "off_mean_total_ms": round(off_mean, 3),
        "best_mean_total_ms": round(best_mean, 3),
        "improvement_vs_off_pct": round(improvement_pct, 2),
        "recommended_affinity_mode": recommended,
    }
    _update_profile_section(settings, "affinity", section)
    return section


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2C: --matrix
# ─────────────────────────────────────────────────────────────────────────────


_MATRIX_STRATA_QUOTAS: tuple[tuple[QuestionType, int], ...] = (
    (QuestionType.FACTUAL, 10),
    (QuestionType.MULTI_HOP, 10),
    (QuestionType.CONTEXT_NOISE, 10),
    (QuestionType.CONFLICT_TRAP, 5),
    (QuestionType.MALFORMED, 3),
    (QuestionType.OUT_OF_SCOPE, 2),
)


def _select_matrix_indices(
    items: list[GoldenItem],
) -> tuple[list[int], dict[str, int]]:
    """Deterministically select 40 stratified golden indices for Stage 2C."""
    selected: list[int] = []
    counts: dict[str, int] = {}
    for qtype, quota in _MATRIX_STRATA_QUOTAS:
        matching = [i for i, it in enumerate(items) if it.question_type == qtype]
        if len(matching) < quota:
            raise RuntimeError(
                f"Insufficient golden items for {qtype.value}: need {quota}, found {len(matching)}"
            )
        # Spread evenly across matching items
        if quota == len(matching):
            picked = matching
        else:
            step = len(matching) / float(quota)
            picked = [matching[int(k * step)] for k in range(quota)]
        selected.extend(picked)
        counts[qtype.value] = len(picked)
    if len(selected) != 40:
        raise RuntimeError(f"Expected 40 stratified matrix indices, got {len(selected)}")
    return selected, counts


def _matrix_worker(tier: str, precision: str, max_length: int) -> dict[str, Any]:
    """Isolated worker measuring 40-query latency and RSS after 5 warmup passes."""
    apply_hardware_guard()
    base = Settings(abstention_enabled=False)
    items = _load_golden_items(base)
    sample_indices, strata_counts = _select_matrix_indices(items)

    rss_before = rss_mb()
    latencies: list[float] = []
    extra: dict[str, Any] = {
        "query_count": len(sample_indices),
        "warmup_passes": 5,
        "strata_counts": strata_counts,
    }

    if tier == "router":
        cfg = base.model_copy(update={"router_precision": precision})
        router = AgentIntentRouter(cfg)
        router.load()
        router.warmup(passes=5)
        rss_after = rss_mb()
        track_a_count = 0
        track_b_count = 0
        for idx in sample_indices:
            q = items[idx].query
            t0 = time.perf_counter()
            dec = router.route(q)
            latencies.append((time.perf_counter() - t0) * 1000.0)
            if dec.fast_path:
                track_a_count += 1
            else:
                track_b_count += 1
        extra["track_a_count"] = track_a_count
        extra["track_b_count"] = track_b_count
    elif tier == "embedder":
        cfg = base.model_copy(update={"embed_precision": precision})
        embedder = Embedder(cfg)
        embedder.warmup(passes=5)
        rss_after = rss_mb()
        for idx in sample_indices:
            q = items[idx].query
            t0 = time.perf_counter()
            embedder.embed([q], kind="query", batch_size=1)
            embedder.clear_cache()
            latencies.append((time.perf_counter() - t0) * 1000.0)
    elif tier == "sq8_search":
        store = ChunkStore(base.data_processed / "corpus_chunks.parquet")
        sq8 = SQ8Index.load(base.models_dir / "index_sq8_mrl512.faiss")
        with np.load(base.data_processed / "ablation_qvecs_ckpt.npz", allow_pickle=False) as d:
            ckpt_512 = np.asarray(d["q_vecs_512"], dtype=np.float32)
        for warmup_idx in sample_indices[:5]:
            hits = sq8.search(ckpt_512[warmup_idx], k=base.candidate_k)
            for cid, _ in hits:
                _ = store.get(cid)
        rss_after = rss_mb()
        search_only_ms: list[float] = []
        for idx in sample_indices:
            qvec = ckpt_512[idx]
            t0 = time.perf_counter()
            hits = sq8.search(qvec, k=base.candidate_k)
            t_search = (time.perf_counter() - t0) * 1000.0
            for cid, _ in hits:
                _ = store.get(cid)
            t_total = (time.perf_counter() - t0) * 1000.0
            search_only_ms.append(t_search)
            latencies.append(t_total)
        extra["search_only_ms"] = _stats_ms(search_only_ms)
    elif tier == "reranker":
        cfg = base.model_copy(
            update={"rerank_precision": precision, "rerank_max_length": max_length}
        )
        store = ChunkStore(cfg.data_processed / "corpus_chunks.parquet")
        ckpt_rows = _load_rerank_checkpoint_rows(cfg)
        reranker = Reranker(cfg)
        reranker.warmup(passes=5)
        reranker.reset_truncation_stats()
        rss_after = rss_mb()
        for idx in sample_indices:
            q = items[idx].query
            cands = [store.get(str(c["chunk_id"])) for c in ckpt_rows[idx]["scored_8"]]
            t0 = time.perf_counter()
            reranker.rerank(q, cands, top_k=3, return_all=True)
            latencies.append((time.perf_counter() - t0) * 1000.0)
        trunc_stats = reranker.truncation_stats
        if trunc_stats["passages_truncated"] != 0:
            raise RuntimeError(
                f"Reranker D2 contract violated in matrix worker: {trunc_stats}"
            )
        extra["truncation_stats"] = trunc_stats
    else:
        raise ValueError(f"Unknown matrix tier: {tier!r}")

    return {
        "tier": tier,
        "precision": precision,
        "max_length": max_length,
        "rss_before_mb": round(rss_before, 2),
        "rss_after_mb": round(rss_after, 2),
        "delta_rss_mb": round(rss_after - rss_before, 2),
        "latency_ms": _stats_ms(latencies),
        "vram_probe": dict(vram_probe()),
        **extra,
    }


def run_matrix(settings: Settings) -> dict[str, Any]:
    """Benchmark 40-query precision x latency x RSS matrix across Tiers 1, 2, 3 (SQ8), and 4."""
    log.info("=== Stage 2C: Precision x Latency x RSS Matrix (40 Stratified Queries, 5 Warmups) ===")
    profile = _load_json(settings.reports_dir / "phase15_profile.json")
    aff_mode = (
        profile.get("affinity", {}).get("recommended_affinity_mode") or "off"
    )
    extra_env = {"KAVACH_CPU_AFFINITY_MODE": str(aff_mode)}

    cells = [
        ("router", "fp32", 128),
        ("router", "bf16", 128),
        ("router", "int8", 128),
        ("embedder", "fp32", 512),
        ("embedder", "bf16", 512),
        ("embedder", "int8", 512),
        ("sq8_search", "sq8_mrl512", 512),
        ("reranker", "fp32", 576),
        ("reranker", "int8", 576),
    ]

    matrix_results: dict[str, dict[str, Any]] = {
        "router": {},
        "embedder": {},
        "sq8_search": {},
        "reranker": {},
    }
    strata_counts: dict[str, int] | None = None
    for tier, prec, max_len in cells:
        idle = _wait_for_cpu_idle()
        label = f"{prec}@{max_len}" if tier == "reranker" else prec
        log.info("Matrix cell: %s %s (affinity=%s, idle=%s)...", tier, label, aff_mode, idle)
        res = _run_subprocess_json(
            [
                "--_matrix-worker",
                "--tier",
                tier,
                "--precision",
                prec,
                "--max-length",
                str(max_len),
            ],
            extra_env=extra_env,
        )
        res["idle_check"] = idle
        if strata_counts is None and "strata_counts" in res:
            strata_counts = dict(res["strata_counts"])
        matrix_results[tier][label] = res

    section = {
        "measured_at": datetime.now(tz=timezone.utc).isoformat(),
        "affinity_mode_used": aff_mode,
        "query_count": 40,
        "warmup_passes": 5,
        "strata_counts": strata_counts or {},
        "reranker_bf16_status": "Excluded per Decision D5 (ModernBERT bfloat16 CPU excluded; candidates: fp32, int8)",
        "tiers": matrix_results,
    }
    _update_profile_section(settings, "matrix", section)
    return section


# ─────────────────────────────────────────────────────────────────────────────
# Stage 3: --gates
# ─────────────────────────────────────────────────────────────────────────────


def _evaluate_embedder_gate(settings: Settings) -> dict[str, Any]:
    """Gate Tier 2 Embedder precisions vs ablation_qvecs_ckpt.npz and Run 4 Recall@8."""
    items = _load_golden_items(settings)
    store = ChunkStore(settings.data_processed / "corpus_chunks.parquet")
    c2doc = store.chunk_to_doc_map()
    sq8 = SQ8Index.load(settings.models_dir / "index_sq8_mrl512.faiss")

    ckpt_path = settings.data_processed / "ablation_qvecs_ckpt.npz"
    with np.load(ckpt_path, allow_pickle=False) as d:
        ckpt_1024 = np.asarray(d["q_vecs_1024"], dtype=np.float32)
        ckpt_512 = np.asarray(d["q_vecs_512"], dtype=np.float32)

    ablation_path = settings.reports_dir / "retrieval_ablation_results.json"
    ab_data = _load_json(ablation_path)
    run4_entry = next(r for r in ab_data["runs"] if r["run_id"] == "Run 4")
    baseline_run4_doc_r8 = float(run4_entry["doc_metrics"]["overall"]["by_k"]["8"]["recall"])

    ckpt_top8 = [
        [cid for cid, _ in sq8.search(ckpt_512[i], k=settings.candidate_k)]
        for i in range(len(items))
    ]
    in_scope_indices = [
        i for i, it in enumerate(items) if it.question_type != QuestionType.OUT_OF_SCOPE
    ]

    variants: dict[str, Any] = {}
    for prec in ("fp32", "int8", "bf16"):
        log.info("Evaluating Embedder gate for precision=%s ...", prec)
        cfg = settings.model_copy(update={"embed_precision": prec})
        embedder = Embedder(cfg)
        embedder.warmup()
        rss_after = round(rss_mb(), 2)

        queries = [it.query for it in items]
        batch = embedder.embed(queries, kind="query", batch_size=8)
        new_1024 = batch.full1024
        new_512 = batch.mrl512

        cos_1024 = np.sum(new_1024 * ckpt_1024, axis=1)
        cos_512 = np.sum(new_512 * ckpt_512, axis=1)
        failed_cosine_test_ids = [
            items[idx].test_id
            for idx in range(len(items))
            if float(cos_1024[idx]) < 0.995 or float(cos_512[idx]) < 0.995
        ]

        overlaps: list[float] = []
        r8_vals: list[float] = []
        for idx in range(len(items)):
            hits = [cid for cid, _ in sq8.search(new_512[idx], k=settings.candidate_k)]
            overlaps.append(len(set(hits) & set(ckpt_top8[idx])) / float(settings.candidate_k))
            if idx in in_scope_indices:
                docs = dedupe_to_docs(hits, c2doc)
                r8_vals.append(recall_at_k(docs, items[idx].expected_doc_ids, 8))

        cos_1024_min = float(np.min(cos_1024))
        cos_512_min = float(np.min(cos_512))
        mean_overlap = float(np.mean(overlaps))
        doc_r8 = float(np.mean(r8_vals))
        r8_delta = doc_r8 - baseline_run4_doc_r8
        passed = bool(
            cos_1024_min >= 0.995
            and cos_512_min >= 0.995
            and mean_overlap >= 0.98
            and abs(r8_delta) <= 0.01
        )

        variants[prec] = {
            "precision": prec,
            "passed": passed,
            "rss_mb": rss_after,
            "cosine_full1024_min": round(cos_1024_min, 6),
            "cosine_full1024_mean": round(float(np.mean(cos_1024)), 6),
            "cosine_mrl512_min": round(cos_512_min, 6),
            "cosine_mrl512_mean": round(float(np.mean(cos_512)), 6),
            "failed_cosine_test_ids": failed_cosine_test_ids,
            "sq8_top8_overlap_mean": round(mean_overlap, 6),
            "sq8_top8_overlap_min": round(float(np.min(overlaps)), 6),
            "run4_doc_recall_at_8": round(doc_r8, 4),
            "baseline_run4_doc_recall_at_8": round(baseline_run4_doc_r8, 4),
            "run4_doc_recall_at_8_delta": round(r8_delta, 4),
        }

        del embedder
        gc.collect()

    return {
        "baseline_run4_doc_recall_at_8": round(baseline_run4_doc_r8, 4),
        "variants": variants,
    }


def _build_ckpt_rerank_reference(
    items: list[GoldenItem],
    ckpt_rows: list[dict[str, Any]],
    c2doc: dict[str, str],
) -> dict[str, Any]:
    """Construct R_cur reference vectors and MRR metrics from ablation_rerank_ckpt.jsonl."""
    in_indices = {
        i for i, it in enumerate(items) if it.question_type != QuestionType.OUT_OF_SCOPE
    }
    doc_mrr_5: list[float] = []
    chunk_mrr_5: list[float] = []
    doc_mrr_6: list[float] = []
    top1_cids: list[str] = []
    top3_sets: list[set[str]] = []
    flat_logits: list[float] = []
    top1_logits: list[float] = []

    for row in ckpt_rows:
        idx = int(row["idx"])
        item = items[idx]
        ckpt_scored = sorted(
            row["scored_8"],
            key=lambda x: (float(x["rerank_raw_logit"]), float(x["rerank_score"])),
            reverse=True,
        )
        cids = [str(c["chunk_id"]) for c in ckpt_scored]
        top1_cids.append(cids[0])
        top3_sets.append(set(cids[:3]))
        top1_logits.append(float(ckpt_scored[0]["rerank_raw_logit"]))
        for c in ckpt_scored:
            flat_logits.append(float(c["rerank_raw_logit"]))
        if idx in in_indices:
            d3 = dedupe_to_docs(cids[:3], c2doc)
            d5 = dedupe_to_docs(cids[:5], c2doc)
            doc_mrr_5.append(reciprocal_rank(d3, item.expected_doc_ids))
            chunk_mrr_5.append(reciprocal_rank(cids[:3], item.expected_source_chunk_ids))
            doc_mrr_6.append(reciprocal_rank(d5, item.expected_doc_ids))

    top1_arr = np.asarray(top1_logits, dtype=np.float64)
    return {
        "run5_doc_mrr": round(float(np.mean(doc_mrr_5)), 4),
        "run5_chunk_mrr": round(float(np.mean(chunk_mrr_5)), 4),
        "run6_doc_mrr": round(float(np.mean(doc_mrr_6)), 4),
        "top1_logit_mean": round(float(np.mean(top1_arr)), 5),
        "top1_logit_median": round(float(np.median(top1_arr)), 5),
        "_raw_top1_cids": top1_cids,
        "_raw_top3_sets": [sorted(s) for s in top3_sets],
        "_raw_logits": flat_logits,
    }


def _score_reranker_variant(
    settings: Settings,
    items: list[GoldenItem],
    store: ChunkStore,
    c2doc: dict[str, str],
    ckpt_rows: list[dict[str, Any]],
    precision: str,
    max_length: int,
    long_query_max_length: int | None = None,
    contingency_l_applied: bool = False,
) -> dict[str, Any]:
    """Score all 150 x 8 stored candidates and compute ranking/logit metrics."""
    cfg = settings.model_copy(
        update={
            "rerank_precision": precision,
            "rerank_max_length": max_length,
            "rerank_long_query_max_length": long_query_max_length,
            "contingency_l_applied": contingency_l_applied,
        }
    )
    reranker = Reranker(cfg)
    reranker.warmup()
    reranker.reset_truncation_stats()
    rss_after = round(rss_mb(), 2)

    in_indices = {
        i for i, it in enumerate(items) if it.question_type != QuestionType.OUT_OF_SCOPE
    }

    doc_mrr_5: list[float] = []
    chunk_mrr_5: list[float] = []
    doc_r3_5: list[float] = []
    doc_ndcg3_5: list[float] = []
    doc_mrr_6: list[float] = []
    chunk_mrr_6: list[float] = []
    lat_ms: list[float] = []

    top1_cids: list[str] = []
    top3_sets: list[set[str]] = []
    flat_logits: list[float] = []
    top1_logits: list[float] = []

    for row in ckpt_rows:
        idx = int(row["idx"])
        item = items[idx]
        ckpt_scored = sorted(
            row["scored_8"],
            key=lambda x: (float(x["rerank_raw_logit"]), float(x["rerank_score"])),
            reverse=True,
        )
        cands = [store.get(str(c["chunk_id"])) for c in ckpt_scored]
        t0 = time.perf_counter()
        scored = reranker.rerank(item.query, cands, top_k=3, return_all=True)
        lat_ms.append((time.perf_counter() - t0) * 1000.0)

        new_cids = [str(c["chunk_id"]) for c in scored]
        top1_cids.append(new_cids[0])
        top3_sets.append(set(new_cids[:3]))
        top1_logits.append(float(scored[0]["rerank_raw_logit"]))

        by_cid_new = {str(c["chunk_id"]): float(c["rerank_raw_logit"]) for c in scored}
        # Align flat_logits identically to ckpt_scored order for pairwise correlation & MAE
        for c in ckpt_scored:
            flat_logits.append(by_cid_new[str(c["chunk_id"])])

        if idx in in_indices:
            c3 = new_cids[:3]
            d3 = dedupe_to_docs(c3, c2doc)
            c5 = new_cids[:5]
            d5 = dedupe_to_docs(c5, c2doc)

            doc_mrr_5.append(reciprocal_rank(d3, item.expected_doc_ids))
            chunk_mrr_5.append(reciprocal_rank(c3, item.expected_source_chunk_ids))
            doc_r3_5.append(recall_at_k(d3, item.expected_doc_ids, 3))
            doc_ndcg3_5.append(ndcg_at_k(d3, item.expected_doc_ids, 3))
            doc_mrr_6.append(reciprocal_rank(d5, item.expected_doc_ids))
            chunk_mrr_6.append(reciprocal_rank(c5, item.expected_source_chunk_ids))

    trunc_stats = reranker.truncation_stats
    del reranker
    gc.collect()

    top1_arr = np.asarray(top1_logits, dtype=np.float64)
    return {
        "precision": precision,
        "max_length": max_length,
        "rerank_long_query_max_length": long_query_max_length,
        "contingency_l_applied": contingency_l_applied,
        "rss_mb": rss_after,
        "truncation_stats": trunc_stats,
        "latency_ms": _stats_ms(lat_ms),
        "top1_logit_mean": round(float(np.mean(top1_arr)), 5),
        "top1_logit_median": round(float(np.median(top1_arr)), 5),
        "run5_doc_mrr": round(float(np.mean(doc_mrr_5)), 4),
        "run5_chunk_mrr": round(float(np.mean(chunk_mrr_5)), 4),
        "run5_doc_recall_at_3": round(float(np.mean(doc_r3_5)), 4),
        "run5_doc_ndcg_at_3": round(float(np.mean(doc_ndcg3_5)), 4),
        "run6_doc_mrr": round(float(np.mean(doc_mrr_6)), 4),
        "run6_chunk_mrr": round(float(np.mean(chunk_mrr_6)), 4),
        "_raw_top1_cids": top1_cids,
        "_raw_top3_sets": [sorted(s) for s in top3_sets],
        "_raw_logits": flat_logits,
    }


def _build_reranker_comparison(
    variant: dict[str, Any],
    ref: dict[str, Any],
) -> dict[str, Any]:
    """Compute exact pairwise drift metrics comparing *variant* against *ref*."""
    v_top1 = variant["_raw_top1_cids"]
    v_top3 = [set(s) for s in variant["_raw_top3_sets"]]
    v_logits = np.asarray(variant["_raw_logits"], dtype=np.float64)

    r_top1 = ref["_raw_top1_cids"]
    r_top3 = [set(s) for s in ref["_raw_top3_sets"]]
    r_logits = np.asarray(ref["_raw_logits"], dtype=np.float64)

    diffs = np.abs(v_logits - r_logits)
    corr = float(np.corrcoef(v_logits, r_logits)[0, 1])

    return {
        "doc_mrr_delta": round(float(variant["run5_doc_mrr"]) - float(ref["run5_doc_mrr"]), 4),
        "chunk_mrr_delta": round(float(variant["run5_chunk_mrr"]) - float(ref["run5_chunk_mrr"]), 4),
        "top1_agreement": round(
            sum(a == b for a, b in zip(v_top1, r_top1)) / len(r_top1), 4
        ),
        "top3_overlap": round(
            float(np.mean([len(a & b) / 3.0 for a, b in zip(v_top3, r_top3)])), 4
        ),
        "top3_exact_match_rate": round(
            sum(a == b for a, b in zip(v_top3, r_top3)) / len(r_top3), 4
        ),
        "logit_mae": round(float(np.mean(diffs)), 5),
        "logit_max_diff": round(float(np.max(diffs)), 5),
        "logit_corr": round(corr, 6),
    }


def _comparison_passes(cmp_dict: dict[str, Any]) -> bool:
    """Enforce strict Stage 3 pairwise drift thresholds (spec top3_overlap >= 0.98)."""
    return bool(
        abs(float(cmp_dict["doc_mrr_delta"])) <= 0.01
        and abs(float(cmp_dict["chunk_mrr_delta"])) <= 0.01
        and float(cmp_dict["top3_overlap"]) >= 0.98
        and float(cmp_dict["logit_corr"]) >= 0.99
    )


def _strip_raw_reranker_arrays(variant: dict[str, Any]) -> dict[str, Any]:
    """Remove internal raw arrays from a reranker variant dict before JSON export."""
    cleaned = dict(variant)
    cleaned.pop("_raw_top1_cids", None)
    cleaned.pop("_raw_top3_sets", None)
    cleaned.pop("_raw_logits", None)
    return cleaned


def _evaluate_reranker_gate(settings: Settings) -> dict[str, Any]:
    """Gate Tier 4 Reranker with full R_cur, fp32@1024, fp32@576, and int8@576 attribution."""
    items = _load_golden_items(settings)
    store = ChunkStore(settings.data_processed / "corpus_chunks.parquet")
    c2doc = store.chunk_to_doc_map()
    ckpt_rows = _load_rerank_checkpoint_rows(settings)

    ab_data = _load_json(settings.reports_dir / "retrieval_ablation_results.json")
    r5_base = next(r for r in ab_data["runs"] if r["run_id"] == "Run 5")
    r6_base = next(r for r in ab_data["runs"] if r["run_id"] == "Run 6")
    base_r5_doc_mrr = float(r5_base["doc_metrics"]["overall"]["mrr"])
    base_r5_chunk_mrr = float(r5_base["chunk_metrics"]["overall"]["mrr"])
    base_r6_doc_mrr = float(r6_base["doc_metrics"]["overall"]["mrr"])

    ckpt_ref = _build_ckpt_rerank_reference(items, ckpt_rows, c2doc)

    log.info("Evaluating Reranker reference fp32@1024 ...")
    ref_1024 = _score_reranker_variant(
        settings, items, store, c2doc, ckpt_rows, precision="fp32", max_length=1024
    )
    ref_vs_ckpt = _build_reranker_comparison(ref_1024, ckpt_ref)
    ref_1024["vs_ckpt"] = ref_vs_ckpt

    doc_mrr_delta_vs_ckpt = round(float(ref_1024["run5_doc_mrr"]) - base_r5_doc_mrr, 4)
    chunk_mrr_delta_vs_ckpt = round(float(ref_1024["run5_chunk_mrr"]) - base_r5_chunk_mrr, 4)
    baseline_drift_detected = bool(abs(doc_mrr_delta_vs_ckpt) > 0.01)
    active_reference = "fp32@1024" if baseline_drift_detected else "R_cur"
    if baseline_drift_detected:
        log.warning(
            "BASELINE DRIFT DETECTED: CPU fp32@1024 doc_mrr=%.4f vs ckpt=%.4f (delta=%.4f). "
            "Promoting fresh CPU fp32@1024 as active reference!",
            ref_1024["run5_doc_mrr"],
            base_r5_doc_mrr,
            doc_mrr_delta_vs_ckpt,
        )

    baseline_drift = {
        "ckpt_run5_doc_mrr": base_r5_doc_mrr,
        "fp32_1024_run5_doc_mrr": ref_1024["run5_doc_mrr"],
        "doc_mrr_delta_vs_ckpt": doc_mrr_delta_vs_ckpt,
        "ckpt_run5_chunk_mrr": base_r5_chunk_mrr,
        "fp32_1024_run5_chunk_mrr": ref_1024["run5_chunk_mrr"],
        "chunk_mrr_delta_vs_ckpt": chunk_mrr_delta_vs_ckpt,
        "ckpt_run6_doc_mrr": base_r6_doc_mrr,
        "fp32_1024_run6_doc_mrr": ref_1024["run6_doc_mrr"],
        "baseline_drift": baseline_drift_detected,
        "active_reference": active_reference,
        "vs_ckpt": ref_vs_ckpt,
    }

    def _base_metrics_ok(v: dict[str, Any]) -> bool:
        return bool(
            v["truncation_stats"]["passages_truncated"] == 0
            and float(v["run5_doc_mrr"]) >= 0.85
        )

    variants_raw: dict[str, dict[str, Any]] = {}

    log.info("Evaluating Reranker fp32@576 ...")
    v_fp32_576 = _score_reranker_variant(
        settings, items, store, c2doc, ckpt_rows, precision="fp32", max_length=576
    )
    v_fp32_576["vs_ckpt"] = _build_reranker_comparison(v_fp32_576, ckpt_ref)
    v_fp32_576["vs_fp32_1024"] = _build_reranker_comparison(v_fp32_576, ref_1024)
    v_fp32_576["passed"] = bool(
        _base_metrics_ok(v_fp32_576)
        and _comparison_passes(v_fp32_576["vs_fp32_1024"])
        and (baseline_drift_detected or _comparison_passes(v_fp32_576["vs_ckpt"]))
    )
    variants_raw["fp32@576"] = v_fp32_576

    contingency_l_triggered = False
    contingency_l_latency_delta_ms: float | None = None
    contingency_l_latency_delta_pct: float | None = None
    active_fp32_576 = v_fp32_576

    if (
        not v_fp32_576["passed"]
        and v_fp32_576["truncation_stats"]["queries_truncated"] > 0
    ):
        contingency_l_triggered = True
        log.warning(
            "fp32@576 failed with queries_truncated=%d; triggering Contingency L (768 for >96 q tokens)!",
            v_fp32_576["truncation_stats"]["queries_truncated"],
        )
        v_fp32_cont = _score_reranker_variant(
            settings,
            items,
            store,
            c2doc,
            ckpt_rows,
            precision="fp32",
            max_length=576,
            long_query_max_length=768,
            contingency_l_applied=True,
        )
        v_fp32_cont["vs_ckpt"] = _build_reranker_comparison(v_fp32_cont, ckpt_ref)
        v_fp32_cont["vs_fp32_1024"] = _build_reranker_comparison(v_fp32_cont, ref_1024)
        base_lat = float(v_fp32_576["latency_ms"]["mean"] or 0.0)
        cont_lat = float(v_fp32_cont["latency_ms"]["mean"] or 0.0)
        contingency_l_latency_delta_ms = round(cont_lat - base_lat, 3)
        contingency_l_latency_delta_pct = (
            round((cont_lat - base_lat) / base_lat * 100.0, 2) if base_lat > 0 else None
        )
        v_fp32_cont["contingency_l_latency_delta_ms"] = contingency_l_latency_delta_ms
        v_fp32_cont["contingency_l_latency_delta_pct"] = contingency_l_latency_delta_pct
        v_fp32_cont["passed"] = bool(
            _base_metrics_ok(v_fp32_cont)
            and _comparison_passes(v_fp32_cont["vs_fp32_1024"])
            and (baseline_drift_detected or _comparison_passes(v_fp32_cont["vs_ckpt"]))
        )
        variants_raw["fp32@576_contingency_l_768"] = v_fp32_cont
        active_fp32_576 = v_fp32_cont

    log.info("Evaluating Reranker int8@576 ...")
    v_int8_576 = _score_reranker_variant(
        settings,
        items,
        store,
        c2doc,
        ckpt_rows,
        precision="int8",
        max_length=576,
        long_query_max_length=768 if contingency_l_triggered else None,
        contingency_l_applied=contingency_l_triggered,
    )
    v_int8_576["vs_ckpt"] = _build_reranker_comparison(v_int8_576, ckpt_ref)
    v_int8_576["vs_fp32_1024"] = _build_reranker_comparison(v_int8_576, ref_1024)
    v_int8_576["vs_fp32_576"] = _build_reranker_comparison(v_int8_576, active_fp32_576)

    int8_active_ref_cmp = (
        v_int8_576["vs_fp32_1024"] if baseline_drift_detected else v_int8_576["vs_ckpt"]
    )
    v_int8_576["passed"] = bool(
        _base_metrics_ok(v_int8_576)
        and _comparison_passes(int8_active_ref_cmp)
        and _comparison_passes(v_int8_576["vs_fp32_576"])
    )
    int8_key = "int8@576" + ("_contingency_l_768" if contingency_l_triggered else "")
    variants_raw[int8_key] = v_int8_576

    return {
        "baseline_drift": baseline_drift,
        "reference_ckpt": _strip_raw_reranker_arrays(ckpt_ref),
        "reference_fp32_1024": _strip_raw_reranker_arrays(ref_1024),
        "contingency_l_triggered": contingency_l_triggered,
        "contingency_l_latency_delta_ms": contingency_l_latency_delta_ms,
        "contingency_l_latency_delta_pct": contingency_l_latency_delta_pct,
        "variants": {k: _strip_raw_reranker_arrays(v) for k, v in variants_raw.items()},
    }


def _evaluate_router_gate(settings: Settings) -> dict[str, Any]:
    """Gate Tier 1 Router precisions against golden_150.jsonl and held-out synthetic split."""
    eval_path = settings.reports_dir / "router_eval.json"
    baseline_eval = _load_json(eval_path)
    baseline_held_out_acc = float(baseline_eval.get("held_out_accuracy", 1.0))

    variants: dict[str, Any] = {}
    for prec in ("fp32", "int8", "bf16"):
        log.info("Evaluating Router gate for precision=%s ...", prec)
        res = evaluate_router_precision(prec, settings=settings)
        delta_acc = float(res["held_out_accuracy"]) - baseline_held_out_acc
        passed = bool(
            int(res["golden_correct"]) >= 149
            and abs(delta_acc) <= 0.005
        )
        res["baseline_held_out_accuracy"] = baseline_held_out_acc
        res["held_out_delta_pp"] = round(delta_acc * 100.0, 4)
        res["passed"] = passed
        variants[prec] = res
        gc.collect()

    return {
        "baseline_held_out_accuracy": baseline_held_out_acc,
        "variants": variants,
    }


def _rss_probe_worker(
    embed_precision: str,
    rerank_precision: str,
    router_precision: str,
    rerank_max_length: int,
    rerank_long_query_max_length: int | None,
    contingency_l_applied: bool,
    cpu_affinity_mode: str,
) -> dict[str, Any]:
    """Isolated worker measuring full 3-model warmed resident RSS."""
    apply_hardware_guard()
    cfg = Settings(
        embed_precision=embed_precision,  # type: ignore[arg-type]
        rerank_precision=rerank_precision,  # type: ignore[arg-type]
        router_precision=router_precision,  # type: ignore[arg-type]
        rerank_max_length=rerank_max_length,
        rerank_long_query_max_length=rerank_long_query_max_length,
        contingency_l_applied=contingency_l_applied,
        cpu_affinity_mode=cpu_affinity_mode,  # type: ignore[arg-type]
        abstention_enabled=False,
    )
    pipe = VectorRetrievalPipeline(cfg)
    return {
        "rss_after_load_mb": pipe.rss_after_load_mb,
        "within_rss_ceiling": pipe.within_rss_ceiling,
        "rss_ceiling_mb": cfg.rss_ceiling_mb,
        "vram_probe": dict(vram_probe()),
    }


def _measure_resident_rss_subprocess(
    embed_precision: str,
    rerank_precision: str,
    router_precision: str,
    rerank_max_length: int,
    rerank_long_query_max_length: int | None,
    contingency_l_applied: bool,
    cpu_affinity_mode: str,
) -> dict[str, Any]:
    """Spawn a clean subprocess to measure 3-model resident RSS without heap carryover."""
    args = [
        "--_rss-worker",
        "--embed-precision",
        embed_precision,
        "--rerank-precision",
        rerank_precision,
        "--router-precision",
        router_precision,
        "--max-length",
        str(rerank_max_length),
        "--mode",
        cpu_affinity_mode,
    ]
    if rerank_long_query_max_length is not None:
        args.extend(["--long-max-length", str(rerank_long_query_max_length)])
    if contingency_l_applied:
        args.append("--contingency-l")
    return _run_subprocess_json(
        args,
        extra_env={"KAVACH_CPU_AFFINITY_MODE": cpu_affinity_mode},
    )


def _rank_passing_by_matrix_latency(
    tier: str,
    passing: list[str],
    matrix_tiers: dict[str, Any],
) -> list[str]:
    """Order passing precisions ascending by Stage 2 --matrix mean latency."""
    tier_cells = matrix_tiers.get(tier)
    if not isinstance(tier_cells, dict) or not tier_cells:
        raise RuntimeError(
            f"Missing Stage 2 --matrix results for tier={tier!r} in reports/phase15_profile.json; "
            "run 'python -m kavach.retrieval.profiling --matrix' before '--gates'."
        )
    scored_items: list[tuple[float, str]] = []
    for prec in passing:
        cell_key = f"{prec}@576" if tier == "reranker" else prec
        cell = tier_cells.get(cell_key)
        mean_ms = (
            cell.get("latency_ms", {}).get("mean")
            if isinstance(cell, dict)
            else None
        )
        if not isinstance(mean_ms, (int, float)):
            raise RuntimeError(
                f"Missing Stage 2 --matrix mean latency for {tier}/{cell_key!r}; "
                "re-run 'python -m kavach.retrieval.profiling --matrix'."
            )
        scored_items.append((float(mean_ms), prec))
    scored_items.sort(key=lambda item: item[0])
    return [prec for _, prec in scored_items]


def run_gates(settings: Settings) -> dict[str, Any]:
    """Execute Stage 3 quality gates, select precision lock under 4096 MB RSS, and persist."""
    log.info("=== Stage 3: Quality Gates & Precision Lock Selection ===")
    profile = _load_json(settings.reports_dir / "phase15_profile.json")
    matrix_tiers = profile.get("matrix", {}).get("tiers")
    if not isinstance(matrix_tiers, dict) or not matrix_tiers:
        raise RuntimeError(
            "Stage 3 --gates requires Stage 2 --matrix results in reports/phase15_profile.json."
        )

    embed_gate = _evaluate_embedder_gate(settings)
    rerank_gate = _evaluate_reranker_gate(settings)
    router_gate = _evaluate_router_gate(settings)

    passing_embed = [
        p for p in ("fp32", "bf16", "int8")
        if embed_gate["variants"][p]["passed"]
    ]
    passing_router = [
        p for p in ("fp32", "bf16", "int8")
        if router_gate["variants"][p]["passed"]
    ]

    cont_l = bool(rerank_gate["contingency_l_triggered"])
    long_max_len = 768 if cont_l else None
    rr_fp32_key = "fp32@576_contingency_l_768" if cont_l else "fp32@576"
    rr_int8_key = "int8@576_contingency_l_768" if cont_l else "int8@576"

    passing_rerank: list[str] = []
    if rerank_gate["variants"].get(rr_fp32_key, {}).get("passed"):
        passing_rerank.append("fp32")
    if rerank_gate["variants"].get(rr_int8_key, {}).get("passed"):
        passing_rerank.append("int8")

    gates_report: dict[str, Any] = {
        "generated_at": datetime.now(tz=timezone.utc).isoformat(),
        "rss_ceiling_mb": float(settings.rss_ceiling_mb),
        "embedder_gate": embed_gate,
        "reranker_gate": rerank_gate,
        "router_gate": router_gate,
        "passing_candidates": {
            "embedder": passing_embed,
            "reranker": passing_rerank,
            "router": passing_router,
        },
        "gate_passed": False,
        "vram_probe": dict(vram_probe()),
    }

    if not passing_embed or not passing_rerank or not passing_router:
        _atomic_write_json(settings.reports_dir / "phase15_quality_gates.json", gates_report)
        raise RuntimeError(
            f"Quality gate failure: passing_candidates={gates_report['passing_candidates']}"
        )

    ranked_embed = _rank_passing_by_matrix_latency("embedder", passing_embed, matrix_tiers)
    ranked_rerank = _rank_passing_by_matrix_latency("reranker", passing_rerank, matrix_tiers)
    ranked_router = _rank_passing_by_matrix_latency("router", passing_router, matrix_tiers)
    gates_report["latency_ranked_passing_candidates"] = {
        "embedder": ranked_embed,
        "reranker": ranked_rerank,
        "router": ranked_router,
    }

    aff_mode = str(
        profile.get("affinity", {}).get("recommended_affinity_mode") or "off"
    )

    sel_embed = ranked_embed[0]
    sel_rerank = ranked_rerank[0]
    sel_router = ranked_router[0]
    downgrades_applied: list[str] = []
    rss_checks: list[dict[str, Any]] = []

    def _probe_current() -> dict[str, Any]:
        probe = _measure_resident_rss_subprocess(
            embed_precision=sel_embed,
            rerank_precision=sel_rerank,
            router_precision=sel_router,
            rerank_max_length=576,
            rerank_long_query_max_length=long_max_len,
            contingency_l_applied=cont_l,
            cpu_affinity_mode=aff_mode,
        )
        probe["config"] = {
            "embed_precision": sel_embed,
            "rerank_precision": sel_rerank,
            "router_precision": sel_router,
        }
        rss_checks.append(probe)
        return probe

    probe = _probe_current()
    # Ordered downgrade strictly over passing candidates: embedder->int8, reranker->int8, router->int8
    if not probe["within_rss_ceiling"] and sel_embed != "int8" and "int8" in passing_embed:
        sel_embed = "int8"
        downgrades_applied.append("embedder->int8")
        probe = _probe_current()

    if not probe["within_rss_ceiling"] and sel_rerank != "int8" and "int8" in passing_rerank:
        sel_rerank = "int8"
        downgrades_applied.append("reranker->int8")
        probe = _probe_current()

    if not probe["within_rss_ceiling"] and sel_router != "int8" and "int8" in passing_router:
        sel_router = "int8"
        downgrades_applied.append("router->int8")
        probe = _probe_current()

    if not probe["within_rss_ceiling"]:
        gates_report["rss_selection"] = {
            "checks": rss_checks,
            "downgrades_applied": downgrades_applied,
        }
        _atomic_write_json(settings.reports_dir / "phase15_quality_gates.json", gates_report)
        raise RuntimeError(
            f"Resident RSS {probe['rss_after_load_mb']} MB exceeds ceiling "
            f"{settings.rss_ceiling_mb} MB even after eligible downgrades {downgrades_applied}."
        )

    quant_engine = get_quantized_engine_info()["engine"]
    locked_at = datetime.now(tz=timezone.utc).isoformat()
    precision_lock = {
        "embed_precision": sel_embed,
        "rerank_precision": sel_rerank,
        "router_precision": sel_router,
        "rerank_max_length": 576,
        "rerank_long_query_max_length": long_max_len,
        "rerank_long_query_token_threshold": int(settings.rerank_long_query_token_threshold),
        "contingency_l_applied": cont_l,
        "cpu_affinity_mode": aff_mode,
        "quantized_engine": quant_engine,
        "torch_version": str(torch.__version__),
        "locked_at": locked_at,
        "gate_report_path": "reports/phase15_quality_gates.json",
        "resident_rss_mb": probe["rss_after_load_mb"],
        "rss_ceiling_mb": float(settings.rss_ceiling_mb),
        "downgrades_applied": downgrades_applied,
        "vram_probe": dict(vram_probe()),
    }
    _atomic_write_json(settings.models_dir / "precision_lock.json", precision_lock)

    gates_report["gate_passed"] = True
    gates_report["rss_selection"] = {
        "checks": rss_checks,
        "downgrades_applied": downgrades_applied,
    }
    gates_report["precision_lock"] = precision_lock
    _atomic_write_json(settings.reports_dir / "phase15_quality_gates.json", gates_report)
    return gates_report


# ─────────────────────────────────────────────────────────────────────────────
# Stage 5: --baseline
# ─────────────────────────────────────────────────────────────────────────────


def run_baseline(settings: Settings) -> dict[str, Any]:
    """Measure warm resident pipeline latency & RSS baseline over all 150 golden queries."""
    log.info("=== Stage 5: Resident Pipeline Latency & RSS Baseline (All 150 Golden Queries) ===")
    idle = _wait_for_cpu_idle()
    items = _load_golden_items(settings)
    if len(items) != 150:
        raise RuntimeError(f"Expected 150 golden queries for Stage 5 baseline, found {len(items)}")

    pipeline = VectorRetrievalPipeline(settings)

    records: list[dict[str, Any]] = []
    lat_by_route: dict[str, list[float]] = {
        "FAST_PATH_EXACT_LOOKUP": [],
        "HARD_REFUSAL": [],
        "SEMANTIC_RAG": [],
    }
    lat_by_expected_route: dict[str, dict[str, list[float]]] = {}
    lat_by_qtype: dict[str, dict[str, list[float]]] = {
        qt.value: {
            "router_ms": [],
            "lookup_ms": [],
            "embed_ms": [],
            "search_ms": [],
            "rerank_ms": [],
            "total_ms": [],
        }
        for qt in QuestionType
    }
    stage_lats: dict[str, list[float]] = {
        "router_ms": [],
        "lookup_ms": [],
        "embed_ms": [],
        "search_ms": [],
        "rerank_ms": [],
        "total_ms": [],
    }
    peak_rss = pipeline.rss_after_load_mb

    for item in items:
        res = pipeline.retrieve(
            item.query,
            rerank=True,
            top_k=settings.rerank_top_k_default,
            candidate_k=settings.candidate_k,
        )
        cur_rss = rss_mb()
        peak_rss = max(peak_rss, cur_rss)

        total_ms_val = float(res.latency_ms["total_ms"])
        lat_by_route.setdefault(res.route.route, []).append(total_ms_val)

        exp_bucket = lat_by_expected_route.setdefault(
            item.expected_route,
            {
                "router_ms": [],
                "lookup_ms": [],
                "embed_ms": [],
                "search_ms": [],
                "rerank_ms": [],
                "total_ms": [],
            },
        )
        qt_bucket = lat_by_qtype[item.question_type.value]
        for k, v in res.latency_ms.items():
            fv = float(v)
            if k in stage_lats:
                stage_lats[k].append(fv)
                qt_bucket[k].append(fv)
                exp_bucket[k].append(fv)

        records.append(
            {
                "test_id": item.test_id,
                "question_type": item.question_type.value,
                "expected_route": item.expected_route,
                "predicted_route": res.route.route,
                "refused": res.refused,
                "abstained": res.abstained,
                "abstain_reason": res.abstain_reason,
                "top1_logit": res.top1_logit,
                "contexts_count": len(res.contexts),
                "latency_ms": res.latency_ms,
            }
        )

    baseline = {
        "generated_at": datetime.now(tz=timezone.utc).isoformat(),
        "idle_check": idle,
        "cpu_affinity": sorted(psutil.Process().cpu_affinity()),
        "active_settings": {
            "embed_precision": settings.embed_precision,
            "rerank_precision": settings.rerank_precision,
            "router_precision": settings.router_precision,
            "rerank_max_length": settings.rerank_max_length,
            "rerank_long_query_max_length": settings.rerank_long_query_max_length,
            "contingency_l_applied": settings.contingency_l_applied,
            "cpu_affinity_mode": settings.cpu_affinity_mode,
            "abstention_enabled": settings.abstention_enabled,
            "abstain_logit_threshold": settings.abstain_logit_threshold,
            "candidate_k": settings.candidate_k,
        },
        "rss": {
            "rss_after_load_mb": pipeline.rss_after_load_mb,
            "rss_after_baseline_mb": round(rss_mb(), 2),
            "peak_rss_mb": round(peak_rss, 2),
            "rss_ceiling_mb": float(settings.rss_ceiling_mb),
            "within_ceiling": peak_rss <= float(settings.rss_ceiling_mb),
        },
        "truncation_stats": (
            pipeline._reranker.truncation_stats if pipeline._reranker is not None else {}
        ),
        "overall_total_latency_ms": _stats_ms(stage_lats["total_ms"]),
        "stage_latency_ms": {
            stage: _stats_ms(vals) for stage, vals in stage_lats.items()
        },
        "route_total_latency_ms": {
            route: _stats_ms(vals) for route, vals in lat_by_route.items()
        },
        "by_expected_route_latency_ms": {
            route: {
                "count": len(buckets["total_ms"]),
                "total_ms": _stats_ms(buckets["total_ms"]),
                "stages": {
                    stage: _stats_ms(vals) for stage, vals in buckets.items()
                },
            }
            for route, buckets in lat_by_expected_route.items()
        },
        "by_question_type_latency_ms": {
            qtype: {
                "count": len(buckets["total_ms"]),
                "total_ms": _stats_ms(buckets["total_ms"]),
                "stages": {
                    stage: _stats_ms(vals) for stage, vals in buckets.items()
                },
            }
            for qtype, buckets in lat_by_qtype.items()
        },
        "query_count": len(records),
        "queries": records,
        "vram_probe": dict(vram_probe()),
    }
    _atomic_write_json(settings.reports_dir / "phase15_latency_baseline.json", baseline)
    return baseline


# ─────────────────────────────────────────────────────────────────────────────
# CLI Entry Point
# ─────────────────────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> None:
    """Run Phase 1.5 CPU profiling, quality gates, or baseline measurement."""
    parser = argparse.ArgumentParser(
        description="KAVACH-RAG Phase 1.5 profiling, quality gates, and baseline CLI."
    )
    parser.add_argument("--audit", action="store_true", help="Run Stage 2A topology & token audit.")
    parser.add_argument("--affinity", action="store_true", help="Run Stage 2B CPU P-core affinity benchmark.")
    parser.add_argument("--matrix", action="store_true", help="Run Stage 2C precision x latency x RSS matrix.")
    parser.add_argument("--gates", action="store_true", help="Run Stage 3 quality gates & write precision_lock.json.")
    parser.add_argument("--baseline", action="store_true", help="Run Stage 5 warm resident latency baseline.")

    # Internal isolated worker flags (invoked while parent holds BenchLock)
    parser.add_argument("--_audit-model-worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--_affinity-worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--_matrix-worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--_rss-worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--mode", default="off", help=argparse.SUPPRESS)
    parser.add_argument("--tier", default="", help=argparse.SUPPRESS)
    parser.add_argument("--precision", default="fp32", help=argparse.SUPPRESS)
    parser.add_argument("--max-length", type=int, default=576, help=argparse.SUPPRESS)
    parser.add_argument("--long-max-length", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--contingency-l", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--embed-precision", default="fp32", help=argparse.SUPPRESS)
    parser.add_argument("--rerank-precision", default="fp32", help=argparse.SUPPRESS)
    parser.add_argument("--router-precision", default="fp32", help=argparse.SUPPRESS)

    args = parser.parse_args(argv)

    if args._audit_model_worker:
        print(json.dumps(_audit_model_worker(args.tier)), flush=True)
        return
    if args._affinity_worker:
        print(json.dumps(_affinity_worker(args.mode)), flush=True)
        return
    if args._matrix_worker:
        print(json.dumps(_matrix_worker(args.tier, args.precision, args.max_length)), flush=True)
        return
    if args._rss_worker:
        print(
            json.dumps(
                _rss_probe_worker(
                    embed_precision=args.embed_precision,
                    rerank_precision=args.rerank_precision,
                    router_precision=args.router_precision,
                    rerank_max_length=args.max_length,
                    rerank_long_query_max_length=args.long_max_length,
                    contingency_l_applied=bool(args.contingency_l),
                    cpu_affinity_mode=args.mode,
                )
            ),
            flush=True,
        )
        return

    if not any((args.audit, args.affinity, args.matrix, args.gates, args.baseline)):
        parser.error("Specify one of: --audit, --affinity, --matrix, --gates, --baseline")

    apply_hardware_guard()
    settings = get_settings()

    with BenchLock(owner="profiling"):
        if args.audit:
            run_audit(settings)
        if args.affinity:
            run_affinity(settings)
        if args.matrix:
            run_matrix(settings)
        if args.gates:
            run_gates(settings)
        if args.baseline:
            run_baseline(settings)


if __name__ == "__main__":
    main()
