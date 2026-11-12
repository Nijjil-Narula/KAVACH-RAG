"""Phase 3 server preflight: validates router LFS, Ollama GPU residency, server lifecycle, live smoke, and clean shutdown."""
from __future__ import annotations

import argparse
import asyncio
import difflib
import json
import math
import os
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

from kavach.generation.gen_config import GenSettings, get_gen_settings
from kavach.generation.prompts import canonical_refusal

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_DEFAULT_OUTPUT = _PROJECT_ROOT / "reports" / "phase3_server_preflight.json"
_BASELINE_MANIFEST = _PROJECT_ROOT / "reports" / "phase3_regression_baseline.json"
_INSCOPE_DEV = _PROJECT_ROOT / "data" / "golden" / "inscope_dev_100.jsonl"
_ROUTER_WEIGHTS = _PROJECT_ROOT / "models" / "router_deberta" / "model.safetensors"
_START_OLLAMA_PS1 = _PROJECT_ROOT / "scripts" / "start_ollama.ps1"
_START_SERVER_PS1 = _PROJECT_ROOT / "scripts" / "start_server.ps1"
_SERVER_PID_FILE = _PROJECT_ROOT / "logs" / ".server.pid"
_SERVER_LAUNCHER_LOG = _PROJECT_ROOT / "logs" / "start_server.log"
_SERVER_ERR_LOG = _PROJECT_ROOT / "logs" / "phase3_server_err.log"
_OLLAMA_ERR_LOG = _PROJECT_ROOT / "logs" / "ollama_server.err.log"

_MIN_LFS_BYTES = 104_857_600
_SERVER_BASE_URL = "http://127.0.0.1:8000"
_MAX_RSS_MB = 4096.0
_HTTP_TIMEOUT = httpx.Timeout(120.0, connect=10.0)


def _atomic_write_json(target: Path, payload: dict[str, Any]) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    fd, tmp_name = tempfile.mkstemp(
        dir=str(target.parent),
        prefix=f".{target.name}.",
        suffix=".tmp",
    )
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(serialized)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_path, target)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise


def _clean_gpu_env() -> dict[str, str]:
    env = os.environ.copy()
    for key in (
        "CUDA_VISIBLE_DEVICES",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
    ):
        env.pop(key, None)
    return env


def _probe_nvidia_smi() -> dict[str, Any] | None:
    try:
        proc = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=5.0,
            check=False,
        )
    except (FileNotFoundError, OSError, subprocess.TimeoutExpired):
        return None
    if proc.returncode != 0:
        return None
    for line in proc.stdout.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 4:
            continue
        try:
            return {
                "gpu_name": parts[0],
                "driver_version": parts[1],
                "memory_used_mib": int(float(parts[2])),
                "memory_total_mib": int(float(parts[3])),
            }
        except ValueError:
            continue
    return None


def _probe_gpu_compute_apps() -> list[dict[str, Any]]:
    try:
        proc = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,process_name",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=5.0,
            check=False,
        )
    except (FileNotFoundError, OSError, subprocess.TimeoutExpired):
        return []
    if proc.returncode != 0:
        return []
    apps: list[dict[str, Any]] = []
    for raw in proc.stdout.splitlines():
        line = raw.strip()
        if not line or "no running processes found" in line.lower():
            continue
        parts = [p.strip() for p in line.split(",", 1)]
        pid_val: int | str = parts[0]
        try:
            pid_val = int(parts[0])
        except ValueError:
            pid_val = parts[0]
        apps.append({"pid": pid_val, "process_name": parts[1] if len(parts) > 1 else ""})
    return apps


def _percentile(values: list[float], pct: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("Cannot compute percentile of empty sample list")
    if len(ordered) == 1:
        return float(ordered[0])
    pos = (pct / 100.0) * (len(ordered) - 1)
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return float(ordered[lo])
    frac = pos - lo
    return float(ordered[lo] * (1.0 - frac) + ordered[hi] * frac)


def _read_log_tail(path: Path, max_lines: int = 30) -> str:
    if not path.is_file():
        return ""
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(lines[-max_lines:])


def _run_ps_script(
    script_path: Path,
    args: list[str],
    *,
    env: dict[str, str] | None = None,
    timeout_s: float = 260.0,
    log_path: Path | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run a PowerShell script without pipe inheritance so background children never block EOF."""
    cmd = [
        "powershell",
        "-NoProfile",
        "-ExecutionPolicy",
        "Bypass",
        "-File",
        str(script_path),
        *args,
    ]
    proc = subprocess.run(
        cmd,
        cwd=str(_PROJECT_ROOT),
        env=env,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        close_fds=True,
        timeout=timeout_s,
        check=False,
    )
    tail = _read_log_tail(log_path) if log_path is not None else ""
    return subprocess.CompletedProcess(cmd, proc.returncode, stdout="", stderr=tail)


def _load_inscope_queries(limit: int = 3) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with _INSCOPE_DEV.open("r", encoding="utf-8") as fh:
        for line in fh:
            stripped = line.strip()
            if not stripped:
                continue
            rows.append(json.loads(stripped))
            if len(rows) >= limit:
                break
    if len(rows) < limit:
        raise RuntimeError(f"Expected at least {limit} rows in {_INSCOPE_DEV}, found {len(rows)}")
    return rows


def _parse_sse_stream(raw_lines: list[str]) -> list[tuple[str, dict[str, Any]]]:
    frames: list[tuple[str, dict[str, Any]]] = []
    event = "message"
    data_lines: list[str] = []
    for line in raw_lines:
        if line == "":
            if data_lines:
                raw = "\n".join(data_lines)
                frames.append((event, json.loads(raw, strict=False)))
                event = "message"
                data_lines = []
        elif line.startswith("event:"):
            event = line[6:].strip()
        elif line.startswith("data:"):
            data_lines.append(line[5:].lstrip(" "))
    if data_lines:
        raw = "\n".join(data_lines)
        frames.append((event, json.loads(raw, strict=False)))
    return frames


def _gate1_lfs() -> dict[str, Any]:
    exists = _ROUTER_WEIGHTS.is_file()
    size_bytes = _ROUTER_WEIGHTS.stat().st_size if exists else 0
    passed = exists and size_bytes > _MIN_LFS_BYTES
    return {
        "path": str(_ROUTER_WEIGHTS.relative_to(_PROJECT_ROOT).as_posix()),
        "exists": exists,
        "bytes": size_bytes,
        "size_mb": round(size_bytes / (1024.0 * 1024.0), 2),
        "min_required_bytes": _MIN_LFS_BYTES,
        "passed": passed,
    }


def _gate2_ollama(gen: GenSettings) -> dict[str, Any]:
    base = gen.ollama_base_url.rstrip("/")
    restarted = False
    version_payload: dict[str, Any] | None = None

    with httpx.Client(timeout=_HTTP_TIMEOUT) as client:
        try:
            r = client.get(f"{base}/api/version")
            if r.status_code == 200:
                version_payload = r.json()
        except httpx.HTTPError:
            version_payload = None

    if version_payload is None:
        restarted = True
        proc = _run_ps_script(
            _START_OLLAMA_PS1,
            ["-Restart"],
            env=_clean_gpu_env(),
            timeout_s=60.0,
            log_path=_OLLAMA_ERR_LOG,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"Failed to start Ollama via {_START_OLLAMA_PS1}: rc={proc.returncode} stderr={proc.stderr.strip()}"
            )

    with httpx.Client(timeout=_HTTP_TIMEOUT) as client:
        r_ver = client.get(f"{base}/api/version")
        r_ver.raise_for_status()
        version_payload = r_ver.json()

        r_tags = client.get(f"{base}/api/tags")
        r_tags.raise_for_status()
        tags_data = r_tags.json()
        available_names = [
            str(m.get("name") or m.get("model") or "")
            for m in tags_data.get("models", [])
            if isinstance(m, dict)
        ]
        matched_model: str | None = None
        for candidate in (gen.model, f"{gen.model}:latest"):
            if candidate in available_names:
                matched_model = candidate
                break
        if matched_model is None:
            raise RuntimeError(f"Model {gen.model!r} not found in /api/tags: {available_names}")

        t0 = time.perf_counter()
        r_chat = client.post(
            f"{base}/api/chat",
            json={
                "model": matched_model,
                "messages": [{"role": "user", "content": "Reply with OK."}],
                "stream": False,
                "keep_alive": gen.keep_alive,
                "options": {"num_predict": 4, "temperature": 0.0},
            },
        )
        r_chat.raise_for_status()
        warm_chat_ms = round((time.perf_counter() - t0) * 1000.0, 2)

        r_ps = client.get(f"{base}/api/ps")
        r_ps.raise_for_status()
        ps_models = r_ps.json().get("models", [])
        ps_entry = next(
            (
                entry
                for entry in ps_models
                if isinstance(entry, dict)
                and str(entry.get("name") or entry.get("model") or "").split(":")[0]
                == matched_model.split(":")[0]
            ),
            None,
        )
        if ps_entry is None:
            raise RuntimeError(f"Model {matched_model!r} not present in /api/ps: {ps_models}")
        size = int(ps_entry.get("size") or 0)
        size_vram = int(ps_entry.get("size_vram") or 0)
        vram_ratio = (size_vram / size) if size > 0 else 0.0
        if size <= 0 or size_vram < int(0.99 * size):
            raise RuntimeError(
                f"Expected full GPU residency (size_vram >= 0.99 * size), got size={size}, size_vram={size_vram}"
            )

    gpu_snap = _probe_nvidia_smi()
    if gpu_snap is None:
        raise RuntimeError("nvidia-smi probe failed during Gate 2")
    if gpu_snap["memory_used_mib"] > gen.vram_hard_limit_mib:
        raise RuntimeError(
            f"GPU VRAM {gpu_snap['memory_used_mib']} MiB exceeds hard limit {gen.vram_hard_limit_mib} MiB"
        )

    return {
        "restarted_ollama": restarted,
        "version": version_payload.get("version"),
        "matched_model": matched_model,
        "warm_chat_ms": warm_chat_ms,
        "size": size,
        "size_vram": size_vram,
        "vram_ratio": round(vram_ratio, 6),
        "nvidia_smi": gpu_snap,
        "vram_hard_limit_mib": gen.vram_hard_limit_mib,
        "passed": True,
    }


def _gate3_start_server() -> dict[str, Any]:
    t0 = time.perf_counter()
    proc = _run_ps_script(
        _START_SERVER_PS1,
        ["-Restart"],
        timeout_s=280.0,
        log_path=_SERVER_LAUNCHER_LOG,
    )
    startup_s = round(time.perf_counter() - t0, 3)
    if proc.returncode != 0:
        err_tail = _read_log_tail(_SERVER_ERR_LOG)
        raise RuntimeError(
            f"start_server.ps1 -Restart failed (rc={proc.returncode}): "
            f"launcher={proc.stderr}\nserver_err={err_tail}"
        )

    with httpx.Client(timeout=_HTTP_TIMEOUT) as client:
        r = client.get(f"{_SERVER_BASE_URL}/health")
        if r.status_code != 200:
            raise RuntimeError(f"/health returned HTTP {r.status_code}: {r.text}")
        health = r.json()

    if health.get("status") != "ok":
        raise RuntimeError(f"/health status is {health.get('status')!r}, expected 'ok': {health}")
    if not health.get("pipeline_ready") or not health.get("generator_ready"):
        raise RuntimeError(f"Server readiness flags not True: {health}")
    rss_mb = float(health["rss_mb"])
    if rss_mb > _MAX_RSS_MB:
        raise RuntimeError(f"Server RSS {rss_mb} MiB exceeds ceiling {_MAX_RSS_MB} MiB")

    return {
        "startup_s": startup_s,
        "rss_mb": rss_mb,
        "rss_limit_mb": _MAX_RSS_MB,
        "health": health,
        "passed": True,
    }


async def _gate4_live_smoke(baseline: dict[str, Any]) -> dict[str, Any]:
    inscope_rows = _load_inscope_queries(3)
    q0 = str(inscope_rows[0]["query"])
    q1 = str(inscope_rows[1]["query"])
    q2 = str(inscope_rows[2]["query"])

    async with httpx.AsyncClient(base_url=_SERVER_BASE_URL, timeout=_HTTP_TIMEOUT) as client:
        # 4a: Fast-path exact lookup
        r_4a = await client.post("/rag/async", json={"query": "CVE-2024-3094", "stream": False, "use_cache": True})
        if r_4a.status_code != 200:
            raise RuntimeError(f"4a failed: HTTP {r_4a.status_code} {r_4a.text}")
        d_4a = r_4a.json()
        if d_4a.get("route") != "FAST_PATH_EXACT_LOOKUP" or not str(d_4a.get("answer") or "").strip():
            raise RuntimeError(f"4a assertion failed: {d_4a}")
        sub_4a = {
            "route": d_4a["route"],
            "answer_chars": len(d_4a["answer"]),
            "total_ms": d_4a["total_ms"],
            "passed": True,
        }

        # 4b: First inscope_dev query cold then warm cache hit
        r_4b_cold = await client.post("/rag/async", json={"query": q0, "stream": False, "use_cache": True})
        if r_4b_cold.status_code != 200:
            raise RuntimeError(f"4b cold failed: HTTP {r_4b_cold.status_code} {r_4b_cold.text}")
        d_4b_cold = r_4b_cold.json()
        if d_4b_cold.get("cache_hit") is not False or not str(d_4b_cold.get("answer") or "").strip():
            raise RuntimeError(f"4b cold assertion failed: {d_4b_cold}")

        r_4b_warm = await client.post("/rag/async", json={"query": q0, "stream": False, "use_cache": True})
        if r_4b_warm.status_code != 200:
            raise RuntimeError(f"4b warm failed: HTTP {r_4b_warm.status_code} {r_4b_warm.text}")
        d_4b_warm = r_4b_warm.json()
        if d_4b_warm.get("cache_hit") is not True or float(d_4b_warm["total_ms"]) >= 50.0:
            raise RuntimeError(f"4b warm assertion failed: {d_4b_warm}")
        if d_4b_warm["answer"] != d_4b_cold["answer"]:
            raise RuntimeError("4b warm cache answer diverged from cold answer")
        sub_4b = {
            "dev_id": inscope_rows[0].get("dev_id"),
            "cold": {
                "retrieval_latency_ms": d_4b_cold["retrieval_latency_ms"],
                "ttft_ms": d_4b_cold["ttft_ms"],
                "total_ms": d_4b_cold["total_ms"],
                "cache_hit": d_4b_cold["cache_hit"],
                "retrieved_chunk_ids": d_4b_cold["retrieved_chunk_ids"],
            },
            "warm": {
                "total_ms": d_4b_warm["total_ms"],
                "cache_hit": d_4b_warm["cache_hit"],
                "answer_matches_cold": True,
            },
            "passed": True,
        }

        # 4c: OOS refusal bypass + generator_calls invariant
        h_before = (await client.get("/health")).json()
        calls_before = int(h_before["generator_calls"])
        r_4c = await client.post(
            "/rag/async",
            json={"query": "How do I bake a sourdough loaf?", "stream": False, "use_cache": True},
        )
        if r_4c.status_code != 200:
            raise RuntimeError(f"4c failed: HTTP {r_4c.status_code} {r_4c.text}")
        d_4c = r_4c.json()
        h_after = (await client.get("/health")).json()
        calls_after = int(h_after["generator_calls"])
        expected_refusal = canonical_refusal("HARD_REFUSAL")
        if (
            d_4c.get("bypassed_llm") is not True
            or d_4c.get("answer") != expected_refusal
            or calls_after != calls_before
        ):
            raise RuntimeError(
                f"4c refusal invariant failed: calls_before={calls_before}, calls_after={calls_after}, resp={d_4c}"
            )
        sub_4c = {
            "bypassed_llm": d_4c["bypassed_llm"],
            "refused": d_4c["refused"],
            "generator_calls_before": calls_before,
            "generator_calls_after": calls_after,
            "total_ms": d_4c["total_ms"],
            "passed": True,
        }

        # 4d: SSE streaming on second inscope_dev query
        raw_lines: list[str] = []
        async with client.stream(
            "POST", "/rag/async", json={"query": q1, "stream": True, "use_cache": True}
        ) as stream_resp:
            if stream_resp.status_code != 200:
                body = (await stream_resp.aread()).decode("utf-8", errors="replace")
                raise RuntimeError(f"4d stream failed: HTTP {stream_resp.status_code} {body}")
            async for line in stream_resp.aiter_lines():
                raw_lines.append(line)

        frames = _parse_sse_stream(raw_lines)
        event_names = [ev for ev, _ in frames]
        token_indices = [i for i, ev in enumerate(event_names) if ev == "token"]
        if (
            len(event_names) < 4
            or event_names[0] != "ttft"
            or not token_indices
            or event_names[-2:] != ["metrics", "done"]
            or any(ev != "token" for ev in event_names[1:-2])
        ):
            raise RuntimeError(f"4d invalid SSE event order: {event_names}")
        streamed_text = "".join(str(data.get("token", "")) for ev, data in frames if ev == "token")
        metrics_payload = frames[-2][1]
        sub_4d = {
            "dev_id": inscope_rows[1].get("dev_id"),
            "event_sequence_valid": True,
            "token_events": len(token_indices),
            "streamed_chars": len(streamed_text),
            "ttft_ms": frames[0][1].get("ttft_ms"),
            "total_ms": metrics_payload.get("total_ms"),
            "passed": True,
        }

        # 4e: /rag/sync latency and parity vs /rag/async for q0
        r_4e = await client.post("/rag/sync", json={"query": q0, "stream": False, "use_cache": False})
        if r_4e.status_code != 200:
            raise RuntimeError(f"4e failed: HTTP {r_4e.status_code} {r_4e.text}")
        d_4e = r_4e.json()
        if d_4e.get("endpoint") != "sync":
            raise RuntimeError(f"4e expected endpoint='sync', got {d_4e.get('endpoint')!r}")
        async_ans = str(d_4b_cold["answer"])
        sync_ans = str(d_4e["answer"])
        seq_ratio = round(difflib.SequenceMatcher(None, async_ans, sync_ans).ratio(), 4)
        async_words = set(async_ans.lower().split())
        sync_words = set(sync_ans.lower().split())
        union = async_words | sync_words
        jaccard = round((len(async_words & sync_words) / len(union)) if union else 1.0, 4)
        sub_4e = {
            "dev_id": inscope_rows[0].get("dev_id"),
            "sync_total_ms": d_4e["total_ms"],
            "sync_ttft_ms": d_4e["ttft_ms"],
            "async_cold_total_ms": d_4b_cold["total_ms"],
            "exact_answer_match": async_ans == sync_ans,
            "sequence_similarity": seq_ratio,
            "token_jaccard": jaccard,
            "retrieved_chunks_identical": d_4e["retrieved_chunk_ids"] == d_4b_cold["retrieved_chunk_ids"],
            "passed": True,
        }

        # 4f: Concurrent cold semantic query (use_cache=false) + 100ms /health probe (p95 < 50 ms)
        done_evt = asyncio.Event()
        health_latencies_ms: list[float] = []

        async def _poll_health() -> None:
            while not done_evt.is_set():
                t_h = time.perf_counter()
                rh = await client.get("/health")
                dt_ms = (time.perf_counter() - t_h) * 1000.0
                if rh.status_code != 200:
                    raise RuntimeError(f"4f /health returned HTTP {rh.status_code} during flight")
                health_latencies_ms.append(round(dt_ms, 3))
                await asyncio.sleep(0.1)

        async def _run_cold_query() -> dict[str, Any]:
            try:
                rq = await client.post("/rag/async", json={"query": q2, "stream": False, "use_cache": False})
                if rq.status_code != 200:
                    raise RuntimeError(f"4f concurrent query failed: HTTP {rq.status_code} {rq.text}")
                return rq.json()
            finally:
                done_evt.set()

        d_4f, _ = await asyncio.gather(_run_cold_query(), _poll_health())
        p95_health = round(_percentile(health_latencies_ms, 95.0), 3)
        if p95_health >= 50.0:
            raise RuntimeError(
                f"4f /health p95 latency {p95_health} ms >= 50.0 ms (samples={health_latencies_ms})"
            )
        sub_4f = {
            "dev_id": inscope_rows[2].get("dev_id"),
            "query_total_ms": d_4f["total_ms"],
            "cache_hit": d_4f["cache_hit"],
            "health_samples_count": len(health_latencies_ms),
            "health_min_ms": round(min(health_latencies_ms), 3),
            "health_p50_ms": round(_percentile(health_latencies_ms, 50.0), 3),
            "health_p95_ms": p95_health,
            "health_max_ms": round(max(health_latencies_ms), 3),
            "passed": True,
        }

        # 4g: Background corpus verification task vs Stage 0 manifest
        r_task = await client.post("/tasks/ingest", json={"task": "verify_corpus", "label": "preflight"})
        if r_task.status_code != 202:
            raise RuntimeError(f"4g POST /tasks/ingest failed: HTTP {r_task.status_code} {r_task.text}")
        job = r_task.json()
        job_id = str(job["job_id"])

        poll_deadline = time.monotonic() + 60.0
        while time.monotonic() < poll_deadline:
            r_poll = await client.get(f"/tasks/{job_id}")
            if r_poll.status_code != 200:
                raise RuntimeError(f"4g GET /tasks/{job_id} failed: HTTP {r_poll.status_code} {r_poll.text}")
            job = r_poll.json()
            if job["status"] in ("succeeded", "failed"):
                break
            await asyncio.sleep(0.2)

        if job["status"] != "succeeded":
            raise RuntimeError(f"4g task {job_id} did not succeed: {job}")

        files_out = (job.get("result") or {}).get("files") or {}
        frozen = baseline["frozen_files"]
        expected_map = {
            "corpus_chunks.parquet": frozen["data/processed/corpus_chunks.parquet"],
            "index_sq8_mrl512.faiss": frozen["models/index_sq8_mrl512.faiss"],
            "index_sq8_mrl512.idmap.json": frozen["models/index_sq8_mrl512.idmap.json"],
        }
        verified_files: dict[str, Any] = {}
        for fname, exp in expected_map.items():
            got = files_out.get(fname)
            if not got or got.get("sha256") != exp["sha256"] or got.get("bytes") != exp["bytes"]:
                raise RuntimeError(f"4g SHA-256/byte mismatch for {fname}: got={got}, expected={exp}")
            verified_files[fname] = {**got, "matches_stage0": True}

        sub_4g = {
            "job_id": job_id,
            "status": job["status"],
            "verified_files": verified_files,
            "passed": True,
        }

    return {
        "4a_fast_path_cve": sub_4a,
        "4b_semantic_cache": sub_4b,
        "4c_refusal_bypass": sub_4c,
        "4d_sse_streaming": sub_4d,
        "4e_sync_vs_async_parity": sub_4e,
        "4f_concurrent_health_probe": sub_4f,
        "4g_verify_corpus_task": sub_4g,
        "passed": True,
    }


def _gate5_stop_server() -> dict[str, Any]:
    proc = _run_ps_script(
        _START_SERVER_PS1,
        ["-Stop"],
        timeout_s=30.0,
        log_path=_SERVER_LAUNCHER_LOG,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"start_server.ps1 -Stop failed: rc={proc.returncode} {proc.stderr}")

    pid_file_exists = _SERVER_PID_FILE.exists()
    health_unreachable = False
    with httpx.Client(timeout=httpx.Timeout(5.0, connect=2.0)) as client:
        try:
            client.get(f"{_SERVER_BASE_URL}/health")
        except httpx.HTTPError:
            health_unreachable = True

    apps = _probe_gpu_compute_apps()
    python_apps = [app for app in apps if "python" in str(app.get("process_name", "")).lower()]
    if pid_file_exists or not health_unreachable or python_apps:
        raise RuntimeError(
            f"Gate 5 clean shutdown failed: pid_file_exists={pid_file_exists}, "
            f"health_unreachable={health_unreachable}, python_gpu_apps={python_apps}"
        )

    return {
        "stop_exit_code": proc.returncode,
        "pid_file_removed": not pid_file_exists,
        "health_unreachable": health_unreachable,
        "gpu_compute_apps": apps,
        "python_gpu_apps": python_apps,
        "passed": True,
    }


def run_preflight(output_path: Path = _DEFAULT_OUTPUT) -> dict[str, Any]:
    if not _BASELINE_MANIFEST.is_file():
        raise FileNotFoundError(f"Stage 0 manifest missing: {_BASELINE_MANIFEST}")
    baseline = json.loads(_BASELINE_MANIFEST.read_text(encoding="utf-8"))
    gen = get_gen_settings()

    report: dict[str, Any] = {
        "schema_version": "1.0",
        "stage": "phase3_server_preflight",
        "created_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "passed": False,
        "gates": {},
        "error": None,
    }

    server_started = False
    try:
        print("[Gate 1/5] Checking router LFS weights...", flush=True)
        g1 = _gate1_lfs()
        report["gates"]["gate1_router_lfs"] = g1
        if not g1["passed"]:
            raise RuntimeError(f"Gate 1 failed: {g1}")

        print("[Gate 2/5] Checking Ollama GPU residency and warmup...", flush=True)
        report["gates"]["gate2_ollama_gpu"] = _gate2_ollama(gen)

        print("[Gate 3/5] Starting FastAPI server via start_server.ps1 -Restart...", flush=True)
        server_started = True
        report["gates"]["gate3_server_start"] = _gate3_start_server()

        print("[Gate 4/5] Running live HTTP smoke checks (4a-4g)...", flush=True)
        report["gates"]["gate4_live_smoke"] = asyncio.run(_gate4_live_smoke(baseline))
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        if server_started:
            print("[Gate 5/5] Stopping server and verifying GPU clean shutdown...", flush=True)
            try:
                report["gates"]["gate5_server_stop"] = _gate5_stop_server()
            except Exception as stop_exc:
                report["gates"]["gate5_server_stop"] = {
                    "passed": False,
                    "error": f"{type(stop_exc).__name__}: {stop_exc}",
                }
                if report["error"] is None:
                    report["error"] = f"{type(stop_exc).__name__}: {stop_exc}"
            else:
                if report["error"] is None:
                    report["passed"] = True
        _atomic_write_json(output_path, report)

    return report


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run KAVACH-RAG Phase 3 server preflight gates and write JSON report."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=_DEFAULT_OUTPUT,
        help="Path for the output JSON preflight report (default: reports/phase3_server_preflight.json).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        report = run_preflight(args.output)
    except Exception as exc:
        print(f"Preflight FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(f"Preflight PASSED. Report written to {args.output}")
    return 0 if report.get("passed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
