"""Locust profile for /rag/sync and /rag/async. Configured by environment variables.

KAVACH_LOAD_ENDPOINT = sync | async      KAVACH_LOAD_MODE = warm | cold
warm: cache primed, use_cache=true (measures stampede protection + replay).
cold: use_cache=false (measures real retrieval+generation capacity). /rag/sync ignores caching.
"""
from __future__ import annotations

import json
import os
import random
import tempfile
import time
from pathlib import Path

from locust import HttpUser, between, events, task

import gevent
import httpx
import requests

ROOT = Path(__file__).resolve().parents[1]
ENDPOINT = os.environ.get("KAVACH_LOAD_ENDPOINT", "async")
MODE = os.environ.get("KAVACH_LOAD_MODE", "warm")
TIMEOUT_S = 120.0
GOLDEN_FREE_SEED = 20260
QUERY_CLASSES = ("cve-id", "semantic", "oos")


def _load_queries(name: str) -> list[str]:
    path = ROOT / "data" / "golden" / name
    with path.open("r", encoding="utf-8") as fh:
        return [json.loads(line)["query"] for line in fh if line.strip()]


def _build_tagged_pool() -> list[tuple[str, str]]:
    """8 tagged queries from the DEV sets (never golden): 2 cve-id, 4 semantic, 2 oos."""
    rng = random.Random(GOLDEN_FREE_SEED)
    inscope = _load_queries("inscope_dev_100.jsonl")
    oos = _load_queries("oos_dev_60.jsonl")
    return [
        ("cve-id", "CVE-2024-3094"),
        ("cve-id", "CVE-2021-4034"),
        *[("semantic", q) for q in rng.sample(inscope, 4)],
        *[("oos", q) for q in rng.sample(oos, 2)],
    ]


def _build_pool() -> list[str]:
    """8 queries from the DEV sets (never the golden set): 2 bare CVE ids, 4 in-scope, 2 off-topic."""
    return [query for _, query in _build_tagged_pool()]


TAGGED_POOL = _build_tagged_pool()
POOL = [query for _, query in TAGGED_POOL]
PATH = "/rag/sync" if ENDPOINT == "sync" else "/rag/async"
USE_CACHE = MODE == "warm"
_probe_greenlet = None
_health_stats: dict[str, int | None] = {
    "health_sample_count": 0,
    "max_gpu_waiting": None,
    "max_gpu_active": None,
    "max_retrieval_waiting": None,
}


def _update_max(key: str, raw_val: object) -> None:
    if isinstance(raw_val, bool) or raw_val is None:
        return
    try:
        val = int(raw_val)
    except (TypeError, ValueError):
        return
    cur = _health_stats[key]
    _health_stats[key] = val if cur is None else max(cur, val)


def _record_health_sample(payload: dict) -> None:
    _health_stats["health_sample_count"] = int(_health_stats["health_sample_count"] or 0) + 1
    _update_max("max_gpu_waiting", payload.get("gpu_waiting"))
    gpu_active = payload.get("gpu_inflight")
    if gpu_active is None:
        gpu_active = payload.get("gpu_active")
    _update_max("max_gpu_active", gpu_active)
    _update_max("max_retrieval_waiting", payload.get("retrieval_waiting"))


def _resolve_meta_path(environment) -> Path | None:
    env_path = os.environ.get("KAVACH_LOCUST_META_PATH")
    if env_path:
        return Path(env_path)
    parsed = getattr(environment, "parsed_options", None)
    csv_prefix = getattr(parsed, "csv_prefix", None)
    if csv_prefix:
        return Path(f"{csv_prefix}_meta.json")
    return None


def _write_probe_meta(environment) -> None:
    meta_path = _resolve_meta_path(environment)
    if meta_path is None:
        return
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    merged: dict[str, object] = {}
    if meta_path.exists():
        try:
            loaded = json.loads(meta_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                merged.update(loaded)
        except (OSError, json.JSONDecodeError):
            pass
    merged.update(
        {
            "max_gpu_waiting": _health_stats["max_gpu_waiting"],
            "max_gpu_active": _health_stats["max_gpu_active"],
            "max_retrieval_waiting": _health_stats["max_retrieval_waiting"],
            "health_sample_count": int(_health_stats["health_sample_count"] or 0),
            "health_sample_rate_hz": 1.0,
            "queue_maxima_bound": "1 Hz lower bound sampled via GET /health",
        }
    )
    fd, tmp_name = tempfile.mkstemp(prefix=f".{meta_path.name}.", suffix=".tmp", dir=str(meta_path.parent))
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(merged, fh, indent=2, ensure_ascii=False)
            fh.write("\n")
            fh.flush()
        os.replace(tmp_path, meta_path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def _probe_loop(host: str, environment) -> None:
    while True:
        t0 = time.perf_counter()
        exc = None
        length = 0
        try:
            resp = requests.get(f"{host}/health", timeout=5.0)
            length = len(resp.content)
            if resp.status_code != 200:
                exc = RuntimeError(f"HTTP {resp.status_code}")
            else:
                try:
                    payload = resp.json()
                except ValueError:
                    payload = None
                if isinstance(payload, dict):
                    _record_health_sample(payload)
        except requests.RequestException as err:
            exc = err
        environment.events.request.fire(
            request_type="GET", name="/health probe", response_time=(time.perf_counter() - t0) * 1000.0,
            response_length=length, exception=exc, context={},
        )
        gevent.sleep(1.0)


@events.test_start.add_listener
def _on_start(environment, **kwargs) -> None:
    global _probe_greenlet
    _health_stats.update(
        {
            "health_sample_count": 0,
            "max_gpu_waiting": None,
            "max_gpu_active": None,
            "max_retrieval_waiting": None,
        }
    )
    host = environment.host
    if ENDPOINT == "async" and MODE == "warm":
        for query in POOL:  # prime the cache once; excluded from the measured stats
            httpx.post(f"{host}/rag/async", json={"query": query, "stream": False, "use_cache": True},
                       timeout=300.0)
    _probe_greenlet = gevent.spawn(_probe_loop, host, environment)


@events.test_stop.add_listener
def _on_stop(environment, **kwargs) -> None:
    global _probe_greenlet
    if _probe_greenlet is not None:
        _probe_greenlet.kill()
        _probe_greenlet = None
    _write_probe_meta(environment)


class RagUser(HttpUser):
    wait_time = between(0.2, 0.8)

    @task(4)
    def ask(self) -> None:
        q_class, query = random.choice(TAGGED_POOL)
        body = {"query": query, "stream": False, "use_cache": USE_CACHE}
        with self.client.post(PATH, json=body, name=f"{PATH}[{MODE}:{q_class}]", timeout=TIMEOUT_S,
                              catch_response=True) as resp:
            if resp.status_code != 200:
                resp.failure(f"HTTP {resp.status_code}: {resp.text[:120]}")
                return
            try:
                payload = resp.json()
            except ValueError:
                resp.failure("response is not JSON")
                return
            if not payload.get("answer"):
                resp.failure("empty answer")
            else:
                resp.success()

    @task(1)
    def ask_stream(self) -> None:
        if ENDPOINT != "async":
            return
        q_class, query = random.choice(TAGGED_POOL)
        body = {"query": query, "stream": True, "use_cache": USE_CACHE}
        name = f"/rag/async stream[{MODE}:{q_class}]"
        t0 = time.perf_counter()
        ttft_ms = None
        done = False
        error = None
        with self.client.post("/rag/async", json=body, name=name, timeout=TIMEOUT_S, stream=True,
                              catch_response=True) as resp:
            if resp.status_code != 200:
                resp.failure(f"HTTP {resp.status_code}")
                return
            event = None
            for line in resp.iter_lines(decode_unicode=True):
                if line.startswith("event:"):
                    event = line[6:].strip()
                    if event == "done":
                        done = True
                    elif event == "error":
                        error = "SSE error event"
                elif line.startswith("data:") and event == "token" and ttft_ms is None:
                    ttft_ms = (time.perf_counter() - t0) * 1000.0
            if done and error is None:
                resp.success()
            else:
                resp.failure(error or "stream ended without done event")
        if ttft_ms is not None:
            events.request.fire(request_type="SSE", name=f"/rag/async first-token[{MODE}:{q_class}]",
                                response_time=ttft_ms, response_length=0, exception=None, context={})
