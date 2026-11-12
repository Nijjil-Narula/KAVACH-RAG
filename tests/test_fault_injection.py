"""Fault-injection tests: circuit breaker, upstream failures, event-loop responsiveness, VRAM, budgets."""
from __future__ import annotations

import asyncio
import threading
import time

import pytest

from kavach.generation.prompts import canonical_refusal
from kavach.serving.resilience import CircuitBreaker, MutexTTLCache
from serving_fakes import FakeClock, FakePipeline, client, make_rig

SERVED = {"query": "What is the CVSS score of CVE-2024-3094?", "use_cache": False}
HOLD_S = 5.0


def test_breaker_state_machine_with_fake_clock() -> None:
    clock = FakeClock()
    breaker = CircuitBreaker(failure_threshold=5, open_seconds=30.0, clock=clock)
    for _ in range(4):
        breaker.record_failure()
    assert breaker.state == "closed" and breaker.allow()
    breaker.record_failure()
    assert breaker.state == "open" and not breaker.allow() and breaker.trips == 1
    clock.advance(29.0)
    assert breaker.state == "open"
    clock.advance(1.0)
    assert breaker.state == "half_open"
    assert breaker.allow() and not breaker.allow()
    breaker.record_failure()
    assert breaker.state == "open" and breaker.trips == 2
    clock.advance(30.0)
    assert breaker.allow()
    breaker.record_success()
    assert breaker.state == "closed" and breaker.allow()


@pytest.mark.parametrize("mode", ["http500", "http429", "connect_error"])
def test_upstream_failures_trip_breaker_then_503_without_calls(mode: str) -> None:
    rig = make_rig(gen_attempts=1)
    rig.ollama.mode = mode

    async def go():
        async with client(rig.app) as c:
            first = [await c.post("/rag/async", json=SERVED) for _ in range(5)]
            calls_before = rig.ollama.calls
            sixth = await c.post("/rag/async", json=SERVED)
            return first, calls_before, sixth

    first, calls_before, sixth = asyncio.run(go())
    assert [r.status_code for r in first] == [502] * 5
    assert rig.rt.breaker.state == "open"
    assert sixth.status_code == 503 and sixth.json()["status"] == "STATUS_TEMPORARILY_DEGRADED"
    assert "Retry-After" in sixth.headers and rig.ollama.calls == calls_before


def test_recovery_after_open_window() -> None:
    rig = make_rig(gen_attempts=1)
    rig.ollama.mode = "http500"

    async def go():
        async with client(rig.app) as c:
            for _ in range(5):
                await c.post("/rag/async", json=SERVED)
            blocked = await c.post("/rag/async", json=SERVED)
            rig.clock.advance(30.0)
            rig.ollama.mode = "ok"
            probe = await c.post("/rag/async", json=SERVED)
            return blocked, probe

    blocked, probe = asyncio.run(go())
    assert blocked.status_code == 503 and probe.status_code == 200
    assert rig.rt.breaker.state == "closed" and rig.rt.cache.stats()["size"] == 0


def test_refusals_still_served_while_circuit_open() -> None:
    rig = make_rig(gen_attempts=1)
    rig.ollama.mode = "http500"

    async def go():
        async with client(rig.app) as c:
            for _ in range(5):
                await c.post("/rag/async", json=SERVED)
            return await c.post("/rag/async", json={"query": "oos: bake bread", "use_cache": False})

    resp = asyncio.run(go())
    assert rig.rt.breaker.state == "open"
    assert resp.status_code == 200 and resp.json()["answer"] == canonical_refusal("HARD_REFUSAL")


def test_health_stays_responsive_while_retrieval_is_blocked() -> None:
    started, release = threading.Event(), threading.Event()
    rig = make_rig(pipeline=FakePipeline(block=release, started=started))

    async def go():
        async with client(rig.app) as c:
            slow = asyncio.create_task(c.post("/rag/async", json=SERVED))
            assert await asyncio.to_thread(started.wait, 5.0)
            worst_ms = 0.0
            end = time.perf_counter() + HOLD_S
            while time.perf_counter() < end:
                t0 = time.perf_counter()
                resp = await c.get("/health")
                worst_ms = max(worst_ms, (time.perf_counter() - t0) * 1000.0)
                assert resp.status_code == 200
                await asyncio.sleep(0.1)
            release.set()
            return worst_ms, await slow

    worst_ms, slow_resp = asyncio.run(go())
    assert worst_ms < 50.0
    assert slow_resp.status_code == 200


def test_key_lock_acquire_times_out_and_recovers() -> None:
    cache = MutexTTLCache()

    async def go():
        held = await cache.acquire("k", 1.0)
        assert held is not None
        timed_out = await cache.acquire("k", 0.05)
        held.release()
        held.release()
        again = await cache.acquire("k", 1.0)
        assert again is not None
        again.release()
        return timed_out

    assert asyncio.run(go()) is None
    assert cache.stats()["lock_timeouts"] == 1


def test_vram_hard_limit_blocks_generation_but_not_refusals() -> None:
    rig = make_rig(vram_used=6000)

    async def go():
        async with client(rig.app) as c:
            blocked = await c.post("/rag/async", json=SERVED)
            refused = await c.post("/rag/async", json={"query": "oos: weather", "use_cache": False})
            return blocked, refused

    blocked, refused = asyncio.run(go())
    assert blocked.status_code == 503 and blocked.json()["status"] == "VRAM_HARD_LIMIT_EXCEEDED"
    assert rig.ollama.calls == 0 and refused.status_code == 200


def test_prompt_budget_error_is_500_and_makes_no_upstream_call() -> None:
    rig = make_rig(counter=lambda text: 10_000)

    async def go():
        async with client(rig.app) as c:
            return await c.post("/rag/async", json=SERVED)

    resp = asyncio.run(go())
    assert resp.status_code == 500 and resp.json()["status"] == "PROMPT_BUDGET_ERROR"
    assert rig.ollama.calls == 0
