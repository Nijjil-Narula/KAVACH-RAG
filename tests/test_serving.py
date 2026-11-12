"""Serving-layer tests (no GPU, no Ollama, no retrieval models)."""
from __future__ import annotations

import asyncio
import time

from kavach.generation.prompts import canonical_refusal
from kavach.serving.app import parse_sse
from kavach.serving.schemas import QueryResponse
from serving_fakes import client, make_rig

SERVED = {"query": "What is the CVSS score of CVE-2024-3094?"}


def sse_events(text: str) -> list[tuple[str, dict]]:
    return [parse_sse(frame) for frame in text.split("\n\n") if frame.strip()]


def test_health_shape_and_speed() -> None:
    rig = make_rig()

    async def go():
        async with client(rig.app) as c:
            t0 = time.perf_counter()
            resp = await c.get("/health")
            return resp, (time.perf_counter() - t0) * 1000.0

    resp, ms = asyncio.run(go())
    body = resp.json()
    assert resp.status_code == 200 and ms < 50.0
    assert body["status"] == "ok" and body["circuit_state"] == "closed"
    assert body["vram_used_mib"] == 3000 and body["vram_hard_limit_mib"] == 5478


def test_async_served_contract_and_headers() -> None:
    rig = make_rig()

    async def go():
        async with client(rig.app) as c:
            return await c.post("/rag/async", json=SERVED)

    resp = asyncio.run(go())
    body = QueryResponse(**resp.json())
    assert resp.status_code == 200 and "10.0" in body.answer and not body.cache_hit
    assert body.endpoint == "async" and not body.bypassed_llm
    assert body.retrieved_chunk_ids == ["CVE-2024-3094_chunk_001"]
    for header in ("X-Retrieval-Latency-MS", "X-TTFT-MS", "X-Total-Tokens", "X-Tokens-Per-Second"):
        assert header in resp.headers
    assert resp.headers["X-Cache"] == "MISS" and rig.ollama.calls == 1


def test_cache_hit_is_fast_and_makes_no_upstream_call() -> None:
    rig = make_rig()

    async def go():
        async with client(rig.app) as c:
            first = await c.post("/rag/async", json=SERVED)
            second = await c.post("/rag/async", json=SERVED)
            return first, second

    first, second = asyncio.run(go())
    assert second.json()["cache_hit"] is True and second.headers["X-Cache"] == "HIT"
    assert second.json()["answer"] == first.json()["answer"]
    assert second.json()["total_ms"] < 50.0
    assert rig.ollama.calls == 1 and rig.pipeline.calls == 1


def test_use_cache_false_bypasses_cache() -> None:
    rig = make_rig()
    body = {**SERVED, "use_cache": False}

    async def go():
        async with client(rig.app) as c:
            await c.post("/rag/async", json=body)
            await c.post("/rag/async", json=body)

    asyncio.run(go())
    assert rig.pipeline.calls == 2 and rig.ollama.calls == 2 and rig.rt.cache.stats()["size"] == 0


def test_refusal_and_abstention_bypass_gpu() -> None:
    rig = make_rig()

    async def go():
        async with client(rig.app) as c:
            refused = await c.post("/rag/async", json={"query": "oos: how do I bake sourdough?"})
            abstained = await c.post("/rag/async", json={"query": "lowconf: obscure thing"})
            return refused.json(), abstained.json()

    refused, abstained = asyncio.run(go())
    assert refused["bypassed_llm"] and refused["answer"] == canonical_refusal("HARD_REFUSAL")
    assert abstained["bypassed_llm"] and abstained["answer"] == canonical_refusal("LOW_RETRIEVAL_CONFIDENCE")
    assert rig.ollama.calls == 0 and rig.rt.generator.calls == 0


def test_sse_event_order_and_headers() -> None:
    rig = make_rig()

    async def go():
        async with client(rig.app) as c:
            return await c.post("/rag/async", json={**SERVED, "stream": True})

    resp = asyncio.run(go())
    events = sse_events(resp.text)
    names = [name for name, _ in events]
    assert resp.headers["content-type"].startswith("text/event-stream")
    assert "X-Retrieval-Latency-MS" in resp.headers
    assert names[0] == "ttft" and names[-2:] == ["metrics", "done"]
    assert set(names[1:-2]) == {"token"}
    streamed = "".join(data["token"] for name, data in events if name == "token")
    assert "10.0" in streamed and rig.ollama.calls == 1


def test_stream_replay_from_cache_matches_answer() -> None:
    rig = make_rig()

    async def go():
        async with client(rig.app) as c:
            first = await c.post("/rag/async", json=SERVED)
            replay = await c.post("/rag/async", json={**SERVED, "stream": True})
            return first.json()["answer"], replay

    answer, replay = asyncio.run(go())
    events = sse_events(replay.text)
    assert replay.headers["X-Cache"] == "HIT"
    assert [name for name, _ in events][0] == "ttft" and events[-1][0] == "done"
    assert "".join(data["token"] for name, data in events if name == "token") == answer
    assert rig.ollama.calls == 1


def test_sync_endpoint_contract_has_no_cache() -> None:
    rig = make_rig()

    async def go():
        async with client(rig.app) as c:
            one = await c.post("/rag/sync", json=SERVED)
            two = await c.post("/rag/sync", json=SERVED)
            return one, two

    one, two = asyncio.run(go())
    body = QueryResponse(**two.json())
    assert one.status_code == 200 and body.endpoint == "sync" and not body.cache_hit
    assert rig.pipeline.calls == 2 and rig.ollama.calls == 2


def test_pipeline_calls_are_serialized() -> None:
    rig = make_rig()

    async def go():
        async with client(rig.app) as c:
            bodies = [{"query": f"question {i}", "use_cache": False} for i in range(6)]
            return await asyncio.gather(*[c.post("/rag/async", json=b) for b in bodies])

    responses = asyncio.run(go())
    assert all(r.status_code == 200 for r in responses)
    assert rig.pipeline.max_active == 1 and rig.pipeline.calls == 6


def test_gpu_semaphore_bounds_inflight_generation() -> None:
    rig = make_rig()
    rig.rt.gpu_sem = asyncio.Semaphore(2)
    rig.ollama.delay_s = 0.05

    async def go():
        async with client(rig.app) as c:
            bodies = [{"query": f"question {i}", "use_cache": False} for i in range(8)]
            return await asyncio.gather(*[c.post("/rag/async", json=b) for b in bodies])

    responses = asyncio.run(go())
    assert all(r.status_code == 200 for r in responses)
    assert rig.ollama.calls == 8 and rig.ollama.max_inflight <= 2


def test_cache_stampede_coalesces_identical_misses() -> None:
    rig = make_rig()

    async def go():
        async with client(rig.app) as c:
            return await asyncio.gather(*[c.post("/rag/async", json=SERVED) for _ in range(10)])

    responses = asyncio.run(go())
    hits = [r.json()["cache_hit"] for r in responses]
    assert all(r.status_code == 200 for r in responses)
    assert rig.pipeline.calls == 1 and rig.ollama.calls == 1
    assert hits.count(False) == 1 and hits.count(True) == 9


def test_stream_stampede_generates_once() -> None:
    rig = make_rig()

    async def go():
        async with client(rig.app) as c:
            return await asyncio.gather(*[c.post("/rag/async", json={**SERVED, "stream": True}) for _ in range(5)])

    responses = asyncio.run(go())
    assert all(r.status_code == 200 and "event: done" in r.text for r in responses)
    assert rig.pipeline.calls == 1 and rig.ollama.calls == 1


def test_ingest_task_lifecycle(tmp_path) -> None:
    target = tmp_path / "corpus.bin"
    target.write_bytes(b"kavach")
    rig = make_rig()
    rig.rt.corpus_paths = [target]

    async def go():
        async with client(rig.app) as c:
            created = await c.post("/tasks/ingest", json={"task": "verify_corpus", "label": "t"})
            job_id = created.json()["job_id"]
            status = None
            for _ in range(50):
                status = await c.get(f"/tasks/{job_id}")
                if status.json()["status"] in ("succeeded", "failed"):
                    break
                await asyncio.sleep(0.02)
            missing = await c.get("/tasks/does-not-exist")
            return created, status, missing

    created, status, missing = asyncio.run(go())
    assert created.status_code == 202 and status.json()["status"] == "succeeded"
    assert status.json()["result"]["files"]["corpus.bin"]["bytes"] == 6
    assert missing.status_code == 404


def test_schema_rejects_bad_requests() -> None:
    rig = make_rig()

    async def go():
        async with client(rig.app) as c:
            empty = await c.post("/rag/async", json={"query": ""})
            extra = await c.post("/rag/async", json={"query": "x", "bogus": 1})
            return empty.status_code, extra.status_code

    assert asyncio.run(go()) == (422, 422)
