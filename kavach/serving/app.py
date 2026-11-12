"""FastAPI serving layer: /rag/sync (baseline), /rag/async (resilient), /health, /tasks.

Retrieval is serialized (one pipeline call at a time); generation is bounded by Semaphore(20).
"""
from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import logging
import math
import re
import threading
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import psutil
from fastapi import BackgroundTasks, FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.background import BackgroundTask

from kavach.exceptions import GenerationError, PromptBudgetError
from kavach.generation.gen_config import GenSettings, get_gen_settings
from kavach.generation.generator import OllamaGenerator
from kavach.generation.prompts import PROMPT_VERSION, QwenTokenCounter, canonical_refusal, plan_prompt
from kavach.serving.resilience import (
    CircuitBreaker,
    CircuitOpenError,
    MutexTTLCache,
    ServiceDegraded,
    TaskRegistry,
    VRAMSampler,
    utc_now,
)
from kavach.serving.schemas import (
    HealthResponse,
    IngestionTaskRequest,
    QueryRequest,
    QueryResponse,
    TaskStatusResponse,
)

log = logging.getLogger(__name__)

GPU_CONCURRENCY = 20
RETRIEVAL_CONCURRENCY = 1
CACHE_LOCK_TIMEOUT_S = 150.0


@dataclass
class Runtime:
    """Everything the endpoints share. Tests inject fakes; production builds it in the lifespan."""

    pipeline: Any
    generator: OllamaGenerator
    counter: Any
    sampler: VRAMSampler
    gen: GenSettings
    cache: MutexTTLCache = field(default_factory=MutexTTLCache)
    breaker: CircuitBreaker = field(default_factory=CircuitBreaker)
    tasks: TaskRegistry = field(default_factory=TaskRegistry)
    gpu_sem: asyncio.Semaphore = field(default_factory=lambda: asyncio.Semaphore(GPU_CONCURRENCY))
    retrieval_sem: asyncio.Semaphore = field(default_factory=lambda: asyncio.Semaphore(RETRIEVAL_CONCURRENCY))
    pipeline_lock: threading.Lock = field(default_factory=threading.Lock)
    started_at: float = field(default_factory=time.monotonic)
    retrieval_waiting: int = 0
    gpu_waiting: int = 0
    gpu_inflight: int = 0
    corpus_paths: list[Path] = field(default_factory=list)


@dataclass
class Retrieved:
    result: Any
    plan: Any | None
    queue_ms: float
    reason: str | None


# ---------------------------------------------------------------- helpers
def cache_key(query: str) -> str:
    return hashlib.sha256(" ".join(query.split()).encode("utf-8")).hexdigest()


def sse(event: str, data: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


def parse_sse(frame: str) -> tuple[str, dict[str, Any]]:
    event = "message"
    data_lines: list[str] = []
    for line in frame.split("\n"):
        if line.startswith("event:"):
            event = line[6:].strip()
        elif line.startswith("data:"):
            data_lines.append(line[5:].lstrip(" "))
    raw = "\n".join(data_lines)
    return event, (json.loads(raw, strict=False) if raw else {})


def _reason(res: Any) -> str | None:
    if res.refused:
        return "HARD_REFUSAL"
    if res.abstained:
        return res.abstain_reason or "LOW_RETRIEVAL_CONFIDENCE"
    if not res.contexts:
        return "LOW_RETRIEVAL_CONFIDENCE"
    return None


def _retrieve_and_plan(rt: Runtime, query: str) -> tuple[Any, Any | None, float]:
    """Worker-thread function: serialized retrieval plus D5 prompt planning. Returns lock wait in ms."""
    t_wait = time.perf_counter()
    with rt.pipeline_lock:
        wait_ms = (time.perf_counter() - t_wait) * 1000.0
        res = rt.pipeline.retrieve(query)
        plan = None
        if _reason(res) is None:
            plan = plan_prompt(
                query, res.contexts, rt.counter,
                num_ctx=rt.gen.num_ctx, max_new=rt.gen.num_predict,
                context_cap=rt.gen.context_token_cap, margin=rt.gen.prompt_safety_margin,
            )
    return res, plan, wait_ms


async def _retrieve_async(rt: Runtime, query: str) -> Retrieved:
    t_queue = time.perf_counter()
    rt.retrieval_waiting += 1
    try:
        await rt.retrieval_sem.acquire()
    except BaseException:
        rt.retrieval_waiting -= 1
        raise
    rt.retrieval_waiting -= 1
    sem_ms = (time.perf_counter() - t_queue) * 1000.0
    try:
        res, plan, lock_ms = await asyncio.to_thread(_retrieve_and_plan, rt, query)
    finally:
        rt.retrieval_sem.release()
    return Retrieved(res, plan, sem_ms + lock_ms, _reason(res))


def _admit(rt: Runtime) -> None:
    latest = rt.sampler.latest()
    if latest is not None and latest[0] > rt.gen.vram_hard_limit_mib:
        raise ServiceDegraded(
            "VRAM_HARD_LIMIT_EXCEEDED",
            f"GPU memory.used {latest[0]} MiB exceeds hard limit {rt.gen.vram_hard_limit_mib} MiB",
            5.0,
        )
    if not rt.breaker.allow():
        raise CircuitOpenError(rt.breaker.retry_after_s())


async def _gen_events(rt: Runtime, plan: Any) -> AsyncIterator[tuple[str, dict[str, Any]]]:
    rt.gpu_waiting += 1
    try:
        await rt.gpu_sem.acquire()
    except BaseException:
        rt.gpu_waiting -= 1
        raise
    rt.gpu_waiting -= 1
    rt.gpu_inflight += 1
    try:
        async for frame in rt.generator.astream(plan.messages):
            yield parse_sse(frame)
    finally:
        rt.gpu_inflight -= 1
        rt.gpu_sem.release()


async def _generate(rt: Runtime, plan: Any, t0: float) -> AsyncIterator[tuple[str, Any]]:
    """Yield ('ttft', ms), ('token', text)..., ('done', metrics). Records breaker outcomes."""
    _admit(rt)
    first = True
    metrics: dict[str, Any] | None = None
    try:
        async with contextlib.aclosing(_gen_events(rt, plan)) as events:
            async for event, data in events:
                if event == "token":
                    if first:
                        first = False
                        yield "ttft", round((time.perf_counter() - t0) * 1000.0, 3)
                    yield "token", data["token"]
                elif event == "metrics":
                    metrics = data
                elif event == "error":
                    raise GenerationError(str(data.get("error", "upstream stream error")))
        if metrics is None:
            raise GenerationError("Ollama stream ended without a metrics frame")
    except GenerationError:
        rt.breaker.record_failure()
        raise
    rt.breaker.record_success()
    yield "done", metrics


def _core(rt: Runtime, rv: Retrieved, *, answer: str, bypassed: bool,
          metrics: dict[str, Any], ttft_ms: float | None) -> dict[str, Any]:
    res, plan = rv.result, rv.plan
    return {
        "answer": answer,
        "route": str(res.route.route),
        "bypassed_llm": bypassed,
        "refused": bool(res.refused),
        "abstained": bool(res.abstained),
        "top1_logit": None if res.top1_logit is None else float(res.top1_logit),
        "retrieved_chunk_ids": list(plan.kept_chunk_ids) if plan is not None else [],
        "dropped_chunk_ids": list(plan.dropped_chunk_ids) if plan is not None else [],
        "retrieval_latency_ms": float(res.latency_ms.get("total_ms", 0.0)),
        "retrieval_queue_ms": round(rv.queue_ms, 3),
        "prompt_tokens": int(metrics.get("prompt_tokens", 0) or 0),
        "completion_tokens": int(metrics.get("completion_tokens", 0) or 0),
        "decode_tps": metrics.get("decode_tps"),
        "finish_reason": metrics.get("finish_reason"),
        "ttft_ms": ttft_ms,
        "prompt_version": PROMPT_VERSION,
    }


_PASSTHROUGH = (
    "answer", "route", "bypassed_llm", "refused", "abstained", "top1_logit", "retrieved_chunk_ids",
    "dropped_chunk_ids", "retrieval_latency_ms", "prompt_tokens", "completion_tokens", "decode_tps",
    "finish_reason", "prompt_version",
)


def _response_dict(core: dict[str, Any], *, cache_hit: bool, total_ms: float, endpoint: str) -> dict[str, Any]:
    d = {k: core[k] for k in _PASSTHROUGH}
    d["retrieval_queue_ms"] = 0.0 if cache_hit else float(core["retrieval_queue_ms"])
    d["ttft_ms"] = round(total_ms, 3) if cache_hit else core["ttft_ms"]
    d["total_ms"] = round(total_ms, 3)
    d["cache_hit"] = cache_hit
    d["endpoint"] = endpoint
    return d


def _diag_headers(d: dict[str, Any]) -> dict[str, str]:
    h = {
        "X-Retrieval-Latency-MS": f"{d['retrieval_latency_ms']:.3f}",
        "X-Total-Tokens": str(int(d["prompt_tokens"]) + int(d["completion_tokens"])),
        "X-Cache": "HIT" if d["cache_hit"] else "MISS",
        "X-Endpoint": str(d["endpoint"]),
    }
    if d["ttft_ms"] is not None:
        h["X-TTFT-MS"] = f"{float(d['ttft_ms']):.3f}"
    if d["decode_tps"] is not None:
        h["X-Tokens-Per-Second"] = f"{float(d['decode_tps']):.3f}"
    return h


def _json_response(d: dict[str, Any]) -> JSONResponse:
    return JSONResponse(QueryResponse(**d).model_dump(), headers=_diag_headers(d))


def _summary(d: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in d.items() if k != "answer"}


def _chunk_text(text: str) -> list[str]:
    return re.findall(r"\S+\s*|\s+", text)


async def _compute_core(rt: Runtime, query: str) -> dict[str, Any]:
    t0 = time.perf_counter()
    rv = await _retrieve_async(rt, query)
    if rv.plan is None:
        ms = round((time.perf_counter() - t0) * 1000.0, 3)
        return _core(rt, rv, answer=canonical_refusal(rv.reason), bypassed=True, metrics={}, ttft_ms=ms)
    parts: list[str] = []
    ttft: float | None = None
    metrics: dict[str, Any] = {}
    async for kind, value in _generate(rt, rv.plan, t0):
        if kind == "ttft":
            ttft = value
        elif kind == "token":
            parts.append(value)
        else:
            metrics = value
    return _core(rt, rv, answer="".join(parts), bypassed=False, metrics=metrics, ttft_ms=ttft)


async def _stream_live(rt: Runtime, req: QueryRequest, key: str, handle: Any,
                       rv: Retrieved, t0: float) -> AsyncIterator[str]:
    try:
        if rv.plan is None:
            answer = canonical_refusal(rv.reason)
            ms = round((time.perf_counter() - t0) * 1000.0, 3)
            core = _core(rt, rv, answer=answer, bypassed=True, metrics={}, ttft_ms=ms)
            yield sse("ttft", {"ttft_ms": ms})
            yield sse("token", {"token": answer})
        else:
            parts: list[str] = []
            ttft: float | None = None
            metrics: dict[str, Any] = {}
            try:
                async for kind, value in _generate(rt, rv.plan, t0):
                    if kind == "ttft":
                        ttft = value
                        yield sse("ttft", {"ttft_ms": value})
                    elif kind == "token":
                        parts.append(value)
                        yield sse("token", {"token": value})
                    else:
                        metrics = value
            except ServiceDegraded as exc:
                yield sse("error", {"code": exc.code, "retry_after_s": round(exc.retry_after_s, 1)})
                return
            except GenerationError as exc:
                yield sse("error", {"code": "UPSTREAM_FAILURE", "message": str(exc)[:300]})
                return
            core = _core(rt, rv, answer="".join(parts), bypassed=False, metrics=metrics, ttft_ms=ttft)
        if req.use_cache and core["answer"]:
            rt.cache.put(key, core)
        d = _response_dict(core, cache_hit=False, total_ms=(time.perf_counter() - t0) * 1000.0, endpoint="async")
        yield sse("metrics", _summary(d))
        yield sse("done", {"cache_hit": False})
    finally:
        if handle is not None:
            handle.release()


async def _replay_frames(core: dict[str, Any], d: dict[str, Any]) -> AsyncIterator[str]:
    yield sse("ttft", {"ttft_ms": d["ttft_ms"]})
    for piece in _chunk_text(core["answer"]):
        yield sse("token", {"token": piece})
    yield sse("metrics", _summary(d))
    yield sse("done", {"cache_hit": True})


def _cached_response(core: dict[str, Any], req: QueryRequest, t0: float) -> Any:
    d = _response_dict(core, cache_hit=True, total_ms=(time.perf_counter() - t0) * 1000.0, endpoint="async")
    if not req.stream:
        return _json_response(d)
    return StreamingResponse(
        _replay_frames(core, d), media_type="text/event-stream",
        headers={**_diag_headers(d), "Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


def _verify_corpus(paths: list[Path]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for path in paths:
        digest = hashlib.sha256()
        size = 0
        with path.open("rb") as fh:
            for block in iter(lambda: fh.read(1 << 20), b""):
                digest.update(block)
                size += len(block)
        out[path.name] = {"sha256": digest.hexdigest(), "bytes": size}
    return out


async def _run_task(rt: Runtime, job_id: str, task: str) -> None:
    rt.tasks.update(job_id, status="running", started_at=utc_now())
    try:
        result = await asyncio.to_thread(_verify_corpus, rt.corpus_paths)
    except Exception as exc:
        rt.tasks.update(job_id, status="failed", finished_at=utc_now(), error=f"{type(exc).__name__}: {exc}")
        return
    rt.tasks.update(job_id, status="succeeded", finished_at=utc_now(), result={"task": task, "files": result})


def build_default_runtime() -> Runtime:
    from kavach.config import get_settings
    from kavach.retrieval.pipeline import VectorRetrievalPipeline

    settings = get_settings()
    gen = get_gen_settings()
    sampler = VRAMSampler(gen.vram_hard_limit_mib, 1.0).start()
    pipeline = VectorRetrievalPipeline(settings)
    generator = OllamaGenerator(gen, probe=sampler.latest)
    counter = QwenTokenCounter(gen.tokenizer_id, cache_dir=settings.models_dir / "hf_cache")
    paths = [
        settings.data_processed / "corpus_chunks.parquet",
        settings.models_dir / "index_sq8_mrl512.faiss",
        settings.models_dir / "index_sq8_mrl512.idmap.json",
    ]
    return Runtime(pipeline=pipeline, generator=generator, counter=counter, sampler=sampler, gen=gen,
                   corpus_paths=paths)


def _prewarm_generator(rt: Runtime) -> None:
    try:
        rt.generator.generate([{"role": "user", "content": "Reply with OK."}], num_predict=4)
    except GenerationError as exc:
        log.warning("Generator prewarm failed; serving starts degraded: %s", exc)


@asynccontextmanager
async def lifespan(app: FastAPI):
    owned = app.state.runtime is None
    if owned:
        app.state.runtime = await asyncio.to_thread(build_default_runtime)
        await asyncio.to_thread(_prewarm_generator, app.state.runtime)
    yield
    rt: Runtime = app.state.runtime
    if owned:
        rt.sampler.stop()
        await rt.generator.aclose()
        rt.generator.close()


def _rt(request: Request) -> Runtime:
    rt = request.app.state.runtime
    if rt is None:
        raise ServiceDegraded("SERVER_STARTING", "Runtime is still loading", 5.0)
    return rt


def create_app(runtime: Runtime | None = None) -> FastAPI:
    app = FastAPI(title="KAVACH-RAG Serving", version="phase3", lifespan=lifespan)
    app.state.runtime = runtime

    @app.exception_handler(ServiceDegraded)
    async def _degraded(request: Request, exc: ServiceDegraded) -> JSONResponse:
        return JSONResponse(
            status_code=503,
            content={"status": exc.code, "detail": exc.message, "retry_after_s": round(exc.retry_after_s, 1)},
            headers={"Retry-After": str(max(1, math.ceil(exc.retry_after_s)))},
        )

    @app.exception_handler(PromptBudgetError)
    async def _budget(request: Request, exc: PromptBudgetError) -> JSONResponse:
        return JSONResponse(status_code=500, content={"status": "PROMPT_BUDGET_ERROR", "detail": str(exc)})

    @app.exception_handler(GenerationError)
    async def _upstream(request: Request, exc: GenerationError) -> JSONResponse:
        return JSONResponse(status_code=502, content={"status": "UPSTREAM_FAILURE", "detail": str(exc)[:300]})

    @app.get("/health")
    async def health(request: Request) -> JSONResponse:
        rt = request.app.state.runtime
        if rt is None:
            body = {"status": "starting"}
            return JSONResponse(status_code=503, content=body)
        snap = rt.sampler.snapshot()
        latest = snap["latest"]
        breaker = rt.breaker.snapshot()
        degraded = breaker["state"] != "closed" or bool(snap["breached"])
        model = HealthResponse(
            status="degraded" if degraded else "ok",
            pipeline_ready=True,
            generator_ready=True,
            circuit_state=breaker["state"],
            circuit=breaker,
            cache=rt.cache.stats(),
            vram_used_mib=None if latest is None else latest[0],
            vram_total_mib=None if latest is None else latest[1],
            vram_peak_mib=snap["peak_mib"],
            vram_hard_limit_mib=rt.gen.vram_hard_limit_mib,
            vram_breached=bool(snap["breached"]),
            rss_mb=round(psutil.Process().memory_info().rss / 1048576.0, 1),
            retrieval_waiting=rt.retrieval_waiting,
            gpu_waiting=rt.gpu_waiting,
            gpu_inflight=rt.gpu_inflight,
            generator_calls=rt.generator.calls,
            uptime_s=round(time.monotonic() - rt.started_at, 1),
        )
        return JSONResponse(model.model_dump())

    @app.post("/rag/sync")
    def rag_sync(req: QueryRequest, request: Request) -> JSONResponse:
        """Intentional baseline: blocking, no cache, no breaker, no semaphores, ignores `stream`."""
        rt = _rt(request)
        t0 = time.perf_counter()
        res, plan, lock_ms = _retrieve_and_plan(rt, req.query)
        rv = Retrieved(res, plan, lock_ms, _reason(res))
        if plan is None:
            ms = round((time.perf_counter() - t0) * 1000.0, 3)
            core = _core(rt, rv, answer=canonical_refusal(rv.reason), bypassed=True, metrics={}, ttft_ms=ms)
        else:
            t_gen = time.perf_counter()
            out = rt.generator.generate(plan.messages)
            ttft = (t_gen - t0) * 1000.0 + float(out.metrics.ttft_ms or 0.0)
            core = _core(rt, rv, answer=out.text, bypassed=False, metrics=out.metrics.to_dict(),
                         ttft_ms=round(ttft, 3))
        d = _response_dict(core, cache_hit=False, total_ms=(time.perf_counter() - t0) * 1000.0, endpoint="sync")
        return _json_response(d)

    @app.post("/rag/async")
    async def rag_async(req: QueryRequest, request: Request) -> Any:
        rt = _rt(request)
        t0 = time.perf_counter()
        key = cache_key(req.query)
        if req.use_cache:
            hit = rt.cache.lookup(key)
            if hit is not None:
                return _cached_response(hit, req, t0)

        if not req.stream:
            if req.use_cache:
                core, was_hit = await rt.cache.get_or_compute(
                    key, lambda: _compute_core(rt, req.query),
                    cacheable=lambda value: bool(value["answer"]), lock_timeout_s=CACHE_LOCK_TIMEOUT_S,
                )
            else:
                core, was_hit = await _compute_core(rt, req.query), False
            d = _response_dict(core, cache_hit=was_hit, total_ms=(time.perf_counter() - t0) * 1000.0,
                               endpoint="async")
            return _json_response(d)

        handle = None
        if req.use_cache:
            handle = await rt.cache.acquire(key, CACHE_LOCK_TIMEOUT_S)
            if handle is not None:
                coalesced = rt.cache.coalesced_hit(key)
                if coalesced is not None:
                    handle.release()
                    return _cached_response(coalesced, req, t0)
        try:
            rv = await _retrieve_async(rt, req.query)
        except BaseException:
            if handle is not None:
                handle.release()
            raise
        headers = {
            "X-Retrieval-Latency-MS": f"{float(rv.result.latency_ms.get('total_ms', 0.0)):.3f}",
            "X-Retrieval-Queue-MS": f"{rv.queue_ms:.3f}",
            "X-Cache": "MISS",
            "X-Endpoint": "async",
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        }
        return StreamingResponse(
            _stream_live(rt, req, key, handle, rv, t0),
            media_type="text/event-stream", headers=headers,
            background=BackgroundTask(handle.release) if handle is not None else None,
        )

    @app.post("/tasks/ingest", status_code=202)
    async def ingest(req: IngestionTaskRequest, background: BackgroundTasks, request: Request) -> JSONResponse:
        rt = _rt(request)
        job = rt.tasks.create(req.task, req.label)
        background.add_task(_run_task, rt, job["job_id"], req.task)
        return JSONResponse(status_code=202, content=TaskStatusResponse(**job).model_dump())

    @app.get("/tasks/{job_id}")
    async def task_status(job_id: str, request: Request) -> JSONResponse:
        rt = _rt(request)
        job = rt.tasks.get(job_id)
        if job is None:
            return JSONResponse(status_code=404, content={"status": "NOT_FOUND", "detail": job_id})
        return JSONResponse(TaskStatusResponse(**job).model_dump())

    return app


app = create_app()
