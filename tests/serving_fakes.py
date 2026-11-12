"""Shared fakes for serving tests: scripted Ollama transport, fake pipeline, runtime rig."""
from __future__ import annotations

import asyncio
import json
import random
import threading
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import httpx

from kavach.generation.gen_config import GenSettings
from kavach.generation.generator import OllamaGenerator
from kavach.serving.app import Runtime, create_app
from kavach.serving.resilience import CircuitBreaker, MutexTTLCache, VRAMSampler

CHUNK = {
    "chunk_id": "CVE-2024-3094_chunk_001",
    "doc_id": "CVE-2024-3094",
    "content": "CVE-2024-3094 CVSS 10.0 vector CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:C/C:H/I:H/A:H liblzma 5.6.0 5.6.1",
    "standard_version": "NVD_5.0",
}


def ollama_frames(tokens: list[str]) -> bytes:
    lines = [json.dumps({"message": {"role": "assistant", "content": t}, "done": False}) for t in tokens]
    lines.append(json.dumps({
        "message": {"content": ""}, "done": True, "done_reason": "stop", "load_duration": 2_000_000,
        "prompt_eval_count": 120, "prompt_eval_duration": 50_000_000,
        "eval_count": 40, "eval_duration": 1_000_000_000,
    }))
    return ("\n".join(lines) + "\n").encode()


class FakeClock:
    def __init__(self) -> None:
        self.t = 1000.0

    def __call__(self) -> float:
        return self.t

    def advance(self, seconds: float) -> None:
        self.t += seconds


class ScriptedOllama:
    """mode: ok | http500 | http429 | connect_error. Counts /api/chat calls and in-flight depth."""

    def __init__(self) -> None:
        self.tokens = ["Score ", "is ", "10.0 ", "[CVE-2024-3094_chunk_001]"]
        self.mode = "ok"
        self.calls = 0
        self.inflight = 0
        self.max_inflight = 0
        self.delay_s = 0.0

    def _respond(self) -> httpx.Response:
        self.calls += 1
        if self.mode == "connect_error":
            raise httpx.ConnectError("scripted connection refused")
        if self.mode == "http500":
            return httpx.Response(500, text="scripted failure")
        if self.mode == "http429":
            return httpx.Response(429, headers={"retry-after": "1"}, text="scripted rate limit")
        return httpx.Response(200, content=ollama_frames(self.tokens))

    def __call__(self, request: httpx.Request) -> httpx.Response:
        return self._respond()

    async def async_call(self, request: httpx.Request) -> httpx.Response:
        self.inflight += 1
        self.max_inflight = max(self.max_inflight, self.inflight)
        try:
            if self.delay_s:
                await asyncio.sleep(self.delay_s)
            return self._respond()
        finally:
            self.inflight -= 1


class FakePipeline:
    """Queries starting 'oos:' are refused, 'lowconf:' abstain, everything else is served."""

    def __init__(self, block: threading.Event | None = None, started: threading.Event | None = None) -> None:
        self.block = block
        self.started = started
        self.calls = 0
        self.active = 0
        self.max_active = 0
        self._lock = threading.Lock()

    def retrieve(self, query: str, **kwargs: Any) -> Any:
        with self._lock:
            self.calls += 1
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        try:
            if self.started is not None:
                self.started.set()
            if self.block is not None:
                self.block.wait(30.0)
            refused = query.startswith("oos:")
            abstained = query.startswith("lowconf:")
            route = "HARD_REFUSAL" if refused else "SEMANTIC_RAG"
            served = not (refused or abstained)
            return SimpleNamespace(
                route=SimpleNamespace(route=route), refused=refused, abstained=abstained,
                abstain_reason="LOW_RETRIEVAL_CONFIDENCE" if abstained else None,
                top1_logit=None if refused else (9.5 if served else 3.0),
                contexts=[dict(CHUNK)] if served else [], candidates=[dict(CHUNK)] if served else [],
                latency_ms={"total_ms": 12.5}, contexts_if_not_gated=[dict(CHUNK)] if served else [],
            )
        finally:
            with self._lock:
                self.active -= 1


@dataclass
class Rig:
    app: Any
    rt: Runtime
    ollama: ScriptedOllama
    pipeline: FakePipeline
    clock: FakeClock


def make_rig(*, pipeline: FakePipeline | None = None, gen_attempts: int = 1,
             counter: Any = None, vram_used: int = 3000) -> Rig:
    ollama = ScriptedOllama()
    pipe = pipeline or FakePipeline()
    clock = FakeClock()
    gen = GenSettings(max_attempts=gen_attempts)
    generator = OllamaGenerator(
        gen, transport=httpx.MockTransport(ollama), async_transport=httpx.MockTransport(ollama.async_call),
        probe=lambda: (vram_used, 6141), sleep=lambda seconds: None, rng=random.Random(1),
    )
    sampler = VRAMSampler(gen.vram_hard_limit_mib, 1.0, probe=lambda: (vram_used, 6141))
    sampler.sample_once()
    rt = Runtime(
        pipeline=pipe, generator=generator, counter=counter or (lambda text: len(text.split())),
        sampler=sampler, gen=gen, cache=MutexTTLCache(), breaker=CircuitBreaker(clock=clock),
    )
    return Rig(create_app(rt), rt, ollama, pipe, clock)


def client(app: Any) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test", timeout=60.0)
