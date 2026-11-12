"""Pydantic v2 request/response contracts for the Phase 3 serving layer."""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class QueryRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1, max_length=8000)
    stream: bool = False
    use_cache: bool = True


class QueryResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    answer: str
    route: str
    bypassed_llm: bool
    refused: bool
    abstained: bool
    top1_logit: float | None
    retrieved_chunk_ids: list[str]
    dropped_chunk_ids: list[str]
    retrieval_latency_ms: float
    retrieval_queue_ms: float
    ttft_ms: float | None
    total_ms: float
    prompt_tokens: int
    completion_tokens: int
    decode_tps: float | None
    finish_reason: str | None
    cache_hit: bool
    endpoint: Literal["sync", "async"]
    prompt_version: str


class IngestionTaskRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task: Literal["verify_corpus"] = "verify_corpus"
    label: str | None = Field(default=None, max_length=80)


class TaskStatusResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    job_id: str
    task: str
    label: str | None
    status: Literal["pending", "running", "succeeded", "failed"]
    created_at: str
    started_at: str | None
    finished_at: str | None
    result: dict[str, Any] | None
    error: str | None


class HealthResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: Literal["ok", "degraded", "starting"]
    pipeline_ready: bool
    generator_ready: bool
    circuit_state: str
    circuit: dict[str, Any]
    cache: dict[str, Any]
    vram_used_mib: int | None
    vram_total_mib: int | None
    vram_peak_mib: int | None
    vram_hard_limit_mib: int
    vram_breached: bool
    rss_mb: float
    retrieval_waiting: int
    gpu_waiting: int
    gpu_inflight: int
    generator_calls: int
    uptime_s: float
