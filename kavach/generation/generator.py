"""Ollama HTTP client, SSE streaming wrapper, Full Jitter retry, and VRAM watchdog."""
from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable, Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass, field
import json
import logging
import random
import subprocess
import threading
import time
from typing import Any

import httpx

from kavach.exceptions import (
    GenerationError,
    OllamaUnavailableError,
    VRAMBudgetExceededError,
)
from kavach.generation.gen_config import GenSettings, get_gen_settings

log = logging.getLogger(__name__)

_RETRYABLE_STATUS_CODES = frozenset({408, 429, 500, 502, 503, 504})


def full_jitter_delay(
    attempt: int,
    base_s: float = 1.5,
    cap_s: float = 30.0,
    rng: random.Random | None = None,
) -> float:
    """Compute AWS Full Jitter backoff delay in seconds for a 0-indexed attempt."""
    safe_attempt = max(0, min(int(attempt), 30))
    safe_base = max(0.0, float(base_s))
    safe_cap = max(0.0, float(cap_s))
    ceiling = min(safe_cap, safe_base * (2**safe_attempt))
    if ceiling <= 0.0:
        return 0.0
    if rng is not None:
        return float(rng.uniform(0.0, ceiling))
    return float(random.uniform(0.0, ceiling))


def probe_vram_mib() -> tuple[int, int] | None:
    """Return ``(memory_used_mib, memory_total_mib)`` via nvidia-smi, or None if unavailable."""
    try:
        proc = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.used,memory.total",
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

    for raw_line in proc.stdout.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 2:
            continue
        try:
            return int(float(parts[0])), int(float(parts[1]))
        except ValueError:
            continue
    return None


@dataclass
class VRAMStats:
    """Summary telemetry collected by :class:`VRAMWatchdog`."""

    baseline_mib: int | None
    peak_mib: int | None
    samples: int
    breached: bool
    mean_mib: float | None = None

    @property
    def over_hard_limit(self) -> bool:
        return self.breached

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def __call__(self) -> VRAMStats:
        return self


class VRAMWatchdog:
    """Background nvidia-smi poller monitoring GPU VRAM against a hard limit."""

    def __init__(
        self,
        hard_limit_mib: int,
        interval_s: float = 0.5,
        probe: Callable[[], tuple[int, int] | None] = probe_vram_mib,
        *,
        target_mib: int | None = None,
        poll_interval_s: float | None = None,
    ) -> None:
        self.hard_limit_mib: int = int(hard_limit_mib)
        self.target_mib: int | None = int(target_mib) if target_mib is not None else None
        effective_interval = poll_interval_s if poll_interval_s is not None else interval_s
        self.interval_s: float = max(0.01, float(effective_interval))
        self._probe: Callable[[], tuple[int, int] | None] = probe
        self.breached: threading.Event = threading.Event()

        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._baseline_mib: int | None = None
        self._peak_mib: int | None = None
        self._sum_mib: int = 0
        self._samples: int = 0
        self._warned_target = False

    def _record(self, sample: tuple[int, int] | int | None, *, is_baseline: bool = False) -> int | None:
        if sample is None:
            return None
        used = int(sample[0]) if isinstance(sample, tuple) else int(sample)
        with self._lock:
            if is_baseline and self._baseline_mib is None:
                self._baseline_mib = used
            if self._peak_mib is None or used > self._peak_mib:
                self._peak_mib = used
            self._sum_mib += used
            self._samples += 1
            if self.target_mib is not None and used > self.target_mib and not self._warned_target:
                self._warned_target = True
                log.warning(
                    "VRAM usage %d MiB exceeded target %d MiB.",
                    used,
                    self.target_mib,
                )
            if used > self.hard_limit_mib:
                self.breached.set()
        return used

    def check_once(self) -> int | None:
        """Synchronously sample VRAM and update watchdog statistics."""
        try:
            sample = self._probe()
        except Exception as exc:
            log.debug("VRAM probe raised exception: %s", exc)
            return None
        return self._record(sample)

    def _poll_loop(self) -> None:
        while not self._stop.wait(self.interval_s):
            self.check_once()

    def start(self) -> VRAMWatchdog:
        """Sample initial baseline VRAM and start background polling."""
        if self._thread is not None and self._thread.is_alive():
            return self
        self._stop.clear()
        try:
            initial = self._probe()
        except Exception:
            initial = None
        self._record(initial, is_baseline=True)
        self._thread = threading.Thread(
            target=self._poll_loop,
            name="kavach-vram-watchdog",
            daemon=True,
        )
        self._thread.start()
        return self

    def stop(self) -> VRAMStats:
        """Stop background polling, take a final probe, and return VRAMStats."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(1.0, self.interval_s * 3.0))
            self._thread = None
        self.check_once()
        return self._build_stats()

    def _build_stats(self) -> VRAMStats:
        with self._lock:
            mean_mib = (
                round(self._sum_mib / self._samples, 2)
                if self._samples > 0
                else None
            )
            return VRAMStats(
                baseline_mib=self._baseline_mib,
                peak_mib=self._peak_mib,
                samples=self._samples,
                breached=self.breached.is_set(),
                mean_mib=mean_mib,
            )

    @property
    def stats(self) -> VRAMStats:
        """Return current VRAMStats (callable as well via VRAMStats.__call__)."""
        return self._build_stats()

    def __enter__(self) -> VRAMWatchdog:
        return self.start()

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.stop()


@dataclass
class GenerationMetrics:
    """Latency, token throughput, and VRAM telemetry for a single generation call."""

    ttft_ms: float | None
    total_ms: float
    server_load_ms: float
    server_prefill_ms: float
    prompt_tokens: int
    completion_tokens: int
    decode_tps: float | None
    finish_reason: str | None
    vram_baseline_mib: int | None = None
    vram_peak_mib: int | None = None
    attempts: int = 1

    @property
    def load_ms(self) -> float:
        return self.server_load_ms

    @property
    def prompt_eval_ms(self) -> float:
        return self.server_prefill_ms

    @property
    def tokens_per_s(self) -> float | None:
        return self.decode_tps

    @property
    def done_reason(self) -> str | None:
        return self.finish_reason

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class GenerationResult:
    """Completed generation output and telemetry."""

    text: str
    metrics: GenerationMetrics
    model: str

    @property
    def done_reason(self) -> str | None:
        return self.metrics.finish_reason

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class _Retryable(Exception):
    """Internal exception signaling a transient retryable HTTP/network failure."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        retry_after: float | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.retry_after = retry_after
        self.message = message


@dataclass
class _StreamState:
    """Mutable state tracked across NDJSON chunks in an Ollama stream."""

    model: str
    t_start: float
    attempts: int = 1
    first_token_ms: float | None = None
    chunks: list[str] = field(default_factory=list)
    emitted_any_token: bool = False
    done: bool = False
    result: GenerationResult | None = None


def _sse(event: str, data: Any) -> str:
    """Format a single Server-Sent Event frame."""
    payload = data if isinstance(data, str) else json.dumps(data, ensure_ascii=False)
    lines = payload.splitlines() or [""]
    data_block = "\n".join(f"data: {line}" for line in lines)
    return f"event: {event}\n{data_block}\n\n"


def _ns_to_ms(value: Any) -> float:
    try:
        return round(float(value) / 1_000_000.0, 3)
    except (TypeError, ValueError):
        return 0.0


def _parse_retry_after(headers: Mapping[str, str]) -> float | None:
    raw = headers.get("retry-after")
    if raw is None:
        return None
    try:
        return max(0.0, float(raw.strip()))
    except ValueError:
        return None


def _build_metrics(
    raw: Mapping[str, Any],
    *,
    total_ms: float,
    ttft_ms: float | None,
    attempts: int,
    vram_baseline_mib: int | None = None,
    vram_peak_mib: int | None = None,
) -> GenerationMetrics:
    server_load_ms = _ns_to_ms(raw.get("load_duration"))
    server_prefill_ms = _ns_to_ms(raw.get("prompt_eval_duration"))
    eval_ms = _ns_to_ms(raw.get("eval_duration"))
    prompt_tokens = int(raw.get("prompt_eval_count") or 0)
    completion_tokens = int(raw.get("eval_count") or 0)
    raw_reason = raw.get("done_reason")
    finish_reason = str(raw_reason) if raw_reason is not None else None

    if eval_ms > 0.0 and completion_tokens > 0:
        decode_tps: float | None = round(completion_tokens / (eval_ms / 1000.0), 3)
    elif total_ms > 0.0 and completion_tokens > 0:
        decode_tps = round(completion_tokens / (total_ms / 1000.0), 3)
    else:
        decode_tps = None

    return GenerationMetrics(
        ttft_ms=round(ttft_ms, 3) if ttft_ms is not None else None,
        total_ms=round(total_ms, 3),
        server_load_ms=server_load_ms,
        server_prefill_ms=server_prefill_ms,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        decode_tps=decode_tps,
        finish_reason=finish_reason,
        vram_baseline_mib=vram_baseline_mib,
        vram_peak_mib=vram_peak_mib,
        attempts=attempts,
    )


class OllamaGenerator:
    """Sync and async Ollama `/api/chat` client with Full Jitter retry, SSE, & VRAM guard."""

    def __init__(
        self,
        s: GenSettings | None = None,
        *,
        transport: httpx.BaseTransport | None = None,
        async_transport: httpx.AsyncBaseTransport | None = None,
        probe: Callable[[], tuple[int, int] | None] = probe_vram_mib,
        sleep: Callable[[float], Any] = time.sleep,
        rng: random.Random | None = None,
    ) -> None:
        self.s: GenSettings = s or get_gen_settings()
        self.base_url: str = self.s.ollama_base_url.rstrip("/")
        self._timeout = httpx.Timeout(
            connect=self.s.connect_timeout_s,
            read=self.s.read_timeout_s,
            write=self.s.connect_timeout_s,
            pool=self.s.connect_timeout_s,
        )
        self._client = httpx.Client(timeout=self._timeout, transport=transport)
        self._async_client = httpx.AsyncClient(
            timeout=self._timeout,
            transport=async_transport,
        )
        self._probe = probe
        self._sleep = sleep
        self._rng = rng
        self.calls: int = 0

    def close(self) -> None:
        """Close the synchronous HTTPX client."""
        self._client.close()

    async def aclose(self) -> None:
        """Close the asynchronous HTTPX client."""
        await self._async_client.aclose()

    def __enter__(self) -> OllamaGenerator:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

    async def __aenter__(self) -> OllamaGenerator:
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        await self.aclose()

    def _payload(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        stream: bool,
        num_predict: int | None = None,
        response_format: dict[str, Any] | str | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.s.model,
            "messages": [
                {"role": str(m["role"]), "content": str(m["content"])}
                for m in messages
            ],
            "stream": bool(stream),
            "keep_alive": self.s.keep_alive,
            "options": {
                "num_ctx": int(self.s.num_ctx),
                "num_predict": int(
                    num_predict if num_predict is not None else self.s.num_predict
                ),
                "temperature": float(self.s.temperature),
                "top_p": float(self.s.top_p),
                "repeat_penalty": float(self.s.repeat_penalty),
                "seed": int(self.s.seed),
            },
        }
        if response_format is not None:
            payload["format"] = response_format
        return payload

    def _backoff(self, attempt: int, retry_after: float | None = None) -> float:
        jitter = full_jitter_delay(
            attempt,
            base_s=self.s.backoff_base_s,
            cap_s=self.s.backoff_cap_s,
            rng=self._rng,
        )
        delay = max(jitter, retry_after) if retry_after is not None else jitter
        self._sleep(delay)
        return delay

    async def _abackoff(self, attempt: int, retry_after: float | None = None) -> float:
        jitter = full_jitter_delay(
            attempt,
            base_s=self.s.backoff_base_s,
            cap_s=self.s.backoff_cap_s,
            rng=self._rng,
        )
        delay = max(jitter, retry_after) if retry_after is not None else jitter
        if self._sleep is not time.sleep:
            self._sleep(delay)
        elif delay > 0.0:
            await asyncio.sleep(delay)
        return delay

    def _check_status(
        self,
        response: httpx.Response,
        body_text: str | None = None,
    ) -> None:
        if 200 <= response.status_code < 300:
            return
        if body_text is None:
            try:
                body_text = response.text[:500]
            except Exception:
                body_text = ""
        if response.status_code in _RETRYABLE_STATUS_CODES:
            raise _Retryable(
                f"Ollama HTTP {response.status_code}: {body_text}",
                status_code=response.status_code,
                retry_after=_parse_retry_after(response.headers),
            )
        raise GenerationError(
            f"Ollama returned HTTP {response.status_code}: {body_text}",
            details={"status_code": response.status_code, "body": body_text},
        )

    def _events(self, chunk: Mapping[str, Any], state: _StreamState) -> list[str]:
        if chunk.get("error"):
            raise GenerationError(
                f"Ollama stream returned error: {chunk['error']}",
                details={"model": state.model, "error": chunk["error"]},
            )

        msg = chunk.get("message")
        token = str(msg.get("content") or "") if isinstance(msg, Mapping) else str(chunk.get("response") or "")

        out: list[str] = []
        if token:
            if state.first_token_ms is None:
                state.first_token_ms = (time.perf_counter() - state.t_start) * 1000.0
            state.chunks.append(token)
            state.emitted_any_token = True
            out.append(_sse("token", {"token": token}))

        if chunk.get("done"):
            total_ms = (time.perf_counter() - state.t_start) * 1000.0
            metrics = _build_metrics(
                chunk,
                total_ms=total_ms,
                ttft_ms=state.first_token_ms,
                attempts=state.attempts,
            )
            result = GenerationResult(
                text="".join(state.chunks),
                metrics=metrics,
                model=str(chunk.get("model") or state.model),
            )
            state.done = True
            state.result = result
            out.append(_sse("metrics", metrics.to_dict()))
        return out

    def _run_stream(
        self,
        payload: dict[str, Any],
    ) -> Iterator[tuple[_StreamState, list[str]]]:
        url = f"{self.base_url}/api/chat"
        max_attempts = max(1, int(self.s.max_attempts))

        for attempt in range(max_attempts):
            state = _StreamState(
                model=payload["model"],
                t_start=time.perf_counter(),
                attempts=attempt + 1,
            )
            self.calls += 1
            try:
                with self._client.stream("POST", url, json=payload) as response:
                    if not (200 <= response.status_code < 300):
                        body = response.read().decode("utf-8", errors="replace")[:500]
                        self._check_status(response, body_text=body)
                    for raw_line in response.iter_lines():
                        line = raw_line.strip()
                        if not line:
                            continue
                        try:
                            chunk = json.loads(line)
                        except ValueError as exc:
                            raise GenerationError(
                                f"Malformed NDJSON chunk from Ollama: {exc}",
                                details={"line": line[:200]},
                            ) from exc
                        if not isinstance(chunk, Mapping):
                            raise GenerationError(
                                f"Expected JSON object chunk, got {type(chunk).__name__}",
                            )
                        frames = self._events(chunk, state)
                        yield state, frames
                        if state.done:
                            return
                if not state.done:
                    raise GenerationError(
                        "Ollama stream ended before final done=true event.",
                        details={"model": state.model},
                    )
                return
            except _Retryable as exc:
                if not state.emitted_any_token and attempt + 1 < max_attempts:
                    self._backoff(attempt, retry_after=exc.retry_after)
                    continue
                raise GenerationError(
                    f"Ollama stream failed: {exc.message}",
                    details={"status_code": exc.status_code, "attempts": attempt + 1},
                ) from exc
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                if not state.emitted_any_token and attempt + 1 < max_attempts:
                    self._backoff(attempt)
                    continue
                raise OllamaUnavailableError(
                    f"Ollama stream unreachable at {url}: {exc}",
                    details={"base_url": self.base_url, "error": str(exc)},
                ) from exc

    def generate(
        self,
        messages: list[dict],
        *,
        num_predict: int | None = None,
        response_format: dict | None = None,
    ) -> GenerationResult:
        """Execute a synchronous `/api/chat` generation request guarded by VRAMWatchdog."""
        payload = self._payload(
            messages,
            stream=True,
            num_predict=num_predict,
            response_format=response_format,
        )

        final_state: _StreamState | None = None
        wd = VRAMWatchdog(
            self.s.vram_hard_limit_mib,
            self.s.vram_poll_interval_s,
            probe=self._probe,
            target_mib=self.s.vram_target_mib,
        )
        wd.start()
        try:
            for state, _ in self._run_stream(payload):
                final_state = state
                if wd.breached.is_set():
                    raise VRAMBudgetExceededError(
                        f"GPU VRAM exceeded hard limit {self.s.vram_hard_limit_mib} MiB.",
                        details=wd.stats().to_dict(),
                    )
        finally:
            wd_stats = wd.stop()

        if wd_stats.breached:
            raise VRAMBudgetExceededError(
                f"GPU VRAM peak {wd_stats.peak_mib} MiB exceeded hard limit {self.s.vram_hard_limit_mib} MiB.",
                details=wd_stats.to_dict(),
            )

        if final_state is None or final_state.result is None:
            raise GenerationError(
                "Ollama stream did not yield a final result.",
                details={"model": payload["model"]},
            )

        result = final_state.result
        result.metrics.vram_baseline_mib = wd_stats.baseline_mib
        result.metrics.vram_peak_mib = wd_stats.peak_mib
        return result

    def stream(
        self,
        messages: list[dict],
        *,
        num_predict: int | None = None,
        response_format: dict | None = None,
    ) -> Iterator[str]:
        """Synchronously stream Server-Sent Events (`token`, then final `metrics`)."""
        payload = self._payload(
            messages,
            stream=True,
            num_predict=num_predict,
            response_format=response_format,
        )
        for _, frames in self._run_stream(payload):
            for sse_frame in frames:
                yield sse_frame

    async def astream(
        self,
        messages: list[dict],
        *,
        num_predict: int | None = None,
        response_format: dict | None = None,
    ) -> AsyncIterator[str]:
        """Asynchronously stream Server-Sent Events (`token`, then final `metrics`)."""
        payload = self._payload(
            messages,
            stream=True,
            num_predict=num_predict,
            response_format=response_format,
        )
        url = f"{self.base_url}/api/chat"
        max_attempts = max(1, int(self.s.max_attempts))

        for attempt in range(max_attempts):
            state = _StreamState(
                model=payload["model"],
                t_start=time.perf_counter(),
                attempts=attempt + 1,
            )
            self.calls += 1
            try:
                async with self._async_client.stream("POST", url, json=payload) as response:
                    if not (200 <= response.status_code < 300):
                        raw_bytes = await response.aread()
                        body = raw_bytes.decode("utf-8", errors="replace")[:500]
                        self._check_status(response, body_text=body)
                    async for raw_line in response.aiter_lines():
                        line = raw_line.strip()
                        if not line:
                            continue
                        try:
                            chunk = json.loads(line)
                        except ValueError as exc:
                            raise GenerationError(
                                f"Malformed NDJSON chunk from Ollama: {exc}",
                                details={"line": line[:200]},
                            ) from exc
                        if not isinstance(chunk, Mapping):
                            raise GenerationError(
                                f"Expected JSON object chunk, got {type(chunk).__name__}",
                            )
                        for sse_frame in self._events(chunk, state):
                            yield sse_frame
                        if state.done:
                            return
                if not state.done:
                    raise GenerationError(
                        "Ollama async stream ended before final done=true event.",
                        details={"model": state.model},
                    )
                return
            except _Retryable as exc:
                if not state.emitted_any_token and attempt + 1 < max_attempts:
                    await self._abackoff(attempt, retry_after=exc.retry_after)
                    continue
                raise GenerationError(
                    f"Ollama async stream failed: {exc.message}",
                    details={"status_code": exc.status_code, "attempts": attempt + 1},
                ) from exc
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                if not state.emitted_any_token and attempt + 1 < max_attempts:
                    await self._abackoff(attempt)
                    continue
                if state.emitted_any_token:
                    yield _sse("error", {"error": str(exc)})
                raise OllamaUnavailableError(
                    f"Ollama async stream unreachable at {url}: {exc}",
                    details={"base_url": self.base_url, "error": str(exc)},
                ) from exc

    def _json(
        self,
        method: str,
        path: str,
        body: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        url = f"{self.base_url}{path}"
        max_attempts = max(1, int(self.s.max_attempts))
        last_exc: Exception | None = None

        for attempt in range(max_attempts):
            self.calls += 1
            try:
                kwargs: dict[str, Any] = {}
                if body is not None:
                    kwargs["json"] = body
                response = self._client.request(method, url, **kwargs)
                self._check_status(response)
                try:
                    data = response.json()
                except ValueError as exc:
                    raise GenerationError(
                        f"Malformed JSON from Ollama {path}: {exc}",
                        details={"path": path, "body": response.text[:500]},
                    ) from exc
                if not isinstance(data, dict):
                    raise GenerationError(
                        f"Expected JSON object from Ollama {path}, got {type(data).__name__}",
                        details={"path": path},
                    )
                return data
            except _Retryable as exc:
                last_exc = exc
                if attempt + 1 < max_attempts:
                    self._backoff(attempt, retry_after=exc.retry_after)
                    continue
                raise GenerationError(
                    f"Ollama {path} failed after {max_attempts} attempts: {exc.message}",
                    details={
                        "path": path,
                        "status_code": exc.status_code,
                        "attempts": max_attempts,
                    },
                ) from exc
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                last_exc = exc
                if attempt + 1 < max_attempts:
                    self._backoff(attempt)
                    continue
                raise OllamaUnavailableError(
                    f"Ollama unreachable at {self.base_url}{path} after {max_attempts} attempts: {exc}",
                    details={
                        "base_url": self.base_url,
                        "path": path,
                        "attempts": max_attempts,
                        "error": str(exc),
                    },
                ) from exc

        raise OllamaUnavailableError(
            f"Ollama request to {path} failed: {last_exc}",
            details={"base_url": self.base_url, "path": path},
        )

    def version(self) -> dict[str, Any]:
        """Query Ollama `/api/version`."""
        return self._json("GET", "/api/version")

    def list_models(self) -> list[dict[str, Any]]:
        """Query Ollama `/api/tags` and return the list of model dicts."""
        data = self._json("GET", "/api/tags")
        models = data.get("models") or []
        return [dict(m) for m in models if isinstance(m, Mapping)]

    def show(self, model: str) -> dict[str, Any]:
        """Query Ollama `/api/show` for model metadata."""
        return self._json("POST", "/api/show", {"model": model})

    def ps(self) -> list[dict[str, Any]]:
        """Query Ollama `/api/ps` and return the list of currently resident model dicts."""
        data = self._json("GET", "/api/ps")
        models = data.get("models") or []
        return [dict(m) for m in models if isinstance(m, Mapping)]
