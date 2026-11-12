"""Resilience primitives: circuit breaker, single-flight TTL cache, VRAM sampler, task registry."""
from __future__ import annotations

import asyncio
import threading
import time
import uuid
from collections import OrderedDict
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable

from kavach.generation.generator import probe_vram_mib


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class ServiceDegraded(Exception):
    """Raised when the service refuses to start generation (HTTP 503)."""

    def __init__(self, code: str, message: str, retry_after_s: float) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.retry_after_s = max(0.0, float(retry_after_s))


class CircuitOpenError(ServiceDegraded):
    def __init__(self, retry_after_s: float) -> None:
        super().__init__("STATUS_TEMPORARILY_DEGRADED", "Upstream generator circuit is open", retry_after_s)


class CircuitBreaker:
    """closed -> open (after N consecutive failures) -> half_open (after open_seconds) -> closed/open."""

    def __init__(self, failure_threshold: int = 5, open_seconds: float = 30.0,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self._threshold = int(failure_threshold)
        self._open_s = float(open_seconds)
        self._clock = clock
        self._lock = threading.Lock()
        self._failures = 0
        self._opened_at: float | None = None
        self._probe_at: float | None = None
        self.trips = 0

    def _state_locked(self) -> str:
        if self._opened_at is None:
            return "closed"
        if self._clock() - self._opened_at >= self._open_s:
            return "half_open"
        return "open"

    @property
    def state(self) -> str:
        with self._lock:
            return self._state_locked()

    def retry_after_s(self) -> float:
        with self._lock:
            if self._opened_at is None:
                return 0.0
            return max(0.0, self._open_s - (self._clock() - self._opened_at))

    def allow(self) -> bool:
        """True if a request may call the upstream. In half_open exactly one probe is admitted."""
        with self._lock:
            state = self._state_locked()
            if state == "closed":
                return True
            if state == "open":
                return False
            now = self._clock()
            if self._probe_at is not None and now - self._probe_at < self._open_s:
                return False
            self._probe_at = now
            return True

    def record_success(self) -> None:
        with self._lock:
            self._failures = 0
            self._opened_at = None
            self._probe_at = None

    def record_failure(self) -> None:
        with self._lock:
            state = self._state_locked()
            self._probe_at = None
            if state == "half_open":
                self._opened_at = self._clock()
                self.trips += 1
                return
            self._failures += 1
            if self._failures >= self._threshold and self._opened_at is None:
                self._opened_at = self._clock()
                self.trips += 1

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "state": self._state_locked(),
                "consecutive_failures": self._failures,
                "trips": self.trips,
                "failure_threshold": self._threshold,
                "open_seconds": self._open_s,
            }


class _LockRec:
    __slots__ = ("lock", "refs")

    def __init__(self) -> None:
        self.lock = asyncio.Lock()
        self.refs = 0


class KeyHandle:
    """Idempotent release handle for a per-key lock."""

    def __init__(self, cache: "MutexTTLCache", key: str, rec: _LockRec) -> None:
        self._cache = cache
        self._key = key
        self._rec = rec
        self._released = False

    def release(self) -> None:
        if self._released:
            return
        self._released = True
        self._rec.lock.release()
        self._cache._unref(self._key, self._rec)


class MutexTTLCache:
    """In-memory LRU+TTL cache with per-key double-checked locking (cache-stampede protection)."""

    def __init__(self, ttl_s: float = 900.0, max_entries: int = 512,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self._ttl_s = float(ttl_s)
        self._max = int(max_entries)
        self._clock = clock
        self._data: OrderedDict[str, tuple[float, dict[str, Any]]] = OrderedDict()
        self._locks: dict[str, _LockRec] = {}
        self.hits = 0
        self.misses = 0
        self.coalesced = 0
        self.evictions = 0
        self.lock_timeouts = 0

    def peek(self, key: str) -> dict[str, Any] | None:
        item = self._data.get(key)
        if item is None:
            return None
        expires, value = item
        if self._clock() >= expires:
            del self._data[key]
            return None
        self._data.move_to_end(key)
        return value

    def lookup(self, key: str) -> dict[str, Any] | None:
        """peek() that counts a hit."""
        value = self.peek(key)
        if value is not None:
            self.hits += 1
        return value

    def coalesced_hit(self, key: str) -> dict[str, Any] | None:
        """peek() after waiting on a key lock: counts a hit and a coalesced request."""
        value = self.peek(key)
        if value is not None:
            self.hits += 1
            self.coalesced += 1
        return value

    def put(self, key: str, value: dict[str, Any]) -> None:
        self._data[key] = (self._clock() + self._ttl_s, value)
        self._data.move_to_end(key)
        while len(self._data) > self._max:
            self._data.popitem(last=False)
            self.evictions += 1

    def clear(self) -> None:
        self._data.clear()

    def _unref(self, key: str, rec: _LockRec) -> None:
        rec.refs -= 1
        if rec.refs <= 0 and self._locks.get(key) is rec:
            del self._locks[key]

    async def acquire(self, key: str, timeout_s: float) -> KeyHandle | None:
        """Take the per-key lock. Returns None if it could not be taken within timeout_s."""
        rec = self._locks.get(key)
        if rec is None:
            rec = _LockRec()
            self._locks[key] = rec
        rec.refs += 1
        try:
            await asyncio.wait_for(rec.lock.acquire(), timeout_s)
        except asyncio.TimeoutError:
            self._unref(key, rec)
            self.lock_timeouts += 1
            return None
        except BaseException:
            self._unref(key, rec)
            raise
        return KeyHandle(self, key, rec)

    async def get_or_compute(
        self,
        key: str,
        compute: Callable[[], Awaitable[dict[str, Any]]],
        *,
        cacheable: Callable[[dict[str, Any]], bool] = lambda value: True,
        lock_timeout_s: float = 150.0,
    ) -> tuple[dict[str, Any], bool]:
        """Return (value, hit). Concurrent identical misses compute once; the rest wait and reuse it."""
        value = self.lookup(key)
        if value is not None:
            return value, True
        handle = await self.acquire(key, lock_timeout_s)
        if handle is None:
            self.misses += 1
            return await compute(), False
        try:
            value = self.coalesced_hit(key)
            if value is not None:
                return value, True
            self.misses += 1
            value = await compute()
            if cacheable(value):
                self.put(key, value)
            return value, False
        finally:
            handle.release()

    def stats(self) -> dict[str, Any]:
        return {
            "size": len(self._data),
            "max_entries": self._max,
            "ttl_s": self._ttl_s,
            "hits": self.hits,
            "misses": self.misses,
            "coalesced": self.coalesced,
            "evictions": self.evictions,
            "lock_timeouts": self.lock_timeouts,
        }


class VRAMSampler:
    """One background nvidia-smi poller. `latest()` is probe-compatible with OllamaGenerator."""

    def __init__(self, hard_limit_mib: int, interval_s: float = 1.0,
                 probe: Callable[[], tuple[int, int] | None] = probe_vram_mib) -> None:
        self.hard_limit_mib = int(hard_limit_mib)
        self.interval_s = max(0.05, float(interval_s))
        self.probe = probe
        self._lock = threading.Lock()
        self._latest: tuple[int, int] | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.baseline_mib: int | None = None
        self.peak_mib: int | None = None
        self.samples = 0
        self.breached = False

    def sample_once(self) -> tuple[int, int] | None:
        reading = self.probe()
        if reading is None:
            return None
        used = int(reading[0])
        with self._lock:
            self._latest = (used, int(reading[1]))
            self.samples += 1
            if self.baseline_mib is None:
                self.baseline_mib = used
            if self.peak_mib is None or used > self.peak_mib:
                self.peak_mib = used
            if used > self.hard_limit_mib:
                self.breached = True
        return self._latest

    def latest(self) -> tuple[int, int] | None:
        return self._latest

    def _loop(self) -> None:
        while not self._stop.wait(self.interval_s):
            self.sample_once()

    def start(self) -> "VRAMSampler":
        self.sample_once()
        if self._thread is None:
            self._thread = threading.Thread(target=self._loop, name="kavach-vram-sampler", daemon=True)
            self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=3.0)
            self._thread = None

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "latest": self._latest,
                "baseline_mib": self.baseline_mib,
                "peak_mib": self.peak_mib,
                "samples": self.samples,
                "breached": self.breached,
                "hard_limit_mib": self.hard_limit_mib,
            }


class TaskRegistry:
    """Thread-safe in-memory registry of background jobs (bounded)."""

    def __init__(self, max_entries: int = 1000) -> None:
        self._max = int(max_entries)
        self._lock = threading.Lock()
        self._data: OrderedDict[str, dict[str, Any]] = OrderedDict()

    def create(self, task: str, label: str | None) -> dict[str, Any]:
        job_id = uuid.uuid4().hex[:12]
        rec: dict[str, Any] = {
            "job_id": job_id, "task": task, "label": label, "status": "pending",
            "created_at": utc_now(), "started_at": None, "finished_at": None,
            "result": None, "error": None,
        }
        with self._lock:
            self._data[job_id] = rec
            while len(self._data) > self._max:
                self._data.popitem(last=False)
            return dict(rec)

    def update(self, job_id: str, **fields: Any) -> None:
        with self._lock:
            rec = self._data.get(job_id)
            if rec is not None:
                rec.update(fields)

    def get(self, job_id: str) -> dict[str, Any] | None:
        with self._lock:
            rec = self._data.get(job_id)
            return dict(rec) if rec is not None else None
