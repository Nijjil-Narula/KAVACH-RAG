"""
kavach.retrieval.benchlock — Exclusive filesystem lock for CPU benchmarks and calibration.

Prevents concurrent heavy neural model loading or timing runs from distorting CPU
latency / RSS measurements across subagents or terminals.
"""
from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import psutil

log = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


class BenchLock:
    """Exclusive context-managed filesystem lock at ``logs/.bench.lock``.

    Uses atomic ``os.open(..., os.O_CREAT | os.O_EXCL | os.O_WRONLY)`` creation
    with JSON metadata ``{pid, acquired_at, owner}``. Dead locks whose owner PID
    no longer exists (verified via ``psutil.pid_exists``) are cleanly broken.
    """

    def __init__(
        self,
        lock_path: Path | None = None,
        owner: str = "kavach",
        timeout_s: float = 3600.0,
        poll_interval_s: float = 0.5,
    ) -> None:
        self.lock_path: Path = (
            Path(lock_path)
            if lock_path is not None
            else (_PROJECT_ROOT / "logs" / ".bench.lock")
        )
        self.owner: str = str(owner)
        self.timeout_s: float = float(timeout_s)
        self.poll_interval_s: float = max(0.05, float(poll_interval_s))
        self._acquired: bool = False

    def acquire(self) -> BenchLock:
        """Acquire the exclusive benchmark lock, waiting up to ``timeout_s``."""
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        deadline = time.monotonic() + self.timeout_s

        while True:
            try:
                fd = os.open(
                    str(self.lock_path),
                    os.O_CREAT | os.O_EXCL | os.O_WRONLY,
                    0o644,
                )
            except FileExistsError:
                if self._try_break_stale_lock():
                    continue
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    holder = self._read_lock_info()
                    raise TimeoutError(
                        f"Timed out after {self.timeout_s:.1f}s waiting for BenchLock "
                        f"at {self.lock_path} (held by {holder})"
                    )
                time.sleep(min(self.poll_interval_s, remaining))
                continue

            try:
                payload = {
                    "pid": os.getpid(),
                    "acquired_at": datetime.now(timezone.utc).isoformat(),
                    "owner": self.owner,
                }
                raw = json.dumps(payload, indent=2).encode("utf-8")
                os.write(fd, raw)
                os.fsync(fd)
            except Exception:
                os.close(fd)
                self.lock_path.unlink(missing_ok=True)
                raise
            else:
                os.close(fd)

            self._acquired = True
            log.info(
                "Acquired BenchLock at %s (pid=%d, owner=%s)",
                self.lock_path,
                os.getpid(),
                self.owner,
            )
            return self

    def release(self) -> None:
        """Release the benchmark lock if held by the current process."""
        if not self._acquired:
            return
        try:
            holder = self._read_lock_info()
            holder_pid = holder.get("pid") if isinstance(holder, dict) else None
            if holder_pid is None or holder_pid == os.getpid():
                self.lock_path.unlink(missing_ok=True)
        except OSError as exc:
            log.warning("Failed removing BenchLock %s: %s", self.lock_path, exc)
        finally:
            self._acquired = False
            log.info("Released BenchLock at %s (owner=%s)", self.lock_path, self.owner)

    def _read_lock_info(self) -> dict[str, Any]:
        """Read lock metadata JSON safely; returns empty dict on transient/missing file."""
        try:
            text = self.lock_path.read_text(encoding="utf-8").strip()
            if not text:
                return {}
            data = json.loads(text)
            return data if isinstance(data, dict) else {}
        except (OSError, json.JSONDecodeError):
            return {}

    def _try_break_stale_lock(self) -> bool:
        """Break lock if its recorded PID no longer exists; return True if broken."""
        info = self._read_lock_info()
        pid = info.get("pid")
        if isinstance(pid, int) and pid > 0:
            if not psutil.pid_exists(pid):
                log.warning(
                    "Breaking stale BenchLock at %s (dead pid=%d, owner=%r)",
                    self.lock_path,
                    pid,
                    info.get("owner"),
                )
                try:
                    self.lock_path.unlink(missing_ok=True)
                    return True
                except OSError:
                    return False
            return False

        # Handle half-written/corrupt empty lock left by crashed creator (> 5s old)
        try:
            age_s = time.time() - self.lock_path.stat().st_mtime
            if age_s > 5.0 and not info:
                log.warning(
                    "Breaking unreadable/empty stale BenchLock at %s (age=%.1fs)",
                    self.lock_path,
                    age_s,
                )
                self.lock_path.unlink(missing_ok=True)
                return True
        except OSError:
            return True
        return False

    def __enter__(self) -> BenchLock:
        return self.acquire()

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.release()
