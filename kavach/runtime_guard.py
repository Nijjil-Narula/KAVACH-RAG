"""
kavach.runtime_guard — Hardware isolation enforcer for KAVACH-RAG.

This module MUST be imported (or apply_hardware_guard() called) before any
other kavach module that touches torch, transformers, or faiss.  It:

  1. Sets all thread-limiting and CUDA-disabling environment variables at
     module-import time (before any downstream import can read them).
  2. Applies optional P-core CPU affinity BEFORE the first torch parallel op.
  3. Provides ``apply_hardware_guard()`` — idempotent, safe to call repeatedly.
  4. Provides diagnostic helpers: ``assert_cpu_only()``, ``rss_mb()``,
     ``vram_probe()``, ``detect_core_topology()``, and ``pin_to_p_cores()``.

Environment variables are set at the TOP of this module, before any import
that could transitively pull in torch or OpenMP shared libraries.
"""
from __future__ import annotations

import os

# ── 1. Set env vars BEFORE any torch/transformers/faiss import ────────────────
# These must be set before the C extensions are dlopen-ed.

# Disable CUDA entirely — prevents any GPU allocation even if drivers exist.
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# Limit thread pools used by CPU math libraries.
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"

# Prevents HuggingFace tokenizer parallelism deadlocks in forked processes.
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Windows: when both Intel MKL OpenMP (libiomp5md.dll) and LLVM OpenMP
# (libomp.dll) are loaded into the same process — which happens with some
# torch + faiss wheel combinations — the runtime raises a fatal error.
# Setting KMP_DUPLICATE_LIB_OK=TRUE suppresses that crash on Windows.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# Point HuggingFace downloads to our local cache inside the project tree.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_HF_HOME = os.path.join(_PROJECT_ROOT, "models", "hf_cache")
os.environ.setdefault("HF_HOME", _HF_HOME)

# ── 2. Standard-library imports (safe; no native code) ───────────────────────
import ctypes
import json
import logging
import subprocess
import threading
from pathlib import Path
from typing import Any, Literal, TypedDict

# ── 3. Third-party imports (now that env vars are locked in) ─────────────────
import psutil

from kavach.exceptions import HardwareIsolationError

log = logging.getLogger(__name__)

# ── Guard state ───────────────────────────────────────────────────────────────
_guard_applied: bool = False
_guard_lock: threading.Lock = threading.Lock()
_pre_pin_affinity: list[int] | None = None
_affinity_pinned_by_guard: bool = False
_last_affinity_status: dict[str, Any] = {
    "mode": "off",
    "applied": False,
    "affinity_unavailable": False,
}

_VALID_AFFINITY_MODES: tuple[str, ...] = ("off", "p_core_distinct", "p_core_smt")
_RELATION_PROCESSOR_CORE = 0
_ERROR_INSUFFICIENT_BUFFER = 122


# ─────────────────────────────────────────────────────────────────────────────
# Win32 CPU Topology & Affinity Helpers
# ─────────────────────────────────────────────────────────────────────────────


def detect_core_topology() -> dict[str, Any]:
    """Inspect Windows hybrid P-core / E-core topology via Win32 ctypes.

    Uses ``GetLogicalProcessorInformationEx(RelationProcessorCore=0)`` to read
    each core's ``EfficiencyClass`` and ``GROUP_AFFINITY`` mask. Highest
    ``EfficiencyClass`` corresponds to P-cores.

    Enforces the verified host contract:
    ``6 P-cores x 2 logical + 4 E-cores x 1 logical = 16 logical processors``.
    On non-Windows hosts, API failures, or any mismatched topology, returns
    ``{"affinity_unavailable": True, ...}`` without guessing.
    """
    if os.name != "nt":
        return {
            "affinity_unavailable": True,
            "reason": f"Win32 core topology API unavailable on os.name={os.name!r}",
        }

    try:
        from ctypes import wintypes

        class _GROUP_AFFINITY(ctypes.Structure):
            _fields_ = [
                ("Mask", ctypes.c_size_t),
                ("Group", wintypes.WORD),
                ("Reserved", wintypes.WORD * 3),
            ]

        class _PROCESSOR_RELATIONSHIP(ctypes.Structure):
            _fields_ = [
                ("Flags", wintypes.BYTE),
                ("EfficiencyClass", wintypes.BYTE),
                ("Reserved", wintypes.BYTE * 20),
                ("GroupCount", wintypes.WORD),
                ("GroupMask", _GROUP_AFFINITY * 1),
            ]

        class _SLPI_EX_HEADER(ctypes.Structure):
            _fields_ = [
                ("Relationship", wintypes.DWORD),
                ("Size", wintypes.DWORD),
            ]

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.GetLogicalProcessorInformationEx.argtypes = [
            wintypes.DWORD,
            ctypes.c_void_p,
            ctypes.POINTER(wintypes.DWORD),
        ]
        kernel32.GetLogicalProcessorInformationEx.restype = wintypes.BOOL

        buf_len = wintypes.DWORD(0)
        ok_probe = kernel32.GetLogicalProcessorInformationEx(
            _RELATION_PROCESSOR_CORE, None, ctypes.byref(buf_len)
        )
        err_probe = ctypes.get_last_error()
        if ok_probe or err_probe != _ERROR_INSUFFICIENT_BUFFER or buf_len.value == 0:
            return {
                "affinity_unavailable": True,
                "reason": (
                    f"GetLogicalProcessorInformationEx size probe failed "
                    f"(ok={ok_probe}, err={err_probe}, bytes={buf_len.value})"
                ),
            }

        buf = (ctypes.c_ubyte * buf_len.value)()
        ctypes.set_last_error(0)
        ok_read = kernel32.GetLogicalProcessorInformationEx(
            _RELATION_PROCESSOR_CORE, ctypes.byref(buf), ctypes.byref(buf_len)
        )
        err_read = ctypes.get_last_error()
        if not ok_read:
            return {
                "affinity_unavailable": True,
                "reason": f"GetLogicalProcessorInformationEx read failed (err={err_read})",
            }

        cores: list[dict[str, Any]] = []
        offset = 0
        base_addr = ctypes.addressof(buf)
        hdr_size = ctypes.sizeof(_SLPI_EX_HEADER)
        gm_offset = _PROCESSOR_RELATIONSHIP.GroupMask.offset
        ga_size = ctypes.sizeof(_GROUP_AFFINITY)

        while offset < buf_len.value:
            hdr = _SLPI_EX_HEADER.from_address(base_addr + offset)
            if hdr.Size == 0 or offset + hdr.Size > buf_len.value:
                return {
                    "affinity_unavailable": True,
                    "reason": f"Corrupt SLPI_EX record size={hdr.Size} at offset={offset}",
                }
            if hdr.Relationship == _RELATION_PROCESSOR_CORE:
                proc = _PROCESSOR_RELATIONSHIP.from_address(
                    base_addr + offset + hdr_size
                )
                logical_cpus: list[int] = []
                for g_idx in range(int(proc.GroupCount)):
                    ga = _GROUP_AFFINITY.from_address(
                        base_addr + offset + hdr_size + gm_offset + g_idx * ga_size
                    )
                    mask_val = int(ga.Mask)
                    group_base = int(ga.Group) * 64
                    for bit in range(64):
                        if (mask_val >> bit) & 1:
                            logical_cpus.append(group_base + bit)
                cores.append(
                    {
                        "core_index": len(cores),
                        "efficiency_class": int(proc.EfficiencyClass),
                        "smt_flags": int(proc.Flags),
                        "logical_cpus": sorted(logical_cpus),
                    }
                )
            offset += int(hdr.Size)
    except Exception as exc:
        return {
            "affinity_unavailable": True,
            "reason": f"Exception probing Win32 processor topology: {exc}",
        }

    if not cores:
        return {
            "affinity_unavailable": True,
            "reason": "No processor cores reported by GetLogicalProcessorInformationEx",
        }

    max_eff = max(c["efficiency_class"] for c in cores)
    min_eff = min(c["efficiency_class"] for c in cores)
    p_cores = [c for c in cores if c["efficiency_class"] == max_eff]
    e_cores = [c for c in cores if c["efficiency_class"] < max_eff]
    total_logical = sum(len(c["logical_cpus"]) for c in cores)

    topology_matches = (
        max_eff > min_eff
        and len(p_cores) == 6
        and all(len(c["logical_cpus"]) == 2 for c in p_cores)
        and len(e_cores) == 4
        and all(len(c["logical_cpus"]) == 1 for c in e_cores)
        and total_logical == 16
    )
    if not topology_matches:
        return {
            "affinity_unavailable": True,
            "reason": (
                f"Host topology mismatch: expected 6 P-cores x 2 + 4 E-cores x 1 = 16, "
                f"observed p_cores={len(p_cores)}, e_cores={len(e_cores)}, "
                f"total_logical={total_logical}, eff_range=[{min_eff}, {max_eff}]"
            ),
            "total_logical_cpus": total_logical,
            "p_core_count": len(p_cores),
            "e_core_count": len(e_cores),
            "cores": cores,
        }

    return {
        "affinity_unavailable": False,
        "total_logical_cpus": total_logical,
        "p_core_count": len(p_cores),
        "e_core_count": len(e_cores),
        "p_cores": [list(c["logical_cpus"]) for c in p_cores],
        "e_cores": [list(c["logical_cpus"]) for c in e_cores],
        "p_core_distinct_cpus": [
            p_cores[0]["logical_cpus"][0],
            p_cores[1]["logical_cpus"][0],
        ],
        "p_core_smt_cpus": list(p_cores[0]["logical_cpus"]),
        "cores": cores,
    }


def pin_to_p_cores(
    mode: Literal["off", "p_core_distinct", "p_core_smt"],
) -> dict[str, Any]:
    """Apply or clear process CPU affinity for P-core pinning.

    Parameters
    ----------
    mode:
        - ``"off"``: No P-core restriction (restores original affinity if pinned).
        - ``"p_core_distinct"``: 1 logical CPU from each of 2 distinct P-cores.
        - ``"p_core_smt"``: Both SMT sibling logical CPUs of 1 P-core.
    """
    global _pre_pin_affinity, _affinity_pinned_by_guard, _last_affinity_status

    if mode not in _VALID_AFFINITY_MODES:
        raise ValueError(
            f"Invalid cpu_affinity_mode {mode!r}; expected one of {_VALID_AFFINITY_MODES}"
        )

    proc = psutil.Process()
    current_affinity = sorted(proc.cpu_affinity())

    if mode == "off":
        if _affinity_pinned_by_guard and _pre_pin_affinity is not None:
            proc.cpu_affinity(list(_pre_pin_affinity))
            current_affinity = sorted(proc.cpu_affinity())
            _affinity_pinned_by_guard = False
        status = {
            "mode": "off",
            "applied": False,
            "affinity_unavailable": False,
            "cpu_affinity": current_affinity,
        }
        _last_affinity_status = status
        return status

    topo = detect_core_topology()
    if topo.get("affinity_unavailable"):
        log.warning(
            "CPU affinity mode %r skipped (%s).",
            mode,
            topo.get("reason", "topology unavailable"),
        )
        status = {
            "mode": mode,
            "applied": False,
            "affinity_unavailable": True,
            "reason": topo.get("reason", "topology unavailable"),
            "cpu_affinity": current_affinity,
        }
        _last_affinity_status = status
        return status

    if _pre_pin_affinity is None:
        _pre_pin_affinity = list(current_affinity)

    target_cpus = (
        list(topo["p_core_distinct_cpus"])
        if mode == "p_core_distinct"
        else list(topo["p_core_smt_cpus"])
    )
    proc.cpu_affinity(target_cpus)
    actual_affinity = sorted(proc.cpu_affinity())
    _affinity_pinned_by_guard = True

    status = {
        "mode": mode,
        "applied": actual_affinity == sorted(target_cpus),
        "affinity_unavailable": False,
        "target_cpus": sorted(target_cpus),
        "cpu_affinity": actual_affinity,
    }
    _last_affinity_status = status
    log.info("Applied CPU affinity mode=%s -> logical CPUs %s", mode, actual_affinity)
    return status


def _resolve_affinity_mode() -> Literal["off", "p_core_distinct", "p_core_smt"]:
    """Resolve CPU affinity mode from KAVACH_CPU_AFFINITY_MODE or precision_lock.json."""
    env_mode = os.environ.get("KAVACH_CPU_AFFINITY_MODE")
    if env_mode is not None:
        cleaned = env_mode.strip()
        if cleaned in _VALID_AFFINITY_MODES:
            return cleaned  # type: ignore[return-value]
        raise HardwareIsolationError(
            f"Invalid KAVACH_CPU_AFFINITY_MODE={env_mode!r}",
            details={"KAVACH_CPU_AFFINITY_MODE": env_mode},
        )

    lock_path = Path(_PROJECT_ROOT) / "models" / "precision_lock.json"
    if lock_path.exists():
        try:
            lock_data = json.loads(lock_path.read_text(encoding="utf-8"))
            if isinstance(lock_data, dict):
                lock_mode = lock_data.get("cpu_affinity_mode")
                if isinstance(lock_mode, str) and lock_mode in _VALID_AFFINITY_MODES:
                    return lock_mode  # type: ignore[return-value]
        except Exception as exc:
            log.warning("Could not read cpu_affinity_mode from %s: %s", lock_path, exc)

    return "off"


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────


def apply_hardware_guard() -> None:
    """Apply P-core CPU affinity (if configured), thread limits, and CUDA isolation.

    This function is idempotent — repeated calls are no-ops after the first
    successful application.  It performs the following actions:

    1. Resolve and apply ``cpu_affinity_mode`` BEFORE any torch parallel op.
    2. Import ``torch`` and set intra-op / inter-op thread counts.
    3. Call :func:`assert_cpu_only` to verify no CUDA build is active.
    4. Import ``faiss`` and set its OpenMP thread count.

    Raises:
        HardwareIsolationError: If a CUDA-enabled torch build is detected,
            or if thread limits cannot be applied.
    """
    global _guard_applied
    with _guard_lock:
        if _guard_applied:
            return
        affinity_mode = _resolve_affinity_mode()
        pin_to_p_cores(affinity_mode)
        _apply_torch_limits()
        assert_cpu_only()
        _apply_faiss_limits()
        _guard_applied = True


def assert_cpu_only() -> None:
    """Assert that the active PyTorch installation is CPU-only.

    Raises:
        HardwareIsolationError: If ``torch.version.cuda`` is not ``None`` or
            ``torch.cuda.is_available()`` returns ``True``.
    """
    import torch  # noqa: PLC0415 — intentional deferred import

    if torch.version.cuda is not None:
        raise HardwareIsolationError(
            "CUDA-enabled PyTorch build detected — CPU-only wheel required.",
            details={"torch_version": torch.__version__, "cuda_version": torch.version.cuda},
        )
    if torch.cuda.is_available():
        raise HardwareIsolationError(
            "torch.cuda.is_available() returned True despite CUDA_VISIBLE_DEVICES=''.",
            details={"CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>")},
        )


def rss_mb() -> float:
    """Return the current process RSS memory usage in mebibytes.

    Returns:
        Resident Set Size of this process in MiB (float).
    """
    return psutil.Process().memory_info().rss / (1024.0 * 1024.0)


class VramProbeResult(TypedDict):
    """Structured result from :func:`vram_probe`."""

    torch_cuda_build: bool
    cuda_available: bool
    pid_in_nvidia_smi: bool | None


def vram_probe() -> VramProbeResult:
    """Probe GPU / VRAM state without causing a crash on headless machines.

    Runs ``nvidia-smi`` in a subprocess to check whether this process appears
    in the compute-app list.  Returns a structured dict rather than raising so
    callers can decide how to handle failures.
    """
    import torch  # noqa: PLC0415

    torch_cuda_build: bool = torch.version.cuda is not None
    cuda_available: bool = torch.cuda.is_available()

    pid_in_nvidia_smi: bool | None = None
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,used_memory",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            current_pid = str(os.getpid())
            pid_in_nvidia_smi = any(
                line.split(",")[0].strip() == current_pid
                for line in result.stdout.splitlines()
                if line.strip()
            )
    except FileNotFoundError:
        pid_in_nvidia_smi = None
    except subprocess.TimeoutExpired:
        pid_in_nvidia_smi = None

    return VramProbeResult(
        torch_cuda_build=torch_cuda_build,
        cuda_available=cuda_available,
        pid_in_nvidia_smi=pid_in_nvidia_smi,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Private helpers
# ─────────────────────────────────────────────────────────────────────────────


def _apply_torch_limits() -> None:
    """Set torch intra-op and inter-op thread counts to safe CPU values.

    Raises:
        HardwareIsolationError: If the inter-op thread count cannot be set to
            the required value (1) after attempting to set it.
    """
    import torch  # noqa: PLC0415

    torch.set_num_threads(2)

    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        # Already set (e.g., guard applied from another entry point); verify.
        pass

    actual = torch.get_num_interop_threads()
    if actual != 1:
        raise HardwareIsolationError(
            f"Could not set inter-op threads to 1; got {actual}.",
            details={"actual_interop_threads": actual, "required": 1},
        )


def _apply_faiss_limits() -> None:
    """Import faiss and constrain its OpenMP thread pool to 2."""
    import faiss  # noqa: PLC0415

    faiss.omp_set_num_threads(2)
