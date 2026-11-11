"""
tests.test_runtime_guard — Verify CPU-only hardware isolation contracts.

Run with:
    .venv\\Scripts\\python.exe -m pytest -q tests/test_runtime_guard.py
"""
from __future__ import annotations

import os

import pytest


def test_env_vars_set() -> None:
    """All mandatory thread-limiting and CUDA-disabling env vars must be present."""
    # Import kavach to trigger apply_hardware_guard()
    import kavach  # noqa: F401

    assert os.environ.get("OMP_NUM_THREADS") == "2", (
        f"OMP_NUM_THREADS is {os.environ.get('OMP_NUM_THREADS')!r}, expected '2'"
    )
    assert os.environ.get("MKL_NUM_THREADS") == "2", (
        f"MKL_NUM_THREADS is {os.environ.get('MKL_NUM_THREADS')!r}, expected '2'"
    )
    assert os.environ.get("OPENBLAS_NUM_THREADS") == "2", (
        f"OPENBLAS_NUM_THREADS is {os.environ.get('OPENBLAS_NUM_THREADS')!r}, expected '2'"
    )
    assert os.environ.get("TOKENIZERS_PARALLELISM") == "false", (
        f"TOKENIZERS_PARALLELISM is {os.environ.get('TOKENIZERS_PARALLELISM')!r}, expected 'false'"
    )
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "", (
        f"CUDA_VISIBLE_DEVICES is {os.environ.get('CUDA_VISIBLE_DEVICES')!r}, expected ''"
    )


def test_torch_threads() -> None:
    """torch.get_num_threads() must equal 2 after guard application."""
    import kavach  # noqa: F401
    import torch

    actual = torch.get_num_threads()
    assert actual == 2, f"torch.get_num_threads() == {actual}, expected 2"


def test_cpu_only_build() -> None:
    """torch.version.cuda must be None for a CPU-only wheel."""
    import kavach  # noqa: F401
    import torch

    assert torch.version.cuda is None, (
        f"CUDA build detected: torch.version.cuda == {torch.version.cuda!r}"
    )


def test_cuda_not_available() -> None:
    """torch.cuda.is_available() must return False with CUDA_VISIBLE_DEVICES=''."""
    import kavach  # noqa: F401
    import torch

    assert torch.cuda.is_available() is False, (
        "torch.cuda.is_available() returned True — GPU isolation has failed."
    )


def test_vram_probe_structure() -> None:
    """vram_probe() must return expected keys and correct CPU-only values."""
    import kavach  # noqa: F401
    from kavach.runtime_guard import vram_probe

    result = vram_probe()

    # Key presence
    assert "torch_cuda_build" in result
    assert "cuda_available" in result
    assert "pid_in_nvidia_smi" in result

    # CPU-only assertions
    assert result["torch_cuda_build"] is False, (
        f"torch_cuda_build should be False, got {result['torch_cuda_build']}"
    )
    assert result["cuda_available"] is False, (
        f"cuda_available should be False, got {result['cuda_available']}"
    )
    # pid_in_nvidia_smi may be None (no nvidia-smi) or False (not in list);
    # it must NOT be True since we have CUDA_VISIBLE_DEVICES=""
    assert result["pid_in_nvidia_smi"] is not True, (
        f"pid_in_nvidia_smi should not be True, got {result['pid_in_nvidia_smi']}"
    )


def test_rss_mb_positive() -> None:
    """rss_mb() must return a positive float."""
    import kavach  # noqa: F401
    from kavach.runtime_guard import rss_mb

    rss = rss_mb()
    assert isinstance(rss, float), f"rss_mb() returned {type(rss)}, expected float"
    assert rss > 0.0, f"rss_mb() returned {rss}, expected > 0"


def test_detect_core_topology_hybrid_host() -> None:
    """Win32 ctypes topology detection must match 6 P-cores x 2 + 4 E-cores x 1 = 16."""
    import kavach  # noqa: F401
    from kavach.runtime_guard import detect_core_topology

    topo = detect_core_topology()
    assert topo["affinity_unavailable"] is False, f"Topology unavailable: {topo}"
    assert topo["total_logical_cpus"] == 16
    assert topo["p_core_count"] == 6
    assert topo["e_core_count"] == 4
    assert len(topo["p_cores"]) == 6
    assert all(len(cpus) == 2 for cpus in topo["p_cores"])
    assert len(topo["e_cores"]) == 4
    assert all(len(cpus) == 1 for cpus in topo["e_cores"])
    assert topo["p_core_distinct_cpus"] == [
        topo["p_cores"][0][0],
        topo["p_cores"][1][0],
    ]
    assert topo["p_core_smt_cpus"] == topo["p_cores"][0]


def test_pin_to_p_cores_modes_and_restore() -> None:
    """pin_to_p_cores() must apply distinct/SMT P-core masks and restore on 'off'."""
    import psutil
    import pytest

    import kavach  # noqa: F401
    from kavach.runtime_guard import (
        apply_hardware_guard,
        detect_core_topology,
        pin_to_p_cores,
    )

    topo = detect_core_topology()
    assert topo["affinity_unavailable"] is False

    proc = psutil.Process()
    # Clear any pinning already applied by import kavach -> apply_hardware_guard()
    pin_to_p_cores("off")
    unpinned_affinity = sorted(proc.cpu_affinity())

    try:
        distinct = pin_to_p_cores("p_core_distinct")
        assert distinct["applied"] is True
        assert distinct["affinity_unavailable"] is False
        assert distinct["cpu_affinity"] == sorted(topo["p_core_distinct_cpus"])
        assert sorted(proc.cpu_affinity()) == sorted(topo["p_core_distinct_cpus"])

        smt = pin_to_p_cores("p_core_smt")
        assert smt["applied"] is True
        assert smt["affinity_unavailable"] is False
        assert smt["cpu_affinity"] == sorted(topo["p_core_smt_cpus"])
        assert sorted(proc.cpu_affinity()) == sorted(topo["p_core_smt_cpus"])

        cleared = pin_to_p_cores("off")
        assert cleared["mode"] == "off"
        assert cleared["applied"] is False
        assert sorted(proc.cpu_affinity()) == unpinned_affinity

        with pytest.raises(ValueError, match="Invalid cpu_affinity_mode"):
            pin_to_p_cores("invalid_mode")  # type: ignore[arg-type]
    finally:
        pin_to_p_cores("off")
        apply_hardware_guard()
