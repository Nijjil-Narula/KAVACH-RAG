"""
tests.test_residency — Static & runtime verification of CPU-only resident execution.

Validates:
  1. Zero legacy un-load / lazy-load / Phase 1 ceiling / obsolete setting tokens
     across KAVACH-RAG S1-owned modules, test suite, and Phase 1.5 runner.
  2. Strict CPU-only environment (no .venv_gpu, CUDA unavailable in vram_probe).
  3. Phase 1.5 Settings contract & lock/env override precedence.
  4. BenchLock exclusive acquisition, release, contention timeout, and stale PID break.
  5. Fail-loud ConfigError when abstention_enabled=True with missing/stale calibration.
  6. Full resident VectorRetrievalPipeline load, warmup, and RSS <= 4096 MB (Stage 7).
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import psutil
import pytest

_ROOT = Path(__file__).resolve().parent.parent

# Dynamically built so this test file never matches its own static scan.
_FORBIDDEN_LITERALS: tuple[str, ...] = (
    "lazy" + "_load",
    "un" + "load",
    "17" + "00",
    "174" + "0.8",
    "1.7" + " GB",
    "embed" + "_dtype",
    "rerank_min" + "_prob",
)


def _iter_residency_scan_paths() -> list[Path]:
    """Return all repo python/runner paths that must be clean of legacy Phase 1 tokens."""
    files: list[Path] = []
    for py_path in sorted((_ROOT / "kavach").rglob("*.py")):
        files.append(py_path)
    for py_path in sorted((_ROOT / "tests").rglob("*.py")):
        files.append(py_path)
    ps1_phase15 = _ROOT / "scripts" / "run_phase15.ps1"
    if ps1_phase15.exists():
        files.append(ps1_phase15)
    return files


def test_no_obsolete_residency_or_ceiling_tokens() -> None:
    """Assert zero obsolete Phase 1 residency or ceiling tokens remain in verified code."""
    pattern = re.compile("|".join(re.escape(tok) for tok in _FORBIDDEN_LITERALS))
    hits: list[str] = []

    for file_path in _iter_residency_scan_paths():
        text = file_path.read_text(encoding="utf-8")
        for lineno, line in enumerate(text.splitlines(), start=1):
            match = pattern.search(line)
            if match:
                rel = file_path.relative_to(_ROOT)
                hits.append(f"{rel}:{lineno}: matched {match.group(0)!r} in {line.strip()!r}")

    assert not hits, "Obsolete Phase 1 tokens found:\n  " + "\n  ".join(hits)


def test_strict_cpu_only_environment() -> None:
    """Verify .venv_gpu does not exist and vram_probe reports CPU-only execution."""
    import kavach  # noqa: F401
    from kavach.runtime_guard import vram_probe

    assert not (_ROOT / ".venv_gpu").exists(), "Forbidden .venv_gpu environment found."
    probe = vram_probe()
    assert probe["torch_cuda_build"] is False
    assert probe["cuda_available"] is False
    assert probe["pid_in_nvidia_smi"] in (False, None)


def test_settings_phase15_contract() -> None:
    """Verify Settings defaults and active runtime contract obey Phase 1.5 bounds."""
    from kavach.config import Settings, get_settings

    defaults = Settings()
    assert defaults.rss_ceiling_mb == 4096.0
    assert defaults.candidate_k == 8
    assert defaults.rerank_max_length == 576
    assert defaults.embed_precision == "fp32"
    assert defaults.rerank_precision == "fp32"
    assert defaults.router_precision == "fp32"
    assert defaults.cpu_affinity_mode == "off"
    assert defaults.abstention_enabled is True
    assert defaults.abstain_logit_threshold is None
    assert not hasattr(defaults, "embed_" + "dtype")
    assert not hasattr(defaults, "rerank_min_" + "prob")

    get_settings.cache_clear()
    active = get_settings()
    assert active.rss_ceiling_mb == 4096.0
    assert active.candidate_k == 8
    assert active.rerank_max_length == 576
    if active.contingency_l_applied:
        assert active.rerank_long_query_max_length == 768


def test_load_runtime_locks_and_env_precedence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Settings.load_runtime_locks() must apply JSON locks unless KAVACH_* env overrides."""
    from kavach.config import Settings
    from kavach.exceptions import ConfigError

    for env_key in (
        "KAVACH_EMBED_PRECISION",
        "KAVACH_RERANK_PRECISION",
        "KAVACH_ROUTER_PRECISION",
        "KAVACH_RERANK_MAX_LENGTH",
        "KAVACH_CPU_AFFINITY_MODE",
        "KAVACH_RERANK_LONG_QUERY_MAX_LENGTH",
        "KAVACH_CONTINGENCY_L_APPLIED",
        "KAVACH_ABSTAIN_LOGIT_THRESHOLD",
    ):
        monkeypatch.delenv(env_key, raising=False)

    precision_lock = {
        "embed_precision": "int8",
        "rerank_precision": "fp32",
        "router_precision": "int8",
        "rerank_max_length": 576,
        "cpu_affinity_mode": "p_core_distinct",
        "rerank_long_query_max_length": 768,
        "rerank_long_query_token_threshold": 96,
        "contingency_l_applied": True,
    }
    abstain_cal = {
        "threshold_T": -0.425,
        "reranker_precision": "fp32",
        "rerank_max_length": 576,
    }
    (tmp_path / "precision_lock.json").write_text(
        json.dumps(precision_lock), encoding="utf-8"
    )
    (tmp_path / "abstain_calibration.json").write_text(
        json.dumps(abstain_cal), encoding="utf-8"
    )

    loaded = Settings(models_dir=tmp_path).load_runtime_locks()
    assert loaded.embed_precision == "int8"
    assert loaded.rerank_precision == "fp32"
    assert loaded.router_precision == "int8"
    assert loaded.cpu_affinity_mode == "p_core_distinct"
    assert loaded.rerank_long_query_max_length == 768
    assert loaded.contingency_l_applied is True
    assert loaded.abstain_logit_threshold == pytest.approx(-0.425)

    monkeypatch.setenv("KAVACH_EMBED_PRECISION", "bf16")
    monkeypatch.setenv("KAVACH_ABSTAIN_LOGIT_THRESHOLD", "1.15")
    overridden = Settings(models_dir=tmp_path).load_runtime_locks()
    assert overridden.embed_precision == "bf16"
    assert overridden.rerank_precision == "fp32"
    assert overridden.abstain_logit_threshold == pytest.approx(1.15)

    (tmp_path / "precision_lock.json").write_text("{corrupt json", encoding="utf-8")
    with pytest.raises(ConfigError, match="Failed to parse"):
        Settings(models_dir=tmp_path).load_runtime_locks()


def test_benchlock_acquire_release_timeout_and_stale_pid(tmp_path: Path) -> None:
    """BenchLock must enforce exclusive ownership, timeout on live holder, and break stale PIDs."""
    from kavach.retrieval.benchlock import BenchLock

    lock_path = tmp_path / "logs" / ".bench.lock"
    with BenchLock(lock_path=lock_path, owner="unit_test") as lock:
        assert lock_path.exists()
        info = json.loads(lock_path.read_text(encoding="utf-8"))
        assert info["owner"] == "unit_test"
        assert isinstance(info["pid"], int) and info["pid"] > 0

        # Live PID holding the lock must cause timeout for a second contender
        with pytest.raises(TimeoutError, match="Timed out"):
            contender = BenchLock(
                lock_path=lock_path,
                owner="contender",
                timeout_s=0.15,
                poll_interval_s=0.05,
            )
            contender.acquire()

    assert not lock_path.exists()

    # Stale PID that does not exist in OS process table should be broken automatically
    stale_pid = 99_999_991
    while psutil.pid_exists(stale_pid):
        stale_pid += 1

    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path.write_text(
        json.dumps({"pid": stale_pid, "acquired_at": "2020-01-01T00:00:00Z", "owner": "dead"}),
        encoding="utf-8",
    )
    with BenchLock(
        lock_path=lock_path,
        owner="recovery",
        timeout_s=1.0,
        poll_interval_s=0.05,
    ):
        recovered = json.loads(lock_path.read_text(encoding="utf-8"))
        assert recovered["owner"] == "recovery"
    assert not lock_path.exists()


def test_pipeline_missing_or_stale_calibration_raises_config_error(tmp_path: Path) -> None:
    """VectorRetrievalPipeline must fail loudly when abstention_enabled=True lacks valid calibration."""
    from kavach.config import Settings
    from kavach.exceptions import ConfigError
    from kavach.retrieval.pipeline import VectorRetrievalPipeline

    missing_settings = Settings(models_dir=tmp_path, abstention_enabled=True)
    with pytest.raises(ConfigError, match="abstain_calibration.json"):
        VectorRetrievalPipeline(missing_settings)

    stale_cal: dict[str, Any] = {
        "threshold_T": -0.5,
        "reranker_precision": "int8",
        "rerank_max_length": 576,
    }
    (tmp_path / "abstain_calibration.json").write_text(
        json.dumps(stale_cal), encoding="utf-8"
    )
    fp32_settings = Settings(
        models_dir=tmp_path,
        rerank_precision="fp32",
        rerank_max_length=576,
        abstention_enabled=True,
    )
    with pytest.raises(ConfigError, match="reranker_precision"):
        VectorRetrievalPipeline(fp32_settings)


def test_pipeline_residency_real_model() -> None:
    """Verify full pipeline residency (all 3 models resident, warmup complete, RSS <= ceiling)."""
    import kavach  # noqa: F401
    from kavach.config import Settings, get_settings
    from kavach.retrieval.pipeline import VectorRetrievalPipeline
    from kavach.runtime_guard import rss_mb, vram_probe

    defaults = Settings()
    required = [
        defaults.data_processed / "corpus_chunks.parquet",
        defaults.models_dir / "index_sq8_mrl512.faiss",
        defaults.models_dir / "index_idmap.json",
        defaults.models_dir / "router_deberta" / "model.safetensors",
    ]
    if not all(p.exists() for p in required):
        pytest.skip("Built corpus/index/router artifacts not found")

    cal_path = defaults.models_dir / "abstain_calibration.json"
    if cal_path.exists():
        get_settings.cache_clear()
        settings = get_settings()
    else:
        settings = Settings(abstention_enabled=False).load_runtime_locks()

    pipeline = VectorRetrievalPipeline(settings)
    assert pipeline._router.is_loaded is True
    assert pipeline._embedder.is_loaded is True
    assert pipeline._reranker is not None
    assert pipeline._reranker.is_loaded is True

    current_rss = rss_mb()
    assert pipeline.rss_after_load_mb <= settings.rss_ceiling_mb
    assert pipeline.within_rss_ceiling is True
    assert current_rss <= settings.rss_ceiling_mb

    probe = vram_probe()
    assert probe["cuda_available"] is False
    assert probe["torch_cuda_build"] is False
