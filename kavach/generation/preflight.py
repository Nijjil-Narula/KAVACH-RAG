"""
kavach.generation.preflight — Phase 2 GPU/Ollama/Prompt/Isolation preflight & baseline lock audit.

Runnable as:
    python -m kavach.generation.preflight
    python -m kavach.generation.preflight --sha256-only
"""
from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
import torch

from kavach.exceptions import GenerationError, OllamaUnavailableError, PromptBudgetError
from kavach.generation.gen_config import GenSettings, get_gen_settings
from kavach.generation.generator import OllamaGenerator
from kavach.retrieval.benchlock import BenchLock

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("kavach.generation.preflight")

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_FORBIDDEN_RETRIEVAL_MODULES: tuple[str, ...] = (
    "kavach.retrieval.pipeline",
    "kavach.retrieval.embedder",
    "kavach.retrieval.reranker",
    "kavach.retrieval.router",
)
_BASELINE_VRAM_LIMIT_MIB = 700
_EXPECTED_PRE_EDIT_EXCEPTIONS = frozenset(
    {
        "GenerationError",
        "OllamaUnavailableError",
        "VRAMBudgetExceededError",
        "PromptBudgetError",
        "JudgeError",
    }
)


def _atomic_write_json(target: Path, payload: dict[str, Any]) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    fd, tmp_name = tempfile.mkstemp(
        dir=str(target.parent),
        prefix=f".{target.name}.",
        suffix=".tmp",
    )
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(serialized)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_path, target)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise


def _clean_gpu_env(*, flash_attention: bool = False) -> dict[str, str]:
    """Return process env with CPU-only isolation vars stripped for Ollama subprocesses."""
    env = os.environ.copy()
    for key in (
        "CUDA_VISIBLE_DEVICES",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
    ):
        env.pop(key, None)
    if flash_attention:
        env["OLLAMA_FLASH_ATTENTION"] = "1"
    return env


def probe_gpu_snapshot() -> dict[str, Any] | None:
    """Query GPU name, driver version, memory.used (MiB), and memory.total (MiB)."""
    try:
        proc = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.used,memory.total",
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
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 4:
            continue
        try:
            used_mib = int(float(parts[2]))
            total_mib = int(float(parts[3]))
        except ValueError:
            continue
        return {
            "gpu_name": parts[0],
            "driver_version": parts[1],
            "memory_used_mib": used_mib,
            "memory_total_mib": total_mib,
        }
    return None


def evict_all_resident(gen: OllamaGenerator, *, wait_s: float = 10.0) -> list[str]:
    """Release all models currently resident in Ollama via `/api/chat` `keep_alive=0`."""
    resident = gen.ps()
    names = [
        str(entry.get("name") or entry.get("model") or "")
        for entry in resident
        if isinstance(entry, dict)
    ]
    names = [name for name in names if name]
    if not names:
        return []

    base_url = gen.s.ollama_base_url.rstrip("/")
    with httpx.Client(timeout=httpx.Timeout(connect=5.0, read=30.0, write=5.0, pool=5.0)) as client:
        for name in names:
            resp = client.post(
                f"{base_url}/api/chat",
                json={"model": name, "messages": [], "keep_alive": 0},
            )
            if resp.status_code >= 400:
                raise GenerationError(
                    f"Failed releasing resident model {name!r}: HTTP {resp.status_code}",
                    details={"model": name, "status_code": resp.status_code, "body": resp.text[:300]},
                )

    waiter = threading.Event()
    deadline = time.monotonic() + max(0.5, float(wait_s))
    while time.monotonic() < deadline:
        if not gen.ps():
            break
        waiter.wait(0.5)
    return names


def _resolve_ollama_exe() -> str:
    found = shutil.which("ollama")
    if found:
        return found
    local_app_data = os.environ.get("LOCALAPPDATA", "")
    if local_app_data:
        candidate = Path(local_app_data) / "Programs" / "Ollama" / "ollama.exe"
        if candidate.is_file():
            return str(candidate)
    return "ollama"


def _run_cli(
    args: list[str],
    *,
    env: dict[str, str] | None = None,
    timeout_s: float = 120.0,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        args,
        cwd=str(_PROJECT_ROOT),
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=timeout_s,
        check=False,
    )


def _parse_ollama_list_names(stdout: str) -> list[str]:
    names: list[str] = []
    for raw_line in stdout.splitlines():
        line = raw_line.strip()
        if not line or line.upper().startswith("NAME"):
            continue
        parts = line.split()
        if parts:
            names.append(parts[0])
    return names


def _match_base_model(installed_names: list[str], preferred: str) -> str | None:
    lower_map = {name.lower(): name for name in installed_names}
    if preferred.lower() in lower_map:
        return lower_map[preferred.lower()]
    if ":" not in preferred and f"{preferred.lower()}:latest" in lower_map:
        return lower_map[f"{preferred.lower()}:latest"]
    for fallback in ("qwen2.5:7b", "qwen2.5:7b:latest"):
        if fallback.lower() in lower_map:
            return lower_map[fallback.lower()]
    return None


def _update_modelfile(
    modelfile_path: Path,
    *,
    from_tag: str | None = None,
    set_params: dict[str, str] | None = None,
) -> None:
    lines = modelfile_path.read_text(encoding="utf-8").splitlines()
    updated: list[str] = []
    seen_params: set[str] = set()

    for line in lines:
        stripped = line.strip()
        if from_tag is not None and stripped.upper().startswith("FROM "):
            updated.append(f"FROM {from_tag}")
            continue
        if set_params and stripped.upper().startswith("PARAMETER "):
            parts = stripped.split(None, 2)
            if len(parts) >= 2 and parts[1] in set_params:
                key = parts[1]
                updated.append(f"PARAMETER {key} {set_params[key]}")
                seen_params.add(key)
                continue
        updated.append(line)

    if set_params:
        for key, val in set_params.items():
            if key not in seen_params:
                updated.append(f"PARAMETER {key} {val}")

    modelfile_path.write_text("\n".join(updated) + "\n", encoding="utf-8", newline="\n")


def _model_name_matches(entry_name: str, target_model: str) -> bool:
    if entry_name == target_model:
        return True
    if ":" not in target_model and entry_name == f"{target_model}:latest":
        return True
    return False


def _lookup_tag_entry(tags_models: list[dict[str, Any]], model_name: str) -> dict[str, Any] | None:
    for entry in tags_models:
        if isinstance(entry, dict):
            name = str(entry.get("name") or entry.get("model") or "")
            if _model_name_matches(name, model_name):
                return entry
    return None


def _lookup_ps_entry(ps_models: list[dict[str, Any]], model_name: str) -> dict[str, Any] | None:
    for entry in ps_models:
        if isinstance(entry, dict):
            name = str(entry.get("name") or entry.get("model") or "")
            if _model_name_matches(name, model_name):
                return entry
    if len(ps_models) == 1 and isinstance(ps_models[0], dict):
        return ps_models[0]
    return None


def _restart_ollama_ps1(*, flash_attention: bool = False) -> None:
    script = _PROJECT_ROOT / "scripts" / "start_ollama.ps1"
    cmd = [
        "powershell",
        "-NoProfile",
        "-ExecutionPolicy",
        "Bypass",
        "-File",
        str(script),
        "-Restart",
    ]
    if flash_attention:
        cmd.append("-FlashAttention")
    proc = _run_cli(cmd, env=_clean_gpu_env(flash_attention=flash_attention), timeout_s=60.0)
    if proc.returncode != 0:
        raise OllamaUnavailableError(
            f"start_ollama.ps1 failed (rc={proc.returncode}): {proc.stderr or proc.stdout}",
            details={"returncode": proc.returncode, "stdout": proc.stdout, "stderr": proc.stderr},
        )


def _warm_load_and_measure(
    settings: GenSettings,
    *,
    num_ctx: int,
) -> dict[str, Any]:
    scoped_settings = settings.model_copy(update={"num_ctx": int(num_ctx)})
    local_gen = OllamaGenerator(scoped_settings)
    try:
        res = local_gen.generate(
            [{"role": "user", "content": "Reply with OK."}],
            num_predict=8,
        )
        snap = probe_gpu_snapshot()
        ps_models = local_gen.ps()
        ps_entry = _lookup_ps_entry(ps_models, scoped_settings.model)
        size_bytes = int((ps_entry or {}).get("size") or 0)
        size_vram_bytes = int((ps_entry or {}).get("size_vram") or 0)
        offload_ratio = (size_vram_bytes / size_bytes) if size_bytes > 0 else 0.0
        return {
            "server_load_ms": res.metrics.server_load_ms,
            "total_ms": res.metrics.total_ms,
            "completion_tokens": res.metrics.completion_tokens,
            "finish_reason": res.metrics.finish_reason,
            "vram_peak_mib": res.metrics.vram_peak_mib,
            "loaded_vram_mib": (snap or {}).get("memory_used_mib"),
            "ps_size_bytes": size_bytes,
            "ps_size_vram_bytes": size_vram_bytes,
            "vram_offload_ratio": round(offload_ratio, 6),
            "full_gpu_offload_ok": size_bytes > 0 and size_vram_bytes >= 0.99 * size_bytes,
        }
    finally:
        local_gen.close()


def _run_step7_prompt_budget(
    settings: GenSettings,
    *,
    effective_num_ctx: int,
) -> dict[str, Any]:
    """Verify prompt token budget over all 140 serveable items without retrieval model imports."""
    from kavach.generation.prompts import QwenTokenCounter, plan_prompt

    hf_cache = _PROJECT_ROOT / "models" / "hf_cache"
    os.environ.setdefault("HF_HOME", str(hf_cache))

    token_counter = QwenTokenCounter(settings.tokenizer_id, cache_dir=hf_cache)

    contexts_path = _PROJECT_ROOT / "data" / "processed" / "retrieved_contexts_150.json"
    items = json.loads(contexts_path.read_text(encoding="utf-8"))
    serveable = [
        item for item in items if not item.get("refused") and not item.get("abstained")
    ]
    if len(serveable) != 140:
        raise GenerationError(
            f"Expected 140 serveable items in {contexts_path}, found {len(serveable)}",
            details={"serveable_count": len(serveable)},
        )

    max_prompt_tokens = 0
    chunk_drop_count = 0
    dropped_test_ids: list[str] = []
    budget_errors: list[dict[str, Any]] = []

    for item in serveable:
        test_id = str(item["test_id"])
        query = str(item["query"])
        chunks = list(item.get("retrieved_context_chunks") or [])

        try:
            plan = plan_prompt(
                query,
                chunks,
                token_counter,
                num_ctx=effective_num_ctx,
                max_new=settings.num_predict,
                context_cap=settings.context_token_cap,
                margin=settings.prompt_safety_margin,
            )
        except PromptBudgetError as exc:
            budget_errors.append({"test_id": test_id, "error": str(exc), "details": exc.details})
            continue

        if plan.prompt_tokens > max_prompt_tokens:
            max_prompt_tokens = plan.prompt_tokens
        if plan.dropped_chunk_ids:
            chunk_drop_count += len(plan.dropped_chunk_ids)
            dropped_test_ids.append(test_id)

    return {
        "serveable_checked": len(serveable),
        "effective_num_ctx": effective_num_ctx,
        "max_prompt_tokens": max_prompt_tokens,
        "chunk_drop_count": chunk_drop_count,
        "dropped_test_ids": dropped_test_ids,
        "budget_errors": budget_errors,
    }


def _check_isolation() -> dict[str, Any]:
    imported_forbidden = [m for m in _FORBIDDEN_RETRIEVAL_MODULES if m in sys.modules]
    venv_gpu_exists = (_PROJECT_ROOT / ".venv_gpu").exists()
    torch_cuda_build = torch.version.cuda is not None
    cuda_available = bool(torch.cuda.is_available())
    gemini_present = bool(
        os.environ.get("GEMINI_API_KEY") or os.environ.get("KAVACH_GEN_GEMINI_API_KEY")
    )
    ok = (
        not imported_forbidden
        and not venv_gpu_exists
        and not torch_cuda_build
        and not cuda_available
    )
    return {
        "ok": ok,
        "torch_version": torch.__version__,
        "torch_cuda_build": torch_cuda_build,
        "cuda_available": cuda_available,
        "venv_gpu_exists": venv_gpu_exists,
        "forbidden_retrieval_modules_imported": imported_forbidden,
        "gemini_api_key_present": gemini_present,
    }


def run_preflight(output_path: Path | None = None) -> tuple[int, dict[str, Any]]:
    """Run all 8 gated Phase 2 preflight checks and write reports/phase2_preflight.json."""
    settings = get_gen_settings()
    report_path = output_path or (_PROJECT_ROOT / "reports" / "phase2_preflight.json")
    modelfile_path = _PROJECT_ROOT / "models" / "Modelfile.tier5"
    ollama_exe = _resolve_ollama_exe()

    report: dict[str, Any] = {
        "schema_version": "1.0",
        "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "status": "RUNNING",
        "model": settings.model,
        "base_model": settings.base_model,
        "ollama_base_url": settings.ollama_base_url,
        "vram_target_mib": settings.vram_target_mib,
        "vram_hard_limit_mib": settings.vram_hard_limit_mib,
        "baseline_vram_limit_mib": _BASELINE_VRAM_LIMIT_MIB,
        "ladder_steps": [],
    }

    # ── Step 1: ollama --version & ollama list CLI check ──────────────────────
    ver_proc = _run_cli([ollama_exe, "--version"], env=_clean_gpu_env(), timeout_s=15.0)
    if ver_proc.returncode != 0:
        report["status"] = "STOP_OLLAMA_CLI_FAILED"
        report["step1_cli"] = {
            "returncode": ver_proc.returncode,
            "stdout": ver_proc.stdout.strip(),
            "stderr": ver_proc.stderr.strip(),
        }
        _atomic_write_json(report_path, report)
        log.error("ollama --version failed: %s", ver_proc.stderr or ver_proc.stdout)
        return 2, report

    list_proc = _run_cli([ollama_exe, "list"], env=_clean_gpu_env(), timeout_s=20.0)
    if list_proc.returncode != 0:
        report["status"] = "STOP_OLLAMA_LIST_FAILED"
        report["step1_cli"] = {
            "version": (ver_proc.stdout or ver_proc.stderr).strip(),
            "list_returncode": list_proc.returncode,
            "list_stderr": list_proc.stderr.strip(),
        }
        _atomic_write_json(report_path, report)
        log.error("ollama list failed: %s", list_proc.stderr or list_proc.stdout)
        return 2, report

    installed_names = _parse_ollama_list_names(list_proc.stdout)
    matched_base = _match_base_model(installed_names, settings.base_model)
    report["step1_cli"] = {
        "ollama_version": (ver_proc.stdout or ver_proc.stderr).strip(),
        "installed_models": installed_names,
        "matched_base_model": matched_base,
    }
    if matched_base is None:
        report["status"] = "STOP_MODEL_ABSENT"
        report["instructions"] = (
            f"Run `ollama pull {settings.base_model}` (~4.7 GB download) manually "
            "before retrying Phase 2 preflight."
        )
        _atomic_write_json(report_path, report)
        print(
            f"STOP_MODEL_ABSENT: Neither '{settings.base_model}' nor 'qwen2.5:7b' is installed.\n"
            f"Please run manually (~4.7 GB download):\n    ollama pull {settings.base_model}"
        )
        return 2, report

    if matched_base.lower() != settings.base_model.lower():
        log.info(
            "Base model %r absent; updating %s FROM to matched tag %r",
            settings.base_model,
            modelfile_path,
            matched_base,
        )
        _update_modelfile(modelfile_path, from_tag=matched_base)
        report["step1_cli"]["modelfile_from_updated"] = matched_base

    # ── Steps 2-6 serialized under BenchLock ──────────────────────────────────
    effective_num_ctx = settings.num_ctx
    gen = OllamaGenerator(settings)
    try:
        with BenchLock(owner="phase2_preflight"):
            # ── Step 2: Connectivity ──────────────────────────────────────────
            restarted = False
            try:
                api_ver = gen.version()
            except (OllamaUnavailableError, GenerationError):
                log.info(
                    "Ollama not reachable at %s; restarting via start_ollama.ps1",
                    settings.ollama_base_url,
                )
                _restart_ollama_ps1(flash_attention=False)
                restarted = True
                api_ver = gen.version()

            report["step2_connectivity"] = {
                "reachable": True,
                "restarted": restarted,
                "api_version": api_ver,
            }

            # ── Step 3: Evict resident models & check <= 700 MiB baseline ─────
            evicted = evict_all_resident(gen, wait_s=10.0)
            baseline_gpu = probe_gpu_snapshot()
            report["step3_baseline_gpu"] = {
                "evicted_models": evicted,
                "gpu_snapshot": baseline_gpu,
            }
            if baseline_gpu is None:
                report["status"] = "STOP_NVIDIA_SMI_UNAVAILABLE"
                _atomic_write_json(report_path, report)
                log.error("nvidia-smi unavailable; cannot verify GPU baseline.")
                return 2, report

            baseline_used = int(baseline_gpu["memory_used_mib"])
            if baseline_used > _BASELINE_VRAM_LIMIT_MIB:
                report["status"] = "STOP_BASELINE_VRAM_HIGH"
                report["instructions"] = (
                    f"Initial GPU baseline VRAM is {baseline_used} MiB (> {_BASELINE_VRAM_LIMIT_MIB} MiB). "
                    "Close GPU-consuming applications (browsers, Electron apps, games, other models) and re-run."
                )
                _atomic_write_json(report_path, report)
                print(
                    f"STOP_BASELINE_VRAM_HIGH: Initial GPU memory.used = {baseline_used} MiB "
                    f"(limit <= {_BASELINE_VRAM_LIMIT_MIB} MiB).\n"
                    "Please close GPU-using desktop apps and retry preflight."
                )
                return 2, report

            # ── Step 4: Create kavach-tier5 & verify Q4_K_M + digest ──────────
            create_proc = _run_cli(
                [ollama_exe, "create", settings.model, "-f", str(modelfile_path)],
                env=_clean_gpu_env(),
                timeout_s=120.0,
            )
            if create_proc.returncode != 0:
                report["status"] = "STOP_OLLAMA_CREATE_FAILED"
                report["step4_model"] = {
                    "returncode": create_proc.returncode,
                    "stdout": create_proc.stdout.strip(),
                    "stderr": create_proc.stderr.strip(),
                }
                _atomic_write_json(report_path, report)
                log.error("ollama create failed: %s", create_proc.stderr or create_proc.stdout)
                return 2, report

            show_info = gen.show(settings.model)
            quant_level = (show_info.get("details") or {}).get("quantization_level")
            tags_models = gen.list_models()
            tag_entry = _lookup_tag_entry(tags_models, settings.model)
            digest = (tag_entry or {}).get("digest")
            tag_size = (tag_entry or {}).get("size")

            report["step4_model"] = {
                "model": settings.model,
                "quantization_level": quant_level,
                "digest": digest,
                "tag_size_bytes": tag_size,
            }
            if tag_entry is None or not digest:
                report["status"] = "STOP_MODEL_TAG_OR_DIGEST_MISSING"
                _atomic_write_json(report_path, report)
                log.error("Model %r missing from /api/tags or lacks digest.", settings.model)
                return 2, report
            if quant_level != "Q4_K_M":
                report["status"] = "STOP_QUANTIZATION_MISMATCH"
                _atomic_write_json(report_path, report)
                log.error("Expected quantization_level='Q4_K_M', got %r", quant_level)
                return 2, report

            # ── Step 5: Warm load (8-token generation) & offload >= 99% ───────
            warm = _warm_load_and_measure(settings, num_ctx=effective_num_ctx)
            report["step5_warm_load"] = warm
            if not warm["full_gpu_offload_ok"]:
                report["status"] = "STOP_PARTIAL_GPU_OFFLOAD"
                report["instructions"] = (
                    "Ollama did not offload >= 99% of model size to VRAM. "
                    "Check that CUDA_VISIBLE_DEVICES was not leaked into the Ollama server."
                )
                _atomic_write_json(report_path, report)
                log.error(
                    "Partial GPU offload detected: size_vram=%s / size=%s",
                    warm["ps_size_vram_bytes"],
                    warm["ps_size_bytes"],
                )
                return 2, report

            # ── Step 6: Hard limit VRAM mitigation ladder (<= 5478 MiB) ───────
            loaded_vram = warm["loaded_vram_mib"]
            ladder_steps: list[dict[str, Any]] = []

            if loaded_vram is not None and loaded_vram > settings.vram_hard_limit_mib:
                log.warning(
                    "Loaded VRAM %d MiB > hard limit %d MiB; trying Step 6a (FlashAttention).",
                    loaded_vram,
                    settings.vram_hard_limit_mib,
                )
                _restart_ollama_ps1(flash_attention=True)
                m_6a = _warm_load_and_measure(settings, num_ctx=effective_num_ctx)
                ladder_steps.append({"step": "6a_flash_attention", **m_6a})
                loaded_vram = m_6a["loaded_vram_mib"]

            if loaded_vram is not None and loaded_vram > settings.vram_hard_limit_mib:
                log.warning(
                    "Loaded VRAM %d MiB > hard limit %d MiB; trying Step 6b (num_batch=256).",
                    loaded_vram,
                    settings.vram_hard_limit_mib,
                )
                _update_modelfile(modelfile_path, set_params={"num_batch": "256"})
                _run_cli(
                    [ollama_exe, "create", settings.model, "-f", str(modelfile_path)],
                    env=_clean_gpu_env(flash_attention=True),
                    timeout_s=120.0,
                )
                m_6b = _warm_load_and_measure(settings, num_ctx=effective_num_ctx)
                ladder_steps.append({"step": "6b_num_batch_256", **m_6b})
                loaded_vram = m_6b["loaded_vram_mib"]

            if loaded_vram is not None and loaded_vram > settings.vram_hard_limit_mib:
                log.warning(
                    "Loaded VRAM %d MiB > hard limit %d MiB; trying Step 6c (num_ctx=3072).",
                    loaded_vram,
                    settings.vram_hard_limit_mib,
                )
                effective_num_ctx = 3072
                _update_modelfile(modelfile_path, set_params={"num_ctx": "3072"})
                _run_cli(
                    [ollama_exe, "create", settings.model, "-f", str(modelfile_path)],
                    env=_clean_gpu_env(flash_attention=True),
                    timeout_s=120.0,
                )
                m_6c = _warm_load_and_measure(settings, num_ctx=effective_num_ctx)
                ladder_steps.append({"step": "6c_num_ctx_3072", **m_6c})
                loaded_vram = m_6c["loaded_vram_mib"]

            report["ladder_steps"] = ladder_steps
            report["effective_num_ctx"] = effective_num_ctx
            report["final_loaded_vram_mib"] = loaded_vram

            if loaded_vram is None or loaded_vram > settings.vram_hard_limit_mib:
                report["status"] = "STOP_VRAM_HARD_LIMIT_EXCEEDED"
                report["instructions"] = (
                    f"Loaded VRAM ({loaded_vram} MiB) still exceeds hard limit "
                    f"{settings.vram_hard_limit_mib} MiB after all 3 mitigation steps. "
                    "Do not alter quantization without explicit user approval."
                )
                _atomic_write_json(report_path, report)
                log.error(report["instructions"])
                return 2, report
    finally:
        gen.close()

    # ── Step 7: Prompt budget verification across all 140 serveable items ─────
    step7 = _run_step7_prompt_budget(settings, effective_num_ctx=effective_num_ctx)
    report["step7_prompt_budget"] = step7
    if step7["budget_errors"]:
        report["status"] = "STOP_PROMPT_BUDGET_ERROR"
        _atomic_write_json(report_path, report)
        log.error("PromptBudgetError encountered on %d items", len(step7["budget_errors"]))
        return 2, report

    # ── Step 8: Isolation checks & secret-safe metadata ───────────────────────
    isolation = _check_isolation()
    report["step8_isolation"] = isolation
    if not isolation["ok"]:
        report["status"] = "STOP_ISOLATION_VIOLATION"
        _atomic_write_json(report_path, report)
        log.error("Isolation check failed: %s", isolation)
        return 2, report

    report["status"] = "PASS"
    _atomic_write_json(report_path, report)
    log.info("Phase 2 preflight PASSED; report written to %s", report_path)
    return 0, report


def verify_sha256_and_amendment1(baseline_path: Path | None = None) -> tuple[int, dict[str, Any]]:
    """Verify frozen SHA-256 digests and Amendment 1 strict diffs against Stage 0 baseline."""
    b_path = baseline_path or (_PROJECT_ROOT / "reports" / "phase2_regression_baseline.json")
    baseline = json.loads(b_path.read_text(encoding="utf-8"))

    frozen_groups: dict[str, dict[str, dict[str, Any]]] = baseline.get("frozen_hashes") or {}
    pre_edit: dict[str, dict[str, Any]] = baseline.get("pre_edit_baselines") or {}

    mismatches: list[dict[str, Any]] = []
    missing_files: list[str] = []
    verified_count = 0

    for group_name, entries in frozen_groups.items():
        for rel_path, meta in entries.items():
            if rel_path == "tests/test_no_placeholders.py":
                continue
            target = _PROJECT_ROOT / rel_path
            if not target.is_file():
                missing_files.append(rel_path)
                continue
            actual_sha = hashlib.sha256(target.read_bytes()).hexdigest()
            expected_sha = meta["sha256"]
            if actual_sha != expected_sha:
                mismatches.append(
                    {
                        "group": group_name,
                        "path": rel_path,
                        "expected_sha256": expected_sha,
                        "actual_sha256": actual_sha,
                    }
                )
            else:
                verified_count += 1

    # Amendment 1 check 1: kavach/exceptions.py strictly append-only
    exc_meta = pre_edit["kavach/exceptions.py"]
    exc_path = _PROJECT_ROOT / "kavach" / "exceptions.py"
    exc_bytes = exc_path.read_bytes()
    prefix_len = int(exc_meta["byte_length"])
    prefix_sha = hashlib.sha256(exc_bytes[:prefix_len]).hexdigest()
    exc_prefix_ok = len(exc_bytes) >= prefix_len and prefix_sha == exc_meta["sha256"]

    appended_text = exc_bytes[prefix_len:].decode("utf-8", errors="replace")
    defined_classes = set(re.findall(r"^class\s+(\w+)\s*\(", appended_text, flags=re.MULTILINE))
    exc_classes_ok = defined_classes == _EXPECTED_PRE_EDIT_EXCEPTIONS

    # Amendment 1 check 2: tests/test_no_placeholders.py only 2 allowlisted lines added
    np_meta = pre_edit["tests/test_no_placeholders.py"]
    np_path = _PROJECT_ROOT / "tests" / "test_no_placeholders.py"
    np_lines = np_path.read_text(encoding="utf-8").splitlines()
    diff_lines = list(
        difflib.unified_diff(np_meta["lines"], np_lines, fromfile="baseline", tofile="current", lineterm="")
    )
    added_lines = [
        line[1:].strip()
        for line in diff_lines
        if line.startswith("+") and not line.startswith("+++")
    ]
    removed_lines = [
        line[1:].strip()
        for line in diff_lines
        if line.startswith("-") and not line.startswith("---")
    ]
    expected_added = {
        '(_ROOT / "kavach" / "generation" / "generator.py").resolve(),',
        '(_ROOT / "kavach" / "generation" / "judge.py").resolve(),',
    }
    np_diff_ok = len(removed_lines) == 0 and set(added_lines) == expected_added and len(added_lines) == 2

    all_ok = (
        not mismatches
        and not missing_files
        and exc_prefix_ok
        and exc_classes_ok
        and np_diff_ok
    )
    summary: dict[str, Any] = {
        "ok": all_ok,
        "baseline_path": b_path.relative_to(_PROJECT_ROOT).as_posix(),
        "frozen_verified_count": verified_count,
        "frozen_mismatches": mismatches,
        "frozen_missing": missing_files,
        "amendment1_exceptions": {
            "prefix_sha256_match": exc_prefix_ok,
            "appended_classes": sorted(defined_classes),
            "expected_classes_match": exc_classes_ok,
        },
        "amendment1_no_placeholders": {
            "removed_count": len(removed_lines),
            "added_lines": added_lines,
            "exact_two_path_allowlist_match": np_diff_ok,
        },
    }
    return (0 if all_ok else 1), summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="KAVACH-RAG Phase 2 preflight & baseline verifier.")
    parser.add_argument(
        "--sha256-only",
        action="store_true",
        help="Verify frozen file SHA-256 digests and Amendment 1 diffs against baseline.",
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        default=None,
        help="Optional path to phase2_regression_baseline.json.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output path for phase2_preflight.json.",
    )
    args = parser.parse_args(argv)

    if args.sha256_only:
        rc, summary = verify_sha256_and_amendment1(args.baseline)
        print(json.dumps(summary, indent=2))
        return rc

    rc, report = run_preflight(args.output)
    print(json.dumps({"status": report.get("status"), "output": str(args.output or "reports/phase2_preflight.json")}, indent=2))
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
