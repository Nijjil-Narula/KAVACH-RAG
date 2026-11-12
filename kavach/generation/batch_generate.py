"""Tier 5 golden batch generation runner, prompt freeze manager, and run summary writer."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from kavach.exceptions import (
    ConfigError,
    GenerationError,
    VRAMBudgetExceededError,
)
from kavach.generation.gen_config import GenSettings, get_gen_settings
from kavach.generation.generator import (
    OllamaGenerator,
    VRAMWatchdog,
    probe_vram_mib,
)
from kavach.generation.prompts import (
    PROMPT_VERSION,
    SYSTEM_PROMPT,
    QwenTokenCounter,
    canonical_refusal,
    plan_prompt,
)
from kavach.retrieval.benchlock import BenchLock

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_DEFAULT_RETRIEVED_PATH = _PROJECT_ROOT / "data" / "processed" / "retrieved_contexts_150.json"
_DEFAULT_OUTPUT_PATH = _PROJECT_ROOT / "data" / "processed" / "generation_results_150.jsonl"
_DEFAULT_FREEZE_PATH = _PROJECT_ROOT / "reports" / "phase2_prompt_freeze.json"
_DEFAULT_SUMMARY_PATH = _PROJECT_ROOT / "reports" / "phase2_generation_run.json"
_DEFAULT_HF_CACHE = _PROJECT_ROOT / "models" / "hf_cache"


def _system_prompt_sha256() -> str:
    return hashlib.sha256(SYSTEM_PROMPT.encode("utf-8")).hexdigest()


def _extract_used_mib(sample: tuple[int, int] | int | None) -> int | None:
    if sample is None:
        return None
    if isinstance(sample, tuple):
        return int(sample[0])
    return int(sample)


def _model_name_matches(entry_name: str, model_name: str) -> bool:
    return entry_name == model_name or (
        ":" not in model_name and entry_name == f"{model_name}:latest"
    )


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write JSON atomically via temp file + fsync + os.replace."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    encoded = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    try:
        with tmp_path.open("w", encoding="utf-8") as fh:
            fh.write(encoded)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_path, path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


def _resolve_live_model_digest(generator: OllamaGenerator, model_name: str) -> str:
    """Resolve live model digest from Ollama /api/tags (OllamaGenerator.list_models())."""
    models = generator.list_models()
    if not isinstance(models, list):
        raise GenerationError(
            "Unexpected response from OllamaGenerator.list_models()",
            details={"type": type(models).__name__},
        )

    available: list[str] = []
    exact_match: dict[str, Any] | None = None
    latest_match: dict[str, Any] | None = None
    for entry in models:
        if not isinstance(entry, dict):
            continue
        name = str(entry.get("name") or entry.get("model") or "")
        if name:
            available.append(name)
        if name == model_name:
            exact_match = entry
            break
        if _model_name_matches(name, model_name) and latest_match is None:
            latest_match = entry

    matched = exact_match or latest_match
    if matched is not None:
        digest = matched.get("digest")
        if isinstance(digest, str) and digest.strip():
            return digest.strip()
        raise GenerationError(
            f"Model {model_name!r} found in /api/tags without a valid digest",
            details={"model": model_name, "entry_keys": sorted(matched.keys())},
        )

    raise GenerationError(
        f"Model {model_name!r} not found in Ollama /api/tags",
        details={"model": model_name, "available_models": available},
    )


def freeze_prompt_contract(
    s: GenSettings,
    generator: OllamaGenerator,
    freeze_path: Path = _DEFAULT_FREEZE_PATH,
) -> dict[str, Any]:
    """Freeze prompt template hash, decode parameters, and live model digest.

    Never reads golden_150.jsonl or retrieved_contexts_150.json.
    """
    model_digest = _resolve_live_model_digest(generator, s.model)
    payload: dict[str, Any] = {
        "prompt_version": PROMPT_VERSION,
        "system_prompt_sha256": _system_prompt_sha256(),
        "model": s.model,
        "num_ctx": int(s.num_ctx),
        "num_predict": int(s.num_predict),
        "repeat_penalty": float(s.repeat_penalty),
        "model_digest": model_digest,
        "frozen_at": datetime.now(timezone.utc).isoformat(),
    }
    _atomic_write_json(freeze_path, payload)
    return payload


def check_prompt_freeze(
    s: GenSettings,
    generator: OllamaGenerator,
    freeze_path: Path = _DEFAULT_FREEZE_PATH,
) -> dict[str, Any]:
    """Verify frozen prompt contract exists and matches current config and live model digest."""
    if not freeze_path.exists():
        raise ConfigError(
            f"Prompt freeze file not found at {freeze_path}; run --freeze first.",
            details={"freeze_path": str(freeze_path)},
        )

    try:
        data = json.loads(freeze_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ConfigError(
            f"Failed to parse prompt freeze file {freeze_path}: {exc}",
            details={"freeze_path": str(freeze_path), "error": str(exc)},
        ) from exc

    if not isinstance(data, dict):
        raise ConfigError(
            f"Prompt freeze file {freeze_path} must contain a JSON object",
            details={"freeze_path": str(freeze_path)},
        )

    live_digest = _resolve_live_model_digest(generator, s.model)
    expected: dict[str, Any] = {
        "prompt_version": PROMPT_VERSION,
        "system_prompt_sha256": _system_prompt_sha256(),
        "num_ctx": int(s.num_ctx),
        "num_predict": int(s.num_predict),
        "repeat_penalty": float(s.repeat_penalty),
        "model_digest": live_digest,
    }
    if "model" in data:
        expected["model"] = s.model

    mismatches: dict[str, dict[str, Any]] = {}
    for key, exp_val in expected.items():
        actual_val = data.get(key)
        if key == "repeat_penalty":
            try:
                match = actual_val is not None and abs(float(actual_val) - float(exp_val)) <= 1e-9
            except (TypeError, ValueError):
                match = False
        else:
            match = actual_val == exp_val
        if not match:
            mismatches[key] = {"expected": exp_val, "frozen": actual_val}

    if mismatches:
        raise ConfigError(
            f"Prompt freeze check failed for {freeze_path}: mismatched keys {sorted(mismatches)}",
            details={"freeze_path": str(freeze_path), "mismatches": mismatches},
        )

    return data


def _load_existing_jsonl(output_path: Path) -> tuple[list[dict[str, Any]], set[str]]:
    """Load existing JSONL lines and enforce strict PROMPT_VERSION parity before resuming."""
    if not output_path.exists():
        return [], set()

    records: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    lines = output_path.read_text(encoding="utf-8").splitlines()
    for lineno, raw_line in enumerate(lines, start=1):
        stripped = raw_line.strip()
        if not stripped:
            continue
        try:
            rec = json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise ConfigError(
                f"Corrupt JSON on line {lineno} of {output_path}: {exc}",
                details={"path": str(output_path), "line": lineno, "error": str(exc)},
            ) from exc

        if not isinstance(rec, dict):
            raise ConfigError(
                f"Expected JSON object on line {lineno} of {output_path}",
                details={"path": str(output_path), "line": lineno},
            )

        rec_prompt_version = rec.get("prompt_version")
        if rec_prompt_version != PROMPT_VERSION:
            raise ConfigError(
                f"Refusing to resume {output_path}: line {lineno} has prompt_version="
                f"{rec_prompt_version!r}, expected {PROMPT_VERSION!r}",
                details={
                    "path": str(output_path),
                    "line": lineno,
                    "test_id": rec.get("test_id"),
                    "found_prompt_version": rec_prompt_version,
                    "expected_prompt_version": PROMPT_VERSION,
                },
            )

        test_id = rec.get("test_id")
        if not isinstance(test_id, str) or not test_id:
            raise ConfigError(
                f"Missing valid test_id on line {lineno} of {output_path}",
                details={"path": str(output_path), "line": lineno},
            )
        if test_id in seen_ids:
            raise ConfigError(
                f"Duplicate test_id {test_id!r} on line {lineno} of {output_path}",
                details={"path": str(output_path), "line": lineno, "test_id": test_id},
            )
        seen_ids.add(test_id)
        records.append(rec)

    return records, seen_ids


def _percentile(values: Sequence[float], pct: float) -> float | None:
    """Compute linear-interpolated percentile rounded to 3 decimals, or None if empty."""
    if not values:
        return None
    ordered = sorted(float(v) for v in values)
    if len(ordered) == 1:
        return round(ordered[0], 3)
    pos = (pct / 100.0) * (len(ordered) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(ordered) - 1)
    frac = pos - lo
    return round(ordered[lo] + frac * (ordered[hi] - ordered[lo]), 3)


def _dist_stats(values: Sequence[float | int | None]) -> dict[str, Any]:
    """Compute min/p50/mean/p95/max summary statistics over non-None numeric values."""
    clean = [float(v) for v in values if v is not None]
    if not clean:
        return {
            "count": 0,
            "min": None,
            "p50": None,
            "mean": None,
            "p95": None,
            "max": None,
        }
    return {
        "count": len(clean),
        "min": round(min(clean), 3),
        "p50": _percentile(clean, 50.0),
        "mean": round(sum(clean) / len(clean), 3),
        "p95": _percentile(clean, 95.0),
        "max": round(max(clean), 3),
    }


def _build_bypass_record(item: dict[str, Any], s: GenSettings) -> dict[str, Any]:
    """Construct output record for refused/abstained items without calling the LLM."""
    reason = "HARD_REFUSAL" if item.get("refused") else item.get("abstain_reason")
    t0_ns = time.perf_counter_ns()
    answer = canonical_refusal(reason)
    bypass_latency_us = round((time.perf_counter_ns() - t0_ns) / 1000.0, 3)

    retrieved_chunks = item.get("retrieved_context_chunks") or []
    return {
        "test_id": item["test_id"],
        "question_type": item["question_type"],
        "query": item["query"],
        "ground_truth_answer": item["ground_truth_answer"],
        "route": item["route"],
        "bypassed_llm": True,
        "bypass_reason": reason,
        "answer": answer,
        "retrieved_chunk_ids": [c["chunk_id"] for c in retrieved_chunks],
        "kept_chunk_ids": [],
        "dropped_chunk_ids": [],
        "expected_chunk_ids": list(
            item.get("expected_source_chunk_ids", item.get("expected_chunk_ids", []))
        ),
        "expected_doc_ids": list(item.get("expected_doc_ids", [])),
        "context_chunks": [],
        "context_tokens": 0,
        "prompt_tokens": 0,
        "ttft_ms": None,
        "total_ms": None,
        "server_load_ms": None,
        "server_prefill_ms": None,
        "completion_tokens": None,
        "decode_tps": None,
        "finish_reason": None,
        "vram_baseline_mib": None,
        "vram_peak_mib": None,
        "attempts": 0,
        "bypass_latency_us": bypass_latency_us,
        "prompt_version": PROMPT_VERSION,
        "model": s.model,
        "error": None,
    }


def _build_served_record(
    item: dict[str, Any],
    s: GenSettings,
    counter: QwenTokenCounter,
    generator: OllamaGenerator,
    watchdog: VRAMWatchdog,
) -> dict[str, Any]:
    """Plan prompt budget, invoke OllamaGenerator, and assemble telemetry record."""
    retrieved_chunks = item.get("retrieved_context_chunks") or []
    retrieved_ids = [c["chunk_id"] for c in retrieved_chunks]
    expected_chunk_ids = list(
        item.get("expected_source_chunk_ids", item.get("expected_chunk_ids", []))
    )
    expected_doc_ids = list(item.get("expected_doc_ids", []))

    kept_chunk_ids: list[str] = []
    dropped_chunk_ids: list[str] = []
    context_chunks: list[dict[str, str]] = []
    context_tokens = 0
    prompt_tokens = 0
    calls_before = int(generator.calls)
    vram_before = _extract_used_mib(probe_vram_mib())

    try:
        plan = plan_prompt(
            item["query"],
            retrieved_chunks,
            counter,
            num_ctx=s.num_ctx,
            max_new=s.num_predict,
            context_cap=s.context_token_cap,
            margin=s.prompt_safety_margin,
        )
        kept_chunk_ids = list(plan.kept_chunk_ids)
        dropped_chunk_ids = list(plan.dropped_chunk_ids)
        kept_set = set(kept_chunk_ids)
        context_chunks = [
            {
                "chunk_id": c["chunk_id"],
                "doc_id": c.get("doc_id") or "",
                "content": c["content"],
                "standard_version": str(c.get("standard_version") or "").strip() or "unknown",
            }
            for c in retrieved_chunks
            if c["chunk_id"] in kept_set
        ]
        context_tokens = int(plan.context_tokens)
        prompt_tokens = int(plan.prompt_tokens)

        res = generator.generate(plan.messages)
        ws = watchdog.stats()
        if ws.breached:
            raise VRAMBudgetExceededError(
                f"GPU VRAM peak {ws.peak_mib} MiB exceeded hard limit {s.vram_hard_limit_mib} MiB.",
                details=ws.to_dict(),
            )

        m = res.metrics
        vram_baseline = m.vram_baseline_mib if m.vram_baseline_mib is not None else vram_before
        vram_peak = m.vram_peak_mib
        if vram_peak is None:
            vram_after = _extract_used_mib(probe_vram_mib())
            candidates = [v for v in (vram_before, vram_after) if v is not None]
            vram_peak = max(candidates) if candidates else None

        return {
            "test_id": item["test_id"],
            "question_type": item["question_type"],
            "query": item["query"],
            "ground_truth_answer": item["ground_truth_answer"],
            "route": item["route"],
            "bypassed_llm": False,
            "bypass_reason": None,
            "answer": res.text,
            "retrieved_chunk_ids": retrieved_ids,
            "kept_chunk_ids": kept_chunk_ids,
            "dropped_chunk_ids": dropped_chunk_ids,
            "expected_chunk_ids": expected_chunk_ids,
            "expected_doc_ids": expected_doc_ids,
            "context_chunks": context_chunks,
            "context_tokens": context_tokens,
            "prompt_tokens": prompt_tokens,
            "ttft_ms": m.ttft_ms,
            "total_ms": m.total_ms,
            "server_load_ms": m.server_load_ms,
            "server_prefill_ms": m.server_prefill_ms,
            "completion_tokens": m.completion_tokens,
            "decode_tps": m.decode_tps,
            "finish_reason": m.finish_reason,
            "vram_baseline_mib": vram_baseline,
            "vram_peak_mib": vram_peak,
            "attempts": m.attempts,
            "bypass_latency_us": None,
            "prompt_version": PROMPT_VERSION,
            "model": s.model,
            "error": None,
        }
    except VRAMBudgetExceededError:
        raise
    except GenerationError as exc:
        ws = watchdog.stats()
        if ws.breached:
            raise VRAMBudgetExceededError(
                f"GPU VRAM peak {ws.peak_mib} MiB exceeded hard limit {s.vram_hard_limit_mib} MiB.",
                details=ws.to_dict(),
            ) from exc
        vram_after = _extract_used_mib(probe_vram_mib())
        vram_samples = [v for v in (vram_before, vram_after) if v is not None]
        vram_peak = max(vram_samples) if vram_samples else None
        attempts_made = max(0, int(generator.calls) - calls_before)
        return {
            "test_id": item["test_id"],
            "question_type": item["question_type"],
            "query": item["query"],
            "ground_truth_answer": item["ground_truth_answer"],
            "route": item["route"],
            "bypassed_llm": False,
            "bypass_reason": None,
            "answer": None,
            "retrieved_chunk_ids": retrieved_ids,
            "kept_chunk_ids": kept_chunk_ids,
            "dropped_chunk_ids": dropped_chunk_ids,
            "expected_chunk_ids": expected_chunk_ids,
            "expected_doc_ids": expected_doc_ids,
            "context_chunks": context_chunks,
            "context_tokens": context_tokens,
            "prompt_tokens": prompt_tokens,
            "ttft_ms": None,
            "total_ms": None,
            "server_load_ms": None,
            "server_prefill_ms": None,
            "completion_tokens": None,
            "decode_tps": None,
            "finish_reason": None,
            "vram_baseline_mib": vram_before,
            "vram_peak_mib": vram_peak,
            "attempts": attempts_made,
            "bypass_latency_us": None,
            "prompt_version": PROMPT_VERSION,
            "model": s.model,
            "error": str(exc),
        }


def _summarize_run(
    records: Sequence[dict[str, Any]],
    s: GenSettings,
    *,
    watchdog: VRAMWatchdog,
    generator_calls_this_run: int,
    served_this_run: int,
    bypassed_this_run: int,
    resumed: bool,
    limit: int | None,
) -> dict[str, Any]:
    """Assemble reports/phase2_generation_run.json summary payload."""
    served = [r for r in records if not r.get("bypassed_llm")]
    served_ok = [r for r in served if r.get("error") is None]
    bypassed = [r for r in records if r.get("bypassed_llm")]
    errors = [r for r in records if r.get("error") is not None]
    chunk_drops = [r for r in served if r.get("dropped_chunk_ids")]

    def _latency_block(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
        return {
            "count": len(rows),
            "ttft_ms": _dist_stats([r.get("ttft_ms") for r in rows]),
            "total_ms": _dist_stats([r.get("total_ms") for r in rows]),
            "server_load_ms": _dist_stats([r.get("server_load_ms") for r in rows]),
            "server_prefill_ms": _dist_stats([r.get("server_prefill_ms") for r in rows]),
        }

    qtypes = sorted({str(r["question_type"]) for r in served_ok})
    by_qtype = {
        qt: _latency_block([r for r in served_ok if r.get("question_type") == qt])
        for qt in qtypes
    }

    wd_stats = watchdog.stats()
    item_baselines = [
        float(r["vram_baseline_mib"]) for r in served if r.get("vram_baseline_mib") is not None
    ]
    item_peaks = [float(r["vram_peak_mib"]) for r in served if r.get("vram_peak_mib") is not None]

    session_baseline_mib = (
        round(float(wd_stats.baseline_mib), 2)
        if wd_stats.baseline_mib is not None
        else (round(min(item_baselines), 2) if item_baselines else None)
    )
    loaded_candidates = [
        float(r["vram_peak_mib"]) for r in served_ok if r.get("vram_peak_mib") is not None
    ]
    session_loaded_mib = round(loaded_candidates[0], 2) if loaded_candidates else None

    peak_candidates = list(item_peaks)
    if wd_stats.peak_mib is not None:
        peak_candidates.append(float(wd_stats.peak_mib))
    session_peak_mib = round(max(peak_candidates), 2) if peak_candidates else None

    bypass_us = [
        float(r["bypass_latency_us"]) for r in bypassed if r.get("bypass_latency_us") is not None
    ]
    finish_counts = dict(Counter(str(r.get("finish_reason")) for r in served_ok))
    bypass_reason_counts = dict(Counter(str(r.get("bypass_reason")) for r in bypassed))

    target_mib = float(s.vram_target_mib)
    hard_limit_mib = float(s.vram_hard_limit_mib)

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "prompt_version": PROMPT_VERSION,
        "system_prompt_sha256": _system_prompt_sha256(),
        "model": s.model,
        "num_ctx": int(s.num_ctx),
        "num_predict": int(s.num_predict),
        "repeat_penalty": float(s.repeat_penalty),
        "resumed": resumed,
        "limit": limit,
        "counts": {
            "total_items": len(records),
            "served_items": len(served),
            "bypassed_items": len(bypassed),
            "succeeded_items": len(served_ok) + len(bypassed),
            "served_succeeded_items": len(served_ok),
            "error_items": len(errors),
            "chunk_drop_items": len(chunk_drops),
        },
        "served_latency_ms": {
            "overall": _latency_block(served_ok),
            "by_question_type": by_qtype,
        },
        "decode_tps": _dist_stats([r.get("decode_tps") for r in served_ok]),
        "vram_mib": {
            "target_mib": target_mib,
            "hard_limit_mib": hard_limit_mib,
            "session_baseline_mib": session_baseline_mib,
            "session_loaded_mib": session_loaded_mib,
            "session_peak_mib": session_peak_mib,
            "watchdog_samples": wd_stats.samples,
            "watchdog_mean_mib": wd_stats.mean_mib,
            "within_target": (session_peak_mib <= target_mib) if session_peak_mib is not None else None,
            "within_hard_limit": (
                (session_peak_mib <= hard_limit_mib) if session_peak_mib is not None else None
            ),
        },
        "finish_reason_counts": finish_counts,
        "bypass": {
            "count": len(bypassed),
            "reason_counts": bypass_reason_counts,
            "worst_us": round(max(bypass_us), 3) if bypass_us else None,
            "mean_us": round(sum(bypass_us) / len(bypass_us), 3) if bypass_us else None,
            "p50_us": _percentile(bypass_us, 50.0),
            "p95_us": _percentile(bypass_us, 95.0),
        },
        "generator_calls": {
            "this_run_calls": int(generator_calls_this_run),
            "this_run_served_items": int(served_this_run),
            "this_run_bypassed_items": int(bypassed_this_run),
            "total_attempts_across_records": sum(int(r.get("attempts") or 0) for r in records),
            "served_items_total": len(served),
        },
    }


def run_batch(
    s: GenSettings,
    *,
    resume: bool = False,
    limit: int | None = None,
    retrieved_path: Path = _DEFAULT_RETRIEVED_PATH,
    output_path: Path = _DEFAULT_OUTPUT_PATH,
    freeze_path: Path = _DEFAULT_FREEZE_PATH,
    summary_path: Path = _DEFAULT_SUMMARY_PATH,
    hf_cache_dir: Path = _DEFAULT_HF_CACHE,
) -> dict[str, Any]:
    """Execute batch generation across retrieved_contexts_150.json inside BenchLock + VRAMWatchdog."""
    if limit is not None and limit < 0:
        raise ConfigError(f"--limit must be >= 0, got {limit}", details={"limit": limit})

    with BenchLock(owner="phase2_batch_generate"):
        with OllamaGenerator(s) as freeze_gen:
            check_prompt_freeze(s, freeze_gen, freeze_path=freeze_path)

        if not retrieved_path.exists():
            raise ConfigError(
                f"Retrieved contexts file not found: {retrieved_path}",
                details={"path": str(retrieved_path)},
            )
        items: list[dict[str, Any]] = json.loads(retrieved_path.read_text(encoding="utf-8"))
        if not isinstance(items, list):
            raise ConfigError(
                f"Expected JSON list in {retrieved_path}",
                details={"path": str(retrieved_path)},
            )

        existing_records: list[dict[str, Any]] = []
        seen_ids: set[str] = set()
        if resume:
            existing_records, seen_ids = _load_existing_jsonl(output_path)

        pending_items = [it for it in items if it["test_id"] not in seen_ids]
        if limit is not None:
            pending_items = pending_items[:limit]

        needs_tokenizer = any(
            not (it.get("refused") or it.get("abstained")) for it in pending_items
        )
        counter = (
            QwenTokenCounter(s.tokenizer_id, cache_dir=str(hf_cache_dir))
            if needs_tokenizer
            else None
        )

        with VRAMWatchdog(
            s.vram_hard_limit_mib,
            s.vram_poll_interval_s,
            target_mib=s.vram_target_mib,
        ) as watchdog:
            with OllamaGenerator(s) as generator:
                output_path.parent.mkdir(parents=True, exist_ok=True)
                open_mode = "a" if resume else "w"
                new_records: list[dict[str, Any]] = []
                served_this_run = 0
                bypassed_this_run = 0
                served_durations_s: list[float] = []
                remaining_serveable = sum(
                    1 for it in pending_items if not (it.get("refused") or it.get("abstained"))
                )

                with output_path.open(open_mode, encoding="utf-8") as fh:
                    total_pending = len(pending_items)
                    for idx, item in enumerate(pending_items, start=1):
                        test_id = item["test_id"]
                        qtype = item["question_type"]
                        is_bypass = bool(item.get("refused") or item.get("abstained"))

                        if is_bypass:
                            calls_before = int(generator.calls)
                            record = _build_bypass_record(item, s)
                            if int(generator.calls) != calls_before:
                                raise GenerationError(
                                    f"Bypassed item {test_id} unexpectedly invoked generator.generate()",
                                    details={"test_id": test_id},
                                )
                            bypassed_this_run += 1
                            status_msg = (
                                f"[{idx:03d}/{total_pending:03d}] {test_id} | {qtype} | "
                                f"bypass({record['bypass_reason']}) | {record['bypass_latency_us']:.3f}us"
                            )
                        else:
                            if counter is None:
                                raise RuntimeError("Token counter not initialized for served item")
                            t_item_start = time.perf_counter()
                            record = _build_served_record(item, s, counter, generator, watchdog)
                            elapsed_s = time.perf_counter() - t_item_start
                            served_this_run += 1
                            remaining_serveable = max(0, remaining_serveable - 1)
                            served_durations_s.append(elapsed_s)
                            mean_s = sum(served_durations_s) / len(served_durations_s)
                            eta_s = mean_s * remaining_serveable
                            if record["error"] is None:
                                status_msg = (
                                    f"[{idx:03d}/{total_pending:03d}] {test_id} | {qtype} | served | "
                                    f"ttft={record['ttft_ms']}ms total={record['total_ms']}ms "
                                    f"tps={record['decode_tps']} vram_peak={record['vram_peak_mib']}MiB | "
                                    f"eta={eta_s:.0f}s"
                                )
                            else:
                                status_msg = (
                                    f"[{idx:03d}/{total_pending:03d}] {test_id} | {qtype} | ERROR | "
                                    f"{record['error']} | eta={eta_s:.0f}s"
                                )

                        fh.write(json.dumps(record, ensure_ascii=False) + "\n")
                        fh.flush()
                        os.fsync(fh.fileno())
                        new_records.append(record)
                        print(status_msg, flush=True)

                watchdog.stop()
                all_records = existing_records + new_records
                summary = _summarize_run(
                    all_records,
                    s,
                    watchdog=watchdog,
                    generator_calls_this_run=int(generator.calls),
                    served_this_run=served_this_run,
                    bypassed_this_run=bypassed_this_run,
                    resumed=resume,
                    limit=limit,
                )
                _atomic_write_json(summary_path, summary)
                return summary


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m kavach.generation.batch_generate",
        description="Tier 5 prompt freeze and golden batch generation runner.",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--freeze",
        action="store_true",
        help="Write reports/phase2_prompt_freeze.json from current prompt and live Ollama digest.",
    )
    mode.add_argument(
        "--freeze-check",
        action="store_true",
        help="Verify reports/phase2_prompt_freeze.json against current prompt and live Ollama digest.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume appending to data/processed/generation_results_150.jsonl by test_id.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        metavar="N",
        help="Process at most N pending items from retrieved_contexts_150.json.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint for prompt freeze, freeze verification, and batch generation."""
    parser = _build_arg_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.freeze and (args.resume or args.limit is not None):
        parser.error("--freeze cannot be combined with --resume or --limit")
    if args.freeze_check and (args.resume or args.limit is not None):
        parser.error("--freeze-check cannot be combined with --resume or --limit")

    s = get_gen_settings()

    if args.freeze:
        with OllamaGenerator(s) as generator:
            frozen = freeze_prompt_contract(s, generator)
        print(
            f"Prompt frozen: version={frozen['prompt_version']} "
            f"model={frozen['model']} digest={frozen['model_digest']}",
            flush=True,
        )
        return 0

    if args.freeze_check:
        with OllamaGenerator(s) as generator:
            checked = check_prompt_freeze(s, generator)
        print(
            f"Prompt freeze OK: version={checked['prompt_version']} "
            f"model={checked.get('model', s.model)} digest={checked['model_digest']}",
            flush=True,
        )
        return 0

    summary = run_batch(s, resume=bool(args.resume), limit=args.limit)
    c = summary["counts"]
    print(
        f"Batch run finished: total={c['total_items']} served={c['served_items']} "
        f"bypassed={c['bypassed_items']} errors={c['error_items']} "
        f"peak_vram={summary['vram_mib']['session_peak_mib']}MiB",
        flush=True,
    )
    return 0 if c["error_items"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
