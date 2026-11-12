"""Two-stage evaluation runner and 9-section report generator for KAVACH-RAG Phase 2.

Runnable as:
    python -m kavach.generation.evaluate --run [--resume] [--limit N]
    python -m kavach.generation.evaluate --report
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Sequence

from kavach.config import Settings, get_settings
from kavach.exceptions import ConfigError, CorpusIntegrityError, JudgeError
from kavach.generation.gen_config import GenSettings, get_gen_settings
from kavach.generation.generator import OllamaGenerator
from kavach.generation.judge import (
    RELEVANCE_SCORE,
    faithfulness,
    select_judge,
    stage1,
    summarize,
)
from kavach.generation.prompts import PROMPT_VERSION
from kavach.retrieval.benchlock import BenchLock

_TTFT_SPEC_MS = 300.0
_TPS_SPEC_LOW = 35.0
_TPS_SPEC_HIGH = 45.0
_Q1_FAITHFULNESS_MIN = 0.85
_Q2_RELEVANCE_MIN = 0.80
_Q3_CITATION_VALIDITY_MIN = 0.95
_ARITHMETIC_E2E_LABEL = "computed from separate runs, not an end-to-end measurement"


def _read_json_optional(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else None
    except (OSError, json.JSONDecodeError):
        return None


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise CorpusIntegrityError(
            f"Required JSONL file not found: {path}",
            details={"path": str(path)},
        )
    rows: list[dict[str, Any]] = []
    for lineno, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = raw_line.strip()
        if not stripped:
            continue
        try:
            obj = json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise CorpusIntegrityError(
                f"Corrupt JSON on line {lineno} of {path}: {exc}",
                details={"path": str(path), "line": lineno},
            ) from exc
        if not isinstance(obj, dict):
            raise CorpusIntegrityError(
                f"Expected JSON object on line {lineno} of {path}",
                details={"path": str(path), "line": lineno},
            )
        rows.append(obj)
    return rows


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        dir=str(path.parent),
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_path, path)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    _atomic_write_text(path, json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


def _percentile(values: Sequence[float], pct: float) -> float | None:
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
    clean = [float(v) for v in values if v is not None]
    if not clean:
        return {"count": 0, "min": None, "p50": None, "mean": None, "p95": None, "max": None}
    return {
        "count": len(clean),
        "min": round(min(clean), 3),
        "p50": _percentile(clean, 50.0),
        "mean": round(sum(clean) / len(clean), 3),
        "p95": _percentile(clean, 95.0),
        "max": round(max(clean), 3),
    }


def _fmt_num(val: float | int | None, digits: int = 3, suffix: str = "") -> str:
    if val is None:
        return "N/A"
    if isinstance(val, int):
        return f"{val}{suffix}"
    return f"{val:.{digits}f}{suffix}"


def _fmt_pct_or_ratio(val: float | None) -> str:
    if val is None:
        return "N/A"
    return f"{val:.4f}"


def _load_joined_results(
    results_path: Path, golden_path: Path
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    golden_rows = _read_jsonl(golden_path)
    golden_by_id: dict[str, dict[str, Any]] = {}
    for row in golden_rows:
        tid = row.get("test_id")
        if not isinstance(tid, str) or not tid:
            raise CorpusIntegrityError(
                f"Golden row missing valid test_id in {golden_path}",
                details={"path": str(golden_path)},
            )
        golden_by_id[tid] = row

    result_rows = _read_jsonl(results_path)
    for row in result_rows:
        tid = row.get("test_id")
        if tid not in golden_by_id:
            raise CorpusIntegrityError(
                f"Result test_id {tid!r} not found in {golden_path}",
                details={"test_id": tid, "golden_path": str(golden_path)},
            )
        golden = golden_by_id[tid]
        if row.get("query") != golden.get("query"):
            raise AssertionError(
                f"Query mismatch for {tid}: result={row.get('query')!r} != golden={golden.get('query')!r}"
            )
    return result_rows, golden_by_id


def run_evaluation(
    gen_settings: GenSettings | None = None,
    settings: Settings | None = None,
    *,
    resume: bool = False,
    limit: int | None = None,
    results_path: Path | None = None,
    golden_path: Path | None = None,
    output_path: Path | None = None,
    judge_override: Any | None = None,
) -> list[dict[str, Any]]:
    """Run two-stage evaluation over generation_results_150.jsonl and write append-only JSONL."""
    if limit is not None and limit < 0:
        raise ConfigError(f"--limit must be >= 0, got {limit}", details={"limit": limit})

    gs = gen_settings or get_gen_settings()
    st = settings or get_settings()

    res_path = results_path or (st.data_processed / "generation_results_150.jsonl")
    gld_path = golden_path or (st.data_golden / "golden_150.jsonl")
    out_path = output_path or (st.data_processed / "generation_judgments_150.jsonl")

    result_rows, golden_by_id = _load_joined_results(res_path, gld_path)

    existing_rows: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    if resume and out_path.is_file():
        existing_rows = _read_jsonl(out_path)
        for row in existing_rows:
            tid = row.get("test_id")
            if isinstance(tid, str) and tid:
                seen_ids.add(tid)

    pending = [r for r in result_rows if r["test_id"] not in seen_ids]
    if limit is not None:
        pending = pending[:limit]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    open_mode = "a" if resume else "w"
    new_rows: list[dict[str, Any]] = []

    generator: OllamaGenerator | None = None
    judge = judge_override
    try:
        if judge is None:
            generator = OllamaGenerator(gs)
            judge = select_judge(gs, generator)

        needs_stage2 = any(
            not bool(r.get("bypassed_llm"))
            and r.get("answer") is not None
            and not stage1(
                {
                    **r,
                    "expected_source_chunk_ids": golden_by_id[r["test_id"]][
                        "expected_source_chunk_ids"
                    ],
                    "question_type": golden_by_id[r["test_id"]]["question_type"],
                },
                golden_by_id[r["test_id"]],
                str(r.get("answer") or ""),
                r.get("context_chunks") or [],
                False,
            )["refusal_emitted"]
            for r in pending
        )

        with ExitStack() as stack:
            if not getattr(judge, "cross_family", False) and needs_stage2:
                stack.enter_context(
                    BenchLock(
                        lock_path=st.logs_dir / ".bench.lock",
                        owner="phase2_evaluate",
                    )
                )

            with out_path.open(open_mode, encoding="utf-8", newline="\n") as fh:
                total_pending = len(pending)
                for idx, result_row in enumerate(pending, start=1):
                    tid = str(result_row["test_id"])
                    golden = golden_by_id[tid]
                    qtype = str(golden["question_type"])
                    chunks = list(result_row.get("context_chunks") or [])
                    raw_answer = result_row.get("answer")
                    answer_str = raw_answer if isinstance(raw_answer, str) else ""
                    bypassed = bool(result_row.get("bypassed_llm"))
                    gen_error = result_row.get("error")

                    item = {
                        **result_row,
                        "expected_source_chunk_ids": golden["expected_source_chunk_ids"],
                        "question_type": qtype,
                    }
                    s1 = stage1(item, golden, answer_str, chunks, bypassed)

                    verdict_dict: dict[str, Any] | None = None
                    faith_val: float | None = None
                    rel_val: float | None = None
                    false_refusal = False
                    row_error: str | None = str(gen_error) if gen_error is not None else None

                    if bypassed:
                        pass_mode = "bypassed"
                    elif raw_answer is None:
                        rel_val = 0.0
                        false_refusal = True
                        if row_error is None:
                            row_error = "missing_answer"
                        pass_mode = "missing_answer"
                    elif s1["refusal_emitted"]:
                        if qtype != "OUT_OF_SCOPE":
                            rel_val = 0.0
                            false_refusal = True
                            pass_mode = "false_refusal"
                        else:
                            pass_mode = "oos_refusal"
                    else:
                        pass_mode = "stage2"
                        try:
                            verdict = judge.judge(golden["query"], chunks, answer_str)
                            verdict_dict = verdict.model_dump()
                            faith_val = faithfulness(verdict)
                            rel_val = RELEVANCE_SCORE[verdict.relevance]
                        except JudgeError as exc:
                            err_msg = f"judge_error: {exc}"
                            row_error = f"{row_error}; {err_msg}" if row_error else err_msg

                    record: dict[str, Any] = {
                        "test_id": tid,
                        "type": qtype,
                        "stage1": s1,
                        "verdict": verdict_dict,
                        "faithfulness": faith_val,
                        "relevance_score": rel_val,
                        "false_refusal": false_refusal,
                        "judge_backend": getattr(judge, "name", "unknown"),
                        "judge_model": getattr(judge, "model_id", "unknown"),
                        "cross_family": bool(getattr(judge, "cross_family", False)),
                        "thinking_applied": getattr(judge, "thinking_applied", None),
                        "error": row_error,
                    }
                    fh.write(json.dumps(record, ensure_ascii=False) + "\n")
                    fh.flush()
                    os.fsync(fh.fileno())
                    new_rows.append(record)
                    print(
                        f"[{idx:03d}/{total_pending:03d}] {tid} | {qtype} | {pass_mode} | "
                        f"faith={_fmt_num(faith_val, 3)} rel={_fmt_num(rel_val, 2)} "
                        f"false_refusal={false_refusal}",
                        flush=True,
                    )
    finally:
        if judge is not None and judge_override is None and hasattr(judge, "close"):
            judge.close()
        if generator is not None:
            generator.close()

    return existing_rows + new_rows


def _failure_tag(stage1_data: dict[str, Any]) -> str:
    if stage1_data.get("retrieval_miss") or not stage1_data.get("context_sufficient", True):
        return "retrieval-limited"
    return "generation-limited"


def _compute_latency_and_throughput(
    result_rows: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    served_ok = [
        r for r in result_rows if not r.get("bypassed_llm") and r.get("error") is None
    ]
    bypassed = [r for r in result_rows if r.get("bypassed_llm")]

    def _block(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
        return {
            "count": len(rows),
            "ttft_ms": _dist_stats([r.get("ttft_ms") for r in rows]),
            "decode_tps": _dist_stats([r.get("decode_tps") for r in rows]),
            "total_ms": _dist_stats([r.get("total_ms") for r in rows]),
        }

    qtypes = sorted({str(r["question_type"]) for r in served_ok})
    by_type = {
        qt: _block([r for r in served_ok if str(r.get("question_type")) == qt])
        for qt in qtypes
    }
    bypass_us = _dist_stats([r.get("bypass_latency_us") for r in bypassed])

    overall = _block(served_ok)
    ttft_mean = overall["ttft_ms"]["mean"]
    tps_mean = overall["decode_tps"]["mean"]
    ttft_p95 = overall["ttft_ms"]["p95"]

    all150_gen_ms = [
        (float(r["bypass_latency_us"]) / 1000.0)
        if r.get("bypassed_llm")
        else r.get("total_ms")
        for r in result_rows
    ]
    all150_clean = [float(v) for v in all150_gen_ms if v is not None]
    all150_mean_ms = round(sum(all150_clean) / len(all150_clean), 3) if all150_clean else None

    return {
        "served_count": len(served_ok),
        "bypassed_count": len(bypassed),
        "overall": overall,
        "by_question_type": by_type,
        "bypass_latency_us": bypass_us,
        "all_150_generation_mean_ms": all150_mean_ms,
        "targets": {
            "ttft_target_ms": _TTFT_SPEC_MS,
            "ttft_mean_meets_target": (ttft_mean <= _TTFT_SPEC_MS) if ttft_mean is not None else None,
            "ttft_p95_meets_target": (ttft_p95 <= _TTFT_SPEC_MS) if ttft_p95 is not None else None,
            "decode_tps_target_range": [_TPS_SPEC_LOW, _TPS_SPEC_HIGH],
            "decode_tps_mean_within_target": (
                (_TPS_SPEC_LOW <= tps_mean <= _TPS_SPEC_HIGH)
                if tps_mean is not None
                else None
            ),
            "decode_tps_mean_meets_floor": (
                (tps_mean >= _TPS_SPEC_LOW) if tps_mean is not None else None
            ),
        },
    }


def _compute_vram_summary(
    gs: GenSettings,
    preflight: dict[str, Any] | None,
    gen_run: dict[str, Any] | None,
    result_rows: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    pre_base = (
        ((preflight or {}).get("step3_baseline_gpu") or {}).get("gpu_snapshot") or {}
    ).get("memory_used_mib")
    pre_loaded = (preflight or {}).get("final_loaded_vram_mib")
    if pre_loaded is None:
        pre_loaded = ((preflight or {}).get("step5_warm_load") or {}).get("loaded_vram_mib")

    run_vram = (gen_run or {}).get("vram_mib") or {}
    run_baseline = run_vram.get("session_baseline_mib")
    run_loaded = run_vram.get("session_loaded_mib")
    run_peak = run_vram.get("session_peak_mib")

    if run_peak is None:
        item_peaks = [
            float(r["vram_peak_mib"])
            for r in result_rows
            if not r.get("bypassed_llm") and r.get("vram_peak_mib") is not None
        ]
        run_peak = round(max(item_peaks), 2) if item_peaks else None

    if run_baseline is None:
        item_bases = [
            float(r["vram_baseline_mib"])
            for r in result_rows
            if not r.get("bypassed_llm") and r.get("vram_baseline_mib") is not None
        ]
        run_baseline = round(min(item_bases), 2) if item_bases else None

    initial_baseline = pre_base if pre_base is not None else run_baseline
    loaded_idle = pre_loaded if pre_loaded is not None else run_loaded
    delta_mib = (
        round(float(run_peak) - float(initial_baseline), 2)
        if (run_peak is not None and initial_baseline is not None)
        else None
    )
    loaded_delta_mib = (
        round(float(loaded_idle) - float(initial_baseline), 2)
        if (loaded_idle is not None and initial_baseline is not None)
        else None
    )

    return {
        "initial_baseline_mib": initial_baseline,
        "loaded_idle_mib": loaded_idle,
        "run_peak_mib": run_peak,
        "delta_mib": delta_mib,
        "loaded_delta_mib": loaded_delta_mib,
        "target_mib": gs.vram_target_mib,
        "hard_limit_mib": gs.vram_hard_limit_mib,
        "within_target": (
            (float(run_peak) <= gs.vram_target_mib) if run_peak is not None else None
        ),
        "within_hard_limit": (
            (float(run_peak) <= gs.vram_hard_limit_mib) if run_peak is not None else None
        ),
    }


def _compute_conflict_and_noise(judgment_rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    conflicts = [r for r in judgment_rows if r.get("type") == "CONFLICT_TRAP"]
    both_in_ctx = sum(
        1 for r in conflicts if (r.get("stage1") or {}).get("both_versions_in_context")
    )
    both_in_ans = sum(
        1 for r in conflicts if (r.get("stage1") or {}).get("both_versions_in_answer")
    )
    both_when_present = sum(
        1
        for r in conflicts
        if (r.get("stage1") or {}).get("both_versions_in_context")
        and (r.get("stage1") or {}).get("both_versions_in_answer")
    )

    noises = [r for r in judgment_rows if r.get("type") == "CONTEXT_NOISE"]
    echo_vals = [
        (r.get("stage1") or {}).get("echo_ratio")
        for r in noises
        if not (r.get("stage1") or {}).get("bypassed_llm")
    ]
    echo_stats = _dist_stats(echo_vals)
    echo_nonzero = [
        {
            "test_id": r["test_id"],
            "echo_ratio": (r.get("stage1") or {}).get("echo_ratio"),
        }
        for r in noises
        if ((r.get("stage1") or {}).get("echo_ratio") or 0.0) > 0.0
    ]
    return {
        "conflict_trap": {
            "count": len(conflicts),
            "both_versions_in_context_count": both_in_ctx,
            "both_versions_in_answer_count": both_in_ans,
            "both_versions_preserved_when_in_context": both_when_present,
            "context_conditional_rate": (
                round(both_when_present / both_in_ctx, 4) if both_in_ctx > 0 else None
            ),
        },
        "context_noise": {
            "count": len(noises),
            "echo_ratio_stats": echo_stats,
            "nonzero_echo_items": echo_nonzero,
            "injection_leakage_detected": bool(echo_nonzero),
        },
    }


def _build_failure_inventory(
    judgment_rows: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    served_with_faith = [
        r
        for r in judgment_rows
        if not (r.get("stage1") or {}).get("bypassed_llm")
        and r.get("type") != "OUT_OF_SCOPE"
        and r.get("faithfulness") is not None
    ]
    lowest_10_src = sorted(
        served_with_faith,
        key=lambda r: (float(r["faithfulness"]), str(r["test_id"])),
    )[:10]

    lowest_10 = []
    for r in lowest_10_src:
        s1 = r.get("stage1") or {}
        lowest_10.append(
            {
                "test_id": r["test_id"],
                "type": r["type"],
                "faithfulness": r["faithfulness"],
                "relevance_score": r.get("relevance_score"),
                "retrieval_miss": s1.get("retrieval_miss"),
                "context_sufficient": s1.get("context_sufficient"),
                "entities_missing": s1.get("entities_missing", []),
                "hallucinated_citations": s1.get("hallucinated_citations", []),
                "tag": _failure_tag(s1),
            }
        )

    hallucinated_items = []
    for r in judgment_rows:
        s1 = r.get("stage1") or {}
        hall = s1.get("hallucinated_citations") or []
        if hall:
            hallucinated_items.append(
                {
                    "test_id": r["test_id"],
                    "type": r["type"],
                    "hallucinated_citations": hall,
                    "retrieval_miss": s1.get("retrieval_miss"),
                    "context_sufficient": s1.get("context_sufficient"),
                    "tag": _failure_tag(s1),
                }
            )

    false_refusals = []
    for r in judgment_rows:
        if r.get("false_refusal"):
            s1 = r.get("stage1") or {}
            false_refusals.append(
                {
                    "test_id": r["test_id"],
                    "type": r["type"],
                    "retrieval_miss": s1.get("retrieval_miss"),
                    "context_sufficient": s1.get("context_sufficient"),
                    "error": r.get("error"),
                    "tag": _failure_tag(s1),
                }
            )

    return {
        "lowest_faithfulness_10": lowest_10,
        "hallucinated_citation_items": hallucinated_items,
        "false_refusal_items": false_refusals,
    }


def _compute_arithmetic_e2e(
    phase15_lat: dict[str, Any] | None,
    lat_summary: dict[str, Any],
) -> dict[str, Any]:
    p15_overall = ((phase15_lat or {}).get("overall_total_latency_ms") or {}).get("mean")
    p15_semantic = (
        ((phase15_lat or {}).get("route_total_latency_ms") or {}).get("SEMANTIC_RAG") or {}
    ).get("mean")
    p2_all150 = lat_summary.get("all_150_generation_mean_ms")
    p2_served = ((lat_summary.get("overall") or {}).get("total_ms") or {}).get("mean")

    all150_e2e = (
        round(float(p15_overall) + float(p2_all150), 3)
        if (p15_overall is not None and p2_all150 is not None)
        else None
    )
    served_e2e = (
        round(float(p15_semantic) + float(p2_served), 3)
        if (p15_semantic is not None and p2_served is not None)
        else None
    )
    return {
        "label": _ARITHMETIC_E2E_LABEL,
        "phase15_retrieval_all150_mean_ms": p15_overall,
        "phase2_generation_all150_mean_ms": p2_all150,
        "arithmetic_e2e_all150_mean_ms": all150_e2e,
        "phase15_semantic_rag_mean_ms": p15_semantic,
        "phase2_generation_served_mean_ms": p2_served,
        "arithmetic_e2e_served_mean_ms": served_e2e,
    }


def _build_engineering_gates(
    gs: GenSettings,
    preflight: dict[str, Any] | None,
    prompt_freeze: dict[str, Any] | None,
    gen_run: dict[str, Any] | None,
    vram_summary: dict[str, Any],
    lat_summary: dict[str, Any],
    result_rows: Sequence[dict[str, Any]],
    gemini_key_present: bool,
) -> list[dict[str, Any]]:
    step4 = (preflight or {}).get("step4_model") or {}
    step5 = (preflight or {}).get("step5_warm_load") or {}
    step8 = (preflight or {}).get("step8_isolation") or {}
    counts = (gen_run or {}).get("counts") or {}
    bypass_worst = (lat_summary.get("bypass_latency_us") or {}).get("max")

    total_items = counts.get("total_items", len(result_rows))
    error_items = counts.get(
        "error_items",
        sum(1 for r in result_rows if r.get("error") is not None),
    )

    quant_val = step4.get("quantization_level")
    offload_ok = step5.get("full_gpu_offload_ok")
    iso_ok = step8.get("ok")
    vram_ok = vram_summary.get("within_hard_limit")
    num_ctx_val = (prompt_freeze or {}).get("num_ctx", gs.num_ctx)
    repeat_pen_val = (prompt_freeze or {}).get("repeat_penalty", gs.repeat_penalty)
    digest_val = (prompt_freeze or {}).get("model_digest") or step4.get("digest")

    def _status(flag: bool | None) -> str:
        if flag is None:
            return "NOT_AVAILABLE"
        return "PASS" if flag else "FAIL"

    return [
        {
            "gate": "Quantization Q4_K_M",
            "observed": quant_val or "N/A",
            "target": "Q4_K_M",
            "status": _status(quant_val == "Q4_K_M" if quant_val is not None else None),
        },
        {
            "gate": "Full GPU offload (>= 99% VRAM)",
            "observed": _fmt_num(step5.get("vram_offload_ratio"), 4)
            if step5.get("vram_offload_ratio") is not None
            else "N/A",
            "target": ">= 0.9900",
            "status": _status(offload_ok),
        },
        {
            "gate": "Peak VRAM <= hard limit",
            "observed": f"{vram_summary.get('run_peak_mib')} MiB"
            if vram_summary.get("run_peak_mib") is not None
            else "N/A",
            "target": f"<= {gs.vram_hard_limit_mib} MiB (target <= {gs.vram_target_mib} MiB)",
            "status": _status(vram_ok),
        },
        {
            "gate": "CPU/GPU isolation & no forbidden retrieval imports",
            "observed": f"torch_cuda={step8.get('torch_cuda_build')}, venv_gpu={step8.get('venv_gpu_exists')}, gemini_api_key_present={gemini_key_present}"
            if step8
            else f"gemini_api_key_present={gemini_key_present}",
            "target": "ok == True",
            "status": _status(iso_ok),
        },
        {
            "gate": "Context window & decoding contract",
            "observed": f"num_ctx={num_ctx_val}, repeat_penalty={repeat_pen_val}, digest={digest_val or 'N/A'}",
            "target": "num_ctx<=4096, repeat_penalty=1.0, digest present",
            "status": _status(
                bool(digest_val)
                and int(num_ctx_val) <= 4096
                and abs(float(repeat_pen_val) - 1.0) <= 1e-6
                if prompt_freeze or preflight
                else None
            ),
        },
        {
            "gate": "Refusal bypass sub-ms & zero LLM calls",
            "observed": f"max={_fmt_num(bypass_worst, 3, ' us')} (n={lat_summary.get('bypassed_count')})",
            "target": "< 1000 us, 0 LLM calls",
            "status": _status(
                (bypass_worst < 1000.0) if bypass_worst is not None else None
            ),
        },
        {
            "gate": "Batch generation 150/150 completion",
            "observed": f"total={total_items}, errors={error_items}",
            "target": "150 items, 0 errors",
            "status": _status(total_items == 150 and error_items == 0),
        },
    ]


def _render_markdown_report(summary_doc: dict[str, Any]) -> str:
    banner = summary_doc["banner"]
    gates = summary_doc["engineering_gates"]
    vram = summary_doc["vram"]
    lat = summary_doc["latency_and_throughput"]
    q = summary_doc["quality"]
    q_gates = summary_doc["quality_gates"]
    cn = summary_doc["conflict_and_noise"]
    fail = summary_doc["failure_inventory"]
    e2e = summary_doc["arithmetic_e2e_latency_estimate"]

    lines: list[str] = [
        "# KAVACH-RAG Phase 2 Generation Evaluation Report",
        "",
        "## 1. Configuration & Judge Banner",
        "",
        f"- **Generated at (UTC):** {summary_doc['generated_at_utc']}",
        f"- **Judge Backend:** `{banner['judge_backend']}` (`{banner['judge_model']}`)",
        f"- **Cross-Family Judge:** `{banner['cross_family']}`",
        f"- **Gemini Thinking Applied:** `{banner['thinking_applied']}`",
        f"- **Gemini API Key Present:** `{banner['gemini_api_key_present']}`",
        f"- **Prompt Version:** `{banner['prompt_version']}`",
        f"- **Model Digest:** `{banner['model_digest']}`",
        f"- **Decoding Window & Penalty:** `num_ctx={banner['num_ctx']}`, `repeat_penalty={banner['repeat_penalty']}`",
    ]
    if not banner["cross_family"]:
        lines.extend(
            [
                "",
                "> **WARNING — Self-Preference Bias (Indicative Only):** `cross_family=False`. "
                "The local fallback judge (`OllamaJudge`) is the same model family judging itself "
                "(self-preference bias: indicative only). Cross-family jury and CI gating belong to Phase 4.",
            ]
        )

    lines.extend(
        [
            "",
            "## 2. Engineering Gate Table",
            "",
            "| Gate | Observed | Target | Status |",
            "|---|---|---|---|",
        ]
    )
    for g in gates:
        lines.append(f"| {g['gate']} | {g['observed']} | {g['target']} | **{g['status']}** |")

    lines.extend(
        [
            "",
            "## 3. GPU VRAM Telemetry",
            "",
            "| Metric | Value (MiB) | Budget / Limit |",
            "|---|---:|---|",
            f"| Initial Baseline | {_fmt_num(vram['initial_baseline_mib'], 2)} | <= 700 MiB preflight limit |",
            f"| Loaded Idle | {_fmt_num(vram['loaded_idle_mib'], 2)} | - |",
            f"| Run Peak | {_fmt_num(vram['run_peak_mib'], 2)} | Target <= {vram['target_mib']} MiB / Hard Limit <= {vram['hard_limit_mib']} MiB |",
            f"| Delta (Run Peak - Baseline) | {_fmt_num(vram['delta_mib'], 2)} | Within target={vram['within_target']}, Within hard limit={vram['within_hard_limit']} |",
        ]
    )

    ov = lat["overall"]
    targets = lat["targets"]
    lines.extend(
        [
            "",
            "## 4. Latency & Throughput",
            "",
            f"- **Served items:** {lat['served_count']} | **Bypassed items:** {lat['bypassed_count']}",
            f"- **Spec Targets:** TTFT <= `{_TTFT_SPEC_MS:.0f} ms`, Decode throughput `{_TPS_SPEC_LOW:.0f}–{_TPS_SPEC_HIGH:.0f} tok/s`.",
            "",
            "| Scope | Count | TTFT P50 / P95 / Mean (ms) | Decode TPS P50 / P95 / Mean | Total P50 / P95 / Mean (ms) |",
            "|---|---:|---|---|---|",
            (
                f"| Overall (Served) | {ov['count']} | "
                f"{_fmt_num(ov['ttft_ms']['p50'], 2)} / {_fmt_num(ov['ttft_ms']['p95'], 2)} / {_fmt_num(ov['ttft_ms']['mean'], 2)} | "
                f"{_fmt_num(ov['decode_tps']['p50'], 2)} / {_fmt_num(ov['decode_tps']['p95'], 2)} / {_fmt_num(ov['decode_tps']['mean'], 2)} | "
                f"{_fmt_num(ov['total_ms']['p50'], 2)} / {_fmt_num(ov['total_ms']['p95'], 2)} / {_fmt_num(ov['total_ms']['mean'], 2)} |"
            ),
        ]
    )
    for qt, block in lat["by_question_type"].items():
        lines.append(
            f"| {qt} | {block['count']} | "
            f"{_fmt_num(block['ttft_ms']['p50'], 2)} / {_fmt_num(block['ttft_ms']['p95'], 2)} / {_fmt_num(block['ttft_ms']['mean'], 2)} | "
            f"{_fmt_num(block['decode_tps']['p50'], 2)} / {_fmt_num(block['decode_tps']['p95'], 2)} / {_fmt_num(block['decode_tps']['mean'], 2)} | "
            f"{_fmt_num(block['total_ms']['p50'], 2)} / {_fmt_num(block['total_ms']['p95'], 2)} / {_fmt_num(block['total_ms']['mean'], 2)} |"
        )

    byp = lat["bypass_latency_us"]
    lines.extend(
        [
            "",
            f"- **Bypass Latency (us):** count={byp['count']}, min={_fmt_num(byp['min'], 3)}, p50={_fmt_num(byp['p50'], 3)}, mean={_fmt_num(byp['mean'], 3)}, p95={_fmt_num(byp['p95'], 3)}, max={_fmt_num(byp['max'], 3)} us",
        ]
    )
    if targets["ttft_mean_meets_target"] is False:
        lines.append(
            f"- **Target Missed (TTFT):** Observed served TTFT mean `{_fmt_num(ov['ttft_ms']['mean'], 2)} ms` exceeds `{_TTFT_SPEC_MS:.0f} ms` target."
        )
    if targets["decode_tps_mean_meets_floor"] is False:
        lines.append(
            f"- **Target Missed (Decode TPS):** Observed decode TPS mean `{_fmt_num(ov['decode_tps']['mean'], 2)} tok/s` is below `{_TPS_SPEC_LOW:.0f} tok/s` floor."
        )

    rc = q["refusal_correct"]
    lines.extend(
        [
            "",
            "## 5. Quality Gates (Stage 1 + Stage 2)",
            "",
            "| Gate | Metric | Observed | Threshold | Status |",
            "|---|---|---:|---:|---|",
            f"| Q1 | Faithfulness Mean | {_fmt_pct_or_ratio(q['faithfulness_mean'])} | >= {_Q1_FAITHFULNESS_MIN:.2f} | **{'PASS' if q_gates['q1_passed'] else 'FAIL'}** |",
            f"| Q2 | Relevance Mean | {_fmt_pct_or_ratio(q['relevance_mean'])} | >= {_Q2_RELEVANCE_MIN:.2f} | **{'PASS' if q_gates['q2_passed'] else 'FAIL'}** |",
            f"| Q3 | Citation Validity Mean | {_fmt_pct_or_ratio(q['citation_validity_mean'])} | >= {_Q3_CITATION_VALIDITY_MIN:.2f} | **{'PASS' if q_gates['q3_passed'] else 'FAIL'}** |",
            "",
            f"- **Evaluated Served (non-OOS):** `{q['n_served']}` | **Context-Sufficient Subset:** `{q['n_context_sufficient']}` (Relevance Mean: `{_fmt_pct_or_ratio(q['relevance_mean_context_sufficient'])}`)",
            f"- **Served Model Refusals:** `{q['model_refusal_items']}` | **Over-Refusals (when context sufficient):** `{q['over_refusal_items']}`",
            f"- **Refusal Correctness (all items):** true={rc.get('true', 0)}, false={rc.get('false', 0)}",
            "",
            "### Faithfulness by Question Type",
            "",
            "| Question Type | Faithfulness Mean |",
            "|---|---:|",
        ]
    )
    for qt, val in q["faithfulness_by_type"].items():
        lines.append(f"| {qt} | {_fmt_pct_or_ratio(val)} |")

    lines.extend(
        [
            "",
            "## 6. Citation & Literal Fidelity Metrics",
            "",
            "| Metric | Value |",
            "|---|---:|",
            f"| Citation Validity Mean | {_fmt_pct_or_ratio(q['citation_validity_mean'])} |",
            f"| Citation Precision (Chunk) Mean | {_fmt_pct_or_ratio(q['citation_precision_mean'])} |",
            f"| Citation Recall (Chunk) Mean | {_fmt_pct_or_ratio(q['citation_recall_mean'])} |",
            f"| Citation Precision (Doc) Mean | {_fmt_pct_or_ratio(q['citation_precision_doc_mean'])} |",
            f"| Citation Recall (Doc) Mean | {_fmt_pct_or_ratio(q['citation_recall_doc_mean'])} |",
            f"| Entity Recall Mean | {_fmt_pct_or_ratio(q['entity_recall_mean'])} |",
            f"| Items with Hallucinated Citations | {q['hallucinated_citation_items']} |",
            f"| Items with Ungrounded Decimals | {q['ungrounded_decimal_items']} |",
        ]
    )

    ct = cn["conflict_trap"]
    ns = cn["context_noise"]
    lines.extend(
        [
            "",
            "## 7. CONFLICT_TRAP & CONTEXT_NOISE Analysis",
            "",
            f"- **CONFLICT_TRAP (`n={ct['count']}`):** `both_versions_in_context={ct['both_versions_in_context_count']}`, "
            f"`both_versions_in_answer={ct['both_versions_in_answer_count']}`, "
            f"`both_versions_preserved_when_in_context={ct['both_versions_preserved_when_in_context']}` "
            f"(rate={_fmt_pct_or_ratio(ct['context_conditional_rate'])}).",
            f"- **CONTEXT_NOISE (`TEST_141`–`TEST_150`, `n={ns['count']}`):** "
            f"`echo_ratio` mean={_fmt_num(ns['echo_ratio_stats']['mean'], 4)}, "
            f"max={_fmt_num(ns['echo_ratio_stats']['max'], 4)}, "
            f"non-zero items={len(ns['nonzero_echo_items'])}, "
            f"injection_leakage_detected={ns['injection_leakage_detected']}.",
        ]
    )

    lines.extend(
        [
            "",
            "## 8. Failure Inventory",
            "",
            "### Lowest Faithfulness Items (Top 10)",
            "",
            "| test_id | Type | Faithfulness | Relevance | Retrieval Miss | Context Sufficient | Tag |",
            "|---|---|---:|---:|---|---|---|",
        ]
    )
    if fail["lowest_faithfulness_10"]:
        for row in fail["lowest_faithfulness_10"]:
            lines.append(
                f"| {row['test_id']} | {row['type']} | {_fmt_num(row['faithfulness'], 3)} | "
                f"{_fmt_num(row['relevance_score'], 2)} | {row['retrieval_miss']} | "
                f"{row['context_sufficient']} | `{row['tag']}` |"
            )
    else:
        lines.append("| - | - | N/A | N/A | - | - | - |")

    lines.extend(
        [
            "",
            "### Hallucinated Citation Items",
            "",
            "| test_id | Type | Hallucinated Citations | Retrieval Miss | Context Sufficient | Tag |",
            "|---|---|---|---|---|---|",
        ]
    )
    if fail["hallucinated_citation_items"]:
        for row in fail["hallucinated_citation_items"]:
            lines.append(
                f"| {row['test_id']} | {row['type']} | {', '.join(row['hallucinated_citations'])} | "
                f"{row['retrieval_miss']} | {row['context_sufficient']} | `{row['tag']}` |"
            )
    else:
        lines.append("| None | - | - | - | - | - |")

    if fail["false_refusal_items"]:
        lines.extend(
            [
                "",
                "### False Refusal / Generation Failure Items",
                "",
                "| test_id | Type | Retrieval Miss | Context Sufficient | Tag | Error |",
                "|---|---|---|---|---|---|",
            ]
        )
        for row in fail["false_refusal_items"]:
            lines.append(
                f"| {row['test_id']} | {row['type']} | {row['retrieval_miss']} | "
                f"{row['context_sufficient']} | `{row['tag']}` | {row.get('error') or '-'} |"
            )

    lines.extend(
        [
            "",
            "## 9. Arithmetic End-to-End Latency Estimate",
            "",
            f"> **Methodology Notice:** `{e2e['label']}`.",
            "",
            "| Component | All-150 Mean (ms) | Served / SEMANTIC_RAG Mean (ms) |",
            "|---|---:|---:|",
            f"| Phase 1.5 Retrieval Baseline (`reports/phase15_latency_baseline.json`) | {_fmt_num(e2e['phase15_retrieval_all150_mean_ms'], 3)} | {_fmt_num(e2e['phase15_semantic_rag_mean_ms'], 3)} |",
            f"| Phase 2 Generation (`data/processed/generation_results_150.jsonl`) | {_fmt_num(e2e['phase2_generation_all150_mean_ms'], 3)} | {_fmt_num(e2e['phase2_generation_served_mean_ms'], 3)} |",
            f"| **Arithmetic End-to-End Estimate** ({e2e['label']}) | **{_fmt_num(e2e['arithmetic_e2e_all150_mean_ms'], 3)}** | **{_fmt_num(e2e['arithmetic_e2e_served_mean_ms'], 3)}** |",
            "",
        ]
    )
    return "\n".join(lines)


def generate_report(
    gen_settings: GenSettings | None = None,
    settings: Settings | None = None,
    *,
    judgments_path: Path | None = None,
    results_path: Path | None = None,
    golden_path: Path | None = None,
    prompt_freeze_path: Path | None = None,
    preflight_path: Path | None = None,
    gen_run_path: Path | None = None,
    phase15_latency_path: Path | None = None,
    summary_out_path: Path | None = None,
    report_out_path: Path | None = None,
) -> tuple[dict[str, Any], str]:
    """Compute summary metrics and write JSON + 9-section Markdown evaluation reports."""
    gs = gen_settings or get_gen_settings()
    st = settings or get_settings()

    jdg_path = judgments_path or (st.data_processed / "generation_judgments_150.jsonl")
    res_path = results_path or (st.data_processed / "generation_results_150.jsonl")
    gld_path = golden_path or (st.data_golden / "golden_150.jsonl")
    frz_path = prompt_freeze_path or (st.reports_dir / "phase2_prompt_freeze.json")
    pre_path = preflight_path or (st.reports_dir / "phase2_preflight.json")
    run_path = gen_run_path or (st.reports_dir / "phase2_generation_run.json")
    p15_path = phase15_latency_path or (st.reports_dir / "phase15_latency_baseline.json")
    sum_out = summary_out_path or (st.reports_dir / "generation_evaluation_summary.json")
    rep_out = report_out_path or (st.reports_dir / "generation_evaluation_report.md")

    judgment_rows = _read_jsonl(jdg_path)
    result_rows, _ = _load_joined_results(res_path, gld_path)

    prompt_freeze = _read_json_optional(frz_path)
    preflight = _read_json_optional(pre_path)
    gen_run = _read_json_optional(run_path)
    phase15_lat = _read_json_optional(p15_path)

    raw_summary = summarize(judgment_rows)
    rc_counter = raw_summary["refusal_correct"]
    quality_summary = {
        **raw_summary,
        "refusal_correct": {
            "true": int(rc_counter.get(True, 0)),
            "false": int(rc_counter.get(False, 0)),
        },
    }

    f_mean = quality_summary.get("faithfulness_mean")
    r_mean = quality_summary.get("relevance_mean")
    cv_mean = quality_summary.get("citation_validity_mean")
    quality_gates = {
        "q1_faithfulness_threshold": _Q1_FAITHFULNESS_MIN,
        "q1_passed": (f_mean >= _Q1_FAITHFULNESS_MIN) if f_mean is not None else False,
        "q2_relevance_threshold": _Q2_RELEVANCE_MIN,
        "q2_passed": (r_mean >= _Q2_RELEVANCE_MIN) if r_mean is not None else False,
        "q3_citation_validity_threshold": _Q3_CITATION_VALIDITY_MIN,
        "q3_passed": (cv_mean >= _Q3_CITATION_VALIDITY_MIN) if cv_mean is not None else False,
    }
    quality_gates["all_passed"] = (
        quality_gates["q1_passed"]
        and quality_gates["q2_passed"]
        and quality_gates["q3_passed"]
    )

    lat_summary = _compute_latency_and_throughput(result_rows)
    vram_summary = _compute_vram_summary(gs, preflight, gen_run, result_rows)
    cn_summary = _compute_conflict_and_noise(judgment_rows)
    fail_inventory = _build_failure_inventory(judgment_rows)
    e2e_summary = _compute_arithmetic_e2e(phase15_lat, lat_summary)

    gemini_key_present = bool(
        (gs.gemini_api_key or os.environ.get("GEMINI_API_KEY") or "").strip()
    )

    first_row = judgment_rows[0] if judgment_rows else {}
    thinking_vals = [
        r.get("thinking_applied")
        for r in judgment_rows
        if r.get("thinking_applied") is not None
    ]
    thinking_applied = thinking_vals[-1] if thinking_vals else first_row.get("thinking_applied")

    banner = {
        "judge_backend": first_row.get("judge_backend", gs.judge_backend),
        "judge_model": first_row.get("judge_model", gs.gemini_model),
        "cross_family": bool(first_row.get("cross_family", False)),
        "thinking_applied": thinking_applied,
        "gemini_api_key_present": gemini_key_present,
        "prompt_version": (prompt_freeze or {}).get("prompt_version")
        or (gen_run or {}).get("prompt_version")
        or PROMPT_VERSION,
        "model_digest": (prompt_freeze or {}).get("model_digest")
        or ((preflight or {}).get("step4_model") or {}).get("digest")
        or "N/A",
        "num_ctx": (prompt_freeze or {}).get("num_ctx", gs.num_ctx),
        "repeat_penalty": (prompt_freeze or {}).get("repeat_penalty", gs.repeat_penalty),
    }

    gates = _build_engineering_gates(
        gs,
        preflight,
        prompt_freeze,
        gen_run,
        vram_summary,
        lat_summary,
        result_rows,
        gemini_key_present,
    )

    summary_doc: dict[str, Any] = {
        "schema_version": "1.0",
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "banner": banner,
        "engineering_gates": gates,
        "vram": vram_summary,
        "latency_and_throughput": lat_summary,
        "quality": quality_summary,
        "quality_gates": quality_gates,
        "conflict_and_noise": cn_summary,
        "failure_inventory": fail_inventory,
        "arithmetic_e2e_latency_estimate": e2e_summary,
    }

    markdown = _render_markdown_report(summary_doc)
    _atomic_write_json(sum_out, summary_doc)
    _atomic_write_text(rep_out, markdown)
    return summary_doc, markdown


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m kavach.generation.evaluate",
        description="Two-stage generation judge runner and report generator.",
    )
    parser.add_argument(
        "--run",
        action="store_true",
        help="Evaluate generation_results_150.jsonl and write generation_judgments_150.jsonl.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume appending to generation_judgments_150.jsonl by test_id.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        metavar="N",
        help="Evaluate at most N pending items during --run.",
    )
    parser.add_argument(
        "--report",
        action="store_true",
        help="Generate generation_evaluation_summary.json and generation_evaluation_report.md.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint for Phase 2 Stage 1 + Stage 2 evaluation and reporting."""
    parser = _build_arg_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    if not args.run and not args.report:
        parser.error("Specify at least one of --run or --report")
    if (args.resume or args.limit is not None) and not args.run:
        parser.error("--resume and --limit require --run")

    gs = get_gen_settings()
    st = get_settings()

    if args.run:
        rows = run_evaluation(gs, st, resume=bool(args.resume), limit=args.limit)
        print(f"Evaluation complete: {len(rows)} rows in generation_judgments_150.jsonl", flush=True)

    if args.report:
        summary_doc, _ = generate_report(gs, st)
        qg = summary_doc["quality_gates"]
        print(
            f"Report written: q1={qg['q1_passed']} q2={qg['q2_passed']} "
            f"q3={qg['q3_passed']} all_passed={qg['all_passed']}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
