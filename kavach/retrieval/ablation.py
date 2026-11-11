"""
kavach.retrieval.ablation — Phase 1.5 CPU-only 6-run retrieval ablation & pipeline eval.

Runnable as:
    python -m kavach.retrieval.ablation [--fresh]

Executes inside BenchLock(owner="ablation") with permanently resident CPU models
(zero model reloading), verifies precision_lock.json and abstain_calibration.json,
archives prior GPU-hybrid outputs to reports/archive/gpu_hybrid/, evaluates:
  - Runs 1..6 over the 150-question golden dataset (ungated + calibrated gate T)
  - Tier 1 router accuracy against expected_route (reports/router_golden_eval.json)
  - Live end-to-end VectorRetrievalPipeline.retrieve() -> data/processed/retrieved_contexts_150.json
  - Reports & comparison vs superseded GPU-hybrid baseline -> reports/retrieval_ablation_report.md
"""
from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from kavach.runtime_guard import apply_hardware_guard, rss_mb, vram_probe  # noqa: E402

import argparse
import gc
import json
import logging
import platform
import re
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import psutil
import torch

from kavach.config import Settings, get_settings
from kavach.exceptions import (
    ConfigError,
    GoldenDatasetValidationError,
    HardwareIsolationError,
    RetrievalError,
)
from kavach.retrieval.benchlock import BenchLock
from kavach.retrieval.calibration import (
    verify_abstain_calibration,
    verify_precision_lock,
)
from kavach.retrieval.metrics import (
    dedupe_to_docs,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)
from kavach.retrieval.pipeline import VectorRetrievalPipeline
from kavach.retrieval.router import (
    ROUTE_FAST_PATH,
    ROUTE_REFUSAL,
    ROUTE_SEMANTIC,
    AgentIntentRouter,
)
from kavach.retrieval.schema import GoldenItem, QuestionType
from kavach.retrieval.vector_store import ChunkStore, FlatIndex

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("kavach.ablation")

_IN_SCOPE_TYPES: list[QuestionType] = [
    QuestionType.FACTUAL,
    QuestionType.MULTI_HOP,
    QuestionType.CONFLICT_TRAP,
    QuestionType.MALFORMED,
    QuestionType.CONTEXT_NOISE,
]

_ALL_TYPES: list[QuestionType] = _IN_SCOPE_TYPES + [QuestionType.OUT_OF_SCOPE]

_ROUTE_ORDER: list[str] = [ROUTE_FAST_PATH, ROUTE_SEMANTIC, ROUTE_REFUSAL]


# ─────────────────────────────────────────────────────────────────────────────
# Atomic I/O & checkpoint helpers
# ─────────────────────────────────────────────────────────────────────────────


def _atomic_write_text(dest: Path, text: str) -> None:
    """Write UTF-8 text atomically via a sibling temp file."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(f"{dest.name}.tmp.{os.getpid()}")
    with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)
    os.replace(tmp, dest)


def _atomic_write_json(dest: Path, payload: Any) -> None:
    """Serialize *payload* to pretty UTF-8 JSON and atomically replace *dest*."""
    _atomic_write_text(dest, json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


def _atomic_write_qvec_checkpoint(
    ckpt_path: Path,
    embed_model_id: str,
    embed_precision: str,
    test_ids: list[str],
    q_vecs_1024: list[np.ndarray],
    q_vecs_512: list[np.ndarray],
    embed_latencies_ms: list[float],
) -> None:
    """Persist Step 1 query embeddings atomically for safe CPU resume."""
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = ckpt_path.with_name(f"{ckpt_path.name}.tmp.{os.getpid()}.npz")
    np.savez(
        tmp,
        embed_model_id=np.asarray([embed_model_id]),
        embed_precision=np.asarray([embed_precision]),
        test_ids=np.asarray(test_ids),
        q_vecs_1024=np.stack(q_vecs_1024).astype(np.float32),
        q_vecs_512=np.stack(q_vecs_512).astype(np.float32),
        embed_latencies_ms=np.asarray(embed_latencies_ms, dtype=np.float64),
    )
    os.replace(tmp, ckpt_path)


def _load_golden(path: Path) -> list[GoldenItem]:
    """Load and Pydantic-validate all 150 golden items."""
    if not path.exists():
        raise GoldenDatasetValidationError(f"Golden dataset missing: {path}")
    items: list[GoldenItem] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line_no, raw in enumerate(fh, start=1):
            stripped = raw.strip()
            if not stripped:
                continue
            try:
                items.append(GoldenItem.model_validate(json.loads(stripped)))
            except Exception as exc:
                raise GoldenDatasetValidationError(
                    f"Golden validation failed at line {line_no} of {path}: {exc}"
                ) from exc
    return items


def _read_preflight(settings: Settings) -> dict[str, Any]:
    """Read reports/preflight.json if available."""
    path = settings.reports_dir / "preflight.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        log.warning("Could not parse preflight.json: %s", exc)
        return {}


def _expected_embed_model_id(settings: Settings, preflight: dict[str, Any]) -> str:
    """Resolve active embed_model_id matching Embedder.__init__."""
    notes_path = settings.models_dir / "embedder_notes.md"
    if notes_path.exists():
        for line in notes_path.read_text(encoding="utf-8").splitlines():
            m_id = re.search(r"\*\*Embed model id:\*\*\s*(.+)", line, re.IGNORECASE)
            if m_id:
                return m_id.group(1).strip()
        for line in notes_path.read_text(encoding="utf-8").splitlines():
            m_hdr = re.match(r"^#\s*Embedder Notes:\s*(.+)$", line.strip())
            if m_hdr:
                return m_hdr.group(1).strip()
    pref_id = preflight.get("embed_model_id")
    if isinstance(pref_id, str) and pref_id:
        return pref_id
    return settings.embed_model_id


def _try_load_qvec_checkpoint(
    ckpt_path: Path,
    items: list[GoldenItem],
    expected_model_id: str,
    cfg: Settings,
) -> tuple[list[np.ndarray], list[np.ndarray], list[float]] | None:
    """Load Step 1 query embedding checkpoint if schema, precision, and model_id match."""
    if not ckpt_path.exists():
        return None
    expected_ids = [it.test_id for it in items]
    try:
        with np.load(ckpt_path, allow_pickle=False) as data:
            required = {
                "embed_model_id",
                "embed_precision",
                "test_ids",
                "q_vecs_1024",
                "q_vecs_512",
                "embed_latencies_ms",
            }
            if not required.issubset(set(data.files)):
                return None
            ckpt_model = str(data["embed_model_id"][0])
            ckpt_prec = str(data["embed_precision"][0])
            ckpt_ids = [str(tid) for tid in data["test_ids"].tolist()]
            q1024 = np.asarray(data["q_vecs_1024"], dtype=np.float32)
            q512 = np.asarray(data["q_vecs_512"], dtype=np.float32)
            lats = [float(v) for v in data["embed_latencies_ms"].tolist()]
    except (OSError, ValueError, KeyError, IndexError) as exc:
        log.warning("Ignoring unreadable CPU query checkpoint %s: %s", ckpt_path, exc)
        return None

    if ckpt_model != expected_model_id or ckpt_prec != cfg.embed_precision:
        log.info("CPU query checkpoint model/precision changed; recomputing.")
        return None
    if ckpt_ids != expected_ids:
        return None
    if q1024.shape != (len(items), cfg.embed_dim_full):
        return None
    if q512.shape != (len(items), cfg.embed_dim_mrl):
        return None
    if len(lats) != len(items):
        return None

    return (
        [np.asarray(q1024[i], dtype=np.float32) for i in range(len(items))],
        [np.asarray(q512[i], dtype=np.float32) for i in range(len(items))],
        lats,
    )


def _load_rerank_checkpoint(
    ckpt_path: Path,
    items: list[GoldenItem],
    sq8_candidates_8: list[list[dict[str, Any]]],
    expected_model_id: str,
    cfg: Settings,
) -> dict[int, dict[str, Any]]:
    """Load valid Step 5 CPU per-query rerank records matching precision & max_length."""
    if not ckpt_path.exists():
        return {}

    loaded: dict[int, dict[str, Any]] = {}
    had_corrupt_tail = False
    raw_lines = ckpt_path.read_text(encoding="utf-8").splitlines()
    for line_no, raw in enumerate(raw_lines, start=1):
        stripped = raw.strip()
        if not stripped:
            continue
        try:
            rec = json.loads(stripped)
        except json.JSONDecodeError as exc:
            log.warning("Skipping corrupt JSON line %d in %s: %s", line_no, ckpt_path, exc)
            had_corrupt_tail = True
            continue

        if not isinstance(rec, dict):
            had_corrupt_tail = True
            continue
        if rec.get("rerank_model_id") != expected_model_id:
            continue
        if rec.get("rerank_precision") != cfg.rerank_precision:
            continue
        if rec.get("rerank_max_length") != cfg.rerank_max_length:
            continue
        if bool(rec.get("contingency_l_applied", False)) != bool(cfg.contingency_l_applied):
            continue
        idx = rec.get("idx")
        if not isinstance(idx, int) or idx < 0 or idx >= len(items):
            continue
        if rec.get("test_id") != items[idx].test_id:
            continue
        scored_entries = rec.get("scored_8")
        if not isinstance(scored_entries, list):
            continue

        expected_cids = {str(c["chunk_id"]) for c in sq8_candidates_8[idx]}
        rec_cids = {
            str(entry.get("chunk_id", ""))
            for entry in scored_entries
            if isinstance(entry, dict)
        }
        if len(scored_entries) != len(expected_cids) or rec_cids != expected_cids:
            continue

        loaded[idx] = rec

    if had_corrupt_tail and loaded:
        sanitized = "\n".join(
            json.dumps(loaded[idx], ensure_ascii=False) for idx in sorted(loaded.keys())
        )
        _atomic_write_text(ckpt_path, sanitized + "\n")
    return loaded


def _rehydrate_scored_8(
    scored_entries: list[dict[str, Any]],
    store_512: ChunkStore,
) -> list[dict[str, Any]]:
    """Reconstruct scored candidate dicts ordered identically to Reranker.rerank."""
    hydrated: list[dict[str, Any]] = []
    for entry in scored_entries:
        chunk_rec = store_512.get(str(entry["chunk_id"]))
        chunk_rec["rerank_score"] = float(entry["rerank_score"])
        chunk_rec["rerank_raw_logit"] = float(entry.get("rerank_raw_logit", 0.0))
        hydrated.append(chunk_rec)
    hydrated.sort(
        key=lambda item: (item["rerank_raw_logit"], item["rerank_score"]),
        reverse=True,
    )
    return hydrated


# ─────────────────────────────────────────────────────────────────────────────
# Start-of-run assertions & GPU-hybrid archival
# ─────────────────────────────────────────────────────────────────────────────


def _check_competing_processes() -> None:
    """Ensure no competing heavy KAVACH/pytest/serving python processes run concurrently."""
    my_pid = os.getpid()
    ancestor_pids: set[int] = {my_pid}
    try:
        proc = psutil.Process(my_pid)
        for parent in proc.parents():
            ancestor_pids.add(parent.pid)
    except psutil.Error:
        pass

    flagged: list[dict[str, Any]] = []
    markers = ("kavach.retrieval.", "pytest", "uvicorn", "vllm")
    for p in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            pid = p.info["pid"]
            if pid in ancestor_pids:
                continue
            name = (p.info.get("name") or "").lower()
            if "python" not in name and "pytest" not in name:
                continue
            cmdline = " ".join(p.info.get("cmdline") or []).lower()
            if any(m in cmdline for m in markers):
                flagged.append({"pid": pid, "name": name, "cmdline": cmdline[:200]})
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            continue

    if flagged:
        raise HardwareIsolationError(
            "Competing KAVACH/Python inference processes detected during serialized ablation",
            details={"competing_processes": flagged},
        )


def _assert_runtime_preconditions(cfg: Settings) -> dict[str, Any]:
    """Assert CPU-only state, no .venv_gpu, no competing processes, precision_lock & calibration."""
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise HardwareIsolationError(
            "CUDA_VISIBLE_DEVICES must be set to empty string '' for CPU-only ablation",
            details={"CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES")},
        )
    project_root = getattr(cfg, "project_root", cfg.models_dir.parent)
    venv_gpu_dir = Path(project_root) / ".venv_gpu"
    if venv_gpu_dir.exists():
        raise HardwareIsolationError(
            f"Forbidden .venv_gpu environment directory detected: {venv_gpu_dir}",
            details={"venv_gpu_path": str(venv_gpu_dir)},
        )
    probe = vram_probe()
    if probe["cuda_available"] or probe["pid_in_nvidia_smi"]:
        raise HardwareIsolationError(
            "GPU attachment detected before Phase 1.5 CPU ablation",
            details=dict(probe),
        )
    _check_competing_processes()
    precision_lock = verify_precision_lock(cfg)
    cal = verify_abstain_calibration(cfg)
    if cal is None:
        raise ConfigError(
            "Phase 1.5 ablation requires calibrated abstention enabled with valid models/abstain_calibration.json",
            details={"abstention_enabled": cfg.abstention_enabled},
        )
    return precision_lock


def _archive_gpu_hybrid_artifacts(cfg: Settings) -> dict[str, str]:
    """Archive legacy GPU-hybrid checkpoints and reports into reports/archive/gpu_hybrid/.

    Called strictly inside live ablation execution (never on module import or T0).
    """
    archive_dir = cfg.reports_dir / "archive" / "gpu_hybrid"
    archive_dir.mkdir(parents=True, exist_ok=True)
    moved: dict[str, str] = {}

    legacy_moves: list[tuple[Path, Path]] = [
        (cfg.data_processed / "ablation_qvecs_ckpt.npz", archive_dir / "ablation_qvecs_ckpt.npz"),
        (cfg.data_processed / "ablation_rerank_ckpt.jsonl", archive_dir / "ablation_rerank_ckpt.jsonl"),
    ]
    for src, dst in legacy_moves:
        if src.exists() and not dst.exists():
            shutil.move(str(src), str(dst))
            moved[src.name] = str(dst)

    results_src = cfg.reports_dir / "retrieval_ablation_results.json"
    results_dst = archive_dir / "retrieval_ablation_results.json"
    report_src = cfg.reports_dir / "retrieval_ablation_report.md"
    report_dst = archive_dir / "retrieval_ablation_report.md"

    should_archive_reports = False
    if results_src.exists() and not results_dst.exists():
        try:
            existing = json.loads(results_src.read_text(encoding="utf-8"))
            is_phase15 = bool(existing.get("hardware", {}).get("cpu_only_phase15", False))
            should_archive_reports = not is_phase15
        except (OSError, json.JSONDecodeError):
            should_archive_reports = True

    if should_archive_reports:
        shutil.move(str(results_src), str(results_dst))
        moved[results_src.name] = str(results_dst)
        if report_src.exists() and not report_dst.exists():
            shutil.move(str(report_src), str(report_dst))
            moved[report_src.name] = str(report_dst)

    if moved:
        log.info("Archived legacy GPU-hybrid artifacts: %s", moved)
    return moved


def _load_archived_gpu_hybrid_reference(cfg: Settings) -> dict[str, Any] | None:
    """Load archived GPU-hybrid ablation JSON if present for delta comparison."""
    ref_path = cfg.reports_dir / "archive" / "gpu_hybrid" / "retrieval_ablation_results.json"
    if not ref_path.exists():
        return None
    try:
        return json.loads(ref_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        log.warning("Could not parse archived GPU-hybrid results %s: %s", ref_path, exc)
        return None


def _collect_hardware_info(cfg: Settings, precision_lock: dict[str, Any]) -> dict[str, Any]:
    """Assemble full CPU hardware and precision provenance banner."""
    return {
        "cpu_only_phase15": True,
        "os_platform": platform.platform(),
        "cpu_model": platform.processor(),
        "physical_cores": psutil.cpu_count(logical=False),
        "logical_cpus": psutil.cpu_count(logical=True),
        "torch_version": torch.__version__,
        "quantized_engine": getattr(torch.backends.quantized, "engine", "unknown"),
        "cpu_affinity_mode": cfg.cpu_affinity_mode,
        "embed_precision": cfg.embed_precision,
        "rerank_precision": cfg.rerank_precision,
        "router_precision": cfg.router_precision,
        "rerank_max_length": cfg.rerank_max_length,
        "contingency_l_applied": bool(cfg.contingency_l_applied),
        "rerank_long_query_max_length": cfg.rerank_long_query_max_length,
        "rerank_long_query_token_threshold": cfg.rerank_long_query_token_threshold,
        "abstention_enabled": cfg.abstention_enabled,
        "abstain_logit_threshold": cfg.abstain_logit_threshold,
        "rss_ceiling_mb": cfg.rss_ceiling_mb,
        "precision_lock_timestamp": precision_lock.get("locked_at"),
        "vram_probe": dict(vram_probe()),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Metric, latency & score-distribution aggregation
# ─────────────────────────────────────────────────────────────────────────────


def _compute_ranking_block(
    items: list[GoldenItem],
    retrieved_lists: list[list[str]],
    output_k: int,
    ground_truth_fn: Any,
    check_both_versions: bool = False,
) -> dict[str, Any]:
    """Compute Recall@k, Precision@k, nDCG@k for every k<=output_k, plus MRR."""
    per_type_indices: dict[str, list[int]] = {qt.value: [] for qt in _IN_SCOPE_TYPES}
    in_scope_indices: list[int] = []

    for idx, item in enumerate(items):
        if item.question_type == QuestionType.OUT_OF_SCOPE:
            continue
        in_scope_indices.append(idx)
        per_type_indices[item.question_type.value].append(idx)

    def _agg_for_indices(indices: list[int], include_conflict_check: bool) -> dict[str, Any]:
        if not indices:
            return {"count": 0, "mrr": 0.0, "by_k": {}}
        mrr_vals: list[float] = []
        by_k: dict[str, dict[str, float]] = {}
        both_ver_hits = 0

        for idx in indices:
            gt = ground_truth_fn(items[idx])
            ret = retrieved_lists[idx][:output_k]
            mrr_vals.append(reciprocal_rank(ret, gt))
            if include_conflict_check and gt:
                ret_set = set(ret)
                if all(doc in ret_set for doc in gt):
                    both_ver_hits += 1

        for k in range(1, output_k + 1):
            r_vals: list[float] = []
            p_vals: list[float] = []
            n_vals: list[float] = []
            for idx in indices:
                gt = ground_truth_fn(items[idx])
                ret = retrieved_lists[idx]
                r_vals.append(recall_at_k(ret, gt, k))
                p_vals.append(precision_at_k(ret, gt, k))
                n_vals.append(ndcg_at_k(ret, gt, k))
            by_k[str(k)] = {
                "recall": round(float(np.mean(r_vals)), 4),
                "precision": round(float(np.mean(p_vals)), 4),
                "ndcg": round(float(np.mean(n_vals)), 4),
            }

        out: dict[str, Any] = {
            "count": len(indices),
            "mrr": round(float(np.mean(mrr_vals)), 4),
            "by_k": by_k,
        }
        if include_conflict_check:
            out["both_versions_retrieved"] = round(both_ver_hits / len(indices), 4)
            out["both_versions_count"] = both_ver_hits
        return out

    overall = _agg_for_indices(in_scope_indices, include_conflict_check=False)
    per_type: dict[str, Any] = {}
    for qt_val, indices in per_type_indices.items():
        is_conflict = check_both_versions and qt_val == QuestionType.CONFLICT_TRAP.value
        per_type[qt_val] = _agg_for_indices(indices, include_conflict_check=is_conflict)

    return {"overall": overall, "per_type": per_type}


def _score_stats(vals: list[float]) -> dict[str, Any]:
    """Compute count, min, median, mean, p95, and max for a numeric list."""
    if not vals:
        return {
            "count": 0,
            "min": None,
            "median": None,
            "mean": None,
            "p95": None,
            "max": None,
        }
    arr = np.asarray(vals, dtype=np.float64)
    return {
        "count": len(vals),
        "min": round(float(np.min(arr)), 6),
        "median": round(float(np.median(arr)), 6),
        "mean": round(float(np.mean(arr)), 6),
        "p95": round(float(np.percentile(arr, 95)), 6),
        "max": round(float(np.max(arr)), 6),
    }


def _compute_reranker_distributions(
    items: list[GoldenItem],
    all_scored_8: list[list[dict[str, Any]]],
    run5_thresh_records: list[list[dict[str, Any]]],
    run6_thresh_records: list[list[dict[str, Any]]],
    top_k_default: int,
    top_k_max: int,
) -> dict[str, Any]:
    """Summarize rerank_score and rerank_raw_logit for in-scope vs OUT_OF_SCOPE."""
    in_indices = [
        i for i, it in enumerate(items) if it.question_type != QuestionType.OUT_OF_SCOPE
    ]
    oos_indices = [
        i for i, it in enumerate(items) if it.question_type == QuestionType.OUT_OF_SCOPE
    ]

    def _pair_stats(records_by_query: list[list[dict[str, Any]]], indices: list[int]) -> dict[str, Any]:
        probs = [float(c["rerank_score"]) for i in indices for c in records_by_query[i]]
        logits = [
            float(c.get("rerank_raw_logit", 0.0)) for i in indices for c in records_by_query[i]
        ]
        return {
            "rerank_score": _score_stats(probs),
            "rerank_raw_logit": _score_stats(logits),
        }

    top1_only = [recs[:1] for recs in all_scored_8]
    run5_unthresh = [recs[:top_k_default] for recs in all_scored_8]
    run6_unthresh = [recs[:top_k_max] for recs in all_scored_8]

    return {
        "top1_in_scope": _pair_stats(top1_only, in_indices),
        "top1_out_of_scope": _pair_stats(top1_only, oos_indices),
        "topk_run5_in_scope": _pair_stats(run5_unthresh, in_indices),
        "topk_run5_out_of_scope": _pair_stats(run5_unthresh, oos_indices),
        "topk_run6_in_scope": _pair_stats(run6_unthresh, in_indices),
        "topk_run6_out_of_scope": _pair_stats(run6_unthresh, oos_indices),
        "all_scored8_in_scope": _pair_stats(all_scored_8, in_indices),
        "all_scored8_out_of_scope": _pair_stats(all_scored_8, oos_indices),
        "thresholded_run5_in_scope": _pair_stats(run5_thresh_records, in_indices),
        "thresholded_run5_out_of_scope": _pair_stats(run5_thresh_records, oos_indices),
        "thresholded_run6_in_scope": _pair_stats(run6_thresh_records, in_indices),
        "thresholded_run6_out_of_scope": _pair_stats(run6_thresh_records, oos_indices),
    }


def _latency_summary(
    embed_ms: list[float],
    stage_ms: list[float],
    items: list[GoldenItem] | None = None,
) -> dict[str, Any]:
    """Summarize P50, P95, and mean latencies overall and per QuestionType."""
    embed_arr = np.asarray(embed_ms, dtype=np.float64)
    stage_arr = np.asarray(stage_ms, dtype=np.float64)
    total_arr = embed_arr + stage_arr

    summary: dict[str, Any] = {
        "mean_embed_ms": round(float(np.mean(embed_arr)), 3),
        "p50_embed_ms": round(float(np.percentile(embed_arr, 50)), 3),
        "p95_embed_ms": round(float(np.percentile(embed_arr, 95)), 3),
        "mean_stage_ms": round(float(np.mean(stage_arr)), 3),
        "p50_stage_ms": round(float(np.percentile(stage_arr, 50)), 3),
        "p95_stage_ms": round(float(np.percentile(stage_arr, 95)), 3),
        "mean_total_ms": round(float(np.mean(total_arr)), 3),
        "p50_total_ms": round(float(np.percentile(total_arr, 50)), 3),
        "p95_total_ms": round(float(np.percentile(total_arr, 95)), 3),
    }
    if items is not None:
        per_type: dict[str, dict[str, Any]] = {}
        for qt in _ALL_TYPES:
            idxs = [i for i, it in enumerate(items) if it.question_type == qt]
            if not idxs:
                continue
            sub = total_arr[idxs]
            per_type[qt.value] = {
                "count": len(idxs),
                "mean_total_ms": round(float(np.mean(sub)), 3),
                "p50_total_ms": round(float(np.percentile(sub, 50)), 3),
                "p95_total_ms": round(float(np.percentile(sub, 95)), 3),
            }
        summary["per_question_type"] = per_type
    return summary


def _evaluate_router(
    items: list[GoldenItem],
    settings: Settings,
    router: AgentIntentRouter | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Evaluate resident AgentIntentRouter across all 150 golden queries."""
    active_router = router
    if active_router is None:
        active_router = AgentIntentRouter(settings)
        active_router.load()
        active_router.warmup()

    decisions: list[dict[str, Any]] = []
    latencies: list[float] = []
    conf_matrix: list[list[int]] = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    route_to_idx = {r: i for i, r in enumerate(_ROUTE_ORDER)}

    per_type_correct: dict[str, int] = {qt.value: 0 for qt in QuestionType}
    per_type_total: dict[str, int] = {qt.value: 0 for qt in QuestionType}
    misclassified: list[dict[str, Any]] = []

    correct = 0
    for item in items:
        dec = active_router.route(item.query)
        latencies.append(float(dec.latency_ms))
        decisions.append(
            {
                "test_id": item.test_id,
                "route": dec.route,
                "confidence": round(float(dec.confidence), 6),
                "matched_identifier": dec.matched_identifier,
                "is_malformed": bool(dec.is_malformed),
                "latency_ms": round(float(dec.latency_ms), 3),
            }
        )

        true_r = item.expected_route
        pred_r = dec.route
        conf_matrix[route_to_idx[true_r]][route_to_idx[pred_r]] += 1
        per_type_total[item.question_type.value] += 1
        if pred_r == true_r:
            correct += 1
            per_type_correct[item.question_type.value] += 1
        else:
            misclassified.append(
                {
                    "test_id": item.test_id,
                    "question_type": item.question_type.value,
                    "query": item.query[:200],
                    "expected_route": true_r,
                    "predicted_route": pred_r,
                    "confidence": round(float(dec.confidence), 4),
                    "is_malformed": bool(dec.is_malformed),
                }
            )

    per_route_metrics: dict[str, dict[str, Any]] = {}
    for r_idx, r_name in enumerate(_ROUTE_ORDER):
        tp = conf_matrix[r_idx][r_idx]
        support = sum(conf_matrix[r_idx])
        pred_total = sum(conf_matrix[row][r_idx] for row in range(3))
        prec = tp / pred_total if pred_total > 0 else 0.0
        rec = tp / support if support > 0 else 0.0
        per_route_metrics[r_name] = {
            "support": support,
            "predicted": pred_total,
            "correct": tp,
            "precision": round(prec, 4),
            "recall": round(rec, 4),
        }

    per_type_acc = {
        qt: {
            "correct": per_type_correct[qt],
            "total": per_type_total[qt],
            "accuracy": round(
                per_type_correct[qt] / per_type_total[qt] if per_type_total[qt] > 0 else 0.0,
                4,
            ),
        }
        for qt in per_type_total
    }

    report = {
        "evaluated_at": datetime.now(tz=timezone.utc).isoformat(),
        "total_queries": len(items),
        "correct_count": correct,
        "overall_accuracy": round(correct / len(items), 4),
        "route_labels": _ROUTE_ORDER,
        "confusion_matrix": conf_matrix,
        "per_route_metrics": per_route_metrics,
        "per_question_type_accuracy": per_type_acc,
        "mean_latency_ms": round(float(np.mean(latencies)), 3),
        "p50_latency_ms": round(float(np.percentile(latencies, 50)), 3),
        "p95_latency_ms": round(float(np.percentile(latencies, 95)), 3),
        "misclassified": misclassified,
    }
    return report, decisions


def _build_gpu_hybrid_comparison(
    current_runs: list[dict[str, Any]],
    ref_results: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Compare current CPU-only Runs 1..6 against superseded GPU-hybrid baseline."""
    if ref_results is None:
        return None
    ref_by_id = {r["run_id"]: r for r in ref_results.get("runs", [])}
    rows: list[dict[str, Any]] = []
    flagged_runs: list[str] = []
    for run in current_runs:
        rid = run["run_id"]
        ref_run = ref_by_id.get(rid)
        if not ref_run:
            continue
        k_str = str(run["output_k"])
        cur_dm = run["doc_metrics"]["overall"]
        ref_dm = ref_run["doc_metrics"]["overall"]
        cur_mrr = float(cur_dm["mrr"])
        ref_mrr = float(ref_dm["mrr"])
        delta_mrr = round(cur_mrr - ref_mrr, 4)
        cur_rk = float(cur_dm["by_k"][k_str]["recall"])
        ref_rk = float(ref_dm["by_k"][k_str]["recall"])
        delta_rk = round(cur_rk - ref_rk, 4)
        flagged = abs(delta_mrr) > 0.01
        if flagged:
            flagged_runs.append(rid)
        rows.append(
            {
                "run_id": rid,
                "output_k": run["output_k"],
                "cpu_doc_mrr": cur_mrr,
                "superseded_ref_doc_mrr": ref_mrr,
                "delta_doc_mrr": delta_mrr,
                "cpu_doc_recall_at_k": cur_rk,
                "superseded_ref_doc_recall_at_k": ref_rk,
                "delta_doc_recall_at_k": delta_rk,
                "cpu_mean_total_ms": run["latency"]["mean_total_ms"],
                "superseded_ref_mean_total_ms": ref_run["latency"]["mean_total_ms"],
                "mrr_delta_gt_0_01": flagged,
            }
        )
    return {
        "reference_label": "superseded_gpu_hybrid_reference",
        "reference_generated_at": ref_results.get("generated_at"),
        "flagged_mrr_drift_runs": flagged_runs,
        "runs": rows,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Markdown report generator
# ─────────────────────────────────────────────────────────────────────────────


def _fmt_stat(val: float | None) -> str:
    """Format an optional distribution statistic cleanly for Markdown tables."""
    if val is None:
        return "N/A"
    return f"{val:.4f}"


def _format_markdown_report(results: dict[str, Any]) -> str:
    """Render reports/retrieval_ablation_report.md from real measured Phase 1.5 CPU runs."""
    hw = results["hardware"]
    lines: list[str] = [
        "# KAVACH-RAG Phase 1.5 CPU-Only Retrieval Ablation Report",
        "",
        f"- **Generated (UTC):** {results['generated_at']}",
        f"- **Hardware Provenance:** CPU-only `{hw['cpu_model']}` ({hw['physical_cores']}P / {hw['logical_cpus']}L cores), "
        f"OS `{hw['os_platform']}`, torch `{hw['torch_version']}` (quantized engine `{hw['quantized_engine']}`), "
        f"affinity `{hw['cpu_affinity_mode']}`",
        f"- **Model Precisions & Truncation:** embed=`{hw['embed_precision']}`, rerank=`{hw['rerank_precision']}`, "
        f"router=`{hw['router_precision']}`, rerank_max_length=`{hw['rerank_max_length']}`, "
        f"contingency_l=`{hw['contingency_l_applied']}`, abstain_threshold_T=`{hw['abstain_logit_threshold']}`",
        f"- **Memory & GPU Guard:** Peak RSS=`{hw['peak_rss_mb']} MB` (ceiling `<= {hw['rss_ceiling_mb']} MB`), "
        f"CUDA available=`{hw['vram_probe']['cuda_available']}`, PID in nvidia-smi=`{hw['vram_probe']['pid_in_nvidia_smi']}`",
        f"- **Corpus Chunks:** w512={results['corpus']['w512_chunks']}, w256={results['corpus']['w256_chunks']}",
        f"- **Golden Queries:** {results['golden_count']} ({results['in_scope_count']} in-scope, {results['oos_count']} OUT_OF_SCOPE)",
        f"- **Embedder:** `{results['models']['embed_model_id']}` | **Reranker:** `{results['models']['rerank_model_id']}` (fallback=`{results['models']['reranker_is_fallback']}`)",
        "",
        "## 1. Headline Document-Level Ablation Results (140 In-Scope Queries)",
        "",
        "| Run | Configuration | k | Recall@1 | Recall@3 | Recall@k | MRR | nDCG@3 | nDCG@k | Mean ms | P50 ms | P95 ms |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    runs = results["runs"]
    for run in runs:
        k = run["output_k"]
        dm = run["doc_metrics"]["overall"]
        by_k = dm["by_k"]
        r1 = f"{by_k['1']['recall']:.4f}"
        r3 = f"{by_k['3']['recall']:.4f}" if "3" in by_k else "N/A"
        rk = f"{by_k[str(k)]['recall']:.4f}"
        n3 = f"{by_k['3']['ndcg']:.4f}" if "3" in by_k else "N/A"
        nk = f"{by_k[str(k)]['ndcg']:.4f}"
        lat = run["latency"]
        lines.append(
            f"| {run['run_id']} | {run['label']} | {k} | {r1} | {r3} | {rk} | "
            f"{dm['mrr']:.4f} | {n3} | {nk} | {lat['mean_total_ms']:.2f} | "
            f"{lat['p50_total_ms']:.2f} | {lat['p95_total_ms']:.2f} |"
        )
    lines.append("")

    lines.extend(
        [
            "## 2. Chunk-Level Ablation Results (Runs 2–6, w512)",
            "",
            "| Run | Configuration | k | Recall@1 | Recall@3 | Recall@k | Prec@3 | Prec@k | MRR | nDCG@3 | nDCG@k |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for run in runs:
        cm = run.get("chunk_metrics")
        if cm is None:
            continue
        k = run["output_k"]
        ov = cm["overall"]
        by_k = ov["by_k"]
        r1 = f"{by_k['1']['recall']:.4f}"
        r3 = f"{by_k['3']['recall']:.4f}" if "3" in by_k else "N/A"
        rk = f"{by_k[str(k)]['recall']:.4f}"
        p3 = f"{by_k['3']['precision']:.4f}" if "3" in by_k else "N/A"
        pk = f"{by_k[str(k)]['precision']:.4f}"
        n3 = f"{by_k['3']['ndcg']:.4f}" if "3" in by_k else "N/A"
        nk = f"{by_k[str(k)]['ndcg']:.4f}"
        lines.append(
            f"| {run['run_id']} | {run['label']} | {k} | {r1} | {r3} | {rk} | "
            f"{p3} | {pk} | {ov['mrr']:.4f} | {n3} | {nk} |"
        )
    lines.append("")

    lines.append("## 3. Per-Question-Type Latency Breakdown (Runs 3 & 5)")
    lines.append("")
    lines.append("| Run | Question Type | Count | Mean ms | P50 ms | P95 ms |")
    lines.append("| --- | --- | ---: | ---: | ---: | ---: |")
    for run in runs:
        if run["run_id"] not in ("Run 3", "Run 5"):
            continue
        for qt, lstat in run["latency"].get("per_question_type", {}).items():
            lines.append(
                f"| {run['run_id']} | `{qt}` | {lstat['count']} | {lstat['mean_total_ms']:.2f} | "
                f"{lstat['p50_total_ms']:.2f} | {lstat['p95_total_ms']:.2f} |"
            )
    lines.append("")

    lines.append("## 4. Per-Question-Type Retrieval Breakdown (Ungated)")
    lines.append("")
    lines.append(
        "| Run | Question Type | Count | Doc Recall@k | Doc MRR | Doc nDCG@k | Chunk Recall@k | Chunk MRR | Both Versions |"
    )
    lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |")
    for run in runs:
        k_str = str(run["output_k"])
        for qt in _IN_SCOPE_TYPES:
            d_pt = run["doc_metrics"]["per_type"][qt.value]
            d_rk = d_pt["by_k"][k_str]["recall"]
            d_nk = d_pt["by_k"][k_str]["ndcg"]
            if run.get("chunk_metrics") is not None:
                c_pt = run["chunk_metrics"]["per_type"][qt.value]
                c_rk = f"{c_pt['by_k'][k_str]['recall']:.4f}"
                c_mrr = f"{c_pt['mrr']:.4f}"
            else:
                c_rk = "N/A"
                c_mrr = "N/A"
            both_str = (
                f"{d_pt['both_versions_retrieved']:.4f} ({d_pt['both_versions_count']}/{d_pt['count']})"
                if "both_versions_retrieved" in d_pt
                else "—"
            )
            lines.append(
                f"| {run['run_id']} | {qt.value} | {d_pt['count']} | {d_rk:.4f} | "
                f"{d_pt['mrr']:.4f} | {d_nk:.4f} | {c_rk} | {c_mrr} | {both_str} |"
            )
    lines.append("")

    lines.append("## 5. Calibrated Logit Abstention & Gated Effective Retrieval (Runs 5 & 6)")
    lines.append("")
    lines.append(
        "| Run | k | Threshold T | OOS Abstention | In-Scope False Abstain | Ungated Doc Recall@k | Gated Effective Doc Recall@k | Gated Doc MRR |"
    )
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for run in runs:
        th = run.get("thresholded_eval")
        if th is None:
            continue
        k_str = str(run["output_k"])
        ung_rk = run["doc_metrics"]["overall"]["by_k"][k_str]["recall"]
        g_dm = th["gated_effective_doc_metrics"]["overall"]
        lines.append(
            f"| {run['run_id']} | {run['output_k']} | {th['threshold_T']} | "
            f"{th['abstention_rate']:.4f} ({th['abstained_oos_count']}/{th['oos_total']}) | "
            f"{th['false_abstain_rate']:.4f} ({th['false_abstain_in_scope_count']}/{th['in_scope_total']}) | "
            f"{ung_rk:.4f} | {g_dm['by_k'][k_str]['recall']:.4f} | {g_dm['mrr']:.4f} |"
        )
    lines.append("")

    dist = results.get("reranker_score_distribution")
    if dist:
        lines.append("### Reranker Score & Raw-Logit Distributions")
        lines.append("")
        lines.append("| Subset | Metric | Count | Min | Median | Mean | P95 | Max |")
        lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |")
        for key, label in [
            ("top1_in_scope", "Top-1 In-Scope"),
            ("top1_out_of_scope", "Top-1 OUT_OF_SCOPE"),
            ("thresholded_run5_in_scope", "Run 5 Gated In-Scope"),
            ("thresholded_run5_out_of_scope", "Run 5 Gated OUT_OF_SCOPE"),
        ]:
            if key not in dist:
                continue
            for metric_key in ("rerank_raw_logit", "rerank_score"):
                st = dist[key][metric_key]
                lines.append(
                    f"| {label} | `{metric_key}` | {st['count']} | "
                    f"{_fmt_stat(st['min'])} | {_fmt_stat(st['median'])} | "
                    f"{_fmt_stat(st['mean'])} | {_fmt_stat(st['p95'])} | {_fmt_stat(st['max'])} |"
                )
        lines.append("")

    rg = results["router_golden_eval"]
    lines.append("## 6. Tier 1 Router & End-to-End Live Pipeline Evaluation")
    lines.append("")
    lines.append(
        f"- **Router Accuracy:** {rg['overall_accuracy']:.4f} ({rg['correct_count']}/{rg['total_queries']}), "
        f"Latency Mean={rg['mean_latency_ms']:.3f} ms / P50={rg['p50_latency_ms']:.3f} ms / P95={rg['p95_latency_ms']:.3f} ms"
    )
    e2e = results.get("end_to_end_pipeline_eval")
    if e2e:
        lines.append(
            f"- **End-to-End Pipeline OOS Block Rate:** {e2e['oos_blocked_rate']:.4f} ({e2e['oos_blocked_count']}/{e2e['oos_total']}), "
            f"**In-Scope False Block Rate:** {e2e['inscope_false_block_rate']:.4f} ({e2e['inscope_false_block_count']}/{e2e['inscope_total']}), "
            f"**Latency:** Mean={e2e['mean_total_ms']:.2f} ms / P50={e2e['p50_total_ms']:.2f} ms / P95={e2e['p95_total_ms']:.2f} ms"
        )
    lines.append("")

    comp = results.get("gpu_hybrid_comparison")
    if comp and comp.get("runs"):
        lines.append("## 7. Comparison vs Superseded GPU-Hybrid Reference")
        lines.append("")
        lines.append(
            "| Run | CPU Doc MRR | Superseded Ref MRR | Delta MRR | CPU Recall@k | Ref Recall@k | Delta Recall@k | Drift > 0.01 |"
        )
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |")
        for row in comp["runs"]:
            flag_str = "FLAGGED" if row["mrr_delta_gt_0_01"] else "ok"
            lines.append(
                f"| {row['run_id']} | {row['cpu_doc_mrr']:.4f} | {row['superseded_ref_doc_mrr']:.4f} | "
                f"{row['delta_doc_mrr']:+.4f} | {row['cpu_doc_recall_at_k']:.4f} | "
                f"{row['superseded_ref_doc_recall_at_k']:.4f} | {row['delta_doc_recall_at_k']:+.4f} | {flag_str} |"
            )
        lines.append("")

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Main ablation execution
# ─────────────────────────────────────────────────────────────────────────────


def run_ablation(
    settings: Settings | None = None,
    fresh: bool = False,
) -> dict[str, Any]:
    """Execute the Phase 1.5 CPU-only 6-run retrieval ablation inside BenchLock."""
    apply_hardware_guard()
    cfg = settings or get_settings()

    cfg.logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = cfg.logs_dir / "ablation.log"
    file_handler = logging.FileHandler(str(log_path), mode="a", encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s", "%H:%M:%S")
    )
    root_logger = logging.getLogger()
    root_logger.addHandler(file_handler)

    try:
        return _run_ablation_impl(cfg, fresh=fresh)
    finally:
        root_logger.removeHandler(file_handler)
        file_handler.flush()
        file_handler.close()


def _run_ablation_impl(cfg: Settings, fresh: bool = False) -> dict[str, Any]:
    """Internal BenchLock-serialized ablation implementation."""
    precision_lock = _assert_runtime_preconditions(cfg)
    _archive_gpu_hybrid_artifacts(cfg)
    archived_ref = _load_archived_gpu_hybrid_reference(cfg)

    golden_path = cfg.data_golden / "golden_150.jsonl"
    p512 = cfg.data_processed / "corpus_chunks.parquet"
    p256 = cfg.data_processed / "corpus_chunks_w256.parquet"
    npz512 = cfg.data_processed / "embeddings" / "w512.npz"
    npz256 = cfg.data_processed / "embeddings" / "w256.npz"
    qvec_ckpt_path = cfg.data_processed / "ablation_cpu_qvecs_ckpt.npz"
    rerank_ckpt_path = cfg.data_processed / "ablation_cpu_rerank_ckpt.jsonl"

    if fresh:
        for ckpt in (qvec_ckpt_path, rerank_ckpt_path):
            if ckpt.exists():
                ckpt.unlink()
                log.info("Removed CPU checkpoint (--fresh): %s", ckpt)

    items = _load_golden(golden_path)
    preflight = _read_preflight(cfg)
    expected_embed_id = _expected_embed_model_id(cfg, preflight)

    peak_rss = rss_mb()

    def _track_rss(stage: str) -> float:
        nonlocal peak_rss
        cur = rss_mb()
        peak_rss = max(peak_rss, cur)
        log.info("[%s] RSS=%.1f MB (peak=%.1f MB, ceiling=%.1f MB)", stage, cur, peak_rss, cfg.rss_ceiling_mb)
        if cur > cfg.rss_ceiling_mb:
            raise HardwareIsolationError(
                f"RSS {cur:.1f} MB exceeded ceiling {cfg.rss_ceiling_mb:.1f} MB at {stage}",
                details={"stage": stage, "rss_mb": cur, "rss_ceiling_mb": cfg.rss_ceiling_mb},
            )
        return cur

    _track_rss("startup")

    with BenchLock(owner="ablation"):
        # Single resident warmed VectorRetrievalPipeline (no model reloading)
        pipeline = VectorRetrievalPipeline(cfg)
        assert pipeline._reranker is not None
        if pipeline._reranker.is_fallback:
            raise RetrievalError(
                "Phase 1.5 CPU ablation requires primary cross-encoder reranker; fallback detected",
                details={"model_id": pipeline._reranker.model_id},
            )
        _track_rss("after_pipeline_resident_warmup")

        embed_model_id = pipeline._embedder.model_id
        rerank_model_id = pipeline._reranker.model_id
        reranker_is_fallback = bool(pipeline._reranker.is_fallback)
        pipeline._reranker.reset_truncation_stats()

        # ── Step 1: Embed 150 golden queries once (full1024 + mrl512) ─────────
        cached_qvecs = (
            None
            if fresh
            else _try_load_qvec_checkpoint(qvec_ckpt_path, items, expected_embed_id, cfg)
        )
        if cached_qvecs is not None:
            q_vecs_1024, q_vecs_512, embed_latencies_ms = cached_qvecs
            log.info("Loaded CPU query embeddings from %s", qvec_ckpt_path)
        else:
            log.info("Embedding 150 golden queries on CPU (full1024 + mrl512)...")
            q_vecs_1024 = []
            q_vecs_512 = []
            embed_latencies_ms = []
            for idx, item in enumerate(items, start=1):
                t0 = time.perf_counter()
                batch = pipeline._embedder.embed([item.query], kind="query", batch_size=1)
                elapsed_ms = (time.perf_counter() - t0) * 1000.0
                q_vecs_1024.append(np.asarray(batch.full1024[0], dtype=np.float32))
                q_vecs_512.append(np.asarray(batch.mrl512[0], dtype=np.float32))
                embed_latencies_ms.append(elapsed_ms)
                if idx == 1 or idx % 25 == 0 or idx == len(items):
                    _track_rss(f"embed_{idx}_{len(items)}")
            pipeline._embedder.clear_cache()
            _atomic_write_qvec_checkpoint(
                qvec_ckpt_path,
                embed_model_id=embed_model_id,
                embed_precision=cfg.embed_precision,
                test_ids=[it.test_id for it in items],
                q_vecs_1024=q_vecs_1024,
                q_vecs_512=q_vecs_512,
                embed_latencies_ms=embed_latencies_ms,
            )

        # ── Step 2: Run 1 — w256 + FlatIndex(1024), k=3 ───────────────────────
        log.info("Executing Run 1: w256 + FlatIndex(1024), k=3...")
        store_256 = ChunkStore(p256)
        c2doc_256 = store_256.chunk_to_doc_map()
        with np.load(npz256, allow_pickle=False) as d256:
            cids_256 = [str(c) for c in d256["chunk_ids"]]
            vecs_256 = np.asarray(d256["full1024"], dtype=np.float32)

        flat_256 = FlatIndex(dim=cfg.embed_dim_full).add(vecs_256, cids_256)
        del vecs_256

        run1_doc_lists: list[list[str]] = []
        run1_stage_ms: list[float] = []
        for qvec in q_vecs_1024:
            t0 = time.perf_counter()
            hits = flat_256.search(qvec, k=3)
            run1_stage_ms.append((time.perf_counter() - t0) * 1000.0)
            run1_doc_lists.append(dedupe_to_docs([cid for cid, _ in hits], c2doc_256))

        run1_doc_metrics = _compute_ranking_block(
            items,
            run1_doc_lists,
            output_k=3,
            ground_truth_fn=lambda it: it.expected_doc_ids,
            check_both_versions=True,
        )
        run1_lat = _latency_summary(embed_latencies_ms, run1_stage_ms, items=items)
        w256_count = len(store_256)
        del flat_256, store_256, c2doc_256
        gc.collect()
        _track_rss("after_run1")

        # ── Step 3: Run 2 — w512 + FlatIndex(1024), k=3 ───────────────────────
        log.info("Executing Run 2: w512 + FlatIndex(1024), k=3...")
        store_512 = pipeline._store
        c2doc_512 = store_512.chunk_to_doc_map()
        with np.load(npz512, allow_pickle=False) as d512:
            cids_512 = [str(c) for c in d512["chunk_ids"]]
            vecs_512 = np.asarray(d512["full1024"], dtype=np.float32)

        flat_512 = FlatIndex(dim=cfg.embed_dim_full).add(vecs_512, cids_512)
        del vecs_512

        run2_chunk_lists: list[list[str]] = []
        run2_doc_lists: list[list[str]] = []
        run2_stage_ms: list[float] = []
        for qvec in q_vecs_1024:
            t0 = time.perf_counter()
            hits = flat_512.search(qvec, k=3)
            run2_stage_ms.append((time.perf_counter() - t0) * 1000.0)
            c_list = [cid for cid, _ in hits]
            run2_chunk_lists.append(c_list)
            run2_doc_lists.append(dedupe_to_docs(c_list, c2doc_512))

        run2_doc_metrics = _compute_ranking_block(
            items,
            run2_doc_lists,
            output_k=3,
            ground_truth_fn=lambda it: it.expected_doc_ids,
            check_both_versions=True,
        )
        run2_chunk_metrics = _compute_ranking_block(
            items,
            run2_chunk_lists,
            output_k=3,
            ground_truth_fn=lambda it: it.expected_source_chunk_ids,
        )
        run2_lat = _latency_summary(embed_latencies_ms, run2_stage_ms, items=items)
        w512_count = len(store_512)
        del flat_512, q_vecs_1024
        gc.collect()
        _track_rss("after_run2")

        # ── Step 4: Runs 3 & 4 — w512 + SQ8Index (shared k=candidate_k search)
        log.info("Executing Runs 3 & 4: w512 + SQ8Index (k=%d)...", cfg.candidate_k)
        sq8_candidates_8: list[list[dict[str, Any]]] = []
        sq8_stage_ms: list[float] = []
        run3_chunk_lists: list[list[str]] = []
        run3_doc_lists: list[list[str]] = []
        run4_chunk_lists: list[list[str]] = []
        run4_doc_lists: list[list[str]] = []

        for qvec in q_vecs_512:
            t0 = time.perf_counter()
            hits = pipeline._index.search(qvec, k=cfg.candidate_k)
            sq8_stage_ms.append((time.perf_counter() - t0) * 1000.0)
            c_list_8 = [cid for cid, _ in hits]
            c_list_3 = c_list_8[: cfg.rerank_top_k_default]
            run3_chunk_lists.append(c_list_3)
            run3_doc_lists.append(dedupe_to_docs(c_list_3, c2doc_512))
            run4_chunk_lists.append(c_list_8)
            run4_doc_lists.append(dedupe_to_docs(c_list_8, c2doc_512))
            sq8_candidates_8.append([store_512.get(cid) for cid in c_list_8])

        run3_doc_metrics = _compute_ranking_block(
            items,
            run3_doc_lists,
            output_k=3,
            ground_truth_fn=lambda it: it.expected_doc_ids,
            check_both_versions=True,
        )
        run3_chunk_metrics = _compute_ranking_block(
            items,
            run3_chunk_lists,
            output_k=3,
            ground_truth_fn=lambda it: it.expected_source_chunk_ids,
        )
        run3_lat = _latency_summary(embed_latencies_ms, sq8_stage_ms, items=items)

        run4_doc_metrics = _compute_ranking_block(
            items,
            run4_doc_lists,
            output_k=cfg.candidate_k,
            ground_truth_fn=lambda it: it.expected_doc_ids,
            check_both_versions=True,
        )
        run4_chunk_metrics = _compute_ranking_block(
            items,
            run4_chunk_lists,
            output_k=cfg.candidate_k,
            ground_truth_fn=lambda it: it.expected_source_chunk_ids,
        )
        run4_lat = _latency_summary(embed_latencies_ms, sq8_stage_ms, items=items)
        del q_vecs_512
        gc.collect()
        _track_rss("after_runs3_4")

        # ── Step 5: Runs 5 & 6 — Single-pass resident Reranker over 8 candidates
        cached_reranks = (
            {}
            if fresh
            else _load_rerank_checkpoint(
                rerank_ckpt_path,
                items=items,
                sq8_candidates_8=sq8_candidates_8,
                expected_model_id=rerank_model_id,
                cfg=cfg,
            )
        )
        log.info(
            "Executing Runs 5 & 6: resident CPU Reranker (%d cached, %d live)...",
            len(cached_reranks),
            len(items) - len(cached_reranks),
        )

        run5_chunk_lists: list[list[str]] = []
        run5_doc_lists: list[list[str]] = []
        run6_chunk_lists: list[list[str]] = []
        run6_doc_lists: list[list[str]] = []
        run5_gated_doc_lists: list[list[str]] = []
        run5_gated_chunk_lists: list[list[str]] = []
        run6_gated_doc_lists: list[list[str]] = []
        run6_gated_chunk_lists: list[list[str]] = []
        rerank_stage_ms: list[float] = []
        all_scored_8: list[list[dict[str, Any]]] = []
        run5_thresh_records: list[list[dict[str, Any]]] = []
        run6_thresh_records: list[list[dict[str, Any]]] = []

        t_rr_start = time.perf_counter()
        remaining = len(items) - len(cached_reranks)
        done_live = 0
        ckpt_fh = (
            open(rerank_ckpt_path, "a", encoding="utf-8")
            if remaining > 0
            else None
        )
        try:
            for idx, item in enumerate(items):
                cands = sq8_candidates_8[idx]
                if idx in cached_reranks:
                    rec = cached_reranks[idx]
                    rr_ms = float(rec["rr_ms"])
                    scored_8 = _rehydrate_scored_8(rec["scored_8"], store_512)
                else:
                    t0 = time.perf_counter()
                    scored_8 = pipeline._reranker.rerank(
                        item.query,
                        cands,
                        top_k=len(cands),
                        return_all=True,
                    )
                    rr_ms = (time.perf_counter() - t0) * 1000.0
                    done_live += 1

                    ckpt_rec = {
                        "rerank_model_id": rerank_model_id,
                        "rerank_precision": cfg.rerank_precision,
                        "rerank_max_length": cfg.rerank_max_length,
                        "contingency_l_applied": bool(cfg.contingency_l_applied),
                        "reranker_is_fallback": reranker_is_fallback,
                        "device": "cpu",
                        "idx": idx,
                        "test_id": item.test_id,
                        "rr_ms": round(rr_ms, 3),
                        "scored_8": [
                            {
                                "chunk_id": str(c["chunk_id"]),
                                "rerank_score": float(c["rerank_score"]),
                                "rerank_raw_logit": float(c["rerank_raw_logit"]),
                            }
                            for c in scored_8
                        ],
                    }
                    assert ckpt_fh is not None
                    ckpt_fh.write(json.dumps(ckpt_rec, ensure_ascii=False) + "\n")
                    ckpt_fh.flush()
                    os.fsync(ckpt_fh.fileno())

                    if done_live == 1 or done_live % 10 == 0 or done_live == remaining:
                        elapsed_s = time.perf_counter() - t_rr_start
                        eta_s = (elapsed_s / done_live) * (remaining - done_live)
                        cur_rss = _track_rss(f"rerank_{idx + 1}_{len(items)}")
                        log.info(
                            "CPU rerank %d/%d (%s): last=%.2fs eta=%.1fmin rss=%.1fMB",
                            idx + 1,
                            len(items),
                            item.test_id,
                            rr_ms / 1000.0,
                            eta_s / 60.0,
                            cur_rss,
                        )

                rerank_stage_ms.append(sq8_stage_ms[idx] + rr_ms)
                all_scored_8.append(scored_8)

                ranked_5 = scored_8[: cfg.rerank_top_k_max]
                ranked_3 = ranked_5[: cfg.rerank_top_k_default]
                c5 = [str(c["chunk_id"]) for c in ranked_5]
                c3 = [str(c["chunk_id"]) for c in ranked_3]

                run5_chunk_lists.append(c3)
                run5_doc_lists.append(dedupe_to_docs(c3, c2doc_512))
                run6_chunk_lists.append(c5)
                run6_doc_lists.append(dedupe_to_docs(c5, c2doc_512))

                top1_logit = float(scored_8[0]["rerank_raw_logit"]) if scored_8 else float("-inf")
                is_abstained = (
                    cfg.abstention_enabled
                    and cfg.abstain_logit_threshold is not None
                    and top1_logit < float(cfg.abstain_logit_threshold)
                )
                thresh_5 = [] if is_abstained else ranked_5
                thresh_3 = [] if is_abstained else ranked_3
                run6_thresh_records.append(thresh_5)
                run5_thresh_records.append(thresh_3)

                gc5 = [str(c["chunk_id"]) for c in thresh_5]
                gc3 = [str(c["chunk_id"]) for c in thresh_3]
                run5_gated_chunk_lists.append(gc3)
                run5_gated_doc_lists.append(dedupe_to_docs(gc3, c2doc_512))
                run6_gated_chunk_lists.append(gc5)
                run6_gated_doc_lists.append(dedupe_to_docs(gc5, c2doc_512))
        finally:
            if ckpt_fh is not None:
                ckpt_fh.close()

        rerank_truncation_stats = pipeline._reranker.truncation_stats
        _track_rss("after_runs5_6")

        run5_doc_metrics = _compute_ranking_block(
            items, run5_doc_lists, output_k=3, ground_truth_fn=lambda it: it.expected_doc_ids, check_both_versions=True
        )
        run5_chunk_metrics = _compute_ranking_block(
            items, run5_chunk_lists, output_k=3, ground_truth_fn=lambda it: it.expected_source_chunk_ids
        )
        run5_lat = _latency_summary(embed_latencies_ms, rerank_stage_ms, items=items)

        run6_doc_metrics = _compute_ranking_block(
            items, run6_doc_lists, output_k=5, ground_truth_fn=lambda it: it.expected_doc_ids, check_both_versions=True
        )
        run6_chunk_metrics = _compute_ranking_block(
            items, run6_chunk_lists, output_k=5, ground_truth_fn=lambda it: it.expected_source_chunk_ids
        )
        run6_lat = _latency_summary(embed_latencies_ms, rerank_stage_ms, items=items)

        def _abstention_stats(
            thresh_lists: list[list[dict[str, Any]]],
            gated_doc_lists: list[list[str]],
            gated_chunk_lists: list[list[str]],
            output_k: int,
        ) -> dict[str, Any]:
            oos_idxs = [i for i, it in enumerate(items) if it.question_type == QuestionType.OUT_OF_SCOPE]
            in_idxs = [i for i, it in enumerate(items) if it.question_type != QuestionType.OUT_OF_SCOPE]
            oos_abs = sum(1 for i in oos_idxs if len(thresh_lists[i]) == 0)
            in_abs = sum(1 for i in in_idxs if len(thresh_lists[i]) == 0)
            return {
                "threshold_T": cfg.abstain_logit_threshold,
                "abstention_enabled": cfg.abstention_enabled,
                "oos_total": len(oos_idxs),
                "abstained_oos_count": oos_abs,
                "abstention_rate": round(oos_abs / len(oos_idxs), 4) if oos_idxs else 0.0,
                "in_scope_total": len(in_idxs),
                "false_abstain_in_scope_count": in_abs,
                "false_abstain_rate": round(in_abs / len(in_idxs), 4) if in_idxs else 0.0,
                "gated_effective_doc_metrics": _compute_ranking_block(
                    items,
                    gated_doc_lists,
                    output_k=output_k,
                    ground_truth_fn=lambda it: it.expected_doc_ids,
                    check_both_versions=True,
                ),
                "gated_effective_chunk_metrics": _compute_ranking_block(
                    items,
                    gated_chunk_lists,
                    output_k=output_k,
                    ground_truth_fn=lambda it: it.expected_source_chunk_ids,
                ),
            }

        run5_thresh_stats = _abstention_stats(
            run5_thresh_records, run5_gated_doc_lists, run5_gated_chunk_lists, output_k=3
        )
        run6_thresh_stats = _abstention_stats(
            run6_thresh_records, run6_gated_doc_lists, run6_gated_chunk_lists, output_k=5
        )
        reranker_dist = _compute_reranker_distributions(
            items=items,
            all_scored_8=all_scored_8,
            run5_thresh_records=run5_thresh_records,
            run6_thresh_records=run6_thresh_records,
            top_k_default=cfg.rerank_top_k_default,
            top_k_max=cfg.rerank_top_k_max,
        )

        # ── Step 6: Evaluate resident Tier 1 Router across golden_150 ─────────
        log.info("Evaluating resident AgentIntentRouter across 150 golden queries...")
        router_eval, _router_decisions = _evaluate_router(items, cfg, router=pipeline._router)
        router_eval_path = cfg.reports_dir / "router_golden_eval.json"
        _atomic_write_json(router_eval_path, router_eval)
        log.info("Written %s (accuracy=%.4f)", router_eval_path, router_eval["overall_accuracy"])

        # ── Step 7: Live VectorRetrievalPipeline.retrieve() across golden_150 ─
        log.info("Running live VectorRetrievalPipeline.retrieve() over 150 golden queries...")
        contexts_payload: list[dict[str, Any]] = []
        e2e_latencies: list[float] = []
        e2e_by_type_lats: dict[str, list[float]] = {qt.value: [] for qt in _ALL_TYPES}
        oos_blocked = 0
        oos_total = 0
        in_false_blocked = 0
        in_total = 0

        for idx, item in enumerate(items, start=1):
            res = pipeline.retrieve(
                item.query,
                rerank=True,
                top_k=cfg.rerank_top_k_default,
                candidate_k=cfg.candidate_k,
            )
            tot_ms = float(res.latency_ms.get("total_ms", 0.0))
            e2e_latencies.append(tot_ms)
            e2e_by_type_lats[item.question_type.value].append(tot_ms)

            is_blocked = bool(res.refused or res.abstained)
            if item.question_type == QuestionType.OUT_OF_SCOPE:
                oos_total += 1
                if is_blocked:
                    oos_blocked += 1
            else:
                in_total += 1
                if is_blocked:
                    in_false_blocked += 1

            def _fmt_chunk(c: dict[str, Any]) -> dict[str, Any]:
                return {
                    "chunk_id": str(c["chunk_id"]),
                    "doc_id": str(c["doc_id"]),
                    "source_file": str(c.get("source_file", "")),
                    "content": str(c["content"]),
                    "rerank_score": round(float(c["rerank_score"]), 6) if "rerank_score" in c else None,
                    "rerank_raw_logit": round(float(c["rerank_raw_logit"]), 6) if "rerank_raw_logit" in c else None,
                    "standard_version": str(c.get("standard_version", "")),
                }

            contexts_payload.append(
                {
                    "test_id": item.test_id,
                    "question_type": item.question_type.value,
                    "query": item.query,
                    "ground_truth_answer": item.ground_truth_answer,
                    "expected_source_chunk_ids": item.expected_source_chunk_ids,
                    "expected_doc_ids": item.expected_doc_ids,
                    "route": res.route.route,
                    "refused": bool(res.refused),
                    "abstained": bool(res.abstained),
                    "abstain_reason": res.abstain_reason,
                    "top1_logit": round(float(res.top1_logit), 6) if res.top1_logit is not None else None,
                    "latency_ms": res.latency_ms,
                    "retrieved_context_chunks": [_fmt_chunk(c) for c in res.contexts],
                    "contexts_if_not_gated": [_fmt_chunk(c) for c in res.contexts_if_not_gated],
                }
            )
            if idx % 25 == 0 or idx == len(items):
                _track_rss(f"live_pipeline_{idx}_{len(items)}")

        contexts_path = cfg.data_processed / "retrieved_contexts_150.json"
        _atomic_write_json(contexts_path, contexts_payload)
        log.info("Written %s (%d records)", contexts_path, len(contexts_payload))

        e2e_summary = {
            "total_queries": len(items),
            "oos_total": oos_total,
            "oos_blocked_count": oos_blocked,
            "oos_blocked_rate": round(oos_blocked / oos_total, 4) if oos_total else 0.0,
            "inscope_total": in_total,
            "inscope_false_block_count": in_false_blocked,
            "inscope_false_block_rate": round(in_false_blocked / in_total, 4) if in_total else 0.0,
            "mean_total_ms": round(float(np.mean(e2e_latencies)), 3),
            "p50_total_ms": round(float(np.percentile(e2e_latencies, 50)), 3),
            "p95_total_ms": round(float(np.percentile(e2e_latencies, 95)), 3),
            "per_question_type_latency": {
                qt: {
                    "count": len(vals),
                    "mean_total_ms": round(float(np.mean(vals)), 3),
                    "p50_total_ms": round(float(np.percentile(vals, 50)), 3),
                    "p95_total_ms": round(float(np.percentile(vals, 95)), 3),
                }
                for qt, vals in e2e_by_type_lats.items()
                if vals
            },
        }

    # ── Step 8: Write ablation JSON + Markdown report ─────────────────────────
    hw_info = _collect_hardware_info(cfg, precision_lock)
    hw_info["peak_rss_mb"] = round(peak_rss, 1)
    hw_info["rerank_truncation_stats"] = rerank_truncation_stats

    ablation_runs = [
        {
            "run_id": "Run 1",
            "label": "w256 | FlatIndex(1024) full1024",
            "window": 256,
            "index_type": "FlatIndex(1024)",
            "reranker": None,
            "output_k": 3,
            "doc_metrics": run1_doc_metrics,
            "chunk_metrics": None,
            "thresholded_eval": None,
            "latency": run1_lat,
        },
        {
            "run_id": "Run 2",
            "label": "w512 | FlatIndex(1024) full1024",
            "window": 512,
            "index_type": "FlatIndex(1024)",
            "reranker": None,
            "output_k": 3,
            "doc_metrics": run2_doc_metrics,
            "chunk_metrics": run2_chunk_metrics,
            "thresholded_eval": None,
            "latency": run2_lat,
        },
        {
            "run_id": "Run 3",
            "label": "w512 | SQ8Index mrl512",
            "window": 512,
            "index_type": "SQ8Index(512)",
            "reranker": None,
            "output_k": 3,
            "doc_metrics": run3_doc_metrics,
            "chunk_metrics": run3_chunk_metrics,
            "thresholded_eval": None,
            "latency": run3_lat,
        },
        {
            "run_id": "Run 4",
            "label": "w512 | SQ8Index mrl512 (expanded)",
            "window": 512,
            "index_type": "SQ8Index(512)",
            "reranker": None,
            "output_k": cfg.candidate_k,
            "doc_metrics": run4_doc_metrics,
            "chunk_metrics": run4_chunk_metrics,
            "thresholded_eval": None,
            "latency": run4_lat,
        },
        {
            "run_id": "Run 5",
            "label": "w512 | SQ8-MRL512 + Reranker",
            "window": 512,
            "index_type": "SQ8Index(512)",
            "reranker": rerank_model_id,
            "output_k": 3,
            "doc_metrics": run5_doc_metrics,
            "chunk_metrics": run5_chunk_metrics,
            "thresholded_eval": run5_thresh_stats,
            "latency": run5_lat,
        },
        {
            "run_id": "Run 6",
            "label": "w512 | SQ8-MRL512 + Reranker (k=5)",
            "window": 512,
            "index_type": "SQ8Index(512)",
            "reranker": rerank_model_id,
            "output_k": 5,
            "doc_metrics": run6_doc_metrics,
            "chunk_metrics": run6_chunk_metrics,
            "thresholded_eval": run6_thresh_stats,
            "latency": run6_lat,
        },
    ]

    results: dict[str, Any] = {
        "generated_at": datetime.now(tz=timezone.utc).isoformat(),
        "golden_count": len(items),
        "in_scope_count": sum(1 for it in items if it.question_type != QuestionType.OUT_OF_SCOPE),
        "oos_count": sum(1 for it in items if it.question_type == QuestionType.OUT_OF_SCOPE),
        "corpus": {
            "w512_chunks": w512_count,
            "w256_chunks": w256_count,
        },
        "models": {
            "embed_model_id": embed_model_id,
            "rerank_model_id": rerank_model_id,
            "reranker_is_fallback": reranker_is_fallback,
        },
        "hardware": hw_info,
        "runs": ablation_runs,
        "reranker_score_distribution": reranker_dist,
        "router_golden_eval": router_eval,
        "end_to_end_pipeline_eval": e2e_summary,
        "gpu_hybrid_comparison": _build_gpu_hybrid_comparison(ablation_runs, archived_ref),
    }

    results_json_path = cfg.reports_dir / "retrieval_ablation_results.json"
    _atomic_write_json(results_json_path, results)
    log.info("Written %s", results_json_path)

    report_md_path = cfg.reports_dir / "retrieval_ablation_report.md"
    _atomic_write_text(report_md_path, _format_markdown_report(results))
    log.info("Written %s", report_md_path)

    return results


def main() -> None:
    """CLI entry point for Phase 1.5 CPU-only retrieval ablation."""
    parser = argparse.ArgumentParser(
        description="Run KAVACH-RAG Phase 1.5 CPU-only 6-run retrieval ablation & pipeline eval."
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Remove CPU query/rerank checkpoints for a clean ablation run.",
    )
    args = parser.parse_args()
    run_ablation(fresh=bool(args.fresh))


if __name__ == "__main__":
    main()
