"""Phase 1.5 calibrated logit abstention threshold selection and reporting.

CLI modes:
- --collect : Collect per-query router decisions and reranker raw top-1 logits
              across inscope_dev_100, oos_dev_60, and golden_150 under BenchLock.
- --select  : Select calibrated logit threshold T strictly on devsplits
              (explicitly excluding golden_150) and write models/abstain_calibration.json.
- --report  : Generate reports/phase15_calibration_report.{md,json} with full
              grid sweep (-2.0..12.0 step 0.25), ROC AUC, error analysis, and
              held-out golden_150 verification at fixed T.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from kavach.config import Settings, get_settings
from kavach.exceptions import ConfigError, HardwareIsolationError, RetrievalError
from kavach.retrieval.benchlock import BenchLock
from kavach.retrieval.devsets import validate_devsets
from kavach.retrieval.embedder import Embedder
from kavach.retrieval.reranker import Reranker
from kavach.retrieval.router import AgentIntentRouter, ROUTE_REFUSAL
from kavach.retrieval.vector_store import ChunkStore, SQ8Index
from kavach.runtime_guard import rss_mb, vram_probe

log = logging.getLogger("kavach.calibration")

GRID_START = -2.0
GRID_END = 12.0
GRID_STEP = 0.25


def _setup_calibration_logging(logs_dir: Path) -> None:
    """Configure console + file logging in logs/calibration.log."""
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = logs_dir / "calibration.log"
    root = logging.getLogger()
    if not any(isinstance(h, logging.StreamHandler) for h in root.handlers):
        stream_h = logging.StreamHandler(sys.stdout)
        stream_h.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
        root.addHandler(stream_h)
    if not any(
        isinstance(h, logging.FileHandler) and Path(getattr(h, "baseFilename", "")) == log_path
        for h in root.handlers
    ):
        file_h = logging.FileHandler(log_path, encoding="utf-8")
        file_h.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
        root.addHandler(file_h)
    root.setLevel(logging.INFO)


def _atomic_write_text(path: Path, text: str) -> None:
    """Write text atomically using a sibling temporary file + os.replace."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    try:
        with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(text)
        os.replace(tmp, path)
    except BaseException:
        if tmp.exists():
            tmp.unlink(missing_ok=True)
        raise


def _atomic_write_json(path: Path, payload: Any) -> None:
    """Serialize JSON cleanly and write atomically."""
    _atomic_write_text(path, json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


def _atomic_write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    """Serialize JSONL records and write atomically."""
    text = "\n".join(json.dumps(r, ensure_ascii=False) for r in rows)
    if text:
        text += "\n"
    _atomic_write_text(path, text)


def _assert_cpu_and_rss(cfg: Settings, stage: str) -> float:
    """Assert CPU-only execution (no GPU attachment) and bounded process RSS."""
    probe = vram_probe()
    if probe["cuda_available"] or probe["pid_in_nvidia_smi"]:
        raise HardwareIsolationError(
            f"GPU leakage detected during calibration stage '{stage}'",
            details={
                "stage": stage,
                "cuda_available": probe["cuda_available"],
                "pid_in_nvidia_smi": probe["pid_in_nvidia_smi"],
            },
        )
    current_rss = rss_mb()
    if current_rss > cfg.rss_ceiling_mb:
        raise HardwareIsolationError(
            f"RSS {current_rss:.1f} MB exceeded ceiling {cfg.rss_ceiling_mb:.1f} MB at '{stage}'",
            details={"stage": stage, "rss_mb": current_rss, "rss_ceiling_mb": cfg.rss_ceiling_mb},
        )
    return current_rss


def verify_precision_lock(cfg: Settings) -> dict[str, Any]:
    """Ensure models/precision_lock.json exists and matches runtime Settings."""
    lock_path = cfg.models_dir / "precision_lock.json"
    if not lock_path.exists():
        raise ConfigError(
            f"Missing precision lock file required for calibration: {lock_path}",
            details={"path": str(lock_path)},
        )
    try:
        lock_data = json.loads(lock_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ConfigError(
            f"Corrupt precision lock JSON at {lock_path}: {exc}",
            details={"path": str(lock_path)},
        ) from exc

    if not isinstance(lock_data, dict):
        raise ConfigError(
            f"Precision lock must be a JSON object: {lock_path}",
            details={"path": str(lock_path)},
        )

    expected_pairs = {
        "embed_precision": cfg.embed_precision,
        "rerank_precision": cfg.rerank_precision,
        "router_precision": cfg.router_precision,
        "rerank_max_length": cfg.rerank_max_length,
        "contingency_l_applied": bool(cfg.contingency_l_applied),
        "rerank_long_query_max_length": cfg.rerank_long_query_max_length,
        "rerank_long_query_token_threshold": cfg.rerank_long_query_token_threshold,
    }
    mismatches: dict[str, Any] = {}
    for key, cfg_val in expected_pairs.items():
        if key not in lock_data:
            if key in (
                "contingency_l_applied",
                "rerank_long_query_max_length",
                "rerank_long_query_token_threshold",
            ):
                lock_val = False if key == "contingency_l_applied" else (None if "max" in key else 96)
            else:
                mismatches[key] = {"lock": "<missing>", "settings": cfg_val}
                continue
        else:
            lock_val = lock_data[key]
        if key == "contingency_l_applied":
            lock_val = bool(lock_val)
        if lock_val != cfg_val:
            mismatches[key] = {"lock": lock_val, "settings": cfg_val}

    if mismatches:
        raise ConfigError(
            "Settings diverge from models/precision_lock.json; recalibration blocked",
            details={"mismatches": mismatches, "lock_path": str(lock_path)},
        )
    return lock_data


def verify_abstain_calibration(cfg: Settings) -> dict[str, Any] | None:
    """Validate models/abstain_calibration.json matches Settings when abstention is enabled."""
    if not cfg.abstention_enabled:
        return None

    cal_path = cfg.models_dir / "abstain_calibration.json"
    if not cal_path.exists():
        raise ConfigError(
            f"Abstention is enabled but calibration file is missing: {cal_path}",
            details={"path": str(cal_path)},
        )
    try:
        cal_data = json.loads(cal_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ConfigError(
            f"Failed to read {cal_path}: {exc}",
            details={"path": str(cal_path)},
        ) from exc

    if not isinstance(cal_data, dict):
        raise ConfigError(
            f"Invalid abstain_calibration JSON structure at {cal_path}",
            details={"path": str(cal_path)},
        )

    raw_threshold = cal_data.get("threshold_T")
    if raw_threshold is None or not isinstance(raw_threshold, (int, float)):
        raise ConfigError(
            f"Missing or non-numeric threshold_T in {cal_path}",
            details={"threshold_T": raw_threshold},
        )

    mismatches: dict[str, Any] = {}
    # Note key spelling difference: abstain_calibration.json uses 'reranker_precision'
    if cal_data.get("reranker_precision") != cfg.rerank_precision:
        mismatches["reranker_precision"] = {
            "calibration": cal_data.get("reranker_precision"),
            "settings_rerank_precision": cfg.rerank_precision,
        }
    if cal_data.get("rerank_max_length") != cfg.rerank_max_length:
        mismatches["rerank_max_length"] = {
            "calibration": cal_data.get("rerank_max_length"),
            "settings": cfg.rerank_max_length,
        }
    if bool(cal_data.get("contingency_l_applied", False)) != bool(cfg.contingency_l_applied):
        mismatches["contingency_l_applied"] = {
            "calibration": cal_data.get("contingency_l_applied"),
            "settings": cfg.contingency_l_applied,
        }
    if cal_data.get("rerank_long_query_max_length") != cfg.rerank_long_query_max_length:
        mismatches["rerank_long_query_max_length"] = {
            "calibration": cal_data.get("rerank_long_query_max_length"),
            "settings": cfg.rerank_long_query_max_length,
        }
    if (
        cal_data.get("rerank_long_query_token_threshold", 96)
        != cfg.rerank_long_query_token_threshold
    ):
        mismatches["rerank_long_query_token_threshold"] = {
            "calibration": cal_data.get("rerank_long_query_token_threshold"),
            "settings": cfg.rerank_long_query_token_threshold,
        }

    if mismatches:
        mismatch_keys = ", ".join(sorted(mismatches.keys()))
        raise ConfigError(
            f"Stale abstain_calibration.json detected ({mismatch_keys}) against active Settings",
            details={"mismatches": mismatches, "calibration_path": str(cal_path)},
        )

    return cal_data


def _load_collection_queries(cfg: Settings) -> list[dict[str, Any]]:
    """Load inscope_dev_100, oos_dev_60, and golden_150 into uniform collection items."""
    oos_path = cfg.data_golden / "oos_dev_60.jsonl"
    inscope_path = cfg.data_golden / "inscope_dev_100.jsonl"
    golden_path = cfg.data_golden / "golden_150.jsonl"

    validate_devsets(
        oos_path=oos_path,
        inscope_path=inscope_path,
        corpus_parquet_path=cfg.data_processed / "corpus_chunks.parquet",
        golden_path=golden_path,
        training_queries_path=cfg.models_dir / "router_deberta" / "training_queries.jsonl",
    )

    items: list[dict[str, Any]] = []
    for path in (inscope_path, oos_path):
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                row = json.loads(line)
                items.append(
                    {
                        "query_id": row["dev_id"],
                        "split": row["split"],
                        "category": row["category"],
                        "query": row["query"],
                        "expected_doc_ids": list(row["expected_doc_ids"]),
                    }
                )

    with open(golden_path, "r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            items.append(
                {
                    "query_id": row["test_id"],
                    "split": "golden_150",
                    "category": row["question_type"],
                    "query": row["query"],
                    "expected_doc_ids": list(row.get("expected_doc_ids", [])),
                }
            )
    return items


def _load_ckpt_rows(ckpt_path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    """Load already-collected records keyed by (split, query_id) for crash-safe resume."""
    completed: dict[tuple[str, str], dict[str, Any]] = {}
    if not ckpt_path.exists():
        return completed
    with open(ckpt_path, "r", encoding="utf-8") as fh:
        for line in fh:
            stripped = line.strip()
            if not stripped:
                continue
            row = json.loads(stripped)
            completed[(str(row["split"]), str(row["query_id"]))] = row
    return completed


def _collect_logits(cfg: Settings, fresh: bool = False) -> dict[str, Any]:
    """Run router + dense retrieval + raw reranker scoring across devsets & golden_150."""
    _setup_calibration_logging(cfg.logs_dir)
    verify_precision_lock(cfg)
    _assert_cpu_and_rss(cfg, "collect_start")

    out_dir = cfg.data_processed / "calibration"
    out_dir.mkdir(parents=True, exist_ok=True)
    logits_path = out_dir / "logits.jsonl"
    router_path = out_dir / "router_oos_decisions.jsonl"
    ckpt_path = out_dir / "logits.ckpt.jsonl"

    if fresh:
        for target in (ckpt_path, logits_path, router_path):
            if target.exists():
                target.unlink()

    items = _load_collection_queries(cfg)
    total = len(items)
    completed = _load_ckpt_rows(ckpt_path)

    log.info(
        "Calibration --collect start: total=%d resumed=%d embed_prec=%s rerank_prec=%s "
        "router_prec=%s rerank_max_len=%d contingency_l=%s rss_ceiling=%.1fMB",
        total,
        len(completed),
        cfg.embed_precision,
        cfg.rerank_precision,
        cfg.router_precision,
        cfg.rerank_max_length,
        cfg.contingency_l_applied,
        cfg.rss_ceiling_mb,
    )

    with BenchLock(owner="calibration"):
        router = AgentIntentRouter(cfg)
        router.load()
        embedder = Embedder(cfg)
        store = ChunkStore(cfg.data_processed / "corpus_chunks.parquet")
        index = SQ8Index.load(cfg.models_dir / "index_sq8_mrl512.faiss")
        reranker = Reranker(cfg)

        # Warmup resident models to prime oneDNN caches and keep truncation stats clean
        router.warmup()
        embedder.warmup()
        reranker.warmup()
        reranker.reset_truncation_stats()
        if reranker.is_fallback:
            raise RetrievalError(
                "Reranker fell back to dummy scoring during calibration warmup",
                details={"model_id": reranker.model_id},
            )

        rss_warm = _assert_cpu_and_rss(cfg, "after_warmup")
        log.info("Calibration models resident & warmed cleanly (rss=%.1fMB)", rss_warm)

        records_by_key: dict[tuple[str, str], dict[str, Any]] = dict(completed)
        pending_items = [
            it for it in items if (str(it["split"]), str(it["query_id"])) not in records_by_key
        ]

        t_start = time.perf_counter()
        if pending_items:
            with open(ckpt_path, "a", encoding="utf-8", newline="\n") as ckpt_fh:
                for step_idx, item in enumerate(pending_items, start=1):
                    decision = router.route(item["query"])

                    t_e0 = time.perf_counter()
                    ebatch = embedder.embed([item["query"]], kind="query", batch_size=1)
                    raw_cands = index.search(ebatch.mrl512[0], cfg.candidate_k)
                    cands: list[dict[str, Any]] = []
                    for rank_idx, (cid, sim) in enumerate(raw_cands, start=1):
                        rec = store.get(cid)
                        if rec is not None:
                            hydrated = dict(rec)
                            hydrated["dense_score"] = float(sim)
                            hydrated["dense_rank"] = rank_idx
                            cands.append(hydrated)
                    embed_ms = (time.perf_counter() - t_e0) * 1000.0

                    t_r0 = time.perf_counter()
                    scored = reranker.rerank(
                        item["query"],
                        cands,
                        top_k=len(cands),
                        return_all=True,
                    )
                    rerank_ms = (time.perf_counter() - t_r0) * 1000.0

                    if reranker.is_fallback:
                        raise RetrievalError(
                            f"Reranker entered fallback mode on {item['query_id']}"
                        )
                    if not scored:
                        raise RetrievalError(
                            f"Reranker returned empty candidate list for {item['query_id']}"
                        )

                    cur_rss = rss_mb()
                    record: dict[str, Any] = {
                        "query_id": item["query_id"],
                        "split": item["split"],
                        "category": item["category"],
                        "query": item["query"],
                        "expected_doc_ids": item["expected_doc_ids"],
                        "router_route": decision.route,
                        "router_confidence": round(float(decision.confidence), 6),
                        "router_matched_identifier": decision.matched_identifier,
                        "router_is_malformed": bool(decision.is_malformed),
                        "router_latency_ms": round(float(decision.latency_ms), 3),
                        "candidate_count": len(cands),
                        "top1_logit": round(float(scored[0]["rerank_raw_logit"]), 6),
                        "top1_prob": round(float(scored[0]["rerank_score"]), 6),
                        "top1_chunk_id": str(scored[0]["chunk_id"]),
                        "top1_doc_id": str(scored[0].get("doc_id") or ""),
                        "top_k_logits": [round(float(c["rerank_raw_logit"]), 6) for c in scored],
                        "top_k_doc_ids": [str(c.get("doc_id") or "") for c in scored],
                        "embed_latency_ms": round(embed_ms, 2),
                        "rerank_latency_ms": round(rerank_ms, 2),
                        "rss_mb": round(cur_rss, 2),
                    }
                    ckpt_fh.write(json.dumps(record, ensure_ascii=False) + "\n")
                    ckpt_fh.flush()
                    records_by_key[(item["split"], item["query_id"])] = record

                    if step_idx % 10 == 0 or step_idx == len(pending_items):
                        _assert_cpu_and_rss(cfg, f"collect_step_{step_idx}")
                        elapsed = time.perf_counter() - t_start
                        rate = elapsed / step_idx
                        eta_s = rate * (len(pending_items) - step_idx)
                        done_total = len(completed) + step_idx
                        log.info(
                            "[collect] %d/%d (%.1f%%) elapsed=%.1fs eta=%.1fs rss=%.1fMB last=%s top1_logit=%.3f",
                            done_total,
                            total,
                            100.0 * done_total / total,
                            elapsed,
                            eta_s,
                            cur_rss,
                            item["query_id"],
                            record["top1_logit"],
                        )

        truncation_stats = reranker.truncation_stats

    ordered_records = [records_by_key[(it["split"], it["query_id"])] for it in items]
    router_oos_rows = [
        {
            "query_id": r["query_id"],
            "split": r["split"],
            "category": r["category"],
            "query": r["query"],
            "router_route": r["router_route"],
            "router_confidence": r["router_confidence"],
            "refused_by_router": r["router_route"] == ROUTE_REFUSAL,
            "top1_logit": r["top1_logit"],
        }
        for r in ordered_records
        if r["split"] == "oos_dev"
    ]

    _atomic_write_jsonl(logits_path, ordered_records)
    _atomic_write_jsonl(router_path, router_oos_rows)

    return {
        "action": "collect",
        "total_records": len(ordered_records),
        "logits_path": str(logits_path),
        "router_oos_path": str(router_path),
        "truncation_stats": truncation_stats,
    }


def _load_logits_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read logits.jsonl records strictly."""
    if not path.exists():
        raise FileNotFoundError(f"Calibration logits file missing: {path}")
    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            stripped = line.strip()
            if stripped:
                rows.append(json.loads(stripped))
    return rows


def _quantile(sorted_vals: list[float], q: float) -> float:
    """Compute linear-interpolated quantile on a pre-sorted non-empty float list."""
    if len(sorted_vals) == 1:
        return float(sorted_vals[0])
    pos = q * (len(sorted_vals) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    frac = pos - lo
    return float(sorted_vals[lo] * (1.0 - frac) + sorted_vals[hi] * frac)


def _logit_summary(vals: list[float]) -> dict[str, float]:
    """Compute distribution statistics without external dependencies."""
    if not vals:
        return {"min": 0.0, "p05": 0.0, "p25": 0.0, "median": 0.0, "p75": 0.0, "p95": 0.0, "max": 0.0, "mean": 0.0}
    s = sorted(float(v) for v in vals)
    return {
        "min": round(s[0], 6),
        "p05": round(_quantile(s, 0.05), 6),
        "p25": round(_quantile(s, 0.25), 6),
        "median": round(_quantile(s, 0.50), 6),
        "p75": round(_quantile(s, 0.75), 6),
        "p95": round(_quantile(s, 0.95), 6),
        "max": round(s[-1], 6),
        "mean": round(sum(s) / len(s), 6),
    }


def compute_roc_auc(inscope_scores: list[float], oos_scores: list[float]) -> float:
    """Compute exact Mann-Whitney U ROC AUC treating in-scope as positive class."""
    if not inscope_scores or not oos_scores:
        return 0.0
    wins = 0.0
    for pos in inscope_scores:
        for neg in oos_scores:
            if pos > neg:
                wins += 1.0
            elif pos == neg:
                wins += 0.5
    return wins / (len(inscope_scores) * len(oos_scores))


def build_threshold_grid(
    inscope_rows: list[dict[str, Any]],
    oos_rows: list[dict[str, Any]],
) -> list[dict[str, float]]:
    """Evaluate threshold grid [-2.0, 12.0] step 0.25 across dev splits."""
    n_steps = int(round((GRID_END - GRID_START) / GRID_STEP)) + 1
    thresholds = [round(GRID_START + idx * GRID_STEP, 4) for idx in range(n_steps)]

    n_in = len(inscope_rows)
    n_oos = len(oos_rows)
    if n_in == 0 or n_oos == 0:
        raise ValueError(f"Non-empty dev splits required (got inscope={n_in}, oos={n_oos})")

    grid_rows: list[dict[str, float]] = []
    for t in thresholds:
        in_false_abstain = sum(1 for r in inscope_rows if float(r["top1_logit"]) < t)
        oos_false_accept = sum(1 for r in oos_rows if float(r["top1_logit"]) >= t)
        sys_oos_false_accept = sum(
            1
            for r in oos_rows
            if r.get("router_route") != ROUTE_REFUSAL and float(r["top1_logit"]) >= t
        )
        fa_rate = in_false_abstain / n_in
        oos_rate = oos_false_accept / n_oos
        sys_oos_rate = sys_oos_false_accept / n_oos
        tpr = 1.0 - fa_rate
        tnr = 1.0 - oos_rate
        bal_acc = 0.5 * (tpr + tnr)
        grid_rows.append(
            {
                "T": t,
                "inscope_false_abstain_rate": round(fa_rate, 6),
                "inscope_tpr": round(tpr, 6),
                "oos_false_accept_rate": round(oos_rate, 6),
                "oos_tnr": round(tnr, 6),
                "balanced_accuracy": round(bal_acc, 6),
                "system_oos_false_accept_rate": round(sys_oos_rate, 6),
            }
        )
    return grid_rows


def select_threshold(
    all_records: list[dict[str, Any]],
    cfg: Settings | None = None,
    logits_sha256: str | None = None,
) -> dict[str, Any]:
    """Select abstention threshold via T = floor((min_inscope - 0.5) * 4.0) / 4.0.

    Explicitly excludes any ``split == "golden_150"`` rows prior to threshold
    computation so held-out test metrics never leak into calibration.
    """
    golden_excluded = [r for r in all_records if r.get("split") == "golden_150"]
    cal_rows = [r for r in all_records if r.get("split") != "golden_150"]
    assert all(r.get("split") != "golden_150" for r in cal_rows)

    inscope_rows = [r for r in cal_rows if r.get("split") == "inscope_dev"]
    oos_rows = [r for r in cal_rows if r.get("split") == "oos_dev"]
    if not inscope_rows or not oos_rows:
        raise ValueError(
            f"Missing dev split rows in calibration data: inscope={len(inscope_rows)}, oos={len(oos_rows)}"
        )

    inscope_logits = [float(r["top1_logit"]) for r in inscope_rows]
    oos_logits = [float(r["top1_logit"]) for r in oos_rows]

    # Deterministic Section 6.2 margin rule anchored below minimum in-scope dev logit
    m = min(inscope_logits)
    t_val = float(math.floor((m - 0.5) * 4.0) / 4.0)

    dev_in_scope_n = len(inscope_rows)
    dev_oos_n = len(oos_rows)
    dev_false_abstain_rate = round(
        sum(1 for logit in inscope_logits if logit < t_val) / dev_in_scope_n,
        6,
    )
    dev_oos_abstain_rate = round(
        sum(1 for logit in oos_logits if logit < t_val) / dev_oos_n,
        6,
    )
    system_oos_abstain_rate = round(
        sum(
            1
            for r in oos_rows
            if r.get("router_route") == ROUTE_REFUSAL or float(r["top1_logit"]) < t_val
        )
        / dev_oos_n,
        6,
    )

    auc = compute_roc_auc(inscope_logits, oos_logits)
    settings_obj = cfg or get_settings()

    if logits_sha256 is None:
        canonical = "\n".join(
            json.dumps(r, sort_keys=True, ensure_ascii=False)
            for r in sorted(cal_rows, key=lambda x: (str(x["split"]), str(x["query_id"])))
        )
        logits_sha256 = hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    return {
        "threshold_T": t_val,
        "reranker_precision": settings_obj.rerank_precision,
        "rerank_max_length": settings_obj.rerank_max_length,
        "reranker_model_id": settings_obj.rerank_model_id,
        "selection_rule": "floor_to_0.25(min_in_scope_dev_logit - 0.5)",
        "dev_in_scope_n": dev_in_scope_n,
        "dev_oos_n": dev_oos_n,
        "dev_false_abstain_rate": dev_false_abstain_rate,
        "dev_oos_abstain_rate": dev_oos_abstain_rate,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "min_in_scope_dev_logit": round(m, 6),
        "embed_precision": settings_obj.embed_precision,
        "router_precision": settings_obj.router_precision,
        "contingency_l_applied": bool(settings_obj.contingency_l_applied),
        "rerank_long_query_max_length": settings_obj.rerank_long_query_max_length,
        "rerank_long_query_token_threshold": settings_obj.rerank_long_query_token_threshold,
        "candidate_k": settings_obj.candidate_k,
        "golden_excluded_count": len(golden_excluded),
        "roc_auc": round(auc, 6),
        "system_oos_abstain_rate": system_oos_abstain_rate,
        "logits_sha256": logits_sha256,
        "inscope_logit_stats": _logit_summary(inscope_logits),
        "oos_logit_stats": _logit_summary(oos_logits),
    }


def run_select(cfg: Settings) -> dict[str, Any]:
    """Load logits.jsonl, run golden-excluded threshold selection, and write abstain_calibration.json."""
    logits_path = cfg.data_processed / "calibration" / "logits.jsonl"
    records = _load_logits_jsonl(logits_path)
    cal_payload = select_threshold(records, cfg=cfg)
    out_path = cfg.models_dir / "abstain_calibration.json"
    _atomic_write_json(out_path, cal_payload)
    return {"action": "select", "output_path": str(out_path), **cal_payload}


def _recall_hit(expected_doc_ids: list[str], top_k_doc_ids: list[str], k: int) -> float:
    """Compute per-query Recall@K across unique expected doc_ids."""
    if not expected_doc_ids:
        return 0.0
    exp_set = set(expected_doc_ids)
    retrieved_set = set(top_k_doc_ids[:k])
    return len(exp_set & retrieved_set) / len(exp_set)


def generate_report(cfg: Settings) -> dict[str, Any]:
    """Generate comprehensive Phase 1.5 calibration markdown and JSON reports."""
    logits_path = cfg.data_processed / "calibration" / "logits.jsonl"
    cal_path = cfg.models_dir / "abstain_calibration.json"
    if not cal_path.exists():
        raise ConfigError(f"Cannot generate calibration report before --select: missing {cal_path}")

    records = _load_logits_jsonl(logits_path)
    cal_data = json.loads(cal_path.read_text(encoding="utf-8"))
    t_val = float(cal_data["threshold_T"])

    inscope_rows = [r for r in records if r["split"] == "inscope_dev"]
    oos_rows = [r for r in records if r["split"] == "oos_dev"]
    golden_rows = [r for r in records if r["split"] == "golden_150"]

    grid = build_threshold_grid(inscope_rows, oos_rows)
    roc_auc = compute_roc_auc(
        [float(r["top1_logit"]) for r in inscope_rows],
        [float(r["top1_logit"]) for r in oos_rows],
    )

    # Per-category OOS breakdown
    oos_by_cat: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in oos_rows:
        oos_by_cat[str(row["category"])].append(row)
    oos_category_stats: dict[str, dict[str, Any]] = {}
    for cat, rows in sorted(oos_by_cat.items()):
        logits = [float(r["top1_logit"]) for r in rows]
        n = len(rows)
        router_ref = sum(1 for r in rows if r["router_route"] == ROUTE_REFUSAL)
        rerank_abs = sum(1 for r in rows if float(r["top1_logit"]) < t_val)
        sys_gated = sum(
            1
            for r in rows
            if r["router_route"] == ROUTE_REFUSAL or float(r["top1_logit"]) < t_val
        )
        oos_category_stats[cat] = {
            "count": n,
            "router_hard_refusal_rate": round(router_ref / n, 4),
            "reranker_abstain_rate_at_T": round(rerank_abs / n, 4),
            "system_refusal_or_abstain_rate": round(sys_gated / n, 4),
            "false_accept_count_system": n - sys_gated,
            "logit_stats": _logit_summary(logits),
        }

    # Per-category inscope_dev breakdown
    in_by_cat: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in inscope_rows:
        in_by_cat[str(row["category"])].append(row)
    inscope_category_stats: dict[str, dict[str, Any]] = {}
    for cat, rows in sorted(in_by_cat.items()):
        logits = [float(r["top1_logit"]) for r in rows]
        n = len(rows)
        false_abs = sum(1 for r in rows if float(r["top1_logit"]) < t_val)
        r3 = sum(_recall_hit(r["expected_doc_ids"], r["top_k_doc_ids"], 3) for r in rows) / n
        r8 = sum(_recall_hit(r["expected_doc_ids"], r["top_k_doc_ids"], 8) for r in rows) / n
        inscope_category_stats[cat] = {
            "count": n,
            "false_abstain_rate_at_T": round(false_abs / n, 4),
            "doc_recall_at_3": round(r3, 4),
            "doc_recall_at_8": round(r8, 4),
            "logit_stats": _logit_summary(logits),
        }

    lowest_inscope = sorted(inscope_rows, key=lambda r: float(r["top1_logit"]))[:10]
    highest_oos = sorted(oos_rows, key=lambda r: float(r["top1_logit"]), reverse=True)[:10]

    # Held-out golden_150 diagnostics at fixed T
    golden_by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in golden_rows:
        golden_by_type[str(row["category"])].append(row)
    golden_type_stats: dict[str, dict[str, Any]] = {}
    for qtype, rows in sorted(golden_by_type.items()):
        n = len(rows)
        router_ref = sum(1 for r in rows if r["router_route"] == ROUTE_REFUSAL)
        rerank_abs_only = sum(
            1
            for r in rows
            if r["router_route"] != ROUTE_REFUSAL and float(r["top1_logit"]) < t_val
        )
        sys_blocked = router_ref + rerank_abs_only
        golden_type_stats[qtype] = {
            "count": n,
            "router_hard_refusal_rate": round(router_ref / n, 4),
            "reranker_abstain_rate_non_refused": round(rerank_abs_only / n, 4),
            "combined_system_block_rate": round(sys_blocked / n, 4),
            "logit_stats": _logit_summary([float(r["top1_logit"]) for r in rows]),
        }

    golden_oos = [r for r in golden_rows if str(r["category"]).upper() == "OUT_OF_SCOPE"]
    golden_inscope = [r for r in golden_rows if str(r["category"]).upper() != "OUT_OF_SCOPE"]
    assert len(golden_oos) == 10 and len(golden_inscope) == 140, (
        f"Expected 10 golden OOS and 140 golden in-scope items, "
        f"got oos={len(golden_oos)}, inscope={len(golden_inscope)}"
    )

    golden_oos_items = [
        {
            "query_id": r["query_id"],
            "query": r["query"],
            "router_route": r["router_route"],
            "top1_logit": float(r["top1_logit"]),
            "gate_abstained": bool(float(r["top1_logit"]) < t_val),
            "system_blocked": bool(r["router_route"] == ROUTE_REFUSAL or float(r["top1_logit"]) < t_val),
        }
        for r in sorted(golden_oos, key=lambda x: str(x["query_id"]))
    ]

    dev_oos_abstain_rate = float(
        cal_data.get(
            "dev_oos_abstain_rate",
            sum(1 for r in oos_rows if float(r["top1_logit"]) < t_val) / max(len(oos_rows), 1),
        )
    )
    dev_false_abstain_rate = float(
        cal_data.get(
            "dev_false_abstain_rate",
            sum(1 for r in inscope_rows if float(r["top1_logit"]) < t_val) / max(len(inscope_rows), 1),
        )
    )
    low_oos_abstain_flag = dev_oos_abstain_rate < 0.50
    low_inscope_outliers = [
        {
            "query_id": r["query_id"],
            "category": r["category"],
            "top1_logit": float(r["top1_logit"]),
            "query_snippet": r["query"][:120],
        }
        for r in lowest_inscope[:5]
    ]
    low_oos_diagnostic: dict[str, Any] = {
        "flagged": low_oos_abstain_flag,
        "dev_oos_abstain_rate": round(dev_oos_abstain_rate, 6),
        "threshold_T": t_val,
        "driving_low_inscope_outliers": low_inscope_outliers,
        "options_without_changing_T": (
            [
                "Add a Phase 2 prompt-level LLM refusal backstop requiring explicit CVE/MITRE grounding verification before answering (without changing T).",
                "Add Phase 2 identifier-existence verification for unindexed_public_security CVE/T-IDs and router hard negatives for lexical_decoy/org_proprietary.",
                "Audit whether low in-scope outliers suffered rerank_max_length query truncation (Contingency L candidate).",
            ]
            if low_oos_abstain_flag
            else []
        ),
    }

    golden_sys_refusal = (
        sum(
            1
            for r in golden_rows
            if r["router_route"] == ROUTE_REFUSAL or float(r["top1_logit"]) < t_val
        )
        / max(len(golden_rows), 1)
    )
    golden_oos_block = (
        sum(
            1
            for r in golden_oos
            if r["router_route"] == ROUTE_REFUSAL or float(r["top1_logit"]) < t_val
        )
        / max(len(golden_oos), 1)
    )
    golden_in_false_abstain = (
        sum(
            1
            for r in golden_inscope
            if r["router_route"] != ROUTE_REFUSAL and float(r["top1_logit"]) < t_val
        )
        / max(len(golden_inscope), 1)
    )

    report_payload: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "threshold_T": t_val,
        "selection_rule": cal_data.get("selection_rule"),
        "reranker_model_id": cal_data.get("reranker_model_id"),
        "dev_in_scope_n": len(inscope_rows),
        "dev_oos_n": len(oos_rows),
        "dev_false_abstain_rate": round(dev_false_abstain_rate, 6),
        "dev_oos_abstain_rate": round(dev_oos_abstain_rate, 6),
        "system_oos_abstain_rate": cal_data.get("system_oos_abstain_rate"),
        "roc_auc_inscope_vs_oos": round(roc_auc, 6),
        "low_oos_abstain_diagnostic": low_oos_diagnostic,
        "precision_config": {
            "reranker_precision": cal_data.get("reranker_precision"),
            "embed_precision": cal_data.get("embed_precision"),
            "router_precision": cal_data.get("router_precision"),
            "rerank_max_length": cal_data.get("rerank_max_length"),
            "contingency_l_applied": cal_data.get("contingency_l_applied"),
            "rerank_long_query_max_length": cal_data.get("rerank_long_query_max_length"),
            "rerank_long_query_token_threshold": cal_data.get("rerank_long_query_token_threshold"),
        },
        "threshold_grid": grid,
        "inscope_dev_summary": _logit_summary([float(r["top1_logit"]) for r in inscope_rows]),
        "oos_dev_summary": _logit_summary([float(r["top1_logit"]) for r in oos_rows]),
        "inscope_dev_by_category": inscope_category_stats,
        "oos_dev_by_category": oos_category_stats,
        "lowest_10_inscope_dev": [
            {
                "query_id": r["query_id"],
                "category": r["category"],
                "top1_logit": r["top1_logit"],
                "router_route": r["router_route"],
                "expected_doc_ids": r["expected_doc_ids"],
                "top1_doc_id": r["top1_doc_id"],
                "query_snippet": r["query"][:120],
            }
            for r in lowest_inscope
        ],
        "highest_10_oos_dev": [
            {
                "query_id": r["query_id"],
                "category": r["category"],
                "top1_logit": r["top1_logit"],
                "router_route": r["router_route"],
                "top1_doc_id": r["top1_doc_id"],
                "query_snippet": r["query"][:120],
            }
            for r in highest_oos
        ],
        "golden_150_heldout_eval": {
            "total": len(golden_rows),
            "system_overall_block_rate": round(golden_sys_refusal, 4),
            "oos_block_rate": round(golden_oos_block, 4),
            "inscope_false_abstain_rate": round(golden_in_false_abstain, 4),
            "by_question_type": golden_type_stats,
            "oos_items": golden_oos_items,
        },
    }

    md_lines: list[str] = [
        "# KAVACH-RAG Phase 1.5 Calibration Report",
        "",
        f"- **Generated at**: `{report_payload['generated_at']}`",
        f"- **Selected logit threshold `T`**: `{t_val}` (`{cal_data.get('selection_rule')}`)",
        f"- **Dev ROC AUC (`inscope_dev_100` vs `oos_dev_60`)**: `{roc_auc:.4f}`",
        f"- **Reranker model / precision / max_len**: `{cal_data.get('reranker_model_id')}` / `{cal_data.get('reranker_precision')}` / `{cal_data.get('rerank_max_length')}` (Contingency L: `{cal_data.get('contingency_l_applied')}`)",
        f"- **Dev OOS abstain rate @ T**: `{dev_oos_abstain_rate:.4f}` (system-level: `{cal_data.get('system_oos_abstain_rate', 0.0):.4f}`)",
        f"- **Dev In-Scope false-abstain rate @ T**: `{dev_false_abstain_rate:.4f}`",
        "",
    ]

    if low_oos_abstain_flag:
        outlier_summary = ", ".join(
            f"`{o['query_id']}` ({o['category']}, logit={o['top1_logit']:.3f})"
            for o in low_inscope_outliers
        )
        md_lines.extend(
            [
                "> [!WARNING]",
                f"> **LOW OOS ABSTAIN RATE DIAGNOSTIC (`dev_oos_abstain_rate={dev_oos_abstain_rate:.2%} < 50%` at margin-rule `T={t_val}`):**",
                f"> Minimum in-scope dev logits pulling `m` downward: {outlier_summary}.",
                "> **Concrete options WITHOUT changing `T`:**",
                "> 1. Add a Phase 2 prompt-level LLM refusal backstop verifying exact CVE/MITRE grounding against retrieved passages before answering.",
                "> 2. Expand Phase 2 identifier-existence checks for `unindexed_public_security` queries and router hard negatives for `lexical_decoy`/`org_proprietary`.",
                "> 3. Inspect whether cluttered/malformed outliers experienced `rerank_max_length` truncation.",
                "",
            ]
        )

    md_lines.extend(
        [
            "## 1. Per-Category Dev Results",
            "",
            "### OOS Dev (`oos_dev_60`)",
            "| Category | N | Router HARD_REFUSAL | Reranker < T | System Blocked | System False Accepts | Mean Logit | Max Logit |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for cat, st in oos_category_stats.items():
        ls = st["logit_stats"]
        md_lines.append(
            f"| `{cat}` | {st['count']} | {st['router_hard_refusal_rate']:.2%} | "
            f"{st['reranker_abstain_rate_at_T']:.2%} | {st['system_refusal_or_abstain_rate']:.2%} | "
            f"{st['false_accept_count_system']} | {ls['mean']:.3f} | {ls['max']:.3f} |"
        )

    md_lines.extend(
        [
            "",
            "### In-Scope Dev (`inscope_dev_100`)",
            "| Category | N | False Abstain @ T | Doc Recall@3 | Doc Recall@8 | Min Logit | Mean Logit |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for cat, st in inscope_category_stats.items():
        ls = st["logit_stats"]
        md_lines.append(
            f"| `{cat}` | {st['count']} | {st['false_abstain_rate_at_T']:.2%} | "
            f"{st['doc_recall_at_3']:.4f} | {st['doc_recall_at_8']:.4f} | "
            f"{ls['min']:.3f} | {ls['mean']:.3f} |"
        )

    md_lines.extend(
        [
            "",
            "## 2. Diagnostic Tail Cases",
            "",
            "### Lowest 10 In-Scope Dev Logits",
            "| ID | Category | Top-1 Logit | Route | Expected Docs | Top-1 Doc | Query |",
            "| --- | --- | ---: | --- | --- | --- | --- |",
        ]
    )
    for item in report_payload["lowest_10_inscope_dev"]:
        q_clean = item["query_snippet"].replace("\n", " ").replace("|", "/")
        md_lines.append(
            f"| `{item['query_id']}` | `{item['category']}` | {item['top1_logit']:.3f} | "
            f"`{item['router_route']}` | `{','.join(item['expected_doc_ids'])}` | `{item['top1_doc_id']}` | {q_clean} |"
        )

    md_lines.extend(
        [
            "",
            "### Highest 10 OOS Dev Logits",
            "| ID | Category | Top-1 Logit | Route | Top-1 Doc | Query |",
            "| --- | --- | ---: | --- | --- | --- |",
        ]
    )
    for item in report_payload["highest_10_oos_dev"]:
        q_clean = item["query_snippet"].replace("\n", " ").replace("|", "/")
        md_lines.append(
            f"| `{item['query_id']}` | `{item['category']}` | {item['top1_logit']:.3f} | "
            f"`{item['router_route']}` | `{item['top1_doc_id']}` | {q_clean} |"
        )

    md_lines.extend(
        [
            "",
            "## 3. Held-Out `golden_150.jsonl` Verification (Fixed `T`)",
            f"- **System overall block rate (router + abstention)**: `{golden_sys_refusal:.2%}`",
            f"- **Golden OOS (`out_of_scope`) block rate**: `{golden_oos_block:.2%}`",
            f"- **Golden In-Scope false-abstain rate**: `{golden_in_false_abstain:.2%}`",
            "",
            "| Question Type | N | Router HARD_REFUSAL | Reranker < T (Non-Refused) | Combined Blocked | Mean Logit |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for qtype, st in golden_type_stats.items():
        ls = st["logit_stats"]
        md_lines.append(
            f"| `{qtype}` | {st['count']} | {st['router_hard_refusal_rate']:.2%} | "
            f"{st['reranker_abstain_rate_non_refused']:.2%} | {st['combined_system_block_rate']:.2%} | {ls['mean']:.3f} |"
        )

    md_lines.extend(
        [
            "",
            "### Golden `OUT_OF_SCOPE` Per-Item Outcomes (`TEST_121..TEST_130`)",
            "| Query ID | Router Route | Top-1 Logit | Gate Abstained (`< T`) | System Blocked | Query |",
            "| --- | --- | ---: | --- | --- | --- |",
        ]
    )
    for g_oos in golden_oos_items:
        q_clean = str(g_oos["query"]).replace("\n", " ").replace("|", "/")[:120]
        md_lines.append(
            f"| `{g_oos['query_id']}` | `{g_oos['router_route']}` | {g_oos['top1_logit']:.3f} | "
            f"`{g_oos['gate_abstained']}` | `{g_oos['system_blocked']}` | {q_clean} |"
        )

    md_lines.extend(
        [
            "",
            "## 4. Full Threshold Sweep Grid (`[-2.0, 12.0]`, step `0.25`)",
            "| T | InScope False Abstain | InScope TPR | OOS False Accept | OOS TNR | Balanced Acc | System OOS False Accept |",
            "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for g in grid:
        marker = " **`<-- T`**" if g["T"] == t_val else ""
        md_lines.append(
            f"| {g['T']:.2f}{marker} | {g['inscope_false_abstain_rate']:.4f} | {g['inscope_tpr']:.4f} | "
            f"{g['oos_false_accept_rate']:.4f} | {g['oos_tnr']:.4f} | {g['balanced_accuracy']:.4f} | {g['system_oos_false_accept_rate']:.4f} |"
        )

    md_lines.extend(
        [
            "",
            "## 5. Phase 2 Recommendations",
            "- Monitor `unindexed_public_security` and `lexical_decoy` OOS queries where `router_route != HARD_REFUSAL`; expand Phase 2 corpus / identifier existence check if non-indexed CVEs share strong semantic priors with indexed CVEs.",
            "- Track `malformed` and `cluttered_context` false abstentions vs reranker `rerank_max_length` truncation; use Contingency L (`rerank_long_query_max_length`) if clutter tokens push core questions beyond the cross-encoder window.",
            "- Add router hard-negative examples specifically contrasting `unindexed_public_security` and `lexical_decoy` phrasing in Phase 2 fine-tuning while preserving strict `golden_150` holdout separation.",
            "",
        ]
    )

    json_path = cfg.reports_dir / "phase15_calibration_report.json"
    md_path = cfg.reports_dir / "phase15_calibration_report.md"
    _atomic_write_json(json_path, report_payload)
    _atomic_write_text(md_path, "\n".join(md_lines))

    return {
        "action": "report",
        "json_path": str(json_path),
        "md_path": str(md_path),
        "threshold_T": t_val,
        "roc_auc": round(roc_auc, 6),
    }


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint for calibration collection, threshold selection, and reporting."""
    parser = argparse.ArgumentParser(
        description="KAVACH-RAG Phase 1.5 calibrated abstention CLI."
    )
    parser.add_argument("--collect", action="store_true", help="Collect logits across devsets and golden_150.")
    parser.add_argument("--select", action="store_true", help="Select threshold T excluding golden_150.")
    parser.add_argument("--report", action="store_true", help="Write Phase 1.5 calibration reports.")
    parser.add_argument("--fresh", action="store_true", help="Discard partial collection checkpoint on --collect.")
    args = parser.parse_args(argv)

    if not (args.collect or args.select or args.report):
        parser.error("Must specify at least one of --collect, --select, --report")

    cfg = get_settings()
    if args.collect:
        res = _collect_logits(cfg, fresh=args.fresh)
        print(json.dumps(res, indent=2))
    if args.select:
        res = run_select(cfg)
        print(json.dumps(res, indent=2))
    if args.report:
        res = generate_report(cfg)
        print(json.dumps(res, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
