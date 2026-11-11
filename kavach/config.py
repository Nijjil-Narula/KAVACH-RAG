"""
kavach.config — Pydantic-settings configuration for KAVACH-RAG.

All settings can be overridden via KAVACH_-prefixed environment variables.
Paths are computed relative to the project root (parent of the kavach/ package).
"""
from __future__ import annotations

import functools
import json
import os
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, ValidationError
from pydantic_settings import BaseSettings, SettingsConfigDict

from kavach.exceptions import ConfigError

# Project root is the parent of the directory containing this file.
_ROOT = Path(__file__).parent.parent

_PRECISION_LOCK_FIELDS: tuple[str, ...] = (
    "embed_precision",
    "rerank_precision",
    "router_precision",
    "rerank_max_length",
    "cpu_affinity_mode",
    "rerank_long_query_max_length",
    "rerank_long_query_token_threshold",
    "contingency_l_applied",
)


class Settings(BaseSettings):
    """
    KAVACH-RAG runtime configuration.

    All fields with Path type are absolute paths derived from the project root.
    Override any field by setting KAVACH_{FIELD_NAME} as an environment variable.
    """

    model_config = SettingsConfigDict(
        env_prefix="KAVACH_",
        extra="ignore",
        validate_assignment=True,
    )

    # ── Paths ─────────────────────────────────────────────────────────────
    data_raw: Path = Field(default=_ROOT / "data" / "raw")
    data_processed: Path = Field(default=_ROOT / "data" / "processed")
    data_golden: Path = Field(default=_ROOT / "data" / "golden")
    models_dir: Path = Field(default=_ROOT / "models")
    reports_dir: Path = Field(default=_ROOT / "reports")
    logs_dir: Path = Field(default=_ROOT / "logs")

    # ── Model identifiers ─────────────────────────────────────────────────
    embed_model_id: str = "codefuse-ai/F2LLM-v2-0.6B"
    rerank_model_id: str = "cross-encoder/ettin-reranker-150m-v1"
    router_base_model_id: str = "microsoft/deberta-v3-small"

    # ── Precision & Hardware Residency ────────────────────────────────────
    rss_ceiling_mb: float = 4096.0
    embed_precision: Literal["bf16", "fp32", "int8"] = "fp32"
    rerank_precision: Literal["bf16", "fp32", "int8"] = "fp32"
    router_precision: Literal["bf16", "fp32", "int8"] = "fp32"
    cpu_affinity_mode: Literal["off", "p_core_distinct", "p_core_smt"] = "off"

    # ── Embedding ─────────────────────────────────────────────────────────
    embed_dim_full: int = 1024
    embed_dim_mrl: int = 512
    embed_max_tokens: int = 512
    query_prefix: str = (
        "Represent this cybersecurity query for retrieving relevant "
        "vulnerability and mitigation context: "
    )
    passage_prefix: str = ""

    # ── Retrieval & Reranking ─────────────────────────────────────────────
    candidate_k: int = 8
    rerank_top_k_default: int = 3
    rerank_top_k_max: int = 5
    rerank_max_length: int = 576
    rerank_long_query_max_length: int | None = None
    rerank_long_query_token_threshold: int = 96
    contingency_l_applied: bool = False
    router_conf_threshold: float = 0.65

    # ── Calibrated Abstention ─────────────────────────────────────────────
    abstention_enabled: bool = True
    abstain_logit_threshold: float | None = None

    # ── Chunking ──────────────────────────────────────────────────────────
    chunk_window: int = 512
    chunk_overlap: int = 64

    # ── Threads ───────────────────────────────────────────────────────────
    torch_threads: int = 2
    interop_threads: int = 1
    build_threads: int = 2

    def load_runtime_locks(self) -> Settings:
        """Overlay precision_lock.json and abstain_calibration.json unless overridden by KAVACH_* env."""
        precision_lock_path = self.models_dir / "precision_lock.json"
        if precision_lock_path.exists():
            try:
                lock_data: dict[str, Any] = json.loads(
                    precision_lock_path.read_text(encoding="utf-8")
                )
            except (OSError, json.JSONDecodeError) as exc:
                raise ConfigError(
                    f"Failed to parse {precision_lock_path}: {exc}",
                    details={"path": str(precision_lock_path), "error": str(exc)},
                ) from exc

            if not isinstance(lock_data, dict):
                raise ConfigError(
                    f"Invalid JSON object in {precision_lock_path}",
                    details={"path": str(precision_lock_path)},
                )

            for field_name in _PRECISION_LOCK_FIELDS:
                env_key = f"KAVACH_{field_name.upper()}"
                if env_key in os.environ:
                    continue
                if field_name in lock_data:
                    try:
                        setattr(self, field_name, lock_data[field_name])
                    except ValidationError as exc:
                        raise ConfigError(
                            f"Invalid '{field_name}' in {precision_lock_path}: {exc}",
                            details={
                                "path": str(precision_lock_path),
                                "field": field_name,
                                "value": lock_data[field_name],
                            },
                        ) from exc

        abstain_cal_path = self.models_dir / "abstain_calibration.json"
        if abstain_cal_path.exists():
            try:
                cal_data: dict[str, Any] = json.loads(
                    abstain_cal_path.read_text(encoding="utf-8")
                )
            except (OSError, json.JSONDecodeError) as exc:
                raise ConfigError(
                    f"Failed to parse {abstain_cal_path}: {exc}",
                    details={"path": str(abstain_cal_path), "error": str(exc)},
                ) from exc

            if not isinstance(cal_data, dict):
                raise ConfigError(
                    f"Invalid JSON object in {abstain_cal_path}",
                    details={"path": str(abstain_cal_path)},
                )

            if "KAVACH_ABSTAIN_LOGIT_THRESHOLD" not in os.environ:
                if "threshold_T" in cal_data and cal_data["threshold_T"] is not None:
                    try:
                        self.abstain_logit_threshold = float(cal_data["threshold_T"])
                    except (TypeError, ValueError, ValidationError) as exc:
                        raise ConfigError(
                            f"Invalid 'threshold_T' in {abstain_cal_path}: {exc}",
                            details={
                                "path": str(abstain_cal_path),
                                "threshold_T": cal_data.get("threshold_T"),
                            },
                        ) from exc

        return self


@functools.lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the singleton Settings instance with runtime locks overlaid."""
    return Settings().load_runtime_locks()
