"""
kavach.exceptions — Project-wide exception hierarchy for KAVACH-RAG.

All exceptions carry a human-readable *message* and an optional *details*
dictionary for structured metadata (e.g., file paths, CVE IDs, metric values).
"""
from __future__ import annotations


class KavachError(Exception):
    """
    Base exception for all KAVACH-RAG errors.

    Attributes:
        message: Human-readable description of the error.
        details: Optional dictionary with structured metadata.
    """

    def __init__(self, message: str, details: dict | None = None) -> None:
        super().__init__(message)
        self.message: str = message
        self.details: dict = details or {}

    def __repr__(self) -> str:  # noqa: D105
        return f"{type(self).__name__}(message={self.message!r}, details={self.details!r})"


class ConfigError(KavachError):
    """Raised when configuration values are missing, invalid, or inconsistent."""


class HardwareIsolationError(KavachError):
    """Raised when the CPU-only hardware isolation contract is violated.

    This includes detection of CUDA-enabled PyTorch builds, non-zero VRAM
    usage by any kavach process, or failure to set thread-count limits.
    """


class CorpusIntegrityError(KavachError):
    """Raised when corpus data fails validation or required files are absent.

    Examples include missing PROVENANCE.json, schema mismatches in parquet
    files, or missing prerequisite artefacts from sibling subagents.
    """


class ModelLoadError(KavachError):
    """Raised when a model or tokenizer cannot be loaded from cache or hub.

    Details typically include ``model_id`` and the underlying exception message.
    """


class IndexIntegrityError(KavachError):
    """Raised when a FAISS index fails consistency checks.

    Details typically include ``index_path``, ``expected_ntotal``, and
    ``actual_ntotal``.
    """


class RetrievalError(KavachError):
    """Raised when retrieval pipeline produces an invalid or empty result set.

    Details typically include ``query``, ``candidate_k``, and the stage at
    which the failure occurred.
    """


class GoldenDatasetValidationError(KavachError):
    """Raised when a golden-dataset item fails schema or semantic validation.

    Details typically include ``test_id``, ``field``, and ``reason``.
    """
