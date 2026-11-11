"""
kavach.retrieval.router — Tier 1 AgentIntentRouter.

Dual-track routing:
  Track A: Deterministic regex fast-path for exact security identifiers.
  Track B: microsoft/deberta-v3-small 3-class classifier for intent
           supporting ``fp32``, ``bf16``, and dynamic ``int8`` precision.

Route values:
  FAST_PATH_EXACT_LOOKUP — exact identifier match
  SEMANTIC_RAG           — semantic vector search needed
  HARD_REFUSAL           — out-of-scope query, refuse immediately

MALFORMED_INPUT from Track B maps to SEMANTIC_RAG with is_malformed=True.
"""
from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from kavach.runtime_guard import apply_hardware_guard  # noqa: E402

import dataclasses
import gc
import logging
import re
import time
import unicodedata
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from kavach.config import Settings, get_settings
from kavach.exceptions import ModelLoadError
from kavach.retrieval.quantization import quantize_linear_layers

log = logging.getLogger(__name__)

# ── Route constants ───────────────────────────────────────────────────────────
ROUTE_FAST_PATH = "FAST_PATH_EXACT_LOOKUP"
ROUTE_SEMANTIC = "SEMANTIC_RAG"
ROUTE_REFUSAL = "HARD_REFUSAL"
_CLASS_NAMES = [ROUTE_SEMANTIC, ROUTE_REFUSAL, "MALFORMED_INPUT"]

_PRECISION_DTYPE_MAP: dict[str, torch.dtype] = {
    "bf16": torch.bfloat16,
    "bfloat16": torch.bfloat16,
    "fp32": torch.float32,
    "float32": torch.float32,
    "int8": torch.float32,
}

# ── Identifier patterns ───────────────────────────────────────────────────────
# For normalisation (repair) — anchored with word boundaries to avoid mangling prose.
_RE_CVE_REPAIR = re.compile(
    r"\bcve[-\u2010-\u2013\s]?(\d{4})[-\u2010-\u2013\s](\d{4,7})\b",
    re.IGNORECASE | re.UNICODE,
)
_RE_CWE_REPAIR = re.compile(
    r"\bcwe[-\u2010-\u2013\s]?(\d{1,5})\b",
    re.IGNORECASE | re.UNICODE,
)
_RE_MITRE_REPAIR = re.compile(
    r"\b(T|TA|M)[-\u2010-\u2013\s]?(\d{4})(?:[-\u2010-\u2013\s.](\d{3}))?\b",
    re.IGNORECASE | re.UNICODE,
)

# For exact match (Track A) — anchored to full string
_RE_CVE_EXACT = re.compile(r"^CVE-\d{4}-\d{4,7}$")
_RE_CWE_EXACT = re.compile(r"^CWE-\d{1,5}$")
_RE_MITRE_EXACT = re.compile(r"^(T\d{4}(\.\d{3})?|TA\d{4}|M\d{4})$")


def _is_control_char(ch: str) -> bool:
    """Return True if *ch* belongs to a Unicode control/format category."""
    return unicodedata.category(ch).startswith("C")


def normalize_query(q: str) -> str:
    """Normalise a raw user query string.

    Steps:
      1. NFKC Unicode normalisation.
      2. Strip control characters.
      3. Collapse whitespace to single spaces and strip.
      4. Repair broken identifiers (CVE, CWE, MITRE) on word boundaries.
    """
    q = unicodedata.normalize("NFKC", q)
    q = "".join(ch for ch in q if not _is_control_char(ch))
    q = " ".join(q.split())

    q = _RE_CVE_REPAIR.sub(lambda m: f"CVE-{m.group(1)}-{m.group(2)}", q)
    q = _RE_CWE_REPAIR.sub(lambda m: f"CWE-{m.group(1)}", q)

    def _mitre_replace(m: re.Match[str]) -> str:
        prefix = m.group(1).upper()
        main_num = m.group(2)
        sub_num = m.group(3)
        if sub_num:
            return f"{prefix}{main_num}.{sub_num}"
        return f"{prefix}{main_num}"

    q = _RE_MITRE_REPAIR.sub(_mitre_replace, q)
    return q.strip()


@dataclasses.dataclass
class RouteDecision:
    """Result of routing a single query through AgentIntentRouter."""

    route: str                         # FAST_PATH_EXACT_LOOKUP | SEMANTIC_RAG | HARD_REFUSAL
    confidence: float                  # 0.0–1.0
    matched_identifier: Optional[str]  # Non-None for FAST_PATH
    normalized_query: str              # Cleaned query text
    is_malformed: bool                 # True when model predicted MALFORMED_INPUT
    latency_ms: float                  # Wall-clock time for this route() call


class AgentIntentRouter:
    """Dual-track intent router: regex fast-path + DeBERTa classifier.

    Track A queries resolve purely via regex without loading Track B weights;
    calling :meth:`load` (used by :class:`VectorRetrievalPipeline`) keeps
    Track B permanently resident in memory.
    """

    _ROUTER_DIR_NAME = "router_deberta"

    def __init__(self, settings: Optional[Settings] = None) -> None:
        apply_hardware_guard()
        self._settings = settings or get_settings()
        self.precision: str = self._settings.router_precision.lower()
        if self.precision not in _PRECISION_DTYPE_MAP:
            raise ModelLoadError(
                f"Unsupported router_precision: {self._settings.router_precision!r}",
                details={"router_precision": self._settings.router_precision},
            )
        self._dtype = _PRECISION_DTYPE_MAP[self.precision]
        self._model: Optional[AutoModelForSequenceClassification] = None
        self._tokenizer: Optional[AutoTokenizer] = None
        self._model_loaded = False

    # ── Lifecycle helpers ─────────────────────────────────────────────────────

    @property
    def is_loaded(self) -> bool:
        """Return True if the Track B classifier weights are currently in memory."""
        return self._model_loaded

    def load(self) -> None:
        """Eagerly load the Track B classifier model and tokenizer onto CPU."""
        self._ensure_model_loaded()

    def free_tokenizer_cache(self) -> None:
        """Run Python garbage collection to clear transient tokenizer references."""
        gc.collect()

    def warmup(self, passes: int = 3) -> None:
        """Run *passes* warmup inferences with a 128-token string to prime oneDNN."""
        self.load()
        assert self._tokenizer is not None
        seed_text = " ".join(
            ["CVE vulnerability advisory CVSS severity mitigation analysis"] * 32
        )
        for _ in range(max(1, int(passes))):
            self._classify(seed_text)

    # ── Public API ────────────────────────────────────────────────────────────

    def needs_model_for(self, query: str) -> bool:
        """Return True if *query* requires the Track B neural classifier."""
        return self._fast_path(normalize_query(query)) is None

    def route(self, query: str) -> RouteDecision:
        """Route *query* and return a :class:`RouteDecision`.

        Track A (regex) is tried first; if no exact identifier is found the
        DeBERTa classifier (Track B) is invoked.
        """
        t_start = time.perf_counter()

        norm = normalize_query(query)

        # ── Track A — Exact identifier fast-path ─────────────────────────────
        fast_path_match = self._fast_path(norm)
        if fast_path_match is not None:
            latency_ms = (time.perf_counter() - t_start) * 1000.0
            return RouteDecision(
                route=ROUTE_FAST_PATH,
                confidence=1.0,
                matched_identifier=fast_path_match,
                normalized_query=norm,
                is_malformed=False,
                latency_ms=latency_ms,
            )

        # ── Track B — DeBERTa classifier ─────────────────────────────────────
        self._ensure_model_loaded()
        probs = self._classify(norm)  # shape (3,) numpy float32

        best_idx = int(probs.argmax())
        best_class = _CLASS_NAMES[best_idx]
        confidence = float(probs[best_idx])

        if best_class == "MALFORMED_INPUT":
            route = ROUTE_SEMANTIC
            is_malformed = True
        elif confidence < self._settings.router_conf_threshold:
            route = ROUTE_SEMANTIC
            is_malformed = False
        else:
            route = best_class  # SEMANTIC_RAG or HARD_REFUSAL
            is_malformed = False

        latency_ms = (time.perf_counter() - t_start) * 1000.0
        return RouteDecision(
            route=route,
            confidence=confidence,
            matched_identifier=None,
            normalized_query=norm,
            is_malformed=is_malformed,
            latency_ms=latency_ms,
        )

    # ── Private helpers ───────────────────────────────────────────────────────

    def _fast_path(self, normalized: str) -> Optional[str]:
        """Return the matched identifier string or None."""
        if _RE_CVE_EXACT.match(normalized):
            return normalized
        if _RE_CWE_EXACT.match(normalized):
            return normalized
        if _RE_MITRE_EXACT.match(normalized):
            return normalized
        return None

    def _ensure_model_loaded(self) -> None:
        """Load the DeBERTa Track B model on demand if not yet resident."""
        if self._model_loaded:
            return

        trained_dir = self._settings.models_dir / self._ROUTER_DIR_NAME

        if trained_dir.exists() and any(trained_dir.iterdir()):
            model_path = str(trained_dir)
            log.info("Loading trained router from %s", model_path)
        else:
            model_path = self._settings.router_base_model_id
            log.info(
                "Trained router not found; loading base model %s with 3-class head.",
                model_path,
            )

        try:
            tok = None
            mdl = None
            for local_only in (True, False):
                try:
                    tok = AutoTokenizer.from_pretrained(
                        model_path, local_files_only=local_only
                    )
                    mdl = AutoModelForSequenceClassification.from_pretrained(
                        model_path,
                        num_labels=3,
                        dtype=self._dtype,
                        ignore_mismatched_sizes=True,
                        local_files_only=local_only,
                    )
                    break
                except OSError:
                    if not local_only or Path(model_path).is_dir():
                        raise
            assert tok is not None and mdl is not None
            mdl.eval()
            if self.precision == "int8":
                q_mdl = quantize_linear_layers(
                    mdl,
                    exclude_names=(
                        "classifier",
                        "pooler",
                        "embeddings",
                        "word_embeddings",
                        "rel_embeddings",
                    ),
                )
                del mdl
                mdl = q_mdl
            self._tokenizer = tok
            self._model = mdl
            self._model_loaded = True
            gc.collect()
            log.info(
                "Router model ready (precision=%s, dtype=%s). Labels: %s",
                self.precision,
                self._dtype,
                _CLASS_NAMES,
            )
        except Exception as exc:
            raise ModelLoadError(
                f"Failed to load router model '{model_path}': {exc}",
                details={
                    "model_id": model_path,
                    "precision": self.precision,
                    "error": str(exc),
                },
            ) from exc

    def _classify(self, text: str) -> np.ndarray:
        """Run the DeBERTa model and return softmax probabilities (shape 3,)."""
        if self._tokenizer is None or self._model is None:
            raise ModelLoadError(
                "Router model not loaded prior to classification.",
                details={"model_id": self._settings.router_base_model_id},
            )
        enc = self._tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=128,
            padding=False,
        )
        with torch.inference_mode():
            logits = self._model(**enc).logits  # shape (1, 3)
        probs = torch.softmax(logits[0].float(), dim=-1).cpu().numpy()
        return probs
