"""
kavach.retrieval.embedder — Tier 2 Dense Bi-Encoder (F2LLM-v2-0.6B + MRL-512).

Eagerly loads the configured bi-encoder on CPU in the configured precision
(``fp32``, ``bf16``, or dynamic ``int8`` on non-embedding Linear layers),
applies asymmetric query/passage prefixes, respects pooling and padding side
recorded in ``models/embedder_notes.md``, truncates to 512 tokens,
L2-normalises both the full 1,024-dim output and the Matryoshka 512-dim slice,
and caches results by content hash (SHA-256 of ``kind + "\\x00" + text``).
"""
from __future__ import annotations

import os

_PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("HF_HOME", os.path.join(_PROJECT_ROOT, "models", "hf_cache"))

from kavach.runtime_guard import apply_hardware_guard  # noqa: E402

import dataclasses
import gc
import hashlib
import json
import logging
import re
from pathlib import Path
from typing import Literal, Optional, Sequence

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from kavach.config import Settings, get_settings
from kavach.exceptions import ModelLoadError
from kavach.retrieval.quantization import quantize_linear_layers

log = logging.getLogger(__name__)

_PRECISION_DTYPE_MAP: dict[str, torch.dtype] = {
    "bf16": torch.bfloat16,
    "bfloat16": torch.bfloat16,
    "fp32": torch.float32,
    "float32": torch.float32,
    "int8": torch.float32,
}


@dataclasses.dataclass
class EmbeddingBatch:
    """Pair of L2-normalised float32 embedding matrices.

    Attributes:
        full1024: Shape ``(N, 1024)`` L2-normalised float32 embeddings.
        mrl512: Shape ``(N, 512)`` Matryoshka slice, L2 re-normalised float32.
    """

    full1024: np.ndarray
    mrl512: np.ndarray


def _l2_normalize_np(arr: np.ndarray) -> np.ndarray:
    """L2-normalise rows of a 2-D float32 array safely."""
    if arr.size == 0:
        return arr.astype(np.float32)
    arr_f32 = arr.astype(np.float32, copy=False)
    norms = np.linalg.norm(arr_f32, axis=1, keepdims=True)
    norms = np.where(norms > 1e-12, norms, 1.0)
    return (arr_f32 / norms).astype(np.float32, copy=False)


def mean_pool(last_hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Attention-masked mean pooling across sequence length."""
    mask_expanded = attention_mask.unsqueeze(-1).float()
    sum_hidden = (last_hidden.float() * mask_expanded).sum(dim=1)
    count = mask_expanded.sum(dim=1).clamp(min=1e-9)
    return sum_hidden / count


def last_token_pool(
    last_hidden: torch.Tensor,
    attention_mask: torch.Tensor,
    padding_side: str = "right",
) -> torch.Tensor:
    """Pool the last non-padding token embedding for left or right padding.

    When ``padding_side == "left"``, every sequence ends at index ``-1``.
    When ``padding_side == "right"``, the final non-padding token sits at
    ``attention_mask.sum(dim=1) - 1``.
    """
    if padding_side == "left":
        return last_hidden[:, -1, :].float()
    seq_len = attention_mask.sum(dim=1) - 1
    batch_size = last_hidden.size(0)
    idx = seq_len.clamp(min=0).long()
    return last_hidden[
        torch.arange(batch_size, device=last_hidden.device), idx, :
    ].float()


def pool_hidden(
    last_hidden: torch.Tensor,
    attention_mask: torch.Tensor,
    pooling_method: str,
    padding_side: str,
) -> torch.Tensor:
    """Dispatch pooling strategy (``last_token``, ``mean``, or ``cls``)."""
    if pooling_method == "last_token":
        return last_token_pool(last_hidden, attention_mask, padding_side=padding_side)
    if pooling_method == "cls":
        return last_hidden[:, 0, :].float()
    return mean_pool(last_hidden, attention_mask)


def content_hash(kind: Literal["query", "passage"], text: str) -> str:
    """Return deterministic SHA-256 hex digest for ``kind + '\\x00' + text``."""
    payload = f"{kind}\x00{text}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _parse_embedder_notes(notes_path: Path) -> dict[str, str]:
    """Parse pooling_method, padding_side, and model_id from embedder_notes.md."""
    parsed: dict[str, str] = {}
    if not notes_path.exists():
        return parsed
    text = notes_path.read_text(encoding="utf-8")
    for line in text.splitlines():
        m_header = re.match(r"^#\s*Embedder Notes:\s*(.+)$", line.strip())
        if m_header and "model_id" not in parsed:
            parsed["model_id"] = m_header.group(1).strip()
        m_id = re.search(r"\*\*Embed model id:\*\*\s*(.+)", line, re.IGNORECASE)
        if m_id:
            parsed["model_id"] = m_id.group(1).strip()
        m_pool = re.search(r"\*\*Pooling method:\*\*\s*(\S+)", line, re.IGNORECASE)
        if m_pool:
            parsed["pooling_method"] = m_pool.group(1).strip().lower()
        m_pad = re.search(r"\*\*Padding side:\*\*\s*(\S+)", line, re.IGNORECASE)
        if m_pad:
            parsed["padding_side"] = m_pad.group(1).strip().lower()
    return parsed


class Embedder:
    """CPU-resident dense bi-encoder wrapper with MRL-512 slicing and memo cache."""

    def __init__(self, settings: Optional[Settings] = None) -> None:
        apply_hardware_guard()
        self._settings = settings or get_settings()
        self._memo: dict[str, tuple[np.ndarray, np.ndarray]] = {}

        notes_meta = _parse_embedder_notes(
            self._settings.models_dir / "embedder_notes.md"
        )
        preflight_meta = self._read_preflight_meta()

        self.model_id: str = (
            notes_meta.get("model_id")
            or preflight_meta.get("embed_model_id")
            or self._settings.embed_model_id
        )
        self.pooling_method: str = (
            notes_meta.get("pooling_method")
            or preflight_meta.get("pooling_method")
            or ""
        )
        self.padding_side: str = (
            notes_meta.get("padding_side")
            or preflight_meta.get("padding_side")
            or ""
        )

        self.precision: str = self._settings.embed_precision.lower()
        if self.precision not in _PRECISION_DTYPE_MAP:
            raise ModelLoadError(
                f"Unsupported embed_precision: {self._settings.embed_precision!r}",
                details={"embed_precision": self._settings.embed_precision},
            )
        self._dtype = _PRECISION_DTYPE_MAP[self.precision]
        self._tokenizer = None
        self._model = None
        self._load_model()

    @property
    def is_loaded(self) -> bool:
        """Return True if the bi-encoder model weights are currently resident."""
        return self._model is not None and self._tokenizer is not None

    def _load_model(self) -> None:
        """Eagerly load the bi-encoder model and tokenizer on CPU."""
        try:
            tok = None
            mdl = None
            for local_only in (True, False):
                try:
                    tok = AutoTokenizer.from_pretrained(
                        self.model_id, local_files_only=local_only
                    )
                    if tok.pad_token is None:
                        tok.pad_token = tok.eos_token or tok.unk_token
                    if self.padding_side not in ("left", "right"):
                        tok_pad = getattr(tok, "padding_side", "right")
                        self.padding_side = (
                            tok_pad if tok_pad in ("left", "right") else "right"
                        )
                    tok.padding_side = self.padding_side
                    if self.pooling_method not in ("mean", "last_token", "cls"):
                        self.pooling_method = (
                            "last_token" if self.padding_side == "left" else "mean"
                        )
                    mdl = AutoModel.from_pretrained(
                        self.model_id,
                        dtype=self._dtype,
                        local_files_only=local_only,
                    )
                    break
                except OSError:
                    if not local_only:
                        raise
            assert tok is not None and mdl is not None
            mdl.eval()
            if self.precision == "int8":
                q_mdl = quantize_linear_layers(
                    mdl,
                    exclude_names=("embed_tokens", "embeddings", "word_embeddings"),
                )
                del mdl
                mdl = q_mdl
            self._tokenizer = tok
            self._model = mdl
            gc.collect()
            log.info(
                "Embedder ready: %s (precision=%s, dtype=%s, pooling=%s, padding_side=%s)",
                self.model_id,
                self.precision,
                self._dtype,
                self.pooling_method,
                self.padding_side,
            )
        except Exception as exc:
            raise ModelLoadError(
                f"Failed to load embedder '{self.model_id}': {exc}",
                details={
                    "model_id": self.model_id,
                    "precision": self.precision,
                    "error": str(exc),
                },
            ) from exc

    def _read_preflight_meta(self) -> dict[str, str]:
        """Load optional metadata from reports/preflight.json."""
        path = self._settings.reports_dir / "preflight.json"
        if not path.exists():
            return {}
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            out: dict[str, str] = {}
            for key in ("embed_model_id", "pooling_method", "padding_side"):
                val = data.get(key)
                if isinstance(val, str) and val:
                    out[key] = val
            return out
        except Exception as exc:
            log.warning("Could not parse %s: %s", path, exc)
            return {}

    def clear_cache(self) -> None:
        """Clear the in-memory embedding memo."""
        self._memo.clear()
        gc.collect()

    def free_tokenizer_caches(self) -> None:
        """Clear cached embeddings and run Python garbage collection."""
        self._memo.clear()
        gc.collect()

    def warmup(self, passes: int = 3) -> None:
        """Prime oneDNN CPU caches across short and ~512-token query inputs."""
        if not self.is_loaded:
            raise ModelLoadError(
                "Embedder must be loaded before warmup.",
                details={"model_id": self.model_id},
            )
        short_q = "What is the CVSS score for CVE-2024-3094?"
        long_q = " ".join(
            ["CVE-2024-3094 XZ Utils liblzma vulnerability mitigation analysis"] * 75
        )
        for _ in range(max(1, int(passes))):
            self._encode_texts([short_q], kind="query")
            self._encode_texts([long_q], kind="query")

    def _encode_texts(
        self, texts: Sequence[str], kind: Literal["query", "passage"]
    ) -> tuple[np.ndarray, np.ndarray]:
        """Encode *texts* without mutating the content-hash memo."""
        assert self._tokenizer is not None and self._model is not None
        prefix = (
            self._settings.query_prefix
            if kind == "query"
            else self._settings.passage_prefix
        )
        prefixed = [prefix + t for t in texts]
        enc = self._tokenizer(
            prefixed,
            padding=True,
            truncation=True,
            max_length=self._settings.embed_max_tokens,
            return_tensors="pt",
        )
        with torch.inference_mode():
            outputs = self._model(**enc)
            pooled = pool_hidden(
                outputs.last_hidden_state,
                enc["attention_mask"],
                pooling_method=self.pooling_method,
                padding_side=self.padding_side,
            )
        raw_np = pooled.cpu().numpy()
        full_vecs = _l2_normalize_np(raw_np)
        mrl_dim = self._settings.embed_dim_mrl
        if full_vecs.shape[1] < mrl_dim:
            raise ModelLoadError(
                f"Embedder output dim {full_vecs.shape[1]} < mrl_dim {mrl_dim}.",
                details={"output_dim": int(full_vecs.shape[1]), "mrl_dim": mrl_dim},
            )
        mrl_vecs = _l2_normalize_np(full_vecs[:, :mrl_dim])
        return full_vecs, mrl_vecs

    def embed(
        self,
        texts: Sequence[str],
        kind: Literal["query", "passage"],
        batch_size: int = 32,
    ) -> EmbeddingBatch:
        """Embed *texts* into L2-normalised 1024-dim and MRL 512-dim vectors.

        Parameters
        ----------
        texts:
            Sequence of strings to encode.
        kind:
            Either ``"query"`` (prepends ``settings.query_prefix``) or
            ``"passage"`` (prepends ``settings.passage_prefix``).
        batch_size:
            Maximum forward-pass batch size (default 32).
        """
        if isinstance(texts, str):
            raise TypeError("texts must be a sequence of strings, not a bare str.")
        if kind not in ("query", "passage"):
            raise ValueError(f"kind must be 'query' or 'passage', got {kind!r}")
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")

        full_dim = self._settings.embed_dim_full
        mrl_dim = self._settings.embed_dim_mrl

        if not texts:
            return EmbeddingBatch(
                full1024=np.empty((0, full_dim), dtype=np.float32),
                mrl512=np.empty((0, mrl_dim), dtype=np.float32),
            )

        keys = [content_hash(kind, t) for t in texts]
        uncached_items: list[tuple[str, str]] = []
        seen_uncached: set[str] = set()
        for key, text in zip(keys, texts):
            if key not in self._memo and key not in seen_uncached:
                seen_uncached.add(key)
                uncached_items.append((key, text))

        if uncached_items:
            if not self.is_loaded:
                raise ModelLoadError(
                    "Embedder weights are not resident.",
                    details={"model_id": self.model_id},
                )
            for start in range(0, len(uncached_items), batch_size):
                batch = uncached_items[start : start + batch_size]
                batch_keys = [k for k, _ in batch]
                batch_texts = [t for _, t in batch]
                full_vecs, mrl_vecs = self._encode_texts(batch_texts, kind=kind)
                for idx, key in enumerate(batch_keys):
                    self._memo[key] = (full_vecs[idx], mrl_vecs[idx])

        full_out = np.vstack([self._memo[k][0] for k in keys]).astype(
            np.float32, copy=False
        )
        mrl_out = np.vstack([self._memo[k][1] for k in keys]).astype(
            np.float32, copy=False
        )
        return EmbeddingBatch(full1024=full_out, mrl512=mrl_out)
