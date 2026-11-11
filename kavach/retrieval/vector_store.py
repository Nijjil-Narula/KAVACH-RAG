"""
kavach.retrieval.vector_store — Tier 3 ChunkStore, FAISS SQ8Index, and FlatIndex.

Provides:
  - ChunkStore: O(1) chunk_id lookup, doc_id mapping, and O(1) hash index over
    primary_identifier and secondary_identifiers backed by Parquet.
  - SQ8Index: FAISS IndexScalarQuantizer (QT_8bit, METRIC_INNER_PRODUCT) with
    atomic save/load and strict ntotal == len(idmap) integrity verification.
  - FlatIndex: Uncompressed FAISS IndexFlatIP wrapper for ablation baselines.

Hardware rule: ``torch`` is always imported before ``faiss``.
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

import json
import logging
import math
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd
import torch  # noqa: F401 — must precede faiss import
import faiss

from kavach.exceptions import CorpusIntegrityError, IndexIntegrityError

log = logging.getLogger(__name__)


def _normalize_row(row: dict[str, Any]) -> dict[str, Any]:
    """Convert Parquet/Pandas cell types (ndarray, NaN) into clean Python values."""
    cleaned: dict[str, Any] = {}
    for key, val in row.items():
        if key == "secondary_identifiers":
            if isinstance(val, (list, tuple, np.ndarray)):
                cleaned[key] = [str(item) for item in val]
            elif val is None or (isinstance(val, float) and math.isnan(val)):
                cleaned[key] = []
            else:
                cleaned[key] = [str(val)]
        elif isinstance(val, np.ndarray):
            cleaned[key] = val.tolist()
        elif isinstance(val, (np.integer,)):
            cleaned[key] = int(val)
        elif isinstance(val, (np.floating, float)):
            cleaned[key] = None if math.isnan(float(val)) else float(val)
        elif val is pd.NA:
            cleaned[key] = None
        else:
            cleaned[key] = val
    return cleaned


class ChunkStore:
    """In-memory indexed view over a KAVACH-RAG chunk Parquet file."""

    def __init__(self, parquet_path: Path | str) -> None:
        self.parquet_path = Path(parquet_path)
        if not self.parquet_path.exists():
            raise CorpusIntegrityError(
                f"Chunk parquet file not found: {self.parquet_path}",
                details={"parquet_path": str(self.parquet_path)},
            )

        df = pd.read_parquet(self.parquet_path)
        self._by_chunk_id: dict[str, dict[str, Any]] = {}
        self._chunk_to_doc: dict[str, str] = {}
        self._chunk_ids: list[str] = []
        self._id_index: dict[str, list[dict[str, Any]]] = {}
        self._seen_per_key: dict[str, set[str]] = {}

        for raw_row in df.to_dict(orient="records"):
            rec = _normalize_row(raw_row)
            cid = str(rec["chunk_id"])
            did = str(rec["doc_id"])
            self._by_chunk_id[cid] = rec
            self._chunk_to_doc[cid] = did
            self._chunk_ids.append(cid)

            identifiers: list[str] = []
            primary = rec.get("primary_identifier")
            if primary:
                identifiers.append(str(primary))
            for sec in rec.get("secondary_identifiers", []):
                if sec:
                    identifiers.append(str(sec))

            for ident in identifiers:
                for key in {ident, ident.strip().upper()}:
                    if not key:
                        continue
                    seen = self._seen_per_key.setdefault(key, set())
                    if cid not in seen:
                        seen.add(cid)
                        self._id_index.setdefault(key, []).append(rec)

        for key, items in self._id_index.items():
            items.sort(
                key=lambda r: (
                    int(r.get("chunk_index", 0) or 0),
                    str(r.get("chunk_id", "")),
                )
            )

        log.info(
            "ChunkStore loaded %d chunks and %d identifier keys from %s",
            len(self._chunk_ids),
            len(self._id_index),
            self.parquet_path,
        )

    def __len__(self) -> int:
        return len(self._chunk_ids)

    def get(self, chunk_id: str) -> dict[str, Any]:
        """Return a shallow copy of the chunk record for *chunk_id*."""
        if chunk_id not in self._by_chunk_id:
            raise KeyError(f"Unknown chunk_id: {chunk_id!r}")
        return dict(self._by_chunk_id[chunk_id])

    def doc_id_of(self, chunk_id: str) -> str:
        """Return the parent ``doc_id`` for *chunk_id*."""
        if chunk_id not in self._chunk_to_doc:
            raise KeyError(f"Unknown chunk_id: {chunk_id!r}")
        return self._chunk_to_doc[chunk_id]

    def lookup_identifier(self, identifier: str) -> list[dict[str, Any]]:
        """O(1) lookup over ``primary_identifier`` and ``secondary_identifiers``."""
        if not identifier:
            return []
        hits = self._id_index.get(identifier)
        if hits is None:
            hits = self._id_index.get(identifier.strip().upper(), [])
        return [dict(item) for item in hits]

    def all_chunk_ids(self) -> list[str]:
        """Return all chunk IDs in corpus order."""
        return list(self._chunk_ids)

    def chunk_to_doc_map(self) -> dict[str, str]:
        """Return a copy of the ``chunk_id -> doc_id`` dictionary."""
        return dict(self._chunk_to_doc)


def _idmap_path_for(index_path: Path) -> Path:
    """Return the companion idmap JSON path for a FAISS index path."""
    return index_path.with_suffix(".idmap.json")


def _prepare_vectors(vectors: np.ndarray, expected_dim: int) -> np.ndarray:
    """Validate and convert vectors to C-contiguous float32."""
    arr = np.asarray(vectors, dtype=np.float32)
    if arr.ndim != 2:
        raise IndexIntegrityError(
            f"Expected 2-D vector matrix, got shape {arr.shape}",
            details={"shape": list(arr.shape)},
        )
    if arr.shape[1] != expected_dim:
        raise IndexIntegrityError(
            f"Vector dimension mismatch: expected {expected_dim}, got {arr.shape[1]}",
            details={"expected_dim": expected_dim, "actual_dim": int(arr.shape[1])},
        )
    return np.ascontiguousarray(arr, dtype=np.float32)


def _prepare_query(qvec: np.ndarray, expected_dim: int) -> np.ndarray:
    """Validate and reshape a single query vector to ``(1, dim)`` float32."""
    arr = np.asarray(qvec, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    elif arr.ndim != 2 or arr.shape[0] != 1:
        raise IndexIntegrityError(
            f"Expected query shape ({expected_dim},) or (1, {expected_dim}), got {arr.shape}",
            details={"shape": list(arr.shape)},
        )
    if arr.shape[1] != expected_dim:
        raise IndexIntegrityError(
            f"Query dim mismatch: expected {expected_dim}, got {arr.shape[1]}",
            details={"expected_dim": expected_dim, "actual_dim": int(arr.shape[1])},
        )
    return np.ascontiguousarray(arr, dtype=np.float32)


class _DualLoadDescriptor:
    """Allows ``SQ8Index.load(path)`` AND ``idx.load(path)``."""

    def __get__(
        self, instance: Optional["SQ8Index"], owner: type["SQ8Index"]
    ) -> Callable[[Path | str], "SQ8Index"]:
        def _loader(path: Path | str) -> "SQ8Index":
            target = instance if instance is not None else owner()
            return target._load_from_disk(path)

        return _loader


class SQ8Index:
    """FAISS 8-bit ScalarQuantizer index using Inner Product similarity."""

    load = _DualLoadDescriptor()

    def __init__(self, dim: int = 512) -> None:
        apply_hardware_guard()
        self.dim: int = dim
        self._index: Optional[faiss.Index] = None
        self._idmap: list[str] = []

    @property
    def ntotal(self) -> int:
        """Number of indexed vectors."""
        return int(self._index.ntotal) if self._index is not None else 0

    def build(self, vectors: np.ndarray, chunk_ids: list[str]) -> "SQ8Index":
        """Train and populate an SQ8 inner-product index from *vectors*."""
        if len(chunk_ids) == 0:
            raise IndexIntegrityError(
                "Cannot build SQ8Index with zero chunk_ids.",
                details={"ntotal": 0},
            )
        arr = _prepare_vectors(vectors, self.dim)
        if arr.shape[0] != len(chunk_ids):
            raise IndexIntegrityError(
                f"Vector count ({arr.shape[0]}) != chunk_ids length ({len(chunk_ids)})",
                details={"vectors": int(arr.shape[0]), "chunk_ids": len(chunk_ids)},
            )

        index = faiss.IndexScalarQuantizer(
            self.dim,
            faiss.ScalarQuantizer.QT_8bit,
            faiss.METRIC_INNER_PRODUCT,
        )
        index.train(arr)
        index.add(arr)

        if index.ntotal != len(chunk_ids):
            raise IndexIntegrityError(
                f"Post-add ntotal ({index.ntotal}) != chunk_ids ({len(chunk_ids)})",
                details={"expected_ntotal": len(chunk_ids), "actual_ntotal": int(index.ntotal)},
            )

        self._index = index
        self._idmap = [str(cid) for cid in chunk_ids]
        log.info("SQ8Index built: dim=%d, ntotal=%d", self.dim, index.ntotal)
        return self

    def save(self, path: Path | str) -> Path:
        """Atomically save FAISS index and companion ``.idmap.json`` to disk."""
        if self._index is None:
            raise IndexIntegrityError(
                "Cannot save uninitialized SQ8Index.",
                details={"index_path": str(path)},
            )
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        idmap_path = _idmap_path_for(target)

        tmp_faiss = target.with_name(target.name + ".tmp")
        faiss.write_index(self._index, str(tmp_faiss))
        os.replace(tmp_faiss, target)

        payload = {
            "dim": int(self.dim),
            "ntotal": int(self._index.ntotal),
            "chunk_ids": self._idmap,
        }
        tmp_json = idmap_path.with_name(idmap_path.name + ".tmp")
        tmp_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        os.replace(tmp_json, idmap_path)

        log.info("Saved SQ8Index (%d vectors) to %s", self._index.ntotal, target)
        return target

    def _load_from_disk(self, path: Path | str) -> "SQ8Index":
        """Load FAISS index + idmap from disk and verify ``ntotal == len(idmap)``."""
        target = Path(path)
        idmap_path = _idmap_path_for(target)

        if not target.exists():
            raise IndexIntegrityError(
                f"FAISS index file not found: {target}",
                details={"index_path": str(target)},
            )
        if not idmap_path.exists():
            raise IndexIntegrityError(
                f"FAISS idmap JSON not found: {idmap_path}",
                details={"index_path": str(target), "idmap_path": str(idmap_path)},
            )

        index = faiss.read_index(str(target))
        idmap_data = json.loads(idmap_path.read_text(encoding="utf-8"))
        chunk_ids = idmap_data.get("chunk_ids")
        if not isinstance(chunk_ids, list):
            raise IndexIntegrityError(
                f"Malformed idmap JSON at {idmap_path}: missing 'chunk_ids' list.",
                details={"idmap_path": str(idmap_path)},
            )

        if int(index.ntotal) != len(chunk_ids):
            raise IndexIntegrityError(
                f"Index integrity check failed: ntotal={index.ntotal} != "
                f"idmap length={len(chunk_ids)}.",
                details={
                    "index_path": str(target),
                    "expected_ntotal": len(chunk_ids),
                    "actual_ntotal": int(index.ntotal),
                },
            )

        self.dim = int(index.d)
        self._index = index
        self._idmap = [str(cid) for cid in chunk_ids]
        log.info("Loaded SQ8Index: dim=%d, ntotal=%d from %s", self.dim, index.ntotal, target)
        return self

    def search(self, qvec: np.ndarray, k: int = 8) -> list[tuple[str, float]]:
        """Return top-*k* ``(chunk_id, inner_product_score)`` pairs."""
        if self._index is None:
            raise IndexIntegrityError("SQ8Index is not built or loaded.")
        if k <= 0 or self._index.ntotal == 0:
            return []
        q = _prepare_query(qvec, self.dim)
        eff_k = min(int(k), int(self._index.ntotal))
        scores, indices = self._index.search(q, eff_k)
        results: list[tuple[str, float]] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self._idmap):
                continue
            results.append((self._idmap[int(idx)], float(score)))
        return results


class FlatIndex:
    """Exact uncompressed FAISS IndexFlatIP wrapper for ablation Runs 1–2."""

    def __init__(self, dim: int) -> None:
        apply_hardware_guard()
        self.dim: int = dim
        self._index = faiss.IndexFlatIP(dim)
        self._idmap: list[str] = []

    @property
    def ntotal(self) -> int:
        """Number of indexed vectors."""
        return int(self._index.ntotal)

    def add(self, vectors: np.ndarray, chunk_ids: list[str]) -> "FlatIndex":
        """Add float32 vectors and corresponding chunk IDs to the flat index."""
        arr = _prepare_vectors(vectors, self.dim)
        if arr.shape[0] != len(chunk_ids):
            raise IndexIntegrityError(
                f"Vector count ({arr.shape[0]}) != chunk_ids length ({len(chunk_ids)})",
                details={"vectors": int(arr.shape[0]), "chunk_ids": len(chunk_ids)},
            )
        if arr.shape[0] == 0:
            return self
        self._index.add(arr)
        self._idmap.extend(str(cid) for cid in chunk_ids)
        if self._index.ntotal != len(self._idmap):
            raise IndexIntegrityError(
                "FlatIndex ntotal mismatch after add.",
                details={
                    "expected_ntotal": len(self._idmap),
                    "actual_ntotal": int(self._index.ntotal),
                },
            )
        return self

    def search(self, qvec: np.ndarray, k: int = 8) -> list[tuple[str, float]]:
        """Return top-*k* exact ``(chunk_id, inner_product_score)`` pairs."""
        if k <= 0 or self._index.ntotal == 0:
            return []
        q = _prepare_query(qvec, self.dim)
        eff_k = min(int(k), int(self._index.ntotal))
        scores, indices = self._index.search(q, eff_k)
        results: list[tuple[str, float]] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self._idmap):
                continue
            results.append((self._idmap[int(idx)], float(score)))
        return results
