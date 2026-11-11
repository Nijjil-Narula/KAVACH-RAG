"""
tests.test_vector_store — Tests for ChunkStore, SQ8Index, and FlatIndex.

Run with:
    .venv\\Scripts\\python.exe -m pytest -q tests/test_vector_store.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from kavach.config import get_settings
from kavach.exceptions import IndexIntegrityError
from kavach.retrieval.schema import PARQUET_SCHEMA
from kavach.retrieval.vector_store import ChunkStore, FlatIndex, SQ8Index


def _make_sample_parquet(path: Path) -> None:
    """Write a small schema-valid Parquet file for ChunkStore tests."""
    rows = [
        {
            "chunk_id": "CVE-2024-3094_001",
            "doc_id": "CVE-2024-3094",
            "document_type": "CVE_RECORD",
            "source_file": "data/raw/cve_json/CVE-2024-3094.json",
            "standard_version": "nvd_2024",
            "primary_identifier": "CVE-2024-3094",
            "secondary_identifiers": ["CWE-506", "T1190"],
            "cvss_base_score": 10.0,
            "severity": "CRITICAL",
            "header_lineage": "CVE > 2024 > CVE-2024-3094",
            "parent_ref": None,
            "chunk_index": 1,
            "content": "XZ Utils liblzma backdoor CVSS 10.0.",
            "token_count": 10,
            "chunk_window": 512,
        },
        {
            "chunk_id": "CVE-2024-3094_002",
            "doc_id": "CVE-2024-3094",
            "document_type": "CVE_RECORD",
            "source_file": "data/raw/cve_json/CVE-2024-3094.json",
            "standard_version": "nvd_2024",
            "primary_identifier": "CVE-2024-3094",
            "secondary_identifiers": ["CWE-506"],
            "cvss_base_score": 10.0,
            "severity": "CRITICAL",
            "header_lineage": "CVE > 2024 > CVE-2024-3094",
            "parent_ref": None,
            "chunk_index": 2,
            "content": "Affected versions 5.6.0 and 5.6.1.",
            "token_count": 8,
            "chunk_window": 512,
        },
        {
            "chunk_id": "MITRE_T1190_001",
            "doc_id": "T1190",
            "document_type": "MITRE_TECHNIQUE",
            "source_file": "data/raw/mitre_markdown/T1190.md",
            "standard_version": "attack_v16",
            "primary_identifier": "T1190",
            "secondary_identifiers": ["TA0001", "M1050"],
            "cvss_base_score": None,
            "severity": None,
            "header_lineage": "TA0001 > T1190",
            "parent_ref": None,
            "chunk_index": 1,
            "content": "Exploit Public-Facing Application.",
            "token_count": 6,
            "chunk_window": 512,
        },
    ]
    table = pa.Table.from_pylist(rows, schema=PARQUET_SCHEMA)
    pq.write_table(table, path)


def test_chunk_store_lookups(tmp_path: Path) -> None:
    """Verify ChunkStore.get, doc_id_of, lookup_identifier, and all_chunk_ids."""
    pq_path = tmp_path / "chunks.parquet"
    _make_sample_parquet(pq_path)

    store = ChunkStore(pq_path)
    assert len(store) == 3
    assert store.all_chunk_ids() == [
        "CVE-2024-3094_001",
        "CVE-2024-3094_002",
        "MITRE_T1190_001",
    ]
    assert store.doc_id_of("CVE-2024-3094_001") == "CVE-2024-3094"

    rec = store.get("MITRE_T1190_001")
    assert rec["cvss_base_score"] is None
    assert rec["secondary_identifiers"] == ["TA0001", "M1050"]

    # Primary identifier lookup returns chunks ordered by chunk_index
    cve_hits = store.lookup_identifier("cve-2024-3094")
    assert [h["chunk_id"] for h in cve_hits] == [
        "CVE-2024-3094_001",
        "CVE-2024-3094_002",
    ]

    # Secondary identifier lookup across chunks
    t1190_hits = store.lookup_identifier("T1190")
    assert {h["chunk_id"] for h in t1190_hits} == {
        "CVE-2024-3094_001",
        "MITRE_T1190_001",
    }
    assert store.lookup_identifier("CVE-1999-9999") == []


def test_sq8_roundtrip_and_integrity_error(tmp_path: Path) -> None:
    """Verify SQ8Index save/load round-trip and ntotal vs idmap mismatch guard."""
    rng = np.random.default_rng(7)
    dim = 64
    n = 200
    vecs = rng.normal(size=(n, dim)).astype(np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    chunk_ids = [f"chunk_{i:04d}" for i in range(n)]

    sq8 = SQ8Index(dim=dim).build(vecs, chunk_ids)
    idx_path = tmp_path / "index_sq8_test.faiss"
    sq8.save(idx_path)

    assert idx_path.exists()
    assert idx_path.with_suffix(".idmap.json").exists()

    loaded_cls = SQ8Index.load(idx_path)
    loaded_inst = SQ8Index(dim=dim).load(idx_path)
    assert loaded_cls.ntotal == n
    assert loaded_inst.ntotal == n

    q = vecs[17]
    hits_orig = sq8.search(q, k=5)
    hits_loaded = loaded_cls.search(q, k=5)
    assert hits_orig == hits_loaded
    assert hits_loaded[0][0] == "chunk_0017"

    # Corrupt idmap JSON to have wrong length and assert IndexIntegrityError
    idmap_path = idx_path.with_suffix(".idmap.json")
    data = json.loads(idmap_path.read_text(encoding="utf-8"))
    data["chunk_ids"] = data["chunk_ids"][:-1]
    idmap_path.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(IndexIntegrityError):
        SQ8Index.load(idx_path)


def test_flat_index_exact_search() -> None:
    """Verify FlatIndex exact inner-product ranking."""
    dim = 16
    rng = np.random.default_rng(12)
    vecs = rng.normal(size=(50, dim)).astype(np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    cids = [f"doc_{i}" for i in range(50)]

    flat = FlatIndex(dim=dim).add(vecs, cids)
    assert flat.ntotal == 50
    hits = flat.search(vecs[5], k=3)
    assert len(hits) == 3
    assert hits[0][0] == "doc_5"
    assert hits[0][1] == pytest.approx(1.0, abs=1e-5)


def test_sq8_vs_flat_overlap_real_corpus() -> None:
    """Verify SQ8 top-8 overlap >= 0.95 against FlatIndex on real w512 embeddings."""
    settings = get_settings()
    w512_npz = settings.data_processed / "embeddings" / "w512.npz"
    if not w512_npz.exists():
        pytest.skip("w512.npz not yet built by build_index.py")

    with np.load(w512_npz, allow_pickle=False) as data:
        chunk_ids = [str(cid) for cid in data["chunk_ids"]]
        mrl512 = np.asarray(data["mrl512"], dtype=np.float32)

    dim = mrl512.shape[1]
    sq8 = SQ8Index(dim=dim).build(mrl512, chunk_ids)
    flat = FlatIndex(dim=dim).add(mrl512, chunk_ids)

    rng = np.random.default_rng(42)
    sample_count = min(32, len(chunk_ids))
    sample_indices = rng.choice(len(chunk_ids), size=sample_count, replace=False)

    overlaps: list[float] = []
    for idx in sample_indices:
        # Perturb slightly to simulate realistic query-passage similarity
        q = mrl512[idx] + rng.normal(scale=0.02, size=dim).astype(np.float32)
        q /= np.linalg.norm(q)
        top_sq8 = {cid for cid, _ in sq8.search(q, k=8)}
        top_flat = {cid for cid, _ in flat.search(q, k=8)}
        overlaps.append(len(top_sq8 & top_flat) / 8.0)

    mean_overlap = float(np.mean(overlaps))
    assert mean_overlap >= 0.95, (
        f"SQ8 top-8 overlap vs FlatIndex was {mean_overlap:.4f}, expected >= 0.95"
    )
