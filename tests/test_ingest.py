"""
tests.test_ingest — Validate ingested parquet corpus files.

Skips gracefully when parquet files have not yet been produced by ingest.py.

Run with:
    .venv\\Scripts\\python.exe -m pytest -q tests/test_ingest.py
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
from pydantic import ValidationError

from kavach.retrieval.schema import PARQUET_SCHEMA, ChunkRecord

_ROOT = Path(__file__).parent.parent
_PARQUET_512 = _ROOT / "data" / "processed" / "corpus_chunks.parquet"
_PARQUET_256 = _ROOT / "data" / "processed" / "corpus_chunks_w256.parquet"

_SKIP_512 = pytest.mark.skipif(
    not _PARQUET_512.exists(),
    reason="corpus_chunks.parquet not yet ingested",
)
_SKIP_256 = pytest.mark.skipif(
    not _PARQUET_256.exists(),
    reason="corpus_chunks_w256.parquet not yet ingested",
)
_SKIP_BOTH = pytest.mark.skipif(
    not (_PARQUET_512.exists() and _PARQUET_256.exists()),
    reason="One or both parquet files not yet ingested",
)


def _load_and_validate(path: Path) -> pd.DataFrame:
    """Load *path* and validate PyArrow schema + every row as :class:`ChunkRecord`."""
    import pyarrow.parquet as pq

    table = pq.read_table(path)
    assert table.schema.equals(PARQUET_SCHEMA), (
        f"PyArrow schema mismatch for {path.name}: {table.schema} != {PARQUET_SCHEMA}"
    )

    df = pd.read_parquet(path)
    errors: list[str] = []
    for idx, row in enumerate(table.to_pylist()):
        if isinstance(row.get("secondary_identifiers"), (list, tuple)):
            row["secondary_identifiers"] = list(row["secondary_identifiers"])
        try:
            ChunkRecord.model_validate(row)
        except ValidationError as exc:
            errors.append(f"Row {idx}: {exc.error_count()} error(s) — {exc.errors()[0]}")
    if errors:
        joined = "\n".join(errors[:20])
        pytest.fail(f"ChunkRecord validation failed for {path.name}:\n{joined}")
    return df


@_SKIP_512
def test_schema_valid_512() -> None:
    """Every row in corpus_chunks.parquet must pass ChunkRecord validation."""
    _load_and_validate(_PARQUET_512)


@_SKIP_256
def test_schema_valid_256() -> None:
    """Every row in corpus_chunks_w256.parquet must pass ChunkRecord validation."""
    _load_and_validate(_PARQUET_256)


@_SKIP_512
def test_token_count_within_window_512() -> None:
    """max(token_count) must not exceed chunk_window in corpus_chunks.parquet."""
    df = pd.read_parquet(_PARQUET_512)
    max_tokens = int(df["token_count"].max())
    window = int(df["chunk_window"].iloc[0])
    assert max_tokens <= window, (
        f"Token count {max_tokens} exceeds window {window} in corpus_chunks.parquet"
    )


@_SKIP_256
def test_token_count_within_window_256() -> None:
    """max(token_count) must not exceed chunk_window in corpus_chunks_w256.parquet."""
    df = pd.read_parquet(_PARQUET_256)
    max_tokens = int(df["token_count"].max())
    window = int(df["chunk_window"].iloc[0])
    assert max_tokens <= window, (
        f"Token count {max_tokens} exceeds window {window} in corpus_chunks_w256.parquet"
    )


@_SKIP_512
def test_chunk_ids_unique_512() -> None:
    """chunk_id must be unique within corpus_chunks.parquet."""
    df = pd.read_parquet(_PARQUET_512)
    dupes = df[df.duplicated("chunk_id", keep=False)]["chunk_id"].unique().tolist()
    assert not dupes, f"Duplicate chunk_ids in corpus_chunks.parquet: {dupes[:10]}"


@_SKIP_256
def test_chunk_ids_unique_256() -> None:
    """chunk_id must be unique within corpus_chunks_w256.parquet."""
    df = pd.read_parquet(_PARQUET_256)
    dupes = df[df.duplicated("chunk_id", keep=False)]["chunk_id"].unique().tolist()
    assert not dupes, f"Duplicate chunk_ids in corpus_chunks_w256.parquet: {dupes[:10]}"


@_SKIP_512
def test_cve_first_chunk_contains_cve_id() -> None:
    """For CVE records with chunk_index==1, content must contain the CVE id."""
    df = pd.read_parquet(_PARQUET_512)
    cve_first = df[(df["document_type"] == "CVE_RECORD") & (df["chunk_index"] == 1)]
    failures: list[str] = []
    for _, row in cve_first.iterrows():
        cve_id: str = row["primary_identifier"]
        content: str = row["content"]
        if cve_id not in content:
            failures.append(f"chunk_id={row['chunk_id']!r}: CVE id {cve_id!r} not in content")
    assert not failures, "CVE chunk_001 CVE-id invariant violated:\n" + "\n".join(failures[:10])


@_SKIP_512
def test_owasp_counts_512() -> None:
    """Exactly 20 OWASP entries: 10 v1.1 and 10 v2.0 (by distinct doc_id)."""
    df = pd.read_parquet(_PARQUET_512)
    owasp = df[df["document_type"] == "OWASP_LLM"]

    v11_docs = owasp[owasp["standard_version"] == "v1.1"]["doc_id"].nunique()
    v20_docs = owasp[owasp["standard_version"] == "v2.0"]["doc_id"].nunique()

    assert v11_docs == 10, f"Expected 10 OWASP v1.1 doc_ids, got {v11_docs}"
    assert v20_docs == 10, f"Expected 10 OWASP v2.0 doc_ids, got {v20_docs}"


@_SKIP_512
def test_owasp_version_tag_in_content_512() -> None:
    """Every OWASP chunk must contain '[OWASP LLM Top 10' version tag."""
    df = pd.read_parquet(_PARQUET_512)
    owasp = df[df["document_type"] == "OWASP_LLM"]
    failures: list[str] = []
    for _, row in owasp.iterrows():
        if "[OWASP LLM Top 10" not in str(row["content"]):
            failures.append(
                f"chunk_id={row['chunk_id']!r}: version tag not found in content"
            )
    assert not failures, (
        "OWASP version tag missing in chunks:\n" + "\n".join(failures[:10])
    )


def _verify_cve_chunk_001_invariant(parquet_path: Path) -> None:
    """Verify CVE chunk_001 retains cve_id, CVSS score/vector, and version operators."""
    df = pd.read_parquet(parquet_path)
    cve_rows = df[df["document_type"] == "CVE_RECORD"]
    assert not cve_rows.empty, f"No CVE_RECORD rows in {parquet_path.name}"

    failures: list[str] = []
    first_chunks = cve_rows[cve_rows["chunk_index"] == 1]
    for _, row in first_chunks.iterrows():
        cve_id = str(row["primary_identifier"])
        content = str(row["content"])
        if cve_id not in content:
            failures.append(f"{row['chunk_id']}: missing {cve_id}")
            continue

        if "\n" not in content:
            failures.append(f"{row['chunk_id']}: missing compact JSON body after header")
            continue
        _, json_part = content.split("\n", 1)
        try:
            payload = json.loads(json_part)
        except json.JSONDecodeError as exc:
            failures.append(f"{row['chunk_id']}: invalid JSON in chunk_001 ({exc})")
            continue

        raw_path = _ROOT / str(row["source_file"])
        raw_cve = json.loads(raw_path.read_text(encoding="utf-8"))["vulnerabilities"][0]["cve"]
        metrics = raw_cve.get("metrics", {}) or {}
        has_v3 = bool(metrics.get("cvssMetricV31") or metrics.get("cvssMetricV30"))
        if has_v3:
            cvss_obj = payload.get("cvss")
            if not isinstance(cvss_obj, dict) or cvss_obj.get("base_score") is None:
                failures.append(f"{row['chunk_id']}: missing CVSS base_score")
            elif str(cvss_obj["base_score"]) not in content:
                failures.append(f"{row['chunk_id']}: CVSS base_score not in content")
            if not isinstance(cvss_obj, dict) or not cvss_obj.get("vector_string"):
                failures.append(f"{row['chunk_id']}: missing CVSS vector_string")
            elif str(cvss_obj["vector_string"]) not in content:
                failures.append(f"{row['chunk_id']}: CVSS vector_string not in content")

        raw_cpes: list[dict] = []
        for cfg in raw_cve.get("configurations", []) or []:
            for node in cfg.get("nodes", []) or []:
                for match in node.get("cpeMatch", []) or []:
                    if match.get("vulnerable") is not False and match.get("criteria"):
                        raw_cpes.append(match)
        if raw_cpes:
            aff = payload.get("affected_versions")
            if not isinstance(aff, list) or not (1 <= len(aff) <= 20):
                failures.append(
                    f"{row['chunk_id']}: expected 1..20 affected_versions, got {len(aff) if isinstance(aff, list) else aff!r}"
                )
            else:
                first_raw = raw_cpes[0]
                first_out = aff[0]
                if first_out.get("criteria") != first_raw.get("criteria"):
                    failures.append(f"{row['chunk_id']}: first affected_version criteria mismatch")
                for op_key in (
                    "versionStartIncluding",
                    "versionStartExcluding",
                    "versionEndIncluding",
                    "versionEndExcluding",
                ):
                    if op_key in first_raw and str(first_raw[op_key]) != str(first_out.get(op_key, "")):
                        failures.append(f"{row['chunk_id']}: {op_key} not preserved intact")

    spill_chunks = cve_rows[cve_rows["chunk_index"] > 1]
    for _, row in spill_chunks.iterrows():
        cve_id = str(row["primary_identifier"])
        expected_parent = f"{cve_id}_chunk_001"
        if row["parent_ref"] != expected_parent:
            failures.append(
                f"{row['chunk_id']}: parent_ref={row['parent_ref']!r}, expected {expected_parent!r}"
            )
        if cve_id not in str(row["content"]):
            failures.append(f"{row['chunk_id']}: spillover missing CVE id header")

    assert not failures, "CVE chunk_001 invariant failures:\n" + "\n".join(failures[:15])


@_SKIP_512
def test_cve_chunk_001_invariant_512() -> None:
    """CVE chunk_001 invariant must hold in corpus_chunks.parquet (w512)."""
    _verify_cve_chunk_001_invariant(_PARQUET_512)


@_SKIP_256
def test_cve_chunk_001_invariant_256() -> None:
    """CVE chunk_001 invariant must hold in corpus_chunks_w256.parquet (w256)."""
    _verify_cve_chunk_001_invariant(_PARQUET_256)

