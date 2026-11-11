"""
kavach.retrieval.ingest — Multi-standard corpus chunking & Parquet serialization.

Parses raw NVD CVE JSON, MITRE ATT&CK Markdown, and OWASP LLM Top 10 (v1.1 / v2.0)
Markdown into standardized, schema-validated ``ChunkRecord`` rows written atomically
to Apache Parquet.

Usage::

    python -m kavach.retrieval.ingest --window 512 --overlap 64
    python -m kavach.retrieval.ingest --window 256 --overlap 32
"""
from __future__ import annotations

import os

# ── Hardware env vars before any torch/transformers/faiss import ──────────────
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("PYTHONUTF8", "1")

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).parent.parent.parent
os.environ.setdefault("HF_HOME", str(_PROJECT_ROOT / "models" / "hf_cache"))

import pyarrow as pa  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402
from langchain_text_splitters import (  # noqa: E402
    MarkdownHeaderTextSplitter,
    RecursiveJsonSplitter,
)
from transformers import AutoTokenizer, logging as hf_logging  # noqa: E402

from kavach.config import Settings, get_settings  # noqa: E402
from kavach.exceptions import ConfigError, CorpusIntegrityError, ModelLoadError  # noqa: E402
from kavach.retrieval.schema import PARQUET_SCHEMA, ChunkRecord  # noqa: E402

_FALLBACK_EMBED_MODEL = "codefuse-ai/F2LLM-0.6B"
_MAX_CHUNK001_AFFECTED_VERSIONS = 20
_VERSION_BOUND_KEYS = (
    "versionStartIncluding",
    "versionStartExcluding",
    "versionEndIncluding",
    "versionEndExcluding",
)


# ─────────────────────────────────────────────────────────────────────────────
# Tokenizer & token-budget utilities
# ─────────────────────────────────────────────────────────────────────────────


def _load_embed_tokenizer(settings: Settings) -> tuple[Any, str]:
    """Load only the bi-encoder tokenizer (no model weights) into hf_cache.

    Tries ``settings.embed_model_id`` first, then falls back to
    ``codefuse-ai/F2LLM-0.6B``.
    """
    cache_dir = settings.models_dir / "hf_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    candidates = [settings.embed_model_id]
    if _FALLBACK_EMBED_MODEL not in candidates:
        candidates.append(_FALLBACK_EMBED_MODEL)

    last_exc: Exception | None = None
    for model_id in candidates:
        try:
            tok = AutoTokenizer.from_pretrained(model_id, cache_dir=str(cache_dir))
            return tok, model_id
        except Exception as exc:
            last_exc = exc
            print(f"  [warn] Could not load tokenizer for {model_id!r}: {exc}", flush=True)

    raise ModelLoadError(
        "Failed to load bi-encoder tokenizer from primary or fallback model ID.",
        details={"candidates": candidates, "error": str(last_exc)},
    )


def _count_tokens(tokenizer: Any, text: str) -> int:
    """Count exact tokens including model special tokens without truncation."""
    prev_level = hf_logging.get_verbosity()
    hf_logging.set_verbosity_error()
    try:
        ids = tokenizer.encode(text, add_special_tokens=True, truncation=False)
    finally:
        hf_logging.set_verbosity(prev_level)
    return len(ids)


def _encode_raw_ids(tokenizer: Any, text: str) -> list[int]:
    """Encode *text* without special tokens or truncation."""
    prev_level = hf_logging.get_verbosity()
    hf_logging.set_verbosity_error()
    try:
        return list(tokenizer.encode(text, add_special_tokens=False, truncation=False))
    finally:
        hf_logging.set_verbosity(prev_level)


def _split_text_by_token_window(
    tokenizer: Any,
    prefix: str,
    body: str,
    window: int,
    overlap: int,
) -> list[str]:
    """Split *body* into chunks each starting with *prefix* and fitting <= *window* tokens.

    Guarantees every returned chunk satisfies ``_count_tokens(tokenizer, chunk) <= window``.
    """
    body_clean = body.strip()
    if not body_clean:
        candidate = prefix.strip()
        if not candidate:
            return []
        if _count_tokens(tokenizer, candidate) > window:
            raise CorpusIntegrityError(
                "Chunk prefix alone exceeds token window.",
                details={"window": window, "prefix": prefix[:120]},
            )
        return [candidate]

    full_text = f"{prefix}{body_clean}" if prefix else body_clean
    if _count_tokens(tokenizer, full_text) <= window:
        return [full_text]

    prefix_tokens = _count_tokens(tokenizer, prefix) if prefix else _count_tokens(tokenizer, "")
    # Reserve 2 tokens margin for BPE boundary merges at prefix/body join.
    body_budget = window - prefix_tokens - 2
    if body_budget < 24:
        raise CorpusIntegrityError(
            "Insufficient token budget remaining after prefix.",
            details={"window": window, "prefix_tokens": prefix_tokens},
        )

    body_ids = _encode_raw_ids(tokenizer, body_clean)
    if not body_ids:
        return [full_text]

    chunks: list[str] = []
    start = 0
    n_ids = len(body_ids)
    effective_overlap = min(overlap, max(0, body_budget // 2))

    while start < n_ids:
        end = min(n_ids, start + body_budget)
        piece = tokenizer.decode(body_ids[start:end], skip_special_tokens=True).strip()
        candidate = f"{prefix}{piece}" if prefix else piece

        # Guard against BPE decode/re-encode expansion exceeding window.
        while _count_tokens(tokenizer, candidate) > window and end > start + 1:
            excess = _count_tokens(tokenizer, candidate) - window
            step_back = max(1, excess)
            end = max(start + 1, end - step_back)
            piece = tokenizer.decode(body_ids[start:end], skip_special_tokens=True).strip()
            candidate = f"{prefix}{piece}" if prefix else piece

        if _count_tokens(tokenizer, candidate) > window:
            raise CorpusIntegrityError(
                "Unable to fit single token slice within window.",
                details={"window": window, "prefix_tokens": prefix_tokens},
            )

        if piece or not chunks:
            chunks.append(candidate)

        if end >= n_ids:
            break

        next_start = end - effective_overlap
        if next_start <= start:
            next_start = start + 1
        start = next_start

    return chunks


# ─────────────────────────────────────────────────────────────────────────────
# CVE ingestion (RecursiveJsonSplitter + chunk_001 invariant)
# ─────────────────────────────────────────────────────────────────────────────


def _extract_cvss(metrics: dict[str, Any]) -> dict[str, Any] | None:
    """Extract Primary CVSS v3.1 (or fallback v3.0) dictionary, else None."""
    for key in ("cvssMetricV31", "cvssMetricV30"):
        entries = metrics.get(key)
        if not isinstance(entries, list) or not entries:
            continue
        selected = next(
            (m for m in entries if isinstance(m, dict) and m.get("type") == "Primary"),
            entries[0],
        )
        if not isinstance(selected, dict):
            continue
        cvss_data = selected.get("cvssData", {})
        if not isinstance(cvss_data, dict):
            continue
        base_score = cvss_data.get("baseScore")
        return {
            "base_score": float(base_score) if base_score is not None else None,
            "severity": cvss_data.get("baseSeverity") or selected.get("baseSeverity"),
            "vector_string": cvss_data.get("vectorString"),
            "attack_vector": cvss_data.get("attackVector"),
        }
    return None


def _reduce_nvd_record(raw_json: dict[str, Any], source_path: Path) -> dict[str, Any]:
    """Reduce a raw NVD 2.0 JSON document to the canonical compact dictionary."""
    vulns = raw_json.get("vulnerabilities")
    if not isinstance(vulns, list) or not vulns:
        raise CorpusIntegrityError(
            f"NVD JSON missing non-empty vulnerabilities array: {source_path}",
            details={"source_file": str(source_path)},
        )
    cve = vulns[0].get("cve", {})
    cve_id = str(cve.get("id", "")).strip()
    if not cve_id:
        raise CorpusIntegrityError(
            f"NVD JSON missing cve.id: {source_path}",
            details={"source_file": str(source_path)},
        )

    published = str(cve.get("published", "")).strip()

    descriptions = cve.get("descriptions", [])
    en_parts = [
        str(d.get("value", "")).strip()
        for d in descriptions
        if isinstance(d, dict) and d.get("lang") == "en" and str(d.get("value", "")).strip()
    ]
    if en_parts:
        description_en = " ".join(en_parts)
    else:
        fallback_parts = [
            str(d.get("value", "")).strip()
            for d in descriptions
            if isinstance(d, dict) and str(d.get("value", "")).strip()
        ]
        description_en = " ".join(fallback_parts)

    cvss = _extract_cvss(cve.get("metrics", {}) or {})

    cwe_ids: list[str] = []
    seen_cwes: set[str] = set()
    for weakness in cve.get("weaknesses", []) or []:
        if not isinstance(weakness, dict):
            continue
        for desc in weakness.get("description", []) or []:
            if not isinstance(desc, dict):
                continue
            val = str(desc.get("value", "")).strip()
            if val.startswith("CWE-") and val not in seen_cwes:
                seen_cwes.add(val)
                cwe_ids.append(val)

    affected_versions: list[dict[str, str]] = []
    seen_cpe_keys: set[str] = set()
    for cfg in cve.get("configurations", []) or []:
        if not isinstance(cfg, dict):
            continue
        for node in cfg.get("nodes", []) or []:
            if not isinstance(node, dict):
                continue
            for match in node.get("cpeMatch", []) or []:
                if not isinstance(match, dict):
                    continue
                if match.get("vulnerable") is False:
                    continue
                criteria = str(match.get("criteria", "")).strip()
                if not criteria:
                    continue
                item: dict[str, str] = {"criteria": criteria}
                for key in _VERSION_BOUND_KEYS:
                    if key in match and match[key] is not None:
                        item[key] = str(match[key])
                dedup_sig = json.dumps(item, sort_keys=True)
                if dedup_sig not in seen_cpe_keys:
                    seen_cpe_keys.add(dedup_sig)
                    affected_versions.append(item)

    return {
        "cve_id": cve_id,
        "published": published,
        "description_en": description_en,
        "cvss": cvss,
        "cwe_ids": cwe_ids,
        "affected_versions": affected_versions,
    }


def _compact_json(obj: Any) -> str:
    """Serialize *obj* to compact single-line UTF-8 JSON."""
    return json.dumps(obj, separators=(",", ":"), ensure_ascii=False)


def _format_cve_chunk(header_line: str, payload: dict[str, Any]) -> str:
    """Format a CVE chunk as one-line header + compact JSON."""
    return f"{header_line}\n{_compact_json(payload)}"


def _split_string_at_word_boundary(
    tokenizer: Any,
    text: str,
    max_tokens: int,
) -> tuple[str, str]:
    """Split *text* into ``(head, tail)`` where *head* fits <= *max_tokens* cleanly."""
    if not text or max_tokens <= 0:
        return "", text
    if _count_tokens(tokenizer, text) <= max_tokens:
        return text, ""

    words = text.split()
    lo, hi = 0, len(words)
    best = 0
    while lo <= hi:
        mid = (lo + hi) // 2
        candidate = " ".join(words[:mid])
        if _count_tokens(tokenizer, candidate) <= max_tokens:
            best = mid
            lo = mid + 1
        else:
            hi = mid - 1

    if best == 0:
        # Single long token/word fallback: split by token IDs directly.
        ids = _encode_raw_ids(tokenizer, text)
        take = min(len(ids), max(1, max_tokens - 4))
        head = tokenizer.decode(ids[:take], skip_special_tokens=True).strip()
        tail = tokenizer.decode(ids[take:], skip_special_tokens=True).strip()
        return head, tail

    head = " ".join(words[:best]).strip()
    tail = " ".join(words[best:]).strip()
    return head, tail


def _split_json_with_token_budget(
    tokenizer: Any,
    header_line: str,
    payload: dict[str, Any],
    window: int,
) -> list[dict[str, Any]]:
    """Split *payload* via ``RecursiveJsonSplitter`` and enforce <= *window* tokens."""
    if _count_tokens(tokenizer, _format_cve_chunk(header_line, payload)) <= window:
        return [payload]

    char_budget = max(120, window * 3)
    for _ in range(10):
        splitter = RecursiveJsonSplitter(
            max_chunk_size=char_budget,
            min_chunk_size=max(40, char_budget // 2),
        )
        pieces = splitter.split_json(json_data=payload, convert_lists=True)
        oversized = [
            p for p in pieces
            if _count_tokens(tokenizer, _format_cve_chunk(header_line, p)) > window
        ]
        if not oversized:
            return pieces
        char_budget = max(60, int(char_budget * 0.72))

    # Deterministic fallback: emit individual sub-chunks with smallest safe splits.
    splitter = RecursiveJsonSplitter(max_chunk_size=80, min_chunk_size=40)
    pieces = splitter.split_json(json_data=payload, convert_lists=True)
    result: list[dict[str, Any]] = []
    for piece in pieces:
        if _count_tokens(tokenizer, _format_cve_chunk(header_line, piece)) <= window:
            result.append(piece)
        else:
            # Split any remaining oversized string values on token boundaries.
            cve_id = payload.get("cve_id", "")
            text_blob = _compact_json(piece)
            prefix = f"{header_line}\n"
            for sub in _split_text_by_token_window(tokenizer, "", text_blob, max(32, window - _count_tokens(tokenizer, prefix) - 16), 0):
                result.append({"cve_id": cve_id, "detail": sub})
    return result


def _chunk_cve_record(
    reduced: dict[str, Any],
    tokenizer: Any,
    window: int,
    overlap: int,
) -> list[dict[str, Any]]:
    """Produce ordered JSON payload dicts for one CVE satisfying the chunk_001 invariant."""
    cve_id = reduced["cve_id"]
    cvss = reduced["cvss"]
    score_str = f"{cvss['base_score']}" if cvss and cvss.get("base_score") is not None else "N/A"
    header_line = f"{cve_id} | CVSS {score_str}"

    affected_all: list[dict[str, str]] = reduced["affected_versions"]
    desc_full: str = reduced["description_en"]

    # Try RecursiveJsonSplitter directly first when the full record fits or splits cleanly.
    full_text = _format_cve_chunk(header_line, reduced)
    if _count_tokens(tokenizer, full_text) <= window and len(affected_all) <= _MAX_CHUNK001_AFFECTED_VERSIONS:
        return [reduced]

    # Build chunk_001 guaranteed to retain cve_id, published, cvss, cwe_ids,
    # and up to 20 affected_versions entries intact within the token window.
    base_001: dict[str, Any] = {
        "cve_id": cve_id,
        "published": reduced["published"],
        "cvss": cvss,
        "cwe_ids": reduced["cwe_ids"],
    }

    # First check how many affected_versions (<= 20) fit with full description_en.
    max_v = min(_MAX_CHUNK001_AFFECTED_VERSIONS, len(affected_all))
    chosen_versions_count = 0
    chosen_desc = desc_full

    # Binary search largest k in [0, max_v] that fits with full description_en.
    lo, hi = 0, max_v
    best_k = -1
    while lo <= hi:
        mid = (lo + hi) // 2
        probe = {
            **base_001,
            "description_en": desc_full,
            "affected_versions": affected_all[:mid],
        }
        if _count_tokens(tokenizer, _format_cve_chunk(header_line, probe)) <= window:
            best_k = mid
            lo = mid + 1
        else:
            hi = mid - 1

    desc_spill = ""
    if best_k == max_v:
        chosen_versions_count = max_v
        chosen_desc = desc_full
    else:
        # Description + <=20 versions didn't all fit together.
        # Reserve space for up to max_v affected_versions first (capped so description still fits),
        # ensuring at least 1 affected_version (when present) and a clean description prefix fit.
        desc_reserve_tokens = min(96, max(48, window // 3)) if desc_full else 0
        lo, hi = (1 if max_v > 0 else 0), max_v
        best_k = 0
        while lo <= hi:
            mid = (lo + hi) // 2
            probe = {
                **base_001,
                "description_en": "",
                "affected_versions": affected_all[:mid],
            }
            tok = _count_tokens(tokenizer, _format_cve_chunk(header_line, probe))
            if tok + desc_reserve_tokens <= window:
                best_k = mid
                lo = mid + 1
            else:
                hi = mid - 1

        # If even 1 version + reserve exceeded window, fit as many versions (>=1 if possible) as fit without reserve.
        if best_k == 0 and max_v > 0:
            probe_one = {**base_001, "description_en": "", "affected_versions": affected_all[:1]}
            if _count_tokens(tokenizer, _format_cve_chunk(header_line, probe_one)) <= window:
                best_k = 1

        chosen_versions_count = best_k
        skeleton = {
            **base_001,
            "description_en": "",
            "affected_versions": affected_all[:chosen_versions_count],
        }
        skel_tokens = _count_tokens(tokenizer, _format_cve_chunk(header_line, skeleton))
        remaining_budget = max(0, window - skel_tokens - 6)
        chosen_desc, desc_spill = _split_string_at_word_boundary(
            tokenizer, desc_full, remaining_budget
        )
        # Verify exact combined token count after adding chosen_desc.
        while _count_tokens(
            tokenizer,
            _format_cve_chunk(
                header_line,
                {
                    **base_001,
                    "description_en": chosen_desc,
                    "affected_versions": affected_all[:chosen_versions_count],
                },
            ),
        ) > window and chosen_desc:
            remaining_budget = max(0, remaining_budget - 8)
            chosen_desc, desc_spill = _split_string_at_word_boundary(
                tokenizer, desc_full, remaining_budget
            )

    chunk_001_obj = {
        **base_001,
        "description_en": chosen_desc,
        "affected_versions": affected_all[:chosen_versions_count],
    }

    # Final verification of chunk_001 token budget.
    if _count_tokens(tokenizer, _format_cve_chunk(header_line, chunk_001_obj)) > window:
        raise CorpusIntegrityError(
            f"CVE {cve_id} chunk_001 exceeds token window {window}.",
            details={"cve_id": cve_id, "window": window},
        )

    chunks_payload: list[dict[str, Any]] = [chunk_001_obj]
    versions_spill = affected_all[chosen_versions_count:]

    if not desc_spill and not versions_spill:
        return chunks_payload

    spillover_obj: dict[str, Any] = {"cve_id": cve_id}
    if desc_spill:
        prefix_tokens = _count_tokens(tokenizer, f"{header_line}\n{{\"cve_id\":\"{cve_id}\",\"description_continuation\":[]}}")
        part_budget = max(32, window - prefix_tokens - 16)
        desc_parts = _split_text_by_token_window(tokenizer, "", desc_spill, part_budget, overlap)
        spillover_obj["description_continuation"] = desc_parts if len(desc_parts) > 1 else desc_parts[0]
    if versions_spill:
        spillover_obj["affected_versions"] = versions_spill

    spill_pieces = _split_json_with_token_budget(tokenizer, header_line, spillover_obj, window)
    for piece in spill_pieces:
        if "cve_id" not in piece:
            piece = {"cve_id": cve_id, **piece}
        # Re-check after adding cve_id key.
        if _count_tokens(tokenizer, _format_cve_chunk(header_line, piece)) > window:
            sub_pieces = _split_json_with_token_budget(tokenizer, header_line, piece, window)
            chunks_payload.extend(sub_pieces)
        else:
            chunks_payload.append(piece)

    return chunks_payload


def _ingest_cves(
    settings: Settings,
    tokenizer: Any,
    window: int,
    overlap: int,
) -> list[ChunkRecord]:
    """Ingest all raw NVD CVE JSON files into validated ChunkRecords."""
    cve_dir = settings.data_raw / "cve_json"
    json_files = sorted(cve_dir.glob("*.json"))
    if not json_files:
        raise CorpusIntegrityError(
            f"No CVE JSON files found in {cve_dir}.",
            details={"cve_dir": str(cve_dir)},
        )

    records: list[ChunkRecord] = []
    for path in json_files:
        raw_json = json.loads(path.read_text(encoding="utf-8"))
        reduced = _reduce_nvd_record(raw_json, path)
        cve_id = reduced["cve_id"]
        cvss = reduced["cvss"]
        base_score = cvss["base_score"] if cvss else None
        severity = cvss["severity"] if cvss else None
        score_str = f"{base_score}" if base_score is not None else "N/A"
        header_line = f"{cve_id} | CVSS {score_str}"
        header_lineage = f"NVD > {cve_id} > CVSS {score_str}"
        rel_path = path.relative_to(_PROJECT_ROOT).as_posix()

        payloads = _chunk_cve_record(reduced, tokenizer, window, overlap)
        for idx, payload in enumerate(payloads, start=1):
            content = _format_cve_chunk(header_line, payload)
            tok_count = _count_tokens(tokenizer, content)
            if tok_count > window:
                raise CorpusIntegrityError(
                    f"Chunk {cve_id}_chunk_{idx:03d} exceeded window ({tok_count} > {window}).",
                    details={"chunk_id": f"{cve_id}_chunk_{idx:03d}", "token_count": tok_count, "window": window},
                )
            records.append(
                ChunkRecord(
                    chunk_id=f"{cve_id}_chunk_{idx:03d}",
                    doc_id=cve_id,
                    document_type="CVE_RECORD",
                    source_file=rel_path,
                    standard_version="NVD_5.0",
                    primary_identifier=cve_id,
                    secondary_identifiers=reduced["cwe_ids"],
                    cvss_base_score=base_score,
                    severity=severity,
                    header_lineage=header_lineage,
                    parent_ref=None if idx == 1 else f"{cve_id}_chunk_001",
                    chunk_index=idx,
                    content=content,
                    token_count=tok_count,
                    chunk_window=window,
                )
            )
    return records


# ─────────────────────────────────────────────────────────────────────────────
# MITRE ATT&CK ingestion (MarkdownHeaderTextSplitter)
# ─────────────────────────────────────────────────────────────────────────────


def _ingest_mitre(
    settings: Settings,
    tokenizer: Any,
    window: int,
    overlap: int,
    attack_version: str,
) -> list[ChunkRecord]:
    """Ingest MITRE ATT&CK Markdown files using ``MarkdownHeaderTextSplitter``."""
    mitre_dir = settings.data_raw / "mitre_markdown"
    md_files = sorted(mitre_dir.glob("*.md"))
    if not md_files:
        raise CorpusIntegrityError(
            f"No MITRE Markdown files found in {mitre_dir}.",
            details={"mitre_dir": str(mitre_dir)},
        )

    available_tech_ids = {p.stem for p in md_files}
    splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[
            ("#", "tactic"),
            ("##", "technique_id"),
            ("###", "mitigation"),
        ],
        strip_headers=True,
    )

    records: list[ChunkRecord] = []
    for path in md_files:
        tech_id = path.stem
        doc_id = f"MITRE_{tech_id}"
        rel_path = path.relative_to(_PROJECT_ROOT).as_posix()
        raw_text = path.read_text(encoding="utf-8")
        parent_tech_id = tech_id.split(".", 1)[0] if "." in tech_id else None
        sub_parent_ref = (
            f"MITRE_{parent_tech_id}_chunk_001"
            if parent_tech_id and parent_tech_id in available_tech_ids
            else None
        )

        sections = splitter.split_text(raw_text)
        if not sections:
            raise CorpusIntegrityError(
                f"MITRE file produced zero Markdown sections: {path}",
                details={"source_file": rel_path},
            )

        chunk_counter = 0
        for sec in sections:
            meta = sec.metadata
            tactic_str = str(meta.get("tactic", "Unknown Tactic")).strip()
            tech_header = str(meta.get("technique_id", tech_id)).strip()
            mit_header = str(meta.get("mitigation", "")).strip()
            body = sec.page_content.strip()

            if mit_header:
                mit_match = re.match(r"^(M\d{4})\b", mit_header)
                mit_id = mit_match.group(1) if mit_match else mit_header.split(":", 1)[0].strip()
                header_lineage = f"MITRE ATT&CK > {tactic_str} > {tech_id} > {mit_id}"
                sec_ids = [mit_id]
                prefix = f"[{header_lineage}]\n## {tech_header}\n### {mit_header}\n"
            else:
                header_lineage = f"MITRE ATT&CK > {tactic_str} > {tech_id}"
                sec_ids = []
                prefix = f"[{header_lineage}]\n## {tech_header}\n"

            sub_chunks = _split_text_by_token_window(tokenizer, prefix, body, window, overlap)
            for content in sub_chunks:
                chunk_counter += 1
                tok_count = _count_tokens(tokenizer, content)
                if tok_count > window:
                    raise CorpusIntegrityError(
                        f"Chunk {doc_id}_chunk_{chunk_counter:03d} exceeded window ({tok_count} > {window}).",
                        details={"chunk_id": f"{doc_id}_chunk_{chunk_counter:03d}", "token_count": tok_count},
                    )
                if chunk_counter == 1:
                    parent_ref = sub_parent_ref
                else:
                    parent_ref = f"{doc_id}_chunk_001"

                records.append(
                    ChunkRecord(
                        chunk_id=f"{doc_id}_chunk_{chunk_counter:03d}",
                        doc_id=doc_id,
                        document_type="MITRE_TECHNIQUE",
                        source_file=rel_path,
                        standard_version=attack_version,
                        primary_identifier=tech_id,
                        secondary_identifiers=sec_ids,
                        cvss_base_score=None,
                        severity=None,
                        header_lineage=header_lineage,
                        parent_ref=parent_ref,
                        chunk_index=chunk_counter,
                        content=content,
                        token_count=tok_count,
                        chunk_window=window,
                    )
                )
    return records


# ─────────────────────────────────────────────────────────────────────────────
# OWASP LLM Top 10 ingestion (conflict-aware Markdown parse)
# ─────────────────────────────────────────────────────────────────────────────


def _ingest_owasp(
    settings: Settings,
    tokenizer: Any,
    window: int,
    overlap: int,
) -> list[ChunkRecord]:
    """Ingest OWASP LLM Top 10 v1.1 and v2.0 Markdown files with explicit version tags."""
    owasp_root = settings.data_raw / "owasp_markdown"
    splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[
            ("#", "standard"),
            ("##", "entry"),
            ("###", "section"),
        ],
        strip_headers=True,
    )

    records: list[ChunkRecord] = []
    for version in ("v1.1", "v2.0"):
        ver_dir = owasp_root / version
        md_files = sorted(ver_dir.glob("*.md"))
        if not md_files:
            raise CorpusIntegrityError(
                f"No OWASP {version} Markdown files found in {ver_dir}.",
                details={"owasp_dir": str(ver_dir)},
            )

        for path in md_files:
            llm_id = path.stem
            doc_id = f"OWASP_{version}_{llm_id}"
            rel_path = path.relative_to(_PROJECT_ROOT).as_posix()
            raw_text = path.read_text(encoding="utf-8")
            cleaned_text = re.sub(
                r"<!--\s*source_provenance:.*?-->\s*",
                "",
                raw_text,
                flags=re.IGNORECASE,
            )

            sections = splitter.split_text(cleaned_text)
            if not sections:
                raise CorpusIntegrityError(
                    f"OWASP file produced zero sections: {path}",
                    details={"source_file": rel_path},
                )

            chunk_counter = 0
            for sec in sections:
                meta = sec.metadata
                entry_str = str(meta.get("entry", llm_id)).strip()
                section_str = str(meta.get("section", "")).strip()
                body = sec.page_content.strip()
                if not body and len(sections) > 1:
                    continue

                header_lineage = (
                    f"OWASP LLM Top 10 {version} > {entry_str}"
                    + (f" > {section_str}" if section_str else "")
                )
                version_tag = f"[OWASP LLM Top 10 {version} | {entry_str}]"
                prefix = f"{version_tag}\n" + (f"### {section_str}\n" if section_str else "")

                sub_chunks = _split_text_by_token_window(tokenizer, prefix, body, window, overlap)
                for content in sub_chunks:
                    chunk_counter += 1
                    tok_count = _count_tokens(tokenizer, content)
                    if tok_count > window:
                        raise CorpusIntegrityError(
                            f"Chunk {doc_id}_chunk_{chunk_counter:03d} exceeded window ({tok_count} > {window}).",
                            details={"chunk_id": f"{doc_id}_chunk_{chunk_counter:03d}", "token_count": tok_count},
                        )
                    records.append(
                        ChunkRecord(
                            chunk_id=f"{doc_id}_chunk_{chunk_counter:03d}",
                            doc_id=doc_id,
                            document_type="OWASP_LLM",
                            source_file=rel_path,
                            standard_version=version,
                            primary_identifier=llm_id,
                            secondary_identifiers=[version],
                            cvss_base_score=None,
                            severity=None,
                            header_lineage=header_lineage,
                            parent_ref=None if chunk_counter == 1 else f"{doc_id}_chunk_001",
                            chunk_index=chunk_counter,
                            content=content,
                            token_count=tok_count,
                            chunk_window=window,
                        )
                    )
    return records


# ─────────────────────────────────────────────────────────────────────────────
# Serialization & manifest
# ─────────────────────────────────────────────────────────────────────────────


def _atomic_write_parquet(dest: Path, records: list[ChunkRecord]) -> None:
    """Serialize *records* to *dest* atomically via ``PARQUET_SCHEMA``."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    rows = [r.model_dump() for r in records]
    table = pa.Table.from_pylist(rows, schema=PARQUET_SCHEMA)
    tmp = dest.with_suffix(".parquet.tmp")
    pq.write_table(table, tmp, compression="snappy")
    os.replace(str(tmp), str(dest))


def _atomic_write_json(dest: Path, data: dict[str, Any]) -> None:
    """Write *data* as pretty UTF-8 JSON atomically."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(str(tmp), str(dest))


def _update_manifest(
    manifest_path: Path,
    window: int,
    overlap: int,
    output_path: Path,
    tokenizer_id: str,
    records: list[ChunkRecord],
    provenance: dict[str, Any],
) -> dict[str, Any]:
    """Build or update ``reports/ingestion_manifest.json`` atomically."""
    existing: dict[str, Any] = {}
    if manifest_path.exists():
        try:
            existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            existing = {}

    runs: dict[str, Any] = existing.get("runs", {}) if isinstance(existing.get("runs"), dict) else {}

    per_source_counts: dict[str, int] = {}
    per_source_docs: dict[str, set[str]] = {}
    for rec in records:
        per_source_counts[rec.document_type] = per_source_counts.get(rec.document_type, 0) + 1
        per_source_docs.setdefault(rec.document_type, set()).add(rec.doc_id)

    token_counts = [rec.token_count for rec in records]
    chunks_over_window = sum(1 for t in token_counts if t > window)
    avg_tokens = round(sum(token_counts) / len(token_counts), 2) if token_counts else 0.0
    max_tokens = max(token_counts) if token_counts else 0

    prov_sources = provenance.get("sources", {})
    source_prov_summary: dict[str, Any] = {}
    for name, info in prov_sources.items():
        if isinstance(info, dict):
            entry: dict[str, Any] = {
                "url": info.get("url"),
                "sha256": info.get("sha256"),
                "fetched_at": info.get("fetched_at"),
                "record_count": info.get("record_count"),
            }
            if "source_provenance" in info:
                entry["source_provenance"] = info["source_provenance"]
            source_prov_summary[name] = entry

    run_summary: dict[str, Any] = {
        "window": window,
        "overlap": overlap,
        "output_file": output_path.relative_to(_PROJECT_ROOT).as_posix(),
        "tokenizer_model_id": tokenizer_id,
        "total_chunks": len(records),
        "total_docs": len({r.doc_id for r in records}),
        "avg_tokens": avg_tokens,
        "max_tokens": max_tokens,
        "chunks_over_window": chunks_over_window,
        "per_source_chunk_counts": per_source_counts,
        "per_source_doc_counts": {k: len(v) for k, v in per_source_docs.items()},
        "generated_at": datetime.now(tz=timezone.utc).isoformat(),
    }
    runs[f"w{window}"] = run_summary

    manifest = {
        **run_summary,
        "provenance": source_prov_summary,
        "runs": runs,
    }
    _atomic_write_json(manifest_path, manifest)
    return manifest


# ─────────────────────────────────────────────────────────────────────────────
# Orchestration & CLI
# ─────────────────────────────────────────────────────────────────────────────


def ingest_corpus(window: int = 512, overlap: int = 64) -> Path:
    """Run full multi-standard ingestion for *window* and *overlap*.

    Args:
        window: Maximum tokens allowed per chunk (e.g. 512 or 256).
        overlap: Token overlap between consecutive split chunks.

    Returns:
        Path to the written Parquet file.

    Raises:
        ConfigError: If *window* or *overlap* parameters are invalid.
        CorpusIntegrityError: If raw files, ``PROVENANCE.json``, or chunk
            constraints fail validation.
    """
    if window < 64:
        raise ConfigError(f"window must be >= 64, got {window}.")
    if overlap < 0 or overlap >= window:
        raise ConfigError(f"overlap must satisfy 0 <= overlap < window, got overlap={overlap}, window={window}.")

    settings = get_settings()
    prov_path = settings.data_raw / "PROVENANCE.json"
    if not prov_path.exists():
        raise CorpusIntegrityError(
            "PROVENANCE.json not found — run `python -m kavach.retrieval.acquire` first.",
            details={"expected_path": str(prov_path)},
        )
    provenance: dict[str, Any] = json.loads(prov_path.read_text(encoding="utf-8"))
    attack_version = (
        provenance.get("sources", {})
        .get("mitre_attack", {})
        .get("attack_version", "attack_v16")
    )

    print(f"[ingest] Loading bi-encoder tokenizer ({settings.embed_model_id}) …", flush=True)
    tokenizer, resolved_tok_id = _load_embed_tokenizer(settings)
    print(f"[ingest] Tokenizer ready: {resolved_tok_id} (window={window}, overlap={overlap})", flush=True)

    cve_records = _ingest_cves(settings, tokenizer, window, overlap)
    print(f"[ingest] CVE_RECORD: {len(cve_records)} chunks", flush=True)

    mitre_records = _ingest_mitre(settings, tokenizer, window, overlap, attack_version)
    print(f"[ingest] MITRE_TECHNIQUE: {len(mitre_records)} chunks", flush=True)

    owasp_records = _ingest_owasp(settings, tokenizer, window, overlap)
    print(f"[ingest] OWASP_LLM: {len(owasp_records)} chunks", flush=True)

    all_records = cve_records + mitre_records + owasp_records
    seen_ids: set[str] = set()
    for rec in all_records:
        if rec.chunk_id in seen_ids:
            raise CorpusIntegrityError(
                f"Duplicate chunk_id generated: {rec.chunk_id}",
                details={"chunk_id": rec.chunk_id},
            )
        seen_ids.add(rec.chunk_id)
        if rec.token_count > window:
            raise CorpusIntegrityError(
                f"Chunk {rec.chunk_id} exceeds window ({rec.token_count} > {window}).",
                details={"chunk_id": rec.chunk_id, "token_count": rec.token_count, "window": window},
            )

    out_name = "corpus_chunks.parquet" if window == 512 else f"corpus_chunks_w{window}.parquet"
    output_path = settings.data_processed / out_name
    _atomic_write_parquet(output_path, all_records)
    print(f"[ingest] Wrote {len(all_records)} chunks to {output_path}", flush=True)

    manifest_path = settings.reports_dir / "ingestion_manifest.json"
    manifest = _update_manifest(
        manifest_path,
        window=window,
        overlap=overlap,
        output_path=output_path,
        tokenizer_id=resolved_tok_id,
        records=all_records,
        provenance=provenance,
    )
    print(
        f"[ingest] Manifest updated ({manifest_path}): "
        f"avg_tokens={manifest['avg_tokens']}, max_tokens={manifest['max_tokens']}, "
        f"chunks_over_window={manifest['chunks_over_window']}",
        flush=True,
    )
    return output_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ingest raw KAVACH-RAG cybersecurity sources into chunked Parquet."
    )
    parser.add_argument(
        "--window",
        type=int,
        default=512,
        help="Maximum tokens per chunk (default: 512).",
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=None,
        help="Token overlap between split chunks (default: 64 for w512, 32 for w256).",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entry point for corpus ingestion."""
    args = _parse_args()
    overlap = args.overlap if args.overlap is not None else (64 if args.window == 512 else max(16, args.window // 8))
    ingest_corpus(window=args.window, overlap=overlap)


if __name__ == "__main__":
    main()
