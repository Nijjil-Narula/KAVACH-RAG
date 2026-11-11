"""
kavach.retrieval.golden — Deterministic 150-question golden dataset builder & validator.

Runnable as:
    python -m kavach.retrieval.golden --build
    python -m kavach.retrieval.golden --validate

Reads ``data/processed/corpus_chunks.parquet`` and ``data/golden/cve_attack_map.yaml``
and assembles every ground-truth answer programmatically from fields extracted
from the ingested corpus chunks. Never fabricates CVE, MITRE, or OWASP facts.
"""
from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import json
import logging
import random
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from pydantic import ValidationError

from kavach.config import Settings, get_settings
from kavach.exceptions import GoldenDatasetValidationError
from kavach.retrieval.router import (
    ROUTE_FAST_PATH,
    ROUTE_REFUSAL,
    ROUTE_SEMANTIC,
    normalize_query,
)
from kavach.retrieval.schema import GoldenItem, QuestionType

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("kavach.golden")

SEED = 1337

_EXPECTED_COUNTS: dict[QuestionType, int] = {
    QuestionType.FACTUAL: 60,
    QuestionType.MULTI_HOP: 45,
    QuestionType.CONFLICT_TRAP: 15,
    QuestionType.OUT_OF_SCOPE: 10,
    QuestionType.MALFORMED: 10,
    QuestionType.CONTEXT_NOISE: 10,
}

_CONFLICT_SPREAD: list[tuple[str, int]] = [
    ("LLM03", 5),
    ("LLM05", 5),
    ("LLM02", 2),
    ("LLM04", 1),
    ("LLM06", 1),
    ("LLM07", 1),
]

_OUT_OF_SCOPE_SPECS: list[tuple[str, str]] = [
    ("How do I bake a sourdough loaf?", ROUTE_REFUSAL),
    ("Who won the 2022 FIFA World Cup final?", ROUTE_REFUSAL),
    ("What is the best way to train for a marathon in six weeks?", ROUTE_REFUSAL),
    ("Recommend a good sci-fi novel for a long flight.", ROUTE_REFUSAL),
    ("What's the weather forecast for Delhi this weekend?", ROUTE_REFUSAL),
    ("How do I fix a leaking kitchen faucet?", ROUTE_REFUSAL),
    ("Explain how compound interest works on a savings account.", ROUTE_REFUSAL),
    ("Translate 'good morning' into Japanese and French.", ROUTE_REFUSAL),
    (
        "What are the internal firewall rules of Acme Corp's production network?",
        ROUTE_SEMANTIC,
    ),
    ("What is the password policy of my company's HR portal?", ROUTE_SEMANTIC),
]

_AV_CODE_MAP: dict[str, str] = {
    "N": "NETWORK",
    "A": "ADJACENT_NETWORK",
    "L": "LOCAL",
    "P": "PHYSICAL",
}

_AV_READABLE_MAP: dict[str, str] = {
    "NETWORK": "Network (AV:N)",
    "ADJACENT_NETWORK": "Adjacent Network (AV:A)",
    "LOCAL": "Local (AV:L)",
    "PHYSICAL": "Physical (AV:P)",
}

_RE_CVSS31_VECTOR = re.compile(
    r"CVSS:3\.1/AV:[NALP]/AC:[LH]/PR:[NLH]/UI:[NR]/S:[UC]/C:[NLH]/I:[NLH]/A:[NLH]"
)
_RE_CVSS3X_VECTOR = re.compile(
    r"CVSS:3\.[01]/AV:[NALP]/AC:[LH]/PR:[NLH]/UI:[NR]/S:[UC]/C:[NLH]/I:[NLH]/A:[NLH]"
)
_RE_ATTACK_VECTOR_KV = re.compile(
    r"""(?i)(?:"attackVector"|attack\s*vector)\s*[:=]\s*"?(NETWORK|ADJACENT_NETWORK|ADJACENT|LOCAL|PHYSICAL)"?"""
)
_RE_CWE = re.compile(r"\bCWE-\d+\b")
_RE_MITIGATION_ID = re.compile(r"^M\d{4}$")
_RE_VERSION_KEYED = re.compile(
    r"""(?i)(versionEndExcluding|versionEndIncluding|versionStartIncluding|versionStartExcluding)["']?\s*[:=]\s*["']?([0-9][0-9A-Za-z._-]*)"""
)
_RE_VERSION_OP = re.compile(
    r"(<=|>=|<|>)\s*([0-9]+\.[0-9][0-9A-Za-z._-]*)"
)
_RE_VERSION_PROSE = re.compile(
    r"(?i)\b(prior to|before|through|up to|earlier than)\s+([0-9]+\.[0-9][0-9A-Za-z._-]*)"
)
_RE_CPE_PRODUCT = re.compile(r"cpe:2\.3:[aho]:([^:]+):([^:]+):")
_RE_EXACT_ID_ONLY = re.compile(
    r"^(CVE-\d{4}-\d{4,7}|CWE-\d{1,5}|T\d{4}(\.\d{3})?|TA\d{4}|M\d{4})$"
)


# ─────────────────────────────────────────────────────────────────────────────
# Extracted corpus representation
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class ChunkRow:
    """Typed wrapper over one row of corpus_chunks.parquet."""

    chunk_id: str
    doc_id: str
    document_type: str
    source_file: str
    standard_version: str
    primary_identifier: str
    secondary_identifiers: list[str]
    cvss_base_score: float | None
    severity: str | None
    header_lineage: str
    parent_ref: str | None
    chunk_index: int
    content: str


@dataclass
class CveInfo:
    """Fields extracted strictly from CVE_RECORD chunks in the corpus."""

    cve_id: str
    chunks: list[ChunkRow]
    first_chunk: ChunkRow
    summary: str
    cvss_score: float | None
    cvss_score_str: str | None
    severity: str | None
    cvss_v31_vector: str | None
    cvss_v31_chunk_ids: list[str]
    attack_vector_code: str | None
    attack_vector_label: str | None
    attack_vector_chunk_ids: list[str]
    cwe_ids: list[str]
    cwe_chunk_map: dict[str, list[str]]
    version_boundary: str | None
    version_token: str | None
    version_product: str | None
    version_chunk_ids: list[str]


@dataclass
class MitreMitigationInfo:
    """Fields extracted strictly from MITRE mitigation H3 chunks."""

    m_id: str
    tech_id: str
    name: str
    description: str
    chunk_id: str
    doc_id: str


@dataclass
class MitreTechInfo:
    """Fields extracted strictly from MITRE_TECHNIQUE chunks."""

    tech_id: str
    doc_id: str
    tactic_id: str | None
    tactic_name: str
    tech_name: str
    summary: str
    overview_chunk: ChunkRow
    chunks: list[ChunkRow]
    mitigations: list[MitreMitigationInfo]


@dataclass
class OwaspInfo:
    """Fields extracted strictly from OWASP_LLM chunks."""

    llm_id: str
    version: str
    doc_id: str
    title: str
    definition: str
    overview_chunk: ChunkRow


# ─────────────────────────────────────────────────────────────────────────────
# Text extraction helpers (purely from ingested corpus content)
# ─────────────────────────────────────────────────────────────────────────────


def _clean_markdown_line(text: str) -> str:
    """Strip markdown heading markers, HTML comments, and OWASP version tags."""
    cleaned = re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL)
    cleaned = re.sub(r"\[OWASP LLM Top 10[^\]]*\]", "", cleaned)
    cleaned = re.sub(r"^#+\s*", "", cleaned.strip())
    cleaned = re.sub(r"\*{1,2}([^*]+)\*{1,2}", r"\1", cleaned)
    cleaned = " ".join(cleaned.split())
    return cleaned.strip()


def _extract_first_sentences(text: str, max_chars: int = 320) -> str:
    """Extract readable summary text from prose or JSON-ish chunk content."""
    # Check ingest.py compact CVE JSON description_en field first
    desc_en_match = re.search(
        r'"description_en"\s*:\s*"((?:[^"\\]|\\.)+)"',
        text,
    )
    if desc_en_match:
        raw = desc_en_match.group(1).replace('\\"', '"').replace("\\n", " ")
        raw = " ".join(raw.split())
        if len(raw) > max_chars:
            cut = raw[:max_chars].rsplit(" ", 1)[0]
            return cut.rstrip(",;:") + "."
        return raw

    # Fallback: check raw NVD English description array field
    desc_match = re.search(
        r'"lang"\s*:\s*"en"\s*,\s*"value"\s*:\s*"((?:[^"\\]|\\.)+)"',
        text,
    )
    if desc_match:
        raw = desc_match.group(1).replace('\\"', '"').replace("\\n", " ")
        raw = " ".join(raw.split())
        if len(raw) > max_chars:
            cut = raw[:max_chars].rsplit(" ", 1)[0]
            return cut.rstrip(",;:") + "."
        return raw

    desc_label_match = re.search(
        r"(?im)^(?:Description|Summary)\s*:\s*(.+)$", text
    )
    if desc_label_match:
        raw = " ".join(desc_label_match.group(1).split())
        if len(raw) > max_chars:
            cut = raw[:max_chars].rsplit(" ", 1)[0]
            return cut.rstrip(",;:") + "."
        return raw

    lines: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith(("<!--", "#", "---", "{", "}", "[", "]", "Platforms:")):
            continue
        if line.startswith("[OWASP LLM Top 10"):
            continue
        cleaned = _clean_markdown_line(line)
        if len(cleaned) < 20:
            continue
        lines.append(cleaned)
        if sum(len(s) for s in lines) >= max_chars:
            break

    combined = " ".join(lines)
    if not combined:
        combined = _clean_markdown_line(text)
    if len(combined) > max_chars:
        cut = combined[:max_chars].rsplit(" ", 1)[0]
        return cut.rstrip(",;:") + "."
    return combined


def _format_score_in_text(score: float, text: str) -> str:
    """Return string representation of CVSS score matching corpus text if present."""
    candidates = [f"{score:.1f}", str(score), f"{score:g}"]
    for cand in candidates:
        if cand in text:
            return cand
    return f"{score:.1f}"


def _chunks_with_token(
    chunks: list[ChunkRow],
    token: str,
    fallback_first: ChunkRow,
) -> list[str]:
    """Return chunk_ids containing *token* (plus fallback_first if needed)."""
    hits = [c.chunk_id for c in chunks if token.lower() in c.content.lower()]
    if not hits:
        return [fallback_first.chunk_id]
    first_hit = hits[0]
    if fallback_first.chunk_id != first_hit and fallback_first.primary_identifier.lower() not in chunks[
        next(i for i, c in enumerate(chunks) if c.chunk_id == first_hit)
    ].content.lower():
        return [fallback_first.chunk_id, first_hit]
    return [first_hit]


def _verify_entities(
    entities: list[str],
    chunk_ids: list[str],
    chunk_by_id: dict[str, ChunkRow],
) -> list[str]:
    """Keep only entities that genuinely appear case-insensitively in expected chunks."""
    concat = "\n".join(
        chunk_by_id[cid].content for cid in chunk_ids if cid in chunk_by_id
    ).lower()
    verified: list[str] = []
    seen: set[str] = set()
    for ent in entities:
        ent_clean = ent.strip()
        if not ent_clean:
            continue
        key = ent_clean.lower()
        if key in seen:
            continue
        if key in concat:
            seen.add(key)
            verified.append(ent_clean)
    return verified


def _extract_cve_info(cve_id: str, chunks: list[ChunkRow]) -> CveInfo:
    """Extract structured facts for one CVE from its corpus chunks."""
    first = chunks[0]
    full_text = "\n".join(c.content for c in chunks)

    # CVSS score & severity from structured column
    score: float | None = None
    severity: str | None = None
    for c in chunks:
        if c.cvss_base_score is not None:
            score = float(c.cvss_base_score)
        if c.severity:
            severity = str(c.severity).strip()
        if score is not None and severity is not None:
            break

    score_str = _format_score_in_text(score, full_text) if score is not None else None

    # CVSS v3.1 vector string
    v31_vector: str | None = None
    v31_chunk_ids: list[str] = []
    for c in chunks:
        m31 = _RE_CVSS31_VECTOR.search(c.content)
        if m31:
            v31_vector = m31.group(0)
            v31_chunk_ids = [c.chunk_id]
            if c.chunk_id != first.chunk_id and cve_id.lower() not in c.content.lower():
                v31_chunk_ids = [first.chunk_id, c.chunk_id]
            break

    # Attack vector from CVSS vector or explicit attackVector field
    av_code: str | None = None
    av_label: str | None = None
    av_chunk_ids: list[str] = []
    for c in chunks:
        m3x = _RE_CVSS3X_VECTOR.search(c.content)
        if m3x:
            vec = m3x.group(0)
            av_m = re.search(r"/AV:([NALP])/", vec)
            if av_m:
                av_code = av_m.group(1)
                av_label = _AV_CODE_MAP.get(av_code)
                av_chunk_ids = [c.chunk_id]
                if c.chunk_id != first.chunk_id and cve_id.lower() not in c.content.lower():
                    av_chunk_ids = [first.chunk_id, c.chunk_id]
                break
        mkv = _RE_ATTACK_VECTOR_KV.search(c.content)
        if mkv:
            raw_av = mkv.group(1).upper()
            av_label = "ADJACENT_NETWORK" if raw_av == "ADJACENT" else raw_av
            av_code = av_label[0]
            av_chunk_ids = [c.chunk_id]
            if c.chunk_id != first.chunk_id and cve_id.lower() not in c.content.lower():
                av_chunk_ids = [first.chunk_id, c.chunk_id]
            break

    # CWE IDs present in chunk text (preferring secondary_identifiers order)
    cwe_candidates: list[str] = []
    for c in chunks:
        for sid in c.secondary_identifiers:
            if _RE_CWE.fullmatch(sid) and sid not in cwe_candidates:
                cwe_candidates.append(sid)
    for found in _RE_CWE.findall(full_text):
        if found not in cwe_candidates:
            cwe_candidates.append(found)

    cwe_chunk_map: dict[str, list[str]] = {}
    valid_cwes: list[str] = []
    for cwe in cwe_candidates:
        matching_chunks = [c.chunk_id for c in chunks if cwe.lower() in c.content.lower()]
        if matching_chunks:
            cid = matching_chunks[0]
            chunk_list = [cid]
            target_chunk = next(c for c in chunks if c.chunk_id == cid)
            if cid != first.chunk_id and cve_id.lower() not in target_chunk.content.lower():
                chunk_list = [first.chunk_id, cid]
            cwe_chunk_map[cwe] = chunk_list
            valid_cwes.append(cwe)

    # Affected-version boundary extraction
    version_boundary: str | None = None
    version_token: str | None = None
    version_product: str | None = None
    version_chunk_ids: list[str] = []

    cpe_m = _RE_CPE_PRODUCT.search(full_text)
    if cpe_m:
        version_product = cpe_m.group(2).replace("_", " ").strip()

    for c in chunks:
        mk = _RE_VERSION_KEYED.search(c.content)
        if mk:
            key_name, ver = mk.group(1), mk.group(2)
            op_map = {
                "versionEndExcluding": "<",
                "versionEndIncluding": "<=",
                "versionStartIncluding": ">=",
                "versionStartExcluding": ">",
            }
            op = op_map.get(key_name, key_name)
            version_boundary = f"{key_name}: {ver} ({op} {ver})"
            version_token = ver
            version_chunk_ids = [c.chunk_id]
            if c.chunk_id != first.chunk_id and cve_id.lower() not in c.content.lower():
                version_chunk_ids = [first.chunk_id, c.chunk_id]
            break

        mo = _RE_VERSION_OP.search(c.content)
        if mo:
            op, ver = mo.group(1), mo.group(2)
            version_boundary = f"{op} {ver}"
            version_token = ver
            version_chunk_ids = [c.chunk_id]
            if c.chunk_id != first.chunk_id and cve_id.lower() not in c.content.lower():
                version_chunk_ids = [first.chunk_id, c.chunk_id]
            break

        mp = _RE_VERSION_PROSE.search(c.content)
        if mp:
            phrase, ver = mp.group(1), mp.group(2)
            version_boundary = f"{phrase} {ver}"
            version_token = ver
            version_chunk_ids = [c.chunk_id]
            if c.chunk_id != first.chunk_id and cve_id.lower() not in c.content.lower():
                version_chunk_ids = [first.chunk_id, c.chunk_id]
            break

    summary = _extract_first_sentences(first.content, max_chars=260)

    return CveInfo(
        cve_id=cve_id,
        chunks=chunks,
        first_chunk=first,
        summary=summary,
        cvss_score=score,
        cvss_score_str=score_str,
        severity=severity,
        cvss_v31_vector=v31_vector,
        cvss_v31_chunk_ids=v31_chunk_ids,
        attack_vector_code=av_code,
        attack_vector_label=av_label,
        attack_vector_chunk_ids=av_chunk_ids,
        cwe_ids=valid_cwes,
        cwe_chunk_map=cwe_chunk_map,
        version_boundary=version_boundary,
        version_token=version_token,
        version_product=version_product,
        version_chunk_ids=version_chunk_ids,
    )


def _extract_mitre_info(doc_id: str, chunks: list[ChunkRow]) -> MitreTechInfo:
    """Extract technique and mitigation fields from MITRE_TECHNIQUE chunks."""
    first = chunks[0]
    tech_id = first.primary_identifier

    tactic_id: str | None = None
    tactic_name = "Enterprise ATT&CK"
    tech_name = tech_id

    for c in chunks:
        m_tac = re.search(r"#\s*(TA\d{4})\s*:\s*([^\n]+)", c.content)
        if m_tac:
            tactic_id = m_tac.group(1).strip()
            tactic_name = _clean_markdown_line(m_tac.group(2))
        elif not tactic_id:
            m_tac_lin = re.search(r"(TA\d{4})\s*:\s*([^>|;\n]+)", c.header_lineage)
            if m_tac_lin:
                tactic_id = m_tac_lin.group(1).strip()
                tactic_name = m_tac_lin.group(2).strip()

        m_tech = re.search(
            rf"##\s*{re.escape(tech_id)}\s*:\s*([^\n]+)", c.content
        )
        if m_tech:
            tech_name = _clean_markdown_line(m_tech.group(1))
        elif tech_name == tech_id:
            m_tech_lin = re.search(
                rf"{re.escape(tech_id)}\s*:\s*([^>|;\n]+)", c.header_lineage
            )
            if m_tech_lin:
                tech_name = m_tech_lin.group(1).strip()

        if tactic_id and tech_name != tech_id:
            break

    # Prefer chunk containing tech_id and tech_name as overview chunk
    overview = first
    for c in chunks:
        if tech_id.lower() in c.content.lower() and tech_name.lower() in c.content.lower():
            overview = c
            break

    summary = _extract_first_sentences(overview.content, max_chars=260)

    # Extract mitigations from chunks whose secondary_identifiers contain M-IDs
    mitigations: list[MitreMitigationInfo] = []
    seen_mids: set[str] = set()
    for c in chunks:
        m_ids = [sid for sid in c.secondary_identifiers if _RE_MITIGATION_ID.fullmatch(sid)]
        for m_id in m_ids:
            if m_id in seen_mids:
                continue
            m_hdr = re.search(
                rf"###\s*{re.escape(m_id)}\s*:\s*([^\n]+)", c.content
            )
            if m_hdr:
                mit_name = _clean_markdown_line(m_hdr.group(1))
                # Extract text immediately following ### M-ID header
                after = c.content[m_hdr.end():]
                next_hdr = re.search(r"\n#{1,3}\s+", after)
                section_body = after[:next_hdr.start()] if next_hdr else after
                mit_desc = _extract_first_sentences(section_body, max_chars=240)
            else:
                m_lin = re.search(
                    rf"{re.escape(m_id)}\s*:\s*([^>|;\n]+)", c.header_lineage
                )
                mit_name = m_lin.group(1).strip() if m_lin else m_id
                mit_desc = _extract_first_sentences(c.content, max_chars=240)

            seen_mids.add(m_id)
            mitigations.append(
                MitreMitigationInfo(
                    m_id=m_id,
                    tech_id=tech_id,
                    name=mit_name,
                    description=mit_desc or mit_name,
                    chunk_id=c.chunk_id,
                    doc_id=doc_id,
                )
            )

    return MitreTechInfo(
        tech_id=tech_id,
        doc_id=doc_id,
        tactic_id=tactic_id,
        tactic_name=tactic_name,
        tech_name=tech_name,
        summary=summary,
        overview_chunk=overview,
        chunks=chunks,
        mitigations=mitigations,
    )


def _extract_owasp_info(doc_id: str, chunks: list[ChunkRow]) -> OwaspInfo:
    """Extract title and definition from OWASP_LLM chunks."""
    first = chunks[0]
    llm_id = first.primary_identifier
    version = first.standard_version

    title = llm_id
    for c in chunks:
        m_tag = re.search(
            rf"\[OWASP[^\]]*\|\s*{re.escape(llm_id)}\s*[:\-–—]\s*([^\]]+)\]",
            c.content,
        )
        if m_tag:
            title = _clean_markdown_line(m_tag.group(1))
            break
        m_title = re.search(
            rf"#+\s*{re.escape(llm_id)}\s*[:\-–—]\s*([^\n]+)", c.content
        )
        if m_title:
            title = _clean_markdown_line(m_title.group(1))
            break
    if title == llm_id:
        m_lin = re.search(
            rf"{re.escape(llm_id)}\s*[:\-–—]\s*([^>|;\n\]]+)", first.header_lineage
        )
        if m_lin:
            title = m_lin.group(1).strip()

    # Prefer chunk 1 if it contains title/overview
    overview = first
    for c in chunks:
        if llm_id.lower() in c.content.lower():
            overview = c
            break

    # Strip mitigation section when extracting definition
    body_before_mit = re.split(r"\n#{2,3}\s+Mitigation", overview.content, maxsplit=1, flags=re.IGNORECASE)[0]
    definition = _extract_first_sentences(body_before_mit, max_chars=260)

    return OwaspInfo(
        llm_id=llm_id,
        version=version,
        doc_id=doc_id,
        title=title,
        definition=definition,
        overview_chunk=overview,
    )


def _load_corpus(
    corpus_path: Path,
) -> tuple[
    dict[str, ChunkRow],
    dict[str, CveInfo],
    dict[str, MitreTechInfo],
    dict[tuple[str, str], OwaspInfo],
    dict[str, list[ChunkRow]],
]:
    """Load corpus_chunks.parquet and extract structured records per doc_id."""
    if not corpus_path.exists():
        raise GoldenDatasetValidationError(
            f"Corpus parquet not found at {corpus_path}. Run ingest.py first.",
            details={"corpus_path": str(corpus_path)},
        )

    df = pd.read_parquet(corpus_path)
    docs_chunks: dict[str, list[ChunkRow]] = defaultdict(list)
    chunk_by_id: dict[str, ChunkRow] = {}

    for _, row in df.iterrows():
        sec_raw = row["secondary_identifiers"]
        sec_list = [str(x) for x in sec_raw] if sec_raw is not None else []
        score_raw = row["cvss_base_score"]
        score_val = float(score_raw) if pd.notna(score_raw) else None
        sev_raw = row["severity"]
        sev_val = str(sev_raw) if pd.notna(sev_raw) and sev_raw is not None else None
        parent_raw = row["parent_ref"]
        parent_val = str(parent_raw) if pd.notna(parent_raw) and parent_raw is not None else None

        rec = ChunkRow(
            chunk_id=str(row["chunk_id"]),
            doc_id=str(row["doc_id"]),
            document_type=str(row["document_type"]),
            source_file=str(row["source_file"]),
            standard_version=str(row["standard_version"]),
            primary_identifier=str(row["primary_identifier"]),
            secondary_identifiers=sec_list,
            cvss_base_score=score_val,
            severity=sev_val,
            header_lineage=str(row["header_lineage"]),
            parent_ref=parent_val,
            chunk_index=int(row["chunk_index"]),
            content=str(row["content"]),
        )
        chunk_by_id[rec.chunk_id] = rec
        docs_chunks[rec.doc_id].append(rec)

    for doc_id in docs_chunks:
        docs_chunks[doc_id].sort(key=lambda c: (c.chunk_index, c.chunk_id))

    cves: dict[str, CveInfo] = {}
    mitre_techs: dict[str, MitreTechInfo] = {}
    owasp_entries: dict[tuple[str, str], OwaspInfo] = {}

    for doc_id in sorted(docs_chunks):
        chunks = docs_chunks[doc_id]
        dtype = chunks[0].document_type
        if dtype == "CVE_RECORD":
            cves[chunks[0].primary_identifier] = _extract_cve_info(
                chunks[0].primary_identifier, chunks
            )
        elif dtype == "MITRE_TECHNIQUE":
            info = _extract_mitre_info(doc_id, chunks)
            mitre_techs[info.tech_id] = info
        elif dtype == "OWASP_LLM":
            o_info = _extract_owasp_info(doc_id, chunks)
            owasp_entries[(o_info.version, o_info.llm_id)] = o_info

    return chunk_by_id, cves, mitre_techs, owasp_entries, docs_chunks


def _load_attack_map(map_path: Path) -> list[dict[str, Any]]:
    """Parse data/golden/cve_attack_map.yaml and return list of technique mapping dicts."""
    if not map_path.exists():
        raise GoldenDatasetValidationError(
            f"cve_attack_map.yaml not found at {map_path}",
            details={"map_path": str(map_path)},
        )
    data = yaml.safe_load(map_path.read_text(encoding="utf-8")) or {}
    mappings = data.get("mappings")
    if not isinstance(mappings, list):
        raise GoldenDatasetValidationError(
            "cve_attack_map.yaml must contain a top-level 'mappings' list.",
            details={"map_path": str(map_path)},
        )
    return mappings


# ─────────────────────────────────────────────────────────────────────────────
# Deterministic builders for each of the 6 question categories
# ─────────────────────────────────────────────────────────────────────────────


def _norm_key(q: str) -> str:
    """Normalized key used to guarantee query uniqueness across all 150 items."""
    return " ".join(normalize_query(q).lower().split())


def _register_item(
    items: list[dict[str, Any]],
    seen_queries: set[str],
    item: dict[str, Any],
) -> bool:
    """Append item if its normalized query is unique; return True if added."""
    key = _norm_key(item["query"])
    if key in seen_queries:
        return False
    seen_queries.add(key)
    items.append(item)
    return True


def _build_factual_items(
    cves: dict[str, CveInfo],
    mitre_techs: dict[str, MitreTechInfo],
    chunk_by_id: dict[str, ChunkRow],
    rng: random.Random,
    seen_queries: set[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """
    Build exactly 60 FACTUAL items (6 bare-id FAST_PATH + 54 SEMANTIC_RAG).

    Returns (factual_items, pool_of_base_specs) where pool_of_base_specs can
    also serve as candidate source material for MALFORMED / CONTEXT_NOISE.
    """
    factual: list[dict[str, Any]] = []
    base_pool: list[dict[str, Any]] = []

    cve_ids_sorted = sorted(cves.keys())
    tech_ids_sorted = sorted(mitre_techs.keys())

    # ── 1. 6 Bare-ID Lookups (FAST_PATH_EXACT_LOOKUP) ────────────────────────
    # 3 CVE IDs (spread deterministically across the sorted CVE pool)
    rng_bare_cves = list(cve_ids_sorted)
    rng.shuffle(rng_bare_cves)
    bare_cves = rng_bare_cves[:3]
    if len(bare_cves) < 3:
        raise GoldenDatasetValidationError(
            "Need at least 3 CVE records in corpus for bare-id lookups.",
            details={"available_cves": len(bare_cves)},
        )
    for cve_id in bare_cves:
        info = cves[cve_id]
        cids = [info.first_chunk.chunk_id]
        score_note = (
            f" CVSS v3 base score: {info.cvss_score_str}"
            + (f" ({info.severity})" if info.severity else "")
            + "."
            if info.cvss_score_str
            else ""
        )
        ans = f"{cve_id}: {info.summary}{score_note}"
        req = _verify_entities([cve_id], cids, chunk_by_id)
        _register_item(
            factual,
            seen_queries,
            {
                "question_type": QuestionType.FACTUAL.value,
                "query": cve_id,
                "ground_truth_answer": ans,
                "expected_source_chunk_ids": cids,
                "expected_doc_ids": [cve_id],
                "conflict_test": False,
                "negative_constraint_test": False,
                "required_entities": req,
                "cvss_threshold": None,
                "expected_route": ROUTE_FAST_PATH,
                "link_basis": "bare_cve_lookup",
            },
        )

    # 2 MITRE Technique IDs
    rng_bare_techs = list(tech_ids_sorted)
    rng.shuffle(rng_bare_techs)
    bare_techs = rng_bare_techs[:2]
    if len(bare_techs) < 2:
        raise GoldenDatasetValidationError(
            "Need at least 2 MITRE technique docs in corpus for bare-id lookups.",
            details={"available_techniques": len(bare_techs)},
        )
    for tech_id in bare_techs:
        tinfo = mitre_techs[tech_id]
        cids = [tinfo.overview_chunk.chunk_id]
        tactic_part = (
            f"{tinfo.tactic_id} ({tinfo.tactic_name})"
            if tinfo.tactic_id
            else tinfo.tactic_name
        )
        ans = (
            f"{tech_id} ({tinfo.tech_name}) belongs to MITRE ATT&CK tactic {tactic_part}. "
            f"{tinfo.summary}"
        )
        req = _verify_entities([tech_id, tinfo.tech_name], cids, chunk_by_id)
        _register_item(
            factual,
            seen_queries,
            {
                "question_type": QuestionType.FACTUAL.value,
                "query": tech_id,
                "ground_truth_answer": ans,
                "expected_source_chunk_ids": cids,
                "expected_doc_ids": [tinfo.doc_id],
                "conflict_test": False,
                "negative_constraint_test": False,
                "required_entities": req,
                "cvss_threshold": None,
                "expected_route": ROUTE_FAST_PATH,
                "link_basis": "bare_mitre_technique_lookup",
            },
        )

    # 1 MITRE Mitigation ID — expected chunks are ALL MITRE chunks having M-ID in secondary_identifiers
    mid_to_chunks: dict[str, list[ChunkRow]] = defaultdict(list)
    mid_to_infos: dict[str, list[MitreMitigationInfo]] = defaultdict(list)
    for tinfo in mitre_techs.values():
        for c in tinfo.chunks:
            for sid in c.secondary_identifiers:
                if _RE_MITIGATION_ID.fullmatch(sid):
                    mid_to_chunks[sid].append(c)
        for minfo in tinfo.mitigations:
            mid_to_infos[minfo.m_id].append(minfo)

    if not mid_to_chunks:
        raise GoldenDatasetValidationError(
            "No MITRE chunks with mitigation M-ID in secondary_identifiers found.",
            details={"mitre_tech_count": len(mitre_techs)},
        )
    # Choose deterministically an M-ID with concise chunk count (1..3 preferred)
    sorted_mids = sorted(
        mid_to_chunks.keys(),
        key=lambda m: (len({c.chunk_id for c in mid_to_chunks[m]}), m),
    )
    chosen_mid = sorted_mids[0]
    m_chunks = sorted(
        {c.chunk_id: c for c in mid_to_chunks[chosen_mid]}.values(),
        key=lambda c: c.chunk_id,
    )
    m_cids = [c.chunk_id for c in m_chunks]
    m_dids = list(dict.fromkeys(c.doc_id for c in m_chunks))
    m_infos = mid_to_infos.get(chosen_mid, [])
    m_name = m_infos[0].name if m_infos else chosen_mid
    m_desc = m_infos[0].description if m_infos else _extract_first_sentences(m_chunks[0].content)
    tech_list = ", ".join(dict.fromkeys(info.tech_id for info in m_infos))
    ans = (
        f"{chosen_mid} ({m_name}) is a MITRE ATT&CK mitigation"
        + (f" mapped to {tech_list}" if tech_list else "")
        + f". {m_desc}"
    )
    req = _verify_entities([chosen_mid, m_name], m_cids, chunk_by_id)
    _register_item(
        factual,
        seen_queries,
        {
            "question_type": QuestionType.FACTUAL.value,
            "query": chosen_mid,
            "ground_truth_answer": ans,
            "expected_source_chunk_ids": m_cids,
            "expected_doc_ids": m_dids,
            "conflict_test": False,
            "negative_constraint_test": False,
            "required_entities": req,
            "cvss_threshold": None,
            "expected_route": ROUTE_FAST_PATH,
            "link_basis": "bare_mitre_mitigation_lookup",
        },
    )

    # ── 2. 54 Semantic Factual Questions ─────────────────────────────────────
    # Prepare candidate pools for each sub-bucket
    v31_cves = [c for c in cve_ids_sorted if cves[c].cvss_v31_vector and cves[c].cvss_score_str]
    av_cves = [c for c in cve_ids_sorted if cves[c].attack_vector_label]
    cwe_cves = [c for c in cve_ids_sorted if cves[c].cwe_ids]
    ver_cves = [c for c in cve_ids_sorted if cves[c].version_boundary and cves[c].version_token]

    rng.shuffle(v31_cves)
    rng.shuffle(av_cves)
    rng.shuffle(cwe_cves)
    rng.shuffle(ver_cves)

    used_cve_for_aspect: set[tuple[str, str]] = set()
    used_tech_factual: set[str] = set()
    used_mit_factual: set[tuple[str, str]] = set()

    def _make_cvss_item(cve_id: str, variant: int) -> dict[str, Any]:
        info = cves[cve_id]
        templates = [
            f"What are the CVSS v3.1 base score and vector string for {cve_id}?",
            f"Provide the CVSS v3.1 numerical score and full vector string assigned to {cve_id}.",
            f"Which CVSS v3.1 score, severity rating, and vector string apply to {cve_id}?",
        ]
        q = templates[variant % len(templates)]
        sev = f" ({info.severity})" if info.severity else ""
        ans = (
            f"{cve_id} has a CVSS v3.1 base score of {info.cvss_score_str}{sev} "
            f"with vector string {info.cvss_v31_vector}."
        )
        cids = list(info.cvss_v31_chunk_ids)
        dids = list(dict.fromkeys(chunk_by_id[cid].doc_id for cid in cids))
        req = _verify_entities(
            [cve_id, info.cvss_score_str or "", info.cvss_v31_vector or ""],
            cids,
            chunk_by_id,
        )
        return {
            "question_type": QuestionType.FACTUAL.value,
            "query": q,
            "ground_truth_answer": ans,
            "expected_source_chunk_ids": cids,
            "expected_doc_ids": dids,
            "conflict_test": False,
            "negative_constraint_test": False,
            "required_entities": req,
            "cvss_threshold": None,
            "expected_route": ROUTE_SEMANTIC,
            "link_basis": "factual_cvss_v31",
        }

    def _make_av_item(cve_id: str, variant: int) -> dict[str, Any]:
        info = cves[cve_id]
        templates = [
            f"What is the CVSS attack vector for {cve_id}?",
            f"Through which attack vector can an adversary exploit {cve_id}?",
            f"Identify the exploitation access/attack vector recorded in CVSS for {cve_id}.",
        ]
        q = templates[variant % len(templates)]
        label = info.attack_vector_label or "NETWORK"
        readable = _AV_READABLE_MAP.get(label, label)
        ans = f"The CVSS attack vector for {cve_id} is {readable}."
        cids = list(info.attack_vector_chunk_ids)
        dids = list(dict.fromkeys(chunk_by_id[cid].doc_id for cid in cids))
        cand_ents = [cve_id, label]
        if info.attack_vector_code:
            cand_ents.append(f"AV:{info.attack_vector_code}")
        req = _verify_entities(cand_ents, cids, chunk_by_id)
        return {
            "question_type": QuestionType.FACTUAL.value,
            "query": q,
            "ground_truth_answer": ans,
            "expected_source_chunk_ids": cids,
            "expected_doc_ids": dids,
            "conflict_test": False,
            "negative_constraint_test": False,
            "required_entities": req,
            "cvss_threshold": None,
            "expected_route": ROUTE_SEMANTIC,
            "link_basis": "factual_attack_vector",
        }

    def _make_cwe_item(cve_id: str, variant: int) -> dict[str, Any]:
        info = cves[cve_id]
        cwe_id = info.cwe_ids[0]
        templates = [
            f"Which Common Weakness Enumeration (CWE) identifier is mapped to {cve_id}?",
            f"What CWE weakness classification is recorded for {cve_id}?",
            f"Identify the CWE ID associated with the root weakness in {cve_id}.",
        ]
        q = templates[variant % len(templates)]
        ans = f"{cve_id} is categorized under {cwe_id}."
        cids = list(info.cwe_chunk_map[cwe_id])
        dids = list(dict.fromkeys(chunk_by_id[cid].doc_id for cid in cids))
        req = _verify_entities([cve_id, cwe_id], cids, chunk_by_id)
        return {
            "question_type": QuestionType.FACTUAL.value,
            "query": q,
            "ground_truth_answer": ans,
            "expected_source_chunk_ids": cids,
            "expected_doc_ids": dids,
            "conflict_test": False,
            "negative_constraint_test": False,
            "required_entities": req,
            "cvss_threshold": None,
            "expected_route": ROUTE_SEMANTIC,
            "link_basis": "factual_cwe_id",
        }

    def _make_version_item(cve_id: str, variant: int) -> dict[str, Any]:
        info = cves[cve_id]
        prod_hint = f" ({info.version_product})" if info.version_product else ""
        templates = [
            f"What affected software version boundary is specified for {cve_id}{prod_hint}?",
            f"Which vulnerable version limit or range is recorded in {cve_id}?",
            f"State the affected version threshold listed in the NVD record for {cve_id}.",
        ]
        q = templates[variant % len(templates)]
        ans = (
            f"The affected version boundary recorded for {cve_id}"
            + (f" ({info.version_product})" if info.version_product else "")
            + f" is {info.version_boundary}."
        )
        cids = list(info.version_chunk_ids)
        dids = list(dict.fromkeys(chunk_by_id[cid].doc_id for cid in cids))
        req = _verify_entities(
            [cve_id, info.version_token or ""],
            cids,
            chunk_by_id,
        )
        return {
            "question_type": QuestionType.FACTUAL.value,
            "query": q,
            "ground_truth_answer": ans,
            "expected_source_chunk_ids": cids,
            "expected_doc_ids": dids,
            "conflict_test": False,
            "negative_constraint_test": False,
            "required_entities": req,
            "cvss_threshold": None,
            "expected_route": ROUTE_SEMANTIC,
            "link_basis": "factual_version_boundary",
        }

    def _make_tech_item(tech_id: str, variant: int) -> dict[str, Any]:
        tinfo = mitre_techs[tech_id]
        templates = [
            f"What is the name and MITRE ATT&CK tactic for technique {tech_id}?",
            f"Which Enterprise ATT&CK tactic and technique title correspond to {tech_id}?",
            f"Describe the MITRE ATT&CK technique {tech_id} and its parent tactic.",
        ]
        q = templates[variant % len(templates)]
        tactic_part = (
            f"{tinfo.tactic_id} ({tinfo.tactic_name})"
            if tinfo.tactic_id
            else tinfo.tactic_name
        )
        ans = (
            f"MITRE ATT&CK {tech_id} is '{tinfo.tech_name}' under tactic {tactic_part}. "
            f"{tinfo.summary}"
        )
        cids = [tinfo.overview_chunk.chunk_id]
        cand_ents = [tech_id, tinfo.tech_name]
        if tinfo.tactic_id:
            cand_ents.append(tinfo.tactic_id)
        req = _verify_entities(cand_ents, cids, chunk_by_id)
        return {
            "question_type": QuestionType.FACTUAL.value,
            "query": q,
            "ground_truth_answer": ans,
            "expected_source_chunk_ids": cids,
            "expected_doc_ids": [tinfo.doc_id],
            "conflict_test": False,
            "negative_constraint_test": False,
            "required_entities": req,
            "cvss_threshold": None,
            "expected_route": ROUTE_SEMANTIC,
            "link_basis": "factual_mitre_technique",
        }

    def _make_mit_item(minfo: MitreMitigationInfo, variant: int) -> dict[str, Any]:
        tinfo = mitre_techs[minfo.tech_id]
        templates = [
            f"What is the name and function of MITRE ATT&CK mitigation {minfo.m_id} for {minfo.tech_id}?",
            f"How does MITRE mitigation {minfo.m_id} reduce risk for {minfo.tech_id} ({tinfo.tech_name})?",
            f"Identify MITRE ATT&CK mitigation {minfo.m_id} as documented under technique {minfo.tech_id}.",
        ]
        q = templates[variant % len(templates)]
        ans = (
            f"MITRE ATT&CK mitigation {minfo.m_id} is '{minfo.name}'. "
            f"For {minfo.tech_id}: {minfo.description}"
        )
        cids = [minfo.chunk_id]
        req = _verify_entities([minfo.m_id, minfo.name], cids, chunk_by_id)
        return {
            "question_type": QuestionType.FACTUAL.value,
            "query": q,
            "ground_truth_answer": ans,
            "expected_source_chunk_ids": cids,
            "expected_doc_ids": [minfo.doc_id],
            "conflict_test": False,
            "negative_constraint_test": False,
            "required_entities": req,
            "cvss_threshold": None,
            "expected_route": ROUTE_SEMANTIC,
            "link_basis": "factual_mitre_mitigation",
        }

    # All (tech_id, m_id) mitigation pairs
    all_mit_pairs: list[MitreMitigationInfo] = []
    for tid in tech_ids_sorted:
        all_mit_pairs.extend(mitre_techs[tid].mitigations)
    rng.shuffle(all_mit_pairs)

    # Prioritise distinct CVEs across buckets where possible
    buckets: dict[str, list[dict[str, Any]]] = {
        "cvss_score_vector": [],
        "attack_vector": [],
        "cwe_id": [],
        "version_boundary": [],
        "technique_name_tactic": [],
        "mitigation_name": [],
    }
    targets: dict[str, int] = {
        "cvss_score_vector": 18,
        "attack_vector": 10,
        "cwe_id": 8,
        "version_boundary": 8,
        "technique_name_tactic": 6,
        "mitigation_name": 4,
    }

    # Fill version_boundary & cwe_id & attack_vector & cvss_score_vector preferring unused CVEs
    used_primary_cves: set[str] = set()

    def _fill_cve_bucket(
        bucket_name: str,
        candidates: list[str],
        builder: Any,
        target: int,
    ) -> None:
        ordered = [c for c in candidates if c not in used_primary_cves] + [
            c for c in candidates if c in used_primary_cves
        ]
        for cve_id in ordered:
            if len(buckets[bucket_name]) >= target:
                break
            if (cve_id, bucket_name) in used_cve_for_aspect:
                continue
            item = builder(cve_id, 0)
            if _norm_key(item["query"]) in seen_queries:
                continue
            buckets[bucket_name].append(item)
            seen_queries.add(_norm_key(item["query"]))
            used_cve_for_aspect.add((cve_id, bucket_name))
            used_primary_cves.add(cve_id)

    _fill_cve_bucket("version_boundary", ver_cves, _make_version_item, targets["version_boundary"])
    _fill_cve_bucket("cwe_id", cwe_cves, _make_cwe_item, targets["cwe_id"])
    _fill_cve_bucket("attack_vector", av_cves, _make_av_item, targets["attack_vector"])
    _fill_cve_bucket("cvss_score_vector", v31_cves, _make_cvss_item, targets["cvss_score_vector"])

    # Fill technique_name_tactic (6)
    tech_pool = list(tech_ids_sorted)
    rng.shuffle(tech_pool)
    for tid in tech_pool:
        if len(buckets["technique_name_tactic"]) >= targets["technique_name_tactic"]:
            break
        item = _make_tech_item(tid, 0)
        if _norm_key(item["query"]) not in seen_queries:
            buckets["technique_name_tactic"].append(item)
            seen_queries.add(_norm_key(item["query"]))
            used_tech_factual.add(tid)

    # Fill mitigation_name (4)
    for minfo in all_mit_pairs:
        if len(buckets["mitigation_name"]) >= targets["mitigation_name"]:
            break
        if (minfo.tech_id, minfo.m_id) in used_mit_factual:
            continue
        item = _make_mit_item(minfo, 0)
        if _norm_key(item["query"]) not in seen_queries:
            buckets["mitigation_name"].append(item)
            seen_queries.add(_norm_key(item["query"]))
            used_mit_factual.add((minfo.tech_id, minfo.m_id))

    # Deterministic rebalancing if any bucket fell short of 54 total
    semantic_count = sum(len(v) for v in buckets.values())
    if semantic_count < 54:
        log.warning(
            "Factual bucket counts before rebalance: %s (total=%d/54)",
            {k: len(v) for k, v in buckets.items()},
            semantic_count,
        )
        # Round 1: unused (cve, aspect) pairs across CVE buckets
        rebalance_cve_builders = [
            ("cvss_score_vector", v31_cves, _make_cvss_item),
            ("attack_vector", av_cves, _make_av_item),
            ("cwe_id", cwe_cves, _make_cwe_item),
            ("version_boundary", ver_cves, _make_version_item),
        ]
        for bucket_name, cand_list, builder in rebalance_cve_builders:
            for cve_id in cand_list:
                if semantic_count >= 54:
                    break
                if (cve_id, bucket_name) in used_cve_for_aspect:
                    continue
                item = builder(cve_id, 0)
                if _norm_key(item["query"]) not in seen_queries:
                    buckets[bucket_name].append(item)
                    seen_queries.add(_norm_key(item["query"]))
                    used_cve_for_aspect.add((cve_id, bucket_name))
                    semantic_count += 1

        # Round 2: remaining unused mitigation pairs & technique phrasings
        for minfo in all_mit_pairs:
            if semantic_count >= 54:
                break
            if (minfo.tech_id, minfo.m_id) in used_mit_factual:
                continue
            item = _make_mit_item(minfo, 0)
            if _norm_key(item["query"]) not in seen_queries:
                buckets["mitigation_name"].append(item)
                seen_queries.add(_norm_key(item["query"]))
                used_mit_factual.add((minfo.tech_id, minfo.m_id))
                semantic_count += 1

        for tid in tech_pool:
            if semantic_count >= 54:
                break
            if tid in used_tech_factual:
                item = _make_tech_item(tid, 1)
            else:
                item = _make_tech_item(tid, 0)
            if _norm_key(item["query"]) not in seen_queries:
                buckets["technique_name_tactic"].append(item)
                seen_queries.add(_norm_key(item["query"]))
                used_tech_factual.add(tid)
                semantic_count += 1

        # Round 3: alternative phrasing variants for distinct questions if still short
        for variant in (1, 2):
            for bucket_name, cand_list, builder in rebalance_cve_builders:
                for cve_id in cand_list:
                    if semantic_count >= 54:
                        break
                    item = builder(cve_id, variant)
                    if _norm_key(item["query"]) not in seen_queries:
                        buckets[bucket_name].append(item)
                        seen_queries.add(_norm_key(item["query"]))
                        semantic_count += 1

    for bucket_name in (
        "cvss_score_vector",
        "attack_vector",
        "cwe_id",
        "version_boundary",
        "technique_name_tactic",
        "mitigation_name",
    ):
        factual.extend(buckets[bucket_name])

    if len(factual) != 60:
        raise GoldenDatasetValidationError(
            f"Could not assemble 60 distinct FACTUAL questions (got {len(factual)}).",
            details={
                "actual_total": len(factual),
                "bucket_counts": {k: len(v) for k, v in buckets.items()},
                "shortfall": 60 - len(factual),
            },
        )

    # Also build diverse pool of base question templates for MALFORMED & CONTEXT_NOISE
    for cve_id in cve_ids_sorted:
        info = cves[cve_id]
        if info.cvss_v31_vector and info.cvss_score_str:
            base_pool.append(_make_cvss_item(cve_id, 2))
        if info.attack_vector_label:
            base_pool.append(_make_av_item(cve_id, 2))
        if info.cwe_ids:
            base_pool.append(_make_cwe_item(cve_id, 2))
    for tid in tech_ids_sorted:
        base_pool.append(_make_tech_item(tid, 2))

    return factual, base_pool


def _build_multi_hop_items(
    mappings: list[dict[str, Any]],
    cves: dict[str, CveInfo],
    mitre_techs: dict[str, MitreTechInfo],
    chunk_by_id: dict[str, ChunkRow],
    seen_queries: set[str],
) -> list[dict[str, Any]]:
    """Build 45 MULTI_HOP items across verified cve_attack_map.yaml pairs."""
    # Collect verified pairs per technique preserving technique representation
    pairs_by_tech: list[tuple[str, list[str]]] = []
    for entry in mappings:
        tech_id = str(entry.get("technique_id", ""))
        cve_list = [str(c) for c in entry.get("cve_ids", [])]
        if tech_id not in mitre_techs:
            log.warning("Multi-hop mapping technique %s missing from corpus; skipping.", tech_id)
            continue
        verified_cves = [c for c in cve_list if c in cves]
        if verified_cves:
            pairs_by_tech.append((tech_id, verified_cves))

    if not pairs_by_tech:
        raise GoldenDatasetValidationError(
            "No verified CVE->MITRE mapping pairs exist in corpus.",
            details={"cves_in_corpus": len(cves), "mitre_in_corpus": len(mitre_techs)},
        )

    # Round-robin across the 9 techniques to guarantee all techniques are represented
    selected_pairs: list[tuple[str, str]] = []
    seen_pairs: set[tuple[str, str]] = set()
    max_len = max(len(c_list) for _, c_list in pairs_by_tech)
    for idx in range(max_len):
        for tech_id, c_list in pairs_by_tech:
            if idx < len(c_list):
                pair = (tech_id, c_list[idx])
                if pair not in seen_pairs:
                    seen_pairs.add(pair)
                    selected_pairs.append(pair)
                    if len(selected_pairs) == 45:
                        break
        if len(selected_pairs) == 45:
            break

    # Primary templates (>= 4 rotating phrasings)
    primary_templates = [
        (
            "Which MITRE ATT&CK technique corresponds to the exploitation mechanism in {cve_id}, "
            "and what enterprise mitigations apply?"
        ),
        (
            "Given the vulnerability described in {cve_id}, identify the mapped MITRE ATT&CK "
            "technique and its recommended mitigations."
        ),
        (
            "How does {cve_id} map to the MITRE ATT&CK Enterprise matrix, and which defensive "
            "mitigations counter that technique?"
        ),
        (
            "For {cve_id}, determine the corresponding MITRE ATT&CK technique ID and name along "
            "with applicable mitigation controls."
        ),
    ]
    secondary_templates = [
        (
            "Identify the MITRE ATT&CK tactic, technique, and top mitigations associated with {cve_id} "
            "to prioritize remediation."
        ),
        (
            "Explain which MITRE ATT&CK defensive mitigations protect against the exploitation pattern "
            "seen in {cve_id}."
        ),
    ]

    def _assemble_multihop(tech_id: str, cve_id: str, query_text: str) -> dict[str, Any]:
        cve_info = cves[cve_id]
        tinfo = mitre_techs[tech_id]
        chosen_mits = tinfo.mitigations[:3]

        # Expected chunks: CVE chunk_001 + technique overview chunk + mitigation chunk(s) (max 3 mits)
        raw_cids: list[str] = [
            cve_info.first_chunk.chunk_id,
            tinfo.overview_chunk.chunk_id,
        ]
        for minfo in chosen_mits:
            raw_cids.append(minfo.chunk_id)
        expected_cids = list(dict.fromkeys(raw_cids))
        expected_dids = [cve_id, tinfo.doc_id]

        mit_ids = [m.m_id for m in chosen_mits]
        mit_phrase = (
            ", ".join(f"{m.m_id} ({m.name})" for m in chosen_mits)
            if chosen_mits
            else "no specific STIX mitigations listed"
        )
        tactic_part = (
            f"{tinfo.tactic_id} ({tinfo.tactic_name})"
            if tinfo.tactic_id
            else tinfo.tactic_name
        )
        score_part = (
            f" (CVSS {cve_info.cvss_score_str})" if cve_info.cvss_score_str else ""
        )
        ans = (
            f"{cve_id}{score_part} ({cve_info.summary}) maps via analyst mapping to MITRE ATT&CK "
            f"technique {tech_id} ({tinfo.tech_name}) under tactic {tactic_part}. "
            f"Applicable enterprise mitigations include: {mit_phrase}."
        )
        req = _verify_entities(
            [cve_id, tech_id] + mit_ids,
            expected_cids,
            chunk_by_id,
        )
        return {
            "question_type": QuestionType.MULTI_HOP.value,
            "query": query_text,
            "ground_truth_answer": ans,
            "expected_source_chunk_ids": expected_cids,
            "expected_doc_ids": expected_dids,
            "conflict_test": False,
            "negative_constraint_test": False,
            "required_entities": req,
            "cvss_threshold": None,
            "expected_route": ROUTE_SEMANTIC,
            "link_basis": "analyst_mapping",
        }

    multihop: list[dict[str, Any]] = []
    for idx, (tech_id, cve_id) in enumerate(selected_pairs):
        tmpl = primary_templates[idx % len(primary_templates)]
        q = tmpl.format(cve_id=cve_id)
        item = _assemble_multihop(tech_id, cve_id, q)
        _register_item(multihop, seen_queries, item)

    # Rebalance if fewer than 45 verified pairs existed (e.g. skipped 404 CVEs)
    if len(multihop) < 45:
        log.warning(
            "Only %d verified CVE->MITRE pairs found; generating secondary distinct phrasings to reach 45.",
            len(selected_pairs),
        )
        for tmpl in secondary_templates:
            for tech_id, cve_id in selected_pairs:
                if len(multihop) >= 45:
                    break
                q = tmpl.format(cve_id=cve_id)
                item = _assemble_multihop(tech_id, cve_id, q)
                _register_item(multihop, seen_queries, item)
            if len(multihop) >= 45:
                break

    if len(multihop) != 45:
        raise GoldenDatasetValidationError(
            f"Could not assemble 45 MULTI_HOP questions (got {len(multihop)}).",
            details={"actual": len(multihop), "verified_pairs": len(selected_pairs)},
        )

    return multihop


def _build_conflict_trap_items(
    owasp_entries: dict[tuple[str, str], OwaspInfo],
    chunk_by_id: dict[str, ChunkRow],
    seen_queries: set[str],
) -> list[dict[str, Any]]:
    """Build 15 CONFLICT_TRAP items comparing OWASP v1.1 vs v2.0 without version in query."""
    templates_by_llm: dict[str, list[str]] = {
        "LLM03": [
            "According to the OWASP Top 10 for LLM Applications, what risk is designated as LLM03?",
            "What vulnerability category and mitigations are specified under OWASP LLM03?",
            "Explain the security issue cataloged as LLM03 in the OWASP GenAI Top 10 framework.",
            "How does the OWASP LLM Top 10 define LLM03 and recommend defending against it?",
            "What threat is classified as OWASP LLM03 for large language model systems?",
        ],
        "LLM05": [
            "According to the OWASP Top 10 for LLM Applications, what vulnerability is LLM05?",
            "What security risk and controls are described under OWASP LLM05?",
            "Define OWASP LLM05 and outline its primary mitigation strategies.",
            "Which threat category is assigned to LLM05 in the OWASP LLM Top 10 guide?",
            "What does OWASP GenAI Top 10 entry LLM05 cover?",
        ],
        "LLM02": [
            "In the OWASP Top 10 for LLM Applications, what vulnerability is listed as LLM02?",
            "Describe the security risk and mitigations associated with OWASP LLM02.",
        ],
        "LLM04": [
            "What vulnerability category does OWASP LLM04 represent in the LLM Top 10?",
        ],
        "LLM06": [
            "How is LLM06 defined in the OWASP Top 10 for Large Language Model Applications?",
        ],
        "LLM07": [
            "What security risk is designated under OWASP LLM07?",
        ],
    }

    conflicts: list[dict[str, Any]] = []
    for llm_id, count in _CONFLICT_SPREAD:
        v11 = owasp_entries.get(("v1.1", llm_id))
        v20 = owasp_entries.get(("v2.0", llm_id))
        if v11 is None or v20 is None:
            raise GoldenDatasetValidationError(
                f"Missing OWASP v1.1 or v2.0 entry for conflict trap {llm_id}.",
                details={"llm_id": llm_id, "has_v1.1": v11 is not None, "has_v2.0": v20 is not None},
            )

        cids = [v11.overview_chunk.chunk_id, v20.overview_chunk.chunk_id]
        dids = [v11.doc_id, v20.doc_id]
        ans = (
            f"The OWASP LLM Top 10 definition of {llm_id} differs between versions. "
            f"Under OWASP v1.1, {llm_id} is '{v11.title}': {v11.definition} "
            f"Under OWASP v2.0, {llm_id} is '{v20.title}': {v20.definition}"
        )
        req = _verify_entities(
            [llm_id, v11.title, v20.title, "v1.1", "v2.0"],
            cids,
            chunk_by_id,
        )

        phrasings = templates_by_llm[llm_id]
        if len(phrasings) < count:
            raise GoldenDatasetValidationError(
                f"Insufficient distinct phrasings for {llm_id}",
                details={"needed": count, "available": len(phrasings)},
            )
        for idx in range(count):
            q = phrasings[idx]
            added = _register_item(
                conflicts,
                seen_queries,
                {
                    "question_type": QuestionType.CONFLICT_TRAP.value,
                    "query": q,
                    "ground_truth_answer": ans,
                    "expected_source_chunk_ids": cids,
                    "expected_doc_ids": dids,
                    "conflict_test": True,
                    "negative_constraint_test": True,
                    "required_entities": req,
                    "cvss_threshold": None,
                    "expected_route": ROUTE_SEMANTIC,
                    "link_basis": f"owasp_version_shift_{llm_id}",
                },
            )
            if not added:
                raise GoldenDatasetValidationError(
                    f"Duplicate conflict trap query generated: {q}",
                    details={"llm_id": llm_id},
                )

    if len(conflicts) != 15:
        raise GoldenDatasetValidationError(
            f"Expected 15 CONFLICT_TRAP items, got {len(conflicts)}."
        )
    return conflicts


def _build_out_of_scope_items(seen_queries: set[str]) -> list[dict[str, Any]]:
    """Build the 10 exact verbatim OUT_OF_SCOPE items."""
    oos: list[dict[str, Any]] = []
    for query_text, expected_route in _OUT_OF_SCOPE_SPECS:
        added = _register_item(
            oos,
            seen_queries,
            {
                "question_type": QuestionType.OUT_OF_SCOPE.value,
                "query": query_text,
                "ground_truth_answer": "INSUFFICIENT_CONTEXT",
                "expected_source_chunk_ids": [],
                "expected_doc_ids": [],
                "conflict_test": False,
                "negative_constraint_test": True,
                "required_entities": [],
                "cvss_threshold": None,
                "expected_route": expected_route,
                "link_basis": "out_of_scope_refusal",
            },
        )
        if not added:
            raise GoldenDatasetValidationError(
                f"Duplicate OUT_OF_SCOPE query: {query_text}"
            )
    return oos


def _corrupt_word(word: str, rng: random.Random) -> str:
    """Apply deterministic typo corruption to a single non-ID word."""
    replacements = {
        "vulnerability": "vuln",
        "vulnerabilities": "vulns",
        "mitigation": "mitgation",
        "mitigations": "mitgations",
        "attack": "atack",
        "vector": "vectr",
        "score": "scre",
        "technique": "technq",
        "techniques": "technqs",
        "enterprise": "entrprse",
        "identifier": "idntfr",
        "weakness": "weakns",
        "corresponding": "corspndg",
        "recommended": "recmded",
        "associated": "assoctd",
        "classification": "clasifctn",
        "affected": "afctd",
        "boundary": "bndry",
        "software": "softwr",
        "recorded": "recrdd",
        "explain": "expln",
        "describe": "dscrbe",
    }
    low = word.lower().strip("?.,;:()")
    if low in replacements:
        return replacements[low]
    if len(low) <= 4 or any(ch.isdigit() for ch in low):
        return word
    mode = rng.randrange(3)
    chars = list(low)
    if mode == 0:
        # Drop first interior vowel
        vowels = set("aeiou")
        for i in range(1, len(chars) - 1):
            if chars[i] in vowels:
                del chars[i]
                break
        return "".join(chars)
    if mode == 1 and len(chars) >= 5:
        # Adjacent swap in interior
        idx = rng.randint(1, len(chars) - 3)
        chars[idx], chars[idx + 1] = chars[idx + 1], chars[idx]
        return "".join(chars)
    # Truncate trailing characters
    return "".join(chars[:-2]) if len(chars) > 6 else "".join(chars[:-1])


def _corrupt_query(query: str, rng: random.Random) -> str:
    """Apply realistic shorthand, broken IDs, and typo noise to a query."""
    # Break formal CVE/MITRE/CWE punctuation
    q = re.sub(r"CVE-(\d{4})-(\d+)", lambda m: f"cve{m.group(1)}-{m.group(2)}", query)
    q = re.sub(r"\bCWE-(\d+)\b", r"cwe \1", q)
    q = re.sub(r"\b(T\d{4}(?:\.\d{3})?)\b", lambda m: m.group(1).lower(), q)

    drop_stopwords = {"what", "is", "the", "are", "and", "for", "to", "in", "of", "which", "with", "that", "a", "an"}
    tokens = q.split()
    out_tokens: list[str] = []
    for tok in tokens:
        clean_tok = tok.strip("?.,;:")
        if clean_tok.lower() in drop_stopwords and rng.random() < 0.7:
            continue
        if re.match(r"^(cve|cwe|t\d|m\d)", clean_tok.lower()):
            out_tokens.append(clean_tok.lower())
        else:
            out_tokens.append(_corrupt_word(clean_tok, rng))

    result = " ".join(t for t in out_tokens if t).strip()
    if not result or _RE_EXACT_ID_ONLY.match(normalize_query(result)):
        result = f"{result} vuln info".strip()
    return result


def _build_malformed_items(
    cves: dict[str, CveInfo],
    base_pool: list[dict[str, Any]],
    chunk_by_id: dict[str, ChunkRow],
    rng: random.Random,
    seen_queries: set[str],
) -> list[dict[str, Any]]:
    """Build 10 MALFORMED items including required CVE-2024-3094 queries if present."""
    malformed: list[dict[str, Any]] = []

    if "CVE-2024-3094" in cves:
        info = cves["CVE-2024-3094"]
        cids_1 = [info.first_chunk.chunk_id]
        score_part = (
            f" CVSS base score {info.cvss_score_str}"
            + (f" ({info.severity})" if info.severity else "")
            if info.cvss_score_str
            else ""
        )
        ans_1 = f"CVE-2024-3094{score_part}: {info.summary}"
        req_1 = _verify_entities(
            ["CVE-2024-3094", info.cvss_score_str or ""],
            cids_1,
            chunk_by_id,
        )
        _register_item(
            malformed,
            seen_queries,
            {
                "question_type": QuestionType.MALFORMED.value,
                "query": "cvs 9.8 vuln xz util liblzma bckdoor mitgation",
                "ground_truth_answer": ans_1,
                "expected_source_chunk_ids": cids_1,
                "expected_doc_ids": ["CVE-2024-3094"],
                "conflict_test": False,
                "negative_constraint_test": False,
                "required_entities": req_1,
                "cvss_threshold": None,
                "expected_route": ROUTE_SEMANTIC,
                "link_basis": "malformed_verbatim_xz",
            },
        )

        cids_2 = info.attack_vector_chunk_ids or info.cvss_v31_chunk_ids or [info.first_chunk.chunk_id]
        av_readable = (
            _AV_READABLE_MAP.get(info.attack_vector_label, info.attack_vector_label)
            if info.attack_vector_label
            else "documented in the CVSS vector"
        )
        ans_2 = (
            f"CVE-2024-3094 has attack vector {av_readable}"
            + (f" (vector {info.cvss_v31_vector})" if info.cvss_v31_vector else "")
            + "."
        )
        cand_ents_2 = ["CVE-2024-3094"]
        if info.attack_vector_label:
            cand_ents_2.append(info.attack_vector_label)
        if info.attack_vector_code:
            cand_ents_2.append(f"AV:{info.attack_vector_code}")
        req_2 = _verify_entities(cand_ents_2, cids_2, chunk_by_id)
        _register_item(
            malformed,
            seen_queries,
            {
                "question_type": QuestionType.MALFORMED.value,
                "query": "cve2024-3094 atack vectr",
                "ground_truth_answer": ans_2,
                "expected_source_chunk_ids": cids_2,
                "expected_doc_ids": ["CVE-2024-3094"],
                "conflict_test": False,
                "negative_constraint_test": False,
                "required_entities": req_2,
                "cvss_threshold": None,
                "expected_route": ROUTE_SEMANTIC,
                "link_basis": "malformed_verbatim_cve2024_3094",
            },
        )

    candidates = list(base_pool)
    rng.shuffle(candidates)
    used_docs: set[str] = {"CVE-2024-3094"} if "CVE-2024-3094" in cves else set()

    for cand in candidates:
        if len(malformed) >= 10:
            break
        doc_key = "|".join(cand["expected_doc_ids"])
        if doc_key in used_docs:
            continue
        corrupted = _corrupt_query(cand["query"], rng)
        if _RE_EXACT_ID_ONLY.match(normalize_query(corrupted)):
            continue
        item = {
            **cand,
            "question_type": QuestionType.MALFORMED.value,
            "query": corrupted,
            "expected_route": ROUTE_SEMANTIC,
            "link_basis": f"malformed_from_{cand.get('link_basis', 'base')}",
        }
        if _register_item(malformed, seen_queries, item):
            used_docs.add(doc_key)

    # Fallback pass if fewer than 10 distinct docs were available
    for cand in candidates:
        if len(malformed) >= 10:
            break
        corrupted = _corrupt_query(cand["query"], rng) + " info"
        if _RE_EXACT_ID_ONLY.match(normalize_query(corrupted)):
            continue
        item = {
            **cand,
            "question_type": QuestionType.MALFORMED.value,
            "query": corrupted,
            "expected_route": ROUTE_SEMANTIC,
            "link_basis": f"malformed_from_{cand.get('link_basis', 'base')}",
        }
        _register_item(malformed, seen_queries, item)

    if len(malformed) != 10:
        raise GoldenDatasetValidationError(
            f"Expected 10 MALFORMED items, got {len(malformed)}."
        )
    return malformed


def _make_synthetic_noise(idx: int, rng: random.Random) -> str:
    """Create 700-1050 chars of deterministic log, stack-trace, and JSON noise."""
    host = f"prod-edge-{idx:02d}.internal"
    pid = 10400 + idx * 37
    req_id = f"req-{rng.randint(100000, 999999)}"
    ip_addr = f"10.24.{rng.randint(1, 250)}.{rng.randint(2, 250)}"

    syslog = (
        f"2026-11-10T14:22:{10 + idx:02d}Z {host} systemd[1]: Started worker-pool-{idx}.service.\n"
        f"2026-11-10T14:22:{11 + idx:02d}Z {host} nginx[{pid}]: {ip_addr} - GET /healthz HTTP/1.1 200 42 \"-\"\n"
        f"2026-11-10T14:22:{12 + idx:02d}Z {host} kernel: [89214.{100 + idx}] TCP: eth0: link speed 10000Mb/s\n"
    )
    stack_trace = (
        "Traceback (most recent call last):\n"
        "  File \"/opt/soc-collector/pipeline/dispatcher.py\", line 142, in process_batch\n"
        "    resp = session.post(endpoint, json=payload, timeout=5.0)\n"
        "  File \"/opt/soc-collector/venv/lib/python3.12/site-packages/httpx/_client.py\", line 897, in post\n"
        f"    return self.request(\"POST\", url, **kwargs)  # correlation={req_id}\n"
        "TimeoutError: downstream telemetry sink ack exceeded 5000ms threshold\n"
    )
    json_blob = json.dumps(
        {
            "event_id": req_id,
            "collector": host,
            "namespace": "secops-telemetry",
            "pod": f"ingest-agent-{idx}",
            "queue_depth": 18 + idx,
            "cpu_pct": round(12.4 + idx * 1.1, 1),
            "tags": ["audit", "heartbeat", "buffer-flush"],
        }
    )

    combined = f"{syslog}\n{stack_trace}\nTELEMETRY_JSON: {json_blob}"
    # Ensure strictly within [600, 1500] chars
    if len(combined) > 1350:
        combined = combined[:1350]
    while len(combined) < 680:
        combined += f"\n2026-11-10T14:23:00Z {host} cron[{pid}]: housekeeping ok"
    return combined


def _build_context_noise_items(
    base_pool: list[dict[str, Any]],
    multihop_items: list[dict[str, Any]],
    rng: random.Random,
    seen_queries: set[str],
) -> list[dict[str, Any]]:
    """Build 10 CONTEXT_NOISE items (3 start, 4 middle, 3 end) with 600-1500 char noise."""
    positions = ["start"] * 3 + ["middle"] * 4 + ["end"] * 3
    # Mix factual + multi-hop sources for varied base questions
    candidate_pool: list[dict[str, Any]] = []
    for mh in multihop_items[::4]:
        candidate_pool.append(mh)
    shuffled_base = list(base_pool)
    rng.shuffle(shuffled_base)
    candidate_pool.extend(shuffled_base)

    noisy_items: list[dict[str, Any]] = []
    used_base_queries: set[str] = set()
    noise_idx = 0

    for cand in candidate_pool:
        if len(noisy_items) >= 10:
            break
        base_q = cand["query"]
        if _norm_key(base_q) in used_base_queries:
            continue

        pos = positions[noise_idx]
        noise_text = _make_synthetic_noise(noise_idx, rng)
        assert 600 <= len(noise_text) <= 1500

        if pos == "start":
            wrapped_q = f"{base_q}\n\n[ATTACHED LOG CONTEXT]\n{noise_text}"
        elif pos == "middle":
            mid = noise_text.find("\n", len(noise_text) // 2)
            if mid == -1:
                mid = len(noise_text) // 2
            part_a = noise_text[:mid].strip()
            part_b = noise_text[mid:].strip()
            wrapped_q = f"{part_a}\n\n{base_q}\n\n{part_b}"
        else:
            wrapped_q = f"[ATTACHED LOG CONTEXT]\n{noise_text}\n\n{base_q}"

        item = {
            **cand,
            "question_type": QuestionType.CONTEXT_NOISE.value,
            "query": wrapped_q,
            "expected_route": ROUTE_SEMANTIC,
            "link_basis": f"context_noise_{pos}",
        }
        if _register_item(noisy_items, seen_queries, item):
            used_base_queries.add(_norm_key(base_q))
            noise_idx += 1

    if len(noisy_items) != 10:
        raise GoldenDatasetValidationError(
            f"Expected 10 CONTEXT_NOISE items, got {len(noisy_items)}."
        )
    return noisy_items


# ─────────────────────────────────────────────────────────────────────────────
# Public build & validate entry points
# ─────────────────────────────────────────────────────────────────────────────


def build_golden_dataset(settings: Settings | None = None) -> list[GoldenItem]:
    """Assemble and atomically write the 150-item golden benchmark dataset."""
    cfg = settings or get_settings()
    corpus_path = cfg.data_processed / "corpus_chunks.parquet"
    map_path = cfg.data_golden / "cve_attack_map.yaml"
    out_path = cfg.data_golden / "golden_150.jsonl"

    rng = random.Random(SEED)
    chunk_by_id, cves, mitre_techs, owasp_entries, _ = _load_corpus(corpus_path)
    mappings = _load_attack_map(map_path)

    log.info(
        "Loaded corpus: %d chunks, %d CVEs, %d MITRE techniques, %d OWASP entries",
        len(chunk_by_id),
        len(cves),
        len(mitre_techs),
        len(owasp_entries),
    )

    seen_queries: set[str] = set()
    factual, base_pool = _build_factual_items(
        cves=cves,
        mitre_techs=mitre_techs,
        chunk_by_id=chunk_by_id,
        rng=rng,
        seen_queries=seen_queries,
    )
    multihop = _build_multi_hop_items(
        mappings=mappings,
        cves=cves,
        mitre_techs=mitre_techs,
        chunk_by_id=chunk_by_id,
        seen_queries=seen_queries,
    )
    conflicts = _build_conflict_trap_items(
        owasp_entries=owasp_entries,
        chunk_by_id=chunk_by_id,
        seen_queries=seen_queries,
    )
    oos = _build_out_of_scope_items(seen_queries=seen_queries)
    malformed = _build_malformed_items(
        cves=cves,
        base_pool=base_pool,
        chunk_by_id=chunk_by_id,
        rng=rng,
        seen_queries=seen_queries,
    )
    context_noise = _build_context_noise_items(
        base_pool=base_pool,
        multihop_items=multihop,
        rng=rng,
        seen_queries=seen_queries,
    )

    raw_items = factual + multihop + conflicts + oos + malformed + context_noise
    golden_items: list[GoldenItem] = []
    for idx, raw in enumerate(raw_items, start=1):
        raw["test_id"] = f"TEST_{idx:03d}"
        golden_items.append(GoldenItem.model_validate(raw))

    # Atomic write
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(".jsonl.tmp")
    lines = [item.model_dump_json() for item in golden_items]
    tmp_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    os.replace(tmp_path, out_path)
    log.info("Written %d items atomically to %s", len(golden_items), out_path)

    # Immediately validate written file
    validate_golden_dataset(cfg)
    return golden_items


def validate_golden_dataset(settings: Settings | None = None) -> list[GoldenItem]:
    """Validate data/golden/golden_150.jsonl against schema, counts, and corpus."""
    cfg = settings or get_settings()
    golden_path = cfg.data_golden / "golden_150.jsonl"
    corpus_path = cfg.data_processed / "corpus_chunks.parquet"

    if not golden_path.exists():
        raise GoldenDatasetValidationError(
            f"Golden dataset not found at {golden_path}. Run --build first.",
            details={"golden_path": str(golden_path)},
        )
    if not corpus_path.exists():
        raise GoldenDatasetValidationError(
            f"Corpus parquet not found at {corpus_path}.",
            details={"corpus_path": str(corpus_path)},
        )

    df = pd.read_parquet(corpus_path)
    chunk_to_doc: dict[str, str] = dict(zip(df["chunk_id"].astype(str), df["doc_id"].astype(str)))
    chunk_to_content: dict[str, str] = dict(zip(df["chunk_id"].astype(str), df["content"].astype(str)))

    items: list[GoldenItem] = []
    failures: list[str] = []

    raw_lines = golden_path.read_text(encoding="utf-8").splitlines()
    for line_no, raw_line in enumerate(raw_lines, start=1):
        stripped = raw_line.strip()
        if not stripped:
            continue
        try:
            obj = json.loads(stripped)
            items.append(GoldenItem.model_validate(obj))
        except (json.JSONDecodeError, ValidationError) as exc:
            failures.append(f"line_{line_no}: schema/JSON error ({exc})")

    if failures:
        raise GoldenDatasetValidationError(
            "Golden dataset failed schema validation.",
            details={"failing_items": failures[:20], "total_failures": len(failures)},
        )

    if len(items) != 150:
        failures.append(f"total_count={len(items)} (expected 150)")

    # Exact counts per QuestionType
    type_counts: dict[QuestionType, int] = {qt: 0 for qt in QuestionType}
    for item in items:
        type_counts[item.question_type] += 1
    for qt, expected_n in _EXPECTED_COUNTS.items():
        if type_counts[qt] != expected_n:
            failures.append(
                f"count({qt.value})={type_counts[qt]} (expected {expected_n})"
            )

    # test_id uniqueness and normalized query uniqueness
    seen_ids: set[str] = set()
    seen_norm_queries: dict[str, str] = {}
    for item in items:
        if item.test_id in seen_ids:
            failures.append(f"{item.test_id}: duplicate test_id")
        seen_ids.add(item.test_id)

        nq = _norm_key(item.query)
        if nq in seen_norm_queries:
            failures.append(
                f"{item.test_id}: duplicate normalized query with {seen_norm_queries[nq]}"
            )
        else:
            seen_norm_queries[nq] = item.test_id

        if item.question_type == QuestionType.OUT_OF_SCOPE:
            if item.expected_source_chunk_ids:
                failures.append(f"{item.test_id}: OUT_OF_SCOPE has non-empty expected_source_chunk_ids")
            if item.expected_doc_ids:
                failures.append(f"{item.test_id}: OUT_OF_SCOPE has non-empty expected_doc_ids")
            if not item.negative_constraint_test:
                failures.append(f"{item.test_id}: OUT_OF_SCOPE negative_constraint_test must be True")
            if item.ground_truth_answer != "INSUFFICIENT_CONTEXT":
                failures.append(f"{item.test_id}: OUT_OF_SCOPE answer must be INSUFFICIENT_CONTEXT")
            continue

        if not item.expected_source_chunk_ids:
            failures.append(f"{item.test_id}: empty expected_source_chunk_ids")
            continue

        missing_cids = [cid for cid in item.expected_source_chunk_ids if cid not in chunk_to_doc]
        if missing_cids:
            failures.append(f"{item.test_id}: missing chunk_ids in parquet: {missing_cids}")
            continue

        derived_docs = {chunk_to_doc[cid] for cid in item.expected_source_chunk_ids}
        if set(item.expected_doc_ids) != derived_docs:
            failures.append(
                f"{item.test_id}: expected_doc_ids {item.expected_doc_ids} != derived {sorted(derived_docs)}"
            )

        concat_text = "\n".join(chunk_to_content[cid] for cid in item.expected_source_chunk_ids).lower()
        missing_ents = [ent for ent in item.required_entities if ent.lower() not in concat_text]
        if missing_ents:
            failures.append(
                f"{item.test_id}: required_entities not found in expected chunks: {missing_ents}"
            )

        if item.question_type == QuestionType.CONFLICT_TRAP:
            if not item.conflict_test or not item.negative_constraint_test:
                failures.append(f"{item.test_id}: CONFLICT_TRAP flags must be True")
            has_v11 = any("_v1.1_" in d for d in item.expected_doc_ids)
            has_v20 = any("_v2.0_" in d for d in item.expected_doc_ids)
            if not (has_v11 and has_v20):
                failures.append(f"{item.test_id}: CONFLICT_TRAP must include both v1.1 and v2.0 doc_ids")

    if failures:
        raise GoldenDatasetValidationError(
            f"Golden dataset validation failed with {len(failures)} error(s).",
            details={"failing_items": failures[:30], "total_failures": len(failures)},
        )

    log.info("Validation PASSED: %d items verified against %s", len(items), corpus_path)
    return items


def main(argv: list[str] | None = None) -> None:
    """CLI entry point supporting --build and --validate."""
    parser = argparse.ArgumentParser(description="Build and validate golden_150.jsonl.")
    parser.add_argument("--build", action="store_true", help="Build golden_150.jsonl from corpus.")
    parser.add_argument("--validate", action="store_true", help="Validate golden_150.jsonl.")
    args = parser.parse_args(argv)

    if not args.build and not args.validate:
        parser.print_help()
        sys.exit(1)

    settings = get_settings()
    if args.build:
        build_golden_dataset(settings)
    elif args.validate:
        validate_golden_dataset(settings)


if __name__ == "__main__":
    main()
