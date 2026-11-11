"""
kavach.retrieval.acquire — Raw corpus acquisition for KAVACH-RAG.

Downloads CVE records from NVD, MITRE ATT&CK techniques/mitigations from the
official STIX GitHub repository, and OWASP LLM Top 10 markdown files.

Usage::

    python -m kavach.retrieval.acquire

Environment variables honoured:
    KAVACH_DATA_RAW    — override raw data directory
    KAVACH_DATA_GOLDEN — override golden data directory
    NVD_API_KEY        — optional NVD API key (raises rate-limit ceiling)

Prerequisite:
    ``data/golden/cve_attack_map.yaml`` must exist (written by Subagent C).
"""
from __future__ import annotations

import gc
import hashlib
import json
import os
import random
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# ── Hardware env vars before any torch/faiss import ──────────────────────────
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("PYTHONUTF8", "1")

_PROJECT_ROOT = Path(__file__).parent.parent.parent
os.environ.setdefault("HF_HOME", str(_PROJECT_ROOT / "models" / "hf_cache"))

import httpx  # noqa: E402
import yaml  # noqa: E402

from kavach.config import get_settings  # noqa: E402
from kavach.exceptions import CorpusIntegrityError  # noqa: E402

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

_NVD_BASE = "https://services.nvd.nist.gov/rest/json/cves/2.0"
_STIX_URL_MASTER = (
    "https://raw.githubusercontent.com/mitre-attack/attack-stix-data"
    "/master/enterprise-attack/enterprise-attack.json"
)
_STIX_URL_MAIN = (
    "https://raw.githubusercontent.com/mitre-attack/attack-stix-data"
    "/main/enterprise-attack/enterprise-attack.json"
)
_GH_OWASP_TREE = (
    "https://api.github.com/repos/OWASP/"
    "www-project-top-10-for-large-language-model-applications"
    "/git/trees/main?recursive=1"
)
_OWASP_RAW_BASE = (
    "https://raw.githubusercontent.com/OWASP/"
    "www-project-top-10-for-large-language-model-applications/main/"
)

_MIN_RESOLVED_CVES = 40
_MAX_MITRE_TECHNIQUES = 30

# Full 49-CVE seed list
_CVE_SEEDS: list[str] = [
    "CVE-2024-3094", "CVE-2021-44228", "CVE-2022-22965", "CVE-2017-5638",
    "CVE-2018-11776", "CVE-2021-26855", "CVE-2022-41040", "CVE-2023-34362",
    "CVE-2018-13379", "CVE-2019-19781", "CVE-2023-3519", "CVE-2023-27997",
    "CVE-2024-3400", "CVE-2022-1388", "CVE-2021-22205", "CVE-2022-26134",
    "CVE-2023-22515", "CVE-2023-20198", "CVE-2021-21972", "CVE-2019-11510",
    "CVE-2020-5902", "CVE-2019-2725", "CVE-2021-41773", "CVE-2021-42013",
    "CVE-2023-46604", "CVE-2024-23897", "CVE-2024-4577", "CVE-2022-47966",
    "CVE-2023-2868", "CVE-2014-6271", "CVE-2024-6387", "CVE-2017-0144",
    "CVE-2019-0708", "CVE-2020-1472", "CVE-2017-11882", "CVE-2022-30190",
    "CVE-2023-38831", "CVE-2023-23397", "CVE-2023-4863", "CVE-2021-4034",
    "CVE-2016-5195", "CVE-2022-0847", "CVE-2024-1086", "CVE-2021-3156",
    "CVE-2024-21626", "CVE-2023-44487", "CVE-2014-0160", "CVE-2021-34527",
    "CVE-2020-0601",
]

_RETRY_STATUSES: frozenset[int] = frozenset([403, 429, 500, 502, 503, 504])

# OWASP LLM Top 10 authoritative listing for both versions
_OWASP_V11_ENTRIES: list[tuple[str, str]] = [
    ("LLM01", "Prompt Injection"),
    ("LLM02", "Insecure Output Handling"),
    ("LLM03", "Training Data Poisoning"),
    ("LLM04", "Model Denial of Service"),
    ("LLM05", "Supply Chain Vulnerabilities"),
    ("LLM06", "Sensitive Information Disclosure"),
    ("LLM07", "Insecure Plugin Design"),
    ("LLM08", "Excessive Agency"),
    ("LLM09", "Overreliance"),
    ("LLM10", "Model Theft"),
]

_OWASP_V20_ENTRIES: list[tuple[str, str]] = [
    ("LLM01", "Prompt Injection"),
    ("LLM02", "Sensitive Information Disclosure"),
    ("LLM03", "Supply Chain"),
    ("LLM04", "Data and Model Poisoning"),
    ("LLM05", "Improper Output Handling"),
    ("LLM06", "Excessive Agency"),
    ("LLM07", "System Prompt Leakage"),
    ("LLM08", "Vector and Embedding Weaknesses"),
    ("LLM09", "Misinformation"),
    ("LLM10", "Unbounded Consumption"),
]

# One-sentence neutral descriptions used ONLY when upstream markdown is unavailable.
_OWASP_NEUTRAL_SENTENCES: dict[str, dict[str, str]] = {
    "v1.1": {
        "LLM01": (
            "LLM01 (Prompt Injection) is the OWASP Top 10 for LLM Applications v1.1 "
            "category covering risks in which crafted inputs alter large language model behavior."
        ),
        "LLM02": (
            "LLM02 (Insecure Output Handling) is the OWASP Top 10 for LLM Applications v1.1 "
            "category covering insufficient validation of model outputs passed to downstream systems."
        ),
        "LLM03": (
            "LLM03 (Training Data Poisoning) is the OWASP Top 10 for LLM Applications v1.1 "
            "category covering manipulation of training datasets that degrades model integrity or safety."
        ),
        "LLM04": (
            "LLM04 (Model Denial of Service) is the OWASP Top 10 for LLM Applications v1.1 "
            "category covering resource exhaustion caused by high-cost model interactions."
        ),
        "LLM05": (
            "LLM05 (Supply Chain Vulnerabilities) is the OWASP Top 10 for LLM Applications v1.1 "
            "category covering risks in third-party datasets, models, and software components."
        ),
        "LLM06": (
            "LLM06 (Sensitive Information Disclosure) is the OWASP Top 10 for LLM Applications v1.1 "
            "category covering unintended exposure of confidential data via model outputs."
        ),
        "LLM07": (
            "LLM07 (Insecure Plugin Design) is the OWASP Top 10 for LLM Applications v1.1 "
            "category covering risks arising from inadequately validated or over-privileged LLM plugins."
        ),
        "LLM08": (
            "LLM08 (Excessive Agency) is the OWASP Top 10 for LLM Applications v1.1 "
            "category covering risks when models are granted overly broad permissions or autonomy."
        ),
        "LLM09": (
            "LLM09 (Overreliance) is the OWASP Top 10 for LLM Applications v1.1 "
            "category covering unverified dependence on potentially inaccurate model outputs."
        ),
        "LLM10": (
            "LLM10 (Model Theft) is the OWASP Top 10 for LLM Applications v1.1 "
            "category covering unauthorized access to or extraction of proprietary model artifacts."
        ),
    },
    "v2.0": {
        "LLM01": (
            "LLM01 (Prompt Injection) is the OWASP Top 10 for LLM Applications v2.0 "
            "category covering risks in which adversarial inputs manipulate model instructions or behavior."
        ),
        "LLM02": (
            "LLM02 (Sensitive Information Disclosure) is the OWASP Top 10 for LLM Applications v2.0 "
            "category covering unintended exposure of confidential or personal data through LLM outputs."
        ),
        "LLM03": (
            "LLM03 (Supply Chain) is the OWASP Top 10 for LLM Applications v2.0 "
            "category covering risks across third-party models, datasets, libraries, and deployment components."
        ),
        "LLM04": (
            "LLM04 (Data and Model Poisoning) is the OWASP Top 10 for LLM Applications v2.0 "
            "category covering manipulation of pre-training, fine-tuning, or embedding data and weights."
        ),
        "LLM05": (
            "LLM05 (Improper Output Handling) is the OWASP Top 10 for LLM Applications v2.0 "
            "category covering inadequate validation or sanitization of LLM outputs prior to downstream use."
        ),
        "LLM06": (
            "LLM06 (Excessive Agency) is the OWASP Top 10 for LLM Applications v2.0 "
            "category covering risks from granting LLM-based systems excessive capabilities or permissions."
        ),
        "LLM07": (
            "LLM07 (System Prompt Leakage) is the OWASP Top 10 for LLM Applications v2.0 "
            "category covering unauthorized disclosure of confidential instructions embedded in system prompts."
        ),
        "LLM08": (
            "LLM08 (Vector and Embedding Weaknesses) is the OWASP Top 10 for LLM Applications v2.0 "
            "category covering risks in vector generation, storage, and retrieval within RAG systems."
        ),
        "LLM09": (
            "LLM09 (Misinformation) is the OWASP Top 10 for LLM Applications v2.0 "
            "category covering false or misleading statements generated by large language models."
        ),
        "LLM10": (
            "LLM10 (Unbounded Consumption) is the OWASP Top 10 for LLM Applications v2.0 "
            "category covering excessive resource, compute, or API usage triggered by LLM workloads."
        ),
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# Jitter backoff
# ─────────────────────────────────────────────────────────────────────────────


def _jitter_sleep(attempt: int, retry_after: float | None = None) -> None:
    """Sleep with full-jitter exponential backoff.

    Args:
        attempt: 0-based attempt number (0 = first retry).
        retry_after: If provided, honour a ``Retry-After`` header value (seconds).
    """
    if retry_after is not None and retry_after > 0:
        delay = min(retry_after, 120.0)
    else:
        delay = random.uniform(0.0, min(60.0, 1.5 * (2 ** attempt)))
    print(f"    [backoff] sleeping {delay:.1f}s (attempt {attempt})", flush=True)
    time.sleep(delay)


# ─────────────────────────────────────────────────────────────────────────────
# HTTP & hashing helpers
# ─────────────────────────────────────────────────────────────────────────────


def _get_with_retry(
    client: httpx.Client,
    url: str,
    *,
    params: dict[str, str] | None = None,
    headers: dict[str, str] | None = None,
    max_attempts: int = 6,
) -> httpx.Response | None:
    """GET *url* with jitter-backoff retry on transient errors.

    Args:
        client: Shared ``httpx.Client`` instance.
        url: Target URL.
        params: Optional query parameters.
        headers: Optional per-request HTTP headers.
        max_attempts: Maximum total attempts (not retries).

    Returns:
        The successful :class:`httpx.Response`, or ``None`` if the resource
        returned HTTP 404 (not found — not retriable).

    Raises:
        httpx.HTTPStatusError: If a non-retriable, non-404 error persists after
            all attempts.
    """
    for attempt in range(max_attempts):
        try:
            resp = client.get(url, params=params, headers=headers, timeout=60.0)
            if resp.status_code == 404:
                return None
            if resp.status_code in _RETRY_STATUSES:
                retry_after: float | None = None
                if "Retry-After" in resp.headers:
                    try:
                        retry_after = float(resp.headers["Retry-After"])
                    except ValueError:
                        retry_after = None
                if attempt < max_attempts - 1:
                    _jitter_sleep(attempt, retry_after)
                    continue
                resp.raise_for_status()
            resp.raise_for_status()
            return resp
        except (httpx.TimeoutException, httpx.NetworkError) as exc:
            if attempt < max_attempts - 1:
                print(f"    [warn] network error on attempt {attempt}: {exc}", flush=True)
                _jitter_sleep(attempt)
                continue
            raise
    return None


def _sha256_of_bytes(data: bytes) -> str:
    """Return hex SHA-256 digest of *data*."""
    return hashlib.sha256(data).hexdigest()


def _aggregate_sha256(hashes: list[str]) -> str:
    """Compute a deterministic SHA-256 over a sorted list of hex digests."""
    joined = "\n".join(sorted(hashes)).encode("utf-8")
    return hashlib.sha256(joined).hexdigest()


# ─────────────────────────────────────────────────────────────────────────────
# Prerequisite check
# ─────────────────────────────────────────────────────────────────────────────


def _load_cve_attack_data(
    golden_dir: Path | None = None,
) -> tuple[dict[str, list[str]], set[str]]:
    """Load ``data/golden/cve_attack_map.yaml`` in either schema format.

    Supports:
      1. List-based schema (Subagent C format)::

             mappings:
               - technique_id: "T1190"
                 cve_ids: ["CVE-2021-44228", ...]

      2. Dict-based fallback schema::

             CVE-2024-3094:
               techniques: ["T1195.002"]

    Args:
        golden_dir: Optional override for the golden data directory.

    Returns:
        Tuple ``(cve_to_techniques, all_technique_ids)``.

    Raises:
        CorpusIntegrityError: If ``cve_attack_map.yaml`` is absent or yields no
            technique mappings.
    """
    base_golden = golden_dir or (get_settings().data_golden)
    map_path = base_golden / "cve_attack_map.yaml"
    if not map_path.exists():
        raise CorpusIntegrityError(
            "cve_attack_map.yaml not found — Subagent C must write it before acquire.py runs",
            details={"expected_path": str(map_path)},
        )

    with map_path.open(encoding="utf-8") as fh:
        raw: Any = yaml.safe_load(fh) or {}

    cve_to_techs: dict[str, list[str]] = {}
    all_techs: set[str] = set()

    if isinstance(raw, dict) and isinstance(raw.get("mappings"), list):
        for entry in raw["mappings"]:
            if not isinstance(entry, dict):
                continue
            tech_id = str(entry.get("technique_id", "")).strip()
            if not tech_id:
                continue
            all_techs.add(tech_id)
            for cve_id in entry.get("cve_ids") or []:
                cid = str(cve_id).strip()
                if not cid:
                    continue
                techs = cve_to_techs.setdefault(cid, [])
                if tech_id not in techs:
                    techs.append(tech_id)
    elif isinstance(raw, dict):
        for cve_id, entry in raw.items():
            if isinstance(entry, dict) and isinstance(entry.get("techniques"), list):
                cid = str(cve_id).strip()
                techs = [str(t).strip() for t in entry["techniques"] if str(t).strip()]
                cve_to_techs[cid] = techs
                all_techs.update(techs)

    if not all_techs:
        raise CorpusIntegrityError(
            "cve_attack_map.yaml contained zero technique mappings.",
            details={"path": str(map_path)},
        )

    return cve_to_techs, all_techs


def _load_cve_attack_map(golden_dir: Path | None = None) -> dict[str, list[str]]:
    """Load ``data/golden/cve_attack_map.yaml`` and return a CVE→technique mapping.

    Args:
        golden_dir: Optional override for the golden data directory.

    Returns:
        A dict mapping CVE IDs to lists of MITRE ATT&CK technique IDs.

    Raises:
        CorpusIntegrityError: If the file does not exist or has no mappings.
    """
    cve_to_techs, _ = _load_cve_attack_data(golden_dir)
    return cve_to_techs


# ─────────────────────────────────────────────────────────────────────────────
# NVD CVE acquisition
# ─────────────────────────────────────────────────────────────────────────────


def _has_valid_vulnerabilities(payload: bytes) -> bool:
    """Return True if *payload* is NVD JSON containing a non-empty vulnerabilities list."""
    try:
        parsed = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return False
    vulns = parsed.get("vulnerabilities")
    return isinstance(vulns, list) and len(vulns) > 0


def acquire_cves(
    out_dir: Path,
    client: httpx.Client,
    provenance: dict[str, Any],
    extra_cve_ids: list[str] | None = None,
) -> None:
    """Download each CVE from the NVD 2.0 API and verify >= 40 resolved records.

    Writes one ``<CVE-ID>.json`` file per record into *out_dir*.  Valid cached
    files are preserved and included in provenance.  Raises
    :class:`CorpusIntegrityError` if fewer than 40 CVE records resolve.

    Args:
        out_dir: Directory for JSON output (``data/raw/cve_json/``).
        client: Shared ``httpx.Client``.
        provenance: Mutable provenance dict updated under ``provenance["sources"]["nvd_cve"]``.
        extra_cve_ids: Additional CVE IDs (e.g. from ``cve_attack_map.yaml``) to merge.

    Raises:
        CorpusIntegrityError: If fewer than 40 CVEs resolve with valid NVD records.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    api_key = os.environ.get("NVD_API_KEY", "").strip()
    req_headers: dict[str, str] = {"Accept": "application/json"}
    if api_key:
        req_headers["apiKey"] = api_key

    # NVD rate limit: 5 req / 30s unauthenticated (6.5s); 50 req / 30s with key (0.7s).
    inter_request_delay = 0.7 if api_key else 6.5

    cve_list: list[str] = list(_CVE_SEEDS)
    if extra_cve_ids:
        seen = set(cve_list)
        for cid in extra_cve_ids:
            if cid not in seen:
                cve_list.append(cid)
                seen.add(cid)

    records: dict[str, dict[str, Any]] = {}
    resolved_hashes: list[str] = []
    latest_fetched_at = datetime.now(tz=timezone.utc).isoformat()

    print(
        f"[NVD] Resolving {len(cve_list)} CVEs "
        f"(api_key={'set' if api_key else 'unset'}, delay={inter_request_delay}s) …",
        flush=True,
    )

    for idx, cve_id in enumerate(cve_list, start=1):
        dest = out_dir / f"{cve_id}.json"
        url = f"{_NVD_BASE}?cveId={cve_id}"

        if dest.exists():
            cached_bytes = dest.read_bytes()
            if _has_valid_vulnerabilities(cached_bytes):
                sha = _sha256_of_bytes(cached_bytes)
                mtime = datetime.fromtimestamp(dest.stat().st_mtime, tz=timezone.utc).isoformat()
                records[cve_id] = {
                    "status": "ok",
                    "cached": True,
                    "url": url,
                    "sha256": sha,
                    "fetched_at": mtime,
                    "size_bytes": len(cached_bytes),
                }
                resolved_hashes.append(sha)
                print(f"  [{idx:02d}/{len(cve_list)}] {cve_id} — cached OK ({len(cached_bytes):,} bytes)")
                continue
            print(f"  [{idx:02d}/{len(cve_list)}] {cve_id} — cached file invalid, re-fetching …", flush=True)

        print(f"  [{idx:02d}/{len(cve_list)}] {cve_id} …", end=" ", flush=True)
        resp = _get_with_retry(
            client,
            _NVD_BASE,
            params={"cveId": cve_id},
            headers=req_headers,
        )

        if resp is None:
            print("SKIP (404 not found)", flush=True)
            records[cve_id] = {"status": "not_found", "url": url}
        else:
            payload: bytes = resp.content
            if not _has_valid_vulnerabilities(payload):
                print("SKIP (empty vulnerabilities array)", flush=True)
                records[cve_id] = {"status": "empty_vulnerabilities", "url": str(resp.url)}
            else:
                _atomic_write_bytes(dest, payload)
                sha = _sha256_of_bytes(payload)
                now_iso = datetime.now(tz=timezone.utc).isoformat()
                latest_fetched_at = now_iso
                records[cve_id] = {
                    "status": "ok",
                    "cached": False,
                    "url": str(resp.url),
                    "sha256": sha,
                    "fetched_at": now_iso,
                    "size_bytes": len(payload),
                }
                resolved_hashes.append(sha)
                print(f"OK ({len(payload):,} bytes)", flush=True)

        if idx < len(cve_list) and inter_request_delay > 0:
            time.sleep(inter_request_delay)

    resolved_count = len(resolved_hashes)
    sources = provenance.setdefault("sources", {})
    sources["nvd_cve"] = {
        "url": _NVD_BASE,
        "sha256": _aggregate_sha256(resolved_hashes) if resolved_hashes else "",
        "fetched_at": latest_fetched_at,
        "record_count": resolved_count,
        "records": records,
    }

    if resolved_count < _MIN_RESOLVED_CVES:
        missing = [cid for cid, meta in records.items() if meta.get("status") != "ok"]
        raise CorpusIntegrityError(
            f"Resolved only {resolved_count} NVD CVE records; minimum {_MIN_RESOLVED_CVES} required.",
            details={"resolved_count": resolved_count, "missing_cves": missing},
        )
    print(f"  [NVD] Verified {resolved_count}/{len(cve_list)} resolved CVE records.", flush=True)


# ─────────────────────────────────────────────────────────────────────────────
# MITRE ATT&CK acquisition
# ─────────────────────────────────────────────────────────────────────────────


def _strip_citations(text: str) -> str:
    """Remove ``(Citation: ...)`` references from STIX description text."""
    return re.sub(r"\(Citation:[^)]*\)", "", text).strip()


def _demote_embedded_headings(text: str) -> str:
    """Demote H1–H3 ATX headings inside STIX text to H4 so splitters stay exact."""
    out_lines: list[str] = []
    in_fence = False
    for line in text.splitlines():
        stripped = line.lstrip()
        if stripped.startswith("```"):
            in_fence = not in_fence
            out_lines.append(line)
            continue
        if not in_fence:
            match = re.match(r"^(#{1,3})\s+(.+)$", stripped)
            if match:
                out_lines.append(f"#### {match.group(2).strip()}")
                continue
        out_lines.append(line)
    return "\n".join(out_lines)


def _extract_external_id(obj: dict[str, Any]) -> str:
    """Extract the first ATT&CK external ID (T-ID, M-ID, TA-ID) from a STIX object."""
    for ref in obj.get("external_references", []):
        if ref.get("source_name") == "mitre-attack":
            return str(ref.get("external_id", "")).strip()
    return ""


def _build_stix_index(
    stix_data: dict[str, Any],
) -> dict[str, Any]:
    """Index STIX objects in a single linear pass for fast deterministic rendering.

    Args:
        stix_data: Parsed JSON from the enterprise-attack STIX bundle.

    Returns:
        Dictionary containing indexed attack-patterns, mitigations, tactics,
        mitigation relationship descriptions, top-level Initial Access IDs,
        and the STIX collection version string.
    """
    obj_by_id: dict[str, dict[str, Any]] = {}
    ext_id_map: dict[str, str] = {}
    attack_version = "attack_v16"

    for obj in stix_data.get("objects", []):
        if obj.get("type") == "x-mitre-collection" and obj.get("x_mitre_version"):
            attack_version = f"attack_v{obj['x_mitre_version']}"
        stix_id: str = obj.get("id", "")
        if not stix_id:
            continue
        if obj.get("revoked") or obj.get("x_mitre_deprecated"):
            continue
        obj_by_id[stix_id] = obj
        ext_id = _extract_external_id(obj)
        if ext_id:
            ext_id_map[stix_id] = ext_id

    tactic_by_shortname: dict[str, tuple[str, str]] = {}
    attack_pattern_by_ext: dict[str, dict[str, Any]] = {}
    mitigation_by_ext: dict[str, dict[str, Any]] = {}

    for stix_id, obj in obj_by_id.items():
        obj_type = obj.get("type")
        ext_id = ext_id_map.get(stix_id, "")
        if obj_type == "x-mitre-tactic":
            shortname = str(obj.get("x_mitre_shortname", "")).strip().lower()
            name = str(obj.get("name", shortname.replace("-", " ").title())).strip()
            if shortname and ext_id:
                tactic_by_shortname[shortname] = (ext_id, name)
        elif obj_type == "attack-pattern" and ext_id:
            attack_pattern_by_ext[ext_id] = obj
        elif obj_type == "course-of-action" and ext_id:
            mitigation_by_ext[ext_id] = obj

    tech_tactic: dict[str, tuple[str, str]] = {}
    initial_access_top_level: list[str] = []

    for ext_id, obj in attack_pattern_by_ext.items():
        phases = [
            p for p in obj.get("kill_chain_phases", [])
            if p.get("kill_chain_name", "mitre-attack") == "mitre-attack"
        ]
        if not phases:
            phases = obj.get("kill_chain_phases", [])
        phase_shortnames = [str(p.get("phase_name", "")).strip().lower() for p in phases]

        if phase_shortnames:
            first_short = phase_shortnames[0]
            if first_short in tactic_by_shortname:
                tech_tactic[ext_id] = tactic_by_shortname[first_short]
            else:
                tech_tactic[ext_id] = ("TA-Unknown", first_short.replace("-", " ").title())

        is_sub = bool(obj.get("x_mitre_is_subtechnique")) or ("." in ext_id)
        if not is_sub and "initial-access" in phase_shortnames:
            initial_access_top_level.append(ext_id)

    tech_mitigations: dict[str, list[str]] = {}
    mit_rel_desc: dict[tuple[str, str], str] = {}
    for obj in obj_by_id.values():
        if obj.get("type") != "relationship" or obj.get("relationship_type") != "mitigates":
            continue
        mit_ext = ext_id_map.get(obj.get("source_ref", ""), "")
        tech_ext = ext_id_map.get(obj.get("target_ref", ""), "")
        if not (mit_ext and tech_ext):
            continue
        if mit_ext not in mitigation_by_ext or tech_ext not in attack_pattern_by_ext:
            continue
        mits = tech_mitigations.setdefault(tech_ext, [])
        if mit_ext not in mits:
            mits.append(mit_ext)
        desc = _demote_embedded_headings(_strip_citations(str(obj.get("description", ""))))
        if desc:
            mit_rel_desc[(mit_ext, tech_ext)] = desc

    return {
        "attack_version": attack_version,
        "attack_pattern_by_ext": attack_pattern_by_ext,
        "mitigation_by_ext": mitigation_by_ext,
        "tech_tactic": tech_tactic,
        "tech_mitigations": tech_mitigations,
        "mit_rel_desc": mit_rel_desc,
        "initial_access_top_level": sorted(set(initial_access_top_level)),
    }


def _render_technique_markdown(tech_ext_id: str, index: dict[str, Any]) -> str | None:
    """Render a MITRE technique and its mitigations as structured Markdown.

    Uses ``# {TA-ID}: {tactic}`` / ``## {T-ID}: {name}`` / ``### {M-ID}: {mitigation}``.
    """
    tech_obj: dict[str, Any] | None = index["attack_pattern_by_ext"].get(tech_ext_id)
    if tech_obj is None:
        return None

    tactic_id, tactic_name = index["tech_tactic"].get(tech_ext_id, ("TA-Unknown", "Unknown Tactic"))
    tech_name = str(tech_obj.get("name", "")).strip()
    tech_desc = _demote_embedded_headings(_strip_citations(str(tech_obj.get("description", ""))))
    platforms: list[str] = [str(p) for p in tech_obj.get("x_mitre_platforms", [])]

    lines: list[str] = [
        f"# {tactic_id}: {tactic_name}",
        f"## {tech_ext_id}: {tech_name}",
    ]
    if "." in tech_ext_id:
        parent_id = tech_ext_id.split(".", 1)[0]
        parent_obj = index["attack_pattern_by_ext"].get(parent_id)
        if parent_obj:
            lines.append(f"Parent Technique: {parent_id}: {parent_obj.get('name', '')}")
    if tech_desc:
        lines.append(tech_desc)
    lines.append(f"Platforms: {', '.join(platforms) if platforms else 'N/A'}")

    for mit_ext_id in sorted(index["tech_mitigations"].get(tech_ext_id, [])):
        mit_obj = index["mitigation_by_ext"].get(mit_ext_id)
        if mit_obj is None:
            continue
        mit_name = str(mit_obj.get("name", "")).strip()
        rel_desc = index["mit_rel_desc"].get((mit_ext_id, tech_ext_id), "")
        if not rel_desc:
            rel_desc = _demote_embedded_headings(_strip_citations(str(mit_obj.get("description", ""))))
        lines.append(f"### {mit_ext_id}: {mit_name}")
        if rel_desc:
            lines.append(rel_desc)

    return "\n".join(lines) + "\n"


def acquire_mitre(
    out_dir: Path,
    client: httpx.Client,
    provenance: dict[str, Any],
    technique_ids: set[str],
) -> None:
    """Download MITRE ATT&CK STIX data and render up to 30 technique Markdown files.

    Selects:
      (a) Every technique in ``cve_attack_map.yaml`` plus the parent technique
          for any sub-technique (e.g. ``T1548`` for ``T1548.003``).
      (b) Top-level Initial Access tactic techniques as distractors, capped at
          30 total technique files.

    Args:
        out_dir: Directory for Markdown output (``data/raw/mitre_markdown/``).
        client: Shared ``httpx.Client``.
        provenance: Mutable provenance dict updated under ``provenance["sources"]["mitre_attack"]``.
        technique_ids: Set of technique IDs from ``cve_attack_map.yaml``.

    Raises:
        CorpusIntegrityError: If STIX cannot be downloaded or any required
            technique from ``cve_attack_map.yaml`` is absent from STIX.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    if not technique_ids:
        raise CorpusIntegrityError(
            "No technique IDs provided to acquire_mitre().",
            details={"technique_ids": []},
        )

    print("[MITRE] Downloading enterprise-attack STIX bundle …", flush=True)
    stix_resp = _get_with_retry(client, _STIX_URL_MASTER, headers={"Accept": "application/json"})
    if stix_resp is None:
        print("  [warn] master branch returned 404, trying main …", flush=True)
        stix_resp = _get_with_retry(client, _STIX_URL_MAIN, headers={"Accept": "application/json"})
    if stix_resp is None:
        raise CorpusIntegrityError("Could not download MITRE ATT&CK enterprise STIX bundle.")

    stix_bytes = stix_resp.content
    stix_sha = _sha256_of_bytes(stix_bytes)
    stix_url = str(stix_resp.url)
    fetched_at = datetime.now(tz=timezone.utc).isoformat()
    stix_data: dict[str, Any] = json.loads(stix_bytes.decode("utf-8"))
    print(f"  STIX bundle: {len(stix_bytes):,} bytes, sha256={stix_sha[:16]}…", flush=True)

    index = _build_stix_index(stix_data)
    del stix_data
    del stix_bytes
    gc.collect()

    required: set[str] = set(technique_ids)
    for tid in list(required):
        if "." in tid:
            required.add(tid.split(".", 1)[0])

    missing_required = sorted(t for t in required if t not in index["attack_pattern_by_ext"])
    if missing_required:
        raise CorpusIntegrityError(
            "Required MITRE ATT&CK techniques missing from STIX bundle.",
            details={"missing_techniques": missing_required},
        )

    distractors = [
        tid for tid in index["initial_access_top_level"]
        if tid not in required
    ]
    selected_ids: list[str] = sorted(required) + distractors
    selected_ids = selected_ids[:_MAX_MITRE_TECHNIQUES]
    selected_set = set(selected_ids)

    # Ensure cap of 30 files by removing any stale non-selected T*.md files.
    for existing_md in out_dir.glob("T*.md"):
        if existing_md.stem not in selected_set:
            existing_md.unlink()

    records: dict[str, dict[str, Any]] = {}
    written = 0
    for tech_id in selected_ids:
        dest = out_dir / f"{tech_id}.md"
        md = _render_technique_markdown(tech_id, index)
        if md is None:
            raise CorpusIntegrityError(
                f"Failed to render MITRE technique {tech_id}.",
                details={"technique_id": tech_id},
            )
        encoded = md.encode("utf-8")
        sha = _sha256_of_bytes(encoded)
        if not dest.exists() or _sha256_of_bytes(dest.read_bytes()) != sha:
            _atomic_write_text(dest, md)
            written += 1
        records[tech_id] = {
            "status": "ok",
            "role": "required_or_parent" if tech_id in required else "initial_access_distractor",
            "sha256": sha,
            "fetched_at": fetched_at,
            "size_bytes": len(encoded),
        }

    sources = provenance.setdefault("sources", {})
    sources["mitre_attack"] = {
        "url": stix_url,
        "sha256": stix_sha,
        "fetched_at": fetched_at,
        "record_count": len(records),
        "attack_version": index["attack_version"],
        "records": records,
    }
    print(
        f"  [MITRE] {len(records)} technique file(s) ready "
        f"({written} written/updated, {len(required)} required+parents, "
        f"{len(records) - len(required)} distractors).",
        flush=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# OWASP LLM Top 10 acquisition
# ─────────────────────────────────────────────────────────────────────────────


def _strip_yaml_frontmatter(text: str) -> str:
    """Remove leading YAML frontmatter block ``--- ... ---`` if present."""
    lines = text.splitlines()
    if lines and lines[0].strip() == "---":
        for idx in range(1, len(lines)):
            if lines[idx].strip() == "---":
                return "\n".join(lines[idx + 1 :]).lstrip("\n")
    return text


def _normalize_owasp_markdown(
    raw_text: str,
    version: str,
    llm_id: str,
    title: str,
) -> str:
    """Normalize upstream OWASP markdown to canonical H1/H2/H3 structure.

    Produces::

        <!-- source_provenance: github_upstream -->
        # OWASP Top 10 for LLM Applications {version}
        ## {llm_id}: {title}
        ### {section}
        ...
    """
    cleaned = _strip_yaml_frontmatter(raw_text.replace("\r\n", "\n").replace("\r", "\n"))
    cleaned = re.sub(r"<!--\s*source_provenance:.*?-->\s*", "", cleaned, flags=re.IGNORECASE)

    lines = cleaned.splitlines()
    in_fence = False
    first_heading_idx: int | None = None
    first_heading_level: int = 0
    has_h2_body = False

    for idx, line in enumerate(lines):
        stripped = line.lstrip()
        if stripped.startswith("```"):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        match = re.match(r"^(#{1,6})\s+(.+)$", stripped)
        if match:
            level = len(match.group(1))
            if first_heading_idx is None:
                first_heading_idx = idx
                first_heading_level = level
            elif level == 2:
                has_h2_body = True

    # Determine if the first heading represents the entry title itself (e.g. '# LLM01...' or '## LLM01...').
    drop_first_heading = False
    if first_heading_idx is not None:
        first_text = re.sub(r"^#{1,6}\s+", "", lines[first_heading_idx].lstrip()).strip()
        if (
            first_heading_level in (1, 2)
            and (
                re.search(rf"\b{re.escape(llm_id)}\b", first_text, re.IGNORECASE)
                or first_heading_level == 1
                or title.lower() in first_text.lower()
            )
        ):
            drop_first_heading = True

    body_lines: list[str] = []
    in_fence = False
    for idx, line in enumerate(lines):
        if drop_first_heading and idx == first_heading_idx:
            continue
        stripped = line.lstrip()
        if stripped.startswith("```"):
            in_fence = not in_fence
            body_lines.append(line)
            continue
        if in_fence:
            body_lines.append(line)
            continue
        match = re.match(r"^(#{1,6})\s+(.+)$", stripped)
        if match:
            level = len(match.group(1))
            heading_text = match.group(2).strip()
            if level <= 2:
                new_level = 3
            elif has_h2_body and first_heading_level == 1:
                new_level = min(6, level + 1)
            else:
                new_level = min(6, max(3, level))
            body_lines.append(f"{'#' * new_level} {heading_text}")
        else:
            body_lines.append(line)

    body_text = "\n".join(body_lines).strip()
    if not body_text:
        body_text = f"### Overview\n{_OWASP_NEUTRAL_SENTENCES[version][llm_id]}"

    header = (
        f"<!-- source_provenance: github_upstream -->\n"
        f"# OWASP Top 10 for LLM Applications {version}\n"
        f"## {llm_id}: {title}\n"
    )
    return header + body_text + "\n"


def _build_authored_summary(version: str, llm_id: str, title: str) -> str:
    """Return a short neutral one-sentence fallback summary tagged authored_summary."""
    sentence = _OWASP_NEUTRAL_SENTENCES[version][llm_id]
    return (
        f"<!-- source_provenance: authored_summary -->\n"
        f"# OWASP Top 10 for LLM Applications {version}\n"
        f"## {llm_id}: {title}\n"
        f"### Overview\n"
        f"{sentence}\n"
    )


def _match_owasp_tree_paths(
    md_paths: dict[str, str],
    folder_token: str,
    llm_id: str,
) -> list[tuple[str, str]]:
    """Return ``(repo_path, raw_url)`` matches strictly filtered to *folder_token*."""
    id_regex = re.compile(rf"(?i)(?:^|[^0-9a-z]){re.escape(llm_id)}(?:[^0-9]|$)")
    matches: list[tuple[str, str]] = []
    for path, url in md_paths.items():
        lower_path = path.lower()
        if folder_token not in lower_path:
            continue
        if "translations/" in lower_path or "/translations/" in lower_path:
            continue
        filename = Path(path).name
        if id_regex.search(filename):
            matches.append((path, url))
    matches.sort(key=lambda item: (len(item[0]), item[0]))
    return matches


def acquire_owasp(
    out_dir_v11: Path,
    out_dir_v20: Path,
    client: httpx.Client,
    provenance: dict[str, Any],
) -> None:
    """Download OWASP LLM Top 10 markdown for v1.1 and v2.0.

    Strictly filters the upstream GitHub tree to ``1_1_vulns`` (v1.1) and
    ``2_0_vulns`` (v2.0), normalizes headers, and falls back to one-sentence
    neutral summaries tagged ``<!-- source_provenance: authored_summary -->``
    for any missing entries.

    Args:
        out_dir_v11: Output directory for v1.1 markdown files.
        out_dir_v20: Output directory for v2.0 markdown files.
        client: Shared ``httpx.Client``.
        provenance: Mutable provenance dict updated with OWASP metadata.
    """
    out_dir_v11.mkdir(parents=True, exist_ok=True)
    out_dir_v20.mkdir(parents=True, exist_ok=True)

    print("[OWASP] Fetching repository file tree …", flush=True)
    tree_resp = _get_with_retry(
        client,
        _GH_OWASP_TREE,
        headers={"Accept": "application/vnd.github+json"},
    )
    if tree_resp is None:
        print("  [warn] GitHub API returned 404 — using authored summaries for missing entries", flush=True)
        file_tree: list[dict[str, Any]] = []
    else:
        tree_data = tree_resp.json()
        file_tree = tree_data.get("tree", [])

    md_paths: dict[str, str] = {}
    for item in file_tree:
        if item.get("type", "blob") != "blob":
            continue
        path_str = str(item.get("path", ""))
        if path_str.lower().endswith(".md"):
            md_paths[path_str] = _OWASP_RAW_BASE + path_str

    def _acquire_version(
        version: str,
        entries: list[tuple[str, str]],
        folder_token: str,
        out_dir: Path,
    ) -> dict[str, Any]:
        records: dict[str, dict[str, Any]] = {}
        hashes: list[str] = []
        provenance_kinds: set[str] = set()
        latest_ts = datetime.now(tz=timezone.utc).isoformat()

        print(f"  [OWASP {version}] processing {len(entries)} entries …", flush=True)
        for llm_id, title in entries:
            dest = out_dir / f"{llm_id}.md"
            expected_h1 = f"# OWASP Top 10 for LLM Applications {version}"

            # Reuse cached file only if it already has normalized canonical headers.
            if dest.exists():
                existing_text = dest.read_text(encoding="utf-8")
                if expected_h1 in existing_text:
                    raw_bytes = dest.read_bytes()
                    sha = _sha256_of_bytes(raw_bytes)
                    prov_tag = (
                        "authored_summary"
                        if "source_provenance: authored_summary" in existing_text
                        else "github_upstream"
                    )
                    provenance_kinds.add(prov_tag)
                    hashes.append(sha)
                    mtime = datetime.fromtimestamp(dest.stat().st_mtime, tz=timezone.utc).isoformat()
                    records[llm_id] = {
                        "url": _OWASP_RAW_BASE if prov_tag == "github_upstream" else None,
                        "sha256": sha,
                        "fetched_at": mtime,
                        "record_count": 1,
                        "source_provenance": prov_tag,
                        "cached": True,
                    }
                    print(f"    {version}/{llm_id} — cached OK ({prov_tag})")
                    continue

            matches = _match_owasp_tree_paths(md_paths, folder_token, llm_id)
            normalized_md: str | None = None
            source_url: str | None = None
            source_prov = "authored_summary"

            for _, candidate_url in matches:
                resp = _get_with_retry(
                    client,
                    candidate_url,
                    headers={"Accept": "text/plain, text/markdown, */*"},
                )
                if resp is not None and resp.content.strip():
                    raw_text = resp.content.decode("utf-8", errors="replace")
                    normalized_md = _normalize_owasp_markdown(raw_text, version, llm_id, title)
                    source_url = str(resp.url)
                    source_prov = "github_upstream"
                    break

            if normalized_md is None:
                normalized_md = _build_authored_summary(version, llm_id, title)
                source_url = None
                source_prov = "authored_summary"

            _atomic_write_text(dest, normalized_md)
            encoded = normalized_md.encode("utf-8")
            sha = _sha256_of_bytes(encoded)
            now_iso = datetime.now(tz=timezone.utc).isoformat()
            latest_ts = now_iso
            hashes.append(sha)
            provenance_kinds.add(source_prov)
            records[llm_id] = {
                "url": source_url,
                "sha256": sha,
                "fetched_at": now_iso,
                "record_count": 1,
                "source_provenance": source_prov,
                "size_bytes": len(encoded),
            }
            print(f"    {version}/{llm_id} — {source_prov} ({len(encoded):,} bytes)", flush=True)

        if len(provenance_kinds) == 1:
            summary_prov = next(iter(provenance_kinds))
        else:
            summary_prov = "mixed"

        return {
            "url": _GH_OWASP_TREE,
            "sha256": _aggregate_sha256(hashes),
            "fetched_at": latest_ts,
            "record_count": len(records),
            "source_provenance": summary_prov,
            "records": records,
        }

    sources = provenance.setdefault("sources", {})
    sources["owasp_llm_v1.1"] = _acquire_version("v1.1", _OWASP_V11_ENTRIES, "1_1_vulns", out_dir_v11)
    sources["owasp_llm_v2.0"] = _acquire_version("v2.0", _OWASP_V20_ENTRIES, "2_0_vulns", out_dir_v20)


# ─────────────────────────────────────────────────────────────────────────────
# Atomic I/O helpers
# ─────────────────────────────────────────────────────────────────────────────


def _atomic_write_bytes(dest: Path, data: bytes) -> None:
    """Write *data* to *dest* atomically via a sibling temp file."""
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    tmp.write_bytes(data)
    os.replace(str(tmp), str(dest))


def _atomic_write_text(dest: Path, text: str) -> None:
    """Write *text* (UTF-8) to *dest* atomically via a sibling temp file."""
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(str(tmp), str(dest))


def _write_provenance(provenance: dict[str, Any], raw_dir: Path) -> None:
    """Write ``data/raw/PROVENANCE.json`` atomically."""
    prov_path = raw_dir / "PROVENANCE.json"
    payload = json.dumps(provenance, indent=2, ensure_ascii=False)
    _atomic_write_text(prov_path, payload)
    print(f"[provenance] Written to {prov_path}", flush=True)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────


def main() -> None:
    """Run the full acquisition pipeline: CVE → MITRE → OWASP → PROVENANCE."""
    settings = get_settings()
    cve_attack_map, all_technique_ids = _load_cve_attack_data(settings.data_golden)
    print(
        f"[acquire] cve_attack_map loaded: {len(cve_attack_map)} CVE entries, "
        f"{len(all_technique_ids)} unique technique IDs",
        flush=True,
    )

    provenance: dict[str, Any] = {
        "generated_at": datetime.now(tz=timezone.utc).isoformat(),
        "sources": {},
    }

    raw_dir = settings.data_raw
    headers: dict[str, str] = {
        "User-Agent": "KAVACH-RAG/1.0 (academic security research)",
        "Accept": "application/json",
    }

    with httpx.Client(
        headers=headers,
        follow_redirects=True,
        http2=True,
        limits=httpx.Limits(max_keepalive_connections=50, keepalive_expiry=60.0),
    ) as client:
        acquire_cves(
            raw_dir / "cve_json",
            client,
            provenance,
            extra_cve_ids=list(cve_attack_map.keys()),
        )
        acquire_mitre(
            raw_dir / "mitre_markdown",
            client,
            provenance,
            all_technique_ids,
        )
        acquire_owasp(
            raw_dir / "owasp_markdown" / "v1.1",
            raw_dir / "owasp_markdown" / "v2.0",
            client,
            provenance,
        )

    _write_provenance(provenance, raw_dir)
    print("[acquire] Acquisition complete.", flush=True)


if __name__ == "__main__":
    main()
