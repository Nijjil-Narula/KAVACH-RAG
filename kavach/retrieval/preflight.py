"""
kavach.retrieval.preflight — Model availability verification and speed benchmark.

Runnable as: python -m kavach.retrieval.preflight

Steps:
  1. Verify and download all three model IDs (Tier 1 router, Tier 2 embedder,
     Tier 4 reranker) into models/hf_cache via HF_HOME.
  2. Inspect embedder config and model card / tokenizer to determine output
     dimension, pooling method, and padding side.
  3. Write models/embedder_notes.md.
  4. Benchmark 16 representative chunks + 16 queries per tier (Tier 2 embedder,
     Tier 1 router, Tier 4 reranker) at 2 CPU threads, freeing weights between
     tiers and tracking RSS and VRAM isolation.
  5. Project corpus embedding ETAs for w512 and w256.
  6. Write reports/preflight.json atomically.
  7. Print summary and await orchestrator go-ahead (AMENDMENT A6).
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

from kavach.runtime_guard import apply_hardware_guard, rss_mb, vram_probe  # noqa: E402

import gc
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import faiss  # noqa: F401 — imported strictly after torch
from huggingface_hub import ModelCard, hf_hub_download, model_info
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

from kavach.config import Settings, get_settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("kavach.preflight")

_FALLBACK_EMBEDDER = "codefuse-ai/F2LLM-0.6B"
_FALLBACK_RERANKER = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# ── 16 representative cybersecurity chunks + 16 queries ────────────────────────
_BENCH_PASSAGES: list[str] = [
    (
        "CVE-2024-3094 describes a supply chain attack embedded in the XZ Utils "
        "compression library. A backdoor was inserted into liblzma via malicious "
        "upstream commits affecting versions 5.6.0 and 5.6.1. The compromised "
        "library was linked by several systemd builds on Linux, allowing an "
        "unauthenticated remote attacker to execute arbitrary code with the "
        "privileges of the sshd process. CVSS base score 10.0 CRITICAL."
    ),
    (
        "MITRE ATT&CK T1190 — Exploit Public-Facing Application. Adversaries may "
        "attempt to exploit weaknesses in an internet-facing host or service using "
        "software, data, or commands in order to cause unintended or unexpected "
        "behaviour. The weakness in the system can be a bug, a glitch, or a design "
        "vulnerability. These applications are often websites, databases, standard "
        "services such as SMB or SSH, and any other application with internet "
        "accessible open sockets, such as web servers and related services."
    ),
    (
        "OWASP LLM03: Training Data Poisoning. When training data is manipulated, "
        "the LLM may propagate misinformation or produce biased outputs. Attackers "
        "can craft poisoned samples that cause the model to behave maliciously in "
        "deployment scenarios. Mitigations include data provenance verification, "
        "differential privacy, and anomaly detection during corpus construction."
    ),
    (
        "CVE-2021-44228 — Log4Shell remote code execution in Apache Log4j 2 prior "
        "to 2.15.0. Attackers can exploit JNDI lookups embedded in log messages to "
        "load a remote class and execute arbitrary Java. CVSS 10.0 CRITICAL. "
        "Mitigations include upgrading to Log4j 2.17.1+, setting "
        "log4j2.formatMsgNoLookups=true, and blocking outbound LDAP traffic."
    ),
    (
        "MITRE ATT&CK M1051 — Update Software. Perform regular software updates to "
        "mitigate exploitation risk. Use central patch management systems, automate "
        "vulnerability scanning, and prioritise patches for internet-facing services "
        "and those addressing known exploited vulnerabilities listed in CISA KEV."
    ),
    (
        "CWE-89: SQL Injection. Improper neutralization of special elements used in "
        "an SQL command. An attacker can alter SQL queries to bypass authentication, "
        "dump the database, or escalate privileges. Preventions include parameterised "
        "queries, prepared statements, and ORM frameworks that handle escaping."
    ),
    (
        "CVE-2022-26134 affects Atlassian Confluence Server and Data Center. An OGNL "
        "injection vulnerability in the Confluence Web Work module allows an "
        "unauthenticated remote attacker to execute arbitrary code. CVSS 9.8 CRITICAL. "
        "Patch to 7.4.17, 7.13.7, 7.14.3, 7.15.2, 7.16.4, 7.17.4, or 7.18.1."
    ),
    (
        "OWASP LLM01: Prompt Injection. User-crafted inputs can override system "
        "prompts causing the LLM to ignore safety guidelines. Direct prompt injection "
        "occurs when the malicious instruction appears in the user turn; indirect "
        "injection occurs when the attack is embedded in documents or web pages the "
        "LLM retrieves during RAG augmentation."
    ),
    (
        "MITRE ATT&CK T1566 — Phishing. Adversaries may send phishing messages to "
        "gain access to victim systems. Spearphishing with attachments, links, or "
        "via service accounts are sub-techniques. Defences include user training, "
        "email filtering, link scanning, and DMARC/DKIM/SPF configuration."
    ),
    (
        "CVE-2023-23397 — Microsoft Outlook NTLM hash leak via crafted calendar "
        "invitations. The UNC path in the reminder sound field is retrieved "
        "automatically by Outlook without user interaction, exposing NTLM credentials "
        "to attacker-controlled servers. CVSS 9.8 CRITICAL. No-click exploitation."
    ),
    (
        "CWE-787: Out-of-bounds Write. Software writes data past the end or before "
        "the beginning of an allocated buffer. This may allow attackers to corrupt "
        "memory, crash the process, or execute code. Memory-safe languages, bounds "
        "checking, and ASLR mitigate exploitation."
    ),
    (
        "OWASP LLM06: Sensitive Information Disclosure. LLMs may reveal confidential "
        "data embedded in training corpora or retrieved documents. PII leakage, "
        "API keys in context, and confidential internal documents are risk vectors. "
        "Mitigation: data sanitisation, output filtering, and strict context scoping."
    ),
    (
        "MITRE ATT&CK TA0001 — Initial Access. The adversary is trying to get into "
        "your network. Initial Access consists of techniques that use various entry "
        "vectors to gain their initial foothold within a network. Techniques used "
        "for initial access include targeted spearphishing and exploiting weaknesses "
        "on public-facing web servers."
    ),
    (
        "CVE-2024-21762 — FortiOS out-of-bounds write vulnerability via crafted HTTP "
        "requests. An unauthenticated remote attacker can execute code or commands. "
        "Affects FortiOS 7.4.0 through 7.4.2, 7.2.0 through 7.2.6, 7.0.x. CVSS 9.6 "
        "CRITICAL. CISA KEV listed. Patch immediately or disable SSL VPN webmode."
    ),
    (
        "CWE-22: Path Traversal. The software uses external input to construct a "
        "pathname that is intended to identify a file or directory that is located "
        "underneath a restricted parent directory, but the software does not properly "
        "neutralize special elements within the pathname. Mitigations include "
        "canonicalisation, allowlists, and chroot jails."
    ),
    (
        "OWASP LLM09: Overreliance. Users and developers may place excessive trust in "
        "LLM outputs without adequate verification, leading to incorrect decisions or "
        "security misconfigurations. Human-in-the-loop review, output confidence "
        "scoring, and citation sourcing are recommended controls."
    ),
]

_BENCH_QUERIES: list[str] = [
    "What is the CVSS score and attack vector for CVE-2024-3094?",
    "Which MITRE ATT&CK techniques does the XZ Utils backdoor employ?",
    "How does OWASP LLM03 training data poisoning work and what mitigations exist?",
    "Describe the Log4Shell vulnerability and recommended remediation steps.",
    "What does MITRE M1051 recommend for software update management?",
    "How can SQL injection via CWE-89 be prevented in web applications?",
    "What versions of Atlassian Confluence are affected by CVE-2022-26134?",
    "Explain prompt injection risk in LLM systems per OWASP LLM01.",
    "What phishing sub-techniques are described under MITRE T1566?",
    "How does CVE-2023-23397 exploit Outlook without user interaction?",
    "What is the difference between CWE-787 out-of-bounds write and buffer overflow?",
    "List mitigations for sensitive information disclosure from LLMs per OWASP LLM06.",
    "Which initial access techniques does MITRE ATT&CK TA0001 catalogue?",
    "What is the patch guidance for FortiOS CVE-2024-21762?",
    "How does path traversal CWE-22 work and how is it prevented?",
    "What controls reduce overreliance on LLM outputs per OWASP LLM09?",
]


def _resolve_model_id(
    primary: str, fallback: str, label: str
) -> tuple[str, bool]:
    """Try primary model id; fall back to fallback if unavailable."""
    try:
        model_info(primary)
        log.info("Model verified on Hub: %s", primary)
        return primary, False
    except Exception as exc:
        log.warning(
            "Primary %s '%s' unavailable (%s). Trying fallback '%s'.",
            label,
            primary,
            exc,
            fallback,
        )
    try:
        model_info(fallback)
        log.info("Fallback %s verified on Hub: %s", label, fallback)
        return fallback, True
    except Exception as exc2:
        raise RuntimeError(
            f"Neither {label} model '{primary}' nor fallback '{fallback}' "
            f"could be verified on HuggingFace Hub: {exc2}"
        ) from exc2


def _verify_router_model(model_id: str) -> None:
    """Verify the router base model exists on HuggingFace Hub."""
    try:
        model_info(model_id)
        log.info("Router base model verified on Hub: %s", model_id)
    except Exception as exc:
        raise RuntimeError(
            f"Router base model '{model_id}' not found on Hub: {exc}"
        ) from exc


def _get_embed_dim_from_config(model_id: str) -> int:
    """Download only the model config and return hidden dimension."""
    cfg = AutoConfig.from_pretrained(model_id)
    dim = (
        getattr(cfg, "hidden_size", None)
        or getattr(cfg, "d_model", None)
        or getattr(cfg, "n_embd", None)
    )
    if dim is None:
        raise RuntimeError(
            f"Cannot determine output dimension from config of {model_id}"
        )
    return int(dim)


def _inspect_embedder_metadata(
    model_id: str, tokenizer: AutoTokenizer
) -> tuple[str, str, str]:
    """Determine pooling method and padding side from Hub metadata and tokenizer.

    Returns (pooling_method, padding_side, source_description).
    """
    pooling_method: str | None = None
    padding_side: str | None = None
    sources: list[str] = []

    # 1. Check Sentence-Transformers 1_Pooling/config.json if present
    try:
        pooling_cfg_path = hf_hub_download(
            repo_id=model_id, filename="1_Pooling/config.json"
        )
        pooling_cfg = json.loads(Path(pooling_cfg_path).read_text(encoding="utf-8"))
        if pooling_cfg.get("pooling_mode_lasttoken"):
            pooling_method = "last_token"
        elif pooling_cfg.get("pooling_mode_cls_token"):
            pooling_method = "cls"
        elif pooling_cfg.get("pooling_mode_mean_tokens"):
            pooling_method = "mean"
        if pooling_method is not None:
            sources.append("1_Pooling/config.json")
    except Exception as exc:
        log.debug("No 1_Pooling/config.json for %s: %s", model_id, exc)

    # 2. Check tokenizer's configured padding_side
    tok_pad_side = getattr(tokenizer, "padding_side", None)
    if tok_pad_side in ("left", "right"):
        padding_side = tok_pad_side
        sources.append(f"tokenizer.padding_side={tok_pad_side}")

    # 3. Check model card text for pooling/padding documentation
    try:
        card_content = ModelCard.load(model_id).content.lower()
        sources.append("ModelCard")
        if pooling_method is None:
            if (
                "last token" in card_content
                or "last_token" in card_content
                or "eos token" in card_content
                or "eos_token" in card_content
            ):
                pooling_method = "last_token"
            elif (
                "mean pooling" in card_content
                or "mean pool" in card_content
                or "average pooling" in card_content
            ):
                pooling_method = "mean"
            elif "cls pooling" in card_content or "cls token" in card_content:
                pooling_method = "cls"

        if "left pad" in card_content or "padding_side='left'" in card_content or 'padding_side="left"' in card_content:
            padding_side = "left"
        elif "right pad" in card_content and padding_side is None:
            padding_side = "right"
    except Exception as exc:
        log.warning("Could not read model card for %s: %s", model_id, exc)

    # 4. Defaults if still undetermined (decoder/Qwen/F2LLM models default to last_token+left if causal)
    if pooling_method is None:
        pooling_method = "last_token" if padding_side == "left" else "mean"
        sources.append("default_heuristic")
    if padding_side is None:
        padding_side = "left" if pooling_method == "last_token" else "right"

    log.info(
        "Embedder metadata for %s: pooling=%s, padding_side=%s (sources: %s)",
        model_id,
        pooling_method,
        padding_side,
        ", ".join(sources),
    )
    return pooling_method, padding_side, ", ".join(sources)


def _write_embedder_notes(
    path: Path,
    model_id: str,
    pooling_method: str,
    padding_side: str,
    embed_dim: int,
    query_prefix: str,
    source: str,
) -> None:
    """Write models/embedder_notes.md atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    content = (
        f"# Embedder Notes: {model_id}\n\n"
        f"- **Embed model id:** {model_id}\n"
        f"- **Pooling method:** {pooling_method}\n"
        f"- **Padding side:** {padding_side}\n"
        f"- **Output dimensions:** {embed_dim}\n"
        f"- **MRL supported:** true\n"
        f"- **Query prefix:** {query_prefix}\n"
        f"- **Passage prefix:** (none)\n"
        f"- **Source:** {source}\n"
    )
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(content, encoding="utf-8")
    os.replace(tmp, path)
    log.info("Written: %s", path)


def _mean_pool(last_hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Mean-pool token hidden states using attention_mask."""
    mask_expanded = attention_mask.unsqueeze(-1).float()
    sum_hidden = (last_hidden.float() * mask_expanded).sum(dim=1)
    count = mask_expanded.sum(dim=1).clamp(min=1e-9)
    return sum_hidden / count


def _last_token_pool(
    last_hidden: torch.Tensor,
    attention_mask: torch.Tensor,
    padding_side: str = "right",
) -> torch.Tensor:
    """Extract the final non-padding token representation.

    Handles both ``padding_side == "left"`` (last position ``[:, -1, :]``)
    and ``padding_side == "right"`` (``attention_mask.sum(dim=1) - 1``).
    """
    if padding_side == "left":
        return last_hidden[:, -1, :].float()
    seq_len = attention_mask.sum(dim=1) - 1
    batch_size = last_hidden.size(0)
    idx = seq_len.clamp(min=0).long()
    return last_hidden[
        torch.arange(batch_size, device=last_hidden.device), idx, :
    ].float()


def _pool_hidden(
    last_hidden: torch.Tensor,
    attention_mask: torch.Tensor,
    pooling_method: str,
    padding_side: str,
) -> torch.Tensor:
    """Apply configured pooling strategy to token embeddings."""
    if pooling_method == "last_token":
        return _last_token_pool(last_hidden, attention_mask, padding_side=padding_side)
    if pooling_method == "cls":
        return last_hidden[:, 0, :].float()
    return _mean_pool(last_hidden, attention_mask)


def _benchmark_embedder(
    model_id: str,
    settings: Settings,
) -> dict[str, Any]:
    """Download embedder weights, inspect metadata, and benchmark 16+16 texts."""
    log.info("--- Benchmarking Tier 2 Embedder: %s ---", model_id)
    rss_before = rss_mb()
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token

    pooling_method, padding_side, source = _inspect_embedder_metadata(
        model_id, tokenizer
    )
    tokenizer.padding_side = padding_side

    model = AutoModel.from_pretrained(model_id, torch_dtype=torch.bfloat16)
    model.eval()
    rss_after = rss_mb()
    log.info("Embedder loaded. RSS: %.1f MB (delta: %.1f MB)", rss_after, rss_after - rss_before)

    passage_tokens = 0
    passage_time = 0.0
    for text in _BENCH_PASSAGES:
        prefixed = settings.passage_prefix + text
        enc = tokenizer(
            prefixed,
            return_tensors="pt",
            truncation=True,
            max_length=settings.embed_max_tokens,
            padding=False,
        )
        n_tok = int(enc["input_ids"].shape[1])
        t0 = time.perf_counter()
        with torch.inference_mode():
            out = model(**enc)
            pooled = _pool_hidden(
                out.last_hidden_state, enc["attention_mask"], pooling_method, padding_side
            )
            _ = torch.nn.functional.normalize(pooled, p=2, dim=-1)
        passage_time += time.perf_counter() - t0
        passage_tokens += n_tok

    query_tokens = 0
    query_time = 0.0
    for q in _BENCH_QUERIES:
        prefixed = settings.query_prefix + q
        enc = tokenizer(
            prefixed,
            return_tensors="pt",
            truncation=True,
            max_length=settings.embed_max_tokens,
            padding=False,
        )
        n_tok = int(enc["input_ids"].shape[1])
        t0 = time.perf_counter()
        with torch.inference_mode():
            out = model(**enc)
            pooled = _pool_hidden(
                out.last_hidden_state, enc["attention_mask"], pooling_method, padding_side
            )
            _ = torch.nn.functional.normalize(pooled, p=2, dim=-1)
        query_time += time.perf_counter() - t0
        query_tokens += n_tok

    total_tokens = passage_tokens + query_tokens
    total_time = passage_time + query_time
    tokens_per_sec = total_tokens / total_time if total_time > 0 else 0.0
    chunk_mean_ms = (passage_time / len(_BENCH_PASSAGES)) * 1000.0
    query_mean_ms = (query_time / len(_BENCH_QUERIES)) * 1000.0

    log.info(
        "Embedder bench: %d tokens in %.2fs -> %.1f tok/s (chunk=%.1f ms, query=%.1f ms)",
        total_tokens,
        total_time,
        tokens_per_sec,
        chunk_mean_ms,
        query_mean_ms,
    )

    del model, tokenizer
    gc.collect()

    return {
        "pooling_method": pooling_method,
        "padding_side": padding_side,
        "metadata_source": source,
        "tokens_per_sec": round(tokens_per_sec, 2),
        "chunk_mean_ms": round(chunk_mean_ms, 2),
        "query_mean_ms": round(query_mean_ms, 2),
        "total_tokens": total_tokens,
        "rss_mb": round(rss_after, 1),
    }


def _benchmark_router(model_id: str) -> dict[str, Any]:
    """Download router weights and benchmark 16 chunks + 16 queries at 2 threads."""
    log.info("--- Benchmarking Tier 1 Router: %s ---", model_id)
    rss_before = rss_mb()
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_id,
        num_labels=3,
        torch_dtype=torch.bfloat16,
        ignore_mismatched_sizes=True,
    )
    model.eval()
    rss_after = rss_mb()
    log.info("Router loaded. RSS: %.1f MB (delta: %.1f MB)", rss_after, rss_after - rss_before)

    chunk_tokens = 0
    chunk_time = 0.0
    for text in _BENCH_PASSAGES:
        enc = tokenizer(
            text, return_tensors="pt", truncation=True, max_length=128, padding=False
        )
        n_tok = int(enc["input_ids"].shape[1])
        t0 = time.perf_counter()
        with torch.inference_mode():
            logits = model(**enc).logits
            _ = torch.softmax(logits[0].float(), dim=-1)
        chunk_time += time.perf_counter() - t0
        chunk_tokens += n_tok

    query_tokens = 0
    query_time = 0.0
    for q in _BENCH_QUERIES:
        enc = tokenizer(
            q, return_tensors="pt", truncation=True, max_length=128, padding=False
        )
        n_tok = int(enc["input_ids"].shape[1])
        t0 = time.perf_counter()
        with torch.inference_mode():
            logits = model(**enc).logits
            _ = torch.softmax(logits[0].float(), dim=-1)
        query_time += time.perf_counter() - t0
        query_tokens += n_tok

    total_tokens = chunk_tokens + query_tokens
    total_time = chunk_time + query_time
    tokens_per_sec = total_tokens / total_time if total_time > 0 else 0.0
    chunk_mean_ms = (chunk_time / len(_BENCH_PASSAGES)) * 1000.0
    query_mean_ms = (query_time / len(_BENCH_QUERIES)) * 1000.0

    log.info(
        "Router bench: %d tokens in %.2fs -> %.1f tok/s (chunk=%.1f ms, query=%.1f ms)",
        total_tokens,
        total_time,
        tokens_per_sec,
        chunk_mean_ms,
        query_mean_ms,
    )

    del model, tokenizer
    gc.collect()

    return {
        "tokens_per_sec": round(tokens_per_sec, 2),
        "chunk_mean_ms": round(chunk_mean_ms, 2),
        "query_mean_ms": round(query_mean_ms, 2),
        "total_tokens": total_tokens,
        "rss_mb": round(rss_after, 1),
    }


def _load_reranker_with_fallback(
    primary_id: str, fallback_id: str, already_fallback: bool
) -> tuple[AutoModelForSequenceClassification, AutoTokenizer, str, bool]:
    """Download and instantiate reranker, falling back on failure."""
    candidate_ids = [primary_id] if already_fallback else [primary_id, fallback_id]
    last_exc: Exception | None = None
    for idx, mid in enumerate(candidate_ids):
        try:
            tok = AutoTokenizer.from_pretrained(mid)
            mdl = AutoModelForSequenceClassification.from_pretrained(
                mid, torch_dtype=torch.bfloat16
            )
            mdl.eval()
            used_fallback = already_fallback or (idx > 0)
            return mdl, tok, mid, used_fallback
        except Exception as exc:
            last_exc = exc
            log.warning("Failed to load reranker '%s': %s", mid, exc)
    raise RuntimeError(
        f"Failed to load reranker '{primary_id}' or fallback '{fallback_id}': {last_exc}"
    ) from last_exc


def _benchmark_reranker(
    model_id: str,
    is_fallback: bool,
    settings: Settings,
) -> tuple[dict[str, Any], str, bool]:
    """Download reranker weights and benchmark 16 (query, chunk) pairs + batch-8."""
    log.info("--- Benchmarking Tier 4 Reranker: %s ---", model_id)
    rss_before = rss_mb()
    model, tokenizer, resolved_id, final_fallback = _load_reranker_with_fallback(
        model_id, _FALLBACK_RERANKER, is_fallback
    )
    rss_after = rss_mb()
    log.info(
        "Reranker '%s' loaded. RSS: %.1f MB (delta: %.1f MB)",
        resolved_id,
        rss_after,
        rss_after - rss_before,
    )

    num_labels = int(getattr(model.config, "num_labels", 1))
    pair_tokens = 0
    pair_time = 0.0

    # Benchmark all 16 queries paired with all 16 representative chunks
    for q, p in zip(_BENCH_QUERIES, _BENCH_PASSAGES):
        enc = tokenizer(
            q,
            p,
            return_tensors="pt",
            truncation=True,
            max_length=settings.rerank_max_length,
            padding=False,
        )
        n_tok = int(enc["input_ids"].shape[1])
        t0 = time.perf_counter()
        with torch.inference_mode():
            logits = model(**enc).logits
            if logits.ndim == 1 or logits.shape[-1] == 1:
                _ = torch.sigmoid(logits.squeeze(-1).float())
            else:
                _ = torch.softmax(logits.float(), dim=-1)[:, 1]
        pair_time += time.perf_counter() - t0
        pair_tokens += n_tok

    # Measure batch-8 reranking latency (1 query against 8 candidate chunks)
    batch_pairs_q = [_BENCH_QUERIES[0]] * 8
    batch_pairs_p = _BENCH_PASSAGES[:8]
    enc_batch = tokenizer(
        batch_pairs_q,
        batch_pairs_p,
        return_tensors="pt",
        truncation=True,
        max_length=settings.rerank_max_length,
        padding=True,
    )
    t0 = time.perf_counter()
    with torch.inference_mode():
        logits_b = model(**enc_batch).logits
        if logits_b.ndim == 1 or logits_b.shape[-1] == 1:
            _ = torch.sigmoid(logits_b.squeeze(-1).float())
        else:
            _ = torch.softmax(logits_b.float(), dim=-1)[:, 1]
    batch8_ms = (time.perf_counter() - t0) * 1000.0

    tokens_per_sec = pair_tokens / pair_time if pair_time > 0 else 0.0
    pair_mean_ms = (pair_time / len(_BENCH_QUERIES)) * 1000.0

    log.info(
        "Reranker bench: %d tokens in %.2fs -> %.1f tok/s (pair=%.1f ms, batch8=%.1f ms)",
        pair_tokens,
        pair_time,
        tokens_per_sec,
        pair_mean_ms,
        batch8_ms,
    )

    del model, tokenizer
    gc.collect()

    return (
        {
            "tokens_per_sec": round(tokens_per_sec, 2),
            "pair_mean_ms": round(pair_mean_ms, 2),
            "batch8_ms": round(batch8_ms, 2),
            "num_labels": num_labels,
            "total_tokens": pair_tokens,
            "rss_mb": round(rss_after, 1),
        },
        resolved_id,
        final_fallback,
    )


def _project_eta(tokens_per_sec: float, n_chunks: int, avg_tokens: int) -> float:
    """Project corpus embedding time in seconds."""
    if tokens_per_sec <= 0:
        return float("inf")
    return (n_chunks * avg_tokens) / tokens_per_sec


def _write_preflight_json(path: Path, data: dict[str, Any]) -> None:
    """Write reports/preflight.json atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    os.replace(tmp, path)
    log.info("Written: %s", path)


def main() -> None:
    """Run preflight checks across all 3 tiers, benchmark, and write artifacts."""
    apply_hardware_guard()
    settings = get_settings()

    preflight_path = settings.reports_dir / "preflight.json"
    notes_path = settings.models_dir / "embedder_notes.md"

    log.info("=== KAVACH-RAG Preflight Check (HF_HOME=%s) ===", os.environ.get("HF_HOME"))

    # ── Step 1: Verify model IDs on HuggingFace Hub ───────────────────────────
    embed_model_id, embed_fallback = _resolve_model_id(
        settings.embed_model_id,
        _FALLBACK_EMBEDDER,
        "embedder",
    )
    rerank_model_id, reranker_is_fallback = _resolve_model_id(
        settings.rerank_model_id,
        _FALLBACK_RERANKER,
        "reranker",
    )
    _verify_router_model(settings.router_base_model_id)
    router_model_id = settings.router_base_model_id

    # ── Step 2: Verify embed dim from config ──────────────────────────────────
    embed_dim = _get_embed_dim_from_config(embed_model_id)
    log.info("Embedder output dim from config: %d", embed_dim)
    if embed_dim != settings.embed_dim_full:
        log.warning(
            "Config reports embed_dim=%d but settings.embed_dim_full=%d. "
            "Proceeding with config value.",
            embed_dim,
            settings.embed_dim_full,
        )

    # ── Step 3: Download & benchmark Tier 2 (Embedder) ────────────────────────
    embed_stats = _benchmark_embedder(embed_model_id, settings)

    # Write embedder_notes.md immediately after inspecting embedder metadata
    _write_embedder_notes(
        notes_path,
        model_id=embed_model_id,
        pooling_method=embed_stats["pooling_method"],
        padding_side=embed_stats["padding_side"],
        embed_dim=embed_dim,
        query_prefix=settings.query_prefix,
        source=embed_stats["metadata_source"],
    )

    # ── Step 4: Download & benchmark Tier 1 (Router) ──────────────────────────
    router_stats = _benchmark_router(router_model_id)

    # ── Step 5: Download & benchmark Tier 4 (Reranker) ────────────────────────
    rerank_stats, rerank_model_id, reranker_is_fallback = _benchmark_reranker(
        rerank_model_id, reranker_is_fallback, settings
    )
    if reranker_is_fallback:
        print(
            f"\n!!! RERANKER FALLBACK: using {rerank_model_id} !!!\n",
            flush=True,
        )

    # ── Step 6: Project corpus embedding ETAs & verify VRAM isolation ─────────
    tokens_per_sec = embed_stats["tokens_per_sec"]
    eta_w512 = _project_eta(tokens_per_sec, n_chunks=1500, avg_tokens=350)
    eta_w256 = _project_eta(tokens_per_sec, n_chunks=2500, avg_tokens=200)
    peak_rss = max(
        embed_stats["rss_mb"],
        router_stats["rss_mb"],
        rerank_stats["rss_mb"],
        round(rss_mb(), 1),
    )
    vram = vram_probe()

    result: dict[str, Any] = {
        "embed_model_id": embed_model_id,
        "embed_fallback": embed_fallback,
        "rerank_model_id": rerank_model_id,
        "reranker_is_fallback": reranker_is_fallback,
        "router_model_id": router_model_id,
        "hf_home": os.environ.get("HF_HOME", ""),
        "pooling_method": embed_stats["pooling_method"],
        "padding_side": embed_stats["padding_side"],
        "embed_dim": embed_dim,
        "mrl_dim": settings.embed_dim_mrl,
        "tokens_per_sec": tokens_per_sec,
        "projected_eta_w512_seconds": round(eta_w512, 1),
        "projected_eta_w256_seconds": round(eta_w256, 1),
        "tiers": {
            "tier1_router": router_stats,
            "tier2_embedder": embed_stats,
            "tier4_reranker": rerank_stats,
        },
        "bench_date": datetime.now(timezone.utc).isoformat(),
        "peak_rss_mb": peak_rss,
        "vram_probe": dict(vram),
    }
    _write_preflight_json(preflight_path, result)

    # ── AMENDMENT A6: Print summary and stop ──────────────────────────────────
    print(
        f"\n{'='*64}\n"
        f"PREFLIGHT COMPLETE — ETA CHECK REQUIRED:\n"
        f"Tier 1 Router:   {router_model_id} "
        f"(query={router_stats['query_mean_ms']:.1f} ms, rss={router_stats['rss_mb']:.1f} MB)\n"
        f"Tier 2 Embedder: {embed_model_id} (fallback={embed_fallback}, "
        f"pooling={embed_stats['pooling_method']}, pad={embed_stats['padding_side']}, "
        f"rss={embed_stats['rss_mb']:.1f} MB)\n"
        f"Tier 4 Reranker: {rerank_model_id} (fallback={reranker_is_fallback}, "
        f"batch8={rerank_stats['batch8_ms']:.1f} ms, rss={rerank_stats['rss_mb']:.1f} MB)\n"
        f"Embedder throughput: {tokens_per_sec:.1f} tokens/sec\n"
        f"Projected ETA w512 corpus (1500x350 tok): {eta_w512:.1f} s "
        f"(~{eta_w512/60:.1f} min)\n"
        f"Projected ETA w256 corpus (2500x200 tok): {eta_w256:.1f} s "
        f"(~{eta_w256/60:.1f} min)\n"
        f"Peak RSS across tiers: {peak_rss:.1f} MB | VRAM probe: {dict(vram)}\n"
        f"{'='*64}\n"
        f"Awaiting orchestrator go-ahead before launching build_index.py.\n",
        flush=True,
    )


if __name__ == "__main__":
    main()
