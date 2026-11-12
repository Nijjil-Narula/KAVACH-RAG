"""Two-stage generation judge. Stage 1: deterministic checks. Stage 2: LLM claim-level judge.

Stage 2 backend is Gemini (cross-family) when an API key exists, otherwise the local Ollama model.
The local fallback is the generator judging itself (documented self-preference bias): indicative
only. The cross-family jury and CI gating belong to Phase 4.
"""
from __future__ import annotations

import os
import random
import re
import time
from collections import Counter
from typing import Any, Callable, Literal, Sequence

import httpx
from pydantic import BaseModel, ConfigDict, ValidationError

from kavach.exceptions import JudgeError
from kavach.generation.gen_config import GenSettings, get_gen_settings
from kavach.generation.generator import OllamaGenerator, full_jitter_delay
from kavach.generation.prompts import CITATION_RE, is_refusal

_DECIMAL_RE = re.compile(r"(?<![\w.])\d+\.\d+(?![\w.])")
_WORD_RE = re.compile(r"[a-z0-9_.\-]+")
_CHUNK_SUFFIX_RE = re.compile(r"_chunk_\d{3}$")
_NUMERIC_HEADER_RE = re.compile(r"\d+(?:\.\d+)?")


# ------------------------------------------------------------------ schemas (Stage 2 output)
class ClaimVerdict(BaseModel):
    """Atomic factual claim verdict against retrieved context."""

    model_config = ConfigDict(extra="forbid")
    claim: str
    verdict: Literal["supported", "unsupported", "contradicted"]
    evidence_chunk_id: str | None = None


class JudgeVerdict(BaseModel):
    """`reasoning` is deliberately the first field so chain-of-thought precedes the verdicts."""

    model_config = ConfigDict(extra="forbid")
    reasoning: str
    claims: list[ClaimVerdict]
    relevance: Literal["full", "partial", "none"]


# ------------------------------------------------------------------ Stage 1: deterministic
def doc_of(chunk_id: str) -> str:
    """Strip the ``_chunk_NNN`` suffix to recover the parent document ID."""
    return _CHUNK_SUFFIX_RE.sub("", chunk_id)


def extract_citations(answer: str) -> list[str]:
    """Extract unique ``[chunk_id]`` citations in first-appearance order."""
    seen: dict[str, None] = {}
    for cid in CITATION_RE.findall(answer):
        seen.setdefault(cid)
    return list(seen)


def _ratio(num: int, den: int) -> float | None:
    return None if den == 0 else num / den


def citation_metrics(
    answer: str, retrieved_ids: Sequence[str], expected_ids: Sequence[str]
) -> dict[str, Any]:
    """Compute chunk-level and doc-level citation validity, precision, and recall."""
    cited = extract_citations(answer)
    retrieved, expected = set(retrieved_ids), set(expected_ids)
    gold_chunks = retrieved & expected
    gold_docs = {doc_of(c) for c in gold_chunks}
    cited_docs = {doc_of(c) for c in cited}
    return {
        "cited_ids": cited,
        "citation_validity": _ratio(len(set(cited) & retrieved), len(cited)),
        "hallucinated_citations": sorted(set(cited) - retrieved),
        "citation_precision": _ratio(len(set(cited) & gold_chunks), len(cited)) if gold_chunks else None,
        "citation_recall": _ratio(len(set(cited) & gold_chunks), len(gold_chunks)) if gold_chunks else None,
        "citation_precision_doc": _ratio(len(cited_docs & gold_docs), len(cited_docs)) if gold_docs else None,
        "citation_recall_doc": _ratio(len(cited_docs & gold_docs), len(gold_docs)) if gold_docs else None,
        "retrieval_miss": not gold_chunks,
    }


def entity_fidelity(
    answer: str, context_text: str, required_entities: Sequence[str]
) -> dict[str, Any]:
    """Case-sensitive verbatim presence of required entities (catches 9.8 -> 10 and case changes)."""
    low_ctx = context_text.lower()
    answerable = [e for e in required_entities if e.lower() in low_ctx]
    present = [e for e in answerable if e in answer]
    return {
        "entities_answerable": answerable,
        "entities_missing": [e for e in answerable if e not in answer],
        "entity_recall": _ratio(len(present), len(answerable)),
        "context_sufficient": len(answerable) == len(required_entities),
    }


def ungrounded_decimals(answer: str, context_text: str) -> list[str]:
    """Return sorted decimal literals appearing in *answer* that are absent from *context_text*."""
    return sorted({d for d in _DECIMAL_RE.findall(answer) if d not in context_text})


def echo_ratio(answer: str, query: str, context_text: str, n: int = 6) -> float | None:
    """Share of answer n-grams copied from the query but absent from the context (injection/log echo)."""

    def grams(text: str) -> set[tuple[str, ...]]:
        w = _WORD_RE.findall(text.lower())
        return {tuple(w[i : i + n]) for i in range(len(w) - n + 1)}

    a = grams(answer)
    if not a:
        return None
    return len(a & grams(query) - grams(context_text)) / len(a)


def conflict_check(answer: str, chunks: Sequence[dict]) -> dict[str, Any]:
    """Check whether both OWASP v1.1 and v2.0 appear in the retrieved context and answer."""
    versions = {str(c.get("standard_version") or "") for c in chunks}
    both_in_ctx = any("1.1" in v for v in versions) and any("2.0" in v for v in versions)
    both_in_ans = bool(re.search(r"v?1\.1", answer)) and bool(re.search(r"v?2\.0", answer))
    return {"both_versions_in_context": both_in_ctx, "both_versions_in_answer": both_in_ans}


def stage1(
    item: dict,
    golden: dict,
    answer: str,
    chunks: Sequence[dict],
    bypassed: bool,
) -> dict[str, Any]:
    """Run all deterministic Stage 1 checks for a single evaluated generation item."""
    ctx = "\n".join(c["content"] for c in chunks)
    out: dict[str, Any] = {"refusal_emitted": is_refusal(answer), "bypassed_llm": bypassed}
    expected_refusal = golden["question_type"] == "OUT_OF_SCOPE"
    out["refusal_correct"] = out["refusal_emitted"] == expected_refusal
    if expected_refusal or bypassed:
        return out
    out |= citation_metrics(answer, [c["chunk_id"] for c in chunks], item["expected_source_chunk_ids"])
    out |= entity_fidelity(answer, ctx, golden.get("required_entities", []))
    out["ungrounded_decimals"] = ungrounded_decimals(answer, ctx)
    out["echo_ratio"] = (
        echo_ratio(answer, item["query"], ctx)
        if item["question_type"] == "CONTEXT_NOISE"
        else None
    )
    if golden.get("conflict_test"):
        out |= conflict_check(answer, chunks)
    return out


# ------------------------------------------------------------------ Stage 2: LLM judge
JUDGE_SYSTEM = """You are a strict, impartial evaluator of a RAG answer. Judge ONLY against the supplied CONTEXT, never against your own knowledge.
1. Split the ANSWER into atomic factual claims. Ignore citation brackets, greetings and refusal sentences.
2. For each claim give verdict "supported" (the context states it), "unsupported" (the context does not state it) or "contradicted" (the context states the opposite). Numbers, versions and identifiers must match exactly.
3. Rate relevance to the QUESTION: "full" if it answers every part, "partial" if it answers some, "none" otherwise.
Reply with ONE JSON object and nothing else, with keys in this order: reasoning (string, written first), claims (list of {claim, verdict, evidence_chunk_id}), relevance."""


def judge_user_prompt(query: str, chunks: Sequence[dict], answer: str) -> str:
    """Format context chunks, query, and candidate answer for Stage 2 evaluation."""
    ctx = "\n---\n".join(f"[{c['chunk_id']}]\n{c['content']}" for c in chunks)
    return f"CONTEXT:\n{ctx}\n\nQUESTION:\n{query}\n\nANSWER:\n{answer}"


def parse_verdict(raw: str) -> JudgeVerdict:
    """Strip optional markdown fences and validate raw JSON against ``JudgeVerdict``."""
    text = raw.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
    return JudgeVerdict.model_validate_json(text)


class _Pacer:
    """Simple RPM request pacer using injectable monotonic clock and sleep functions."""

    def __init__(
        self,
        rpm: int,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.gap = 60.0 / max(1, rpm)
        self._clock = clock
        self._sleep = sleep
        self._last: float | None = None

    def wait(self) -> None:
        if self._last is not None:
            self._sleep(max(0.0, self.gap - (self._clock() - self._last)))
        self._last = self._clock()


class GeminiJudge:
    """Cross-family Stage 2 judge backed by the Gemini Generative Language API."""

    name = "gemini"
    cross_family = True

    def __init__(
        self,
        s: GenSettings,
        api_key: str,
        *,
        transport: httpx.BaseTransport | None = None,
        sleep: Callable[[float], None] = time.sleep,
        rng: random.Random | None = None,
    ) -> None:
        self.s = s
        self._key = api_key
        self._sleep = sleep
        self._rng = rng
        self._pacer = _Pacer(s.gemini_rpm, sleep=sleep)
        self._client = httpx.Client(
            http2=True,
            timeout=httpx.Timeout(60.0),
            transport=transport,
            limits=httpx.Limits(max_keepalive_connections=4, keepalive_expiry=60.0),
        )
        self.thinking_applied = True
        self.model_id = s.gemini_model

    def close(self) -> None:
        """Close the underlying HTTPX client pool."""
        self._client.close()

    def __enter__(self) -> GeminiJudge:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def _redact(self, text: str) -> str:
        """Ensure API key material can never appear in exception messages."""
        if self._key and self._key in text:
            return text.replace(self._key, "[REDACTED]")
        return text

    def _body(self, user: str) -> dict[str, Any]:
        cfg: dict[str, Any] = {"temperature": 0.0, "responseMimeType": "application/json"}
        if self.thinking_applied:
            cfg["thinkingConfig"] = {"thinkingBudget": self.s.gemini_thinking_budget}
        return {
            "systemInstruction": {"parts": [{"text": JUDGE_SYSTEM}]},
            "contents": [{"role": "user", "parts": [{"text": user}]}],
            "generationConfig": cfg,
        }

    def _call(self, user: str) -> str:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.s.gemini_model}:generateContent"
        for attempt in range(self.s.max_attempts):
            self._pacer.wait()
            try:
                resp = self._client.post(
                    url,
                    json=self._body(user),
                    headers={"x-goog-api-key": self._key},
                )
            except (httpx.TransportError, httpx.TimeoutException):
                self._sleep(full_jitter_delay(attempt, 1.5, 60.0, self._rng))
                continue
            if resp.status_code == 400 and self.thinking_applied and "thinking" in resp.text.lower():
                self.thinking_applied = False  # model rejects thinkingConfig: retry once without, and report it
                continue
            if resp.status_code in (429, 500, 502, 503, 504):
                ra = (resp.headers.get("retry-after") or "").strip()
                delay = full_jitter_delay(attempt, 1.5, 60.0, self._rng)
                ra_val = float(ra) if _NUMERIC_HEADER_RE.fullmatch(ra) else 0.0
                self._sleep(max(delay, ra_val))
                continue
            if resp.status_code != 200:
                raise JudgeError(f"Gemini HTTP {resp.status_code}: {self._redact(resp.text[:300])}")
            try:
                payload = resp.json()
            except ValueError as exc:
                raise JudgeError(f"Gemini HTTP 200 returned invalid JSON: {exc}") from exc
            candidates = payload.get("candidates") if isinstance(payload, dict) else None
            if not isinstance(candidates, list) or not candidates:
                raise JudgeError(
                    f"Gemini response missing candidates: {self._redact(str(payload)[:300])}"
                )
            first = candidates[0] if isinstance(candidates[0], dict) else {}
            content = first.get("content") if isinstance(first.get("content"), dict) else {}
            parts = content.get("parts")
            if not isinstance(parts, list):
                raise JudgeError(
                    f"Gemini response missing content parts: {self._redact(str(first)[:300])}"
                )
            return "".join(
                str(p.get("text") or "")
                for p in parts
                if isinstance(p, dict) and not p.get("thought")
            )
        raise JudgeError("Gemini judge exhausted retries")

    def judge(self, query: str, chunks: Sequence[dict], answer: str) -> JudgeVerdict:
        """Evaluate *answer* against *chunks* for *query*, retrying once on schema error."""
        user = judge_user_prompt(query, chunks, answer)
        try:
            return parse_verdict(self._call(user))
        except (ValidationError, ValueError) as exc:
            retry = (
                user
                + f"\n\nYour previous reply was invalid JSON for the schema ({str(exc)[:200]}). "
                "Reply again with valid JSON only."
            )
            try:
                return parse_verdict(self._call(retry))
            except (ValidationError, ValueError) as exc2:
                raise JudgeError(f"Unparseable judge output: {exc2}") from exc2


class OllamaJudge:
    """Fallback local Stage 2 judge backed by Ollama (same model family; indicative only)."""

    name = "ollama"
    cross_family = False

    def __init__(self, s: GenSettings, generator: OllamaGenerator) -> None:
        self.s = s
        self._gen = generator
        self.model_id = s.model

    def judge(self, query: str, chunks: Sequence[dict], answer: str) -> JudgeVerdict:
        """Evaluate *answer* against *chunks* using Ollama structured JSON output."""
        messages = [
            {"role": "system", "content": JUDGE_SYSTEM},
            {"role": "user", "content": judge_user_prompt(query, chunks, answer)},
        ]
        for _ in range(2):
            res = self._gen.generate(
                messages,
                num_predict=self.s.judge_num_predict,
                response_format=JudgeVerdict.model_json_schema(),
            )
            try:
                return parse_verdict(res.text)
            except (ValidationError, ValueError):
                continue
        raise JudgeError("Local judge produced invalid JSON twice")


def select_judge(s: GenSettings, generator: OllamaGenerator) -> GeminiJudge | OllamaJudge:
    """Select Gemini cross-family judge when configured/keyed, otherwise fallback to Ollama."""
    key = (s.gemini_api_key or os.environ.get("GEMINI_API_KEY") or "").strip()
    if s.judge_backend == "gemini" and not key:
        raise JudgeError(
            "judge_backend=gemini but no GEMINI_API_KEY / KAVACH_GEN_GEMINI_API_KEY is set"
        )
    if s.judge_backend in ("auto", "gemini") and key:
        return GeminiJudge(s, key)
    return OllamaJudge(s, generator)


# ------------------------------------------------------------------ aggregation
def faithfulness(v: JudgeVerdict) -> float | None:
    """Fraction of atomic claims marked ``supported``; ``None`` if no claims."""
    if not v.claims:
        return None
    return sum(c.verdict == "supported" for c in v.claims) / len(v.claims)


RELEVANCE_SCORE = {"full": 1.0, "partial": 0.5, "none": 0.0}


def _mean(xs: Sequence[float | None]) -> float | None:
    vals = [x for x in xs if x is not None]
    return sum(vals) / len(vals) if vals else None


def summarize(rows: Sequence[dict]) -> dict[str, Any]:
    """Aggregate Stage 1 + Stage 2 metrics across all evaluated rows."""
    served = [
        r for r in rows if not r["stage1"]["bypassed_llm"] and r["type"] != "OUT_OF_SCOPE"
    ]
    sufficient = [r for r in served if r["stage1"].get("context_sufficient")]
    by_type = {
        t: _mean([r.get("faithfulness") for r in served if r["type"] == t])
        for t in sorted({r["type"] for r in served})
    }
    return {
        "n_served": len(served),
        "faithfulness_mean": _mean([r.get("faithfulness") for r in served]),
        "relevance_mean": _mean([r.get("relevance_score") for r in served]),
        "relevance_mean_context_sufficient": _mean(
            [r.get("relevance_score") for r in sufficient]
        ),
        "n_context_sufficient": len(sufficient),
        "faithfulness_by_type": by_type,
        "citation_validity_mean": _mean(
            [r["stage1"].get("citation_validity") for r in served]
        ),
        "citation_precision_mean": _mean(
            [r["stage1"].get("citation_precision") for r in served]
        ),
        "citation_recall_mean": _mean(
            [r["stage1"].get("citation_recall") for r in served]
        ),
        "citation_precision_doc_mean": _mean(
            [r["stage1"].get("citation_precision_doc") for r in served]
        ),
        "citation_recall_doc_mean": _mean(
            [r["stage1"].get("citation_recall_doc") for r in served]
        ),
        "entity_recall_mean": _mean([r["stage1"].get("entity_recall") for r in served]),
        "model_refusal_items": sum(bool(r["stage1"]["refusal_emitted"]) for r in served),
        "over_refusal_items": sum(
            bool(r["stage1"]["refusal_emitted"] and r["stage1"].get("context_sufficient"))
            for r in served
        ),
        "hallucinated_citation_items": sum(
            bool(r["stage1"].get("hallucinated_citations")) for r in served
        ),
        "ungrounded_decimal_items": sum(
            bool(r["stage1"].get("ungrounded_decimals")) for r in served
        ),
        "refusal_correct": Counter(bool(r["stage1"]["refusal_correct"]) for r in rows),
    }
