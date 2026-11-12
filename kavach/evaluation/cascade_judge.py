"""Tier 6 two-stage cross-family judge cascade with atomic, resumable checkpointing.

Stage 1 grades every non-deterministic item with one Gemini model. Borderline items escalate to a
panel of Stage 1 + Gemini jury + Groq jury: majority vote on the verdict, minimum on the scores.
Run:  python -m kavach.evaluation.cascade_judge --ping | --limit 5 | (no flags = full batch)
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import random
import re
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable, Sequence

import httpx
from pydantic import ValidationError

from kavach.evaluation.schemas import (
    WIRE_SCHEMA,
    CyberRAGEvaluationSchema,
    VerdictEnum,
    gemini_schema,
)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = ROOT / "data" / "processed" / "generation_results_150.jsonl"
DEFAULT_GOLDEN = ROOT / "data" / "golden" / "golden_150.jsonl"
DEFAULT_CHECKPOINT = ROOT / "data" / "processed" / "tier6_eval_checkpoints.jsonl"

STAGE1_MODEL = "gemini-3.5-flash-lite"
STAGE1_FALLBACK_MODEL = "gemini-3.1-flash-lite"
JURY_GEMINI_MODEL = "gemini-3.8-flash"
JURY_GROQ_MODEL = "openai/gpt-oss-120b"
PACING_S = {
    STAGE1_MODEL: 4.5,
    STAGE1_FALLBACK_MODEL: 4.5,
    JURY_GEMINI_MODEL: 12.5,
    JURY_GROQ_MODEL: 17.5,
}
THINKING_BUDGET = 1024
MAX_ATTEMPTS = 5
BACKOFF_BASE_S = 1.5
BACKOFF_CAP_S = 60.0
REQUEST_TIMEOUT_S = 120.0
JURY_CONCURRENCY = 3
RETRYABLE_STATUS = frozenset({408, 429, 500, 502, 503, 504})
FAITHFUL_PASS_FLOOR = 0.85
FAITHFUL_BORDERLINE_FLOOR = 0.50
REFUSAL_TOKEN = "INSUFFICIENT_CONTEXT"
REQUIRED_ITEM_KEYS = (
    "test_id", "question_type", "query", "ground_truth_answer",
    "bypassed_llm", "answer", "context_chunks",
)

SYSTEM_RUBRIC = """You are a Principal Cybersecurity Auditor grading one answer produced by an enterprise retrieval-augmented generation pipeline.
You receive: QUESTION CATEGORY, USER QUERY, GROUND TRUTH REFERENCE ANSWER, RETRIEVED CONTEXT CHUNKS and CANDIDATE ANSWER.

AUDITING RULES:
1. BOUNDED VERIFICATION. Judge candidate claims ONLY against the RETRIEVED CONTEXT CHUNKS. Never use your own knowledge to confirm CVE details, CVSS scores or attack techniques. A claim that is true in reality but absent from the chunks is unsupported. The reference answer is NOT evidence: use it only to decide whether the context contains what is needed and whether a grounded answer disagrees with the expected result.
2. FAITHFULNESS. Split the candidate into atomic factual claims (ignore citation brackets and refusal wording). faithfulness_score = supported claims / total claims, rounded to two decimals. Any altered number, identifier, version or casing (for example 10 instead of 10.0, or a rewritten CWE or technique ID) makes that claim unsupported and sets faithfulness_score = 0.0.
3. REFUSALS. The candidate refuses by starting with INSUFFICIENT_CONTEXT.
   - Refusal and the context lacks the evidence (or the category is OUT_OF_SCOPE): faithfulness_score = 1.0, negative_constraint_passed = true, verdict = PASS.
   - Refusal although the context contains the evidence: faithfulness_score = 1.0, negative_constraint_passed = true, verdict = PARTIAL, and state that it is an over-refusal.
   - A substantive answer although the context lacks the evidence (or the category is OUT_OF_SCOPE): faithfulness_score = 0.0, negative_constraint_passed = false, verdict = FAIL.
   negative_constraint_passed is true whenever the refusal rule is satisfied or does not apply.
4. VERSION CONFLICTS. If the context holds contradictory versions of one standard (for example OWASP LLM Top 10 v1.1 and v2.0), the candidate must state each version separately. Blending them, or silently choosing one, sets conflict_resolution_passed = false, faithfulness_score no higher than 0.50 and verdict = PARTIAL. If the context holds no cross-version conflict, conflict_resolution_passed = true (not applicable).
5. CONTEXT RELEVANCE. context_relevance_score measures the retrieved chunks, not the answer: 1.0 if they contain the evidence for every part of the query, 0.5 if for some parts, 0.0 if for none. Use exactly 0.0, 0.5 or 1.0.
6. VERDICT. PASS: faithfulness_score >= 0.85 and no rule above forced a lower verdict. FAIL: faithfulness_score < 0.50. Otherwise PARTIAL. A grounded answer that clearly disagrees with the reference answer is at most PARTIAL; explain why.
7. OUTPUT ORDER. Write the "reasoning" audit trace first, one claim at a time, before any score, flag or verdict. Reply with one JSON object only."""

PING_PAYLOAD = (
    "### QUESTION CATEGORY:\nFACTUAL\n\n"
    "### USER QUERY:\nWhat is the CVSS v3.1 base score of CVE-0000-0000?\n\n"
    "### GROUND TRUTH REFERENCE ANSWER:\n9.8\n\n"
    "### RETRIEVED CONTEXT CHUNKS:\n[PING_chunk_001]\nCVE-0000-0000 has a CVSS v3.1 base score of 9.8.\n\n"
    "### CANDIDATE ANSWER TO GRADE:\nThe CVSS v3.1 base score is 9.8 [PING_chunk_001]."
)

_REFUSAL_RE = re.compile(r"^[\s\"'\x60*#>\-]*" + re.escape(REFUSAL_TOKEN), re.IGNORECASE)
_FENCE = "\x60" * 3
_FENCE_RE = re.compile(rf"^{_FENCE}(?:json)?\s*|\s*{_FENCE}$")
_SECRETS: list[str] = []


# ------------------------------------------------------------------ errors and small helpers
class JudgeUnavailable(Exception):
    """A judge could not return a response (non-retryable failure or retries exhausted)."""


class RateLimited(JudgeUnavailable):
    """HTTP 429 persisted through every retry."""


class JudgeOutputError(Exception):
    """A judge answered, but never with schema-valid JSON."""


class CascadeError(Exception):
    """The cascade could not produce a trustworthy grade for an item."""


def register_secrets(values: Sequence[str]) -> None:
    _SECRETS[:] = [v for v in values if v]


def redact(text: str) -> str:
    for secret in _SECRETS:
        if secret in text:
            text = text.replace(secret, "[REDACTED]")
    return text


def full_jitter_delay(
    attempt: int,
    base_s: float = BACKOFF_BASE_S,
    cap_s: float = BACKOFF_CAP_S,
    rng: random.Random | None = None,
) -> float:
    ceiling = min(cap_s, base_s * (2 ** max(0, min(int(attempt), 30))))
    return float((rng or random).uniform(0.0, ceiling))


def _read_windows_user_env(name: str) -> str:
    try:
        import winreg
    except ImportError:
        return ""
    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as key:
            value, _ = winreg.QueryValueEx(key, name)
    except OSError:
        return ""
    return str(value).strip()


def load_user_env(names: Sequence[str] = ("GEMINI_API_KEY", "GROQ_API_KEY")) -> dict[str, bool]:
    """Populate missing keys from the Windows User environment. Returns presence booleans only."""
    present: dict[str, bool] = {}
    for name in names:
        value = os.environ.get(name, "").strip()
        if not value:
            value = _read_windows_user_env(name)
            if value:
                os.environ[name] = value
        present[name] = bool(value)
    return present


def classify_error(exc: BaseException) -> tuple[int | None, float | None]:
    """Return (HTTP-like status, Retry-After seconds) for an SDK or network exception."""
    status: int | None = None
    for attr in ("status_code", "code"):
        value = getattr(exc, attr, None)
        if isinstance(value, int) and not isinstance(value, bool):
            status = value
            break
    response = getattr(exc, "response", None)
    if status is None and response is not None:
        resp_status = getattr(response, "status_code", None)
        if isinstance(resp_status, int) and not isinstance(resp_status, bool):
            status = resp_status
    retry_after: float | None = None
    headers = getattr(response, "headers", None)
    if headers is not None:
        try:
            raw = headers.get("retry-after")
        except AttributeError:
            raw = None
        if raw is not None:
            try:
                retry_after = max(0.0, float(str(raw).strip()))
            except ValueError:
                retry_after = None
    if status is None:
        text = str(exc).upper()
        name = type(exc).__name__
        if "429" in text or "RESOURCE_EXHAUSTED" in text:
            status = 429
        elif (
            isinstance(
                exc,
                (asyncio.TimeoutError, TimeoutError, ConnectionError, httpx.TimeoutException, httpx.TransportError),
            )
            or "Timeout" in name
            or "Connection" in name
        ):
            status = 503
    return status, retry_after


def parse_evaluation(raw: str) -> CyberRAGEvaluationSchema:
    return CyberRAGEvaluationSchema.model_validate_json(_FENCE_RE.sub("", raw.strip()))


def _verdict(result: CyberRAGEvaluationSchema) -> str:
    value = result.verdict
    return value.value if isinstance(value, VerdictEnum) else str(value)


# ------------------------------------------------------------------ pacing and judges
class Pacer:
    """Start-to-start minimum gap between calls to one model (RPM / TPM guard)."""

    def __init__(
        self,
        gap_s: float,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        self.gap_s = float(gap_s)
        self._clock = clock
        self._sleep = sleep
        self._lock = asyncio.Lock()
        self._last: float | None = None

    async def wait(self) -> None:
        async with self._lock:
            if self._last is not None:
                remaining = self.gap_s - (self._clock() - self._last)
                if remaining > 0:
                    await self._sleep(remaining)
            self._last = self._clock()


@dataclass
class ThinkingState:
    applied: bool = True


@dataclass
class Vote:
    role: str
    model_id: str
    result: CyberRAGEvaluationSchema
    thinking_applied: bool | None

    def brief(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "model": self.model_id,
            "verdict": _verdict(self.result),
            "faithfulness_score": self.result.faithfulness_score,
            "context_relevance_score": self.result.context_relevance_score,
            "thinking_applied": self.thinking_applied,
        }


class Judge:
    """One model behind a pacer, with Full Jitter retries, timeout and one format-repair retry."""

    def __init__(
        self,
        role: str,
        model_id: str,
        attempt: Callable[[str], Awaitable[str]],
        pacer: Pacer,
        *,
        thinking: ThinkingState | None = None,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        rng: random.Random | None = None,
    ) -> None:
        self.role = role
        self.model_id = model_id
        self._attempt = attempt
        self._pacer = pacer
        self._thinking = thinking
        self._sleep = sleep
        self._rng = rng

    @property
    def thinking_applied(self) -> bool | None:
        return None if self._thinking is None else self._thinking.applied

    async def _call(self, user_text: str) -> str:
        last_status: int | None = None
        for attempt_no in range(MAX_ATTEMPTS):
            await self._pacer.wait()
            try:
                return await asyncio.wait_for(self._attempt(user_text), REQUEST_TIMEOUT_S)
            except JudgeOutputError:
                raise
            except Exception as exc:
                status, retry_after = classify_error(exc)
                if status not in RETRYABLE_STATUS:
                    raise JudgeUnavailable(
                        f"{self.model_id}: non-retryable failure {type(exc).__name__} ({status}): {redact(str(exc))[:200]}"
                    ) from exc
                last_status = status
                if attempt_no + 1 < MAX_ATTEMPTS:
                    await self._sleep(max(full_jitter_delay(attempt_no, rng=self._rng), retry_after or 0.0))
        if last_status == 429:
            raise RateLimited(f"{self.model_id}: HTTP 429 after {MAX_ATTEMPTS} attempts")
        raise JudgeUnavailable(f"{self.model_id}: HTTP {last_status} after {MAX_ATTEMPTS} attempts")

    async def evaluate(self, payload: str) -> Vote:
        raw = await self._call(payload)
        try:
            result = parse_evaluation(raw)
        except (ValidationError, ValueError) as first:
            repair = (
                payload + "\n\n### FORMAT ERROR\nYour previous reply failed schema validation: "
                f"{str(first)[:200]}\nReply again with one valid JSON object only."
            )
            raw = await self._call(repair)
            try:
                result = parse_evaluation(raw)
            except (ValidationError, ValueError) as second:
                raise JudgeOutputError(f"{self.model_id}: unparseable judge output: {str(second)[:200]}") from second
        return Vote(self.role, self.model_id, result, self.thinking_applied)


def make_gemini_attempt(
    client: Any,
    model_id: str,
    state: ThinkingState,
    budget: int = THINKING_BUDGET,
) -> Callable[[str], Awaitable[str]]:
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_id}:generateContent"

    async def raw_call(user_text: str, thinking: bool) -> str:
        gen_config: dict[str, Any] = {
            "temperature": 0.0,
            "responseMimeType": "application/json",
            "responseSchema": gemini_schema(),
        }
        if thinking:
            gen_config["thinkingConfig"] = {"thinkingBudget": budget}
        body = {
            "systemInstruction": {"parts": [{"text": SYSTEM_RUBRIC}]},
            "contents": [{"role": "user", "parts": [{"text": user_text}]}],
            "generationConfig": gen_config,
        }
        resp = await client.post(url, json=body)
        resp.raise_for_status()
        try:
            payload = resp.json()
        except ValueError as exc:
            raise JudgeOutputError(f"{model_id}: Gemini HTTP 200 returned invalid JSON: {exc}") from exc
        candidates = payload.get("candidates") if isinstance(payload, dict) else None
        if not isinstance(candidates, list) or not candidates:
            raise JudgeOutputError(f"{model_id}: Gemini response missing candidates: {redact(str(payload)[:200])}")
        first = candidates[0] if isinstance(candidates[0], dict) else {}
        content = first.get("content") if isinstance(first.get("content"), dict) else {}
        parts = content.get("parts")
        if not isinstance(parts, list) or not parts:
            raise JudgeOutputError(f"{model_id}: Gemini response missing content parts: {redact(str(first)[:200])}")
        return "".join(str(p.get("text") or "") for p in parts if isinstance(p, dict) and not p.get("thought"))

    async def attempt(user_text: str) -> str:
        try:
            return await raw_call(user_text, state.applied)
        except Exception as exc:
            status, _ = classify_error(exc)
            resp_text = getattr(getattr(exc, "response", None), "text", "") or ""
            if status == 400 and state.applied and "thinking" in f"{exc} {resp_text}".lower():
                state.applied = False
                return await raw_call(user_text, False)
            raise

    return attempt


def make_groq_attempt(client: Any, model_id: str) -> Callable[[str], Awaitable[str]]:
    async def attempt(user_text: str) -> str:
        completion = await client.chat.completions.create(
            model=model_id,
            messages=[
                {"role": "system", "content": SYSTEM_RUBRIC},
                {"role": "user", "content": user_text},
            ],
            temperature=0.0,
            response_format={
                "type": "json_schema",
                "json_schema": {"name": "cyber_rag_evaluation", "strict": True, "schema": WIRE_SCHEMA},
            },
        )
        return completion.choices[0].message.content or ""

    return attempt


# ------------------------------------------------------------------ aggregation and escalation
def escalation_reasons(result: CyberRAGEvaluationSchema) -> list[str]:
    reasons: list[str] = []
    if _verdict(result) == "PARTIAL":
        reasons.append("VERDICT_PARTIAL")
    if FAITHFUL_BORDERLINE_FLOOR <= result.faithfulness_score < FAITHFUL_PASS_FLOOR:
        reasons.append("FAITHFULNESS_BORDERLINE")
    if not result.conflict_resolution_passed:
        reasons.append("CONFLICT_RESOLUTION_FAILED")
    return reasons


def aggregate(votes: Sequence[Vote]) -> dict[str, Any]:
    n = len(votes)
    if n == 0:
        raise ValueError("aggregate() needs at least one vote")
    counts = Counter(_verdict(v.result) for v in votes)
    final = next((name for name in ("PASS", "PARTIAL", "FAIL") if counts[name] * 2 > n), "PARTIAL")
    need = n // 2 + 1
    labels = {"stage1": "Stage 1", "gemini_jury": "Jury Gemini", "groq_jury": "Jury Groq"}
    if n == 1:
        reasoning = votes[0].result.reasoning
    else:
        reasoning = " | ".join(f"[{labels.get(v.role, v.role)} {v.model_id}]: {v.result.reasoning}" for v in votes)
    return {
        "reasoning": reasoning,
        "faithfulness_score": min(v.result.faithfulness_score for v in votes),
        "context_relevance_score": min(v.result.context_relevance_score for v in votes),
        "negative_constraint_passed": sum(v.result.negative_constraint_passed for v in votes) >= need,
        "conflict_resolution_passed": sum(v.result.conflict_resolution_passed for v in votes) >= need,
        "verdict": final,
    }


@dataclass
class Cascade:
    stage1: Judge
    stage1_fallback: Judge | None
    jury_gemini: Judge
    jury_groq: Judge
    semaphore: asyncio.BoundedSemaphore = field(default_factory=lambda: asyncio.BoundedSemaphore(JURY_CONCURRENCY))
    _closers: tuple[Callable[[], Awaitable[None]], ...] = field(default=(), repr=False)

    async def aclose(self) -> None:
        for closer in self._closers:
            await closer()

    async def _stage1(self, payload: str) -> tuple[Vote | None, list[str]]:
        for judge in (self.stage1, self.stage1_fallback):
            if judge is None:
                continue
            try:
                return await judge.evaluate(payload), []
            except RateLimited:
                continue
        return None, ["STAGE1_HTTP_429"]

    async def _jury(self, judge: Judge, payload: str) -> Vote:
        async with self.semaphore:
            return await judge.evaluate(payload)

    @staticmethod
    def _finish(votes: list[Vote], stage: str, reasons: list[str], failures: list[str]) -> dict[str, Any]:
        return {
            **aggregate(votes),
            "evaluation_stage": stage,
            "escalation_reasons": reasons,
            "panel": [v.brief() for v in votes],
            "panel_failures": failures,
            "stage1_model": votes[0].model_id if votes[0].role == "stage1" else None,
        }

    async def grade(self, payload: str) -> dict[str, Any]:
        stage1_vote, reasons = await self._stage1(payload)
        if stage1_vote is not None:
            reasons = escalation_reasons(stage1_vote.result)
            if not reasons:
                return self._finish([stage1_vote], "STAGE_1_BASELINE", [], [])
        jurors = (self.jury_gemini, self.jury_groq)
        outcomes = await asyncio.gather(*(self._jury(j, payload) for j in jurors), return_exceptions=True)
        votes: list[Vote] = [stage1_vote] if stage1_vote is not None else []
        failures: list[str] = []
        for judge, outcome in zip(jurors, outcomes):
            if isinstance(outcome, Vote):
                votes.append(outcome)
            elif isinstance(outcome, (JudgeUnavailable, JudgeOutputError)):
                failures.append(f"{judge.model_id}: {redact(str(outcome))[:160]}")
            else:
                raise outcome
        if len(votes) < 2:
            raise CascadeError(f"jury produced {len(votes)} usable vote(s): {failures}")
        return self._finish(votes, "STAGE_2_JURY_CONSENSUS", reasons, failures)


# ------------------------------------------------------------------ items, Stage 0, records
def is_refusal(answer: str) -> bool:
    return bool(_REFUSAL_RE.match(answer))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}: invalid JSON on line {lineno}: {exc}") from exc
    return rows


def load_items(path: Path) -> list[dict[str, Any]]:
    items = load_jsonl(path)
    seen: set[str] = set()
    for item in items:
        missing = [k for k in REQUIRED_ITEM_KEYS if k not in item]
        if missing:
            raise ValueError(f"{item.get('test_id', '?')}: missing keys {missing}")
        if item["test_id"] in seen:
            raise ValueError(f"duplicate test_id {item['test_id']}")
        seen.add(item["test_id"])
        if not isinstance(item["answer"], str) or not isinstance(item["context_chunks"], list):
            raise ValueError(f"{item['test_id']}: answer must be text and context_chunks a list")
    return items


def load_golden(path: Path) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        return {}
    return {row["test_id"]: row for row in load_jsonl(path)}


def input_hash(item: dict[str, Any]) -> str:
    blob = json.dumps(
        [
            item["query"],
            item["ground_truth_answer"],
            item["answer"],
            [[c.get("chunk_id"), c.get("content")] for c in item["context_chunks"]],
        ],
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def compute_context_sufficient(item: dict[str, Any], golden_row: dict[str, Any] | None) -> bool | None:
    if item["question_type"] == "OUT_OF_SCOPE" or golden_row is None:
        return None
    text = "\n".join(str(c.get("content", "")) for c in item["context_chunks"]).lower()
    return all(str(entity).lower() in text for entity in (golden_row.get("required_entities") or []))


def build_payload(item: dict[str, Any]) -> str:
    chunks = item["context_chunks"]
    context = "\n---\n".join(f"[{c.get('chunk_id', '?')}]\n{c.get('content', '')}" for c in chunks)
    return (
        f"### QUESTION CATEGORY:\n{item['question_type']}\n\n"
        f"### USER QUERY:\n{item['query']}\n\n"
        f"### GROUND TRUTH REFERENCE ANSWER:\n{item['ground_truth_answer']}\n\n"
        f"### RETRIEVED CONTEXT CHUNKS:\n{context or '(no context chunks were retrieved)'}\n\n"
        f"### CANDIDATE ANSWER TO GRADE:\n{item['answer']}"
    )


def deterministic_grade(item: dict[str, Any]) -> dict[str, Any] | None:
    refusal = is_refusal(item["answer"])
    out_of_scope = item["question_type"] == "OUT_OF_SCOPE"
    if out_of_scope and refusal:
        return {
            "reasoning": "Deterministic: out-of-scope question and the answer is the required refusal.",
            "faithfulness_score": 1.0,
            "context_relevance_score": 1.0,
            "negative_constraint_passed": True,
            "conflict_resolution_passed": True,
            "verdict": "PASS",
        }
    if item["bypassed_llm"] and refusal and not out_of_scope:
        return {
            "reasoning": "Deterministic: retrieval abstained and the pipeline returned the canonical refusal for an answerable question; no generation to audit.",
            "faithfulness_score": 1.0,
            "context_relevance_score": 0.0,
            "negative_constraint_passed": True,
            "conflict_resolution_passed": True,
            "verdict": "PARTIAL",
        }
    return None


async def evaluate_item(cascade: Cascade, item: dict[str, Any], golden_row: dict[str, Any] | None) -> dict[str, Any]:
    started = time.perf_counter()
    graded = deterministic_grade(item)
    if graded is not None:
        graded = {
            **graded,
            "evaluation_stage": "STAGE_0_DETERMINISTIC",
            "escalation_reasons": [],
            "panel": [],
            "panel_failures": [],
            "stage1_model": None,
        }
    else:
        graded = await cascade.grade(build_payload(item))
    return {
        "test_id": item["test_id"],
        "question_type": item["question_type"],
        "bypassed_llm": bool(item["bypassed_llm"]),
        "answer_is_refusal": is_refusal(item["answer"]),
        "context_sufficient": compute_context_sufficient(item, golden_row),
        "input_sha256": input_hash(item),
        **graded,
        "audit_latency_sec": round(time.perf_counter() - started, 2),
    }


# ------------------------------------------------------------------ atomic checkpoint
def repair_tail(path: Path) -> None:
    """Truncate a torn final line left by a crash so the next append starts on a clean line."""
    if not path.exists():
        return
    data = path.read_bytes()
    if not data or data.endswith(b"\n"):
        return
    cut = data.rfind(b"\n")
    with path.open("r+b") as fh:
        fh.truncate(cut + 1 if cut >= 0 else 0)
        fh.flush()
        os.fsync(fh.fileno())


def load_checkpoint(path: Path) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return records
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            record = json.loads(line)
            records[record["test_id"]] = record
    return records


def append_record(handle: Any, record: dict[str, Any]) -> None:
    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    handle.flush()
    os.fsync(handle.fileno())


async def run_batch(
    cascade: Cascade,
    items: list[dict[str, Any]],
    golden: dict[str, dict[str, Any]],
    checkpoint: Path,
    *,
    limit: int | None = None,
    log: Callable[[str], None] = print,
) -> dict[str, Any]:
    repair_tail(checkpoint)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    done = load_checkpoint(checkpoint)
    pending = [it for it in items if done.get(it["test_id"], {}).get("input_sha256") != input_hash(it)]
    cached = len(items) - len(pending)
    if limit is not None:
        pending = pending[:limit]
    log(f"[INIT] {len(items)} items | {cached} already checkpointed | {len(pending)} to evaluate now")
    failed: list[dict[str, str]] = []
    stages: Counter[str] = Counter()
    written = 0
    with checkpoint.open("a", encoding="utf-8", newline="\n") as handle:
        for idx, item in enumerate(pending, start=1):
            try:
                record = await evaluate_item(cascade, item, golden.get(item["test_id"]))
            except (JudgeUnavailable, JudgeOutputError, CascadeError) as exc:
                failed.append({"test_id": item["test_id"], "error": redact(str(exc))[:300]})
                log(f"[{idx}/{len(pending)}] {item['test_id']} FAILED (not checkpointed): {redact(str(exc))[:160]}")
                continue
            append_record(handle, record)
            written += 1
            stages[record["evaluation_stage"]] += 1
            log(
                f"[{idx}/{len(pending)}] {record['test_id']} ({record['question_type']}) -> {record['verdict']} | "
                f"faith={record['faithfulness_score']:.2f} | {record['evaluation_stage']} ({record['audit_latency_sec']}s)"
            )
    return {
        "total": len(items),
        "cached": cached,
        "attempted": len(pending),
        "written": written,
        "failed": failed,
        "stages": dict(stages),
    }


# ------------------------------------------------------------------ wiring and CLI
def build_default_cascade() -> Cascade:
    from openai import AsyncOpenAI

    gemini_key = os.environ["GEMINI_API_KEY"].strip()
    groq_key = os.environ["GROQ_API_KEY"].strip()
    register_secrets([gemini_key, groq_key])
    gemini_client = httpx.AsyncClient(
        http2=True,
        timeout=httpx.Timeout(REQUEST_TIMEOUT_S),
        headers={"x-goog-api-key": gemini_key},
        limits=httpx.Limits(max_keepalive_connections=6, keepalive_expiry=60.0),
    )
    groq_client = AsyncOpenAI(
        base_url="https://api.groq.com/openai/v1",
        api_key=groq_key,
        max_retries=0,
        timeout=REQUEST_TIMEOUT_S,
    )

    def gemini(role: str, model_id: str) -> Judge:
        state = ThinkingState(applied=True)
        return Judge(
            role,
            model_id,
            make_gemini_attempt(gemini_client, model_id, state),
            Pacer(PACING_S[model_id]),
            thinking=state,
        )

    return Cascade(
        stage1=gemini("stage1", STAGE1_MODEL),
        stage1_fallback=gemini("stage1", STAGE1_FALLBACK_MODEL),
        jury_gemini=gemini("gemini_jury", JURY_GEMINI_MODEL),
        jury_groq=Judge(
            "groq_jury",
            JURY_GROQ_MODEL,
            make_groq_attempt(groq_client, JURY_GROQ_MODEL),
            Pacer(PACING_S[JURY_GROQ_MODEL]),
        ),
        _closers=(gemini_client.aclose, groq_client.close),
    )


async def ping(cascade: Cascade) -> bool:
    ok = True
    for judge in (cascade.stage1, cascade.jury_gemini, cascade.jury_groq):
        started = time.perf_counter()
        try:
            vote = await judge.evaluate(PING_PAYLOAD)
        except (JudgeUnavailable, JudgeOutputError) as exc:
            ok = False
            print(f"[PING] {judge.role:<12} {judge.model_id:<26} FAILED: {redact(str(exc))[:200]}")
            continue
        print(
            f"[PING] {judge.role:<12} {judge.model_id:<26} OK {time.perf_counter() - started:5.1f}s "
            f"verdict={_verdict(vote.result)} thinking_applied={vote.thinking_applied}"
        )
    return ok


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m kavach.evaluation.cascade_judge",
        description="Tier 6 two-stage cross-family judge cascade.",
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--golden", type=Path, default=DEFAULT_GOLDEN)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        metavar="N",
        help="Evaluate at most N pending items (smoke test).",
    )
    parser.add_argument(
        "--ping",
        action="store_true",
        help="Verify keys, model names and structured output; no checkpoint writes.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(list(argv) if argv is not None else None)
    present = load_user_env()
    missing = [name for name, ok in present.items() if not ok]
    if missing:
        print(f"[CONFIG] missing environment variables: {', '.join(missing)}", file=sys.stderr)
        return 2
    try:
        cascade = build_default_cascade()
    except ImportError as exc:
        print(f"[CONFIG] provider SDK not installed: {exc}", file=sys.stderr)
        return 2

    async def _run() -> int:
        try:
            if args.ping:
                return 0 if await ping(cascade) else 1
            items = load_items(args.input)
            summary = await run_batch(
                cascade,
                items,
                load_golden(args.golden),
                args.checkpoint,
                limit=args.limit,
            )
            print(
                f"[DONE] written={summary['written']} cached={summary['cached']} "
                f"failed={len(summary['failed'])} stages={summary['stages']}"
            )
            return 1 if summary["failed"] else 0
        finally:
            await cascade.aclose()

    return asyncio.run(_run())


if __name__ == "__main__":
    raise SystemExit(main())
