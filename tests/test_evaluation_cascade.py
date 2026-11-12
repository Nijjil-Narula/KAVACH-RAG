"""Offline unit tests for kavach.evaluation.schemas and kavach.evaluation.cascade_judge."""
from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
import random
import subprocess
import sys
from typing import Any

import httpx
import pytest
from pydantic import ValidationError

from kavach.evaluation import cascade_judge as cj
from kavach.evaluation.schemas import (
    CyberRAGEvaluationSchema,
    FIELD_ORDER,
    WIRE_SCHEMA,
    VerdictEnum,
    gemini_schema,
)
from kavach.generation.prompts import is_refusal as frozen_is_refusal


def _valid_eval_dict(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "reasoning": "All claims match retrieved chunks accurately.",
        "faithfulness_score": 0.95,
        "context_relevance_score": 1.0,
        "negative_constraint_passed": True,
        "conflict_resolution_passed": True,
        "verdict": "PASS",
    }
    base.update(overrides)
    return base


def _valid_eval_json(**overrides: Any) -> str:
    return json.dumps(_valid_eval_dict(**overrides))


def _make_vote(
    role: str = "stage1",
    model_id: str = "m1",
    thinking_applied: bool | None = True,
    **overrides: Any,
) -> cj.Vote:
    result = CyberRAGEvaluationSchema(**_valid_eval_dict(**overrides))
    return cj.Vote(role=role, model_id=model_id, result=result, thinking_applied=thinking_applied)


def _sample_item(test_id: str = "TEST_001", **overrides: Any) -> dict[str, Any]:
    item: dict[str, Any] = {
        "test_id": test_id,
        "question_type": "FACTUAL",
        "query": "What is the CVSS score of CVE-2021-44228?",
        "ground_truth_answer": "CVE-2021-44228 has CVSS 10.0 (CRITICAL).",
        "answer": "CVE-2021-44228 has CVSS 10.0 (CRITICAL). [CVE-2021-44228_chunk_001]",
        "bypassed_llm": False,
        "context_chunks": [
            {
                "chunk_id": "CVE-2021-44228_chunk_001",
                "doc_id": "CVE-2021-44228",
                "standard_version": "NVD_5.0",
                "content": "CVE-2021-44228 | CVSS 10.0 (CRITICAL)",
            }
        ],
    }
    item.update(overrides)
    return item


def _fast_pacer(gap_s: float = 0.0) -> cj.Pacer:
    async def noop_sleep(_: float) -> None:
        return None

    return cj.Pacer(gap_s, clock=lambda: 0.0, sleep=noop_sleep)


# --------------------------------------------------------------------------- 1. Schema
def test_schema_order_bounds_and_wire_formats() -> None:
    model_fields = tuple(CyberRAGEvaluationSchema.model_fields.keys())
    assert model_fields == FIELD_ORDER
    assert model_fields[0] == "reasoning"

    # extra="forbid"
    with pytest.raises(ValidationError):
        CyberRAGEvaluationSchema(**_valid_eval_dict(extra_key="nope"))

    # Empty reasoning rejected (min_length=1)
    with pytest.raises(ValidationError):
        CyberRAGEvaluationSchema(**_valid_eval_dict(reasoning=""))

    # Numeric bounds enforced on Pydantic model (0.0 <= x <= 1.0)
    ok_low = CyberRAGEvaluationSchema(**_valid_eval_dict(faithfulness_score=0.0, context_relevance_score=0.0))
    ok_high = CyberRAGEvaluationSchema(**_valid_eval_dict(faithfulness_score=1.0, context_relevance_score=1.0))
    assert ok_low.faithfulness_score == 0.0
    assert ok_high.context_relevance_score == 1.0
    assert ok_high.verdict == VerdictEnum.PASS.value

    for field_name in ("faithfulness_score", "context_relevance_score"):
        for bad_val in (-0.01, 1.01):
            with pytest.raises(ValidationError):
                CyberRAGEvaluationSchema(**_valid_eval_dict(**{field_name: bad_val}))

    # WIRE_SCHEMA checks
    assert tuple(WIRE_SCHEMA["properties"].keys()) == FIELD_ORDER
    assert WIRE_SCHEMA["required"] == list(FIELD_ORDER)
    assert WIRE_SCHEMA["additionalProperties"] is False

    # gemini_schema checks
    g = gemini_schema()
    assert g["type"] == "OBJECT"
    assert tuple(g["properties"].keys()) == FIELD_ORDER
    assert g["propertyOrdering"] == list(FIELD_ORDER)
    assert g["required"] == list(FIELD_ORDER)
    assert "additionalProperties" not in g
    for prop in g["properties"].values():
        assert prop["type"] in {"STRING", "NUMBER", "INTEGER", "BOOLEAN", "ARRAY", "OBJECT"}
        assert "additionalProperties" not in prop
        for bound_key in ("minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum"):
            assert bound_key not in prop


# --------------------------------------------------------------------------- 2. escalation_reasons
@pytest.mark.parametrize(
    ("verdict", "faithfulness", "conflict_ok", "expected"),
    [
        ("PASS", 0.85, True, []),
        ("PASS", 0.99, True, []),
        ("FAIL", 0.49, True, []),
        ("PASS", 0.50, True, ["FAITHFULNESS_BORDERLINE"]),
        ("PASS", 0.849, True, ["FAITHFULNESS_BORDERLINE"]),
        ("PARTIAL", 0.90, True, ["VERDICT_PARTIAL"]),
        ("PASS", 0.95, False, ["CONFLICT_RESOLUTION_FAILED"]),
        (
            "PARTIAL",
            0.70,
            False,
            ["VERDICT_PARTIAL", "FAITHFULNESS_BORDERLINE", "CONFLICT_RESOLUTION_FAILED"],
        ),
    ],
)
def test_escalation_reasons_truth_table(
    verdict: str,
    faithfulness: float,
    conflict_ok: bool,
    expected: list[str],
) -> None:
    ev = CyberRAGEvaluationSchema(
        **_valid_eval_dict(
            verdict=verdict,
            faithfulness_score=faithfulness,
            conflict_resolution_passed=conflict_ok,
        )
    )
    assert cj.escalation_reasons(ev) == expected


# --------------------------------------------------------------------------- 3. aggregate
def test_aggregate_voting_scores_and_empty_error() -> None:
    with pytest.raises(ValueError):
        cj.aggregate([])

    # 3-vote majority PASS
    v_pass_1 = _make_vote("stage1", "m1", faithfulness_score=0.95, context_relevance_score=1.0, verdict="PASS")
    v_pass_2 = _make_vote("gemini_jury", "m2", faithfulness_score=0.80, context_relevance_score=0.5, verdict="PASS")
    v_fail_1 = _make_vote(
        "groq_jury",
        "m3",
        faithfulness_score=0.40,
        context_relevance_score=0.5,
        negative_constraint_passed=True,
        conflict_resolution_passed=False,
        verdict="FAIL",
    )
    agg3 = cj.aggregate([v_pass_1, v_pass_2, v_fail_1])
    assert agg3["verdict"] == "PASS"
    assert agg3["faithfulness_score"] == 0.40
    assert agg3["context_relevance_score"] == 0.5
    assert agg3["negative_constraint_passed"] is True
    assert agg3["conflict_resolution_passed"] is True
    assert "[Stage 1 m1]:" in agg3["reasoning"]
    assert "[Jury Groq m3]:" in agg3["reasoning"]

    # 3-way split -> PARTIAL
    v_partial = _make_vote(
        "gemini_jury",
        "m2",
        verdict="PARTIAL",
        faithfulness_score=0.60,
        context_relevance_score=0.5,
        negative_constraint_passed=False,
    )
    split = cj.aggregate([v_pass_1, v_partial, v_fail_1])
    assert split["verdict"] == "PARTIAL"
    assert split["negative_constraint_passed"] is True

    # 2-vote disagree -> PARTIAL, boolean majority requires 2/2
    disagree2 = cj.aggregate([v_pass_1, v_fail_1])
    assert disagree2["verdict"] == "PARTIAL"
    assert disagree2["negative_constraint_passed"] is True
    assert disagree2["conflict_resolution_passed"] is False


# --------------------------------------------------------------------------- 4. Clean Stage 1
def test_clean_stage1_makes_one_call_zero_jury() -> None:
    calls: list[str] = []

    async def ok_s1(_: str) -> str:
        calls.append("s1")
        return _valid_eval_json(verdict="PASS", faithfulness_score=0.92)

    async def fail_if_called(_: str) -> str:
        calls.append("unexpected")
        raise AssertionError("Fallback or jury must not be called")

    cascade = cj.Cascade(
        stage1=cj.Judge("stage1", cj.STAGE1_MODEL, ok_s1, _fast_pacer()),
        stage1_fallback=cj.Judge("stage1", cj.STAGE1_FALLBACK_MODEL, fail_if_called, _fast_pacer()),
        jury_gemini=cj.Judge("gemini_jury", cj.JURY_GEMINI_MODEL, fail_if_called, _fast_pacer()),
        jury_groq=cj.Judge("groq_jury", cj.JURY_GROQ_MODEL, fail_if_called, _fast_pacer()),
    )

    res = asyncio.run(cascade.grade("payload"))
    assert calls == ["s1"]
    assert res["evaluation_stage"] == "STAGE_1_BASELINE"
    assert res["escalation_reasons"] == []
    assert len(res["panel"]) == 1
    assert res["verdict"] == "PASS"


# --------------------------------------------------------------------------- 5. Borderline escalation + concurrency
def test_borderline_escalates_and_runs_jurors_concurrently() -> None:
    active_jurors = 0
    max_concurrent = 0
    both_started = asyncio.Event()

    async def s1_borderline(_: str) -> str:
        return _valid_eval_json(verdict="PASS", faithfulness_score=0.72)

    async def juror_fn(verdict: str) -> str:
        nonlocal active_jurors, max_concurrent
        active_jurors += 1
        max_concurrent = max(max_concurrent, active_jurors)
        if active_jurors == 2:
            both_started.set()
        await asyncio.wait_for(both_started.wait(), timeout=1.0)
        active_jurors -= 1
        return _valid_eval_json(verdict=verdict, faithfulness_score=0.78)

    async def unused(_: str) -> str:
        raise AssertionError("Unused judge called")

    cascade = cj.Cascade(
        stage1=cj.Judge("stage1", cj.STAGE1_MODEL, s1_borderline, _fast_pacer()),
        stage1_fallback=cj.Judge("stage1", cj.STAGE1_FALLBACK_MODEL, unused, _fast_pacer()),
        jury_gemini=cj.Judge("gemini_jury", cj.JURY_GEMINI_MODEL, lambda t: juror_fn("PASS"), _fast_pacer()),
        jury_groq=cj.Judge("groq_jury", cj.JURY_GROQ_MODEL, lambda t: juror_fn("PASS"), _fast_pacer()),
    )

    res = asyncio.run(cascade.grade("payload"))
    assert max_concurrent == 2
    assert res["evaluation_stage"] == "STAGE_2_JURY_CONSENSUS"
    assert res["escalation_reasons"] == ["FAITHFULNESS_BORDERLINE"]
    assert len(res["panel"]) == 3


# --------------------------------------------------------------------------- 6. Stage 1 HTTP 429 fallback & escalation
def test_stage1_429_fallback_and_double_429_escalation() -> None:
    class Http429(Exception):
        status_code = 429

    async def noop_sleep(_: float) -> None:
        return None

    async def always_429(_: str) -> str:
        raise Http429("rate limited")

    async def ok_fallback(_: str) -> str:
        return _valid_eval_json(verdict="PASS", faithfulness_score=0.90)

    async def ok_jury(_: str) -> str:
        return _valid_eval_json(verdict="PASS", faithfulness_score=0.91)

    # Primary 429s -> fallback gemini-3.1-flash-lite succeeds
    cascade_fb_ok = cj.Cascade(
        stage1=cj.Judge("stage1", cj.STAGE1_MODEL, always_429, _fast_pacer(), sleep=noop_sleep),
        stage1_fallback=cj.Judge("stage1", "gemini-3.1-flash-lite", ok_fallback, _fast_pacer(), sleep=noop_sleep),
        jury_gemini=cj.Judge("gemini_jury", cj.JURY_GEMINI_MODEL, ok_jury, _fast_pacer()),
        jury_groq=cj.Judge("groq_jury", cj.JURY_GROQ_MODEL, ok_jury, _fast_pacer()),
    )
    res_fb = asyncio.run(cascade_fb_ok.grade("payload"))
    assert res_fb["evaluation_stage"] == "STAGE_1_BASELINE"
    assert res_fb["panel"][0]["model"] == "gemini-3.1-flash-lite"

    # Primary + fallback both 429 -> escalate with STAGE1_HTTP_429 and 2-vote jury
    cascade_both_429 = cj.Cascade(
        stage1=cj.Judge("stage1", cj.STAGE1_MODEL, always_429, _fast_pacer(), sleep=noop_sleep),
        stage1_fallback=cj.Judge("stage1", "gemini-3.1-flash-lite", always_429, _fast_pacer(), sleep=noop_sleep),
        jury_gemini=cj.Judge("gemini_jury", cj.JURY_GEMINI_MODEL, ok_jury, _fast_pacer()),
        jury_groq=cj.Judge("groq_jury", cj.JURY_GROQ_MODEL, ok_jury, _fast_pacer()),
    )
    res_esc = asyncio.run(cascade_both_429.grade("payload"))
    assert res_esc["evaluation_stage"] == "STAGE_2_JURY_CONSENSUS"
    assert res_esc["escalation_reasons"] == ["STAGE1_HTTP_429"]
    assert len(res_esc["panel"]) == 2


# --------------------------------------------------------------------------- 7. Juror failures
def test_juror_failure_tolerated_at_2_votes_and_fails_below_2() -> None:
    class Http429(Exception):
        status_code = 429

    async def noop_sleep(_: float) -> None:
        return None

    async def s1_partial(_: str) -> str:
        return _valid_eval_json(verdict="PARTIAL", faithfulness_score=0.75)

    async def ok_juror(_: str) -> str:
        return _valid_eval_json(verdict="PASS", faithfulness_score=0.88)

    async def bad_json(_: str) -> str:
        return "not json"

    async def down_juror(_: str) -> str:
        raise ConnectionError("network unreachable")

    # Stage 1 + 1 healthy juror -> 2 votes + 1 panel_failure
    cascade_one_fail = cj.Cascade(
        stage1=cj.Judge("stage1", "g1", s1_partial, _fast_pacer()),
        stage1_fallback=cj.Judge("stage1", "gemini-3.1-flash-lite", ok_juror, _fast_pacer()),
        jury_gemini=cj.Judge("gemini_jury", "g2", bad_json, _fast_pacer(), sleep=noop_sleep),
        jury_groq=cj.Judge("groq_jury", "groq1", ok_juror, _fast_pacer()),
    )
    res = asyncio.run(cascade_one_fail.grade("payload"))
    assert res["evaluation_stage"] == "STAGE_2_JURY_CONSENSUS"
    assert len(res["panel"]) == 2
    assert len(res["panel_failures"]) == 1
    assert "g2:" in res["panel_failures"][0]

    # Both jurors fail -> 1 vote < 2 -> CascadeError
    cascade_both_fail = cj.Cascade(
        stage1=cj.Judge("stage1", "g1", s1_partial, _fast_pacer()),
        stage1_fallback=cj.Judge("stage1", "gemini-3.1-flash-lite", ok_juror, _fast_pacer()),
        jury_gemini=cj.Judge("gemini_jury", "g2", bad_json, _fast_pacer(), sleep=noop_sleep),
        jury_groq=cj.Judge("groq_jury", "groq1", down_juror, _fast_pacer(), sleep=noop_sleep),
    )
    with pytest.raises(cj.CascadeError):
        asyncio.run(cascade_both_fail.grade("payload"))

    # Stage 1 429 + 1 juror fail -> 1 vote < 2 -> CascadeError
    async def always_429(_: str) -> str:
        raise Http429("429")

    cascade_429_plus_one = cj.Cascade(
        stage1=cj.Judge("stage1", "g1", always_429, _fast_pacer(), sleep=noop_sleep),
        stage1_fallback=cj.Judge("stage1", "gemini-3.1-flash-lite", always_429, _fast_pacer(), sleep=noop_sleep),
        jury_gemini=cj.Judge("gemini_jury", "g2", down_juror, _fast_pacer(), sleep=noop_sleep),
        jury_groq=cj.Judge("groq_jury", "groq1", ok_juror, _fast_pacer()),
    )
    with pytest.raises(cj.CascadeError):
        asyncio.run(cascade_429_plus_one.grade("payload"))


# --------------------------------------------------------------------------- 8. Retries, Retry-After, HTTPStatusError & format repair
def test_retry_backoff_retry_after_http_status_and_format_repair() -> None:
    delays: list[float] = []

    async def record_sleep(s: float) -> None:
        delays.append(s)

    call_idx = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal call_idx
        call_idx += 1
        if call_idx == 1:
            return httpx.Response(500, json={"error": "transient"})
        if call_idx == 2:
            return httpx.Response(429, headers={"retry-after": "120"}, json={"error": "slow down"})
        return httpx.Response(
            200,
            json={"candidates": [{"content": {"parts": [{"text": _valid_eval_json()}]}}]},
        )

    async def run_mock() -> cj.Vote:
        state = cj.ThinkingState(applied=True)
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            judge = cj.Judge(
                "stage1",
                "gemini-test",
                cj.make_gemini_attempt(client, "gemini-test", state),
                _fast_pacer(),
                thinking=state,
                sleep=record_sleep,
                rng=random.Random(42),
            )
            return await judge.evaluate("hello")

    vote = asyncio.run(run_mock())
    assert call_idx == 3
    assert len(delays) == 2
    assert 0.0 <= delays[0] <= cj.BACKOFF_CAP_S
    assert delays[1] == 120.0
    assert vote.result.verdict == "PASS"

    # Also verify full_jitter_delay uses base_s and cap_s bounds
    rng = random.Random(7)
    for attempt in range(8):
        d = cj.full_jitter_delay(attempt, base_s=1.5, cap_s=60.0, rng=rng)
        assert 0.0 <= d <= min(60.0, 1.5 * (2**attempt))

    # Format repair: 1 bad JSON then valid JSON succeeds; 2 bad JSONs raises JudgeOutputError
    repair_calls = 0

    async def one_bad_then_good(_: str) -> str:
        nonlocal repair_calls
        repair_calls += 1
        if repair_calls == 1:
            return "{not valid json"
        return _valid_eval_json()

    judge_repair = cj.Judge("stage1", "m", one_bad_then_good, _fast_pacer(), sleep=record_sleep)
    repaired_vote = asyncio.run(judge_repair.evaluate("q"))
    assert repair_calls == 2
    assert repaired_vote.result.verdict == "PASS"

    async def always_bad(_: str) -> str:
        return "{still broken"

    judge_fail = cj.Judge("stage1", "m", always_bad, _fast_pacer(), sleep=record_sleep)
    with pytest.raises(cj.JudgeOutputError):
        asyncio.run(judge_fail.evaluate("q"))


# --------------------------------------------------------------------------- 9. Gemini HTTP 400 thinking fallback
def test_gemini_http400_thinking_disables_thinking_and_records_brief() -> None:
    seen_thinking_configs: list[bool] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8"))
        has_thinking = "thinkingConfig" in body["generationConfig"]
        seen_thinking_configs.append(has_thinking)
        if has_thinking:
            return httpx.Response(400, json={"error": {"message": "thinkingConfig is not supported"}})
        return httpx.Response(
            200,
            json={"candidates": [{"content": {"parts": [{"text": _valid_eval_json()}]}}]},
        )

    async def run_test() -> tuple[cj.Vote, cj.Vote]:
        state = cj.ThinkingState(applied=True)
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            judge = cj.Judge(
                "stage1",
                "gemini-no-think",
                cj.make_gemini_attempt(client, "gemini-no-think", state),
                _fast_pacer(),
                thinking=state,
            )
            v1 = await judge.evaluate("item 1")
            v2 = await judge.evaluate("item 2")
            return v1, v2

    v1, v2 = asyncio.run(run_test())
    assert seen_thinking_configs == [True, False, False]
    assert v1.thinking_applied is False
    assert v1.brief()["thinking_applied"] is False
    assert v2.brief()["thinking_applied"] is False


# --------------------------------------------------------------------------- 10. Pacer
def test_pacer_sleeps_exact_remaining_gap() -> None:
    now = 100.0
    sleeps: list[float] = []

    def fake_clock() -> float:
        return now

    async def fake_sleep(delay: float) -> None:
        nonlocal now
        sleeps.append(delay)
        now += delay

    pacer = cj.Pacer(gap_s=4.0, clock=fake_clock, sleep=fake_sleep)

    async def run_calls() -> None:
        nonlocal now
        await pacer.wait()  # t=100.0 -> no sleep
        now += 1.25         # t=101.25 -> gap=4.0, remaining=2.75
        await pacer.wait()
        now += 5.0          # t=109.0 -> elapsed=5.0 >= 4.0 -> no sleep
        await pacer.wait()

    asyncio.run(run_calls())
    assert len(sleeps) == 1
    assert sleeps[0] == pytest.approx(2.75)


# --------------------------------------------------------------------------- 11. Atomic checkpointing
def test_atomic_checkpoint_fsync_resume_repair_tail_and_retry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fsync_calls = 0
    real_fsync = os.fsync

    def counting_fsync(fd: int) -> None:
        nonlocal fsync_calls
        fsync_calls += 1
        real_fsync(fd)

    monkeypatch.setattr(cj.os, "fsync", counting_fsync)

    eval_calls: list[str] = []
    fail_t2 = True

    async def s1_fn(payload: str) -> str:
        marker = "### CANDIDATE ANSWER TO GRADE:\n"
        ans = payload.split(marker, 1)[1]
        eval_calls.append(ans)
        if "FAIL_ME" in ans and fail_t2:
            raise ConnectionError("transient crash")
        return _valid_eval_json()

    async def unused(_: str) -> str:
        raise AssertionError("Unused")

    async def noop_sleep(_: float) -> None:
        return None

    cascade = cj.Cascade(
        stage1=cj.Judge("stage1", "m1", s1_fn, _fast_pacer(), sleep=noop_sleep),
        stage1_fallback=cj.Judge("stage1", "m2", unused, _fast_pacer()),
        jury_gemini=cj.Judge("gemini_jury", "m3", unused, _fast_pacer()),
        jury_groq=cj.Judge("groq_jury", "m4", unused, _fast_pacer()),
    )

    ckpt = tmp_path / "checkpoints.jsonl"
    items = [
        _sample_item("TEST_001", answer="Answer 1 [CVE-2021-44228_chunk_001]"),
        _sample_item("TEST_002", answer="FAIL_ME Answer 2 [CVE-2021-44228_chunk_001]"),
        _sample_item("TEST_003", answer="Answer 3 [CVE-2021-44228_chunk_001]"),
    ]
    golden = {it["test_id"]: {"test_id": it["test_id"], "required_entities": ["CVE-2021-44228"]} for it in items}

    # Run 1: TEST_002 fails after MAX_ATTEMPTS -> nothing written for TEST_002; TEST_001 & TEST_003 fsynced once each
    summary1 = asyncio.run(cj.run_batch(cascade, items, golden, ckpt))
    assert summary1["written"] == 2
    assert len(summary1["failed"]) == 1
    assert summary1["failed"][0]["test_id"] == "TEST_002"
    assert fsync_calls == 2
    assert len(eval_calls) == 2 + cj.MAX_ATTEMPTS

    # Append a torn final line and verify repair_tail truncates it cleanly
    with ckpt.open("a", encoding="utf-8", newline="\n") as fh:
        fh.write('{"test_id": "TORN_TAIL", "broken": ')
    cj.repair_tail(ckpt)
    assert fsync_calls == 3
    loaded = cj.load_checkpoint(ckpt)
    assert set(loaded.keys()) == {"TEST_001", "TEST_003"}

    # Run 2 with TEST_002 succeeding and TEST_003 answer changed -> TEST_001 skipped, TEST_002 & TEST_003 evaluated
    fail_t2 = False
    eval_calls.clear()
    items_updated = [
        _sample_item("TEST_001", answer="Answer 1 [CVE-2021-44228_chunk_001]"),
        _sample_item("TEST_002", answer="FAIL_ME Answer 2 [CVE-2021-44228_chunk_001]"),
        _sample_item("TEST_003", answer="Updated grounded answer. [CVE-2021-44228_chunk_001]"),
    ]
    summary2 = asyncio.run(cj.run_batch(cascade, items_updated, golden, ckpt))
    assert summary2["cached"] == 1
    assert summary2["written"] == 2
    assert summary2["failed"] == []
    assert len(eval_calls) == 2
    assert fsync_calls == 5
    final_records = cj.load_checkpoint(ckpt)
    assert set(final_records.keys()) == {"TEST_001", "TEST_002", "TEST_003"}


# --------------------------------------------------------------------------- 12. Stage 0 deterministic rules
def test_stage0_deterministic_rules() -> None:
    judge_calls: list[str] = []

    async def s1_fn(payload: str) -> str:
        judge_calls.append(payload)
        return _valid_eval_json(verdict="FAIL", faithfulness_score=0.10)

    async def unused(_: str) -> str:
        raise AssertionError("Should not reach fallback/jury")

    cascade = cj.Cascade(
        stage1=cj.Judge("stage1", "m1", s1_fn, _fast_pacer()),
        stage1_fallback=cj.Judge("stage1", "m2", unused, _fast_pacer()),
        jury_gemini=cj.Judge("gemini_jury", "m3", unused, _fast_pacer()),
        jury_groq=cj.Judge("groq_jury", "m4", unused, _fast_pacer()),
    )

    # 1) OUT_OF_SCOPE refusal -> PASS, STAGE_0_DETERMINISTIC, 0 judge calls
    oos_refused = _sample_item(
        "OOS_1",
        question_type="OUT_OF_SCOPE",
        bypassed_llm=True,
        answer="INSUFFICIENT_CONTEXT: Out of scope.",
    )
    r1 = asyncio.run(cj.evaluate_item(cascade, oos_refused, None))
    assert r1["evaluation_stage"] == "STAGE_0_DETERMINISTIC"
    assert r1["verdict"] == "PASS"
    assert r1["escalation_reasons"] == []
    assert judge_calls == []

    # 2) bypassed_llm=True + non-OUT_OF_SCOPE refusal -> PARTIAL, STAGE_0_DETERMINISTIC, 0 judge calls
    inscope_bypassed = _sample_item(
        "BYP_1",
        question_type="FACTUAL",
        bypassed_llm=True,
        answer="INSUFFICIENT_CONTEXT: Low retrieval confidence.",
    )
    r2 = asyncio.run(cj.evaluate_item(cascade, inscope_bypassed, None))
    assert r2["evaluation_stage"] == "STAGE_0_DETERMINISTIC"
    assert r2["verdict"] == "PARTIAL"
    assert r2["escalation_reasons"] == []
    assert judge_calls == []

    # 3) In-scope model refusal (bypassed_llm=False) -> reaches judge
    inscope_model_refusal = _sample_item(
        "MOD_REF",
        question_type="FACTUAL",
        bypassed_llm=False,
        answer="INSUFFICIENT_CONTEXT: Model refused.",
    )
    r3 = asyncio.run(cj.evaluate_item(cascade, inscope_model_refusal, None))
    assert r3["evaluation_stage"] == "STAGE_1_BASELINE"
    assert len(judge_calls) == 1

    # 4) Answered OUT_OF_SCOPE (non-refusal) -> reaches judge
    oos_answered = _sample_item(
        "OOS_ANS",
        question_type="OUT_OF_SCOPE",
        bypassed_llm=False,
        answer="Here is an ungrounded answer about quantum computing.",
    )
    r4 = asyncio.run(cj.evaluate_item(cascade, oos_answered, None))
    assert r4["evaluation_stage"] == "STAGE_1_BASELINE"
    assert len(judge_calls) == 2


# --------------------------------------------------------------------------- 13. redact & load_user_env
def test_redact_and_load_user_env() -> None:
    secret_a = "fake-gemini-secret-xyz-123456"
    secret_b = "fake-groq-secret-abc-987654"
    cj.register_secrets([secret_a, secret_b])
    msg = f"Failure calling {secret_a} and {secret_b} in payload"
    redacted = cj.redact(msg)
    assert secret_a not in redacted
    assert secret_b not in redacted
    assert redacted.count("[REDACTED]") == 2

    presence = cj.load_user_env()
    assert presence == {"GEMINI_API_KEY": True, "GROQ_API_KEY": True}


# --------------------------------------------------------------------------- 14. Clean subprocess import isolation
def test_clean_subprocess_excludes_torch_and_transformers() -> None:
    # Normal import: kavach/__init__.py runs apply_hardware_guard() which imports CPU torch
    # and faiss to configure thread limits, while transformers, kavach.retrieval, and
    # kavach.generation must stay unloaded.
    normal_code = (
        "import sys; "
        "import kavach.evaluation.schemas, kavach.evaluation.cascade_judge; "
        "assert 'transformers' not in sys.modules, 'transformers imported'; "
        "assert 'kavach.retrieval' not in sys.modules, 'retrieval imported'; "
        "assert 'kavach.generation' not in sys.modules, 'generation imported'"
    )
    proc_normal = subprocess.run(
        [sys.executable, "-c", normal_code],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc_normal.returncode == 0, proc_normal.stderr

    # Stubbed package root (bypassing frozen kavach/__init__.py hardware guard):
    # proves kavach/evaluation/* itself imports neither torch nor transformers.
    isolated_code = (
        "import pathlib, sys, types; "
        "pkg = types.ModuleType('kavach'); "
        "pkg.__path__ = [str(pathlib.Path('kavach').resolve())]; "
        "sys.modules['kavach'] = pkg; "
        "import kavach.evaluation.schemas, kavach.evaluation.cascade_judge; "
        "assert 'torch' not in sys.modules, 'torch imported by evaluation module'; "
        "assert 'transformers' not in sys.modules, 'transformers imported by evaluation module'"
    )
    proc_iso = subprocess.run(
        [sys.executable, "-c", isolated_code],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc_iso.returncode == 0, proc_iso.stderr


# --------------------------------------------------------------------------- 15. is_refusal parity on 150 rows
def test_is_refusal_matches_prompts_across_150_results() -> None:
    results_path = Path(__file__).resolve().parent.parent / "data" / "processed" / "generation_results_150.jsonl"
    rows = cj.load_items(results_path)
    assert len(rows) == 150
    for row in rows:
        ans = row["answer"]
        assert cj.is_refusal(ans) is frozen_is_refusal(ans), row["test_id"]
