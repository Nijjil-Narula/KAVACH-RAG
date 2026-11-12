"""Offline CI regression gate over the Tier 6 evaluation artifacts. Makes no network calls."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINTS = ROOT / "data" / "processed" / "tier6_eval_checkpoints.jsonl"
GENERATION = ROOT / "data" / "processed" / "generation_results_150.jsonl"
CALIBRATION = ROOT / "reports" / "human_calibration_30.json"

EXPECTED_ITEMS = 150
MIN_AVG_FAITHFULNESS = 0.85
MAX_FAIL_VERDICTS = 12
MIN_OOS_REFUSAL_RATE = 1.00
MIN_KAPPA = 0.70
LABELS = ("PASS", "PARTIAL", "FAIL")
REQUIRED_FIELDS = (
    "test_id", "question_type", "faithfulness_score", "context_relevance_score",
    "negative_constraint_passed", "conflict_resolution_passed", "verdict", "evaluation_stage",
)


def _jsonl(path: Path) -> list[dict]:
    assert path.is_file(), f"required artifact is missing: {path.relative_to(ROOT)}"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _kappa(human: list[str], judge: list[str]) -> float | None:
    n = len(human)
    observed = sum(h == j for h, j in zip(human, judge)) / n
    h_counts, j_counts = Counter(human), Counter(judge)
    expected = sum((h_counts[x] / n) * (j_counts[x] / n) for x in LABELS)
    return None if expected >= 1.0 else (observed - expected) / (1.0 - expected)


@pytest.fixture(scope="session")
def records() -> list[dict]:
    last: dict[str, dict] = {}
    for row in _jsonl(CHECKPOINTS):
        last[row["test_id"]] = row
    return list(last.values())


def test_checkpoint_covers_exactly_the_150_generated_items(records):
    assert len(records) == EXPECTED_ITEMS, f"expected {EXPECTED_ITEMS} evaluated items, found {len(records)}"
    generated = {row["test_id"] for row in _jsonl(GENERATION)}
    assert {r["test_id"] for r in records} == generated, "checkpoint ids differ from generation_results_150.jsonl"


def test_record_schema_integrity(records):
    for rec in records:
        for field in REQUIRED_FIELDS:
            assert field in rec, f"{rec.get('test_id')} lacks {field}"
        assert rec["verdict"] in LABELS
        assert 0.0 <= rec["faithfulness_score"] <= 1.0
        assert 0.0 <= rec["context_relevance_score"] <= 1.0


def test_average_faithfulness_meets_floor(records):
    average = sum(r["faithfulness_score"] for r in records) / len(records)
    answered = [r["faithfulness_score"] for r in records if not r.get("answer_is_refusal")]
    answered_avg = sum(answered) / len(answered) if answered else float("nan")
    print(f"\n[CI METRIC] faithfulness all={average:.4f} answered_only={answered_avg:.4f} (n_answered={len(answered)})")
    assert average >= MIN_AVG_FAITHFULNESS, f"average faithfulness {average:.4f} < {MIN_AVG_FAITHFULNESS}"


def test_out_of_scope_refusals_are_absolute(records):
    oos = [r for r in records if r["question_type"] == "OUT_OF_SCOPE"]
    assert oos, "no out-of-scope records found"
    rate = sum(1 for r in oos if r["negative_constraint_passed"] is True) / len(oos)
    print(f"\n[CI METRIC] out-of-scope refusal rate={rate:.2%} (n={len(oos)})")
    assert rate >= MIN_OOS_REFUSAL_RATE, f"refusal rate {rate:.2%} below 100%"


def test_fail_verdict_budget(records):
    fails = sum(1 for r in records if r["verdict"] == "FAIL")
    print(f"\n[CI METRIC] FAIL verdicts={fails}/{len(records)}")
    assert fails <= MAX_FAIL_VERDICTS, f"{fails} FAIL verdicts exceed the budget of {MAX_FAIL_VERDICTS}"


def test_context_relevance_is_reported_not_gated(records):
    """Retrieval-bound metric: documented honestly, never tuned. Overall relevance is expected below 0.80."""
    in_scope = [r for r in records if r["question_type"] != "OUT_OF_SCOPE"]
    sufficient = [r for r in in_scope if r.get("context_sufficient") is True]
    assert sufficient, "no context-sufficient items recorded; golden required_entities were not applied"
    overall = sum(r["context_relevance_score"] for r in in_scope) / len(in_scope)
    subset = sum(r["context_relevance_score"] for r in sufficient) / len(sufficient)
    print(f"\n[CI METRIC] relevance in_scope={overall:.4f} context_sufficient={subset:.4f} "
          f"(n={len(sufficient)}; target on the subset >= 0.90, overall is retrieval-limited)")
    assert 0.0 <= overall <= 1.0 and 0.0 <= subset <= 1.0


def test_human_judge_kappa_meets_floor():
    assert CALIBRATION.is_file(), "reports/human_calibration_30.json is missing"
    data = json.loads(CALIBRATION.read_text(encoding="utf-8"))
    assert data.get("status") == "SCORED", "human calibration is not complete (status: %s)" % data.get("status")
    pairs = data["pairs"]
    assert len(pairs) == 30, f"expected 30 calibration pairs, found {len(pairs)}"
    kappa = _kappa([p["human"] for p in pairs], [p["judge"] for p in pairs])
    assert kappa is not None, "kappa is undefined for these labels"
    print(f"\n[CI METRIC] Cohen's kappa (recomputed)={kappa:.4f}")
    assert abs(kappa - data["cohen_kappa"]) < 1e-5, "stored kappa does not match the stored label pairs"
    assert kappa >= MIN_KAPPA, f"kappa {kappa:.4f} < {MIN_KAPPA}"
