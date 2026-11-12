"""Offline unit tests for scripts/calibrate_judge.py. Makes no network or API calls."""
from __future__ import annotations

import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "calibrate_judge.py"

_SPEC = importlib.util.spec_from_file_location("calibrate_judge", SCRIPT_PATH)
assert _SPEC is not None and _SPEC.loader is not None
calibrate_judge = importlib.util.module_from_spec(_SPEC)
sys.modules["calibrate_judge"] = calibrate_judge
_SPEC.loader.exec_module(calibrate_judge)

STRATUM_SIZES = {
    "FACTUAL": 60,
    "MULTI_HOP": 45,
    "CONFLICT_TRAP": 15,
    "OUT_OF_SCOPE": 10,
    "MALFORMED": 10,
    "CONTEXT_NOISE": 10,
}

EXPECTED_QUOTAS = {
    "FACTUAL": 12,
    "MULTI_HOP": 9,
    "CONFLICT_TRAP": 3,
    "OUT_OF_SCOPE": 2,
    "MALFORMED": 2,
    "CONTEXT_NOISE": 2,
}

FORBIDDEN_JUDGE_FIELDS = {
    "verdict",
    "faithfulness_score",
    "context_relevance_score",
    "panel",
    "evaluation_stage",
    "escalation_reasons",
    "negative_constraint_passed",
    "conflict_resolution_passed",
}


def _make_synthetic_dataset() -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    records: dict[str, dict[str, Any]] = {}
    results: dict[str, dict[str, Any]] = {}
    idx = 1
    for qtype, count in STRATUM_SIZES.items():
        for offset in range(count):
            test_id = f"TEST_{idx:03d}"
            idx += 1
            if offset == 0:
                verdict = "FAIL"
                faith = 0.0
            elif offset in (1, 2):
                verdict = "PARTIAL"
                faith = 0.5
            else:
                verdict = "PASS"
                faith = 1.0
            is_oos = qtype == "OUT_OF_SCOPE"
            records[test_id] = {
                "test_id": test_id,
                "question_type": qtype,
                "faithfulness_score": faith,
                "context_relevance_score": 0.9 if not is_oos else 0.1,
                "negative_constraint_passed": True,
                "conflict_resolution_passed": qtype != "CONFLICT_TRAP" or verdict == "PASS",
                "verdict": verdict,
                "evaluation_stage": "STAGE_2_JURY" if verdict != "PASS" else "STAGE_1_PRIMARY",
                "answer_is_refusal": is_oos,
                "context_sufficient": not is_oos,
                "escalation_reasons": ["verdict_partial"] if verdict == "PARTIAL" else [],
                "panel": [
                    {"model": "gemini-3.5-flash-lite", "thinking_applied": True, "verdict": verdict}
                ],
                "panel_failures": [],
            }
            results[test_id] = {
                "test_id": test_id,
                "question_type": qtype,
                "query": f"Synthetic query for {test_id}",
                "ground_truth_answer": f"Ground truth answer for {test_id}",
                "context_chunks": [
                    {"chunk_id": f"{test_id}_chunk_001", "content": f"Evidence content for {test_id}"}
                ],
                "answer": "INSUFFICIENT_CONTEXT" if is_oos else f"Candidate answer for {test_id}",
                "bypassed_llm": is_oos,
            }
    return records, results


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(json.dumps(r) for r in rows) + "\n"
    path.write_text(text, encoding="utf-8")


def _patch_paths(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> dict[str, Path]:
    paths = {
        "CHECKPOINTS": tmp_path / "data" / "processed" / "tier6_eval_checkpoints.jsonl",
        "RESULTS": tmp_path / "data" / "processed" / "generation_results_150.jsonl",
        "REVIEW_SET": tmp_path / "data" / "calibration" / "human_review_set_30.json",
        "LABELS_FILE": tmp_path / "data" / "calibration" / "human_expert_labels_30.json",
        "CALIBRATION_JSON": tmp_path / "reports" / "human_calibration_30.json",
        "AUDIT_REPORT": tmp_path / "reports" / "phase4_audit_report.md",
        "PHASE2_SUMMARY": tmp_path / "reports" / "generation_evaluation_summary.json",
    }
    for attr, p in paths.items():
        monkeypatch.setattr(calibrate_judge, attr, p)
    return paths


def test_cohen_kappa_known_contingency_matrix() -> None:
    """3x3 matrix [[19, 1, 0], [1, 4, 1], [0, 0, 4]] gives Po=0.9, Pe=0.5, kappa=0.8."""
    labels = calibrate_judge.LABELS
    matrix = [
        [19, 1, 0],
        [1, 4, 1],
        [0, 0, 4],
    ]
    human: list[str] = []
    judge: list[str] = []
    for r_idx, h_label in enumerate(labels):
        for c_idx, j_label in enumerate(labels):
            count = matrix[r_idx][c_idx]
            human.extend([h_label] * count)
            judge.extend([j_label] * count)

    stats = calibrate_judge.cohen_kappa(human, judge)
    assert stats["n"] == 30
    assert stats["observed_agreement"] == pytest.approx(0.9)
    assert stats["expected_agreement"] == pytest.approx(0.5)
    assert stats["kappa"] == pytest.approx(0.8)
    assert calibrate_judge.contingency(human, judge) == {
        "PASS": {"PASS": 19, "PARTIAL": 1, "FAIL": 0},
        "PARTIAL": {"PASS": 1, "PARTIAL": 4, "FAIL": 1},
        "FAIL": {"PASS": 0, "PARTIAL": 0, "FAIL": 4},
    }


def test_cohen_kappa_perfect_agreement_and_degenerate_pe() -> None:
    """Perfect mixed agreement gives kappa=1.0; expected_agreement>=1.0 gives None."""
    mixed = ["PASS"] * 18 + ["PARTIAL"] * 7 + ["FAIL"] * 5
    perfect = calibrate_judge.cohen_kappa(mixed, mixed)
    assert perfect["observed_agreement"] == pytest.approx(1.0)
    assert perfect["kappa"] == pytest.approx(1.0)

    all_pass = ["PASS"] * 30
    degenerate = calibrate_judge.cohen_kappa(all_pass, all_pass)
    assert degenerate["observed_agreement"] == pytest.approx(1.0)
    assert degenerate["expected_agreement"] == pytest.approx(1.0)
    assert degenerate["kappa"] is None


def test_select_review_ids_stratified_deterministic_and_non_pass() -> None:
    """Stratified selection returns 30 IDs matching 12/9/3/2/2/2, is deterministic, and includes non-PASS."""
    records, _ = _make_synthetic_dataset()
    ids_first = calibrate_judge.select_review_ids(records, seed=calibrate_judge.SEED)
    ids_second = calibrate_judge.select_review_ids(records, seed=calibrate_judge.SEED)

    assert ids_first == ids_second
    assert len(ids_first) == 30
    assert len(set(ids_first)) == 30

    counts = Counter(records[tid]["question_type"] for tid in ids_first)
    assert dict(counts) == EXPECTED_QUOTAS

    for qtype in STRATUM_SIZES:
        non_pass_in_stratum = [
            tid for tid in ids_first
            if records[tid]["question_type"] == qtype and records[tid]["verdict"] != "PASS"
        ]
        assert len(non_pass_in_stratum) >= 1


def test_ensure_review_set_strips_judge_fields_and_creates_null_template(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """human_review_set_30.json has zero judge fields; human_expert_labels_30.json has null labels."""
    paths = _patch_paths(monkeypatch, tmp_path)
    records, results = _make_synthetic_dataset()

    review = calibrate_judge.ensure_review_set(records, results)
    assert paths["REVIEW_SET"].is_file()
    assert paths["LABELS_FILE"].is_file()

    review_disk = json.loads(paths["REVIEW_SET"].read_text(encoding="utf-8"))
    assert review_disk == review
    assert review_disk["n"] == 30
    assert len(review_disk["items"]) == 30
    assert FORBIDDEN_JUDGE_FIELDS.isdisjoint(review_disk.keys())
    for item in review_disk["items"]:
        assert FORBIDDEN_JUDGE_FIELDS.isdisjoint(item.keys())

    labels_disk = json.loads(paths["LABELS_FILE"].read_text(encoding="utf-8"))
    assert labels_disk["review_set_sha256"] == review_disk["review_set_sha256"]
    assert labels_disk["status"] == "PENDING_HUMAN_REVIEW"
    assert len(labels_disk["labels"]) == 30
    assert all(v is None for v in labels_disk["labels"].values())


def test_main_pending_labels_missing_and_null_return_exit_3(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Unset (null) or missing labels write PENDING_HUMAN_LABELS and return exit code 3."""
    paths = _patch_paths(monkeypatch, tmp_path)
    records, results = _make_synthetic_dataset()
    _write_jsonl(paths["CHECKPOINTS"], list(records.values()))
    _write_jsonl(paths["RESULTS"], list(results.values()))

    # First run creates review set and null template -> exit 3
    rc1 = calibrate_judge.main([])
    assert rc1 == 3
    cal1 = json.loads(paths["CALIBRATION_JSON"].read_text(encoding="utf-8"))
    assert cal1["status"] == "PENDING_HUMAN_LABELS"
    assert cal1["cohen_kappa"] is None
    assert cal1["passed"] is False
    assert paths["AUDIT_REPORT"].is_file()
    assert "PENDING_HUMAN_LABELS" in paths["AUDIT_REPORT"].read_text(encoding="utf-8")

    # Missing labels file -> also PENDING_HUMAN_LABELS and exit 3
    paths["LABELS_FILE"].unlink()
    rc2 = calibrate_judge.main([])
    assert rc2 == 3
    cal2 = json.loads(paths["CALIBRATION_JSON"].read_text(encoding="utf-8"))
    assert cal2["status"] == "PENDING_HUMAN_LABELS"


def test_mismatched_review_set_hash_raises_and_returns_exit_2(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Mismatched review_set_sha256 raises ValueError in load_labels and makes main() exit 2."""
    paths = _patch_paths(monkeypatch, tmp_path)
    records, results = _make_synthetic_dataset()
    _write_jsonl(paths["CHECKPOINTS"], list(records.values()))
    _write_jsonl(paths["RESULTS"], list(results.values()))

    review = calibrate_judge.ensure_review_set(records, results)
    labels_payload = json.loads(paths["LABELS_FILE"].read_text(encoding="utf-8"))
    labels_payload["review_set_sha256"] = "0" * 64
    for tid in labels_payload["labels"]:
        labels_payload["labels"][tid] = "PASS"
    paths["LABELS_FILE"].write_text(json.dumps(labels_payload, indent=2), encoding="utf-8")

    with pytest.raises(ValueError, match="hash mismatch"):
        calibrate_judge.load_labels(review)

    rc = calibrate_judge.main([])
    assert rc == 2


def test_main_scored_pass_and_fail_exit_codes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Valid 30 labels compute SCORED kappa, audit report, and exit 0 (>=0.70) or 1 (<0.70)."""
    paths = _patch_paths(monkeypatch, tmp_path)
    records, results = _make_synthetic_dataset()
    _write_jsonl(paths["CHECKPOINTS"], list(records.values()))
    _write_jsonl(paths["RESULTS"], list(results.values()))

    review = calibrate_judge.ensure_review_set(records, results)
    review_ids = [item["test_id"] for item in review["items"]]

    # Exact match -> kappa == 1.0 -> exit 0
    labels_payload = json.loads(paths["LABELS_FILE"].read_text(encoding="utf-8"))
    labels_payload["annotator"] = "reviewer"
    labels_payload["labels"] = {tid: records[tid]["verdict"] for tid in review_ids}
    paths["LABELS_FILE"].write_text(json.dumps(labels_payload, indent=2), encoding="utf-8")

    rc_pass = calibrate_judge.main([])
    assert rc_pass == 0
    cal_pass = json.loads(paths["CALIBRATION_JSON"].read_text(encoding="utf-8"))
    assert cal_pass["status"] == "SCORED"
    assert cal_pass["cohen_kappa"] == pytest.approx(1.0)
    assert cal_pass["passed"] is True
    assert len(cal_pass["pairs"]) == 30
    audit_md = paths["AUDIT_REPORT"].read_text(encoding="utf-8")
    assert "Cohen's kappa" in audit_md
    assert "**PASS**" in audit_md

    # Invert labels -> low kappa < 0.70 -> exit 1
    invert = {"PASS": "FAIL", "PARTIAL": "PASS", "FAIL": "PARTIAL"}
    labels_payload["labels"] = {tid: invert[records[tid]["verdict"]] for tid in review_ids}
    paths["LABELS_FILE"].write_text(json.dumps(labels_payload, indent=2), encoding="utf-8")

    rc_fail = calibrate_judge.main([])
    assert rc_fail == 1
    cal_fail = json.loads(paths["CALIBRATION_JSON"].read_text(encoding="utf-8"))
    assert cal_fail["status"] == "SCORED"
    assert cal_fail["cohen_kappa"] < calibrate_judge.THRESHOLD
    assert cal_fail["passed"] is False
    assert "**FAIL**" in paths["AUDIT_REPORT"].read_text(encoding="utf-8")


def test_bootstrap_ci_deterministic() -> None:
    """bootstrap_ci produces identical bounds for a fixed seed."""
    human = ["PASS"] * 20 + ["PARTIAL"] * 6 + ["FAIL"] * 4
    judge = ["PASS"] * 19 + ["PARTIAL"] * 5 + ["FAIL"] * 5 + ["PARTIAL"]
    ci1 = calibrate_judge.bootstrap_ci(human, judge, iterations=200, seed=12345)
    ci2 = calibrate_judge.bootstrap_ci(human, judge, iterations=200, seed=12345)
    assert ci1 is not None
    assert ci1 == ci2
    assert len(ci1) == 2 and ci1[0] <= ci1[1]


def test_atomic_write_json_leaves_no_tmp_files(tmp_path: Path) -> None:
    """atomic_write_json writes valid JSON and cleans up temporary files."""
    target = tmp_path / "nested" / "payload.json"
    calibrate_judge.atomic_write_json(target, {"ok": True, "count": 30})
    assert target.is_file()
    assert json.loads(target.read_text(encoding="utf-8")) == {"ok": True, "count": 30}
    leftovers = [p.name for p in target.parent.iterdir() if p.name.endswith(".tmp")]
    assert leftovers == []
