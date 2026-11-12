"""Human-to-judge calibration: blind review set, Cohen's kappa, calibration JSON and Phase 4 audit report.

Exit codes: 0 scored and kappa >= threshold, 1 scored but below threshold, 2 input error, 3 labels pending.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINTS = ROOT / "data" / "processed" / "tier6_eval_checkpoints.jsonl"
RESULTS = ROOT / "data" / "processed" / "generation_results_150.jsonl"
REVIEW_SET = ROOT / "data" / "calibration" / "human_review_set_30.json"
LABELS_FILE = ROOT / "data" / "calibration" / "human_expert_labels_30.json"
CALIBRATION_JSON = ROOT / "reports" / "human_calibration_30.json"
AUDIT_REPORT = ROOT / "reports" / "phase4_audit_report.md"
PHASE2_SUMMARY = ROOT / "reports" / "generation_evaluation_summary.json"

LABELS = ("PASS", "PARTIAL", "FAIL")
SEED = 20261112
THRESHOLD = 0.70
BOOTSTRAP_ITERATIONS = 2000
STRATA = (
    ("FACTUAL", 12), ("MULTI_HOP", 9), ("CONFLICT_TRAP", 3),
    ("OUT_OF_SCOPE", 2), ("MALFORMED", 2), ("CONTEXT_NOISE", 2),
)
HUMAN_RUBRIC = (
    "Judge ONLY against the retrieved context. PASS: every claim in the candidate answer is stated in the context, "
    "numbers and identifiers match exactly, a correct INSUFFICIENT_CONTEXT refusal counts as PASS, and conflicting "
    "standard versions (for example OWASP v1.1 vs v2.0) are kept separate. FAIL: invented or altered facts, or an "
    "answer where a refusal was required. PARTIAL: mostly grounded but with an omission, a blended version conflict, "
    "or an unnecessary refusal when the context did contain the answer."
)


# ------------------------------------------------------------------ io helpers
def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_last_per_id(path: Path) -> dict[str, dict[str, Any]]:
    return {row["test_id"]: row for row in read_jsonl(path)}


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_path, path)
    finally:
        tmp_path.unlink(missing_ok=True)


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    atomic_write_text(path, json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


# ------------------------------------------------------------------ review set
def select_review_ids(records: dict[str, dict[str, Any]], seed: int = SEED) -> list[str]:
    """Taxonomy-stratified, deterministic. Within each stratum about one third of the quota (at least one)
    comes from items the judge did not mark PASS, so kappa is not inflated by the PASS base rate."""
    rng = random.Random(seed)
    chosen: list[str] = []
    for qtype, quota in STRATA:
        pool = sorted(tid for tid, rec in records.items() if rec["question_type"] == qtype)
        if len(pool) < quota:
            raise ValueError(f"stratum {qtype} has {len(pool)} items, need {quota}")
        hard = [t for t in pool if records[t]["verdict"] != "PASS"]
        take_hard = min(len(hard), max(1, quota // 3))
        picked = rng.sample(hard, take_hard) if take_hard else []
        rest = [t for t in pool if t not in picked]
        picked += rng.sample(rest, quota - len(picked))
        chosen.extend(sorted(picked))
    return chosen


def build_review_items(ids: Sequence[str], results: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for test_id in ids:
        row = results[test_id]
        items.append({
            "test_id": test_id,
            "question_type": row["question_type"],
            "query": row["query"],
            "ground_truth_answer": row["ground_truth_answer"],
            "retrieved_context": [{"chunk_id": c.get("chunk_id"), "content": c.get("content")} for c in row["context_chunks"]],
            "candidate_answer": row["answer"],
            "bypassed_llm": bool(row["bypassed_llm"]),
        })
    return items


def items_sha256(items: list[dict[str, Any]]) -> str:
    return sha256_text(json.dumps(items, sort_keys=True, ensure_ascii=False))


def ensure_review_set(records: dict[str, dict[str, Any]], results: dict[str, dict[str, Any]],
                      *, regenerate: bool = False) -> dict[str, Any]:
    if REVIEW_SET.exists() and not regenerate:
        return json.loads(REVIEW_SET.read_text(encoding="utf-8"))
    ids = select_review_ids(records)
    items = build_review_items(ids, results)
    review = {
        "n": len(items), "seed": SEED, "strata": {q: n for q, n in STRATA},
        "selection_note": "Taxonomy-stratified; one third of each stratum quota (minimum one) is drawn from items "
                          "the automated judge did not mark PASS. Judge output is deliberately absent from this file.",
        "rubric": HUMAN_RUBRIC,
        "review_set_sha256": items_sha256(items),
        "items": items,
    }
    atomic_write_json(REVIEW_SET, review)
    template = {
        "review_set_sha256": review["review_set_sha256"],
        "annotator": None,
        "status": "PENDING_HUMAN_REVIEW",
        "instructions": "Set every value in labels to PASS, PARTIAL or FAIL after reading human_review_set_30.json, then re-run scripts/calibrate_judge.py.",
        "labels": {item["test_id"]: None for item in items},
    }
    atomic_write_json(LABELS_FILE, template)
    return review


def load_labels(review: dict[str, Any]) -> dict[str, str] | None:
    """Return labels, or None while any label is still unset. Raises ValueError on inconsistent input."""
    if not LABELS_FILE.exists():
        return None
    data = json.loads(LABELS_FILE.read_text(encoding="utf-8"))
    labels = data.get("labels", {})
    expected = [item["test_id"] for item in review["items"]]
    if sorted(labels) != sorted(expected):
        raise ValueError("labels file test_ids do not match the review set")
    if data.get("review_set_sha256") not in (None, review["review_set_sha256"]):
        raise ValueError("labels file was written for a different review set (hash mismatch)")
    if any(value is None for value in labels.values()):
        return None
    bad = {tid: v for tid, v in labels.items() if v not in LABELS}
    if bad:
        raise ValueError(f"invalid labels (must be one of {LABELS}): {bad}")
    return dict(labels)


# ------------------------------------------------------------------ statistics
def cohen_kappa(human: Sequence[str], judge: Sequence[str]) -> dict[str, Any]:
    n = len(human)
    if n == 0 or n != len(judge):
        raise ValueError("kappa needs two equally long, non-empty label lists")
    observed = sum(h == j for h, j in zip(human, judge)) / n
    h_counts, j_counts = Counter(human), Counter(judge)
    expected = sum((h_counts[label] / n) * (j_counts[label] / n) for label in LABELS)
    kappa = None if expected >= 1.0 else (observed - expected) / (1.0 - expected)
    return {"n": n, "observed_agreement": observed, "expected_agreement": expected, "kappa": kappa}


def bootstrap_ci(human: Sequence[str], judge: Sequence[str], iterations: int = BOOTSTRAP_ITERATIONS,
                 seed: int = SEED) -> list[float] | None:
    rng = random.Random(seed)
    n = len(human)
    values: list[float] = []
    for _ in range(iterations):
        idx = [rng.randrange(n) for _ in range(n)]
        kappa = cohen_kappa([human[i] for i in idx], [judge[i] for i in idx])["kappa"]
        if kappa is not None:
            values.append(kappa)
    if len(values) < iterations // 2:
        return None
    values.sort()
    return [round(values[int(0.025 * len(values))], 4), round(values[min(len(values) - 1, int(0.975 * len(values)))], 4)]


def contingency(human: Sequence[str], judge: Sequence[str]) -> dict[str, dict[str, int]]:
    table = {h: {j: 0 for j in LABELS} for h in LABELS}
    for h, j in zip(human, judge):
        table[h][j] += 1
    return table


def mean(values: Sequence[float | None]) -> float | None:
    vals = [v for v in values if v is not None]
    return sum(vals) / len(vals) if vals else None


def fmt(value: float | None, digits: int = 4) -> str:
    return "N/A" if value is None else f"{value:.{digits}f}"


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    in_scope = [r for r in records if r["question_type"] != "OUT_OF_SCOPE"]
    answered = [r for r in records if not r.get("answer_is_refusal")]
    sufficient = [r for r in in_scope if r.get("context_sufficient") is True]
    oos = [r for r in records if r["question_type"] == "OUT_OF_SCOPE"]
    conflicts = [r for r in records if r["question_type"] == "CONFLICT_TRAP"]
    types = sorted({r["question_type"] for r in records})
    return {
        "n": len(records),
        "verdicts": Counter(r["verdict"] for r in records),
        "by_type": {t: Counter(r["verdict"] for r in records if r["question_type"] == t) for t in types},
        "faith_by_type": {t: mean([r["faithfulness_score"] for r in records if r["question_type"] == t]) for t in types},
        "stages": Counter(r["evaluation_stage"] for r in records),
        "faith_all": mean([r["faithfulness_score"] for r in records]),
        "faith_answered": mean([r["faithfulness_score"] for r in answered]),
        "n_answered": len(answered),
        "rel_in_scope": mean([r["context_relevance_score"] for r in in_scope]),
        "rel_context_sufficient": mean([r["context_relevance_score"] for r in sufficient]),
        "n_context_sufficient": len(sufficient),
        "oos_total": len(oos),
        "oos_refused": sum(1 for r in oos if r["negative_constraint_passed"] is True),
        "conflict_total": len(conflicts),
        "conflict_passed": sum(1 for r in conflicts if r["conflict_resolution_passed"] is True),
        "fail_ids": sorted(r["test_id"] for r in records if r["verdict"] == "FAIL"),
        "degraded_panels": sum(1 for r in records if r.get("panel_failures")),
        "judges": sorted({(p["model"], p["thinking_applied"]) for r in records for p in r.get("panel", [])}, key=str),
    }


# ------------------------------------------------------------------ reports
def score_calibration(review: dict[str, Any], labels: dict[str, str], records: dict[str, dict[str, Any]]) -> dict[str, Any]:
    ids = sorted(labels)
    human = [labels[i] for i in ids]
    judge = [records[i]["verdict"] for i in ids]
    stats = cohen_kappa(human, judge)
    kappa = stats["kappa"]
    return {
        "status": "SCORED", "n": stats["n"],
        "observed_agreement": round(stats["observed_agreement"], 6),
        "expected_agreement": round(stats["expected_agreement"], 6),
        "cohen_kappa": None if kappa is None else round(kappa, 6),
        "kappa_ci95_bootstrap": bootstrap_ci(human, judge),
        "threshold": THRESHOLD,
        "passed": bool(kappa is not None and kappa >= THRESHOLD),
        "contingency_human_rows_judge_cols": contingency(human, judge),
        "pairs": [{"test_id": i, "human": h, "judge": j} for i, h, j in zip(ids, human, judge)],
    }


def build_audit_report(records: list[dict[str, Any]], calibration: dict[str, Any], phase2: dict[str, Any] | None) -> str:
    s = summarize(records)
    verdicts = s["verdicts"]
    lines = [
        "# KAVACH-RAG Phase 4 Audit Report: Two-Stage Cross-Family Judge Cascade", "",
        f"- Generated at (UTC): {datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}",
        f"- Items graded: {s['n']} (checkpoint file `data/processed/tier6_eval_checkpoints.jsonl`)",
        "- Judges: " + ", ".join(f"`{m}` (thinking_applied={t})" for m, t in s["judges"]) + ". No Qwen model judges Qwen output.",
        f"- Stage mix: {dict(s['stages'])}; jury panels with a failed member: {s['degraded_panels']}", "",
        "## 1. Verdict distribution", "",
        "| Question type | PASS | PARTIAL | FAIL | Faithfulness mean |", "|---|---:|---:|---:|---:|",
    ]
    for qtype, counts in s["by_type"].items():
        lines.append(f"| {qtype} | {counts['PASS']} | {counts['PARTIAL']} | {counts['FAIL']} | {fmt(s['faith_by_type'][qtype])} |")
    lines += [
        f"| **Total** | **{verdicts['PASS']}** | **{verdicts['PARTIAL']}** | **{verdicts['FAIL']}** | **{fmt(s['faith_all'])}** |", "",
        "## 2. CI gate metrics", "",
        "| Metric | Observed | Threshold | Gated |", "|---|---:|---:|---|",
        f"| Average faithfulness, all {s['n']} items | {fmt(s['faith_all'])} | >= 0.85 | yes |",
        f"| Out-of-scope refusal rate | {s['oos_refused']}/{s['oos_total']} | 100% | yes |",
        f"| FAIL verdicts | {verdicts['FAIL']} | <= 12 | yes |",
        f"| Faithfulness, answered items only (n={s['n_answered']}) | {fmt(s['faith_answered'])} | reported | no |",
        f"| Context relevance, in-scope items | {fmt(s['rel_in_scope'])} | reported (retrieval-bound) | no |",
        f"| Context relevance, context-sufficient subset (n={s['n_context_sufficient']}) | {fmt(s['rel_context_sufficient'])} | target >= 0.90 | no |",
        f"| Conflict-trap items with versions kept separate | {s['conflict_passed']}/{s['conflict_total']} | reported | no |", "",
        "Faithfulness over all items counts a correct refusal as faithful. Answered-only faithfulness is the stricter view of hallucination risk. "
        "Overall relevance is dominated by retrieval misses on multi-hop and context-noise items, not by generation quality.", "",
        "## 3. Failures", "", f"FAIL test_ids: {', '.join(s['fail_ids']) or 'none'}", "",
    ]
    if phase2:
        quality = phase2.get("quality", {})
        lines += [
            "## 4. Reference: Phase 2 single-stage judge", "",
            f"- Faithfulness mean {fmt(quality.get('faithfulness_mean'))}, relevance mean {fmt(quality.get('relevance_mean'))}, "
            f"relevance on context-sufficient items {fmt(quality.get('relevance_mean_context_sufficient'))}. "
            "The Phase 2 judge scores claim-level support on served items only; the Phase 4 cascade scores all items on a PASS/PARTIAL/FAIL scale, so the numbers are related but not identical.", "",
        ]
    lines += ["## 5. Human calibration", ""]
    if calibration["status"] == "SCORED":
        ci = calibration["kappa_ci95_bootstrap"]
        lines += [
            f"- Items: {calibration['n']} | observed agreement {fmt(calibration['observed_agreement'])} | chance agreement {fmt(calibration['expected_agreement'])}",
            f"- Cohen's kappa: **{fmt(calibration['cohen_kappa'])}** (bootstrap 95% CI {ci if ci else 'N/A'}) against the {THRESHOLD:.2f} floor: **{'PASS' if calibration['passed'] else 'FAIL'}**", "",
            "| Human \\ Judge | PASS | PARTIAL | FAIL |", "|---|---:|---:|---:|",
        ]
        for human_label, row in calibration["contingency_human_rows_judge_cols"].items():
            lines.append(f"| {human_label} | {row['PASS']} | {row['PARTIAL']} | {row['FAIL']} |")
        lines += ["", "Limits: n=30 gives a wide confidence interval; the review set over-samples non-PASS judge verdicts to avoid a PASS-dominated chance baseline."]
    else:
        lines += [
            "- Status: **PENDING_HUMAN_LABELS**. No kappa has been computed and none is claimed.",
            "- Action: fill every label in `data/calibration/human_expert_labels_30.json` (PASS, PARTIAL or FAIL) after reading `data/calibration/human_review_set_30.json`, then re-run `scripts/calibrate_judge.py`.",
        ]
    return "\n".join(lines) + "\n"


# ------------------------------------------------------------------ CLI
def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Calibrate the Tier 6 judge against 30 human labels.")
    parser.add_argument("--regenerate-review-set", action="store_true",
                        help="Replace the blind review set. Existing human labels become invalid.")
    args = parser.parse_args(list(argv) if argv is not None else None)
    for required in (CHECKPOINTS, RESULTS):
        if not required.is_file():
            print(f"[ERROR] missing input: {required}", file=sys.stderr)
            return 2
    records = load_last_per_id(CHECKPOINTS)
    results = load_last_per_id(RESULTS)
    if len(records) != 150 or set(records) != set(results):
        print(f"[ERROR] checkpoint holds {len(records)} items; run the full cascade before calibrating.", file=sys.stderr)
        return 2
    review = ensure_review_set(records, results, regenerate=args.regenerate_review_set)
    try:
        labels = load_labels(review)
    except ValueError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 2
    if labels is None:
        calibration: dict[str, Any] = {
            "status": "PENDING_HUMAN_LABELS", "n": review["n"], "cohen_kappa": None,
            "threshold": THRESHOLD, "passed": False, "pairs": [],
        }
    else:
        calibration = score_calibration(review, labels, records)
    calibration.update({
        "seed": SEED,
        "review_set_sha256": review["review_set_sha256"],
        "judge_checkpoint_sha256": sha256_file(CHECKPOINTS),
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    })
    atomic_write_json(CALIBRATION_JSON, calibration)
    phase2 = json.loads(PHASE2_SUMMARY.read_text(encoding="utf-8")) if PHASE2_SUMMARY.is_file() else None
    atomic_write_text(AUDIT_REPORT, build_audit_report(list(records.values()), calibration, phase2))
    print(f"[CALIBRATION] status={calibration['status']} kappa={calibration.get('cohen_kappa')} passed={calibration['passed']}")
    if calibration["status"] != "SCORED":
        return 3
    return 0 if calibration["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
