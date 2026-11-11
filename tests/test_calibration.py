"""Phase 1.5 calibration devsets, threshold selection, and stale-guard tests."""

from __future__ import annotations

import json
import math
import shutil
from pathlib import Path

import pytest

from kavach.config import Settings, get_settings
from kavach.exceptions import ConfigError, GoldenDatasetValidationError, HardwareIsolationError
from kavach.retrieval.ablation import _assert_runtime_preconditions
from kavach.retrieval.calibration import (
    compute_roc_auc,
    generate_report,
    select_threshold,
    verify_abstain_calibration,
    verify_precision_lock,
)
from kavach.retrieval.devsets import (
    INSCOPE_CATEGORY_TARGETS,
    OOS_CATEGORY_TARGETS,
    DevItem,
    validate_devsets,
)
from kavach.retrieval.pipeline import VectorRetrievalPipeline


def test_devsets_exist_and_validate() -> None:
    """Generated oos_dev_60 and inscope_dev_100 pass all strict disjointness checks."""
    cfg = get_settings()
    summary = validate_devsets(
        oos_path=cfg.data_golden / "oos_dev_60.jsonl",
        inscope_path=cfg.data_golden / "inscope_dev_100.jsonl",
        corpus_parquet_path=cfg.data_processed / "corpus_chunks.parquet",
        golden_path=cfg.data_golden / "golden_150.jsonl",
        training_queries_path=cfg.models_dir / "router_deberta" / "training_queries.jsonl",
    )
    assert summary["oos_count"] == 60
    assert summary["inscope_count"] == 100
    assert summary["oos_categories"] == OOS_CATEGORY_TARGETS
    assert summary["inscope_categories"] == INSCOPE_CATEGORY_TARGETS
    assert float(summary["max_oos_golden_ws_jaccard"]) < 0.6
    assert float(summary["max_oos_golden_alnum_jaccard"]) < 0.6


def test_devsets_strict_pydantic_schema() -> None:
    """Every row in both devsets conforms to the 5-field DevItem contract."""
    cfg = get_settings()
    oos_lines = (cfg.data_golden / "oos_dev_60.jsonl").read_text(encoding="utf-8").splitlines()
    in_lines = (cfg.data_golden / "inscope_dev_100.jsonl").read_text(encoding="utf-8").splitlines()

    assert len(oos_lines) == 60
    assert len(in_lines) == 100

    for raw in oos_lines:
        item = DevItem.model_validate(json.loads(raw))
        assert item.split == "oos_dev"
        assert item.expected_doc_ids == []

    for raw in in_lines:
        item = DevItem.model_validate(json.loads(raw))
        assert item.split == "inscope_dev"
        assert len(item.expected_doc_ids) >= 1


def test_validate_devsets_rejects_overlap_and_indexed_entities(tmp_path: Path) -> None:
    """Negative validation tests verify hard failures on duplicates, Jaccard>=0.6, and corpus leaks."""
    cfg = get_settings()
    oos_src = cfg.data_golden / "oos_dev_60.jsonl"
    in_src = cfg.data_golden / "inscope_dev_100.jsonl"
    golden_src = cfg.data_golden / "golden_150.jsonl"
    train_src = cfg.models_dir / "router_deberta" / "training_queries.jsonl"
    corpus_src = cfg.data_processed / "corpus_chunks.parquet"

    oos_tmp = tmp_path / "oos.jsonl"
    in_tmp = tmp_path / "in.jsonl"

    # 1. Inject training_queries normalized substring leak into inscope_dev
    first_train = json.loads(train_src.read_text(encoding="utf-8").splitlines()[0])["query"]
    in_rows = [json.loads(line) for line in in_src.read_text(encoding="utf-8").splitlines()]
    in_rows[0]["query"] = f"Analyst question: {first_train}"
    shutil.copy2(oos_src, oos_tmp)
    in_tmp.write_text("\n".join(json.dumps(r) for r in in_rows) + "\n", encoding="utf-8")

    with pytest.raises(GoldenDatasetValidationError, match="substring overlap"):
        validate_devsets(oos_tmp, in_tmp, corpus_src, golden_src, train_src)

    # 2. Inject indexed CVE-2021-44228 into unindexed_public_security
    shutil.copy2(in_src, in_tmp)
    oos_rows = [json.loads(line) for line in oos_src.read_text(encoding="utf-8").splitlines()]
    for r in oos_rows:
        if r["category"] == "unindexed_public_security":
            r["query"] = "Explain details for indexed vulnerability CVE-2021-44228."
            break
    oos_tmp.write_text("\n".join(json.dumps(r) for r in oos_rows) + "\n", encoding="utf-8")
    with pytest.raises(GoldenDatasetValidationError, match="matched corpus_chunks.parquet"):
        validate_devsets(oos_tmp, in_tmp, corpus_src, golden_src, train_src)

    # 3. Inject high-Jaccard (>=0.6) near-duplicate of a golden_150 query into oos_dev
    golden_queries = [
        json.loads(line)["query"]
        for line in golden_src.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    long_golden_q = next(q for q in golden_queries if len(q.split()) >= 7)
    oos_rows = [json.loads(line) for line in oos_src.read_text(encoding="utf-8").splitlines()]
    oos_rows[0]["query"] = f"{long_golden_q} extra"
    oos_tmp.write_text("\n".join(json.dumps(r) for r in oos_rows) + "\n", encoding="utf-8")
    with pytest.raises(GoldenDatasetValidationError, match="Jaccard"):
        validate_devsets(oos_tmp, in_tmp, corpus_src, golden_src, train_src)


def _pipeline_guard_call(cfg: Settings) -> float | None:
    """Invoke VectorRetrievalPipeline._validate_abstention_contract without heavy model loads."""
    stub = object.__new__(VectorRetrievalPipeline)
    stub._settings = cfg
    return stub._validate_abstention_contract()


def test_stale_calibration_guard_isolated(tmp_path: Path) -> None:
    """Missing or mismatched abstain_calibration.json raises ConfigError in both guard & pipeline."""
    models_dir = tmp_path / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    cfg = Settings(
        models_dir=models_dir,
        abstention_enabled=True,
        rerank_precision="fp32",
        rerank_max_length=576,
        contingency_l_applied=False,
    )

    # Missing file -> ConfigError
    with pytest.raises(ConfigError, match="missing"):
        verify_abstain_calibration(cfg)
    with pytest.raises(ConfigError, match="abstain_calibration.json"):
        _pipeline_guard_call(cfg)

    cal_path = models_dir / "abstain_calibration.json"
    base_cal = {
        "threshold_T": 1.25,
        "reranker_precision": "fp32",
        "rerank_max_length": 576,
        "contingency_l_applied": False,
        "rerank_long_query_max_length": None,
        "rerank_long_query_token_threshold": 96,
    }

    # Valid calibration -> passes cleanly
    cal_path.write_text(json.dumps(base_cal), encoding="utf-8")
    parsed = verify_abstain_calibration(cfg)
    assert parsed is not None and parsed["threshold_T"] == 1.25
    assert _pipeline_guard_call(cfg) == 1.25

    # Stale reranker_precision -> ConfigError
    bad_prec = dict(base_cal, reranker_precision="int8")
    cal_path.write_text(json.dumps(bad_prec), encoding="utf-8")
    with pytest.raises(ConfigError, match="reranker_precision"):
        verify_abstain_calibration(cfg)
    with pytest.raises(ConfigError, match="reranker_precision"):
        _pipeline_guard_call(cfg)

    # Stale rerank_max_length -> ConfigError
    bad_len = dict(base_cal, rerank_max_length=512)
    cal_path.write_text(json.dumps(bad_len), encoding="utf-8")
    with pytest.raises(ConfigError, match="rerank_max_length"):
        verify_abstain_calibration(cfg)
    with pytest.raises(ConfigError, match="rerank_max_length"):
        _pipeline_guard_call(cfg)

    # Stale contingency_l_applied -> ConfigError
    bad_cont = dict(base_cal, contingency_l_applied=True)
    cal_path.write_text(json.dumps(bad_cont), encoding="utf-8")
    with pytest.raises(ConfigError, match="contingency_l_applied"):
        verify_abstain_calibration(cfg)
    with pytest.raises(ConfigError, match="contingency_l_applied"):
        _pipeline_guard_call(cfg)


def test_verify_precision_lock_checks_mismatch(tmp_path: Path) -> None:
    """verify_precision_lock rejects missing files and precision mismatches cleanly."""
    models_dir = tmp_path / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    cfg = Settings(models_dir=models_dir, rerank_precision="fp32", embed_precision="fp32")

    with pytest.raises(ConfigError, match="Missing precision lock"):
        verify_precision_lock(cfg)

    lock_path = models_dir / "precision_lock.json"
    lock_path.write_text(
        json.dumps(
            {
                "embed_precision": "fp32",
                "rerank_precision": "int8",
                "router_precision": "fp32",
                "rerank_max_length": 576,
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="diverge"):
        verify_precision_lock(cfg)


def test_select_threshold_formula_and_excludes_golden_150(tmp_path: Path) -> None:
    """select_threshold enforces floor_to_0.25(m - 0.5), ignores golden_150, and emits exact schema."""
    cfg = Settings(models_dir=tmp_path)
    dev_rows = [
        {"split": "inscope_dev", "query_id": "IN_1", "router_route": "SEMANTIC_RAG", "top1_logit": 2.1},
        {"split": "inscope_dev", "query_id": "IN_2", "router_route": "SEMANTIC_RAG", "top1_logit": 3.0},
        {"split": "inscope_dev", "query_id": "IN_3", "router_route": "SEMANTIC_RAG", "top1_logit": 4.0},
        {"split": "inscope_dev", "query_id": "IN_4", "router_route": "SEMANTIC_RAG", "top1_logit": 5.0},
        {"split": "oos_dev", "query_id": "OOS_1", "router_route": "HARD_REFUSAL", "top1_logit": -2.5},
        {"split": "oos_dev", "query_id": "OOS_2", "router_route": "SEMANTIC_RAG", "top1_logit": -1.0},
        {"split": "oos_dev", "query_id": "OOS_3", "router_route": "SEMANTIC_RAG", "top1_logit": 0.0},
        {"split": "oos_dev", "query_id": "OOS_4", "router_route": "SEMANTIC_RAG", "top1_logit": 2.2},
    ]
    poisoned_golden = [
        {"split": "golden_150", "query_id": "G_1", "router_route": "SEMANTIC_RAG", "top1_logit": -99.0},
        {"split": "golden_150", "query_id": "G_2", "router_route": "HARD_REFUSAL", "top1_logit": 99.0},
        {"split": "golden_150", "query_id": "G_3", "router_route": "SEMANTIC_RAG", "top1_logit": 0.1},
    ]

    res_clean = select_threshold(dev_rows, cfg=cfg)
    res_with_golden = select_threshold(dev_rows + poisoned_golden, cfg=cfg)

    expected_t = math.floor((2.1 - 0.5) * 4.0) / 4.0  # 1.5
    assert res_clean["threshold_T"] == res_with_golden["threshold_T"] == expected_t == 1.5
    assert res_clean["selection_rule"] == "floor_to_0.25(min_in_scope_dev_logit - 0.5)"
    assert res_clean["reranker_model_id"] == cfg.rerank_model_id
    assert res_clean["reranker_precision"] == cfg.rerank_precision
    assert res_clean["rerank_max_length"] == cfg.rerank_max_length
    assert res_clean["dev_in_scope_n"] == 4
    assert res_clean["dev_oos_n"] == 4
    assert res_clean["dev_false_abstain_rate"] == 0.0
    assert res_clean["dev_oos_abstain_rate"] == 0.75
    assert "created_at" in res_clean
    assert res_with_golden["golden_excluded_count"] == 3

    # Verify negative min logit quantization: floor((-0.1 - 0.5) * 4) / 4 = -0.75
    neg_rows = [
        {"split": "inscope_dev", "query_id": "IN_1", "router_route": "SEMANTIC_RAG", "top1_logit": -0.1},
        {"split": "inscope_dev", "query_id": "IN_2", "router_route": "SEMANTIC_RAG", "top1_logit": 5.0},
        {"split": "oos_dev", "query_id": "OOS_1", "router_route": "SEMANTIC_RAG", "top1_logit": 0.5},
        {"split": "oos_dev", "query_id": "OOS_2", "router_route": "SEMANTIC_RAG", "top1_logit": -2.0},
    ]
    res_neg = select_threshold(neg_rows, cfg=cfg)
    assert res_neg["threshold_T"] == -0.75
    assert res_neg["dev_false_abstain_rate"] == 0.0
    assert res_neg["dev_oos_abstain_rate"] == 0.5

    # Verify Mann-Whitney U AUC tie behavior
    assert compute_roc_auc([1.0, 2.0], [1.0, 2.0]) == 0.5


def test_generate_report_oos_table_and_low_oos_warning(tmp_path: Path) -> None:
    """generate_report handles uppercase OUT_OF_SCOPE (10/140), per-item OOS table, and <50% flag."""
    processed_dir = tmp_path / "processed"
    cal_dir = processed_dir / "calibration"
    models_dir = tmp_path / "models"
    reports_dir = tmp_path / "reports"
    cal_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    cfg = Settings(
        data_processed=processed_dir,
        models_dir=models_dir,
        reports_dir=reports_dir,
    )

    records: list[dict[str, object]] = []
    # In-scope dev with low outlier (-0.2) pulling T to -0.75 while OOS logits are mostly > -0.75
    records.append(
        {
            "query_id": "DEV_IN_001",
            "split": "inscope_dev",
            "category": "malformed",
            "query": "outlier query",
            "expected_doc_ids": ["CVE-2021-44228"],
            "router_route": "SEMANTIC_RAG",
            "top1_logit": -0.2,
            "top1_doc_id": "CVE-2021-44228",
            "top_k_doc_ids": ["CVE-2021-44228"],
        }
    )
    for idx in range(2, 11):
        records.append(
            {
                "query_id": f"DEV_IN_{idx:03d}",
                "split": "inscope_dev",
                "category": "factual",
                "query": f"normal query {idx}",
                "expected_doc_ids": ["CVE-2021-44228"],
                "router_route": "SEMANTIC_RAG",
                "top1_logit": 4.5,
                "top1_doc_id": "CVE-2021-44228",
                "top_k_doc_ids": ["CVE-2021-44228"],
            }
        )
    for idx in range(1, 11):
        records.append(
            {
                "query_id": f"DEV_OOS_{idx:03d}",
                "split": "oos_dev",
                "category": "lexical_decoy" if idx > 2 else "general_off_topic",
                "query": f"oos dev {idx}",
                "expected_doc_ids": [],
                "router_route": "HARD_REFUSAL" if idx <= 2 else "SEMANTIC_RAG",
                "top1_logit": -1.5 if idx <= 2 else 0.5,
                "top1_doc_id": "CVE-2021-44228",
                "top_k_doc_ids": ["CVE-2021-44228"],
            }
        )
    for idx in range(1, 141):
        records.append(
            {
                "query_id": f"TEST_{idx:03d}",
                "split": "golden_150",
                "category": "FACTUAL",
                "query": f"golden in {idx}",
                "expected_doc_ids": ["CVE-2021-44228"],
                "router_route": "SEMANTIC_RAG",
                "top1_logit": 5.0,
                "top1_doc_id": "CVE-2021-44228",
                "top_k_doc_ids": ["CVE-2021-44228"],
            }
        )
    for idx in range(121, 131):
        records.append(
            {
                "query_id": f"TEST_OOS_{idx}",
                "split": "golden_150",
                "category": "OUT_OF_SCOPE",
                "query": f"golden oos {idx}",
                "expected_doc_ids": [],
                "router_route": "HARD_REFUSAL" if idx <= 128 else "SEMANTIC_RAG",
                "top1_logit": -2.0 if idx <= 129 else 1.0,
                "top1_doc_id": "CVE-2021-44228",
                "top_k_doc_ids": ["CVE-2021-44228"],
            }
        )

    logits_path = cal_dir / "logits.jsonl"
    logits_path.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")
    cal_payload = select_threshold(records, cfg=cfg)
    assert cal_payload["dev_oos_abstain_rate"] == 0.2
    (models_dir / "abstain_calibration.json").write_text(
        json.dumps(cal_payload, indent=2), encoding="utf-8"
    )

    generate_report(cfg)
    rpt_json = json.loads((reports_dir / "phase15_calibration_report.json").read_text(encoding="utf-8"))
    rpt_md = (reports_dir / "phase15_calibration_report.md").read_text(encoding="utf-8")

    assert len(rpt_json["golden_150_heldout_eval"]["oos_items"]) == 10
    assert rpt_json["low_oos_abstain_diagnostic"]["flagged"] is True
    assert rpt_json["low_oos_abstain_diagnostic"]["driving_low_inscope_outliers"][0]["query_id"] == "DEV_IN_001"
    assert "DEV_IN_001" in rpt_md
    assert "prompt-level LLM refusal backstop" in rpt_md
    assert "TEST_OOS_130" in rpt_md


def test_ablation_runtime_preconditions_isolation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """_assert_runtime_preconditions enforces CUDA_VISIBLE_DEVICES='', no .venv_gpu, and enabled calibration."""
    models_dir = tmp_path / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    cfg = Settings(models_dir=models_dir, abstention_enabled=False)

    # 1. CUDA_VISIBLE_DEVICES != "" -> HardwareIsolationError
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    with pytest.raises(HardwareIsolationError, match="CUDA_VISIBLE_DEVICES"):
        _assert_runtime_preconditions(cfg)

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")

    # 2. .venv_gpu directory present -> HardwareIsolationError
    (tmp_path / ".venv_gpu").mkdir()
    with pytest.raises(HardwareIsolationError, match=".venv_gpu"):
        _assert_runtime_preconditions(cfg)
    (tmp_path / ".venv_gpu").rmdir()

    # 3. abstention_enabled=False -> ConfigError (must be enabled for Phase 1.5 ablation)
    (models_dir / "precision_lock.json").write_text(
        json.dumps(
            {
                "embed_precision": cfg.embed_precision,
                "rerank_precision": cfg.rerank_precision,
                "router_precision": cfg.router_precision,
                "rerank_max_length": cfg.rerank_max_length,
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="calibrated abstention enabled"):
        _assert_runtime_preconditions(cfg)


def test_post_stage4_calibration_recomputes_identically() -> None:
    """When Stage 4 calibration outputs exist on disk, recomputation matches abstain_calibration.json."""
    cfg = get_settings()
    cal_path = cfg.models_dir / "abstain_calibration.json"
    logits_path = cfg.data_processed / "calibration" / "logits.jsonl"
    if not cal_path.exists() or not logits_path.exists():
        pytest.skip("Stage 4 calibration artifacts not generated yet")

    saved = json.loads(cal_path.read_text(encoding="utf-8"))
    rows = [json.loads(line) for line in logits_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    recomputed = select_threshold(rows, cfg=cfg)

    assert saved["dev_in_scope_n"] == 100
    assert saved["dev_oos_n"] == 60
    assert saved["golden_excluded_count"] == 150
    assert saved["selection_rule"] == "floor_to_0.25(min_in_scope_dev_logit - 0.5)"
    assert saved["reranker_model_id"] == recomputed["reranker_model_id"]
    assert saved["threshold_T"] == recomputed["threshold_T"]
    assert saved["roc_auc"] == recomputed["roc_auc"]
    assert saved["dev_oos_abstain_rate"] == recomputed["dev_oos_abstain_rate"]
    assert saved["dev_false_abstain_rate"] == recomputed["dev_false_abstain_rate"]
    assert "created_at" in saved
    verify_abstain_calibration(cfg)
