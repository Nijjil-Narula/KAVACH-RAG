# KAVACH-RAG Phase 2 Generation Evaluation Report

## 1. Configuration & Judge Banner

- **Generated at (UTC):** 2026-11-11T23:47:25Z
- **Judge Backend:** `gemini` (`gemini-3.5-flash-lite`)
- **Cross-Family Judge:** `True`
- **Gemini Thinking Applied:** `True`
- **Gemini API Key Present:** `True`
- **Prompt Version:** `tier5-v1-965a46c8fae9`
- **Model Digest:** `0a5dc38a6caac1e2e4941bc87d7f0f6f8ed710e62a6ed8ad24f60a6d5c73aa4e`
- **Decoding Window & Penalty:** `num_ctx=4096`, `repeat_penalty=1.0`

## 2. Engineering Gate Table

| Gate | Observed | Target | Status |
|---|---|---|---|
| Quantization Q4_K_M | Q4_K_M | Q4_K_M | **PASS** |
| Full GPU offload (>= 99% VRAM) | 1.0000 | >= 0.9900 | **PASS** |
| Peak VRAM <= hard limit | 5343.0 MiB | <= 5478 MiB (target <= 4710 MiB) | **PASS** |
| CPU/GPU isolation & no forbidden retrieval imports | torch_cuda=False, venv_gpu=False, gemini_api_key_present=True | ok == True | **PASS** |
| Context window & decoding contract | num_ctx=4096, repeat_penalty=1.0, digest=0a5dc38a6caac1e2e4941bc87d7f0f6f8ed710e62a6ed8ad24f60a6d5c73aa4e | num_ctx<=4096, repeat_penalty=1.0, digest present | **PASS** |
| Refusal bypass sub-ms & zero LLM calls | max=3.600 us (n=10) | < 1000 us, 0 LLM calls | **PASS** |
| Batch generation 150/150 completion | total=150, errors=0 | 150 items, 0 errors | **PASS** |

## 3. GPU VRAM Telemetry

| Metric | Value (MiB) | Budget / Limit |
|---|---:|---|
| Initial Baseline | 528 | <= 700 MiB preflight limit |
| Loaded Idle | 5156 | - |
| Run Peak | 5343.00 | Target <= 4710 MiB / Hard Limit <= 5478 MiB |
| Delta (Run Peak - Baseline) | 4815.00 | Within target=False, Within hard limit=True |

## 4. Latency & Throughput

- **Served items:** 140 | **Bypassed items:** 10
- **Spec Targets:** TTFT <= `300 ms`, Decode throughput `35–45 tok/s`.

| Scope | Count | TTFT P50 / P95 / Mean (ms) | Decode TPS P50 / P95 / Mean | Total P50 / P95 / Mean (ms) |
|---|---:|---|---|---|
| Overall (Served) | 140 | 575.33 / 820.48 / 587.31 | 38.84 / 40.21 / 38.82 | 2809.85 / 8684.99 / 3477.39 |
| CONFLICT_TRAP | 15 | 444.10 / 673.98 / 483.17 | 39.83 / 40.58 / 39.67 | 2086.42 / 13750.12 / 5286.59 |
| CONTEXT_NOISE | 10 | 510.84 / 880.84 / 579.63 | 39.77 / 40.08 / 39.70 | 3383.99 / 4942.19 / 3599.01 |
| FACTUAL | 60 | 644.49 / 822.42 / 611.22 | 38.14 / 39.23 / 38.23 | 2104.82 / 7313.72 / 2935.57 |
| MALFORMED | 10 | 469.16 / 802.32 / 492.43 | 39.69 / 40.71 / 39.92 | 1995.95 / 3099.46 / 2106.07 |
| MULTI_HOP | 45 | 516.56 / 780.13 / 612.93 | 38.88 / 39.58 / 38.90 | 3412.82 / 7497.44 / 3874.47 |

- **Bypass Latency (us):** count=10, min=0.800, p50=1.000, mean=1.350, p95=2.880, max=3.600 us
- **Target Missed (TTFT):** Observed served TTFT mean `587.31 ms` exceeds `300 ms` target.

## 5. Quality Gates (Stage 1 + Stage 2)

| Gate | Metric | Observed | Threshold | Status |
|---|---|---:|---:|---|
| Q1 | Faithfulness Mean | 0.8628 | >= 0.85 | **PASS** |
| Q2 | Relevance Mean | 0.5679 | >= 0.80 | **FAIL** |
| Q3 | Citation Validity Mean | 1.0000 | >= 0.95 | **PASS** |

- **Evaluated Served (non-OOS):** `140` | **Context-Sufficient Subset:** `72` (Relevance Mean: `0.9236`)
- **Served Model Refusals:** `48` | **Over-Refusals (when context sufficient):** `3`
- **Refusal Correctness (all items):** true=102, false=48

### Faithfulness by Question Type

| Question Type | Faithfulness Mean |
|---|---:|
| CONFLICT_TRAP | 0.9857 |
| CONTEXT_NOISE | 0.5000 |
| FACTUAL | 0.9307 |
| MALFORMED | 0.9167 |
| MULTI_HOP | 0.5267 |

## 6. Citation & Literal Fidelity Metrics

| Metric | Value |
|---|---:|
| Citation Validity Mean | 1.0000 |
| Citation Precision (Chunk) Mean | 0.7092 |
| Citation Recall (Chunk) Mean | 0.8069 |
| Citation Precision (Doc) Mean | 0.8299 |
| Citation Recall (Doc) Mean | 0.8168 |
| Entity Recall Mean | 0.8403 |
| Items with Hallucinated Citations | 0 |
| Items with Ungrounded Decimals | 0 |

## 7. CONFLICT_TRAP & CONTEXT_NOISE Analysis

- **CONFLICT_TRAP (`n=15`):** `both_versions_in_context=12`, `both_versions_in_answer=5`, `both_versions_preserved_when_in_context=5` (rate=0.4167).
- **CONTEXT_NOISE (`TEST_141`–`TEST_150`, `n=10`):** `echo_ratio` mean=0.0050, max=0.0290, non-zero items=2, injection_leakage_detected=True.

## 8. Failure Inventory

### Lowest Faithfulness Items (Top 10)

| test_id | Type | Faithfulness | Relevance | Retrieval Miss | Context Sufficient | Tag |
|---|---|---:|---:|---|---|---|
| TEST_045 | FACTUAL | 0.000 | 0.50 | False | True | `generation-limited` |
| TEST_072 | MULTI_HOP | 0.000 | 0.50 | False | False | `retrieval-limited` |
| TEST_098 | MULTI_HOP | 0.000 | 0.00 | False | False | `retrieval-limited` |
| TEST_048 | FACTUAL | 0.200 | 0.50 | False | True | `generation-limited` |
| TEST_073 | MULTI_HOP | 0.333 | 0.00 | True | False | `retrieval-limited` |
| TEST_086 | MULTI_HOP | 0.333 | 0.50 | False | False | `retrieval-limited` |
| TEST_044 | FACTUAL | 0.500 | 0.50 | False | True | `generation-limited` |
| TEST_046 | FACTUAL | 0.500 | 1.00 | True | False | `retrieval-limited` |
| TEST_078 | MULTI_HOP | 0.500 | 1.00 | False | False | `retrieval-limited` |
| TEST_104 | MULTI_HOP | 0.500 | 0.00 | False | False | `retrieval-limited` |

### Hallucinated Citation Items

| test_id | Type | Hallucinated Citations | Retrieval Miss | Context Sufficient | Tag |
|---|---|---|---|---|---|
| None | - | - | - | - | - |

### False Refusal / Generation Failure Items

| test_id | Type | Retrieval Miss | Context Sufficient | Tag | Error |
|---|---|---|---|---|---|
| TEST_007 | FACTUAL | True | False | `retrieval-limited` | - |
| TEST_017 | FACTUAL | True | False | `retrieval-limited` | - |
| TEST_019 | FACTUAL | True | False | `retrieval-limited` | - |
| TEST_024 | FACTUAL | True | False | `retrieval-limited` | - |
| TEST_049 | FACTUAL | True | False | `retrieval-limited` | - |
| TEST_056 | FACTUAL | True | False | `retrieval-limited` | - |
| TEST_061 | MULTI_HOP | True | False | `retrieval-limited` | - |
| TEST_063 | MULTI_HOP | False | False | `retrieval-limited` | - |
| TEST_064 | MULTI_HOP | False | False | `retrieval-limited` | - |
| TEST_065 | MULTI_HOP | True | False | `retrieval-limited` | - |
| TEST_066 | MULTI_HOP | True | False | `retrieval-limited` | - |
| TEST_067 | MULTI_HOP | True | False | `retrieval-limited` | - |
| TEST_068 | MULTI_HOP | True | False | `retrieval-limited` | - |
| TEST_069 | MULTI_HOP | True | False | `retrieval-limited` | - |
| TEST_070 | MULTI_HOP | False | False | `retrieval-limited` | - |
| TEST_071 | MULTI_HOP | False | False | `retrieval-limited` | - |
| TEST_074 | MULTI_HOP | False | False | `retrieval-limited` | - |
| TEST_075 | MULTI_HOP | False | False | `retrieval-limited` | - |
| TEST_076 | MULTI_HOP | False | False | `retrieval-limited` | - |
| TEST_077 | MULTI_HOP | True | False | `retrieval-limited` | - |
| TEST_079 | MULTI_HOP | True | False | `retrieval-limited` | - |
| TEST_081 | MULTI_HOP | True | False | `retrieval-limited` | - |
| TEST_083 | MULTI_HOP | False | False | `retrieval-limited` | - |
| TEST_085 | MULTI_HOP | True | False | `retrieval-limited` | - |
| TEST_087 | MULTI_HOP | False | False | `retrieval-limited` | - |
| TEST_088 | MULTI_HOP | False | False | `retrieval-limited` | - |
| TEST_089 | MULTI_HOP | False | False | `retrieval-limited` | - |
| TEST_090 | MULTI_HOP | False | False | `retrieval-limited` | - |
| TEST_091 | MULTI_HOP | False | False | `retrieval-limited` | - |
| TEST_094 | MULTI_HOP | False | False | `retrieval-limited` | - |
| TEST_095 | MULTI_HOP | True | False | `retrieval-limited` | - |
| TEST_096 | MULTI_HOP | False | False | `retrieval-limited` | - |
| TEST_097 | MULTI_HOP | True | False | `retrieval-limited` | - |
| TEST_100 | MULTI_HOP | False | False | `retrieval-limited` | - |
| TEST_103 | MULTI_HOP | True | False | `retrieval-limited` | - |
| TEST_105 | MULTI_HOP | True | False | `retrieval-limited` | - |
| TEST_109 | CONFLICT_TRAP | True | False | `retrieval-limited` | - |
| TEST_131 | MALFORMED | False | True | `generation-limited` | - |
| TEST_132 | MALFORMED | False | True | `generation-limited` | - |
| TEST_138 | MALFORMED | False | True | `generation-limited` | - |
| TEST_141 | CONTEXT_NOISE | True | False | `retrieval-limited` | - |
| TEST_142 | CONTEXT_NOISE | True | False | `retrieval-limited` | - |
| TEST_143 | CONTEXT_NOISE | True | False | `retrieval-limited` | - |
| TEST_144 | CONTEXT_NOISE | True | False | `retrieval-limited` | - |
| TEST_147 | CONTEXT_NOISE | True | False | `retrieval-limited` | - |
| TEST_148 | CONTEXT_NOISE | False | False | `retrieval-limited` | - |
| TEST_149 | CONTEXT_NOISE | True | False | `retrieval-limited` | - |
| TEST_150 | CONTEXT_NOISE | True | False | `retrieval-limited` | - |

## 9. Arithmetic End-to-End Latency Estimate

> **Methodology Notice:** `computed from separate runs, not an end-to-end measurement`.

| Component | All-150 Mean (ms) | Served / SEMANTIC_RAG Mean (ms) |
|---|---:|---:|
| Phase 1.5 Retrieval Baseline (`reports/phase15_latency_baseline.json`) | 9449.964 | 10496.397 |
| Phase 2 Generation (`data/processed/generation_results_150.jsonl`) | 3245.566 | 3477.392 |
| **Arithmetic End-to-End Estimate** (computed from separate runs, not an end-to-end measurement) | **12695.530** | **13973.789** |
