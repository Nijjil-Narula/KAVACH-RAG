# KAVACH-RAG Phase 1.5 CPU-Only Hotfix — Completion Report

**Status:** COMPLETE (`PRECISION_LOCKED`, `CALIBRATED`, `BASELINE_DONE`, `ABLATION_DONE`, `FINAL_VERIFICATION_PASSED`)  
**Host Hardware:** Windows 11 (`Windows-11-10.0.26200-SP0`), `Intel64 Family 6 Model 186 Stepping 2, GenuineIntel` (13th Gen hybrid: 6 P-cores / 12 SMT threads + 4 E-cores = 16 logical CPUs), 16 GB RAM  
**Runtime Stack:** `.venv\Scripts\python.exe` (Python 3.12.6, `torch==2.15.0+cpu`, `quantized_engine=onednn`), strict 100% CPU-only execution (`CUDA_VISIBLE_DEVICES=""`, `OMP_NUM_THREADS=2`, `MKL_NUM_THREADS=2`, `OPENBLAS_NUM_THREADS=2`, `TOKENIZERS_PARALLELISM=false`, `.venv_gpu` absent)

---

## 7.1 Full Test Suite & Direct CPU Isolation Probe Verification

All 13 test suites in [`tests/`](file:///c:/Users/Nijjil%20Narula/DOC/PROJECTS/KAVACH-RAG/tests) (`test_calibration.py`, `test_embedder.py`, `test_golden.py`, `test_ingest.py`, `test_metrics.py`, `test_no_placeholders.py`, `test_pipeline_smoke.py`, `test_quantization_gate.py`, `test_reranker.py`, `test_residency.py`, `test_router.py`, `test_runtime_guard.py`, `test_vector_store.py`) and PowerShell runner AST parsing passed with **zero failures and zero skips**:

- **Runner AST Check (`scripts/run_phase15.ps1`):** `PS1_PARSE_OK tokens=363`
- **Full Pytest (`.\.venv\Scripts\python.exe -m pytest -q -rs`):** **`108 passed, 0 failed, 0 skipped, 1 warning`**
  - Only warning: `FutureWarning: 'torch.jit.script' is deprecated. Please switch to 'torch.compile' or 'torch.export'.` (triggered internally during `test_pipeline_smoke_five_queries_plus_oos_real_model`)
- **Direct CPU Isolation Probe (`kavach.runtime_guard.vram_probe()`):**
  - `torch_version= 2.15.0+cpu`
  - `cuda_available= False`
  - `quantized_engine= onednn`
  - `vram_probe= {'torch_cuda_build': False, 'cuda_available': False, 'pid_in_nvidia_smi': False}`
  - `venv_gpu_exists= False`
  - `bench_lock_exists= False`

### Verbatim Calibrated Pipeline Smoke Outputs (`test_pipeline_smoke_five_queries_plus_oos_real_model`, `-s`)

```text
[SMOKE] exact_cve        | route=FAST_PATH_EXACT_LOOKUP | refused=False | abstained=False | reason=None | top1_logit=None | contexts=1 | ungated=0 | latency_ms={'router_ms': 0.342, 'lookup_ms': 0.005, 'total_ms': 0.441}
[SMOKE] multi_hop        | route=SEMANTIC_RAG           | refused=False | abstained=False | reason=None | top1_logit=8.495824813842773 | contexts=3 | ungated=3 | latency_ms={'router_ms': 74.647, 'embed_ms': 1916.611, 'search_ms': 5.428, 'rerank_ms': 4025.886, 'total_ms': 6022.632}
[SMOKE] golden_oos_1     | route=HARD_REFUSAL           | refused=True  | abstained=False | reason=None | top1_logit=None | contexts=0 | ungated=0 | latency_ms={'router_ms': 61.33, 'total_ms': 61.342}
[SMOKE] typo_query       | route=SEMANTIC_RAG           | refused=False | abstained=False | reason=None | top1_logit=9.49897289276123 | contexts=3 | ungated=3 | latency_ms={'router_ms': 56.737, 'embed_ms': 1369.507, 'search_ms': 0.415, 'rerank_ms': 5941.933, 'total_ms': 7368.631}
[SMOKE] owasp_llm03      | route=SEMANTIC_RAG           | refused=False | abstained=False | reason=None | top1_logit=10.52985954284668 | contexts=3 | ungated=3 | latency_ms={'router_ms': 61.653, 'embed_ms': 1503.457, 'search_ms': 0.381, 'rerank_ms': 3822.214, 'total_ms': 5387.738}
[SMOKE] golden_oos_item9 | route=SEMANTIC_RAG           | refused=False | abstained=True  | reason=LOW_RETRIEVAL_CONFIDENCE | top1_logit=3.666347026824951 | contexts=0 | ungated=3 | latency_ms={'router_ms': 58.15, 'embed_ms': 1209.178, 'search_ms': 0.969, 'rerank_ms': 1058.543, 'total_ms': 2326.872}
[SMOKE] rss_after_load_mb=2796.2 | rss_after_queries_mb=2859.4 | ceiling_mb=4096.0 | vram_probe={'torch_cuda_build': False, 'cuda_available': False, 'pid_in_nvidia_smi': False}
```

---

## 7.2 Cold-vs-Warm CPU Audit & Tokenization Audit ([`reports/phase15_profile.json`](file:///c:/Users/Nijjil%20Narula/DOC/PROJECTS/KAVACH-RAG/reports/phase15_profile.json))

### OS Preflight & Quantization Backend Smoke Check

- **Power & Thermal State:** AC plugged (`power_plugged=true`, battery `79.0%`), active power scheme `Power Scheme GUID: 381b4222-f694-41f0-9685-ff5bb260df2e (Balanced)` (observed honestly; not mutated).
- **Process Hygiene:** `competing_python_count=0`; `cpu_idle_wait`: settled in `11.03 s` at `13.8%` CPU (`< 15.0%` threshold).
- **Dynamic Quantization Safety (`torch.ao.quantization.quantize_dynamic`):** Supported engines `['onednn']` (`fbgemm` absent in `torch==2.15.0+cpu`; fallback logged once to `onednn`); linear smoke test quantized `2` Linear modules with output cosine similarity `0.999898` (`smoke_ok=true`).

### Isolated Cold-vs-Warm Model Audit (`fp32`, Single-Tier Subprocesses)

| Tier | `rss_before_mb` | `rss_after_load_mb` | `rss_after_warm_mb` | `load_time_ms` | `first_call_ms` | `warmup_3_passes_ms` | `warm_call_ms` (Mean) | `warm_call` P50 / P95 (ms) | First/Warm Ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `router` | 284.57 | 1029.12 | 1059.75 | 6856.871 | 313.095 | 1083.421 | 187.838 | 188.338 / 189.451 | 1.67× |
| `embedder` | 284.09 | 2726.47 | 2745.96 | 7048.843 | 1499.462 | 21518.178 | 470.836 | 452.495 / 539.975 | 3.18× |
| `reranker` | 283.91 | 965.52 | 1064.10 | 3533.832 | 11270.140 | 30590.221 | 7155.029 | 5675.218 / 11798.127 | 1.58× |
| **3-Tier Sum** | — | — | — | **17439.546** | **13082.697** | **53191.820** | **7813.703** | — | — |

- **Cold Load + First Inference Total:** `30522.243 ms` (`30.52 s`) vs **Steady-State Warm Single-Query Sum:** `7813.703 ms` (`7.81 s`).
- **Cold-vs-Warm Thrash Delta (`cold_vs_warm_delta_ms`):** **`22708.54 ms` (`22.71 s`)** eliminated per cold reload by keeping all 3 models resident and warmed up inside [`VectorRetrievalPipeline`](file:///c:/Users/Nijjil%20Narula/DOC/PROJECTS/KAVACH-RAG/kavach/retrieval/pipeline.py).

### ModernBERT Tokenization & Decision D2 Truncation Audit

| Distribution (`ettin-reranker-150m-v1`, `special_tokens=3`) | Count | Min | Mean | P50 | P95 | Max | Threshold Overflow Counts |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Golden Query Tokens (`golden_150`) | 150 | 3.0 | 45.847 | 23.0 | 374.0 | 384.0 | `10` queries `> 96` (`CONTEXT_NOISE`) |
| Corpus Passage Tokens (`corpus_chunks.parquet`) | 505 | 52.0 | 345.756 | 409.0 | 540.8 | 582.0 | **`78`** `> 512`, **`0`** `> 600` |
| Checkpoint Rerank Pairs (Query + Passage + 3) | 1200 | 62.0 | 359.799 | 448.0 | 582.05 | 923.0 | **`21`** `> 640` (8 queries: `TEST_141..143, 145..149`), **`0`** `> 1024` |

Because the frozen corpus chunks were segmented under the F2LLM tokenizer rather than ModernBERT BPE, `78/505` chunks exceed `512` tokens (up to `582`). Raising `rerank_max_length` to `640` and `_PASSAGE_MAX_TOKENS` to `600` (user-approved Option A) guarantees `passages_truncated == 0` across every frozen chunk without re-ingesting the index.

---

## 7.3 Hybrid CPU P-Core / E-Core Affinity Benchmark ([`reports/phase15_profile.json`](file:///c:/Users/Nijjil%20Narula/DOC/PROJECTS/KAVACH-RAG/reports/phase15_profile.json))

Win32 `GetLogicalProcessorInformationEx` (`RelationProcessorCore`) detected `affinity_unavailable=false`, `16` logical CPUs across **`6` P-cores × 2 SMT siblings** (`[0,1], [2,3], [4,5], [6,7], [8,9], [10,11]`) and **`4` E-cores × 1** (`[12], [13], [14], [15]`):

| Affinity Mode | CPU Mask | Idle Wait (CPU %) | Router Mean (ms) | Embed Mean (ms) | Rerank-8 Mean (ms) | Total Neural Mean (ms) | Total P50 / P95 (ms) | Probe RSS (`fp32`) |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `off` | `[0..15]` | 12.53s (9.7%) | 317.728 | 1252.272 | 17607.454 | **19177.454** | 19900.887 / 22298.603 | 2837.58 MB |
| `p_core_smt` | `[0, 1]` | 29.58s (6.7%) | 84.933 | 537.391 | 10033.054 | **10655.378** | 10405.437 / 14390.445 | 4102.95 MB |
| **`p_core_distinct`** | **`[0, 2]`** | 5.52s (7.4%) | **67.861** | **431.809** | **6970.709** | **7470.378** | **8112.732 / 8968.419** | 4119.23 MB |

- **Speedup (`p_core_distinct` vs `off`):** **`-61.05%`** total neural latency (`19177.454 ms -> 7470.378 ms`; `2.57×` faster), avoiding both E-core scheduling thrash and SMT sibling contention. Selected `recommended_affinity_mode = "p_core_distinct"`.

---

## 7.4 Precision Matrix, Quality Gates & Precision Lock ([`models/precision_lock.json`](file:///c:/Users/Nijjil%20Narula/DOC/PROJECTS/KAVACH-RAG/models/precision_lock.json))

### Stage 2C Stratified 40-Query CPU Matrix (`affinity=p_core_distinct`, 5 Warmups)

| Tier | Precision / Config | Mean (ms) | P50 (ms) | P95 (ms) | Isolated RSS (MB) | Notes |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| `router@128` | `int8` | 52.486 | 45.580 | 84.700 | 1440.38 | Failed accuracy gate (-22.22 pp) |
| **`router@128`** | **`fp32`** | **70.925** | **59.968** | **116.015** | **1060.09** | **Selected (#1 passing latency)** |
| `router@128` | `bf16` | 810.329 | 623.774 | 1533.015 | 773.79 | Passed gate; 11.4× slower than `fp32` |
| `embedder@512` | `int8` | 445.192 | 223.783 | 1153.986 | 2768.38 | Failed accuracy gate (`cos_min=0.053`) |
| `embedder@512` | `fp32` | 963.349 | 496.682 | 3055.613 | 2745.97 | Failed `sq8_top8_overlap` (`0.9733 < 0.98`) |
| **`embedder@512`** | **`bf16`** | **5521.392** | **1856.308** | **17224.049** | **1601.22** | **Selected (ONLY passing embedder; saves 1.07 GB)** |
| `sq8_search` | `sq8_mrl512` | 0.268 | 0.209 | 0.591 | 299.71 | `search_only_ms.mean=0.264` |
| `reranker` | `int8@640` | 3014.117 | 3263.136 | 4037.411 | 994.03 | Failed ranking/logit gate (`top3_overlap=0.68`) |
| **`reranker`** | **`fp32@640`** | **5343.920** | **5807.173** | **7021.012** | **1061.91** | **Selected (ONLY passing reranker)** |
| `reranker` | `bf16@640` | — | — | — | — | Excluded per Decision D5 (ModernBERT CPU) |

### Stage 3 Quality Gate Results ([`reports/phase15_quality_gates.json`](file:///c:/Users/Nijjil%20Narula/DOC/PROJECTS/KAVACH-RAG/reports/phase15_quality_gates.json))

1. **Tier 2 Embedder Gate (`baseline_run4_doc_recall_at_8 = 0.775`):**
   - `bf16` (**PASSED**): `rss=1696.08 MB`, `cos_1024_min=0.998346` (`mean=0.999618`), `cos_512_min=0.99836` (`mean=0.999609`), `failed_cosine_test_ids=[]`, `sq8_top8_overlap_mean=0.985833 >= 0.98`, `run4_doc_recall_at_8=0.775` (`delta=0.0`).
   - `fp32` (**FAILED**): `rss=2765.44 MB`, `cos_1024_min=0.998345`, `cos_512_min=0.998045`, `run4_doc_recall_at_8=0.775` (`delta=0.0`), but `sq8_top8_overlap_mean=0.973333 < 0.98` because the frozen SQ8 index and checkpoint vectors were generated originally under `bfloat16`.
   - `int8` (**FAILED**): `cos_1024_min=0.053376`, `cos_512_min=0.025045`, `sq8_top8_overlap_mean=0.071667`, `run4_doc_recall_at_8=0.2071` (`delta=-0.5679`), `150/150` queries failed `cosine >= 0.995`. Dynamic Linear `onednn` INT8 destroys residual/RMSNorm scaling in `F2LLM-v2-0.6B` decoder blocks.
2. **Tier 4 Reranker Gate (strict `top3_overlap >= 0.98`, `|ΔMRR| <= 0.01`, `logit_corr >= 0.99`, `passages_truncated == 0`):**
   - **Baseline Drift Audit (`fp32@1024` vs `R_cur`):** `ckpt_run5_doc_mrr=0.8762` vs `fp32_1024_run5_doc_mrr=0.8762` (`delta=0.0`), `chunk_mrr_delta=-0.0011`, `ckpt_run6_doc_mrr=0.8875` vs `0.8875`, `top1_agreement=0.9933`, `top3_overlap=0.9867`, `top3_exact_match_rate=0.96`, `logit_mae=0.01617`, `logit_corr=0.999977`. Result: **`baseline_drift=false`**, `active_reference="R_cur"`.
   - **`fp32@640` (PASSED):** `rss=1086.33 MB`, `pairs_scored=1200`, `passages_truncated=0`, `queries_truncated=21`, `run5_doc_mrr=0.8726` (`delta=-0.0036`), `run5_chunk_mrr=0.6762` (`delta=-0.0059`), `run5_doc_recall_at_3=0.7286`, `run5_doc_ndcg_at_3=0.7576`, `run6_doc_mrr=0.8821`. Pairwise vs `R_cur`: `top1_agreement=0.9867`, **`top3_overlap=0.98`**, `top3_exact_match_rate=0.9467`, `logit_mae=0.02677`, `logit_corr=0.99929`. Pairwise vs `fp32@1024`: `top1_agreement=0.9933`, `top3_overlap=0.9933`, `top3_exact_match_rate=0.9867`, `logit_corr=0.999294`. Result: **`contingency_l_triggered=false`**.
   - `int8@640` (**FAILED**): `run5_doc_mrr=0.8571` (`delta=-0.0191`), `run5_chunk_mrr=0.544` (`delta=-0.1381`), `top1_agreement=0.58`, `top3_overlap=0.68`, `top3_exact_match_rate=0.32`, `logit_mae=1.18611`, `logit_corr=0.922018`. Dynamic `onednn` INT8 severely perturbs ModernBERT classification logits.
3. **Tier 1 Router Gate (`baseline_held_out_accuracy = 1.0` across `153` synthetic + `150` golden):**
   - `fp32` (**PASSED**, selected #1): `golden_correct=149/150` (`0.993333`), `held_out_accuracy=1.0` (`153/153`, `delta=0.0 pp`), `rss=1368.3 MB`.
   - `bf16` (**PASSED**): `golden_correct=149/150` (`0.993333`), `held_out_accuracy=1.0` (`153/153`, `delta=0.0 pp`), `rss=1077.35 MB`.
   - `int8` (**FAILED**): `golden_correct=139/150` (`0.926667`, 11 misclassified incl. OWASP LLM02/LLM03/LLM05/LLM07 -> `HARD_REFUSAL`), `held_out_accuracy=0.777778` (`119/153`, `delta=-22.2222 pp`).

### Final Locked Configuration ([`models/precision_lock.json`](file:///c:/Users/Nijjil%20Narula/DOC/PROJECTS/KAVACH-RAG/models/precision_lock.json))

- `embed_precision`: `"bf16"` | `rerank_precision`: `"fp32"` | `router_precision`: `"fp32"`
- `rerank_max_length`: `640` | `rerank_long_query_max_length`: `null` | `rerank_long_query_token_threshold`: `96` | `contingency_l_applied`: `false`
- `reranker_is_fallback`: `false` (`cross-encoder/ettin-reranker-150m-v1`)
- `cpu_affinity_mode`: `"p_core_distinct"` (`[0, 2]`) | `quantized_engine`: `"onednn"` | `torch_version`: `"2.15.0+cpu"`
- `resident_rss_mb`: **`2953.3 MB`** (`<= 4096.0 MB` ceiling; `1142.7 MB` headroom; `downgrades_applied=[]`)

---

## 7.5 Calibrated Abstention Threshold ([`models/abstain_calibration.json`](file:///c:/Users/Nijjil%20Narula/DOC/PROJECTS/KAVACH-RAG/models/abstain_calibration.json), [`reports/phase15_calibration_report.json`](file:///c:/Users/Nijjil%20Narula/DOC/PROJECTS/KAVACH-RAG/reports/phase15_calibration_report.json))

Calibrated strictly on isolated dev sets (`dev_in_scope_100` + `dev_oos_60`, `golden_excluded_count=150`):

- **Threshold Rule:** `floor_to_0.25(min_in_scope_dev_logit - 0.5)` = `floor_to_0.25(7.525618 - 0.5)` -> **`threshold_T = 7.0`**
- **Dev Distributions:**
  - `inscope_dev_100`: `min=7.525618` (`DEV_IN_069`, `owasp_version_agnostic`), `p05=8.958076`, `p25=10.214424`, `median=10.753274`, `mean=10.679231`, `max=12.397745`.
  - `oos_dev_60`: `min=-2.631799`, `p25=-0.589555`, `median=0.915371`, `p75=2.347824`, `p95=6.022878`, `max=7.99494`, `mean=1.42766`.
- **Dev Separation Metrics:**
  - **ROC AUC (`inscope_dev_100` vs `oos_dev_60`):** **`0.999667`** (`~0.9997`)
  - **`dev_false_abstain_rate` at `T=7.0`:** **`0.0`** (`0/100`)
  - **`dev_oos_abstain_rate` at `T=7.0`:** **`0.966667`** (`58/60`; `system_oos_abstain_rate=0.966667`, balanced accuracy `0.983333`)
- **Two `oos_dev_60` False-Accept Outliers (`unindexed_public_security`, near-domain lexical overlap):**
  1. `DEV_OOS_056` (`top1_logit=7.99494` mapped to `MITRE_T1078`): *"Describe adversary persistence via SSH authorized keys covered by ATT&CK T1098.004."*
  2. `DEV_OOS_051` (`top1_logit=7.26043` mapped to `CVE-2023-4863`): *"Summarize the session token memory leakage vulnerability tracked as CVE-2023-4966."*
- **Held-Out `golden_150.jsonl` Verification (Fixed `T=7.0`, Zero Retuning):**
  - **`inscope_false_abstain_rate`:** **`0.0`** (`0/140`; lowest in-scope category min was `7.403697` on `CONTEXT_NOISE`)
  - **`oos_block_rate`:** **`1.0`** (`10/10` — `9/10` router `HARD_REFUSAL`: `TEST_121..TEST_128, TEST_130`; `1/10` reranker gate abstention: `TEST_129` at `top1_logit=3.666347 < 7.0`)

---

## 7.6 Fresh CPU-Only 6-Run Ablation & Reference Comparison ([`reports/retrieval_ablation_results.json`](file:///c:/Users/Nijjil%20Narula/DOC/PROJECTS/KAVACH-RAG/reports/retrieval_ablation_results.json))

Legacy mixed CUDA/CPU Phase 1 artifacts are archived cleanly under [`reports/archive/gpu_hybrid/`](file:///c:/Users/Nijjil%20Narula/DOC/PROJECTS/KAVACH-RAG/reports/archive/gpu_hybrid/) and cited strictly as a **superseded reference**. Fresh CPU-only evaluation (`peak_rss_mb=2948.9 MB <= 4096.0 MB`, `passages_truncated=0`, `queries_truncated=21`):

### Runs 1–6 Headline Document-Level Table (140 In-Scope Queries) vs Superseded GPU-Hybrid Reference

| Run | Configuration | `k` | CPU R@1 | CPU R@3 | CPU R@k | CPU MRR | Superseded Ref MRR | `ΔMRR` | CPU nDCG@k | Chunk MRR | CPU Mean (ms) | Ref Mean (ms) | Drift `>0.01` |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Run 1 | `w256 \| Flat(1024)` | 3 | 0.6536 | 0.7071 | 0.7071 | 0.8714 | 0.8714 | +0.0000 | 0.7316 | N/A | 3240.66 | 2792.52 | `false` |
| Run 2 | `w512 \| Flat(1024)` | 3 | 0.6393 | 0.7071 | 0.7071 | 0.8524 | 0.8524 | +0.0000 | 0.7267 | 0.4607 | 3240.57 | 2792.44 | `false` |
| Run 3 | `w512 \| SQ8 mrl512` | 3 | 0.6107 | 0.6929 | 0.6929 | 0.8107 | 0.8107 | +0.0000 | 0.7033 | 0.4369 | 3240.75 | 2792.61 | `false` |
| Run 4 | `w512 \| SQ8 mrl512` | 8 | 0.6107 | 0.7071 | 0.7750 | 0.8301 | 0.8301 | +0.0000 | 0.7437 | 0.4751 | 3240.75 | 2792.61 | `false` |
| Run 5 | `w512 \| SQ8+Rerank` | 3 | 0.6786 | 0.7286 | 0.7286 | 0.8726 | 0.8762 | -0.0036 | 0.7576 | 0.6762 | 8714.65 | 3688.75 | `false` |
| Run 6 | `w512 \| SQ8+Rerank` | 5 | 0.6786 | 0.7321 | 0.7571 | 0.8821 | 0.8875 | -0.0054 | 0.7726 | 0.6876 | 8714.65 | 3688.75 | `false` |

*(Zero runs flagged `|ΔMRR| > 0.01`; Run 6 `ΔRecall@5 = -0.0036` (`0.7571` vs `0.7607`).)*

### Calibrated Abstention & Router Golden Evaluation

- **Standalone Raw Runs 5–6 (`T=7.0`, un-normalized `item.query`):**
  - `oos_abstention_rate`: **`1.0000` (`10/10`)** in both Run 5 and Run 6.
  - `false_abstain_in_scope`: **`0.0214` (`3/140`)** — specifically 3 un-normalized `CONTEXT_NOISE` queries (`TEST_148=6.596936`, `TEST_149=6.858266`, `TEST_150=6.894686`). Gated effective metrics: Run 5 `Doc MRR=0.8690, R@3=0.7250`; Run 6 `Doc MRR=0.8786, R@5=0.7536`.
- **Live End-to-End Pipeline (`end_to_end_pipeline_eval` via [`VectorRetrievalPipeline.retrieve()`](file:///c:/Users/Nijjil%20Narula/DOC/PROJECTS/KAVACH-RAG/kavach/retrieval/pipeline.py)):**
  - `oos_blocked_rate`: **`1.0` (`10/10`)**
  - `inscope_false_block_rate`: **`0.0` (`0/140`)** — Tier 1 `normalize_query()` cleans excessive whitespace/tabs/control chars in noisy SOC logs before embedding & reranking.
  - Live mean total latency: `8060.21 ms` (`p50=7852.967 ms`, `p95=25308.909 ms`).
- **Tier 1 Router Evaluation ([`reports/router_golden_eval.json`](file:///c:/Users/Nijjil%20Narula/DOC/PROJECTS/KAVACH-RAG/reports/router_golden_eval.json)):**
  - **`149/150` (`0.9933`)** overall accuracy (`FAST_PATH` `6/6`, `HARD_REFUSAL` `8/8`, `SEMANTIC_RAG` `135/136`; only mismatch `TEST_130` predicted `HARD_REFUSAL` at `conf=0.9976` on an `OUT_OF_SCOPE` question labeled `expected_route="SEMANTIC_RAG"`). Mean latency `65.007 ms` (`p50=63.753`, `p95=116.755`).

---

## 7.7 Phase 3 Calibrated End-to-End CPU Latency Baseline ([`reports/phase15_latency_baseline.json`](file:///c:/Users/Nijjil%20Narula/DOC/PROJECTS/KAVACH-RAG/reports/phase15_latency_baseline.json))

Measured on all 150 golden queries under `BenchLock` (`idle_wait_s=4.5, observed_cpu_pct=7.6%`, `cpu_affinity=[0, 2]`, `T=7.0`, `passages_truncated=0`, `queries_truncated=34`, `pairs_scored=1080`):
- **Memory & Guard:** `rss_after_load_mb=2945.28 MB`, `peak_rss_mb=2951.5 MB` (`<= 4096.0 MB`), `rss_after_baseline_mb=1904.83 MB`, `vram_probe={torch_cuda_build: false, cuda_available: false, pid_in_nvidia_smi: false}`.
- **Abstention & Refusal (`150` queries):** `0/140` (`0.00%`) in-scope false abstains/refusals; **`10/10` (`100.00%`)** `OUT_OF_SCOPE` blocked (`9` router `HARD_REFUSAL` + `TEST_129` reranker abstention at `top1_logit=3.666347 < 7.0`).

### Overall & Per-Stage Latency (ms)

| Slice | Count | Min | Mean | P50 | P95 | Max |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **Overall (`total_ms`)** | **150** | **0.009** | **9449.964** | **9231.709** | **24884.009** | **35318.110** |
| Stage: `router_ms` | 150 | 0.005 | 83.673 | 83.312 | 140.673 | 255.692 |
| Stage: `lookup_ms` | 6 | 0.002 | 0.009 | 0.004 | 0.024 | 0.026 |
| Stage: `embed_ms` | 135 | 1217.708 | 3725.897 | 2445.350 | 19522.031 | 24509.727 |
| Stage: `search_ms` | 135 | 0.275 | 0.645 | 0.604 | 1.070 | 6.590 |
| Stage: `rerank_ms` | 135 | 1064.964 | 6680.406 | 7397.743 | 9236.432 | 11303.836 |

### Per-Route & Per-Question-Type Latency (ms)

| Breakdown | Count | Min | Mean | P50 | P95 | Max | Target Status on CPU |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Route: `FAST_PATH_EXACT_LOOKUP` | 6 | 0.009 | **0.088** | 0.021 | 0.306 | 0.381 | **MET** (`< 10 ms`) |
| Route: `HARD_REFUSAL` | 9 | 50.594 | **53.382** | 52.848 | 56.524 | 56.935 | **SLIGHTLY OVER** (~8 ms above 45 ms router target) |
| Route: `SEMANTIC_RAG` | 135 | 2335.415 | **10496.397** | 9960.168 | 26587.675 | 35318.110 | **UNMET** (vs sub-second / 500 ms interactive GPU budget) |
| QType: `FACTUAL` | 60 | 0.009 | 8567.054 | 10366.178 | 11941.185 | 13625.806 | `embed=2263.85`, `rerank=7168.61` |
| QType: `MULTI_HOP` | 45 | 4654.931 | 8880.028 | 8539.819 | 11086.735 | 11643.913 | `embed=2594.75`, `rerank=6199.11` |
| QType: `CONFLICT_TRAP` | 15 | 5632.634 | 7400.375 | 7976.394 | 8583.015 | 8775.604 | `embed=1946.17`, `rerank=5376.94` |
| QType: `OUT_OF_SCOPE` | 10 | 50.594 | 281.586 | 53.283 | 1310.099 | 2335.415 | 9 fast refusals + `TEST_129` full gate (`2335.4 ms`) |
| QType: `MALFORMED` | 10 | 4291.613 | 9123.070 | 9999.452 | 11311.897 | 11394.393 | `embed=2041.01`, `rerank=7002.60` |
| QType: `CONTEXT_NOISE` | 10 | 22708.101 | **29881.785** | 31556.613 | 34522.732 | 35318.110 | `embed=21316.41` (380-tok SW BF16), `rerank=8404.47` |

---

## 7.8 Honest Deviations, Rejected Precisions & Unmet Targets

1. **User-Approved Option A (Decision D2):** `rerank_max_length` increased `576 -> 640`, `_PASSAGE_MAX_TOKENS` raised `512 -> 600`, and Contingency L long-query cap updated `768 -> 1024`. Reason: `78/505` frozen corpus chunks exceed `512` tokens under ModernBERT BPE (max `582`). This guaranteed `passages_truncated == 0` across every run without re-chunking the frozen corpus; `fp32@640` cleared strict `top3_overlap >= 0.98`, so Contingency L did not trigger (`contingency_l_applied=false`).
2. **Dynamic `onednn` INT8 Rejected on All 3 Neural Tiers:** PyTorch `2.15.0+cpu` exposes only `'onednn'` (`fbgemm` absent). Dynamic Linear INT8 degraded F2LLM embedder cosine to `0.053`, ModernBERT reranker `top3_overlap` to `0.68`, and DeBERTa router held-out accuracy by `-22.22 pp`. All three were rejected by automated quality gates rather than loosening thresholds.
3. **CPU `bf16` Embedder Throughput vs Quality & Memory:** Intel i7-13620H lacks AVX-512 BF16 instructions, so `bf16` emulates in software (~`3.73 s` mean embedder vs `21.32 s` on 380-token `CONTEXT_NOISE`). However, `bf16` was the **only** embedder precision to pass `sq8_top8_overlap_mean >= 0.98` (`0.9858` vs `0.9733` on `fp32`), matching the CUDA-`bf16` SQ8 index while cutting embedder RSS by `~1.07 GB` (`1696 MB` vs `2765 MB`), keeping 3-model resident RSS at `2953.3 MB <= 4096.0 MB` with zero downgrades.
4. **CPU Latency Targets Honestly Unmet for `SEMANTIC_RAG`:** Pure CPU execution of a 0.6B decoder embedder (BF16 SW emulation) plus `fp32@640` 150M cross-encoder across 8 pairs yields `10.50 s` mean / `9.96 s` P50 (`SEMANTIC_RAG`) and `53.38 ms` (`HARD_REFUSAL`). Interactive `< 500 ms` semantic latency is **not met on CPU alone** and requires dedicated GPU/NPU acceleration in production deployment; `FAST_PATH_EXACT_LOOKUP` meets `< 1 ms` (`0.088 ms` mean).
5. **Standalone vs Live Pipeline Normalization Difference (`CONTEXT_NOISE`):** Standalone un-normalized `ablation.py` Runs 5–6 logged `3/140` false abstains (`TEST_148..150` at `6.597..6.895 < 7.0`), whereas live `VectorRetrievalPipeline.retrieve()` (which applies `normalize_query()` in Tier 1) achieved `0/140` (`0.00%`) across both Stage 5 and Stage 6.
6. **OS Power Plan:** Host ran on Windows `Balanced` AC power (`79%`); observed and recorded faithfully without mutating OS power settings.

---

## 7.9 Exact End-to-End Reproduction Command (PowerShell)

Execute from the repository root (`c:\Users\Nijjil Narula\DOC\PROJECTS\KAVACH-RAG`):

```powershell
.\scripts\run_phase15.ps1
```

[`scripts/run_phase15.ps1`](file:///c:/Users/Nijjil%20Narula/DOC/PROJECTS/KAVACH-RAG/scripts/run_phase15.ps1) verifies `.venv_gpu` is absent, enforces `PYTHONUTF8=1`, `CUDA_VISIBLE_DEVICES=""`, `OMP_NUM_THREADS=2`, `MKL_NUM_THREADS=2`, `OPENBLAS_NUM_THREADS=2`, `TOKENIZERS_PARALLELISM=false`, and `HF_HOME=<repo>\models\hf_cache`, then executes Stages 0–7 (`devsets --validate`, `profiling --audit`, `--affinity`, `--matrix`, `--gates`, `calibration --collect --select --report`, `--baseline`, `ablation --fresh`, and `pytest -q`) sequentially under `BenchLock`.
