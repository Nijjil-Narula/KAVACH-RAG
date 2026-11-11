# KAVACH-RAG Phase 1 Completion Report

**Project:** KAVACH-RAG (Cybersecurity Retrieval-Augmented Generation)  
**Scope:** Phase 1 — Offline Corpus Acquisition, Token-Window Chunking, 150-Item Golden Benchmark, Tier 1 Intent Router, Tier 2/3 Vector Store, Tier 4 Cross-Encoder Reranker, 6-Run Ablation Study, Smoke Test & Final Test Gate  
**Exclusions Enforced:** Tier 5 (Ollama / LLM Generation), FastAPI Service Layer, Tier 6 (Guardrails), and CUDA production execution remain untouched and excluded.  
**Report Basis:** Every metric, latency figure, memory reading, file path, and test count below is taken exclusively from verified workspace artifacts and direct runtime outputs observed in `.venv`.

---

## 1. Executive Summary & Final Gate Verdict

Phase 1 of KAVACH-RAG is **COMPLETE** and **VERIFIED**. All 10 domain components and evaluation deliverables are built on disk, `requirements.lock.txt` matches `.venv` `pip freeze` with **0 drift**, the 9-step `scripts/run_phase1.ps1` orchestrator parses cleanly (`0` PowerShell syntax errors), `.venv_gpu` has been removed (`Test-Path .venv_gpu == False`), and the test suite achieves **100% pass rate (85 passed, 0 failed, 0 skipped)** under strict 2-thread CPU isolation. **Latency Gate Callout:** While `FAST_PATH_EXACT_LOOKUP` (`0.066 ms`) and bare Tier 3 FAISS SQ8 search (`0.313–3.562 ms`) meet the **`< 45 ms`** per-query latency target, full neural CPU execution **MISSES** `< 45 ms` (`38.96s–83.59s` per semantic/rerank query in `test_pipeline_smoke.py`; `28.27s` and `93.32s` CPU rerank on `TEST_001` and `TEST_002`).

| Phase 1 Gate Criterion | Target / Constraint | Observed Result | Status |
| --- | --- | --- | --- |
| Document-level MRR (In-Scope, Run 5 `k=3` / Run 6 `k=5`) | `>= 0.85` | **0.8762** (Run 5) / **0.8875** (Run 6) | **MET** |
| Chunk-level MRR (In-Scope, Run 5 `k=3` / Run 6 `k=5`) | Reported transparently | **0.6821** (Run 5) / **0.6946** (Run 6) | **OBSERVED (< 0.85)** |
| `FACTUAL` Subset Precision/Ranking (Run 5 `k=3`) | High factual precision | Doc MRR **0.9833**, Doc Recall@k **0.9833**, Chunk MRR **0.8083** | **STRONG** |
| Tier 1 Router Accuracy (`golden_150.jsonl`) | High routing accuracy | **99.33% (149/150)**, `mean=606.189 ms`, `p95=1418.899 ms` | **MET** |
| Tier 1 Router Held-Out Eval (`router_eval.json`) | Disjoint train/eval split | **100.0% (153/153)**; 24 substring overlaps purged via `--verify-disjoint` | **MET** |
| Per-Query CPU Latency (`< 45 ms` Target) | `< 45 ms` per query | **MISSED** on semantic/rerank queries (`38.96s–83.59s` in smoke test; `28.27s`/`93.32s` CPU rerank in `TEST_001`/`TEST_002`); **MET** on `FAST_PATH_EXACT_LOOKUP` (`0.066 ms`) & bare FAISS SQ8 search (`< 4 ms`) | **MISSED (NEURAL) / MET (FAST-PATH & FAISS)** |
| Production Runtime RSS Ceiling | `<= 1.7 GB` (`1740.8 MiB`) | Smoke RSS **736.1 MB** (load) / **706.2 MB** (post-smoke); `build_index` Batch 1 CPU hit **1901.9 MB** (disclosed) | **MET IN PROD PIPELINE (1 OFFLINE BATCH BREACH)** |
| Production VRAM Footprint (`.venv` CPU runs) | `0 MB VRAM` | `torch_cuda_build=False`, `cuda_available=False`, `pid_in_nvidia_smi=False` | **MET (WITH 2 APPROVED BATCH GPU OVERRIDES DISCLOSED)** |
| Automated Test Suite & Placeholder Gate (`pytest -q`) | Zero failures / zero placeholders | **85 passed, 0 failed, 0 skipped** (`test_no_placeholders.py` included; `187.72s` in pre-polish gate run) | **PASSED** |

---

## 2. Corpus Acquisition, Ingestion & Golden Benchmark Architecture

### 2.1 Raw Corpus Acquisition (`kavach/retrieval/acquire.py`, `logs/acquire.log`)
All 90 curated cybersecurity source documents were fetched cleanly with cryptographic provenance logged in `data/raw/PROVENANCE.json` (`reports/ingestion_manifest.json`):
- **NVD CVEs:** **49 / 49** curated CVE advisories acquired.
- **MITRE ATT&CK STIX 2.1 (`attack_v20.1`):** **21** technique documents acquired (**12** target + parent techniques, **9** hard distractors).
- **OWASP LLM Top 10 (`v1.1` & `v2.0`):** **10 / 10** `v1.1` + **10 / 10** `v2.0` (**20** documents total, 100% `github_upstream`), establishing controlled version divergence for conflict evaluation.
- **Total Corpus:** **90** curated documents across 3 authoritative cybersecurity sources.

### 2.2 Token-Window Chunking & Deduplication (`kavach/retrieval/ingest.py`, `reports/ingestion_manifest.json`)
Two deterministic token-windowed chunk corpora were produced and verified with zero chunks exceeding window limits:
- **Primary Corpus (`data/processed/corpus_chunks.parquet`, `w=512, overlap=64`):**
  - Total chunks: **505** across **90** documents (`CVE: 268`, `MITRE: 114`, `OWASP: 123`)
  - Token statistics: `avg_tokens = 326.26`, `max_tokens = 512`, `chunks_over_window = 0`
- **Short-Window Baseline (`data/processed/corpus_chunks_w256.parquet`, `w=256, overlap=32`):**
  - Total chunks: **1,028** across **90** documents (`CVE: 688`, `MITRE: 136`, `OWASP: 204`)
  - Token statistics: `avg_tokens = 183.9`, `max_tokens = 256`, `chunks_over_window = 0`
- **Content Deduplication:** Across all **1,533** combined chunks (`505 + 1,028`), SHA-256 passage deduplication identified **1,401** unique text passages, eliminating **132** redundant bi-encoder forward passes during index build.

### 2.3 Golden Benchmark Dataset (`kavach/retrieval/golden.py`, `data/golden/golden_150.jsonl`)
A validated 150-query benchmark (`golden_150.jsonl`, paired with `data/processed/retrieved_contexts_150.json`) covers 6 evaluation regimes and all 3 routing paths:
- **By Question Type:** `FACTUAL` (**60**), `MULTI_HOP` (**45**), `CONFLICT_TRAP` (**15**), `OUT_OF_SCOPE` (**10**), `MALFORMED` (**10**), `CONTEXT_NOISE` (**10**).
- **By Expected Route:** `SEMANTIC_RAG` (**136**), `HARD_REFUSAL` (**8**), `FAST_PATH_EXACT_LOOKUP` (**6**).

---

## 3. Model Stack, Embeddings, Vector Store & Intent Router (Tiers 1–4)

### 3.1 Model Selection & Preflight Benchmarks (`reports/preflight.json`, `models/embedder_notes.md`)
All three production transformer models loaded natively without fallback (`embed_fallback=false`, `reranker_is_fallback=false`) under 2-thread CPU preflight profiling (peak preflight RSS **1,566.7 MB**, `vram_probe` all `False`):

| Tier | Component & Model ID | Precision / Architecture | CPU Throughput | Mean Single-Item Latency | Preflight RSS |
| --- | --- | --- | --- | --- | --- |
| Tier 1 | `microsoft/deberta-v3-small` (`AgentIntentRouter`) | 3-class classifier + exact regex/hash fast path | `26.22 tok/s` | chunk `1,499.36 ms` / query `1,812.07 ms` | `750.4 MB` |
| Tier 2 | `codefuse-ai/F2LLM-v2-0.6B` (`Embedder`) | `last_token` pooling (`padding_side="right"`), `1024`-dim + MRL-`512` | `14.97 tok/s` | chunk `5,915.41 ms` / query `1,163.07 ms` | `1,566.7 MB` |
| Tier 4 | `cross-encoder/ettin-reranker-150m-v1` (`Reranker`) | Option A `_ModularCrossEncoder` (`bfloat16` ModernBERT + `float32` CLS/Dense/LN/Dense head, `num_labels=1`) | `81.14 tok/s` | pair `1,145.38 ms` / batch-8 `11,580.84 ms` | `711.2 MB` |

- **Query Instruction Prefix (`models/embedder_notes.md`):** `"Represent this cybersecurity query for retrieving relevant vulnerability and mitigation context: "`.
- **Preflight Full-Corpus CPU Projections:** At `14.97 tok/s`, full CPU embedding of `w512` and `w256` was projected at `35,070.1s` and `33,400.1s` respectively.

### 3.2 Tier 1 Router Training & Evaluation (`reports/router_eval.json`, `reports/router_golden_eval.json`)
- **Disjoint Verification (`--verify-disjoint` in `kavach/retrieval/train_router.py`):** Checked all synthetic training examples against `data/golden/golden_150.jsonl`. Zero verbatim (`exact_normalised`) overlaps existed; **24** substring overlaps (e.g., overlapping advisory identifiers such as `T1548`, `CVE-2014-6271`, `CVE-2021-4034`, `T1203`) were automatically purged prior to splitting, and an assertion confirmed zero remaining overlap with golden `OUT_OF_SCOPE` queries.
- **Held-Out Synthetic Evaluation (`train_examples=610`, `eval_examples=153`):**
  - Accuracy: **1.0000 (153/153)**; confusion matrix `[[92, 0, 0], [0, 41, 0], [0, 0, 20]]`; `peak_rss_mb = 861.4`.
- **Golden 150 End-to-End Router Evaluation (`router_golden_eval.json`):**
  - Overall Accuracy: **0.9933 (149/150)**; latency `mean = 606.189 ms`, `p95 = 1,418.899 ms`.
  - Confusion Matrix (`[FAST_PATH_EXACT_LOOKUP, SEMANTIC_RAG, HARD_REFUSAL]`): `[[6, 0, 0], [0, 135, 1], [0, 0, 8]]`.
  - Per-Route Precision / Recall:
    - `FAST_PATH_EXACT_LOOKUP`: Precision **1.0000**, Recall **1.0000** (`6/6`)
    - `SEMANTIC_RAG`: Precision **1.0000**, Recall **0.9926** (`135/136`)
    - `HARD_REFUSAL`: Precision **0.8889**, Recall **1.0000** (`8/8` true positives, `9` predicted)
  - Per-Question-Type Accuracy: `FACTUAL` **60/60 (1.0)**, `MULTI_HOP` **45/45 (1.0)**, `CONFLICT_TRAP` **15/15 (1.0)**, `MALFORMED` **10/10 (1.0)**, `CONTEXT_NOISE` **10/10 (1.0)**, `OUT_OF_SCOPE` **9/10 (0.9)**.
  - Sole Misclassification (`TEST_130`): `"What is the password policy of my company's HR portal?"` (`OUT_OF_SCOPE`) — expected `SEMANTIC_RAG`, predicted `HARD_REFUSAL` (`confidence = 0.9976`).

### 3.3 Tier 3 FAISS SQ8 Index (`kavach/retrieval/build_index.py`, `kavach/retrieval/vector_store.py`)
- **Production Index (`models/index_sq8_mrl512.faiss`):** FAISS 8-bit scalar quantizer (`SQ8`) over MRL-512 normalized embeddings: `ntotal = 505`, `dim = 512`, `is_trained = True`, paired with `models/index_sq8_mrl512.idmap.json` and `ChunkStore` exact identifier index.
- **Precomputed Vector Stores:** `data/processed/vectors_mrl512.parquet` (`505` rows), `w512.npz` (`(505, 1024)`), and `w256.npz` (`(1028, 1024)`).

---

## 4. Retrieval Ablation & Empirical Analysis (Runs 1–6)

All 6 retrieval configurations (`reports/retrieval_ablation_results.json`, `reports/retrieval_ablation_report.md`) were evaluated with Tier 1 routing bypassed so retrieval, quantization, expansion, reranking, and probability-threshold behavior are isolated cleanly across the **140** in-scope queries and **10** `OUT_OF_SCOPE` queries. Note that Run 5/6 stage latencies reflect mixed CPU/CUDA-checkpointed reranking (detailed in Section 6).

### 4.1 Overall Document-Level & Chunk-Level Ablation Results (140 In-Scope Queries)

| Run | Configuration | `k` | Doc R@1 | Doc R@3 | Doc R@k | Doc MRR | Doc nDCG@3 | Doc nDCG@k | Chunk R@1 | Chunk R@3 | Chunk R@k | Chunk P@k | Chunk MRR | Chunk nDCG@k | Mean Lat (ms) | P95 Lat (ms) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Run 1 | `w256 \| FlatIndex(1024) full1024` | 3 | 0.6536 | 0.7071 | 0.7071 | 0.8714 | 0.7316 | 0.7316 | N/A | N/A | N/A | N/A | N/A | N/A | 2,792.52 | 15,437.10 |
| Run 2 | `w512 \| FlatIndex(1024) full1024` | 3 | 0.6393 | 0.7071 | 0.7071 | 0.8524 | 0.7267 | 0.7267 | 0.2439 | 0.4089 | 0.4089 | 0.2024 | 0.4607 | 0.3906 | 2,792.44 | 15,437.05 |
| Run 3 | `w512 \| SQ8Index mrl512` | 3 | 0.6107 | 0.6929 | 0.6929 | 0.8107 | 0.7033 | 0.7033 | 0.2296 | 0.3989 | 0.3989 | 0.1929 | 0.4369 | 0.3756 | 2,792.61 | 15,437.20 |
| Run 4 | `w512 \| SQ8Index mrl512 (expanded)` | 8 | 0.6107 | 0.7071 | 0.7750 | 0.8301 | 0.7119 | 0.7437 | 0.2296 | 0.3989 | 0.5361 | 0.1045 | 0.4751 | 0.4140 | 2,792.61 | 15,437.20 |
| **Run 5** | **`w512 \| SQ8-MRL512 + Reranker`** | **3** | **0.6821** | **0.7286** | **0.7286** | **0.8762** | **0.7592** | **0.7592** | **0.4739** | **0.5096** | **0.5096** | **0.2476** | **0.6821** | **0.5632** | 3,688.75* | 15,705.81* |
| **Run 6** | **`w512 \| SQ8-MRL512 + Reranker (k=5)`** | **5** | **0.6821** | **0.7321** | **0.7607** | **0.8875** | **0.7614** | **0.7761** | **0.4739** | **0.5096** | **0.5289** | **0.1600** | **0.6946** | **0.5445** | 3,688.75* | 15,705.81* |

*Key ablation findings:*
- Compressing from `FlatIndex(1024)` (Run 2) to `SQ8(512)` (Run 3) introduced a `0.0417` Doc MRR and `0.0238` Chunk MRR quantization/Matryoshka drop.
- Expanding candidate pool to `candidate_k=8` (Run 4) recovered Doc Recall@8 to **0.7750** and Chunk Recall@8 to **0.5361**.
- Tier 4 cross-encoder reranking (`Run 5` at `k=3`) more than **doubled Chunk Recall@1** (`0.2296 -> 0.4739`) and boosted **Chunk MRR by +0.2452** (`0.4369 -> 0.6821`) and **Doc MRR by +0.0655** (`0.8107 -> 0.8762`), surpassing the unquantized 1024-dim baseline (`Run 1` `0.8714`, `Run 2` `0.8524`). Increasing output cap to `k=5` (`Run 6`) lifted Doc Recall@k to **0.7607** and Doc MRR to **0.8875**.

### 4.2 Per-Question-Type Breakdown (Run 5 `k=3` & Run 6 `k=5`)

| Question Type | Count | Run 5 Doc R@k | Run 5 Doc MRR | Run 5 Chunk R@k | Run 5 Chunk MRR | Run 6 Doc R@k | Run 6 Doc MRR | Run 6 Chunk R@k | Run 6 Chunk MRR | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `FACTUAL` | 60 | **0.9833** | **0.9833** | 0.8167 | 0.8083 | **0.9833** | **0.9833** | 0.8333 | 0.8117 | Near-ceiling precision & rank |
| `MULTI_HOP` | 45 | 0.4667 | 0.8370 | 0.1478 | 0.6444 | 0.5444 | 0.8611 | 0.1700 | 0.6689 | Gains `+0.0777` Doc R@k at `k=5` |
| `CONFLICT_TRAP` | 15 | 0.7333 | 0.9333 | 0.3667 | 0.4667 | 0.7333 | 0.9333 | 0.4000 | 0.4833 | Dual-version (`v1.1 + v2.0`): **8/15 (0.5333)** in both runs |
| `MALFORMED` | 10 | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | Perfect typo/malformed robustness |
| `CONTEXT_NOISE` | 10 | 0.1000 | 0.2000 | 0.0200 | 0.1000 | 0.2000 | 0.2500 | 0.0400 | 0.1200 | Adversarial distractor pressure |

### 4.3 Reranker Calibration & Probability-Threshold Abstention (`rerank_min_prob = 0.35`)
- **Score Separation across All Candidate-8 Pools (`reports/retrieval_ablation_results.json`):**
  - `all_scored8_in_scope` (`n = 1,120`): `rerank_score` min **0.869643**, median **0.999625**, mean **0.996695**, max **0.999994** (`raw_logit` min **1.89781**, median **7.887037**, mean **7.684167**, max **12.074607**).
  - `all_scored8_out_of_scope` (`n = 80`): `rerank_score` min **0.019109**, median **0.117958**, mean **0.360766**, max **0.996421** (`raw_logit` min **-3.938283**, median **-2.013397**, mean **-0.910111**, max **5.629014**).
- **Thresholded Abstention Outcomes (Runs 5 & 6, Router Disabled):**
  - **False Abstention Rate (140 In-Scope Queries):** **0.0000 (0 / 140)** — every in-scope scored candidate exceeded `0.35`.
  - **OOS Abstention Rate (10 `OUT_OF_SCOPE` Queries):** **0.3000 (3 / 10)** pure Tier 4 abstention; the other 7 OOS queries matched superficial security vocabulary scoring up to `0.996421`, demonstrating why Tier 1 `AgentIntentRouter` (`HARD_REFUSAL` recall `1.0000` on refusal-labeled OOS queries) is required alongside Tier 4 threshold filtering.

---

## 5. Runtime Memory Engineering, End-to-End Smoke Test & Final Test Gate

### 5.1 Ordered 5-Stage RSS Mitigation (`kavach/retrieval/pipeline.py`)
`VectorRetrievalPipeline` enforces `_RSS_CEILING_MB = 1700.0` (strictly below `1.7 * 1024 = 1,740.8 MiB`) via deterministic ordered mitigations in `_enforce_rss_ceiling()`:
1. `bf16_router` (cast DeBERTa router weights to `bfloat16`)
2. `lazy_load_router` (unload router and reload on demand)
3. `free_tokenizer_caches` (release embedder, reranker, and router tokenizer caches)
4. `lazy_load_reranker` (unload cross-encoder until Tier 4 execution)
5. `lazy_load_embedder` (unload bi-encoder until Tier 2 execution, swapping cleanly between Tier 1 -> Tier 2 -> Tier 4 in `retrieve()`)

### 5.2 Live End-to-End CPU Smoke Test (`tests/test_pipeline_smoke.py::test_pipeline_smoke_five_queries`)
Executed directly in `.venv` under strict 2-thread CPU isolation (`pytest -q -s tests/test_pipeline_smoke.py`, passed in **184.70s**):
- **RSS Telemetry:** All 5 mitigations engaged during initialization (`['bf16_router', 'lazy_load_router', 'free_tokenizer_caches', 'lazy_load_reranker', 'lazy_load_embedder']`), reducing process memory to **736.1 MB after load** and **706.2 MB post-smoke** (vs `1,740.8 MB` ceiling).
- **VRAM Telemetry:** `{'torch_cuda_build': False, 'cuda_available': False, 'pid_in_nvidia_smi': False}`.
- **5-Query Single-Query Pure-CPU Latencies (including lazy load/unload transitions):**
  - `exact_cve`: `FAST_PATH_EXACT_LOOKUP` | `refused=False` | `contexts=1` | `total_ms = 0.066` (`router=0.01`, `lookup=0.002`)
  - `multi_hop`: `SEMANTIC_RAG` | `refused=False` | `contexts=3` | `total_ms = 38,964.002` (`router=1,542.654`, `embed=1,591.55`, `search=3.562`, `rerank=33,286.343`)
  - `golden_oos_1`: `HARD_REFUSAL` | `refused=True` | `contexts=0` | `total_ms = 1,649.898` (`router=1,385.041`)
  - `typo_query`: `SEMANTIC_RAG` | `refused=False` | `contexts=3` | `total_ms = 83,591.867` (`router=1,419.755`, `embed=1,317.43`, `search=0.339`, `rerank=77,978.131`)
  - `owasp_llm03`: `SEMANTIC_RAG` | `refused=False` | `contexts=3` | `total_ms = 48,623.963` (`router=1,548.054`, `embed=1,421.419`, `search=0.313`, `rerank=42,310.252`)

### 5.3 Environment Lock, Hardware Probe & Test Suite Verification
- **Dependency Lock (`requirements.lock.txt`):** **76** pinned lines (`torch==2.15.0+cpu`, `transformers==5.22.2`, `faiss-cpu==1.16.0`, `httpx==0.28.1`, `numpy==2.5.4`, `pandas==3.1.1`, `psutil==7.2.2`, `pyarrow==26.0.1`, `pydantic==2.14.3`, `pytest==9.2.0`, `safetensors==0.8.0`, `tokenizers==0.23.3`); `Compare-Object` against `.\.venv\Scripts\python.exe -m pip freeze` confirmed **0** differences.
- **Direct Runtime Hardware Probe (`.venv`):** `torch: 2.15.0+cpu`, `cuda: None`, `avail: False`, `vram: {'torch_cuda_build': False, 'cuda_available': False, 'pid_in_nvidia_smi': False}`, idle `rss_mb: 204.7`. Temporary GPU venv deleted (`Test-Path .venv_gpu == False`).
- **Orchestration Script (`scripts/run_phase1.ps1`):** Aligned with the full 9-step Phase 1 pipeline (`preflight`, `acquire`, `ingest w512`, `ingest w256`, `golden --build` & `golden --validate`, `train_router --verify-disjoint`, `build_index --build-threads 2`, `ablation`, `pytest -q`) with strict UTF-8 and 2-thread CPU environment variables and **0** PowerShell parse errors.
- **Full Automated Test Suite (`.\.venv\Scripts\python.exe -m pytest -q`):** **85 passed, 0 failed, 0 skipped** (1 `FutureWarning` on `torch.jit.script` inside `test_pipeline_smoke.py`; observed in **187.72s** on the pre-polish gate run, and re-verified after this report edit) across all 10 test modules (`test_embedder.py`, `test_golden.py`, `test_ingest.py`, `test_metrics.py`, `test_no_placeholders.py`, `test_pipeline_smoke.py`, `test_reranker.py`, `test_router.py`, `test_runtime_guard.py`, `test_vector_store.py`).

---

## 6. Deviations, Latency Target Miss, Overrides & Honest Engineering Disclosures

1. **`< 45 ms` Per-Query CPU Latency Target MISSED on Neural Paths (`SEMANTIC_RAG` / Reranking):**
   - **Missed on semantic + rerank CPU queries:** In `tests/test_pipeline_smoke.py`, full single-query CPU execution measured **38,964.002 ms** (`multi_hop`), **48,623.963 ms** (`owasp_llm03`), and **83,591.867 ms** (`typo_query`), dominated by Tier 4 cross-encoder reranking (`33.29s–77.98s` per 8 candidates on 2 CPU threads, alongside `1.39s–1.55s` DeBERTa router inference and `1.32s–1.59s` F2LLM query embedding). Native CPU reranking in `ablation.py` likewise measured **28,265.072 ms** (`TEST_001`) and **93,324.384 ms** (`TEST_002`). Pure-CPU `HARD_REFUSAL` (`golden_oos_1`) took **1,649.898 ms** (`router = 1,385.041 ms`).
   - **Met on exact fast path and vector index lookup:** `FAST_PATH_EXACT_LOOKUP` (`exact_cve`) completed in **0.066 ms** (`router = 0.01 ms`, `lookup = 0.002 ms`), and bare Tier 3 FAISS `SQ8Index` search completed in **0.313–3.562 ms** during smoke testing (`0.229 ms` mean `stage_ms` in Ablation Runs 3 & 4).
2. **`torch==2.15.0+cpu` Upgrade from `2.5.1+cpu`:** `transformers==5.22.2` disables PyTorch support when `torch < 2.6`. `requirements.txt` and `requirements.lock.txt` were updated cleanly to the official CPU wheel index (`https://download.pytorch.org/whl/cpu`, `torch==2.15.0+cpu`).
3. **`ettin-reranker-150m-v1` Modular Loader Patch (Option A):** `cross-encoder/ettin-reranker-150m-v1` packages its scoring head across Sentence-Transformers modules (`1_Pooling`, `2_Dense`, `3_LayerNorm`, `4_Dense`) rather than a standard `ForSequenceClassification` config. `_ModularCrossEncoder` in `kavach/retrieval/reranker.py` implements the exact `bfloat16` ModernBERT backbone + `float32` head architecture, preserving full fidelity without fallback (`reranker_is_fallback = false`).
4. **`build_index` Batch-1 CPU RSS Breach (`1,901.9 MB` > `1,700.0 MB`):** During `logs/build_index.log` Batch `1/44` (`batch_00000`, `32/1,401` passages) with `F2LLM-v2-0.6B` on 2 CPU threads, processing took **1,048s** (`0.03 chunks/s`, projected CPU ETA `44,834.8s`) and RSS peaked at **1,901.9 MB**, breaching the `1.7 GB` hardware ceiling on large batch passage encoding.
5. **Explicit User-Approved Temporary `.venv_gpu` CUDA Overrides (Offline Batch Generation Only):**
   - *Corpus Index Embedding (`logs/build_index.log`):* Numerical parity against CPU `batch_00000` was first confirmed (`n=4`: `cos_full min=0.998467, mean=0.998887 | cos_mrl min=0.998348, mean=0.998855`; `n=32`: `cos_full min=0.998403, mean=0.999000 | cos_mrl min=0.998274, mean=0.998962`). Remaining Batches 1–43 (**1,369** passages) executed on `NVIDIA GeForce RTX 4050 Laptop GPU` in **46.38s** (up to `29.52 chunks/s`, `VRAM alloc = 1,145.0 MB`, `peak = 1,585.5 MB`). Final FAISS SQ8 training, quantization, and parquet export returned to `.venv` CPU (`RSS = 354.0 MB`, `VRAM = 0 MB`).
   - *Ablation Reranker Pre-Scoring (`logs/ablation.log`, `reports/retrieval_ablation_results.json`):* Queries `TEST_001` (**28,265.072 ms**) and `TEST_002` (**93,324.384 ms**) executed natively on 2 CPU threads (`cpu_mean_rr_ms = 60,794.728 ms`, `cpu_p95_rr_ms = 90,071.418 ms`). After confirming numerical parity, the remaining **148** ablation queries were pre-scored via `cuda:rtx4050_bf16` (`gpu_override_mean_rr_ms = 86.702 ms`, `gpu_override_p95_rr_ms = 115.329 ms`) into `data/processed/ablation_rerank_ckpt.jsonl`. Metric compilation, golden routing evaluation, and all reports/tests finished strictly on CPU (`final RSS = 467.3 MB`, `VRAM = 0 MB`). Ablation Run 5/6 latency averages (`mean = 3,688.748 ms`) therefore reflect mixed cached execution and must not be interpreted as pure 2-thread CPU query latency.
   - Temporary `.venv_gpu` was deleted after batch generation (`Test-Path .venv_gpu == False`), leaving the workspace strictly CPU-bound in `.venv`.
6. **Phase 2 Readiness & Open Engineering Focus Areas:** All Phase 1 data artifacts, indices, models, and automated tests are locked and ready for Phase 2 integration. Key follow-up opportunities for subsequent phases are: (a) reducing pure-CPU Tier 4 reranker latency (`33–78s` per query for 8 chunks) via `candidate_k` tuning or quantized ONNX/OpenVINO export, and (b) strengthening distractor resistance on `CONTEXT_NOISE` (`Run 5` Doc R@k `0.1000`, `Run 6` `0.2000`) and dual-version coverage on `CONFLICT_TRAP` (`53.3%`).

---

## 7. Project File Tree (Verified Post-Cleanup)

Verified directly via `Get-ChildItem` across the workspace (`Test-Path .venv_gpu == False`):

```text
KAVACH-RAG/
├── .gitignore
├── requirements.txt
├── requirements.lock.txt                      # 76 pinned lines, 0 drift vs .venv pip freeze
├── .venv/                                     # CPU-only Python 3.12.6 + torch==2.15.0+cpu (no .venv_gpu)
├── app/                                       # Deferred to Phase 2+
│   ├── .gitkeep
│   └── __init__.py
├── kavach/
│   ├── .gitkeep
│   ├── __init__.py
│   ├── config.py
│   ├── exceptions.py
│   ├── runtime_guard.py
│   └── retrieval/
│       ├── .gitkeep
│       ├── __init__.py
│       ├── ablation.py
│       ├── acquire.py
│       ├── build_index.py
│       ├── embedder.py
│       ├── golden.py
│       ├── ingest.py
│       ├── metrics.py
│       ├── pipeline.py
│       ├── preflight.py
│       ├── reranker.py
│       ├── router.py
│       ├── schema.py
│       ├── train_router.py
│       └── vector_store.py
├── scripts/
│   ├── .gitkeep
│   └── run_phase1.ps1                         # 9-step deterministic Phase 1 orchestrator
├── tests/
│   ├── .gitkeep
│   ├── test_embedder.py
│   ├── test_golden.py
│   ├── test_ingest.py
│   ├── test_metrics.py
│   ├── test_no_placeholders.py
│   ├── test_pipeline_smoke.py
│   ├── test_reranker.py
│   ├── test_router.py
│   ├── test_runtime_guard.py
│   └── test_vector_store.py
├── data/
│   ├── golden/
│   │   ├── .gitkeep
│   │   ├── cve_attack_map.yaml
│   │   └── golden_150.jsonl                   # 150 validated benchmark items
│   ├── processed/
│   │   ├── .gitkeep
│   │   ├── ablation_qvecs_ckpt.npz
│   │   ├── ablation_rerank_ckpt.jsonl
│   │   ├── corpus_chunks.parquet              # 505 chunks (w=512, overlap=64)
│   │   ├── corpus_chunks_w256.parquet         # 1,028 chunks (w=256, overlap=32)
│   │   ├── retrieved_contexts_150.json
│   │   ├── vectors_mrl512.parquet             # 505 rows x 512-dim MRL embeddings
│   │   └── embeddings/
│   │       ├── .gitkeep
│   │       ├── w256.npz
│   │       ├── w512.npz
│   │       └── checkpoint/                    # 45 files (.gitkeep + batch_00000.npz .. batch_00043.npz)
│   └── raw/                                   # 95 files total
│       ├── PROVENANCE.json
│       ├── cve_json/                          # 50 files
│       ├── mitre_markdown/                    # 22 files
│       └── owasp_markdown/                    # 22 files
├── models/
│   ├── .gitkeep
│   ├── embedder_notes.md
│   ├── index_sq8_mrl512.faiss                 # ntotal=505, dim=512, is_trained=True
│   ├── index_sq8_mrl512.idmap.json
│   ├── hf_cache/                              # 110 cached model files
│   └── router_deberta/                        # 4 trained router checkpoint files
├── logs/
│   ├── .gitkeep
│   ├── ablation.log
│   ├── acquire.log
│   ├── acquire_err.log
│   ├── build_index.log
│   ├── build_index_stderr.log
│   └── build_index_stdout.log
└── reports/
    ├── .gitkeep
    ├── ingestion_manifest.json
    ├── phase1_completion_report.md
    ├── preflight.json
    ├── retrieval_ablation_report.md
    ├── retrieval_ablation_results.json
    ├── router_eval.json
    └── router_golden_eval.json
```

---

## 8. Reproducibility & Exact PowerShell Re-Run Commands

From the repository root (`c:\Users\Nijjil Narula\DOC\PROJECTS\KAVACH-RAG`) in PowerShell:

### Full 9-Step Phase 1 Pipeline Run
```powershell
.\scripts\run_phase1.ps1
```

### Re-Run Skipping Model Preflight & Upstream Network Acquisition
```powershell
.\scripts\run_phase1.ps1 -SkipPreflight -SkipAcquire
```

### Strict CPU/UTF-8 Isolated Test Suite Verification Only
```powershell
$env:PYTHONUTF8 = "1"
$env:CUDA_VISIBLE_DEVICES = ""
$env:OMP_NUM_THREADS = "2"
$env:MKL_NUM_THREADS = "2"
$env:OPENBLAS_NUM_THREADS = "2"
$env:TOKENIZERS_PARALLELISM = "false"
.\.venv\Scripts\python.exe -m pytest -q
```

