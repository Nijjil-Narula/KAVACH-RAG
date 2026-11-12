# KAVACH-RAG Phase 2: Generation & Grounded Synthesis — Completion Report

## 1. Executive Sign-Off Verdict

**Verdict:** `PHASE2_ENGINEERING_PASSED_QUALITY_UNMET`

- **Phase 2 Generation Engineering Gates (`E1–E8`) & Live Generation Tests:** All **PASS**. The quantized `kavach-tier5` model (`Q4_K_M`, `qwen2.5:7b-instruct-q4_K_M`) loaded `100%` onto GPU (`1.0000` VRAM offload ratio, `num_gpu 999`, `num_ctx 4096`, `repeat_penalty 1.0`), stayed below the `5478 MiB` VRAM hard limit (`5343.0 MiB` batch peak), completed `150/150` frozen golden queries (`140` served + `10` zero-LLM sub-ms bypasses) with `0` generation errors, preserved strict CPU-only PyTorch isolation, passed `26/26` tests in `tests/test_generation.py` (`KAVACH_LIVE=1`), and kept all `39` Phase 1.5 frozen files SHA-256 identical.
- **Full Regression Gate (`E8`):** After verifying Git LFS hydration of `models/router_deberta/model.safetensors` (`283,807,294` bytes), `.\.venv\Scripts\python.exe -m pytest -q -rs` (`KAVACH_LIVE=1`) passed **`134 passed, 0 failed, 0 skipped, 1 warning in 331.19s`** (`exit code 0` — all **`108/108` Phase 1.5 tests** and all **`26/26` Phase 2 tests**).
- **Quality Gates (`Q1–Q3`, Cross-Family `GeminiJudge`):**
  - **Q1 (Faithfulness Mean >= 0.85):** **`0.8628`** (`0.862780`) — **PASS**
  - **Q2 (Relevance Mean >= 0.80):** **`0.5679`** (`0.567857`) — **FAIL** (driven primarily by `45/48` `retrieval-limited` false refusals where Phase 1.5 retrieved insufficient context on `MULTI_HOP` (`30`) and `CONTEXT_NOISE` (`8`); on the `n=72` `context_sufficient=True` subset, relevance mean is **`0.9236`**)
  - **Q3 (Citation Validity Mean >= 0.95):** **`1.0000`** (`0` hallucinated citations) — **PASS**
  - **Judge Backend:** Evaluated across all `140` served items with `GeminiJudge` (`gemini-3.5-flash-lite`, `cross_family=True`, `thinking_applied=True`, `gemini_api_key_present=True`, paced at `15 RPM`), **`0` Stage 2 parse/API errors**, and `10` zero-LLM bypassed refusals.

---

## 2. Engineering Gates (`E1–E8`)

| Gate | Requirement | Observed Measurement | Source | Status |
|---|---|---|---|---|
| **E1** | Phase 1.5 frozen SHA-256 & Amendment 1 strictly append-only diff | `39/39` frozen files intact (`0` mismatches, `0` missing); `kavach/exceptions.py` `prefix_sha256_match=true` (+5 classes); `tests/test_no_placeholders.py` `removed_count=0`, `exact_two_path_allowlist_match=true` | `preflight --sha256-only`, `git diff` | **PASS** |
| **E2** | Target quantization lock | `Q4_K_M` (`kavach-tier5` from `qwen2.5:7b-instruct-q4_K_M`, digest `0a5dc38a6caac1e2e4941bc87d7f0f6f8ed710e62a6ed8ad24f60a6d5c73aa4e`) | `reports/phase2_preflight.json` | **PASS** |
| **E3** | Full GPU offload (`>= 99%` VRAM) | `vram_offload_ratio = 1.0000` (`4755397016 / 4755397016` bytes, `num_gpu 999`, `full_gpu_offload_ok=true`) | `reports/phase2_preflight.json` | **PASS** |
| **E4** | Baseline & peak GPU VRAM within hard limit | Initial baseline `528 MiB` (`<= 700 MiB`); loaded idle `5156 MiB`; batch peak `5343.0 MiB` (`<= 5478 MiB` hard limit; missed `4710 MiB` soft warning target) | `reports/phase2_preflight.json`, `reports/phase2_generation_run.json` | **PASS** *(hard limit)* / **SOFT TARGET MISSED** |
| **E5** | CPU-only Python isolation & zero secret leakage | `torch==2.15.0+cpu`, `torch.version.cuda is None`, `cuda_available=False`, `.venv_gpu` absent, `0` forbidden `kavach.retrieval` imports, `0` Python CUDA processes in `nvidia-smi`, `0` secret matches in outputs (`gemini_api_key_present=True` boolean only) | Stage 6 probe, `nvidia-smi`, secret scan | **PASS** |
| **E6** | Context window, decoding & prompt budget contract | `num_ctx=4096`, `num_predict=512`, `temperature=0`, `top_p=1`, `repeat_penalty=1.0`, `seed=0`, `prompt_version=tier5-v1-965a46c8fae9`; `140` serveable checked with `0` `PromptBudgetError` (`11` chunks dropped cleanly) | `reports/phase2_prompt_freeze.json`, `reports/phase2_preflight.json` | **PASS** |
| **E7** | Sub-ms refusal bypass (`< 1000 us`, `0` LLM calls) & `150/150` batch completion | `10` bypassed (`9` `HARD_REFUSAL`, `1` `LOW_RETRIEVAL_CONFIDENCE`), max `3.600 us`, mean `1.350 us`, `generator_calls=140 == served_items`, `150/150` succeeded, `0` errors | `reports/phase2_generation_run.json` | **PASS** |
| **E8** | Full regression & live generation pytest suite (`KAVACH_LIVE=1`) | **`134 passed, 0 failed, 0 skipped, 1 warning in 331.19s`** (`108/108` Phase 1.5 tests + `26/26` Phase 2 tests) after Git LFS hydration of `models/router_deberta/model.safetensors` (`283,807,294` bytes) | `pytest -q -rs` (`KAVACH_LIVE=1`), `git lfs pull` | **PASS** |

---

## 3. Quality Gates (`Q1–Q3`) & Cross-Family Automated Evaluation

Evaluated across `150` frozen golden outputs (`140` served via `kavach-tier5`, `10` bypassed via canonical refusal templates). All Stage 2 LLM verdicts were produced by `GeminiJudge` (`gemini-3.5-flash-lite`, `cross_family=True`, `thinking_applied=True`, `15 RPM` pacer, `0` errors). *(Note: An earlier fallback run using `OllamaJudge` with `cross_family=False` had scored Q1=`0.9513` / Q2=`0.5515` / `Q2_ctx_suff=0.8768`; cross-family Gemini scoring gives a stricter, unbiased Q1=`0.8628` still passing Q1 while raising `Q2_ctx_suff` to `0.9236`.)*

| Gate | Metric | Observed | Threshold | Status | Notes |
|---|---|---|---|---|---|
| **Q1** | Faithfulness Mean | **`0.8628`** | `>= 0.85` | **PASS** | By type: `CONFLICT_TRAP=0.9857`, `FACTUAL=0.9307`, `MALFORMED=0.9167`, `MULTI_HOP=0.5267`, `CONTEXT_NOISE=0.5000` |
| **Q2** | Relevance Mean | **`0.5679`** | `>= 0.80` | **FAIL** | **`0.9236`** on `context_sufficient=True` (`n=72`); `48` false refusals (`45` `retrieval-limited`, `3` `generation-limited`) |
| **Q3** | Citation Validity Mean | **`1.0000`** | `>= 0.95` | **PASS** | `0` items with hallucinated citations; `0` items with ungrounded decimals |

### Stage 1 Deterministic & Literal Fidelity Metrics (`n=140` served)

| Metric | Observed Value |
|---|---:|
| Citation Precision (Chunk) Mean | `0.7092` |
| Citation Recall (Chunk) Mean | `0.8069` |
| Citation Precision (Doc) Mean | `0.8299` |
| Citation Recall (Doc) Mean | `0.8168` |
| Entity Recall Mean | `0.8403` |
| Items with Hallucinated Citations | `0` |
| Items with Ungrounded Decimals | `0` |
| Refusal Correctness (`all 150`) | `102` true / `48` false |
| Model Refusal Count (`served`) | `48` (`3` over-refusals with `context_sufficient=True`: `TEST_131`, `TEST_132`, `TEST_138`) |
| Stage 2 Gemini Judge Errors / Parse Failures | `0` (`0/140`) |

### False Refusal Breakdown (`n=48`) & Failure Diagnostics

| Root Cause Tag | Count | Breakdown by Question Type |
|---|---:|---|
| `retrieval-limited` (`retrieval_miss=True` or `context_sufficient=False`) | **`45`** | `MULTI_HOP: 30`, `CONTEXT_NOISE: 8`, `FACTUAL: 6`, `CONFLICT_TRAP: 1` |
| `generation-limited` (`context_sufficient=True`, model still refused) | **`3`** | `MALFORMED: 3` (`TEST_131`, `TEST_132`, `TEST_138`) |

- **`CONFLICT_TRAP` (`n=15`):** `both_versions_in_context=12`, `both_versions_in_answer=5`, `both_versions_preserved_when_in_context=5` (`context_conditional_rate=0.4167`).
- **`CONTEXT_NOISE` (`TEST_141–TEST_150`, `n=10`):** `echo_ratio` min `0.0`, p50 `0.0`, mean `0.005`, p95 `0.026`, max `0.029`; `2` non-zero items (`TEST_145`: `0.0222`, `TEST_146`: `0.0290`), `injection_leakage_detected=True`.
- **Lowest Faithfulness Generation-Limited Items (`context_sufficient=True`):** `TEST_045` (`FACTUAL`, faith `0.000`, rel `0.50`), `TEST_048` (`FACTUAL`, faith `0.200`, rel `0.50`), `TEST_044` (`FACTUAL`, faith `0.500`, rel `0.50`).

---

## 4. Latency, Throughput, Bypass & GPU VRAM Telemetry

### Served Generation Latency & Decode Throughput (`n=140`)

| Question Type | Count | TTFT Min / P50 / Mean / P95 / Max (ms) | Decode TPS Min / P50 / Mean / P95 / Max | Total Min / P50 / Mean / P95 / Max (ms) |
|---|---:|---|---|---|
| **Overall (Served)** | **140** | `105.31` / `575.33` / **`587.31`** / `820.48` / `5876.35` | `37.09` / `38.84` / **`38.82`** / `40.21` / `40.92` | `1019.74` / `2809.85` / **`3477.39`** / `8684.99` / `14513.13` |
| `FACTUAL` | 60 | `105.31` / `644.49` / `611.22` / `822.42` / `867.35` | `37.09` / `38.14` / `38.23` / `39.23` / `39.45` | `1053.26` / `2104.82` / `2935.57` / `7313.72` / `14513.13` |
| `MULTI_HOP` | 45 | `179.90` / `516.56` / `612.93` / `780.13` / `5876.35` | `38.36` / `38.88` / `38.90` / `39.58` / `39.73` | `2015.59` / `3412.82` / `3874.47` / `7497.44` / `8608.43` |
| `CONFLICT_TRAP` | 15 | `371.32` / `444.10` / `483.17` / `673.98` / `751.71` | `38.06` / `39.83` / `39.67` / `40.58` / `40.59` | `1272.75` / `2086.42` / `5286.59` / `13750.12` / `14206.04` |
| `MALFORMED` | 10 | `118.77` / `469.16` / `492.43` / `802.32` / `812.68` | `39.31` / `39.69` / `39.92` / `40.71` / `40.92` | `1019.74` / `1995.95` / `2106.07` / `3099.46` / `3231.77` |
| `CONTEXT_NOISE` | 10 | `404.14` / `510.84` / `579.63` / `880.84` / `926.21` | `39.08` / `39.77` / `39.70` / `40.08` / `40.20` | `2319.61` / `3383.99` / `3599.01` / `4942.19` / `4995.26` |

- **Server Load & Prefill (`n=140`):** `server_load_ms` mean `3.975 ms` (`p50=3.490`, `p95=6.978`, `max=13.117`); `server_prefill_ms` mean `514.010 ms` (`p50=501.573`, `p95=740.248`, `max=5769.388`).
- **Finish Reasons (`n=140`):** `stop: 137`, `length: 3`.
- **Target Assessment:**
  - **Decode TPS (`35–45 tok/s` target):** **`38.823 tok/s` mean** (`min 37.094`, `max 40.915`) — **PASS**.
  - **TTFT (`<= 300 ms` target):** **`587.31 ms` mean** (`p50=575.33 ms`, `p95=820.48 ms`) — **TARGET MISSED** (dominated by `514.01 ms` mean server prefill over retrieved multi-chunk prompts).
- **Refusal Bypass Latency (`n=10`):** `9` `HARD_REFUSAL` + `1` `LOW_RETRIEVAL_CONFIDENCE`; min `0.800 us`, p50 `1.000 us`, mean `1.350 us`, p95 `2.880 us`, max `3.600 us`; `0` LLM calls.
- **GPU VRAM (`RTX 4050 Laptop GPU, 6141 MiB total`):**
  - Preflight clean desktop baseline: `528 MiB` (`<= 700 MiB` gate)
  - Preflight loaded idle (`kavach-tier5`, `num_ctx 4096`): `5156 MiB` (`vram_offload_ratio=1.0`, `ladder_steps=[]`)
  - Batch run telemetry (`910` watchdog samples): baseline `5261.0 MiB`, loaded `5297.0 MiB`, watchdog mean `5242.52 MiB`, **peak `5343.0 MiB`** (`<= 5478 MiB` hard limit **PASS**; `> 4710 MiB` soft warning target **MISSED**).
- **Arithmetic End-to-End Latency Estimate** *(Methodology Notice: `computed from separate runs, not an end-to-end measurement`)*:
  - **All-150 Mean:** Phase 1.5 Retrieval `9449.964 ms` + Phase 2 Generation `3245.566 ms` = **`12695.530 ms`**
  - **Served / `SEMANTIC_RAG` Mean:** Phase 1.5 `SEMANTIC_RAG` `10496.397 ms` + Phase 2 Served `3477.392 ms` = **`13973.789 ms`**

---

## 5. Stage 3A Prompt Iteration Log (Round 1 / Max 3)

Prompt optimization followed the strict pre-freeze rule (`<= 3` rounds using unit/oracle-context live tests only; `retrieved_contexts_150.json` outputs unseen).

1. **Initial Prompt (`tier5-v1-baab0ec87060`):**
   - `tests/test_generation.py` with `KAVACH_LIVE=1` exited `1` (`25 passed, 1 failed`): `test_live_conflict_owasp_both_versions_and_citations` passed `conflict_check(ans, ctx)["both_versions_in_answer"]` but failed `assert {v1, v2} <= set(P.CITATION_RE.findall(ans))` (`P.CITATION_RE.findall(ans)` returned `set()`).
2. **Round 1 Fix (`tier5-v1-965a46c8fae9`, SHA-256 `965a46c8fae98d634af2e91ad8b60179e65edddb17d6bd3934878e8af89d84fe`):**
   - Updated `format_context()` header from `[CHUNK ID: ...]` to `CHUNK ID: ...` (preventing bracket syntax confusion with inline `[id]` citations) and clarified `SYSTEM_PROMPT` Rules 4 and 5 so `[id]` bracket citations are explicitly emitted on every claim and every line of multi-version answers (`v1.1: ... [id]` and `v2.0: ... [id]`).
3. **Re-Verification & Freeze:**
   - `tests/test_generation.py -rs` (`KAVACH_LIVE=1`): **`26 passed, 0 failed, 0 skipped`** in `34.64s` (`tests/test_no_placeholders.py`: `1 passed in 0.53s`).
   - Prompt frozen via `kavach.generation.batch_generate --freeze` to `reports/phase2_prompt_freeze.json`.

---

## 6. Stage 6 Verification Commands & Audit Outputs

### 6.1 Amendment 1 Diff & Frozen SHA-256 Audit

- **`.\.venv\Scripts\python.exe -m kavach.generation.preflight --sha256-only` (`exit code 0`):**
  - `ok: true`, `frozen_verified_count: 39`, `frozen_mismatches: []`, `frozen_missing: []`
  - `kavach/exceptions.py`: `prefix_sha256_match: true`, `expected_classes_match: true`, appended `[GenerationError, JudgeError, OllamaUnavailableError, PromptBudgetError, VRAMBudgetExceededError]`
  - `tests/test_no_placeholders.py`: `removed_count: 0`, `exact_two_path_allowlist_match: true` (added ONLY `kavach/generation/generator.py` and `kavach/generation/judge.py`)
- **`git diff -- kavach/exceptions.py tests/test_no_placeholders.py` (`exit code 0`):** Verified zero deletions/edits outside the strictly append-only 5 exception classes and 2 sleep-allowlist entries.

### 6.2 Full Sequential Pytest Suite (`KAVACH_LIVE=1`)

- Command: `.\.venv\Scripts\python.exe -m pytest -q -rs` (with `KAVACH_LIVE=1` scoped in `try / finally`, `PYTHONUTF8=1`) after confirming `git lfs pull` (`exit code 0`) and `models/router_deberta/model.safetensors` hydrated at `283,807,294` bytes
- Outcome: **`134 passed, 0 failed, 0 skipped, 1 warning in 331.19s (0:05:31)`** (`exit code 0`)
  - **`26/26` Phase 2 tests (`tests/test_generation.py`) PASSED** (including all `6/6` `@live` tests).
  - **`108/108` Phase 1.5 tests PASSED** (including `tests/test_no_placeholders.py`, all `7/7` `tests/test_residency.py`, `tests/test_pipeline_smoke.py`, and `tests/test_router.py`).
  - *(Earlier prior-to-LFS-hydration run had failed only on those `3` router tests when `models/router_deberta/model.safetensors` was an unhydrated `134`-byte LFS pointer.)*

### 6.3 Runtime Isolation & Secret Hygiene Audit

- **`.venv` Python Isolation Probe:**
  - `torch_version = "2.15.0+cpu"`, `torch_cuda_is_none = true`, `cuda_available = false`, `venv_gpu_exists = false`
  - Importing all 7 `kavach.generation.*` modules (`gen_config`, `prompts`, `generator`, `batch_generate`, `judge`, `evaluate`, `preflight`) added `[]` forbidden retrieval modules (`pipeline`, `embedder`, `reranker`, `router`) to `sys.modules`; only `kavach.retrieval` and `kavach.retrieval.benchlock` were loaded.
- **GPU Process Audit (`nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader`):** Zero Python processes present on GPU; only Ollama's `llama-server.exe` (PID `7540`) and desktop WDDM processes appeared.
- **Secret Hygiene Scan:** Scanned Stage 5 & evaluation artifacts (`data/processed/generation_judgments_150.jsonl`, `reports/generation_evaluation_summary.json`, `reports/generation_evaluation_report.md`, `logs/phase2_evaluate.log`) after User-scoped keys loaded — **`0` secret findings** (`gemini_api_key_present=True`).

---

## 7. Amendment 1–4 Compliance Summary

- **Amendment 1 (Stage 6 E1 Diff Proof):** `kavach/exceptions.py` is strictly append-only (+5 Phase 2 exception classes, `prefix_sha256_match=true`) and `tests/test_no_placeholders.py` added ONLY the two allowlist paths (`kavach/generation/generator.py`, `kavach/generation/judge.py`, `removed_count=0`), proved via `git diff` and `preflight --sha256-only` (all `39` frozen files intact).
- **Amendment 2 (Megaprompt R4 Narrow Logged Allowlist Contingency):** Contingency retained (narrow & logged if `tests/test_no_placeholders.py` ever flagged scripted transport unit tests); not triggered because `tests/test_no_placeholders.py` passed (`1 passed`) with zero extra allowlist entries.
- **Amendment 3 (Stage 3A Prompt Edit Log, <= 3 Rounds, Oracle Contexts Only):** Logged all prompt edits in Section 5 (`1` round of max `3`: `tier5-v1-baab0ec87060` -> `tier5-v1-965a46c8fae9`), using unit tests and `corpus_chunks.parquet` oracle contexts exclusively prior to `--freeze` (per the oracle-only `corpus_chunks.parquet` / `inscope_dev_100.jsonl` rule).
- **Amendment 4 (No Gemini Key Written + Stage 2 Stop Gate):** `0` secret findings across all generated reports, judgments JSONL, and logs even when User-scoped `GEMINI_API_KEY` and `KAVACH_GEN_GEMINI_API_KEY` were loaded (`gemini_api_key_present=True` boolean only); Stage 2 stop gate enforced and halted fail-closed on initial preflight when desktop GPU baseline exceeded `700 MiB` (`STOP_BASELINE_VRAM_HIGH`: `1326 MiB`, exit `2`) until host desktop VRAM cleared to `528 MiB` (`exit 0`) without modifying limits or auto-pulling models.

---

## 8. File Manifest (Phase 2 Created & Modified Artifacts)

| Category | Path | Role / Change |
|---|---|---|
| Modified (Amendment 1) | `kavach/exceptions.py` | Append-only +5 Phase 2 exceptions |
| Modified (Amendment 1) | `tests/test_no_placeholders.py` | +2 `_ALLOWED_SLEEP_PATHS` (`generator.py`, `judge.py`) |
| Created (Generation Package) | `kavach/generation/__init__.py`, `gen_config.py`, `prompts.py`, `generator.py`, `batch_generate.py`, `judge.py`, `evaluate.py`, `preflight.py` | Tier 5 generator, prompt planner/budgets, Ollama/Gemini judges, preflight & evaluation pipeline |
| Created (Model & Scripts) | `models/Modelfile.tier5`, `scripts/start_ollama.ps1`, `scripts/run_phase2.ps1` | Ollama `kavach-tier5` Modelfile (`num_gpu 999`, `num_ctx 4096`) and PowerShell runners |
| Created (Tests) | `tests/test_generation.py` | `26` unit & `@live` Phase 2 tests |
| Created (Data Outputs) | `data/processed/generation_results_150.jsonl`, `data/processed/generation_judgments_150.jsonl` | `150` golden synthesis outputs & Stage 1+2 judgments |
| Created (Reports & Logs) | `reports/phase2_regression_baseline.json`, `reports/phase2_preflight.json`, `reports/phase2_prompt_freeze.json`, `reports/phase2_generation_run.json`, `reports/generation_evaluation_summary.json`, `reports/generation_evaluation_report.md`, `reports/phase2_completion_report.md`, `logs/phase2_batch.log`, `logs/phase2_evaluate.log` | Stage 0–6 verification records, metrics summaries, and logs |
