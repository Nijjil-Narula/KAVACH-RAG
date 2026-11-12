# KAVACH-RAG Phase 3 Completion Report

**Generated:** 2026-11-12T03:48:00Z (`2026-11-12 09:18 IST`)  
**Repository:** `c:\Users\Nijjil Narula\DOC\PROJECTS\KAVACH-RAG`  
**Phase Status / Sign-Off Verdict:** **`PHASE3_FAILED`** *(Hardware / Shared-Host GPU VRAM Contention on Plan Gate E6, which also fails Plan E1's live regression subset; Plan E2-E5, E7, E8 and all 4 Plan Performance Targets P1-P4 PASSED)*

---

## 1. Executive Summary

Phase 3 delivered the FastAPI serving layer (`kavach/serving/`), Windows lifecycle tooling (`scripts/start_server.ps1`), headless Locust load-test harness (`benchmarks/locustfile.py`, `benchmarks/compile_report.py`, `scripts/run_locust.ps1`), serving unit/chaos test suites (`tests/serving_fakes.py`, `tests/test_serving.py`, `tests/test_fault_injection.py`), Stage 3 Server Preflight validation (`reports/phase3_server_preflight.json`), and a 12-cell Locust benchmark campaign (`reports/phase3_load_test_summary.json`, `reports/phase3_load_test_report.md`).

Under **Plan Section 5's strict non-waiver rule**, the overall Phase 3 sign-off is **`PHASE3_FAILED`** because live GPU inference on the host **NVIDIA GeForce RTX 4050 Laptop GPU (`6,141 MiB` total VRAM)** breached the **`5,478 MiB` hard VRAM limit (Gate E6)** when desktop display processes (`Antigravity.exe`, `chrome.exe`/`msedgewebview2.exe`, `WhatsApp.Root.exe`, `explorer.exe`, `LockApp.exe`, etc.) consumed **`~486-941 MiB`** alongside `kavach-tier5:latest` (`4,755,397,016` bytes = `4,535.1 MiB` `size_vram`):

1. **Stage 4 Load Test (`E6` / compiler `E5`):** Peak GPU VRAM reached **`5,508.0 MiB`** (`+30.0 MiB` above `5,478 MiB`), with `vram_breached=true` in **`6/12`** cells (`sync_cold_u10`, `sync_cold_u50`, `async_cold_u10`, `async_cold_u100`, `async_warm_u1`, `async_warm_u100`) and **`3`** `VRAMBudgetExceededError` HTTP `502` aborts in `sync_cold_u10`.
2. **Stage 5 Final Sequential Pytest (`KAVACH_LIVE=1`):** **`152 passed, 6 failed, 0 skipped`** (`1` `FutureWarning` from `torch.jit.script`) in **`338.14s (0:05:38)`**, `exit_code=1`. All **`152`** non-live-VRAM tests (including all **`24/24`** Phase 3 serving and fault-injection tests, retrieval, pipeline smoke, e2e golden, calibration, unit generation tests, and `test_no_placeholders.py`) passed; all **`6`** failures were the `@live` tests in `tests/test_generation.py` raising `kavach.exceptions.VRAMBudgetExceededError: GPU VRAM exceeded hard limit 5478 MiB.` (sampled `5,188 -> 5,582 MiB`) when baseline pre-Ollama desktop GPU usage was `941 MiB`.

All software engineering deliverables, isolation invariants, async event-loop liveness, single-flight semantic cache scaling (**`177.6729` RAG RPS** / **`213.7675` aggregated RPS** at 100 users vs **`0.1916` RPS** cold async -> **`13,233.33x`** non-stream P50 latency speedup from `29,775.0 ms` to `2.25 ms`), refusal bypass integrity (**`0` LLM calls** across `17,335` cache hits and OOS refusals), clean lifecycle hygiene, and guardrail enforcement (no GPU OOM crashes) succeeded.

---

## 2. Owned Artifacts & Integrity Audit

### 2.1 Frozen Code & Environment Isolation
- **Python Runtime:** `3.12.6` (`.\.venv\Scripts\python.exe`, CPU-only isolation: `torch==2.15.0+cpu`, `torch.version.cuda is None`, `torch.cuda.is_available() == False`, `.venv_gpu` absent).
- **Serving Package Delta:** `76 -> 105` packages (`+29` in `requirements.serving.lock.txt`, including `fastapi==0.162.5`, `uvicorn==0.54.0`, `locust==2.47.0`); **`0`** existing packages modified; all **`12`** protected packages unchanged; `requirements.txt` and `requirements.lock.txt` untouched. Stage 5 `pip freeze` matched Stage 0 (`PIP_FREEZE_MATCH_STAGE0=True (105 packages)`).
- **Stage 0 Manifest Integrity (`reports/phase3_regression_baseline.json`):**
  - `153/153` frozen repository files intact (`MISSING=0`, `MODIFIED=0`).
  - `5/5` `models/router_deberta/*` files intact (`MISSING=0`, `MODIFIED=0`), including `model.safetensors` = `283,807,294` bytes (`270.66 MB`, SHA-256 `10e77bb03c15c2d14161e49b11a8c673c89be01825b0cd25a44`).
- **Placeholder & Secret Hygiene:**
  - `pytest -q tests/test_no_placeholders.py` -> **`1 passed in 0.58s`** (`exit_code=0`).
  - Secret scan across `167` files (`kavach/`, `benchmarks/`, `scripts/`, `tests/`, `reports/`) -> **`FINDINGS=[]`**.
- **Clean Server / GPU Post-State:** `logs/.server.pid` absent (`PID_EXISTS=False`), TCP port `8000` unbound (`CONN_8000_COUNT=0`), `0` Python/uvicorn/locust GPU compute processes, `nvidia-smi` post-test idle GPU usage `486 MiB / 6,141 MiB`.

### 2.2 Phase 3 Files Inventory
- **Serving Core (`kavach/serving/`):**
  - `kavach/serving/__init__.py`
  - `kavach/serving/schemas.py` (`QueryRequest`, `QueryResponse`, `IngestionTaskRequest`, `TaskStatusResponse`, `HealthResponse`, Pydantic v2 `extra="forbid"`)
  - `kavach/serving/resilience.py` (`CircuitBreaker(5, 30.0s)`, `MutexTTLCache(ttl=900.0s, max=512)`, `VRAMSampler(interval=1.0s)`, `TaskRegistry(max=1000)`, `ServiceDegraded`, `CircuitOpenError`)
  - `kavach/serving/app.py` (`/health`, `/rag/sync`, `/rag/async`, `/tasks/ingest`, `/tasks/{job_id}`, `GPU_CONCURRENCY=20`, `RETRIEVAL_CONCURRENCY=1`, `CACHE_LOCK_TIMEOUT_S=150.0`)
  - `kavach/serving/server_preflight.py` (5-gate automated preflight verifier)
- **Lifecycle & Benchmark Tooling:**
  - `scripts/start_server.ps1`, `scripts/run_locust.ps1`
  - `benchmarks/locustfile.py`, `benchmarks/compile_report.py`
  - `requirements.serving.txt`, `requirements.serving.lock.txt`
- **Tests:**
  - `tests/serving_fakes.py`, `tests/test_serving.py` (`14` tests), `tests/test_fault_injection.py` (`10` tests)
- **Reports & Raw Telemetry:**
  - `reports/phase3_regression_baseline.json` (Stage 0)
  - `reports/phase3_test_report.json` (Stage 2)
  - `reports/phase3_server_preflight.json` (Stage 3)
  - `reports/locust/*` + `reports/phase3_load_test_summary.json` + `reports/phase3_load_test_report.md` (Stage 4)
  - `reports/phase3_completion_report.md` (this report, Stage 5)

---

## 3. Plan Section 5 Gate Evaluation Matrix & `compile_report.py` Reconciliation

### 3.1 Plan Section 5 Engineering Gates (`E1`-`E8`)

| Gate | Requirement Summary | Evidence / Observed Metrics | Verdict |
|---|---|---|---|
| **E1** | FastAPI Serving Contracts & Endpoints (`/rag/sync`, `/rag/async` sync+SSE, `/health`, `/tasks/*`, Pydantic v2 `extra="forbid"`) | Verified in `kavach/serving/schemas.py`, `app.py`; `14/14` contract/endpoint tests passed (`tests/test_serving.py`); Stage 3 Gates `4a-4g` all green | **PASS** |
| **E2** | Bounded Concurrency (`Semaphore(20)`), Serialized Retrieval (`RETRIEVAL_CONCURRENCY=1` + `pipeline_lock`), Single-Flight TTL Cache, Circuit Breaker | `10/10` chaos tests passed (`tests/test_fault_injection.py`); Stage 4 event-loop liveness `/health` P95 max `20.0 ms` under `213.77` agg RPS (`0` probe failures) | **PASS** |
| **E3** | Router LFS Weight Integrity, CPU-Only Torch Isolation & Zero Frozen Mutations | `model.safetensors` = `283,807,294` B (`sha256=10e7...a44`); `torch 2.15.0+cpu` (`cuda is None`); `153/153` frozen files & `5/5` DeBERTa weights unchanged | **PASS** |
| **E4** | Stage 3 Server Preflight & End-to-End Parity (`reports/phase3_server_preflight.json`) | `passed=true`; startup `156.304s`, RSS `3,002.9 MB` (`<=4096`), cold-vs-warm `0.019 ms`, sync-vs-async `sequence_similarity=0.9372`, identical chunks, `/health` p95 `6.582 ms` | **PASS** |
| **E5** | Refusal/Abstain LLM-Bypass Integrity & Stampede/Cache Safety | Stage 3 Gate `4c`: `bypassed_llm=true`, `generator_calls` unchanged (`3 -> 3`); Stage 4 across all `4` warm cells (`17,335` hits): `generator_calls == 7` (`0` leak calls) | **PASS** |
| **E6** | GPU VRAM Hard Cap (`<= 5,478 MiB`, Soft Target `<= 4,710 MiB`) Under Live Inference | Stage 3 Preflight passed (`5,370 MiB`), **but** Stage 4 peaked at **`5,508.0 MiB`** (`6/12` runs breached, `3` HTTP 502s in `sync_cold_u10`) and Stage 5 live pytest peaked at **`5,582 MiB`** (`6/6` `@live` tests in `tests/test_generation.py` raised `VRAMBudgetExceededError`) | **FAIL** *(Host shared-GPU desktop contention)* |
| **E7** | Server Stop Hygiene, Zero Placeholders & Secret Cleanliness | Stage 3 Gate `5` (`pid_file_removed=true`, `health_unreachable=true`, `python_gpu_apps=[]`); Stage 5 final check: `PID_EXISTS=False`, port `8000` free, `test_no_placeholders.py` `1 passed`, `0` secret findings (`167` files) | **PASS** |
| **E8** | Full Sequential Regression Suite (`KAVACH_LIVE=1`) & Honest Completion Gate | Stage 0: `134/134` passed (`357.65s`). Stage 2: `24/24` passed (`8.68s`). Stage 5 sequential pytest: **`152 passed, 6 failed, 0 skipped`** in `338.14s` (`exit=1`), where all `6` failures are `E6` `VRAMBudgetExceededError` in `tests/test_generation.py` | **FAIL** *(Triggered solely by E6 live VRAM failures)* |

### 3.2 Stage 4 Load Test Performance Targets (`P1`-`P4`)

| Target | Definition | Observed Metric | Verdict |
|---|---|---|---|
| **P1** | Event-loop liveness: async `/health` probe P95 `<= 500 ms`, `0` probe failures | Async max probe P95 = **`20.0 ms`** (u100 warm @ `213.77` agg RPS), **`0`** failures across `786` total probes (`sync` max P95 = `17.0 ms`) | **PASS** *(Note: `compile_report.py` logged `WARN` — see Section 3.3)* |
| **P2** | Throughput scaling: `async-warm` > `async-cold` & > `sync`; cold RPS bounded cleanly | Peak RAG RPS: `async-warm` = **`177.6729`** (`213.7675` agg RPS), `sync` = **`0.2187`**, `async-cold` = **`0.1916`** (near `u1` serial e2e cap `0.1682` & Phase 1.5 retrieval cap `0.1083`) | **PASS** |
| **P3** | Warm cache replay latency vs cold + SSE TTFT capture | Mean non-stream P50: `async-warm` **`2.25 ms`** vs `async-cold` **`29,775.0 ms`** (**`13,233.33x`** speedup); warm SSE TTFT P50 **`2.0-4.0 ms`** | **PASS** |
| **P4** | Saturation reliability (`u >= 50`): `<= 5%` warm error rate, `0` empty answers | `0.0%` error rate across all `async-warm` (`17,292` RAG reqs total, `u>=50`: `16,055` reqs, `0` fails), `async-cold`, and `sync` `u50`/`u100`; **`0`** empty answers | **PASS** |

### 3.3 ID Reconciliation Between `benchmarks/compile_report.py` and Plan Section 5

`benchmarks/compile_report.py` uses an internal scorecard numbering scheme for the Stage 4 telemetry (`E5`-`E7`, `P1`-`P4`). The exact mapping to Plan Section 5 is:

| `compile_report.py` ID & Status | Meaning in `phase3_load_test_summary.json` | Plan Section 5 Equivalent | Reconciliation Note |
|---|---|---|---|
| **`E5` (`FAIL`)** | GPU VRAM Hard Cap (`<= 5,478 MiB`) & Guardrail Telemetry (`5,508.0 MiB` peak, `6/12` runs breached) | **Gate E6** (`FAIL`) | Identical underlying metric & failure (`5,508 > 5,478 MiB`). |
| **`E6` (`PASS`)** | Bounded GPU `Semaphore(20)`, Retrieval Serialization & Head-of-Line Isolation (`20.0 ms` max probe P95) | **Gates E2 & E4** (`PASS`) | Verifies event-loop liveness and bounded admission under concurrency. |
| **`E7` (`PASS`)** | Cache-Hit Distribution (`17,335` warm hits), Stampede Guard & Failure Accounting (`0` empty answers, `3` HTTP 502s) | **Gates E2 & E5** (`PASS`) | Confirms stampede protection, zero empty answers, and exact failure accounting. |
| **`P1` (`WARN`)** | Event-loop liveness: `async_max_probe_p95_ms = 20.0 ms <= 500 ms` with `0` failures, `WARN` only because `20.0 ms > sync_max_probe_p95_ms (17.0 ms)` | **Target P1** (`PASS`) | Both values sit >25x below the `500 ms` target (`6.582 ms < 50 ms` in preflight, `20.0 ms` under `213.77` agg RPS). Strictly passes Plan P1. |


### 3.4 Canonical `kavach_phase3_plan.md` Section 5 Matrix (`E1`-`E8` & `P1`-`P4`)

Mapping every directly observed artifact and test metric strictly onto the canonical `kavach_phase3_plan.md` Section 5 Gate IDs:

| Plan ID | Canonical Plan Gate / Performance Criterion | Threshold / Requirement | Directly Observed Metrics | Verdict |
|---|---|---|---|---|
| **E1** | **Regression** | Stage 0 hashes intact, `0` edits to frozen files, `134` prior tests pass (`KAVACH_LIVE=1`) | `153/153` frozen files & `5/5` DeBERTa weights intact (`MISSING=0`, `MODIFIED=0`); `0` edited tracked files; Stage 0 passed `134/134` (`357.65s`), but Stage 5 sequential pytest saw **`128` prior tests pass + `6` `@live` `tests/test_generation.py` failures** (`VRAMBudgetExceededError > 5478 MiB`, linked to `E6`) | **FAIL** *(6 `@live` VRAM breaches from `E6`; 128/128 prior non-VRAM tests & all hashes PASSED)* |
| **E2** | **Packages & Isolation** | CPU-only `torch`, `.venv_gpu` absent, serving delta in lockfile only, `requirements*.txt` untouched | `torch==2.15.0+cpu`, `cuda is None`, `cuda_available=False`, `.venv_gpu` absent; `fastapi==0.162.5`, `uvicorn==0.54.0`, `locust==2.47.0` (`+29` packages in `requirements.serving.lock.txt`); `pip freeze` matches Stage 0 (`105` pkgs); `requirements.txt` & `requirements.lock.txt` untouched | **PASS** |
| **E3** | **Unit & Chaos Tests** | Phase 3 serving contract & fault-injection suites pass (`0` failures/skips) | **`24/24` passed** in `8.68s` (`14` in `tests/test_serving.py`, `10` in `tests/test_fault_injection.py`, `reports/phase3_test_report.json`); `24/24` also passed inside Stage 5 sequential pytest | **PASS** |
| **E4** | **Server Preflight** | Automated 5-gate preflight (`gates 1-5`, `4a-4g`) passes end-to-end | `reports/phase3_server_preflight.json`: `passed=true`, `error=null`, startup `156.304s`, RSS `3,002.9 MB` (`<=4096`), peak VRAM `5,370 MiB` (`<=5,478`), `/health` P95 `6.582 ms` | **PASS** |
| **E5** | **Load Test Complete** | Full 12-cell Locust sweep completes cleanly with compiled artifacts | `run_count=12` (`sync`, `async-cold`, `async-warm` x `1, 10, 50, 100` users); `reports/phase3_load_test_summary.json` & `phase3_load_test_report.md` compiled; `0` client timeouts, `0` empty answers | **PASS** |
| **E6** | **VRAM** | Live GPU VRAM `<= 5,478 MiB` hard limit (soft target `<= 4,710 MiB`) | Stage 3 Preflight passed (`5,370 MiB`), but Stage 4 peaked at **`5,508.0 MiB`** (`6/12` runs `vram_breached=true`, `3` HTTP 502s in `sync_cold_u10`) and Stage 5 live pytest peaked at **`5,582 MiB`** (`6/6` `@live` `tests/test_generation.py` raised `VRAMBudgetExceededError`) | **FAIL** *(Host shared desktop GPU contention; guardrails functioned without OOM)* |
| **E7** | **Bypass Integrity** | Refusals / abstain & cache hits execute `0` LLM calls | Stage 3 Gate `4c`: `bypassed_llm=true`, `generator_calls` unchanged (`3 -> 3`); Stage 4 across all `4` `async_warm` cells (`u1..u100`): **`generator_calls == 7`** (`1` prewarm + `2` CVE + `4` in-scope + `0` OOS refusals + `0` across `17,335` cache hits) | **PASS** |
| **E8** | **Audit** | Zero placeholders, zero leaked secrets, clean post-run server/GPU state | `test_no_placeholders.py` (`1 passed in 0.58s`); `167`-file secret scan `FINDINGS=[]`; `logs/.server.pid` absent, port `8000` unbound, `0` Python/uvicorn/locust GPU apps | **PASS** |
| **P1** | **`async-warm u100` Fail Rate** | Failure rate `<= 1%` | **`0.00%`** (`0` failures across `10,534` RAG requests & `58` health probes) | **PASS** |
| **P2** | **`async-warm u100` P95 Latency** | End-to-end P95 latency `<= 250 ms` | **`6.29 ms`** (`rag_only.p95_ms`; non-stream P95 `6.0 ms`, SSE P95 `7.0 ms`, SSE TTFT P95 `8.0 ms`) | **PASS** |
| **P3** | **`async-cold` vs `sync` Fail Rate** | `async-cold` failure rate `<= sync` at `1, 10, 50, 100` users | `u1`: **`0.0% <= 0.0%`**, `u10`: **`0.0% <= 20.0%`**, `u50`: **`0.0% <= 0.0%`**, `u100`: **`0.0% <= 0.0%`** | **PASS** |
| **P4** | **`/health` Liveness Under `async-cold u100`** | `/health` probe P95 `< 50 ms` during `async-cold u100` saturation | **`17.0 ms`** (`0` failures across `68` probes; `max=99.46 ms`, `95` coroutines waiting on `retrieval_sem`) | **PASS** |

---

## 4. Test Campaign Metrics (Stages 0, 2, 3, and 5)

### 4.1 Stage 0 Baseline & Stage 2 Serving/Chaos Suites
- **Stage 0 Baseline (`reports/phase3_regression_baseline.json`, `2026-11-12T00:56:40Z`):**
  - Command: `PYTHONUTF8=1 KAVACH_LIVE=1 .\.venv\Scripts\python.exe -m pytest -q -rs`
  - Result: **`134 passed, 0 failed, 0 skipped`** (`1` FutureWarning) in **`357.65s`**, `exit_code=0`.
- **Stage 2 Phase 3 Unit & Fault-Injection (`reports/phase3_test_report.json`, `2026-11-12T01:07:46Z`):**
  - Command: `PYTHONUTF8=1 .\.venv\Scripts\python.exe -m pytest -v -rs tests/test_serving.py tests/test_fault_injection.py`
  - Result: **`24 passed, 0 failed, 0 skipped`** (`14` in `tests/test_serving.py`, `10` in `tests/test_fault_injection.py`) in **`8.68s`**, `exit_code=0`.

### 4.2 Stage 3 Live Server Preflight (`reports/phase3_server_preflight.json`, `2026-11-12T01:37:15Z`)
- **Overall Result:** `passed=true`, `error=null`, `exit_code=0`.
- **Gate 1 (Router LFS):** `models/router_deberta/model.safetensors` verified (`283,807,294` B / `270.66 MB`).
- **Gate 2 (Ollama GPU):** `version="0.40.16"`, `matched_model="kavach-tier5:latest"`, `warm_chat_ms=1628.39 ms`, `size=4,755,397,016` B, `size_vram=4,755,397,016` B (`vram_ratio=1.0`), `nvidia_smi` `5,333 / 6,141 MiB` (`<= 5,478 MiB`).
- **Gate 3 (Server Boot):** `startup_s=156.304s`, `rss_mb=3,002.9 MB` (`<= 4,096.0`), `/health` `status="ok"`, `vram_used_mib=5,340`, `vram_peak_mib=5,370`, `vram_breached=false`, `generator_calls=1`.
- **Gate 4 (Live Smoke 4a-4g):**
  - `4a_fast_path_cve`: `route="FAST_PATH_EXACT_LOOKUP"`, `answer_chars=511`, `total_ms=3,745.711`.
  - `4b_semantic_cache` (`DEV_IN_001`): cold `retrieval_latency_ms=9,311.206`, `ttft_ms=10,283.134`, `total_ms=11,743.615`, chunks `["CVE-2014-0160_chunk_001", "CVE-2014-0160_chunk_003"]`; warm replay `total_ms=0.019 ms`, `cache_hit=true`, `answer_matches_cold=true`.
  - `4c_refusal_bypass`: `bypassed_llm=true`, `refused=true`, `generator_calls` `3 -> 3`, `total_ms=92.947 ms`.
  - `4d_sse_streaming` (`DEV_IN_002`): `event_sequence_valid=true`, `token_events=92`, `streamed_chars=470`, `ttft_ms=11,865.779`, `total_ms=14,291.013`.
  - `4e_sync_vs_async_parity` (`DEV_IN_001`): `sync_total_ms=12,246.342` (`ttft=10,836.173`), `async_cold_total_ms=11,743.615`, `sequence_similarity=0.9372`, `token_jaccard=0.5882`, `retrieved_chunks_identical=true`, `exact_answer_match=false` (non-greedy decode).
  - `4f_concurrent_health_probe` (`DEV_IN_003`): `query_total_ms=11,431.677`, `cache_hit=false`, `health_samples_count=103`, `health_min=1.309 ms`, `p50=1.594 ms`, `p95=6.582 ms` (`< 50 ms`), `max=18.294 ms`.
  - `4g_verify_corpus_task`: `job_id="c7d823717680"`, `status="succeeded"`, exact SHA-256 & byte matches (`corpus_chunks.parquet` `199,073` B, `index_sq8_mrl512.faiss` `262,737` B, `index_sq8_mrl512.idmap.json` `16,538` B).
- **Gate 5 (Server Stop):** `stop_exit_code=0`, `pid_file_removed=true`, `health_unreachable=true`, `python_gpu_apps=[]`.

### 4.3 Stage 5 Full Sequential Pytest Verification (`KAVACH_LIVE=1`)
- **Authoritative Run (`task-218`):** Launched `scripts\start_ollama.ps1 -Restart` (`version=0.40.16`, `pid=2608`) inside the same PowerShell session, verified `/api/version`, then ran `PYTHONUTF8=1 KAVACH_LIVE=1 .\.venv\Scripts\python.exe -m pytest -q -rs`.
  - **Outcome:** **`152 passed, 6 failed, 0 skipped, 1 warning`** (`torch.jit.script` FutureWarning) in **`338.14s (0:05:38)`**, `exit_code=1`.
  - **Failed Tests (all 6 `@live` tests in `tests/test_generation.py` — identical `VRAMBudgetExceededError`):**
    1. `test_live_model_is_q4km_and_fully_gpu_offloaded` (`VRAMBudgetExceededError: GPU VRAM exceeded hard limit 5478 MiB.`, logged `5188 MiB > 4710 MiB` at load then breached `5478 MiB` during streaming)
    2. `test_live_negative_constraint` (`VRAMBudgetExceededError`, logged `5576 MiB`)
    3. `test_live_literal_decimals` (`VRAMBudgetExceededError`, logged `5578 MiB`)
    4. `test_live_prompt_injection_resistance` (`VRAMBudgetExceededError`, logged `5578 MiB`)
    5. `test_live_conflict_owasp_both_versions_and_citations` (`VRAMBudgetExceededError`, logged `5578 MiB`)
    6. `test_live_vram_stays_under_hard_limit_and_decode_speed_recorded` (`VRAMBudgetExceededError`, logged `5582 MiB`)
  - *Operational diagnostic note:* Pre-run GPU baseline was `941 MiB` (`4,980 MiB free`) vs `6,141 MiB` total. An earlier pytest attempt (`task-145`) failed the same 6 tests immediately (`[WinError 10061]`) because the Windows command runner's Job Object terminated the `ollama serve` child process spawned in an earlier separate `run_command`; unifying `start_ollama.ps1 -Restart` and `pytest` inside `task-218` resolved daemon lifetime and exposed the real shared-GPU VRAM breach.

---

## 5. Stage 4 Locust 12-Cell Matrix & Telemetry (`reports/phase3_load_test_summary.json`)

Each cell ran with a fresh uvicorn restart/stop cycle (`run_seconds=60`; cold cells ran ~69-71s including tail in-flight wait). Client timeout was `120.0s`; `GOLDEN_FREE_SEED=20260` 8-query pool (`2` bare CVE IDs, `4` `inscope_dev_100` queries, `2` `oos_dev_60` refusals).

| Cell (`slug`) | Mode / Endpoint | Users | RAG Reqs | RAG Fails (Err %) | RAG RPS (Agg RPS) | Avg (ms) | P50 (ms) | P95 (ms) | Max (ms) | `/health` Probe P50 / P95 / Max (ms) | Peak VRAM (MiB) / Breached | RSS (MB) | Gen Calls | Cache Hits / Miss |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|
| `sync_cold_u1` | `sync` / cold | 1 | 5 | 0 (0.0%) | 0.0761 (1.0799) | 9,197.44 | 12,000.0 | 14,000.0 | 14,275.83 | 3.0 / 17.0 / 17.41 | 5,452 / False | 3,066.1 | 5 | 0 / 0 |
| `sync_cold_u10` | `sync` / cold | 10 | 15 | **3 (20.0%)** | 0.2187 (1.2102) | 27,441.39 | 20,000.0 | 51,000.0 | 51,416.78 | 3.0 / 17.0 / 37.27 | **5,489 / True** | 3,230.5 | 17 | 0 / 0 |
| `sync_cold_u50` | `sync` / cold | 50 | 13 | 0 (0.0%) | 0.1911 (1.1906) | 41,143.68 | 42,000.0 | 64,000.0 | 64,262.82 | 3.0 / 16.0 / 98.96 | **5,492 / True** | 3,378.4 | 10 | 0 / 0 |
| `sync_cold_u100` | `sync` / cold | 100 | 14 | 0 (0.0%) | 0.2043 (1.1965) | 38,922.46 | 44,000.0 | 68,000.0 | 67,517.61 | 3.0 / 17.0 / 40.84 | 5,475 / False | 3,397.7 | 9 | 0 / 0 |
| `async_cold_u1` | `async` / cold | 1 | 8 | 0 (0.0%) | 0.1179 (1.1347) | 5,944.49 | 7,809.43 | 10,935.57 | 10,935.57 | 3.0 / 17.0 / 21.55 | 5,459 / False | 3,027.3 | 4 | 0 / 0 |
| `async_cold_u10` | `async` / cold | 10 | 13 | 0 (0.0%) | 0.1916 (1.2230) | 28,338.13 | 32,220.21 | 54,454.55 | 54,571.95 | 3.0 / 16.0 / 22.82 | **5,508 / True** | 3,059.4 | 12 | 0 / 0 |
| `async_cold_u50` | `async` / cold | 50 | 13 | 0 (0.0%) | 0.1914 (1.1924) | 43,714.18 | 49,000.0 | 62,000.0 | 61,819.45 | 3.0 / 17.0 / 88.38 | 5,472 / False | 3,081.5 | 11 | 0 / 0 |
| `async_cold_u100` | `async` / cold | 100 | 13 | 0 (0.0%) | 0.1911 (1.2203) | 33,894.63 | 35,681.82 | 54,272.73 | 54,700.43 | 3.0 / 17.0 / 99.46 | **5,489 / True** | 3,071.4 | 14 | 0 / 0 |
| `async_warm_u1` | `async` / warm | 1 | 109 | 0 (0.0%) | 1.8659 (3.3552) | 1.94 | 2.0 | 3.95 | 6.81 | 3.0 / 17.0 / 17.22 | **5,486 / True** | 3,016.0 | 7 | 119 / 8 |
| `async_warm_u10` | `async` / warm | 10 | 1,128 | 0 (0.0%) | 18.9990 (23.2098) | 2.28 | 2.0 | 4.0 | 28.55 | 3.0 / 17.0 / 18.52 | 5,458 / False | 3,025.5 | 7 | 1,136 / 8 |
| `async_warm_u50` | `async` / warm | 50 | 5,521 | 0 (0.0%) | 93.0342 (112.4129) | 2.51 | 2.0 | 5.0 | 24.21 | 5.0 / 18.0 / 19.69 | 5,471 / False | 3,045.6 | 7 | 5,529 / 8 |
| `async_warm_u100` | `async` / warm | 100 | 10,534 | 0 (0.0%) | **177.6729 (213.7675)** | 3.08 | 3.0 | 6.29 | 25.12 | 5.0 / 20.0 / 28.75 | **5,488 / True** | 3,029.9 | 7 | 10,551 / 8 |

**SSE TTFT (`sse_ttft` P50 / P95 ms across streaming 20%-weight traffic):**
- Cold: `async_cold_u1` `7,400.0 / 7,400.0` (1 req), `async_cold_u10` `42,000.0 / 42,000.0` (2 reqs), `async_cold_u50` `N/A` (0 streams completed in window), `async_cold_u100` `55,000.0 / 55,000.0` (2 reqs).
- Warm: `async_warm_u1` `2.0 / 4.0` (30 reqs), `async_warm_u10` `2.0 / 5.0` (192 reqs), `async_warm_u50` `3.0 / 7.0` (1,092 reqs), `async_warm_u100` `4.0 / 8.0` (2,082 reqs).

---

## 6. Deep Technical Analysis

### 6.1 Gate E6 Hardware VRAM Root-Cause Analysis & Guardrail Behavior
- **Hardware Margin Arithmetic:** The RTX 4050 Laptop GPU has **`6,141 MiB`** total VRAM and a Phase 2 frozen `vram_hard_limit_mib = 5,478 MiB` (`soft target = 4,710 MiB`). `kavach-tier5:latest` (`Q4_K_M`) allocates **`4,755,397,016` bytes (`4,535.1 MiB`)** strictly in `size_vram` (`vram_ratio = 1.0`), leaving **`942.9 MiB`** total headroom below `5,478 MiB`.
- **Shared Desktop Contention:** Windows desktop/editor/browser/messaging processes (`Antigravity.exe`, `msedgewebview2.exe`, `WhatsApp.Root.exe`, `explorer.exe`, `ShellExperienceHost.exe`, `TextInputHost.exe`, `LockApp.exe`, etc.) fluctuate between **`486 MiB`** (idle display) and **`803-941 MiB`** during active desktop sessions. Whenever background desktop VRAM rises above ~`900 MiB` while `kavach-tier5` is resident and KV/compute allocations engage, system `memory.used` crosses `5,478 MiB` (`5,486-5,508 MiB` in Stage 4; `5,576-5,582 MiB` in Stage 5 pytest).
- **Critical Proof (`async_warm_u1` & `async_warm_u100`):** In `async_warm_u1` (`peak=5,486 MiB`) and `async_warm_u100` (`peak=5,488 MiB`), `generator_calls == 7` remained constant before and after the 60s load test (all live traffic hit CPU `MutexTTLCache` with zero GPU generation). Even zero-GPU cache replay cells tripped `vram_breached=true` due solely to transient desktop DWM/compositor VRAM spikes pushing total GPU memory `+8-10 MiB` above `5,478 MiB`.
- **Guardrails Validated Under Breach:**
  - `VRAMSampler` (`1.0s` daemon poll) latched `vram_breached=true` and automatically transitioned `/health` to `"status": "degraded"` without dropping HTTP health probes.
  - On `sync_cold_u10`, `OllamaGenerator`'s `VRAMWatchdog` (`kavach/generation/generator.py:620`) aborted `3` active generations cleanly with `VRAMBudgetExceededError`, mapped by `@app.exception_handler(GenerationError)` (`kavach/serving/app.py:433-435`) to `HTTP 502: {"status":"UPSTREAM_FAILURE","detail":"GPU VRAM exceeded hard limit 5478 MiB."}` — zero Ollama crashes, zero CUDA OOMs, and automatic recovery on subsequent requests (`sync_cold_u50` & `sync_cold_u100` had `0` failures).
  - Note: `/rag/sync` intentionally has no pre-request `_admit` check (`kavach/serving/app.py:468-485`), which is why mid-stream `VRAMBudgetExceededError` produced `502 UPSTREAM_FAILURE` (`vram_hard_limit_503_count == 0`) rather than pre-admission `503 VRAM_HARD_LIMIT_EXCEEDED` (`_admit(rt)` on `/rag/async`, `app.py:149-156`).

### 6.2 Gate E7 Refusal/Bypass Integrity — Exact Mathematical Proof
- The Stage 4 Locust 8-query pool (`benchmarks/locustfile.py`) contains **`2` bare CVE IDs**, **`4` `inscope_dev_100` queries**, and **`2` `oos_dev_60` refusals**.
- During `Invoke-WarmCachePrewarm` (`scripts/run_locust.ps1`), priming executes 1 startup prewarm (`_prewarm_generator` = **`1`** call) plus 1 pass per unique pool query:
  - `2` bare CVE queries -> `FAST_PATH_EXACT_LOOKUP` (`2` generation calls)
  - `4` `inscope_dev_100` queries -> standard RAG generation (`4` generation calls)
  - `2` `oos_dev_60` queries -> `HARD_REFUSAL` / abstention (`0` generation calls; bypassed via `_reason(res)` -> `plan is None`)
- Therefore, exact theoretical LLM call count = `1 + 2 + 4 + 0 =` **`7` calls**.
- Observed `generator_calls` across all four `async_warm` runs (`u1`, `u10`, `u50`, `u100`) = **`7`, `7`, `7`, `7`**, alongside **`8` cache misses** (all 8 queries cached, including canonical refusals!) and **`119 + 1,136 + 5,529 + 10,551 = 17,335` cache hits** (`99.92%` hit rate at `u100`).
- Combined with Stage 3 Gate `4c` (`"How do I bake a sourdough loaf?"` -> `bypassed_llm=true`, `refused=true`, `generator_calls` `3 -> 3`), **zero LLM calls ever leaked past refusal/abstain guards or semantic cache hits**.

### 6.3 D2 Serialization Bottleneck & Head-of-Line Isolation
- **Retrieval Serialization Dominance:** Phase 1.5 baseline retrieval latency is **`9,231.71 ms` median / `9,449.96 ms` mean** (`serialized_retrieval_cap_rps = 0.1083`), using CPU DeBERTa + BGE + CrossEncoder under `RETRIEVAL_CONCURRENCY=1` (`asyncio.Semaphore(1)`) + `pipeline_lock`. Thus, cold async RPS (`0.1179` at `u1`, plateauing at **`0.1911-0.1916` RPS** for `u10-u100`) is physically bounded by serialized CPU retrieval + single-model Ollama execution, not by FastAPI or `Semaphore(20)`.
- **Queue Telemetry Proof Under Saturation:**
  - `async_cold_u10`: `retrieval_waiting=6`, `gpu_inflight=1` (`10` virtual users)
  - `async_cold_u50`: `retrieval_waiting=44`, `gpu_inflight=2` (`50` virtual users)
  - `async_cold_u100`: `retrieval_waiting=95`, `gpu_inflight=2` (`100` virtual users)
- **Head-of-Line Event-Loop Isolation:** Even with `95` coroutines queued on `retrieval_sem` and `u100` warm flooding **`213.7675` aggregated RPS** (`177.6729` RAG RPS), background `asyncio.to_thread` retrieval and `MutexTTLCache` kept the ASGI event loop completely non-blocking: `/health` P95 stayed **`<= 20.0 ms`** (`max = 28.75 ms` warm `u100`, `99.46 ms` cold `u100`) with **`0` failed probes across all 12 runs**.

---

## 7. Deviations & Repairs Log (Stages 0-5)

1. **Windows Inherited-Pipe Hang Fix (Stage 3):**
   - *Symptom:* Initial `server_preflight.py` (`task-82`) hung for 23+ minutes after launching `scripts/start_server.ps1` because `subprocess.run(..., capture_output=True)` blocked waiting for pipe EOF held open by the detached `uvicorn` child process inheriting stdout/stderr handles.
   - *Repair:* Updated `_run_ps_script` in `kavach/serving/server_preflight.py` to use `stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, close_fds=True`, added file-backed diagnostics (`logs/start_server.log`) in `scripts/start_server.ps1`, and enforced finite `httpx.Timeout(120.0, connect=10.0)` (`5.0s / 2.0s` on stop check).
2. **Locust Gevent Import Ordering Fix (Stage 4):**
   - *Symptom:* Importing `gevent` or `httpx` before `from locust import ...` broke Locust's internal `gevent.monkey.patch_all()` ordering.
   - *Repair:* Placed `from locust import ...` strictly first at top of `benchmarks/locustfile.py`.
3. **PowerShell 5.1 Inline Quoting, GPU Idle Polling & Warm Prewarm Serialization (Stage 4):**
   - *Repair:* `scripts/run_locust.ps1` writes temporary `.py` helper scripts (avoiding Windows PowerShell 5.1 inline `python -c` quote mangling) for `Capture-HealthSnapshot` and `Invoke-WarmCachePrewarm`, runs `Wait-GpuIdle` (`gpu_idle_timed_out` cleanly bounded and recorded per run in `meta`), and cleanly starts/stops uvicorn per matrix cell.
4. **Stage 5 Child-Process Daemon Lifetime in Windows Command Runner:**
   - *Observation:* Running `scripts\start_ollama.ps1 -Restart` in a standalone `run_command` terminated spawned `ollama serve` on shell exit under the runner Job Object (`task-145` saw `WinError 10061`).
   - *Resolution:* Invoked `scripts\start_ollama.ps1 -Restart`, `/api/version` check, and `pytest -q -rs` in the single unified `task-218` shell, capturing clean live execution metrics.

---

## 8. Inputs & Recommendations for Phase 3.5 and Phase 4

1. **Dedicated Headless / Non-Contended GPU Environment (Unblocks Gate E6):**
   - Re-run `KAVACH_LIVE=1` pytest and the 12-cell Locust sweep with non-essential Windows desktop GPU apps closed (target baseline VRAM `< 350 MiB`, leaving `> 600 MiB` headroom below `5,478 MiB` above `kavach-tier5`'s `4,535.1 MiB`) or deploy on an 8+ GiB VRAM GPU / Linux headless runner.
   - Alternatively, in Phase 3.5+ runtime tuning, consider deducting or isolating display overhead, or evaluating `OLLAMA_KV_CACHE_TYPE=q8_0` / smaller context footprint only if re-calibrated against Phase 2 accuracy baselines (do **not** silently relax frozen `5,478 MiB` thresholds without formal calibration sign-off).
2. **CPU Retrieval Throughput Optimization (Target `9.23s` Serialization Bottleneck):**
   - Because Phase 1.5 CPU retrieval (`~9.2s/query`, serialized by `RETRIEVAL_CONCURRENCY=1` + `pipeline_lock`) caps cold throughput at `~0.19 RPS`, Phase 4 observability & Phase 3.5 retrieval tuning should profile DeBERTa router / BGE embedder ONNX/OpenVINO quantization or batch micro-concurrency once RAM budget (`~3.0-3.4 GB` RSS) allows.
3. **Phase 4 Observability Hooks Already Exposed:**
   - `/health` already surfaces structured Prometheus-ready counters: `vram_used_mib`, `vram_peak_mib`, `vram_breached`, `rss_mb`, `retrieval_waiting`, `gpu_waiting`, `gpu_inflight`, `generator_calls`, `cache.{hits,misses,coalesced,evictions,lock_timeouts,size}`, and `circuit.{state,consecutive_failures,trips}`.
   - Diagnostic response headers (`X-Retrieval-Latency-MS`, `X-Retrieval-Queue-MS`, `X-TTFT-MS`, `X-Total-Tokens`, `X-Tokens-Per-Second`, `X-Cache`, `X-Endpoint`) are active on both JSON and SSE responses.
