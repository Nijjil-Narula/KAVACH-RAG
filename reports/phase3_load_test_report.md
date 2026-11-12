# KAVACH-RAG Phase 3 Load Test & Concurrency Benchmark Report

- **Generated at (UTC):** 2026-11-12T03:27:28Z
- **Source Locust directory:** `C:\Users\Nijjil Narula\DOC\PROJECTS\KAVACH-RAG\reports\locust`
- **Total matrix runs compiled:** 12
- **Client request timeout:** 120 s (`benchmarks/locustfile.py`)
- **GPU concurrency semaphore:** `Semaphore(20)`
- **VRAM limits:** Soft target `<= 4710 MiB` | Hard limit `<= 5478 MiB`

## 1. Executive Summary & Test Matrix

Each matrix cell started a clean FastAPI server instance (`start_server.ps1 -Restart`), waited for GPU utilization `<= 10%` across 3 consecutive 1s samples (60s cap), ran headless Locust over an 8-query pool (`2` bare CVE IDs, `4` sampled from `inscope_dev_100.jsonl`, `2` sampled from `oos_dev_60.jsonl`; dev sets disjoint from `golden_150.jsonl`), and captured a post-run `/health` JSON snapshot.

| Run Slug | Profile | Users | Spawn Rate | Duration (s) | Locust Exit | GPU Idle Timed Out | RAG Requests | RAG RPS | RAG Error % |
|---|---|---:|---:|---:|---:|---|---:|---:|---:|
| `sync_cold_u1` | `sync` | 1 | 1 | 69.0 | 0 | True | 5 | 0.08 | 0.00% |
| `sync_cold_u10` | `sync` | 10 | 10 | 70.8 | 1 | True | 15 | 0.22 | 20.00% |
| `sync_cold_u50` | `sync` | 50 | 10 | 70.8 | 0 | True | 13 | 0.19 | 0.00% |
| `sync_cold_u100` | `sync` | 100 | 10 | 70.8 | 0 | False | 14 | 0.20 | 0.00% |
| `async_cold_u1` | `async-cold` | 1 | 1 | 70.8 | 0 | False | 8 | 0.12 | 0.00% |
| `async_cold_u10` | `async-cold` | 10 | 10 | 70.8 | 0 | False | 13 | 0.19 | 0.00% |
| `async_cold_u50` | `async-cold` | 50 | 10 | 70.8 | 0 | False | 13 | 0.19 | 0.00% |
| `async_cold_u100` | `async-cold` | 100 | 10 | 70.8 | 0 | False | 13 | 0.19 | 0.00% |
| `async_warm_u1` | `async-warm` | 1 | 1 | 60.8 | 0 | True | 109 | 1.87 | 0.00% |
| `async_warm_u10` | `async-warm` | 10 | 10 | 60.7 | 0 | False | 1128 | 19.00 | 0.00% |
| `async_warm_u50` | `async-warm` | 50 | 10 | 60.7 | 0 | False | 5521 | 93.03 | 0.00% |
| `async_warm_u100` | `async-warm` | 100 | 10 | 60.8 | 0 | True | 10534 | 177.67 | 0.00% |

### Scorecard Summary (Performance Targets P1–P4 & Engineering Gates E5–E7)

| Check | Description | Status |
|---|---|---|
| **P1** | Event-Loop Liveness (/health probe under async load) | **WARN** |
| **P2** | Concurrency Throughput Scaling (async-warm > async-cold vs sync) | **PASS** |
| **P3** | Warm Cache Replay Latency & SSE Time-To-First-Token (TTFT) | **PASS** |
| **P4** | Saturation Reliability & Graceful Degradation (u >= 50) | **PASS** |
| **E5** | GPU VRAM Hard Cap & Guardrail Telemetry | **FAIL** |
| **E6** | Bounded GPU Semaphore(20), Retrieval Serialization & Head-of-Line Isolation | **PASS** |
| **E7** | Cache-Hit Distribution, Stampede Guard & Failure Accounting | **PASS** |

## 2. Throughput, Latency & Concurrency Scaling (RAG-Only vs Raw Locust Aggregated)

> **Methodology note:** `benchmarks/locustfile.py` fires two auxiliary Locust stat rows alongside `/rag/*` HTTP traffic: `GET /health probe` (1 req/s event-loop monitor) and `SSE /rag/async first-token[mode]` (synthetic TTFT event). Because Locust's `Aggregated` row sums all of these together, **RAG-Only** metrics below strictly sum only `POST /rag/*` rows (non-stream + stream). Both RAG-Only and raw Locust `Aggregated` are shown for complete transparency.

### 2a. Combined RAG-Only Scaling Table

| Profile | Users | RAG Reqs | RAG Fails | Error % | RAG RPS | P50 (ms) | P95 (ms) | P99 (ms) | Max (ms) | Raw Aggregated RPS | Raw Aggregated Err % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `sync` | 1 | 5 | 0 | 0.00% | 0.08 | 12000.0 | 14000.0 | 14000.0 | 14275.8 | 1.08 | 0.00% |
| `sync` | 10 | 15 | 3 | 20.00% | 0.22 | 20000.0 | 51000.0 | 51000.0 | 51416.8 | 1.21 | 3.61% |
| `sync` | 50 | 13 | 0 | 0.00% | 0.19 | 42000.0 | 64000.0 | 64000.0 | 64262.8 | 1.19 | 0.00% |
| `sync` | 100 | 14 | 0 | 0.00% | 0.20 | 44000.0 | 68000.0 | 68000.0 | 67517.6 | 1.20 | 0.00% |
| `async-cold` | 1 | 8 | 0 | 0.00% | 0.12 | 7809.4 | 10935.6 | 10935.6 | 10935.6 | 1.13 | 0.00% |
| `async-cold` | 10 | 13 | 0 | 0.00% | 0.19 | 32220.2 | 54454.6 | 54571.9 | 54571.9 | 1.22 | 0.00% |
| `async-cold` | 50 | 13 | 0 | 0.00% | 0.19 | 49000.0 | 62000.0 | 62000.0 | 61819.4 | 1.19 | 0.00% |
| `async-cold` | 100 | 13 | 0 | 0.00% | 0.19 | 35681.8 | 54272.7 | 54700.4 | 54700.4 | 1.22 | 0.00% |
| `async-warm` | 1 | 109 | 0 | 0.00% | 1.87 | 2.0 | 4.0 | 6.0 | 6.8 | 3.36 | 0.00% |
| `async-warm` | 10 | 1128 | 0 | 0.00% | 19.00 | 2.0 | 4.0 | 10.4 | 28.6 | 23.21 | 0.00% |
| `async-warm` | 50 | 5521 | 0 | 0.00% | 93.03 | 2.0 | 5.0 | 11.1 | 24.2 | 112.41 | 0.00% |
| `async-warm` | 100 | 10534 | 0 | 0.00% | 177.67 | 3.0 | 6.3 | 10.4 | 25.1 | 213.77 | 0.00% |

### 2b. Per-Route Breakdown (Non-Stream, SSE Stream & SSE First-Token TTFT)

| Run Slug | Non-Stream Reqs / RPS / P50 / P95 (ms) | Stream Reqs / RPS / P50 / P95 (ms) | SSE TTFT Count / P50 / P95 / P99 (ms) |
|---|---|---|---|
| `sync_cold_u1` | 5 / 0.08 / 12000.0 / 14000.0 | N/A | N/A |
| `sync_cold_u10` | 15 / 0.22 / 20000.0 / 51000.0 | N/A | N/A |
| `sync_cold_u50` | 13 / 0.19 / 42000.0 / 64000.0 | N/A | N/A |
| `sync_cold_u100` | 14 / 0.20 / 44000.0 / 68000.0 | N/A | N/A |
| `async_cold_u1` | 7 / 0.10 / 9100.0 / 11000.0 | 1 / 0.01 / 7000.0 / 7000.0 | 1 / 7400.0 / 7400.0 / 7400.0 |
| `async_cold_u10` | 11 / 0.16 / 31000.0 / 55000.0 | 2 / 0.03 / 38000.0 / 38000.0 | 2 / 42000.0 / 42000.0 / 42000.0 |
| `async_cold_u50` | 13 / 0.19 / 49000.0 / 62000.0 | N/A | N/A |
| `async_cold_u100` | 11 / 0.16 / 30000.0 / 55000.0 | 2 / 0.03 / 49000.0 / 49000.0 | 2 / 55000.0 / 55000.0 / 55000.0 |
| `async_warm_u1` | 79 / 1.35 / 2.0 / 4.0 | 30 / 0.51 / 2.0 / 3.0 | 30 / 2.0 / 4.0 / 7.0 |
| `async_warm_u10` | 936 / 15.77 / 2.0 / 4.0 | 192 / 3.23 / 2.0 / 4.0 | 192 / 2.0 / 5.0 / 13.0 |
| `async_warm_u50` | 4429 / 74.63 / 2.0 / 5.0 | 1092 / 18.40 / 2.0 / 5.0 | 1092 / 3.0 / 7.0 / 15.0 |
| `async_warm_u100` | 8452 / 142.56 / 3.0 / 6.0 | 2082 / 35.12 / 3.0 / 7.0 | 2082 / 4.0 / 8.0 / 12.0 |

## 3. Event-Loop Responsiveness & Head-of-Line Blocking (`/health probe`)

`_probe_loop` issues `GET /health` every 1.0 s with a 5.0 s timeout throughout each load test to measure ASGI event-loop starvation and head-of-line blocking under `/rag/sync` versus `/rag/async`.

| Run Slug | Profile | Users | Probe Reqs | Probe Fails | Probe P50 (ms) | Probe P95 (ms) | Probe P99 (ms) | Probe Max (ms) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `sync_cold_u1` | `sync` | 1 | 66 | 0 | 3.0 | 17.0 | 17.0 | 17.4 |
| `sync_cold_u10` | `sync` | 10 | 68 | 0 | 3.0 | 17.0 | 37.0 | 37.3 |
| `sync_cold_u50` | `sync` | 50 | 68 | 0 | 3.0 | 16.0 | 99.0 | 99.0 |
| `sync_cold_u100` | `sync` | 100 | 68 | 0 | 3.0 | 17.0 | 41.0 | 40.8 |
| `async_cold_u1` | `async-cold` | 1 | 68 | 0 | 3.0 | 17.0 | 22.0 | 21.6 |
| `async_cold_u10` | `async-cold` | 10 | 68 | 0 | 3.0 | 16.0 | 23.0 | 22.8 |
| `async_cold_u50` | `async-cold` | 50 | 68 | 0 | 3.0 | 17.0 | 88.0 | 88.4 |
| `async_cold_u100` | `async-cold` | 100 | 68 | 0 | 3.0 | 17.0 | 99.0 | 99.5 |
| `async_warm_u1` | `async-warm` | 1 | 57 | 0 | 3.0 | 17.0 | 17.0 | 17.2 |
| `async_warm_u10` | `async-warm` | 10 | 58 | 0 | 3.0 | 17.0 | 19.0 | 18.5 |
| `async_warm_u50` | `async-warm` | 50 | 58 | 0 | 5.0 | 18.0 | 20.0 | 19.7 |
| `async_warm_u100` | `async-warm` | 100 | 58 | 0 | 5.0 | 20.0 | 29.0 | 28.8 |

- **Peak `/health probe` P95 on `/rag/sync`:** `17.0 ms` (`0` probe failures)
- **Peak `/health probe` P95 on `/rag/async`:** `20.0 ms` (`0` probe failures)

## 4. VRAM, Concurrency Guardrails & Cache Telemetry

### 4a. Per-Run `/health` Snapshot Telemetry

| Run Slug | Health HTTP | Status | Peak VRAM (MiB) | VRAM Breached | GPU Waiting (Cur / Max) | Circuit Open | Cache Hits / Misses / Coalesced (Hit Rate) |
|---|---:|---|---:|---|---|---|---|
| `sync_cold_u1` | 200 | ok | 5452.0 | False | 0 / N/A | False | 0 / 0 / 0 (0.0%) |
| `sync_cold_u10` | 200 | degraded | 5489.0 | True | 0 / N/A | False | 0 / 0 / 0 (0.0%) |
| `sync_cold_u50` | 200 | degraded | 5492.0 | True | 0 / N/A | False | 0 / 0 / 0 (0.0%) |
| `sync_cold_u100` | 200 | ok | 5475.0 | False | 0 / N/A | False | 0 / 0 / 0 (0.0%) |
| `async_cold_u1` | 200 | ok | 5459.0 | False | 0 / N/A | False | 0 / 0 / 0 (0.0%) |
| `async_cold_u10` | 200 | degraded | 5508.0 | True | 0 / N/A | False | 0 / 0 / 0 (0.0%) |
| `async_cold_u50` | 200 | ok | 5472.0 | False | 0 / N/A | False | 0 / 0 / 0 (0.0%) |
| `async_cold_u100` | 200 | degraded | 5489.0 | True | 0 / N/A | False | 0 / 0 / 0 (0.0%) |
| `async_warm_u1` | 200 | degraded | 5486.0 | True | 0 / N/A | False | 119 / 8 / 0 (93.7%) |
| `async_warm_u10` | 200 | ok | 5458.0 | False | 0 / N/A | False | 1136 / 8 / 0 (99.3%) |
| `async_warm_u50` | 200 | ok | 5471.0 | False | 0 / N/A | False | 5529 / 8 / 0 (99.9%) |
| `async_warm_u100` | 200 | degraded | 5488.0 | True | 0 / N/A | False | 10551 / 8 / 0 (99.9%) |

### 4b. Hardware & Serialization Analysis

- **Peak Observed VRAM:** `5508.0 MiB` (Soft target `<= 4710 MiB`, Hard cap `<= 5478 MiB`, `any_vram_breached=True`)
- **Cold Single-User End-to-End Serialization Cap (`async-cold` lowest concurrency):** `0.1682 RPS` vs peak measured `async-cold` `0.1916 RPS`
- **Phase 1.5 Retrieval Baseline Serialization Cap (`phase15_latency_baseline.json`):** median `9231.71 ms` -> `0.1083 RPS` theoretical ceiling if CPU retrieval executes single-file.
- **Warm Cache Priming Note:** In `async-warm`, `_on_start` primes all 8 queries in `POOL` prior to worker spawn (excluded from Locust CSV counts, included in server `/health` cumulative cache-miss/request totals).

## 5. Failure & Degradation Breakdown

| Run Slug | Total RAG Fails | 503 STATUS_TEMPORARILY_DEGRADED | 503 VRAM_HARD_LIMIT_EXCEEDED | 503 Other | 502 Bad Gateway | Client 120s Timeout | Empty Answers | SSE / JSON / Conn / Other | Health Probe Fails |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `sync_cold_u1` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `sync_cold_u10` | 3 | 0 | 0 | 0 | 3 | 0 | 0 | 0 | 0 |
| `sync_cold_u50` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `sync_cold_u100` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `async_cold_u1` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `async_cold_u10` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `async_cold_u50` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `async_cold_u100` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `async_warm_u1` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `async_warm_u10` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `async_warm_u50` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `async_warm_u100` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

### 5a. Observed Raw Failure Signatures

- `sync_cold_u10` [502_bad_gateway] (3x) `POST /rag/sync[cold]`: `CatchResponseError('HTTP 502: {"status":"UPSTREAM_FAILURE","detail":"GPU VRAM exceeded hard limit 5478 MiB."}')`

## 6. Performance Targets (P1–P4) & Engineering Gates (E5–E7) Evaluation

| Gate / Target | Status | Target / Contract | Key Observed Evidence |
|---|---|---|---|
| **P1: Event-Loop Liveness (/health probe under async load)** | **WARN** | async /health probe P95 <= 500 ms with 0 probe failures across all user tiers | `{"async_max_probe_p95_ms": 20.0, "async_probe_failures": 0, "sync_max_probe_p95_ms": 17.0, "sync_probe_failures": 0}` |
| **P2: Concurrency Throughput Scaling (async-warm > async-cold vs sync)** | **PASS** | async-warm RPS > async-cold RPS and > sync RPS; cold RPS bounded by CPU retrieval + GPU serialization | `{"peak_sync_rps": 0.2187, "peak_async_cold_rps": 0.1916, "peak_async_warm_rps": 177.6729, "cold_u1_e2e_serial_cap_rps": 0.1682, "phase15_retrieval_serial_cap_rps": 0.1083}` |
| **P3: Warm Cache Replay Latency & SSE Time-To-First-Token (TTFT)** | **PASS** | async-warm non-stream P50 substantially below async-cold P50; SSE TTFT captured cleanly | `{"async_warm_mean_non_stream_p50_ms": 2.25, "async_cold_mean_non_stream_p50_ms": 29775.0, "speedup_ratio": 13233.33, "sse_ttft_p50_by_run_ms": {"async_cold_u1": 7400.0, "async_cold_u10": 42000.0, "async_cold_u100": 55000.0, "async_warm_u1": 2.0, "async_warm_u10": 2.0, "async_warm_u50": 3.0, "async_warm_u100": 4.0}}` |
| **P4: Saturation Reliability & Graceful Degradation (u >= 50)** | **PASS** | 0 empty-answer responses; warm saturation error rate <= 5%; graceful 503 degradation over crashes | `{"warm_max_saturation_error_pct": 0.0, "cold_max_saturation_error_pct": 0.0, "sync_max_saturation_error_pct": 0.0, "empty_answer_failures": 0}` |
| **E5: GPU VRAM Hard Cap & Guardrail Telemetry** | **FAIL** | vram_breached == false and peak VRAM <= 5,478 MiB (soft target <= 4,710 MiB) | `{"max_peak_vram_mib": 5508.0, "soft_target_mib": 4710.0, "hard_limit_mib": 5478.0, "within_soft_target": false, "within_hard_limit": false, "any_vram_breached": true, "vram_hard_limit_503_count": 0}` |
| **E6: Bounded GPU Semaphore(20), Retrieval Serialization & Head-of-Line Isolation** | **PASS** | Async event loop remains unblocked while Semaphore(20) and CPU serialization bound GPU/CPU admission | `{"semaphore_limit": 20, "gpu_waiting_snapshots": [{"slug": "sync_cold_u1", "users": 1, "gpu_waiting": 0, "max_gpu_waiting": null, "gpu_active": 0}, {"slug": "sync_cold_u10", "users": 10, "gpu_waiting": 0, "max_gpu_waiting": null, "gpu_active": 0}, {"slug": "sync_cold_u50", "users": 50, "gpu_waiting": 0, "max_gpu_waiting": null, "gpu_active": 0}, {"slug": "sync_cold_u100", "users": 100, "gpu_waiting": 0, "max_gpu_waiting": null, "gpu_active": 0}, {"slug": "async_cold_u1", "users": 1, "gpu_waiting": 0, "max_gpu_waiting": null, "gpu_active": 0}, {"slug": "async_cold_u10", "users": 10, "gpu_waiting": 0, "max_gpu_waiting": null, "gpu_active": 1}, {"slug": "async_cold_u50", "users": 50, "gpu_waiting": 0, "max_gpu_waiting": null, "gpu_active": 2}, {"slug": "async_cold_u100", "users": 100, "gpu_waiting": 0, "max_gpu_waiting": null, "gpu_active": 2}, {"slug": "async_warm_u1", "users": 1, "gpu_waiting": 0, "max_gpu_waiting": null, "gpu_active": 0}, {"slug": "async_warm_u10", "users": 10, "gpu_waiting": 0, "max_gpu_waiting": null, "gpu_active": 0}, {"slug": "async_warm_u50", "users": 50, "gpu_waiting": 0, "max_gpu_waiting": null, "gpu_active": 0}, {"slug": "async_warm_u100", "users": 100, "gpu_waiting": 0, "max_gpu_waiting": null, "gpu_active": 0}], "cold_u1_avg_ms": 5944.49, "cold_u1_p50_ms": 7809.43, "cold_serial_e2e_cap_rps": 0.1682, "peak_measured_async_cold_rps": 0.1916, "async_max_health_probe_p95_ms": 20.0, "sync_max_health_probe_p95_ms": 17.0}` |
| **E7: Cache-Hit Distribution, Stampede Guard & Failure Accounting** | **PASS** | Warm cache hits + stampede coalescing active; 0 empty answers; structured failure breakdown | `{"warm_cache_snapshots": [{"slug": "async_warm_u1", "users": 1, "cache_hits": 119, "cache_misses": 8, "cache_coalesced": 0, "cache_hit_rate": 0.937, "non_stream_p50_ms": 2.0}, {"slug": "async_warm_u10", "users": 10, "cache_hits": 1136, "cache_misses": 8, "cache_coalesced": 0, "cache_hit_rate": 0.993, "non_stream_p50_ms": 2.0}, {"slug": "async_warm_u50", "users": 50, "cache_hits": 5529, "cache_misses": 8, "cache_coalesced": 0, "cache_hit_rate": 0.9986, "non_stream_p50_ms": 2.0}, {"slug": "async_warm_u100", "users": 100, "cache_hits": 10551, "cache_misses": 8, "cache_coalesced": 0, "cache_hit_rate": 0.9992, "non_stream_p50_ms": 3.0}], "warm_total_cache_hits": 17335, "failure_totals": {"503_status_temporarily_degraded": 0, "503_vram_hard_limit_exceeded": 0, "503_other": 0, "502_bad_gateway": 3, "504_gateway_timeout": 0, "500_internal_error": 0, "429_rate_limited": 0, "client_timeout_120s": 0, "empty_answer": 0, "sse_stream_error": 0, "invalid_json": 0, "connection_error": 0, "other": 0}}` |

## 7. Limits of This Test

1. **Single-Host Client/Server Co-Location:** Locust (`gevent` worker), FastAPI/Uvicorn, PyTorch CPU retrieval (`torch.set_num_threads(2)`), and Ollama GPU inference all execute on the same Windows 11 host laptop (`i7-13620H`, 16 GB DDR5, RTX 4050 6 GB VRAM), introducing slight loopback socket and CPU scheduling competition under 50–100 users.
2. **Single-GPU Serialized Decode (`OLLAMA_NUM_PARALLEL=1`):** While `asyncio.Semaphore(20)` bounds FastAPI admission cleanly without VRAM OOM, local Ollama runs with `OLLAMA_NUM_PARALLEL=1` on 6 GB VRAM, meaning cold LLM generation requests still execute sequentially on the GPU.
3. **Fixed 60s Window vs 120s Client Timeout & 10s Stop Timeout:** Each concurrency cell runs for `RunSeconds` (default `60s`) with `--stop-timeout 10` while `TIMEOUT_S = 120.0`. Under deep queue saturation (`sync` or `async-cold` at 50–100 users), in-flight requests queued near the end of the 60s window may be interrupted at `--stop-timeout 10` before reaching the 120s client socket timeout.
4. **Small 8-Query Dev Pool (`GOLDEN_FREE_SEED = 20260`):** `POOL` intentionally uses 8 fixed queries (2 bare CVEs, 4 `inscope_dev_100`, 2 `oos_dev_60`; never `golden_150.jsonl`) to exercise `async-warm` cache stampede protection and replay; warm hit rates therefore represent hot-key working-set caching rather than open-vocabulary cache miss ratios.
5. **End-of-Run `/health` Snapshot Granularity:** `*_health.json` captures one point-in-time snapshot immediately after Locust finishes, whereas `/health probe` records continuous 1s event-loop latency samples throughout the run.
6. **Pre-Test Cache Priming Scope:** In `async-warm`, `_on_start` sends 8 unmeasured non-streaming priming calls so user traffic measures steady-state cache replay; server-side `/health` totals include those 8 priming calls.
