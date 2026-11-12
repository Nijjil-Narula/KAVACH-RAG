# KAVACH-RAG

Retrieval-augmented question answering for cybersecurity. It answers from NVD CVE records, MITRE ATT&CK techniques and the OWASP Top 10 for LLM applications (v1.1 and v2.0), cites the chunk behind every claim, keeps conflicting standard versions separate, and refuses with `INSUFFICIENT_CONTEXT` when the evidence is missing. It runs on one Windows 11 laptop (Intel i7-13620H, 16 GB RAM, RTX 4050 6 GB) with no paid services.

## Architecture

```
user query
  |
  v
Tier 1  intent router        exact CVE/CWE/ATT&CK id -> direct lookup; off-topic -> refusal   CPU
  |
  v
Tier 2  bi-encoder           F2LLM-v2-0.6B, 1024-d vectors cut to 512-d, L2-normalized        CPU
  |
  v
Tier 3  FAISS SQ8 index      8-bit scalar quantization, top-8 candidates                      CPU, RAM
  |
  v
Tier 4  cross-encoder        ettin-reranker-150m-v1, abstention gate T=7.0, top-3 chunks      CPU
  |
  v
Tier 5  generator            Qwen2.5-7B-Instruct Q4_K_M via Ollama, num_ctx 4096              GPU
  |
  v
FastAPI service: /rag/sync, /rag/async (SSE), /health, /tasks/ingest
Offline: Tier 6 judge cascade (API only) -> human calibration -> CI gate
```

| Tier | Component | Runs on | Notes |
|---|---|---|---|
| 1-4 | Retrieval pipeline | CPU, 2 threads | No GPU memory is used at serving time |
| 5 | Ollama, one decode slot | RTX 4050 | `OLLAMA_NUM_PARALLEL=1`, context clamped to 4096 |
| 6 | Judge cascade | Remote APIs | Cross-family: Gemini models and a Groq-hosted open model, never the generator family |

Serving design: retrieval is serialized behind a single lock; generation entry is bounded by `asyncio.Semaphore(20)`; repeated queries are served from a TTL cache with per-key single flight; a circuit breaker returns `STATUS_TEMPORARILY_DEGRADED` after five consecutive upstream failures; a VRAM guard refuses generation above the configured limit.

## Repository layout

```text
.github/                  CI workflow (rag_eval_regression.yml)
app/                      Package stub (FastAPI service lives in kavach/serving/app.py)
benchmarks/               Locust load-test suite (locustfile.py) and report compiler (compile_report.py)
data/                     Golden benchmark (golden/), calibration sets (calibration/), raw downloads (raw/), derived files (processed/)
deliverables/             Generated release PDFs (reports 01 to 04)
kavach/                   Core package: retrieval/, generation/, serving/, evaluation/, config.py, exceptions.py, runtime_guard.py
logs/                     Runtime and evaluation logs
models/                   FAISS SQ8 index, DeBERTa router weights, Modelfile.tier5, precision_lock.json, abstain_calibration.json
reports/                  Retrieval ablation, generation evaluation, load-test, calibration, and baseline audit reports
scripts/                  PowerShell runners, Ollama/server launchers, calibration CLI, and PDF deliverable builder
tests/                    Offline and live pytest test suites
requirements*.txt         Base, serving, and evaluation dependency specifications and pinned lock files
.gitattributes            Git LFS tracking rules
.gitignore                Ignore rules for caches, logs, and regenerable derived artifacts
```

## Setup

Requirements: Windows 11, PowerShell, Python 3.12, Git with Git LFS, Ollama, an NVIDIA GPU with 6 GB for the generator.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.lock.txt
.\.venv\Scripts\python.exe -m pip install -r requirements.serving.lock.txt -r requirements.evaluation.lock.txt
git lfs install
git lfs pull
```

`requirements.txt` pins the CPU-only PyTorch wheel (`torch==2.15.0+cpu`, verified in `requirements.lock.txt`) via the PyTorch CPU index (`--extra-index-url https://download.pytorch.org/whl/cpu`). To install or verify that exact CPU build directly:

```powershell
.\.venv\Scripts\python.exe -m pip install --extra-index-url https://download.pytorch.org/whl/cpu torch==2.15.0+cpu
```

Set the judge keys once as user environment variables. Never commit them.

```powershell
[Environment]::SetEnvironmentVariable("GEMINI_API_KEY", "<your key>", "User")
[Environment]::SetEnvironmentVariable("GROQ_API_KEY", "<your key>", "User")
```

## Build the retrieval corpus and indexes

```powershell
.\scripts\run_phase1.ps1 [-SkipPreflight] [-SkipAcquire]
.\scripts\run_phase15.ps1
```

`scripts\run_phase1.ps1` enforces CPU-only isolation (`PYTHONUTF8=1`, `CUDA_VISIBLE_DEVICES=""`, `OMP_NUM_THREADS=2`, `MKL_NUM_THREADS=2`, `OPENBLAS_NUM_THREADS=2`, `TOKENIZERS_PARALLELISM=false`, `HF_HOME=models\hf_cache`) and runs in order: preflight (`kavach.retrieval.preflight`), raw corpus acquisition (`kavach.retrieval.acquire`), chunk ingestion for `--window 512 --overlap 64` and `--window 256 --overlap 32` (`kavach.retrieval.ingest`), golden benchmark build and validation (`kavach.retrieval.golden --build` and `--validate`), disjoint router training (`kavach.retrieval.train_router --verify-disjoint`), FAISS SQ8 and baseline indexing (`kavach.retrieval.build_index --build-threads 2`), retrieval ablation (`kavach.retrieval.ablation`), and `pytest -q`. Pass `-SkipPreflight` to skip the preflight check and `-SkipAcquire` to reuse existing raw downloads. `kavach.retrieval.acquire` requires network access. Derived files under `data/processed/` are regenerable and not tracked (except the two evaluation files kept for the CI gate).

`scripts\run_phase15.ps1` (no switches) runs Phase 1.5 CPU-only profiling, precision locking (`models/precision_lock.json`), abstention threshold calibration (`models/abstain_calibration.json`), a fresh 6-variant ablation (`kavach.retrieval.ablation --fresh`), and `pytest -q`.

## Run the generator

```powershell
.\scripts\start_ollama.ps1 [-Restart] [-FlashAttention]
ollama create kavach-tier5 -f models\Modelfile.tier5
.\scripts\run_phase2.ps1 [-RestartOllama]
```

`scripts\start_ollama.ps1` strips `CUDA_VISIBLE_DEVICES`, `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, and `OPENBLAS_NUM_THREADS` before launching `ollama serve` on `127.0.0.1:11434` with `OLLAMA_NUM_PARALLEL=1`, `OLLAMA_MAX_LOADED_MODELS=1`, and `OLLAMA_KEEP_ALIVE=30m` (optional `-Restart` stops existing Ollama processes first; `-FlashAttention` sets `OLLAMA_FLASH_ATTENTION=1`), then restores the caller's environment. `models\Modelfile.tier5` builds `kavach-tier5` from `qwen2.5:7b-instruct-q4_K_M` with `num_ctx 4096`, `num_predict 512`, `temperature 0`, `top_p 1`, `repeat_penalty 1.0`, `seed 0`, and `num_gpu 999`.

`scripts\run_phase2.ps1` (optional `-RestartOllama`) starts Ollama, runs Phase 2 preflight, executes live generation tests, freezes the prompt contract, generates the 150-item batch, runs Phase 2 evaluation, runs the live pytest suite, and verifies frozen SHA-256 hashes. To run the prompt freeze and 150-item batch directly:

```powershell
.\.venv\Scripts\python.exe -m kavach.generation.preflight
.\.venv\Scripts\python.exe -m kavach.generation.batch_generate --freeze
.\.venv\Scripts\python.exe -m kavach.generation.batch_generate --resume
```

## Evaluate

The golden dataset has 150 questions: 60 factual, 45 multi-hop, 15 version-conflict traps, 10 out-of-scope, 10 malformed and 10 context-noise.

```powershell
.\.venv\Scripts\python.exe -m kavach.generation.evaluate --run --resume
.\.venv\Scripts\python.exe -m kavach.generation.evaluate --report
.\.venv\Scripts\python.exe -m kavach.evaluation.cascade_judge --ping
.\.venv\Scripts\python.exe -m kavach.evaluation.cascade_judge --limit 5
.\.venv\Scripts\python.exe -m kavach.evaluation.cascade_judge
.\.venv\Scripts\python.exe scripts\calibrate_judge.py
```

The cascade grades every item with a first judge and escalates borderline items to a three-model panel (majority vote on the verdict, minimum on the scores). Progress is checkpointed after every item, so an interrupted run resumes without repeating calls. Calibration draws 30 items, writes a blind review set to `data/calibration/human_review_set_30.json`, and scores your labels in `data/calibration/human_expert_labels_30.json` with Cohen's kappa. No kappa is reported until all 30 labels are filled.

## Serve and load-test

```powershell
.\scripts\start_ollama.ps1 -Restart
.\scripts\start_server.ps1 -Restart
.\scripts\start_server.ps1 -Stop
```

| Endpoint | Purpose |
|---|---|
| `POST /rag/async` | Cached, bounded, streaming (`stream: true` returns SSE events `ttft`, `token`, `metrics`, `done`, `error`) |
| `POST /rag/sync` | Blocking baseline used only for comparison |
| `GET /health` | Event-loop probe plus VRAM, cache, circuit and queue state |
| `POST /tasks/ingest` | Read-only corpus integrity job; poll `GET /tasks/{job_id}` |

```powershell
Invoke-RestMethod -Uri http://127.0.0.1:8000/rag/async -Method Post -ContentType "application/json" `
  -Body '{"query":"What is the CVSS v3.1 base score of CVE-2024-3094?","stream":false,"use_cache":true}'
```

The load test restarts the server before every run and takes about 30 minutes for the full matrix:

```powershell
.\scripts\run_locust.ps1 -Users 1,10,50,100 -RunSeconds 60
.\.venv\Scripts\python.exe benchmarks\compile_report.py
```

## Test and CI

```powershell
.\.venv\Scripts\python.exe -m pytest -q
$env:KAVACH_LIVE = "1"; .\.venv\Scripts\python.exe -m pytest -q -rs; Remove-Item Env:KAVACH_LIVE
.\.venv\Scripts\python.exe -m pytest -q --noconftest tests\test_ci_regression.py
```

The CI workflow in `.github/workflows` runs only the offline regression gate over the stored evaluation records: 150 items, average faithfulness at least 0.85, 100% out-of-scope refusals, at most 12 FAIL verdicts, and kappa at least 0.70. Overall context relevance is reported but not gated because it is limited by retrieval.

## Measured results

All values were read from the checked-in reports when this file was written.

| Area | Metric | Value | Source |
|---|---|---|---|
| Retrieval, Run 5 (k=3) | Document MRR / chunk MRR | 0.8726 (gated 0.8690) / 0.6762 | reports/retrieval_ablation_report.md |
| Generation (single-stage judge) | Faithfulness / relevance / citation validity | 0.8628 / 0.5679 (context-sufficient 0.9236) / 1.0000 | reports/generation_evaluation_summary.json |
| Cascade | Faithfulness all items / answered only / FAIL count / out-of-scope refusals | 0.8366 / 0.7336 (n=92) / 16 FAIL (115 PASS, 19 PARTIAL) / 10/10 (100%) | data/processed/tier6_eval_checkpoints.jsonl |
| Judge calibration | Cohen's kappa | 0.8929 (95% bootstrap CI [0.7207, 1.0], n=30, status SCORED) | reports/human_calibration_30.json |
| Serving | Async warm, 100 users: P95 and error rate | 6.29 ms P95, 0.00% error rate (177.67 RPS) | reports/phase3_load_test_summary.json |
| Serving | Async cold: peak RPS across runs | 0.1916 RPS (at 10 users) | reports/phase3_load_test_summary.json |
| GPU | Peak memory.used vs hard limit | 5508 MiB peak vs 5478 MiB hard limit (6/12 runs breached, 0 VRAM failures) | reports/phase3_load_test_summary.json |

## Known limits

- Semantic retrieval on CPU takes seconds (`8,714.65 ms` mean in Run 5; `27.77 s` mean on `CONTEXT_NOISE` queries) and is serialized behind a single lock, bounding cold throughput to `0.1916 RPS` peak.
- Multi-hop (`0.4667` document Recall@3) and context-noise (`0.1000` document Recall@3) retrieval recall limit overall context relevance (`0.5679` in Phase 2, `0.5893` in-scope in Phase 4 vs `0.9792` on context-sufficient items).
- On the current 150-item cascade evaluation, the offline CI gate reports two failing thresholds: all-item faithfulness is `0.8366` (below the `0.85` floor) and `FAIL` verdicts total `16` (above the `12` ceiling), while out-of-scope refusals (`10/10`) and human calibration (`kappa = 0.8929 >= 0.70`) pass.
- The GPU VRAM margin is small (`5508 MiB` peak `memory.used` vs `5478 MiB` hard limit; `6` of `12` runs flagged `vram_breached`), and other desktop applications on the shared host contribute to `memory.used`.
- Load-test cells run for `60 s` against a `120 s` client timeout, so high-concurrency cold runs cut off queued requests near the end; the `8`-query warm pool (`177.67 RPS`, `6.29 ms` P95 at 100 users) measures fixed-pool cache replay only.
- Human calibration (`kappa = 0.8929`) uses `30` items stratified to over-sample non-PASS judge verdicts, yielding a wide bootstrap interval (`[0.7207, 1.0]`).
