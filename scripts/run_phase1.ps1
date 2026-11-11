<#
.SYNOPSIS
    Run the full KAVACH-RAG Phase 1 CPU-bound retrieval pipeline in deterministic order.
.DESCRIPTION
    Enforces CPU-only hardware isolation and UTF-8 encoding, then executes
    preflight -> acquire -> ingest (w512 & w256) -> golden -> router training ->
    indexing -> ablation -> pytest using the local virtual environment interpreter.
#>
[CmdletBinding()]
param(
    [switch]$SkipPreflight,
    [switch]$SkipAcquire
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
Push-Location $ProjectRoot

try {
    # Hardware isolation and UTF-8 environment contract
    $env:PYTHONUTF8 = "1"
    $env:CUDA_VISIBLE_DEVICES = ""
    $env:OMP_NUM_THREADS = "2"
    $env:MKL_NUM_THREADS = "2"
    $env:OPENBLAS_NUM_THREADS = "2"
    $env:TOKENIZERS_PARALLELISM = "false"
    $env:KMP_DUPLICATE_LIB_OK = "TRUE"
    $env:HF_HOME = Join-Path $ProjectRoot "models\hf_cache"

    $Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
    if (-not (Test-Path $Python)) {
        throw "Virtual environment Python not found at: $Python"
    }

    function Invoke-KavachStep {
        param(
            [string]$Label,
            [string[]]$PyArgs
        )
        Write-Host "`n=== [$Label] ===" -ForegroundColor Cyan
        & $Python @PyArgs
        if ($LASTEXITCODE -ne 0) {
            throw "Step '$Label' failed with exit code $LASTEXITCODE."
        }
    }

    if (-not $SkipPreflight) {
        Invoke-KavachStep -Label "1/9 Preflight Model & Throughput Check" -PyArgs @("-m", "kavach.retrieval.preflight")
    }

    if (-not $SkipAcquire) {
        Invoke-KavachStep -Label "2/9 Raw Corpus Acquisition" -PyArgs @("-m", "kavach.retrieval.acquire")
    }

    Invoke-KavachStep -Label "3/9 Corpus Ingest (w=512, overlap=64)" -PyArgs @("-m", "kavach.retrieval.ingest", "--window", "512", "--overlap", "64")
    Invoke-KavachStep -Label "4/9 Corpus Ingest (w=256, overlap=32)" -PyArgs @("-m", "kavach.retrieval.ingest", "--window", "256", "--overlap", "32")

    Invoke-KavachStep -Label "5/9a Build Golden Benchmark (150 items)" -PyArgs @("-m", "kavach.retrieval.golden", "--build")
    Invoke-KavachStep -Label "5/9b Validate Golden Benchmark" -PyArgs @("-m", "kavach.retrieval.golden", "--validate")

    Invoke-KavachStep -Label "6/9 Train & Verify-Disjoint Tier 1 AgentIntentRouter" -PyArgs @("-m", "kavach.retrieval.train_router", "--verify-disjoint")

    Invoke-KavachStep -Label "7/9 Build FAISS SQ8 & Baseline Indices" -PyArgs @("-m", "kavach.retrieval.build_index", "--build-threads", "2")

    Invoke-KavachStep -Label "8/9 Run 6-Run Retrieval Ablation & Golden Router Eval" -PyArgs @("-m", "kavach.retrieval.ablation")

    Invoke-KavachStep -Label "9/9 Run Final Test Suite" -PyArgs @("-m", "pytest", "-q")

    Write-Host "`nPhase 1 pipeline completed successfully." -ForegroundColor Green
}
finally {
    Pop-Location
}
