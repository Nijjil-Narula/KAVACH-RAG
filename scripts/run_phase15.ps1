<#
.SYNOPSIS
    KAVACH-RAG Phase 1.5 Hotfix End-to-End CPU-Only Execution Runner.
.DESCRIPTION
    Executes all Phase 1.5 calibration, CPU-only profiling, precision locking,
    abstention calibration, 6-variant ablation, and full pytest verification under
    strict deterministic CPU-only runtime settings.
#>

$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
Push-Location $ProjectRoot

try {
    $GpuVenv = Join-Path $ProjectRoot ".venv_gpu"
    if (Test-Path $GpuVenv) {
        throw "Forbidden .venv_gpu directory detected at $GpuVenv. Phase 1.5 must run strict CPU-only."
    }

    $Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
    if (-not (Test-Path $Python)) {
        throw "Python virtual environment not found at $Python"
    }

    $env:PYTHONUTF8 = "1"
    $env:CUDA_VISIBLE_DEVICES = ""
    $env:OMP_NUM_THREADS = "2"
    $env:MKL_NUM_THREADS = "2"
    $env:OPENBLAS_NUM_THREADS = "2"
    $env:TOKENIZERS_PARALLELISM = "false"
    $env:KMP_DUPLICATE_LIB_OK = "TRUE"
    if (-not $env:HF_HOME) {
        $env:HF_HOME = Join-Path $ProjectRoot "models\hf_cache"
    }

    Write-Host "==========================================================" -ForegroundColor Cyan
    Write-Host " KAVACH-RAG Phase 1.5 Hotfix: Strict CPU-Only Execution" -ForegroundColor Cyan
    Write-Host " Root:   $ProjectRoot" -ForegroundColor Cyan
    Write-Host " Python: $Python" -ForegroundColor Cyan
    Write-Host " CUDA_VISIBLE_DEVICES='$env:CUDA_VISIBLE_DEVICES' THREADS=2" -ForegroundColor Cyan
    Write-Host "==========================================================" -ForegroundColor Cyan

    function Invoke-KavachStep {
        param(
            [Parameter(Mandatory = $true)][string]$StepName,
            [Parameter(Mandatory = $true)][string[]]$ModuleArgs
        )
        Write-Host "`n>>> [$StepName] Running: $Python $($ModuleArgs -join ' ')" -ForegroundColor Yellow
        $sw = [System.Diagnostics.Stopwatch]::StartNew()
        & $Python @ModuleArgs
        $exitCode = $LASTEXITCODE
        $sw.Stop()
        if ($exitCode -ne 0) {
            throw "Step '$StepName' failed with exit code $exitCode"
        }
        Write-Host "<<< [$StepName] Completed cleanly in $($sw.Elapsed.ToString('mm\:ss\.ff'))" -ForegroundColor Green
    }

    Invoke-KavachStep -StepName "Stage 0: Devset Integrity Gate" -ModuleArgs @("-m", "kavach.retrieval.devsets", "--validate")
    Invoke-KavachStep -StepName "Stage 1: CPU Thread & Memory Baseline Audit" -ModuleArgs @("-m", "kavach.retrieval.profiling", "--audit")
    Invoke-KavachStep -StepName "Stage 2a: P-Core / E-Core Affinity Benchmark" -ModuleArgs @("-m", "kavach.retrieval.profiling", "--affinity")
    Invoke-KavachStep -StepName "Stage 2b: Precision Matrix Benchmark" -ModuleArgs @("-m", "kavach.retrieval.profiling", "--matrix")
    Invoke-KavachStep -StepName "Stage 3: Quality Gates & Precision Lock" -ModuleArgs @("-m", "kavach.retrieval.profiling", "--gates")
    Invoke-KavachStep -StepName "Stage 4: Abstention Calibration (Dev Sets Only)" -ModuleArgs @("-m", "kavach.retrieval.calibration", "--collect", "--select", "--report")
    Invoke-KavachStep -StepName "Stage 5: Calibrated End-to-End Baseline" -ModuleArgs @("-m", "kavach.retrieval.profiling", "--baseline")
    Invoke-KavachStep -StepName "Stage 6: 6-Variant CPU-Only Ablation (Fresh)" -ModuleArgs @("-m", "kavach.retrieval.ablation", "--fresh")
    Invoke-KavachStep -StepName "Stage 7: Full Regression Test Suite" -ModuleArgs @("-m", "pytest", "-q")

    Write-Host "`n==========================================================" -ForegroundColor Green
    Write-Host " ALL PHASE 1.5 STEPS AND VERIFICATION GATES PASSED" -ForegroundColor Green
    Write-Host "==========================================================" -ForegroundColor Green
}
finally {
    Pop-Location
}
