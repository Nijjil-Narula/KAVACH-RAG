<#
.SYNOPSIS
    KAVACH-RAG Phase 2 End-to-End Generation, Evaluation, and Regression Runner.
.DESCRIPTION
    Executes Phase 2 Ollama startup, preflight gating, live smoke validation,
    prompt freeze, batch synthesis over retrieved_contexts_150.json, two-stage
    evaluation, full pytest regression suite, and Stage 0 SHA-256 baseline audit.

    Ensures Ollama is never started from a shell with CUDA_VISIBLE_DEVICES set,
    while Python processes continue enforcing CPU-only PyTorch via kavach.runtime_guard.
#>
[CmdletBinding()]
param(
    [switch]$RestartOllama
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
Push-Location $ProjectRoot

try {
    $GpuVenv = Join-Path $ProjectRoot ".venv_gpu"
    if (Test-Path $GpuVenv) {
        throw "Forbidden .venv_gpu directory detected at $GpuVenv. Python must remain CPU-only."
    }

    $Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
    if (-not (Test-Path $Python)) {
        throw "Python virtual environment not found at $Python"
    }

    # Ensure Ollama never inherits CUDA_VISIBLE_DEVICES="" from the parent shell.
    Remove-Item Env:\CUDA_VISIBLE_DEVICES -ErrorAction SilentlyContinue

    $env:PYTHONUTF8 = "1"
    $env:HF_HOME = Join-Path $ProjectRoot "models\hf_cache"
    $env:TOKENIZERS_PARALLELISM = "false"

    Write-Host "==========================================================" -ForegroundColor Cyan
    Write-Host " KAVACH-RAG Phase 2: Generation & Grounded Synthesis" -ForegroundColor Cyan
    Write-Host " Root:    $ProjectRoot" -ForegroundColor Cyan
    Write-Host " Python:  $Python" -ForegroundColor Cyan
    Write-Host " HF_HOME: $env:HF_HOME" -ForegroundColor Cyan
    Write-Host "==========================================================" -ForegroundColor Cyan

    function Invoke-KavachPythonStep {
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

    # Stage 1a: Ensure Ollama server is started cleanly without CUDA_VISIBLE_DEVICES leakage
    $StartOllamaScript = Join-Path $PSScriptRoot "start_ollama.ps1"
    Write-Host "`n>>> [Stage 1a: Ollama Server Startup] Running: $StartOllamaScript" -ForegroundColor Yellow
    if ($RestartOllama) {
        & $StartOllamaScript -Restart
    }
    else {
        & $StartOllamaScript
    }
    if ($LASTEXITCODE -ne 0) {
        throw "Ollama startup script failed with exit code $LASTEXITCODE"
    }

    # Stage 1b / Step 1: Phase 2 GPU/Ollama/Prompt/Isolation Preflight
    Invoke-KavachPythonStep -StepName "Stage 1b: Phase 2 Preflight Gate" -ModuleArgs @("-m", "kavach.generation.preflight")

    # Stage 3 Step 1: Live generation tests BEFORE prompt freeze (sequential)
    $SavedKavachLive = [Environment]::GetEnvironmentVariable("KAVACH_LIVE", "Process")
    try {
        $env:KAVACH_LIVE = "1"
        Invoke-KavachPythonStep -StepName "Stage 3 Step 1: Live Generation Tests" -ModuleArgs @("-m", "pytest", "-q", "tests/test_generation.py", "-rs")
    }
    finally {
        [Environment]::SetEnvironmentVariable("KAVACH_LIVE", $SavedKavachLive, "Process")
    }

    # Stage 3 Step 2: Freeze prompt contract & live model digest
    Invoke-KavachPythonStep -StepName "Stage 3 Step 2: Prompt Freeze" -ModuleArgs @("-m", "kavach.generation.batch_generate", "--freeze")

    # Stage 4: Batch synthesis over retrieved_contexts_150.json
    Invoke-KavachPythonStep -StepName "Stage 4: Golden 150 Batch Synthesis" -ModuleArgs @("-m", "kavach.generation.batch_generate", "--resume")

    # Stage 5 Step 1: Two-stage evaluation & judgment
    Invoke-KavachPythonStep -StepName "Stage 5 Step 1: Evaluation Run" -ModuleArgs @("-m", "kavach.generation.evaluate", "--run", "--resume")

    # Stage 5 Step 2: Evaluation summary & 9-section report
    Invoke-KavachPythonStep -StepName "Stage 5 Step 2: Evaluation Report" -ModuleArgs @("-m", "kavach.generation.evaluate", "--report")

    # Stage 6a: Full pytest suite with KAVACH_LIVE=1 (restoring env var afterward)
    $SavedKavachLiveStage6 = [Environment]::GetEnvironmentVariable("KAVACH_LIVE", "Process")
    try {
        $env:KAVACH_LIVE = "1"
        Invoke-KavachPythonStep -StepName "Stage 6a: Full Pytest Suite (Live)" -ModuleArgs @("-m", "pytest", "-q", "-rs")
    }
    finally {
        [Environment]::SetEnvironmentVariable("KAVACH_LIVE", $SavedKavachLiveStage6, "Process")
    }

    # Stage 6b: Frozen SHA-256 & Amendment 1 Baseline Verification
    Invoke-KavachPythonStep -StepName "Stage 6b: Frozen SHA-256 & Amendment 1 Audit" -ModuleArgs @("-m", "kavach.generation.preflight", "--sha256-only")

    Write-Host "`n==========================================================" -ForegroundColor Green
    Write-Host " ALL PHASE 2 STEPS AND VERIFICATION GATES PASSED" -ForegroundColor Green
    Write-Host "==========================================================" -ForegroundColor Green
}
finally {
    Pop-Location
}
