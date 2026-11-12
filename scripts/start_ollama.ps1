<#
.SYNOPSIS
    Start or restart the local Ollama server with Phase 2 GPU residency settings.
.DESCRIPTION
    Strips CPU-only isolation variables (CUDA_VISIBLE_DEVICES, OMP_NUM_THREADS,
    MKL_NUM_THREADS, OPENBLAS_NUM_THREADS) before launching `ollama serve` so
    Ollama never sees CUDA_VISIBLE_DEVICES="", then restores the caller's process
    environment in a finally block.
#>
[CmdletBinding()]
param(
    [switch]$Restart,
    [switch]$FlashAttention
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$LogsDir = Join-Path $ProjectRoot "logs"
$StdoutLog = Join-Path $LogsDir "ollama_server.log"
$StderrLog = Join-Path $LogsDir "ollama_server.err.log"
$VersionUrl = "http://127.0.0.1:11434/api/version"

function Test-OllamaAlive {
    param([string]$Url)
    try {
        $resp = Invoke-WebRequest -Uri $Url -Method Get -TimeoutSec 2 -UseBasicParsing
        return ($resp.StatusCode -eq 200)
    }
    catch {
        return $false
    }
}

function Resolve-OllamaExe {
    $cmd = Get-Command "ollama" -ErrorAction SilentlyContinue
    if ($cmd -and $cmd.Source) {
        return $cmd.Source
    }
    $localFallback = Join-Path $env:LOCALAPPDATA "Programs\Ollama\ollama.exe"
    if (Test-Path $localFallback) {
        return $localFallback
    }
    throw "Ollama executable not found on PATH or at $localFallback"
}

$EnvVarsToStrip = @(
    "CUDA_VISIBLE_DEVICES",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS"
)

$SavedEnv = @{}
foreach ($name in $EnvVarsToStrip) {
    $existing = [Environment]::GetEnvironmentVariable($name, "Process")
    $SavedEnv[$name] = $existing
}

try {
    foreach ($name in $EnvVarsToStrip) {
        [Environment]::SetEnvironmentVariable($name, $null, "Process")
    }

    if (-not $Restart) {
        if (Test-OllamaAlive -Url $VersionUrl) {
            Write-Warning (
                "Ollama daemon is already running at $VersionUrl; leaving pre-existing " +
                "daemon running. OLLAMA_NUM_PARALLEL=1 could not be verified on the " +
                "pre-existing daemon (pass -Restart to enforce)."
            )
            exit 0
        }
    }
    else {
        $targets = Get-Process -Name "ollama", "ollama app" -ErrorAction SilentlyContinue
        if ($targets) {
            Write-Host "Stopping existing Ollama process(es)..." -ForegroundColor Yellow
            $targets | Stop-Process -Force -ErrorAction SilentlyContinue
            $stopDeadline = (Get-Date).AddSeconds(10)
            while ((Get-Date) -lt $stopDeadline) {
                $remaining = Get-Process -Name "ollama", "ollama app" -ErrorAction SilentlyContinue
                if (-not $remaining) {
                    break
                }
                Start-Sleep -Milliseconds 250
            }
        }
    }

    New-Item -ItemType Directory -Path $LogsDir -Force | Out-Null
    $OllamaExe = Resolve-OllamaExe

    $env:OLLAMA_NUM_PARALLEL = "1"
    $env:OLLAMA_MAX_LOADED_MODELS = "1"
    $env:OLLAMA_KEEP_ALIVE = "30m"
    $env:OLLAMA_HOST = "127.0.0.1:11434"
    if ($FlashAttention) {
        $env:OLLAMA_FLASH_ATTENTION = "1"
    }

    Write-Host "Starting Ollama server ($OllamaExe serve)..." -ForegroundColor Cyan
    $proc = Start-Process `
        -FilePath $OllamaExe `
        -ArgumentList "serve" `
        -WindowStyle Hidden `
        -RedirectStandardOutput $StdoutLog `
        -RedirectStandardError $StderrLog `
        -PassThru

    $deadline = (Get-Date).AddSeconds(30)
    $ready = $false
    while ((Get-Date) -lt $deadline) {
        if (Test-OllamaAlive -Url $VersionUrl) {
            $ready = $true
            break
        }
        if ($proc.HasExited) {
            break
        }
        Start-Sleep -Milliseconds 500
    }

    if (-not $ready) {
        $errTail = ""
        if (Test-Path $StderrLog) {
            $errTail = (Get-Content -Path $StderrLog -Tail 20 -ErrorAction SilentlyContinue) -join "`n"
        }
        Write-Error "Ollama failed to answer $VersionUrl within 30s. Stderr tail:`n$errTail"
        exit 1
    }

    $verInfo = Invoke-RestMethod -Uri $VersionUrl -Method Get -TimeoutSec 2
    Write-Host "Ollama ready at $VersionUrl (version=$($verInfo.version), pid=$($proc.Id))" -ForegroundColor Green
    exit 0
}
finally {
    foreach ($name in $EnvVarsToStrip) {
        [Environment]::SetEnvironmentVariable($name, $SavedEnv[$name], "Process")
    }
}
