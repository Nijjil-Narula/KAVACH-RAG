<#
.SYNOPSIS
    Start, restart, or stop the KAVACH-RAG Phase 3 FastAPI serving process.
.DESCRIPTION
    Manages the single-worker Uvicorn server for kavach.serving.app:app on 127.0.0.1:8000.
    Tracks only the exact PID written atomically to logs\.server.pid and never kills
    processes by name. Requires Ollama to already be running via scripts\start_ollama.ps1.
#>
[CmdletBinding()]
param(
    [switch]$Restart,
    [switch]$Stop
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$LogsDir = Join-Path $ProjectRoot "logs"
$PidFile = Join-Path $LogsDir ".server.pid"
$LauncherLog = Join-Path $LogsDir "start_server.log"
$StdoutLog = Join-Path $LogsDir "phase3_server.log"
$StderrLog = Join-Path $LogsDir "phase3_server_err.log"
$HealthUrl = "http://127.0.0.1:8000/health"
$OllamaVersionUrl = "http://127.0.0.1:11434/api/version"
$VenvPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$HfCacheDir = Join-Path $ProjectRoot "models\hf_cache"

New-Item -ItemType Directory -Path $LogsDir -Force | Out-Null

function Write-LauncherLog {
    param(
        [string]$Level,
        [string]$Message
    )
    $ts = [DateTime]::UtcNow.ToString("yyyy-MM-ddTHH:mm:ssZ")
    $line = "[$ts] [$Level] $Message`n"
    [System.IO.File]::AppendAllText($LauncherLog, $line, [System.Text.UTF8Encoding]::new($false))
}

function Exit-LauncherFailure {
    param([string]$Message)
    Write-LauncherLog -Level "ERROR" -Message $Message
    [Console]::Error.WriteLine($Message)
    exit 1
}

Write-LauncherLog -Level "INFO" -Message ("Invoked start_server.ps1 (Restart={0}, Stop={1})" -f $Restart.IsPresent, $Stop.IsPresent)

function Test-Endpoint200 {
    param(
        [string]$Url,
        [int]$TimeoutSec = 3
    )
    try {
        $resp = Invoke-WebRequest -Uri $Url -Method Get -TimeoutSec $TimeoutSec -UseBasicParsing
        return ($resp.StatusCode -eq 200)
    }
    catch {
        return $false
    }
}

function Get-HealthPayload {
    param(
        [string]$Url,
        [int]$TimeoutSec = 3
    )
    try {
        $resp = Invoke-WebRequest -Uri $Url -Method Get -TimeoutSec $TimeoutSec -UseBasicParsing
        if ($resp.StatusCode -eq 200) {
            return ($resp.Content | ConvertFrom-Json)
        }
    }
    catch {
        return $null
    }
    return $null
}

function Stop-TrackedServer {
    param([int]$SpecificPid = 0)

    $targetPid = $SpecificPid
    if ($targetPid -le 0 -and (Test-Path -LiteralPath $PidFile)) {
        $raw = (Get-Content -LiteralPath $PidFile -Raw -ErrorAction SilentlyContinue)
        if ($null -ne $raw) {
            $parsed = 0
            if ([int]::TryParse($raw.Trim(), [ref]$parsed) -and $parsed -gt 0) {
                $targetPid = $parsed
            }
        }
    }

    if ($targetPid -gt 0) {
        $existing = Get-Process -Id $targetPid -ErrorAction SilentlyContinue
        if ($null -ne $existing) {
            Write-Host "Stopping KAVACH-RAG server PID $targetPid..." -ForegroundColor Yellow
            Stop-Process -Id $targetPid -ErrorAction SilentlyContinue
            $graceDeadline = (Get-Date).AddSeconds(10)
            while ((Get-Date) -lt $graceDeadline) {
                $stillUp = Get-Process -Id $targetPid -ErrorAction SilentlyContinue
                if ($null -eq $stillUp) {
                    break
                }
                Start-Sleep -Milliseconds 250
            }

            $stillAlive = Get-Process -Id $targetPid -ErrorAction SilentlyContinue
            if ($null -ne $stillAlive) {
                Write-Host "Force-stopping KAVACH-RAG server PID $targetPid..." -ForegroundColor Yellow
                Stop-Process -Id $targetPid -Force -ErrorAction SilentlyContinue
                $forceDeadline = (Get-Date).AddSeconds(5)
                while ((Get-Date) -lt $forceDeadline) {
                    $check = Get-Process -Id $targetPid -ErrorAction SilentlyContinue
                    if ($null -eq $check) {
                        break
                    }
                    Start-Sleep -Milliseconds 200
                }
            }
        }
    }

    if (Test-Path -LiteralPath $PidFile) {
        Remove-Item -LiteralPath $PidFile -Force -ErrorAction SilentlyContinue
    }
}

function Get-LogTails {
    $outTail = ""
    $errTail = ""
    if (Test-Path -LiteralPath $StdoutLog) {
        $outTail = (Get-Content -LiteralPath $StdoutLog -Tail 20 -ErrorAction SilentlyContinue) -join "`n"
    }
    if (Test-Path -LiteralPath $StderrLog) {
        $errTail = (Get-Content -LiteralPath $StderrLog -Tail 20 -ErrorAction SilentlyContinue) -join "`n"
    }
    return "=== stdout ($StdoutLog) ===`n$outTail`n=== stderr ($StderrLog) ===`n$errTail"
}

if ($Stop) {
    Stop-TrackedServer
    Write-LauncherLog -Level "INFO" -Message "KAVACH-RAG server stopped."
    Write-Host "KAVACH-RAG server stopped." -ForegroundColor Green
    exit 0
}

if (-not (Test-Path -LiteralPath $VenvPython)) {
    Exit-LauncherFailure -Message "Virtualenv Python not found at $VenvPython"
}

if (-not (Test-Endpoint200 -Url $OllamaVersionUrl -TimeoutSec 5)) {
    Exit-LauncherFailure -Message (
        "Ollama is unreachable at $OllamaVersionUrl. " +
        "Start Ollama first with: powershell -NoProfile -ExecutionPolicy Bypass -File scripts\start_ollama.ps1"
    )
}

if ($Restart) {
    Stop-TrackedServer
}
elseif (Test-Path -LiteralPath $PidFile) {
    $rawPid = (Get-Content -LiteralPath $PidFile -Raw -ErrorAction SilentlyContinue)
    $existingPid = 0
    if ($null -ne $rawPid -and [int]::TryParse($rawPid.Trim(), [ref]$existingPid) -and $existingPid -gt 0) {
        $procObj = Get-Process -Id $existingPid -ErrorAction SilentlyContinue
        if ($null -ne $procObj -and (Test-Endpoint200 -Url $HealthUrl -TimeoutSec 3)) {
            $msg = "KAVACH-RAG server is already running (PID $existingPid) at $HealthUrl."
            Write-LauncherLog -Level "INFO" -Message $msg
            Write-Host $msg -ForegroundColor Green
            exit 0
        }
    }
    Remove-Item -LiteralPath $PidFile -Force -ErrorAction SilentlyContinue
}

New-Item -ItemType Directory -Path $LogsDir -Force | Out-Null

$SavedPythonUtf8 = [Environment]::GetEnvironmentVariable("PYTHONUTF8", "Process")
$SavedHfHome = [Environment]::GetEnvironmentVariable("HF_HOME", "Process")
$proc = $null

try {
    [Environment]::SetEnvironmentVariable("PYTHONUTF8", "1", "Process")
    [Environment]::SetEnvironmentVariable("HF_HOME", $HfCacheDir, "Process")

    Write-LauncherLog -Level "INFO" -Message "Starting KAVACH-RAG FastAPI server on http://127.0.0.1:8000 ..."
    Write-Host "Starting KAVACH-RAG FastAPI server on http://127.0.0.1:8000 ..." -ForegroundColor Cyan
    $proc = Start-Process `
        -PassThru `
        -NoNewWindow `
        -WorkingDirectory $ProjectRoot `
        -FilePath $VenvPython `
        -ArgumentList @(
            "-m", "uvicorn", "kavach.serving.app:app",
            "--host", "127.0.0.1",
            "--port", "8000",
            "--workers", "1",
            "--log-level", "warning",
            "--no-access-log",
            "--backlog", "2048"
        ) `
        -RedirectStandardOutput $StdoutLog `
        -RedirectStandardError $StderrLog

    $tmpPid = Join-Path $LogsDir ".server.pid.tmp"
    [System.IO.File]::WriteAllText($tmpPid, "$($proc.Id)`n", [System.Text.UTF8Encoding]::new($false))
    Move-Item -LiteralPath $tmpPid -Destination $PidFile -Force
    Write-LauncherLog -Level "INFO" -Message "Spawned uvicorn PID $($proc.Id); polling $HealthUrl ..."

    $deadline = (Get-Date).AddSeconds(240)
    $health = $null
    while ((Get-Date) -lt $deadline) {
        $health = Get-HealthPayload -Url $HealthUrl -TimeoutSec 3
        if ($null -ne $health) {
            break
        }
        if ($proc.HasExited) {
            Write-LauncherLog -Level "ERROR" -Message "Spawned uvicorn PID $($proc.Id) exited prematurely with code $($proc.ExitCode)."
            break
        }
        Start-Sleep -Seconds 2
    }

    if ($null -eq $health) {
        $tails = Get-LogTails
        $spawnedId = if ($null -ne $proc) { $proc.Id } else { 0 }
        Stop-TrackedServer -SpecificPid $spawnedId
        Exit-LauncherFailure -Message "KAVACH-RAG server failed to become healthy at $HealthUrl within 240s.`n$tails"
    }

    if ($health.status -eq "ok") {
        $okMsg = ("KAVACH-RAG server ready (PID {0}, status={1}, rss_mb={2})." -f $proc.Id, $health.status, $health.rss_mb)
        Write-LauncherLog -Level "INFO" -Message $okMsg
        Write-Host $okMsg -ForegroundColor Green
        exit 0
    }
    elseif ($health.status -eq "degraded") {
        $warnMsg = ("KAVACH-RAG server answered HTTP 200 with status=degraded (PID {0}, circuit={1}, vram_breached={2})." -f $proc.Id, $health.circuit_state, $health.vram_breached)
        Write-LauncherLog -Level "WARN" -Message $warnMsg
        Write-Warning $warnMsg
        exit 0
    }
    else {
        $tails = Get-LogTails
        Stop-TrackedServer -SpecificPid $proc.Id
        Exit-LauncherFailure -Message ("KAVACH-RAG server returned unexpected health status '{0}'.`n{1}" -f $health.status, $tails)
    }
}
finally {
    [Environment]::SetEnvironmentVariable("PYTHONUTF8", $SavedPythonUtf8, "Process")
    [Environment]::SetEnvironmentVariable("HF_HOME", $SavedHfHome, "Process")
}
