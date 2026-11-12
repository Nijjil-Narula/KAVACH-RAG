<#
.SYNOPSIS
    KAVACH-RAG Phase 3 Stepped Concurrency Load Test & Report Runner.
.DESCRIPTION
    Executes the stepped Locust load matrix across:
      1) sync       (KAVACH_LOAD_ENDPOINT="sync",  KAVACH_LOAD_MODE="cold")
      2) async-cold (KAVACH_LOAD_ENDPOINT="async", KAVACH_LOAD_MODE="cold")
      3) async-warm (KAVACH_LOAD_ENDPOINT="async", KAVACH_LOAD_MODE="warm")
    for each user count in -Users (default: 1, 10, 50, 100), restarting and
    stopping the FastAPI server cleanly per cell, waiting for GPU idle <= 10%
    (3 consecutive 1s samples, 60s timeout), capturing /health and run metadata,
    and compiling reports/phase3_load_test_summary.json + .md at completion.
#>
[CmdletBinding()]
param(
    [int[]]$Users = @(1, 10, 50, 100),
    [int]$RunSeconds = 60
)

$ErrorActionPreference = "Stop"

if (-not $Users -or $Users.Count -eq 0) {
    throw "Parameter -Users must contain at least one positive integer."
}
foreach ($uCount in $Users) {
    if ($uCount -le 0) {
        throw "Invalid user count '$uCount' in -Users; all values must be > 0."
    }
}
if ($RunSeconds -le 0) {
    throw "Parameter -RunSeconds must be > 0 (got $RunSeconds)."
}

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$StartServerScript = Join-Path $PSScriptRoot "start_server.ps1"
$LocustDir = Join-Path $ProjectRoot "reports\locust"
$LogsDir = Join-Path $ProjectRoot "logs"
$BaseUrl = "http://127.0.0.1:8000"

if (-not (Test-Path $Python)) {
    throw "Python virtual environment not found at $Python"
}
if (-not (Test-Path $StartServerScript)) {
    throw "Server control script not found at $StartServerScript"
}

$SavedEnv = @{
    KAVACH_LOAD_ENDPOINT   = [Environment]::GetEnvironmentVariable("KAVACH_LOAD_ENDPOINT", "Process")
    KAVACH_LOAD_MODE       = [Environment]::GetEnvironmentVariable("KAVACH_LOAD_MODE", "Process")
    KAVACH_LOCUST_META_PATH = [Environment]::GetEnvironmentVariable("KAVACH_LOCUST_META_PATH", "Process")
    PYTHONUTF8             = [Environment]::GetEnvironmentVariable("PYTHONUTF8", "Process")
}

function Get-NvidiaSmiGpuSnapshot {
    $prevNativePref = $null
    $hasNativePref = Test-Path variable:PSNativeCommandUseErrorActionPreference
    if ($hasNativePref) {
        $prevNativePref = $PSNativeCommandUseErrorActionPreference
        $PSNativeCommandUseErrorActionPreference = $false
    }
    $prevErr = $ErrorActionPreference
    try {
        $ErrorActionPreference = "Continue"
        $global:LASTEXITCODE = 0
        $raw = & nvidia-smi --query-gpu=memory.used,memory.free,memory.total,utilization.gpu --format=csv,noheader,nounits 2>&1
        $rc = $LASTEXITCODE
    }
    finally {
        $ErrorActionPreference = $prevErr
        if ($hasNativePref) {
            $PSNativeCommandUseErrorActionPreference = $prevNativePref
        }
    }
    $rawText = ($raw | ForEach-Object { if ($_ -is [System.Management.Automation.ErrorRecord]) { $_.Exception.Message } else { "$_" } }) -join "`n"
    if ($rc -ne 0 -or -not $rawText) {
        throw "nvidia-smi failed (exit=$rc): $rawText"
    }
    foreach ($line in ($rawText -split "`r?`n")) {
        $parts = $line.Split(",") | ForEach-Object { $_.Trim() }
        if ($parts.Count -ge 4 -and $parts[0] -match '^\d+(\.\d+)?$' -and $parts[1] -match '^\d+(\.\d+)?$' -and $parts[2] -match '^\d+(\.\d+)?$' -and $parts[3] -match '^\d+(\.\d+)?$') {
            return [ordered]@{
                memory_used_mib     = [int][Math]::Round([double]$parts[0])
                memory_free_mib     = [int][Math]::Round([double]$parts[1])
                memory_total_mib    = [int][Math]::Round([double]$parts[2])
                utilization_gpu_pct = [int][Math]::Round([double]$parts[3])
            }
        }
    }
    throw "Unable to parse nvidia-smi output: $rawText"
}

function Get-OllamaResidentModels {
    $psUrl = "http://127.0.0.1:11434/api/ps"
    try {
        $resp = Invoke-RestMethod -Uri $psUrl -Method Get -TimeoutSec 5
        $names = @()
        if ($null -ne $resp -and $null -ne $resp.models) {
            foreach ($m in $resp.models) {
                $n = if ($m.name) { "$($m.name)" } elseif ($m.model) { "$($m.model)" } else { "" }
                if ($n) {
                    $names += $n
                }
            }
        }
        return [ordered]@{
            reachable = $true
            models    = $names
        }
    }
    catch {
        return [ordered]@{
            reachable = $false
            models    = @()
        }
    }
}

function Write-Utf8Atomic {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][string]$Content
    )
    $parent = Split-Path -Parent $Path
    if (-not (Test-Path $parent)) {
        New-Item -ItemType Directory -Path $parent -Force | Out-Null
    }
    $tmp = Join-Path $parent (".tmp_" + [System.IO.Path]::GetFileName($Path) + "_" + [Guid]::NewGuid().ToString("N"))
    $utf8NoBom = [System.Text.UTF8Encoding]::new($false)
    try {
        [System.IO.File]::WriteAllText($tmp, $Content, $utf8NoBom)
        Move-Item -Path $tmp -Destination $Path -Force
    }
    finally {
        if (Test-Path $tmp) {
            Remove-Item -Path $tmp -Force -ErrorAction SilentlyContinue
        }
    }
}

function Wait-GpuIdle {
    param(
        [int]$ThresholdPct = 10,
        [int]$RequiredConsecutive = 3,
        [int]$TimeoutSeconds = 60
    )
    $sw = [System.Diagnostics.Stopwatch]::StartNew()
    $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
    $consecutive = 0
    $samples = 0
    $lastUtil = $null
    $timedOut = $false

    while ($true) {
        $samples += 1
        $util = $null
        try {
            $raw = & nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>$null
            if ($LASTEXITCODE -eq 0 -and $raw) {
                $parsed = @()
                foreach ($line in ($raw -split "`r?`n")) {
                    $trimmed = $line.Trim()
                    if ($trimmed -match '^\d+$') {
                        $parsed += [int]$trimmed
                    }
                }
                if ($parsed.Count -gt 0) {
                    $util = ($parsed | Measure-Object -Maximum).Maximum
                    $lastUtil = [int]$util
                }
            }
        }
        catch {
            $util = $null
        }

        if ($null -ne $util -and $util -le $ThresholdPct) {
            $consecutive += 1
            if ($consecutive -ge $RequiredConsecutive) {
                break
            }
        }
        else {
            $consecutive = 0
        }

        if ((Get-Date) -ge $deadline) {
            $timedOut = $true
            break
        }
        Start-Sleep -Seconds 1
    }

    $sw.Stop()
    return [ordered]@{
        timed_out     = $timedOut
        wait_s        = [Math]::Round($sw.Elapsed.TotalSeconds, 2)
        samples       = $samples
        last_util_pct = $lastUtil
    }
}

function Capture-HealthSnapshot {
    param(
        [Parameter(Mandatory = $true)][string]$PythonExe,
        [Parameter(Mandatory = $true)][string]$Url,
        [Parameter(Mandatory = $true)][string]$OutPath
    )
    $pyScript = @'
import json
import os
import sys
import tempfile
from pathlib import Path
import httpx

url, out_path_str = sys.argv[1], sys.argv[2]
out_path = Path(out_path_str)
out_path.parent.mkdir(parents=True, exist_ok=True)
status_code = None
payload = None
try:
    resp = httpx.get(url, timeout=15.0)
    status_code = resp.status_code
    try:
        payload = resp.json()
    except ValueError:
        payload = {"status": "non_json_health", "http_status": status_code, "raw_text": resp.text[:1000]}
except Exception as exc:
    payload = {"status": "unreachable", "http_status": None, "error": f"{type(exc).__name__}: {exc}"}

fd, tmp_name = tempfile.mkstemp(prefix=".health_", suffix=".tmp", dir=str(out_path.parent))
try:
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)
        fh.write("\n")
    os.replace(tmp_name, out_path)
finally:
    if os.path.exists(tmp_name):
        os.remove(tmp_name)

print(status_code if status_code is not None else "ERR", flush=True)
if status_code != 200 or not isinstance(payload, dict):
    sys.exit(2)
'@

    $tmpPy = Join-Path ([System.IO.Path]::GetTempPath()) ("kavach_health_" + [Guid]::NewGuid().ToString("N") + ".py")
    Write-Utf8Atomic -Path $tmpPy -Content ($pyScript + "`n")
    try {
        $lastError = ""
        for ($attempt = 1; $attempt -le 3; $attempt++) {
            $prevNativePref = $null
            $hasNativePref = Test-Path variable:PSNativeCommandUseErrorActionPreference
            if ($hasNativePref) {
                $prevNativePref = $PSNativeCommandUseErrorActionPreference
                $PSNativeCommandUseErrorActionPreference = $false
            }
            $prevErr = $ErrorActionPreference
            try {
                $ErrorActionPreference = "Continue"
                $global:LASTEXITCODE = 0
                $out = & $PythonExe $tmpPy $Url $OutPath 2>&1
                $pyExit = $LASTEXITCODE
            }
            finally {
                $ErrorActionPreference = $prevErr
                if ($hasNativePref) {
                    $PSNativeCommandUseErrorActionPreference = $prevNativePref
                }
            }

            $outText = ($out | ForEach-Object { if ($_ -is [System.Management.Automation.ErrorRecord]) { $_.Exception.Message } else { "$_" } }) -join "`n"
            $lastLine = (($outText -split "`r?`n") | Where-Object { $_.Trim() -ne "" } | Select-Object -Last 1)
            if ($pyExit -eq 0 -and $lastLine -match '^\d+$' -and [int]$lastLine -eq 200 -and (Test-Path $OutPath) -and (Get-Item $OutPath).Length -gt 0) {
                try {
                    $parsedHealth = (Get-Content -Path $OutPath -Raw -Encoding UTF8) | ConvertFrom-Json
                    if ($null -ne $parsedHealth) {
                        return [int]$lastLine
                    }
                    $lastError = "Parsed JSON was null in $OutPath"
                }
                catch {
                    $lastError = "Invalid JSON written to $OutPath : $_"
                }
            }
            else {
                $lastError = "attempt=$attempt exit=$pyExit status=$lastLine output=$outText"
            }

            if ($attempt -lt 3) {
                Start-Sleep -Seconds 1
            }
        }
        throw "Capture-HealthSnapshot failed for $Url -> $OutPath ($lastError)"
    }
    finally {
        if (Test-Path $tmpPy) {
            Remove-Item -Path $tmpPy -Force -ErrorAction SilentlyContinue
        }
    }
}

function Invoke-WarmCachePrewarm {
    param(
        [Parameter(Mandatory = $true)][string]$PythonExe,
        [Parameter(Mandatory = $true)][string]$BaseUrl,
        [Parameter(Mandatory = $true)][string]$ProjectRoot
    )
    $pyScript = @'
import sys
import httpx

base_url = sys.argv[1].rstrip("/")
project_root = sys.argv[2]
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from benchmarks.locustfile import _build_pool

pool = _build_pool()
for idx, query in enumerate(pool, start=1):
    resp = httpx.post(
        f"{base_url}/rag/async",
        json={"query": query, "stream": False, "use_cache": True},
        timeout=120.0,
    )
    if resp.status_code != 200:
        raise RuntimeError(f"Pre-warm [{idx}/{len(pool)}] HTTP {resp.status_code}: {resp.text[:200]}")
    data = resp.json()
    answer = data.get("answer")
    if not answer:
        raise RuntimeError(f"Pre-warm [{idx}/{len(pool)}] returned empty answer for query={query!r}")
    print(
        f"    Pre-warm [{idx}/{len(pool)}] HTTP 200 (cache_hit={data.get('cache_hit')}, "
        f"total_ms={data.get('total_ms')}, chars={len(answer)})",
        flush=True,
    )

health_resp = httpx.get(f"{base_url}/health", timeout=15.0)
if health_resp.status_code != 200:
    raise RuntimeError(f"Post-prewarm /health returned HTTP {health_resp.status_code}")
health = health_resp.json()
cache_info = health.get("cache") or {}
cache_size = cache_info.get("size", 0)
if not isinstance(cache_size, int) or cache_size < 1:
    raise RuntimeError(f"Expected /health cache.size >= 1 after pre-warm, got {cache_size!r} (cache={cache_info})")
print(f"    Pre-warm verified: /health cache={cache_info}", flush=True)
'@

    $tmpPy = Join-Path ([System.IO.Path]::GetTempPath()) ("kavach_prewarm_" + [Guid]::NewGuid().ToString("N") + ".py")
    Write-Utf8Atomic -Path $tmpPy -Content ($pyScript + "`n")
    try {
        $global:LASTEXITCODE = 0
        & $PythonExe $tmpPy $BaseUrl $ProjectRoot
        if ($LASTEXITCODE -ne 0) {
            throw "Invoke-WarmCachePrewarm failed with exit code $LASTEXITCODE"
        }
    }
    finally {
        if (Test-Path $tmpPy) {
            Remove-Item -Path $tmpPy -Force -ErrorAction SilentlyContinue
        }
    }
}

$Matrix = @(
    @{ Endpoint = "sync";  Mode = "cold"; Label = "sync" },
    @{ Endpoint = "async"; Mode = "cold"; Label = "async-cold" },
    @{ Endpoint = "async"; Mode = "warm"; Label = "async-warm" }
)

Push-Location $ProjectRoot
try {
    New-Item -ItemType Directory -Path $LocustDir -Force | Out-Null
    New-Item -ItemType Directory -Path $LogsDir -Force | Out-Null

    $env:PYTHONUTF8 = "1"
    $totalRuns = $Matrix.Count * $Users.Count
    $runIdx = 0

    Write-Host "==========================================================" -ForegroundColor Cyan
    Write-Host " KAVACH-RAG Phase 3 Pre-Run GPU Baseline Gate" -ForegroundColor Cyan
    Write-Host "==========================================================" -ForegroundColor Cyan

    $baselinePath = Join-Path $LocustDir "baseline_gpu.json"
    $prevNativePref = $null
    $hasNativePref = Test-Path variable:PSNativeCommandUseErrorActionPreference
    if ($hasNativePref) {
        $prevNativePref = $PSNativeCommandUseErrorActionPreference
        $PSNativeCommandUseErrorActionPreference = $false
    }
    $prevErr = $ErrorActionPreference
    $stopExit = -1
    $stopRaw = @()
    try {
        $ErrorActionPreference = "Continue"
        $global:LASTEXITCODE = 0
        $stopRaw = & ollama stop kavach-tier5 2>&1
        $stopExit = $LASTEXITCODE
    }
    catch {
        $stopRaw = @($_.Exception.Message)
    }
    finally {
        $ErrorActionPreference = $prevErr
        if ($hasNativePref) {
            $PSNativeCommandUseErrorActionPreference = $prevNativePref
        }
    }
    $stopText = (($stopRaw | ForEach-Object { if ($_ -is [System.Management.Automation.ErrorRecord]) { $_.Exception.Message } else { "$_" } }) -join "`n").Trim()
    $psCheck = Get-OllamaResidentModels
    if ($stopExit -ne 0) {
        if ($psCheck.reachable -and $psCheck.models.Count -gt 0) {
            throw "ollama stop kavach-tier5 failed (exit=$stopExit) and /api/ps still reports resident model(s): $($psCheck.models -join ', '). Output: $stopText"
        }
        Write-Warning "ollama stop kavach-tier5 exited with code $stopExit, but /api/ps reports no resident model (reachable=$($psCheck.reachable)). Continuing baseline GPU check."
    }

    $baselineIdle = Wait-GpuIdle -ThresholdPct 10 -RequiredConsecutive 3 -TimeoutSeconds 60
    if ($baselineIdle.timed_out) {
        Write-Warning "Pre-run baseline GPU idle wait timed out at 60s (last_util=$($baselineIdle.last_util_pct)%)."
    }
    $baselineGpu = Get-NvidiaSmiGpuSnapshot
    $baselineThresholdMib = 600
    $baselinePassed = ($baselineGpu.memory_used_mib -le $baselineThresholdMib)
    $baselinePayload = [ordered]@{
        schema_version         = "1.0"
        recorded_at_utc        = (Get-Date).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ssZ")
        ollama_stop_command    = "ollama stop kavach-tier5"
        ollama_stop_exit_code  = $stopExit
        ollama_stop_output     = $stopText
        ollama_ps_reachable    = [bool]$psCheck.reachable
        ollama_resident_models = @($psCheck.models)
        gpu_idle               = $baselineIdle
        memory_used_mib        = $baselineGpu.memory_used_mib
        memory_free_mib        = $baselineGpu.memory_free_mib
        memory_total_mib       = $baselineGpu.memory_total_mib
        utilization_gpu_pct    = $baselineGpu.utilization_gpu_pct
        threshold_mib          = $baselineThresholdMib
        passed                 = [bool]$baselinePassed
    }
    Write-Utf8Atomic -Path $baselinePath -Content (($baselinePayload | ConvertTo-Json -Depth 5) + "`n")
    Write-Host "Baseline GPU memory.used = $($baselineGpu.memory_used_mib) MiB (threshold <= $baselineThresholdMib MiB) -> $baselinePath" -ForegroundColor Cyan
    if (-not $baselinePassed) {
        throw "Baseline GPU memory.used $($baselineGpu.memory_used_mib) MiB exceeds $baselineThresholdMib MiB. Saved $baselinePath and halting without starting any run."
    }

    Write-Host "==========================================================" -ForegroundColor Cyan
    Write-Host " KAVACH-RAG Phase 3 Locust Load Matrix ($totalRuns runs)" -ForegroundColor Cyan
    Write-Host " Users: $($Users -join ', ') | RunSeconds: $RunSeconds" -ForegroundColor Cyan
    Write-Host "==========================================================" -ForegroundColor Cyan

    foreach ($profile in $Matrix) {
        $ep = $profile.Endpoint
        $mode = $profile.Mode
        $label = $profile.Label

        foreach ($u in $Users) {
            $runIdx += 1
            $slug = "${ep}_${mode}_u${u}"
            $csvBaseRel = "reports\locust\$slug"
            $csvBaseAbs = Join-Path $LocustDir $slug
            $statsCsv = "${csvBaseAbs}_stats.csv"
            $healthPath = "${csvBaseAbs}_health.json"
            $metaPath = "${csvBaseAbs}_meta.json"
            $logPath = Join-Path $LogsDir "phase3_locust_${slug}.log"
            $spawnRate = [Math]::Min($u, 10)

            Write-Host "`n>>> [$runIdx/$totalRuns] Profile=$label (endpoint=$ep, mode=$mode, users=$u, spawn_rate=$spawnRate, duration=${RunSeconds}s)" -ForegroundColor Yellow

            # Remove stale outputs for this specific matrix cell before execution
            foreach ($suffix in @("_stats.csv", "_stats_history.csv", "_failures.csv", "_exceptions.csv", "_health.json", "_meta.json")) {
                $candidate = "${csvBaseAbs}${suffix}"
                if (Test-Path $candidate) {
                    Remove-Item -Path $candidate -Force
                }
            }
            if (Test-Path $logPath) {
                Remove-Item -Path $logPath -Force
            }

            try {
                # 1. Fresh server per run
                Write-Host "    [1/4] Restarting FastAPI server via start_server.ps1 -Restart..." -ForegroundColor Cyan
                $global:LASTEXITCODE = 0
                & $StartServerScript -Restart
                if ($LASTEXITCODE -ne 0) {
                    throw "start_server.ps1 -Restart failed with exit code $LASTEXITCODE for $slug"
                }

                # 2. Wait for GPU idle (<= 10% for 3 consecutive 1s samples, 60s timeout) and record loaded_idle_mib
                Write-Host "    [2/4] Waiting for GPU idle (<= 10% for 3 consecutive samples, 60s max)..." -ForegroundColor Cyan
                $gpuIdle = Wait-GpuIdle -ThresholdPct 10 -RequiredConsecutive 3 -TimeoutSeconds 60
                if ($gpuIdle.timed_out) {
                    Write-Warning "GPU idle wait timed out at 60s for $slug (last_util=$($gpuIdle.last_util_pct)%). Continuing benchmark."
                }
                else {
                    Write-Host "    GPU idle confirmed in $($gpuIdle.wait_s)s (last_util=$($gpuIdle.last_util_pct)%)." -ForegroundColor Green
                }
                $loadedIdleSnap = Get-NvidiaSmiGpuSnapshot
                $loadedIdleMib = $loadedIdleSnap.memory_used_mib
                Write-Host "    Post-restart loaded_idle_mib = $loadedIdleMib MiB" -ForegroundColor Cyan

                # 2.5 Pre-warm cache for async-warm before Locust starts its 60s timer
                $prewarmPerformed = $false
                $postPrewarmGpuIdle = $null
                if ($ep -eq "async" -and $mode -eq "warm") {
                    Write-Host "    [2.5/4] Pre-warming cache for async-warm (8 POOL queries)..." -ForegroundColor Cyan
                    Invoke-WarmCachePrewarm -PythonExe $Python -BaseUrl $BaseUrl -ProjectRoot $ProjectRoot
                    $prewarmPerformed = $true

                    Write-Host "    [2.5/4] Waiting for post-prewarm GPU idle (<= 10% for 3 consecutive samples, 60s max)..." -ForegroundColor Cyan
                    $postPrewarmGpuIdle = Wait-GpuIdle -ThresholdPct 10 -RequiredConsecutive 3 -TimeoutSeconds 60
                    if ($postPrewarmGpuIdle.timed_out) {
                        Write-Warning "Post-prewarm GPU idle wait timed out at 60s for $slug (last_util=$($postPrewarmGpuIdle.last_util_pct)%). Continuing benchmark."
                    }
                    else {
                        Write-Host "    Post-prewarm GPU idle confirmed in $($postPrewarmGpuIdle.wait_s)s (last_util=$($postPrewarmGpuIdle.last_util_pct)%)." -ForegroundColor Green
                    }
                }

                # 3. Execute Locust headless benchmark
                $env:KAVACH_LOAD_ENDPOINT = $ep
                $env:KAVACH_LOAD_MODE = $mode
                $env:KAVACH_LOCUST_META_PATH = $metaPath
                $env:PYTHONUTF8 = "1"

                Write-Host "    [3/4] Running Locust -> $csvBaseRel (log: logs\phase3_locust_${slug}.log)..." -ForegroundColor Cyan
                $startedUtc = (Get-Date).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ssZ")
                $runSw = [System.Diagnostics.Stopwatch]::StartNew()

                $locustArgs = @(
                    "-m", "locust",
                    "-f", "benchmarks\locustfile.py",
                    "--headless",
                    "-u", "$u",
                    "-r", "$spawnRate",
                    "-t", "${RunSeconds}s",
                    "--stop-timeout", "10",
                    "--host", $BaseUrl,
                    "--csv", $csvBaseRel,
                    "--only-summary"
                )

                $prevNativePref = $null
                $hasNativePref = Test-Path variable:PSNativeCommandUseErrorActionPreference
                if ($hasNativePref) {
                    $prevNativePref = $PSNativeCommandUseErrorActionPreference
                    $PSNativeCommandUseErrorActionPreference = $false
                }
                $prevErrPref = $ErrorActionPreference
                $locustExit = -1
                $utf8NoBom = [System.Text.UTF8Encoding]::new($false)
                $logWriter = [System.IO.StreamWriter]::new($logPath, $false, $utf8NoBom)
                $logWriter.AutoFlush = $true
                try {
                    $ErrorActionPreference = "Continue"
                    $global:LASTEXITCODE = 0
                    & $Python @locustArgs 2>&1 | ForEach-Object {
                        $lineText = if ($_ -is [System.Management.Automation.ErrorRecord]) { $_.Exception.Message } else { "$_" }
                        $logWriter.WriteLine($lineText)
                        Write-Host $lineText
                    }
                    $locustExit = $LASTEXITCODE
                }
                finally {
                    $logWriter.Dispose()
                    $ErrorActionPreference = $prevErrPref
                    if ($hasNativePref) {
                        $PSNativeCommandUseErrorActionPreference = $prevNativePref
                    }
                }

                $runSw.Stop()
                $finishedUtc = (Get-Date).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ssZ")
                $wallDurationS = [Math]::Round($runSw.Elapsed.TotalSeconds, 3)

                if (-not (Test-Path $statsCsv) -or (Get-Item $statsCsv).Length -eq 0) {
                    $logTail = ""
                    if (Test-Path $logPath) {
                        $logTail = (Get-Content -Path $logPath -Tail 30 -Encoding UTF8 -ErrorAction SilentlyContinue) -join "`n"
                    }
                    throw "Locust failed to produce non-empty $statsCsv (exit=$locustExit). Log tail:`n$logTail"
                }

                Write-Host "    Locust completed in ${wallDurationS}s (exit=$locustExit; non-zero expected when HTTP errors/timeouts occur)." -ForegroundColor Cyan

                # 4. Capture clean /health JSON snapshot and companion _meta.json
                Write-Host "    [4/4] Capturing /health snapshot and run metadata..." -ForegroundColor Cyan
                $healthStatus = Capture-HealthSnapshot -PythonExe $Python -Url "$BaseUrl/health" -OutPath $healthPath

                $probeMeta = $null
                if (Test-Path $metaPath) {
                    try {
                        $probeMeta = (Get-Content -Path $metaPath -Raw -Encoding UTF8) | ConvertFrom-Json
                    }
                    catch {
                        $probeMeta = $null
                    }
                }

                $metaObj = [ordered]@{
                    slug                          = $slug
                    label                         = $label
                    endpoint                      = $ep
                    mode                          = $mode
                    users                         = $u
                    spawn_rate                    = $spawnRate
                    run_seconds                   = $RunSeconds
                    started_at_utc                = $startedUtc
                    finished_at_utc               = $finishedUtc
                    duration_s                    = $wallDurationS
                    locust_exit_code              = $locustExit
                    loaded_idle_mib               = $loadedIdleMib
                    gpu_idle_timed_out            = [bool]$gpuIdle.timed_out
                    gpu_idle_wait_s               = $gpuIdle.wait_s
                    gpu_idle_samples              = $gpuIdle.samples
                    gpu_idle_last_util_pct        = $gpuIdle.last_util_pct
                    prewarm_performed             = $prewarmPerformed
                    post_prewarm_gpu_idle_timeout = if ($null -ne $postPrewarmGpuIdle) { [bool]$postPrewarmGpuIdle.timed_out } else { $null }
                    health_http_status            = $healthStatus
                    max_gpu_waiting               = if ($null -ne $probeMeta) { $probeMeta.max_gpu_waiting } else { $null }
                    max_gpu_active                = if ($null -ne $probeMeta) { $probeMeta.max_gpu_active } else { $null }
                    max_retrieval_waiting         = if ($null -ne $probeMeta) { $probeMeta.max_retrieval_waiting } else { $null }
                    health_sample_count           = if ($null -ne $probeMeta -and $null -ne $probeMeta.health_sample_count) { $probeMeta.health_sample_count } else { 0 }
                    health_sample_rate_hz         = if ($null -ne $probeMeta -and $null -ne $probeMeta.health_sample_rate_hz) { $probeMeta.health_sample_rate_hz } else { 1.0 }
                    queue_maxima_bound            = if ($null -ne $probeMeta -and $probeMeta.queue_maxima_bound) { $probeMeta.queue_maxima_bound } else { "1 Hz lower bound sampled via GET /health" }
                    log_file                      = "logs/phase3_locust_${slug}.log"
                }
                $metaJson = $metaObj | ConvertTo-Json -Depth 4
                Write-Utf8Atomic -Path $metaPath -Content ($metaJson + "`n")
            }
            finally {
                try {
                    $global:LASTEXITCODE = 0
                    & $StartServerScript -Stop
                    if ($LASTEXITCODE -ne 0) {
                        Write-Warning "start_server.ps1 -Stop returned exit code $LASTEXITCODE after $slug"
                    }
                }
                catch {
                    Write-Warning "start_server.ps1 -Stop failed after $slug : $_"
                }
            }
        }
    }

    Write-Host "`n>>> Compiling Phase 3 Load Test Report via benchmarks\compile_report.py..." -ForegroundColor Yellow
    $global:LASTEXITCODE = 0
    & $Python "benchmarks\compile_report.py"
    if ($LASTEXITCODE -ne 0) {
        throw "benchmarks\compile_report.py failed with exit code $LASTEXITCODE"
    }

    Write-Host "`n==========================================================" -ForegroundColor Green
    Write-Host " PHASE 3 LOCUST BENCHMARK MATRIX & REPORT COMPLETE" -ForegroundColor Green
    Write-Host "==========================================================" -ForegroundColor Green
}
finally {
    foreach ($key in $SavedEnv.Keys) {
        [Environment]::SetEnvironmentVariable($key, $SavedEnv[$key], "Process")
    }
    Pop-Location
}
