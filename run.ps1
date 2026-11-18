# Smart Library Intelligence System - PowerShell Launcher
$ErrorActionPreference = "Continue"
$projectDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $projectDir
$host.UI.RawUI.WindowTitle = "Smart Library Intelligence System"

$venvPy = Join-Path $projectDir ".venv\Scripts\python.exe"
$port = 8501

function Get-AppPid {
    try {
        $conn = Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue | Select-Object -First 1
        if ($conn) { return $conn.OwningProcess }
    } catch {}
    return $null
}

function Stop-AppServer {
    param([bool]$Quiet = $false)
    $p = Get-AppPid
    if ($p) {
        Stop-Process -Id $p -Force -ErrorAction SilentlyContinue
        if (-not $Quiet) { Write-Host "[OK] Stopped Smart Library (PID: $p)." -ForegroundColor Green }
    } else {
        if (-not $Quiet) { Write-Host "[INFO] No running instance found on port $port." -ForegroundColor Yellow }
    }
}

function Start-AppServer {
    $existing = Get-AppPid
    if ($existing) {
        Write-Host "[INFO] Smart Library is already running (PID: $existing)." -ForegroundColor Yellow
        return
    }

    $env:STREAMLIT_SERVER_HEADLESS = "true"
    $env:STREAMLIT_BROWSER_GATHER_USAGE_STATS = "false"

    Start-Process -FilePath $venvPy -ArgumentList "-m streamlit run app.py --server.port $port --server.headless true --browser.gatherUsageStats false" -WindowStyle Minimized
    Start-Sleep -Seconds 3
    Start-Process "http://localhost:$port"
    Write-Host "[OK] Smart Library started at http://localhost:$port" -ForegroundColor Green
}

# Handle arguments
if ($args.Count -gt 0) {
    switch ($args[0].ToLower()) {
        "stop" { Stop-AppServer; exit 0 }
        "status" {
            $p = Get-AppPid
            if ($p) { Write-Host "[RUNNING] Smart Library is active on port $port (PID: $p)." -ForegroundColor Green }
            else { Write-Host "[STOPPED] Smart Library is not running." -ForegroundColor Yellow }
            exit 0
        }
        "restart" {
            Stop-AppServer -Quiet $true
            Start-Sleep -Seconds 2
            Start-AppServer
            exit 0
        }
        "browser" { Start-Process "http://localhost:$port"; exit 0 }
        default {
            Write-Host "Usage: .\run.ps1 [stop|restart|status|browser]"
            exit 0
        }
    }
}

Clear-Host
Write-Host "===================================================" -ForegroundColor Cyan
Write-Host "       SMART LIBRARY INTELLIGENCE SYSTEM" -ForegroundColor Cyan
Write-Host "===================================================" -ForegroundColor Cyan
Write-Host " [1] New Installation (Install all dependencies & setup)"
Write-Host " [2] Normal Start     (Fast launch - recommended)"
Write-Host "==================================================="
Write-Host ""
$choice = Read-Host "Select option [1/2] (Default is 2)"
if ([string]::IsNullOrWhiteSpace($choice)) { $choice = "2" }

if ($choice -eq "1") {
    Write-Host "`n[INFO] Running New Installation..." -ForegroundColor Cyan
    if (-not (Test-Path $venvPy)) {
        Write-Host "[INFO] Creating virtual environment at .venv..."
        python -m venv .venv
    }
    Write-Host "[INFO] Upgrading pip..."
    & $venvPy -m pip install --upgrade pip --quiet
    Write-Host "[INFO] Installing required dependencies..."
    & $venvPy -m pip install -r requirements.txt
}

if (-not (Test-Path ".env") -and (Test-Path ".env.example")) {
    Copy-Item ".env.example" ".env"
}

if (-not (Test-Path "data\processed\hourly.parquet")) {
    Write-Host "[INFO] Preparing dataset..."
    & $venvPy "src\prepare_data.py"
}

if (-not (Test-Path "models\occupancy.joblib")) {
    Write-Host "[INFO] Training model..."
    & $venvPy "src\train.py"
}

# Launch App
Stop-AppServer -Quiet $true
Start-AppServer

Write-Host "`n===================================================" -ForegroundColor Cyan
Write-Host "   SMART LIBRARY IS RUNNING AT http://localhost:$port" -ForegroundColor Green
Write-Host "===================================================" -ForegroundColor Cyan
Write-Host " Console stays open. Type a command anytime:"
Write-Host "   stop     : Stop the library server"
Write-Host "   start    : Start the library server"
Write-Host "   restart  : Restart the library server"
Write-Host "   status   : Check server status"
Write-Host "   browser  : Reopen app in browser"
Write-Host "   exit     : Stop server and exit"
Write-Host "==================================================="

while ($true) {
    $cmd = Read-Host "`n[SmartLibrary]"
    if ([string]::IsNullOrWhiteSpace($cmd)) { continue }
    switch ($cmd.Trim().ToLower()) {
        "stop" { Stop-AppServer }
        "start" { Start-AppServer }
        "restart" {
            Stop-AppServer -Quiet $true
            Start-Sleep -Seconds 2
            Start-AppServer
        }
        "status" {
            $p = Get-AppPid
            if ($p) { Write-Host "[RUNNING] Active on port $port (PID: $p)." -ForegroundColor Green }
            else { Write-Host "[STOPPED] Smart Library is not running." -ForegroundColor Yellow }
        }
        "browser" { Start-Process "http://localhost:$port" }
        "open" { Start-Process "http://localhost:$port" }
        "cls" { Clear-Host }
        "clear" { Clear-Host }
        { $_ -in "exit", "quit", "q" } {
            Stop-AppServer -Quiet $true
            Write-Host "[OK] Smart Library stopped. Goodbye!" -ForegroundColor Green
            exit 0
        }
        default {
            Write-Host "[INFO] Unknown command: $cmd" -ForegroundColor Yellow
            Write-Host "Commands: stop, start, restart, status, browser, exit"
        }
    }
}
