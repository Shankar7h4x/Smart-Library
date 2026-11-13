@echo off
setlocal
cd /d "%~dp0"
title Smart Library Intelligence

set "CMD=%~1"
if /i "%CMD%"=="stop" goto do_stop
if /i "%CMD%"=="restart" goto do_restart
if /i "%CMD%"=="status" goto do_status
if /i "%CMD%"=="help" goto do_help
if /i "%CMD%"=="" goto do_start

echo [ERROR] Unknown command: %CMD%
goto do_help

:do_start
echo ===================================================
echo     SMART LIBRARY INTELLIGENCE SYSTEM
echo ===================================================
echo  Ctrl+C       Close the app
echo  run stop     Close from another terminal
echo  run restart  Restart the app
echo  run status   Check if running
echo  F5 / R       Refresh in browser
echo ===================================================
echo.

set "PY=python"
where python >nul 2>nul
if errorlevel 1 (
    where py >nul 2>nul
    if not errorlevel 1 (
        set "PY=py"
    ) else (
        echo [ERROR] Python not found. Install Python 3.10+ and try again.
        pause
        exit /b 1
    )
)

if not exist "data\processed\hourly.parquet" (
    echo [INFO] Preparing dataset...
    %PY% src\prepare_data.py
    if errorlevel 1 (
        echo [ERROR] Data preparation failed.
        pause
        exit /b 1
    )
)

if not exist "models\occupancy.joblib" (
    echo [INFO] Training model...
    %PY% src\train.py
    if errorlevel 1 (
        echo [ERROR] Model training failed.
        pause
        exit /b 1
    )
)

echo [INFO] Starting Smart Library...
echo [INFO] Opening http://localhost:8501 in your browser...
echo.
start "" http://localhost:8501
%PY% -m streamlit run app.py --server.headless false --browser.gatherUsageStats false
pause
exit /b 0

:do_stop
echo [INFO] Stopping Smart Library...
python -c "import subprocess; res = subprocess.run(['powershell', '-NoProfile', '-Command', 'Get-NetTCPConnection -LocalPort 8501 -State Listen -ErrorAction SilentlyContinue | Select-Object -ExpandProperty OwningProcess -Unique'], capture_output=True, text=True); pids = [p.strip() for p in res.stdout.strip().splitlines() if p.strip()]; [subprocess.run(['taskkill', '/F', '/PID', pid], capture_output=True) for pid in pids]; print(f'[OK] Stopped {len(pids)} process(es).' if pids else '[INFO] No running instance found on port 8501.')"
exit /b 0

:do_restart
call :do_stop
echo.
timeout /t 2 /nobreak >nul
goto do_start

:do_status
python -c "import subprocess; res = subprocess.run(['powershell', '-NoProfile', '-Command', 'Get-NetTCPConnection -LocalPort 8501 -State Listen -ErrorAction SilentlyContinue | Select-Object -ExpandProperty OwningProcess -Unique'], capture_output=True, text=True); pids = [p.strip() for p in res.stdout.strip().splitlines() if p.strip()]; print(f'[RUNNING] Smart Library is active on port 8501 (PID {pids[0]}).' if pids else '[STOPPED] Smart Library is not running.')"
exit /b 0

:do_help
echo.
echo Usage: run.bat [command]
echo.
echo Commands:
echo   (none)    Start the application
echo   stop      Kill the running app
echo   restart   Stop then start again
echo   status    Check if the app is running
echo   help      Show this help
echo.
exit /b 0
