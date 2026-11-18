@echo off
setlocal
cd /d "%~dp0"
title Smart Library Intelligence System

set "VENV_DIR=%~dp0.venv"
set "VENV_PY=%~dp0.venv\Scripts\python.exe"
set "PORT=8501"     

rem Check CLI arguments first
set "CMD=%~1"
if /i "%CMD%"=="stop" goto do_stop
if /i "%CMD%"=="restart" goto do_restart
if /i "%CMD%"=="status" goto do_status
if /i "%CMD%"=="browser" goto do_open_browser
if /i "%CMD%"=="help" goto do_help
if not "%CMD%"=="" (
    echo [ERROR] Unknown command: %CMD%
    goto do_help
)

rem Interactive Menu
cls
echo ===================================================
echo        SMART LIBRARY INTELLIGENCE SYSTEM
echo ===================================================
echo  [1] New Installation (Install all dependencies)
echo  [2] Normal Start     (Fast launch - recommended)
echo ===================================================
echo.
set "CHOICE=2"
set /p "CHOICE=Select option [1/2] (Default is 2): "

if "%CHOICE%"=="1" goto setup_new
goto setup_fast

rem ----------------------------------------------------
rem NEW INSTALLATION
rem ----------------------------------------------------
:setup_new
echo.
echo [INFO] Running New Installation...
call :find_sys_python

if not exist "%VENV_PY%" (
    echo [INFO] Creating virtual environment at .venv...
    "%SYS_PY%" -m venv "%VENV_DIR%"
    if errorlevel 1 (
        echo [ERROR] Failed to create virtual environment.
        pause
        exit /b 1
    )
)

echo [INFO] Upgrading pip...
"%VENV_PY%" -m pip install --upgrade pip --quiet

echo [INFO] Installing required dependencies from requirements.txt...
"%VENV_PY%" -m pip install -r requirements.txt
if errorlevel 1 (
    echo [ERROR] Dependency installation encountered an issue.
    pause
    exit /b 1
)

call :check_env_file
call :check_data_and_model
goto launch_app

rem ----------------------------------------------------
rem NORMAL FAST START
rem ----------------------------------------------------
:setup_fast
echo.
echo [INFO] Starting Smart Library (Fast Mode)...

if not exist "%VENV_PY%" (
    echo [WARNING] Virtual environment not found. Switching to setup...
    goto setup_new
)

call :check_env_file
call :check_data_and_model
goto launch_app

rem ----------------------------------------------------
rem APP LAUNCH & INTERACTIVE CONTROLLER
rem ----------------------------------------------------
:launch_app
set "PY=%VENV_PY%"

rem Verify streamlit exists
"%PY%" -c "import streamlit" >nul 2>nul
if errorlevel 1 (
    echo [WARNING] Streamlit not found in .venv. Installing dependencies...
    goto setup_new
)

rem Stop any old instance on port 8501
call :kill_port_quiet

echo [INFO] Starting Streamlit server...
set "STREAMLIT_SERVER_HEADLESS=true"
set "STREAMLIT_BROWSER_GATHER_USAGE_STATS=false"

start "Smart Library Server" /min cmd /c ""%PY%" -m streamlit run app.py --server.port %PORT% --server.headless true --browser.gatherUsageStats false"

echo [INFO] Waiting for server to initialize...
timeout /t 3 /nobreak >nul

call :do_open_browser

echo.
echo ===================================================
echo   SMART LIBRARY IS RUNNING AT http://localhost:%PORT%
echo ===================================================
echo   Console stays open. Type a command anytime:
echo     stop     : Stop the library server
echo     start    : Start the library server
echo     restart  : Restart the library server
echo     status   : Check if server is running
echo     browser  : Reopen app in browser
echo     reinstall: Re-run full dependency installation
echo     exit     : Stop server and exit
echo ===================================================

:cmd_loop
echo.
set "USER_CMD="
set /p "USER_CMD=[SmartLibrary]> "

if /i "%USER_CMD%"=="stop" goto action_stop
if /i "%USER_CMD%"=="start" goto action_start
if /i "%USER_CMD%"=="restart" goto action_restart
if /i "%USER_CMD%"=="status" goto action_status
if /i "%USER_CMD%"=="browser" goto action_browser
if /i "%USER_CMD%"=="open" goto action_browser
if /i "%USER_CMD%"=="reinstall" goto setup_new
if /i "%USER_CMD%"=="setup" goto setup_new
if /i "%USER_CMD%"=="exit" goto action_exit
if /i "%USER_CMD%"=="quit" goto action_exit
if /i "%USER_CMD%"=="q" goto action_exit
if /i "%USER_CMD%"=="cls" (
    cls
    goto cmd_loop
)

if "%USER_CMD%"=="" goto cmd_loop

echo [INFO] Unknown command: %USER_CMD%
echo Valid commands: stop, start, restart, status, browser, reinstall, exit
goto cmd_loop

rem ----------------------------------------------------
rem INTERACTIVE COMMAND ACTIONS
rem ----------------------------------------------------
:action_stop
call :kill_port
goto cmd_loop

:action_start
call :is_running
if not "%RUNNING_PID%"=="" (
    echo [INFO] Smart Library is already running [PID %RUNNING_PID%].
    goto cmd_loop
)
start "Smart Library Server" /min cmd /c ""%VENV_PY%" -m streamlit run app.py --server.port %PORT% --server.headless true --browser.gatherUsageStats false"
timeout /t 2 /nobreak >nul
call :do_open_browser
echo [OK] Smart Library started.
goto cmd_loop

:action_restart
echo [INFO] Restarting Smart Library...
call :kill_port_quiet
timeout /t 2 /nobreak >nul
start "Smart Library Server" /min cmd /c ""%VENV_PY%" -m streamlit run app.py --server.port %PORT% --server.headless true --browser.gatherUsageStats false"
timeout /t 2 /nobreak >nul
call :do_open_browser
echo [OK] Restarted successfully.
goto cmd_loop

:action_status
call :is_running
if not "%RUNNING_PID%"=="" (
    echo [STATUS] RUNNING - Active on port %PORT% [PID %RUNNING_PID%]
    echo          URL: http://localhost:%PORT%
) else (
    echo [STATUS] STOPPED - Smart Library is not running.
    echo          Type 'start' to launch it.
)
goto cmd_loop

:action_browser
call :do_open_browser
goto cmd_loop

:action_exit
echo [INFO] Stopping Smart Library...
call :kill_port_quiet
echo [OK] Smart Library stopped. Goodbye!
timeout /t 1 >nul
exit /b 0

rem ----------------------------------------------------
rem DIRECT CLI ACTIONS
rem ----------------------------------------------------
:do_stop
call :kill_port
exit /b 0

:do_restart
call :kill_port_quiet
timeout /t 2 /nobreak >nul
goto setup_fast

:do_status
call :is_running
if not "%RUNNING_PID%"=="" (
    echo [RUNNING] Smart Library is active on port %PORT% [PID %RUNNING_PID%].
) else (
    echo [STOPPED] Smart Library is not running.
)
exit /b 0

:do_open_browser
echo [INFO] Opening http://localhost:%PORT% in your browser...
start "" "http://localhost:%PORT%"
exit /b 0

:do_help
echo.
echo Usage: run.bat [command]
echo.
echo Commands:
echo   (none)    Interactive menu (New vs Normal Fast start)
echo   stop      Stop the running app
echo   restart   Restart the app
echo   status    Check if the app is currently running
echo   browser   Open http://localhost:8501 in default browser
echo   help      Show this help message
echo.
exit /b 0

rem ----------------------------------------------------
rem HELPER FUNCTIONS
rem ----------------------------------------------------
:find_sys_python
set "SYS_PY=python"
where python >nul 2>nul
if errorlevel 1 (
    where py >nul 2>nul
    if not errorlevel 1 (
        set "SYS_PY=py"
    ) else (
        echo [ERROR] Python not found on your system. Please install Python 3.10+.
        pause
        exit /b 1
    )
)
exit /b 0

:check_env_file
if not exist ".env" (
    if exist ".env.example" (
        echo [INFO] Creating .env from .env.example...
        copy ".env.example" ".env" >nul
    )
)
exit /b 0

:check_data_and_model
if not exist "data\processed\hourly.parquet" (
    echo [INFO] Preparing dataset...
    "%VENV_PY%" src\prepare_data.py
    if errorlevel 1 (
        echo [ERROR] Data preparation failed.
        pause
        exit /b 1
    )
)
if not exist "models\occupancy.joblib" (
    echo [INFO] Training occupancy prediction model...
    "%VENV_PY%" src\train.py
    if errorlevel 1 (
        echo [ERROR] Model training failed.
        pause
        exit /b 1
    )
)
exit /b 0

:is_running
set "RUNNING_PID="
for /f "tokens=5" %%a in ('netstat -ano 2^>nul ^| findstr /c:":%PORT% " ^| findstr LISTENING') do (
    set "RUNNING_PID=%%a"
)
exit /b 0

:kill_port
set "KILLED=0"
for /f "tokens=5" %%a in ('netstat -ano 2^>nul ^| findstr /c:":%PORT% " ^| findstr LISTENING') do (
    taskkill /F /PID %%a >nul 2>&1
    set "KILLED=1"
)
if "%KILLED%"=="1" (
    echo [OK] Stopped running instance on port %PORT%.
) else (
    echo [INFO] No running instance found on port %PORT%.
)
exit /b 0

:kill_port_quiet
for /f "tokens=5" %%a in ('netstat -ano 2^>nul ^| findstr /c:":%PORT% " ^| findstr LISTENING') do (
    taskkill /F /PID %%a >nul 2>&1
)
exit /b 0
