@echo off
REM ===================================================================
REM  Windows setup for monoculer_camera_yolo.
REM  Double-click this file, or run it from a Command Prompt.
REM  It creates venv\, installs the dependencies, and runs both tests.
REM ===================================================================
setlocal

cd /d "%~dp0"
echo.
echo ==== monoculer_camera_yolo setup =================================
echo Folder: %CD%
echo.

REM ---- 1. Is Python here, and is it new enough? ---------------------
where py >nul 2>nul
if errorlevel 1 (
    set "PY=python"
) else (
    set "PY=py -3"
)

%PY% --version >nul 2>nul
if errorlevel 1 (
    echo [X] Python was not found.
    echo.
    echo     Install Python 3.12 from https://www.python.org/downloads/
    echo     and TICK "Add python.exe to PATH" on the first screen.
    echo     Then close this window, open a new one, and run setup again.
    goto :fail
)

for /f "tokens=2" %%v in ('%PY% --version 2^>^&1') do set "PYVER=%%v"
echo [1/4] Found Python %PYVER%

%PY% -c "import sys; sys.exit(0 if sys.version_info[0]==3 and sys.version_info[1] in range(11,40) else 1)"
if errorlevel 1 (
    echo.
    echo [X] Python %PYVER% is too old. This needs 3.11 or newer; 3.12 is recommended.
    echo     Get it from https://www.python.org/downloads/
    goto :fail
)

REM ---- 2. Virtual environment --------------------------------------
if exist "venv\Scripts\python.exe" (
    echo [2/4] venv already exists, reusing it
) else (
    echo [2/4] Creating venv ...
    %PY% -m venv venv
    if errorlevel 1 (
        echo [X] Could not create the virtual environment.
        goto :fail
    )
)

REM ---- 3. Dependencies ---------------------------------------------
echo [3/4] Installing packages ^(a few minutes the first time^) ...
venv\Scripts\python.exe -m pip install --upgrade pip --quiet
venv\Scripts\python.exe -m pip install -r requirements.txt
if errorlevel 1 (
    echo [X] Package installation failed. Check your internet connection.
    goto :fail
)

REM ---- 4. Prove it works -------------------------------------------
echo.
echo [4/4] Running the self-tests ^(no API key or camera needed^) ...
echo.
venv\Scripts\python.exe selftest.py
if errorlevel 1 goto :fail
venv\Scripts\python.exe pipeline_test.py
if errorlevel 1 goto :fail

echo.
echo ==================================================================
echo  Setup finished. Next:
echo.
echo    1. Get a free key: https://app.roboflow.com/settings/api
echo    2. setx ROBOFLOW_API_KEY "your_key_here"
echo    3. Close this window, open a new Command Prompt, then:
echo.
echo       cd /d "%CD%"
echo       venv\Scripts\python.exe run.py --source 0 --model-id drone-detection-rchy7/8 --keep-classes 1 --conf 0.40 --hfov 70
echo.
echo  See SETUP_WINDOWS.md for the full walkthrough.
echo ==================================================================
echo.
pause
exit /b 0

:fail
echo.
echo Setup did not finish. See SETUP_WINDOWS.md, section "If something breaks".
echo.
pause
exit /b 1
