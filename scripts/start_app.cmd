@echo off
rem ---------------------------------------------------------------------------
rem  GeoData AI - one-click start (Windows)
rem  Starts the API + map frontend and opens it in the browser.
rem  Close this window (or press Ctrl+C) to stop the server.
rem  Optional: GEODATA_PORT=8001 to use another port, GEODATA_NO_BROWSER=1 to skip the browser.
rem ---------------------------------------------------------------------------
setlocal
title GeoData AI - Fire Severity Map
cd /d "%~dp0.."

if defined GEODATA_PORT (set "PORT=%GEODATA_PORT%") else (set "PORT=8000")
set "URL=http://127.0.0.1:%PORT%/"

rem Already running? Just open the browser.
powershell -NoProfile -Command "try { Invoke-WebRequest -UseBasicParsing '%URL%health' -TimeoutSec 2 | Out-Null; exit 0 } catch { exit 1 }"
if %errorlevel%==0 (
    echo GeoData AI is already running. Opening %URL%
    if not defined GEODATA_NO_BROWSER start "" "%URL%"
    exit /b 0
)

rem First run: create the virtual environment and install the requirements.
if not exist ".venv\Scripts\python.exe" (
    echo First run: setting up the Python environment. This takes a few minutes...
    python -m venv .venv || goto :python_missing
    ".venv\Scripts\python.exe" -m pip install --upgrade pip
    ".venv\Scripts\python.exe" -m pip install -r requirements.txt || goto :install_failed
)

rem Open the browser as soon as the API answers (loading the data takes ~20 s).
if not defined GEODATA_NO_BROWSER start "" /b powershell -NoProfile -WindowStyle Hidden -Command "for ($i = 0; $i -lt 180; $i++) { try { Invoke-WebRequest -UseBasicParsing '%URL%health' -TimeoutSec 2 | Out-Null; Start-Process '%URL%'; break } catch { Start-Sleep -Seconds 1 } }"

echo.
echo  GeoData AI is starting on %URL%
echo  The browser opens by itself when the data has loaded.
echo  Keep this window open while you use the app. Close it to stop the server.
echo.
".venv\Scripts\python.exe" -m uvicorn api.main:app --host 127.0.0.1 --port %PORT%
exit /b %errorlevel%

:python_missing
echo.
echo Python was not found. Install Python 3.10 or newer from https://www.python.org and try again.
pause
exit /b 1

:install_failed
echo.
echo Installing the requirements failed. See the messages above.
pause
exit /b 1
