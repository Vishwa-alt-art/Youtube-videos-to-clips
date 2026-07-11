@echo off
REM ============================================================
REM  Free YouTube Clipper - one-click launcher for Windows
REM  First run: creates a virtual env + installs everything.
REM  Later runs: just starts the app.
REM ============================================================
setlocal
cd /d "%~dp0"

REM --- check python ---
where python >nul 2>nul
if errorlevel 1 (
  echo [X] Python is not installed or not on PATH.
  echo     Install Python 3.9+ from https://www.python.org/downloads/
  echo     ^(tick "Add Python to PATH" during setup^), then run this again.
  pause
  exit /b 1
)

REM --- check ffmpeg ---
where ffmpeg >nul 2>nul
if errorlevel 1 (
  echo [!] FFmpeg was not found on PATH.
  echo     Download from https://www.gyan.dev/ffmpeg/builds/ ^(get the "full" build^),
  echo     unzip it, and add the bin\ folder to your PATH. Then run this again.
  pause
  exit /b 1
)

REM --- create venv on first run ---
if not exist ".venv\Scripts\python.exe" (
  echo [*] First-time setup: creating virtual environment...
  python -m venv .venv
  echo [*] Installing packages ^(this can take a few minutes^)...
  call .venv\Scripts\python.exe -m pip install --upgrade pip
  call .venv\Scripts\python.exe -m pip install -r requirements.txt
)

echo.
echo ============================================================
echo   Starting Free YouTube Clipper...
echo   Open your browser at:  http://localhost:8000
echo   ^(Press Ctrl+C in this window to stop^)
echo ============================================================
echo.

REM open the browser automatically after a short delay
start "" cmd /c "timeout /t 3 >nul & start http://localhost:8000"

call .venv\Scripts\python.exe app.py
pause
