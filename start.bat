@echo off
REM SpillWatch / Marine Trace AI - one-click start for demos
REM Starts the backend (port 8000) and the website (port 5173) in two windows, then opens the browser.
cd /d "%~dp0"
if not exist venv\Scripts\activate.bat (
  echo Python environment "venv" not found. Create it first: python -m venv venv
  pause
  exit /b 1
)
start "SpillWatch backend" cmd /k "cd /d "%~dp0backend" && call ..\venv\Scripts\activate.bat && uvicorn app.main:app --reload"
start "SpillWatch website" cmd /k "cd /d "%~dp0frontend" && npm run dev"
echo Waiting for the AI models to load (about 20 seconds)...
timeout /t 20 /nobreak > nul
start http://localhost:5173
echo SpillWatch is running. Close the two black windows to stop it.
