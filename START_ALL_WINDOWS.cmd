@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo First run INSTALL_WINDOWS.ps1 from PowerShell.
  pause
  exit /b 1
)
start "Continual AI - Learning" cmd /k "cd /d ""%~dp0"" && .venv\Scripts\python.exe -m continual_ai.cli loop --pause 2"
start "Continual AI - Dashboard" cmd /k "cd /d ""%~dp0"" && .venv\Scripts\python.exe -m continual_ai.cli dashboard"
timeout /t 3 /nobreak >nul
start "" "http://127.0.0.1:8765"
".venv\Scripts\python.exe" -m continual_ai.cli chat
pause
