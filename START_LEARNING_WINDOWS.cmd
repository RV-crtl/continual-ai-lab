@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Run INSTALL_WINDOWS.ps1 first.
  pause
  exit /b 1
)
".venv\Scripts\python.exe" -m continual_ai.cli loop --pause 2
pause
