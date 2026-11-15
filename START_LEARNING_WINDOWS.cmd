@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Run INSTALL_WINDOWS.ps1 first.
  pause
  exit /b 1
)
".venv\Scripts\python.exe" -m continual_ai.inference_doctor --model qwen3:8b
if errorlevel 1 (
 echo Preflight failed. Learning NOT started.
 pause
 exit /b 1
)
".venv\Scripts\python.exe" -m continual_ai.cli loop --model qwen3:8b --pause 2
pause
