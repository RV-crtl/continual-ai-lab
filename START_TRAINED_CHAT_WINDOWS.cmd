@echo off
cd /d "%~dp0"
if not exist ".venv-train\Scripts\python.exe" (
 echo First run INSTALL_TRAINING_WINDOWS.ps1 from PowerShell.
 pause
 exit /b 1
)
".venv-train\Scripts\python.exe" -m continual_ai.chat_trained
pause
