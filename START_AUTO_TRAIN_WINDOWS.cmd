@echo off
cd /d "%~dp0"
if not exist ".venv-train\Scripts\python.exe" (
 echo First run INSTALL_TRAINING_WINDOWS.ps1 from PowerShell.
 pause
 exit /b 1
)
".venv-train\Scripts\python.exe" -m continual_ai.training_pipeline --check-every 600 --min-new 96 --max-steps 40 --test-count 60
pause
