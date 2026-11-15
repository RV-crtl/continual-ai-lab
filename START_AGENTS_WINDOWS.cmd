@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
 echo First run INSTALL_WINDOWS.ps1 from PowerShell.
 pause
 exit /b 1
)
".venv\Scripts\python.exe" -m continual_ai.multi_agent local --pause 2
pause
