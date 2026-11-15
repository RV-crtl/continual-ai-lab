@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
 echo First run INSTALL_WINDOWS.ps1 from PowerShell.
 pause
 exit /b 1
)
start "Continual AI - Multi Agents" cmd /k "cd /d ""%~dp0"" && .venv\Scripts\python.exe -m continual_ai.multi_agent local --pause 2"
start "Continual AI - Dashboard" cmd /k "cd /d ""%~dp0"" && .venv\Scripts\python.exe -m continual_ai.cli dashboard"
if exist ".venv-train\Scripts\python.exe" (
 start "Continual AI - GPU Training Governor" cmd /k "cd /d ""%~dp0"" && .venv-train\Scripts\python.exe -m continual_ai.training_pipeline --check-every 600"
)
timeout /t 3 /nobreak >nul
start "" "http://127.0.0.1:8765"
".venv\Scripts\python.exe" -m continual_ai.cli chat
pause
