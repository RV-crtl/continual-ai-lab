@echo off
cd /d "%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0PUBLISH_GITHUB_WINDOWS.ps1"
set EXITCODE=%ERRORLEVEL%
echo.
if not "%EXITCODE%"=="0" echo Publishing failed (exit code %EXITCODE%). Read the error above.
pause
exit /b %EXITCODE%
