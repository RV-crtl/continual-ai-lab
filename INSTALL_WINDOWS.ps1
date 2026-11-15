# Run from PowerShell in the extracted project folder:
# powershell -NoProfile -ExecutionPolicy Bypass -File .\INSTALL_WINDOWS.ps1
$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
Write-Host '=== Continual AI local setup ===' -ForegroundColor Cyan
$pythonCmd = $null
if (Get-Command py -ErrorAction SilentlyContinue) {
  & py -3 --version
  if ($LASTEXITCODE -eq 0) { $pythonCmd = 'py' }
}
if (-not $pythonCmd -and (Get-Command python -ErrorAction SilentlyContinue)) {
  & python --version
  if ($LASTEXITCODE -eq 0) { $pythonCmd = 'python' }
}
if (-not $pythonCmd) {
  throw 'Python 3.11-3.13 not found. Install Python from https://www.python.org/downloads/ (check Add to PATH), then rerun.'
}
if ($pythonCmd -eq 'py') { & py -3 -m venv .venv } else { & python -m venv .venv }
if ($LASTEXITCODE -ne 0) { throw 'Could not create virtual environment.' }
$venvPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
& $venvPython -m unittest discover -s tests -v
if ($LASTEXITCODE -ne 0) { throw 'Local project self-tests failed.' }
if (-not (Get-Command ollama -ErrorAction SilentlyContinue)) {
  Write-Host 'Ollama not on PATH. Install from https://ollama.com/download/windows' -ForegroundColor Yellow
  Write-Host 'Then open a NEW PowerShell window, return to this folder and run:'
  Write-Host '  ollama pull qwen3:8b'
  Write-Host '  .\.venv\Scripts\python.exe -m continual_ai.cli doctor'
  exit 0
}
Write-Host 'Downloading an initial local model (approximately 5 GB)...'
& ollama pull qwen3:8b
if ($LASTEXITCODE -ne 0) { throw 'Model download failed; try ollama pull qwen3:8b later.' }
& $venvPython -m continual_ai.cli doctor
if ($LASTEXITCODE -ne 0) { throw 'Diagnostics failed.' }
Write-Host 'Setup complete. Next: .\.venv\Scripts\python.exe -m continual_ai.cli chat' -ForegroundColor Green
