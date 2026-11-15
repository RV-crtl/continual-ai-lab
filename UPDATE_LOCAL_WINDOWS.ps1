# Copy source-only release files from extracted GitHub ZIP into EXISTING install.
# Preserves all local data, models, virtual environments, and chat history.
param([Parameter(Mandatory=$true)][string]$Target)
$ErrorActionPreference = 'Stop'
$source=(Resolve-Path $PSScriptRoot).Path
$dest=(Resolve-Path $Target).Path
if (-not (Test-Path (Join-Path $dest 'continual_ai\cli.py'))) {
  throw 'Target is not an existing Continual AI installation. Pass the directory containing continual_ai\cli.py.'
}
if ($source -eq $dest) { throw 'Source and destination must differ.' }
$backup=Join-Path $dest ('backups\source-'+(Get-Date -Format 'yyyyMMdd-HHmmss'))
New-Item -ItemType Directory -Path $backup -Force | Out-Null
$paths=@('continual_ai','tests','.github\workflows','README.md','.gitignore',
 'requirements-train.txt','INSTALL_WINDOWS.ps1','INSTALL_TRAINING_WINDOWS.ps1',
 'CHECK_HARDWARE_WINDOWS.ps1','UPDATE_LOCAL_WINDOWS.ps1',
 'START_ALL_WINDOWS.cmd','START_CHAT_WINDOWS.cmd','START_DASHBOARD_WINDOWS.cmd',
 'START_LEARNING_WINDOWS.cmd','START_AGENTS_WINDOWS.cmd','START_AUTO_TRAIN_WINDOWS.cmd',
 'START_TRAINED_CHAT_WINDOWS.cmd')
foreach($item in $paths){
  $from=Join-Path $source $item
  $to=Join-Path $dest $item
  if(-not (Test-Path $from)){ throw "Release file missing: $from" }
  if(Test-Path $to){
    $old=Join-Path $backup $item
    New-Item -ItemType Directory -Path (Split-Path $old) -Force | Out-Null
    Copy-Item $to $old -Recurse -Force
  }
}
foreach($item in $paths){
  $from=Join-Path $source $item
  $to=Join-Path $dest $item
  if(Test-Path $from -PathType Container){
    New-Item -ItemType Directory -Path $to -Force | Out-Null
    Copy-Item (Join-Path $from '*') $to -Recurse -Force
  } else {
    Copy-Item $from $to -Force
  }
}
Write-Host "Source backup: $backup" -ForegroundColor Cyan
Write-Host 'Existing chat SQLite database and training checkpoints remain unchanged.'
$py=Join-Path $dest '.venv\Scripts\python.exe'
if(Test-Path $py){
  Push-Location $dest
  try { & $py -m unittest discover -s tests -v; if($LASTEXITCODE -ne 0){throw 'Updated code tests failed; review backup before restarting'} }
  finally {Pop-Location}
}
Write-Host 'Local source update installed. Restart START_ALL_WINDOWS.cmd.' -ForegroundColor Green
