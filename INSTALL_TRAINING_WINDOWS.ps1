# OPTIONAL: separate GPU fine-tuning environment. Run from Windows PowerShell.
$ErrorActionPreference='Stop'
Set-Location $PSScriptRoot
if (-not (Get-Command nvidia-smi -ErrorAction SilentlyContinue)) { throw 'NVIDIA GPU driver/nvidia-smi unavailable. Install a compatible driver first.' }
if (Get-Command py -ErrorAction SilentlyContinue) { $py = 'py' } else { $py = 'python' }
if ($py -eq 'py') { & py -3.12 -m venv .venv-train } else { & python -m venv .venv-train }
if ($LASTEXITCODE -ne 0) { throw 'Python 3.12 not available. Install 64-bit Python 3.12 and retry.' }
$trainPy = Join-Path $PSScriptRoot '.venv-train\Scripts\python.exe'
& $trainPy -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) { throw 'pip upgrade failed' }
Write-Host 'Installing CUDA 12.8 PyTorch wheel for Blackwell (sm_120) ...' -ForegroundColor Cyan
& $trainPy -m pip install --index-url https://download.pytorch.org/whl/cu128 torch
if ($LASTEXITCODE -ne 0) { throw 'CUDA PyTorch install failed. Check https://pytorch.org/get-started/locally/' }
& $trainPy -m pip install -r requirements-train.txt
if ($LASTEXITCODE -ne 0) { throw 'Training dependencies failed' }
& $trainPy -c "import torch, transformers, peft, bitsandbytes; print('Torch:',torch.__version__, 'CUDA:',torch.version.cuda); print('GPU:',torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE'); print('Compute capability:',torch.cuda.get_device_capability(0) if torch.cuda.is_available() else 'NONE'); assert torch.cuda.is_available(), 'CUDA unavailable'; assert torch.cuda.get_device_capability(0)[0] >= 12, 'Expected Blackwell sm_120'; x=torch.randn(16,16,device='cuda'); print('GPU smoke test:',float(x.square().mean()))"
if ($LASTEXITCODE -ne 0) { throw 'GPU training smoke test failed. Training disabled. Keep local learning running without weight updates.' }
Write-Host 'GPU training environment ready. Run START_AUTO_TRAIN_WINDOWS.cmd when you are on AC power.' -ForegroundColor Green
