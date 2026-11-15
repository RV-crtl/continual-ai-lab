$ErrorActionPreference = 'Continue'
Write-Host '=== Windows and CPU ===' -ForegroundColor Cyan
Get-CimInstance Win32_ComputerSystem | Select-Object Manufacturer, Model, TotalPhysicalMemory | Format-List
Get-CimInstance Win32_Processor | Select-Object Name, NumberOfCores, NumberOfLogicalProcessors | Format-List
Write-Host '=== Graphics adaptors (AdapterRAM can be inaccurate on Windows) ===' -ForegroundColor Cyan
Get-CimInstance Win32_VideoController | Select-Object Name, AdapterRAM, DriverVersion | Format-Table -AutoSize
Write-Host '=== NVIDIA VRAM (preferred if available) ===' -ForegroundColor Cyan
if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
  & nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
} else {
  Write-Host 'nvidia-smi not found. This may mean there is no NVIDIA GPU or drivers are not installed.'
}
Write-Host '=== Local storage ===' -ForegroundColor Cyan
Get-PSDrive -PSProvider FileSystem | Select-Object Name,Used,Free | Format-Table -AutoSize
