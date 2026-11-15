"""Minimal hardware diagnostics without requiring a GPU library."""
from __future__ import annotations
import platform
import shutil
import subprocess


def detect() -> dict:
    info = {"os": platform.platform(), "cpu": platform.processor(), "python": platform.python_version(),
            "nvidia_smi_present": shutil.which("nvidia-smi") is not None}
    if info["nvidia_smi_present"]:
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=10, check=True,
            )
            info["nvidia_gpus"] = out.stdout.strip().splitlines()
        except (OSError, subprocess.SubprocessError):
            info["nvidia_gpus"] = ["Could not query NVIDIA GPU"]
    if platform.system() == "Windows":
        try:
            cmd = ["powershell", "-NoProfile", "-Command",
                   "Get-CimInstance Win32_VideoController | Select-Object Name,AdapterRAM | Format-Table -AutoSize"]
            x = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
            info["windows_graphics"] = x.stdout.strip()
        except (OSError, subprocess.SubprocessError):
            pass
    return info
