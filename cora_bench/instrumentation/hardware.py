"""Hardware capture for the run manifest and GPU-cost accounting."""

from __future__ import annotations

import platform
import subprocess


def gpu_info() -> dict:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total,count", "--format=csv,noheader,nounits"],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=10,
        )
        rows = [r.strip() for r in out.strip().splitlines() if r.strip()]
        names = [r.split(",")[0].strip() for r in rows]
        return {"gpu_type": names[0] if names else None, "gpu_count": len(names)}
    except Exception:
        return {"gpu_type": None, "gpu_count": 0}


def peak_vram_gb() -> float | None:
    try:
        import torch

        if torch.cuda.is_available():
            return round(torch.cuda.max_memory_allocated() / 1e9, 3)
    except Exception:
        pass
    return None


def host_info() -> dict:
    return {"platform": platform.platform(), "python": platform.python_version(), **gpu_info()}
