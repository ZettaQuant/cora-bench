"""Latency measurement."""

from __future__ import annotations

import time
from contextlib import contextmanager


@contextmanager
def timed(sync_cuda: bool = False):
    """Yields a dict; on exit dict['seconds'] holds wall-clock. Syncs CUDA if requested."""
    if sync_cuda:
        _cuda_sync()
    out: dict = {}
    t0 = time.perf_counter()
    try:
        yield out
    finally:
        if sync_cuda:
            _cuda_sync()
        out["seconds"] = time.perf_counter() - t0


def _cuda_sync() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        pass
