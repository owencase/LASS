"""Shared inference scheduling and device/memory management."""

from __future__ import annotations

import gc
import sys
from functools import wraps
from threading import RLock
from typing import Any


# Streamlit sessions share a process. Never run two model jobs simultaneously.
INFERENCE_LOCK = RLock()


def release_memory() -> None:
    gc.collect()
    torch = sys.modules.get("torch")
    if torch is None:
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        torch.mps.empty_cache()


def resolve_device(torch: Any, requested: str = "auto") -> str:
    requested = requested.strip().lower()
    mps_available = bool(
        getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
    )
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "mps" if mps_available else "cpu"
    if requested == "mps" and not mps_available:
        raise RuntimeError("Apple GPU(MPS)를 사용할 수 없습니다. CPU 또는 자동을 선택해 주세요.")
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU를 사용할 수 없습니다. CPU 또는 자동을 선택해 주세요.")
    if requested not in {"cpu", "mps", "cuda", "cuda:0"}:
        raise ValueError(f"지원하지 않는 장치: {requested}")
    return requested


def memory_managed(method):
    """Release model weights on success AND failure; serialize across profiles."""
    @wraps(method)
    def run(self, *args, **kwargs):
        with INFERENCE_LOCK:
            try:
                return method(self, *args, **kwargs)
            finally:
                self.unload()
    return run
