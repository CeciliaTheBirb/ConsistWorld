"""Device adapter for CUDA, Ascend NPU, and CPU-only validation.

Provides a single source of truth for `device_type` / `device_module` so the rest
of the code can stay device-agnostic.
"""
from __future__ import annotations

import os
from typing import Any

import torch
from torch._utils import _get_available_device_type, _get_device_module


def _detect() -> tuple[str, Any]:
    dtype = _get_available_device_type()
    if dtype is None:
        # Keep imports and argument parsing usable on a CPU-only host.  The
        # entry points that require a 14B accelerator run call
        # ``require_accelerator`` before allocating a model.
        dtype = "cpu"
    mod = _get_device_module(dtype)
    return dtype, mod


device_type, device_module = _detect()

if device_type == "npu":
    import torch_npu  # noqa: F401  (registers torch.npu / hccl backend)


def is_npu() -> bool:
    return device_type == "npu"


def is_cuda() -> bool:
    return device_type == "cuda"


def has_accelerator() -> bool:
    """Whether a supported CUDA or Ascend device is available."""
    return device_type in {"cuda", "npu"}


def require_accelerator(operation: str) -> None:
    """Fail before model construction when an accelerator is unavailable."""
    if has_accelerator():
        return
    raise RuntimeError(
        f"{operation} requires a CUDA GPU or Ascend NPU. "
        "CPU-only execution is supported only for metadata and format checks."
    )


def current_device() -> int:
    if hasattr(device_module, "current_device"):
        return int(device_module.current_device())
    return 0


def empty_cache() -> None:
    if hasattr(device_module, "empty_cache"):
        device_module.empty_cache()


def synchronize(device: Any | None = None) -> None:
    if not hasattr(device_module, "synchronize"):
        return
    if device is None:
        device_module.synchronize()
    else:
        device_module.synchronize(device)


def manual_seed(seed: int) -> None:
    torch.manual_seed(seed=seed)
    if hasattr(device_module, "manual_seed"):
        device_module.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed % 2**32)


def manual_seed_all(seed: int) -> None:
    torch.manual_seed(seed=seed)
    if hasattr(device_module, "manual_seed_all"):
        device_module.manual_seed_all(seed)
    elif hasattr(device_module, "manual_seed"):
        device_module.manual_seed(seed)


__all__ = [
    "device_type",
    "device_module",
    "is_npu",
    "is_cuda",
    "has_accelerator",
    "require_accelerator",
    "current_device",
    "empty_cache",
    "synchronize",
    "manual_seed",
    "manual_seed_all",
]
