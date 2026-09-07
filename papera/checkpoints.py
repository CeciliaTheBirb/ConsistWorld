"""Strict full-model checkpoint loading for the PaperA release."""
from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import nn


_WRAPPER_PREFIXES = (
    "model.",
    "_checkpoint_wrapped_module.",
    "_fsdp_wrapped_module.",
    "_orig_mod.",
)


def _strip_wrapper_prefixes(key: str) -> str:
    while True:
        prefix = next((item for item in _WRAPPER_PREFIXES if key.startswith(item)), None)
        if prefix is None:
            return key
        key = key[len(prefix):]


def load_full_state_dict(path: str) -> dict[str, torch.Tensor]:
    """Load a full ``model_full.pt`` checkpoint and normalize wrapper prefixes."""
    state = torch.load(path, map_location="cpu", mmap=True, weights_only=True)
    if not isinstance(state, Mapping):
        raise TypeError(f"checkpoint must be a state dict, got {type(state).__name__}")

    normalized: dict[str, torch.Tensor] = {}
    for raw_key, value in state.items():
        if not isinstance(raw_key, str) or not isinstance(value, torch.Tensor):
            raise TypeError("checkpoint must map string parameter names to tensors")
        key = _strip_wrapper_prefixes(raw_key)
        if key in normalized:
            raise RuntimeError(f"duplicate checkpoint key after normalization: {key}")
        normalized[key] = value
    return normalized


def load_strict_full_checkpoint(model: nn.Module, path: str) -> None:
    """Load a release checkpoint without filling or ignoring missing tensors."""
    state = load_full_state_dict(path)
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as exc:
        raise RuntimeError(
            f"checkpoint does not match the PaperA model architecture: {path}"
        ) from exc
