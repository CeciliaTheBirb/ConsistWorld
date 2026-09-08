"""Strict full-model checkpoint loading for the ConsistWorld release."""
from __future__ import annotations

from collections.abc import Mapping
import re

import torch
from torch import nn


_WRAPPER_PREFIXES = (
    "model.",
    "_checkpoint_wrapped_module.",
    "_fsdp_wrapped_module.",
    "_orig_mod.",
)

_MEM_KEY_MARKER = re.compile(r"^blocks\.\d+\.mem_key_marker$")


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
            f"checkpoint does not match the ConsistWorld model architecture: {path}"
        ) from exc


def load_consistworld_warm_start(model: nn.Module, path: str) -> list[str]:
    """Load a Stage-1 warm start, initializing only new zero-init P-Mem tags.

    The released Stage-1 trainers save the tags too. This narrow compatibility
    path exists for the original Stage-1A/1B checkpoints, which predate them.
    Every other missing or unexpected tensor remains an error.
    """
    state = load_full_state_dict(path)
    expected = set(model.state_dict())
    actual = set(state)
    missing = expected - actual
    unexpected = actual - expected
    allowed_missing = {name for name in missing if _MEM_KEY_MARKER.fullmatch(name)}
    if unexpected or missing != allowed_missing:
        details = []
        if missing - allowed_missing:
            details.append(f"missing={sorted(missing - allowed_missing)[:8]}")
        if unexpected:
            details.append(f"unexpected={sorted(unexpected)[:8]}")
        raise RuntimeError(
            f"checkpoint does not match the ConsistWorld warm-start architecture: {path}; "
            + "; ".join(details)
        )

    result = model.load_state_dict(state, strict=False)
    if set(result.missing_keys) != allowed_missing or result.unexpected_keys:
        raise RuntimeError(f"checkpoint loading changed unexpectedly: {path}")
    parameters = dict(model.named_parameters())
    for name in allowed_missing:
        with torch.no_grad():
            parameters[name].zero_()
    return sorted(allowed_missing)
