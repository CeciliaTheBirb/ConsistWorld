"""Batch transport for sequence-parallel training.

Every sequence-parallel rank needs the same sample before the model shards its
token sequence.  The cache loader yields one CPU sample at a time; this module
moves it to the accelerator and exchanges a copy across the SP group.
"""
from __future__ import annotations

import torch

from wan.commons.communications import all_to_all
from wan.commons.parallel_states import get_parallel_state


def _exchange(tensor: torch.Tensor, group, sp_size: int) -> list[torch.Tensor]:
    copies = tensor.repeat(sp_size, *([1] * (tensor.ndim - 1)))
    return all_to_all(copies, group=group, scatter_dim=0, gather_dim=0)


def prepare_sequence_parallel_data(batch, sp_size: int):
    """Return one full batch per sequence-parallel rank."""
    latents, cond_y, text_emb, control, chunk_text, prompt_b, prompt_g, extra = batch
    parallel = get_parallel_state()
    if not parallel.sp_enabled:
        return [(latents, cond_y, text_emb, control, chunk_text, prompt_b, prompt_g, extra)]

    group = parallel.sp_group
    base = [
        _exchange(tensor, group, sp_size)
        for tensor in (latents, cond_y, text_emb, control, chunk_text, prompt_b, prompt_g)
    ]
    extras = {name: _exchange(value, group, sp_size) for name, value in extra.items()}
    return [
        (*[values[index] for values in base], {name: values[index] for name, values in extras.items()})
        for index in range(sp_size)
    ]


def sequence_parallel_batches(dataloader, device, sp_size: int):
    """Yield accelerator batches forever from a cache-backed iterable dataset."""
    while True:
        for cpu_batch in dataloader:
            moved = tuple(
                value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
                for value in cpu_batch[:-1]
            )
            extra = {
                name: value.to(device, non_blocking=True)
                for name, value in cpu_batch[-1].items()
            }
            for batch in prepare_sequence_parallel_data((*moved, extra), sp_size):
                yield batch
