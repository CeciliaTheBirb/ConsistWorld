"""Self-resampling helpers for Stage-1 AR training (MaineCoon / Resampling-Forcing).

Pure, trainer-agnostic utilities: rho curriculum, clean-biased history sigma
sampling, on-schedule timestep snapping, and the AR sequence-length helper. The
autoregressive rollout that produces model-degraded history lives as a trainer
method (`_self_resample_history_kv`) because it needs the model-call wiring.
"""

from __future__ import annotations

import math


def _ceil_div(a: int, b: int) -> int:
    return (int(a) + int(b) - 1) // int(b)


def rho_curriculum(step: int, config) -> float:
    """Self-resampling ratio rho at a given global step (MaineCoon eq 11).

    Anneals from ``sr_rho_start`` to ``sr_rho_end`` over ``sr_rho_anneal_steps``
    using ``sr_rho_schedule`` ("linear" or "cosine"). Constant afterwards.
    """
    start = float(config.sr_rho_start)
    end = float(config.sr_rho_end)
    total = int(config.sr_rho_anneal_steps)
    if total <= 0:
        return end
    frac = min(max(int(step) / total, 0.0), 1.0)
    schedule = getattr(config, "sr_rho_schedule", "linear")
    if schedule == "cosine":
        frac = 0.5 * (1.0 - math.cos(math.pi * frac))
    elif schedule != "linear":
        raise ValueError(f"Unsupported sr_rho_schedule={schedule!r}")
    return start + (end - start) * frac


def resolve_history_timestep(noise_scheduler, sigma_target: float, device):
    """Pick the schedule timestep whose flow sigma is closest to ``sigma_target``.

    Returns ``(timestep_scalar_tensor, sigma_value)``: the scalar tensor is a
    valid schedule timestep the AR forward accepts, and ``sigma_value`` is its
    exact flow sigma for noising / the single Euler step. Snapping to an
    on-schedule timestep matches how ``get_sigmas`` looks up sigmas during
    training, so the resampling step stays numerically consistent.
    """
    import torch

    sigmas = noise_scheduler.sigmas.to(device=device, dtype=torch.float32)
    schedule_timesteps = noise_scheduler.timesteps.to(device)
    n = schedule_timesteps.numel()
    sig = sigmas[:n]
    idx = int(torch.argmin((sig - float(sigma_target)).abs()).item())
    timestep = schedule_timesteps[idx].reshape(1)
    sigma_value = float(sig[idx].item())
    return timestep, sigma_value


def sample_history_sigma(shift: float, device):
    """Sample a clean-biased history noise level sigma_h (Paper 1 eq 6-7).

    logit(sigma_h) ~ N(0, 1); then apply the standard timestep shift with
    ``s = shift`` (< 1 puts more weight on the low-noise / clean region):
        sigma_h <- s * sigma_h / (1 + (s - 1) * sigma_h).
    Returns a scalar float tensor on ``device`` (caller SP-syncs it).
    """
    import torch

    u = torch.randn(1, device=device)
    sigma = torch.sigmoid(u)
    s = float(shift)
    sigma = s * sigma / (1.0 + (s - 1.0) * sigma)
    return sigma


def ar_seq_len(n_frames: int, lat_h: int, lat_w: int, patch_size) -> int:
    """Token count for ``n_frames`` uniform (1x) latent frames at ``patch_size``.

    Mirrors ``Stage1ARGeometry.compute_seq_len`` but for an arbitrary frame count.
    """
    p_t, p_h, p_w = patch_size
    return int(_ceil_div(n_frames, p_t) * _ceil_div(lat_h, p_h) * _ceil_div(lat_w, p_w))
