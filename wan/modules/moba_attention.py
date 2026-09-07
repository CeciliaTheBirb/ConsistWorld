"""MoBA (Mixture of Bidirectional and Autoregressive) attention over the TRIPLED
sequence ``[clean history | TF-target | Bi-target]`` (paper §3.2), computed in a
SINGLE attention call for self and a single call for cross.

Segments (chunk_tok tokens per chunk):
    clean history : chunks 0..H-1        at [0, hist_len)
    TF-target     : N chunks             at [hist_len, hist_len + N*ct)
    Bi-target     : N chunks (duplicate) at [hist_len + N*ct, hist_len + 2*N*ct)

Self-attention (one call):
    * clean chunk c        -> clean[0:c] (causal)
    * TF-target chunk t (p)-> clean[<p] + its own TF chunk
    * Bi-target            -> all Bi-target only (full among the Bi copy; paper
                              Fig. 4 bottom-right block: Bi rows attend the Bi
                              columns only, NOT clean history)
    (TF and Bi never attend each other, so this equals the paper's combined mask.)

Cross-attention (one call), KV = [a_B, a_G, a_{cti[0]}..a_{cti[N-1]}]:
    * clean & TF tokens (chunk p) -> a_B + a_1..a_p  (cumulative)
    * Bi tokens                   -> a_G

Backend via ``LINGBOT_MOBA_ATTN``: "varlen" (FA4 cu_seqlens, default) | "flex"
(compiled flex_attention + mask_mod) | "loop" (per-chunk reference/fallback).
"""
from __future__ import annotations

import os

import torch

from wan.modules.attention import FLASH_ATTN_4_AVAILABLE, _npu_varlen_attention, flash_attention
from wan.utils.accel import is_npu

try:
    from flash_attn.cute import flash_attn_varlen_func as _fa4_varlen
except Exception:  # pragma: no cover
    _fa4_varlen = None

try:
    from torch.nn.attention.flex_attention import create_block_mask, flex_attention
    _FLEX_AVAILABLE = not is_npu()
except ImportError:  # pragma: no cover
    _FLEX_AVAILABLE = False

_flex_attention = None


def moba_impl() -> str:
    return os.environ.get("LINGBOT_MOBA_ATTN", "varlen").lower()


def _compiled_flex_attention():
    """Compile FlexAttention only when the optional backend is selected."""
    global _flex_attention
    if _flex_attention is None:
        _flex_attention = torch.compile(flex_attention)
    return _flex_attention


def _packed_varlen_attn(q0, k_packed, v_packed, cu_q, cu_k, max_q, max_k):
    """Packed variable-length attention over one MoBA (self or cross) call.

    q0/k_packed/v_packed are flat `(total_tokens, n_heads, head_dim)` (TND); cu_q/cu_k
    are int32 cumulative-offset tensors with a leading 0. Dispatches to
    `torch_npu.npu_fusion_attention` (TND) on NPU and FlashAttention-4 varlen on CUDA.
    Returns `(total_q, n_heads, head_dim)`.
    """
    if is_npu():
        return _npu_varlen_attention(
            q0, k_packed, v_packed,
            cu_seqlens_q=cu_q, cu_seqlens_k=cu_k,
            dropout_p=0.0, softmax_scale=None,
        )
    out = _fa4_varlen(
        q0, k_packed, v_packed,
        cu_seqlens_q=cu_q, cu_seqlens_k=cu_k,
        max_seqlen_q=max_q, max_seqlen_k=max_k, causal=False,
    )
    if isinstance(out, (tuple, list)):
        out = out[0]
    return out


# --------------------------------------------------------------------------- #
# Self-attention over the tripled sequence.
# --------------------------------------------------------------------------- #
def _self_loop(q, k, v, chunk_tok, num_hist, tgt_abs, include_bi=True):
    ct = chunk_tok
    n = len(tgt_abs)
    hist_len = num_hist * ct
    tf0 = hist_len
    bi0 = hist_len + n * ct
    outs = []
    for c in range(1, num_hist + 1):
        outs.append(flash_attention(q[:, (c - 1) * ct : c * ct], k[:, : c * ct], v[:, : c * ct]))
    for t, p in enumerate(tgt_abs):
        qs = tf0 + t * ct
        qe = qs + ct
        hke = (int(p) - 1) * ct
        kc = torch.cat([k[:, :hke], k[:, qs:qe]], dim=1)
        vc = torch.cat([v[:, :hke], v[:, qs:qe]], dim=1)
        outs.append(flash_attention(q[:, qs:qe], kc, vc))
    if include_bi and n > 0:
        # Bi rows attend the Bi columns ONLY (paper Fig. 4 bottom-right full block);
        # NO clean history.
        outs.append(flash_attention(q[:, bi0:], k[:, bi0 : bi0 + n * ct], v[:, bi0 : bi0 + n * ct]))
    return torch.cat(outs, dim=1)


def _self_varlen(q, k, v, chunk_tok, num_hist, tgt_abs, include_bi=True):
    assert q.size(0) == 1, "MoBA varlen path assumes B=1"
    ct = chunk_tok
    n = len(tgt_abs)
    hist_len = num_hist * ct
    tf0 = hist_len
    bi0 = hist_len + n * ct
    q0, k0, v0 = q[0], k[0], v[0]
    ql, kl, kp, vp = [], [], [], []
    for c in range(num_hist):
        e = (c + 1) * ct
        ql.append(ct); kp.append(k0[:e]); vp.append(v0[:e]); kl.append(e)
    for t, p in enumerate(tgt_abs):
        qs = tf0 + t * ct
        qe = qs + ct
        hke = (int(p) - 1) * ct
        ql.append(ct)
        kp.append(torch.cat([k0[:hke], k0[qs:qe]], dim=0))
        vp.append(torch.cat([v0[:hke], v0[qs:qe]], dim=0))
        kl.append(hke + ct)
    if include_bi and n > 0:
        # Bi group: attend the Bi tokens ONLY (no clean history).
        ql.append(n * ct)
        kb = k0[bi0 : bi0 + n * ct]
        vb = v0[bi0 : bi0 + n * ct]
        kp.append(kb); vp.append(vb); kl.append(kb.size(0))
    k_packed = torch.cat(kp, dim=0)
    v_packed = torch.cat(vp, dim=0)
    cu_q = torch.tensor([0, *ql], device=q.device, dtype=torch.int32).cumsum(0).to(torch.int32)
    cu_k = torch.tensor([0, *kl], device=q.device, dtype=torch.int32).cumsum(0).to(torch.int32)
    out = _packed_varlen_attn(q0, k_packed, v_packed, cu_q, cu_k, max(ql), max(kl))
    return out.unsqueeze(0).type(q.dtype)


def _self_flex(q, k, v, chunk_tok, num_hist, tgt_abs, include_bi=True):
    device = q.device
    seq = q.size(1)
    ct = chunk_tok
    n = len(tgt_abs)
    hist_len = num_hist * ct
    tf0 = hist_len
    bi0 = hist_len + n * ct
    tgt_abs0 = torch.tensor([int(p) - 1 for p in tgt_abs] or [0], device=device, dtype=torch.long)

    def mask_mod(b, h, qi, ki):
        q_clean = qi < tf0
        q_tf = (qi >= tf0) & (qi < bi0)
        q_bi = qi >= bi0
        kv_clean = ki < tf0
        kv_tf = (ki >= tf0) & (ki < bi0)
        kv_bi = ki >= bi0
        q_cc = qi // ct
        kv_cc = ki // ct
        clean_rule = q_clean & kv_clean & (kv_cc <= q_cc)
        q_tfc = ((qi - tf0) // ct).clamp(0, max(n - 1, 0))
        kv_tfc = (ki - tf0) // ct
        q_absp = tgt_abs0[q_tfc]
        tf_hist = q_tf & kv_clean & (kv_cc < q_absp)
        tf_own = q_tf & kv_tf & (kv_tfc == q_tfc)
        bi_rule = q_bi & kv_bi
        return clean_rule | tf_hist | tf_own | bi_rule

    bm = create_block_mask(
        mask_mod, B=None, H=None, Q_LEN=seq, KV_LEN=seq, device=device, _compile=True
    )
    out = _compiled_flex_attention()(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), block_mask=bm)
    return out.transpose(1, 2)


def moba_self_attention(q, k, v, chunk_tok, num_hist, tgt_abs, include_bi=True):
    impl = moba_impl()
    if impl == "flex" and _FLEX_AVAILABLE:
        return _self_flex(q, k, v, chunk_tok, num_hist, tgt_abs, include_bi)
    if impl == "varlen" and (is_npu() or (FLASH_ATTN_4_AVAILABLE and _fa4_varlen is not None)):
        return _self_varlen(q, k, v, chunk_tok, num_hist, tgt_abs, include_bi)
    return _self_loop(q, k, v, chunk_tok, num_hist, tgt_abs, include_bi)


# --------------------------------------------------------------------------- #
# Generalized self-attention driven by explicit query groups. Each group is
# ``(q_start, q_end, kv_ranges)`` and groups tile the sequence in order.
# --------------------------------------------------------------------------- #
def _gather_ranges(t0, ranges):
    return torch.cat([t0[s:e] for (s, e) in ranges], dim=0)


def _self_loop_groups(q, k, v, groups):
    outs = []
    for (qs, qe, kv_ranges) in groups:
        kc = torch.cat([k[:, s:e] for (s, e) in kv_ranges], dim=1)
        vc = torch.cat([v[:, s:e] for (s, e) in kv_ranges], dim=1)
        outs.append(flash_attention(q[:, qs:qe], kc, vc))
    return torch.cat(outs, dim=1)


def _self_varlen_groups(q, k, v, groups):
    assert q.size(0) == 1, "MoBA varlen path assumes B=1"
    q0, k0, v0 = q[0], k[0], v[0]
    ql, kl, kp, vp = [], [], [], []
    off = 0
    for (qs, qe, kv_ranges) in groups:
        assert qs == off, f"self-attn groups must tile [0,L) in order (gap at {qs}!={off})"
        ql.append(qe - qs)
        kk = _gather_ranges(k0, kv_ranges)
        vv = _gather_ranges(v0, kv_ranges)
        kp.append(kk); vp.append(vv); kl.append(kk.size(0))
        off = qe
    k_packed = torch.cat(kp, dim=0)
    v_packed = torch.cat(vp, dim=0)
    cu_q = torch.tensor([0, *ql], device=q.device, dtype=torch.int32).cumsum(0).to(torch.int32)
    cu_k = torch.tensor([0, *kl], device=q.device, dtype=torch.int32).cumsum(0).to(torch.int32)
    out = _packed_varlen_attn(q0, k_packed, v_packed, cu_q, cu_k, max(ql), max(kl))
    return out.unsqueeze(0).type(q.dtype)


def moba_self_attention_groups(q, k, v, groups):
    """Generalized self-attention over explicit query groups."""
    impl = moba_impl()
    if impl == "varlen" and (is_npu() or (FLASH_ATTN_4_AVAILABLE and _fa4_varlen is not None)):
        return _self_varlen_groups(q, k, v, groups)
    return _self_loop_groups(q, k, v, groups)


def gated_cross_now(q, k, v, x_full, groups_without_peer, gate):
    """Blend full attention with attention that excludes peer-current keys.

    The geometry gate belongs only to the equal-time cross-view cell.  Re-running
    the affected query groups without that cell keeps the two endpoints exact:
    ``gate=1`` is the original attention result and ``gate=0`` is the layout
    without peer-current attention.
    """
    if gate is None:
        return x_full
    gate = gate.to(device=x_full.device, dtype=x_full.dtype)
    if gate.ndim != 1 or gate.numel() != x_full.shape[1]:
        raise ValueError(
            "cross-view gate must provide one value per sequence token: "
            f"got {tuple(gate.shape)} for sequence length {x_full.shape[1]}"
        )

    out = x_full
    for qs, qe, key_ranges in groups_without_peer:
        weights = gate[qs:qe]
        if bool((weights >= 1.0).all()):
            continue
        keys = torch.cat([k[:, start:end] for start, end in key_ranges], dim=1)
        values = torch.cat([v[:, start:end] for start, end in key_ranges], dim=1)
        without_peer = flash_attention(q[:, qs:qe], keys, values)
        blend = weights[None, :, None, None]
        out = torch.cat(
            [
                out[:, :qs],
                without_peer + blend * (out[:, qs:qe] - without_peer),
                out[:, qe:],
            ],
            dim=1,
        )
    return out



# --------------------------------------------------------------------------- #
# Cross-attention over the tripled sequence.
# KV blocks (each text_len tokens): 0=a_B, 1=a_G, 2..N+1 = per-chunk prompts.
# Routing per query token: is_bi -> a_G; else -> a_B + a_1..a_p (cumulative).
# --------------------------------------------------------------------------- #
def _cross_flex(q, k, v, chunk_pos, is_bi, text_len):
    device = q.device
    Lq = q.size(1)
    Kp = k.size(1)

    def mask_mod(b, h, qi, ki):
        p = chunk_pos[qi]
        block = ki // text_len
        cumulative = (block == 0) | ((block >= 2) & ((block - 2) <= p))
        bi = is_bi[qi] > 0
        return torch.where(bi, block == 1, cumulative)

    bm = create_block_mask(
        mask_mod, B=None, H=None, Q_LEN=Lq, KV_LEN=Kp, device=device, _compile=True
    )
    out = _compiled_flex_attention()(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), block_mask=bm)
    return out.transpose(1, 2)


def _cross_varlen(q, k, v, moba_groups, text_len):
    q0, k0, v0 = q[0], k[0], v[0]

    def blk(i):
        return slice(i * text_len, (i + 1) * text_len)

    ql, kl, kp, vp = [], [], [], []
    for (start, end, p, is_bi) in moba_groups:
        ql.append(end - start)
        if is_bi:
            kb = [k0[blk(1)]]  # a_G
            vb = [v0[blk(1)]]
        else:
            kb = [k0[blk(0)]]  # a_B
            vb = [v0[blk(0)]]
            for c in range(p + 1):
                kb.append(k0[blk(2 + c)])
                vb.append(v0[blk(2 + c)])
        kk = torch.cat(kb, dim=0)
        vv = torch.cat(vb, dim=0)
        kp.append(kk); vp.append(vv); kl.append(kk.size(0))
    k_packed = torch.cat(kp, dim=0)
    v_packed = torch.cat(vp, dim=0)
    cu_q = torch.tensor([0, *ql], device=q.device, dtype=torch.int32).cumsum(0).to(torch.int32)
    cu_k = torch.tensor([0, *kl], device=q.device, dtype=torch.int32).cumsum(0).to(torch.int32)
    out = _packed_varlen_attn(q0, k_packed, v_packed, cu_q, cu_k, max(ql), max(kl))
    return out.unsqueeze(0)


def moba_cross_attention(cross, normed, a_b_ctx, a_g_ctx, prompts, moba_cti, moba_groups, chunk_pos, is_bi):
    """Single-call MoBA cross-attention; returns [B, L, dim] or None (use loop)."""
    impl = moba_impl()
    if impl not in ("varlen", "flex"):
        return None
    B = 1
    H = cross.num_heads
    hd = cross.head_dim
    text_len = a_b_ctx.size(1)
    dim = a_b_ctx.size(2)
    a_g = a_g_ctx if a_g_ctx is not None else a_b_ctx
    kv = torch.cat([a_b_ctx, a_g, prompts[moba_cti]], dim=0)  # [N+2, text_len, dim]
    kv_flat = kv.reshape(1, kv.size(0) * text_len, dim)
    q = cross.norm_q(cross.q(normed)).view(B, -1, H, hd)
    k = cross.norm_k(cross.k(kv_flat)).view(B, -1, H, hd)
    v = cross.v(kv_flat).view(B, -1, H, hd)
    if impl == "flex" and _FLEX_AVAILABLE:
        out = _cross_flex(q, k, v, chunk_pos, is_bi, text_len)
    else:
        out = _cross_varlen(q, k, v, moba_groups, text_len)
    return cross.o(out.flatten(2))
