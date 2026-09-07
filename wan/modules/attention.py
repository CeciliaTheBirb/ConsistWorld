import math
import warnings

import torch

from wan.utils.accel import is_npu

try:
    # FlashAttention-4 (CuTeDSL, Hopper/Blackwell). Use the varlen entry point
    # so it slots into the same cu_seqlens path as FA3/FA2. A broad except
    # guards against partial/broken cute installs so they degrade to FA3/FA2
    # instead of crashing training. Never available on NPU.
    from flash_attn.cute import flash_attn_varlen_func as flash_attn_varlen_func_v4
    FLASH_ATTN_4_AVAILABLE = True
except Exception:
    FLASH_ATTN_4_AVAILABLE = False

try:
    import flash_attn_interface
    FLASH_ATTN_3_AVAILABLE = True
except ImportError:
    FLASH_ATTN_3_AVAILABLE = False

try:
    import flash_attn
    FLASH_ATTN_2_AVAILABLE = True
except ImportError:
    FLASH_ATTN_2_AVAILABLE = False

if is_npu():
    import torch_npu  # noqa: F401  (registers torch_npu.npu_fusion_attention)

__all__ = [
    'flash_attention',
    'attention',
]


def _npu_varlen_attention(
    q,
    k,
    v,
    cu_seqlens_q,
    cu_seqlens_k,
    dropout_p,
    softmax_scale,
):
    """Variable-length attention on Ascend NPU via `torch_npu.npu_fusion_attention`.

    q/k/v are flat `(total_tokens, n_heads, head_dim)` tensors (TND layout).
    cu_seqlens_*: int32 tensors with leading 0, e.g. `[0, 2, 5, 9]`.
    """
    import torch_npu

    # See _npu_dense_attention: fused attention needs contiguous q/k/v. No-op when
    # already contiguous.
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    head_num = q.shape[1]
    head_dim = q.shape[-1]
    scale = softmax_scale if softmax_scale is not None else 1.0 / math.sqrt(head_dim)
    actual_seq_qlen = cu_seqlens_q[1:].to(dtype=torch.int64, device="cpu").tolist()
    actual_seq_kvlen = cu_seqlens_k[1:].to(dtype=torch.int64, device="cpu").tolist()

    out = torch_npu.npu_fusion_attention(
        q,
        k,
        v,
        head_num=head_num,
        input_layout="TND",
        scale=scale,
        keep_prob=1.0 - float(dropout_p),
        actual_seq_qlen=actual_seq_qlen,
        actual_seq_kvlen=actual_seq_kvlen,
        sparse_mode=0,
    )
    return out[0]


def _npu_dense_attention(
    q,
    k,
    v,
    dropout_p,
    softmax_scale,
):
    """Dense equal-length attention on Ascend NPU.

    The training batches in this project are padded/validated to equal sequence
    lengths.  Prefer the dense BSND path over TND varlen because the latter
    requires per-call host-side actual_seq lists and can accumulate large CANN
    host workspaces in long-running training.
    """
    import torch_npu

    # npu_fusion_attention requires contiguous inputs. Slicing q/k/v along the
    # sequence dim (e.g. block-causal history/target split) yields non-contiguous
    # tensors when batch>1, which the fused op cannot consume correctly. This is a
    # no-op when the tensor is already contiguous.
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    head_num = q.shape[2]
    head_dim = q.shape[-1]
    scale = softmax_scale if softmax_scale is not None else 1.0 / math.sqrt(head_dim)
    out = torch_npu.npu_fusion_attention(
        q,
        k,
        v,
        head_num=head_num,
        input_layout="BSND",
        scale=scale,
        keep_prob=1.0 - float(dropout_p),
        sparse_mode=0,
    )
    return out[0]


def flash_attention(
    q,
    k,
    v,
    q_lens=None,
    k_lens=None,
    dropout_p=0.,
    softmax_scale=None,
    q_scale=None,
    causal=False,
    window_size=(-1, -1),
    deterministic=False,
    dtype=torch.bfloat16,
    version=None,
):
    """
    q:              [B, Lq, Nq, C1].
    k:              [B, Lk, Nk, C1].
    v:              [B, Lk, Nk, C2]. Nq must be divisible by Nk.
    q_lens:         [B].
    k_lens:         [B].
    dropout_p:      float. Dropout probability.
    softmax_scale:  float. The scaling of QK^T before applying softmax.
    causal:         bool. Whether to apply causal attention mask.
    window_size:    (left right). If not (-1, -1), apply sliding window local attention.
    deterministic:  bool. If True, slightly slower and uses more memory.
    dtype:          torch.dtype. Apply when dtype of q/k/v is not float16/bfloat16.
    """
    half_dtypes = (torch.float16, torch.bfloat16)
    assert dtype in half_dtypes
    assert q.device.type in ('cuda', 'npu') and q.size(-1) <= 256

    b, lq, lk, out_dtype = q.size(0), q.size(1), k.size(1), q.dtype

    def half(x):
        return x if x.dtype in half_dtypes else x.to(dtype)

    # ---- Ascend NPU dense (equal-length) path ----
    # Operates on the original [B, L, N, C] (BSND) layout, so it must run BEFORE
    # the varlen flatten preprocessing below. The training/inference batches here
    # are validated to equal sequence lengths, so the dense fused op applies.
    if is_npu():
        if causal or window_size != (-1, -1):
            raise NotImplementedError(
                "NPU attention path does not yet support causal or windowed masking; "
                "current Lingbot training does not require them."
            )
        dense_equal_lengths = q_lens is None
        if dense_equal_lengths and k_lens is not None:
            try:
                dense_equal_lengths = (
                    int(k_lens.min().item()) == lk and int(k_lens.max().item()) == lk
                )
            except Exception:
                dense_equal_lengths = False
        if dense_equal_lengths:
            q_dense = half(q).to(v.dtype)
            k_dense = half(k).to(v.dtype)
            v_dense = half(v)
            x = _npu_dense_attention(
                q_dense,
                k_dense,
                v_dense,
                dropout_p=dropout_p,
                softmax_scale=softmax_scale,
            )
            return x.type(out_dtype)

    # preprocess query
    if q_lens is None:
        q = half(q.flatten(0, 1))
        q_lens = torch.tensor(
            [lq] * b, dtype=torch.int32).to(
                device=q.device, non_blocking=True)
    else:
        q = half(torch.cat([u[:v] for u, v in zip(q, q_lens)]))

    # preprocess key, value
    if k_lens is None:
        k = half(k.flatten(0, 1))
        v = half(v.flatten(0, 1))
        k_lens = torch.tensor(
            [lk] * b, dtype=torch.int32).to(
                device=k.device, non_blocking=True)
    else:
        k = half(torch.cat([u[:v] for u, v in zip(k, k_lens)]))
        v = half(torch.cat([u[:v] for u, v in zip(v, k_lens)]))

    q = q.to(v.dtype)
    k = k.to(v.dtype)

    if q_scale is not None:
        q = q * q_scale

    cu_seqlens_q = torch.cat([q_lens.new_zeros([1]), q_lens]).cumsum(
        0, dtype=torch.int32).to(q.device, non_blocking=True)
    cu_seqlens_k = torch.cat([k_lens.new_zeros([1]), k_lens]).cumsum(
        0, dtype=torch.int32).to(q.device, non_blocking=True)

    # ---- Ascend NPU varlen path (TND) ----
    if is_npu():
        x = _npu_varlen_attention(
            q, k, v,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
        ).unflatten(0, (b, lq))
        return x.type(out_dtype)

    if version is not None and version == 4 and not FLASH_ATTN_4_AVAILABLE:
        warnings.warn(
            'Flash attention 4 is not available, fall back to flash attention 3/2.'
        )

    # FlashAttention-4 (highest priority on CUDA). Handled in its own early-return
    # branch so the FA3 / FA2 code below stays untouched.
    if (version is None or version == 4) and FLASH_ATTN_4_AVAILABLE:
        # FA4 uses `None` (not -1) to denote an unbounded window side.
        fa4_window = tuple(None if s is None or s < 0 else s for s in window_size)
        x = flash_attn_varlen_func_v4(
            q,
            k,
            v,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=lq,
            max_seqlen_k=lk,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=fa4_window,
            deterministic=deterministic)
        # FA4's varlen entry point returns (out, softmax_lse); older builds
        # return the tensor directly.
        if isinstance(x, (tuple, list)):
            x = x[0]
        x = x.unflatten(0, (b, lq))
        return x.type(out_dtype)

    if version is not None and version == 3 and not FLASH_ATTN_3_AVAILABLE:
        warnings.warn(
            'Flash attention 3 is not available, use flash attention 2 instead.'
        )

    # apply attention
    if (version is None or version == 3) and FLASH_ATTN_3_AVAILABLE:
        # Note: dropout_p, window_size are not supported in FA3 now.
        x = flash_attn_interface.flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            seqused_q=None,
            seqused_k=None,
            max_seqlen_q=lq,
            max_seqlen_k=lk,
            softmax_scale=softmax_scale,
            causal=causal,
            deterministic=deterministic).unflatten(0, (b, lq))
    else:
        assert FLASH_ATTN_2_AVAILABLE
        x = flash_attn.flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=lq,
            max_seqlen_k=lk,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,
            deterministic=deterministic).unflatten(0, (b, lq))

    # output
    return x.type(out_dtype)


def attention(
    q,
    k,
    v,
    q_lens=None,
    k_lens=None,
    dropout_p=0.,
    softmax_scale=None,
    q_scale=None,
    causal=False,
    window_size=(-1, -1),
    deterministic=False,
    dtype=torch.bfloat16,
    fa_version=None,
):
    if is_npu() or FLASH_ATTN_4_AVAILABLE or FLASH_ATTN_2_AVAILABLE or FLASH_ATTN_3_AVAILABLE:
        return flash_attention(
            q=q,
            k=k,
            v=v,
            q_lens=q_lens,
            k_lens=k_lens,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            q_scale=q_scale,
            causal=causal,
            window_size=window_size,
            deterministic=deterministic,
            dtype=dtype,
            version=fa_version,
        )
    else:
        # Fallback when no fused attention backend is available: PyTorch SDPA.
        if q_lens is not None or k_lens is not None:
            warnings.warn(
                'Padding mask is disabled when using scaled_dot_product_attention. It can have a significant impact on performance.'
            )
        attn_mask = None

        # [B, L, N, C] -> [B, N, L, C]
        q = q.transpose(1, 2).to(dtype)
        k = k.transpose(1, 2).to(dtype)
        v = v.transpose(1, 2).to(dtype)

        out = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, is_causal=causal, dropout_p=dropout_p)

        out = out.transpose(1, 2).contiguous()
        return out
