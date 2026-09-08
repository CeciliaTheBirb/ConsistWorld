import math

import torch
import torch.nn as nn
import torch.nn.functional as torch_F
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin
from einops import rearrange

from wan.commons.communications import all_gather, all_to_all_4D
from wan.commons.parallel_states import get_parallel_state
from wan.modules.attention import flash_attention
from wan.modules.moba_attention import (
    gated_cross_now,
    moba_cross_attention,
    moba_self_attention,
    moba_self_attention_groups,
)
from wan.utils.accel import device_type

__all__ = ["WanModelAR", "WanModel"]


def sinusoidal_embedding_1d(dim, position):
    assert dim % 2 == 0
    half = dim // 2
    position = position.type(torch.float64)
    sinusoid = torch.outer(
        position,
        torch.pow(10000, -torch.arange(half).to(position).div(half)),
    )
    return torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)


@torch.amp.autocast(device_type, enabled=False)
def rope_params(max_seq_len, dim, theta=10000):
    assert dim % 2 == 0
    # NPU's aclnnCat does not support complex128; downcast to float32 -> complex64.
    rope_dtype = torch.float64 if device_type == "cuda" else torch.float32
    freqs = torch.outer(
        torch.arange(max_seq_len),
        1.0 / torch.pow(theta, torch.arange(0, dim, 2).to(rope_dtype).div(dim)),
    )
    return torch.polar(torch.ones_like(freqs), freqs)


@torch.amp.autocast(device_type, enabled=False)
def rope_apply(x, grid_sizes, freqs):
    num_heads = x.size(2)
    half_dim = x.size(3) // 2
    freq_splits = freqs.split(
        [
            half_dim - 2 * (half_dim // 3),
            half_dim // 3,
            half_dim // 3,
        ],
        dim=1,
    )

    outputs = []
    rope_dtype = torch.float64 if device_type == "cuda" else torch.float32
    for batch_idx, (frames, height, width) in enumerate(grid_sizes.tolist()):
        seq_len = frames * height * width
        x_item = torch.view_as_complex(
            x[batch_idx, :seq_len].to(rope_dtype).reshape(seq_len, num_heads, -1, 2)
        )
        freqs_item = torch.cat(
            [
                freq_splits[0][:frames].view(frames, 1, 1, -1).expand(frames, height, width, -1).contiguous(),
                freq_splits[1][:height].view(1, height, 1, -1).expand(frames, height, width, -1).contiguous(),
                freq_splits[2][:width].view(1, 1, width, -1).expand(frames, height, width, -1).contiguous(),
            ],
            dim=-1,
        ).reshape(seq_len, 1, -1)
        x_item = torch.view_as_real(x_item * freqs_item).flatten(2)
        x_item = torch.cat([x_item, x[batch_idx, seq_len:]])
        outputs.append(x_item)
    return torch.stack(outputs).float()


def _pad_freqs(freqs, target_len):
    seq_len, s1, s2 = freqs.shape
    if seq_len >= target_len:
        return freqs
    pad = torch.ones(
        target_len - seq_len,
        s1,
        s2,
        dtype=freqs.dtype,
        device=freqs.device,
    )
    return torch.cat([freqs, pad], dim=0)


def pad_for_3d_conv(x, kernel_size):
    _, _, frames, height, width = x.shape
    patch_t, patch_h, patch_w = kernel_size
    pad_t = (patch_t - (frames % patch_t)) % patch_t
    pad_h = (patch_h - (height % patch_h)) % patch_h
    pad_w = (patch_w - (width % patch_w)) % patch_w
    if device_type == "cuda":
        return torch.nn.functional.pad(x, (0, pad_w, 0, pad_h, 0, pad_t), mode="replicate")
    # NPU: aclnnReplicationPad3d (error 161002) is unreliable for 5-D bf16. Replicate
    # the trailing edges manually via slice + cat (numerically identical to mode="replicate").
    if pad_t > 0:
        x = torch.cat([x, x[:, :, -1:, :, :].expand(-1, -1, pad_t, -1, -1)], dim=2)
    if pad_h > 0:
        x = torch.cat([x, x[:, :, :, -1:, :].expand(-1, -1, -1, pad_h, -1)], dim=3)
    if pad_w > 0:
        x = torch.cat([x, x[:, :, :, :, -1:].expand(-1, -1, -1, -1, pad_w)], dim=4)
    return x.contiguous()


def _avg_pool3d_complex(x, kernel_size):
    x_real = torch.nn.functional.avg_pool3d(x.real, kernel_size, stride=kernel_size)
    x_imag = torch.nn.functional.avg_pool3d(x.imag, kernel_size, stride=kernel_size)
    return torch.complex(x_real, x_imag)


@torch.amp.autocast(device_type, enabled=False)
def rope_apply_token_freqs(x, token_freqs):
    batch_size, seq_len, num_heads, head_dim = x.shape
    if token_freqs.ndim != 4:
        raise ValueError(
            f"token_freqs must have shape [B, L, 1, D/2], got {tuple(token_freqs.shape)}"
        )
    expected_shape = (batch_size, seq_len, 1, head_dim // 2)
    if tuple(token_freqs.shape) != expected_shape:
        raise ValueError(
            "RoPE token frequency shape mismatch. "
            f"Expected {expected_shape}, got {tuple(token_freqs.shape)}"
        )

    rope_dtype = torch.float64 if device_type == "cuda" else torch.float32
    x_complex = torch.view_as_complex(
        x.to(rope_dtype).reshape(batch_size, seq_len, num_heads, -1, 2)
    )
    freqs = token_freqs.to(device=x.device, dtype=x_complex.dtype)
    x_complex = x_complex * freqs
    return torch.view_as_real(x_complex).flatten(3).float()


def _local_mem_key_marker(mem_key_marker, local_num_heads, sp_size=1, sp_rank=0):
    """Return the marker block that matches this sequence-parallel head shard."""
    if mem_key_marker is None:
        return None
    if mem_key_marker.ndim != 2:
        raise ValueError(
            "mem_key_marker must be [num_heads, head_dim], got "
            f"{tuple(mem_key_marker.shape)}"
        )
    local_num_heads = int(local_num_heads)
    sp_size = int(sp_size)
    sp_rank = int(sp_rank)
    if mem_key_marker.shape[0] == local_num_heads:
        return mem_key_marker
    if (
        sp_size > 1
        and mem_key_marker.shape[0] == local_num_heads * sp_size
        and 0 <= sp_rank < sp_size
    ):
        start = sp_rank * local_num_heads
        return mem_key_marker[start : start + local_num_heads]
    raise ValueError(
        "mem_key_marker head count does not match attention heads: "
        f"marker={mem_key_marker.shape[0]}, local={local_num_heads}, sp_size={sp_size}, "
        f"sp_rank={sp_rank}"
    )


@torch.amp.autocast(device_type, enabled=False)
def rope_apply_sequence_parallel(x, grid_sizes, freqs, sp_size, sp_rank):
    local_seq_len = x.size(1)
    num_heads = x.size(2)
    half_dim = x.size(3) // 2
    parallel_dims = get_parallel_state()
    split_sizes = getattr(parallel_dims, "sp_split_sizes", None)
    if split_sizes is not None and len(split_sizes) == sp_size:
        expected_local_seq_len = int(split_sizes[sp_rank])
        if local_seq_len != expected_local_seq_len:
            raise ValueError(
                f"SP RoPE shard length mismatch on rank {sp_rank}: "
                f"expected {expected_local_seq_len}, got {local_seq_len}"
            )
        shard_start = sum(int(size) for size in split_sizes[:sp_rank])
        total_padded_seq_len = sum(int(size) for size in split_sizes)
    else:
        shard_start = sp_rank * local_seq_len
        total_padded_seq_len = local_seq_len * sp_size

    freq_splits = freqs.split(
        [
            half_dim - 2 * (half_dim // 3),
            half_dim // 3,
            half_dim // 3,
        ],
        dim=1,
    )

    outputs = []
    rope_dtype = torch.float64 if device_type == "cuda" else torch.float32
    for batch_idx, (frames, height, width) in enumerate(grid_sizes.tolist()):
        x_item = torch.view_as_complex(
            x[batch_idx, :local_seq_len].to(rope_dtype).reshape(local_seq_len, num_heads, -1, 2)
        )
        freqs_item = torch.cat(
            [
                freq_splits[0][:frames].view(frames, 1, 1, -1).expand(frames, height, width, -1),
                freq_splits[1][:height].view(1, height, 1, -1).expand(frames, height, width, -1),
                freq_splits[2][:width].view(1, 1, width, -1).expand(frames, height, width, -1),
            ],
            dim=-1,
        ).reshape(frames * height * width, 1, -1)
        freqs_item = _pad_freqs(freqs_item, total_padded_seq_len)
        freqs_item = freqs_item[shard_start : shard_start + local_seq_len]
        x_item = torch.view_as_real(x_item * freqs_item).flatten(2)
        x_item = torch.cat([x_item, x[batch_idx, local_seq_len:]])
        outputs.append(x_item)
    return torch.stack(outputs, dim=0).float()


class WanRMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return self._norm(x.float()).type_as(x) * self.weight

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)


class WanLayerNorm(nn.LayerNorm):
    def __init__(self, dim, eps=1e-6, elementwise_affine=False):
        super().__init__(dim, elementwise_affine=elementwise_affine, eps=eps)

    def forward(self, x):
        return super().forward(x.float()).type_as(x)


class WanSelfAttention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        window_size=(-1, -1),
        qk_norm=True,
        eps=1e-6,
        local_attn_size=-1,
        sink_size=0,
    ):
        assert dim % num_heads == 0
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.eps = eps
        self.local_attn_size = local_attn_size
        self.sink_size = sink_size

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

    def forward(
        self,
        x,
        seq_lens,
        grid_sizes,
        freqs,
        causal_layout=None,
        mem_key_marker=None,
    ):
        batch_size, seq_len, num_heads, head_dim = *x.shape[:2], self.num_heads, self.head_dim

        def qkv_fn(hidden_states):
            q = self.norm_q(self.q(hidden_states)).view(batch_size, seq_len, num_heads, head_dim)
            k = self.norm_k(self.k(hidden_states)).view(batch_size, seq_len, num_heads, head_dim)
            v = self.v(hidden_states).view(batch_size, seq_len, num_heads, head_dim)
            return q, k, v

        parallel_dims = get_parallel_state()
        if parallel_dims.sp_enabled:
            q, k, v = qkv_fn(x)
            sp_group = parallel_dims.sp_group
            if freqs.ndim == 4:
                q = rope_apply_token_freqs(q, freqs).type_as(v)
                k = rope_apply_token_freqs(k, freqs).type_as(v)
            else:
                q = rope_apply_sequence_parallel(
                    q, grid_sizes, freqs, parallel_dims.sp, parallel_dims.sp_rank
                ).type_as(v)
                k = rope_apply_sequence_parallel(
                    k, grid_sizes, freqs, parallel_dims.sp, parallel_dims.sp_rank
                ).type_as(v)
            q = all_to_all_4D(q, sp_group, scatter_dim=2, gather_dim=1)
            k = all_to_all_4D(k, sp_group, scatter_dim=2, gather_dim=1)
            v = all_to_all_4D(v, sp_group, scatter_dim=2, gather_dim=1)
        else:
            q, k, v = qkv_fn(x)
            if freqs.ndim == 4:
                q = rope_apply_token_freqs(q, freqs)
                k = rope_apply_token_freqs(k, freqs)
            else:
                q = rope_apply(q, grid_sizes, freqs)
                k = rope_apply(k, grid_sizes, freqs)
            sp_group = None

        if causal_layout is None:
            raise ValueError("WanSelfAttention requires a chunk-causal layout (uniform AR only)")
        mv_groups = causal_layout.get("mv_groups")
        if mv_groups is not None:
            local_marker = _local_mem_key_marker(
                mem_key_marker,
                k.shape[2],
                parallel_dims.sp if parallel_dims.sp_enabled else 1,
                parallel_dims.sp_rank if parallel_dims.sp_enabled else 0,
            )
            x = moba_self_attention_groups(q, k, v, mv_groups, local_marker)
            if causal_layout.get("xnow_gate") is not None:
                x = gated_cross_now(
                    q,
                    k,
                    v,
                    x,
                    causal_layout["mv_groups_without_peer"],
                    causal_layout["xnow_gate"],
                    local_marker,
                )
        else:
            x = moba_self_attention(
                q,
                k,
                v,
                causal_layout["chunk_tok"],
                causal_layout["num_hist_chunks"],
                causal_layout["tgt_abs_chunks"],
                causal_layout.get("include_bi", True),
            )

        if sp_group is not None:
            x = all_to_all_4D(x, sp_group, scatter_dim=1, gather_dim=2)

        x = x.flatten(2)
        x = self.o(x)
        return x


class WanCrossAttention(WanSelfAttention):
    def forward(self, x, context, context_lens):
        batch_size, num_heads, head_dim = x.size(0), self.num_heads, self.head_dim
        q = self.norm_q(self.q(x)).view(batch_size, -1, num_heads, head_dim)
        k = self.norm_k(self.k(context)).view(batch_size, -1, num_heads, head_dim)
        v = self.v(context).view(batch_size, -1, num_heads, head_dim)
        x = flash_attention(q, k, v, k_lens=context_lens)
        x = x.flatten(2)
        x = self.o(x)
        return x


class WanAttentionBlock(nn.Module):
    def __init__(
        self,
        dim,
        ffn_dim,
        num_heads,
        window_size=(-1, -1),
        qk_norm=True,
        cross_attn_norm=False,
        eps=1e-6,
        local_attn_size=-1,
        sink_size=0,
    ):
        super().__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        self.norm1 = WanLayerNorm(dim, eps)
        self.self_attn = WanSelfAttention(
            dim,
            num_heads,
            window_size,
            qk_norm,
            eps,
            local_attn_size=local_attn_size,
            sink_size=sink_size,
        )
        self.norm3 = (
            WanLayerNorm(dim, eps, elementwise_affine=True) if cross_attn_norm else nn.Identity()
        )
        self.cross_attn = WanCrossAttention(dim, num_heads, (-1, -1), qk_norm, eps)
        self.norm2 = WanLayerNorm(dim, eps)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(ffn_dim, dim),
        )

        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

        # The final ConsistWorld recipe gives retrieved P-Mem keys a small, zero-init
        # per-head marker. It distinguishes retrieval keys from phase-zero
        # rolling-history keys while preserving the Stage-1 warm start exactly.
        self.mem_key_marker = nn.Parameter(torch.zeros(num_heads, dim // num_heads))

        self.cam_injector_layer1 = nn.Linear(dim, dim)
        self.cam_injector_layer2 = nn.Linear(dim, dim)
        self.cam_scale_layer = nn.Linear(dim, dim)
        self.cam_shift_layer = nn.Linear(dim, dim)

    def _moba_cross_loop(
        self, normed, encoder_hidden_states, moba_groups, moba_cti,
        a_b_ctx, a_g_ctx,
    ):
        # Per-group MoBA cross-attention (fallback when LINGBOT_MOBA_ATTN="loop").
        # groups over the tripled [clean | TF | Bi] sequence: is_bi=1 -> a_G,
        # else cumulative a_B + a_1..a_p.
        text_len = a_b_ctx.size(1)
        dim = a_b_ctx.size(2)
        outs = []
        for (start, end, p, is_bi) in moba_groups:
            q_slice = normed[:, start:end]
            if is_bi and a_g_ctx is not None:
                outs.append(self.cross_attn(q_slice, a_g_ctx, None))
            else:
                prompt_ids = moba_cti[: p + 1]
                kv = torch.cat([a_b_ctx, encoder_hidden_states[prompt_ids]], dim=0)
                kv = kv.reshape(1, (p + 2) * text_len, dim)
                outs.append(self.cross_attn(q_slice, kv, None))
        return torch.cat(outs, dim=1)

    def forward(
        self,
        x,
        e,
        seq_lens,
        grid_sizes,
        freqs,
        context,
        context_lens,
        dit_cond_dict=None,
        causal_layout=None,
        a_b_ctx=None,
        a_g_ctx=None,
        moba_groups=None,
        moba_cti=None,
        moba_chunk_pos=None,
        moba_is_bi=None,
    ):
        if e.dtype != torch.float32:
            e = e.float()
        with torch.amp.autocast(device_type, dtype=torch.float32):
            e = (self.modulation.unsqueeze(0) + e).chunk(6, dim=2)

        y = self.self_attn(
            self.norm1(x).float() * (1 + e[1].squeeze(2)) + e[0].squeeze(2),
            seq_lens,
            grid_sizes,
            freqs,
            causal_layout=causal_layout,
            mem_key_marker=self.mem_key_marker,
        )
        with torch.amp.autocast(device_type, dtype=torch.float32):
            x = x + y * e[2].squeeze(2)

        if dit_cond_dict is not None and "c2ws_plucker_emb" in dit_cond_dict:
            c2ws_plucker_emb = dit_cond_dict["c2ws_plucker_emb"]
            if c2ws_plucker_emb.shape[:2] != x.shape[:2]:
                raise ValueError(
                    "control token length must strictly match hidden states. "
                    f"control={tuple(c2ws_plucker_emb.shape)}, hidden={tuple(x.shape)}"
                )
            # Camera injection runs in ambient dtype (bf16) — matches both the
            # bidirectional base (Lingbot_train model.py) and the released fast model
            # (model_fast.py). Only the residual add above is an fp32 island.
            c2ws_hidden_states = self.cam_injector_layer2(
                torch_F.silu(self.cam_injector_layer1(c2ws_plucker_emb))
            )
            c2ws_hidden_states = c2ws_hidden_states + c2ws_plucker_emb
            cam_scale = self.cam_scale_layer(c2ws_hidden_states)
            cam_shift = self.cam_shift_layer(c2ws_hidden_states)
            x = (1.0 + cam_scale) * x + cam_shift

        def cross_attn_ffn(hidden_states, encoder_hidden_states, modulation):
            # MoBA cross-attention (paper §3.2) over the tripled sequence. Per
            # (SP-local, contiguous) chunk group:
            #   * clean history + TF-target tokens (is_bi=0): a token in absolute
            #     chunk position p attends to a_B + a_1..a_p (background + cumulative
            #     chunk prompts, lower-triangular).
            #   * Bi-target tokens (is_bi=1): attend the single global a_G.
            # Cross-attn MUST be one joint attention over the concatenated KV
            # (a masked sum of separate calls is wrong: softmax normalises per-call),
            # so we run one flash call per group. Inference / non-MoBA falls back to
            # single-prompt cross-attn below.
            normed = self.norm3(hidden_states)
            if (
                moba_groups is not None
                and a_b_ctx is not None
                and moba_cti is not None
            ):
                single = moba_cross_attention(
                    self.cross_attn,
                    normed,
                    a_b_ctx,
                    a_g_ctx,
                    encoder_hidden_states,
                    moba_cti,
                    moba_groups,
                    moba_chunk_pos,
                    moba_is_bi,
                )
                if single is not None:
                    attn_out = single
                else:
                    attn_out = self._moba_cross_loop(
                        normed, encoder_hidden_states, moba_groups, moba_cti,
                        a_b_ctx, a_g_ctx,
                    )
            else:
                # Inference / non-MoBA uses one shared prompt.
                attn_out = self.cross_attn(
                    normed,
                    encoder_hidden_states,
                    None,
                )
            hidden_states = hidden_states + attn_out
            y = self.ffn(
                self.norm2(hidden_states).float() * (1 + modulation[4].squeeze(2))
                + modulation[3].squeeze(2)
            )
            with torch.amp.autocast(device_type, dtype=torch.float32):
                hidden_states = hidden_states + y * modulation[5].squeeze(2)
            return hidden_states

        return cross_attn_ffn(x, context, e)


class Head(nn.Module):
    def __init__(self, dim, out_dim, patch_size, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim
        self.patch_size = patch_size
        self.eps = eps

        out_dim = math.prod(patch_size) * out_dim
        self.norm = WanLayerNorm(dim, eps)
        self.head = nn.Linear(dim, out_dim)
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def forward(self, x, e):
        if e.dtype != torch.float32:
            e = e.float()
        with torch.amp.autocast(device_type, dtype=torch.float32):
            e = (self.modulation.unsqueeze(0) + e.unsqueeze(2)).chunk(2, dim=2)
            x = self.head(self.norm(x) * (1 + e[1].squeeze(2)) + e[0].squeeze(2))
        return x


class WanModelAR(ModelMixin, ConfigMixin):
    ignore_for_config = ["patch_size", "cross_attn_norm", "qk_norm", "text_dim", "window_size"]
    _no_split_modules = ["WanAttentionBlock"]

    @register_to_config
    def __init__(
        self,
        model_type="t2v",
        control_type="cam",
        patch_size=(1, 2, 2),
        text_len=512,
        in_dim=16,
        dim=2048,
        ffn_dim=8192,
        freq_dim=256,
        text_dim=4096,
        out_dim=16,
        num_heads=16,
        num_layers=32,
        window_size=(-1, -1),
        qk_norm=True,
        cross_attn_norm=True,
        eps=1e-6,
        zero_history_timestep=True,
        local_attn_size=-1,
        sink_size=0,
        max_views=8,
    ):
        super().__init__()

        assert model_type in ["t2v", "i2v", "ti2v", "s2v"]
        self.model_type = model_type

        # i2v checkpoints concatenate latent + i2v mask + cond_y into 36 input
        # channels. The model instantiates with that shape so `from_pretrained`
        # can load the base weights without mismatch; Stage-1 AR keeps the full
        # 36-channel layout end-to-end (the AR-window cond_y stream is fed via
        # the `y=` kwarg at forward time).
        if model_type == "i2v" and in_dim == 16:
            in_dim = 36

        self.patch_size = patch_size
        self.text_len = text_len
        self.in_dim = in_dim
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.freq_dim = freq_dim
        self.text_dim = text_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps
        self.zero_history_timestep = zero_history_timestep
        self.local_attn_size = local_attn_size
        self.sink_size = sink_size
        self.max_views = max_views

        if control_type == "cam":
            control_dim = 6
        elif control_type == "act":
            control_dim = 7
        else:
            raise ValueError(f"Unsupported control_type: {control_type}")

        self.patch_embedding = nn.Conv3d(in_dim, dim, kernel_size=patch_size, stride=patch_size)
        self.patch_embedding_wancamctrl = nn.Linear(
            control_dim * 64 * patch_size[0] * patch_size[1] * patch_size[2],
            dim,
        )
        self.control_dim = control_dim
        self.control_channels = control_dim * 64
        if tuple(patch_size) != (1, 2, 2):
            raise ValueError(
                "Stage-1 AR uniform layout requires Wan patch_size=(1, 2, 2), "
                f"got {patch_size}"
            )
        self.c2ws_hidden_states_layer1 = nn.Linear(dim, dim)
        self.c2ws_hidden_states_layer2 = nn.Linear(dim, dim)
        # Kept only to load the historical checkpoint ABI. The release does not
        # apply learned per-view identities.
        self.view_embedding = nn.Embedding(max_views, dim)
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(dim, dim),
        )
        self.time_embedding = nn.Sequential(nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 6))
        self.blocks = nn.ModuleList(
            [
                WanAttentionBlock(
                    dim,
                    ffn_dim,
                    num_heads,
                    window_size,
                    qk_norm,
                    cross_attn_norm,
                    eps,
                    local_attn_size=local_attn_size,
                    sink_size=sink_size,
                )
                for _ in range(num_layers)
            ]
        )
        self.head = Head(dim, out_dim, patch_size, eps)

        assert (dim % num_heads) == 0 and (dim // num_heads) % 2 == 0
        head_dim = dim // num_heads
        self.freqs = torch.cat(
            [
                rope_params(1024, head_dim - 4 * (head_dim // 6)),
                rope_params(1024, 2 * (head_dim // 6)),
                rope_params(1024, 2 * (head_dim // 6)),
            ],
            dim=1,
        )

        self.init_weights()

    def _normalize_indices(self, indices, batch_size, expected_frames, device, name):
        if indices is None:
            indices = torch.arange(expected_frames, device=device).unsqueeze(0).expand(batch_size, -1)
        elif indices.ndim == 1:
            indices = indices.to(device=device).unsqueeze(0).expand(batch_size, -1)
        else:
            indices = indices.to(device=device)

        expected_shape = (batch_size, expected_frames)
        if tuple(indices.shape) != expected_shape:
            raise ValueError(f"{name} must have shape {expected_shape}, got {tuple(indices.shape)}")
        if indices.numel() > 0 and indices.min().item() < 0:
            raise ValueError(f"{name} contains negative frame indices")
        return indices.long()

    def _rope_grid(self, frame_indices, height, width, device):
        frame_indices = frame_indices.to(device=device, dtype=torch.long)
        if frame_indices.numel() > 0 and frame_indices.max().item() >= self.freqs.size(0):
            raise ValueError(
                f"frame index {frame_indices.max().item()} exceeds RoPE table length {self.freqs.size(0)}"
            )
        if height > self.freqs.size(0) or width > self.freqs.size(0):
            raise ValueError(
                f"spatial RoPE size {(height, width)} exceeds RoPE table length {self.freqs.size(0)}"
            )

        half_dim = self.freqs.size(1)
        freq_splits = self.freqs.split(
            [
                half_dim - 2 * (half_dim // 3),
                half_dim // 3,
                half_dim // 3,
            ],
            dim=1,
        )
        batch_size, frames = frame_indices.shape
        freq_t = freq_splits[0].index_select(0, frame_indices.reshape(-1)).view(
            batch_size, frames, -1
        )
        freq_y = freq_splits[1][:height].to(device=device)
        freq_x = freq_splits[2][:width].to(device=device)
        freq_t = freq_t[:, :, None, None, :].expand(batch_size, frames, height, width, -1)
        freq_y = freq_y[None, None, :, None, :].expand(batch_size, frames, height, width, -1)
        freq_x = freq_x[None, None, None, :, :].expand(batch_size, frames, height, width, -1)
        return torch.cat([freq_t, freq_y, freq_x], dim=-1)

    def _downsample_rope_grid(self, freqs_grid, kernel_size):
        freqs_grid = freqs_grid.permute(0, 4, 1, 2, 3).contiguous()
        freqs_grid = pad_for_3d_conv(freqs_grid, kernel_size)
        freqs_grid = _avg_pool3d_complex(freqs_grid, kernel_size)
        return freqs_grid.permute(0, 2, 3, 4, 1).contiguous()

    @staticmethod
    def _freq_grid_to_tokens(freqs_grid):
        batch_size, frames, height, width, half_dim = freqs_grid.shape
        return freqs_grid.reshape(batch_size, frames * height * width, 1, half_dim)

    def _patch_history_segment(
        self,
        latents,
        indices,
        patch_layer,
        patch_kernel,
        rope_kernel,
        batch_size,
        base_grid_h,
        base_grid_w,
        device,
        name,
    ):
        if latents.ndim != 5:
            raise ValueError(f"{name} must have shape [B, C, T, H, W], got {tuple(latents.shape)}")
        if latents.shape[0] != batch_size or latents.shape[1] != self.in_dim:
            raise ValueError(
                f"{name} must have batch={batch_size} and {self.in_dim} latent/cond channels, "
                f"got {tuple(latents.shape)}"
            )

        raw_frames = latents.shape[2]
        indices = self._normalize_indices(indices, batch_size, raw_frames, device, f"indices_{name}")
        latents = pad_for_3d_conv(latents.to(device=device, dtype=self.patch_embedding.weight.dtype), patch_kernel)
        tokens_5d = patch_layer(latents)
        tokens = tokens_5d.flatten(2).transpose(1, 2)

        freqs_grid = self._rope_grid(indices, base_grid_h, base_grid_w, device)
        if rope_kernel is not None:
            freqs_grid = self._downsample_rope_grid(freqs_grid, rope_kernel)
        freqs = self._freq_grid_to_tokens(freqs_grid)
        if tokens.size(1) != freqs.size(1):
            raise ValueError(
                f"{name} token/RoPE length mismatch: tokens={tokens.size(1)}, freqs={freqs.size(1)}"
            )
        return tokens, freqs

    def _control_to_tensor(self, value, batch_size, name, expected_raw_shape=None):
        if isinstance(value, (list, tuple)):
            value = torch.cat(list(value), dim=0)
        if not isinstance(value, torch.Tensor):
            raise ValueError(f"{name} must be a Tensor or a list/tuple of Tensor chunks")
        if value.ndim != 5 or value.shape[0] != batch_size or value.shape[1] != self.control_channels:
            raise ValueError(
                f"{name} must have shape [B, {self.control_channels}, T, H, W], "
                f"got {tuple(value.shape)}"
            )
        if expected_raw_shape is not None and tuple(value.shape[2:]) != tuple(expected_raw_shape):
            raise ValueError(
                f"{name} raw T/H/W must strictly match its latent branch. "
                f"Expected {tuple(expected_raw_shape)}, got {tuple(value.shape[2:])}"
            )
        return value

    def _patch_control_segment(self, control, patch_kernel, linear, expected_tokens, device, name):
        control = pad_for_3d_conv(control.to(device=device, dtype=self.patch_embedding.weight.dtype), patch_kernel)
        tokens = rearrange(
            control,
            "b c (f c1) (h c2) (w c3) -> b (f h w) (c c1 c2 c3)",
            c1=patch_kernel[0],
            c2=patch_kernel[1],
            c3=patch_kernel[2],
        )
        if tokens.size(1) != expected_tokens:
            raise ValueError(
                f"{name} control token length mismatch: expected {expected_tokens}, got {tokens.size(1)}"
            )
        tokens = linear(tokens)
        hidden_states = self.c2ws_hidden_states_layer2(torch_F.silu(self.c2ws_hidden_states_layer1(tokens)))
        return tokens + hidden_states

    def _build_control_tokens(
        self,
        dit_cond_dict,
        batch_size,
        target_tokens,
        history_tokens,
        target_raw_shape,
        history_raw_shapes,
        device,
        ar_history=False,
        source_tokens=None,
        source_raw_shape=None,
    ):
        if dit_cond_dict is None or "c2ws_plucker_emb" not in dit_cond_dict:
            return None

        pieces = []
        # Keep controls aligned with x = [source | history | target].
        if source_tokens is not None:
            if "c2ws_plucker_emb_source" not in dit_cond_dict:
                raise ValueError("Missing required source control key: c2ws_plucker_emb_source")
            control_source = self._control_to_tensor(
                dit_cond_dict["c2ws_plucker_emb_source"],
                batch_size,
                "c2ws_plucker_emb_source",
                expected_raw_shape=source_raw_shape,
            )
            pieces.append(
                self._patch_control_segment(
                    control_source,
                    self.patch_size,
                    self.patch_embedding_wancamctrl,
                    source_tokens.size(1),
                    device,
                    "c2ws_plucker_emb_source",
                )
            )
        if history_tokens:
            if "c2ws_plucker_emb_history_short" not in dit_cond_dict:
                raise ValueError("Missing required AR control key: c2ws_plucker_emb_history_short")
            if history_raw_shapes is None:
                raise ValueError("history_raw_shapes are required when AR history tokens are provided")
            control_short = self._control_to_tensor(
                dit_cond_dict["c2ws_plucker_emb_history_short"],
                batch_size,
                "c2ws_plucker_emb_history_short",
                expected_raw_shape=history_raw_shapes["short"],
            )
            pieces.append(
                self._patch_control_segment(
                    control_short,
                    self.patch_size,
                    self.patch_embedding_wancamctrl,
                    history_tokens[0].size(1),
                    device,
                    "c2ws_plucker_emb_history_short",
                )
            )

        control_target = self._control_to_tensor(
            dit_cond_dict["c2ws_plucker_emb"],
            batch_size,
            "c2ws_plucker_emb",
            expected_raw_shape=target_raw_shape,
        )
        pieces.append(
            self._patch_control_segment(
                control_target,
                self.patch_size,
                self.patch_embedding_wancamctrl,
                target_tokens.size(1),
                device,
                "c2ws_plucker_emb",
            )
        )
        return {"c2ws_plucker_emb": torch.cat(pieces, dim=1)}

    def _mv_self_groups(self, num_views, s_tok, hv_tok, tv_tok, chunk_tok, tgt_abs0, use_bi,
                        window_chunks=-1, sink_chunks=0,
                        mem_hist_sets=None, mem_src_chunks=0, mem_src_sets=None,
                        cross_now=True):
        """Build self-attention query groups for the source-conditioned
        multi-view layout, ``num_views`` (=k) targets grouped as
        [ source(s_tok) | h_0..h_{k-1} (hv_tok each) | t_0..t_{k-1} (tv_tok each)
          | (Bi: k*tv_tok) ]. Returns [(q_start,q_end,[(k_start,k_end),...]),...]
        tiling [0, L) in ascending q order. Rules:
          * SOURCE is a fully-observed given reference (NOT generated) -> no causality:
            each source chunk attends the WHOLE source (bidirectional); every target
            chunk (any view) attends the WHOLE source. Lets targets borrow scene
            content the source revealed at ANY time; "source runs out" (free-run) is
            the natural case where the finite source is simply all attended.
          * per-view own-history: block-causal within that view (cp<=c).
          * target view v, chunk p: WHOLE source + for EVERY view w (incl. v):
              - w's clean history over the WINDOW: sink chunks [0, sink_chunks) UNION
                sliding window [p-window_chunks, p). window_chunks<=0 => full [0, p)
                (== the committed generated chunks other views hold at rollout time).
                Windowing target-to-history attention matches rolling inference and
                preserves shift-invariance. The finite source stream is unwindowed.
              - w's noised chunk == p        (equal-time lockstep coupling: the
                                               chunk being co-denoised across views;
                                               w==v is this view's own noised chunk)
            It deliberately does NOT attend other views' noised target cp<p — at
            rollout those are clean committed history (attended via w's history),
            so attending the noised copy would break train/inference consistency.
          * Bi target: attend the whole Bi block only (unchanged v2 rule).
        At num_views==1 AND window_chunks<=0 this is byte-identical to the previous
        single-target layout (regression-safe default).

        P-Mem M1 (worldweaver_spatial_memory_plan §3.2/§3.5), both default-off:
          * mem_hist_sets (training): per-target-chunk, PER-TARGET-VIEW list of
            (view w, chunk c) retrieved GT memory entries — mem_hist_sets[t][v] is
            read ONLY by target view v (M1.1 per-view read, not shared). Entries
            must satisfy sink <= c <= p - win - 1 (anti-leak, disjoint sink∪window).
          * mem_src_chunks / mem_src_sets (inference): the LAST mem_src_chunks
            chunks of the source stream are committed memory anchors — anchors are
            self-only (the sink stays bidirectional over the sink part only), and
            target chunk t, view v attends the anchors listed in mem_src_sets[t][v]
            (per-view read, mirroring mem_hist_sets' M1.1 contract).
        """
        ct = int(chunk_tok)
        k = int(num_views)
        win = int(window_chunks)
        sink = int(sink_chunks)
        mem_hist = list(mem_hist_sets) if mem_hist_sets is not None else None
        mem_src = int(mem_src_chunks)
        if mem_hist is not None and win <= 0:
            raise ValueError("P-Mem mem_hist_sets requires windowed history (window_chunks > 0)")
        if mem_hist is not None and len(mem_hist) != len(tgt_abs0):
            raise ValueError(
                f"mem_hist_sets must have one entry per target chunk: "
                f"{len(mem_hist)} != {len(tgt_abs0)}"
            )
        if mem_hist is not None and any(len(s) != int(num_views) for s in mem_hist):
            raise ValueError(
                "mem_hist_sets[t] must be per-view: one (w,c) list per target view "
                f"(len {int(num_views)})"
            )
        if mem_src > 0 and mem_src_sets is not None and len(mem_src_sets) != len(tgt_abs0):
            raise ValueError(
                f"mem_src_sets must have one entry per target chunk: "
                f"{len(mem_src_sets)} != {len(tgt_abs0)}"
            )
        if mem_src > 0 and mem_src_sets is not None \
                and any(len(s) != int(num_views) for s in mem_src_sets):
            raise ValueError(
                "mem_src_sets[t] must be per-view: one anchor-index list per target "
                f"view (len {int(num_views)})"
            )
        sink_tok = s_tok - mem_src * ct
        if mem_src > 0 and sink_tok < 0:
            raise ValueError(f"mem_src_chunks={mem_src} exceeds source stream ({s_tok} tokens)")
        # Each group is (query_start, query_end, ordinary_key_ranges,
        # memory_key_ranges). Keeping P-Mem separate lets attention tag only
        # retrieved keys; it is deliberately not applied to the rolling window.
        groups = []
        src0 = 0
        histk0 = s_tok
        tf0 = s_tok + k * hv_tok
        cs = s_tok // ct
        ch = hv_tok // ct
        # source: bidirectional within the (fully-observed) source stream. P-Mem
        # anchors ride the source tail and are SELF-ONLY (§3.2); the sink part
        # stays bidirectional over the sink part only.
        for c in range(cs):
            q0 = src0 + c * ct
            if mem_src > 0 and q0 >= sink_tok:
                groups.append((q0, q0 + ct, [(q0, q0 + ct)], []))
            else:
                groups.append(
                    (q0, q0 + ct, [(src0, sink_tok if mem_src > 0 else s_tok)], [])
                )
        # per-view history: block-causal within its own view. Windowed/rolling mode
        # (win>0): each history chunk attends ONLY ITSELF — at rollout the kept
        # prev chunk is re-encoded alone, and rolling slots give every history
        # chunk the same RoPE phase, so cross-chunk history attention would alias.
        for v in range(k):
            base = histk0 + v * hv_tok
            for c in range(ch):
                if win > 0:
                    groups.append((base + c * ct, base + (c + 1) * ct,
                                   [(base + c * ct, base + (c + 1) * ct)], []))
                else:
                    groups.append(
                        (base + c * ct, base + (c + 1) * ct,
                         [(base, base + (c + 1) * ct)], [])
                    )
        # per-view target
        for v in range(k):
            tbase = tf0 + v * tv_tok
            for t, p in enumerate(int(x) for x in tgt_abs0):
                qs = tbase + t * ct
                qe = qs + ct
                ranges = []
                memory_ranges = []
                if s_tok > 0:                       # WHOLE source (non-causal)
                    s_end = sink_tok if mem_src > 0 else s_tok
                    if s_end > 0:
                        ranges.append((src0, s_end))
                p_end = min(p * ct, hv_tok)         # clean history up to (exclusive) chunk p
                for w in range(k):
                    hb = histk0 + w * hv_tok
                    if win > 0 and p_end > 0:
                        # sink [0, sink) UNION sliding window [p-win, p)
                        sink_end = min(sink * ct, p_end)
                        if sink_end > 0:
                            ranges.append((hb, hb + sink_end))
                        lo = max(max(0, p - win) * ct, sink_end)
                        if p_end > lo:
                            ranges.append((hb + lo, hb + p_end))
                    elif p_end > 0:                 # full history (regression default)
                        ranges.append((hb, hb + p_end))
                    # Every query always sees its own current noised chunk.  The
                    # alternate layout for the geometry gate removes only peers.
                    if w == v or cross_now:
                        wtb = tf0 + w * tv_tok
                        ranges.append((wtb + t * ct, wtb + t * ct + ct))
                # P-Mem memory reads (per target view v; §3.6 anti-leak checks).
                if mem_hist is not None:
                    for (w_m, c_m) in mem_hist[t][v]:
                        if not (0 <= w_m < k) or not (sink <= c_m <= p - win - 1) \
                                or (c_m + 1) * ct > hv_tok:
                            raise ValueError(
                                f"P-Mem leak/range: mem entry (view={w_m}, chunk={c_m}) "
                                f"invalid for target p={p} (win={win}, sink={sink}, "
                                f"hist_chunks={hv_tok // ct})"
                            )
                        mb = histk0 + w_m * hv_tok + c_m * ct
                        memory_ranges.append((mb, mb + ct))
                if mem_src > 0 and mem_src_sets is not None:
                    for m in mem_src_sets[t][v]:
                        if not (0 <= m < mem_src):
                            raise ValueError(
                                f"P-Mem: mem_src_sets[{t}][{v}] index {m} out of range "
                                f"[0, {mem_src})"
                            )
                        mb = sink_tok + m * ct
                        memory_ranges.append((mb, mb + ct))
                groups.append((qs, qe, ranges, memory_ranges))
        if use_bi:
            bi0 = tf0 + k * tv_tok
            groups.append((bi0, bi0 + k * tv_tok, [(bi0, bi0 + k * tv_tok)], []))
        return groups

    def _build_time_embeddings(self, t, batch_size, target_seq_len, history_seq_len, device, history_timestep):
        if t.dim() == 1:
            target_t = t.to(device=device)[:, None].expand(batch_size, target_seq_len)
        elif t.dim() == 2 and tuple(t.shape) == (batch_size, target_seq_len):
            target_t = t.to(device=device)
        else:
            raise ValueError(
                f"t must have shape [B] or [B, target_seq_len={target_seq_len}], got {tuple(t.shape)}"
            )

        if history_seq_len > 0:
            if self.zero_history_timestep:
                history_t = torch.full(
                    (batch_size, history_seq_len),
                    float(history_timestep),
                    device=device,
                    dtype=target_t.dtype,
                )
            else:
                history_t = target_t[:, :1].expand(batch_size, history_seq_len)
            token_t = torch.cat([history_t, target_t], dim=1)
        else:
            token_t = target_t

        with torch.amp.autocast(device_type, dtype=torch.float32):
            flat_t = token_t.flatten()
            e = self.time_embedding(
                sinusoidal_embedding_1d(self.freq_dim, flat_t)
                .unflatten(0, (batch_size, token_t.size(1)))
                .float()
            )
            e0 = self.time_projection(e).unflatten(2, (6, self.dim))
        return e.float(), e0.float()

    def forward(
        self,
        x,
        t,
        context,
        seq_len,
        y=None,
        dit_cond_dict=None,
        indices_hidden_states=None,
        indices_latents_history_short=None,
        latents_history_short=None,
        history_timestep=0.0,
        ar_history=True,
        ar_layout=None,
        chunk_text_idx=None,
        a_b_emb=None,
        a_g_emb=None,
        enable_bi=True,
        latents_source=None,
        indices_source=None,
        num_views=1,
        x_bi=None,
        t_bi=None,
    ):
        # Seed-less chunk-by-chunk AR: a single 1x history stream (the N chunks).
        has_history = latents_history_short is not None
        if has_history and indices_latents_history_short is None:
            raise ValueError(
                "AR history requires latents_history_short and "
                "indices_latents_history_short together"
            )

        device = self.patch_embedding.weight.device
        if self.freqs.device != device:
            self.freqs = self.freqs.to(device)

        batch_size = len(x)
        if batch_size == 0:
            raise ValueError("x must contain at least one video latent")
        if (x_bi is None) != (t_bi is None):
            raise ValueError("x_bi and t_bi must be provided together")
        # Pure 16-channel latent path: x is the full input; `y` (legacy i2v
        # cond_y) is only concatenated when the patch_embedding still expects
        # the full 36-channel i2v input. When the caller has stripped the model
        # to 16 channels, `y` is ignored.
        if y is not None and self.in_dim == x[0].shape[0] + y[0].shape[0]:
            if len(y) != batch_size:
                raise ValueError(f"x/y batch mismatch: len(x)={batch_size}, len(y)={len(y)}")
            x = [torch.cat([latent, cond], dim=0) for latent, cond in zip(x, y)]
        if x_bi is not None:
            if len(x_bi) != batch_size:
                raise ValueError(
                    f"x/x_bi batch mismatch: len(x)={batch_size}, len(x_bi)={len(x_bi)}"
                )
            if y is not None and self.in_dim == x_bi[0].shape[0] + y[0].shape[0]:
                if len(y) != batch_size:
                    raise ValueError(
                        f"x_bi/y batch mismatch: len(x_bi)={len(x_bi)}, len(y)={len(y)}"
                    )
                x_bi = [
                    torch.cat([latent, cond], dim=0)
                    for latent, cond in zip(x_bi, y)
                ]
        target_raw_shape = tuple(x[0].shape[1:])
        for batch_idx, latent in enumerate(x):
            if latent.ndim != 4:
                raise ValueError(f"x[{batch_idx}] must have shape [C, T, H, W], got {tuple(latent.shape)}")
            if tuple(latent.shape[1:]) != target_raw_shape:
                raise ValueError(
                    "Stage-1 AR requires equal target raw T/H/W in a batch. "
                    f"Expected {target_raw_shape}, got x[{batch_idx}]={tuple(latent.shape[1:])}"
                )

        target_5d = [self.patch_embedding(latent.unsqueeze(0)) for latent in x]
        grid_sizes = torch.stack([torch.tensor(latent.shape[2:], dtype=torch.long) for latent in target_5d])
        if len({tuple(item.tolist()) for item in grid_sizes}) != 1:
            raise ValueError(f"Stage-1 AR requires equal target grid sizes in a batch, got {grid_sizes.tolist()}")
        target_tokens = torch.cat([latent.flatten(2).transpose(1, 2) for latent in target_5d], dim=0)
        target_seq_len = target_tokens.size(1)
        target_frames, target_grid_h, target_grid_w = grid_sizes[0].tolist()

        # The Bi regularizer is a genuinely separate noisy view of the target.
        # It must not reuse TF's per-chunk-noised tokens: Bi self-attention is full
        # over this whole block, so all of its tokens must come from one video-level
        # sigma and use the matching scalar t_bi. RoPE/camera/view identity remain
        # identical to the TF target because both represent the same clean frames.
        bi_target_tokens = None
        if x_bi is not None:
            for batch_idx, latent in enumerate(x_bi):
                if latent.ndim != 4:
                    raise ValueError(
                        f"x_bi[{batch_idx}] must have shape [C, T, H, W], "
                        f"got {tuple(latent.shape)}"
                    )
                if tuple(latent.shape[1:]) != target_raw_shape:
                    raise ValueError(
                        "TF and Bi targets must have identical raw T/H/W. "
                        f"Expected {target_raw_shape}, got "
                        f"x_bi[{batch_idx}]={tuple(latent.shape[1:])}"
                    )
            bi_target_5d = [
                self.patch_embedding(latent.unsqueeze(0)) for latent in x_bi
            ]
            bi_grid_sizes = torch.stack(
                [
                    torch.tensor(latent.shape[2:], dtype=torch.long)
                    for latent in bi_target_5d
                ]
            )
            if not torch.equal(bi_grid_sizes, grid_sizes):
                raise ValueError(
                    "TF and Bi targets must patchify to identical grids. "
                    f"TF={grid_sizes.tolist()}, Bi={bi_grid_sizes.tolist()}"
                )
            bi_target_tokens = torch.cat(
                [latent.flatten(2).transpose(1, 2) for latent in bi_target_5d],
                dim=0,
            )
            if bi_target_tokens.size(1) != target_seq_len:
                raise ValueError(
                    "TF and Bi target token lengths differ: "
                    f"TF={target_seq_len}, Bi={bi_target_tokens.size(1)}"
                )

        indices_hidden_states = self._normalize_indices(
            indices_hidden_states,
            batch_size,
            target_frames,
            device,
            "indices_hidden_states",
        )
        target_freqs = self._freq_grid_to_tokens(
            self._rope_grid(indices_hidden_states, target_grid_h, target_grid_w, device)
        )

        history_tokens = []
        history_freqs = []
        history_raw_shapes = None
        if has_history:
            history_raw_shapes = {
                "short": tuple(latents_history_short.shape[2:]),
            }
            # Uniform layout: history uses the SAME base patchify as the target
            # (single 1x patch size), so no AR-specific patch_short/mid/long.
            short_tokens, short_freqs = self._patch_history_segment(
                latents_history_short,
                indices_latents_history_short,
                self.patch_embedding,
                self.patch_size,
                None,
                batch_size,
                target_grid_h,
                target_grid_w,
                device,
                "latents_history_short",
            )
            history_tokens = [short_tokens]
            history_freqs = [short_freqs]

        # Optional clean source stream, prepended before target-view history.
        source_tokens = None
        source_freqs = None
        source_raw_shape = None
        if latents_source is not None:
            source_raw_shape = tuple(latents_source.shape[2:])
            source_tokens, source_freqs = self._patch_history_segment(
                latents_source,
                indices_source,
                self.patch_embedding,
                self.patch_size,
                None,
                batch_size,
                target_grid_h,
                target_grid_w,
                device,
                "latents_source",
            )

        prefix_tokens = ([source_tokens] if source_tokens is not None else []) + list(history_tokens)
        prefix_freqs = ([source_freqs] if source_freqs is not None else []) + list(history_freqs)
        x = torch.cat([*prefix_tokens, target_tokens], dim=1) if prefix_tokens else target_tokens
        token_freqs = torch.cat([*prefix_freqs, target_freqs], dim=1) if prefix_freqs else target_freqs
        total_seq_len = x.size(1)
        history_seq_len = total_seq_len - target_seq_len
        source_seq_len = source_tokens.size(1) if source_tokens is not None else 0
        own_hist_seq_len = history_seq_len - source_seq_len
        if int(seq_len) != total_seq_len:
            raise ValueError(
                f"seq_len must equal AR token length exactly: expected {total_seq_len}, got {seq_len}"
            )

        seq_lens = torch.full((batch_size,), total_seq_len, dtype=torch.long, device=device)
        e, e0 = self._build_time_embeddings(
            t,
            batch_size,
            target_seq_len,
            history_seq_len,
            device,
            history_timestep,
        )

        context_lens = None
        context = self.text_embedding(
            torch.stack(
                [torch.cat([item, item.new_zeros(self.text_len - item.size(0), item.size(1))]) for item in context]
            )
        )

        # Per-token prompt id over the global doubled sequence (history half then
        # target half). Chunk c of either half maps to prompt chunk_text_idx[c']:
        # history chunk c' = c; target chunk c' = tgt_abs_chunks[c]-1 (0-indexed
        # own chunk). None => single-prompt cross-attn (inference / no per-chunk).
        context_chunk_pos = None
        context_is_bi = None
        moba_cti = None
        a_b_ctx = None
        if a_b_emb is not None:
            # a_B (background prompt) embedded like `context`: [1, text_len, dim].
            a_b_ctx = self.text_embedding(a_b_emb)
        a_g_ctx = None
        if a_g_emb is not None:
            # a_G (global prompt) for the MoBA Bi segment: [1, text_len, dim].
            a_g_ctx = self.text_embedding(a_g_emb)
        if ar_history and ar_layout is not None and chunk_text_idx is not None:
            ftok_ctx = int(target_grid_h) * int(target_grid_w)
            chunk_tok_ctx = int(ar_layout["chunk_size"]) * ftok_ctx
            cti = chunk_text_idx.to(device=device, dtype=torch.long).reshape(-1)
            tgt_abs0 = torch.tensor(
                [int(p) - 1 for p in ar_layout["tgt_abs_chunks"]],
                device=device,
                dtype=torch.long,
            )
            if history_seq_len > 0:
                # The clean prefix is source followed by target-view history; each
                # stream starts its local chunk positions at zero.
                if source_seq_len > 0:
                    source_cp = torch.arange(source_seq_len, device=device) // chunk_tok_ctx
                else:
                    source_cp = cti.new_empty(0)
                if own_hist_seq_len > 0:
                    # History is concatenated by view, with local chunk positions.
                    hv_tok_cp = own_hist_seq_len // num_views
                    own_hist_cp = (
                        torch.arange(own_hist_seq_len, device=device) % hv_tok_cp
                    ) // chunk_tok_ctx
                else:
                    own_hist_cp = cti.new_empty(0)
                hist_chunk = torch.cat([source_cp, own_hist_cp], dim=0)
            else:
                hist_chunk = cti.new_empty(0)
            # Targets are concatenated by view, with local chunk positions.
            tv_tok_cp = target_seq_len // num_views
            tgt_local_chunk = (
                torch.arange(target_seq_len, device=device) % tv_tok_cp
            ) // chunk_tok_ctx
            # Per-token ABSOLUTE chunk position (0-indexed) for MoBA cumulative
            # cross-attn: a token in (history or target) chunk at absolute position
            # p attends to a_B + a_1..a_p (the chunk prompts of chunks 0..p).
            # moba_cti maps absolute chunk position -> unique-prompt id.
            tgt_pos = tgt_abs0[tgt_local_chunk]
            context_chunk_pos = torch.cat([hist_chunk, tgt_pos], dim=0)
            moba_cti = cti

        dit_cond_dict = self._build_control_tokens(
            dit_cond_dict,
            batch_size,
            target_tokens,
            history_tokens,
            target_raw_shape,
            history_raw_shapes,
            device,
            ar_history=ar_history,
            source_tokens=source_tokens,
            source_raw_shape=source_raw_shape,
        )

        # ---- MoBA sequence [ clean history | TF-target (| Bi-target) ] ----
        # The target half is DUPLICATED (Bi copy) ONLY when enable_bi is set, so
        # self- and cross-attention each run in a SINGLE call over the combined mask
        # (paper §3.2): the TF copy uses causal self-attn + cumulative a_B+a_<=p
        # cross-attn; the Bi copy uses full (clean+Bi) self-attn + global a_G
        # cross-attn. When enable_bi is False the Bi copy is omitted entirely — the
        # sequence stays [clean | TF] (doubled), which cuts ~1/3 of the activation
        # memory (used to fit longer clips on memory-bound NPUs).
        moba_active = (
            a_b_ctx is not None
            and context_chunk_pos is not None
            and moba_cti is not None
        )
        use_bi = moba_active and bool(enable_bi)
        if use_bi:
            if bi_target_tokens is None or t_bi is None:
                raise ValueError(
                    "MoBA Bi training requires x_bi plus one scalar t_bi per sample; "
                    "reusing TF's per-chunk-noised target would leak across noise levels"
                )
            if t_bi.dim() != 1 or tuple(t_bi.shape) != (batch_size,):
                raise ValueError(
                    "t_bi must contain exactly one full-video timestep per sample, "
                    f"expected {(batch_size,)}, got {tuple(t_bi.shape)}"
                )
            bi_e, bi_e0 = self._build_time_embeddings(
                t_bi,
                batch_size,
                target_seq_len,
                0,
                device,
                history_timestep,
            )
            x = torch.cat([x, bi_target_tokens], dim=1)
            token_freqs = torch.cat([token_freqs, target_freqs], dim=1)
            e = torch.cat([e, bi_e], dim=1)
            e0 = torch.cat([e0, bi_e0], dim=1)
            if dit_cond_dict is not None and "c2ws_plucker_emb" in dit_cond_dict:
                dit_cond_dict = dict(dit_cond_dict)
                dit_cond_dict["c2ws_plucker_emb"] = torch.cat(
                    [
                        dit_cond_dict["c2ws_plucker_emb"],
                        dit_cond_dict["c2ws_plucker_emb"][:, history_seq_len:],
                    ],
                    dim=1,
                )
            context_chunk_pos = torch.cat(
                [context_chunk_pos, context_chunk_pos[history_seq_len:]], dim=0
            )
            # is_bi: 0 for clean history + TF-target, 1 for the Bi-target copy.
            context_is_bi = torch.cat(
                [
                    torch.zeros(total_seq_len, device=device, dtype=torch.long),
                    torch.ones(target_seq_len, device=device, dtype=torch.long),
                ],
                dim=0,
            )
            seq_lens = torch.full(
                (batch_size,), x.size(1), dtype=torch.long, device=device
            )
        elif x_bi is not None or t_bi is not None:
            raise ValueError(
                "x_bi/t_bi were provided but the MoBA Bi branch is inactive "
                "(requires enable_bi=True and complete MoBA conditioning)"
            )

        parallel_dims = get_parallel_state()
        if parallel_dims.sp_enabled:
            sp_size = parallel_dims.sp
            sp_rank = parallel_dims.sp_rank
            x_chunks = torch.chunk(x, sp_size, dim=1)
            split_sizes = [chunk.shape[1] for chunk in x_chunks]
            if len(split_sizes) != sp_size:
                raise ValueError(
                    f"Too short context length for SP {sp_size}: seq_len={x.shape[1]}, "
                    f"chunks={len(split_sizes)}"
                )
            parallel_dims.set_sp_sequence_info(split_sizes)
            x = x_chunks[sp_rank]
            e = torch.split(e, split_sizes, dim=1)[sp_rank]
            e0 = torch.split(e0, split_sizes, dim=1)[sp_rank]
            token_freqs = torch.split(token_freqs, split_sizes, dim=1)[sp_rank]
            if context_chunk_pos is not None:
                context_chunk_pos = torch.split(context_chunk_pos, split_sizes, dim=0)[sp_rank]
            if context_is_bi is not None:
                context_is_bi = torch.split(context_is_bi, split_sizes, dim=0)[sp_rank]
            if dit_cond_dict is not None:
                dit_cond_dict = dict(dit_cond_dict)
                dit_cond_dict["c2ws_plucker_emb"] = torch.split(
                    dit_cond_dict["c2ws_plucker_emb"],
                    split_sizes,
                    dim=1,
                )[sp_rank]

        # Precompute MoBA cross-attn chunk groups ONCE (shared by every layer):
        # contiguous runs of equal (absolute chunk position, is_target) ->
        # (start, end, p, is_target). Doing this here (single GPU->CPU sync per
        # forward) avoids a per-layer .tolist() sync inside cross_attn_ffn.
        moba_groups = None
        if context_chunk_pos is not None:
            pos_list = context_chunk_pos.tolist()
            bi_list = (
                context_is_bi.tolist()
                if context_is_bi is not None
                else [0] * len(pos_list)
            )
            moba_groups = []
            i = 0
            L = len(pos_list)
            while i < L:
                p = int(pos_list[i])
                is_bi = int(bi_list[i])
                j = i
                while j < L and int(pos_list[j]) == p and int(bi_list[j]) == is_bi:
                    j += 1
                moba_groups.append((i, j, p, is_bi))
                i = j

        causal_layout = None
        if ar_history and ar_layout is not None:
            ftok = int(target_grid_h) * int(target_grid_w)
            chunk_size = int(ar_layout["chunk_size"])
            # History chunk count from the actual history-half frames (seed-less:
            # H chunks). Target absolute chunk positions come from the caller
            # (training: [1..N]; rollout/generation: [j+1]). No history (rollout
            # chunk 0) -> latents_history_short is None -> H = 0.
            hist_frames = int(latents_history_short.shape[2]) if has_history else 0
            num_hist_chunks = hist_frames // chunk_size
            causal_layout = {
                "chunk_tok": chunk_size * ftok,
                "num_hist_chunks": num_hist_chunks,
                "tgt_abs_chunks": list(ar_layout["tgt_abs_chunks"]),
                "include_bi": use_bi,
            }
            # Multi-view layout uses explicit query groups over
            # [optional source | h_0..h_{k-1} | t_0..t_{k-1} | (Bi)].
            # The source-free shared-image path is the normal K-agent case:
            # source_seq_len=0, while current chunks still communicate across
            # views and clean histories are visible across views.
            if latents_source is not None or num_views > 1:
                tgt_abs0_list = [int(p) - 1 for p in ar_layout["tgt_abs_chunks"]]
                if own_hist_seq_len % num_views != 0 or target_seq_len % num_views != 0:
                    raise ValueError(
                        f"multi-view history/target must divide evenly by num_views={num_views}: "
                        f"own_hist={own_hist_seq_len}, target={target_seq_len}"
                    )
                # Training provides a dense history and applies the rolling window;
                # rollout and inference provide an already-windowed history.
                causal_layout["mv_groups"] = self._mv_self_groups(
                    num_views,
                    source_seq_len,
                    own_hist_seq_len // num_views,
                    target_seq_len // num_views,
                    chunk_size * ftok,
                    tgt_abs0_list,
                    use_bi,
                    window_chunks=int(ar_layout.get("window_chunks", -1)),
                    sink_chunks=int(ar_layout.get("sink_chunks", 0)),
                    mem_hist_sets=ar_layout.get("mem_hist_sets"),
                    mem_src_chunks=int(ar_layout.get("mem_src_chunks", 0)),
                    mem_src_sets=ar_layout.get("mem_src_sets"),
                )
                gate = ar_layout.get("xnow_gate")
                if gate is not None:
                    gate = torch.as_tensor(gate, device=device, dtype=torch.float32).reshape(-1)
                    if gate.numel() != target_seq_len:
                        raise ValueError(
                            "xnow_gate must provide one value per target token: "
                            f"got {gate.numel()}, expected {target_seq_len}"
                        )
                    without_peer = self._mv_self_groups(
                        num_views,
                        source_seq_len,
                        own_hist_seq_len // num_views,
                        target_seq_len // num_views,
                        chunk_size * ftok,
                        tgt_abs0_list,
                        use_bi,
                        window_chunks=int(ar_layout.get("window_chunks", -1)),
                        sink_chunks=int(ar_layout.get("sink_chunks", 0)),
                        mem_hist_sets=ar_layout.get("mem_hist_sets"),
                        mem_src_chunks=int(ar_layout.get("mem_src_chunks", 0)),
                        mem_src_sets=ar_layout.get("mem_src_sets"),
                        cross_now=False,
                    )
                    full_gate = torch.ones(total_seq_len, device=device, dtype=torch.float32)
                    full_gate[history_seq_len:history_seq_len + target_seq_len] = gate
                    if use_bi:
                        full_gate = torch.cat([full_gate, gate], dim=0)
                    causal_layout["mv_groups_without_peer"] = [
                        group
                        for group in without_peer
                        if history_seq_len <= group[0] < history_seq_len + target_seq_len
                    ]
                    causal_layout["xnow_gate"] = full_gate

        kwargs = dict(
            e=e0,
            seq_lens=seq_lens,
            grid_sizes=grid_sizes,
            freqs=token_freqs,
            context=context,
            context_lens=context_lens,
            dit_cond_dict=dit_cond_dict,
            causal_layout=causal_layout,
            a_b_ctx=a_b_ctx,
            a_g_ctx=a_g_ctx,
            moba_groups=moba_groups,
            moba_cti=moba_cti,
            moba_chunk_pos=context_chunk_pos,
            moba_is_bi=context_is_bi,
        )

        for block in self.blocks:
            x = block(x, **kwargs)

        if parallel_dims.sp_enabled:
            x = all_gather(x, dim=1, group=parallel_dims.sp_group)
            e = all_gather(e, dim=1, group=parallel_dims.sp_group)

        tf0 = history_seq_len
        tf1 = history_seq_len + target_seq_len
        if use_bi:
            # Tripled [clean | TF-target | Bi-target]: slice the two target copies
            # apart and run the head on each. Returns (tf_pred_list, bi_pred_list).
            bi0 = tf1
            bi1 = tf1 + target_seq_len
            x_tf = self.head(x[:, tf0:tf1], e[:, tf0:tf1])
            bi_hidden = self.head(x[:, bi0:bi1], e[:, bi0:bi1])
            tf_out = [item.float() for item in self.unpatchify(x_tf, grid_sizes)]
            bi_out = [item.float() for item in self.unpatchify(bi_hidden, grid_sizes)]
            return tf_out, bi_out

        if moba_active:
            # MoBA with Bi disabled: doubled [clean | TF-target]. Only the TF target
            # is predicted; return (tf_pred_list, None) so callers can skip the Bi loss.
            x_tf = self.head(x[:, tf0:tf1], e[:, tf0:tf1])
            tf_out = [item.float() for item in self.unpatchify(x_tf, grid_sizes)]
            return tf_out, None

        x = x[:, tf0:tf1]
        e = e[:, tf0:tf1]
        x = self.head(x, e)
        x = self.unpatchify(x, grid_sizes)
        return [item.float() for item in x]

    def unpatchify(self, x, grid_sizes):
        outputs = []
        for item, grid_size in zip(x, grid_sizes.tolist()):
            item = item[: math.prod(grid_size)].view(*grid_size, *self.patch_size, self.out_dim)
            item = torch.einsum("fhwpqrc->cfphqwr", item)
            item = item.reshape(
                self.out_dim,
                *[grid * patch for grid, patch in zip(grid_size, self.patch_size)],
            )
            outputs.append(item)
        return outputs

    def init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        nn.init.xavier_uniform_(self.patch_embedding.weight.flatten(1))
        for module in self.text_embedding.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
        for module in self.time_embedding.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)

        nn.init.zeros_(self.head.head.weight)

        # Zero initialization makes this optional embedding inert at initialization.
        nn.init.zeros_(self.view_embedding.weight)

        nn.init.xavier_uniform_(self.patch_embedding_wancamctrl.weight)
        nn.init.zeros_(self.patch_embedding_wancamctrl.bias)
        nn.init.xavier_uniform_(self.c2ws_hidden_states_layer1.weight)
        nn.init.zeros_(self.c2ws_hidden_states_layer1.bias)
        nn.init.xavier_uniform_(self.c2ws_hidden_states_layer2.weight)
        nn.init.zeros_(self.c2ws_hidden_states_layer2.bias)

        for block in self.blocks:
            nn.init.xavier_uniform_(block.cam_injector_layer1.weight)
            nn.init.zeros_(block.cam_injector_layer1.bias)
            nn.init.xavier_uniform_(block.cam_injector_layer2.weight)
            nn.init.zeros_(block.cam_injector_layer2.bias)
            nn.init.xavier_uniform_(block.cam_scale_layer.weight)
            nn.init.zeros_(block.cam_scale_layer.bias)
            nn.init.xavier_uniform_(block.cam_shift_layer.weight)
            nn.init.zeros_(block.cam_shift_layer.bias)


WanModel = WanModelAR
