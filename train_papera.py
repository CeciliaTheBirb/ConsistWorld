import argparse
import gc
import math
import os
import random
import time
from dataclasses import dataclass

import torch
import numpy as np
import torch.distributed as dist
import torch.nn as nn
from diffusers.optimization import get_scheduler
from torch.distributed._composable.fsdp import MixedPrecisionPolicy, fully_shard
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    CheckpointImpl,
    apply_activation_checkpointing,
    checkpoint_wrapper,
)

from wan.commons.parallel_states import get_parallel_state, initialize_parallel_state
from wan.configs import WAN_CONFIGS
from wan.dataset.pmem_retrieval import token_xnow_gate
from wan.dataset.sequence_parallel import sequence_parallel_batches
from wan.modules.model_ar import WanModelAR
from wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
from wan.utils.stage1_ar_flow import DEFAULT_FLOW_SHIFT
from wan.utils.stage1_ar_geometry import Stage1ARGeometry
from wan.utils.stage1_ar_selfresample import (
    ar_seq_len,
    resolve_history_timestep,
    rho_curriculum,
    sample_history_sigma,
)
from wan.utils.accel import (
    device_module,
    device_type,
    empty_cache as accel_empty_cache,
    is_cuda,
    synchronize as accel_synchronize,
)
from papera.checkpoints import load_strict_full_checkpoint


def compute_density_for_timestep_sampling(
    weighting_scheme: str,
    batch_size: int,
    generator=None,
    logit_mean: float | None = None,
    logit_std: float | None = None,
    mode_scale: float | None = None,
):
    if weighting_scheme == "logit_normal":
        u = torch.normal(
            mean=logit_mean,
            std=logit_std,
            size=(batch_size,),
            device="cpu",
            generator=generator,
        )
        u = torch.nn.functional.sigmoid(u)
    elif weighting_scheme == "mode":
        u = torch.rand(size=(batch_size,), device="cpu", generator=generator)
        u = 1 - u - mode_scale * (torch.cos(math.pi * u / 2) ** 2 - 1 + u)
    else:
        u = torch.rand(size=(batch_size,), device="cpu", generator=generator)
    return u


def compute_loss_weighting_for_sd3(weighting_scheme: str, sigmas: torch.Tensor):
    if weighting_scheme == "sigma_sqrt":
        weighting = (sigmas**-2.0).float()
    elif weighting_scheme == "cosmap":
        bot = 1 - 2 * sigmas + 2 * sigmas**2
        weighting = 2 / (math.pi * bot)
    else:
        weighting = torch.ones_like(sigmas)
    return weighting


def get_sigmas(noise_scheduler, device, timesteps, n_dim=4, dtype=torch.float32):
    sigmas = noise_scheduler.sigmas.to(device=device, dtype=dtype)
    schedule_timesteps = noise_scheduler.timesteps.to(device)
    timesteps = timesteps.to(device)
    step_indices = [(schedule_timesteps == timestep).nonzero().item() for timestep in timesteps]
    sigma = sigmas[step_indices].flatten()
    while len(sigma.shape) < n_dim:
        sigma = sigma.unsqueeze(-1)
    return sigma


@dataclass
class TrainingConfig:
    pretrained_model_root: str
    control_type: str
    learning_rate: float = 1e-5
    weight_decay: float = 0.01
    max_steps: int = 4000
    warmup_steps: int = 100
    gradient_accumulation_steps: int = 2
    max_grad_norm: float = 1.0
    train_timestep_shift: float = DEFAULT_FLOW_SHIFT
    uniform_sampling: bool = False
    weighting_scheme: str = "logit_normal"
    logit_mean: float = 0.0
    logit_std: float = 1.0
    mode_scale: float = 1.29
    enable_fsdp: bool = True
    enable_gradient_checkpointing: bool = True
    sp_size: int = 8
    dp_replicate: int = 1
    output_dir: str = ""
    save_interval: int = 500
    dtype: str = "bf16"
    seed: int = 42
    clip_cache_dir: str = ""
    clip_cache_k_min: int = 1
    clip_cache_k_max: int = 4
    clip_cache_max_total_view_chunks: int = 20
    clip_cache_target_chunks: int = 9
    clip_cache_cfg_rate: float = 0.0
    scene_filter: str = ""
    save_final_checkpoint: bool = True
    init_model_pt: str = ""


class LingbotSingleDiT(nn.Module):
    def __init__(self):
        super().__init__()
        self.model: WanModelAR | None = None

    @staticmethod
    def load_pretrained(
        pretrained_model_root: str,
        control_type: str,
        torch_dtype: torch.dtype,
        subfolder: str,
        device: torch.device,
        zero_history_timestep: bool,
        local_attn_size: int = -1,
        sink_size: int = 0,
        load_base_weights: bool = False,
    ) -> WanModelAR:
        """Build the AR model, optionally loading the foundation weights."""
        overrides = dict(
            control_type=control_type,
            zero_history_timestep=zero_history_timestep,
            local_attn_size=local_attn_size,
            sink_size=sink_size,
        )
        if load_base_weights:
            model = WanModelAR.from_pretrained(
                pretrained_model_root,
                subfolder=subfolder,
                torch_dtype=torch_dtype,
                device_map=None,
                low_cpu_mem_usage=False,
                **overrides,
            )
            return model.to(device)

        previous_dtype = torch.get_default_dtype()
        torch.set_default_dtype(torch_dtype)
        try:
            model = WanModelAR.from_config(
                WanModelAR.load_config(pretrained_model_root, subfolder=subfolder),
                **overrides,
            )
        finally:
            torch.set_default_dtype(previous_dtype)
        return model.to(device=device, dtype=torch_dtype)


def sync_tensor_for_sp(tensor: torch.Tensor, sp_group):
    """Broadcast `tensor` from SP-group rank 0 to all other ranks in the group.

    Assumes `tensor` is a Tensor (not an arbitrary Python object). The
    SP-rank-0 tensor's shape/dtype is broadcast first via object_list so
    receivers can allocate matching buffers.
    """
    if sp_group is None:
        return tensor
    is_src = dist.get_rank() == dist.get_global_rank(sp_group, 0)
    if is_src:
        tensor = tensor.to(device_type).contiguous()
        meta = [(tensor.shape, tensor.dtype)]
    else:
        meta = [None]
    dist.broadcast_object_list(meta, group_src=0, group=sp_group)
    shape, dtype = meta[0]
    buffer = tensor if is_src else torch.empty(shape, device=device_type, dtype=dtype)
    dist.broadcast(buffer, group_src=0, group=sp_group)
    return buffer


@dataclass
class Stage1ARTrainingConfig(TrainingConfig):
    """Fixed training recipe used for PaperA."""

    chunk_size: int = 4

    # Resampling-forcing rollout settings.
    self_resample: bool = True
    sr_shift: float = 0.25
    sr_rollout_steps: int = 4
    sr_rho_start: float = 0.0
    sr_rho_end: float = 0.2
    sr_rho_anneal_steps: int = 4000
    sr_rho_schedule: str = "linear"
    sr_sink_size: int = 0
    sr_local_attn_size: int = 4

    # Per-target-view top-1 P-Mem and equal-time geometry gate.
    pmem: bool = True
    pmem_r: int = 1
    xnow_gate: bool = True
    xnow_gate_start: float = 0.0
    xnow_gate_end: float = 1.0
    xnow_gate_anneal_steps: int = 500

class LingbotStage1ARTrainer:
    def __init__(self, config: Stage1ARTrainingConfig):
        self.config = config
        if not config.clip_cache_dir:
            raise ValueError("--clip_cache_dir is required")
        if int(config.sr_local_attn_size) != int(config.chunk_size):
            raise ValueError(
                "PaperA uses a one-chunk rolling window: "
                "sr_local_attn_size must equal chunk_size"
            )
        if int(config.sr_sink_size) != 0:
            raise ValueError("PaperA uses the shared image as its only sink (sr_sink_size=0)")
        if int(config.chunk_size) != 4:
            raise ValueError("PaperA uses chunk_size=4")
        self.wan_config = WAN_CONFIGS["i2v-A14B"]
        self.device = torch.device(device_type)

        if "RANK" in os.environ:
            self.rank = int(os.environ["RANK"])
            self.world_size = int(os.environ.get("WORLD_SIZE", "1"))
            self.local_rank = int(os.environ.get("LOCAL_RANK", "0"))
            self.device = torch.device(f"{device_type}:{self.local_rank}")
            self.is_main_process = self.rank == 0
        else:
            self.rank = 0
            self.world_size = 1
            self.local_rank = 0
            self.is_main_process = True

        if config.sp_size > self.world_size:
            raise ValueError(
                f"sp_size ({config.sp_size}) cannot be greater than world_size ({self.world_size})"
            )
        if self.world_size % config.sp_size != 0:
            raise ValueError(
                f"sp_size ({config.sp_size}) must evenly divide world_size ({self.world_size}). "
                f"world_size % sp_size = {self.world_size % config.sp_size}"
            )
        if self.wan_config.num_heads % config.sp_size != 0:
            raise ValueError(
                f"num_heads ({self.wan_config.num_heads}) must be divisible by sp_size ({config.sp_size})"
            )
        self.geometry = Stage1ARGeometry(
            chunk_size=int(config.chunk_size),
            vae_temporal_stride=int(self.wan_config.vae_stride[0]),
            patch_size=tuple(self.wan_config.patch_size),
        )

        initialize_parallel_state(sp=config.sp_size, dp_replicate=config.dp_replicate)
        device_module.set_device(self.local_rank)
        if is_cuda():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        self.parallel_state = get_parallel_state()
        self.dp_rank = self.parallel_state.world_mesh["dp"].get_local_rank()
        self.dp_size = self.parallel_state.world_mesh["dp"].size()
        self.sp_enabled = self.parallel_state.sp_enabled
        self.sp_group = self.parallel_state.sp_group if self.sp_enabled else None

        if self.is_main_process:
            print(self.parallel_state.world_mesh)
            print(self.parallel_state.fsdp_mesh)

        self._set_seed(config.seed + self.dp_rank)
        self.num_train_timesteps = int(self.wan_config.num_train_timesteps)
        self.checkpoint_module: nn.Module | None = None

        self._build_models()
        self._build_optimizer()

        # Initialize collective communicators before the model fills device memory.
        if self.world_size > 1:
            dist.all_reduce(torch.zeros(1, device=self.device))
            try:
                mesh = self.parallel_state.fsdp_mesh
                for dim in range(mesh.ndim):
                    dist.all_reduce(torch.zeros(1, device=self.device),
                                    group=mesh.get_group(dim))
            except Exception as exc:
                if self.is_main_process:
                    print(f"[train] fsdp communicator initialization skipped: {exc}", flush=True)
            accel_synchronize()

        self.noise_schedule = FlowUniPCMultistepScheduler(
            num_train_timesteps=self.num_train_timesteps,
            shift=self.config.train_timestep_shift,
            use_dynamic_shifting=False,
        )
        self.global_step = 0
        self.last_grad_norm = 0.0
        if self.is_main_process:
            os.makedirs(config.output_dir, exist_ok=True)

    def _set_seed(self, seed: int):
        random.seed(seed)
        torch.manual_seed(seed)
        if device_type != "cpu" and hasattr(device_module, "manual_seed_all"):
            device_module.manual_seed_all(seed)

    def _build_models(self):
        if self.config.dtype == "bf16":
            transformer_dtype = torch.bfloat16
        elif self.config.dtype == "fp32":
            transformer_dtype = torch.float32
        else:
            raise ValueError(f"Unsupported dtype: {self.config.dtype}")

        self.single_dit = LingbotSingleDiT()

        subfolder = "transformers"

        if self.is_main_process:
            print("[train] building from the Stage-1 model config")

        self.single_dit.model = self._load_and_wrap_model(
            subfolder=subfolder,
            torch_dtype=transformer_dtype,
        )

        self.checkpoint_module = self.single_dit

        self.single_dit.train()

        if self.is_main_process:
            total_params = sum(param.numel() for param in self.single_dit.parameters())
            trainable_params = sum(
                param.numel() for param in self.single_dit.parameters() if param.requires_grad
            )
            print(
                f"[train] single DiT loaded from {subfolder}. "
                f"total={total_params:,}, trainable={trainable_params:,}, "
                f"sp_size={self.config.sp_size}, dp_size={self.dp_size}"
            )

    def _load_and_wrap_model(self, subfolder: str, torch_dtype: torch.dtype) -> WanModelAR:
        init_pt = getattr(self.config, "init_model_pt", "") or ""
        if self.is_main_process:
            source = "warm-start architecture" if init_pt else "foundation weights"
            print(f"[train] loading {source} from subfolder={subfolder}")
        sr_local_attn_size = int(self.config.sr_local_attn_size)
        sr_sink_size = int(self.config.sr_sink_size)
        model = LingbotSingleDiT.load_pretrained(
            pretrained_model_root=self.config.pretrained_model_root,
            control_type=self.config.control_type,
            torch_dtype=torch_dtype,
            subfolder=subfolder,
            device=self.device,
            zero_history_timestep=True,
            local_attn_size=sr_local_attn_size,
            sink_size=sr_sink_size,
            load_base_weights=not bool(init_pt),
        )
        model.train()

        # Load a full architecture-matched warm start before checkpoint/FSDP wrapping.
        if init_pt:
            load_strict_full_checkpoint(model, init_pt)
            if self.is_main_process:
                print(f"[train] strict warm-start loaded from {init_pt}", flush=True)

        # Gradient checkpointing must not depend on the warm-start path: apply it
        # whenever the flag is on (base-safetensors loads included), before FSDP.
        if self.config.enable_gradient_checkpointing:
            if self.is_main_process:
                print(f"[train] applying gradient checkpointing: {subfolder}")
            self._apply_gradient_checkpointing_to_model(model)

        if self.config.enable_fsdp and self.world_size > 1:
            if self.is_main_process:
                print(f"[train] applying fsdp sharding: {subfolder}")
            self._apply_fsdp_to_model(model)

        return model

    def _apply_gradient_checkpointing_to_model(self, model: WanModelAR):
        block_type = None
        for block in model.blocks:
            if block is not None:
                block_type = type(block)
                break

        if block_type is None:
            return

        def non_reentrant_wrapper(module):
            return checkpoint_wrapper(module, checkpoint_impl=CheckpointImpl.NO_REENTRANT)

        def selective_checkpointing(submodule):
            return isinstance(submodule, block_type)

        apply_activation_checkpointing(
            model,
            checkpoint_wrapper_fn=non_reentrant_wrapper,
            check_fn=selective_checkpointing,
        )

    def _apply_fsdp_to_model(self, model: WanModelAR):
        param_dtype = torch.bfloat16 if self.config.dtype == "bf16" else torch.float32
        reduce_dtype = torch.float32
        mp_policy = MixedPrecisionPolicy(
            param_dtype=param_dtype,
            reduce_dtype=reduce_dtype,
            cast_forward_inputs=False,
        )
        fsdp_config = {"mp_policy": mp_policy}
        try:
            fsdp_config["mesh"] = self.parallel_state.fsdp_mesh
        except Exception as exc:
            raise RuntimeError(
                f"Cannot fetch fsdp_mesh from parallel_state "
                f"(sp_size={self.config.sp_size}, dp_replicate={self.config.dp_replicate}, "
                f"world_size={self.world_size}). Falling back to the default group would shard "
                f"parameters across SP ranks and break sequence parallelism. Original error: {exc}"
            ) from exc

        # Keep FSDP master parameters and optimizer states in fp32 while bf16 is
        # used for forward computation. Cast each block before sharding so the
        # temporary full fp32 model is never materialized on one device.
        cast_master_fp32 = param_dtype == torch.bfloat16
        for block in list(model.blocks):
            if cast_master_fp32:
                block.to(torch.float32)
            fully_shard(block, **fsdp_config)
        if cast_master_fp32:
            for p in model.parameters():
                # remaining root (non-block) params are still plain tensors
                if not hasattr(p, "full_tensor") and p.dtype == torch.bfloat16:
                    p.data = p.data.to(torch.float32)
        fully_shard(model, **fsdp_config)

    def _build_optimizer(self):
        # This unused tensor is retained only for strict compatibility with the
        # Stage-1 checkpoint ABI; it is never part of the released forward path.
        for name, param in self.single_dit.named_parameters():
            param.requires_grad = "view_embedding" not in name

        if self.is_main_process:
            trainable_count = sum(p.numel() for p in self.single_dit.parameters() if p.requires_grad)
            total_count = sum(p.numel() for p in self.single_dit.parameters())
            print(
                "[train] trainable=full: "
                f"{trainable_count/1e9:.3f}B / total={total_count/1e9:.2f}B "
                f"({100*trainable_count/max(total_count, 1):.2f}%)",
                flush=True,
            )

        params_to_optimize = [param for param in self.single_dit.parameters() if param.requires_grad]
        if self.is_main_process:
            master_dtypes = {str(p.dtype) for p in params_to_optimize}
            print(f"[train] master param dtypes: {sorted(master_dtypes)} "
                  f"(optimizer states follow via zeros_like)", flush=True)
        self.optimizer = torch.optim.AdamW(
            params_to_optimize,
            lr=self.config.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=self.config.weight_decay,
        )
        self.lr_scheduler = get_scheduler(
            "constant",
            optimizer=self.optimizer,
            num_warmup_steps=self.config.warmup_steps,
            num_training_steps=self.config.max_steps,
        )
        self.optimizer.zero_grad(set_to_none=True)

    def _sample_timesteps(self, batch_size: int) -> torch.Tensor:
        if self.config.uniform_sampling:
            indices = torch.randint(0, self.num_train_timesteps, (batch_size,), device="cpu")
        else:
            u = compute_density_for_timestep_sampling(
                weighting_scheme=self.config.weighting_scheme,
                batch_size=batch_size,
                generator=None,
                logit_mean=self.config.logit_mean,
                logit_std=self.config.logit_std,
                mode_scale=self.config.mode_scale,
            )
            indices = torch.clamp(
                (u * self.num_train_timesteps).long(),
                0,
                self.num_train_timesteps - 1,
            )
        return self.noise_schedule.timesteps[indices].to(self.device)

    @staticmethod
    def _tensor_to_video_list(tensor: torch.Tensor):
        return [item for item in tensor]

    @staticmethod
    def _tensor_to_text_list(tensor: torch.Tensor):
        return [item for item in tensor]

    def _expected_control_channels(self) -> int:
        if self.config.control_type == "cam":
            control_dim = 6
        elif self.config.control_type == "act":
            control_dim = 7
        else:
            raise ValueError(f"Unsupported control_type: {self.config.control_type}")
        return control_dim * int(self.wan_config.vae_stride[1]) * int(self.wan_config.vae_stride[2])

    def _num_chunks_from_shape(self, latents: torch.Tensor) -> int:
        """Derive the per-item chunk count N from the full-clip latent frames.

        The cache emits a whole clip of ``L = N * chunk_size`` latent frames
        (seed-less); N is variable per video. All N chunks are targets."""
        chunk = int(self.geometry.chunk_size)
        L = int(latents.shape[2])
        if L % chunk != 0 or L < chunk:
            raise ValueError(
                f"latents temporal length {L} must be a positive multiple of "
                f"chunk_size {chunk} (seed-less full-clip layout)"
            )
        return L // chunk

    def _validate_batch_shapes(
        self,
        latents: torch.Tensor,
        cond_y: torch.Tensor,
        control_tensor: torch.Tensor,
        height: int,
        width: int,
    ) -> None:
        L = int(latents.shape[2])
        if latents.ndim != 5 or latents.shape[0] != 1 or latents.shape[1] != 16:
            raise ValueError(
                f"latents must be [1, 16, N*chunk, H, W], got {tuple(latents.shape)}"
            )
        if cond_y.ndim != 5 or cond_y.shape[1] != 20 or cond_y.shape[2] != L:
            raise ValueError(
                f"cond_y must be [1, 20, {L}, H, W] aligned with latents; "
                f"got {tuple(cond_y.shape)}"
            )
        expected_control = (1, self._expected_control_channels(), L, height, width)
        if control_tensor.ndim != 5 or tuple(control_tensor.shape) != expected_control:
            raise ValueError(
                f"control_tensor must be {expected_control}, got {tuple(control_tensor.shape)}"
            )

    def _masked_flow_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        sigmas: torch.Tensor,
    ) -> torch.Tensor:
        weighting = compute_loss_weighting_for_sd3(
            weighting_scheme=self.config.weighting_scheme,
            sigmas=sigmas,
        ).float()
        per_element = nn.functional.mse_loss(pred.float(), target.float(), reduction="none")
        return (per_element * weighting).mean()

    def _sync_for_sp(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.sp_enabled:
            return sync_tensor_for_sp(tensor, self.sp_group)
        return tensor

    def _xnow_sigma(self) -> float:
        if self.config.xnow_gate_anneal_steps <= 0:
            return float(self.config.xnow_gate_end)
        progress = min(self.global_step / self.config.xnow_gate_anneal_steps, 1.0)
        return float(
            self.config.xnow_gate_start
            + progress * (self.config.xnow_gate_end - self.config.xnow_gate_start)
        )

    def _build_xnow_gate(self, extra, memory_sets, num_views, num_chunks, chunk,
                         window_chunks, sink_chunks, latent_shape):
        """Build the training and self-resample cross-view gates from camera geometry."""
        self._xnow_rollout = None
        if not self.config.xnow_gate:
            return None
        sigma = self._xnow_sigma()
        if sigma <= 0.0:
            return None
        required = {"gate_tgt_abs", "gate_tgt_K4", "gate_video_hw"}
        if not required.issubset(extra):
            missing = ", ".join(sorted(required - set(extra)))
            raise ValueError(f"xnow_gate requires cache geometry: missing {missing}")

        latent_h, latent_w = (int(value) for value in latent_shape[-2:])
        patch_h, patch_w = self.wan_config.patch_size[1:]
        grid_hw = (latent_h // patch_h, latent_w // patch_w)
        target_poses = extra["gate_tgt_abs"].to("cpu", torch.float64).numpy()
        target_intrinsics = extra["gate_tgt_K4"].to("cpu", torch.float64).numpy()
        video_hw = extra["gate_video_hw"].to("cpu").tolist()
        source_poses = extra.get("gate_src_abs")
        source_intrinsics = extra.get("gate_src_K4")
        if source_poses is not None:
            source_poses = source_poses.to("cpu", torch.float64).numpy()
            source_intrinsics = source_intrinsics.to("cpu", torch.float64).numpy()

        window = int(window_chunks)
        sink = max(0, int(sink_chunks))
        train_gates = []
        rollout_gates = {}
        for view in range(num_views):
            for position in range(num_chunks):
                clean_poses = []
                clean_intrinsics = []
                if source_poses is not None:
                    clean_poses.extend(source_poses)
                    clean_intrinsics.extend([source_intrinsics] * len(source_poses))

                own_chunks = list(range(min(sink, position)))
                if window > 0:
                    own_chunks.extend(
                        index for index in range(max(0, position - window), position)
                        if index not in own_chunks
                    )
                else:
                    own_chunks.extend(index for index in range(position) if index not in own_chunks)
                for history_chunk in own_chunks:
                    clean_poses.extend(target_poses[view][history_chunk * chunk:(history_chunk + 1) * chunk])
                    clean_intrinsics.extend([target_intrinsics[view]] * chunk)

                current_target = target_poses[view][position * chunk:(position + 1) * chunk]
                peer_poses = []
                peer_intrinsics = []
                for peer in range(num_views):
                    if peer == view:
                        continue
                    peer_poses.extend(target_poses[peer][position * chunk:(position + 1) * chunk])
                    peer_intrinsics.extend([target_intrinsics[peer]] * chunk)

                rollout_gates[(view, position)] = token_xnow_gate(
                    current_target,
                    target_intrinsics[view],
                    clean_poses,
                    clean_intrinsics,
                    video_hw,
                    grid_hw,
                    sigma=sigma,
                    peer_c2ws=peer_poses,
                    peer_intrinsics=peer_intrinsics,
                ).reshape(-1)

                memory_poses = list(clean_poses)
                memory_intrinsics = list(clean_intrinsics)
                for memory_view, memory_chunk in memory_sets[position][view]:
                    memory_poses.extend(
                        target_poses[memory_view][memory_chunk * chunk:(memory_chunk + 1) * chunk]
                    )
                    memory_intrinsics.extend([target_intrinsics[memory_view]] * chunk)
                train_gates.append(
                    token_xnow_gate(
                        current_target,
                        target_intrinsics[view],
                        memory_poses,
                        memory_intrinsics,
                        video_hw,
                        grid_hw,
                        sigma=sigma,
                        peer_c2ws=peer_poses,
                        peer_intrinsics=peer_intrinsics,
                    )
                )

        self._xnow_rollout = {
            key: torch.as_tensor(value, device=self.device, dtype=torch.float32)
            for key, value in rollout_gates.items()
        }
        return torch.as_tensor(
            np.concatenate([gate.reshape(-1) for gate in train_gates]),
            device=self.device,
            dtype=torch.float32,
        )

    def _rollout_layout(self, chunk: int, position: int, num_views: int) -> dict:
        layout = {"chunk_size": chunk, "tgt_abs_chunks": [position + 1]}
        gates = getattr(self, "_xnow_rollout", None)
        if gates:
            per_view = [gates.get((view, position)) for view in range(num_views)]
            if all(gate is not None for gate in per_view):
                layout["xnow_gate"] = torch.cat(per_view, dim=0)
        return layout

    def _assemble_history_input(self, latent: torch.Tensor, cond_y: torch.Tensor) -> torch.Tensor:
        """Build the 36-ch history input ``[latent(16) | mask(4) | cond_latent(16)]``.

        The trailing 20 channels are the cached i2v cond_y stream
        (``[build_i2v_mask | VAE([first_frame, black×rest])]`` — mask=1 only at the
        true first frame, ``VAE(black)`` elsewhere), byte-identical to the
        Stage-1 base. ``latent`` is the (possibly corrupted / self-resampled)
        clean history latent occupying the front 16 channels; ``cond_y`` supplies
        the trailing mask+cond_latent, so committed history frames read exactly as
        the base's cached frames do (only the first frame is an i2v anchor).
        """
        return torch.cat([latent, cond_y], dim=1).contiguous()

    def prepare_batch(self, batch):
        return self._prepare_batch_ar(batch)

    def _prepare_batch_ar(self, batch):
        """Seed-less chunk-by-chunk batch prep (MaineCoon doubled-sequence, all-N).

        Full clip = [chunk_0 | ... | chunk_{N-1}] (N*chunk frames; N derived from
        the clip length, variable per item). ALL N chunks are prediction targets
        (parallel). Builds:
          - history half latents (clean) = the N chunks (16-ch, N*chunk frames)
          - target latents = the N chunks, noised at per-chunk levels tau_c
        train_step feeds the doubled sequence [history || target] with a
        block-causal mask (model ar_history path) and predicts all N chunks.
        Frame 0 is the first frame of chunk 0; its i2v anchor is delivered only
        through cond_y (mask=1 at frame 0), never as a clean seed latent.
        """
        latents, cond_y, text_emb_unique, control_tensor, chunk_text_idx, a_b_emb, a_g_emb, *rest = batch
        extra = rest[0] if rest else {}
        if latents.ndim != 5:
            raise ValueError(
                "Stage-1 AR expects full-clip latents [1, C, N*chunk, H, W], "
                f"got shape {tuple(latents.shape)}"
            )
        batch_size, _, _, height, width = latents.shape
        if batch_size != 1:
            raise ValueError(f"Stage-1 AR full-clip trains one video per item (B=1), got B={batch_size}")
        self._validate_batch_shapes(latents, cond_y, control_tensor, height, width)
        latents = latents.contiguous()

        chunk = int(self.geometry.chunk_size)
        num_chunks = self._num_chunks_from_shape(latents)
        _, p_h, p_w = tuple(self.wan_config.patch_size)
        gh = -(-height // p_h)
        gw = -(-width // p_w)
        ftok = gh * gw

        if not extra or "latents_tgt" not in extra:
            raise ValueError("PaperA batches require a packed multi-view target")
        target_latents_by_view = extra["latents_tgt"].to(self.device)
        target_conditions_by_view = extra["cond_tgt"].to(self.device)
        target_controls_by_view = extra["control_tgt"].to(self.device)
        num_views = int(target_latents_by_view.shape[0])
        if num_views < 1:
            raise ValueError(
                f"latents_tgt must have at least one view, got {tuple(target_latents_by_view.shape)}"
            )
        latents = torch.cat(
            [target_latents_by_view[view : view + 1] for view in range(num_views)], dim=2
        ).contiguous()
        cond_y = torch.cat(
            [target_conditions_by_view[view : view + 1] for view in range(num_views)], dim=2
        ).contiguous()
        control_tensor = torch.cat(
            [target_controls_by_view[view : view + 1] for view in range(num_views)], dim=2
        ).contiguous()

        # Seed-less: the whole clip is BOTH the clean history half and (noised) the
        # target half. No seed carve-out or left padding is used.
        history_clean = latents.contiguous()
        cond_y = cond_y.contiguous()
        target_latents = latents

        # Equal-time target chunks share their timestep across views.
        taus = self._sample_timesteps(batch_size * num_chunks).view(batch_size, num_chunks)
        taus_full = taus.repeat(1, num_views)  # [B, k*num_chunks]
        noise = torch.randn_like(target_latents)
        if self.sp_enabled:
            taus = sync_tensor_for_sp(taus, self.sp_group)
            taus_full = taus.repeat(1, num_views)
            noise = sync_tensor_for_sp(noise, self.sp_group)

        total_chunks = num_views * num_chunks
        sigmas_chunk = get_sigmas(
            self.noise_schedule,
            target_latents.device,
            taus_full.reshape(-1),
            n_dim=1,
            dtype=target_latents.dtype,
        ).view(batch_size, total_chunks)
        sigma_frame = sigmas_chunk.repeat_interleave(chunk, dim=1).view(
            batch_size, 1, total_chunks * chunk, 1, 1
        )
        noised = (1.0 - sigma_frame) * target_latents + sigma_frame * noise

        # Per-token target timesteps [B, target_tokens] (frame-major, ftok/frame).
        tau_frame = taus_full.repeat_interleave(chunk, dim=1)  # [B, k*N*chunk]
        t_tokens = tau_frame.repeat_interleave(ftok, dim=1).contiguous()

        # History 36-ch = [clean history latent(16) | cached cond_y(20)]; cond_y
        # supplies mask=1 only at frame 0 (the i2v anchor) and VAE([f0,black])
        # elsewhere, byte-identical to the base's cached-frame layout. The target
        # half reuses the SAME cond_y (fed via `y=`).
        hist_cond = cond_y
        hist_ctrl_tensor = control_tensor
        if num_chunks < 2:
            raise ValueError("rolling PaperA training needs clips with at least two chunks")
        # Keep only the previous chunk for each target position. The final history
        # chunk is never read and is intentionally omitted.
        frames_per_view = num_chunks * chunk
        kept_frames = (num_chunks - 1) * chunk
        keep_idx = torch.cat(
            [
                torch.arange(
                    view * frames_per_view,
                    view * frames_per_view + kept_frames,
                    device=latents.device,
                )
                for view in range(num_views)
            ]
        )
        sr_full_clean = latents
        history_clean = latents.index_select(2, keep_idx).contiguous()
        hist_cond = cond_y.index_select(2, keep_idx).contiguous()
        hist_ctrl_tensor = control_tensor.index_select(2, keep_idx).contiguous()
        history_short_full = self._assemble_history_input(history_clean, hist_cond)
        cond_y_target = cond_y

        # Each rolling history chunk uses the previous-slot RoPE phase; each target
        # chunk uses the current-slot phase. Indices restart for every target view.
        hist_frames = kept_frames
        tgt_frames = num_chunks * chunk
        per_view_hist = torch.arange(0, hist_frames, device=self.device) % chunk
        per_view_tgt = torch.arange(0, tgt_frames, device=self.device) % chunk + chunk
        hist_idx = (
            per_view_hist.repeat(num_views)
            .unsqueeze(0)
            .expand(batch_size, -1)
            .contiguous()
        )
        tgt_idx = (
            per_view_tgt.repeat(num_views)
            .unsqueeze(0)
            .expand(batch_size, -1)
            .contiguous()
        )

        control_dict = {
            "c2ws_plucker_emb": control_tensor.chunk(control_tensor.shape[0], dim=0),
            "c2ws_plucker_emb_history_short": hist_ctrl_tensor.chunk(hist_ctrl_tensor.shape[0], dim=0),
        }

        if not extra or "latents_source" not in extra:
            raise ValueError("PaperA batches require the shared source-image stream")
        src_lat = extra["latents_source"].to(self.device).contiguous()
        src_cond = extra["cond_source"].to(self.device).contiguous()
        src_ctrl = extra["control_source"].to(self.device).contiguous()
        latents_source = torch.cat([src_lat, src_cond], dim=1).contiguous()
        src_frames = int(latents_source.shape[2])
        indices_source = (
            torch.arange(0, src_frames, device=self.device)
            .unsqueeze(0)
            .expand(batch_size, -1)
            .contiguous()
        )
        control_dict["c2ws_plucker_emb_source"] = src_ctrl

        # Sequence length covers the shared source, history, and current targets.
        seq_len = ar_seq_len(
            src_frames + num_views * hist_frames + num_views * num_chunks * chunk,
            height, width, self.wan_config.patch_size,
        )

        win_chunks = 1
        sink_chunks = 0

        # Each target view reads only its own top-1 retrieved history entry. The
        # cache enforces the anti-leak bound ``candidate_chunk <= query_chunk - 2``.
        if "mem_pairs" not in extra:
            raise ValueError("PaperA batches require per-view P-Mem pairs")
        mp = extra["mem_pairs"].to("cpu", torch.long)
        if mp.ndim != 4 or mp.shape[0] != num_chunks or mp.shape[1] != num_views or mp.shape[3] != 2:
            raise ValueError(
                f"mem_pairs must be [num_chunks, num_views, r_per_view, 2], got {tuple(mp.shape)}"
            )
        mem_hist_sets = [
            [[(int(view), int(position)) for view, position in mp[p, v].tolist() if view >= 0]
             for v in range(num_views)]
            for p in range(num_chunks)
        ]

        ar_layout = {
            "chunk_size": chunk,
            "tgt_abs_chunks": list(range(1, num_chunks + 1)),
            "window_chunks": win_chunks,
            "sink_chunks": sink_chunks,
        }
        ar_layout["mem_hist_sets"] = mem_hist_sets
        gate = self._build_xnow_gate(
            extra,
            mem_hist_sets,
            num_views,
            num_chunks,
            chunk,
            win_chunks,
            sink_chunks,
            history_clean.shape,
        )
        if gate is not None:
            ar_layout["xnow_gate"] = gate

        return {
            "latents_history_short": history_short_full,
            "cond_y": cond_y_target,
            "history_clean_16": history_clean,
            "cond_y_full": cond_y,
            "text_emb_unique": text_emb_unique,
            "chunk_text_idx": chunk_text_idx,
            "a_b_emb": a_b_emb,
            "a_g_emb": a_g_emb,
            "seq_len": seq_len,
            "control_dict": control_dict,
            "timesteps": t_tokens,
            "noised": noised,
            "sigmas": sigma_frame,
            "target": noise - target_latents,
            "indices_hidden_states": tgt_idx,
            "indices_latents_history_short": hist_idx,
            "latents_source": latents_source,
            "indices_source": indices_source,
            "num_views": num_views,
            "sr_full_clean_16": sr_full_clean,
            "sr_hist_cond": hist_cond,
            "ar_layout": ar_layout,
        }

    @torch.no_grad()
    def _self_resample_history_kv(self, context, a_b_emb, a_g_emb, history_clean_16, cond_y_full,
                                  chunk_text_idx, lat_h, lat_w, num_views, control_dict,
                                  latents_source, indices_source):
        """Run the multi-view rolling self-resample used by PaperA training.

        Per chunk j (all k views JOINTLY): noise the clean chunk j at a clean-biased
        sigma_h, condition on the accumulated DEGRADED history (this view's own x̂
        chunks <j) + the source + the co-denoised k-view chunk (lockstep via
        _mv_self_groups), then denoise to sigma=0 with ``sr_rollout_steps`` Euler
        steps -> x̂_j. Returns the degraded history laid out exactly like the clean
        history the caller will replace ([v0 chunks | v1 chunks | ...]).

        Multi-node safety: sigma_h/eps SP-synced (identical x̂ across an SP group);
        length-synced to world-max chunk count with discarded padding forwards so all
        FSDP ranks issue identical all-gather counts.
        """
        cfg = self.config
        chunk = int(self.geometry.chunk_size)
        k = int(num_views)
        device = history_clean_16.device
        model_dtype = torch.bfloat16 if cfg.dtype == "bf16" else torch.float32
        total = int(history_clean_16.shape[2])
        if total % (k * chunk) != 0:
            raise ValueError(f"history frames {total} not divisible by k*chunk={k*chunk}")
        N = total // (k * chunk)                       # per-view chunk count
        per = N * chunk
        clean_v = [history_clean_16[:, :, v * per:(v + 1) * per] for v in range(k)]  # [1,16,N*chunk,h,w]
        condy_v = [cond_y_full[:, :, v * per:(v + 1) * per] for v in range(k)]        # [1,20,N*chunk,h,w]
        tgt_ctrl = control_dict["c2ws_plucker_emb"]                                    # [1,C,k*N*chunk,h,w]
        if isinstance(tgt_ctrl, (list, tuple)):                                        # _prepare_batch_ar stores it .chunk()'d
            tgt_ctrl = torch.cat(list(tgt_ctrl), dim=0)
        ctrl_v = [tgt_ctrl[:, :, v * per:(v + 1) * per] for v in range(k)]            # [1,C,N*chunk,h,w]
        src_ctrl = control_dict["c2ws_plucker_emb_source"]                            # [1,C,src_frames,h,w]
        if isinstance(src_ctrl, (list, tuple)):
            src_ctrl = torch.cat(list(src_ctrl), dim=0)
        src_frames = int(latents_source.shape[2])
        # Clean-biased history sigma (SP-synced so an SP group resamples identically).
        sigma_sample = self._sync_for_sp(sample_history_sigma(cfg.sr_shift, device))
        t_h, sigma_h = resolve_history_timestep(self.noise_schedule, float(sigma_sample.item()), device)
        t_h = t_h.to(device).reshape(1).contiguous()

        # Use a few uniformly spaced levels from
        # sigma_h down to 0, each SNAPPED to the schedule grid so the timestep the
        # model is conditioned on is the exact noise level the latent carries. A
        # single Euler step returns the posterior mean E[x0|z], which is
        # systematically blurry and (at high rho) teaches the model a "videos decay
        # over time" prior -> progressive inference blur; a few-step chain tracks
        # the ODE and lands the degraded history near the real inference product.
        n_roll = max(1, int(getattr(cfg, "sr_rollout_steps", 1)))
        roll_levels = [(t_h, sigma_h)]                  # [(timestep, sigma), ...] non-increasing
        for i in range(1, n_roll):
            t_i, sig_i = resolve_history_timestep(
                self.noise_schedule, sigma_h * (n_roll - i) / n_roll, device
            )
            # ALWAYS exactly n_roll levels, even where a coarse grid region snaps two
            # levels onto the same sigma (that step is then a no-op): the forward
            # count must not depend on sigma_h, which is only SP-group-synced — DP
            # replicas drawing different sigma_h would then issue different numbers
            # of FSDP all-gathers and hang.
            roll_levels.append((t_i.to(device).reshape(1).contiguous(), sig_i))

        # Only chunks [0, N-1) ever reach the model as conditioning
        # history (the rolling drop removes each view's last chunk), so rolling the
        # last chunk is pure waste — 1/N of an sr_rollout_steps-times-more-expensive
        # rollout, and 1/2 of it at N=2.
        roll_c = N - 1
        if roll_c < 1:
            raise ValueError(f"self-resample needs >= 2 chunks per view in rolling mode, got N={N}")

        # World-max chunk count (length-sync): pad short ranks with discarded forwards.
        global_C = roll_c
        if self.world_size > 1:
            gC = torch.tensor([roll_c], device=device, dtype=torch.long)
            dist.all_reduce(gC, op=dist.ReduceOp.MAX)
            global_C = int(gC.item())

        # All chunks share one prompt in this recipe (the assembler emits an
        # all-zero chunk_text_idx against a single unique text embedding), which is
        # what lets the rollout hand the model a fresh per-chunk id vector below.
        if int(chunk_text_idx.max().item()) != 0:
            raise ValueError(
                "self-resample rollout assumes a single shared prompt (chunk_text_idx all zero)"
            )
        committed_v = [[] for _ in range(k)]           # degraded x̂ chunks per view
        for j in range(global_C):
            is_pad = j >= roll_c
            jj = j if not is_pad else (roll_c - 1)
            s, e = jj * chunk, (jj + 1) * chunk
            # noise the clean chunk jj (k views), SP-synced eps
            z_v = []
            for v in range(k):
                cj = clean_v[v][:, :, s:e]
                eps = self._sync_for_sp(torch.randn_like(cj))
                z_v.append((1.0 - sigma_h) * cj + sigma_h * eps)
            z = torch.cat(z_v, dim=2)                  # [1,16,k*chunk,h,w]
            # PaperA always uses the immediately preceding generated chunk as
            # history. The source image remains in its own fixed stream.
            kept = [jj - 1] if jj > 0 else []
            slots, jj_eff = ([0] if kept else []), 1
            hist36 = None
            idx_hist = None
            ctrl_hist = None
            hist_frames = 0
            if kept:
                hist_frames = len(kept) * chunk
                hs, ch_ = [], []
                for v in range(k):
                    hv = torch.cat([committed_v[v][kc] for kc in kept], dim=2)  # [1,16,|kept|*chunk,h,w]
                    cy = torch.cat([condy_v[v][:, :, kc * chunk:(kc + 1) * chunk] for kc in kept], dim=2)
                    hs.append(torch.cat([hv[0], cy[0]], dim=0))                 # [36,|kept|*chunk,h,w]
                    ch_.append(torch.cat([ctrl_v[v][:, :, kc * chunk:(kc + 1) * chunk] for kc in kept], dim=2))
                hist36 = torch.cat(hs, dim=1)[None]                       # [1,36,k*|kept|*chunk,h,w]
                ctrl_hist = torch.cat(ch_, dim=2)                         # [1,C,k*|kept|*chunk,h,w]
                abs_h = torch.cat([torch.arange(sl * chunk, (sl + 1) * chunk, device=device) for sl in slots])
                idx_hist = abs_h.repeat(k)[None]
            idx_tgt = torch.arange(jj_eff * chunk, (jj_eff + 1) * chunk, device=device).repeat(k)[None]
            # Cross-attn prompt ids: the model reads moba_cti[:p+1] for a target chunk
            # at abs index p == jj_eff, so this MUST hold jj_eff+1 entries. A shorter
            # tensor slices short and silently under-fills the cross-attn KV (the
            # reshape to (p+2)*text_len then fails), so size it from jj_eff — NOT from
            # the rollout's chunk count, which is smaller than jj_eff+1 at N=2.
            cti_slots = torch.zeros(jj_eff + 1, dtype=torch.long, device=device)
            ctrl_tgt = torch.cat([ctrl_v[v][:, :, s:e] for v in range(k)], dim=2)
            cond_tgt = torch.cat([condy_v[v][0, :, s:e] for v in range(k)], dim=1)   # [20,k*chunk,h,w]
            cd = {"c2ws_plucker_emb": ctrl_tgt, "c2ws_plucker_emb_source": src_ctrl}
            if ctrl_hist is not None:
                cd["c2ws_plucker_emb_history_short"] = ctrl_hist
            seq_len = ar_seq_len(src_frames + k * hist_frames + k * chunk, lat_h, lat_w, self.wan_config.patch_size)
            fwd_kwargs = dict(
                context=context, seq_len=seq_len,
                y=[cond_tgt], dit_cond_dict=cd,
                indices_hidden_states=idx_tgt,
                indices_latents_history_short=idx_hist,
                latents_history_short=hist36,
                history_timestep=0.0, ar_history=True,
                ar_layout=self._rollout_layout(chunk, jj_eff, k),
                chunk_text_idx=cti_slots, a_b_emb=a_b_emb, a_g_emb=a_g_emb,
                enable_bi=False, latents_source=latents_source, indices_source=indices_source,
                num_views=k,
            )
            # Euler chain down the ladder, accumulated in fp32; a 1-level ladder is
            # the original single Euler step x̂ = z - sigma_h * v.
            zc = z[0].float()
            for li, (t_i, sig_i) in enumerate(roll_levels):
                sig_next = roll_levels[li + 1][1] if li + 1 < len(roll_levels) else 0.0
                with torch.autocast(device_type, dtype=model_dtype, enabled=(device_type != "cpu" and model_dtype == torch.bfloat16)):
                    out = self.single_dit.model(x=[zc.to(z.dtype)], t=t_i, **fwd_kwargs)
                    v_pred = out[0][0].float()             # [16,k*chunk,h,w]
                zc = zc - (sig_i - sig_next) * v_pred
            x0 = zc.to(z.dtype)                            # [16,k*chunk,h,w] at sigma=0
            if not is_pad:
                for v in range(k):
                    committed_v[v].append(x0[:, v * chunk:(v + 1) * chunk].unsqueeze(0))  # [1,16,chunk,h,w]

        degraded = torch.cat([torch.cat(committed_v[v], dim=2) for v in range(k)], dim=2).contiguous()
        accel_empty_cache()
        return degraded

    def train_step(self, batch):
        inputs = self.prepare_batch(batch)
        model_dtype = torch.bfloat16 if self.config.dtype == "bf16" else torch.float32

        context = self._tensor_to_text_list(inputs["text_emb_unique"])
        chunk_text_idx = inputs["chunk_text_idx"]

        # With probability rho, replace clean history with a stop-gradient rollout.
        rho = rho_curriculum(self.global_step, self.config)
        use_self_resample = False
        if self.config.self_resample:
            # All ranks take the same branch because it changes FSDP collective counts.
            draw_t = torch.rand(1, device=self.device)
            if self.world_size > 1:
                dist.broadcast(draw_t, src=0)
            use_self_resample = draw_t.item() < rho

        latents_history_short = inputs["latents_history_short"]
        if use_self_resample:
            # The rollout consumes the full target layout and returns the same
            # one-chunk-per-view history layout used by the training forward.
            clean_16 = inputs["sr_full_clean_16"]
            lat_h, lat_w = clean_16.shape[-2:]
            degraded_16 = self._self_resample_history_kv(
                context,
                inputs["a_b_emb"],
                inputs["a_g_emb"],
                clean_16,
                inputs["cond_y_full"],
                chunk_text_idx,
                int(lat_h),
                int(lat_w),
                inputs.get("num_views", 1),
                inputs["control_dict"],
                inputs["latents_source"],
                inputs["indices_source"],
            )
            latents_history_short = self._assemble_history_input(
                degraded_16, inputs["sr_hist_cond"]
            )

        with torch.autocast(
            device_type=device_type,
            dtype=model_dtype,
            enabled=(device_type != "cpu" and model_dtype == torch.bfloat16),
        ):
            model_kwargs = dict(
                x=self._tensor_to_video_list(inputs["noised"]),
                t=inputs["timesteps"],
                context=context,
                seq_len=inputs["seq_len"],
                y=self._tensor_to_video_list(inputs["cond_y"]),
                dit_cond_dict=inputs["control_dict"],
                indices_hidden_states=inputs["indices_hidden_states"],
                indices_latents_history_short=inputs["indices_latents_history_short"],
                latents_history_short=latents_history_short,
                ar_history=True,
                ar_layout=inputs["ar_layout"],
                chunk_text_idx=chunk_text_idx,
                a_b_emb=inputs["a_b_emb"],
                a_g_emb=inputs["a_g_emb"],
                enable_bi=False,
                history_timestep=0.0,
                num_views=inputs.get("num_views", 1),
            )
            model_kwargs["latents_source"] = inputs["latents_source"]
            model_kwargs["indices_source"] = inputs["indices_source"]
            pred_tf_list, pred_bi_list = self.single_dit.model(**model_kwargs)
            pred_tf = torch.stack(pred_tf_list, dim=0)
            if pred_bi_list is not None:
                raise RuntimeError("PaperA training must not return a bidirectional branch")

        loss = self._masked_flow_loss(pred_tf, inputs["target"], inputs["sigmas"])

        total_loss = loss / self.config.gradient_accumulation_steps
        total_loss.backward()
        # fp32-master chain: FSDP2 computes in bf16 but keeps the sharded master
        # params and reduced grads in fp32. Recast defensively after EVERY backward
        # so (a) optimizer.step sees matching fp32 param/grad and (b) gradient
        # accumulation across ga steps adds in fp32.
        for param in self.single_dit.parameters():
            if param.grad is not None and param.grad.dtype != param.dtype:
                param.grad = param.grad.to(param.dtype)

        if (self.global_step + 1) % self.config.gradient_accumulation_steps == 0:
            # Definitive fp32 grad recast at the update boundary: FSDP2's
            # post-backward callback can (re)set .grad in orig-dtype bf16 AFTER
            # backward() returns, overwriting the per-backward recast above.
            for param in self.single_dit.parameters():
                if param.grad is not None and param.grad.dtype != param.dtype:
                    param.grad = param.grad.to(param.dtype)
            if self.config.max_grad_norm > 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.single_dit.parameters(),
                    self.config.max_grad_norm,
                )
            else:
                grad_norm = torch.tensor(0.0, device=self.device)
            grad_norm_val = grad_norm.item() if isinstance(grad_norm, torch.Tensor) else float(grad_norm)
            self.last_grad_norm = grad_norm_val
            if not getattr(self, "_logged_fp32_optimizer_chain", False):
                param_dtypes = {str(param.dtype) for param in self.single_dit.parameters()}
                grad_dtypes = {
                    str(param.grad.dtype)
                    for param in self.single_dit.parameters()
                    if param.grad is not None
                }
                if param_dtypes != {"torch.float32"} or grad_dtypes != {"torch.float32"}:
                    raise RuntimeError(
                        "fp32 optimizer chain validation failed before optimizer.step: "
                        f"params={sorted(param_dtypes)} grads={sorted(grad_dtypes)}"
                    )
            self.optimizer.step()
            if not getattr(self, "_logged_fp32_optimizer_chain", False):
                state_dtypes = {
                    str(value.dtype)
                    for state in self.optimizer.state.values()
                    for key, value in state.items()
                    if key in {"exp_avg", "exp_avg_sq"} and torch.is_tensor(value)
                }
                if state_dtypes != {"torch.float32"}:
                    raise RuntimeError(
                        "fp32 optimizer chain validation failed after optimizer.step: "
                        f"Adam states={sorted(state_dtypes)}"
                    )
                self._logged_fp32_optimizer_chain = True
                if self.is_main_process:
                    print(
                        "[train] fp32 optimizer chain verified: "
                        "master_params=float32 grads=float32 adam_states=float32",
                        flush=True,
                    )
            self.lr_scheduler.step()
            self.optimizer.zero_grad(set_to_none=True)

        mem_sets = inputs["ar_layout"].get("mem_hist_sets", [])
        pmem_positions = sum(
            any(entries for entries in per_view) for per_view in mem_sets
        )
        pmem_views = sum(
            bool(entries) for per_view in mem_sets for entries in per_view
        )
        pmem_entries = sum(
            len(entries) for per_view in mem_sets for entries in per_view
        )
        return {
            "loss": loss.item(),
            "grad_norm": self.last_grad_norm,
            "rho": rho,
            "self_resample": float(use_self_resample),
            "num_views": int(inputs.get("num_views", 1)),
            "view_chunks": int(
                inputs["target"].shape[2]
                // (max(1, int(inputs.get("num_views", 1))) * self.geometry.chunk_size)
            ),
            "pmem_positions": pmem_positions,
            "pmem_views": pmem_views,
            "pmem_entries": pmem_entries,
        }

    def save_checkpoint(self, step: int) -> None:
        """Gather the FSDP model into the portable ``model_full.pt`` format."""
        checkpoint_dir = os.path.join(self.config.output_dir, f"checkpoint-{step}")
        if self.is_main_process:
            os.makedirs(checkpoint_dir, exist_ok=True)

        gc.collect()
        accel_empty_cache()
        accel_synchronize()
        if self.world_size > 1:
            dist.barrier()

        state: dict[str, torch.Tensor] = {}
        for name, parameter in self.checkpoint_module.named_parameters():
            tensor = parameter.detach()
            if hasattr(tensor, "full_tensor"):
                tensor = tensor.full_tensor()
            if self.is_main_process:
                key = name.removeprefix("model.")
                state[key] = tensor.to(torch.bfloat16).cpu()
        for name, buffer in self.checkpoint_module.named_buffers():
            tensor = buffer.detach()
            if hasattr(tensor, "full_tensor"):
                tensor = tensor.full_tensor()
            if self.is_main_process:
                state[name.removeprefix("model.")] = tensor.cpu()

        if self.is_main_process:
            output_path = os.path.join(checkpoint_dir, "model_full.pt")
            torch.save(state, output_path)
            print(f"[train] saved {len(state)} tensors to {output_path}", flush=True)
        accel_synchronize()
        if self.world_size > 1:
            dist.barrier()

    def train(self, dataloader):
        if self.is_main_process:
            print(
                f"[train] start max_steps={self.config.max_steps} lr={self.config.learning_rate} "
                f"control_type={self.config.control_type} sp_size={self.config.sp_size} "
                "shared_image=true "
                "self_resample=on "
                f"flow_shift={self.config.train_timestep_shift}"
            )

        sp_dataloader = sequence_parallel_batches(dataloader, self.device, self.config.sp_size)
        sp_iterator = iter(sp_dataloader)
        self.single_dit.train()

        accel_empty_cache()

        while self.global_step < self.config.max_steps:
            batch = next(sp_iterator)
            step_start = time.perf_counter()
            metrics = self.train_step(batch)
            duration = time.perf_counter() - step_start
            del batch

            if self.is_main_process:
                print(
                    f"[train] step={self.global_step}/{self.config.max_steps} "
                    f"time={duration:.2f}s "
                    f"loss={metrics['loss']:.6f} "
                    f"grad={metrics['grad_norm']:.4f} "
                    f"rho={metrics['rho']:.3f} sr={metrics['self_resample']:.0f} "
                    f"K={metrics['num_views']} view_chunks={metrics['view_chunks']} "
                    f"pmem=positions:{metrics['pmem_positions']} "
                    f"views:{metrics['pmem_views']} entries:{metrics['pmem_entries']} "
                    f"mem={device_module.memory_allocated() / 2**30:.1f}/"
                    f"{device_module.memory_reserved() / 2**30:.1f}G",
                    flush=True,
                )

            checkpoint_step = self.global_step + 1
            if checkpoint_step % self.config.save_interval == 0:
                self.save_checkpoint(checkpoint_step)
                if self.world_size > 1:
                    dist.barrier()

            self.global_step += 1
            del metrics

        if self.config.save_final_checkpoint and self.global_step % self.config.save_interval != 0:
            self.save_checkpoint(self.global_step)
        if self.is_main_process:
            print("[train] finished")

        if self.world_size > 1:
            dist.barrier()
            dist.destroy_process_group()



def create_dataloader(config: Stage1ARTrainingConfig):
    if not config.clip_cache_dir:
        raise ValueError("--clip_cache_dir is required; PaperA trains from the packed clip cache")
    from wan.dataset.clip_cache_assembler import ClipCacheConsumer

    return ClipCacheConsumer(
        config.clip_cache_dir,
        target_chunks=config.clip_cache_target_chunks,
        k_min=config.clip_cache_k_min,
        k_max=config.clip_cache_k_max,
        max_total_view_chunks=config.clip_cache_max_total_view_chunks,
        cfg_rate=config.clip_cache_cfg_rate,
        pmem=config.pmem,
        pmem_r=config.pmem_r,
        scene_filter=config.scene_filter,
        chunk_size=config.chunk_size,
    )


def main():
    parser = argparse.ArgumentParser(description="Train PaperA from a packed multi-view clip cache")
    parser.add_argument("--pretrained_model_root", required=True)
    parser.add_argument("--init_model_pt", required=True,
                        help="Stage-1 rolling base checkpoint (model_full.pt)")
    parser.add_argument("--clip_cache_dir", required=True,
                        help="packed cache produced by build_clip_cache.py")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--max_steps", type=int, default=4000)
    parser.add_argument("--save_interval", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sp_size", type=int, default=8)
    parser.add_argument("--dp_replicate", type=int, default=1)
    args = parser.parse_args()

    config = Stage1ARTrainingConfig(
        pretrained_model_root=args.pretrained_model_root,
        control_type="cam",
        max_steps=args.max_steps,
        output_dir=args.output_dir,
        save_interval=args.save_interval,
        seed=args.seed,
        sp_size=args.sp_size,
        dp_replicate=args.dp_replicate,
        clip_cache_dir=args.clip_cache_dir,
        init_model_pt=args.init_model_pt,
    )

    trainer = LingbotStage1ARTrainer(config)
    dataloader = create_dataloader(config)
    trainer.train(dataloader)


if __name__ == "__main__":
    main()
