"""Assemble one PaperA multi-view training item from a packed clip cache."""
from __future__ import annotations

import glob
import os
import random

import numpy as np
import torch
from einops import rearrange

from wan.configs import WAN_CONFIGS
from wan.dataset.pmem_retrieval import retrieve_gt_mem_pairs
from wan.utils.cam_utils import compute_relative_poses, get_plucker_embeddings, interpolate_camera_poses
from wan.utils.infer_data import broadcast_intrinsics_to_length
from wan.utils.stage1_ar_geometry import build_i2v_mask


def _intrinsics4(intrinsics: torch.Tensor) -> list[float]:
    values = torch.as_tensor(intrinsics, dtype=torch.float64).reshape(-1)
    if values.numel() == 9:
        return [float(values[0]), float(values[4]), float(values[2]), float(values[5])]
    if values.numel() != 4:
        raise ValueError(f"intrinsics must be [fx, fy, cx, cy] or 3x3, got {tuple(values.shape)}")
    return [float(value) for value in values]


def _interpolate_poses(poses: torch.Tensor, rgb_indices: torch.Tensor) -> torch.Tensor:
    source = poses.detach().cpu().numpy()
    return interpolate_camera_poses(
        src_indices=np.arange(len(source), dtype=np.float64),
        src_rot_mat=source[:, :3, :3],
        src_trans_vec=source[:, :3, 3],
        tgt_indices=rgb_indices.detach().cpu().numpy().astype(np.float64),
    )


def _camera_control(
    poses: torch.Tensor,
    intrinsics: torch.Tensor,
    video_hw: tuple[int, int],
    num_chunks: int,
    latent_hw: tuple[int, int],
    chunk_size: int,
    temporal_stride: int,
    dtype: torch.dtype,
    reference_pose: torch.Tensor,
    translation_scale: float,
    absolute_poses: torch.Tensor,
) -> torch.Tensor:
    height, width = video_hw
    latent_h, latent_w = latent_hw
    latent_frames = num_chunks * chunk_size
    relative = compute_relative_poses(
        absolute_poses,
        framewise=False,
        ref_pose=reference_pose,
        trans_scale=translation_scale,
        normalize_trans=False,
    )
    intrinsics = broadcast_intrinsics_to_length(intrinsics, latent_frames)
    plucker = get_plucker_embeddings(relative, intrinsics, height, width, only_rays_d=False)
    plucker = rearrange(
        plucker,
        "f (h ph) (w pw) c -> (f h w) (c ph pw)",
        ph=height // latent_h,
        pw=width // latent_w,
    )[None]
    control = rearrange(
        plucker,
        "b (f h w) c -> b c f h w",
        f=latent_frames,
        h=latent_h,
        w=latent_w,
    )
    return control[0].to(dtype=dtype).contiguous()


def _conditioning_stream(
    condition_latents: torch.Tensor,
    latent_hw: tuple[int, int],
    temporal_stride: int,
) -> torch.Tensor:
    latent_h, latent_w = latent_hw
    latent_frames = int(condition_latents.shape[1])
    rgb_frames = (latent_frames - 1) * temporal_stride + 1
    mask = build_i2v_mask(
        frame_num=rgb_frames,
        lat_h=latent_h,
        lat_w=latent_w,
        device=condition_latents.device,
        dtype=condition_latents.dtype,
        vae_temporal_stride=temporal_stride,
    )
    return torch.cat([mask, condition_latents], dim=0).contiguous()


class ClipCacheConsumer:
    """Infinite rank-strided iterable over shared-image PaperA examples.

    Each item selects one fixed source camera and a random number of target
    cameras in ``[k_min, k_max]``. The source contributes only its first image
    chunk; target views are jointly denoised.
    """

    def __init__(
        self,
        clip_cache_dir: str,
        *,
        target_chunks: int,
        k_min: int,
        k_max: int,
        max_total_view_chunks: int,
        cfg_rate: float = 0.0,
        pmem: bool = False,
        pmem_r: int = 1,
        scene_filter: str = "",
        chunk_size: int = 4,
    ) -> None:
        if int(chunk_size) != 4:
            raise ValueError("PaperA uses chunk_size=4")
        if not 1 <= int(k_min) <= int(k_max):
            raise ValueError(f"invalid target-view range [{k_min}, {k_max}]")
        if int(target_chunks) < 2:
            raise ValueError("target_chunks must be at least two for rolling training")
        if int(pmem_r) != 1:
            raise ValueError("PaperA uses one retrieved chunk per target view")

        self.clips_dir = os.path.join(clip_cache_dir, "clips")
        self.text_dir = os.path.join(clip_cache_dir, "text")
        self.target_chunks = int(target_chunks)
        self.k_min = int(k_min)
        self.k_max = int(k_max)
        self.max_total_view_chunks = int(max_total_view_chunks)
        self.cfg_rate = float(cfg_rate)
        self.pmem = bool(pmem)
        self.chunk_size = int(chunk_size)

        config = WAN_CONFIGS["i2v-A14B"]
        self.temporal_stride = int(config.vae_stride[0])
        self.parameter_dtype = config.param_dtype
        self.scene_paths = {
            os.path.basename(path)[:-3]: path
            for path in glob.glob(os.path.join(self.clips_dir, "*.pt"))
        }
        requested = {name.strip() for name in scene_filter.split(",") if name.strip()}
        self.scene_keys = sorted(
            key for key in self.scene_paths if not requested or key in requested
        )
        if not self.scene_keys:
            raise FileNotFoundError(f"no packed scene cache found under {self.clips_dir}")
        self.empty_embedding = torch.load(
            os.path.join(self.text_dir, "_empty.pt"), map_location="cpu"
        )
        self.rank = int(os.environ.get("RANK", "0"))
        self.world_size = int(os.environ.get("WORLD_SIZE", "1"))
        self.index = self.rank

    def __iter__(self):
        return self

    def _text_embedding(self, scene_key: str) -> torch.Tensor:
        return torch.load(os.path.join(self.text_dir, f"{scene_key}.pt"), map_location="cpu")

    def __next__(self):
        scene_key = self.scene_keys[self.index % len(self.scene_keys)]
        self.index += self.world_size
        scene = torch.load(self.scene_paths[scene_key], map_location="cpu")
        camera_names = sorted(scene)
        if len(camera_names) < self.k_min:
            raise RuntimeError(f"{scene_key} has only {len(camera_names)} cameras")

        view_count = random.randint(self.k_min, min(self.k_max, len(camera_names)))
        source = scene[camera_names[0]]
        targets = [scene[name] for name in random.sample(camera_names, view_count)]
        target_chunks = self.target_chunks
        if self.max_total_view_chunks > 0:
            target_chunks = min(target_chunks, self.max_total_view_chunks // view_count)
        if target_chunks < 2:
            raise ValueError("max_total_view_chunks leaves fewer than two chunks per view")
        num_chunks = min([target_chunks, int(source["num_chunks"])] + [int(item["num_chunks"]) for item in targets])
        if num_chunks < 2:
            raise ValueError(f"{scene_key} does not contain two full chunks in every selected view")

        latent_h, latent_w = int(source["lat_h"]), int(source["lat_w"])
        video_hw = tuple(int(value) for value in source["video_hw"])
        latent_frames = num_chunks * self.chunk_size
        rgb_indices = torch.arange(latent_frames, dtype=torch.long) * self.temporal_stride
        source_absolute = _interpolate_poses(source["poses"], rgb_indices)
        reference_pose = source_absolute[0]
        target_absolute = [_interpolate_poses(item["poses"], rgb_indices) for item in targets]

        translations = [
            compute_relative_poses(
                source_absolute, framewise=False, ref_pose=reference_pose, normalize_trans=False
            )[:, :3, 3]
        ]
        translations.extend(
            compute_relative_poses(
                poses, framewise=False, ref_pose=reference_pose, normalize_trans=False
            )[:, :3, 3]
            for poses in target_absolute
        )
        translation_scale = float(torch.norm(torch.cat(translations, dim=0), dim=-1).max().item())
        if translation_scale < 1e-4:
            translation_scale = 1.0

        target_condition = _conditioning_stream(
            source["cond_i2v_latents"][:, :latent_frames],
            (latent_h, latent_w),
            self.temporal_stride,
        )
        source_latents = source["cond_i2v_latents"][:, :self.chunk_size].contiguous()
        source_condition = target_condition[:, :self.chunk_size].contiguous()
        source_control = _camera_control(
            source["poses"],
            source["intrinsics"],
            video_hw,
            num_chunks,
            (latent_h, latent_w),
            self.chunk_size,
            self.temporal_stride,
            self.parameter_dtype,
            reference_pose,
            translation_scale,
            source_absolute,
        )[:, :self.chunk_size].contiguous()

        target_latents = []
        target_conditions = []
        target_controls = []
        for target, absolute in zip(targets, target_absolute):
            target_latents.append(target["latents_full"][:, :latent_frames].contiguous())
            target_conditions.append(target_condition.clone())
            target_controls.append(
                _camera_control(
                    target["poses"],
                    target["intrinsics"],
                    video_hw,
                    num_chunks,
                    (latent_h, latent_w),
                    self.chunk_size,
                    self.temporal_stride,
                    self.parameter_dtype,
                    reference_pose,
                    translation_scale,
                    absolute,
                )
            )
        target_latents = torch.stack(target_latents)
        target_conditions = torch.stack(target_conditions)
        target_controls = torch.stack(target_controls)

        scene_embedding = self._text_embedding(scene_key)
        prompt_embedding = self.empty_embedding if self.cfg_rate > 0 and random.random() < self.cfg_rate else scene_embedding
        extra = {
            "latents_tgt": target_latents,
            "cond_tgt": target_conditions,
            "control_tgt": target_controls,
            "latents_source": source_latents.unsqueeze(0),
            "cond_source": source_condition.unsqueeze(0),
            "control_source": source_control.unsqueeze(0),
            "gate_tgt_abs": torch.stack(target_absolute).float(),
            "gate_tgt_K4": torch.tensor(
                [_intrinsics4(target["intrinsics"]) for target in targets], dtype=torch.float32
            ),
            "gate_src_abs": source_absolute[:self.chunk_size].float(),
            "gate_src_K4": torch.tensor(_intrinsics4(source["intrinsics"]), dtype=torch.float32),
            "gate_video_hw": torch.tensor(video_hw, dtype=torch.long),
        }
        if self.pmem:
            extra["mem_pairs"] = retrieve_gt_mem_pairs(
                target_absolute,
                [target["intrinsics"] for target in targets],
                video_hw,
                num_chunks,
                lw=self.chunk_size,
                win=1,
                r_per_view=1,
            )

        chunk_text_index = torch.zeros(num_chunks, dtype=torch.long)
        return (
            target_latents[:1],
            target_conditions[:1],
            self.empty_embedding.unsqueeze(0),
            target_controls[:1],
            chunk_text_index,
            prompt_embedding.unsqueeze(0),
            prompt_embedding.unsqueeze(0),
            extra,
        )
