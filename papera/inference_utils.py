"""Camera and checkpoint-independent helpers used by PaperA inference."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass

import numpy as np
import torch
from decord import VideoReader, cpu
from einops import rearrange

from wan.utils.cam_utils import (
    compute_relative_poses,
    get_plucker_embeddings,
    interpolate_camera_poses,
)
from wan.utils.infer_data import (
    ResizeCropAspectCenter,
    broadcast_intrinsics_to_length,
    extract_spatialvid_meta,
    round_down_4n_plus_1,
)


@dataclass(frozen=True)
class CameraClip:
    """One SpatialVID camera stream after the release resize/crop transform."""

    video: torch.Tensor
    poses: torch.Tensor
    intrinsics: torch.Tensor
    text: str


def load_camera(
    data_root: str,
    scene: str,
    camera: str,
    resize_crop: ResizeCropAspectCenter,
) -> CameraClip:
    """Load ``<scene>__<camera>.mp4/.json`` and return ``[C,T,H,W]`` frames."""
    stem = f"{scene}__{camera}"
    json_path = os.path.join(data_root, f"{stem}.json")
    video_path = os.path.join(data_root, f"{stem}.mp4")
    with open(json_path, "r", encoding="utf-8") as handle:
        parsed = extract_spatialvid_meta(json.load(handle))

    reader = VideoReader(video_path, ctx=cpu(0), num_threads=4)
    frame_count = round_down_4n_plus_1(len(reader))
    if frame_count < 5:
        raise ValueError(f"{video_path} contains fewer than five usable frames")
    frames = reader.get_batch(list(range(frame_count))).asnumpy()
    original_height, original_width = frames.shape[1:3]
    if original_height > original_width:
        raise ValueError(f"portrait input is not supported: {video_path}")
    if len(parsed["poses"]) < frame_count:
        raise ValueError(
            f"{json_path} has {len(parsed['poses'])} poses for {frame_count} frames"
        )

    intrinsics = resize_crop.transform_intrinsics(
        parsed["intrinsics"], original_height, original_width
    )
    video = torch.from_numpy(frames).permute(0, 3, 1, 2).float() / 127.5 - 1.0
    video = resize_crop(video).clamp(-1.0, 1.0).permute(1, 0, 2, 3).contiguous()
    return CameraClip(
        video=video,
        poses=torch.from_numpy(parsed["poses"][:frame_count]).float(),
        intrinsics=torch.from_numpy(intrinsics).float(),
        text=parsed["text"],
    )


def camera_control(
    poses: torch.Tensor,
    intrinsics: torch.Tensor,
    num_chunks: int,
    chunk_size: int,
    temporal_stride: int,
    video_hw: tuple[int, int],
    latent_hw: tuple[int, int],
    reference_pose: torch.Tensor,
    translation_scale: float,
    device: torch.device | str,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build the camera Plucker tensor and interpolated latent-frame poses."""
    latent_frames = int(num_chunks) * int(chunk_size)
    target_rgb_indices = np.arange(latent_frames, dtype=np.float64) * int(temporal_stride)
    poses_np = poses.detach().cpu().numpy()
    absolute = interpolate_camera_poses(
        src_indices=np.arange(len(poses_np), dtype=np.float64),
        src_rot_mat=poses_np[:, :3, :3],
        src_trans_vec=poses_np[:, :3, 3],
        tgt_indices=target_rgb_indices,
    ).to(device)
    relative = compute_relative_poses(
        absolute,
        framewise=False,
        ref_pose=reference_pose.to(device),
        trans_scale=float(translation_scale),
        normalize_trans=False,
    )
    height, width = (int(v) for v in video_hw)
    latent_height, latent_width = (int(v) for v in latent_hw)
    intrinsics = broadcast_intrinsics_to_length(intrinsics, latent_frames).to(device)
    plucker = get_plucker_embeddings(
        relative, intrinsics, height, width, only_rays_d=False
    )
    plucker = rearrange(
        plucker,
        "f (h ph) (w pw) c -> (f h w) (c ph pw)",
        ph=height // latent_height,
        pw=width // latent_width,
    )[None]
    control = rearrange(
        plucker,
        "b (f h w) c -> b c f h w",
        f=latent_frames,
        h=latent_height,
        w=latent_width,
    )[0].to(device=device, dtype=dtype).contiguous()
    return control, absolute


def load_trajectory(
    path: str,
    target_cameras: list[str],
    frame_count: int,
) -> list[torch.Tensor]:
    """Read absolute c2w trajectories ordered by target camera.

    Poses are already in the world coordinate frame and are therefore passed
    directly to camera-control construction.  A ``cam`` field selects views by
    name when present; otherwise the ``views`` order must match ``target_cameras``.
    """
    with open(path, "r", encoding="utf-8") as handle:
        document = json.load(handle)
    views = document.get("views")
    if not isinstance(views, list) or len(views) != len(target_cameras):
        raise ValueError(
            f"trajectory must contain one view per target camera ({len(target_cameras)}), "
            f"got {len(views) if isinstance(views, list) else 'none'}"
        )

    by_camera = {str(view["cam"]): view for view in views if "cam" in view}
    if by_camera and set(by_camera) != set(target_cameras):
        raise ValueError(
            "trajectory camera names must exactly match --target_cams: "
            f"trajectory={sorted(by_camera)} target={sorted(target_cameras)}"
        )
    ordered = [by_camera[camera] for camera in target_cameras] if by_camera else views

    result = []
    for index, view in enumerate(ordered):
        poses = torch.as_tensor(view.get("poses"), dtype=torch.float32)
        expected = (int(frame_count), 4, 4)
        if tuple(poses.shape) != expected:
            raise ValueError(
                f"trajectory view {index} must have shape {expected}, got {tuple(poses.shape)}"
            )
        result.append(poses.contiguous())
    return result
