"""Run PaperA multi-view autoregressive inference on an authored trajectory."""
from __future__ import annotations

import argparse
import json
import os

import imageio.v3 as iio
import numpy as np
import torch
import torch.distributed as dist

from papera.checkpoints import load_strict_full_checkpoint
from papera.inference_utils import camera_control, load_camera, load_trajectory
from wan.commons.parallel_states import initialize_parallel_state
from wan.configs import WAN_CONFIGS
from wan.dataset.pmem_retrieval import rank_mem_candidates, token_xnow_gate
from wan.modules.model_ar import WanModelAR
from wan.modules.t5 import T5EncoderModel
from wan.modules.vae2_1 import Wan2_1_VAE
from wan.utils.accel import device_module, device_type, empty_cache, is_npu, manual_seed_all
from wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
from wan.utils.infer_data import ResizeCropAspectCenter
from wan.utils.prompt_template import compose_scene_text_condition
from wan.utils.stage1_ar_flow import DEFAULT_FLOW_SHIFT
from wan.utils.stage1_ar_geometry import build_i2v_mask
from wan.utils.stage1_ar_selfresample import ar_seq_len


CHUNK_SIZE = 4
WINDOW_CHUNKS = 1
PMEM_PER_VIEW = 1
XNOW_GATE = 1.0


def parse_target_cameras(value: str) -> list[str]:
    cameras = [item.strip() for item in value.split(",") if item.strip()]
    if not cameras:
        raise ValueError("--target_cams must contain at least one camera")
    return cameras


def trajectory_geometry(path: str, temporal_stride: int) -> tuple[int, int]:
    with open(path, "r", encoding="utf-8") as handle:
        document = json.load(handle)
    chunks = document.get("n_chunks")
    frames = document.get("n_frames")
    if not isinstance(chunks, int) or chunks < 1:
        raise ValueError("trajectory must define a positive integer n_chunks")
    if not isinstance(frames, int) or frames < 1:
        raise ValueError("trajectory must define a positive integer n_frames")
    expected_frames = (chunks * CHUNK_SIZE - 1) * temporal_stride + 1
    if frames != expected_frames:
        raise ValueError(
            f"trajectory n_frames={frames} does not match n_chunks={chunks}: "
            f"expected {expected_frames} for chunk_size={CHUNK_SIZE}"
        )
    return chunks, frames


def intrinsics4(intrinsics: torch.Tensor) -> list[float]:
    values = torch.as_tensor(intrinsics, dtype=torch.float64).reshape(-1)
    if values.numel() == 9:
        return [float(values[0]), float(values[4]), float(values[2]), float(values[5])]
    if values.numel() == 4:
        return [float(value) for value in values]
    raise ValueError(f"intrinsics must be [fx, fy, cx, cy] or 3x3, got {tuple(intrinsics.shape)}")


def scene_scale(pose_sequences: list[torch.Tensor]) -> float:
    centers = torch.cat([poses[:, :3, 3].detach().cpu() for poses in pose_sequences], dim=0)
    span = torch.cdist(centers.float(), centers.float()).max().item()
    return float(span) if span > 1e-8 else 1.0


def padded_text_embedding(encoder: T5EncoderModel, text: str, config, device: torch.device) -> torch.Tensor:
    embedding = encoder([text], device)[0].to(device)
    if embedding.shape[0] > config.text_len:
        raise ValueError(f"text encoder returned {embedding.shape[0]} tokens, expected <= {config.text_len}")
    if embedding.shape[0] < config.text_len:
        padding = embedding.new_zeros(config.text_len - embedding.shape[0], embedding.shape[1])
        embedding = torch.cat([embedding, padding], dim=0)
    return embedding.contiguous()


def load_first_frame(path: str, resize_crop: ResizeCropAspectCenter, device: torch.device) -> torch.Tensor:
    image = iio.imread(path)
    if image.ndim != 3 or image.shape[-1] < 3:
        raise ValueError(f"first-frame image must have at least three channels: {path}")
    tensor = torch.from_numpy(image[..., :3]).permute(2, 0, 1).float()[None] / 127.5 - 1.0
    return resize_crop(tensor).clamp(-1.0, 1.0)[0].to(device)


def initialize_single_process() -> torch.device:
    if dist.is_initialized():
        if dist.get_world_size() != 1:
            raise RuntimeError("PaperA release inference supports one process")
    else:
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        os.environ.setdefault("LOCAL_RANK", "0")
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29500")
        dist.init_process_group(backend="hccl" if is_npu() else "nccl", rank=0, world_size=1)
    device_module.set_device(0)
    initialize_parallel_state(sp=1, dp_replicate=1)
    return torch.device(device_type, 0)


def build_model(pretrained_model_root: str, checkpoint: str, device: torch.device) -> WanModelAR:
    config = WanModelAR.load_config(pretrained_model_root, subfolder="transformers")
    previous_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        model = WanModelAR.from_config(
            config,
            control_type="cam",
            zero_history_timestep=True,
            local_attn_size=CHUNK_SIZE,
            sink_size=0,
        )
    finally:
        torch.set_default_dtype(previous_dtype)
    load_strict_full_checkpoint(model, checkpoint)
    return model.to(device=device, dtype=torch.bfloat16).eval()


def build_gate(
    *,
    target_poses: list[torch.Tensor],
    target_intrinsics: list[list[float]],
    source_poses: torch.Tensor,
    source_intrinsics: list[float],
    selected_memory: list[tuple[int, int]],
    per_view_memory: list[list[int]],
    chunk_index: int,
    latent_hw: tuple[int, int],
    video_hw: tuple[int, int],
    device: torch.device,
) -> torch.Tensor:
    grid_hw = (latent_hw[0] // 2, latent_hw[1] // 2)
    gates = []
    for view, query_poses in enumerate(target_poses):
        clean_poses = [pose for pose in source_poses]
        clean_intrinsics = [source_intrinsics] * len(source_poses)
        if chunk_index > 0:
            start = (chunk_index - 1) * CHUNK_SIZE
            stop = chunk_index * CHUNK_SIZE
            clean_poses.extend(query_poses[start:stop])
            clean_intrinsics.extend([target_intrinsics[view]] * CHUNK_SIZE)
        for memory_index in per_view_memory[view]:
            memory_view, memory_chunk = selected_memory[memory_index]
            start = memory_chunk * CHUNK_SIZE
            stop = start + CHUNK_SIZE
            clean_poses.extend(target_poses[memory_view][start:stop])
            clean_intrinsics.extend([target_intrinsics[memory_view]] * CHUNK_SIZE)

        peers = []
        peer_intrinsics = []
        start = chunk_index * CHUNK_SIZE
        stop = start + CHUNK_SIZE
        for peer, peer_poses in enumerate(target_poses):
            if peer == view:
                continue
            peers.extend(peer_poses[start:stop])
            peer_intrinsics.extend([target_intrinsics[peer]] * CHUNK_SIZE)

        gate = token_xnow_gate(
            query_poses[start:stop],
            target_intrinsics[view],
            clean_poses,
            clean_intrinsics,
            video_hw,
            grid_hw,
            sigma=XNOW_GATE,
            peer_c2ws=peers,
            peer_intrinsics=peer_intrinsics,
        )
        gates.append(gate.reshape(-1))
    return torch.as_tensor(np.concatenate(gates), device=device, dtype=torch.float32)


def run(args: argparse.Namespace, device: torch.device) -> None:
    config = WAN_CONFIGS["i2v-A14B"]
    temporal_stride = int(config.vae_stride[0])
    target_cameras = parse_target_cameras(args.target_cams)
    num_chunks, num_frames = trajectory_geometry(args.trajectory, temporal_stride)
    resize_crop = ResizeCropAspectCenter(args.target_height, args.target_width)

    source = load_camera(args.data_root, args.scene, args.src_cam, resize_crop)
    targets = [load_camera(args.data_root, args.scene, camera, resize_crop) for camera in target_cameras]
    video_hw = (int(source.video.shape[-2]), int(source.video.shape[-1]))
    for camera, target in zip(target_cameras, targets):
        if tuple(target.video.shape[-2:]) != video_hw:
            raise ValueError(f"source and {camera} have different resized dimensions")

    target_paths = load_trajectory(args.trajectory, target_cameras, num_frames)
    model = build_model(args.pretrained_model_root, args.ckpt, device)
    vae = Wan2_1_VAE(
        vae_pth=os.path.join(args.pretrained_model_root, config.vae_checkpoint),
        dtype=torch.float32,
        device=device,
    )
    text_encoder = T5EncoderModel(
        text_len=config.text_len,
        dtype=config.t5_dtype,
        device=device,
        checkpoint_path=os.path.join(args.pretrained_model_root, config.t5_checkpoint),
        tokenizer_path=os.path.join(args.pretrained_model_root, config.t5_tokenizer),
    )

    reference_pose = source.poses[0].to(device)
    source_control, source_poses = camera_control(
        source.poses,
        source.intrinsics,
        num_chunks=1,
        chunk_size=CHUNK_SIZE,
        temporal_stride=temporal_stride,
        video_hw=video_hw,
        latent_hw=(args.target_height // int(config.vae_stride[1]), args.target_width // int(config.vae_stride[2])),
        reference_pose=reference_pose,
        translation_scale=1.0,
        device=device,
        dtype=config.param_dtype,
    )
    provisional = [source_poses]
    target_pose_sequences = []
    for path, target in zip(target_paths, targets):
        _, poses = camera_control(
            path,
            target.intrinsics,
            num_chunks=num_chunks,
            chunk_size=CHUNK_SIZE,
            temporal_stride=temporal_stride,
            video_hw=video_hw,
            latent_hw=(args.target_height // int(config.vae_stride[1]), args.target_width // int(config.vae_stride[2])),
            reference_pose=reference_pose,
            translation_scale=1.0,
            device=device,
            dtype=config.param_dtype,
        )
        provisional.append(poses)
        target_pose_sequences.append(poses)
    translation_scale = scene_scale(provisional)

    source_control, source_poses = camera_control(
        source.poses,
        source.intrinsics,
        num_chunks=1,
        chunk_size=CHUNK_SIZE,
        temporal_stride=temporal_stride,
        video_hw=video_hw,
        latent_hw=(args.target_height // int(config.vae_stride[1]), args.target_width // int(config.vae_stride[2])),
        reference_pose=reference_pose,
        translation_scale=translation_scale,
        device=device,
        dtype=config.param_dtype,
    )
    target_controls = []
    target_pose_sequences = []
    for path, target in zip(target_paths, targets):
        control, poses = camera_control(
            path,
            target.intrinsics,
            num_chunks=num_chunks,
            chunk_size=CHUNK_SIZE,
            temporal_stride=temporal_stride,
            video_hw=video_hw,
            latent_hw=(args.target_height // int(config.vae_stride[1]), args.target_width // int(config.vae_stride[2])),
            reference_pose=reference_pose,
            translation_scale=translation_scale,
            device=device,
            dtype=config.param_dtype,
        )
        target_controls.append(control)
        target_pose_sequences.append(poses)

    conditioning_video = torch.zeros(
        3,
        num_frames,
        video_hw[0],
        video_hw[1],
        device=device,
        dtype=source.video.dtype,
    )
    conditioning_video[:, 0] = source.video[:, 0].to(device)
    if args.first_frame_image:
        conditioning_video[:, 0] = load_first_frame(args.first_frame_image, resize_crop, device)
    with torch.no_grad():
        condition_latents = vae.encode([conditioning_video])[0]
    latent_hw = (int(condition_latents.shape[-2]), int(condition_latents.shape[-1]))
    expected_latent_frames = num_chunks * CHUNK_SIZE
    if int(condition_latents.shape[1]) != expected_latent_frames:
        raise RuntimeError(
            f"VAE returned {condition_latents.shape[1]} latent frames, expected {expected_latent_frames}"
        )
    mask = build_i2v_mask(
        num_frames,
        latent_hw[0],
        latent_hw[1],
        device,
        condition_latents.dtype,
        temporal_stride,
    )
    target_conditions = torch.cat([mask, condition_latents], dim=0).to(config.param_dtype)
    source_stream = torch.cat(
        [condition_latents[:, :CHUNK_SIZE], target_conditions[:, :CHUNK_SIZE]], dim=0
    )[None].to(config.param_dtype)
    source_indices = torch.arange(CHUNK_SIZE, device=device)[None]
    source_control = source_control[:, :CHUNK_SIZE][None]

    empty_prompt = padded_text_embedding(text_encoder, "", config, device)
    scene_prompt = padded_text_embedding(
        text_encoder, compose_scene_text_condition(source.text), config, device
    )[None]
    del text_encoder
    empty_cache()

    target_intrinsics = [intrinsics4(target.intrinsics) for target in targets]
    source_intrinsics = intrinsics4(source.intrinsics)
    generated: list[list[torch.Tensor]] = [[] for _ in targets]
    chunk_prompt_indices = torch.zeros(2, dtype=torch.long, device=device)
    manual_seed_all(args.seed)

    for chunk_index in range(num_chunks):
        start = chunk_index * CHUNK_SIZE
        stop = start + CHUNK_SIZE
        kept = [chunk_index - 1] if chunk_index > 0 else []
        history = None
        history_control = None
        history_indices = None
        if kept:
            history_parts = []
            control_parts = []
            for view in range(len(targets)):
                latents = torch.cat([generated[view][index] for index in kept], dim=1)
                conditions = torch.cat(
                    [target_conditions[:, index * CHUNK_SIZE:(index + 1) * CHUNK_SIZE] for index in kept],
                    dim=1,
                )
                history_parts.append(torch.cat([latents, conditions], dim=0))
                control_parts.append(
                    torch.cat(
                        [target_controls[view][:, index * CHUNK_SIZE:(index + 1) * CHUNK_SIZE] for index in kept],
                        dim=1,
                    )
                )
            history = torch.cat(history_parts, dim=1)[None]
            history_control = torch.cat(control_parts, dim=2)[None]
            history_indices = torch.arange(CHUNK_SIZE, device=device).repeat(len(targets))[None]

        candidates = [
            (view, earlier)
            for view in range(len(targets))
            for earlier in range(max(0, chunk_index - WINDOW_CHUNKS))
        ]
        per_view_pairs: list[list[tuple[int, int]]] = []
        for view in range(len(targets)):
            if not candidates:
                per_view_pairs.append([])
                continue
            query = target_pose_sequences[view][start + CHUNK_SIZE // 2]
            candidate_poses = [
                target_pose_sequences[candidate_view][candidate_chunk * CHUNK_SIZE + CHUNK_SIZE // 2]
                for candidate_view, candidate_chunk in candidates
            ]
            candidate_intrinsics = [target_intrinsics[candidate_view] for candidate_view, _ in candidates]
            selected_indices, _ = rank_mem_candidates(
                query,
                target_intrinsics[view],
                candidate_poses,
                candidate_intrinsics,
                video_hw,
                r=PMEM_PER_VIEW,
                scene_scale=translation_scale,
            )
            per_view_pairs.append([candidates[index] for index in selected_indices])
        selected_memory = sorted({pair for pairs in per_view_pairs for pair in pairs})
        per_view_memory = [
            [selected_memory.index(pair) for pair in pairs] for pairs in per_view_pairs
        ]

        memory_stream = source_stream
        memory_control = source_control
        memory_indices = source_indices
        if selected_memory:
            anchors = []
            anchor_controls = []
            for memory_view, memory_chunk in selected_memory:
                conditions = target_conditions[:, memory_chunk * CHUNK_SIZE:(memory_chunk + 1) * CHUNK_SIZE]
                anchors.append(torch.cat([generated[memory_view][memory_chunk], conditions], dim=0))
                anchor_controls.append(
                    target_controls[memory_view][:, memory_chunk * CHUNK_SIZE:(memory_chunk + 1) * CHUNK_SIZE]
                )
            memory_stream = torch.cat([source_stream, torch.cat(anchors, dim=1)[None]], dim=2)
            memory_control = torch.cat([source_control, torch.cat(anchor_controls, dim=1)[None]], dim=2)
            memory_indices = torch.cat(
                [source_indices, torch.arange(CHUNK_SIZE, device=device).repeat(len(selected_memory))[None]],
                dim=1,
            )

        target_control = torch.cat(
            [control[:, start:stop] for control in target_controls], dim=1
        )[None]
        target_condition = torch.cat(
            [target_conditions[:, start:stop] for _ in targets], dim=1
        )
        controls = {
            "c2ws_plucker_emb": target_control,
            "c2ws_plucker_emb_source": memory_control,
        }
        if history_control is not None:
            controls["c2ws_plucker_emb_history_short"] = history_control

        gate = build_gate(
            target_poses=target_pose_sequences,
            target_intrinsics=target_intrinsics,
            source_poses=source_poses,
            source_intrinsics=source_intrinsics,
            selected_memory=selected_memory,
            per_view_memory=per_view_memory,
            chunk_index=chunk_index,
            latent_hw=latent_hw,
            video_hw=video_hw,
            device=device,
        )
        layout = {
            "chunk_size": CHUNK_SIZE,
            "tgt_abs_chunks": [2],
            "xnow_gate": gate,
        }
        if selected_memory:
            layout["mem_src_chunks"] = len(selected_memory)
            layout["mem_src_sets"] = [per_view_memory]

        source_frames = CHUNK_SIZE * (1 + len(selected_memory))
        history_frames = CHUNK_SIZE * len(kept)
        sequence_length = ar_seq_len(
            source_frames + len(targets) * history_frames + len(targets) * CHUNK_SIZE,
            latent_hw[0],
            latent_hw[1],
            config.patch_size,
        )
        target_indices = torch.arange(CHUNK_SIZE, 2 * CHUNK_SIZE, device=device).repeat(len(targets))[None]
        scheduler = FlowUniPCMultistepScheduler(
            num_train_timesteps=config.num_train_timesteps,
            shift=DEFAULT_FLOW_SHIFT,
            use_dynamic_shifting=False,
        )
        scheduler.set_timesteps(args.sampling_steps, device=device, shift=DEFAULT_FLOW_SHIFT)
        sample = torch.randn(
            1,
            16,
            len(targets) * CHUNK_SIZE,
            latent_hw[0],
            latent_hw[1],
            device=device,
            dtype=torch.float32,
        )
        with torch.no_grad():
            for timestep in scheduler.timesteps:
                with torch.autocast(device_type, dtype=torch.bfloat16, enabled=device_type != "cpu"):
                    prediction, bidirectional_prediction = model(
                        x=[sample[0].to(config.param_dtype)],
                        t=torch.stack([timestep]).to(device),
                        context=[empty_prompt],
                        seq_len=sequence_length,
                        y=[target_condition],
                        dit_cond_dict=controls,
                        indices_hidden_states=target_indices,
                        indices_latents_history_short=history_indices,
                        latents_history_short=history,
                        history_timestep=0.0,
                        ar_history=True,
                        ar_layout=layout,
                        chunk_text_idx=chunk_prompt_indices,
                        a_b_emb=scene_prompt,
                        a_g_emb=scene_prompt,
                        enable_bi=False,
                        latents_source=memory_stream,
                        indices_source=memory_indices,
                        num_views=len(targets),
                    )
                if bidirectional_prediction is not None:
                    raise RuntimeError("PaperA inference must not produce a bidirectional branch")
                sample = scheduler.step(
                    prediction[0].float().unsqueeze(0), timestep, sample, return_dict=False
                )[0]
        for view in range(len(targets)):
            generated[view].append(sample[0, :, view * CHUNK_SIZE:(view + 1) * CHUNK_SIZE].to(config.param_dtype))
        print(
            f"[infer] chunk {chunk_index + 1}/{num_chunks} "
            f"pmem={per_view_pairs}",
            flush=True,
        )

    with torch.no_grad():
        videos = [vae.decode([torch.cat(chunks, dim=1)])[0] for chunks in generated]
    frames = np.concatenate(
        [
            ((video.float().clamp(-1, 1) + 1) * 127.5).round().byte().permute(1, 2, 3, 0).cpu().numpy()
            for video in videos
        ],
        axis=2,
    )
    output_dir = os.path.dirname(os.path.abspath(args.out))
    os.makedirs(output_dir, exist_ok=True)
    iio.imwrite(args.out, frames, fps=16, codec="libx264", output_params=["-crf", "12"])
    print(f"[infer] wrote {frames.shape[0]} frames to {args.out}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the released PaperA inference recipe")
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--pretrained_model_root", required=True)
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--target_cams", required=True, help="comma-separated camera names")
    parser.add_argument("--trajectory", required=True)
    parser.add_argument("--src_cam", default="cam01")
    parser.add_argument("--first_frame_image", default="")
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sampling_steps", type=int, default=30)
    parser.add_argument("--target_height", type=int, default=240)
    parser.add_argument("--target_width", type=int, default=416)
    args = parser.parse_args()
    if args.sampling_steps < 1:
        raise ValueError("--sampling_steps must be positive")

    device = initialize_single_process()
    try:
        run(args, device)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
