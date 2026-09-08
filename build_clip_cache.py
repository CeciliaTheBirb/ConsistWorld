"""Build the packed latent cache consumed by ConsistWorld training.

The input is a SpatialVID-style directory containing matching ``.mp4`` and
``.json`` files named ``<scene>__camNN``. Every output scene is self-contained:

    <out_dir>/clips/<scene>.pt
    <out_dir>/text/<scene>.pt
    <out_dir>/text/_empty.pt

The builder only loads the VAE and text encoder. It intentionally does not
start a model-serving process or load the diffusion transformer.
"""
from __future__ import annotations

import argparse
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor

import torch

from wan.configs import WAN_CONFIGS
from wan.dataset.local_dataset import SpatialVidDataset
from wan.modules.t5 import T5EncoderModel
from wan.modules.vae2_1 import Wan2_1_VAE
from wan.utils.accel import device_module, device_type, is_cuda, require_accelerator
from wan.utils.prompt_template import compose_scene_text_condition
from wan.utils.stage1_ar_geometry import Stage1ARGeometry


SCENE_PATTERN = re.compile(r"^(.*)__cam(\d+)$")


def _atomic_save(value, path: str) -> None:
    temporary = f"{path}.tmp.{os.getpid()}"
    torch.save(value, temporary)
    os.replace(temporary, path)


class ClipEncoder:
    """Local VAE/T5 encoder shared by one cache-building rank."""

    def __init__(self, pretrained_model_root: str, chunk_size: int) -> None:
        self.rank = int(os.environ.get("RANK", "0"))
        self.world_size = int(os.environ.get("WORLD_SIZE", "1"))
        self.local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        if device_type == "cpu":
            self.device = torch.device("cpu")
        else:
            self.device = torch.device(f"{device_type}:{self.local_rank}")
            device_module.set_device(self.device)
            if is_cuda():
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True

        self.config = WAN_CONFIGS["i2v-A14B"]
        self.geometry = Stage1ARGeometry(
            chunk_size=int(chunk_size),
            vae_temporal_stride=int(self.config.vae_stride[0]),
            patch_size=tuple(self.config.patch_size),
        )
        self.text_encoder = T5EncoderModel(
            text_len=self.config.text_len,
            dtype=self.config.t5_dtype,
            device=self.device,
            checkpoint_path=os.path.join(pretrained_model_root, self.config.t5_checkpoint),
            tokenizer_path=os.path.join(pretrained_model_root, self.config.t5_tokenizer),
        )
        self.vae = Wan2_1_VAE(
            vae_pth=os.path.join(pretrained_model_root, self.config.vae_checkpoint),
            dtype=torch.float32,
            device=self.device,
        )

    @property
    def latent_window_size(self) -> int:
        return self.geometry.latent_window_size

    def encode_text(self, prompt: str) -> torch.Tensor:
        with torch.no_grad():
            embedding = self.text_encoder([prompt], self.device)[0].detach().cpu()
        if embedding.shape[0] > self.config.text_len:
            raise ValueError(
                f"text embedding length {embedding.shape[0]} exceeds {self.config.text_len}"
            )
        if embedding.shape[0] < self.config.text_len:
            padding = embedding.new_zeros(
                self.config.text_len - embedding.shape[0], embedding.shape[1]
            )
            embedding = torch.cat([embedding, padding], dim=0)
        return embedding.contiguous()

    def encode_clip(
        self, video: torch.Tensor, max_chunks: int
    ) -> tuple[torch.Tensor, torch.Tensor, int, int]:
        """Return full latents, i2v latents, kept RGB frames, and chunk count."""
        stride = int(self.config.vae_stride[0])
        frame_count = int(video.shape[1])
        if (frame_count - 1) % stride != 0:
            raise ValueError(
                f"video length must be 1+{stride}k after loading, got {frame_count}"
            )
        latent_frames = (frame_count - 1) // stride + 1
        num_chunks = self.geometry.num_chunks(latent_frames)
        if max_chunks > 0:
            num_chunks = min(num_chunks, int(max_chunks))
        if num_chunks < 1:
            raise ValueError("clip does not contain one full autoregressive chunk")

        kept_latents = num_chunks * self.latent_window_size
        kept_frames = (kept_latents - 1) * stride + 1
        video = video[:, :kept_frames].to(self.device, non_blocking=True).contiguous()
        first_frame = torch.cat(
            [video[:, :1], torch.zeros_like(video[:, 1:])], dim=1
        ).contiguous()
        with torch.no_grad():
            latents = self.vae.encode([video])[0].detach().cpu().contiguous()
            condition_latents = self.vae.encode([first_frame])[0].detach().cpu().contiguous()
        if latents.shape[1] != kept_latents or condition_latents.shape[1] != kept_latents:
            raise RuntimeError(
                "VAE temporal output does not match the requested cache geometry: "
                f"expected {kept_latents}, got {latents.shape[1]} and {condition_latents.shape[1]}"
            )
        return latents, condition_latents, kept_frames, num_chunks


def _camera_name(sample: dict) -> str:
    match = SCENE_PATTERN.match(sample["json_path"].stem)
    if match is None:
        raise ValueError(f"sample name must end in __camNN: {sample['json_path']}")
    return f"cam{match.group(2)}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build ConsistWorld packed VAE/T5 clip caches")
    parser.add_argument("--pretrained_model_root", required=True)
    parser.add_argument("--navigation_roots", nargs="+", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--target_height", type=int, default=240)
    parser.add_argument("--target_width", type=int, default=416)
    parser.add_argument("--chunk_size", type=int, default=4)
    parser.add_argument("--max_chunks", type=int, default=0,
                        help="maximum chunks per camera; 0 keeps every full chunk")
    parser.add_argument("--decode_workers", type=int, default=8)
    parser.add_argument("--scene_filter", default="")
    parser.add_argument("--max_scenes", type=int, default=0)
    args = parser.parse_args()

    require_accelerator("Cache construction")
    encoder = ClipEncoder(args.pretrained_model_root, args.chunk_size)
    dataset = SpatialVidDataset(
        roots=args.navigation_roots,
        target_size=(args.target_height, args.target_width),
    )

    scenes: dict[str, list[dict]] = {}
    for sample in dataset.samples:
        match = SCENE_PATTERN.match(sample["json_path"].stem)
        if match is not None:
            scenes.setdefault(match.group(1), []).append(sample)
    scene_keys = sorted(scenes)
    if args.scene_filter:
        scene_keys = [key for key in scene_keys if args.scene_filter in key]
    if args.max_scenes > 0:
        scene_keys = scene_keys[:args.max_scenes]
    assigned_scenes = scene_keys[encoder.rank::encoder.world_size]

    clips_dir = os.path.join(args.out_dir, "clips")
    text_dir = os.path.join(args.out_dir, "text")
    os.makedirs(clips_dir, exist_ok=True)
    os.makedirs(text_dir, exist_ok=True)
    empty_text_path = os.path.join(text_dir, "_empty.pt")
    if encoder.rank == 0 and not os.path.exists(empty_text_path):
        _atomic_save(encoder.encode_text(""), empty_text_path)

    print(
        f"[cache] rank {encoder.rank}/{encoder.world_size}: "
        f"{len(assigned_scenes)} of {len(scene_keys)} scenes", flush=True
    )
    completed = skipped = failed = 0
    started = time.perf_counter()

    def decode(sample: dict):
        try:
            item = dataset.load(sample)
            return _camera_name(sample), item, None
        except Exception as exc:
            return _camera_name(sample), None, exc

    with ThreadPoolExecutor(max_workers=max(1, int(args.decode_workers))) as pool:
        for scene_key in assigned_scenes:
            clip_path = os.path.join(clips_dir, f"{scene_key}.pt")
            text_path = os.path.join(text_dir, f"{scene_key}.pt")
            if os.path.exists(clip_path) and os.path.exists(text_path):
                skipped += 1
                continue

            decoded = list(pool.map(decode, sorted(scenes[scene_key], key=_camera_name)))
            packed_scene: dict[str, dict] = {}
            caption = None
            for camera, item, error in decoded:
                if error is not None:
                    print(f"[cache] skip {scene_key}/{camera}: {error}", flush=True)
                    continue
                try:
                    latents, condition_latents, kept_frames, num_chunks = encoder.encode_clip(
                        item["video"], args.max_chunks
                    )
                except Exception as exc:
                    print(f"[cache] skip {scene_key}/{camera}: {exc}", flush=True)
                    continue
                packed_scene[camera] = {
                    "latents_full": latents,
                    "cond_i2v_latents": condition_latents,
                    "poses": item["poses"][:kept_frames].float().contiguous(),
                    "intrinsics": item["intrinsics"].contiguous(),
                    "video_hw": (int(item["video"].shape[-2]), int(item["video"].shape[-1])),
                    "num_chunks": int(num_chunks),
                    "lat_h": int(latents.shape[-2]),
                    "lat_w": int(latents.shape[-1]),
                    "sample_id": item["sample_id"],
                }
                caption = item["text"] if caption is None else caption

            if len(packed_scene) < 2:
                failed += 1
                print(f"[cache] skip {scene_key}: requires at least two valid cameras", flush=True)
                continue
            _atomic_save(packed_scene, clip_path)
            if not os.path.exists(text_path):
                _atomic_save(
                    encoder.encode_text(compose_scene_text_condition(caption or "")), text_path
                )
            completed += 1
            if completed % 20 == 0:
                seconds_per_scene = (time.perf_counter() - started) / completed
                print(
                    f"[cache] rank {encoder.rank}: {completed}/{len(assigned_scenes)} "
                    f"done, {skipped} skipped, {failed} failed, {seconds_per_scene:.2f}s/scene",
                    flush=True,
                )

    print(
        f"[cache] rank {encoder.rank} complete: {completed} written, {skipped} skipped, "
        f"{failed} failed", flush=True
    )


if __name__ == "__main__":
    main()
