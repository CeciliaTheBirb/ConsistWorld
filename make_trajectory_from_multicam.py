"""Export a ConsistWorld inference trajectory by replaying SpatialVID camera poses.

The input camera JSON stores OpenCV world-to-camera matrices.  ConsistWorld inference
expects absolute camera-to-world matrices in a separate trajectory document, so
this helper converts the convention and selects a common usable prefix from the
requested target views.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
from decord import VideoReader, cpu

from wan.utils.infer_data import extract_spatialvid_meta, round_down_4n_plus_1


CHUNK_SIZE = 4
TEMPORAL_STRIDE = 4
DEFAULT_MAX_CHUNKS = 9


def parse_cameras(value: str) -> list[str]:
    cameras = [item.strip() for item in value.split(",") if item.strip()]
    if not cameras:
        raise ValueError("--target_cams must contain at least one camera")
    if len(set(cameras)) != len(cameras):
        raise ValueError("--target_cams must not contain duplicate camera names")
    if len(cameras) > 4:
        raise ValueError("ConsistWorld inference supports one to four target cameras")
    return cameras


def required_rgb_frames(n_chunks: int) -> int:
    if n_chunks < 1:
        raise ValueError("n_chunks must be positive")
    return (n_chunks * CHUNK_SIZE - 1) * TEMPORAL_STRIDE + 1


def full_chunks_from_frames(frame_count: int) -> int:
    usable = round_down_4n_plus_1(frame_count)
    latent_frames = (usable - 1) // TEMPORAL_STRIDE + 1
    return latent_frames // CHUNK_SIZE


def load_camera_poses(data_root: Path, scene: str, camera: str) -> tuple[np.ndarray, int]:
    stem = f"{scene}__{camera}"
    json_path = data_root / f"{stem}.json"
    video_path = data_root / f"{stem}.mp4"
    if not json_path.is_file() or not video_path.is_file():
        raise FileNotFoundError(
            f"expected matching SpatialVID files for {stem}: {json_path} and {video_path}"
        )
    with json_path.open("r", encoding="utf-8") as handle:
        parsed = extract_spatialvid_meta(json.load(handle))
    poses = np.asarray(parsed["poses"], dtype=np.float32)
    if not np.isfinite(poses).all():
        raise ValueError(f"{json_path} contains non-finite poses")

    reader = VideoReader(str(video_path), ctx=cpu(0), num_threads=1)
    usable_frames = round_down_4n_plus_1(len(reader))
    if usable_frames < 5:
        raise ValueError(f"{video_path} has fewer than five usable frames")
    if poses.shape[0] < usable_frames:
        raise ValueError(
            f"{json_path} has {poses.shape[0]} poses for {usable_frames} usable video frames"
        )
    return poses[:usable_frames], usable_frames


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a valid ConsistWorld trajectory JSON from SpatialVID/MultiCamData poses"
    )
    parser.add_argument("--data_root", required=True, help="directory containing <scene>__camNN.mp4/.json")
    parser.add_argument("--scene", required=True)
    parser.add_argument("--target_cams", required=True, help="comma-separated target camera names")
    parser.add_argument(
        "--n_chunks",
        type=int,
        default=0,
        help="trajectory chunks; 0 selects the common prefix, capped at 9",
    )
    parser.add_argument("--start_frame", type=int, default=0, help="first RGB-frame pose to replay")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    if args.n_chunks < 0:
        parser.error("--n_chunks must be non-negative")
    if args.start_frame < 0:
        parser.error("--start_frame must be non-negative")

    data_root = Path(args.data_root)
    if not data_root.is_dir():
        parser.error(f"--data_root is not a directory: {data_root}")
    cameras = parse_cameras(args.target_cams)
    loaded = [load_camera_poses(data_root, args.scene, camera) for camera in cameras]
    available_chunks = min(
        full_chunks_from_frames(frame_count - args.start_frame)
        for _, frame_count in loaded
        if frame_count > args.start_frame
    )
    if available_chunks < 1:
        parser.error("the requested views have no full ConsistWorld chunk after --start_frame")
    n_chunks = int(args.n_chunks) if args.n_chunks else min(DEFAULT_MAX_CHUNKS, available_chunks)
    if n_chunks > available_chunks:
        parser.error(
            f"--n_chunks={n_chunks} needs {required_rgb_frames(n_chunks)} RGB frames after "
            f"--start_frame, but the common prefix supports only {available_chunks} chunks"
        )
    n_frames = required_rgb_frames(n_chunks)

    views = []
    for camera, (poses, _) in zip(cameras, loaded):
        selected = poses[args.start_frame : args.start_frame + n_frames]
        if selected.shape != (n_frames, 4, 4):
            raise RuntimeError(f"internal trajectory selection error for {camera}: {selected.shape}")
        views.append({"cam": camera, "poses": selected.tolist()})
    document = {"n_chunks": n_chunks, "n_frames": n_frames, "views": views}

    output = Path(args.out)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing trajectory: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f"{output.name}.tmp.{os.getpid()}")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(document, handle, indent=2, allow_nan=False)
            handle.write("\n")
        os.replace(temporary, output)
    except Exception:
        if temporary.exists():
            temporary.unlink()
        raise
    print(
        f"[trajectory] wrote {output}: {len(cameras)} view(s), "
        f"{n_chunks} chunk(s), {n_frames} RGB frames",
        flush=True,
    )


if __name__ == "__main__":
    main()
