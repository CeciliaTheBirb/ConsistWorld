"""Convert an Infinigen render into ConsistWorld's SpatialVID/MultiCamData layout.

Infinigen writes RGB frames and camera parameters below::

    <scene>/frames/Image/camera_0/Image_<rig>_<subcam>_<frame>_<resample>.png
    <scene>/frames/camview/camera_0/camview_<rig>_<subcam>_<frame>_<resample>.npz

Each rig becomes one output view named ``<scene_name>__camNN.mp4/.json``.  The
JSON uses the same normalized intrinsics and OpenCV world-to-camera convention
accepted by ``build_clip_cache.py``. Existing output pairs are never overwritten.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
from pathlib import Path

import imageio.v3 as iio
import numpy as np


BLENDER_TO_OPENCV = np.diag([1.0, -1.0, -1.0, 1.0]).astype(np.float64)
CAMVIEW_RE = re.compile(r"^camview_(\d+)_(\d+)_(\d+)_(\d+)\.npz$")
IMAGE_RE = re.compile(r"^Image_(\d+)_(\d+)_(\d+)_(\d+)\.png$")


def _parse_name(path: str | Path, pattern: re.Pattern[str], kind: str) -> tuple[int, int, int, int]:
    match = pattern.match(Path(path).name)
    if match is None:
        raise ValueError(f"unexpected {kind} filename: {path}")
    return tuple(int(value) for value in match.groups())


def _files_for_rig(directory: Path, pattern: re.Pattern[str], rig: int, kind: str) -> list[Path]:
    records: list[tuple[tuple[int, int, int, int], Path]] = []
    for path in directory.glob("*.npz" if kind == "camview" else "*.png"):
        match = pattern.match(path.name)
        if match is None:
            continue
        values = tuple(int(value) for value in match.groups())
        if values[0] == rig:
            records.append((values, path))
    records.sort(key=lambda record: (record[0][2], record[0][1], record[0][3]))
    if not records:
        raise FileNotFoundError(f"no {kind} files for rig {rig} in {directory}")
    duplicate_frames = [
        frame
        for frame in {record[0][2] for record in records}
        if sum(record[0][2] == frame for record in records) != 1
    ]
    if duplicate_frames:
        raise ValueError(
            f"rig {rig} has duplicate {kind} frames {duplicate_frames[:8]}; "
            "select one subcamera/resample before conversion"
        )
    return [path for _, path in records]


def _rigs_in(camview_dir: Path) -> list[int]:
    rigs = set()
    for path in camview_dir.glob("camview_*.npz"):
        match = CAMVIEW_RE.match(path.name)
        if match is not None:
            rigs.add(int(match.group(1)))
    return sorted(rigs)


def _load_camviews(paths: list[Path]) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    transforms: list[np.ndarray] = []
    intrinsics: np.ndarray | None = None
    image_hw: tuple[int, int] | None = None
    for path in paths:
        with np.load(path) as data:
            if not {"T", "K", "HW"}.issubset(data.files):
                raise ValueError(f"{path} must contain T, K, and HW; found {sorted(data.files)}")
            transform = np.asarray(data["T"], dtype=np.float64)
            matrix = np.asarray(data["K"], dtype=np.float64)
            hw = tuple(int(value) for value in np.asarray(data["HW"]).reshape(-1))
        if transform.shape != (4, 4):
            raise ValueError(f"{path}: T must be [4, 4], got {transform.shape}")
        if matrix.shape != (3, 3):
            raise ValueError(f"{path}: K must be [3, 3], got {matrix.shape}")
        if len(hw) != 2 or min(hw) <= 0:
            raise ValueError(f"{path}: HW must be [H, W], got {hw}")
        if intrinsics is None:
            intrinsics = matrix
            image_hw = hw
        elif not np.allclose(matrix, intrinsics, rtol=1e-5, atol=1e-5) or hw != image_hw:
            raise ValueError(f"rig camera calibration changes across frames: {path}")
        transforms.append(transform)
    assert intrinsics is not None and image_hw is not None
    return np.stack(transforms), intrinsics, image_hw


def _write_view(
    *,
    camview_paths: list[Path],
    image_paths: list[Path],
    out_stem: Path,
    caption: str,
    fps: int,
) -> tuple[int, tuple[int, int]]:
    if len(camview_paths) != len(image_paths):
        raise ValueError(
            f"{out_stem.name}: {len(image_paths)} RGB frames but {len(camview_paths)} camera records"
        )
    camview_frames = [_parse_name(path, CAMVIEW_RE, "camview")[2] for path in camview_paths]
    image_frames = [_parse_name(path, IMAGE_RE, "image")[2] for path in image_paths]
    if camview_frames != image_frames:
        raise ValueError(f"{out_stem.name}: image/camera frame indices do not match")

    c2w_blender, K, (height, width) = _load_camviews(camview_paths)
    c2w_opencv = c2w_blender @ BLENDER_TO_OPENCV[None]
    w2c_opencv = np.linalg.inv(c2w_opencv).astype(np.float32)
    intrinsics = np.array(
        [[K[0, 0] / width, 0.0, K[0, 2] / width], [0.0, K[1, 1] / height, K[1, 2] / height], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )

    frames = []
    for path in image_paths:
        image = np.asarray(iio.imread(path))
        if image.ndim != 3 or image.shape[-1] < 3:
            raise ValueError(f"{path} is not an RGB(A) image")
        if tuple(image.shape[:2]) != (height, width):
            raise ValueError(f"{path}: image size {tuple(image.shape[:2])} does not match camview HW {(height, width)}")
        frames.append(image[..., :3])
    video = np.stack(frames)

    out_stem.parent.mkdir(parents=True, exist_ok=True)
    video_path = out_stem.with_suffix(".mp4")
    metadata_path = out_stem.with_suffix(".json")
    temporary_stem = out_stem.with_name(f".{out_stem.name}.tmp.{os.getpid()}")
    temporary_video = temporary_stem.with_suffix(".mp4")
    temporary_metadata = temporary_stem.with_suffix(".json")
    metadata = {
        "intrinsics_vipe": intrinsics.tolist(),
        "poses_w2c_vipe": w2c_opencv.tolist(),
        "caption": {"SceneDescription": caption},
    }
    try:
        iio.imwrite(
            temporary_video,
            video,
            fps=int(fps),
            codec="libx264",
            # 960x540 is already codec-compatible but its height is not divisible
            # by imageio's default macro block size (16). Disable implicit padding
            # so decoded video and calibration describe the same size.
            macro_block_size=1,
            output_params=["-crf", "14", "-pix_fmt", "yuv420p"],
        )
        with temporary_metadata.open("w", encoding="utf-8") as handle:
            json.dump(metadata, handle)
        os.replace(temporary_video, video_path)
        os.replace(temporary_metadata, metadata_path)
    except Exception:
        for path in (temporary_video, temporary_metadata):
            if path.exists():
                path.unlink()
        raise
    return len(video), (height, width)


def convert_scene(
    scene_dir: str | Path,
    out_dir: str | Path,
    scene_name: str,
    caption: str,
    fps: int,
    expected_views: int = 0,
) -> int:
    """Convert every rig below one Infinigen scene directory."""
    scene_dir = Path(scene_dir)
    out_dir = Path(out_dir)
    camview_dirs = sorted((scene_dir / "frames" / "camview").glob("camera_*"))
    if not camview_dirs:
        raise FileNotFoundError(f"no frames/camview/camera_* directory under {scene_dir}")

    planned_views: list[tuple[Path, Path, int]] = []
    for camview_dir in camview_dirs:
        image_dir = scene_dir / "frames" / "Image" / camview_dir.name
        if not image_dir.is_dir():
            raise FileNotFoundError(f"missing RGB directory for {camview_dir}: {image_dir}")
        for rig in _rigs_in(camview_dir):
            planned_views.append((camview_dir, image_dir, rig))

    if expected_views and len(planned_views) != int(expected_views):
        raise ValueError(
            f"expected {expected_views} views, found {len(planned_views)} under {scene_dir}"
        )
    output_stems = [out_dir / f"{scene_name}__cam{index:02d}" for index in range(1, len(planned_views) + 1)]
    existing = [
        path
        for stem in output_stems
        for path in (stem.with_suffix(".mp4"), stem.with_suffix(".json"))
        if path.exists()
    ]
    if existing:
        raise FileExistsError(
            "refusing to overwrite existing converted output; first: "
            + ", ".join(str(path) for path in existing[:4])
        )

    for view_index, (camview_dir, image_dir, rig) in enumerate(planned_views, start=1):
        stem = out_dir / f"{scene_name}__cam{view_index:02d}"
        count, hw = _write_view(
            camview_paths=_files_for_rig(camview_dir, CAMVIEW_RE, rig, "camview"),
            image_paths=_files_for_rig(image_dir, IMAGE_RE, rig, "image"),
            out_stem=stem,
            caption=caption,
            fps=fps,
        )
        print(
            f"{camview_dir.name} rig={rig} -> {stem.name} "
            f"({count} frames, {hw[1]}x{hw[0]})",
            flush=True,
        )
    print(f"converted {len(planned_views)} view(s) from {scene_dir}", flush=True)
    return len(planned_views)


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert Infinigen RGB/camview frames to ConsistWorld SpatialVID clips")
    parser.add_argument("--scene_dir", required=True, help="Infinigen scene directory containing frames/")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--scene_name", required=True, help="output prefix before __camNN")
    parser.add_argument("--caption", default="a procedural natural outdoor scene, multi-view camera exploration")
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument("--expected_views", type=int, default=0, help="0 accepts every rig; otherwise require this count")
    args = parser.parse_args()
    if args.fps < 1:
        parser.error("--fps must be positive")
    if args.expected_views < 0:
        parser.error("--expected_views must be non-negative")
    convert_scene(
        args.scene_dir,
        args.out_dir,
        args.scene_name,
        args.caption,
        args.fps,
        args.expected_views,
    )


if __name__ == "__main__":
    main()
