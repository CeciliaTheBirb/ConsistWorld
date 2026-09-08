"""Validate ConsistWorld SpatialVID/MultiCamData clips before cache construction."""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from decord import VideoReader, cpu


PATTERN = re.compile(r"^(.*)__cam(\d+)$")


def _usable_frames(count: int) -> int:
    return ((count - 1) // 4) * 4 + 1 if count > 0 else 0


def validate_view(json_path: Path, expected_frames: int) -> tuple[bool, str]:
    video_path = json_path.with_suffix(".mp4")
    if not video_path.is_file():
        return False, "missing mp4"
    try:
        with json_path.open("r", encoding="utf-8") as handle:
            meta = json.load(handle)
        intrinsics = np.asarray(meta["intrinsics_vipe"], dtype=np.float32)
        poses = np.asarray(meta["poses_w2c_vipe"], dtype=np.float32)
        if intrinsics.shape != (3, 3):
            return False, f"intrinsics shape {intrinsics.shape}, expected (3, 3)"
        if poses.ndim != 3 or poses.shape[1:] != (4, 4):
            return False, f"pose shape {poses.shape}, expected [T, 4, 4]"
        if not np.isfinite(intrinsics).all() or not np.isfinite(poses).all():
            return False, "non-finite camera metadata"
        if not isinstance(meta.get("caption", {}), dict):
            return False, "caption must be an object"
        reader = VideoReader(str(video_path), ctx=cpu(0), num_threads=1)
        frame_count = len(reader)
        if frame_count < 1:
            return False, "video contains no frames"
        first_frame = reader[0].asnumpy()
        if first_frame.ndim != 3 or first_frame.shape[-1] < 3:
            return False, f"unexpected decoded frame shape {first_frame.shape}"
        if first_frame.shape[0] > first_frame.shape[1]:
            return False, f"portrait video {first_frame.shape[1]}x{first_frame.shape[0]} is unsupported"
        usable = _usable_frames(frame_count)
        if usable < expected_frames:
            return False, f"only {usable} usable video frames, expected >= {expected_frames}"
        if poses.shape[0] < usable:
            return False, f"only {poses.shape[0]} poses for {usable} usable video frames"
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"
    return True, f"frames={usable} poses={poses.shape[0]}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate SpatialVID/MultiCamData input for ConsistWorld")
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--expected_views", type=int, default=0, help="0 permits any scene with >=2 views")
    parser.add_argument("--expected_frames", type=int, default=5)
    args = parser.parse_args()
    if args.expected_views < 0 or args.expected_frames < 5:
        parser.error("--expected_views must be >= 0 and --expected_frames must be >= 5")

    root = Path(args.data_root)
    if not root.is_dir():
        parser.error(f"not a directory: {root}")
    scenes: dict[str, list[Path]] = defaultdict(list)
    malformed: list[Path] = []
    for path in sorted(root.rglob("*.json")):
        match = PATTERN.match(path.stem)
        if match is None:
            malformed.append(path)
        else:
            scenes[str(path.parent / match.group(1))].append(path)

    bad = 0
    valid_scenes = 0
    for scene, paths in sorted(scenes.items()):
        scene_bad = False
        if len(paths) < 2 or (args.expected_views and len(paths) != args.expected_views):
            print(f"BAD  {scene}: found {len(paths)} views")
            bad += 1
            continue
        for path in paths:
            ok, message = validate_view(path, args.expected_frames)
            state = "OK  " if ok else "BAD "
            print(f"{state} {path}: {message}")
            if not ok:
                bad += 1
                scene_bad = True
        if not scene_bad:
            valid_scenes += 1
    for path in malformed:
        print(f"BAD  {path}: expected filename <scene>__camNN.json")
        bad += 1
    print(f"validated scenes={valid_scenes} invalid_entries={bad}")
    raise SystemExit(1 if bad else 0)


if __name__ == "__main__":
    main()
