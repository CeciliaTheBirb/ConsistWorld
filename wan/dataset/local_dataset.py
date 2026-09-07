"""Local SpatialVID reader used to build a PaperA clip cache."""
from __future__ import annotations

import json
import os
from pathlib import Path

import torch
from decord import VideoReader, cpu

from wan.utils.infer_data import (
    ResizeCropAspectCenter,
    extract_spatialvid_meta,
    round_down_4n_plus_1,
)


class SpatialVidDataset:
    """Scan and decode matching SpatialVID ``.mp4``/``.json`` camera clips."""

    def __init__(self, roots: list[str], target_size: tuple[int, int], video_num_threads: int = 4):
        self.roots = [Path(root) for root in roots]
        if not self.roots:
            raise ValueError("at least one SpatialVID root is required")
        self.resize_crop = ResizeCropAspectCenter(*target_size)
        self.video_num_threads = max(1, int(video_num_threads))
        self.samples = self._scan()
        if not self.samples:
            roots_text = ", ".join(str(root) for root in self.roots)
            raise FileNotFoundError(f"no matching .mp4/.json SpatialVID clips found under: {roots_text}")

    def _scan(self) -> list[dict[str, Path | str]]:
        samples: list[dict[str, Path | str]] = []
        for root in self.roots:
            if not root.exists():
                continue
            for directory, _, filenames in os.walk(root):
                names = set(filenames)
                for filename in sorted(name for name in filenames if name.endswith(".json")):
                    stem = filename[:-5]
                    if f"{stem}.mp4" not in names:
                        continue
                    json_path = Path(directory) / filename
                    samples.append(
                        {
                            "sample_id": f"{root.name}/{json_path.relative_to(root).with_suffix('')}",
                            "json_path": json_path,
                            "video_path": json_path.with_suffix(".mp4"),
                        }
                    )
        return sorted(samples, key=lambda sample: str(sample["json_path"]))

    def load(self, sample: dict[str, Path | str]) -> dict[str, torch.Tensor | str]:
        json_path = Path(sample["json_path"])
        video_path = Path(sample["video_path"])
        with json_path.open("r", encoding="utf-8") as handle:
            metadata = extract_spatialvid_meta(json.load(handle))

        reader = VideoReader(str(video_path), ctx=cpu(0), num_threads=self.video_num_threads)
        frame_count = round_down_4n_plus_1(len(reader))
        if frame_count < 5:
            raise ValueError(f"{video_path} has fewer than five usable frames")
        if len(metadata["poses"]) < frame_count:
            raise ValueError(
                f"{json_path} has {len(metadata['poses'])} poses for {frame_count} video frames"
            )

        frames = reader.get_batch(list(range(frame_count))).asnumpy()
        height, width = frames.shape[1:3]
        if height > width:
            raise ValueError(f"portrait video is not supported: {video_path}")

        intrinsics = self.resize_crop.transform_intrinsics(metadata["intrinsics"], height, width)
        video = torch.from_numpy(frames).permute(0, 3, 1, 2).float() / 127.5 - 1.0
        video = self.resize_crop(video).clamp(-1.0, 1.0).permute(1, 0, 2, 3).contiguous()
        return {
            "video": video,
            "text": metadata["text"],
            "intrinsics": torch.from_numpy(intrinsics).float(),
            "poses": torch.from_numpy(metadata["poses"][:frame_count]).float(),
            "sample_id": str(sample["sample_id"]),
        }
