import numpy as np
import torch
import torch.nn.functional as torch_nn_functional


def round_down_4n_plus_1(frame_num: int) -> int:
    if frame_num <= 0:
        raise ValueError("frame_num must be positive")
    return ((frame_num - 1) // 4) * 4 + 1


def broadcast_intrinsics_to_length(intrinsics: torch.Tensor, target_len: int) -> torch.Tensor:
    """Broadcast a 1D or (1, 4) pixel-space [fx, fy, cx, cy] tensor to (target_len, 4).

    Accepts:
      * 1D shape (4,)
      * 2D shape (1, 4)
    Anything else raises. Used by both the AR pipeline and the latent-pipe
    cache builder and inference share one canonical broadcast contract.
    """
    if intrinsics.ndim == 1:
        intrinsics = intrinsics.unsqueeze(0)
    if intrinsics.shape != (1, 4):
        raise ValueError(
            "Expected transformed intrinsics with shape [4] (stored internally as [1, 4]), "
            f"got {tuple(intrinsics.shape)}"
        )
    return intrinsics.repeat(target_len, 1)


class ResizeCropAspectCenter:
    def __init__(self, target_height: int, target_width: int):
        self.th = target_height
        self.tw = target_width

    def get_resize_crop_params(self, height: int, width: int):
        orig_ar = width / height
        target_ar = self.tw / self.th
        if orig_ar < target_ar:
            scale = self.tw / width
        else:
            scale = self.th / height
        new_w = int(round(width * scale))
        new_h = int(round(height * scale))
        top = (new_h - self.th) // 2
        left = (new_w - self.tw) // 2
        return new_h, new_w, top, left

    def transform_intrinsics(self, intrinsics: np.ndarray, height: int, width: int) -> np.ndarray:
        intrinsics = np.array(intrinsics, dtype=np.float32, copy=True)
        new_h, new_w, top, left = self.get_resize_crop_params(height, width)
        scale_x = new_w / float(width)
        scale_y = new_h / float(height)

        if intrinsics.shape[-2:] == (3, 3):
            fx = intrinsics[..., 0, 0] * width
            fy = intrinsics[..., 1, 1] * height
            cx = intrinsics[..., 0, 2] * width
            cy = intrinsics[..., 1, 2] * height
        elif intrinsics.shape[-1] == 4:
            fx = intrinsics[..., 0]
            fy = intrinsics[..., 1]
            cx = intrinsics[..., 2]
            cy = intrinsics[..., 3]
        else:
            raise ValueError(
                "Unsupported intrinsics format; expected [..., 3, 3] or [..., 4], "
                f"got shape {intrinsics.shape}"
            )

        fx = fx * scale_x
        fy = fy * scale_y
        cx = cx * scale_x - left
        cy = cy * scale_y - top
        return np.stack([fx, fy, cx, cy], axis=-1).astype(np.float32)

    def __call__(self, video: torch.Tensor) -> torch.Tensor:
        if video.dim() != 4:
            raise ValueError(f"Expected video tensor with shape [T, C, H, W], got {video.shape}")
        _, _, height, width = video.shape
        new_h, new_w, top, left = self.get_resize_crop_params(height, width)
        video = torch_nn_functional.interpolate(
            video,
            size=(new_h, new_w),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )
        return video[:, :, top : top + self.th, left : left + self.tw]


def extract_spatialvid_meta(meta: dict) -> dict:
    intrinsics = np.asarray(meta["intrinsics_vipe"], dtype=np.float32)
    if intrinsics.shape != (3, 3):
        raise ValueError(f"intrinsics_vipe must have shape [3, 3], got {intrinsics.shape}")
    poses_w2c = np.asarray(meta["poses_w2c_vipe"], dtype=np.float32)
    if poses_w2c.ndim != 3 or poses_w2c.shape[-2:] != (4, 4):
        raise ValueError(f"poses_w2c_vipe must have shape [T, 4, 4], got {poses_w2c.shape}")
    caption = meta.get("caption", {})
    text = (caption.get("SceneDescription") or caption.get("SceneSummary") or "").strip()
    return {
        "text": text,
        "intrinsics": intrinsics,
        "poses": np.linalg.inv(poses_w2c).astype(np.float32),
    }
