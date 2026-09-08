"""Deterministic per-view P-Mem retrieval for ConsistWorld."""
import numpy as np
import torch

_GRID_W, _GRID_H = 8, 5
_DEPTH_FACTORS = (0.5, 1.0, 2.0, 4.0)
_AXIS_WEIGHT = 0.2
_NMS_ROT_DEG = 10.0
_NMS_TRANS_FRAC = 0.05


def _as_np_c2ws(abs_c2ws) -> np.ndarray:
    if isinstance(abs_c2ws, torch.Tensor):
        return abs_c2ws.detach().cpu().double().numpy()
    if isinstance(abs_c2ws, (list, tuple)) and abs_c2ws and isinstance(abs_c2ws[0], torch.Tensor):
        return torch.stack([pose.detach().cpu() for pose in abs_c2ws]).double().numpy()
    return np.asarray(abs_c2ws, dtype=np.float64)


def _frustum_points(c2w: np.ndarray, K4: np.ndarray, video_hw, depths) -> np.ndarray:
    """World points on a pixel grid x depth samples of one camera. [G*D, 3]"""
    h, w = int(video_hw[0]), int(video_hw[1])
    fx, fy, cx, cy = (float(x) for x in K4)
    us = np.linspace(0.5, w - 0.5, _GRID_W)
    vs = np.linspace(0.5, h - 0.5, _GRID_H)
    uu, vv = np.meshgrid(us, vs)
    dirs = np.stack([(uu - cx) / fx, (vv - cy) / fy, np.ones_like(uu)], axis=-1).reshape(-1, 3)
    dirs = dirs @ c2w[:3, :3].T                       # camera->world rotation
    o = c2w[:3, 3]
    return (o[None, None] + dirs[None, :, :] * np.asarray(depths)[:, None, None]).reshape(-1, 3)


def _visible_fraction(points_w: np.ndarray, c2w: np.ndarray, K4: np.ndarray, video_hw) -> float:
    """Fraction of world points that project inside the camera image (z>0)."""
    h, w = int(video_hw[0]), int(video_hw[1])
    fx, fy, cx, cy = (float(x) for x in K4)
    R, t = c2w[:3, :3], c2w[:3, 3]
    pc = (points_w - t[None]) @ R                     # world->camera (R is orthonormal)
    z = pc[:, 2]
    ok = z > 1e-6
    u = fx * pc[:, 0] / np.where(ok, z, 1.0) + cx
    v = fy * pc[:, 1] / np.where(ok, z, 1.0) + cy
    ok &= (u >= 0) & (u < w) & (v >= 0) & (v < h)
    return float(ok.mean())


def _pose_close(a: np.ndarray, b: np.ndarray, scene_scale: float) -> bool:
    cos = (np.trace(a[:3, :3].T @ b[:3, :3]) - 1.0) / 2.0
    ang = np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))
    dt = np.linalg.norm(a[:3, 3] - b[:3, 3])
    return ang < _NMS_ROT_DEG and dt < _NMS_TRANS_FRAC * scene_scale


def rank_mem_candidates(
    query_c2w,
    query_intrinsics,
    cand_c2ws,
    cand_intrinsics,
    video_hw,
    r: int = 1,
    scene_scale: float = None,
):
    """Rank memory candidates for ONE query camera (inference-side counterpart of
    retrieve_gt_mem_pairs): same symmetric frustum-overlap + optical-axis score
    and the same pose-NMS, so a target view picks at inference what training
    taught it to read. Returns (chosen_indices best-first, all scores).
    """
    q = _as_np_c2ws(query_c2w)
    qK = np.asarray(query_intrinsics.detach().cpu() if isinstance(query_intrinsics, torch.Tensor)
                    else query_intrinsics, dtype=np.float64).reshape(-1)[:4]
    cs = [_as_np_c2ws(c) for c in cand_c2ws]
    cKs = [np.asarray(i.detach().cpu() if isinstance(i, torch.Tensor) else i,
                      dtype=np.float64).reshape(-1)[:4] for i in cand_intrinsics]
    if scene_scale is None:
        centers = np.stack([q[:3, 3]] + [c[:3, 3] for c in cs])
        scene_scale = float(np.linalg.norm(centers[None] - centers[:, None], axis=-1).max())
        if scene_scale <= 1e-8:
            scene_scale = 1.0
    else:
        scene_scale = float(scene_scale)
        if scene_scale <= 1e-8:
            scene_scale = 1.0
    depths = [f * scene_scale for f in _DEPTH_FACTORS]
    q_frustum = _frustum_points(q, qK, video_hw, depths)
    scores = []
    for c, cK in zip(cs, cKs):
        a = _visible_fraction(_frustum_points(c, cK, video_hw, depths), q, qK, video_hw)
        b = _visible_fraction(q_frustum, c, cK, video_hw)
        cos = float(np.dot(c[:3, :3][:, 2], q[:3, :3][:, 2]))
        scores.append(0.5 * (a + b) + _AXIS_WEIGHT * max(0.0, cos))
    ranked = sorted(range(len(cs)), key=lambda i: (-scores[i], i))
    chosen = []
    for i in ranked:
        if len(chosen) >= int(r):
            break
        if any(_pose_close(cs[i], cs[j], scene_scale) for j in chosen):
            continue
        chosen.append(i)
    for i in ranked:                                       # backfill NMS-suppressed
        if len(chosen) >= int(r):
            break
        if i not in chosen:
            chosen.append(i)
    return chosen, scores


def retrieve_gt_mem_pairs(
    abs_c2ws_list,
    intrinsics_list,
    video_hw,
    num_chunks: int,
    lw: int = 4,
    win: int = 1,
    r_per_view: int = 1,
) -> torch.Tensor:
    """Per-(target-chunk, target-view) GT retrieval sets over the sample's own views.

    abs_c2ws_list: K per-view absolute c2w on the latent frame grid,
        each [num_chunks*lw, 4, 4] (clip_cache_assembler tgt_abs_list).
    intrinsics_list: K per-view pixel-space [fx, fy, cx, cy] on video_hw.
    Returns LongTensor [num_chunks, K, r_per_view, 2] of (view w, chunk c),
    -1-padded; out[p, v] holds TARGET view v's own top-r_per_view picks (M1.1:
    per-view read, no shared union — so one view's strong hit can't crowd out
    another's, and a view only reads memory relevant to itself).
    """
    K = len(abs_c2ws_list)
    N = int(num_chunks)
    c2ws = [_as_np_c2ws(a) for a in abs_c2ws_list]
    Ks = [np.asarray(i.detach().cpu() if isinstance(i, torch.Tensor) else i,
                     dtype=np.float64).reshape(-1)[:4] for i in intrinsics_list]
    mid = [[c2ws[w][c * lw + lw // 2] for c in range(N)] for w in range(K)]

    centers = np.stack([mid[w][c][:3, 3] for w in range(K) for c in range(N)])
    scene_scale = float(np.linalg.norm(centers[None] - centers[:, None], axis=-1).max())
    if scene_scale <= 1e-8:
        scene_scale = 1.0
    depths = [f * scene_scale for f in _DEPTH_FACTORS]

    frustum = [[_frustum_points(mid[w][c], Ks[w], video_hw, depths) for c in range(N)]
               for w in range(K)]

    rp = int(r_per_view)
    out = torch.full((N, K, rp, 2), -1, dtype=torch.long)
    for p in range(N):
        c_max = p - win - 1                            # §3.6: chunk_id < p - win
        if c_max < 0:
            continue
        cands = [(w, c) for w in range(K) for c in range(c_max + 1)]
        # score[v][cand]: symmetric frustum overlap + optical-axis alignment,
        # scored per TARGET view v (relevance of past chunk (w,c) to view v@p).
        score = {v: {} for v in range(K)}
        for (w, c) in cands:
            for v in range(K):
                q = mid[v][p]
                a = _visible_fraction(frustum[w][c], q, Ks[v], video_hw)
                b = _visible_fraction(frustum[v][p], mid[w][c], Ks[w], video_hw)
                cos = float(np.dot(mid[w][c][:3, :3][:, 2], q[:3, :3][:, 2]))
                score[v][(w, c)] = 0.5 * (a + b) + _AXIS_WEIGHT * max(0.0, cos)
        # M1.1: each target view keeps ITS OWN top-r_per_view (pose-NMS within
        # the view), independent of the other views — no shared union/truncate.
        for v in range(K):
            ranked = sorted(cands, key=lambda wc: (-score[v][wc], wc))
            chosen = []
            for (w, c) in ranked:
                if len(chosen) >= rp:
                    break
                if any(_pose_close(mid[w][c], mid[cw][cc], scene_scale) for cw, cc in chosen):
                    continue
                chosen.append((w, c))
            for (w, c) in ranked:                      # backfill NMS-suppressed
                if len(chosen) >= rp:
                    break
                if (w, c) not in chosen:
                    chosen.append((w, c))
            for j, (w, c) in enumerate(chosen):
                if c > c_max:
                    raise AssertionError(f"P-Mem leak: chunk {c} >= p-win for p={p}, win={win}")
                out[p, v, j, 0] = w
                out[p, v, j, 1] = c
    return out


def _token_dirs(c2w: np.ndarray, intrinsics, video_hw, grid_hw) -> np.ndarray:
    """World-space rays through the centers of one latent token grid."""
    height, width = (int(value) for value in video_hw)
    grid_h, grid_w = (int(value) for value in grid_hw)
    fx, fy, cx, cy = (float(value) for value in intrinsics)
    us = (np.arange(grid_w) + 0.5) * width / grid_w
    vs = (np.arange(grid_h) + 0.5) * height / grid_h
    uu, vv = np.meshgrid(us, vs)
    rays = np.stack(
        [(uu - cx) / fx, (vv - cy) / fy, np.ones_like(uu)], axis=-1
    ).reshape(-1, 3)
    rays = rays @ c2w[:3, :3].T
    return rays / np.linalg.norm(rays, axis=1, keepdims=True)


def _seen_score(rays: np.ndarray, c2w: np.ndarray, intrinsics, video_hw, grid_hw) -> np.ndarray:
    """Score whether each world-space ray falls within a camera image.

    The transition spans half a latent token on each side of the image border.
    This makes a frame cover its own outermost token with score exactly one.
    """
    height, width = (int(value) for value in video_hw)
    grid_h, grid_w = (int(value) for value in grid_hw)
    fx, fy, cx, cy = (float(value) for value in intrinsics)
    camera_rays = rays @ c2w[:3, :3]
    depth = camera_rays[:, 2]
    valid = depth > 1e-6
    depth = np.where(valid, depth, 1.0)
    u = fx * camera_rays[:, 0] / depth + cx
    v = fy * camera_rays[:, 1] / depth + cy
    half_token = 0.5 * min(width / grid_w, height / grid_h)
    border_distance = np.minimum.reduce([u, width - u, v, height - v])
    score = np.clip((border_distance + half_token) / (2.0 * half_token), 0.0, 1.0)
    return np.where(valid, score, 0.0)


def token_history_coverage(target_c2ws, target_intrinsics, history_c2ws, history_intrinsics,
                           video_hw, grid_hw) -> np.ndarray:
    """Per-token clean-evidence coverage for each target latent frame."""
    targets = _as_np_c2ws(target_c2ws)
    if targets.ndim == 2:
        targets = targets[None]
    coverage = np.zeros((len(targets), int(grid_hw[0]) * int(grid_hw[1])), dtype=np.float64)
    histories = [_as_np_c2ws(pose) for pose in history_c2ws]
    if not histories:
        return coverage
    if len(histories) != len(history_intrinsics):
        raise ValueError("history camera poses and intrinsics must have the same length")

    for frame_index, target_pose in enumerate(targets):
        rays = _token_dirs(target_pose, target_intrinsics, video_hw, grid_hw)
        for history_pose, intrinsics in zip(histories, history_intrinsics):
            np.maximum(
                coverage[frame_index],
                _seen_score(rays, history_pose, intrinsics, video_hw, grid_hw),
                out=coverage[frame_index],
            )
    return coverage


def token_xnow_gate(target_c2ws, target_intrinsics, history_c2ws, history_intrinsics,
                    video_hw, grid_hw, sigma: float, peer_c2ws, peer_intrinsics) -> np.ndarray:
    """Two-factor gate for equal-time cross-view attention.

    ``coverage`` measures clean evidence visible to this target.  ``peer_sees``
    measures the peer-current chunk's coverage of the same target ray.  The gate
    is zero only where clean evidence already covers a ray or peers cannot help.
    """
    coverage = token_history_coverage(
        target_c2ws, target_intrinsics, history_c2ws, history_intrinsics, video_hw, grid_hw
    )
    if peer_c2ws:
        peer = token_history_coverage(
            target_c2ws, target_intrinsics, peer_c2ws, peer_intrinsics, video_hw, grid_hw
        )
    else:
        peer = np.ones_like(coverage)
    # Preserve the exact single-factor result when peer coverage is one.
    closed = (1.0 - peer) + coverage * peer
    return np.clip(1.0 - float(sigma) * closed, 0.0, 1.0)
