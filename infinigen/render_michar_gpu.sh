#!/usr/bin/env bash
# Render the eight ConsistWorld michar camera rigs one at a time.
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: render_michar_gpu.sh --infinigen_root DIR --scene_dir DIR --seed SEED [options]

Required:
  --infinigen_root DIR  Patched Princeton Infinigen checkout (or set INFINIGEN_ROOT).
  --scene_dir DIR       Scene created by prepare_scene_cpu.sh.
  --seed SEED           The seed used for preparation.

Options:
  --scene_type NAME     Infinigen nature configuration. Default: plain
  --gpu_id ID           Set CUDA_VISIBLE_DEVICES for every sequential rig render.
  --frames N            Inclusive frame count. Default: 144
  --width N             Render width. Default: 960
  --height N            Render height. Default: 540
  --samples N           Cycles samples. Default: 64
  --python PATH         Python executable for Infinigen. Default: python

This intentionally renders rigs 0..7 sequentially. Successful output is
collected under <scene_dir>/frames/. The patched checkout supplies the
ConsistWorld mi-character trajectory; sequential rendering keeps this public
workflow independent of the production pipeline's parallel-render machinery.
EOF
}

INFINIGEN_ROOT="${INFINIGEN_ROOT:-}"
SCENE_DIR=""
SEED=""
SCENE_TYPE="plain"
GPU_ID=""
FRAMES=144
WIDTH=960
HEIGHT=540
SAMPLES=64
PYTHON_BIN="python"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --infinigen_root) INFINIGEN_ROOT="$2"; shift 2 ;;
        --scene_dir) SCENE_DIR="$2"; shift 2 ;;
        --seed) SEED="$2"; shift 2 ;;
        --scene_type) SCENE_TYPE="$2"; shift 2 ;;
        --gpu_id) GPU_ID="$2"; shift 2 ;;
        --frames) FRAMES="$2"; shift 2 ;;
        --width) WIDTH="$2"; shift 2 ;;
        --height) HEIGHT="$2"; shift 2 ;;
        --samples) SAMPLES="$2"; shift 2 ;;
        --python) PYTHON_BIN="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
done

is_positive_integer() { [[ "$1" =~ ^[1-9][0-9]*$ ]]; }
[[ -n "$INFINIGEN_ROOT" && -n "$SCENE_DIR" && -n "$SEED" ]] || { usage >&2; exit 2; }
[[ "$SEED" =~ ^[A-Za-z0-9._-]+$ ]] || { echo "--seed may contain only A-Z, a-z, 0-9, ., _, and -" >&2; exit 2; }
[[ "$SCENE_TYPE" =~ ^[A-Za-z0-9_-]+$ ]] || { echo "--scene_type may contain only A-Z, a-z, 0-9, _, and -" >&2; exit 2; }
for value in "$FRAMES" "$WIDTH" "$HEIGHT" "$SAMPLES"; do
    is_positive_integer "$value" || { echo "numeric options must be positive integers" >&2; exit 2; }
done
[[ -f "$INFINIGEN_ROOT/src/infinigen_examples/generate_nature.py" ]] || {
    echo "Not an Infinigen source checkout: $INFINIGEN_ROOT" >&2
    exit 2
}
grep -q '^def _michar_sweep_walk' "$INFINIGEN_ROOT/src/infinigen/core/placement/camera_trajectories.py" || {
    echo "The ConsistWorld michar patch is missing. Run infinigen/apply_michar_patch.sh first." >&2
    exit 2
}
[[ -f "$SCENE_DIR/fine/scene.blend" ]] || { echo "Missing prepared scene: $SCENE_DIR/fine/scene.blend" >&2; exit 2; }
[[ -f "$SCENE_DIR/consistworld_michar_manifest.json" ]] || {
    echo "No ConsistWorld preparation manifest at $SCENE_DIR. Re-run prepare_scene_cpu.sh before rendering." >&2
    exit 2
}

LOG_DIR="$SCENE_DIR/logs"
mkdir -p "$LOG_DIR"
export PYTHONPATH="$INFINIGEN_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export INFNGN_MICHAR_SWEEP=1
unset INFNGN_REBAKE_TRAJ || true
if [[ -n "$GPU_ID" ]]; then
    export CUDA_VISIBLE_DEVICES="$GPU_ID"
fi

COMMON_CONFIGS=("$SCENE_TYPE" "no_creatures.gin" "fast_terrain_assets.gin" "high_quality_terrain.gin")
COMMON_OVERRIDES=(
    "render.render_image_func=@full/render_image"
    "LOG_DIR=$LOG_DIR"
    "fine_terrain.mesher_backend=OcMesher"
    "OcMesher.pixels_per_cube=16"
    "OcMesher.coarse_count=100000"
    "configure_render_cycles.num_samples=$SAMPLES"
    "configure_render_cycles.denoise=True"
    "full/render_image.passes_to_save=[]"
    "execute_tasks.generate_resolution=($WIDTH,$HEIGHT)"
    "camera.spawn_camera_rigs.n_camera_rigs=8"
    "compose_nature.ground_creatures_chance=0.0"
    "execute_tasks.frame_range=[1,$FRAMES]"
    "execute_tasks.resample_idx=0"
    "execute_tasks.point_trajectory_src_frame=1"
)

count_files() {
    local directory="$1"
    local pattern="$2"
    if [[ ! -d "$directory" ]]; then
        printf '0\n'
        return
    fi
    find "$directory" -maxdepth 1 -type f -name "$pattern" -printf . | wc -c
}

for rig in $(seq 0 7); do
    image_count=$(count_files "$SCENE_DIR/frames/Image/camera_0" "Image_${rig}_0_*_0.png")
    camview_count=$(count_files "$SCENE_DIR/frames/camview/camera_0" "camview_${rig}_0_*_0.npz")
    if [[ "$image_count" -eq "$FRAMES" && "$camview_count" -eq "$FRAMES" ]]; then
        echo "[render] rig $rig already complete"
        continue
    fi
    if [[ "$image_count" -ne 0 || "$camview_count" -ne 0 ]]; then
        echo "Partial output exists for rig $rig (RGB=$image_count camview=$camview_count); refusing to overwrite it." >&2
        exit 1
    fi

    render_dir="$SCENE_DIR/frames_${rig}_0_0001_0"
    if [[ -e "$render_dir" ]]; then
        echo "Incomplete Infinigen render directory exists: $render_dir; inspect it before retrying." >&2
        exit 1
    fi
    echo "[render] rig $rig/7"
    "$PYTHON_BIN" -m infinigen_examples.generate_nature -- \
        --input_folder "$SCENE_DIR/fine" --output_folder "$render_dir" \
        --seed "$SEED" --task render --task_uniqname "michar_render_${rig}" \
        -g "${COMMON_CONFIGS[@]}" \
        -p "${COMMON_OVERRIDES[@]}" "execute_tasks.camera_id=[$rig,0]"

    image_count=$(count_files "$SCENE_DIR/frames/Image/camera_0" "Image_${rig}_0_*_0.png")
    camview_count=$(count_files "$SCENE_DIR/frames/camview/camera_0" "camview_${rig}_0_*_0.npz")
    if [[ "$image_count" -ne "$FRAMES" || "$camview_count" -ne "$FRAMES" ]]; then
        echo "Rig $rig finished without the expected frame count (RGB=$image_count camview=$camview_count)." >&2
        exit 1
    fi
done

echo "[render] complete: $SCENE_DIR/frames"
