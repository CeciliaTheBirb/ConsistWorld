#!/usr/bin/env bash
# Prepare one ConsistWorld Infinigen scene. Rendering is deliberately separate.
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: prepare_scene_cpu.sh --infinigen_root DIR --work_dir DIR --scene_id ID --seed SEED [options]

Required:
  --infinigen_root DIR  Patched Princeton Infinigen checkout (or set INFINIGEN_ROOT).
  --work_dir DIR        Parent directory for generated scenes.
  --scene_id ID         Safe output directory name, for example desert_000001.
  --seed SEED           Infinigen scene seed.

Options:
  --scene_type NAME     Infinigen nature configuration. Default: plain
  --frames N            Inclusive frame count. Default: 144
  --width N             Render width stored in the scene. Default: 960
  --height N            Render height stored in the scene. Default: 540
  --samples N           Cycles samples configured for the later render. Default: 64
  --python PATH         Python executable for Infinigen. Default: python

The script runs coarse -> fine_terrain -> populate only. It does not launch a
render task. It never removes an existing incomplete stage; inspect or move
that scene before retrying.
EOF
}

INFINIGEN_ROOT="${INFINIGEN_ROOT:-}"
WORK_DIR=""
SCENE_ID=""
SEED=""
SCENE_TYPE="plain"
FRAMES=144
WIDTH=960
HEIGHT=540
SAMPLES=64
PYTHON_BIN="python"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --infinigen_root) INFINIGEN_ROOT="$2"; shift 2 ;;
        --work_dir) WORK_DIR="$2"; shift 2 ;;
        --scene_id) SCENE_ID="$2"; shift 2 ;;
        --seed) SEED="$2"; shift 2 ;;
        --scene_type) SCENE_TYPE="$2"; shift 2 ;;
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
[[ -n "$INFINIGEN_ROOT" && -n "$WORK_DIR" && -n "$SCENE_ID" && -n "$SEED" ]] || { usage >&2; exit 2; }
[[ "$SCENE_ID" =~ ^[A-Za-z0-9._-]+$ ]] || { echo "--scene_id may contain only A-Z, a-z, 0-9, ., _, and -" >&2; exit 2; }
[[ "$SEED" =~ ^[A-Za-z0-9._-]+$ ]] || { echo "--seed may contain only A-Z, a-z, 0-9, ., _, and -" >&2; exit 2; }
[[ "$SCENE_TYPE" =~ ^[A-Za-z0-9_-]+$ ]] || { echo "--scene_type may contain only A-Z, a-z, 0-9, _, and -" >&2; exit 2; }
for value in "$FRAMES" "$WIDTH" "$HEIGHT" "$SAMPLES"; do
    is_positive_integer "$value" || { echo "numeric options must be positive integers" >&2; exit 2; }
done
[[ -f "$INFINIGEN_ROOT/src/infinigen_examples/generate_nature.py" ]] || {
    echo "Not an Infinigen source checkout: $INFINIGEN_ROOT" >&2
    exit 2
}
TRAJECTORY_SOURCE="$INFINIGEN_ROOT/src/infinigen/core/placement/camera_trajectories.py"
grep -q '^def _michar_sweep_walk' "$TRAJECTORY_SOURCE" || {
    echo "The ConsistWorld michar patch is missing. Run infinigen/apply_michar_patch.sh first." >&2
    exit 2
}

SCENE_DIR="$WORK_DIR/$SCENE_ID"
LOG_DIR="$SCENE_DIR/logs"
mkdir -p "$LOG_DIR"

export PYTHONPATH="$INFINIGEN_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
# Scene construction is the CPU phase of the recipe.  Keep a mixed CPU/GPU
# machine from accidentally reserving a render GPU while Blender prepares
# terrain and assets.
export CUDA_VISIBLE_DEVICES=""
export INFNGN_MICHAR_SWEEP=1
unset INFNGN_REBAKE_TRAJ || true

COMMON_CONFIGS=("$SCENE_TYPE" "no_creatures.gin" "fast_terrain_assets.gin" "high_quality_terrain.gin")
COMMON_OVERRIDES=(
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
    "execute_tasks.camera_id=[0,0]"
)

run_nature() {
    "$PYTHON_BIN" -m infinigen_examples.generate_nature -- "$@"
}

if [[ -f "$LOG_DIR/FINISH_coarse" && -f "$SCENE_DIR/coarse/scene.blend" ]]; then
    echo "[prepare] coarse already complete"
else
    [[ ! -e "$SCENE_DIR/coarse" ]] || {
        echo "Incomplete coarse stage exists at $SCENE_DIR/coarse; refusing to overwrite it." >&2
        exit 1
    }
    echo "[prepare] coarse"
    run_nature \
        --output_folder "$SCENE_DIR/coarse" --seed "$SEED" --task coarse --task_uniqname coarse \
        -g "${COMMON_CONFIGS[@]}" -p "${COMMON_OVERRIDES[@]}"
fi

if [[ -f "$LOG_DIR/FINISH_fineterrain" && -f "$SCENE_DIR/fine/scene.blend" ]]; then
    echo "[prepare] fine_terrain already complete"
else
    [[ -f "$SCENE_DIR/coarse/scene.blend" ]] || { echo "Missing coarse/scene.blend" >&2; exit 1; }
    [[ ! -e "$SCENE_DIR/fine" ]] || {
        echo "Incomplete fine stage exists at $SCENE_DIR/fine; refusing to overwrite it." >&2
        exit 1
    }
    echo "[prepare] fine_terrain"
    run_nature \
        --input_folder "$SCENE_DIR/coarse" --output_folder "$SCENE_DIR/fine" \
        --seed "$SEED" --task fine_terrain --task_uniqname fineterrain \
        -g "${COMMON_CONFIGS[@]}" -p "${COMMON_OVERRIDES[@]}"
fi

if [[ -f "$LOG_DIR/FINISH_populate" && -f "$SCENE_DIR/fine/scene.blend" ]]; then
    echo "[prepare] populate already complete"
else
    [[ -f "$LOG_DIR/FINISH_fineterrain" && -f "$SCENE_DIR/fine/scene.blend" ]] || {
        echo "fine_terrain did not finish successfully" >&2
        exit 1
    }
    echo "[prepare] populate"
    run_nature \
        --input_folder "$SCENE_DIR/fine" --output_folder "$SCENE_DIR/fine" \
        --seed "$SEED" --task populate --task_uniqname populate \
        -g "${COMMON_CONFIGS[@]}" -p "${COMMON_OVERRIDES[@]}"
fi

[[ -f "$LOG_DIR/FINISH_populate" && -f "$SCENE_DIR/fine/scene.blend" ]] || {
    echo "populate did not produce fine/scene.blend" >&2
    exit 1
}

printf '%s\n' \
    '{' \
    "  \"scene_id\": \"$SCENE_ID\"," \
    "  \"seed\": \"$SEED\"," \
    "  \"scene_type\": \"$SCENE_TYPE\"," \
    '  "trajectory": "michar_sweep",' \
    '  "camera_rigs": 8,' \
    "  \"frames\": $FRAMES," \
    "  \"resolution\": [$WIDTH, $HEIGHT]" \
    '}' > "$SCENE_DIR/consistworld_michar_manifest.json"
echo "[prepare] complete: $SCENE_DIR"
