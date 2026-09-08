#!/usr/bin/env bash
# Convert complete Infinigen scene folders to ConsistWorld SpatialVID/MultiCamData clips.
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: convert_batch.sh --scenes_dir DIR --out_dir DIR [options]

Options:
  --prefix NAME          Output scene prefix. Default: infngn_michar8
  --expected_views N     Require N converted views per scene. Default: 8
  --expected_frames N    Require at least N RGB and camview files for every rig. Default: 144
  --fps N                Output video FPS. Default: 16
  --caption TEXT         Shared caption written to every JSON.
  --python PATH          Python executable. Default: python
EOF
}

SCENES_DIR=""
OUT_DIR=""
PREFIX="infngn_michar8"
EXPECTED_VIEWS=8
EXPECTED_FRAMES=144
FPS=16
CAPTION="a procedural natural outdoor scene, multi-view camera exploration"
PYTHON_BIN="python"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --scenes_dir) SCENES_DIR="$2"; shift 2 ;;
        --out_dir) OUT_DIR="$2"; shift 2 ;;
        --prefix) PREFIX="$2"; shift 2 ;;
        --expected_views) EXPECTED_VIEWS="$2"; shift 2 ;;
        --expected_frames) EXPECTED_FRAMES="$2"; shift 2 ;;
        --fps) FPS="$2"; shift 2 ;;
        --caption) CAPTION="$2"; shift 2 ;;
        --python) PYTHON_BIN="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
done

[[ -n "$SCENES_DIR" && -n "$OUT_DIR" ]] || { usage >&2; exit 2; }
[[ -d "$SCENES_DIR" ]] || { echo "Scene directory does not exist: $SCENES_DIR" >&2; exit 2; }
[[ "$EXPECTED_VIEWS" =~ ^[1-9][0-9]*$ ]] || { echo "--expected_views must be a positive integer" >&2; exit 2; }
[[ "$EXPECTED_FRAMES" =~ ^[1-9][0-9]*$ ]] || { echo "--expected_frames must be a positive integer" >&2; exit 2; }
[[ "$FPS" =~ ^[1-9][0-9]*$ ]] || { echo "--fps must be a positive integer" >&2; exit 2; }
mkdir -p "$OUT_DIR"

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
converted=0
skipped=0
failed=0

shopt -s nullglob
for scene_dir in "$SCENES_DIR"/*; do
    [[ -d "$scene_dir" ]] || continue
    seed=$(basename -- "$scene_dir")
    camview_dir="$scene_dir/frames/camview/camera_0"
    image_dir="$scene_dir/frames/Image/camera_0"
    if [[ ! -d "$camview_dir" || ! -d "$image_dir" ]]; then
        echo "SKIP $seed: no frames/camview/camera_0 and frames/Image/camera_0"
        skipped=$((skipped + 1))
        continue
    fi

    complete=1
    for rig in $(seq 0 $((EXPECTED_VIEWS - 1))); do
        camviews=("$camview_dir"/camview_"$rig"_0_*.npz)
        images=("$image_dir"/Image_"$rig"_0_*.png)
        if (( ${#camviews[@]} < EXPECTED_FRAMES || ${#images[@]} < EXPECTED_FRAMES )); then
            echo "SKIP $seed: rig $rig incomplete (RGB=${#images[@]}, camview=${#camviews[@]})"
            complete=0
            break
        fi
    done
    [[ $complete -eq 1 ]] || { skipped=$((skipped + 1)); continue; }

    scene_name="${PREFIX}_${seed}"
    existing=0
    complete_output=1
    for view in $(seq 1 "$EXPECTED_VIEWS"); do
        stem="$OUT_DIR/${scene_name}__cam$(printf '%02d' "$view")"
        if [[ -f "$stem.mp4" && -f "$stem.json" ]]; then
            continue
        fi
        complete_output=0
        if [[ -e "$stem.mp4" || -e "$stem.json" ]]; then
            existing=1
        fi
    done
    if [[ "$complete_output" -eq 1 ]]; then
        echo "SKIP $seed: output already exists"
        skipped=$((skipped + 1))
        continue
    fi
    if [[ "$existing" -eq 1 ]]; then
        echo "FAIL $seed: partial converted output exists; refusing to overwrite it" >&2
        failed=$((failed + 1))
        continue
    fi
    if "$PYTHON_BIN" "$SCRIPT_DIR/infinigen_to_multicam.py" \
        --scene_dir "$scene_dir" --out_dir "$OUT_DIR" --scene_name "$scene_name" \
        --caption "$CAPTION" --fps "$FPS" --expected_views "$EXPECTED_VIEWS"; then
        echo "OK   $seed"
        converted=$((converted + 1))
    else
        echo "FAIL $seed" >&2
        failed=$((failed + 1))
    fi
done

echo "converted=$converted skipped=$skipped failed=$failed"
[[ $failed -eq 0 ]]
