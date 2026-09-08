#!/usr/bin/env bash
# Apply the minimal ConsistWorld trajectory patch to the pinned Infinigen checkout.
set -euo pipefail

EXPECTED_REVISION="25a7d284dc21fdea6525cdfc6be4c10e4d79f28f"
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PATCH_FILE="$SCRIPT_DIR/patches/0001-michar-sweep.patch"

usage() {
    cat <<'EOF'
Usage: apply_michar_patch.sh --infinigen_root PATH

The target must be the Princeton Infinigen repository at the pinned revision
documented in infinigen/README.md. The script is idempotent when this exact
ConsistWorld patch has already been applied.
EOF
}

INFINIGEN_ROOT="${INFINIGEN_ROOT:-}"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --infinigen_root) INFINIGEN_ROOT="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
done

[[ -n "$INFINIGEN_ROOT" ]] || { usage >&2; exit 2; }
if ! git -C "$INFINIGEN_ROOT" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    echo "Not an Infinigen git checkout: $INFINIGEN_ROOT" >&2
    exit 2
fi
TARGET="$INFINIGEN_ROOT/src/infinigen/core/placement/camera_trajectories.py"
[[ -f "$TARGET" ]] || { echo "Missing target source file: $TARGET" >&2; exit 2; }

if grep -q '^def _michar_sweep_walk' "$TARGET"; then
    echo "ConsistWorld michar patch is already present: $TARGET"
    exit 0
fi

REVISION=$(git -C "$INFINIGEN_ROOT" rev-parse HEAD)
if [[ "$REVISION" != "$EXPECTED_REVISION" ]]; then
    cat >&2 <<EOF
Refusing to patch an unpinned Infinigen revision.
Expected: $EXPECTED_REVISION
Found:    $REVISION
Checkout the documented revision, then rerun this script.
EOF
    exit 2
fi

git -C "$INFINIGEN_ROOT" apply --check "$PATCH_FILE"
git -C "$INFINIGEN_ROOT" apply "$PATCH_FILE"
echo "Applied ConsistWorld michar patch to $INFINIGEN_ROOT"
