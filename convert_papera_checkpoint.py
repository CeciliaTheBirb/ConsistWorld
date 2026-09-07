"""Convert a legacy PaperA checkpoint to the clean release architecture."""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch

from papera.checkpoints import load_full_state_dict


# These names identify legacy checkpoint-only extensions. They are not part of
# the release model; any other unexpected tensor makes conversion fail.
_REMOVED_COMPONENTS = frozenset(
    {
        "mem_key_marker",
        "mem_proj",
        "mem_patch_embed",
        "mem_feat_embed",
        "mem_select",
        "mem_null_logit",
        "ore",
        "ore_v",
        "ore_o",
        "ore_film",
        "scope_pe",
        "evidence_router",
    }
)


def _is_removed_extension(key: str) -> bool:
    return any(component in _REMOVED_COMPONENTS for component in key.split("."))


def convert(source_checkpoint: Path, reference_checkpoint: Path, output_checkpoint: Path) -> None:
    """Write a checkpoint that strictly matches the public model ABI.

    The clean Stage-1 checkpoint supplies the complete key and shape contract.
    Conversion keeps every matching core tensor and discards only declared legacy
    extensions, preventing accidental release of a hidden branch.
    """
    if output_checkpoint.exists():
        raise FileExistsError(f"refusing to overwrite an existing checkpoint: {output_checkpoint}")

    source_state = load_full_state_dict(str(source_checkpoint))
    reference_state = load_full_state_dict(str(reference_checkpoint))
    expected_keys = set(reference_state)
    source_keys = set(source_state)

    missing = sorted(expected_keys - source_keys)
    if missing:
        raise RuntimeError(
            f"source checkpoint is missing {len(missing)} core tensors; first: {missing[:8]}"
        )

    shape_mismatches = [
        key
        for key in sorted(expected_keys)
        if tuple(source_state[key].shape) != tuple(reference_state[key].shape)
    ]
    if shape_mismatches:
        details = [
            f"{key}: source={tuple(source_state[key].shape)} reference={tuple(reference_state[key].shape)}"
            for key in shape_mismatches[:8]
        ]
        raise RuntimeError(
            f"source checkpoint has {len(shape_mismatches)} incompatible core tensors: "
            + "; ".join(details)
        )

    extensions = sorted(source_keys - expected_keys)
    unknown = [key for key in extensions if not _is_removed_extension(key)]
    if unknown:
        raise RuntimeError(
            "refusing to discard unknown checkpoint tensors; first: "
            + ", ".join(unknown[:12])
        )

    clean_state = {key: source_state[key] for key in reference_state}
    output_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_checkpoint.with_name(f"{output_checkpoint.name}.tmp")
    try:
        torch.save(clean_state, temporary)
        os.replace(temporary, output_checkpoint)
    except Exception:
        if temporary.exists():
            temporary.unlink()
        raise

    print(
        f"[convert] wrote {len(clean_state)} core tensors to {output_checkpoint}; "
        f"removed {len(extensions)} declared legacy tensors",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert legacy PaperA weights to the release ABI")
    parser.add_argument("--source", required=True, help="legacy full-model checkpoint")
    parser.add_argument("--reference", required=True, help="clean Stage-1 SR-v3 model_full.pt")
    parser.add_argument("--out", required=True, help="new clean model_full.pt path")
    args = parser.parse_args()
    convert(Path(args.source), Path(args.reference), Path(args.out))


if __name__ == "__main__":
    main()
