# ConsistWorld Infinigen Data Workflow

This directory provides the small, reproducible part of the ConsistWorld Infinigen
pipeline:

1. prepare terrain and assets on CPU;
2. render eight camera rigs sequentially on one GPU at a time;
3. convert RGB frames and camera metadata to the SpatialVID/MultiCamData format;
4. validate the converted clips before ConsistWorld cache construction.

It intentionally does not include the private job supervisors, machine-specific
paths, background processes, or destructive cleanup scripts from the production
pipeline.

## Prerequisites

Start from a clean Princeton Infinigen checkout at the pinned revision:

```bash
git clone https://github.com/princeton-vl/infinigen.git /path/to/infinigen
git -C /path/to/infinigen checkout 25a7d284dc21fdea6525cdfc6be4c10e4d79f28f
bash /path/to/ConsistWorld/infinigen/apply_michar_patch.sh \
  --infinigen_root /path/to/infinigen
```

The patch adds only the `INFNGN_MICHAR_SWEEP=1` trajectory. It is idempotent
when already applied and refuses a checkout at another revision. Follow the
upstream Infinigen installation instructions for Blender, Python packages, and
renderer dependencies.

## Generate One Scene

Run CPU scene preparation first. It creates `coarse/`, `fine/`, logs, and a
manifest, but does not render frames:

```bash
bash prepare_scene_cpu.sh \
  --infinigen_root /path/to/infinigen \
  --work_dir /path/to/infinigen_work \
  --scene_id plain_000001 --seed 100001 \
  --scene_type plain --frames 144 --width 960 --height 540 --samples 64
```

The preparation script sets `CUDA_VISIBLE_DEVICES=""` to prevent terrain and
asset construction from accidentally reserving a render GPU. It refuses to
overwrite an incomplete stage.

Render the prepared scene. Rigs 0 through 7 run sequentially, so the public
workflow does not require the private production parallel-render machinery.
`--gpu_id` is optional and maps to `CUDA_VISIBLE_DEVICES`.

```bash
bash render_michar_gpu.sh \
  --infinigen_root /path/to/infinigen \
  --scene_dir /path/to/infinigen_work/plain_000001 \
  --seed 100001 --gpu_id 0 \
  --scene_type plain --frames 144 --width 960 --height 540 --samples 64
```

The result is the standard Infinigen hierarchy:

```text
<scene>/frames/Image/camera_0/Image_<rig>_0_<frame>_0.png
<scene>/frames/camview/camera_0/camview_<rig>_0_<frame>_0.npz
```

The mi-character trajectory keeps the eight rigs at the validated base-view
location and performs paired yaw/pitch sweeps along eight 45-degree axes. It is
the minimal earlier ConsistWorld trajectory, not a replacement for the other private
trajectory families.

## Convert And Validate

Convert every complete scene below a work directory to the common ConsistWorld data
contract. The output contains one MP4 and JSON pair per rig, named
`<prefix>_<seed>__cam01` through `__cam08`.

```bash
bash convert_batch.sh \
  --scenes_dir /path/to/infinigen_work \
  --out_dir /path/to/rendered_revisit_data \
  --prefix infngn_michar8 --expected_views 8 --expected_frames 144 --fps 16
```

The converter changes Blender camera-to-world poses to the OpenCV convention,
writes normalized intrinsics, checks for duplicate or mismatched frame records,
and writes 960x540 MP4 files without implicit macroblock padding.

Validate the result before building a ConsistWorld cache:

```bash
python validate_multicam.py \
  --data_root /path/to/rendered_revisit_data \
  --expected_views 8 --expected_frames 141
```

The renderer produces 144 RGB frames. ConsistWorld uses the largest `1 + 4*n` prefix,
so 141 usable frames become 36 VAE latents and nine four-latent chunks.

The top-level [`README.md`](../README.md) documents cache construction, Stage
1/2 training, and inference using the resulting data.
