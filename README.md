# ConsistWorld

ConsistWorld is a multi-view, autoregressive video recipe built on
[lingbot-world-v2-14b-causal-fast](https://huggingface.co/robbyant/lingbot-world-v2-14b-causal-fast).
This release contains the actual cache builder, the three training stages,
P-Mem retrieval, the cross-view attention gate, checkpoint conversion, and
joint multi-camera inference. It also contains a small, reproducible
Infinigen CPU-preparation and GPU-rendering workflow under
[`infinigen/`](infinigen/README.md).

The 14B model is not practical on a CPU. Cache construction, training, and
inference require CUDA or Ascend NPU; CPU-only hosts can run metadata, format,
trajectory, and checkpoint-format checks. The entry points fail early with a
clear error when no supported accelerator is available.

## Recipe Lineage

| Phase | Data and behavior | Checkpoint used next |
| --- | --- | --- |
| Foundation | `lingbot-world-v2-14b-causal-fast` | foundation model |
| Stage 1A | MultiCamData `multicam_vipe_f18`; shared first image, 1-4 target views, one-chunk rolling teacher forcing, bidirectional loss | step 6000 |
| Stage 1B | Same data; four-step self-resampling rollout, rho linearly ramped from 0 to 0.4 over 4000 steps | step 4000 |
| Stage 2 | Rendered Infinigen revisit clips; P-Mem and cross-view attention gate enabled | available final checkpoint: step 3000 |

The original Stage 1B run continued to 6000 steps, but the portable warm start
used by ConsistWorld is its step-4000 checkpoint. Stage 2 was configured for 4000
steps but received SIGTERM after saving checkpoint 3000. Do not describe that
checkpoint as the result of a completed 4000-step run.

The Stage 2 defaults in `train_consistworld.py` reproduce the released recipe:

- 4 latent frames per chunk, nine chunks per input view, and 1-4 target views;
- a shared first-image source chunk plus one rolling history chunk per target;
- top-1 P-Mem independently selected for each target view, with candidates
  restricted to `candidate_chunk <= query_chunk - 2`;
- a parameter-free cross-view-current gate ramped from 0 to 1 over 500 steps;
- four-step self-resampling with rho ramped from 0 to 0.2 over 4000 steps;
- full-model fine-tuning at `1e-5`, 100 warmup steps, and gradient accumulation
  of 2. No learned view identity embedding or bidirectional loss is used here.

The retained `mem_key_marker` is the active, zero-initialized P-Mem key tag in
the original final model. Experimental ORE, router, and related branches are
not part of this release.

## Setup

Install dependencies in an accelerator-enabled environment:

```bash
pip install -r requirements.txt
```

The original training used 16 Ascend NPUs. CUDA requires a compatible attention
backend; the conservative reference setting below works on the original NPU
path:

```bash
export LINGBOT_MOBA_ATTN=loop
```

For the original-scale run, use 16 devices with `sp_size=8` and
`dp_replicate=1`. This gives a world mesh of DP=2 x SP=8; the FSDP mesh keeps
one replica dimension and shards across all 16 ranks. Smaller configurations
are not an equivalent reproduction of the 21.79B full-finetuning run.

The model root must contain the upstream `transformers/` configuration, VAE,
T5 weights, and tokenizer. The upstream project is
[lingbot-world-v2](https://github.com/robbyant/lingbot-world-v2).

## Data Contract

Both MultiCamData and converted Infinigen data use the same SpatialVID layout.
Each scene has at least two paired files, recursively under a data root:

```text
<scene>__cam01.mp4
<scene>__cam01.json
<scene>__cam02.mp4
<scene>__cam02.json
```

Each JSON must contain normalized `intrinsics_vipe` and OpenCV
world-to-camera `poses_w2c_vipe`:

```json
{
  "intrinsics_vipe": [[fx_norm, 0, cx_norm], [0, fy_norm, cy_norm], [0, 0, 1]],
  "poses_w2c_vipe": [[[...], [...], [...], [...]]],
  "caption": {"SceneDescription": "optional scene description"}
}
```

The loader converts poses to camera-to-world internally and applies the same
center crop and intrinsic transform during cache building and inference. Video
lengths are rounded down to `1 + 4*n` RGB frames for the Wan temporal stride.
This is a model-format check, not an arbitrary dataset restriction: each
four-frame latent chunk requires a complete temporal-VAE window, and each usable
RGB frame must have a matching pose. For example, a 144-frame Infinigen render
supplies 141 usable frames, which is 36 latents or nine ConsistWorld chunks.

Validate a dataset before allocating the VAE or text encoder:

```bash
python infinigen/validate_multicam.py \
  --data_root /path/to/multicam_data \
  --expected_views 8 --expected_frames 141
```

Use `--expected_views 0` when validating a MultiCamData root with a varying
number of cameras.

## Repository Layout

`consistworld_runtime/` contains the small release-specific runtime package:

- `checkpoints.py` strictly loads, normalizes, and converts full-model
  checkpoints, including the zero-initialized P-Mem key marker compatibility
  rule;
- `inference_utils.py` reads the common SpatialVID camera format, converts
  poses, constructs camera controls, and validates authored trajectories.

The actual model implementation remains under `wan/`; the training, cache, and
inference entry points stay at the repository root. This keeps release-only
runtime and data-contract helpers separate from the model implementation.

## Build Caches

Run cache construction once for each source dataset. It writes packed scene
latents to `clips/` and text embeddings to `text/`.

```bash
LINGBOT_MOBA_ATTN=loop torchrun --nproc_per_node=16 build_clip_cache.py \
  --pretrained_model_root /path/to/lingbot-world-v2-14b-causal-fast \
  --navigation_roots /path/to/multicam_vipe_f18 \
  --out_dir /path/to/multicam_cache \
  --target_height 240 --target_width 416 --max_chunks 5
```

```bash
LINGBOT_MOBA_ATTN=loop torchrun --nproc_per_node=16 build_clip_cache.py \
  --pretrained_model_root /path/to/lingbot-world-v2-14b-causal-fast \
  --navigation_roots /path/to/rendered_revisit_data \
  --out_dir /path/to/consistworld_cache \
  --target_height 240 --target_width 416 --max_chunks 9
```

## Train

Stage 1A trains the rolling teacher-forced initializer from the foundation
model:

```bash
LINGBOT_MOBA_ATTN=loop torchrun --nproc_per_node=16 train_multicam_base.py \
  --pretrained_model_root /path/to/lingbot-world-v2-14b-causal-fast \
  --clip_cache_dir /path/to/multicam_cache \
  --output_dir /path/to/multicam_base_output \
  --sp_size 8 --dp_replicate 1 --max_steps 6000 --save_interval 250
```

Stage 1B continues from `checkpoint-6000/model_full.pt`. The command below
stops at the selected step-4000 warm start:

```bash
LINGBOT_MOBA_ATTN=loop torchrun --nproc_per_node=16 train_multicam_stage1.py \
  --pretrained_model_root /path/to/lingbot-world-v2-14b-causal-fast \
  --init_model_pt /path/to/multicam_base_output/checkpoint-6000/model_full.pt \
  --clip_cache_dir /path/to/multicam_cache \
  --output_dir /path/to/multicam_sr_output \
  --sp_size 8 --dp_replicate 1 --max_steps 4000 --save_interval 250
```

Stage 2 adapts that warm start on the rendered Infinigen cache:

```bash
LINGBOT_MOBA_ATTN=loop torchrun --nproc_per_node=16 train_consistworld.py \
  --pretrained_model_root /path/to/lingbot-world-v2-14b-causal-fast \
  --init_model_pt /path/to/multicam_sr_output/checkpoint-4000/model_full.pt \
  --clip_cache_dir /path/to/consistworld_cache \
  --output_dir /path/to/consistworld_output \
  --sp_size 8 --dp_replicate 1 --max_steps 4000 --save_interval 500
```

All trainers save portable `checkpoint-<step>/model_full.pt` files. Old Stage
1 checkpoints that predate the P-Mem marker can only omit those zero-init
tensors during a warm start; any other missing or unexpected tensor is an
error.

## Convert A Final Checkpoint

The private final ConsistWorld checkpoint has experimental tensors that are not part
of the public architecture. Convert it once before publishing it. The converter
retains every released core tensor, including `mem_key_marker`, validates
shapes, and refuses to discard an unrecognized tensor.

```bash
python convert_consistworld_checkpoint.py \
  --source /path/to/raw_consistworld/checkpoint-3000/model_full.pt \
  --reference /path/to/multicam_sr_output/checkpoint-4000/model_full.pt \
  --out /path/to/consistworld_checkpoint-3000/model_full.pt
```

`infer_consistworld.py` loads the converted final checkpoint strictly. This keeps a
published weight file tied to the released ABI instead of silently ignoring
architecture differences.

## Inference

Create a trajectory by replaying camera poses from either MultiCamData or
converted Infinigen data. The helper emits absolute camera-to-world poses and
limits the result to a valid common video prefix.

```bash
python make_trajectory_from_multicam.py \
  --data_root /path/to/multicam_data \
  --scene infngn_michar8_example \
  --target_cams cam02,cam03 \
  --n_chunks 9 \
  --out /path/to/trajectory.json
```

Then jointly generate the requested target views. The source video supplies
the conditioning first frame unless `--first_frame_image` is set.

```bash
LINGBOT_MOBA_ATTN=loop python infer_consistworld.py \
  --ckpt /path/to/consistworld_checkpoint-3000/model_full.pt \
  --pretrained_model_root /path/to/lingbot-world-v2-14b-causal-fast \
  --data_root /path/to/multicam_data \
  --scene infngn_michar8_example \
  --src_cam cam01 --target_cams cam02,cam03 \
  --trajectory /path/to/trajectory.json \
  --out /path/to/output.mp4 --seed 42 --sampling_steps 30
```

Inference is intentionally single-process. It keeps one generated history
chunk per target view, materializes each selected P-Mem anchor in the source
tail for that denoise step, and performs retrieval independently for every
target view.

## Infinigen Data

See [`infinigen/README.md`](infinigen/README.md) for the complete CPU scene
preparation, sequential GPU rendering, conversion, and validation workflow.
It uses the minimal eight-rig, in-place "mi-character" sweep from the earlier
pipeline and intentionally excludes private supervisors, hard-coded paths,
background services, and destructive cleanup behavior.

## Verification Scope

The release has been checked with command-line parsing, Python syntax checks,
P-Mem layout and checkpoint compatibility tests, Infinigen-to-SpatialVID
conversion, decoded MP4 dimensions, and dataset validation. A full 14B cache,
training, or inference run still requires a compatible CUDA GPU or Ascend NPU;
it cannot be truthfully verified on a CPU-only host.

## License

The release follows the terms in [`LICENSE.txt`](LICENSE.txt). Upstream and
third-party notices are listed in [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).
