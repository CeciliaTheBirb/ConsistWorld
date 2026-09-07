# PaperA

This is a compact release of the PaperA multi-view autoregressive training,
cache-building, checkpoint-conversion, and inference path. It contains only the
code used by the final recipe. Experimental services, router-style extensions,
ablations, evaluation outputs, and data are deliberately excluded.

## Training Lineage

PaperA has two high-level training stages. Stage 1 builds the multi-view rolling
capability on MultiCamData; Stage 2 adapts it to rendered revisit trajectories.

| Phase | Data and recipe | Checkpoint used downstream |
| --- | --- | --- |
| Foundation | `lingbot-world-v2-14b-causal-fast` | foundation model |
| Stage 1A | MultiCamData / `multicam_vipe_f18`; shared first image, 1-4 target views, one-chunk rolling teacher forcing, bidirectional loss | `window_rolling_ckpt6000_model_full.pt` |
| Stage 1B | Same MultiCamData; SR-v3 continuation with four rollout steps and rho linearly ramped from 0 to 0.4 over 4000 steps | selected step 4000: `sr_v3_ckpt4000_model_full.pt` |
| Stage 2 | Rendered Infinigen multi-camera revisit data | selected step 3000 from the 4000-step PaperA run |

`train_multicam_base.py` is Stage 1A. `train_multicam_stage1.py` is Stage 1B
and starts from the Stage 1A checkpoint. `train_papera.py` is Stage 2 and starts
from the selected Stage 1B checkpoint.

The fixed Stage 2 recipe is:

- latent chunks of four frames; up to nine chunks per view
- one shared source-image chunk and one rolling history chunk per target view
- 1-4 target views, with at most 20 target-view chunks per step
- per-view top-1 pose retrieval with `candidate_chunk <= query_chunk - 2`
- geometry gate ramped from 0 to 1 over the first 500 steps
- self-resampling with four rollout steps, shift 0.25, and rho from 0 to 0.2
  over 4000 steps
- no learned view identity embedding and no bidirectional loss
- full-model fine-tuning with learning rate `1e-5`, 100-step warmup, gradient
  accumulation 2, sequence parallel size 8, and one 16-device FSDP replica

## Environment

Install the Python dependencies in an accelerator-enabled environment:

```bash
pip install -r requirements.txt
```

The original runs used 16 Ascend NPUs. CUDA requires a compatible attention
backend; Ascend requires the matching `torch_npu` and CANN runtime. `ffmpeg` is
needed for MP4 output. The model root must contain the original transformer
configuration, VAE, and T5 files from `lingbot-world-v2-14b-causal-fast`.

For the original NPU attention path, set `LINGBOT_MOBA_ATTN=loop`. Run all
commands below from this directory.

## Build Caches

The cache builder reads SpatialVID-style paired files named
`<scene>__camNN.mp4` and `<scene>__camNN.json`. Each JSON must provide
`intrinsics_vipe` and `poses_w2c_vipe`. It writes one packed scene under
`clips/` and its text embedding under `text/`.

Build the five-chunk MultiCamData cache for Stage 1:

```bash
LINGBOT_MOBA_ATTN=loop torchrun --nproc_per_node=16 build_clip_cache.py \
  --pretrained_model_root /path/to/lingbot-world-v2-14b-causal-fast \
  --navigation_roots /path/to/multicam_vipe_f18 \
  --out_dir /path/to/multicam_cache \
  --target_height 240 --target_width 416 --max_chunks 5
```

Build the nine-chunk cache for Stage 2 from the rendered revisit dataset:

```bash
LINGBOT_MOBA_ATTN=loop torchrun --nproc_per_node=16 build_clip_cache.py \
  --pretrained_model_root /path/to/lingbot-world-v2-14b-causal-fast \
  --navigation_roots /path/to/rendered_revisit_data \
  --out_dir /path/to/papera_cache \
  --target_height 240 --target_width 416 --max_chunks 9
```

## Stage 1: MultiCamData

Train the rolling teacher-forced initializer from the foundation model:

```bash
LINGBOT_MOBA_ATTN=loop torchrun --nproc_per_node=16 train_multicam_base.py \
  --pretrained_model_root /path/to/lingbot-world-v2-14b-causal-fast \
  --clip_cache_dir /path/to/multicam_cache \
  --output_dir /path/to/multicam_base_output \
  --sp_size 8 --dp_replicate 1 --max_steps 6000 --save_interval 250
```

Use `checkpoint-6000/model_full.pt` as the Stage 1B initializer. Continue with
SR-v3 and select its step-4000 portable checkpoint as
`sr_v3_ckpt4000_model_full.pt`:

```bash
LINGBOT_MOBA_ATTN=loop torchrun --nproc_per_node=16 train_multicam_stage1.py \
  --pretrained_model_root /path/to/lingbot-world-v2-14b-causal-fast \
  --init_model_pt /path/to/multicam_base_output/checkpoint-6000/model_full.pt \
  --clip_cache_dir /path/to/multicam_cache \
  --output_dir /path/to/multicam_sr_v3_output \
  --sp_size 8 --dp_replicate 1 --max_steps 6000 --save_interval 250
```

## Stage 2: Rendered Revisit Data

```bash
LINGBOT_MOBA_ATTN=loop torchrun --nproc_per_node=16 train_papera.py \
  --pretrained_model_root /path/to/lingbot-world-v2-14b-causal-fast \
  --init_model_pt /path/to/sr_v3_ckpt4000_model_full.pt \
  --clip_cache_dir /path/to/papera_cache \
  --output_dir /path/to/papera_output \
  --sp_size 8 --dp_replicate 1 --max_steps 4000 --save_interval 500
```

All trainers write portable `checkpoint-<step>/model_full.pt` files. The release
loads them strictly; missing or extra model tensors are errors rather than being
silently initialized or ignored.

## Convert the Final Checkpoint

The original final PaperA checkpoint contains legacy extension tensors that are
not present in this release, including the removed memory marker. It cannot be
loaded directly. Convert it once against the clean Stage 1B checkpoint:

```bash
python convert_papera_checkpoint.py \
  --source /path/to/raw/paperA_mem_20260901/checkpoint-3000/model_full.pt \
  --reference /path/to/sr_v3_ckpt4000_model_full.pt \
  --out /path/to/papera_clean_checkpoint-3000/model_full.pt
```

The converter keeps only the exact Stage 1B model ABI, validates every retained
tensor shape, and rejects unknown extras. Publish only the converted checkpoint
with this code. Removing the legacy extension tensors intentionally changes the
model, so the clean checkpoint is suitable for reproducing the released
architecture and recipe, not for bit-identical reproduction of the raw
extension checkpoint's videos.

## Inference

The trajectory JSON must define `n_chunks`, `n_frames`, and an absolute c2w pose
sequence for each target camera in `views[].poses`.

```bash
LINGBOT_MOBA_ATTN=loop python infer_papera.py \
  --ckpt /path/to/papera_clean_checkpoint-3000/model_full.pt \
  --pretrained_model_root /path/to/lingbot-world-v2-14b-causal-fast \
  --data_root /path/to/multicam_vipe_f18 \
  --scene f18_aperture10__scene4 \
  --src_cam cam01 --target_cams cam02,cam03 \
  --trajectory /path/to/trajectory.json \
  --first_frame_image /path/to/first_frame.png \
  --out /path/to/output.mp4 --seed 2027048 --sampling_steps 30
```

All target views are generated jointly. Retrieval is per target view and the
retrieved history is materialized only for the corresponding inference step.

## Release Scope and Licensing

This repository includes no model weights, data, trajectories, experiment logs,
or rendered videos. It is distributed under the upstream CC BY-NC-SA 4.0 terms
in `LICENSE.txt`. Files with separate upstream terms and their attributions are
listed in `THIRD_PARTY_NOTICES.md`.
