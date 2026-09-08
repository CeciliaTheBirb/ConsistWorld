"""Train the MultiCamData rolling SR-v3 warm start used by ConsistWorld."""
from __future__ import annotations

import argparse
import time
from dataclasses import dataclass

import torch
import torch.distributed as dist

from train_consistworld import (
    LingbotStage1ARTrainer,
    Stage1ARTrainingConfig,
    create_dataloader,
    get_sigmas,
)
from wan.dataset.sequence_parallel import sequence_parallel_batches
from wan.utils.accel import device_module, device_type, empty_cache as accel_empty_cache
from wan.utils.stage1_ar_selfresample import ar_seq_len, rho_curriculum


@dataclass
class MultiCamStage1Config(Stage1ARTrainingConfig):
    """The fixed rolling recipe that produced the Stage-1 ConsistWorld initializer."""

    max_steps: int = 6000
    warmup_steps: int = 500
    save_interval: int = 250
    clip_cache_k_min: int = 1
    clip_cache_k_max: int = 4
    clip_cache_max_total_view_chunks: int = 10
    clip_cache_target_chunks: int = 5
    sr_rho_end: float = 0.4
    sr_rho_anneal_steps: int = 4000
    pmem: bool = False
    xnow_gate: bool = False


class MultiCamStage1Trainer(LingbotStage1ARTrainer):
    """Stage 1 without the ConsistWorld-only retrieval and geometry-gate paths."""

    def _prepare_batch_ar(self, batch):
        (
            latents,
            cond_y,
            text_emb_unique,
            control_tensor,
            chunk_text_idx,
            a_b_emb,
            a_g_emb,
            *rest,
        ) = batch
        extra = rest[0] if rest else {}
        if not extra or "latents_tgt" not in extra:
            raise ValueError("MultiCamData batches must contain packed target views")
        if latents.ndim != 5 or latents.shape[0] != 1:
            raise ValueError(
                "expected one full target clip shaped [1, 16, N*4, H, W], "
                f"got {tuple(latents.shape)}"
            )

        batch_size, _, _, height, width = latents.shape
        self._validate_batch_shapes(latents, cond_y, control_tensor, height, width)
        chunk = int(self.geometry.chunk_size)
        num_chunks = self._num_chunks_from_shape(latents)
        if num_chunks < 2:
            raise ValueError("rolling MultiCamData training needs at least two chunks per view")

        target_latents = extra["latents_tgt"].to(self.device).contiguous()
        target_conditions = extra["cond_tgt"].to(self.device).contiguous()
        target_controls = extra["control_tgt"].to(self.device).contiguous()
        num_views = int(target_latents.shape[0])
        expected_frames = num_chunks * chunk
        if (
            num_views < 1
            or tuple(target_latents.shape[1:3]) != (16, expected_frames)
            or tuple(target_conditions.shape[1:3]) != (20, expected_frames)
        ):
            raise ValueError("packed target views do not match the requested full-clip layout")
        if target_controls.shape[0] != num_views or target_controls.shape[2] != expected_frames:
            raise ValueError("packed camera controls do not match the target views")

        latents = torch.cat([target_latents[view : view + 1] for view in range(num_views)], dim=2)
        cond_y = torch.cat([target_conditions[view : view + 1] for view in range(num_views)], dim=2)
        control_tensor = torch.cat([target_controls[view : view + 1] for view in range(num_views)], dim=2)

        _, patch_height, patch_width = self.wan_config.patch_size
        tokens_per_frame = (-(-height // patch_height)) * (-(-width // patch_width))

        taus = self._sample_timesteps(batch_size * num_chunks).view(batch_size, num_chunks)
        bi_timestep = self._sample_timesteps(batch_size)
        noise = torch.randn_like(latents)
        if self.sp_enabled:
            taus = self._sync_for_sp(taus)
            bi_timestep = self._sync_for_sp(bi_timestep)
            noise = self._sync_for_sp(noise)
        taus_full = taus.repeat(1, num_views)

        total_chunks = num_views * num_chunks
        sigmas_chunk = get_sigmas(
            self.noise_schedule,
            latents.device,
            taus_full.reshape(-1),
            n_dim=1,
            dtype=latents.dtype,
        ).view(batch_size, total_chunks)
        sigmas = sigmas_chunk.repeat_interleave(chunk, dim=1).view(
            batch_size, 1, total_chunks * chunk, 1, 1
        )
        noised = (1.0 - sigmas) * latents + sigmas * noise
        timesteps = taus_full.repeat_interleave(chunk, dim=1).repeat_interleave(
            tokens_per_frame, dim=1
        )

        bi_sigma = get_sigmas(
            self.noise_schedule,
            latents.device,
            bi_timestep,
            n_dim=1,
            dtype=latents.dtype,
        ).view(batch_size, 1, 1, 1, 1)
        bi_sigmas = bi_sigma.expand_as(sigmas).contiguous()
        bi_noised = (1.0 - bi_sigmas) * latents + bi_sigmas * noise

        frames_per_view = num_chunks * chunk
        kept_frames = (num_chunks - 1) * chunk
        history_indices = torch.cat(
            [
                torch.arange(
                    view * frames_per_view,
                    view * frames_per_view + kept_frames,
                    device=self.device,
                )
                for view in range(num_views)
            ]
        )
        history_clean = latents.index_select(2, history_indices).contiguous()
        history_condition = cond_y.index_select(2, history_indices).contiguous()
        history_control = control_tensor.index_select(2, history_indices).contiguous()
        history = self._assemble_history_input(history_clean, history_condition)

        history_rope = torch.arange(kept_frames, device=self.device).remainder(chunk)
        target_rope = torch.arange(frames_per_view, device=self.device).remainder(chunk).add(chunk)
        history_rope = history_rope.repeat(num_views)[None].expand(batch_size, -1).contiguous()
        target_rope = target_rope.repeat(num_views)[None].expand(batch_size, -1).contiguous()

        source_latents = extra["latents_source"].to(self.device).contiguous()
        source_condition = extra["cond_source"].to(self.device).contiguous()
        source_control = extra["control_source"].to(self.device).contiguous()
        if source_latents.shape[2] != chunk or source_condition.shape[2] != chunk:
            raise ValueError("the MultiCamData source stream must contain exactly one chunk")
        source = torch.cat([source_latents, source_condition], dim=1).contiguous()
        source_rope = torch.arange(chunk, device=self.device)[None].expand(batch_size, -1).contiguous()

        controls = {
            "c2ws_plucker_emb": control_tensor.chunk(batch_size, dim=0),
            "c2ws_plucker_emb_history_short": history_control.chunk(batch_size, dim=0),
            "c2ws_plucker_emb_source": source_control,
        }
        sequence_length = ar_seq_len(
            chunk + num_views * kept_frames + num_views * frames_per_view,
            height,
            width,
            self.wan_config.patch_size,
        )

        return {
            "latents_history_short": history,
            "cond_y": cond_y,
            "cond_y_full": cond_y,
            "text_emb_unique": text_emb_unique,
            "chunk_text_idx": chunk_text_idx,
            "a_b_emb": a_b_emb,
            "a_g_emb": a_g_emb,
            "seq_len": sequence_length,
            "control_dict": controls,
            "timesteps": timesteps,
            "noised": noised,
            "sigmas": sigmas,
            "target": noise - latents,
            "bi_timesteps": bi_timestep,
            "bi_noised": bi_noised,
            "bi_sigmas": bi_sigmas,
            "bi_target": noise - latents,
            "indices_hidden_states": target_rope,
            "indices_latents_history_short": history_rope,
            "latents_source": source,
            "indices_source": source_rope,
            "num_views": num_views,
            "sr_full_clean_16": latents,
            "sr_hist_cond": history_condition,
            "ar_layout": {
                "chunk_size": chunk,
                "tgt_abs_chunks": list(range(1, num_chunks + 1)),
                "window_chunks": 1,
                "sink_chunks": 0,
            },
        }

    def train_step(self, batch):
        inputs = self.prepare_batch(batch)
        model_dtype = torch.bfloat16 if self.config.dtype == "bf16" else torch.float32
        context = self._tensor_to_text_list(inputs["text_emb_unique"])
        chunk_text_idx = inputs["chunk_text_idx"]

        rho = rho_curriculum(self.global_step, self.config)
        use_self_resample = False
        if self.config.self_resample:
            draw = torch.rand(1, device=self.device)
            if self.world_size > 1:
                dist.broadcast(draw, src=0)
            use_self_resample = draw.item() < rho

        history = inputs["latents_history_short"]
        if use_self_resample:
            clean = inputs["sr_full_clean_16"]
            degraded = self._self_resample_history_kv(
                context,
                inputs["a_b_emb"],
                inputs["a_g_emb"],
                clean,
                inputs["cond_y_full"],
                chunk_text_idx,
                int(clean.shape[-2]),
                int(clean.shape[-1]),
                inputs["num_views"],
                inputs["control_dict"],
                inputs["latents_source"],
                inputs["indices_source"],
            )
            history = self._assemble_history_input(degraded, inputs["sr_hist_cond"])

        with torch.autocast(
            device_type=device_type,
            dtype=model_dtype,
            enabled=device_type != "cpu" and model_dtype == torch.bfloat16,
        ):
            prediction, bidirectional_prediction = self.single_dit.model(
                x=self._tensor_to_video_list(inputs["noised"]),
                x_bi=self._tensor_to_video_list(inputs["bi_noised"]),
                t=inputs["timesteps"],
                t_bi=inputs["bi_timesteps"],
                context=context,
                seq_len=inputs["seq_len"],
                y=self._tensor_to_video_list(inputs["cond_y"]),
                dit_cond_dict=inputs["control_dict"],
                indices_hidden_states=inputs["indices_hidden_states"],
                indices_latents_history_short=inputs["indices_latents_history_short"],
                latents_history_short=history,
                ar_history=True,
                ar_layout=inputs["ar_layout"],
                chunk_text_idx=chunk_text_idx,
                a_b_emb=inputs["a_b_emb"],
                a_g_emb=inputs["a_g_emb"],
                enable_bi=True,
                history_timestep=0.0,
                latents_source=inputs["latents_source"],
                indices_source=inputs["indices_source"],
                num_views=inputs["num_views"],
            )
            if bidirectional_prediction is None:
                raise RuntimeError("the Stage-1 recipe requires the bidirectional loss branch")
            pred_tf = torch.stack(prediction, dim=0)
            pred_bi = torch.stack(bidirectional_prediction, dim=0)

        loss_tf = self._masked_flow_loss(pred_tf, inputs["target"], inputs["sigmas"])
        loss_bi = self._masked_flow_loss(pred_bi, inputs["bi_target"], inputs["bi_sigmas"])
        loss = 0.5 * (loss_tf + loss_bi)
        (loss / self.config.gradient_accumulation_steps).backward()

        for parameter in self.single_dit.parameters():
            if parameter.grad is not None and parameter.grad.dtype != parameter.dtype:
                parameter.grad = parameter.grad.to(parameter.dtype)

        if (self.global_step + 1) % self.config.gradient_accumulation_steps == 0:
            for parameter in self.single_dit.parameters():
                if parameter.grad is not None and parameter.grad.dtype != parameter.dtype:
                    parameter.grad = parameter.grad.to(parameter.dtype)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.single_dit.parameters(), self.config.max_grad_norm
            )
            self.last_grad_norm = float(grad_norm.item())
            self.optimizer.step()
            self.lr_scheduler.step()
            self.optimizer.zero_grad(set_to_none=True)

        return {
            "loss": float(loss.item()),
            "loss_tf": float(loss_tf.item()),
            "loss_bi": float(loss_bi.item()),
            "grad_norm": self.last_grad_norm,
            "rho": rho,
            "self_resample": float(use_self_resample),
            "num_views": inputs["num_views"],
            "view_chunks": inputs["target"].shape[2] // (inputs["num_views"] * self.geometry.chunk_size),
        }

    def train(self, dataloader) -> None:
        if self.is_main_process:
            print(
                f"[stage1] max_steps={self.config.max_steps} lr={self.config.learning_rate} "
                "source=image rolling_window=1 bidirectional_loss=on "
                f"self_resample={'on' if self.config.self_resample else 'off'}",
                flush=True,
            )

        iterator = iter(sequence_parallel_batches(dataloader, self.device, self.config.sp_size))
        self.single_dit.train()
        accel_empty_cache()
        while self.global_step < self.config.max_steps:
            started = time.perf_counter()
            metrics = self.train_step(next(iterator))
            elapsed = time.perf_counter() - started
            if self.is_main_process:
                print(
                    f"[stage1] step={self.global_step}/{self.config.max_steps} "
                    f"time={elapsed:.2f}s loss={metrics['loss']:.6f} "
                    f"(tf={metrics['loss_tf']:.4f} bi={metrics['loss_bi']:.4f}) "
                    f"grad={metrics['grad_norm']:.4f} rho={metrics['rho']:.3f} "
                    f"sr={metrics['self_resample']:.0f} K={metrics['num_views']} "
                    f"view_chunks={metrics['view_chunks']} "
                    f"mem={device_module.memory_allocated() / 2**30:.1f}/"
                    f"{device_module.memory_reserved() / 2**30:.1f}G",
                    flush=True,
                )

            checkpoint_step = self.global_step + 1
            if checkpoint_step % self.config.save_interval == 0:
                self.save_checkpoint(checkpoint_step)
            self.global_step += 1

        if self.config.save_final_checkpoint and self.global_step % self.config.save_interval != 0:
            self.save_checkpoint(self.global_step)
        if self.world_size > 1:
            dist.barrier()
            dist.destroy_process_group()


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the MultiCamData Stage-1 SR-v3 initializer")
    parser.add_argument("--pretrained_model_root", required=True)
    parser.add_argument("--init_model_pt", required=True, help="rolling TF checkpoint at step 6000")
    parser.add_argument("--clip_cache_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--max_steps", type=int, default=6000)
    parser.add_argument("--save_interval", type=int, default=250)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sp_size", type=int, default=8)
    parser.add_argument("--dp_replicate", type=int, default=1)
    args = parser.parse_args()

    config = MultiCamStage1Config(
        pretrained_model_root=args.pretrained_model_root,
        control_type="cam",
        init_model_pt=args.init_model_pt,
        clip_cache_dir=args.clip_cache_dir,
        output_dir=args.output_dir,
        max_steps=args.max_steps,
        save_interval=args.save_interval,
        seed=args.seed,
        sp_size=args.sp_size,
        dp_replicate=args.dp_replicate,
    )
    trainer = MultiCamStage1Trainer(config)
    trainer.train(create_dataloader(config))


if __name__ == "__main__":
    main()
