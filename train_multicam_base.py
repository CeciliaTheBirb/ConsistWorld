"""Train the MultiCamData rolling teacher-forced initializer for ConsistWorld."""
from __future__ import annotations

import argparse
from dataclasses import dataclass

from train_multicam_stage1 import MultiCamStage1Config, MultiCamStage1Trainer
from train_consistworld import create_dataloader


@dataclass
class MultiCamBaseConfig(MultiCamStage1Config):
    """Fixed MultiCamData recipe before the SR continuation."""

    max_steps: int = 6000
    self_resample: bool = False
    sr_rho_start: float = 0.0
    sr_rho_end: float = 0.0
    sr_rho_anneal_steps: int = 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train the rolling MultiCamData initializer from the foundation model"
    )
    parser.add_argument("--pretrained_model_root", required=True)
    parser.add_argument("--clip_cache_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--max_steps", type=int, default=6000)
    parser.add_argument("--save_interval", type=int, default=250)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sp_size", type=int, default=8)
    parser.add_argument("--dp_replicate", type=int, default=1)
    args = parser.parse_args()

    config = MultiCamBaseConfig(
        pretrained_model_root=args.pretrained_model_root,
        control_type="cam",
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
