import argparse
import os
from pathlib import Path

import yaml
from loguru import logger

import lightning.pytorch as pl
import torch

from src.training.kfold_runner import run_kfold
from src.training.logging_utils import init_logging
from src.training.single_run import (
    parse_image_size,
    resolve_dataset_area_m2,
    train_single_split,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default=str(Path(__file__).parent / "configs" / "train.yaml"),
        help="Path to YAML config file",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    version = cfg.get("version", None)
    version = None if version in (None, "", "null") else str(version)

    base_log_dir = Path(cfg["logging"]["log_dir"]).expanduser()
    base_ckpt_dir = Path(cfg["logging"]["ckpt_dir"]).expanduser()

    # Route outputs to versioned subfolders if specified
    log_dir = base_log_dir / version if version else base_log_dir
    ckpt_dir = base_ckpt_dir / version if version else base_ckpt_dir

    log_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # --- Snapshot training config into log_dir for reproducible packaging/inference ---
    try:
        import shutil as _shutil

        snapshot_path = log_dir / "train.yaml"
        # Always overwrite to reflect the exact config used for this run
        _shutil.copyfile(str(args.config), str(snapshot_path))
    except Exception as e:
        # Do not fail training if snapshotting fails; just warn.
        print(f"[WARN] Failed to snapshot training config to {log_dir}/train.yaml: {e}")

    init_logging(log_dir, use_loguru=cfg["logging"].get("use_loguru", True))
    logger.info("Loaded config from {}", args.config)

    pl.seed_everything(cfg.get("seed", 42), workers=True)

    # Ensure DINOv3 weights are copied to dinov3_weights for consistent, frozen backbone reuse
    try:
        import shutil
        dinov3_src = cfg["model"].get("weights_path")
        if dinov3_src and str(dinov3_src).strip() and os.path.isfile(str(dinov3_src)):
            repo_root = Path(__file__).parent
            dinov3_dir = repo_root / "dinov3_weights"
            dinov3_dir.mkdir(parents=True, exist_ok=True)
            dinov3_dst = dinov3_dir / "dinov3_vith16plus_pretrain_lvd1689m-7c1da9a5.pt"
            if not dinov3_dst.is_file() or os.path.getsize(dinov3_dst) == 0:
                shutil.copyfile(str(dinov3_src), str(dinov3_dst))
    except Exception as e:
        logger.warning(f"Copying DINOv3 weights failed: {e}")

    # Optional k-fold / train-all configuration
    kfold_cfg = cfg.get("kfold", {})
    kfold_enabled = bool(kfold_cfg.get("enabled", False))

    train_all_cfg = cfg.get("train_all", {})
    train_all_enabled = bool(train_all_cfg.get("enabled", False))

    # 1) Run k-fold training if enabled (uses per-fold subdirectories under log_dir/ckpt_dir)
    if kfold_enabled:
        run_kfold(cfg, log_dir, ckpt_dir)

    # 2) Train-all or plain single-split training
    if train_all_enabled:
        # Build full dataframe once when train_all is enabled to construct splits.
        try:
            from src.data.datamodule import PastureDataModule

            area_m2 = resolve_dataset_area_m2(cfg)
            base_dm = PastureDataModule(
                data_root=cfg["data"]["root"],
                train_csv=cfg["data"]["train_csv"],
                image_size=parse_image_size(cfg["data"]["image_size"]),
                batch_size=int(cfg["data"]["batch_size"]),
                val_batch_size=int(
                    cfg["data"].get("val_batch_size", cfg["data"]["batch_size"])
                ),
                num_workers=int(cfg["data"]["num_workers"]),
                prefetch_factor=int(cfg["data"].get("prefetch_factor", 2)),
                val_split=float(cfg["data"]["val_split"]),
                target_order=list(cfg["data"]["target_order"]),
                mean=list(cfg["data"]["normalization"]["mean"]),
                std=list(cfg["data"]["normalization"]["std"]),
                train_scale=tuple(cfg["data"]["augment"]["random_resized_crop_scale"]),
                sample_area_m2=float(area_m2),
                zscore_output_path=str((log_dir / "train_all") / "z_score.json"),
                log_scale_targets=bool(
                    cfg["model"].get("log_scale_targets", False)
                ),
                hflip_prob=float(cfg["data"]["augment"]["horizontal_flip_prob"]),
                shuffle=bool(cfg["data"].get("shuffle", True)),
            )
            try:
                full_df = base_dm.build_full_dataframe()
            except Exception as e:
                logger.warning(f"Building full dataframe failed: {e}")
                raise

            import pandas as pd  # local import to avoid global dependency if unused

            rng_seed = int(cfg.get("seed", 42))
            if len(full_df) < 1:
                raise ValueError(
                    "No samples available to construct train_all splits."
                )
            dummy_val = full_df.sample(n=1, random_state=rng_seed)
            # Optionally duplicate the single row to avoid degenerate loader corner cases
            val_df = pd.concat([dummy_val], ignore_index=True)
            train_df = full_df.reset_index(drop=True)
        except Exception as e:
            logger.warning(f"Constructing train_all splits failed: {e}")
            raise

        train_all_log_dir = log_dir / "train_all"
        train_all_ckpt_dir = ckpt_dir / "train_all"
        train_all_log_dir.mkdir(parents=True, exist_ok=True)
        train_all_ckpt_dir.mkdir(parents=True, exist_ok=True)

        # Delegate the actual training pipeline for train_all to the shared single-run helper.
        train_single_split(
            cfg,
            train_all_log_dir,
            train_all_ckpt_dir,
            train_df=train_df,
            val_df=val_df,
            train_all_mode=True,
        )
    elif not kfold_enabled:
        # Regular single-split training without k-fold or train_all.
        train_single_split(
            cfg,
            log_dir,
            ckpt_dir,
            train_df=None,
            val_df=None,
            train_all_mode=False,
        )


if __name__ == "__main__":
    main()


