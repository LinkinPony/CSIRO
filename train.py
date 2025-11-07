import argparse
import os
from pathlib import Path

import yaml
from loguru import logger

import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor, StochasticWeightAveraging
import torch

from src.data.datamodule import PastureDataModule
from src.models.regressor import BiomassRegressor
from src.callbacks.head_checkpoint import HeadCheckpoint
from src.training.kfold_runner import run_kfold
from src.training.logging_utils import init_logging, create_lightning_loggers, plot_epoch_metrics


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default=str(Path(__file__).parent / "configs" / "train.yaml"),
        help="Path to YAML config file",
    )
    return parser.parse_args()


def _parse_image_size(value):
    # Accept int (square) or [width, height]; return (height, width)
    try:
        if isinstance(value, (list, tuple)) and len(value) == 2:
            w, h = int(value[0]), int(value[1])
            return (int(h), int(w))
        v = int(value)
        return (v, v)
    except Exception:
        v = int(value)
        return (v, v)


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

    # Train-all mode may redirect outputs under a single fold_0 for consistency
    train_all_cfg = cfg.get("train_all", {})
    train_all_enabled = bool(train_all_cfg.get("enabled", False))
    if train_all_enabled:
        log_dir = log_dir / "fold_0"
        ckpt_dir = ckpt_dir / "fold_0"

    log_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

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

    # Optional k-fold configuration (ignored when train_all is enabled)
    kfold_cfg = cfg.get("kfold", {})
    use_kfold = bool(kfold_cfg.get("enabled", False)) and not train_all_enabled

    if use_kfold:
        run_kfold(cfg, log_dir, ckpt_dir)
    else:
        # Regular single-split training
        dm = PastureDataModule(
            data_root=cfg["data"]["root"],
            train_csv=cfg["data"]["train_csv"],
            image_size=_parse_image_size(cfg["data"]["image_size"]),
            batch_size=int(cfg["data"]["batch_size"]),
            val_batch_size=int(cfg["data"].get("val_batch_size", cfg["data"]["batch_size"])),
            num_workers=int(cfg["data"]["num_workers"]),
            prefetch_factor=int(cfg["data"].get("prefetch_factor", 2)),
            val_split=float(cfg["data"]["val_split"]),
            target_order=list(cfg["data"]["target_order"]),
            mean=list(cfg["data"]["normalization"]["mean"]),
            std=list(cfg["data"]["normalization"]["std"]),
            train_scale=tuple(cfg["data"]["augment"]["random_resized_crop_scale"]),
            hflip_prob=float(cfg["data"]["augment"]["horizontal_flip_prob"]),
            shuffle=bool(cfg["data"].get("shuffle", True)),
        )

        # MTL toggle: when disabled, train only reg3 task
        mtl_cfg = cfg.get("mtl", {})
        mtl_enabled = bool(mtl_cfg.get("enabled", True))

        # Build full dataframe once (also used by train_all)
        try:
            full_df = dm.build_full_dataframe()
        except Exception as e:
            logger.warning(f"Building full dataframe failed: {e}")
            raise

        # Train-all: use all samples for train and a single duplicated sample as dummy val
        if train_all_enabled:
            try:
                import pandas as pd  # local import to avoid global dependency if unused
                rng_seed = int(cfg.get("seed", 42))
                if len(full_df) < 1:
                    raise ValueError("No samples available to construct train_all splits.")
                dummy_val = full_df.sample(n=1, random_state=rng_seed)
                # Optionally duplicate the single row to avoid degenerate loader corner cases
                val_df = pd.concat([dummy_val], ignore_index=True)
                train_df = full_df.reset_index(drop=True)
                # Recreate datamodule with predefined splits
                dm = PastureDataModule(
                    data_root=cfg["data"]["root"],
                    train_csv=cfg["data"]["train_csv"],
                    image_size=_parse_image_size(cfg["data"]["image_size"]),
                    batch_size=int(cfg["data"]["batch_size"]),
                    val_batch_size=int(cfg["data"].get("val_batch_size", cfg["data"]["batch_size"])),
                    num_workers=int(cfg["data"]["num_workers"]),
                    prefetch_factor=int(cfg["data"].get("prefetch_factor", 2)),
                    val_split=0.0,
                    target_order=list(cfg["data"]["target_order"]),
                    mean=list(cfg["data"]["normalization"]["mean"]),
                    std=list(cfg["data"]["normalization"]["std"]),
                    train_scale=tuple(cfg["data"]["augment"]["random_resized_crop_scale"]),
                    hflip_prob=float(cfg["data"]["augment"]["horizontal_flip_prob"]),
                    shuffle=bool(cfg["data"].get("shuffle", True)),
                    predefined_train_df=train_df,
                    predefined_val_df=val_df,
                )
            except Exception as e:
                logger.warning(f"Constructing train_all splits failed: {e}")
                raise

        # Task toggles (default off)
        tasks_cfg = cfg.get("mtl", {}).get("tasks", {})
        height_enabled = bool(tasks_cfg.get("height", False)) and mtl_enabled
        ndvi_enabled = bool(tasks_cfg.get("ndvi", False)) and mtl_enabled
        species_enabled = bool(tasks_cfg.get("species", False)) and mtl_enabled
        state_enabled = bool(tasks_cfg.get("state", False)) and mtl_enabled

        # Infer number of species/state classes only if corresponding tasks are enabled
        if mtl_enabled:
            try:
                if species_enabled:
                    num_species_classes = int(len(sorted(full_df["Species"].dropna().astype(str).unique().tolist())))
                    if num_species_classes <= 1:
                        raise ValueError("Species column has <=1 unique values")
                else:
                    num_species_classes = None
            except Exception as e:
                if species_enabled:
                    logger.warning(f"Falling back to num_species_classes=2 (reason: {e})")
                    num_species_classes = 2
                else:
                    num_species_classes = None
            try:
                if state_enabled:
                    num_state_classes = int(len(sorted(full_df["State"].dropna().astype(str).unique().tolist())))
                    if num_state_classes <= 1:
                        raise ValueError("State column has <=1 unique values")
                else:
                    num_state_classes = None
            except Exception as e:
                if state_enabled:
                    logger.warning(f"Falling back to num_state_classes=2 (reason: {e})")
                    num_state_classes = 2
                else:
                    num_state_classes = None
        else:
            num_species_classes = None
            num_state_classes = None

        model = BiomassRegressor(
            backbone_name=str(cfg["model"]["backbone"]),
            embedding_dim=int(cfg["model"]["embedding_dim"]),
            num_outputs=3,
            dropout=float(cfg["model"]["head"].get("dropout", 0.0)),
            head_hidden_dims=list(cfg["model"]["head"].get("hidden_dims", [512, 256])),
            head_activation=str(cfg["model"]["head"].get("activation", "relu")),
            use_output_softplus=bool(cfg["model"]["head"].get("use_output_softplus", True)),
            pretrained=bool(cfg["model"].get("pretrained", True)),
            weights_url=cfg["model"].get("weights_url", None),
            weights_path=cfg["model"].get("weights_path", None),
            freeze_backbone=bool(cfg["model"].get("freeze_backbone", True)),
            learning_rate=float(cfg["optimizer"]["lr"]),
            weight_decay=float(cfg["optimizer"]["weight_decay"]),
            scheduler_name=str(cfg.get("scheduler", {}).get("name", "")).lower() or None,
            max_epochs=int(cfg["trainer"]["max_epochs"]),
            loss_weighting=(str(cfg.get("loss", {}).get("weighting", "")).lower() or None),
            num_species_classes=num_species_classes,
            num_state_classes=num_state_classes,
            mtl_enabled=mtl_enabled,
            enable_height=height_enabled,
            enable_ndvi=ndvi_enabled,
            enable_species=species_enabled,
            enable_state=state_enabled,
            peft_cfg=dict(cfg.get("peft", {})),
        )

        checkpoint_cb = ModelCheckpoint(
            dirpath=str(ckpt_dir),
            filename="best",
            auto_insert_metric_name=False,
            monitor="val_loss",
            mode="min",
            save_top_k=1,
            save_last=True,
        )

        head_ckpt_dir = ckpt_dir / "head"
        callbacks = [
            checkpoint_cb,
            LearningRateMonitor(logging_interval="epoch"),
            HeadCheckpoint(output_dir=str(head_ckpt_dir)),
        ]

        # Optional SWA to stabilize small-batch updates (match k-fold behavior)
        swa_cfg = cfg.get("trainer", {}).get("swa", {})
        if bool(swa_cfg.get("enabled", False)):
            try:
                swa_cb = StochasticWeightAveraging(
                    swa_lrs=float(swa_cfg.get("swa_lrs", cfg["optimizer"]["lr"])),
                    swa_epoch_start=float(swa_cfg.get("swa_epoch_start", 0.8)),
                    annealing_epochs=int(swa_cfg.get("annealing_epochs", 5)),
                    annealing_strategy=str(swa_cfg.get("annealing_strategy", "cos")),
                )
                callbacks.append(swa_cb)
            except Exception as e:
                logger.warning(f"SWA callback creation failed, continuing without SWA: {e}")

        csv_logger, tb_logger = create_lightning_loggers(log_dir)

        trainer = pl.Trainer(
            max_epochs=int(cfg["trainer"]["max_epochs"]),
            accelerator=str(cfg["trainer"]["accelerator"]),
            devices=cfg["trainer"]["devices"],
            precision=cfg["trainer"]["precision"],
            callbacks=callbacks,
            logger=[csv_logger, tb_logger],
            log_every_n_steps=int(cfg["trainer"]["log_every_n_steps"]),
            accumulate_grad_batches=int(cfg["trainer"].get("accumulate_grad_batches", 1)),
            gradient_clip_val=float(cfg["trainer"].get("gradient_clip_val", 0.0)),
            gradient_clip_algorithm=str(cfg["trainer"].get("gradient_clip_algorithm", "norm")),
        )

        last_ckpt = ckpt_dir / "last.ckpt"
        resume_from_cfg = cfg.get("trainer", {}).get("resume_from", None)
        if last_ckpt.is_file():
            resume_path = str(last_ckpt)
            logger.info(f"Auto-resuming from last checkpoint: {resume_path}")
        elif resume_from_cfg is not None and resume_from_cfg != "null" and str(resume_from_cfg).strip() != "":
            resume_path = str(resume_from_cfg)
            logger.info(f"Resuming from checkpoint (config): {resume_path}")
        else:
            resume_path = None

        trainer.fit(model=model, datamodule=dm, ckpt_path=resume_path)

    # Generate metric plots for non-kfold run
    if not use_kfold:
        metrics_csv = Path(csv_logger.log_dir) / "metrics.csv"
        plots_dir = Path(log_dir) / "plots"
        plot_epoch_metrics(metrics_csv, plots_dir)


if __name__ == "__main__":
    main()


