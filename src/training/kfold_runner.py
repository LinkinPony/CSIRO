from __future__ import annotations

import os
from pathlib import Path
from typing import Dict

import numpy as np
from loguru import logger
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor

from src.data.datamodule import PastureDataModule
from src.models.regressor import BiomassRegressor
from src.callbacks.head_checkpoint import HeadCheckpoint
from src.training.logging_utils import create_lightning_loggers, plot_epoch_metrics


def _build_model(cfg: Dict) -> BiomassRegressor:
    return BiomassRegressor(
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
    )


def run_kfold(cfg: Dict, log_dir: Path, ckpt_dir: Path) -> None:
    # Build once to get the full dataframe and other settings
    base_dm = PastureDataModule(
        data_root=cfg["data"]["root"],
        train_csv=cfg["data"]["train_csv"],
        image_size=int(cfg["data"]["image_size"]),
        batch_size=int(cfg["data"]["batch_size"]),
        num_workers=int(cfg["data"]["num_workers"]),
        val_split=float(cfg["data"]["val_split"]),
        target_order=list(cfg["data"]["target_order"]),
        mean=list(cfg["data"]["normalization"]["mean"]),
        std=list(cfg["data"]["normalization"]["std"]),
        train_scale=tuple(cfg["data"]["augment"]["random_resized_crop_scale"]),
        hflip_prob=float(cfg["data"]["augment"]["horizontal_flip_prob"]),
        shuffle=bool(cfg["data"].get("shuffle", True)),
    )
    full_df = base_dm.build_full_dataframe()

    num_folds = int(cfg.get("kfold", {}).get("k", 5))

    # Prepare fold indices
    n_samples = len(full_df)
    indices = np.arange(n_samples)
    rng = np.random.default_rng(seed=int(cfg.get("seed", 42)))
    rng.shuffle(indices)
    folds = np.array_split(indices, num_folds)

    # Export fold splits
    splits_root = log_dir / "folds"
    splits_root.mkdir(parents=True, exist_ok=True)

    # Optional test listing export
    test_listing_df = None
    try:
        import pandas as pd
        test_csv_path = Path(cfg["data"]["root"]) / "test.csv"
        if test_csv_path.is_file():
            tdf = pd.read_csv(str(test_csv_path))
            tdf = tdf.copy()
            tdf["image_id"] = tdf["sample_id"].astype(str).str.split("__", n=1, expand=True)[0]
            tdf = tdf.groupby("image_id")["image_path"].first().reset_index()[["image_id", "image_path"]]
            test_listing_df = tdf
    except Exception as e:
        logger.warning(f"Reading test.csv for split export failed: {e}")

    for fold_idx in range(num_folds):
        val_idx = folds[fold_idx]
        train_idx = np.concatenate([folds[i] for i in range(num_folds) if i != fold_idx])

        train_df = full_df.iloc[train_idx].reset_index(drop=True)
        val_df = full_df.iloc[val_idx].reset_index(drop=True)

        # Save split CSVs
        try:
            import pandas as pd
            fold_splits_dir = splits_root / f"fold_{fold_idx}"
            fold_splits_dir.mkdir(parents=True, exist_ok=True)
            train_df[["image_id", "image_path"]].to_csv(fold_splits_dir / "train.csv", index=False)
            val_df[["image_id", "image_path"]].to_csv(fold_splits_dir / "val.csv", index=False)
            if test_listing_df is not None:
                test_listing_df.to_csv(fold_splits_dir / "test.csv", index=False)
        except Exception as e:
            logger.warning(f"Saving fold split CSVs failed (fold {fold_idx}): {e}")

        # Per-fold dirs
        fold_log_dir = log_dir / f"fold_{fold_idx}"
        fold_ckpt_dir = ckpt_dir / f"fold_{fold_idx}"
        fold_log_dir.mkdir(parents=True, exist_ok=True)
        fold_ckpt_dir.mkdir(parents=True, exist_ok=True)

        # Datamodule with predefined splits
        dm = PastureDataModule(
            data_root=cfg["data"]["root"],
            train_csv=cfg["data"]["train_csv"],
            image_size=int(cfg["data"]["image_size"]),
            batch_size=int(cfg["data"]["batch_size"]),
            num_workers=int(cfg["data"]["num_workers"]),
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

        model = _build_model(cfg)

        checkpoint_cb = ModelCheckpoint(
            dirpath=str(fold_ckpt_dir),
            filename="best",
            auto_insert_metric_name=False,
            monitor="val_loss",
            mode="min",
            save_top_k=1,
            save_last=True,
        )

        head_ckpt_dir = fold_ckpt_dir / "head"
        callbacks = [
            checkpoint_cb,
            LearningRateMonitor(logging_interval="epoch"),
            HeadCheckpoint(output_dir=str(head_ckpt_dir)),
        ]

        csv_logger, tb_logger = create_lightning_loggers(fold_log_dir)

        trainer = pl.Trainer(
            max_epochs=int(cfg["trainer"]["max_epochs"]),
            accelerator=str(cfg["trainer"]["accelerator"]),
            devices=cfg["trainer"]["devices"],
            precision=cfg["trainer"]["precision"],
            callbacks=callbacks,
            logger=[csv_logger, tb_logger],
            log_every_n_steps=int(cfg["trainer"]["log_every_n_steps"]),
        )

        last_ckpt = fold_ckpt_dir / "last.ckpt"
        resume_from_cfg = cfg.get("trainer", {}).get("resume_from", None)
        if last_ckpt.is_file():
            resume_path = str(last_ckpt)
            logger.info(f"[Fold {fold_idx}] Auto-resuming from last checkpoint: {resume_path}")
        elif resume_from_cfg is not None and resume_from_cfg != "null" and str(resume_from_cfg).strip() != "":
            resume_path = str(resume_from_cfg)
            logger.info(f"[Fold {fold_idx}] Resuming from checkpoint (config): {resume_path}")
        else:
            resume_path = None

        trainer.fit(model=model, datamodule=dm, ckpt_path=resume_path)

        # plots per fold
        metrics_csv = Path(csv_logger.log_dir) / "metrics.csv"
        plots_dir = fold_log_dir / "plots"
        plot_epoch_metrics(metrics_csv, plots_dir)


