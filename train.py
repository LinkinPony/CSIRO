import argparse
import os
from pathlib import Path

import yaml
from loguru import logger

import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import CSVLogger

from src.data.datamodule import PastureDataModule
from src.models.regressor import BiomassRegressor


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

    log_dir = Path(cfg["logging"]["log_dir"]).expanduser()
    ckpt_dir = Path(cfg["logging"]["ckpt_dir"]).expanduser()
    log_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    if cfg["logging"].get("use_loguru", True):
        logger.add(log_dir / "train.log", rotation="10 MB", retention="7 days")
        logger.info("Loaded config from {}", args.config)

    pl.seed_everything(cfg.get("seed", 42), workers=True)

    dm = PastureDataModule(
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

    model = BiomassRegressor(
        backbone_name=str(cfg["model"]["backbone"]),
        embedding_dim=int(cfg["model"]["embedding_dim"]),
        num_outputs=3,
        dropout=float(cfg["model"]["head"].get("dropout", 0.0)),
        pretrained=bool(cfg["model"].get("pretrained", True)),
        weights_url=cfg["model"].get("weights_url", None),
        freeze_backbone=bool(cfg["model"].get("freeze_backbone", True)),
        learning_rate=float(cfg["optimizer"]["lr"]),
        weight_decay=float(cfg["optimizer"]["weight_decay"]),
        scheduler_name=str(cfg.get("scheduler", {}).get("name", "")).lower() or None,
        max_epochs=int(cfg["trainer"]["max_epochs"]),
    )

    callbacks = [
        ModelCheckpoint(
            dirpath=str(ckpt_dir),
            filename="biomass-{epoch:02d}-{val_loss:.4f}",
            monitor="val_loss",
            mode="min",
            save_top_k=3,
        ),
        LearningRateMonitor(logging_interval="epoch"),
    ]

    csv_logger = CSVLogger(save_dir=str(log_dir), name="lightning")

    trainer = pl.Trainer(
        max_epochs=int(cfg["trainer"]["max_epochs"]),
        accelerator=str(cfg["trainer"]["accelerator"]),
        devices=cfg["trainer"]["devices"],
        precision=cfg["trainer"]["precision"],
        callbacks=callbacks,
        logger=csv_logger,
        log_every_n_steps=int(cfg["trainer"]["log_every_n_steps"]),
    )

    trainer.fit(model=model, datamodule=dm)


if __name__ == "__main__":
    main()


