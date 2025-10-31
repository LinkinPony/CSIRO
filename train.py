import argparse
import os
from pathlib import Path

import yaml
from loguru import logger

import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor, Callback
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger
import torch

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

    version = cfg.get("version", None)
    version = None if version in (None, "", "null") else str(version)

    base_log_dir = Path(cfg["logging"]["log_dir"]).expanduser()
    base_ckpt_dir = Path(cfg["logging"]["ckpt_dir"]).expanduser()

    # Route outputs to versioned subfolders if specified
    log_dir = base_log_dir / version if version else base_log_dir
    ckpt_dir = base_ckpt_dir / version if version else base_ckpt_dir
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

    class ExportTorchScriptCallback(Callback):
        def __init__(self, ckpt_dir: Path, image_size: int) -> None:
            super().__init__()
            self.ckpt_dir = ckpt_dir
            self.best_path = ckpt_dir / "best.ckpt"
            self.last_path = ckpt_dir / "last.ckpt"
            self.best_mtime = -1.0
            self.last_mtime = -1.0
            self.image_size = int(image_size)

        def _export_ts(self, model: pl.LightningModule, out_path: Path):
            try:
                example = torch.randn(1, 3, self.image_size, self.image_size)
                model_cpu = model.to("cpu").eval()
                ts = model_cpu.to_torchscript(method="trace", example_inputs=example)
                out_path.parent.mkdir(parents=True, exist_ok=True)
                torch.jit.save(ts, str(out_path))
                logger.info(f"Exported TorchScript to {out_path}")
            except Exception as e:
                logger.warning(f"TorchScript export failed for {out_path.name}: {e}")

        def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
            # Save and export 'last.ckpt' every epoch
            try:
                trainer.save_checkpoint(str(self.last_path))
                if self.last_path.exists():
                    mtime = self.last_path.stat().st_mtime
                    if mtime != self.last_mtime:
                        self.last_mtime = mtime
                        self._export_ts(pl_module, self.ckpt_dir / "last.ts")
            except Exception as e:
                logger.warning(f"Saving last.ckpt failed: {e}")

        def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
            # If best.ckpt updated, export
            try:
                if self.best_path.exists():
                    mtime = self.best_path.stat().st_mtime
                    if mtime != self.best_mtime:
                        self.best_mtime = mtime
                        # Load best weights to a fresh module to ensure consistency
                        best_model = type(pl_module).load_from_checkpoint(str(self.best_path))
                        self._export_ts(best_model, self.ckpt_dir / "best.ts")
            except Exception as e:
                logger.warning(f"Exporting best.ts failed: {e}")

    checkpoint_cb = ModelCheckpoint(
        dirpath=str(ckpt_dir),
        filename="best",
        auto_insert_metric_name=False,
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        save_last=True,
    )

    callbacks = [
        checkpoint_cb,
        LearningRateMonitor(logging_interval="epoch"),
        ExportTorchScriptCallback(ckpt_dir=ckpt_dir, image_size=int(cfg["data"]["image_size"])),
    ]

    csv_logger = CSVLogger(save_dir=str(log_dir), name="lightning", version=None)
    tb_logger = TensorBoardLogger(save_dir=str(log_dir), name="tensorboard", version=None)

    trainer = pl.Trainer(
        max_epochs=int(cfg["trainer"]["max_epochs"]),
        accelerator=str(cfg["trainer"]["accelerator"]),
        devices=cfg["trainer"]["devices"],
        precision=cfg["trainer"]["precision"],
        callbacks=callbacks,
        logger=[csv_logger, tb_logger],
        log_every_n_steps=int(cfg["trainer"]["log_every_n_steps"]),
    )

    resume_path = cfg.get("trainer", {}).get("resume_from", None)
    if resume_path is not None and resume_path != "null" and str(resume_path).strip() != "":
        resume_path = str(resume_path)
        logger.info(f"Resuming from checkpoint: {resume_path}")
    else:
        resume_path = None

    trainer.fit(model=model, datamodule=dm, ckpt_path=resume_path)

    # Generate metric plots into outputs
    try:
        import pandas as pd
        import matplotlib.pyplot as plt

        metrics_csv = Path(csv_logger.log_dir) / "metrics.csv"
        if metrics_csv.is_file():
            df = pd.read_csv(metrics_csv)
            # take last value per epoch
            if "epoch" not in df.columns:
                raise ValueError("metrics.csv missing epoch column")
            gb = df.groupby("epoch").tail(1).reset_index(drop=True)

            def get_col(candidates):
                for c in candidates:
                    if c in gb.columns:
                        return c
                return None

            cols = {
                "train_loss": get_col(["train_loss_epoch", "train_loss"]),
                "val_loss": get_col(["val_loss"]),
                "train_mae": get_col(["train_mae_epoch", "train_mae"]),
                "val_mae": get_col(["val_mae"]),
                "train_mse": get_col(["train_mse_epoch", "train_mse", "train_loss_epoch", "train_loss"]),
                "val_mse": get_col(["val_mse", "val_loss"]),
                "val_r2": get_col(["val_r2"]),
            }

            plots_dir = Path(log_dir) / "plots"
            plots_dir.mkdir(parents=True, exist_ok=True)

            # loss
            plt.figure()
            if cols["train_loss"]:
                plt.plot(gb["epoch"], gb[cols["train_loss"]], label="train")
            if cols["val_loss"]:
                plt.plot(gb["epoch"], gb[cols["val_loss"]], label="val")
            plt.xlabel("epoch")
            plt.ylabel("loss")
            plt.legend()
            plt.title("Loss")
            plt.tight_layout()
            plt.savefig(plots_dir / "loss.png")
            plt.close()

            # MAE
            plt.figure()
            if cols["train_mae"]:
                plt.plot(gb["epoch"], gb[cols["train_mae"]], label="train")
            if cols["val_mae"]:
                plt.plot(gb["epoch"], gb[cols["val_mae"]], label="val")
            plt.xlabel("epoch")
            plt.ylabel("MAE")
            plt.legend()
            plt.title("MAE")
            plt.tight_layout()
            plt.savefig(plots_dir / "mae.png")
            plt.close()

            # MSE
            plt.figure()
            if cols["train_mse"]:
                plt.plot(gb["epoch"], gb[cols["train_mse"]], label="train")
            if cols["val_mse"]:
                plt.plot(gb["epoch"], gb[cols["val_mse"]], label="val")
            plt.xlabel("epoch")
            plt.ylabel("MSE")
            plt.legend()
            plt.title("MSE")
            plt.tight_layout()
            plt.savefig(plots_dir / "mse.png")
            plt.close()

            # R^2 (val only)
            if cols["val_r2"]:
                plt.figure()
                plt.plot(gb["epoch"], gb[cols["val_r2"]], label="val_r2")
                plt.xlabel("epoch")
                plt.ylabel("R^2")
                plt.legend()
                plt.title("R^2")
                plt.tight_layout()
                plt.savefig(plots_dir / "r2.png")
                plt.close()
    except Exception as e:
        logger.warning(f"Metric plotting failed: {e}")


if __name__ == "__main__":
    main()


