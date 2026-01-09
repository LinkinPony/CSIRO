from __future__ import annotations

from pathlib import Path
from typing import Tuple

from loguru import logger
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger


_LOGURU_FILE_SINK_ID = None


def init_logging(log_dir: Path, use_loguru: bool = True) -> None:
    if use_loguru:
        global _LOGURU_FILE_SINK_ID
        try:
            if _LOGURU_FILE_SINK_ID is not None:
                logger.remove(_LOGURU_FILE_SINK_ID)
        except Exception:
            pass
        _LOGURU_FILE_SINK_ID = logger.add(
            log_dir / "train.log", rotation="10 MB", retention="7 days"
        )


def create_lightning_loggers(log_dir: Path) -> Tuple[CSVLogger, TensorBoardLogger]:
    csv_logger = CSVLogger(save_dir=str(log_dir), name="lightning", version=None)
    tb_logger = TensorBoardLogger(save_dir=str(log_dir), name="tensorboard", version=None)
    return csv_logger, tb_logger


def plot_epoch_metrics(metrics_csv_path: Path, out_dir: Path) -> None:
    try:
        import pandas as pd
        import matplotlib.pyplot as plt

        if not metrics_csv_path.is_file():
            return

        df = pd.read_csv(metrics_csv_path)
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

        out_dir.mkdir(parents=True, exist_ok=True)

        # Loss
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
        plt.savefig(out_dir / "loss.png")
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
        plt.savefig(out_dir / "mae.png")
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
        plt.savefig(out_dir / "mse.png")
        plt.close()

        # R2 (val only)
        if cols["val_r2"]:
            plt.figure()
            plt.plot(gb["epoch"], gb[cols["val_r2"]], label="val_r2")
            plt.xlabel("epoch")
            plt.ylabel("R^2")
            plt.legend()
            plt.title("R^2")
            plt.tight_layout()
            plt.savefig(out_dir / "r2.png")
            plt.close()
    except Exception as e:
        logger.warning(f"Metric plotting failed: {e}")


