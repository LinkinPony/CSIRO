from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
from loguru import logger

from src.metrics import TARGETS_5D_ORDER, weighted_r2_logspace
from src.tabular.tabpfn_utils import parse_tabpfn_inference_precision


@dataclass(frozen=True)
class FinetuneOptimizerConfig:
    name: str = "adam"
    lr: float = 1.5e-6
    weight_decay: float = 0.0


@dataclass(frozen=True)
class FinetuneGradientClipConfig:
    enabled: bool = True
    max_norm: float = 1.0


@dataclass(frozen=True)
class FinetuneCheckpointConfig:
    enabled: bool = True
    save_last: bool = True
    save_best: bool = True
    save_every_n_epochs: int = 0  # 0 => disabled
    monitor: str = "val_weighted_r2_log"


@dataclass(frozen=True)
class TabPFNFinetuneConfig:
    enabled: bool = False
    n_estimators: int = 2
    inference_precision: str = "float32"
    max_epochs: int = 10
    # Ensemble size used for per-epoch outer validation evaluation during fine-tuning.
    # This can be much smaller than `tabpfn.n_estimators` to keep epochs fast.
    # Final evaluation (outside the fine-tune loop) can still use a larger ensemble.
    eval_n_estimators: int = 8
    eval_every_n_epochs: int = 1
    inner_valid_ratio: float = 0.3
    meta_batch_size: int = 1
    max_data_size: Optional[int] = None
    equal_split_size: bool = True
    optimizer: FinetuneOptimizerConfig = FinetuneOptimizerConfig()
    gradient_clip: FinetuneGradientClipConfig = FinetuneGradientClipConfig()
    checkpoint: FinetuneCheckpointConfig = FinetuneCheckpointConfig()


@dataclass(frozen=True)
class TabPFNLightningTrainerConfig:
    """
    Subset of Lightning Trainer knobs used for TabPFN fine-tuning, kept compatible with
    the layout in `configs/train.yaml` (top-level `trainer:`).
    """

    max_epochs: int = 10
    accelerator: str = "auto"
    devices: Any = 1
    precision: Any = "32-true"
    log_every_n_steps: int = 1
    accumulate_grad_batches: int = 1
    limit_train_batches: Any = 1.0
    gradient_clip_val: float = 0.0
    gradient_clip_algorithm: str = "norm"
    deterministic: bool = False
    enable_progress_bar: bool = True


def parse_finetune_config(cfg: Dict[str, Any]) -> TabPFNFinetuneConfig:
    raw = dict(cfg or {})

    opt_raw = dict(raw.get("optimizer", {}) or {})
    optimizer = FinetuneOptimizerConfig(
        name=str(opt_raw.get("name", "adam") or "adam").strip().lower(),
        lr=float(opt_raw.get("lr", 1.5e-6)),
        weight_decay=float(opt_raw.get("weight_decay", 0.0)),
    )

    gc_raw = dict(raw.get("gradient_clip", {}) or {})
    gradient_clip = FinetuneGradientClipConfig(
        enabled=bool(gc_raw.get("enabled", True)),
        max_norm=float(gc_raw.get("max_norm", 1.0)),
    )

    ck_raw = dict(raw.get("checkpoint", {}) or {})
    checkpoint = FinetuneCheckpointConfig(
        enabled=bool(ck_raw.get("enabled", True)),
        save_last=bool(ck_raw.get("save_last", True)),
        save_best=bool(ck_raw.get("save_best", True)),
        save_every_n_epochs=int(ck_raw.get("save_every_n_epochs", 0) or 0),
        monitor=str(ck_raw.get("monitor", "val_weighted_r2_log") or "val_weighted_r2_log").strip(),
    )

    max_data_size_raw = raw.get("max_data_size", None)
    max_data_size = None
    if max_data_size_raw not in (None, "", "null"):
        max_data_size = int(max_data_size_raw)

    return TabPFNFinetuneConfig(
        enabled=bool(raw.get("enabled", False)),
        n_estimators=int(raw.get("n_estimators", 2)),
        inference_precision=str(raw.get("inference_precision", "float32") or "float32").strip(),
        max_epochs=int(raw.get("max_epochs", 10)),
        eval_n_estimators=int(raw.get("eval_n_estimators", 8)),
        eval_every_n_epochs=max(1, int(raw.get("eval_every_n_epochs", 1))),
        inner_valid_ratio=float(raw.get("inner_valid_ratio", 0.3)),
        meta_batch_size=int(raw.get("meta_batch_size", 1)),
        max_data_size=max_data_size,
        equal_split_size=bool(raw.get("equal_split_size", True)),
        optimizer=optimizer,
        gradient_clip=gradient_clip,
        checkpoint=checkpoint,
    )


def parse_tabpfn_lightning_trainer_config(
    cfg: Dict[str, Any], *, finetune_cfg: TabPFNFinetuneConfig
) -> TabPFNLightningTrainerConfig:
    raw = dict(cfg or {})

    def _norm_optional(v: Any) -> Any:
        return None if v in (None, "", "null") else v

    max_epochs = int(_norm_optional(raw.get("max_epochs", None)) or finetune_cfg.max_epochs)
    accelerator = str(_norm_optional(raw.get("accelerator", None)) or "auto")
    devices = _norm_optional(raw.get("devices", 1))
    precision = _norm_optional(raw.get("precision", "32-true")) or "32-true"
    log_every_n_steps = int(_norm_optional(raw.get("log_every_n_steps", None)) or 1)
    accumulate_grad_batches = int(_norm_optional(raw.get("accumulate_grad_batches", None)) or 1)

    limit_train_batches = _norm_optional(raw.get("limit_train_batches", None))
    if limit_train_batches is None:
        limit_train_batches = 1.0
    else:
        # Accept int / float / numeric strings (Lightning supports both int and float)
        if isinstance(limit_train_batches, str):
            s = limit_train_batches.strip()
            try:
                limit_train_batches = float(s) if ("." in s) else int(s)
            except Exception:
                limit_train_batches = 1.0

    # Gradient clip: prefer trainer.* (train.yaml style); fall back to finetune.gradient_clip
    gc_val = _norm_optional(raw.get("gradient_clip_val", None))
    if gc_val is None:
        gradient_clip_val = (
            float(finetune_cfg.gradient_clip.max_norm) if finetune_cfg.gradient_clip.enabled else 0.0
        )
    else:
        gradient_clip_val = float(gc_val)
    gradient_clip_algorithm = str(_norm_optional(raw.get("gradient_clip_algorithm", None)) or "norm")

    deterministic = bool(_norm_optional(raw.get("deterministic", None)) or False)
    enable_progress_bar = bool(_norm_optional(raw.get("enable_progress_bar", None)) or True)

    return TabPFNLightningTrainerConfig(
        max_epochs=int(max_epochs),
        accelerator=str(accelerator),
        devices=devices,
        precision=precision,
        log_every_n_steps=int(log_every_n_steps),
        accumulate_grad_batches=int(accumulate_grad_batches),
        limit_train_batches=limit_train_batches,
        gradient_clip_val=float(gradient_clip_val),
        gradient_clip_algorithm=str(gradient_clip_algorithm),
        deterministic=bool(deterministic),
        enable_progress_bar=bool(enable_progress_bar),
    )


def finetune_tabpfn_regressor_on_fold(
    *,
    TabPFNRegressor: type,
    base_model_path: str,
    device: str,
    ignore_pretraining_limits: bool,
    seed: int,
    fold_idx: int,
    finetune_cfg: TabPFNFinetuneConfig,
    eval_n_estimators: int,
    eval_fit_mode: str,
    eval_inference_precision: str,
    X_train: np.ndarray,
    y_train_5d: np.ndarray,
    X_val: np.ndarray,
    y_val_5d: np.ndarray,
    fold_log_dir: Path,
    fold_ckpt_dir: Path,
    trainer_cfg_raw: Optional[Dict[str, Any]] = None,
    tb_log_dir: Optional[Path] = None,
    tb_run_name: str = "tabpfn_finetune",
) -> tuple[str, dict[str, Any]]:
    """
    Fine-tune TabPFNRegressor weights on one fold and return a checkpoint path to use for evaluation.

    Returns:
        (best_or_last_ckpt_path, finetune_summary_dict)
    """
    if not finetune_cfg.enabled:
        raise ValueError("finetune_cfg.enabled is False")

    if finetune_cfg.meta_batch_size != 1:
        raise ValueError(
            "TabPFN fine-tuning currently only supports meta_batch_size=1 reliably. "
            f"Got meta_batch_size={finetune_cfg.meta_batch_size}."
        )

    import torch
    from sklearn.model_selection import train_test_split
    from torch.utils.data import DataLoader

    from tabpfn.model_loading import save_tabpfn_model
    from tabpfn.utils import meta_dataset_collator

    # Best-effort reduce fragmentation before allocating large graphs
    try:
        if torch.cuda.is_available() and str(device).strip().lower() != "cpu":
            torch.cuda.empty_cache()
    except Exception:
        pass

    # Create a finetuning regressor (batched) with requested precision.
    ft = TabPFNRegressor(
        n_estimators=int(finetune_cfg.n_estimators),
        device=str(device),
        fit_mode="batched",
        inference_precision=parse_tabpfn_inference_precision(finetune_cfg.inference_precision),
        random_state=int(seed + fold_idx),
        ignore_pretraining_limits=bool(ignore_pretraining_limits),
        model_path=str(base_model_path),
    )

    # Build one dataset per target so the fine-tuned weights see all 5 targets.
    X_list = [X_train for _ in range(len(TARGETS_5D_ORDER))]
    y_list = [y_train_5d[:, j] for j in range(len(TARGETS_5D_ORDER))]

    split_rng = np.random.default_rng(int(seed + 10_000 + fold_idx))

    def split_fn(X: np.ndarray, y: np.ndarray):  # type: ignore[no-untyped-def]
        rs = int(split_rng.integers(0, 2**31 - 1))
        return train_test_split(
            X, y, test_size=float(finetune_cfg.inner_valid_ratio), random_state=rs
        )

    datasets = ft.get_preprocessed_datasets(
        X_list,
        y_list,
        split_fn,
        max_data_size=finetune_cfg.max_data_size,
        equal_split_size=bool(finetune_cfg.equal_split_size),
    )

    # Ensure model is initialized (get_preprocessed_datasets does this on first call).
    if ft.models_ is None or len(ft.models_) != 1:
        raise RuntimeError(
            f"Fine-tuning expects a single internal model. Got models_={None if ft.models_ is None else len(ft.models_)}"
        )

    dl = DataLoader(
        datasets,
        batch_size=int(finetune_cfg.meta_batch_size),
        shuffle=True,
        collate_fn=meta_dataset_collator,
        num_workers=0,
    )

    tabpfn_ckpt_dir = (fold_ckpt_dir / "tabpfn").resolve()
    tabpfn_ckpt_dir.mkdir(parents=True, exist_ok=True)

    best_metric = -float("inf")
    best_path = tabpfn_ckpt_dir / "tabpfn_finetuned_best.ckpt"
    last_path = tabpfn_ckpt_dir / "tabpfn_finetuned_last.ckpt"

    eval_n_estimators_effective = int(getattr(finetune_cfg, "eval_n_estimators", 8) or 8)
    if eval_n_estimators_effective <= 0:
        eval_n_estimators_effective = 1

    def _eval_val_metric() -> float:
        # Evaluate using a fresh estimator that loads the *current* in-memory weights:
        # save to a temporary path in the fold ckpt dir to reuse the regular predict path.
        tmp_path = tabpfn_ckpt_dir / "tabpfn_finetuned_tmp_eval.ckpt"
        try:
            save_tabpfn_model(ft, tmp_path)
        except Exception as e:
            raise RuntimeError(
                f"Failed to save temporary TabPFN checkpoint for eval: {e}"
            ) from e

        # Predict each target independently (TabPFNRegressor is scalar-output).
        preds = np.zeros((len(X_val), len(TARGETS_5D_ORDER)), dtype=np.float64)
        for j in range(len(TARGETS_5D_ORDER)):
            reg = TabPFNRegressor(
                n_estimators=int(eval_n_estimators_effective),
                device=str(device),
                fit_mode=str(eval_fit_mode),
                inference_precision=parse_tabpfn_inference_precision(
                    eval_inference_precision
                ),
                random_state=int(seed + fold_idx + j),
                ignore_pretraining_limits=bool(ignore_pretraining_limits),
                model_path=str(tmp_path),
            )
            reg.fit(X_train, y_train_5d[:, j])
            preds[:, j] = reg.predict(X_val)

        return float(weighted_r2_logspace(y_val_5d, preds))

    # -------------------------
    # Lightning fine-tuning
    # -------------------------
    try:
        import lightning.pytorch as pl
        from lightning.pytorch.loggers import TensorBoardLogger
    except Exception as e:
        raise RuntimeError(
            "TabPFN fine-tuning requires Lightning (package `lightning`). "
            f"Import failed: {e}"
        ) from e

    # Optional TensorBoard logger (write under the provided tb_log_dir, usually <log_dir>/tensorboard/)
    tb_logger = None
    if tb_log_dir is not None:
        try:
            tb_log_dir = Path(tb_log_dir).resolve()
            tb_log_dir.mkdir(parents=True, exist_ok=True)
            tb_logger = TensorBoardLogger(
                save_dir=str(tb_log_dir.parent),
                name=str(tb_log_dir.name),
                version=None,
            )
        except Exception as e:
            logger.warning(f"[TabPFN][finetune] TensorBoardLogger unavailable, continuing without TB: {e}")
            tb_logger = None

    trainer_cfg = parse_tabpfn_lightning_trainer_config(
        trainer_cfg_raw or {}, finetune_cfg=finetune_cfg
    )

    class _TabPFNFinetuneLitModule(pl.LightningModule):
        def __init__(self) -> None:
            super().__init__()
            # Register the actual torch.nn.Module so Lightning can optimize it.
            self.model = ft.models_[0]

            self._best_metric = -float("inf")
            self._history: list[dict[str, Any]] = []

            self._loss_sum = 0.0
            self._loss_count = 0

        # Avoid Lightning moving this complex nested batch; TabPFN handles devices internally.
        def transfer_batch_to_device(self, batch: Any, device: Any, dataloader_idx: int) -> Any:  # type: ignore[override]
            return batch

        def configure_optimizers(self):  # type: ignore[no-untyped-def]
            opt_name = str(finetune_cfg.optimizer.name or "adam").strip().lower()
            lr = float(finetune_cfg.optimizer.lr)
            wd = float(finetune_cfg.optimizer.weight_decay)

            if opt_name == "adamw":
                from torch.optim import AdamW

                return AdamW(self.model.parameters(), lr=lr, weight_decay=wd)
            if opt_name == "adam":
                from torch.optim import Adam

                return Adam(self.model.parameters(), lr=lr, weight_decay=wd)
            raise ValueError(
                f"Unsupported finetune.optimizer.name: {opt_name!r} (expected 'adam' or 'adamw')"
            )

        def training_step(self, batch: Any, batch_idx: int):  # type: ignore[no-untyped-def]
            (
                X_trains_preprocessed,
                X_tests_preprocessed,
                y_trains_znorm,
                y_test_znorm,
                cat_ixs,
                confs,
                raw_space_bardist_,
                znorm_space_bardist_,
                _x_test_raw,
                _y_test_raw,
            ) = batch

            # These are wrapped by meta_dataset_collator; meta_batch_size=1 => index 0.
            ft.raw_space_bardist_ = raw_space_bardist_[0]
            ft.znorm_space_bardist_ = znorm_space_bardist_[0]

            ft.fit_from_preprocessed(X_trains_preprocessed, y_trains_znorm, cat_ixs, confs)

            # CRITICAL: TabPFN 2.5 creates the batched inference engine under torch.inference_mode(True) by default.
            # For weight fine-tuning we must explicitly disable it to allow autograd.
            try:
                if getattr(ft, "executor_", None) is not None:
                    ft.executor_.use_torch_inference_mode(use_inference=False)  # type: ignore[attr-defined]
            except Exception:
                # Best-effort: if TabPFN internals change, we still want a clear error at backward time.
                pass

            logits, _raw_outputs, _borders = ft.forward(X_tests_preprocessed)
            loss_fn = znorm_space_bardist_[0]
            y_target = y_test_znorm.to(self.device)
            loss = loss_fn(logits, y_target).mean()

            # Track epoch mean ourselves (stable scalar for history + checkpointing).
            try:
                self._loss_sum += float(loss.detach().cpu().item())
                self._loss_count += 1
            except Exception:
                pass

            # Log to TensorBoard via Lightning
            self.log(
                f"{tb_run_name}/train/loss",
                loss,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                logger=True,
            )
            self.log(
                "train_loss",
                loss,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                logger=True,
            )
            return loss

        def on_train_epoch_start(self) -> None:
            self._loss_sum = 0.0
            self._loss_count = 0

        def on_train_epoch_end(self) -> None:
            nonlocal best_metric

            epoch = int(self.current_epoch + 1)
            record: dict[str, Any] = {"epoch": int(epoch)}

            if self._loss_count > 0:
                record["train_loss"] = float(self._loss_sum / max(1, self._loss_count))

            do_eval = epoch % int(finetune_cfg.eval_every_n_epochs) == 0
            if do_eval:
                try:
                    import time

                    t0 = time.time()
                    val_r2 = _eval_val_metric()
                    dt = time.time() - t0
                    logger.info(
                        "[TabPFN][finetune] fold={} epoch={} outer-val eval done (eval_n_estimators={}): {:.1f}s",
                        int(fold_idx),
                        int(epoch),
                        int(eval_n_estimators_effective),
                        float(dt),
                    )
                except Exception as e:
                    logger.warning(f"[TabPFN][finetune] fold={fold_idx} epoch={epoch} eval failed: {e}")
                    val_r2 = float("nan")

                record["val_weighted_r2_log"] = float(val_r2)

                # Log both an unprefixed metric (for monitoring) and a prefixed one (for TB organization)
                self.log(
                    "val_weighted_r2_log",
                    float(val_r2),
                    on_step=False,
                    on_epoch=True,
                    prog_bar=True,
                    logger=True,
                )
                self.log(
                    f"{tb_run_name}/val/weighted_r2_log",
                    float(val_r2),
                    on_step=False,
                    on_epoch=True,
                    prog_bar=False,
                    logger=True,
                )

                if np.isfinite(val_r2) and float(val_r2) > float(best_metric):
                    best_metric = float(val_r2)
                    self._best_metric = float(val_r2)
                    if finetune_cfg.checkpoint.enabled and finetune_cfg.checkpoint.save_best:
                        try:
                            save_tabpfn_model(ft, best_path)
                        except Exception as e:
                            logger.warning(f"[TabPFN][finetune] saving best checkpoint failed: {e}")

            # Optional periodic checkpoint
            if finetune_cfg.checkpoint.enabled and int(finetune_cfg.checkpoint.save_every_n_epochs) > 0:
                every = int(finetune_cfg.checkpoint.save_every_n_epochs)
                if epoch % every == 0:
                    try:
                        save_tabpfn_model(ft, tabpfn_ckpt_dir / f"tabpfn_finetuned_epoch{epoch:03d}.ckpt")
                    except Exception as e:
                        logger.warning(f"[TabPFN][finetune] saving epoch checkpoint failed: {e}")

            self._history.append(record)

        def on_train_end(self) -> None:
            # Save last (always best-effort)
            if finetune_cfg.checkpoint.enabled and finetune_cfg.checkpoint.save_last:
                try:
                    save_tabpfn_model(ft, last_path)
                except Exception as e:
                    logger.warning(f"[TabPFN][finetune] saving last checkpoint failed: {e}")

        @property
        def history(self) -> list[dict[str, Any]]:
            return list(self._history)

        @property
        def best_metric(self) -> float:
            return float(self._best_metric)

    module = _TabPFNFinetuneLitModule()

    trainer = pl.Trainer(
        default_root_dir=str(fold_ckpt_dir),
        max_epochs=int(trainer_cfg.max_epochs),
        accelerator=str(trainer_cfg.accelerator),
        devices=trainer_cfg.devices,
        precision=trainer_cfg.precision,
        log_every_n_steps=int(trainer_cfg.log_every_n_steps),
        accumulate_grad_batches=int(trainer_cfg.accumulate_grad_batches),
        limit_train_batches=trainer_cfg.limit_train_batches,
        deterministic=bool(trainer_cfg.deterministic),
        gradient_clip_val=float(trainer_cfg.gradient_clip_val),
        gradient_clip_algorithm=str(trainer_cfg.gradient_clip_algorithm),
        enable_checkpointing=False,  # we save TabPFN-native checkpoints ourselves
        enable_progress_bar=bool(trainer_cfg.enable_progress_bar),
        enable_model_summary=False,
        logger=tb_logger if tb_logger is not None else False,
    )

    try:
        trainer.fit(module, train_dataloaders=dl)
    except torch.OutOfMemoryError:
        # No CPU fallback (user requested). Provide actionable hints.
        logger.error(
            "[TabPFN][finetune] CUDA OOM (no fallback). Tips: "
            "set finetune.n_estimators=1, set finetune.max_data_size (e.g. 64/128), "
            "set trainer.precision=16-mixed, and/or reduce input feature dim before TabPFN (e.g. PCA)."
        )
        raise
    except Exception:
        raise

    # Choose which checkpoint to return
    chosen = str(
        best_path if (finetune_cfg.checkpoint.save_best and best_path.is_file()) else last_path
    )
    summary = {
        "enabled": True,
        "n_estimators_train": int(finetune_cfg.n_estimators),
        "max_epochs": int(trainer_cfg.max_epochs),
        "inner_valid_ratio": float(finetune_cfg.inner_valid_ratio),
        "best_val_weighted_r2_log": float(best_metric) if np.isfinite(best_metric) else None,
        "checkpoint_best": str(best_path) if best_path.is_file() else None,
        "checkpoint_last": str(last_path) if last_path.is_file() else None,
        "checkpoint_chosen": chosen if Path(chosen).is_file() else None,
        "engine": "lightning",
        "trainer": {
            "max_epochs": int(trainer_cfg.max_epochs),
            "accelerator": str(trainer_cfg.accelerator),
            "devices": trainer_cfg.devices,
            "precision": trainer_cfg.precision,
            "log_every_n_steps": int(trainer_cfg.log_every_n_steps),
            "accumulate_grad_batches": int(trainer_cfg.accumulate_grad_batches),
            "limit_train_batches": trainer_cfg.limit_train_batches,
            "gradient_clip_val": float(trainer_cfg.gradient_clip_val),
            "gradient_clip_algorithm": str(trainer_cfg.gradient_clip_algorithm),
            "deterministic": bool(trainer_cfg.deterministic),
        },
        "history": module.history,
    }

    # Persist per-fold finetune history
    try:
        fold_log_dir.mkdir(parents=True, exist_ok=True)
        import json

        with open(fold_log_dir / "finetune_history.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
    except Exception as e:
        logger.warning(f"[TabPFN][finetune] saving finetune_history.json failed: {e}")

    if not summary["checkpoint_chosen"]:
        raise RuntimeError(
            "Fine-tuning finished but no checkpoint was saved (checkpoint.enabled may be false)."
        )

    return str(summary["checkpoint_chosen"]), summary


