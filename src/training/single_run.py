from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import lightning.pytorch as pl
from lightning.pytorch.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    StochasticWeightAveraging,
)
from loguru import logger

from src.callbacks.head_checkpoint import HeadCheckpoint
from src.data.datamodule import PastureDataModule
from src.models.regressor import BiomassRegressor
from src.training.logging_utils import create_lightning_loggers, plot_epoch_metrics


def parse_image_size(value: Any) -> Tuple[int, int]:
    """
    Accept int (square) or [width, height]; return (height, width).
    This keeps backward-compatible behavior with the original train.py helper.
    """
    try:
        if isinstance(value, (list, tuple)) and len(value) == 2:
            w, h = int(value[0]), int(value[1])
            return int(h), int(w)
        v = int(value)
        return v, v
    except Exception:
        v = int(value)
        return v, v


def resolve_dataset_area_m2(cfg: Dict) -> float:
    """
    Resolve dataset area (m^2) from config for unit conversion g <-> g/m^2.
    Shared between single-run training and k-fold split construction.
    """
    ds_name = str(cfg.get("data", {}).get("dataset", "csiro"))
    ds_map = dict(cfg.get("data", {}).get("datasets", {}))
    ds_info = dict(ds_map.get(ds_name, {}))
    try:
        width_m = float(ds_info.get("width_m", ds_info.get("width", 1.0)))
    except Exception:
        width_m = 1.0
    try:
        length_m = float(ds_info.get("length_m", ds_info.get("length", 1.0)))
    except Exception:
        length_m = 1.0
    try:
        area_m2 = float(ds_info.get("area_m2", width_m * length_m))
    except Exception:
        area_m2 = max(
            1.0,
            width_m * length_m if (width_m > 0.0 and length_m > 0.0) else 1.0,
        )
    if not (area_m2 > 0.0):
        area_m2 = 1.0
    return area_m2


def _build_datamodule(
    cfg: Dict,
    log_dir: Path,
    *,
    area_m2: float,
    train_df=None,
    val_df=None,
) -> PastureDataModule:
    """
    Build a PastureDataModule for a single train/val split.

    If train_df/val_df are provided, they are used as predefined splits.
    Otherwise, random splitting is controlled by cfg['data']['val_split'].
    """
    log_scale_targets_cfg = bool(cfg["model"].get("log_scale_targets", False))
    irish_cfg = cfg.get("irish_glass_clover", {})
    ndvi_dense_cfg = cfg.get("ndvi_dense", {})

    dm = PastureDataModule(
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
        zscore_output_path=str(log_dir / "z_score.json"),
        log_scale_targets=log_scale_targets_cfg,
        hflip_prob=float(cfg["data"]["augment"]["horizontal_flip_prob"]),
        augment_cfg=dict(cfg["data"].get("augment", {})),
        shuffle=bool(cfg["data"].get("shuffle", True)),
        # Optional predefined splits (used by train_all and k-fold)
        predefined_train_df=train_df,
        predefined_val_df=val_df,
        # NDVI dense (optional)
        ndvi_dense_enabled=bool(ndvi_dense_cfg.get("enabled", False)),
        ndvi_dense_root=str(ndvi_dense_cfg.get("root", ""))  # type: ignore[arg-type]
        if ndvi_dense_cfg.get("root", None) is not None
        else None,
        ndvi_dense_tile_size=int(ndvi_dense_cfg.get("tile_size", 512)),
        ndvi_dense_stride=int(
            ndvi_dense_cfg.get("tile_stride", ndvi_dense_cfg.get("stride", 448))
        ),
        ndvi_dense_batch_size=int(
            ndvi_dense_cfg.get("batch_size", cfg["data"]["batch_size"])
        ),
        ndvi_dense_num_workers=int(
            ndvi_dense_cfg.get("num_workers", cfg["data"]["num_workers"])
        ),
        ndvi_dense_mean=list(
            ndvi_dense_cfg.get("normalization", {}).get(
                "mean", cfg["data"]["normalization"]["mean"]
            )
        ),
        ndvi_dense_std=list(
            ndvi_dense_cfg.get("normalization", {}).get(
                "std", cfg["data"]["normalization"]["std"]
            )
        ),
        ndvi_dense_hflip_prob=float(
            ndvi_dense_cfg.get("augment", {}).get(
                "horizontal_flip_prob", cfg["data"]["augment"]["horizontal_flip_prob"]
            )
        ),
        ndvi_dense_vflip_prob=float(
            ndvi_dense_cfg.get("augment", {}).get("vertical_flip_prob", 0.0)
        ),
        # Irish Glass Clover (optional mixed dataset)
        irish_enabled=bool(irish_cfg.get("enabled", False)),
        irish_root=str(irish_cfg.get("root", ""))  # type: ignore[arg-type]
        if irish_cfg.get("root", None) is not None
        else None,
        irish_csv=str(irish_cfg.get("csv", "data.csv"))  # type: ignore[arg-type]
        if irish_cfg.get("csv", None) is not None
        else None,
        irish_image_dir=str(irish_cfg.get("image_dir", "images")),
        irish_image_size=parse_image_size(
            irish_cfg.get("image_size", cfg["data"]["image_size"])
        )
        if irish_cfg.get("image_size", None) is not None
        else None,
    )
    return dm


def _infer_aux_class_counts(
    dm: PastureDataModule,
    mtl_enabled: bool,
    tasks_cfg: Dict,
) -> Tuple[Optional[int], Optional[int]]:
    """
    Infer number of species/state classes using the datamodule utilities,
    matching the previous behavior that used the full dataframe.
    """
    if not mtl_enabled:
        return None, None

    species_enabled = bool(tasks_cfg.get("species", False)) and mtl_enabled
    state_enabled = bool(tasks_cfg.get("state", False)) and mtl_enabled

    num_species_classes: Optional[int]
    num_state_classes: Optional[int]

    # Species
    if species_enabled:
        try:
            num_species_classes = int(dm.num_species_classes)
            if num_species_classes <= 1:
                raise ValueError("Species column has <=1 unique values")
        except Exception as e:
            logger.warning(
                f"Falling back to num_species_classes=2 (reason: {e})"
            )
            num_species_classes = 2
    else:
        num_species_classes = None

    # State
    if state_enabled:
        try:
            num_state_classes = int(dm.num_state_classes)
            if num_state_classes <= 1:
                raise ValueError("State column has <=1 unique values")
        except Exception as e:
            logger.warning(f"Falling back to num_state_classes=2 (reason: {e})")
            num_state_classes = 2
    else:
        num_state_classes = None

    return num_species_classes, num_state_classes


def train_single_split(
    cfg: Dict,
    log_dir: Path,
    ckpt_dir: Path,
    *,
    train_df=None,
    val_df=None,
    train_all_mode: bool = False,
    num_species_classes: Optional[int] = None,
    num_state_classes: Optional[int] = None,
) -> None:
    """
    Train a single model on a given train/val split.

    This function encapsulates the full training pipeline so that both
    plain single-split training, train_all mode, and k-fold folds can
    reuse identical code (DRY).
    """
    area_m2 = resolve_dataset_area_m2(cfg)

    # Build datamodule, optionally with predefined splits.
    dm = _build_datamodule(
        cfg,
        log_dir,
        area_m2=area_m2,
        train_df=train_df,
        val_df=val_df,
    )

    # MTL toggle: when disabled, train only reg3 task.
    mtl_cfg = cfg.get("mtl", {})
    mtl_enabled = bool(mtl_cfg.get("enabled", True))
    tasks_cfg = mtl_cfg.get("tasks", {})

    height_enabled = bool(tasks_cfg.get("height", False)) and mtl_enabled
    ndvi_enabled = bool(tasks_cfg.get("ndvi", False)) and mtl_enabled
    ndvi_dense_enabled = bool(tasks_cfg.get("ndvi_dense", False)) and mtl_enabled

    # Infer auxiliary class counts if not provided explicitly (e.g., from k-fold)
    if mtl_enabled:
        # If caller did not pre-compute class counts, infer from datamodule.
        if num_species_classes is None or num_state_classes is None:
            inferred_species, inferred_state = _infer_aux_class_counts(
                dm, mtl_enabled, tasks_cfg
            )
            if num_species_classes is None:
                num_species_classes = inferred_species
            if num_state_classes is None:
                num_state_classes = inferred_state
    else:
        num_species_classes = None
        num_state_classes = None

    # Ensure z-score stats are computed before model init.
    try:
        dm.setup()
    except Exception:
        # For safety, keep going and let downstream code handle missing stats.
        pass

    loss_cfg = cfg.get("loss", {})
    ratio_head_enabled = bool(loss_cfg.get("use_ratio_head", True))
    loss_5d_enabled = bool(loss_cfg.get("use_5d_weighted_mse", True))
    ratio_kl_weight = float(loss_cfg.get("ratio_kl_weight", 1.0))
    mse_5d_weight = float(loss_cfg.get("mse_5d_weight", 1.0))
    mse_5d_weights_per_target = list(
        loss_cfg.get(
            "mse_5d_weights_per_target",
            [0.1, 0.1, 0.1, 0.2, 0.5],
        )
    )

    model = BiomassRegressor(
        backbone_name=str(cfg["model"]["backbone"]),
        embedding_dim=int(cfg["model"]["embedding_dim"]),
        num_outputs=len(cfg["data"]["target_order"]),
        dropout=float(cfg["model"]["head"].get("dropout", 0.0)),
        head_hidden_dims=list(cfg["model"]["head"].get("hidden_dims", [512, 256])),
        # Use a fixed SwiGLU bottleneck; activation is no longer configurable via YAML.
        head_activation="swiglu",
        use_output_softplus=bool(cfg["model"]["head"].get("use_output_softplus", True)),
        log_scale_targets=bool(cfg["model"].get("log_scale_targets", False)),
        area_m2=float(area_m2),
        reg3_zscore_mean=list(dm.reg3_zscore_mean or [])
        if hasattr(dm, "reg3_zscore_mean")
        else None,
        reg3_zscore_std=list(dm.reg3_zscore_std or [])
        if hasattr(dm, "reg3_zscore_std")
        else None,
        ndvi_zscore_mean=float(dm.ndvi_zscore_mean)
        if getattr(dm, "ndvi_zscore_mean", None) is not None
        else None,
        ndvi_zscore_std=float(dm.ndvi_zscore_std)
        if getattr(dm, "ndvi_zscore_std", None) is not None
        else None,
        biomass_5d_zscore_mean=list(dm.biomass_5d_zscore_mean or [])
        if hasattr(dm, "biomass_5d_zscore_mean")
        else None,
        biomass_5d_zscore_std=list(dm.biomass_5d_zscore_std or [])
        if hasattr(dm, "biomass_5d_zscore_std")
        else None,
        uw_learning_rate=float(
            cfg.get("optimizer", {}).get("uw_lr", cfg["optimizer"]["lr"])
        ),
        uw_weight_decay=float(
            cfg.get("optimizer", {}).get(
                "uw_weight_decay", cfg["optimizer"]["weight_decay"]
            )
        ),
        pretrained=bool(cfg["model"].get("pretrained", True)),
        weights_url=cfg["model"].get("weights_url", None),
        weights_path=cfg["model"].get("weights_path", None),
        freeze_backbone=bool(cfg["model"].get("freeze_backbone", True)),
        learning_rate=float(cfg["optimizer"]["lr"]),
        weight_decay=float(cfg["optimizer"]["weight_decay"]),
        scheduler_name=str(cfg.get("scheduler", {}).get("name", "")).lower() or None,
        scheduler_warmup_epochs=int(
            cfg.get("scheduler", {}).get("warmup_epochs", 0) or 0
        ),
        scheduler_warmup_start_factor=float(
            cfg.get("scheduler", {}).get("warmup_start_factor", 0.1)
        ),
        max_epochs=int(cfg["trainer"]["max_epochs"]),
        loss_weighting=(str(cfg.get("loss", {}).get("weighting", "")).lower() or None),
        num_species_classes=num_species_classes,
        num_state_classes=num_state_classes,
        mtl_enabled=mtl_enabled,
        # Task sampling ratio: probability to include NDVI-dense on a training step
        ndvi_dense_prob=float(
            cfg.get("mtl", {})
            .get("sample_ratio", {})
            .get("ndvi_dense", 1.0 if ndvi_dense_enabled else 0.0)
        ),
        enable_height=height_enabled,
        enable_ndvi=ndvi_enabled,
        enable_ndvi_dense=ndvi_dense_enabled,
        enable_species=bool(tasks_cfg.get("species", False)) and mtl_enabled,
        enable_state=bool(tasks_cfg.get("state", False)) and mtl_enabled,
        peft_cfg=dict(cfg.get("peft", {})),
        # Feature-level manifold mixup & CutMix configs (augmentation-level)
        manifold_mixup_cfg=dict(
            cfg.get("data", {}).get("augment", {}).get("manifold_mixup", {})
        ),
        cutmix_cfg=dict(cfg.get("data", {}).get("augment", {}).get("cutmix", {})),
        ndvi_dense_cutmix_cfg=dict(
            cfg.get("ndvi_dense", {})
            .get("augment", {})
            .get("cutmix", cfg.get("data", {}).get("augment", {}).get("cutmix", {}))
        ),
        # Biomass ratio / 5D loss configuration
        enable_ratio_head=ratio_head_enabled,
        ratio_kl_weight=ratio_kl_weight,
        enable_5d_loss=loss_5d_enabled,
        loss_5d_weight=mse_5d_weight,
        biomass_5d_weights=mse_5d_weights_per_target,
    )

    head_ckpt_dir = ckpt_dir / "head"
    callbacks = []
    # Lightweight Lightning checkpoint (last + best) to preserve full training state, excluding backbone weights.
    # When multiple val dataloaders are used (NDVI-dense enabled), Lightning logs metrics with
    # the suffix `/dataloader_idx_0`. Otherwise, it logs plain `val_loss`.
    use_multi_val = bool(cfg.get("ndvi_dense", {}).get("enabled", False))
    monitor_metric = "val_loss/dataloader_idx_0" if use_multi_val else "val_loss"
    checkpoint_cb = ModelCheckpoint(
        dirpath=str(ckpt_dir),
        filename="best",
        auto_insert_metric_name=False,
        monitor=monitor_metric,
        mode="min",
        save_top_k=1,
        save_last=True,
    )
    callbacks.append(checkpoint_cb)
    callbacks.append(LearningRateMonitor(logging_interval="epoch"))
    callbacks.append(HeadCheckpoint(output_dir=str(head_ckpt_dir)))

    # Optional SWA to stabilize small-batch updates
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
            logger.warning(
                f"SWA callback creation failed, continuing without SWA: {e}"
            )

    csv_logger, tb_logger = create_lightning_loggers(log_dir)

    trainer = pl.Trainer(
        max_epochs=int(cfg["trainer"]["max_epochs"]),
        accelerator=str(cfg["trainer"]["accelerator"]),
        devices=cfg["trainer"]["devices"],
        precision=cfg["trainer"]["precision"],
        limit_train_batches=cfg["trainer"].get("limit_train_batches", 1.0),
        limit_val_batches=cfg["trainer"].get(
            "limit_val_batches", 1 if train_all_mode else 1.0
        ),
        callbacks=callbacks,
        logger=[csv_logger, tb_logger],
        log_every_n_steps=int(cfg["trainer"]["log_every_n_steps"]),
        accumulate_grad_batches=int(cfg["trainer"].get("accumulate_grad_batches", 1)),
        gradient_clip_val=float(cfg["trainer"].get("gradient_clip_val", 0.0)),
        gradient_clip_algorithm=str(
            cfg["trainer"].get("gradient_clip_algorithm", "norm")
        ),
        enable_checkpointing=True,
    )

    last_ckpt = ckpt_dir / "last.ckpt"
    resume_from_cfg = cfg.get("trainer", {}).get("resume_from", None)
    if last_ckpt.is_file():
        resume_path = str(last_ckpt)
        logger.info(f"Auto-resuming from last checkpoint: {resume_path}")
    elif (
        resume_from_cfg is not None
        and resume_from_cfg != "null"
        and str(resume_from_cfg).strip() != ""
    ):
        resume_path = str(resume_from_cfg)
        logger.info(f"Resuming from checkpoint (config): {resume_path}")
    else:
        resume_path = None

    trainer.fit(model=model, datamodule=dm, ckpt_path=resume_path)

    # Generate metric plots for this run
    metrics_csv = Path(csv_logger.log_dir) / "metrics.csv"
    plots_dir = Path(log_dir) / "plots"
    plot_epoch_metrics(metrics_csv, plots_dir)


