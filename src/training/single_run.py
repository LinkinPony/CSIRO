from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import lightning.pytorch as pl
from lightning.pytorch.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    StochasticWeightAveraging,
)
from loguru import logger

from src.callbacks.adaptive_swa_lrs import AdaptiveSwaLrsOnStart
from src.callbacks.ema import ExponentialMovingAverage
from src.callbacks.freeze_lora_on_swa import FreezeLoraOnSwaStart
from src.callbacks.head_checkpoint import HeadCheckpoint
from src.data.datamodule import PastureDataModule
from src.models.regressor import BiomassRegressor
from src.training.logging_utils import create_lightning_loggers, plot_epoch_metrics


def _extract_optimizer_from_conf(opt_conf: Any):
    """
    Handle LightningModule.configure_optimizers() return types:
      - Optimizer
      - dict with key "optimizer"
    """
    if isinstance(opt_conf, dict) and "optimizer" in opt_conf:
        return opt_conf["optimizer"]
    return opt_conf


def _build_swa_lrs_for_model(
    model: pl.LightningModule,
    *,
    head_swa_lr: float,
    uw_swa_lr: Optional[float] = None,
    freeze_lora: bool = True,
    lora_swa_lr: float = 0.0,
) -> List[float]:
    """
    Build a per-parameter-group SWA LR list (方案 B), aligned with the model's optimizer param groups.

    - head / other params  -> head_swa_lr
    - UW params (if present) -> uw_swa_lr (defaults to head_swa_lr)
    - LoRA params -> lora_swa_lr (defaults to 0.0 when freeze_lora=True; 方案 C)
    """
    try:
        opt_conf = model.configure_optimizers()
        optimizer = _extract_optimizer_from_conf(opt_conf)
        param_groups = list(getattr(optimizer, "param_groups", []) or [])
    except Exception as e:
        logger.warning(f"SWA: failed to inspect optimizer param groups for auto_lrs (fallback to scalar): {e}")
        return [float(head_swa_lr)]

    # Lightning's StochasticWeightAveraging validates that `swa_lrs` is a positive float
    # (or a list of positive floats). Zero will raise:
    #   "The `swa_lrs` should a positive float, or a list of positive floats"
    #
    # We keep LoRA freezing semantics via the dedicated callback `FreezeLoraOnSwaStart`
    # (requires_grad=False + lr=0), so the SWA LR list should never contain zeros.
    eps = 1e-12
    head_lr_eff = float(head_swa_lr)
    uw_lr_eff = float(head_lr_eff) if uw_swa_lr is None else float(uw_swa_lr)
    # When freeze_lora=True we still return a positive LR here; the freeze callback
    # will hard-set LoRA lr to 0.0 at SWA start (and disable grads).
    lora_lr_eff = float(lora_swa_lr) if (not freeze_lora) else float(head_lr_eff)
    head_lr_eff = max(head_lr_eff, eps)
    uw_lr_eff = max(uw_lr_eff, eps)
    lora_lr_eff = max(lora_lr_eff, eps)

    swa_lrs: List[float] = []
    for group in param_groups:
        gtype = str(group.get("group_type", "") or "").strip().lower()
        gname = str(group.get("name", "") or "").strip().lower()
        if gtype == "lora" or gname.startswith("lora"):
            swa_lrs.append(float(lora_lr_eff))
        elif gtype == "uw" or gname == "uw":
            swa_lrs.append(float(uw_lr_eff))
        else:
            swa_lrs.append(float(head_lr_eff))
    return swa_lrs


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
    Otherwise, the datamodule performs a grouped split by (Sampling_Date, State),
    controlled by cfg['data']['val_split'] and seeded by cfg['seed'].
    """
    log_scale_targets_cfg = bool(cfg["model"].get("log_scale_targets", False))
    irish_cfg = cfg.get("irish_glass_clover", {})
    ndvi_dense_cfg = cfg.get("ndvi_dense", {})
    aigc_cfg = cfg.get("data", {}).get("aigc_aug", {}) or {}
    # Normalize optional list-like config keys
    types_val = aigc_cfg.get("types", None)
    if types_val in (None, "", "null"):
        aigc_types = None
    elif isinstance(types_val, (list, tuple)):
        aigc_types = [str(x) for x in types_val]
    else:
        aigc_types = [str(types_val)]

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
        irish_supervise_ratio=bool(irish_cfg.get("supervise_ratio", False)),
        irish_image_size=parse_image_size(
            irish_cfg.get("image_size", cfg["data"]["image_size"])
        )
        if irish_cfg.get("image_size", None) is not None
        else None,
        # Seed for reproducible internal split when predefined splits are not supplied
        random_seed=int(cfg.get("seed", 42)),
        # Optional AIGC (Nano Banana Pro) offline augmentations
        aigc_aug_enabled=bool(aigc_cfg.get("enabled", False)),
        aigc_aug_subdir=str(aigc_cfg.get("subdir", "nano_banana_pro/train"))
        if aigc_cfg.get("subdir", None) is not None
        else None,
        aigc_aug_manifest=str(aigc_cfg.get("manifest", "manifest.csv")),
        aigc_aug_types=aigc_types,
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
    extra_callbacks: Optional[list] = None,
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
    date_enabled = bool(tasks_cfg.get("date", False)) and mtl_enabled

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
    mse_5d_weights_per_target = list(
        loss_cfg.get(
            "mse_5d_weights_per_target",
            [0.1, 0.1, 0.1, 0.2, 0.5],
        )
    )

    # Multi-layer backbone / layer-wise head configuration (optional)
    backbone_layers_cfg = (
        cfg["model"].get("backbone_layers", {})
        if isinstance(cfg.get("model", {}), dict)
        else {}
    )
    use_layerwise_heads = bool(backbone_layers_cfg.get("enabled", False))
    # Layer fusion across selected backbone layers:
    # - "mean" (default): uniform average (legacy behavior)
    # - "learned": learn softmax weights over layers
    backbone_layers_fusion = str(
        backbone_layers_cfg.get(
            "layer_fusion",
            backbone_layers_cfg.get("fusion", backbone_layers_cfg.get("fusion_mode", "mean")),
        )
        or "mean"
    ).strip().lower()

    # Layer indices can be provided either as:
    #  - backbone_layers.indices (explicit list)
    #  - backbone_layers.indices_by_backbone.<backbone_name> (map, avoids manual edits when switching backbones)
    backbone_layer_indices = backbone_layers_cfg.get("indices", None)
    if (not backbone_layer_indices) and isinstance(backbone_layers_cfg.get("indices_by_backbone", None), dict):
        try:
            bb_name = str(cfg.get("model", {}).get("backbone", "")).strip()
        except Exception:
            bb_name = ""
        indices_map = dict(backbone_layers_cfg.get("indices_by_backbone", {}))
        if bb_name and bb_name in indices_map:
            backbone_layer_indices = indices_map.get(bb_name, None)
        else:
            # Optional fallback: allow keys without the "dinov3_" prefix.
            short = bb_name.replace("dinov3_", "") if bb_name else ""
            if short and short in indices_map:
                backbone_layer_indices = indices_map.get(short, None)
    # When true (default), use an independent bottleneck MLP for each selected layer.
    # When false, all layers share a single bottleneck (legacy behavior).
    use_separate_bottlenecks = bool(
        backbone_layers_cfg.get("separate_bottlenecks", True)
    )

    # Ratio head coupling mode (enum). Prefer the new single-field config, but
    # fall back to legacy boolean flags for backward compatibility with old YAMLs.
    head_cfg = cfg.get("model", {}).get("head", {}) if isinstance(cfg.get("model", {}), dict) else {}
    try:
        from src.models.regressor.heads.ratio_mode import resolve_ratio_head_mode

        ratio_head_mode = resolve_ratio_head_mode(
            head_cfg.get("ratio_head_mode", None),
            separate_ratio_head=head_cfg.get("separate_ratio_head", None),
            separate_ratio_spatial_head=head_cfg.get("separate_ratio_spatial_head", None),
        )
    except Exception:
        ratio_head_mode = str(head_cfg.get("ratio_head_mode", "shared") or "shared").strip().lower()

    # Optional dual-branch fusion for MLP patch-mode:
    # combine patch-based main prediction with a global prediction from CLS+mean(patch).
    dual_branch_cfg = head_cfg.get("dual_branch", {})
    if not isinstance(dual_branch_cfg, dict):
        dual_branch_cfg = {}
    dual_branch_enabled = bool(dual_branch_cfg.get("enabled", False))
    try:
        dual_branch_alpha_init = float(dual_branch_cfg.get("alpha_init", 0.2))
    except Exception:
        dual_branch_alpha_init = 0.2

    # Optimizer / SAM configuration
    optimizer_cfg = cfg.get("optimizer", {})
    optimizer_name = str(optimizer_cfg.get("name", "adamw"))
    use_sam = bool(optimizer_cfg.get("use_sam", False) or optimizer_name.lower() == "sam")
    sam_rho = float(optimizer_cfg.get("sam_rho", 0.05))
    sam_adaptive = bool(optimizer_cfg.get("sam_adaptive", False))

    model = BiomassRegressor(
        backbone_name=str(cfg["model"]["backbone"]),
        embedding_dim=int(cfg["model"]["embedding_dim"]),
        head_type=str(cfg["model"]["head"].get("type", "mlp")),
        vitdet_dim=int(cfg["model"]["head"].get("vitdet_dim", 256)),
        vitdet_patch_size=int(
            cfg["model"]["head"].get(
                "vitdet_patch_size",
                cfg["model"]["head"].get("fpn_patch_size", 16),
            )
        ),
        vitdet_scale_factors=list(cfg["model"]["head"].get("vitdet_scale_factors", [2.0, 1.0, 0.5])),
        # EoMT-style query pooling head (optional)
        eomt_num_queries=int(
            (cfg["model"]["head"].get("eomt", {}) or {}).get("num_queries", 16)
        )
        if isinstance(cfg["model"]["head"].get("eomt", {}), dict)
        else 16,
        eomt_num_layers=int(
            # Backward compatibility:
            # - new injected-query mode uses `num_blocks` (matches `third_party/eomt`)
            # - older configs used `num_layers` (from the previous query-decoder variant)
            (cfg["model"]["head"].get("eomt", {}) or {}).get(
                "num_blocks",
                (cfg["model"]["head"].get("eomt", {}) or {}).get("num_layers", 4),
            )
        )
        if isinstance(cfg["model"]["head"].get("eomt", {}), dict)
        else 4,
        eomt_num_heads=int(
            (cfg["model"]["head"].get("eomt", {}) or {}).get("num_heads", 8)
        )
        if isinstance(cfg["model"]["head"].get("eomt", {}), dict)
        else 8,
        eomt_ffn_dim=int(
            (cfg["model"]["head"].get("eomt", {}) or {}).get("ffn_dim", 2048)
        )
        if isinstance(cfg["model"]["head"].get("eomt", {}), dict)
        else 2048,
        eomt_query_pool=str(
            (cfg["model"]["head"].get("eomt", {}) or {}).get("query_pool", "mean")
        )
        if isinstance(cfg["model"]["head"].get("eomt", {}), dict)
        else "mean",
        eomt_use_mean_query=bool(
            (cfg["model"]["head"].get("eomt", {}) or {}).get("use_mean_query", True)
        )
        if isinstance(cfg["model"]["head"].get("eomt", {}), dict)
        else True,
        eomt_use_mean_patch=bool(
            (cfg["model"]["head"].get("eomt", {}) or {}).get("use_mean_patch", False)
        )
        if isinstance(cfg["model"]["head"].get("eomt", {}), dict)
        else False,
        eomt_use_cls_token=bool(
            (cfg["model"]["head"].get("eomt", {}) or {}).get(
                "use_cls_token",
                (cfg["model"]["head"].get("eomt", {}) or {}).get("use_cls", False),
            )
        )
        if isinstance(cfg["model"]["head"].get("eomt", {}), dict)
        else False,
        eomt_proj_dim=int(
            (cfg["model"]["head"].get("eomt", {}) or {}).get("proj_dim", 0)
        )
        if isinstance(cfg["model"]["head"].get("eomt", {}), dict)
        else 0,
        eomt_proj_activation=str(
            (cfg["model"]["head"].get("eomt", {}) or {}).get("proj_activation", "relu")
        )
        if isinstance(cfg["model"]["head"].get("eomt", {}), dict)
        else "relu",
        eomt_proj_dropout=float(
            (cfg["model"]["head"].get("eomt", {}) or {}).get("proj_dropout", 0.0)
        )
        if isinstance(cfg["model"]["head"].get("eomt", {}), dict)
        else 0.0,
        fpn_dim=int(cfg["model"]["head"].get("fpn_dim", 256)),
        fpn_num_levels=int(cfg["model"]["head"].get("fpn_num_levels", 3)),
        fpn_patch_size=int(cfg["model"]["head"].get("fpn_patch_size", 16)),
        fpn_reverse_level_order=bool(
            cfg["model"]["head"].get("fpn_reverse_level_order", True)
        ),
        dpt_features=int(cfg["model"]["head"].get("dpt_features", 256)),
        dpt_patch_size=int(
            cfg["model"]["head"].get(
                "dpt_patch_size",
                cfg["model"]["head"].get("fpn_patch_size", 16),
            )
        ),
        dpt_readout=str(cfg["model"]["head"].get("dpt_readout", "ignore")),
        num_outputs=len(cfg["data"]["target_order"]),
        dropout=float(cfg["model"]["head"].get("dropout", 0.0)),
        head_hidden_dims=list(cfg["model"]["head"].get("hidden_dims", [512, 256])),
        # Head activation is configurable via YAML; default to ReLU for backward compatibility.
        head_activation=str(cfg["model"]["head"].get("activation", "relu")),
        use_output_softplus=bool(cfg["model"]["head"].get("use_output_softplus", True)),
        # Whether to include CLS token in global features (CLS+mean(patch) vs mean(patch) only)
        use_cls_token=bool(cfg["model"]["head"].get("use_cls_token", True)),
        # Optional patch-based main regression path (scheme A: only main task uses patch path).
        use_patch_reg3=bool(cfg["model"]["head"].get("use_patch_reg3", False)),
        # Optional dual-branch fusion (MLP patch-mode only).
        dual_branch_enabled=bool(dual_branch_enabled),
        dual_branch_alpha_init=float(dual_branch_alpha_init),
        # Optional separate ratio head branch (decouples ratio MLP trunk from reg3 trunk).
        ratio_head_mode=str(ratio_head_mode),
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
        backbone_weights_dtype=str(cfg["model"].get("backbone_weights_dtype", "fp32")),
        gradient_checkpointing=bool(cfg["model"].get("gradient_checkpointing", False)),
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
        enable_date=date_enabled,
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
        # Keep ratio_kl_weight argument for checkpoint compatibility; it is not used for manual task weighting.
        enable_5d_loss=loss_5d_enabled,
        biomass_5d_weights=mse_5d_weights_per_target,
        # Multi-layer heads
        use_layerwise_heads=use_layerwise_heads,
        backbone_layer_indices=list(backbone_layer_indices)
        if isinstance(backbone_layer_indices, (list, tuple))
        else None,
        use_separate_bottlenecks=use_separate_bottlenecks,
        backbone_layers_fusion=str(backbone_layers_fusion),
        optimizer_name=optimizer_name,
        use_sam=use_sam,
        sam_rho=sam_rho,
        sam_adaptive=sam_adaptive,
        # Gradient surgery (multi-task)
        pcgrad_cfg=dict(cfg.get("pcgrad", {})),
        # Debug: optionally dump the final model input images (after CutMix) for sanity checks.
        input_image_mean=list(cfg.get("data", {}).get("normalization", {}).get("mean", [0.485, 0.456, 0.406])),
        input_image_std=list(cfg.get("data", {}).get("normalization", {}).get("std", [0.229, 0.224, 0.225])),
        run_log_dir=str(log_dir),
        debug_input_dump_cfg=dict(cfg.get("data", {}).get("augment", {}).get("debug_dump", {})),
        augmix_consistency_cfg=dict(
            cfg.get("data", {}).get("augment", {}).get("augmix", {}).get("consistency", {})
        ),
    )

    head_ckpt_dir = ckpt_dir / "head"
    callbacks = []
    # Lightweight Lightning checkpoint (last + best) to preserve full training state, excluding backbone weights.
    # When multiple val dataloaders are used (NDVI-dense enabled), Lightning logs metrics with
    # the suffix `/dataloader_idx_0`. Otherwise, it logs plain `val_loss`.
    use_multi_val = bool(cfg.get("ndvi_dense", {}).get("enabled", False))
    monitor_metric = "val_loss/dataloader_idx_0" if use_multi_val else "val_loss"
    # Checkpoint policy:
    # - Do NOT save any "best.ckpt" (or any top-k monitored checkpoints).
    # - Keep Lightning's "last.ckpt" for backward compatibility / resume.
    #
    # NOTE:
    #   We only need `last.ckpt` updated at least once per epoch (no step-based saving).
    ckpt_cfg_raw = cfg.get("trainer", {}).get("checkpoint", None)
    # Default: save `last.ckpt` once per epoch.
    if ckpt_cfg_raw is None:
        every_n_train_steps = 0
        every_n_epochs = 1
    else:
        ckpt_cfg = dict(ckpt_cfg_raw or {})
        every_n_train_steps = int(
            ckpt_cfg.get("every_n_train_steps", ckpt_cfg.get("every_n_steps", 0)) or 0
        )
        every_n_epochs = int(ckpt_cfg.get("every_n_epochs", 1) or 0)
        # Lightning requires checkpoint triggers to be mutually exclusive.
        # We intentionally prefer epoch-based saving (updates `last.ckpt` once per epoch).
        if every_n_train_steps > 0 and every_n_epochs > 0:
            logger.warning(
                "Both trainer.checkpoint.every_n_train_steps and every_n_epochs are set; "
                "disabling step-based checkpointing and keeping epoch-based saving."
            )
            every_n_train_steps = 0
        if every_n_train_steps <= 0 and every_n_epochs <= 0:
            every_n_train_steps = 0
            every_n_epochs = 1

    checkpoint_cb = ModelCheckpoint(
        dirpath=str(ckpt_dir),
        auto_insert_metric_name=False,
        save_top_k=0,
        save_last=True,
        every_n_train_steps=int(every_n_train_steps),
        every_n_epochs=int(every_n_epochs),
        # Save on train epoch end too (helps when validation is disabled).
        save_on_train_epoch_end=True,
    )
    callbacks.append(checkpoint_cb)
    callbacks.append(LearningRateMonitor(logging_interval="epoch"))
    callbacks.append(HeadCheckpoint(output_dir=str(head_ckpt_dir)))
    if extra_callbacks:
        try:
            callbacks.extend(list(extra_callbacks))
        except Exception:
            # Best-effort; avoid failing training due to callback injection issues.
            pass

    # Optional EMA to smooth weights and (optionally) evaluate/report metrics using EMA weights.
    # This is particularly useful for HPO: Ray Tune will see `val_r2` computed with EMA weights
    # when `trainer.ema.eval_with_ema=true`, aligning the selection metric with final exported weights.
    ema_cfg = cfg.get("trainer", {}).get("ema", {}) or {}
    ema_enabled = bool(ema_cfg.get("enabled", False))
    if ema_enabled:
        try:
            callbacks.append(
                ExponentialMovingAverage(
                    decay=float(ema_cfg.get("decay", 0.999)),
                    update_every_n_steps=int(ema_cfg.get("update_every_n_steps", 1)),
                    start_step=int(ema_cfg.get("start_step", 0)),
                    eval_with_ema=bool(ema_cfg.get("eval_with_ema", True)),
                    apply_at_end=bool(ema_cfg.get("apply_at_end", True)),
                    trainable_only=bool(ema_cfg.get("trainable_only", True)),
                )
            )
        except Exception as e:
            logger.warning(f"EMA callback creation failed, continuing without EMA: {e}")
            ema_enabled = False

    # Optional SWA to stabilize small-batch updates
    swa_cfg = cfg.get("trainer", {}).get("swa", {})
    if bool(swa_cfg.get("enabled", False)) and not ema_enabled:
        try:
            adaptive_lrs = bool(swa_cfg.get("adaptive_lrs", True))
            # 方案 B: build SWA per-param-group LRs (head/UW/LoRA) instead of forcing a single scalar LR.
            auto_lrs = bool(swa_cfg.get("auto_lrs", True))
            freeze_lora = bool(swa_cfg.get("freeze_lora", True))
            head_swa_lr = float(swa_cfg.get("head_swa_lr", swa_cfg.get("swa_lrs", cfg["optimizer"]["lr"])))
            uw_swa_lr = (
                float(swa_cfg.get("uw_swa_lr"))
                if swa_cfg.get("uw_swa_lr", None) is not None
                else None
            )
            # When not freezing LoRA, allow overriding its SWA LR; otherwise it is forced to 0.
            lora_swa_lr = float(swa_cfg.get("lora_swa_lr", 0.0))

            swa_lrs_val: Any
            if auto_lrs:
                swa_lrs_val = _build_swa_lrs_for_model(
                    model,
                    head_swa_lr=head_swa_lr,
                    uw_swa_lr=uw_swa_lr,
                    freeze_lora=freeze_lora,
                    lora_swa_lr=lora_swa_lr,
                )
            else:
                swa_lrs_val = swa_cfg.get("swa_lrs", cfg["optimizer"]["lr"])

            swa_cb = StochasticWeightAveraging(
                swa_lrs=swa_lrs_val,
                swa_epoch_start=float(swa_cfg.get("swa_epoch_start", 0.8)),
                annealing_epochs=int(swa_cfg.get("annealing_epochs", 5)),
                annealing_strategy=str(swa_cfg.get("annealing_strategy", "cos")),
            )
            # IMPORTANT: add adaptive LR patcher BEFORE SWA callback, so it can update
            # SWA's internal target LRs before SWA initializes its scheduler.
            if adaptive_lrs:
                callbacks.append(AdaptiveSwaLrsOnStart(swa_callback=swa_cb))
            callbacks.append(swa_cb)
            # 方案 C: freeze LoRA during SWA (hard stop on grads + lr=0).
            #
            # IMPORTANT: append this callback AFTER SWA so it can override any LR updates
            # performed by SWA schedulers at/after the SWA start boundary.
            if freeze_lora:
                callbacks.append(
                    FreezeLoraOnSwaStart(
                        swa_epoch_start=float(swa_cfg.get("swa_epoch_start", 0.8)),
                        set_lora_lr_to=float(swa_cfg.get("freeze_lora_lr", 0.0)),
                    )
                )
        except Exception as e:
            logger.warning(
                f"SWA callback creation failed, continuing without SWA: {e}"
            )
    elif bool(swa_cfg.get("enabled", False)) and ema_enabled:
        logger.warning("Both EMA and SWA are enabled; ignoring SWA and using EMA only.")

    csv_logger, tb_logger = create_lightning_loggers(log_dir)

    trainer_cfg = cfg["trainer"]
    # For train_all_mode we respect the configured limit_val_batches (default: 1 to
    # only run a minimal dummy validation). For all other modes (single-split and
    # k-fold), always validate on the full validation set (1.0) regardless of the
    # config value.
    if train_all_mode:
        limit_val_batches = trainer_cfg.get("limit_val_batches", 1)
    else:
        limit_val_batches = 1.0

    trainer = pl.Trainer(
        max_epochs=int(trainer_cfg["max_epochs"]),
        accelerator=str(trainer_cfg["accelerator"]),
        devices=trainer_cfg["devices"],
        precision=trainer_cfg["precision"],
        limit_train_batches=trainer_cfg.get("limit_train_batches", 1.0),
        limit_val_batches=limit_val_batches,
        callbacks=callbacks,
        logger=[csv_logger, tb_logger],
        log_every_n_steps=int(trainer_cfg["log_every_n_steps"]),
        accumulate_grad_batches=int(trainer_cfg.get("accumulate_grad_batches", 1)),
        gradient_clip_val=float(trainer_cfg.get("gradient_clip_val", 0.0)),
        gradient_clip_algorithm=str(
            trainer_cfg.get("gradient_clip_algorithm", "norm")
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

