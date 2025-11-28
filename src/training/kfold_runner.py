from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Optional

import numpy as np
from loguru import logger

from src.data.datamodule import PastureDataModule
from src.training.single_run import (
    parse_image_size,
    resolve_dataset_area_m2,
    train_single_split,
)


def run_kfold(cfg: Dict, log_dir: Path, ckpt_dir: Path) -> None:
    # Build once to get the full dataframe and other settings used for splitting.
    area_m2 = resolve_dataset_area_m2(cfg)

    base_dm = PastureDataModule(
        data_root=cfg["data"]["root"],
        train_csv=cfg["data"]["train_csv"],
        image_size=parse_image_size(cfg["data"]["image_size"]),
        batch_size=int(cfg["data"]["batch_size"]),
        val_batch_size=int(cfg["data"].get("val_batch_size", cfg["data"]["batch_size"])),
        num_workers=int(cfg["data"]["num_workers"]),
        prefetch_factor=int(cfg["data"].get("prefetch_factor", 2)),
        val_split=float(cfg["data"]["val_split"]),
        target_order=list(cfg["data"]["target_order"]),
        mean=list(cfg["data"]["normalization"]["mean"]),
        std=list(cfg["data"]["normalization"]["std"]),
        train_scale=tuple(cfg["data"]["augment"]["random_resized_crop_scale"]),
        sample_area_m2=float(area_m2),
        zscore_output_path=str(log_dir / "z_score.json"),
        log_scale_targets=bool(cfg["model"].get("log_scale_targets", False)),
        hflip_prob=float(cfg["data"]["augment"]["horizontal_flip_prob"]),
        shuffle=bool(cfg["data"].get("shuffle", True)),
    )
    # MTL toggle
    mtl_cfg = cfg.get("mtl", {})
    mtl_enabled = bool(mtl_cfg.get("enabled", True))
    tasks_cfg = mtl_cfg.get("tasks", {})
    height_enabled = bool(tasks_cfg.get("height", False)) and mtl_enabled
    ndvi_enabled = bool(tasks_cfg.get("ndvi", False)) and mtl_enabled
    species_enabled = bool(tasks_cfg.get("species", False)) and mtl_enabled
    state_enabled = bool(tasks_cfg.get("state", False)) and mtl_enabled

    full_df = base_dm.build_full_dataframe()

    # Infer number of species/state classes (only if tasks enabled).
    num_species_classes: Optional[int]
    num_state_classes: Optional[int]
    if mtl_enabled and (species_enabled or state_enabled):
        try:
            if species_enabled:
                uniques = full_df["Species"].dropna().astype(str).unique().tolist()
                num_species_classes = int(len(sorted(uniques)))
                if num_species_classes <= 1:
                    raise ValueError("Species column has <=1 unique values")
            else:
                num_species_classes = None
        except Exception as e:
            if species_enabled:
                logger.warning(
                    f"Falling back to num_species_classes=2 (reason: {e})"
                )
                num_species_classes = 2
            else:
                num_species_classes = None

        try:
            if state_enabled:
                uniques = full_df["State"].dropna().astype(str).unique().tolist()
                num_state_classes = int(len(sorted(uniques)))
                if num_state_classes <= 1:
                    raise ValueError("State column has <=1 unique values")
            else:
                num_state_classes = None
        except Exception as e:
            if state_enabled:
                logger.warning(
                    f"Falling back to num_state_classes=2 (reason: {e})"
                )
                num_state_classes = 2
            else:
                num_state_classes = None
    else:
        num_species_classes = None
        num_state_classes = None

    num_folds = int(cfg.get("kfold", {}).get("k", 5))
    even_split = bool(cfg.get("kfold", {}).get("even_split", False))

    # Prepare fold indices
    n_samples = len(full_df)
    indices = np.arange(n_samples)
    rng = np.random.default_rng(seed=int(cfg.get("seed", 42)))
    # When even_split is disabled, we use classic K-fold style where each fold
    # uses one contiguous chunk as validation and the remainder as training.
    # When enabled, for each fold we generate a fresh random 50/50 split.
    if not even_split:
        rng.shuffle(indices)
        folds = np.array_split(indices, num_folds)
    else:
        folds = None  # not used under even_split

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
        if even_split:
            perm = rng.permutation(indices)
            n_train = n_samples // 2
            train_idx = perm[:n_train]
            val_idx = perm[n_train:]
        else:
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

        # Delegate actual training of this fold to the shared single-run helper.
        train_single_split(
            cfg,
            fold_log_dir,
            fold_ckpt_dir,
            train_df=train_df,
            val_df=val_df,
            train_all_mode=False,
            num_species_classes=num_species_classes,
            num_state_classes=num_state_classes,
        )


