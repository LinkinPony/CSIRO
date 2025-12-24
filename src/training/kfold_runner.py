from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Optional, List

import numpy as np
from loguru import logger

from src.data.datamodule import PastureDataModule
from src.training.single_run import (
    parse_image_size,
    resolve_dataset_area_m2,
    train_single_split,
)


def _find_latest_metrics_csv(log_dir: Path) -> Optional[Path]:
    """
    Locate the most recent metrics.csv under a given log_dir (any depth).
    Returns None if not found.
    """
    candidates: List[Path] = []
    if not log_dir.exists():
        return None
    for root, _dirs, files in os.walk(log_dir):
        for name in files:
            if name == "metrics.csv":
                candidates.append(Path(root) / name)
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def _extract_final_metrics(metrics_csv: Path) -> Dict[str, float]:
    """
    Read a Lightning CSVLogger metrics.csv and extract the last-epoch
    validation metrics for this fold (loss/mae/mse/r2 when available).
    """
    import pandas as pd

    df = pd.read_csv(str(metrics_csv))
    if "epoch" not in df.columns:
        raise ValueError(f"metrics.csv missing epoch column: {metrics_csv}")
    # Take the last logged row per epoch, then keep the final epoch.
    gb = df.groupby("epoch").tail(1).reset_index(drop=True)
    last = gb.iloc[-1]

    def _pick(col_candidates: List[str]) -> Optional[str]:
        for c in col_candidates:
            if c in gb.columns:
                return c
        return None

    # Handle potential multi-val suffixes
    col_loss = _pick(["val_loss", "val_loss/dataloader_idx_0"])
    col_mae = _pick(["val_mae", "val_mae/dataloader_idx_0"])
    col_mse = _pick(["val_mse", "val_mse/dataloader_idx_0"])
    col_r2 = _pick(["val_r2"])

    out: Dict[str, float] = {}
    out["epoch"] = float(last["epoch"])
    if col_loss:
        out["val_loss"] = float(last[col_loss])
    if col_mae:
        out["val_mae"] = float(last[col_mae])
    if col_mse:
        out["val_mse"] = float(last[col_mse])
    if col_r2:
        out["val_r2"] = float(last[col_r2])
    return out


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

    kfold_cfg = cfg.get("kfold", {})
    num_folds = int(kfold_cfg.get("k", 5))
    even_split = bool(kfold_cfg.get("even_split", False))
    # By default, perform grouped k-fold so that all samples from the same
    # (Sampling_Date, State) pair stay in the same fold.
    group_by_date_state = bool(kfold_cfg.get("group_by_date_state", True))

    # Prepare indices and RNG
    n_samples = len(full_df)
    indices = np.arange(n_samples)
    rng = np.random.default_rng(seed=int(cfg.get("seed", 42)))

    # Optionally build group assignments based on Sampling_Date + State
    use_grouped_kfold = (
        group_by_date_state
        and "Sampling_Date" in full_df.columns
        and "State" in full_df.columns
    )
    if group_by_date_state and not use_grouped_kfold:
        logger.warning(
            "k-fold config requests grouped splitting by (Sampling_Date, State) "
            "but the dataframe is missing one or both columns. Falling back to per-image k-fold. "
            "This can significantly inflate CV metrics. "
            "Ensure your datamodule includes 'Sampling_Date' and 'State' in build_full_dataframe()."
        )

    group_indices = None  # type: ignore[assignment]
    fold_group_ids = None  # type: ignore[assignment]
    group_sizes = None  # type: ignore[assignment]

    if use_grouped_kfold:
        # Build a group id per sample: (Sampling_Date, State)
        dates = full_df["Sampling_Date"].astype(str).to_numpy()
        states = full_df["State"].astype(str).to_numpy()
        group_labels = np.array(
            [f"{d}__{s}" for d, s in zip(dates, states)], dtype=object
        )
        unique_groups, group_indices = np.unique(
            group_labels, return_inverse=True
        )
        n_groups = len(unique_groups)
        group_order = np.arange(n_groups)
        rng.shuffle(group_order)
        group_sizes = np.bincount(group_indices, minlength=n_groups)

        # For classic k-fold, assign whole groups to folds greedily by size.
        if not even_split:
            fold_group_ids = [[] for _ in range(num_folds)]
            fold_sizes = [0 for _ in range(num_folds)]
            for g in group_order:
                # Always assign the next group to the currently smallest fold.
                best_fold = int(np.argmin(fold_sizes))
                fold_group_ids[best_fold].append(int(g))
                fold_sizes[best_fold] += int(group_sizes[g])
        else:
            fold_group_ids = None
    else:
        # Fallback: original per-sample k-fold behaviour.
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

    fold_summaries: List[Dict[str, float]] = []

    for fold_idx in range(num_folds):
        if use_grouped_kfold:
            assert group_indices is not None
            assert group_sizes is not None
            if even_split:
                # For even_split under grouped k-fold, generate a fresh random
                # 50/50 split of groups for each fold.
                n_groups = int(group_sizes.shape[0])
                perm_groups = rng.permutation(np.arange(n_groups))
                # Greedy accumulate group sizes until ~50% of samples are covered.
                target_train = n_samples // 2
                cum_sizes = group_sizes[perm_groups].cumsum()
                train_mask_groups = cum_sizes <= target_train
                if not train_mask_groups.any():
                    # Ensure at least one group ends up in the training set
                    train_mask_groups[0] = True
                train_group_ids = set(perm_groups[train_mask_groups])
                train_mask = np.isin(group_indices, list(train_group_ids))
                val_mask = ~train_mask
                train_idx = np.where(train_mask)[0]
                val_idx = np.where(val_mask)[0]
            else:
                assert fold_group_ids is not None
                val_group_ids = set(fold_group_ids[fold_idx])
                val_mask = np.isin(group_indices, list(val_group_ids))
                val_idx = np.where(val_mask)[0]
                train_idx = np.where(~val_mask)[0]
        else:
            if even_split:
                perm = rng.permutation(indices)
                n_train = n_samples // 2
                train_idx = perm[:n_train]
                val_idx = perm[n_train:]
            else:
                val_idx = folds[fold_idx]
                train_idx = np.concatenate(
                    [folds[i] for i in range(num_folds) if i != fold_idx]
                )

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

        # After training this fold, try to extract its final validation metrics.
        try:
            metrics_csv = _find_latest_metrics_csv(fold_log_dir)
            if metrics_csv is not None:
                m = _extract_final_metrics(metrics_csv)
                m["fold"] = float(fold_idx)
                fold_summaries.append(m)
            else:
                logger.warning(f"No metrics.csv found for fold {fold_idx} under {fold_log_dir}")
        except Exception as e:
            logger.warning(f"Failed to extract metrics for fold {fold_idx}: {e}")

    # Write a summary JSON aggregating per-fold metrics at the root log_dir.
    if fold_summaries:
        import json
        import math

        summary_path = log_dir / "kfold_metrics.json"
        # Compute simple averages for shared keys across folds.
        keys = set().union(*(m.keys() for m in fold_summaries)) - {"fold"}
        avg: Dict[str, float] = {}
        for k in keys:
            vals: List[float] = []
            for m in fold_summaries:
                v = m.get(k, None)
                if v is None:
                    continue
                try:
                    vv = float(v)
                except Exception:
                    continue
                if math.isfinite(vv):
                    vals.append(vv)
            if vals:
                avg[k] = float(sum(vals) / len(vals))
        payload = {
            "num_folds": num_folds,
            "per_fold": fold_summaries,
            "average": avg,
        }
        try:
            with open(summary_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
            logger.info("Saved k-fold metrics summary -> {}", summary_path)
        except Exception as e:
            logger.warning(f"Failed to write k-fold metrics summary JSON: {e}")


