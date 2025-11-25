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

    # Group-aware k-fold: ensure that all samples from the same (Sampling_Date, State)
    # combination are kept together in the same fold. If anything goes wrong, we fall
    # back to the previous per-sample splitting behavior.
    group_to_indices: Optional[Dict[tuple, list[int]]] = None
    try:
        dates = full_df["Sampling_Date"]
        states = full_df["State"]
        group_map: Dict[tuple, list[int]] = {}
        for idx, (d, s) in enumerate(zip(dates.tolist(), states.tolist())):
            key = (str(d), str(s))
            if key not in group_map:
                group_map[key] = []
            group_map[key].append(idx)
        if group_map:
            group_to_indices = group_map
    except Exception as e:
        logger.warning(
            f"Grouped k-fold by Sampling_Date+State disabled; "
            f"falling back to per-sample split. Reason: {e}"
        )
        group_to_indices = None

    # Prepare fold indices
    n_samples = len(full_df)
    indices = np.arange(n_samples)
    rng = np.random.default_rng(seed=int(cfg.get("seed", 42)))

    if group_to_indices is None:
        # Original behavior: operate at individual-sample level.
        # When even_split is disabled, we use classic K-fold style where each fold
        # uses one contiguous chunk as validation and the remainder as training.
        # When enabled, for each fold we generate a fresh random 50/50 split.
        if not even_split:
            rng.shuffle(indices)
            folds = np.array_split(indices, num_folds)
        else:
            folds = None  # not used under even_split
    else:
        # Group-aware behavior: assign whole (date, state) groups to folds.
        if not even_split:
            # One global assignment of groups to folds, greedily balancing by sample count.
            group_keys = list(group_to_indices.keys())
            perm = rng.permutation(len(group_keys))
            fold_lists = [[] for _ in range(num_folds)]
            fold_sizes = [0 for _ in range(num_folds)]
            for gi in perm:
                key = group_keys[int(gi)]
                idxs = group_to_indices[key]
                j = int(np.argmin(fold_sizes))
                fold_lists[j].extend(idxs)
                fold_sizes[j] += len(idxs)
            folds = [np.array(sorted(lst)) for lst in fold_lists]
        else:
            # For even_split, we will construct fresh random 50/50 splits by group
            # inside the per-fold loop below.
            folds = None

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
            if group_to_indices is not None:
                # Random 50/50 split at the (date, state) group level.
                group_keys = list(group_to_indices.keys())
                perm = rng.permutation(len(group_keys))
                target_train = n_samples // 2
                train_idx_list: list[int] = []
                val_idx_list: list[int] = []
                train_count = 0
                for gi in perm:
                    key = group_keys[int(gi)]
                    idxs = group_to_indices[key]
                    if train_count + len(idxs) <= target_train or not train_idx_list:
                        train_idx_list.extend(idxs)
                        train_count += len(idxs)
                    else:
                        val_idx_list.extend(idxs)
                train_idx = np.array(sorted(train_idx_list))
                val_idx = np.array(sorted(val_idx_list))
            else:
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
    # After all folds are trained, summarize per-fold metrics and run an additional
    # train_all pass using the full dataset when train_all is disabled in the config.
    # 1) Aggregate per-fold metrics into a JSON file with:
    #    - metrics at the epoch with minimal val_loss ("best")
    #    - metrics at the final epoch ("last")
    try:
        import pandas as pd  # type: ignore[import]
        import json
        from package_artifacts import (  # type: ignore[import]
            find_latest_metrics_csv,
            pick_best_epoch_from_metrics,
        )

        per_fold_records = []
        best_rows = []
        last_rows = []
        for fold_idx in range(num_folds):
            fold_log_dir = log_dir / f"fold_{fold_idx}"
            metrics_csv = find_latest_metrics_csv(fold_log_dir)
            if metrics_csv is None:
                logger.warning(f"No metrics.csv found for fold {fold_idx} under {fold_log_dir}")
                continue
            try:
                df = pd.read_csv(str(metrics_csv))
            except Exception as e:
                logger.warning(f"Reading metrics.csv for fold {fold_idx} failed: {e}")
                continue
            if "epoch" not in df.columns:
                logger.warning(f"metrics.csv for fold {fold_idx} missing 'epoch' column")
                continue

            # Prefer rows that actually contain validation metrics; Lightning typically
            # logs separate train/val rows per epoch and the last row is often the
            # training metrics only (val_* columns are NaN). We therefore filter to
            # rows where `val_loss` is non-null when available, and then keep the
            # last logged validation row per epoch.
            metrics_df = df
            if "val_loss" in df.columns:
                df_val = df[df["val_loss"].notna()]
                if not df_val.empty:
                    metrics_df = df_val
            metrics_df = metrics_df.groupby("epoch").tail(1).reset_index(drop=True)

            # Helper to extract numeric metrics from a row.
            def _extract_metrics(row) -> dict:
                out: dict = {}
                try:
                    out["epoch"] = int(row["epoch"])
                except Exception:
                    pass
                for name in ["val_loss", "val_mae", "val_mse", "val_r2"]:
                    if name in row.index:
                        try:
                            value = float(row[name])
                        except Exception:
                            continue
                        if value == value:  # skip NaN
                            out[name] = value
                return out

            # Best epoch: epoch with minimal val_loss (using helper from package_artifacts
            # when possible, otherwise local argmin over metrics_df).
            best_epoch = pick_best_epoch_from_metrics(metrics_csv)
            if best_epoch is not None and (metrics_df["epoch"] == best_epoch).any():
                best_row = metrics_df[metrics_df["epoch"] == best_epoch].iloc[-1]
            else:
                if "val_loss" in metrics_df.columns and not metrics_df["val_loss"].isna().all():
                    best_row = metrics_df.loc[metrics_df["val_loss"].idxmin()]
                else:
                    # Fallback: use the last epoch with metrics available
                    best_row = metrics_df.iloc[metrics_df["epoch"].idxmax()]

            # Last epoch: maximal epoch index among validation rows.
            last_row = metrics_df.iloc[metrics_df["epoch"].idxmax()]

            best_metrics = _extract_metrics(best_row)
            last_metrics = _extract_metrics(last_row)

            best_rows.append(best_metrics)
            last_rows.append(last_metrics)
            per_fold_records.append(
                {
                    "fold": int(fold_idx),
                    "best": best_metrics,
                    "last": last_metrics,
                }
            )

        records = per_fold_records

        # Append mean metrics across folds (for best and last separately).
        if best_rows:
            best_df = pd.DataFrame(best_rows)
            mean_best: dict = {"epoch": -1}
            for name in ["val_loss", "val_mae", "val_mse", "val_r2"]:
                if name in best_df.columns:
                    try:
                        mv = float(best_df[name].mean())
                    except Exception:
                        continue
                    if mv == mv:
                        mean_best[name] = mv
        else:
            mean_best = {}

        if last_rows:
            last_df = pd.DataFrame(last_rows)
            mean_last: dict = {"epoch": -1}
            for name in ["val_loss", "val_mae", "val_mse", "val_r2"]:
                if name in last_df.columns:
                    try:
                        mv = float(last_df[name].mean())
                    except Exception:
                        continue
                    if mv == mv:
                        mean_last[name] = mv
        else:
            mean_last = {}

        if records and (mean_best or mean_last):
            records.append(
                {
                    "fold": "mean",
                    "best": mean_best if mean_best else None,
                    "last": mean_last if mean_last else None,
                }
            )

            summary_path = log_dir / "kfold_metrics_summary.json"
            try:
                with open(summary_path, "w", encoding="utf-8") as f:
                    json.dump(records, f, indent=2)
                logger.info(f"Saved k-fold metrics summary to {summary_path}")
            except Exception as e:
                logger.warning(f"Saving k-fold metrics summary failed: {e}")
    except Exception as e:
        logger.warning(f"K-fold metrics summary step failed: {e}")

    # 2) Train a final model on all data (train_all-style) after k-fold training.
    #    This uses the same full_df built above and constructs a dummy validation
    #    split (single sample) so that training can proceed with minimal val overhead.
    try:
        import pandas as pd  # type: ignore[import]

        rng_seed = int(cfg.get("seed", 42))
        if len(full_df) < 1:
            raise ValueError("No samples available to construct train_all splits.")
        dummy_val = full_df.sample(n=1, random_state=rng_seed)
        # Optionally duplicate the single row to avoid degenerate loader corner cases
        val_all_df = pd.concat([dummy_val], ignore_index=True)
        train_all_df = full_df.reset_index(drop=True)

        all_log_dir = log_dir / "train_all"
        all_ckpt_dir = ckpt_dir / "train_all"
        all_log_dir.mkdir(parents=True, exist_ok=True)
        all_ckpt_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Starting additional train_all run on full dataset after k-fold training...")
        train_single_split(
            cfg,
            all_log_dir,
            all_ckpt_dir,
            train_df=train_all_df,
            val_df=val_all_df,
            train_all_mode=True,
            num_species_classes=num_species_classes,
            num_state_classes=num_state_classes,
        )
    except Exception as e:
        logger.warning(f"Post k-fold train_all run failed or skipped: {e}")

