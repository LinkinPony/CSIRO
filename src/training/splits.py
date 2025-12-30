from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Optional

import numpy as np
import pandas as pd
from loguru import logger


def build_kfold_splits(
    df: pd.DataFrame,
    cfg: Optional[Mapping[str, Any]] = None,
    *,
    k: Optional[int] = None,
    even_split: Optional[bool] = None,
    group_by_date_state: Optional[bool] = None,
    seed: Optional[int] = None,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Build fold splits (train_idx, val_idx) matching the project's `run_kfold()` logic.

    Notes
    -----
    - When `group_by_date_state=True` (default), we keep all samples from the same
      (Sampling_Date, State) pair in the same fold.
    - When `even_split=True`, this is **not** classic k-fold. For each fold, a fresh
      random 50/50 split is generated (still respecting grouping if enabled).
    """

    cfg = cfg or {}
    kfold_cfg = dict(cfg.get("kfold", {}) or {})

    num_folds = int(k if k is not None else kfold_cfg.get("k", 5))
    even_split_flag = bool(
        even_split if even_split is not None else kfold_cfg.get("even_split", False)
    )
    group_by_date_state_flag = bool(
        group_by_date_state
        if group_by_date_state is not None
        else kfold_cfg.get("group_by_date_state", True)
    )
    rng_seed = int(seed if seed is not None else cfg.get("seed", 42))

    if num_folds <= 0:
        raise ValueError(f"Invalid k: {num_folds}")
    if len(df) < 2:
        raise ValueError(f"Need at least 2 samples for k-fold, got: {len(df)}")

    # Prepare indices and RNG (must match `src/training/kfold_runner.py` semantics)
    n_samples = int(len(df))
    indices = np.arange(n_samples)
    rng = np.random.default_rng(seed=rng_seed)

    # Optionally build group assignments based on Sampling_Date + State
    use_grouped_kfold = (
        group_by_date_state_flag
        and "Sampling_Date" in df.columns
        and "State" in df.columns
    )
    if group_by_date_state_flag and not use_grouped_kfold:
        logger.warning(
            "k-fold config requests grouped splitting by (Sampling_Date, State) "
            "but the dataframe is missing one or both columns. Falling back to per-image k-fold. "
            "This can significantly inflate CV metrics."
        )

    group_indices = None  # type: ignore[assignment]
    fold_group_ids = None  # type: ignore[assignment]
    group_sizes = None  # type: ignore[assignment]

    if use_grouped_kfold:
        # Build a group id per sample: (Sampling_Date, State)
        dates = df["Sampling_Date"].astype(str).to_numpy()
        states = df["State"].astype(str).to_numpy()
        group_labels = np.array([f"{d}__{s}" for d, s in zip(dates, states)], dtype=object)
        unique_groups, group_indices = np.unique(group_labels, return_inverse=True)
        n_groups = int(len(unique_groups))
        group_order = np.arange(n_groups)
        rng.shuffle(group_order)
        group_sizes = np.bincount(group_indices, minlength=n_groups)

        # For classic k-fold, assign whole groups to folds greedily by size.
        if not even_split_flag:
            fold_group_ids = [[] for _ in range(num_folds)]
            fold_sizes = [0 for _ in range(num_folds)]
            for g in group_order:
                best_fold = int(np.argmin(fold_sizes))
                fold_group_ids[best_fold].append(int(g))
                fold_sizes[best_fold] += int(group_sizes[g])
        else:
            fold_group_ids = None
    else:
        # Fallback: original per-sample k-fold behaviour.
        if not even_split_flag:
            rng.shuffle(indices)
            folds = np.array_split(indices, num_folds)
        else:
            folds = None  # not used under even_split

    splits: list[tuple[np.ndarray, np.ndarray]] = []
    for fold_idx in range(num_folds):
        if use_grouped_kfold:
            assert group_indices is not None
            assert group_sizes is not None
            if even_split_flag:
                # Fresh random 50/50 split of groups for each fold.
                n_groups = int(group_sizes.shape[0])
                perm_groups = rng.permutation(np.arange(n_groups))
                target_train = n_samples // 2
                cum_sizes = group_sizes[perm_groups].cumsum()
                train_mask_groups = cum_sizes <= target_train
                if not train_mask_groups.any():
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
            if even_split_flag:
                perm = rng.permutation(indices)
                n_train = n_samples // 2
                train_idx = perm[:n_train]
                val_idx = perm[n_train:]
            else:
                val_idx = folds[fold_idx]  # type: ignore[index]
                train_idx = np.concatenate(
                    [folds[i] for i in range(num_folds) if i != fold_idx]  # type: ignore[arg-type]
                )

        splits.append((train_idx.astype(int), val_idx.astype(int)))

    return splits


