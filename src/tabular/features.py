from __future__ import annotations

from collections.abc import Sequence
from typing import Optional

import numpy as np
import pandas as pd

from src.metrics import TARGETS_5D_ORDER


DEFAULT_FEATURE_COLS: list[str] = [
    "Sampling_Date",
    "State",
    "Species",
    "Pre_GSHH_NDVI",
    "Height_Ave_cm",
]


def resolve_feature_cols(
    df: pd.DataFrame, feature_cols: Optional[Sequence[str]] = None
) -> list[str]:
    cols = list(feature_cols) if feature_cols is not None else list(DEFAULT_FEATURE_COLS)
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing feature columns in dataframe: {missing}")
    return cols


def build_X(df: pd.DataFrame, *, feature_cols: Optional[Sequence[str]] = None) -> pd.DataFrame:
    cols = resolve_feature_cols(df, feature_cols)
    # Keep as DataFrame so TabPFN can infer categorical/text columns properly.
    return df.loc[:, cols].copy()


def build_y_5d(
    df: pd.DataFrame,
    *,
    targets_5d_order: Optional[Sequence[str]] = None,
    fillna: float = 0.0,
) -> np.ndarray:
    targets = list(targets_5d_order) if targets_5d_order is not None else TARGETS_5D_ORDER
    missing = [c for c in targets if c not in df.columns]
    if missing:
        raise KeyError(f"Missing target columns in dataframe: {missing}")
    y = df.loc[:, targets].to_numpy(dtype=np.float64, copy=True)
    if fillna is not None:
        y = np.nan_to_num(y, nan=float(fillna), posinf=float(fillna), neginf=float(fillna))
    return y


