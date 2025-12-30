from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


# Canonical 5D biomass ordering used across the project:
# [Dry_Clover_g, Dry_Dead_g, Dry_Green_g, GDM_g, Dry_Total_g]
TARGETS_5D_ORDER: list[str] = [
    "Dry_Clover_g",
    "Dry_Dead_g",
    "Dry_Green_g",
    "GDM_g",
    "Dry_Total_g",
]

# Competition-style weights in the same order as TARGETS_5D_ORDER
BIOMASS_5D_WEIGHTS: np.ndarray = np.asarray([0.1, 0.1, 0.1, 0.2, 0.5], dtype=np.float64)


def weighted_r2_logspace(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    weights: Optional[np.ndarray] = None,
    eps: float = 1e-8,
    return_per_target: bool = False,
) -> float | Tuple[float, np.ndarray]:
    """
    Weighted R^2 in log-space, matching the project's internal metric definition.

    - Evaluate on `log1p(clamp(x, min=0))`, per target dimension.
    - R^2 baseline uses the mean of the evaluation targets (standard R^2).
    - Aggregate across targets with fixed weights (competition-style).
    """

    yt = np.asarray(y_true, dtype=np.float64)
    yp = np.asarray(y_pred, dtype=np.float64)

    if yt.shape != yp.shape:
        raise ValueError(f"Shape mismatch: y_true={yt.shape} vs y_pred={yp.shape}")
    if yt.ndim != 2:
        raise ValueError(f"Expected 2D arrays (N, D), got shape: {yt.shape}")

    w = BIOMASS_5D_WEIGHTS if weights is None else np.asarray(weights, dtype=np.float64)
    if w.ndim != 1 or w.shape[0] != yt.shape[1]:
        raise ValueError(f"Invalid weights shape: {w.shape} for D={yt.shape[1]}")

    yt = np.maximum(yt, 0.0)
    yp = np.maximum(yp, 0.0)

    yt_log = np.log1p(yt)
    yp_log = np.log1p(yp)

    mean_log = np.mean(yt_log, axis=0)
    ss_res = np.sum((yt_log - yp_log) ** 2, axis=0)
    ss_tot = np.sum((yt_log - mean_log) ** 2, axis=0)
    r2_per = 1.0 - (ss_res / (ss_tot + float(eps)))

    valid = np.isfinite(r2_per)
    w_eff = w * valid.astype(np.float64)
    denom = max(float(np.sum(w_eff)), float(eps))
    r2_weighted = float(np.sum(w_eff * r2_per) / denom)

    if return_per_target:
        return r2_weighted, r2_per
    return r2_weighted


