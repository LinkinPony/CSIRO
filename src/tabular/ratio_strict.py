from __future__ import annotations

from typing import Any


def apply_ratio_strict_5d(y_pred: Any, *, eps: float = 1e-8):
    """
    Apply the "ratio_strict" post-processing constraint to 5D biomass predictions.

    Expected column order matches `src.metrics.TARGETS_5D_ORDER`:
      [Dry_Clover_g, Dry_Dead_g, Dry_Green_g, GDM_g, Dry_Total_g]

    Logic (per row):
      1) total = predicted Dry_Total_g
      2) total_sum = clover + dead + green
      3) total_final = (total + total_sum) / 2
      4) Recompute (clover, dead, green) by preserving their proportions and scaling to total_final
      5) gdm = clover + green
    """
    import numpy as np

    yp = np.asarray(y_pred, dtype=np.float64)
    if yp.ndim != 2 or yp.shape[1] < 5:
        raise ValueError(f"apply_ratio_strict_5d expects array shape (N,>=5); got {yp.shape}")

    # Copy so we don't mutate caller-owned buffers.
    out = yp.copy()

    # Canonical indices (TARGETS_5D_ORDER)
    IDX_CLOVER = 0
    IDX_DEAD = 1
    IDX_GREEN = 2
    IDX_GDM = 3
    IDX_TOTAL = 4

    clover = np.maximum(out[:, IDX_CLOVER], 0.0)
    dead = np.maximum(out[:, IDX_DEAD], 0.0)
    green = np.maximum(out[:, IDX_GREEN], 0.0)
    total = np.maximum(out[:, IDX_TOTAL], 0.0)

    total_sum = clover + dead + green
    total_final = 0.5 * (total + total_sum)

    denom = np.maximum(total_sum, float(eps))
    r_clover = clover / denom
    r_dead = dead / denom
    r_green = green / denom

    # If total_sum is ~0, proportions are undefined: fall back to uniform split.
    mask = total_sum <= float(eps)
    if np.any(mask):
        r_clover[mask] = 1.0 / 3.0
        r_dead[mask] = 1.0 / 3.0
        r_green[mask] = 1.0 / 3.0

    clover_new = total_final * r_clover
    dead_new = total_final * r_dead
    green_new = total_final * r_green
    gdm_new = clover_new + green_new

    out[:, IDX_CLOVER] = clover_new
    out[:, IDX_DEAD] = dead_new
    out[:, IDX_GREEN] = green_new
    out[:, IDX_GDM] = gdm_new
    out[:, IDX_TOTAL] = total_final

    out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    out = np.maximum(out, 0.0)
    return out


