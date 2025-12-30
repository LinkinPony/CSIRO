"""
Tabular (non-image) experiments and utilities.

This package should only contain logic that is specific to **tabular** models /
tabular baselines. Project-wide shared logic (dataset pivoting, split building,
metrics, etc.) lives elsewhere under `src/`.
"""

from .features import DEFAULT_FEATURE_COLS, build_X, build_y_5d, resolve_feature_cols

__all__ = [
    "DEFAULT_FEATURE_COLS",
    "resolve_feature_cols",
    "build_X",
    "build_y_5d",
]


