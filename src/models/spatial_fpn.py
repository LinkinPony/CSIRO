"""
Backward-compatible re-export.

The implementation has been moved under `src.models.regressor.heads` to keep the
regressor head-related code in one place, while preserving the historical import path:

    from src.models.spatial_fpn import FPNHeadConfig, FPNScalarHead
"""

from src.models.regressor.heads.spatial_fpn import (  # noqa: F401
    FPNHeadConfig,
    FPNScalarHead,
    _infer_patch_grid_hw,
)

__all__ = ["FPNHeadConfig", "FPNScalarHead", "_infer_patch_grid_hw"]

