"""
Backward-compatible re-export.

The implementation lives under `src.models.regressor.heads` to keep the regressor
head-related code in one place, while preserving a stable import path:

    from src.models.vitdet_head import ViTDetHeadConfig, ViTDetScalarHead, ViTDetMultiLayerScalarHead
"""

from __future__ import annotations

from src.models.regressor.heads.vitdet import (  # noqa: F401
    ViTDetHeadConfig,
    ViTDetMultiLayerScalarHead,
    ViTDetScalarHead,
)

__all__ = ["ViTDetHeadConfig", "ViTDetScalarHead", "ViTDetMultiLayerScalarHead"]


