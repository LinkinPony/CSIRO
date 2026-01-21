"""
Backward-compatible re-export.

The implementation lives under `src.models.regressor.heads` to keep the regressor
head-related code in one place, while preserving a stable import path:

    from src.models.mamba_head import MambaHeadConfig, MambaAxialScalarHead, MambaMultiLayerScalarHead
"""

from __future__ import annotations

from src.models.regressor.heads.mamba import (  # noqa: F401
    MambaAxialScalarHead,
    MambaHeadConfig,
    MambaMultiLayerScalarHead,
)

__all__ = ["MambaHeadConfig", "MambaAxialScalarHead", "MambaMultiLayerScalarHead"]

