"""
Backward-compatible re-export.

The implementation has been moved under `src.models.regressor.heads` to keep the
regressor head-related code in one place, while preserving the historical import path:

    from src.models.dpt_scalar_head import DPTHeadConfig, DPTScalarHead
"""

from __future__ import annotations

from src.models.regressor.heads.dpt_scalar_head import (  # noqa: F401
    DPTHeadConfig,
    DPTScalarHead,
)

__all__ = ["DPTHeadConfig", "DPTScalarHead"]


