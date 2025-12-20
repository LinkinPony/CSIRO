"""
Head-type specific construction helpers for `BiomassRegressor`.

We keep the public LightningModule API stable while moving large head-specific
branches into dedicated modules:
  - mlp.py
  - fpn.py
  - dpt.py
  - vitdet.py

IMPORTANT:
----------
This package also contains *inference* head modules (e.g. `dpt_scalar_head.py`).
Python executes this `__init__.py` when importing any submodule under
`src.models.regressor.heads.*`. To keep lightweight inference environments
(e.g. Kaggle) working without optional training dependencies (Lightning,
Detectron2), we avoid importing those dependencies at module import time.
"""

from __future__ import annotations

from typing import List


def init_head_by_type(
    model,
    *,
    head_type: str,
    embedding_dim: int,
    hidden_dims: List[int],
    head_activation: str,
    dropout: float,
    # FPN config
    fpn_dim: int,
    fpn_num_levels: int,
    fpn_patch_size: int,
    fpn_reverse_level_order: bool,
    # DPT config
    dpt_features: int,
    dpt_patch_size: int,
    dpt_readout: str,
    # ViTDet config
    vitdet_dim: int,
    vitdet_patch_size: int,
    vitdet_scale_factors: List[float],
) -> int:
    """
    Initialize head-specific modules on `model` and return `bottleneck_dim`
    (the feature dim consumed by auxiliary heads like species/state/height).
    """
    ht = str(head_type or "mlp").strip().lower()
    if ht == "mlp":
        from .mlp import init_mlp_head, init_mlp_task_heads

        bottleneck_dim = init_mlp_head(
            model,
            embedding_dim=embedding_dim,
            hidden_dims=list(hidden_dims),
            head_activation=str(head_activation),
            dropout=float(dropout),
        )
        init_mlp_task_heads(model, bottleneck_dim=int(bottleneck_dim))
        return int(bottleneck_dim)
    if ht == "fpn":
        from .fpn import init_fpn_head, init_fpn_task_heads

        bottleneck_dim = init_fpn_head(
            model,
            embedding_dim=embedding_dim,
            fpn_dim=int(fpn_dim),
            fpn_num_levels=int(fpn_num_levels),
            fpn_patch_size=int(fpn_patch_size),
            fpn_reverse_level_order=bool(fpn_reverse_level_order),
            hidden_dims=list(hidden_dims),
            head_activation=str(head_activation),
            dropout=float(dropout),
        )
        init_fpn_task_heads(model, bottleneck_dim=int(bottleneck_dim))
        return int(bottleneck_dim)
    if ht == "dpt":
        from .dpt import init_dpt_head, init_dpt_task_heads

        bottleneck_dim = init_dpt_head(
            model,
            embedding_dim=embedding_dim,
            dpt_features=int(dpt_features),
            dpt_patch_size=int(dpt_patch_size),
            dpt_readout=str(dpt_readout),
            hidden_dims=list(hidden_dims),
            head_activation=str(head_activation),
            dropout=float(dropout),
        )
        init_dpt_task_heads(model, bottleneck_dim=int(bottleneck_dim))
        return int(bottleneck_dim)
    if ht == "vitdet":
        from .vitdet import init_vitdet_head, init_vitdet_task_heads

        bottleneck_dim = init_vitdet_head(
            model,
            embedding_dim=embedding_dim,
            vitdet_dim=int(vitdet_dim),
            vitdet_patch_size=int(vitdet_patch_size),
            vitdet_scale_factors=list(vitdet_scale_factors or [2.0, 1.0, 0.5]),
            hidden_dims=list(hidden_dims),
            head_activation=str(head_activation),
            dropout=float(dropout),
        )
        init_vitdet_task_heads(model, bottleneck_dim=int(bottleneck_dim))
        return int(bottleneck_dim)
    raise RuntimeError(f"Unexpected head type: {ht!r}")


__all__ = ["init_head_by_type"]


