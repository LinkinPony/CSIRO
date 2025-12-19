"""
Head-type specific construction helpers for `BiomassRegressor`.

We keep the public LightningModule API stable while moving large head-specific
branches into dedicated modules:
  - mlp.py
  - fpn.py
  - dpt.py
"""

from __future__ import annotations

from typing import List

from .dpt import init_dpt_head, init_dpt_task_heads
from .fpn import init_fpn_head, init_fpn_task_heads
from .mlp import init_mlp_head, init_mlp_task_heads


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
) -> int:
    """
    Initialize head-specific modules on `model` and return `bottleneck_dim`
    (the feature dim consumed by auxiliary heads like species/state/height).
    """
    ht = str(head_type or "mlp").strip().lower()
    if ht == "mlp":
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
    raise RuntimeError(f"Unexpected head type: {ht!r}")


__all__ = ["init_head_by_type"]


