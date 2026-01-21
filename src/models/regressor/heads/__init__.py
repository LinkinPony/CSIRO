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
    # Mamba axial head config (PyTorch-only)
    mamba_dim: int = 320,
    mamba_depth: int = 4,
    mamba_patch_size: int = 16,
    mamba_d_conv: int = 3,
    mamba_bidirectional: bool = True,
    # EoMT-style query pooling config
    eomt_num_queries: int = 16,
    eomt_num_layers: int = 2,
    eomt_num_heads: int = 8,
    eomt_ffn_dim: int = 2048,
    eomt_query_pool: str = "mean",
    eomt_use_mean_query: bool = True,
    eomt_use_mean_patch: bool = False,
    eomt_use_cls_token: bool = False,
    eomt_proj_dim: int = 0,
    eomt_proj_activation: str = "relu",
    eomt_proj_dropout: float = 0.0,
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
    if ht in ("mamba", "mamba_axial", "mamba_head"):
        from .mamba import init_mamba_head, init_mamba_task_heads

        bottleneck_dim = init_mamba_head(
            model,
            embedding_dim=int(embedding_dim),
            mamba_dim=int(mamba_dim),
            mamba_depth=int(mamba_depth),
            mamba_patch_size=int(mamba_patch_size),
            mamba_d_conv=int(mamba_d_conv),
            mamba_bidirectional=bool(mamba_bidirectional),
            hidden_dims=list(hidden_dims),
            head_activation=str(head_activation),
            dropout=float(dropout),
        )
        init_mamba_task_heads(model, bottleneck_dim=int(bottleneck_dim))
        return int(bottleneck_dim)
    if ht in ("eomt", "eomt_query", "query_pool", "qpool"):
        from .eomt import init_eomt_head, init_eomt_task_heads

        bottleneck_dim = init_eomt_head(
            model,
            embedding_dim=int(embedding_dim),
            eomt_num_queries=int(eomt_num_queries),
            eomt_num_layers=int(eomt_num_layers),
            eomt_num_heads=int(eomt_num_heads),
            eomt_ffn_dim=int(eomt_ffn_dim),
            eomt_query_pool=str(eomt_query_pool),
            eomt_use_mean_query=bool(eomt_use_mean_query),
            eomt_use_mean_patch=bool(eomt_use_mean_patch),
            eomt_use_cls_token=bool(eomt_use_cls_token),
            eomt_proj_dim=int(eomt_proj_dim),
            eomt_proj_activation=str(eomt_proj_activation),
            eomt_proj_dropout=float(eomt_proj_dropout or 0.0),
            hidden_dims=list(hidden_dims),
            head_activation=str(head_activation),
            dropout=float(dropout),
        )
        init_eomt_task_heads(model, bottleneck_dim=int(bottleneck_dim))
        return int(bottleneck_dim)
    raise RuntimeError(f"Unexpected head type: {ht!r}")


__all__ = ["init_head_by_type"]


