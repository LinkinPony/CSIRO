from __future__ import annotations

from typing import List

from torch import nn

from ...eomt_injected_head import EoMTInjectedQueryHeadConfig, EoMTInjectedQueryScalarHead


def init_eomt_head(
    model,
    *,
    embedding_dim: int,
    eomt_num_queries: int,
    eomt_num_layers: int,
    eomt_num_heads: int,
    eomt_ffn_dim: int,
    eomt_query_pool: str,
    eomt_use_mean_query: bool = True,
    eomt_use_mean_patch: bool = False,
    eomt_use_cls_token: bool = False,
    eomt_proj_dim: int = 0,
    eomt_proj_activation: str = "relu",
    eomt_proj_dropout: float = 0.0,
    hidden_dims: List[int],
    head_activation: str,
    dropout: float,
) -> int:
    """
    Initialize an EoMT-style injected-query head (`eomt_head`) on `model`.

    This matches the design in `third_party/eomt`:
      - run the backbone for the first (depth - k) blocks without queries
      - prepend Q learnable query tokens
      - run the remaining k blocks jointly
      - pool final query tokens into z and regress scalars from z

    Notes:
      - `eomt_num_layers` is interpreted as `num_blocks` (k, last-k backbone blocks).
      - `eomt_num_heads` / `eomt_ffn_dim` are unused for injected-query mode (kept for config/ckpt compatibility).
    """
    # Ensure mutually-exclusive head modules exist.
    model.fpn_head = None  # type: ignore[assignment]
    model.dpt_head = None  # type: ignore[assignment]
    model.vitdet_head = None  # type: ignore[assignment]

    # Placeholders for legacy attributes referenced elsewhere (export, etc.)
    model.shared_bottleneck = None  # type: ignore[assignment]
    model.reg3_heads = None  # type: ignore[assignment]
    model.layer_bottlenecks = None  # type: ignore[assignment]

    use_layerwise_heads = bool(getattr(model, "use_layerwise_heads", False))
    backbone_layer_indices = list(getattr(model, "backbone_layer_indices", []))
    num_layers_eff = int(max(1, len(backbone_layer_indices))) if use_layerwise_heads else 1

    enable_ratio = bool(getattr(model, "enable_ratio_head", False))
    num_outputs_main = int(getattr(model, "num_outputs", 1))
    num_ratio_outputs = int(getattr(model, "num_ratio_outputs", 3)) if enable_ratio else 0
    enable_ndvi = bool(getattr(model, "enable_ndvi", False)) and bool(
        getattr(model, "mtl_enabled", True)
    )

    # Store config on hparams-compatible attributes (used by export/inference meta).
    model.eomt_num_queries = int(eomt_num_queries)  # type: ignore[assignment]
    model.eomt_num_layers = int(eomt_num_layers)  # type: ignore[assignment]
    model.eomt_num_heads = int(eomt_num_heads)  # type: ignore[assignment]
    model.eomt_ffn_dim = int(eomt_ffn_dim)  # type: ignore[assignment]
    model.eomt_query_pool = str(eomt_query_pool)  # type: ignore[assignment]
    model.eomt_use_mean_query = bool(eomt_use_mean_query)  # type: ignore[assignment]
    model.eomt_use_mean_patch = bool(eomt_use_mean_patch)  # type: ignore[assignment]
    model.eomt_use_cls_token = bool(eomt_use_cls_token)  # type: ignore[assignment]
    model.eomt_proj_dim = int(eomt_proj_dim)  # type: ignore[assignment]
    model.eomt_proj_activation = str(eomt_proj_activation)  # type: ignore[assignment]
    model.eomt_proj_dropout = float(eomt_proj_dropout or 0.0)  # type: ignore[assignment]
    model.eomt_num_layers_eff = int(num_layers_eff)  # type: ignore[assignment]

    cfg = EoMTInjectedQueryHeadConfig(
        embedding_dim=int(embedding_dim),
        num_queries=int(eomt_num_queries),
        num_blocks=int(eomt_num_layers),
        dropout=float(dropout or 0.0),
        query_pool=str(eomt_query_pool or "mean"),
        use_mean_query=bool(eomt_use_mean_query),
        use_mean_patch=bool(eomt_use_mean_patch),
        use_cls_token=bool(eomt_use_cls_token),
        proj_dim=int(eomt_proj_dim),
        proj_activation=str(eomt_proj_activation or "relu"),
        proj_dropout=float(eomt_proj_dropout or 0.0),
        head_hidden_dims=list(hidden_dims),
        head_activation=str(head_activation or "relu"),
        num_outputs_main=int(num_outputs_main),
        num_outputs_ratio=int(num_ratio_outputs),
        enable_ndvi=bool(enable_ndvi),
    )
    model.eomt_head = EoMTInjectedQueryScalarHead(cfg)  # type: ignore[assignment]

    bottleneck_dim = int(getattr(model.eomt_head, "scalar_dim", int(embedding_dim)))  # type: ignore[attr-defined]
    return int(bottleneck_dim)


def init_eomt_task_heads(
    model,
    *,
    bottleneck_dim: int,
) -> None:
    """
    Initialize auxiliary heads for the EoMT query pooling head type.

    Notes:
      - NDVI is produced by the EoMT head itself; keep legacy `ndvi_head` as None.
      - Ratio logits are produced by the EoMT head itself; keep legacy `ratio_head` as None.
    """
    enable_height = bool(getattr(model, "enable_height", False))
    enable_species = bool(getattr(model, "enable_species", False))
    enable_state = bool(getattr(model, "enable_state", False))

    model.height_head = (
        nn.Linear(int(bottleneck_dim), 1) if enable_height else None
    )  # type: ignore[assignment]
    model.ndvi_head = None  # type: ignore[assignment]
    model.ratio_head = None  # type: ignore[assignment]

    if enable_species:
        num_species_classes = getattr(model, "num_species_classes", None)
        if num_species_classes is None or int(num_species_classes) <= 1:
            raise ValueError(
                "num_species_classes must be provided (>1) when species task is enabled"
            )
        model.num_species_classes = int(num_species_classes)  # type: ignore[assignment]
        model.species_head = nn.Linear(
            int(bottleneck_dim), int(model.num_species_classes)
        )  # type: ignore[assignment]
    else:
        model.num_species_classes = 0  # type: ignore[assignment]
        model.species_head = None  # type: ignore[assignment]

    if enable_state:
        num_state_classes = getattr(model, "num_state_classes", None)
        if num_state_classes is None or int(num_state_classes) <= 1:
            raise ValueError(
                "num_state_classes must be provided (>1) when state task is enabled"
            )
        model.num_state_classes = int(num_state_classes)  # type: ignore[assignment]
        model.state_head = nn.Linear(
            int(bottleneck_dim), int(model.num_state_classes)
        )  # type: ignore[assignment]
    else:
        model.num_state_classes = 0  # type: ignore[assignment]
        model.state_head = None  # type: ignore[assignment]

    # No layer-wise scalar heads for EoMT query pooling (multi-layer tokens fused inside the head)
    model.layer_reg3_heads = None  # type: ignore[assignment]
    model.layer_ratio_heads = None  # type: ignore[assignment]
    model.layer_height_heads = None  # type: ignore[assignment]
    model.layer_ndvi_heads = None  # type: ignore[assignment]
    model.layer_species_heads = None  # type: ignore[assignment]
    model.layer_state_heads = None  # type: ignore[assignment]


__all__ = ["init_eomt_head", "init_eomt_task_heads"]


