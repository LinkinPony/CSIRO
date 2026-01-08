from __future__ import annotations

from typing import List

from torch import nn

from .dpt_scalar_head import DPTHeadConfig, DPTScalarHead


def init_dpt_head(
    model,
    *,
    embedding_dim: int,
    dpt_features: int,
    dpt_patch_size: int,
    dpt_readout: str,
    hidden_dims: List[int],
    head_activation: str,
    dropout: float,
) -> int:
    """
    Initialize the DPT-style dense prediction head on `model` (`dpt_head`).
    Returns bottleneck_dim (scalar feature dim produced by the head).
    """
    # Ensure mutually-exclusive head modules exist.
    model.fpn_head = None  # type: ignore[assignment]
    model.vitdet_head = None  # type: ignore[assignment]

    # Placeholders for legacy attributes referenced elsewhere (export, etc.)
    model.shared_bottleneck = None  # type: ignore[assignment]
    model.reg3_heads = None  # type: ignore[assignment]
    model.layer_bottlenecks = None  # type: ignore[assignment]

    use_layerwise_heads = bool(getattr(model, "use_layerwise_heads", False))
    backbone_layer_indices = list(getattr(model, "backbone_layer_indices", []))
    num_layers_eff = int(max(1, len(backbone_layer_indices))) if use_layerwise_heads else 1

    enable_ndvi = bool(getattr(model, "enable_ndvi", False)) and bool(getattr(model, "mtl_enabled", True))
    enable_ratio_head = bool(getattr(model, "enable_ratio_head", True))
    num_outputs_main = int(getattr(model, "num_outputs", 1))
    num_ratio_outputs = int(getattr(model, "num_ratio_outputs", 3)) if bool(enable_ratio_head) else 0

    dpt_cfg = DPTHeadConfig(
        embedding_dim=int(embedding_dim),
        features=int(dpt_features),
        patch_size=int(dpt_patch_size),
        readout=str(dpt_readout),
        num_layers=int(max(1, num_layers_eff)),
        num_outputs_main=int(num_outputs_main),
        num_outputs_ratio=int(num_ratio_outputs),
        enable_ndvi=bool(enable_ndvi),
        separate_ratio_head=bool(getattr(model, "separate_ratio_head", False)),
        separate_ratio_spatial_head=bool(getattr(model, "separate_ratio_spatial_head", False)),
        head_hidden_dims=list(hidden_dims),
        head_activation=str(head_activation),
        dropout=float(dropout or 0.0),
    )
    model.dpt_head = DPTScalarHead(dpt_cfg)  # type: ignore[assignment]

    bottleneck_dim = int(getattr(model.dpt_head, "scalar_dim", int(embedding_dim)))  # type: ignore[attr-defined]
    return bottleneck_dim


def init_dpt_task_heads(
    model,
    *,
    bottleneck_dim: int,
) -> None:
    """
    Initialize auxiliary heads for the DPT head type.

    Notes:
      - NDVI is produced by the DPT head itself; keep legacy `ndvi_head` as None.
      - Ratio logits are produced by the DPT head itself; keep legacy `ratio_head` as None.
    """
    enable_height = bool(getattr(model, "enable_height", False))
    enable_species = bool(getattr(model, "enable_species", False))
    enable_state = bool(getattr(model, "enable_state", False))

    model.height_head = nn.Linear(int(bottleneck_dim), 1) if enable_height else None  # type: ignore[assignment]
    model.ndvi_head = None  # type: ignore[assignment]
    model.ratio_head = None  # type: ignore[assignment]

    if enable_species:
        num_species_classes = getattr(model, "num_species_classes", None)
        if num_species_classes is None or int(num_species_classes) <= 1:
            raise ValueError("num_species_classes must be provided (>1) when species task is enabled")
        model.num_species_classes = int(num_species_classes)
        model.species_head = nn.Linear(int(bottleneck_dim), int(model.num_species_classes))  # type: ignore[assignment]
    else:
        model.num_species_classes = 0  # type: ignore[assignment]
        model.species_head = None  # type: ignore[assignment]

    if enable_state:
        num_state_classes = getattr(model, "num_state_classes", None)
        if num_state_classes is None or int(num_state_classes) <= 1:
            raise ValueError("num_state_classes must be provided (>1) when state task is enabled")
        model.num_state_classes = int(num_state_classes)
        model.state_head = nn.Linear(int(bottleneck_dim), int(model.num_state_classes))  # type: ignore[assignment]
    else:
        model.num_state_classes = 0  # type: ignore[assignment]
        model.state_head = None  # type: ignore[assignment]

    # No layer-wise scalar heads for DPT (multi-layer tokens fused inside the head)
    model.layer_reg3_heads = None  # type: ignore[assignment]
    model.layer_ratio_heads = None  # type: ignore[assignment]
    model.layer_height_heads = None  # type: ignore[assignment]
    model.layer_ndvi_heads = None  # type: ignore[assignment]
    model.layer_species_heads = None  # type: ignore[assignment]
    model.layer_state_heads = None  # type: ignore[assignment]


