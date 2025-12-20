from __future__ import annotations

from typing import List

from torch import nn

from ...head_builder import SwiGLU


def _build_mlp_bottleneck(
    *,
    embedding_dim: int,
    use_patch_reg3: bool,
    use_cls_token: bool,
    hidden_dims: List[int],
    act_name: str,
    dropout: float,
) -> nn.Sequential:
    layers: List[nn.Module] = []
    use_patch = bool(use_patch_reg3)
    if use_patch:
        in_dim = int(embedding_dim)
    else:
        in_dim = int(embedding_dim) * 2 if bool(use_cls_token) else int(embedding_dim)
    if dropout and dropout > 0:
        layers.append(nn.Dropout(float(dropout)))

    if str(act_name).lower() == "swiglu":
        for hd in hidden_dims:
            layers.append(nn.Linear(in_dim, int(hd) * 2))
            layers.append(SwiGLU())
            if dropout and dropout > 0:
                layers.append(nn.Dropout(float(dropout)))
            in_dim = int(hd)
    else:
        def _act():
            name = str(act_name).lower()
            if name == "relu":
                return nn.ReLU(inplace=True)
            if name == "gelu":
                return nn.GELU()
            if name in ("silu", "swish"):
                return nn.SiLU(inplace=True)
            return nn.ReLU(inplace=True)

        for hd in hidden_dims:
            layers.append(nn.Linear(in_dim, int(hd)))
            layers.append(_act())
            if dropout and dropout > 0:
                layers.append(nn.Dropout(float(dropout)))
            in_dim = int(hd)

    return nn.Sequential(*layers)


def init_mlp_head(
    model,
    *,
    embedding_dim: int,
    hidden_dims: List[int],
    head_activation: str,
    dropout: float,
) -> int:
    """
    Initialize the classic MLP head on `model`:
      - shared bottleneck MLP (`shared_bottleneck`)
      - independent scalar heads (`reg3_heads`)
      - optional per-layer bottlenecks (`layer_bottlenecks`) when enabled

    Returns:
        bottleneck_dim used by downstream auxiliary heads.
    """
    # Ensure mutually-exclusive head modules exist.
    model.fpn_head = None  # type: ignore[assignment]
    model.dpt_head = None  # type: ignore[assignment]
    model.vitdet_head = None  # type: ignore[assignment]

    bottleneck = _build_mlp_bottleneck(
        embedding_dim=int(embedding_dim),
        use_patch_reg3=bool(getattr(model, "use_patch_reg3", False)),
        use_cls_token=bool(getattr(model, "use_cls_token", True)),
        hidden_dims=list(hidden_dims),
        act_name=str(head_activation or "relu").lower(),
        dropout=float(dropout or 0.0),
    )
    model.shared_bottleneck = bottleneck  # type: ignore[assignment]

    bottleneck_dim = int(hidden_dims[-1]) if len(hidden_dims) > 0 else int(embedding_dim)
    num_outputs = int(getattr(model, "num_outputs", 1))
    model.reg3_heads = nn.ModuleList([nn.Linear(bottleneck_dim, 1) for _ in range(num_outputs)])  # type: ignore[assignment]

    use_layerwise_heads = bool(getattr(model, "use_layerwise_heads", False))
    use_separate_bottlenecks = bool(getattr(model, "use_separate_bottlenecks", True))
    num_layers = int(getattr(model, "num_layers", 0))
    if use_layerwise_heads and use_separate_bottlenecks:
        model.layer_bottlenecks = nn.ModuleList(  # type: ignore[assignment]
            [
                _build_mlp_bottleneck(
                    embedding_dim=int(embedding_dim),
                    use_patch_reg3=bool(getattr(model, "use_patch_reg3", False)),
                    use_cls_token=bool(getattr(model, "use_cls_token", True)),
                    hidden_dims=list(hidden_dims),
                    act_name=str(head_activation or "relu").lower(),
                    dropout=float(dropout or 0.0),
                )
                for _ in range(num_layers)
            ]
        )
    else:
        model.layer_bottlenecks = None  # type: ignore[assignment]

    return bottleneck_dim


def init_mlp_task_heads(
    model,
    *,
    bottleneck_dim: int,
) -> None:
    """
    Initialize auxiliary heads for the MLP head type:
      - height_head / ndvi_head (optional)
      - species_head / state_head (optional)
      - ratio_head (optional)
      - layer-wise heads (optional, when use_layerwise_heads=True)
    """
    enable_height = bool(getattr(model, "enable_height", False))
    enable_ndvi = bool(getattr(model, "enable_ndvi", False))
    enable_species = bool(getattr(model, "enable_species", False))
    enable_state = bool(getattr(model, "enable_state", False))
    enable_ratio_head = bool(getattr(model, "enable_ratio_head", True))

    # Scalar auxiliary heads
    model.height_head = nn.Linear(int(bottleneck_dim), 1) if enable_height else None  # type: ignore[assignment]
    model.ndvi_head = nn.Linear(int(bottleneck_dim), 1) if enable_ndvi else None  # type: ignore[assignment]

    # Species / state classification heads
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

    # Ratio head (MLP-only; FPN/DPT produce ratio logits internally)
    if enable_ratio_head:
        num_ratio_outputs = int(getattr(model, "num_ratio_outputs", 3))
        model.ratio_head = nn.Linear(int(bottleneck_dim), int(num_ratio_outputs))  # type: ignore[assignment]
    else:
        model.ratio_head = None  # type: ignore[assignment]

    # Layer-wise heads per backbone layer (optional)
    use_layerwise_heads = bool(getattr(model, "use_layerwise_heads", False))
    if use_layerwise_heads:
        L = int(getattr(model, "num_layers", 0))
        num_outputs = int(getattr(model, "num_outputs", 1))
        model.layer_reg3_heads = nn.ModuleList(  # type: ignore[assignment]
            nn.ModuleList([nn.Linear(int(bottleneck_dim), 1) for _ in range(num_outputs)])
            for _ in range(L)
        )
        if enable_ratio_head:
            num_ratio_outputs = int(getattr(model, "num_ratio_outputs", 3))
            model.layer_ratio_heads = nn.ModuleList(  # type: ignore[assignment]
                nn.Linear(int(bottleneck_dim), int(num_ratio_outputs)) for _ in range(L)
            )
        else:
            model.layer_ratio_heads = None  # type: ignore[assignment]
        model.layer_height_heads = (  # type: ignore[assignment]
            nn.ModuleList(nn.Linear(int(bottleneck_dim), 1) for _ in range(L)) if enable_height else None
        )
        model.layer_ndvi_heads = (  # type: ignore[assignment]
            nn.ModuleList(nn.Linear(int(bottleneck_dim), 1) for _ in range(L)) if enable_ndvi else None
        )
        model.layer_species_heads = (  # type: ignore[assignment]
            nn.ModuleList(nn.Linear(int(bottleneck_dim), int(getattr(model, "num_species_classes", 0))) for _ in range(L))
            if enable_species and int(getattr(model, "num_species_classes", 0)) > 0
            else None
        )
        model.layer_state_heads = (  # type: ignore[assignment]
            nn.ModuleList(nn.Linear(int(bottleneck_dim), int(getattr(model, "num_state_classes", 0))) for _ in range(L))
            if enable_state and int(getattr(model, "num_state_classes", 0)) > 0
            else None
        )
    else:
        model.layer_reg3_heads = None  # type: ignore[assignment]
        model.layer_ratio_heads = None  # type: ignore[assignment]
        model.layer_height_heads = None  # type: ignore[assignment]
        model.layer_ndvi_heads = None  # type: ignore[assignment]
        model.layer_species_heads = None  # type: ignore[assignment]
        model.layer_state_heads = None  # type: ignore[assignment]


