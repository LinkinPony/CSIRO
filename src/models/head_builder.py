from typing import List, Optional

from torch import nn


def _build_activation(name: str) -> nn.Module:
    name = (name or "").lower()
    if name == "relu":
        return nn.ReLU(inplace=True)
    if name == "gelu":
        return nn.GELU()
    if name == "silu" or name == "swish":
        return nn.SiLU(inplace=True)
    return nn.ReLU(inplace=True)


def build_head_layer(
    embedding_dim: int,
    num_outputs: int,
    head_hidden_dims: Optional[List[int]] = None,
    head_activation: str = "relu",
    dropout: float = 0.0,
    use_output_softplus: bool = True,
) -> nn.Sequential:
    hidden_dims: List[int] = list(head_hidden_dims or [512, 256])
    layers: List[nn.Module] = []
    in_dim = embedding_dim
    act = _build_activation(head_activation)

    if dropout and dropout > 0:
        layers.append(nn.Dropout(dropout))

    for hidden_dim in hidden_dims:
        layers.append(nn.Linear(in_dim, hidden_dim))
        layers.append(act.__class__())
        if dropout and dropout > 0:
            layers.append(nn.Dropout(dropout))
        in_dim = hidden_dim

    layers.append(nn.Linear(in_dim, num_outputs))
    if use_output_softplus:
        layers.append(nn.Softplus())
    return nn.Sequential(*layers)


