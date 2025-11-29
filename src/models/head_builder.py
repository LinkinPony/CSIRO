from typing import List, Optional

from torch import nn
import torch.nn.functional as F


class SwiGLU(nn.Module):
    """
    Parameter-free SwiGLU gate operating on the last dimension.

    Expects input with size (..., 2 * d) and returns (..., d):
        x, gate = x.chunk(2, dim=-1)
        return silu(gate) * x
    """

    def forward(self, x):
        x_main, x_gate = x.chunk(2, dim=-1)
        return F.silu(x_gate) * x_main


def _build_activation(name: str) -> nn.Module:
    name = (name or "").lower()
    if name == "relu":
        return nn.ReLU(inplace=True)
    if name == "gelu":
        return nn.GELU()
    if name == "silu" or name == "swish":
        return nn.SiLU(inplace=True)
    # Fallback for unknown names: ReLU
    return nn.ReLU(inplace=True)


def build_head_layer(
    embedding_dim: int,
    num_outputs: int,
    head_hidden_dims: Optional[List[int]] = None,
    head_activation: str = "relu",
    dropout: float = 0.0,
    use_output_softplus: bool = True,
    input_dim: Optional[int] = None,
) -> nn.Sequential:
    hidden_dims: List[int] = list(head_hidden_dims or [512, 256])
    layers: List[nn.Module] = []
    # Default assumption for legacy heads: feature extractor provides
    # CLS concat mean(patch) → 2 * embedding_dim. When input_dim is
    # explicitly provided (e.g., patch-only heads), it overrides this.
    if input_dim is not None:
        in_dim = int(input_dim)
    else:
        in_dim = embedding_dim * 2
    act_name = (head_activation or "").lower()

    if dropout and dropout > 0:
        layers.append(nn.Dropout(dropout))

    if act_name == "swiglu":
        # SwiGLU MLP: for each hidden_dim we use a Linear(in_dim, 2 * hidden_dim)
        # followed by a SwiGLU gate which halves the dimension back to hidden_dim.
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim * 2))
            layers.append(SwiGLU())
            if dropout and dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim
    else:
        # Legacy MLP: Linear + pointwise activation
        act = _build_activation(act_name)
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


