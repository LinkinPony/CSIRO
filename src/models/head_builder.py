from typing import List, Optional

import torch
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
    """
    Build a simple MLP head used for legacy single-layer and packed multi-layer heads.

    This helper assumes a single shared bottleneck MLP whose output is mapped to
    `num_outputs` via a final Linear layer (optionally followed by Softplus).
    """
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


def build_bottleneck_mlp(
    embedding_dim: int,
    head_hidden_dims: Optional[List[int]] = None,
    head_activation: str = "relu",
    dropout: float = 0.0,
    use_patch_reg3: bool = False,
    use_cls_token: bool = True,
) -> nn.Sequential:
    """
    Build the shared/ per-layer bottleneck MLP used between DINO features and
    scalar heads. This mirrors the bottleneck construction in BiomassRegressor.

    Args:
        embedding_dim: Backbone embedding dimension (C).
        head_hidden_dims: Hidden sizes for the bottleneck MLP.
        head_activation: Activation name ('relu', 'gelu', 'silu', 'swiglu').
        dropout: Dropout probability applied after each hidden layer.
        use_patch_reg3: If True, the bottleneck expects patch-token dimensionality
                        (C) as input; otherwise it expects CLS+mean(patch) with
                        2 * C features.
        use_cls_token: If False (and use_patch_reg3 is False), the bottleneck expects
                       mean(patch) with C features instead of CLS+mean(patch) with 2C.
    """
    hidden_dims: List[int] = list(head_hidden_dims or [512, 256])
    act_name = (head_activation or "").lower()

    layers: List[nn.Module] = []
    # When use_patch_reg3 is enabled, the bottleneck operates directly on patch
    # token dimensionality (C). Otherwise, it expects:
    #   - CLS concat mean(patch) (2C) when use_cls_token is True
    #   - mean(patch) only (C) when use_cls_token is False
    in_dim = embedding_dim if (use_patch_reg3 or (not use_cls_token)) else (embedding_dim * 2)

    if dropout and dropout > 0:
        layers.append(nn.Dropout(dropout))

    if act_name == "swiglu":
        # SwiGLU bottleneck: for each hidden_dim we use Linear(in_dim, 2 * hidden_dim)
        # followed by a SwiGLU gate, which halves the dimension back to hidden_dim.
        for hd in hidden_dims:
            layers.append(nn.Linear(in_dim, hd * 2))
            layers.append(SwiGLU())
            if dropout and dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = hd
    else:
        # Legacy MLP: Linear + pointwise activation (ReLU/GELU/SiLU).
        act = _build_activation(act_name)
        for hd in hidden_dims:
            layers.append(nn.Linear(in_dim, hd))
            # Create a fresh activation instance per layer
            layers.append(act.__class__())
            if dropout and dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = hd

    return nn.Sequential(*layers)


class MultiLayerHeadExport(nn.Module):
    """
    Export-only multi-layer regression head with optional per-layer bottlenecks.

    This module is designed to be lightweight and to mirror the structure of
    the training-time layer-wise heads:
      - layer_bottlenecks[l]: bottleneck MLP for backbone layer l
      - layer_reg3_heads[l][k]: scalar reg3 head k for layer l
      - layer_ratio_heads[l]: ratio head for layer l (optional)

    It is used only for offline inference when `use_layerwise_heads` and
    `use_separate_bottlenecks` are both enabled.
    """

    def __init__(
        self,
        embedding_dim: int,
        num_outputs_main: int,
        num_outputs_ratio: int,
        head_hidden_dims: Optional[List[int]] = None,
        head_activation: str = "relu",
        dropout: float = 0.0,
        use_patch_reg3: bool = False,
        use_cls_token: bool = True,
        num_layers: int = 1,
    ) -> None:
        super().__init__()
        if num_layers <= 0:
            raise ValueError("num_layers must be >= 1 for MultiLayerHeadExport")

        self.embedding_dim = int(embedding_dim)
        self.num_outputs_main = int(num_outputs_main)
        self.num_outputs_ratio = int(num_outputs_ratio)
        self.use_patch_reg3 = bool(use_patch_reg3)
        self.use_cls_token = bool(use_cls_token)
        self.num_layers = int(num_layers)

        # Build one bottleneck MLP per selected backbone layer.
        self.layer_bottlenecks = nn.ModuleList(
            [
                build_bottleneck_mlp(
                    embedding_dim=self.embedding_dim,
                    head_hidden_dims=head_hidden_dims,
                    head_activation=head_activation,
                    dropout=dropout,
                    use_patch_reg3=self.use_patch_reg3,
                    use_cls_token=self.use_cls_token,
                )
                for _ in range(self.num_layers)
            ]
        )

        # All bottlenecks share the same output dimension by construction.
        # Infer bottleneck dim from the last Linear layer of the first bottleneck.
        sample_bottleneck = self.layer_bottlenecks[0]
        bottleneck_dim: Optional[int] = None
        for m in sample_bottleneck.modules():
            if isinstance(m, nn.Linear):
                bottleneck_dim = m.out_features
        if bottleneck_dim is None:
            bottleneck_dim = self.embedding_dim

        # Per-layer reg3 heads: each is a list of scalar Linear(bottleneck_dim, 1).
        self.layer_reg3_heads = nn.ModuleList(
            nn.ModuleList([nn.Linear(bottleneck_dim, 1) for _ in range(self.num_outputs_main)])
            for _ in range(self.num_layers)
        )

        # Optional per-layer ratio heads: Linear(bottleneck_dim, num_outputs_ratio)
        if self.num_outputs_ratio > 0:
            self.layer_ratio_heads = nn.ModuleList(
                nn.Linear(bottleneck_dim, self.num_outputs_ratio)
                for _ in range(self.num_layers)
            )
        else:
            self.layer_ratio_heads = None  # type: ignore[assignment]

    @staticmethod
    def _forward_reg3_logits_for_heads(
        z: torch.Tensor,
        heads: List[nn.Linear],
    ) -> torch.Tensor:
        preds = [h(z) for h in heads]
        return torch.cat(preds, dim=-1) if preds else torch.empty(
            (z.size(0), 0), dtype=z.dtype, device=z.device
        )

    def forward_global_layer(
        self,
        cls: torch.Tensor,
        patch_mean: torch.Tensor,
        layer_idx: int,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Global path for a single backbone layer.
        """
        if not (0 <= layer_idx < self.num_layers):
            raise IndexError(f"layer_idx {layer_idx} out of range for num_layers={self.num_layers}")

        # When use_patch_reg3 is False, the bottleneck expects either:
        #   - CLS+mean(patch) (2C) when use_cls_token is True
        #   - mean(patch)     (C)  when use_cls_token is False
        feats = torch.cat([cls, patch_mean], dim=-1) if self.use_cls_token else patch_mean
        bottleneck = self.layer_bottlenecks[layer_idx]
        z = bottleneck(feats)

        main = self._forward_reg3_logits_for_heads(
            z, list(self.layer_reg3_heads[layer_idx])
        )

        ratio: Optional[torch.Tensor]
        if self.num_outputs_ratio > 0 and self.layer_ratio_heads is not None:
            ratio = self.layer_ratio_heads[layer_idx](z)
        else:
            ratio = None
        return main, ratio

    def forward_patch_layer(
        self,
        pt: torch.Tensor,
        layer_idx: int,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Patch-mode path for a single backbone layer.

        Args:
            pt:       Patch tokens of shape (B, N, C).
            layer_idx: Index of the backbone block / bottleneck layer.

        Returns:
            main:  (B, num_outputs_main) averaged over patches
            ratio: (B, num_outputs_ratio) or None if ratio heads are disabled
        """
        if not (0 <= layer_idx < self.num_layers):
            raise IndexError(f"layer_idx {layer_idx} out of range for num_layers={self.num_layers}")
        if pt.dim() != 3:
            raise RuntimeError(f"Unexpected patch tokens shape in forward_patch_layer: {tuple(pt.shape)}")

        B, N, C = pt.shape
        bottleneck = self.layer_bottlenecks[layer_idx]

        # Main reg3: per-patch bottleneck + scalar heads, then average over patches.
        patch_features_flat = pt.reshape(B * N, C)
        z_patches_flat = bottleneck(patch_features_flat)
        main_flat = self._forward_reg3_logits_for_heads(
            z_patches_flat,
            list(self.layer_reg3_heads[layer_idx]),
        )
        if main_flat.numel() > 0:
            main = main_flat.view(B, N, self.num_outputs_main).mean(dim=1)
        else:
            main = torch.empty((B, 0), dtype=pt.dtype, device=pt.device)

        # Ratio logits (if any) from global mean-patch bottleneck.
        ratio: Optional[torch.Tensor]
        if self.num_outputs_ratio > 0 and self.layer_ratio_heads is not None:
            patch_mean = pt.mean(dim=1)  # (B, C)
            z_global = bottleneck(patch_mean)
            ratio = self.layer_ratio_heads[layer_idx](z_global)
        else:
            ratio = None

        return main, ratio


