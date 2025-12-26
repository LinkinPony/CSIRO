from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .spatial_fpn import _infer_patch_grid_hw


def _build_activation(name: str) -> nn.Module:
    n = (name or "").lower().strip()
    if n == "relu":
        return nn.ReLU(inplace=True)
    if n == "gelu":
        return nn.GELU()
    if n in ("silu", "swish"):
        return nn.SiLU(inplace=True)
    return nn.ReLU(inplace=True)


class LayerNorm2d(nn.Module):
    """
    Channel-wise LayerNorm for NCHW tensors (B, C, H, W).

    This matches Detectron2's ViTDet LayerNorm implementation: point-wise mean/variance
    normalization over the channel dimension.
    """

    def __init__(self, normalized_shape: int, eps: float = 1e-6) -> None:
        super().__init__()
        c = int(normalized_shape)
        if c <= 0:
            raise ValueError("LayerNorm2d normalized_shape must be a positive integer.")
        self.weight = nn.Parameter(torch.ones(c))
        self.bias = nn.Parameter(torch.zeros(c))
        self.eps = float(eps)

    def forward(self, x: Tensor) -> Tensor:
        if x.dim() != 4:
            raise RuntimeError(f"LayerNorm2d expects (B,C,H,W), got: {tuple(x.shape)}")
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


def _get_norm(norm: Optional[str], channels: int) -> Optional[nn.Module]:
    """
    Minimal `get_norm` helper (Detectron2-compatible semantics for LN).

    Supported values:
    - None / "" : no norm
    - "LN"      : LayerNorm2d (channel-wise LN over NCHW)
    """

    n = (norm or "").strip()
    if n == "":
        return None
    if n.upper() == "LN":
        return LayerNorm2d(int(channels))
    raise ValueError(f"Unsupported vitdet norm={norm!r}. Supported: '' | 'LN'")


@dataclass
class ViTDetHeadConfig:
    embedding_dim: int
    vitdet_dim: int
    scale_factors: Sequence[float]
    patch_size: int
    num_outputs_main: int
    num_outputs_ratio: int
    enable_ndvi: bool
    # When True, predict ratio logits from an independent MLP branch fed by the
    # pooled pyramid features (no shared scalar MLP trunk with reg3).
    separate_ratio_head: bool = False
    # When True, predict ratio logits from a completely separate spatial head:
    # duplicate the conv/pyramid stack (and its pooled-feature MLP), so ratio does not
    # share any head parameters with reg3.
    separate_ratio_spatial_head: bool = False
    # When True (default), require (H//patch_size)*(W//patch_size) == N exactly.
    strict_patch_grid: bool = True
    # Scalar head (GAP -> MLP -> Linear)
    head_hidden_dims: Sequence[int] = ()
    head_activation: str = "relu"
    dropout: float = 0.0
    # SimpleFeaturePyramid norm (Detectron2 ViTDet default is LN).
    norm: str = "LN"


class ViTDetScalarHead(nn.Module):
    """
    ViTDet-inspired "SimpleFeaturePyramid" style scalar head:

      - consumes ViT/DINO patch tokens (B, N, C) from a *single* backbone layer
      - reshapes to (B, C, Hp, Wp)
      - builds a Detectron2-aligned SimpleFeaturePyramid (SFP) over the token map:
          * 4.0: ConvTranspose2d x2 with channel reduction (dim->dim//2->dim//4),
                 with LN+GELU between the two upsampling ops (Detectron2 ViTDet)
          * 2.0: ConvTranspose2d x1 with channel reduction (dim->dim//2)
          * 1.0: Identity
          * 0.5: MaxPool2d stride=2
          * 0.25/0.125: repeated MaxPool2d (repo extension beyond Detectron2)
        Each scale then applies: 1x1 Conv -> LN -> 3x3 Conv -> LN to produce `vitdet_dim` channels.
      - global-average-pools each scale, concatenates, passes through an MLP, and predicts:
          * reg3 logits: (B, num_outputs_main)
          * ratio logits: (B, num_outputs_ratio) or None
          * ndvi pred: (B, 1) or None
    """

    def __init__(self, cfg: ViTDetHeadConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.embedding_dim = int(cfg.embedding_dim)
        self.vitdet_dim = int(cfg.vitdet_dim)
        self.patch_size = int(max(1, cfg.patch_size))
        self.scale_factors = tuple(float(x) for x in (cfg.scale_factors or (1.0,)))
        self.strict_patch_grid = bool(getattr(cfg, "strict_patch_grid", True))

        self.num_outputs_main = int(max(1, cfg.num_outputs_main))
        self.num_outputs_ratio = int(max(0, cfg.num_outputs_ratio))
        self.enable_ndvi = bool(cfg.enable_ndvi)
        self.separate_ratio_spatial_head = bool(getattr(cfg, "separate_ratio_spatial_head", False))
        self.separate_ratio_head = bool(getattr(cfg, "separate_ratio_head", False)) or self.separate_ratio_spatial_head

        # Detectron2-aligned SFP stages.
        # Important difference vs the repo's previous implementation:
        #   - We do NOT pre-project to `vitdet_dim` before pyramid generation.
        #     Instead, we build the pyramid in `embedding_dim` space and then project each
        #     scale to `vitdet_dim` (= Detectron2's `out_channels`).
        self.norm: str = str(getattr(cfg, "norm", "LN") or "LN")
        use_bias = (self.norm or "").strip() == ""  # Detectron2: bias only when no norm
        dim = int(self.embedding_dim)
        out_channels = int(self.vitdet_dim)

        def _build_stage(scale: float) -> nn.Sequential:
            layers: List[nn.Module] = []
            out_dim = dim
            s_key = float(scale)

            if s_key == 4.0:
                # Detectron2 ViTDet: two upsampling steps with channel reduction.
                half = max(1, dim // 2)
                quarter = max(1, dim // 4)
                layers.append(nn.ConvTranspose2d(dim, half, kernel_size=2, stride=2))
                n1 = _get_norm(self.norm, half)
                if n1 is not None:
                    layers.append(n1)
                layers.append(nn.GELU())
                layers.append(nn.ConvTranspose2d(half, quarter, kernel_size=2, stride=2))
                out_dim = quarter
            elif s_key == 2.0:
                half = max(1, dim // 2)
                layers.append(nn.ConvTranspose2d(dim, half, kernel_size=2, stride=2))
                out_dim = half
            elif s_key == 1.0:
                out_dim = dim
            elif s_key == 0.5:
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
                out_dim = dim
            elif s_key == 0.25:
                # Repo extension (not in Detectron2): downsample x4.
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
                out_dim = dim
            elif s_key == 0.125:
                # Repo extension (not in Detectron2): downsample x8.
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
                out_dim = dim
            else:
                raise ValueError(
                    f"Unsupported vitdet scale_factor={s_key}. Use one of: 4.0, 2.0, 1.0, 0.5, 0.25, 0.125"
                )

            # 1x1 + 3x3 conv blocks with LN (Detectron2 ViTDet).
            layers.append(nn.Conv2d(out_dim, out_channels, kernel_size=1, bias=use_bias))
            n2 = _get_norm(self.norm, out_channels)
            if n2 is not None:
                layers.append(n2)
            layers.append(
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=use_bias)
            )
            n3 = _get_norm(self.norm, out_channels)
            if n3 is not None:
                layers.append(n3)
            return nn.Sequential(*layers)

        self.stages = nn.ModuleList([_build_stage(float(s)) for s in self.scale_factors])

        # Optional separate spatial branch for ratio: duplicate SFP stages (independent params).
        if self.separate_ratio_spatial_head and self.num_outputs_ratio > 0:
            self.ratio_stages = nn.ModuleList([_build_stage(float(s)) for s in self.scale_factors])
        else:
            self.ratio_stages = None  # type: ignore[assignment]
        # Legacy placeholder (kept for backward compatibility with older checkpoints).
        self.ratio_in_proj = None  # type: ignore[assignment]

        # Scalar bottleneck MLP(s) on concatenated pooled pyramid features.
        in_dim = self.vitdet_dim * len(self.scale_factors)
        hidden_dims = list(getattr(cfg, "head_hidden_dims", None) or [])
        drop = float(getattr(cfg, "dropout", 0.0) or 0.0)
        act_name = str(getattr(cfg, "head_activation", "relu"))

        def _build_scalar_mlp() -> tuple[nn.Module, int]:
            layers: List[nn.Module] = []
            cur = in_dim
            if drop > 0:
                layers.append(nn.Dropout(drop))
            for hd in hidden_dims:
                hd_i = int(hd)
                layers.append(nn.Linear(cur, hd_i))
                layers.append(_build_activation(act_name))
                if drop > 0:
                    layers.append(nn.Dropout(drop))
                cur = hd_i
            mlp = nn.Sequential(*layers) if layers else nn.Identity()
            dim = cur if layers else in_dim
            return mlp, dim

        self.scalar_mlp, self.scalar_dim = _build_scalar_mlp()

        # Optional independent ratio MLP branch.
        # - separate_ratio_head: ratio gets its own MLP, but shares pooled features `g`
        # - separate_ratio_spatial_head: ratio gets its own conv/pyramid -> pooled features `g_ratio`,
        #   and its own MLP
        if self.separate_ratio_head and self.num_outputs_ratio > 0:
            self.ratio_mlp, ratio_dim = _build_scalar_mlp()
        else:
            self.ratio_mlp = None
            ratio_dim = int(self.scalar_dim)

        self.reg3_head = nn.Linear(int(self.scalar_dim), self.num_outputs_main)
        self.ratio_head = nn.Linear(int(ratio_dim), self.num_outputs_ratio) if self.num_outputs_ratio > 0 else None
        self.ndvi_head = nn.Linear(self.scalar_dim, 1) if self.enable_ndvi else None

    def _tokens_to_map(self, pt: Tensor, *, image_hw: Tuple[int, int]) -> Tensor:
        if pt.dim() != 3:
            raise RuntimeError(f"Expected patch tokens (B,N,C), got: {tuple(pt.shape)}")
        B, N, C = pt.shape
        if C != self.embedding_dim:
            raise RuntimeError(
                f"Patch token dim mismatch: got C={C}, expected embedding_dim={self.embedding_dim}"
            )
        Hp, Wp = _infer_patch_grid_hw(
            num_patches=int(N),
            image_hw=image_hw,
            patch_size=self.patch_size,
            strict_primary=self.strict_patch_grid,
        )
        if Hp * Wp != int(N):
            raise RuntimeError(
                f"Cannot reshape tokens to grid: N={N}, inferred Hp={Hp}, Wp={Wp}, Hp*Wp={Hp*Wp}"
            )
        x = pt.transpose(1, 2).contiguous().view(B, C, Hp, Wp)
        return x

    def forward(
        self,
        pt_tokens: Tensor,
        *,
        image_hw: Tuple[int, int],
    ) -> Dict[str, Optional[Tensor]]:
        x = self._tokens_to_map(pt_tokens, image_hw=image_hw)

        # Best-effort dtype alignment if autocast is off (e.g., frozen backbone in fp16).
        try:
            w_dtype = next(self.parameters()).dtype
        except Exception:
            w_dtype = x.dtype
        if (not torch.is_autocast_enabled()) and x.dtype != w_dtype:
            x = x.to(dtype=w_dtype)

        # Main branch
        feats = [stage(x) for stage in self.stages]
        pooled = [F.adaptive_avg_pool2d(f, output_size=1).flatten(1) for f in feats]
        g = torch.cat(pooled, dim=-1) if pooled else torch.empty((x.size(0), 0), device=x.device, dtype=x.dtype)
        z = self.scalar_mlp(g) if not isinstance(self.scalar_mlp, nn.Identity) else g

        # Ratio branch (mode-dependent)
        if self.separate_ratio_spatial_head and self.ratio_head is not None:
            if self.ratio_stages is None:
                raise RuntimeError("Missing ratio spatial modules for separate_ratio_spatial_head")
            feats_r = [stage(x) for stage in self.ratio_stages]
            pooled_r = [F.adaptive_avg_pool2d(f, output_size=1).flatten(1) for f in feats_r]
            g_ratio = (
                torch.cat(pooled_r, dim=-1)
                if pooled_r
                else torch.empty((x.size(0), 0), device=x.device, dtype=x.dtype)
            )
            if self.ratio_mlp is not None:
                z_ratio = self.ratio_mlp(g_ratio) if not isinstance(self.ratio_mlp, nn.Identity) else g_ratio
            else:
                z_ratio = g_ratio
        elif self.ratio_mlp is not None:
            z_ratio = self.ratio_mlp(g) if not isinstance(self.ratio_mlp, nn.Identity) else g
        else:
            z_ratio = z

        reg3 = self.reg3_head(z)
        ratio = self.ratio_head(z_ratio) if self.ratio_head is not None else None
        ndvi = self.ndvi_head(z) if self.ndvi_head is not None else None
        return {"z": z, "reg3": reg3, "ratio": ratio, "ndvi": ndvi}


class ViTDetMultiLayerScalarHead(nn.Module):
    """
    Multi-layer wrapper: one independent ViTDetScalarHead per backbone layer,
    and fuse the final outputs across layers (mean or learned softmax weights).
    """

    def __init__(
        self,
        cfg: ViTDetHeadConfig,
        *,
        num_layers: int,
        layer_fusion: str = "mean",
    ) -> None:
        super().__init__()
        if int(num_layers) <= 0:
            raise ValueError("num_layers must be >= 1 for ViTDetMultiLayerScalarHead")
        self.cfg = cfg
        self.num_layers = int(num_layers)
        self.layer_fusion: str = str(layer_fusion or "mean").strip().lower()
        self.heads = nn.ModuleList([ViTDetScalarHead(cfg) for _ in range(self.num_layers)])
        # Expose scalar_dim for downstream aux heads
        self.scalar_dim = int(getattr(self.heads[0], "scalar_dim", cfg.vitdet_dim))
        # Optional learnable fusion weights. Only create when requested so legacy mean-fusion
        # heads still load with strict=True.
        if self.layer_fusion == "learned":
            self.layer_logits = nn.Parameter(torch.zeros(self.num_layers, dtype=torch.float32))
        else:
            self.layer_logits = None  # type: ignore[assignment]

    @staticmethod
    def _avg_optional(xs: List[Optional[Tensor]]) -> Optional[Tensor]:
        ts = [t for t in xs if t is not None]
        if not ts:
            return None
        return torch.stack(ts, dim=0).mean(dim=0)

    def get_layer_weights(
        self,
        *,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> Tensor:
        """
        Return fusion weights over layers as a tensor of shape (L,).
          - mean: uniform weights
          - learned: softmax(layer_logits)
        """
        L = int(self.num_layers)
        if L <= 0:
            w = torch.ones(1, dtype=torch.float32)
        elif self.layer_fusion == "learned" and isinstance(self.layer_logits, torch.Tensor):
            w = F.softmax(self.layer_logits, dim=0)
        else:
            w = torch.full((L,), 1.0 / float(L), dtype=torch.float32)
        if device is not None:
            w = w.to(device=device)
        if dtype is not None:
            w = w.to(dtype=dtype)
        return w

    def _fuse_optional(self, xs: List[Optional[Tensor]], *, weights: Tensor) -> Optional[Tensor]:
        """
        Fuse optional per-layer tensors with provided weights.
        If some entries are None, renormalize weights over present layers.
        """
        ts: List[Tensor] = []
        idxs: List[int] = []
        for i, t in enumerate(xs):
            if t is None:
                continue
            ts.append(t)
            idxs.append(i)
        if not ts:
            return None
        stacked = torch.stack(ts, dim=0)  # (K, ...)
        if stacked.shape[0] == 1:
            return stacked[0]
        try:
            idx_t = torch.tensor(idxs, device=weights.device, dtype=torch.long)
            w = weights.index_select(0, idx_t)
        except Exception:
            w = weights[: stacked.shape[0]]
        w = w.to(device=stacked.device, dtype=stacked.dtype)
        w = w / w.sum().clamp_min(1e-8)
        while w.dim() < stacked.dim():
            w = w.view(*w.shape, 1)
        return (w * stacked).sum(dim=0)

    def forward(
        self,
        pt_tokens: Union[Tensor, Sequence[Tensor]],
        *,
        image_hw: Tuple[int, int],
    ) -> Dict[str, Optional[Tensor]]:
        if isinstance(pt_tokens, Tensor):
            pt_list = [pt_tokens]
        else:
            pt_list = list(pt_tokens)
        if len(pt_list) != int(self.num_layers):
            raise RuntimeError(
                f"ViTDetMultiLayerScalarHead expected {int(self.num_layers)} layers but got {len(pt_list)}"
            )
        outs = [h(pt_list[i], image_hw=image_hw) for i, h in enumerate(self.heads)]
        z_list = [o.get("z", None) for o in outs]
        reg_list = [o.get("reg3", None) for o in outs]
        ratio_list = [o.get("ratio", None) for o in outs]
        ndvi_list = [o.get("ndvi", None) for o in outs]
        # Choose weights for fusion (global, per-layer).
        ref = next((t for t in reg_list if isinstance(t, torch.Tensor)), None)
        w = self.get_layer_weights(
            device=(ref.device if isinstance(ref, torch.Tensor) else None),
            dtype=(ref.dtype if isinstance(ref, torch.Tensor) else None),
        )

        if self.layer_fusion == "learned":
            z = self._fuse_optional(z_list, weights=w)
            reg3 = self._fuse_optional(reg_list, weights=w)
            ratio = self._fuse_optional(ratio_list, weights=w)
            ndvi = self._fuse_optional(ndvi_list, weights=w)
        else:
            z = self._avg_optional(z_list)
            reg3 = self._avg_optional(reg_list)
            ratio = self._avg_optional(ratio_list)
            ndvi = self._avg_optional(ndvi_list)

        # Expose both fused outputs (legacy keys) and per-layer outputs for downstream
        # constraint-aware component fusion.
        return {
            "z": z,
            "reg3": reg3,
            "ratio": ratio,
            "ndvi": ndvi,
            "z_layers": z_list,
            "reg3_layers": reg_list,
            "ratio_layers": ratio_list,
            "ndvi_layers": ndvi_list,
            "layer_weights": w,
        }


def init_vitdet_head(
    model,
    *,
    embedding_dim: int,
    vitdet_dim: int,
    vitdet_patch_size: int,
    vitdet_scale_factors: Sequence[float],
    hidden_dims: List[int],
    head_activation: str,
    dropout: float,
) -> int:
    """
    Initialize the ViTDet-style head on `model` (`vitdet_head`).
    Returns bottleneck_dim (scalar feature dim produced by the head).
    """
    # Ensure mutually-exclusive head modules exist.
    model.fpn_head = None  # type: ignore[assignment]
    model.dpt_head = None  # type: ignore[assignment]

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

    cfg = ViTDetHeadConfig(
        embedding_dim=int(embedding_dim),
        vitdet_dim=int(vitdet_dim),
        scale_factors=tuple(float(x) for x in (vitdet_scale_factors or (2.0, 1.0, 0.5))),
        patch_size=int(vitdet_patch_size),
        num_outputs_main=int(num_outputs_main),
        num_outputs_ratio=3 if bool(enable_ratio_head) else 0,
        enable_ndvi=bool(enable_ndvi),
        separate_ratio_head=bool(getattr(model, "separate_ratio_head", False)),
        separate_ratio_spatial_head=bool(getattr(model, "separate_ratio_spatial_head", False)),
        head_hidden_dims=list(hidden_dims),
        head_activation=str(head_activation),
        dropout=float(dropout or 0.0),
    )

    if use_layerwise_heads:
        layer_fusion = str(getattr(model, "backbone_layers_fusion", "mean") or "mean").strip().lower()
        model.vitdet_head = ViTDetMultiLayerScalarHead(cfg, num_layers=num_layers_eff, layer_fusion=layer_fusion)  # type: ignore[assignment]
        bottleneck_dim = int(getattr(model.vitdet_head, "scalar_dim", int(vitdet_dim)))  # type: ignore[attr-defined]
    else:
        model.vitdet_head = ViTDetScalarHead(cfg)  # type: ignore[assignment]
        bottleneck_dim = int(getattr(model.vitdet_head, "scalar_dim", int(vitdet_dim)))  # type: ignore[attr-defined]
    return bottleneck_dim


def init_vitdet_task_heads(
    model,
    *,
    bottleneck_dim: int,
) -> None:
    """
    Initialize auxiliary heads for the ViTDet head type.

    Notes:
      - NDVI is produced by the ViTDet head itself; keep legacy `ndvi_head` as None.
      - Ratio logits are produced by the ViTDet head itself; keep legacy `ratio_head` as None.
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

    # No layer-wise scalar heads for ViTDet (multi-layer averaging happens inside vitdet_head)
    model.layer_reg3_heads = None  # type: ignore[assignment]
    model.layer_ratio_heads = None  # type: ignore[assignment]
    model.layer_height_heads = None  # type: ignore[assignment]
    model.layer_ndvi_heads = None  # type: ignore[assignment]
    model.layer_species_heads = None  # type: ignore[assignment]
    model.layer_state_heads = None  # type: ignore[assignment]


__all__ = [
    "ViTDetHeadConfig",
    "ViTDetScalarHead",
    "ViTDetMultiLayerScalarHead",
    "init_vitdet_head",
    "init_vitdet_task_heads",
]



