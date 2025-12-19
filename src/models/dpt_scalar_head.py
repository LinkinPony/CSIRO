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


class ProjectReadout(nn.Module):
    """
    DPT-style project readout:
      - repeat CLS token over all patches
      - concat [patch ; cls] along channel dim -> Linear(2C -> C) -> GELU
    """

    def __init__(self, embedding_dim: int) -> None:
        super().__init__()
        c = int(embedding_dim)
        self.proj = nn.Sequential(nn.Linear(2 * c, c), nn.GELU())

    def forward(self, pt: Tensor, cls: Tensor) -> Tensor:
        # pt:  (B, N, C)
        # cls: (B, C)
        if pt.dim() != 3:
            raise RuntimeError(f"Expected pt tokens (B,N,C), got {tuple(pt.shape)}")
        if cls.dim() != 2:
            raise RuntimeError(f"Expected cls tokens (B,C), got {tuple(cls.shape)}")
        B, N, C = pt.shape
        if cls.shape[0] != B or cls.shape[1] != C:
            raise RuntimeError(
                f"CLS shape mismatch: pt={tuple(pt.shape)} cls={tuple(cls.shape)}"
            )
        cls_rep = cls.unsqueeze(1).expand(B, N, C)
        x = torch.cat([pt, cls_rep], dim=-1)
        return self.proj(x)


class ResidualConvUnitCustom(nn.Module):
    def __init__(self, features: int, activation: nn.Module, bn: bool = False) -> None:
        super().__init__()
        f = int(features)
        self.bn = bool(bn)
        self.activation = activation
        self.conv1 = nn.Conv2d(f, f, kernel_size=3, stride=1, padding=1, bias=not self.bn)
        self.conv2 = nn.Conv2d(f, f, kernel_size=3, stride=1, padding=1, bias=not self.bn)
        if self.bn:
            self.bn1 = nn.BatchNorm2d(f)
            self.bn2 = nn.BatchNorm2d(f)

    def forward(self, x: Tensor) -> Tensor:
        out = self.activation(x)
        out = self.conv1(out)
        if self.bn:
            out = self.bn1(out)
        out = self.activation(out)
        out = self.conv2(out)
        if self.bn:
            out = self.bn2(out)
        return out + x


class FeatureFusionBlockCustom(nn.Module):
    """
    DPT refinenet-style fusion block:
      - optional skip input is processed by a residual conv unit and added
      - another residual conv unit
      - upsample x2 (bilinear)
      - 1x1 conv
    """

    def __init__(
        self,
        features: int,
        *,
        activation: Optional[nn.Module] = None,
        bn: bool = False,
        align_corners: bool = True,
    ) -> None:
        super().__init__()
        f = int(features)
        act = activation if activation is not None else nn.ReLU(inplace=False)
        self.align_corners = bool(align_corners)
        self.res1 = ResidualConvUnitCustom(f, act, bn=bn)
        self.res2 = ResidualConvUnitCustom(f, act, bn=bn)
        self.out_conv = nn.Conv2d(f, f, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x: Tensor, skip: Optional[Tensor] = None) -> Tensor:
        out = x
        if skip is not None:
            out = out + self.res1(skip)
        out = self.res2(out)
        out = F.interpolate(out, scale_factor=2.0, mode="bilinear", align_corners=self.align_corners)
        out = self.out_conv(out)
        return out


@dataclass
class DPTHeadConfig:
    embedding_dim: int
    features: int
    patch_size: int
    readout: str  # "ignore" | "project"
    num_layers: int
    num_outputs_main: int
    num_outputs_ratio: int
    enable_ndvi: bool
    # When True (default), require (H//patch_size)*(W//patch_size) == N exactly.
    strict_patch_grid: bool = True
    # Scalar head (GAP -> MLP -> Linear)
    head_hidden_dims: Sequence[int] = ()
    head_activation: str = "relu"
    dropout: float = 0.0


class DPTScalarHead(nn.Module):
    """
    DPT-style dense-prediction head adapted for scalar regression:
      - takes 4 (recommended) layers of ViT patch tokens (+ optional CLS tokens)
      - "reassembles" each layer to a different spatial scale using ConvTranspose/stride-2 Conv
      - fuses features via RefineNet-style top-down blocks
      - performs one final upsample (DPT-style)
      - produces a global scalar representation via GAP -> MLP -> Linear heads
    """

    def __init__(self, cfg: DPTHeadConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.embedding_dim = int(cfg.embedding_dim)
        self.features = int(cfg.features)
        self.patch_size = int(max(1, cfg.patch_size))
        self.num_layers = int(max(1, cfg.num_layers))
        self.num_outputs_main = int(max(1, cfg.num_outputs_main))
        self.num_outputs_ratio = int(max(0, cfg.num_outputs_ratio))
        self.enable_ndvi = bool(cfg.enable_ndvi)
        self.strict_patch_grid = bool(getattr(cfg, "strict_patch_grid", True))

        readout = str(cfg.readout or "ignore").strip().lower()
        if readout not in ("ignore", "project"):
            raise ValueError("DPTScalarHead readout must be one of: ignore, project")
        self.readout = readout

        # DPT "project readout" modules (one per selected layer).
        if self.readout == "project":
            self.readout_proj = nn.ModuleList([ProjectReadout(self.embedding_dim) for _ in range(self.num_layers)])
        else:
            self.readout_proj = None  # type: ignore[assignment]

        # Per-layer token->map projection (C -> features).
        self.layer_proj = nn.ModuleList(
            [nn.Conv2d(self.embedding_dim, self.features, kernel_size=1, bias=True) for _ in range(self.num_layers)]
        )

        # ConvTranspose reassembly (DPT-style):
        #  - layer_1: upsample x4 -> H/4
        #  - layer_2: upsample x2 -> H/8
        #  - layer_3: keep        -> H/16
        #  - layer_4: downsample x2 -> H/32
        # If num_layers != 4, we still build per-index rules in order (best-effort).
        self.reassemble = nn.ModuleList()
        for i in range(self.num_layers):
            if i == 0:
                self.reassemble.append(nn.ConvTranspose2d(self.features, self.features, kernel_size=4, stride=4))
            elif i == 1:
                self.reassemble.append(nn.ConvTranspose2d(self.features, self.features, kernel_size=2, stride=2))
            elif i == (self.num_layers - 1):
                # last layer: downsample
                self.reassemble.append(nn.Conv2d(self.features, self.features, kernel_size=3, stride=2, padding=1))
            else:
                self.reassemble.append(nn.Identity())

        # Scratch convs (DPT "layer*_rn"): unify channels (already `features`) and add locality.
        self.scratch = nn.ModuleList(
            [nn.Conv2d(self.features, self.features, kernel_size=3, stride=1, padding=1, bias=False) for _ in range(self.num_layers)]
        )

        # RefineNet-style fusion blocks (top-down).
        act = nn.ReLU(inplace=False)
        self.refinenet4 = FeatureFusionBlockCustom(self.features, activation=act, bn=False, align_corners=True)
        self.refinenet3 = FeatureFusionBlockCustom(self.features, activation=act, bn=False, align_corners=True)
        self.refinenet2 = FeatureFusionBlockCustom(self.features, activation=act, bn=False, align_corners=True)
        self.refinenet1 = FeatureFusionBlockCustom(self.features, activation=act, bn=False, align_corners=True)

        # DPT-style "last upsample once" feature path.
        # IMPORTANT: we reduce channels BEFORE upsampling to keep memory reasonable at large resolutions.
        self._pool_ch: int = 32
        self.up_proj = nn.Conv2d(self.features, self._pool_ch, kernel_size=3, stride=1, padding=1)
        self.up_refine = nn.Conv2d(self._pool_ch, self._pool_ch, kernel_size=3, stride=1, padding=1)
        self.up_act = nn.ReLU(inplace=True)

        # Scalar bottleneck MLP (GAP -> MLP -> Linear).
        # We concatenate pooled fused features at H/2 with pooled upsampled features at H.
        in_dim = int(self.features + self._pool_ch)
        hidden_dims = list(getattr(cfg, "head_hidden_dims", None) or [])
        drop = float(getattr(cfg, "dropout", 0.0) or 0.0)
        act_name = str(getattr(cfg, "head_activation", "relu"))

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
        self.scalar_mlp = nn.Sequential(*layers) if layers else nn.Identity()
        self.scalar_dim = cur if layers else in_dim

        self.reg3_head = nn.Linear(self.scalar_dim, self.num_outputs_main)
        self.ratio_head = (
            nn.Linear(self.scalar_dim, self.num_outputs_ratio) if self.num_outputs_ratio > 0 else None
        )
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
        cls_tokens: Optional[Union[Tensor, Sequence[Tensor]]],
        pt_tokens: Union[Tensor, Sequence[Tensor]],
        *,
        image_hw: Tuple[int, int],
    ) -> Dict[str, Optional[Tensor]]:
        # Normalize inputs to lists.
        if isinstance(pt_tokens, Tensor):
            pt_list: List[Tensor] = [pt_tokens]
        else:
            pt_list = list(pt_tokens)

        if cls_tokens is None:
            cls_list: List[Optional[Tensor]] = [None for _ in range(len(pt_list))]
        elif isinstance(cls_tokens, Tensor):
            cls_list = [cls_tokens]
        else:
            cls_list = list(cls_tokens)

        if len(cls_list) != len(pt_list):
            raise RuntimeError(
                f"DPT head expected matching cls/pt lists, got len(cls)={len(cls_list)} len(pt)={len(pt_list)}"
            )
        if len(pt_list) != int(self.num_layers):
            raise RuntimeError(
                "DPT head received an unexpected number of layers. "
                f"Expected num_layers={int(self.num_layers)} but got {len(pt_list)}. "
                "This usually indicates a mismatch between backbone_layer_indices and the exported head meta."
            )

        # Convert each layer tokens to a projected map and reassemble to a specific scale.
        maps: List[Tensor] = []
        for li, pt in enumerate(pt_list):
            cls = cls_list[li]
            if self.readout == "project":
                if cls is None:
                    raise RuntimeError("DPT head readout='project' requires cls_tokens")
                pt = self.readout_proj[li](pt, cls)  # type: ignore[index]

            x = self._tokens_to_map(pt, image_hw=image_hw)
            # Best-effort dtype alignment if autocast is off.
            try:
                w_dtype = next(self.parameters()).dtype
            except Exception:
                w_dtype = x.dtype
            if (not torch.is_autocast_enabled()) and x.dtype != w_dtype:
                x = x.to(dtype=w_dtype)

            x = self.layer_proj[li](x)  # (B, features, Hp, Wp)
            x = self.reassemble[li](x)  # -> target scale
            x = self.scratch[li](x)     # local refinement
            maps.append(x)

        # Expect DPT ordering: [H/4, H/8, H/16, H/32] (or best-effort for non-4).
        # We fuse from the coarsest map (last) to the finest (first).
        if len(maps) < 1:
            raise RuntimeError("DPT head got empty maps list")

        # Coarsest is last
        path_4 = self.refinenet4(maps[-1], None)
        # Next
        path_3 = self.refinenet3(path_4, maps[-2] if len(maps) >= 2 else None)
        path_2 = self.refinenet2(path_3, maps[-3] if len(maps) >= 3 else None)
        path_1 = self.refinenet1(path_2, maps[-4] if len(maps) >= 4 else None)

        # GAP at H/2 on fused features (keeps rich channels).
        g_fused = F.adaptive_avg_pool2d(path_1, output_size=1).flatten(1)

        # DPT-style last upsample once (on reduced channels).
        y = self.up_proj(path_1)
        y = F.interpolate(y, scale_factor=2.0, mode="bilinear", align_corners=True)
        y = self.up_act(self.up_refine(y))
        g_up = F.adaptive_avg_pool2d(y, output_size=1).flatten(1)

        g = torch.cat([g_fused, g_up], dim=-1)
        z = self.scalar_mlp(g) if not isinstance(self.scalar_mlp, nn.Identity) else g

        reg3 = self.reg3_head(z)
        ratio = self.ratio_head(z) if self.ratio_head is not None else None
        ndvi = self.ndvi_head(z) if self.ndvi_head is not None else None
        return {"z": z, "reg3": reg3, "ratio": ratio, "ndvi": ndvi}


