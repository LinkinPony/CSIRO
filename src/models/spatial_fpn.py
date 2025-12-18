from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def _build_activation(name: str) -> nn.Module:
    n = (name or "").lower().strip()
    if n == "relu":
        return nn.ReLU(inplace=True)
    if n == "gelu":
        return nn.GELU()
    if n in ("silu", "swish"):
        return nn.SiLU(inplace=True)
    return nn.ReLU(inplace=True)


def _auto_group_layers(num_layers: int, num_levels: int) -> List[List[int]]:
    """
    Deterministically group `num_layers` layer indices into `num_levels` contiguous groups.

    - If num_layers >= num_levels: split roughly evenly.
    - If num_layers < num_levels: reuse the last layer for remaining groups.
    """
    L = int(max(1, num_layers))
    K = int(max(1, num_levels))
    if L <= 0:
        return [[0] for _ in range(K)]
    if L < K:
        out: List[List[int]] = []
        for i in range(K):
            out.append([min(i, L - 1)])
        return out
    # sizes like numpy.array_split
    base = L // K
    rem = L % K
    sizes = [base + (1 if i < rem else 0) for i in range(K)]
    groups: List[List[int]] = []
    cur = 0
    for s in sizes:
        groups.append(list(range(cur, cur + s)))
        cur += s
    return groups


def _infer_patch_grid_hw(
    *,
    num_patches: int,
    image_hw: Tuple[int, int],
    patch_size: int,
    strict_primary: bool = True,
) -> Tuple[int, int]:
    """
    Infer (Hp, Wp) patch grid from (H, W) and patch_size.

    Primary rule (ViT patch-embed conv stride=patch_size):
      Hp = H // patch_size
      Wp = W // patch_size

    If Hp*Wp != num_patches:
      - when strict_primary=True: raise immediately (safer; catches wrong patch_size or wrong image_hw)
      - when strict_primary=False: fall back to factor search guided by aspect ratio
    """
    H, W = int(image_hw[0]), int(image_hw[1])
    N = int(num_patches)
    ps = int(max(1, patch_size))

    # Primary: compute from input resolution and patch size.
    hp = max(1, H // ps)
    wp = max(1, W // ps)
    if hp * wp == N:
        return hp, wp
    if bool(strict_primary):
        raise RuntimeError(
            "Patch grid primary inference mismatch. "
            f"Got num_patches={N} but (H//ps)*(W//ps)={(hp * wp)} "
            f"from image_hw=({H},{W}) and patch_size={ps}. "
            "This usually indicates that `image_hw` is not the actual tensor size fed into the backbone, "
            "or `fpn_patch_size` does not match the backbone patch size."
        )

    # Fallback: factor search guided by aspect ratio.
    target_ratio = float(H) / float(max(1, W))
    best = None
    best_err = float("inf")

    # Try a quick closed-form guess first.
    try:
        hp_guess = int(round(math.sqrt(float(N) * float(H) / float(max(1, W)))))
        hp_guess = max(1, min(hp_guess, N))
        for delta in range(-8, 9):
            hpc = hp_guess + delta
            if hpc <= 0:
                continue
            if N % hpc != 0:
                continue
            wpc = N // hpc
            r = float(hpc) / float(max(1, wpc))
            err = abs(r - target_ratio)
            if err < best_err:
                best_err = err
                best = (hpc, wpc)
    except Exception:
        pass

    if best is not None:
        return best

    # Full divisor scan (N is usually small, e.g., 1800).
    for hpc in range(1, N + 1):
        if N % hpc != 0:
            continue
        wpc = N // hpc
        r = float(hpc) / float(max(1, wpc))
        err = abs(r - target_ratio)
        if err < best_err:
            best_err = err
            best = (hpc, wpc)
    if best is not None:
        return best

    # Last resort: assume square-ish.
    side = int(round(math.sqrt(max(1, N))))
    side = max(1, side)
    return side, max(1, N // side)


@dataclass
class FPNHeadConfig:
    embedding_dim: int
    fpn_dim: int
    num_levels: int
    num_layers: int
    use_separate_bottlenecks: bool
    head_hidden_dims: Sequence[int]
    head_activation: str
    dropout: float
    num_outputs_main: int
    num_outputs_ratio: int
    enable_ndvi: bool
    patch_size: int = 16
    # When True (default), require that (H//patch_size)*(W//patch_size) == N exactly.
    # This avoids silent grid mis-inference when patch_size or image_hw are wrong.
    strict_patch_grid: bool = True


class FPNScalarHead(nn.Module):
    """
    Phase-A FPN head:
      - consumes one or more layers of ViT patch tokens (B, N, C)
      - converts to (B, C, Hp, Wp)
      - builds a pseudo multi-scale pyramid (P2/P3/P4...) via group-wise aggregation + downsample
      - global pools each P-level, concatenates, passes through a small MLP, then predicts:
          * reg3 logits: (B, num_outputs_main)
          * ratio logits: (B, num_outputs_ratio) or None
          * ndvi pred: (B, 1) or None
    """

    def __init__(self, cfg: FPNHeadConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.embedding_dim = int(cfg.embedding_dim)
        self.fpn_dim = int(cfg.fpn_dim)
        self.num_levels = int(max(1, cfg.num_levels))
        self.num_layers = int(max(1, cfg.num_layers))
        self.use_separate_bottlenecks = bool(cfg.use_separate_bottlenecks)
        self.patch_size = int(max(1, cfg.patch_size))
        self.strict_patch_grid = bool(getattr(cfg, "strict_patch_grid", True))

        # Per-layer projection (separate_bottlenecks semantics).
        if self.use_separate_bottlenecks:
            self.layer_proj = nn.ModuleList(
                [
                    nn.Conv2d(self.embedding_dim, self.fpn_dim, kernel_size=1, bias=False)
                    for _ in range(self.num_layers)
                ]
            )
        else:
            self.shared_proj = nn.Conv2d(self.embedding_dim, self.fpn_dim, kernel_size=1, bias=False)

        # FPN lateral convs (one per C-level) and smoothing convs (one per P-level).
        self.lateral = nn.ModuleList(
            [nn.Conv2d(self.fpn_dim, self.fpn_dim, kernel_size=1, bias=True) for _ in range(self.num_levels)]
        )
        self.smooth = nn.ModuleList(
            [
                nn.Conv2d(self.fpn_dim, self.fpn_dim, kernel_size=3, padding=1, bias=True)
                for _ in range(self.num_levels)
            ]
        )

        # Scalar bottleneck MLP on concatenated pooled pyramid features.
        in_dim = self.fpn_dim * self.num_levels
        hidden_dims = list(cfg.head_hidden_dims or [])
        drop = float(cfg.dropout or 0.0)

        layers: List[nn.Module] = []
        cur = in_dim
        if drop > 0:
            layers.append(nn.Dropout(drop))
        for hd in hidden_dims:
            hd_i = int(hd)
            layers.append(nn.Linear(cur, hd_i))
            layers.append(_build_activation(cfg.head_activation))
            if drop > 0:
                layers.append(nn.Dropout(drop))
            cur = hd_i
        self.scalar_mlp = nn.Sequential(*layers) if layers else nn.Identity()
        self.scalar_dim = cur if layers else in_dim

        self.reg3_head = nn.Linear(self.scalar_dim, int(cfg.num_outputs_main))
        self.ratio_head = (
            nn.Linear(self.scalar_dim, int(cfg.num_outputs_ratio))
            if int(cfg.num_outputs_ratio) > 0
            else None
        )
        self.ndvi_head = nn.Linear(self.scalar_dim, 1) if bool(cfg.enable_ndvi) else None

    def _project_layer(self, x: Tensor, layer_idx: int) -> Tensor:
        if self.use_separate_bottlenecks:
            return self.layer_proj[layer_idx](x)
        return self.shared_proj(x)  # type: ignore[attr-defined]

    def _tokens_to_map(self, pt: Tensor, *, image_hw: Tuple[int, int]) -> Tensor:
        if pt.dim() != 3:
            raise RuntimeError(f"Expected patch tokens (B,N,C), got: {tuple(pt.shape)}")
        B, N, C = pt.shape
        if C != self.embedding_dim:
            # Be tolerant: some backbones may differ slightly (but this is likely a config error).
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

    def _build_pyramid(
        self,
        pt_list: Sequence[Tensor],
        *,
        image_hw: Tuple[int, int],
    ) -> List[Tensor]:
        """
        Return P-levels as a list [P2, P3, ...] with length = num_levels.
        """
        if len(pt_list) == 0:
            raise ValueError("pt_list must contain at least one tensor")
        if len(pt_list) != int(self.num_layers):
            raise RuntimeError(
                "FPN head received an unexpected number of patch-token layers. "
                f"Expected num_layers={int(self.num_layers)} but got len(pt_list)={len(pt_list)}. "
                "This usually indicates a mismatch between the exported head configuration "
                "(use_layerwise_heads / backbone_layer_indices) and the tokens provided at runtime."
            )

        # Convert each layer tokens to projected maps at full resolution.
        proj_maps: List[Tensor] = []
        for li, pt in enumerate(pt_list):
            x = self._tokens_to_map(pt, image_hw=image_hw)
            # Avoid dtype mismatch in fp32 mode when backbone weights are fp16:
            # if autocast is not enabled, cast tokens to match head weights.
            try:
                w_dtype = next(self.parameters()).dtype
            except Exception:
                w_dtype = x.dtype
            if (not torch.is_autocast_enabled()) and x.dtype != w_dtype:
                x = x.to(dtype=w_dtype)
            proj_maps.append(self._project_layer(x, li))

        # Group layers into num_levels groups and aggregate within each group (mean).
        groups = _auto_group_layers(num_layers=len(proj_maps), num_levels=self.num_levels)
        group_maps: List[Tensor] = []
        for g in groups:
            xs = [proj_maps[i] for i in g if 0 <= i < len(proj_maps)]
            if not xs:
                xs = [proj_maps[-1]]
            group_maps.append(torch.stack(xs, dim=0).mean(dim=0))

        # Downsample each group map to create C2/C3/... like inputs.
        # Level 0 keeps full resolution; level i downsamples by 2**i.
        B, C, Hp, Wp = group_maps[0].shape
        c_levels: List[Tensor] = []
        for i, gm in enumerate(group_maps):
            scale = 2 ** int(i)
            th = max(1, Hp // scale)
            tw = max(1, Wp // scale)
            if gm.shape[-2:] != (th, tw):
                gm = F.interpolate(gm, size=(th, tw), mode="bilinear", align_corners=False)
            c_levels.append(self.lateral[i](gm))

        # Top-down FPN fusion.
        p_levels: List[Tensor] = [torch.empty(0)] * self.num_levels
        prev = None
        for i in reversed(range(self.num_levels)):
            c = c_levels[i]
            if prev is None:
                p = c
            else:
                up = F.interpolate(prev, size=c.shape[-2:], mode="nearest")
                p = c + up
            p = self.smooth[i](p)
            p_levels[i] = p
            prev = p

        return p_levels

    def forward(
        self,
        pt_tokens: Union[Tensor, Sequence[Tensor]],
        *,
        image_hw: Tuple[int, int],
    ) -> Dict[str, Optional[Tensor]]:
        """
        Args:
            pt_tokens: patch tokens (B,N,C) or list of per-layer tokens [(B,N,C), ...]
            image_hw:  (H, W) of the tensor fed into the backbone (after transforms)
        Returns:
            dict with keys:
              - z: (B, scalar_dim) global representation (after MLP)
              - reg3: (B, num_outputs_main) logits in normalized domain
              - ratio: (B, num_outputs_ratio) logits or None
              - ndvi: (B, 1) prediction or None
        """
        if isinstance(pt_tokens, Tensor):
            pt_list: List[Tensor] = [pt_tokens]
        else:
            pt_list = list(pt_tokens)
        p_levels = self._build_pyramid(pt_list, image_hw=image_hw)
        pooled = [F.adaptive_avg_pool2d(p, output_size=1).flatten(1) for p in p_levels]
        g = torch.cat(pooled, dim=-1)
        z = self.scalar_mlp(g) if not isinstance(self.scalar_mlp, nn.Identity) else g

        reg3 = self.reg3_head(z)
        ratio = self.ratio_head(z) if self.ratio_head is not None else None
        ndvi = self.ndvi_head(z) if self.ndvi_head is not None else None
        return {"z": z, "reg3": reg3, "ratio": ratio, "ndvi": ndvi}

