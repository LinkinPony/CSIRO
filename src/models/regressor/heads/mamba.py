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


class _DiagonalSSMScan1D(nn.Module):
    """
    A small, PyTorch-only diagonal state-space scan used as a Mamba-like mixing primitive.

    This is intentionally lightweight and avoids custom CUDA extensions:
      s_t = a * s_{t-1} + b * x_t
      y_t = c * s_t     + d * x_t

    where (a,b,c,d) are learnable per-channel vectors.
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        d = int(dim)
        if d <= 0:
            raise ValueError("dim must be positive for _DiagonalSSMScan1D")
        # a in (0,1) via sigmoid for stability
        self.a_logit = nn.Parameter(torch.zeros(d, dtype=torch.float32))
        self.b = nn.Parameter(torch.ones(d, dtype=torch.float32))
        self.c = nn.Parameter(torch.ones(d, dtype=torch.float32))
        self.d = nn.Parameter(torch.ones(d, dtype=torch.float32))
        # Numerical safety for the vectorized scan (avoid log(0) and div-by-0).
        self._a_eps: float = 1.0e-4
        # Lower bound for a^t to avoid underflow -> inf in u/a^t. This effectively truncates
        # extremely long horizons when `a` is very small, which is fine for this lightweight head.
        self._pow_min: float = 1.0e-20

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (B, L, D)
        Returns:
            y: (B, L, D)
        """
        if x.dim() != 3:
            raise RuntimeError(f"_DiagonalSSMScan1D expects (B,L,D), got: {tuple(x.shape)}")
        B, L, D = x.shape
        if int(D) != int(self.a_logit.numel()):
            raise RuntimeError(
                f"Channel dim mismatch: got D={int(D)} expected {int(self.a_logit.numel())}"
            )

        # Vectorized scan (PyTorch-only): avoid Python loops (major speedup).
        #
        # Recurrence:
        #   s_t = a * s_{t-1} + b * x_t,  s_-1 = 0
        #   y_t = c * s_t     + d * x_t
        #
        # Closed form:
        #   s_t = a^t * sum_{k<=t} (b*x_k / a^k)
        # so we can compute:
        #   a_pow[k] = a^k
        #   r = cumsum((b*x)/a_pow, dim=time)
        #   s = r * a_pow
        #
        # Note: we compute in fp32 for speed (AMP-friendly) and clamp a_pow away from 0
        # to prevent div-by-zero when `a` is very small.
        x_f = x.to(dtype=torch.float32)
        a = torch.sigmoid(self.a_logit).to(device=x.device, dtype=torch.float32)
        a = a.clamp(min=float(self._a_eps), max=float(1.0 - self._a_eps))  # (D,)
        b = self.b.to(device=x.device, dtype=torch.float32)
        c = self.c.to(device=x.device, dtype=torch.float32)
        d = self.d.to(device=x.device, dtype=torch.float32)

        if int(L) <= 0:
            return x_f.to(dtype=x.dtype)

        # Build a_pow: (L, D)
        t = torch.arange(int(L), device=x.device, dtype=torch.float32)  # (L,)
        log_a = torch.log(a)  # (D,)
        a_pow = torch.exp(t.view(-1, 1) * log_a.view(1, -1))  # (L,D)
        a_pow = a_pow.clamp_min(float(self._pow_min))
        a_pow_b = a_pow.unsqueeze(0)  # (1,L,D)

        u = x_f * b.view(1, 1, -1)  # (B,L,D)
        r = torch.cumsum(u / a_pow_b, dim=1)  # (B,L,D)
        s = r * a_pow_b  # (B,L,D)
        y = (s * c.view(1, 1, -1)) + (x_f * d.view(1, 1, -1))
        return y.to(dtype=x.dtype)


class _MambaMixBlock2D(nn.Module):
    """
    A minimal Mamba-like residual block operating on a 2D token map (B, H, W, D):
      LN -> Linear(2D) -> depthwise Conv2d (same padding) -> SiLU -> 2D scan mixing -> gate -> Linear(D) -> residual

    Notes:
      - This block is **non-causal** (symmetric padding) which is more appropriate for image regression.
      - "2D mixing" is implemented by running the diagonal SSM scan over the flattened HxW grid
        in both row-major and column-major orderings and averaging the results.
      - We keep this implementation PyTorch-only (no custom CUDA extensions).
    """

    def __init__(
        self,
        dim: int,
        *,
        d_conv: int = 3,
        dropout: float = 0.0,
        bidirectional: bool = True,
    ) -> None:
        super().__init__()
        d = int(dim)
        if d <= 0:
            raise ValueError("dim must be positive for _MambaMixBlock2D")
        k = int(max(1, d_conv))
        # For "same" padding with symmetric padding, prefer odd kernels.
        if k % 2 == 0:
            k += 1

        self.dim = d
        self.d_conv = k
        self.bidirectional = bool(bidirectional)

        self.norm = nn.LayerNorm(d)
        self.in_proj = nn.Linear(d, 2 * d, bias=True)
        # Non-causal depthwise Conv2d (same padding)
        self.conv2d = nn.Conv2d(
            in_channels=d,
            out_channels=d,
            kernel_size=k,
            padding=k // 2,
            groups=d,
            bias=True,
        )
        self.ssm = _DiagonalSSMScan1D(d)
        self.out_proj = nn.Linear(d, d, bias=True)
        self.drop = nn.Dropout(float(dropout)) if float(dropout or 0.0) > 0.0 else nn.Identity()

    def _conv_act_2d(self, x: Tensor) -> Tensor:
        # x: (B,H,W,D) -> (B,D,H,W) for conv2d
        if x.dim() != 4:
            raise RuntimeError(f"_conv_act_2d expects (B,H,W,D), got: {tuple(x.shape)}")
        x_c = x.permute(0, 3, 1, 2).contiguous()  # (B,D,H,W)
        y = self.conv2d(x_c)  # (B,D,H,W)
        y = F.silu(y)
        return y.permute(0, 2, 3, 1).contiguous()  # (B,H,W,D)

    def _scan2d(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (B,H,W,D)
        Returns:
            y: (B,H,W,D)
        """
        if x.dim() != 4:
            raise RuntimeError(f"_scan2d expects (B,H,W,D), got: {tuple(x.shape)}")
        B, H, W, D = x.shape
        L = int(H * W)
        if L <= 0:
            return x

        # Row-major sequence: (h,w) with w varying fastest.
        seq_row = x.reshape(int(B), int(L), int(D))  # (B,L,D)

        # Column-major sequence: (w,h) with h varying fastest (permute H/W first).
        x_col = x.permute(0, 2, 1, 3).contiguous()  # (B,W,H,D)
        seq_col = x_col.reshape(int(B), int(L), int(D))  # (B,L,D)

        # Run the diagonal scan on both orderings in a single batched call.
        if self.bidirectional:
            seqs = torch.cat([seq_row, seq_col], dim=0)  # (2B,L,D)
            seqs_rev = torch.flip(seqs, dims=[1])  # (2B,L,D)
            y2 = self.ssm(torch.cat([seqs, seqs_rev], dim=0))  # (4B,L,D)
            y_f, y_rev = y2.chunk(2, dim=0)  # each (2B,L,D)
            y_b = torch.flip(y_rev, dims=[1])
            y = 0.5 * (y_f + y_b)  # (2B,L,D)
        else:
            y = self.ssm(torch.cat([seq_row, seq_col], dim=0))  # (2B,L,D)

        y_row, y_col = y.chunk(2, dim=0)  # each (B,L,D)
        y_row_map = y_row.reshape(int(B), int(H), int(W), int(D))
        y_col_map = y_col.reshape(int(B), int(W), int(H), int(D)).permute(0, 2, 1, 3).contiguous()
        return 0.5 * (y_row_map + y_col_map)

    def forward(self, x: Tensor) -> Tensor:
        if x.dim() != 4:
            raise RuntimeError(f"_MambaMixBlock2D expects (B,H,W,D), got: {tuple(x.shape)}")

        # Best-effort dtype alignment when autocast is disabled:
        # - frozen backbone may emit fp16 tokens while the head params are fp32.
        try:
            w_dtype = next(self.parameters()).dtype
        except Exception:
            w_dtype = x.dtype
        if (not torch.is_autocast_enabled()) and x.dtype != w_dtype:
            x = x.to(dtype=w_dtype)

        x0 = x
        h = self.norm(x)
        x_gate = self.in_proj(h)
        x_in, gate = x_gate.chunk(2, dim=-1)
        gate = torch.sigmoid(gate)

        u = self._conv_act_2d(x_in)
        y = self._scan2d(u)
        y = y * gate
        y = self.out_proj(y)
        y = self.drop(y)
        return x0 + y


@dataclass
class MambaHeadConfig:
    embedding_dim: int
    mamba_dim: int
    depth: int
    patch_size: int
    d_conv: int
    bidirectional: bool
    num_outputs_main: int
    num_outputs_ratio: int
    enable_ndvi: bool
    # Ratio coupling
    separate_ratio_head: bool = False
    separate_ratio_spatial_head: bool = False
    # Patch grid inference strictness
    strict_patch_grid: bool = True
    # Scalar head (pool -> MLP -> Linear)
    head_hidden_dims: Sequence[int] = ()
    head_activation: str = "relu"
    dropout: float = 0.0


class Mamba2DScalarHead(nn.Module):
    """
    PyTorch-only Mamba-like scalar head over ViT patch tokens with 2D mixing.

    - Consumes patch tokens (B,N,C) from a single backbone layer.
    - Reshapes to a 2D grid (B,Hp,Wp,*) using image_hw + patch_size.
    - Applies non-causal 2D Mamba-like mixing blocks (depthwise Conv2d + 2D scan).
    - Pools to a global vector and predicts reg3/ratio/ndvi.
    """

    def __init__(self, cfg: MambaHeadConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.embedding_dim = int(cfg.embedding_dim)
        self.mamba_dim = int(cfg.mamba_dim)
        self.depth = int(max(0, cfg.depth))
        self.patch_size = int(max(1, cfg.patch_size))
        self.d_conv = int(max(1, cfg.d_conv))
        self.bidirectional = bool(cfg.bidirectional)
        self.strict_patch_grid = bool(getattr(cfg, "strict_patch_grid", True))

        self.num_outputs_main = int(max(1, cfg.num_outputs_main))
        self.num_outputs_ratio = int(max(0, cfg.num_outputs_ratio))
        self.enable_ndvi = bool(cfg.enable_ndvi)

        self.separate_ratio_spatial_head = bool(getattr(cfg, "separate_ratio_spatial_head", False))
        # separate spatial implies separate scalar branch
        self.separate_ratio_head = bool(getattr(cfg, "separate_ratio_head", False)) or self.separate_ratio_spatial_head

        if self.embedding_dim <= 0:
            raise ValueError("MambaHeadConfig.embedding_dim must be positive")
        if self.mamba_dim <= 0:
            raise ValueError("MambaHeadConfig.mamba_dim must be positive")

        # Main spatial trunk
        self.in_proj = nn.Linear(self.embedding_dim, self.mamba_dim, bias=True)
        self.blocks = nn.ModuleList(
            [
                _MambaMixBlock2D(
                    self.mamba_dim,
                    d_conv=self.d_conv,
                    dropout=float(cfg.dropout or 0.0),
                    bidirectional=self.bidirectional,
                )
                for _ in range(int(self.depth))
            ]
        )
        self.out_norm = nn.LayerNorm(self.mamba_dim)

        # Optional separate spatial trunk for ratio
        if self.separate_ratio_spatial_head and self.num_outputs_ratio > 0:
            self.ratio_in_proj = nn.Linear(self.embedding_dim, self.mamba_dim, bias=True)
            self.ratio_blocks = nn.ModuleList(
                [
                    _MambaMixBlock2D(
                        self.mamba_dim,
                        d_conv=self.d_conv,
                        dropout=float(cfg.dropout or 0.0),
                        bidirectional=self.bidirectional,
                    )
                    for _ in range(int(self.depth))
                ]
            )
            self.ratio_out_norm = nn.LayerNorm(self.mamba_dim)
        else:
            self.ratio_in_proj = None  # type: ignore[assignment]
            self.ratio_blocks = None  # type: ignore[assignment]
            self.ratio_out_norm = None  # type: ignore[assignment]

        # Scalar bottleneck MLP(s) after pooling
        hidden_dims = list(getattr(cfg, "head_hidden_dims", None) or [])
        act_name = str(getattr(cfg, "head_activation", "relu") or "relu")
        drop = float(getattr(cfg, "dropout", 0.0) or 0.0)

        def _build_scalar_mlp() -> tuple[nn.Module, int]:
            layers: List[nn.Module] = []
            cur = int(self.mamba_dim)
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
            dim = cur if layers else int(self.mamba_dim)
            return mlp, dim

        self.scalar_mlp, self.scalar_dim = _build_scalar_mlp()

        if self.separate_ratio_head and self.num_outputs_ratio > 0:
            self.ratio_mlp, ratio_dim = _build_scalar_mlp()
        else:
            self.ratio_mlp = None
            ratio_dim = int(self.scalar_dim)

        self.reg3_head = nn.Linear(int(self.scalar_dim), self.num_outputs_main)
        self.ratio_head = nn.Linear(int(ratio_dim), self.num_outputs_ratio) if self.num_outputs_ratio > 0 else None
        self.ndvi_head = nn.Linear(int(self.scalar_dim), 1) if self.enable_ndvi else None

    def _infer_hw(self, pt: Tensor, *, image_hw: Tuple[int, int]) -> Tuple[int, int]:
        if pt.dim() != 3:
            raise RuntimeError(f"Expected patch tokens (B,N,C), got: {tuple(pt.shape)}")
        _b, N, _c = pt.shape
        Hp, Wp = _infer_patch_grid_hw(
            num_patches=int(N),
            image_hw=image_hw,
            patch_size=int(self.patch_size),
            strict_primary=bool(self.strict_patch_grid),
        )
        return int(Hp), int(Wp)

    def _run_trunk(self, pt_tokens: Tensor, *, image_hw: Tuple[int, int]) -> Tensor:
        # Returns token map features (B, Hp, Wp, D)
        if pt_tokens.dim() != 3:
            raise RuntimeError(f"Expected patch tokens (B,N,C), got: {tuple(pt_tokens.shape)}")
        B, N, C = pt_tokens.shape
        if int(C) != int(self.embedding_dim):
            raise RuntimeError(
                f"Patch token dim mismatch: got C={int(C)}, expected embedding_dim={int(self.embedding_dim)}"
            )
        Hp, Wp = self._infer_hw(pt_tokens, image_hw=image_hw)
        if int(Hp * Wp) != int(N):
            raise RuntimeError(
                f"Cannot reshape tokens to grid: N={int(N)}, inferred Hp={Hp}, Wp={Wp}, Hp*Wp={Hp*Wp}"
            )

        x = self.in_proj(pt_tokens)  # (B,N,D)
        x = x.reshape(int(B), int(Hp), int(Wp), int(self.mamba_dim))
        for blk in self.blocks:
            x = blk(x)
        # Normalize across channels (last dim)
        x = self.out_norm(x)
        return x

    def _run_ratio_trunk(self, pt_tokens: Tensor, *, image_hw: Tuple[int, int]) -> Tensor:
        if self.ratio_in_proj is None or self.ratio_blocks is None or self.ratio_out_norm is None:
            raise RuntimeError("ratio trunk is not initialized (separate_ratio_spatial_head is False)")
        B, N, C = pt_tokens.shape
        Hp, Wp = self._infer_hw(pt_tokens, image_hw=image_hw)
        x = self.ratio_in_proj(pt_tokens)
        x = x.reshape(int(B), int(Hp), int(Wp), int(self.mamba_dim))
        for blk in self.ratio_blocks:
            x = blk(x)
        x = self.ratio_out_norm(x)
        return x

    def forward(self, pt_tokens: Tensor, *, image_hw: Tuple[int, int]) -> Dict[str, Optional[Tensor]]:
        # Main trunk
        x_map = self._run_trunk(pt_tokens, image_hw=image_hw)
        g = x_map.mean(dim=(1, 2))  # (B, D)
        z = self.scalar_mlp(g) if not isinstance(self.scalar_mlp, nn.Identity) else g

        # Ratio branch (mode-dependent)
        if self.separate_ratio_spatial_head and self.ratio_head is not None:
            x_ratio = self._run_ratio_trunk(pt_tokens, image_hw=image_hw)
            g_ratio = x_ratio.mean(dim=(1, 2))
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


class MambaMultiLayerScalarHead(nn.Module):
    """
    Multi-layer wrapper: one independent Mamba2DScalarHead per backbone layer,
    and fuse the final outputs across layers (mean or learned softmax weights).
    """

    def __init__(
        self,
        cfg: MambaHeadConfig,
        *,
        num_layers: int,
        layer_fusion: str = "mean",
    ) -> None:
        super().__init__()
        if int(num_layers) <= 0:
            raise ValueError("num_layers must be >= 1 for MambaMultiLayerScalarHead")
        self.cfg = cfg
        self.num_layers = int(num_layers)
        self.layer_fusion: str = str(layer_fusion or "mean").strip().lower()
        self.heads = nn.ModuleList([Mamba2DScalarHead(cfg) for _ in range(self.num_layers)])
        # Expose scalar_dim for downstream aux heads
        self.scalar_dim = int(getattr(self.heads[0], "scalar_dim", cfg.mamba_dim))
        if self.layer_fusion == "learned":
            self.layer_logits = nn.Parameter(torch.zeros(self.num_layers, dtype=torch.float32))
        else:
            self.layer_logits = None  # type: ignore[assignment]

    def get_layer_weights(
        self,
        *,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> Tensor:
        L = int(self.num_layers)
        if L <= 0:
            w = torch.ones(1, dtype=torch.float32)
        elif self.layer_fusion == "learned" and isinstance(self.layer_logits, torch.Tensor):
            w = torch.softmax(self.layer_logits.to(dtype=torch.float32), dim=0)
        else:
            w = torch.full((L,), 1.0 / float(L), dtype=torch.float32)
        if device is not None:
            w = w.to(device=device)
        if dtype is not None:
            w = w.to(dtype=dtype)
        return w

    @staticmethod
    def _avg_optional(xs: List[Optional[Tensor]]) -> Optional[Tensor]:
        ts = [t for t in xs if isinstance(t, torch.Tensor)]
        if not ts:
            return None
        return torch.stack(ts, dim=0).mean(dim=0)

    def _fuse_optional(self, xs: List[Optional[Tensor]], *, weights: Tensor) -> Optional[Tensor]:
        ts: List[Tensor] = []
        idxs: List[int] = []
        for i, t in enumerate(xs):
            if not isinstance(t, torch.Tensor):
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
                f"MambaMultiLayerScalarHead expected {int(self.num_layers)} layers but got {len(pt_list)}"
            )
        outs = [h(pt_list[i], image_hw=image_hw) for i, h in enumerate(self.heads)]
        z_list = [o.get("z", None) for o in outs]
        reg_list = [o.get("reg3", None) for o in outs]
        ratio_list = [o.get("ratio", None) for o in outs]
        ndvi_list = [o.get("ndvi", None) for o in outs]

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


def init_mamba_head(
    model,
    *,
    embedding_dim: int,
    mamba_dim: int,
    mamba_depth: int,
    mamba_patch_size: int,
    mamba_d_conv: int,
    mamba_bidirectional: bool,
    hidden_dims: List[int],
    head_activation: str,
    dropout: float,
) -> int:
    """
    Initialize the Mamba 2D-mixing head on `model` (`mamba_head`).
    Returns bottleneck_dim (scalar feature dim produced by the head).
    """
    # Ensure mutually-exclusive head modules exist.
    model.fpn_head = None  # type: ignore[assignment]
    model.dpt_head = None  # type: ignore[assignment]
    model.vitdet_head = None  # type: ignore[assignment]
    model.eomt_head = None  # type: ignore[assignment]

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

    cfg = MambaHeadConfig(
        embedding_dim=int(embedding_dim),
        mamba_dim=int(mamba_dim),
        depth=int(mamba_depth),
        patch_size=int(mamba_patch_size),
        d_conv=int(mamba_d_conv),
        bidirectional=bool(mamba_bidirectional),
        num_outputs_main=int(num_outputs_main),
        num_outputs_ratio=int(num_ratio_outputs),
        enable_ndvi=bool(enable_ndvi),
        separate_ratio_head=bool(getattr(model, "separate_ratio_head", False)),
        separate_ratio_spatial_head=bool(getattr(model, "separate_ratio_spatial_head", False)),
        head_hidden_dims=list(hidden_dims),
        head_activation=str(head_activation),
        dropout=float(dropout or 0.0),
    )

    if use_layerwise_heads:
        layer_fusion = str(getattr(model, "backbone_layers_fusion", "mean") or "mean").strip().lower()
        model.mamba_head = MambaMultiLayerScalarHead(cfg, num_layers=num_layers_eff, layer_fusion=layer_fusion)  # type: ignore[assignment]
        bottleneck_dim = int(getattr(model.mamba_head, "scalar_dim", int(mamba_dim)))  # type: ignore[attr-defined]
    else:
        model.mamba_head = Mamba2DScalarHead(cfg)  # type: ignore[assignment]
        bottleneck_dim = int(getattr(model.mamba_head, "scalar_dim", int(mamba_dim)))  # type: ignore[attr-defined]
    return int(bottleneck_dim)


def init_mamba_task_heads(
    model,
    *,
    bottleneck_dim: int,
) -> None:
    """
    Initialize auxiliary heads for the Mamba head type.

    Notes:
      - NDVI is produced by the Mamba head itself; keep legacy `ndvi_head` as None.
      - Ratio logits are produced by the Mamba head itself; keep legacy `ratio_head` as None.
    """
    enable_height = bool(getattr(model, "enable_height", False))
    enable_species = bool(getattr(model, "enable_species", False))
    enable_state = bool(getattr(model, "enable_state", False))
    enable_date = bool(getattr(model, "enable_date", False))

    model.height_head = nn.Linear(int(bottleneck_dim), 1) if enable_height else None  # type: ignore[assignment]
    model.date_head = nn.Linear(int(bottleneck_dim), 2) if enable_date else None  # type: ignore[assignment]
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

    # No layer-wise scalar heads for Mamba (multi-layer averaging happens inside mamba_head).
    model.layer_reg3_heads = None  # type: ignore[assignment]
    model.layer_ratio_heads = None  # type: ignore[assignment]
    model.layer_height_heads = None  # type: ignore[assignment]
    model.layer_ndvi_heads = None  # type: ignore[assignment]
    model.layer_date_heads = None  # type: ignore[assignment]
    model.layer_species_heads = None  # type: ignore[assignment]
    model.layer_state_heads = None  # type: ignore[assignment]


__all__ = [
    "MambaHeadConfig",
    "Mamba2DScalarHead",
    "MambaMultiLayerScalarHead",
    "init_mamba_head",
    "init_mamba_task_heads",
]

