from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch
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


@dataclass
class EoMTQueryHeadConfig:
    """
    EoMT-inspired low-risk head for scalar regression:

    - Keep the DINOv3 backbone frozen.
    - Treat patch tokens as the "memory" sequence.
    - Learn a small set of query tokens and use a lightweight Transformer decoder
      (self-attn over queries + cross-attn to patch tokens) to pool information.
    - Produce a global representation z and predict scalar outputs from it.
    """

    embedding_dim: int
    num_queries: int = 16
    num_layers: int = 2
    num_heads: int = 8
    ffn_dim: int = 2048
    dropout: float = 0.0
    # Pooling strategy over query outputs: "mean" | "first"
    query_pool: str = "mean"
    # Scalar MLP after pooling
    head_hidden_dims: Sequence[int] = ()
    head_activation: str = "relu"
    # Outputs
    num_outputs_main: int = 1
    num_outputs_ratio: int = 0
    enable_ndvi: bool = False


class EoMTQueryScalarHead(nn.Module):
    """
    Query-pooling head that consumes DINOv3 patch tokens (B, N, C) and returns a dict:
      - reg3: (B, num_outputs_main)
      - ratio: (B, num_outputs_ratio) or None
      - ndvi: (B, 1) or None
      - z: (B, scalar_dim)

    For multi-layer inference/training, the head also accepts a list of patch token tensors
    and fuses them via a simple mean (low-risk default).
    """

    def __init__(self, cfg: EoMTQueryHeadConfig) -> None:
        super().__init__()
        self.cfg = cfg

        self.embedding_dim = int(cfg.embedding_dim)
        self.num_queries = int(max(1, cfg.num_queries))
        self.num_layers = int(max(0, cfg.num_layers))
        self.num_heads = int(max(1, cfg.num_heads))
        self.ffn_dim = int(max(1, cfg.ffn_dim))
        self.dropout = float(cfg.dropout or 0.0)
        self.query_pool = str(cfg.query_pool or "mean").strip().lower()

        self.num_outputs_main = int(max(1, cfg.num_outputs_main))
        self.num_outputs_ratio = int(max(0, cfg.num_outputs_ratio))
        self.enable_ndvi = bool(cfg.enable_ndvi)

        if self.embedding_dim <= 0:
            raise ValueError("EoMTQueryHeadConfig.embedding_dim must be positive")
        if self.num_heads <= 0:
            raise ValueError("EoMTQueryHeadConfig.num_heads must be positive")
        if (self.embedding_dim % self.num_heads) != 0:
            raise ValueError(
                f"embedding_dim ({self.embedding_dim}) must be divisible by num_heads ({self.num_heads})"
            )

        # Learnable query tokens (Q, C)
        self.query_embed = nn.Embedding(self.num_queries, self.embedding_dim)

        # Lightweight decoder: self-attn on queries + cross-attn to patch tokens
        if self.num_layers > 0:
            layer = nn.TransformerDecoderLayer(
                d_model=self.embedding_dim,
                nhead=self.num_heads,
                dim_feedforward=self.ffn_dim,
                dropout=self.dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.decoder = nn.TransformerDecoder(layer, num_layers=self.num_layers)
        else:
            self.decoder = None

        # Scalar bottleneck MLP after pooling query outputs
        hidden_dims = list(getattr(cfg, "head_hidden_dims", None) or [])
        act_name = str(getattr(cfg, "head_activation", "relu") or "relu")
        drop = float(getattr(cfg, "dropout", 0.0) or 0.0)

        layers: List[nn.Module] = []
        cur = int(self.embedding_dim)
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
        self.scalar_dim = int(cur if layers else self.embedding_dim)

        self.reg3_head = nn.Linear(self.scalar_dim, self.num_outputs_main)
        self.ratio_head = (
            nn.Linear(self.scalar_dim, self.num_outputs_ratio)
            if self.num_outputs_ratio > 0
            else None
        )
        self.ndvi_head = nn.Linear(self.scalar_dim, 1) if self.enable_ndvi else None

    def _fuse_pt_tokens(self, pt_tokens: Union[Tensor, List[Tensor]]) -> Tensor:
        if isinstance(pt_tokens, Tensor):
            return pt_tokens
        # list/tuple path (multi-layer): simple mean fusion (low-risk default)
        if not isinstance(pt_tokens, (list, tuple)) or len(pt_tokens) == 0:
            raise ValueError("pt_tokens must be a Tensor or a non-empty list of Tensors")
        pt0 = pt_tokens[0]
        if not isinstance(pt0, Tensor):
            raise TypeError("pt_tokens list must contain torch.Tensor elements")
        try:
            pt_stack = torch.stack(list(pt_tokens), dim=1)  # (B, L, N, C)
        except Exception as e:
            raise RuntimeError(f"Failed to stack pt_tokens list for fusion: {e}") from e
        return pt_stack.mean(dim=1)

    def forward(
        self,
        pt_tokens: Union[Tensor, List[Tensor]],
        *,
        image_hw: Optional[Tuple[int, int]] = None,
    ) -> Dict[str, Optional[Tensor]]:
        _ = image_hw  # kept for API compatibility with other heads; unused here

        pt = self._fuse_pt_tokens(pt_tokens)
        if pt.dim() != 3:
            raise RuntimeError(f"Expected patch tokens (B,N,C), got: {tuple(pt.shape)}")
        B, _N, C = pt.shape
        if int(C) != int(self.embedding_dim):
            raise RuntimeError(
                f"Patch token dim mismatch: got C={int(C)}, expected embedding_dim={int(self.embedding_dim)}"
            )

        # Best-effort dtype alignment when autocast is off (avoid matmul dtype mismatch).
        try:
            w_dtype = next(self.parameters()).dtype
        except Exception:
            w_dtype = pt.dtype
        if (not torch.is_autocast_enabled()) and pt.dtype != w_dtype:
            pt = pt.to(dtype=w_dtype)

        # Queries: (B, Q, C)
        q = self.query_embed.weight.unsqueeze(0).expand(B, -1, -1)
        if (not torch.is_autocast_enabled()) and q.dtype != pt.dtype:
            q = q.to(dtype=pt.dtype)

        if self.decoder is not None:
            q_out = self.decoder(tgt=q, memory=pt)
        else:
            q_out = q

        if self.query_pool in ("first", "cls", "q0"):
            pooled = q_out[:, 0]
        else:
            # default: mean pooling
            pooled = q_out.mean(dim=1)

        z = pooled if isinstance(self.scalar_mlp, nn.Identity) else self.scalar_mlp(pooled)
        reg3 = self.reg3_head(z)
        ratio = self.ratio_head(z) if self.ratio_head is not None else None
        ndvi = self.ndvi_head(z) if self.ndvi_head is not None else None
        return {"reg3": reg3, "ratio": ratio, "ndvi": ndvi, "z": z}


__all__ = ["EoMTQueryHeadConfig", "EoMTQueryScalarHead"]


