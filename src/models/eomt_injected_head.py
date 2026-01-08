from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

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
class EoMTInjectedQueryHeadConfig:
    """
    EoMT-style injected-query head for scalar regression:

    - Run the frozen DINOv3 backbone for the first (depth - k) blocks WITHOUT queries.
    - Prepend Q learnable query tokens.
    - Run the remaining k blocks jointly over [queries + prefix tokens + patch tokens].
    - Pool the final query tokens into a global representation z and regress scalars from it.

    This matches the "insert queries into last-k blocks" design used in `third_party/eomt`.
    """

    embedding_dim: int
    num_queries: int = 16
    # Number of LAST backbone blocks to run jointly with injected queries (k in the description above).
    num_blocks: int = 4
    # Pooling strategy over query outputs: "mean" | "first"
    query_pool: str = "mean"

    # ---------------------------------------------------------------------
    # Global representation construction (new; backward compatible defaults)
    # ---------------------------------------------------------------------
    # Which sources to include when building the global vector before the scalar MLP.
    # - mean_query: mean pooled injected query outputs (B, C)
    # - mean_patch: mean pooled patch tokens after the injected-query blocks (B, C)
    # - cls:        CLS token after the injected-query blocks (B, C)
    #
    # NOTE: By default we keep legacy behavior: use only mean_query.
    use_mean_query: bool = True
    use_mean_patch: bool = False
    use_cls_token: bool = False

    # Project concatenated sources down to this dim before the scalar MLP.
    #
    # - If <= 0:
    #   - legacy mode (only mean_query selected): projection is Identity
    #   - multi-source mode: projection defaults to embedding_dim
    proj_dim: int = 0
    proj_activation: str = "relu"
    proj_dropout: float = 0.0

    # Scalar MLP after pooling
    head_hidden_dims: Sequence[int] = ()
    head_activation: str = "relu"
    dropout: float = 0.0

    # Outputs
    num_outputs_main: int = 1
    num_outputs_ratio: int = 0
    enable_ndvi: bool = False


class EoMTInjectedQueryScalarHead(nn.Module):
    """
    Injected-query EoMT head.

    Inputs:
      - x_base: token sequence AFTER running the backbone up to (depth - k) blocks,
                shape (B, N_tokens, C) with tokens ordered as:
                  [CLS, (storage tokens...), patch tokens]
      - backbone: the (possibly PEFT-wrapped) DINOv3 backbone module
      - patch_hw: (H_p, W_p) patch grid size used to build RoPE embeddings

    Outputs (dict):
      - reg3:  (B, num_outputs_main)
      - ratio: (B, num_outputs_ratio) or None
      - ndvi:  (B, 1) or None
      - z:     (B, scalar_dim)
    """

    def __init__(self, cfg: EoMTInjectedQueryHeadConfig) -> None:
        super().__init__()
        self.cfg = cfg

        self.embedding_dim = int(cfg.embedding_dim)
        self.num_queries = int(max(1, cfg.num_queries))
        self.num_blocks = int(max(0, cfg.num_blocks))
        self.query_pool = str(cfg.query_pool or "mean").strip().lower()

        # Which sources to use for the global representation.
        self.use_mean_query = bool(getattr(cfg, "use_mean_query", True))
        self.use_mean_patch = bool(getattr(cfg, "use_mean_patch", False))
        # Accept both `use_cls_token` (preferred) and legacy `use_cls`.
        self.use_cls_token = bool(
            getattr(cfg, "use_cls_token", getattr(cfg, "use_cls", False))
        )

        # Projection config (concat -> proj_dim). May be Identity for legacy mode.
        proj_dim = int(getattr(cfg, "proj_dim", 0) or 0)
        self._proj_dim_cfg = int(proj_dim)
        self.proj_activation = str(getattr(cfg, "proj_activation", "relu") or "relu")
        self.proj_dropout = float(getattr(cfg, "proj_dropout", 0.0) or 0.0)

        self.num_outputs_main = int(max(1, cfg.num_outputs_main))
        self.num_outputs_ratio = int(max(0, cfg.num_outputs_ratio))
        self.enable_ndvi = bool(cfg.enable_ndvi)

        if self.embedding_dim <= 0:
            raise ValueError("EoMTInjectedQueryHeadConfig.embedding_dim must be positive")

        if not (self.use_mean_query or self.use_mean_patch or self.use_cls_token):
            raise ValueError(
                "At least one of use_mean_query/use_mean_patch/use_cls_token must be True"
            )

        # Learnable query tokens (Q, C)
        self.query_embed = nn.Embedding(self.num_queries, self.embedding_dim)

        # Concat feature dim (mean_query / mean_patch / cls).
        concat_dim = 0
        concat_dim += self.embedding_dim if self.use_mean_query else 0
        concat_dim += self.embedding_dim if self.use_mean_patch else 0
        concat_dim += self.embedding_dim if self.use_cls_token else 0
        self.concat_dim = int(concat_dim)

        # Determine effective projection dim:
        # - cfg.proj_dim > 0: use that
        # - cfg.proj_dim <= 0:
        #   - legacy single-source mode: Identity projection (proj_dim = concat_dim = embedding_dim)
        #   - multi-source mode: default to embedding_dim
        if self._proj_dim_cfg > 0:
            self.proj_dim = int(self._proj_dim_cfg)
        else:
            self.proj_dim = int(self.embedding_dim if self.concat_dim != self.embedding_dim else self.concat_dim)

        need_proj = bool(self.concat_dim != self.proj_dim) or bool(self._proj_dim_cfg > 0 and self.concat_dim == self.proj_dim)
        if need_proj:
            proj_layers: List[nn.Module] = []
            if self.proj_dropout > 0:
                proj_layers.append(nn.Dropout(self.proj_dropout))
            proj_layers.append(nn.Linear(self.concat_dim, self.proj_dim))
            # Light non-linearity; can be disabled by setting proj_activation to ""/None.
            if str(self.proj_activation or "").strip():
                proj_layers.append(_build_activation(self.proj_activation))
            self.proj = nn.Sequential(*proj_layers)
        else:
            self.proj = nn.Identity()

        # Scalar bottleneck MLP after pooling query outputs
        hidden_dims = list(getattr(cfg, "head_hidden_dims", None) or [])
        act_name = str(getattr(cfg, "head_activation", "relu") or "relu")
        drop = float(getattr(cfg, "dropout", 0.0) or 0.0)

        layers: List[nn.Module] = []
        cur = int(self.proj_dim if not isinstance(self.proj, nn.Identity) else self.concat_dim)
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
        self.scalar_dim = int(cur if layers else (self.proj_dim if not isinstance(self.proj, nn.Identity) else self.concat_dim))

        self.reg3_head = nn.Linear(self.scalar_dim, self.num_outputs_main)
        self.ratio_head = (
            nn.Linear(self.scalar_dim, self.num_outputs_ratio)
            if self.num_outputs_ratio > 0
            else None
        )
        self.ndvi_head = nn.Linear(self.scalar_dim, 1) if self.enable_ndvi else None

    @staticmethod
    def _resolve_base_backbone(backbone: nn.Module) -> nn.Module:
        # 1) Direct model
        if hasattr(backbone, "blocks"):
            return backbone
        # 2) PEFT-style: backbone.base_model.model
        base_model = getattr(backbone, "base_model", None)
        if isinstance(base_model, nn.Module):
            cand = getattr(base_model, "model", None)
            if isinstance(cand, nn.Module) and hasattr(cand, "blocks"):
                return cand
            if hasattr(base_model, "blocks"):
                return base_model
        # 3) Some wrappers use `.model`
        cand2 = getattr(backbone, "model", None)
        if isinstance(cand2, nn.Module) and hasattr(cand2, "blocks"):
            return cand2
        return backbone

    def _forward_x_norm(
        self,
        x_base: Tensor,
        *,
        backbone: nn.Module,
        patch_hw: Tuple[int, int],
    ) -> Tensor:
        """
        Run the injected-query transformer and return the *normalized* token sequence.

        Returns:
            x_norm: Tensor of shape (B, Q + N_tokens, C) where the first Q tokens are
                   the injected query outputs.
        """
        if x_base.dim() != 3:
            raise RuntimeError(
                f"Expected x_base tokens (B,N,C), got: {tuple(x_base.shape)}"
            )
        B, _N, C = x_base.shape
        if int(C) != int(self.embedding_dim):
            raise RuntimeError(
                f"Token dim mismatch: got C={int(C)}, expected embedding_dim={int(self.embedding_dim)}"
            )

        bb = self._resolve_base_backbone(backbone)
        blocks = getattr(bb, "blocks", None)
        if not isinstance(blocks, (nn.ModuleList, list)) or len(blocks) == 0:
            raise RuntimeError(
                "Injected EoMT head requires a DINOv3 ViT backbone with `.blocks`"
            )
        depth = len(blocks)
        k = int(max(0, min(self.num_blocks, depth)))
        start = int(depth - k)

        H_p, W_p = int(patch_hw[0]), int(patch_hw[1])
        if H_p <= 0 or W_p <= 0:
            raise ValueError(f"Invalid patch_hw={patch_hw}; expected positive ints")

        # Prepend learnable query tokens (B, Q, C) to the token sequence.
        q = self.query_embed.weight.unsqueeze(0).expand(B, -1, -1)
        if (not torch.is_autocast_enabled()) and q.dtype != x_base.dtype:
            q = q.to(dtype=x_base.dtype)
        if q.device != x_base.device:
            q = q.to(device=x_base.device)
        x = torch.cat((q, x_base), dim=1)

        rope_embed = getattr(bb, "rope_embed", None)
        rope = rope_embed(H=H_p, W=W_p) if rope_embed is not None else None

        for i in range(start, depth):
            x = blocks[i](x, rope)  # type: ignore[call-arg]

        # Normalize tokens the same way the backbone would.
        norm = getattr(bb, "norm", None)
        if norm is None or not isinstance(norm, nn.Module):
            raise RuntimeError(
                "Backbone does not expose a `.norm` module needed for token normalization"
            )

        untie = bool(getattr(bb, "untie_cls_and_patch_norms", False))
        cls_norm = getattr(bb, "cls_norm", None)
        if untie and isinstance(cls_norm, nn.Module):
            n_storage = int(getattr(bb, "n_storage_tokens", 0) or 0)
            prefix_len = int(self.num_queries + (n_storage + 1))
            x_norm_prefix = cls_norm(x[:, :prefix_len])
            x_norm_patch = norm(x[:, prefix_len:])
            x_norm = torch.cat((x_norm_prefix, x_norm_patch), dim=1)
        else:
            x_norm = norm(x)

        return x_norm

    @staticmethod
    def _split_tokens(
        x_tokens: Tensor,
        *,
        backbone: nn.Module,
        num_queries: int,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Split a token sequence into (q_out, cls_token, patch_tokens).

        Args:
            x_tokens: (B, Q + N_tokens, C) normalized token sequence.
            backbone: backbone module (used for n_storage_tokens if present).
            num_queries: Q

        Returns:
            q_out: (B, Q, C)
            cls_tok: (B, C)
            pt: (B, N_patch, C)
        """
        if x_tokens.dim() != 3:
            raise RuntimeError(f"Expected tokens (B,N,C), got: {tuple(x_tokens.shape)}")
        if int(num_queries) <= 0:
            raise ValueError("num_queries must be positive")
        B, N, _C = x_tokens.shape
        if N <= int(num_queries):
            raise RuntimeError(
                f"Token sequence too short: N={int(N)} <= Q={int(num_queries)}"
            )

        bb = EoMTInjectedQueryScalarHead._resolve_base_backbone(backbone)
        n_storage = int(getattr(bb, "n_storage_tokens", 0) or 0)
        # After query tokens: [CLS, (storage tokens...), patch tokens]
        base = x_tokens[:, int(num_queries) :, :]
        if base.size(1) < 1:
            raise RuntimeError("Missing CLS token in base token sequence")
        cls_tok = base[:, 0, :]
        # Remaining after CLS: storage + patch
        rest = base[:, 1:, :]
        if n_storage > 0 and rest.size(1) >= n_storage:
            pt = rest[:, n_storage:, :]
        else:
            # If backbone has no storage tokens (or mismatch), treat all as patch tokens.
            pt = rest
        if pt.numel() == 0 or pt.size(1) <= 0:
            raise RuntimeError("No patch tokens found in token sequence")
        q_out = x_tokens[:, : int(num_queries), :]
        return q_out, cls_tok, pt

    def forward_query_tokens(
        self,
        x_base: Tensor,
        *,
        backbone: nn.Module,
        patch_hw: Tuple[int, int],
    ) -> Tensor:
        """
        Return the final normalized injected-query outputs.

        Returns:
            q_out: Tensor of shape (B, Q, C)
        """
        x_norm = self._forward_x_norm(x_base, backbone=backbone, patch_hw=patch_hw)
        return x_norm[:, : self.num_queries, :]

    def _pool_query(self, q_out: Tensor) -> Tensor:
        if self.query_pool in ("first", "cls", "q0"):
            return q_out[:, 0]
        return q_out.mean(dim=1)

    def build_concat_features_from_tokens(
        self,
        x_norm: Tensor,
        *,
        backbone: nn.Module,
    ) -> Tensor:
        """
        Build the concatenated feature vector (before projection/MLP) from the normalized
        token sequence.

        Returns:
            fused_concat: (B, concat_dim)
        """
        q_out, cls_tok, pt = self._split_tokens(
            x_norm, backbone=backbone, num_queries=int(self.num_queries)
        )
        parts: List[Tensor] = []
        if self.use_mean_query:
            parts.append(self._pool_query(q_out))
        if self.use_mean_patch:
            parts.append(pt.mean(dim=1))
        if self.use_cls_token:
            parts.append(cls_tok)
        if not parts:
            raise RuntimeError("No feature sources selected for EoMT fusion")
        fused_concat = torch.cat(parts, dim=-1) if len(parts) > 1 else parts[0]
        return fused_concat

    def forward_from_concat_features(
        self,
        fused_concat: Tensor,
    ) -> Dict[str, Optional[Tensor]]:
        """
        Compute predictions from the concatenated feature vector (before projection).
        This is useful for applying manifold mixup on the exact representation used by the head.
        """
        if fused_concat.dim() != 2:
            raise RuntimeError(
                f"Expected fused_concat (B,F), got: {tuple(fused_concat.shape)}"
            )
        if int(fused_concat.size(-1)) != int(self.concat_dim):
            raise RuntimeError(
                f"Concat feature dim mismatch: got F={int(fused_concat.size(-1))}, expected concat_dim={int(self.concat_dim)}"
            )
        fused = self.proj(fused_concat) if not isinstance(self.proj, nn.Identity) else fused_concat
        z = fused if isinstance(self.scalar_mlp, nn.Identity) else self.scalar_mlp(fused)
        reg3 = self.reg3_head(z)
        ratio = self.ratio_head(z) if self.ratio_head is not None else None
        ndvi = self.ndvi_head(z) if self.ndvi_head is not None else None
        return {"reg3": reg3, "ratio": ratio, "ndvi": ndvi, "z": z}

    def forward_from_query_tokens(self, q_out: Tensor) -> Dict[str, Optional[Tensor]]:
        """
        Compute scalar predictions from (possibly mixed) query-token outputs.

        Args:
            q_out: (B, Q, C) normalized query-token outputs
        """
        if q_out.dim() != 3:
            raise RuntimeError(
                f"Expected q_out tokens (B,Q,C), got: {tuple(q_out.shape)}"
            )
        if int(q_out.size(-1)) != int(self.embedding_dim):
            raise RuntimeError(
                f"Query token dim mismatch: got C={int(q_out.size(-1))}, expected embedding_dim={int(self.embedding_dim)}"
            )
        if int(q_out.size(1)) != int(self.num_queries):
            raise RuntimeError(
                f"Query token count mismatch: got Q={int(q_out.size(1))}, expected num_queries={int(self.num_queries)}"
            )

        # Legacy API: when only query tokens are provided, we can only use mean_query/first.
        if (not self.use_mean_query) or self.use_mean_patch or self.use_cls_token:
            raise RuntimeError(
                "This EoMT head is configured to use mean_patch/cls and requires `forward(...)` "
                "(or `forward_from_tokens(...)`) so it can access patch/CLS tokens. "
                "Got `forward_from_query_tokens(...)` with only q_out."
            )
        pooled_q = self._pool_query(q_out)
        if pooled_q.dim() != 2 or int(pooled_q.size(-1)) != int(self.embedding_dim):
            raise RuntimeError("Internal error: pooled query features have unexpected shape")
        fused_concat = pooled_q
        return self.forward_from_concat_features(fused_concat)

    def forward_from_tokens(
        self,
        x_norm: Tensor,
        *,
        backbone: nn.Module,
    ) -> Dict[str, Optional[Tensor]]:
        """
        Compute scalar predictions from the full normalized token sequence (queries + base tokens).
        This enables using mean_query + mean_patch + cls together.
        """
        fused_concat = self.build_concat_features_from_tokens(x_norm, backbone=backbone)
        return self.forward_from_concat_features(fused_concat)

    def forward(
        self,
        x_base: Tensor,
        *,
        backbone: nn.Module,
        patch_hw: Tuple[int, int],
    ) -> Dict[str, Optional[Tensor]]:
        x_norm = self._forward_x_norm(x_base, backbone=backbone, patch_hw=patch_hw)
        return self.forward_from_tokens(x_norm, backbone=backbone)


__all__ = ["EoMTInjectedQueryHeadConfig", "EoMTInjectedQueryScalarHead"]


