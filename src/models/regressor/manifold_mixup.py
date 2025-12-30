from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch
from torch import Tensor


class ManifoldMixup:
    """
    Feature-level ("manifold") mixup applied on the shared bottleneck representation z,
    together with consistent mixing of regression and 5D/ratio targets.
    """

    def __init__(
        self,
        *,
        enabled: bool,
        prob: float,
        alpha: float,
        mix_cls_token: bool = True,
        # When True, stop gradients through the "paired" sample in in-batch mixup (bsz>=2).
        # This makes bsz>=2 closer to the bsz==1 cache-mode behavior where the cached sample
        # is detached (i.e., acts like a feature/tokens perturbation rather than a coupled update).
        detach_pair: bool = False,
    ) -> None:
        self.enabled: bool = bool(enabled)
        self.prob: float = float(prob)
        self.alpha: float = float(alpha)
        # When True (default), manifold mixup is applied to the full feature vector,
        # including the CLS token when present. When False (and use_cls_token=True),
        # we can keep CLS intact and mix only the patch features.
        self.mix_cls_token: bool = bool(mix_cls_token)
        self.detach_pair: bool = bool(detach_pair)
        # Previous-sample cache to support batch_size == 1 manifold mixup
        self._prev: Optional[Dict[str, Tensor]] = None

    @staticmethod
    def from_cfg(cfg: Optional[Dict[str, Any]]) -> Optional["ManifoldMixup"]:
        if cfg is None:
            return None
        enabled = bool(cfg.get("enabled", False))
        prob = float(cfg.get("prob", 0.0))
        alpha = float(cfg.get("alpha", 1.0))
        mix_cls_token = bool(cfg.get("mix_cls_token", True))
        detach_pair = bool(cfg.get("detach_pair", False))
        if (not enabled) or prob <= 0.0:
            return None
        return ManifoldMixup(
            enabled=enabled,
            prob=prob,
            alpha=alpha,
            mix_cls_token=mix_cls_token,
            detach_pair=detach_pair,
        )

    def sample_params(self, *, bsz: int, device: torch.device) -> Tuple[float, Tensor]:
        """
        Sample a mixup coefficient and permutation for view-consistent mixup.

        Returns:
            lam: float in [0, 1]
            perm: LongTensor of shape (bsz,)
        """
        b = int(bsz)
        if b < 2:
            return 1.0, torch.arange(b, device=device, dtype=torch.long)
        lam = 1.0
        if self.alpha > 0.0:
            try:
                lam = float(torch.distributions.Beta(self.alpha, self.alpha).sample().item())
            except Exception:
                lam = 1.0
        if not (0.0 <= lam <= 1.0):
            lam = 1.0 if lam >= 1.0 else 0.0
        perm = torch.randperm(b, device=device)
        return lam, perm

    def apply(
        self,
        z: Tensor,
        batch: Dict[str, Tensor],
        *,
        force: bool = False,
        lam: Optional[float] = None,
        perm: Optional[Tensor] = None,
        mix_labels: bool = True,
    ) -> Tuple[Tensor, Dict[str, Tensor], bool]:
        """
        Apply manifold mixup on feature tensor z and relevant regression/5D/ratio targets.

        Returns:
            z_mixed, batch_mixed, applied_flag
        """
        if not self.enabled:
            return z, batch, False
        if z.dim() < 2 or z.size(0) <= 0:
            return z, batch, False
        if not force:
            if self.prob <= 0.0:
                return z, batch, False
            if torch.rand(()) > self.prob:
                return z, batch, False

        bsz = z.size(0)
        lam_eff = 1.0
        if lam is not None:
            try:
                lam_eff = float(lam)
            except Exception:
                lam_eff = 1.0
        elif self.alpha > 0.0:
            lam_eff = float(torch.distributions.Beta(self.alpha, self.alpha).sample().item())
        # Numeric safety
        if not (0.0 <= lam_eff <= 1.0):
            lam_eff = 1.0 if lam_eff >= 1.0 else 0.0

        # Case 1: standard in-batch mixup for batch_size >= 2 (unchanged behavior).
        if bsz >= 2:
            if perm is None:
                perm_eff = torch.randperm(bsz, device=z.device)
            else:
                try:
                    perm_eff = perm.to(device=z.device)
                except Exception:
                    perm_eff = torch.randperm(bsz, device=z.device)
                if perm_eff.dim() != 1 or perm_eff.numel() != bsz:
                    perm_eff = torch.randperm(bsz, device=z.device)

            z_other = z[perm_eff]
            if bool(getattr(self, "detach_pair", False)):
                z_other = z_other.detach()
            z_mixed = lam_eff * z + (1.0 - lam_eff) * z_other

            # View-consistent mode: allow mixing features/tokens without mixing labels.
            if not bool(mix_labels):
                return z_mixed, batch, True

            # Mix main scalar regression targets (already in normalized space when applicable)
            for key in ("y_reg3", "y_height", "y_ndvi"):
                if key in batch:
                    y = batch[key]
                    if (
                        isinstance(y, torch.Tensor)
                        and y.dim() >= 1
                        and y.size(0) == bsz
                    ):
                        y_safe = torch.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
                        y_perm = torch.nan_to_num(
                            y_safe[perm_eff], nan=0.0, posinf=0.0, neginf=0.0
                        )
                        batch[key] = lam_eff * y_safe + (1.0 - lam_eff) * y_perm

            # Mix/AND supervision masks when present (keeps label mixing consistent with missing targets)
            for key in ("reg3_mask", "ndvi_mask"):
                if key in batch:
                    m = batch[key]
                    if (
                        isinstance(m, torch.Tensor)
                        and m.dim() >= 1
                        and m.size(0) == bsz
                    ):
                        if m.dtype == torch.bool:
                            batch[key] = m & m[perm_eff]
                        else:
                            batch[key] = m * m[perm_eff]

            # Mix 5D biomass grams and recompute ratio targets from the mixed grams.
            if "y_biomass_5d_g" in batch:
                y_5d = batch["y_biomass_5d_g"]
                if (
                    isinstance(y_5d, torch.Tensor)
                    and y_5d.dim() == 2
                    and y_5d.size(0) == bsz
                    and y_5d.size(1) >= 5
                ):
                    y_5d_safe = torch.nan_to_num(
                        y_5d, nan=0.0, posinf=0.0, neginf=0.0
                    )
                    y_5d_perm = torch.nan_to_num(
                        y_5d_safe[perm_eff], nan=0.0, posinf=0.0, neginf=0.0
                    )
                    mixed_5d = lam_eff * y_5d_safe + (1.0 - lam_eff) * y_5d_perm
                    mixed_5d = torch.nan_to_num(
                        mixed_5d, nan=0.0, posinf=0.0, neginf=0.0
                    )
                    batch["y_biomass_5d_g"] = mixed_5d

                    # Mix 5D masks conservatively (only keep supervision where both were valid)
                    mask_5d = batch.get("biomass_5d_mask", None)
                    if (
                        isinstance(mask_5d, torch.Tensor)
                        and mask_5d.shape == mixed_5d.shape
                        and mask_5d.size(0) == bsz
                    ):
                        mask_5d_safe = mask_5d.to(
                            device=mixed_5d.device, dtype=mixed_5d.dtype
                        )
                    else:
                        # Fall back to finite-mask if not provided / incompatible.
                        mask_5d_safe = torch.isfinite(y_5d_safe).to(
                            device=mixed_5d.device, dtype=mixed_5d.dtype
                        )
                    mask_5d_perm = mask_5d_safe[perm_eff]
                    mixed_mask_5d = mask_5d_safe * mask_5d_perm
                    batch["biomass_5d_mask"] = mixed_mask_5d

                    # --- Ratio supervision mixing ---
                    # Preferred path (CSIRO): recompute ratios from mixed 5D grams when available.
                    # Fallback path (e.g., Irish): mix provided y_ratio directly when no 5D supervision exists.
                    mixed_total = mixed_5d[:, 4].clamp(min=1e-6)
                    ratio_mask_from_5d = (
                        (mixed_mask_5d[:, 0] > 0.0)
                        & (mixed_mask_5d[:, 1] > 0.0)
                        & (mixed_mask_5d[:, 2] > 0.0)
                        & (mixed_mask_5d[:, 4] > 0.0)
                        & (mixed_5d[:, 4] > 0.0)
                    ).to(dtype=mixed_mask_5d.dtype).unsqueeze(-1)

                    # If we have any 5D supervision after mixing, we gate ratio supervision by 5D completeness.
                    has_any_5d = (mixed_mask_5d.sum(dim=-1, keepdim=True) > 0.0).to(
                        dtype=mixed_mask_5d.dtype
                    )

                    y_ratio_existing = batch.get("y_ratio", None)
                    ratio_mask_existing = batch.get("ratio_mask", None)

                    y_ratio_mixed: Optional[Tensor] = None
                    ratio_mask_mixed: Optional[Tensor] = None
                    if (
                        isinstance(y_ratio_existing, torch.Tensor)
                        and y_ratio_existing.dim() == 2
                        and y_ratio_existing.size(0) == bsz
                        and y_ratio_existing.size(1) >= 3
                    ):
                        y_ratio_safe = torch.nan_to_num(
                            y_ratio_existing, nan=0.0, posinf=0.0, neginf=0.0
                        )
                        y_ratio_perm = torch.nan_to_num(
                            y_ratio_safe[perm_eff], nan=0.0, posinf=0.0, neginf=0.0
                        )
                        y_ratio_mixed = lam_eff * y_ratio_safe + (1.0 - lam_eff) * y_ratio_perm

                    if (
                        isinstance(ratio_mask_existing, torch.Tensor)
                        and ratio_mask_existing.dim() >= 1
                        and ratio_mask_existing.size(0) == bsz
                    ):
                        if ratio_mask_existing.dtype == torch.bool:
                            ratio_mask_mixed = (
                                ratio_mask_existing & ratio_mask_existing[perm_eff]
                            ).to(dtype=mixed_mask_5d.dtype)
                        else:
                            ratio_mask_mixed = ratio_mask_existing.to(
                                dtype=mixed_mask_5d.dtype
                            ) * ratio_mask_existing[perm_eff].to(dtype=mixed_mask_5d.dtype)
                        if ratio_mask_mixed.dim() == 1:
                            ratio_mask_mixed = ratio_mask_mixed.view(bsz, 1)

                    # Final ratio mask: gate by 5D completeness ONLY when any 5D supervision exists.
                    if ratio_mask_mixed is None:
                        ratio_mask_out = ratio_mask_from_5d
                    else:
                        ratio_mask_out = torch.where(
                            has_any_5d > 0.0,
                            ratio_mask_mixed * ratio_mask_from_5d,
                            ratio_mask_mixed,
                        )
                    batch["ratio_mask"] = ratio_mask_out

                    if isinstance(y_ratio_existing, torch.Tensor):
                        # Ratio targets derived from mixed 5D grams (CSIRO path)
                        y_ratio_from_5d = torch.stack(
                            [
                                mixed_5d[:, 0] / mixed_total,
                                mixed_5d[:, 1] / mixed_total,
                                mixed_5d[:, 2] / mixed_total,
                            ],
                            dim=-1,
                        )
                        y_ratio_from_5d = torch.nan_to_num(
                            y_ratio_from_5d, nan=0.0, posinf=0.0, neginf=0.0
                        )

                        if y_ratio_mixed is None:
                            y_ratio_out = y_ratio_from_5d
                        else:
                            use_from_5d = (has_any_5d > 0.0) & (ratio_mask_from_5d > 0.0)
                            y_ratio_out = torch.where(use_from_5d, y_ratio_from_5d, y_ratio_mixed)

                        # Zero-out when ratio supervision is disabled.
                        rm_f = ratio_mask_out.to(device=y_ratio_out.device, dtype=y_ratio_out.dtype)
                        if rm_f.dim() == 1:
                            rm_f = rm_f.view(bsz, 1)
                        y_ratio_out = torch.where(rm_f > 0.0, y_ratio_out, torch.zeros_like(y_ratio_out))
                        batch["y_ratio"] = y_ratio_out

            return z_mixed, batch, True

        # Case 2: batch_size == 1. Use previous cached sample to perform mixing, similar to CutMix.
        # Cache current sample (detached) for potential use on the next step.
        current: Dict[str, Tensor] = {
            "z": z.detach().clone(),
        }
        for key in (
            "y_reg3",
            "reg3_mask",
            "y_height",
            "y_ndvi",
            "ndvi_mask",
            "y_biomass_5d_g",
            "biomass_5d_mask",
            "y_ratio",
            "ratio_mask",
        ):
            val = batch.get(key, None)
            if isinstance(val, torch.Tensor):
                # Avoid caching NaNs/infs for numeric targets (masks can stay as-is).
                if key in ("y_reg3", "y_height", "y_ndvi", "y_biomass_5d_g", "y_ratio"):
                    current[key] = torch.nan_to_num(
                        val, nan=0.0, posinf=0.0, neginf=0.0
                    ).detach().clone()
                else:
                    current[key] = val.detach().clone()

        # If no previous sample or incompatible shape, only update cache and skip mixing this time.
        if self._prev is None or "z" not in self._prev or self._prev["z"].shape != z.shape:
            self._prev = current
            return z, batch, False

        prev = self._prev
        z_prev = prev["z"].to(z.device, dtype=z.dtype)
        z_mixed = lam_eff * z + (1.0 - lam_eff) * z_prev

        # View-consistent mode: allow mixing features without mixing labels.
        if not bool(mix_labels):
            self._prev = current
            return z_mixed, batch, True

        # Mix main scalar regression targets with the cached sample.
        for key in ("y_reg3", "y_height", "y_ndvi"):
            if key in batch and key in prev:
                y = batch[key]
                y_prev = prev[key].to(y.device, dtype=y.dtype)
                if (
                    isinstance(y, torch.Tensor)
                    and isinstance(y_prev, torch.Tensor)
                    and y.dim() >= 1
                    and y_prev.shape == y.shape
                ):
                    y_safe = torch.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
                    y_prev_safe = torch.nan_to_num(
                        y_prev, nan=0.0, posinf=0.0, neginf=0.0
                    )
                    batch[key] = lam_eff * y_safe + (1.0 - lam_eff) * y_prev_safe

        # Mix/AND supervision masks when present.
        for key in ("reg3_mask", "ndvi_mask"):
            if key in batch and key in prev:
                m = batch[key]
                m_prev = prev[key].to(m.device)
                if (
                    isinstance(m, torch.Tensor)
                    and isinstance(m_prev, torch.Tensor)
                    and m.dim() >= 1
                    and m_prev.shape == m.shape
                ):
                    if m.dtype == torch.bool or m_prev.dtype == torch.bool:
                        batch[key] = m.to(torch.bool) & m_prev.to(torch.bool)
                    else:
                        batch[key] = m * m_prev.to(dtype=m.dtype)

        # Mix 5D biomass grams and recompute ratio targets using the cached sample.
        if "y_biomass_5d_g" in batch and "y_biomass_5d_g" in prev:
            y_5d = batch["y_biomass_5d_g"]
            y_5d_prev = prev["y_biomass_5d_g"].to(y_5d.device, dtype=y_5d.dtype)
            if (
                isinstance(y_5d, torch.Tensor)
                and isinstance(y_5d_prev, torch.Tensor)
                and y_5d.dim() == 2
                and y_5d_prev.shape == y_5d.shape
                and y_5d.size(1) >= 5
            ):
                y_5d_safe = torch.nan_to_num(
                    y_5d, nan=0.0, posinf=0.0, neginf=0.0
                )
                y_5d_prev_safe = torch.nan_to_num(
                    y_5d_prev, nan=0.0, posinf=0.0, neginf=0.0
                )
                mixed_5d = lam_eff * y_5d_safe + (1.0 - lam_eff) * y_5d_prev_safe
                mixed_5d = torch.nan_to_num(
                    mixed_5d, nan=0.0, posinf=0.0, neginf=0.0
                )
                batch["y_biomass_5d_g"] = mixed_5d

                mask_5d = batch.get("biomass_5d_mask", None)
                mask_5d_prev = prev.get("biomass_5d_mask", None)
                if (
                    isinstance(mask_5d, torch.Tensor)
                    and mask_5d.shape == mixed_5d.shape
                    and mask_5d.size(0) == bsz
                ):
                    mask_5d_safe = mask_5d.to(
                        device=mixed_5d.device, dtype=mixed_5d.dtype
                    )
                else:
                    mask_5d_safe = torch.isfinite(y_5d_safe).to(
                        device=mixed_5d.device, dtype=mixed_5d.dtype
                    )

                if (
                    isinstance(mask_5d_prev, torch.Tensor)
                    and mask_5d_prev.shape == mixed_5d.shape
                ):
                    mask_5d_prev_safe = mask_5d_prev.to(
                        device=mixed_5d.device, dtype=mixed_5d.dtype
                    )
                else:
                    mask_5d_prev_safe = torch.isfinite(y_5d_prev_safe).to(
                        device=mixed_5d.device, dtype=mixed_5d.dtype
                    )

                mixed_mask_5d = mask_5d_safe * mask_5d_prev_safe
                batch["biomass_5d_mask"] = mixed_mask_5d

                # --- Ratio supervision mixing (batch_size == 1 cache mode) ---
                mixed_total = mixed_5d[:, 4].clamp(min=1e-6)
                ratio_mask_from_5d = (
                    (mixed_mask_5d[:, 0] > 0.0)
                    & (mixed_mask_5d[:, 1] > 0.0)
                    & (mixed_mask_5d[:, 2] > 0.0)
                    & (mixed_mask_5d[:, 4] > 0.0)
                    & (mixed_5d[:, 4] > 0.0)
                ).to(dtype=mixed_mask_5d.dtype).unsqueeze(-1)

                has_any_5d = (mixed_mask_5d.sum(dim=-1, keepdim=True) > 0.0).to(
                    dtype=mixed_mask_5d.dtype
                )

                y_ratio_existing = batch.get("y_ratio", None)
                y_ratio_prev = prev.get("y_ratio", None)
                ratio_mask_existing = batch.get("ratio_mask", None)
                ratio_mask_prev = prev.get("ratio_mask", None)

                y_ratio_mixed: Optional[Tensor] = None
                ratio_mask_mixed: Optional[Tensor] = None

                if (
                    isinstance(y_ratio_existing, torch.Tensor)
                    and isinstance(y_ratio_prev, torch.Tensor)
                    and y_ratio_prev.shape == y_ratio_existing.shape
                    and y_ratio_existing.dim() == 2
                    and y_ratio_existing.size(0) == bsz
                    and y_ratio_existing.size(1) >= 3
                ):
                    y_ratio_safe = torch.nan_to_num(
                        y_ratio_existing, nan=0.0, posinf=0.0, neginf=0.0
                    )
                    y_ratio_prev_safe = torch.nan_to_num(
                        y_ratio_prev.to(y_ratio_safe.device, dtype=y_ratio_safe.dtype),
                        nan=0.0,
                        posinf=0.0,
                        neginf=0.0,
                    )
                    y_ratio_mixed = lam_eff * y_ratio_safe + (1.0 - lam_eff) * y_ratio_prev_safe

                if (
                    isinstance(ratio_mask_existing, torch.Tensor)
                    and isinstance(ratio_mask_prev, torch.Tensor)
                    and ratio_mask_prev.shape == ratio_mask_existing.shape
                    and ratio_mask_existing.dim() >= 1
                    and ratio_mask_existing.size(0) == bsz
                ):
                    rm = ratio_mask_existing
                    rm_prev = ratio_mask_prev.to(rm.device)
                    if rm.dtype == torch.bool or rm_prev.dtype == torch.bool:
                        ratio_mask_mixed = (rm.to(torch.bool) & rm_prev.to(torch.bool)).to(
                            dtype=mixed_mask_5d.dtype
                        )
                    else:
                        ratio_mask_mixed = rm.to(dtype=mixed_mask_5d.dtype) * rm_prev.to(
                            dtype=mixed_mask_5d.dtype
                        )
                    if ratio_mask_mixed.dim() == 1:
                        ratio_mask_mixed = ratio_mask_mixed.view(bsz, 1)

                if ratio_mask_mixed is None:
                    ratio_mask_out = ratio_mask_from_5d
                else:
                    ratio_mask_out = torch.where(
                        has_any_5d > 0.0,
                        ratio_mask_mixed * ratio_mask_from_5d,
                        ratio_mask_mixed,
                    )
                batch["ratio_mask"] = ratio_mask_out

                if isinstance(y_ratio_existing, torch.Tensor):
                    y_ratio_from_5d = torch.stack(
                        [
                            mixed_5d[:, 0] / mixed_total,
                            mixed_5d[:, 1] / mixed_total,
                            mixed_5d[:, 2] / mixed_total,
                        ],
                        dim=-1,
                    )
                    y_ratio_from_5d = torch.nan_to_num(
                        y_ratio_from_5d, nan=0.0, posinf=0.0, neginf=0.0
                    )

                    if y_ratio_mixed is None:
                        y_ratio_out = y_ratio_from_5d
                    else:
                        use_from_5d = (has_any_5d > 0.0) & (ratio_mask_from_5d > 0.0)
                        y_ratio_out = torch.where(use_from_5d, y_ratio_from_5d, y_ratio_mixed)

                    rm_f = ratio_mask_out.to(device=y_ratio_out.device, dtype=y_ratio_out.dtype)
                    if rm_f.dim() == 1:
                        rm_f = rm_f.view(bsz, 1)
                    y_ratio_out = torch.where(rm_f > 0.0, y_ratio_out, torch.zeros_like(y_ratio_out))
                    batch["y_ratio"] = y_ratio_out

        # Update cache with current sample for the next step.
        self._prev = current
        return z_mixed, batch, True


