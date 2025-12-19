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
    ) -> None:
        self.enabled: bool = bool(enabled)
        self.prob: float = float(prob)
        self.alpha: float = float(alpha)
        # When True (default), manifold mixup is applied to the full feature vector,
        # including the CLS token when present. When False (and use_cls_token=True),
        # we can keep CLS intact and mix only the patch features.
        self.mix_cls_token: bool = bool(mix_cls_token)
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
        if (not enabled) or prob <= 0.0:
            return None
        return ManifoldMixup(
            enabled=enabled, prob=prob, alpha=alpha, mix_cls_token=mix_cls_token
        )

    def apply(
        self, z: Tensor, batch: Dict[str, Tensor], *, force: bool = False
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
        lam = 1.0
        if self.alpha > 0.0:
            lam = float(
                torch.distributions.Beta(self.alpha, self.alpha).sample().item()
            )

        # Case 1: standard in-batch mixup for batch_size >= 2 (unchanged behavior).
        if bsz >= 2:
            perm = torch.randperm(bsz, device=z.device)
            z_mixed = lam * z + (1.0 - lam) * z[perm]

            # Mix main scalar regression targets (already in normalized space when applicable)
            for key in ("y_reg3", "y_height", "y_ndvi"):
                if key in batch:
                    y = batch[key]
                    if (
                        isinstance(y, torch.Tensor)
                        and y.dim() >= 1
                        and y.size(0) == bsz
                    ):
                        batch[key] = lam * y + (1.0 - lam) * y[perm]

            # Mix 5D biomass grams and recompute ratio targets from the mixed grams.
            if "y_biomass_5d_g" in batch:
                y_5d = batch["y_biomass_5d_g"]
                if (
                    isinstance(y_5d, torch.Tensor)
                    and y_5d.dim() == 2
                    and y_5d.size(0) == bsz
                    and y_5d.size(1) >= 5
                ):
                    y_5d_perm = y_5d[perm]
                    mixed_5d = lam * y_5d + (1.0 - lam) * y_5d_perm
                    batch["y_biomass_5d_g"] = mixed_5d

                    # Mix 5D masks conservatively (only keep supervision where both were valid)
                    mask_5d = batch.get("biomass_5d_mask", None)
                    if isinstance(mask_5d, torch.Tensor) and mask_5d.dim() == mixed_5d.dim():
                        mask_5d_perm = mask_5d[perm]
                        batch["biomass_5d_mask"] = mask_5d * mask_5d_perm

                    # Recompute ratio labels from mixed grams to keep physical consistency
                    mixed_total = mixed_5d[:, 4].clamp(min=1e-6)
                    if "y_ratio" in batch:
                        batch["y_ratio"] = torch.stack(
                            [
                                mixed_5d[:, 0] / mixed_total,
                                mixed_5d[:, 1] / mixed_total,
                                mixed_5d[:, 2] / mixed_total,
                            ],
                            dim=-1,
                        )
                    if "ratio_mask" in batch:
                        ratio_mask = batch["ratio_mask"]
                        if isinstance(ratio_mask, torch.Tensor) and ratio_mask.size(0) == bsz:
                            batch["ratio_mask"] = ratio_mask * ratio_mask[perm]

            return z_mixed, batch, True

        # Case 2: batch_size == 1. Use previous cached sample to perform mixing, similar to CutMix.
        # Cache current sample (detached) for potential use on the next step.
        current: Dict[str, Tensor] = {
            "z": z.detach().clone(),
        }
        for key in (
            "y_reg3",
            "y_height",
            "y_ndvi",
            "y_biomass_5d_g",
            "biomass_5d_mask",
            "y_ratio",
            "ratio_mask",
        ):
            val = batch.get(key, None)
            if isinstance(val, torch.Tensor):
                current[key] = val.detach().clone()

        # If no previous sample or incompatible shape, only update cache and skip mixing this time.
        if self._prev is None or "z" not in self._prev or self._prev["z"].shape != z.shape:
            self._prev = current
            return z, batch, False

        prev = self._prev
        z_prev = prev["z"].to(z.device, dtype=z.dtype)
        z_mixed = lam * z + (1.0 - lam) * z_prev

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
                    batch[key] = lam * y + (1.0 - lam) * y_prev

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
                mixed_5d = lam * y_5d + (1.0 - lam) * y_5d_prev
                batch["y_biomass_5d_g"] = mixed_5d

                mask_5d = batch.get("biomass_5d_mask", None)
                mask_5d_prev = prev.get("biomass_5d_mask", None)
                if (
                    isinstance(mask_5d, torch.Tensor)
                    and isinstance(mask_5d_prev, torch.Tensor)
                    and mask_5d.shape == mixed_5d.shape
                ):
                    batch["biomass_5d_mask"] = mask_5d * mask_5d_prev.to(
                        mask_5d.device, dtype=mask_5d.dtype
                    )

                # Recompute ratio labels from mixed grams to keep physical consistency
                mixed_total = mixed_5d[:, 4].clamp(min=1e-6)
                if "y_ratio" in batch:
                    batch["y_ratio"] = torch.stack(
                        [
                            mixed_5d[:, 0] / mixed_total,
                            mixed_5d[:, 1] / mixed_total,
                            mixed_5d[:, 2] / mixed_total,
                        ],
                        dim=-1,
                    )
                if "ratio_mask" in batch and "ratio_mask" in prev:
                    ratio_mask = batch["ratio_mask"]
                    ratio_mask_prev = prev["ratio_mask"].to(
                        ratio_mask.device, dtype=ratio_mask.dtype
                    )
                    if (
                        isinstance(ratio_mask, torch.Tensor)
                        and isinstance(ratio_mask_prev, torch.Tensor)
                        and ratio_mask_prev.shape == ratio_mask.shape
                    ):
                        batch["ratio_mask"] = ratio_mask * ratio_mask_prev

        # Update cache with current sample for the next step.
        self._prev = current
        return z_mixed, batch, True


