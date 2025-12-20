from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch


@dataclass
class CutMixConfig:
    enabled: bool = True
    prob: float = 1.0
    alpha: float = 1.0
    # Optional bbox area fraction clamp (min, max). If None, sample by lam only.
    minmax: Optional[Tuple[float, float]] = None
    # For batch_size == 1, use a cache of previous sample to mix.
    use_prev_for_bsz1: bool = True

    @staticmethod
    def from_dict(d: Optional[Dict[str, Any]]) -> "CutMixConfig":
        if d is None:
            return CutMixConfig(enabled=False)
        return CutMixConfig(
            enabled=bool(d.get("enabled", True)),
            prob=float(d.get("prob", 1.0)),
            alpha=float(d.get("alpha", 1.0)),
            minmax=tuple(d["minmax"]) if d.get("minmax", None) is not None else None,
            use_prev_for_bsz1=bool(d.get("use_prev_for_bsz1", True)),
        )


class CutMixBatchAugment:
    """
    Batch-level CutMix that supports:
      - Standard image + scalar targets (regression) via label interpolation with bbox area ratio.
      - Dense maps (e.g., NDVI) via patch replacement of both image and target maps.
      - batch_size == 1 by caching previous sample.
    """

    def __init__(self, cfg: CutMixConfig) -> None:
        self.cfg = cfg
        # Previous-sample caches for batch_size == 1
        self._prev_main: Optional[Dict[str, torch.Tensor]] = None
        self._prev_dense: Optional[Dict[str, torch.Tensor]] = None

    @staticmethod
    def from_cfg(d: Optional[Dict[str, Any]]) -> Optional["CutMixBatchAugment"]:
        cfg = CutMixConfig.from_dict(d)
        if not cfg.enabled:
            return None
        return CutMixBatchAugment(cfg)

    @staticmethod
    def _rand_bbox(h: int, w: int, lam: float, minmax: Optional[Tuple[float, float]] = None) -> Tuple[int, int, int, int, float]:
        """
        Sample a rectangular bbox for CutMix.
        Returns (yl, xl, yu, xu, lam_eff) where lam_eff is the effective lambda computed from the area.
        """
        if minmax is not None:
            min_ratio, max_ratio = float(minmax[0]), float(minmax[1])
            cut_ratio = float(torch.empty(()).uniform_(min_ratio, max_ratio).item())
            cut_w = int(w * cut_ratio)
            cut_h = int(h * cut_ratio)
        else:
            cut_ratio = math.sqrt(max(0.0, 1.0 - lam))
            cut_w = int(w * cut_ratio)
            cut_h = int(h * cut_ratio)
        # Uniform center
        cx = int(torch.randint(low=0, high=max(1, w), size=(1,)).item())
        cy = int(torch.randint(low=0, high=max(1, h), size=(1,)).item())
        xl = max(0, cx - cut_w // 2)
        yl = max(0, cy - cut_h // 2)
        xu = min(w, xl + cut_w)
        yu = min(h, yl + cut_h)
        bbx_w = max(0, xu - xl)
        bbx_h = max(0, yu - yl)
        box_area = float(bbx_w * bbx_h)
        total_area = float(h * w) if h > 0 and w > 0 else 1.0
        lam_eff = 1.0 - (box_area / total_area)
        return yl, xl, yu, xu, lam_eff

    @staticmethod
    def _mix_images_with_bbox(dst_imgs: torch.Tensor, src_imgs: torch.Tensor, bbox: Tuple[int, int, int, int]) -> None:
        yl, xl, yu, xu = bbox
        dst_imgs[..., yl:yu, xl:xu] = src_imgs[..., yl:yu, xl:xu]

    def _maybe_update_prev_main(self, batch: Dict[str, torch.Tensor]) -> None:
        self._prev_main = {
            "image": batch["image"].detach().clone(),
            "y_reg3": torch.nan_to_num(batch["y_reg3"], nan=0.0, posinf=0.0, neginf=0.0).detach().clone(),
            "y_height": torch.nan_to_num(batch["y_height"], nan=0.0, posinf=0.0, neginf=0.0).detach().clone(),
            "y_ndvi": torch.nan_to_num(batch["y_ndvi"], nan=0.0, posinf=0.0, neginf=0.0).detach().clone(),
            "reg3_mask": batch.get("reg3_mask", torch.empty(0)).detach().clone(),
            "ndvi_mask": batch.get("ndvi_mask", torch.empty(0)).detach().clone(),
            # Cache auxiliary targets for consistent mixing
            "y_biomass_5d_g": torch.nan_to_num(batch.get("y_biomass_5d_g", torch.empty(0)), nan=0.0, posinf=0.0, neginf=0.0).detach().clone(),
            "biomass_5d_mask": batch.get("biomass_5d_mask", torch.empty(0)).detach().clone(),
            "y_ratio": torch.nan_to_num(batch.get("y_ratio", torch.empty(0)), nan=0.0, posinf=0.0, neginf=0.0).detach().clone(),
            "ratio_mask": batch.get("ratio_mask", torch.empty(0)).detach().clone(),
        }

    def _maybe_update_prev_dense(self, batch: Dict[str, torch.Tensor]) -> None:
        self._prev_dense = {
            "image": batch["image"].detach().clone(),
            "ndvi_dense": batch["ndvi_dense"].detach().clone(),
            "ndvi_mask": batch.get("ndvi_mask", torch.ones_like(batch["ndvi_dense"], dtype=torch.bool)).detach().clone(),
        }

    def apply_main_batch(self, batch: Dict[str, torch.Tensor], *, force: bool = False) -> Tuple[Dict[str, torch.Tensor], bool]:
        """
        Apply CutMix to main batch dict with keys:
          - image: (B,C,H,W)
          - y_reg3: (B,3), y_height: (B,1), y_ndvi: (B,1)
        Classification targets, if any, are left unchanged.

        Returns:
            (batch, applied_flag)
        """
        applied = False
        if not self.cfg.enabled:
            return batch, applied
        if not force:
            if self.cfg.prob <= 0.0:
                return batch, applied
            if torch.rand(()) > self.cfg.prob:
                # Still update cache
                if self.cfg.use_prev_for_bsz1 and batch["image"].size(0) == 1:
                    self._maybe_update_prev_main(batch)
                return batch, applied

        images = batch["image"]
        bsz, _, h, w = images.shape
        if bsz <= 0:
            return batch, applied

        # Sample mix coefficient
        lam = 1.0
        if self.cfg.alpha > 0.0:
            # Beta(alpha, alpha)
            lam = float(torch.distributions.Beta(self.cfg.alpha, self.cfg.alpha).sample().item())
        yl, xl, yu, xu, lam_eff = self._rand_bbox(h, w, lam, self.cfg.minmax)

        if bsz >= 2:
            perm = torch.randperm(bsz, device=images.device)
            images_perm = images[perm]
            self._mix_images_with_bbox(images, images_perm, (yl, xl, yu, xu))  # in-place
            # Mix scalar regression targets by lam_eff
            y_reg3 = torch.nan_to_num(batch["y_reg3"], nan=0.0, posinf=0.0, neginf=0.0)
            y_height = torch.nan_to_num(batch["y_height"], nan=0.0, posinf=0.0, neginf=0.0)
            y_ndvi = torch.nan_to_num(batch["y_ndvi"], nan=0.0, posinf=0.0, neginf=0.0)
            batch["y_reg3"] = lam_eff * y_reg3 + (1.0 - lam_eff) * y_reg3[perm]
            batch["y_height"] = lam_eff * y_height + (1.0 - lam_eff) * y_height[perm]
            batch["y_ndvi"] = lam_eff * y_ndvi + (1.0 - lam_eff) * y_ndvi[perm]

            # AND supervision masks when present (keeps label mixing consistent with missing targets)
            if "reg3_mask" in batch:
                m = batch["reg3_mask"]
                if isinstance(m, torch.Tensor) and m.size(0) == bsz:
                    batch["reg3_mask"] = (m & m[perm]) if m.dtype == torch.bool else (m * m[perm])
            if "ndvi_mask" in batch:
                m = batch["ndvi_mask"]
                if isinstance(m, torch.Tensor) and m.size(0) == bsz:
                    batch["ndvi_mask"] = (m & m[perm]) if m.dtype == torch.bool else (m * m[perm])

            # Mix 5D biomass and update ratios
            if "y_biomass_5d_g" in batch:
                y_5d = batch["y_biomass_5d_g"]
                finite_5d = torch.isfinite(y_5d).to(dtype=y_5d.dtype)
                y_5d = torch.nan_to_num(y_5d, nan=0.0, posinf=0.0, neginf=0.0)
                mask_5d = batch.get("biomass_5d_mask", finite_5d).to(dtype=y_5d.dtype) * finite_5d

                y_5d_perm = y_5d[perm]
                mask_5d_perm = mask_5d[perm]

                # Linear mix of grams
                mixed_5d = lam_eff * y_5d + (1.0 - lam_eff) * y_5d_perm
                mixed_5d = torch.nan_to_num(mixed_5d, nan=0.0, posinf=0.0, neginf=0.0)
                batch["y_biomass_5d_g"] = mixed_5d
                mixed_mask_5d = mask_5d * mask_5d_perm
                batch["biomass_5d_mask"] = mixed_mask_5d

                # Re-calculate ratios from mixed 5D
                # 5D: [Clover, Dead, Green, GDM, Total]
                mixed_total = mixed_5d[:, 4].clamp(min=1e-6)

                # Ratio supervision is only valid when all 3 components and total are present and total > 0.
                ratio_mask_from_5d = (
                    (mixed_mask_5d[:, 0] > 0.0)
                    & (mixed_mask_5d[:, 1] > 0.0)
                    & (mixed_mask_5d[:, 2] > 0.0)
                    & (mixed_mask_5d[:, 4] > 0.0)
                    & (mixed_5d[:, 4] > 0.0)
                ).to(dtype=mixed_mask_5d.dtype).unsqueeze(-1)

                if "ratio_mask" in batch:
                    rm = batch["ratio_mask"]
                    if isinstance(rm, torch.Tensor) and rm.size(0) == bsz:
                        rm_mixed = (rm & rm[perm]) if rm.dtype == torch.bool else (rm * rm[perm])
                        if rm_mixed.dim() == 1:
                            rm_mixed = rm_mixed.view(bsz, 1)
                        batch["ratio_mask"] = rm_mixed.to(dtype=mixed_mask_5d.dtype) * ratio_mask_from_5d
                    else:
                        batch["ratio_mask"] = ratio_mask_from_5d
                else:
                    batch["ratio_mask"] = ratio_mask_from_5d

                if "y_ratio" in batch:
                    y_ratio_new = torch.stack([
                        mixed_5d[:, 0] / mixed_total,
                        mixed_5d[:, 1] / mixed_total,
                        mixed_5d[:, 2] / mixed_total,
                    ], dim=-1)
                    y_ratio_new = torch.nan_to_num(y_ratio_new, nan=0.0, posinf=0.0, neginf=0.0)
                    rm = batch.get("ratio_mask", None)
                    if isinstance(rm, torch.Tensor):
                        rm_f = rm.to(device=y_ratio_new.device, dtype=y_ratio_new.dtype)
                        if rm_f.dim() == 1:
                            rm_f = rm_f.view(bsz, 1)
                        y_ratio_new = torch.where(rm_f > 0.0, y_ratio_new, torch.zeros_like(y_ratio_new))
                    batch["y_ratio"] = y_ratio_new
            applied = True
        else:
            # bsz == 1
            if self.cfg.use_prev_for_bsz1 and self._prev_main is not None:
                prev = self._prev_main
                if (
                    isinstance(prev, dict)
                    and prev["image"].shape[-2:] == images.shape[-2:]
                    and prev["image"].shape[1] == images.shape[1]
                ):
                    self._mix_images_with_bbox(images, prev["image"].to(images.device, dtype=images.dtype), (yl, xl, yu, xu))
                    # Mix scalar targets with cache
                    y_reg3 = torch.nan_to_num(batch["y_reg3"], nan=0.0, posinf=0.0, neginf=0.0)
                    y_height = torch.nan_to_num(batch["y_height"], nan=0.0, posinf=0.0, neginf=0.0)
                    y_ndvi = torch.nan_to_num(batch["y_ndvi"], nan=0.0, posinf=0.0, neginf=0.0)
                    batch["y_reg3"] = lam_eff * y_reg3 + (1.0 - lam_eff) * prev["y_reg3"].to(batch["y_reg3"].device, dtype=batch["y_reg3"].dtype)
                    batch["y_height"] = lam_eff * y_height + (1.0 - lam_eff) * prev["y_height"].to(batch["y_height"].device, dtype=batch["y_height"].dtype)
                    batch["y_ndvi"] = lam_eff * y_ndvi + (1.0 - lam_eff) * prev["y_ndvi"].to(batch["y_ndvi"].device, dtype=batch["y_ndvi"].dtype)

                    # AND supervision masks when present
                    if "reg3_mask" in batch and "reg3_mask" in prev and prev["reg3_mask"].numel() > 0:
                        m = batch["reg3_mask"]
                        pm = prev["reg3_mask"].to(m.device)
                        if pm.shape == m.shape:
                            batch["reg3_mask"] = (m.to(torch.bool) & pm.to(torch.bool)) if (m.dtype == torch.bool or pm.dtype == torch.bool) else (m * pm.to(dtype=m.dtype))
                    if "ndvi_mask" in batch and "ndvi_mask" in prev and prev["ndvi_mask"].numel() > 0:
                        m = batch["ndvi_mask"]
                        pm = prev["ndvi_mask"].to(m.device)
                        if pm.shape == m.shape:
                            batch["ndvi_mask"] = (m.to(torch.bool) & pm.to(torch.bool)) if (m.dtype == torch.bool or pm.dtype == torch.bool) else (m * pm.to(dtype=m.dtype))

                    # Mix 5D biomass and update ratios
                    if "y_biomass_5d_g" in batch and "y_biomass_5d_g" in prev and prev["y_biomass_5d_g"].numel() > 0:
                        y_5d = batch["y_biomass_5d_g"]
                        finite_5d = torch.isfinite(y_5d).to(dtype=y_5d.dtype)
                        y_5d = torch.nan_to_num(y_5d, nan=0.0, posinf=0.0, neginf=0.0)
                        prev_y_5d = torch.nan_to_num(prev["y_biomass_5d_g"].to(y_5d.device, dtype=y_5d.dtype), nan=0.0, posinf=0.0, neginf=0.0)

                        mask_5d = batch.get("biomass_5d_mask", finite_5d).to(dtype=y_5d.dtype) * finite_5d
                        prev_mask_5d = prev["biomass_5d_mask"].to(mask_5d.device, dtype=mask_5d.dtype)
                        if prev_mask_5d.numel() == 0:
                            prev_mask_5d = torch.ones_like(prev_y_5d)

                        mixed_5d = lam_eff * y_5d + (1.0 - lam_eff) * prev_y_5d
                        mixed_5d = torch.nan_to_num(mixed_5d, nan=0.0, posinf=0.0, neginf=0.0)
                        batch["y_biomass_5d_g"] = mixed_5d
                        mixed_mask_5d = mask_5d * prev_mask_5d
                        batch["biomass_5d_mask"] = mixed_mask_5d

                        mixed_total = mixed_5d[:, 4].clamp(min=1e-6)

                        ratio_mask_from_5d = (
                            (mixed_mask_5d[:, 0] > 0.0)
                            & (mixed_mask_5d[:, 1] > 0.0)
                            & (mixed_mask_5d[:, 2] > 0.0)
                            & (mixed_mask_5d[:, 4] > 0.0)
                            & (mixed_5d[:, 4] > 0.0)
                        ).to(dtype=mixed_mask_5d.dtype).unsqueeze(-1)

                        if "y_ratio" in batch:
                            y_ratio_new = torch.stack([
                                mixed_5d[:, 0] / mixed_total,
                                mixed_5d[:, 1] / mixed_total,
                                mixed_5d[:, 2] / mixed_total,
                            ], dim=-1)
                            y_ratio_new = torch.nan_to_num(y_ratio_new, nan=0.0, posinf=0.0, neginf=0.0)

                            rm = batch.get("ratio_mask", None)
                            pm = prev.get("ratio_mask", None)
                            if isinstance(rm, torch.Tensor) and isinstance(pm, torch.Tensor) and pm.numel() > 0 and pm.shape == rm.shape:
                                rm_mixed = (rm.to(torch.bool) & pm.to(torch.bool).to(rm.device)) if (rm.dtype == torch.bool or pm.dtype == torch.bool) else (rm * pm.to(rm.device, dtype=rm.dtype))
                                if rm_mixed.dim() == 1:
                                    rm_mixed = rm_mixed.view(bsz, 1)
                                batch["ratio_mask"] = rm_mixed.to(dtype=mixed_mask_5d.dtype) * ratio_mask_from_5d
                            else:
                                batch["ratio_mask"] = ratio_mask_from_5d
                            rm_f = batch.get("ratio_mask", None)
                            if isinstance(rm_f, torch.Tensor):
                                rm_f2 = rm_f.to(device=y_ratio_new.device, dtype=y_ratio_new.dtype)
                                if rm_f2.dim() == 1:
                                    rm_f2 = rm_f2.view(bsz, 1)
                                y_ratio_new = torch.where(rm_f2 > 0.0, y_ratio_new, torch.zeros_like(y_ratio_new))
                            batch["y_ratio"] = y_ratio_new
                        applied = True

            # Update cache for next time
            if self.cfg.use_prev_for_bsz1:
                self._maybe_update_prev_main(batch)

        return batch, applied

    def apply_ndvi_dense_batch(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Apply CutMix to NDVI-dense batch dict with keys:
          - image: (B,C,H,W)
          - ndvi_dense: (B,1,H,W)
          - ndvi_mask: (B,1,H,W) [bool]
        We replace the bbox region for both image and target maps; loss is computed against the mixed target map.
        """
        if not self.cfg.enabled or self.cfg.prob <= 0.0:
            return batch
        if torch.rand(()) > self.cfg.prob:
            if self.cfg.use_prev_for_bsz1 and batch["image"].size(0) == 1:
                self._maybe_update_prev_dense(batch)
            return batch

        images = batch["image"]
        labels = batch["ndvi_dense"]
        mask = batch.get("ndvi_mask", torch.ones_like(labels, dtype=torch.bool))
        bsz, _, h, w = images.shape
        if bsz <= 0:
            return batch

        # Sample mix coefficient
        lam = 1.0
        if self.cfg.alpha > 0.0:
            lam = float(torch.distributions.Beta(self.cfg.alpha, self.cfg.alpha).sample().item())
        yl, xl, yu, xu, _ = self._rand_bbox(h, w, lam, self.cfg.minmax)

        if bsz >= 2:
            perm = torch.randperm(bsz, device=images.device)
            images_perm = images[perm]
            labels_perm = labels[perm]
            mask_perm = mask[perm]
            self._mix_images_with_bbox(images, images_perm, (yl, xl, yu, xu))
            self._mix_images_with_bbox(labels, labels_perm, (yl, xl, yu, xu))
            self._mix_images_with_bbox(mask, mask_perm, (yl, xl, yu, xu))
            batch["ndvi_mask"] = mask.to(torch.bool)
        else:
            # bsz == 1
            if self.cfg.use_prev_for_bsz1 and self._prev_dense is not None:
                prev = self._prev_dense
                if (
                    isinstance(prev, dict)
                    and prev["image"].shape[-2:] == images.shape[-2:]
                    and prev["image"].shape[1] == images.shape[1]
                ):
                    self._mix_images_with_bbox(images, prev["image"].to(images.device, dtype=images.dtype), (yl, xl, yu, xu))
                    self._mix_images_with_bbox(labels, prev["ndvi_dense"].to(labels.device, dtype=labels.dtype), (yl, xl, yu, xu))
                    self._mix_images_with_bbox(mask, prev["ndvi_mask"].to(mask.device, dtype=mask.dtype), (yl, xl, yu, xu))
                    batch["ndvi_mask"] = mask.to(torch.bool)
            if self.cfg.use_prev_for_bsz1:
                self._maybe_update_prev_dense(batch)

        return batch


