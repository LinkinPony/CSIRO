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
            # Optional debug/traceability: keep the permutation indices so downstream
            # tooling (e.g., input image dumping) can associate mixed samples with both
            # source IDs. This is safe to ignore in training logic.
            try:
                batch["_cutmix_perm"] = perm.detach().cpu()
                batch["_cutmix_lam"] = float(lam_eff)
                batch["_cutmix_bbox"] = (int(yl), int(xl), int(yu), int(xu))
            except Exception:
                pass
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

                has_any_5d = (mixed_mask_5d.sum(dim=-1, keepdim=True) > 0.0).to(
                    dtype=mixed_mask_5d.dtype
                )

                y_ratio_existing = batch.get("y_ratio", None)
                ratio_mask_existing = batch.get("ratio_mask", None)

                y_ratio_mixed = None
                ratio_mask_mixed = None
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
                        y_ratio_safe[perm], nan=0.0, posinf=0.0, neginf=0.0
                    )
                    y_ratio_mixed = lam_eff * y_ratio_safe + (1.0 - lam_eff) * y_ratio_perm

                if (
                    isinstance(ratio_mask_existing, torch.Tensor)
                    and ratio_mask_existing.size(0) == bsz
                    and ratio_mask_existing.dim() >= 1
                ):
                    if ratio_mask_existing.dtype == torch.bool:
                        ratio_mask_mixed = (ratio_mask_existing & ratio_mask_existing[perm]).to(
                            dtype=mixed_mask_5d.dtype
                        )
                    else:
                        ratio_mask_mixed = ratio_mask_existing.to(
                            dtype=mixed_mask_5d.dtype
                        ) * ratio_mask_existing[perm].to(dtype=mixed_mask_5d.dtype)
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

                        y_ratio_mixed = None
                        ratio_mask_mixed = None
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
                            pm = ratio_mask_prev.to(rm.device)
                            if rm.dtype == torch.bool or pm.dtype == torch.bool:
                                ratio_mask_mixed = (rm.to(torch.bool) & pm.to(torch.bool)).to(
                                    dtype=mixed_mask_5d.dtype
                                )
                            else:
                                ratio_mask_mixed = rm.to(dtype=mixed_mask_5d.dtype) * pm.to(
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
                            y_ratio_out = torch.where(
                                rm_f > 0.0, y_ratio_out, torch.zeros_like(y_ratio_out)
                            )
                            batch["y_ratio"] = y_ratio_out
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


