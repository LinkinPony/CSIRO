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
            "y_reg3": batch["y_reg3"].detach().clone(),
            "y_height": batch["y_height"].detach().clone(),
            "y_ndvi": batch["y_ndvi"].detach().clone(),
        }

    def _maybe_update_prev_dense(self, batch: Dict[str, torch.Tensor]) -> None:
        self._prev_dense = {
            "image": batch["image"].detach().clone(),
            "ndvi_dense": batch["ndvi_dense"].detach().clone(),
            "ndvi_mask": batch.get("ndvi_mask", torch.ones_like(batch["ndvi_dense"], dtype=torch.bool)).detach().clone(),
        }

    def apply_main_batch(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Apply CutMix to main batch dict with keys:
          - image: (B,C,H,W)
          - y_reg3: (B,3), y_height: (B,1), y_ndvi: (B,1)
        Classification targets, if any, are left unchanged.
        """
        if not self.cfg.enabled or self.cfg.prob <= 0.0:
            return batch
        if torch.rand(()) > self.cfg.prob:
            # Still update cache
            if self.cfg.use_prev_for_bsz1 and batch["image"].size(0) == 1:
                self._maybe_update_prev_main(batch)
            return batch

        images = batch["image"]
        bsz, _, h, w = images.shape
        if bsz <= 0:
            return batch

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
            batch["y_reg3"] = lam_eff * batch["y_reg3"] + (1.0 - lam_eff) * batch["y_reg3"][perm]
            batch["y_height"] = lam_eff * batch["y_height"] + (1.0 - lam_eff) * batch["y_height"][perm]
            batch["y_ndvi"] = lam_eff * batch["y_ndvi"] + (1.0 - lam_eff) * batch["y_ndvi"][perm]
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
                    batch["y_reg3"] = lam_eff * batch["y_reg3"] + (1.0 - lam_eff) * prev["y_reg3"].to(batch["y_reg3"].device, dtype=batch["y_reg3"].dtype)
                    batch["y_height"] = lam_eff * batch["y_height"] + (1.0 - lam_eff) * prev["y_height"].to(batch["y_height"].device, dtype=batch["y_height"].dtype)
                    batch["y_ndvi"] = lam_eff * batch["y_ndvi"] + (1.0 - lam_eff) * prev["y_ndvi"].to(batch["y_ndvi"].device, dtype=batch["y_ndvi"].dtype)
            # Update cache for next time
            if self.cfg.use_prev_for_bsz1:
                self._maybe_update_prev_main(batch)

        return batch

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


@dataclass
class CMixupConfig:
    """
    Configuration for C-Mixup (label-distribution aware mixup for regression),
    adapted from the official C-Mixup implementation.
    """

    enabled: bool = False
    # Per-batch probability to apply C-Mixup
    prob: float = 1.0
    # Beta(alpha, alpha) distribution parameter for mix coefficient
    alpha: float = 1.0
    # Gaussian kernel bandwidth in label space (on the chosen label tensor)
    bandwidth: float = 1.0
    # Label key to use for KDE / mixing (default: main regression targets)
    label_key: str = "y_reg3"

    @staticmethod
    def from_dict(d: Optional[Dict[str, Any]]) -> "CMixupConfig":
        if d is None:
            return CMixupConfig(enabled=False)
        return CMixupConfig(
            enabled=bool(d.get("enabled", False)),
            prob=float(d.get("prob", 1.0)),
            alpha=float(d.get("alpha", 1.0)),
            bandwidth=float(d.get("bandwidth", 1.0)),
            label_key=str(d.get("label_key", "y_reg3")),
        )


class CMixupBatchAugment:
    """
    Batch-level C-Mixup for regression tasks.

    For a batch of labels Y in R^{B x D}, we construct, for each sample i,
    a label-conditional sampling distribution over the batch via a
    Gaussian kernel in label space (following the spirit of C-Mixup),
    then sample a partner j ~ p(j | i) and mix:

        x_mix = lam * x_i + (1 - lam) * x_j
        y_mix = lam * y_i + (1 - lam) * y_j

    This operates purely in-batch (no global pre-computation), which keeps
    the integration lightweight while preserving the core idea of
    label-distribution-aware mixup.
    """

    def __init__(self, cfg: CMixupConfig) -> None:
        self.cfg = cfg

    @staticmethod
    def from_cfg(d: Optional[Dict[str, Any]]) -> Optional["CMixupBatchAugment"]:
        cfg = CMixupConfig.from_dict(d)
        if not cfg.enabled:
            return None
        return CMixupBatchAugment(cfg)

    def _compute_kde_probs(self, labels: torch.Tensor) -> torch.Tensor:
        """
        Compute per-sample KDE-based sampling probabilities over the batch.

        Args:
            labels: (B, D) tensor of regression labels (no grad required).

        Returns:
            mix_probs: (B, B) tensor where mix_probs[i] is a probability
                       distribution over partner indices for sample i.
        """
        if labels.ndim != 2:
            labels = labels.view(labels.size(0), -1)
        bsz, dim = labels.shape
        if bsz < 2:
            # Nothing to mix
            return torch.zeros((bsz, bsz), device=labels.device, dtype=labels.dtype)
        h = float(self.cfg.bandwidth)
        if not torch.isfinite(torch.tensor(h)) or h <= 0.0:
            h = 1.0
        two_h2 = 2.0 * (h ** 2)

        mix_probs = labels.new_zeros((bsz, bsz))
        for i in range(bsz):
            # Gaussian kernel centered at labels[i]
            diff = labels - labels[i].view(1, dim)
            dist2 = (diff * diff).sum(dim=-1)  # (B,)
            scores = torch.exp(-dist2 / two_h2)
            # Avoid degenerate all-zero scores
            scores_sum = scores.sum()
            if scores_sum <= 0:
                probs = torch.full_like(scores, 1.0 / float(bsz))
            else:
                probs = scores / scores_sum
            mix_probs[i] = probs
        return mix_probs

    def apply_main_batch(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Apply C-Mixup to the main regression batch.

        Expected keys in batch (when not NDVI-only):
          - "image": (B,C,H,W)
          - label_key (default "y_reg3"): (B, D_reg)
          - Optionally "y_height", "y_ndvi" (regression scalars) which will also be mixed.

        NDVI-only scalar batches (with "ndvi_only" == True) are left untouched.
        """
        if not self.cfg.enabled or self.cfg.prob <= 0.0:
            return batch
        # Skip NDVI-only scalar batches
        if bool(batch.get("ndvi_only", False)):
            return batch
        if torch.rand(()) > self.cfg.prob:
            return batch

        if "image" not in batch:
            return batch
        if self.cfg.label_key not in batch:
            return batch

        images: torch.Tensor = batch["image"]
        labels: torch.Tensor = batch[self.cfg.label_key]
        if not torch.is_tensor(images) or not torch.is_tensor(labels):
            return batch
        if images.size(0) != labels.size(0):
            return batch

        bsz = images.size(0)
        if bsz < 2:
            return batch

        # Detach labels for KDE computation (no gradients needed)
        labels_detached = labels.detach()
        if labels_detached.ndim == 1:
            labels_detached = labels_detached.view(bsz, 1)
        mix_probs = self._compute_kde_probs(labels_detached)

        # Sample partner indices j ~ p(j | i) for each i
        # mix_probs is (B,B) with rows being distributions
        try:
            idx2 = torch.multinomial(mix_probs, num_samples=1).squeeze(-1)  # (B,)
        except RuntimeError:
            # Fallback to uniform random pairing if anything goes wrong
            idx2 = torch.randint(low=0, high=bsz, size=(bsz,), device=images.device)

        # Sample global mix coefficient from Beta(alpha, alpha)
        lam = 1.0
        if self.cfg.alpha > 0.0:
            lam = float(torch.distributions.Beta(self.cfg.alpha, self.cfg.alpha).sample().item())
        lam = float(max(0.0, min(1.0, lam)))

        # Mix images
        images2 = images[idx2]
        batch["image"] = lam * images + (1.0 - lam) * images2

        # Mix main regression targets and auxiliary scalar regressions
        def _mix_tensor_key(key: str) -> None:
            if key not in batch:
                return
            t = batch[key]
            if not torch.is_tensor(t):
                return
            if t.size(0) != bsz:
                return
            t2 = t[idx2]
            # Broadcast lam over trailing dims
            while t.dim() > 1 and t.shape[1] != 1:
                break
            lam_t = torch.tensor(lam, dtype=t.dtype, device=t.device)
            batch[key] = lam_t * t + (1.0 - lam_t) * t2

        # Main normalized regression targets
        _mix_tensor_key(self.cfg.label_key)
        # Auxiliary scalar regressions (if present)
        _mix_tensor_key("y_height")
        _mix_tensor_key("y_ndvi")

        # To keep validation metrics consistent, drop original-scale targets if present;
        # downstream code will fall back to inverting from normalized labels.
        for k in ("y_reg3_g", "y_reg3_g_m2"):
            if k in batch:
                batch.pop(k, None)

        return batch


