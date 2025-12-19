from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch
from torch import Tensor


class SharedStepMixin:
    """
    Orchestrates batch selection, augmentation decision, prediction, and loss computation.
    """

    def _shared_step(self, batch: Any, stage: str) -> Dict[str, Tensor]:
        # When multiple dataloaders are used, Lightning may deliver a list/tuple of batches.
        # For alternating training, process only ONE sub-batch per step to avoid holding multiple graphs.
        if isinstance(batch, (list, tuple)):
            flat_batches: List[Any] = []
            for sub in batch:
                if isinstance(sub, (list, tuple)):
                    for sb in sub:
                        flat_batches.append(sb)
                else:
                    flat_batches.append(sub)

            use_ndvi_only = False
            if stage == "train" and self.enable_ndvi_dense:
                try:
                    use_ndvi_only = bool(torch.rand(()) < self._ndvi_dense_prob)
                except Exception:
                    use_ndvi_only = False

            selected: Optional[Any] = None
            if use_ndvi_only:
                selected = next(
                    (x for x in flat_batches if isinstance(x, dict) and bool(x.get("ndvi_only", False))),
                    None,
                )
            else:
                selected = next(
                    (x for x in flat_batches if isinstance(x, dict) and not bool(x.get("ndvi_only", False))),
                    None,
                )
            if selected is None:
                selected = next((x for x in flat_batches if isinstance(x, dict)), flat_batches[0])
            return self._shared_step(selected, stage)

        # batch is a dict from the dataset
        is_ndvi_only: bool = bool(batch.get("ndvi_only", False))

        # Decide which augmentation (CutMix / manifold mixup) to apply for this batch.
        use_cutmix = False
        use_mixup = False
        if stage == "train" and (not is_ndvi_only):
            cutmix_enabled = self._cutmix_main is not None
            cutmix_prob = 0.0
            if cutmix_enabled:
                try:
                    cutmix_enabled = bool(getattr(self._cutmix_main, "cfg", None) and self._cutmix_main.cfg.enabled)
                    cutmix_prob = float(self._cutmix_main.cfg.prob)
                except Exception:
                    cutmix_enabled = False
                    cutmix_prob = 0.0

            mixup_enabled = self._manifold_mixup is not None and bool(self._manifold_mixup.enabled)
            mixup_prob = 0.0
            if mixup_enabled:
                try:
                    mixup_prob = float(self._manifold_mixup.prob)
                except Exception:
                    mixup_prob = 0.0

            cut_trigger = False
            mix_trigger = False
            if cutmix_enabled and cutmix_prob > 0.0:
                try:
                    cut_trigger = bool(torch.rand(()) < cutmix_prob)
                except Exception:
                    cut_trigger = False
            if mixup_enabled and mixup_prob > 0.0:
                try:
                    mix_trigger = bool(torch.rand(()) < mixup_prob)
                except Exception:
                    mix_trigger = False

            if cut_trigger and mix_trigger:
                try:
                    choose_cut = bool(torch.rand(()) < 0.5)
                except Exception:
                    choose_cut = True
                if choose_cut:
                    use_cutmix, use_mixup = True, False
                else:
                    use_cutmix, use_mixup = False, True
            elif cut_trigger:
                use_cutmix = True
            elif mix_trigger:
                use_mixup = True

        images: Tensor = batch["image"]
        if use_cutmix and self._cutmix_main is not None:
            try:
                batch, _ = self._cutmix_main.apply_main_batch(batch, force=True)  # type: ignore[assignment]
                images = batch["image"]
            except Exception:
                pass

        head_type = str(getattr(self, "_head_type", "mlp")).lower()
        pred_reg3, z, z_layers, ratio_logits_pred, ndvi_pred, batch = self._predict_reg3_and_z(
            images=images,
            batch=batch,
            stage=stage,
            use_mixup=use_mixup,
            is_ndvi_only=is_ndvi_only,
        )

        if is_ndvi_only:
            return self._loss_ndvi_only(
                stage=stage,
                head_type=head_type,
                z=z,
                z_layers=z_layers,
                ndvi_pred_from_head=ndvi_pred,
                batch=batch,
            )

        return self._loss_supervised(
            stage=stage,
            head_type=head_type,
            pred_reg3=pred_reg3,
            z=z,
            z_layers=z_layers,
            ratio_logits_pred=ratio_logits_pred,
            ndvi_pred_from_head=ndvi_pred,
            batch=batch,
        )


