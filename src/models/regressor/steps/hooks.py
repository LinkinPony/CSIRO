from __future__ import annotations

from typing import Any


class StepHooksMixin:
    def training_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0):
        return self._shared_step(batch, stage="train")["loss"]

    def validation_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0):
        out = self._shared_step(batch, stage="val")
        # Only aggregate main regression predictions for val_r2
        if "preds" in out and "targets" in out:
            self._val_preds.append(out["preds"].detach().float().cpu())
            self._val_targets.append(out["targets"].detach().float().cpu())


