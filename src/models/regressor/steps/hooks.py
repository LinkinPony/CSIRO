from __future__ import annotations

from typing import Any


class StepHooksMixin:
    def training_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0):
        out = self._shared_step(batch, stage="train")
        # When PCGrad is enabled, stash per-task terms for the custom backward hook.
        try:
            if bool(getattr(self, "_pcgrad_enabled", False)) and isinstance(out, dict):
                setattr(self, "_pcgrad_terms", out.get("pcgrad_terms", None))
                setattr(self, "_pcgrad_unscaled_loss", out.get("loss", None))
            else:
                setattr(self, "_pcgrad_terms", None)
                setattr(self, "_pcgrad_unscaled_loss", None)
        except Exception:
            pass
        return out["loss"]

    def validation_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0):
        out = self._shared_step(batch, stage="val")
        # Only aggregate main regression predictions for val_r2
        if "preds" in out and "targets" in out:
            self._val_preds.append(out["preds"].detach().float().cpu())
            self._val_targets.append(out["targets"].detach().float().cpu())


