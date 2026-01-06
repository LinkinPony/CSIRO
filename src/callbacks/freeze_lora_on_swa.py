from __future__ import annotations

import math
from typing import Optional

import lightning.pytorch as pl
from lightning.pytorch.callbacks import Callback
from loguru import logger


def _compute_swa_start_epoch(max_epochs: int, swa_epoch_start: float | int) -> int:
    """
    Mirror Lightning's common convention:
      - if swa_epoch_start is a float in [0, 1), treat it as a fraction of max_epochs
      - otherwise treat it as an absolute epoch index
    """
    try:
        v = float(swa_epoch_start)
    except Exception:
        try:
            return int(swa_epoch_start)  # type: ignore[arg-type]
        except Exception:
            return 0

    if v < 1.0:
        if not (max_epochs > 0):
            return 0
        if v <= 0.0:
            return 0
        return int(v * float(max_epochs))
    return int(v)


class FreezeLoraOnSwaStart(Callback):
    """
    Freeze LoRA parameters when SWA starts (方案 C).

    This intentionally leaves the backbone frozen (as before) and prevents LoRA adapters
    from updating during the SWA phase, so SWA can focus on stabilizing the head updates.
    """

    def __init__(
        self,
        *,
        swa_epoch_start: float | int = 0.8,
        set_lora_lr_to: float = 0.0,
    ) -> None:
        super().__init__()
        self.swa_epoch_start = swa_epoch_start
        self.set_lora_lr_to = float(set_lora_lr_to)
        self._start_epoch: Optional[int] = None
        self._done: bool = False

    def on_fit_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:  # type: ignore[override]
        self._start_epoch = _compute_swa_start_epoch(
            int(getattr(trainer, "max_epochs", 0) or 0),
            self.swa_epoch_start,
        )

    def on_train_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:  # type: ignore[override]
        if self._done:
            return
        start_epoch = (
            int(self._start_epoch)
            if self._start_epoch is not None
            else _compute_swa_start_epoch(int(getattr(trainer, "max_epochs", 0) or 0), self.swa_epoch_start)
        )
        if int(getattr(trainer, "current_epoch", 0) or 0) < start_epoch:
            return

        # 1) Freeze LoRA parameters (requires_grad=False) to stop gradients & updates.
        num_frozen = 0
        try:
            fe = getattr(pl_module, "feature_extractor", None)
            backbone = getattr(fe, "backbone", None) if fe is not None else None
            if backbone is not None:
                for name, p in backbone.named_parameters():
                    if ("lora_" in name) or ("lora_magnitude_vector" in name):
                        if getattr(p, "requires_grad", False):
                            p.requires_grad = False
                            num_frozen += 1
        except Exception:
            # Best-effort; still try LR patch below.
            pass

        # 2) Hard-set LoRA param group LR to 0.0 (or configured) as an extra safety net.
        num_lr_groups = 0
        try:
            for opt in getattr(trainer, "optimizers", []) or []:
                for group in getattr(opt, "param_groups", []) or []:
                    gtype = str(group.get("group_type", "") or "").lower()
                    gname = str(group.get("name", "") or "").lower()
                    if gtype == "lora" or gname.startswith("lora"):
                        group["lr"] = float(self.set_lora_lr_to)
                        num_lr_groups += 1
        except Exception:
            pass

        try:
            logger.info(
                "SWA start reached (epoch {} >= {}): froze {} LoRA params; set lr={} for {} LoRA param groups.",
                int(getattr(trainer, "current_epoch", 0) or 0),
                int(start_epoch),
                int(num_frozen),
                float(self.set_lora_lr_to),
                int(num_lr_groups),
            )
        except Exception:
            pass

        self._done = True


