from __future__ import annotations

from typing import Any, Optional

import lightning.pytorch as pl
from lightning.pytorch.callbacks import Callback
from loguru import logger


def _compute_swa_start_epoch(max_epochs: int, swa_epoch_start: float | int) -> int:
    """
    Mirror Lightning's convention for `swa_epoch_start`:
      - if swa_epoch_start is a float in [0, 1), treat it as a fraction of max_epochs
      - otherwise treat it as an absolute (0-based) epoch index
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


class AdaptiveSwaLrsOnStart(Callback):
    """
    Make Lightning SWA LR targets follow the *current* optimizer param-group LRs at SWA start.

    Why:
    - In this repo, HPO samples `optimizer.lr`/`peft.lora_lr` and uses cosine LR scheduling.
    - A fixed `swa_lrs` (e.g., 1e-4) can be *much larger* than the current LR near the end,
      causing a sharp LR increase at SWA start -> unstable training / metric spikes.

    Implementation:
    - We mutate Lightning's SWA callback private field `_swa_lrs` right before SWA activates,
      setting it to the optimizer's current per-param-group learning rates.
    - This keeps SWA averaging behavior but avoids an LR discontinuity.
    """

    def __init__(
        self,
        *,
        swa_callback: Any,
        eps: float = 1e-12,
        log: bool = True,
    ) -> None:
        super().__init__()
        self.swa_callback = swa_callback
        self.eps = float(eps)
        self.log = bool(log)
        self._start_epoch: Optional[int] = None
        self._done: bool = False

    def on_fit_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:  # type: ignore[override]
        try:
            swa_epoch_start = getattr(self.swa_callback, "_swa_epoch_start", 0.8)
        except Exception:
            swa_epoch_start = 0.8
        self._start_epoch = _compute_swa_start_epoch(
            int(getattr(trainer, "max_epochs", 0) or 0),
            swa_epoch_start,
        )

    def on_train_epoch_start(  # type: ignore[override]
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        if self._done:
            return

        # Only act at/after the SWA boundary, and only if SWA hasn't constructed its scheduler yet.
        start_epoch = (
            int(self._start_epoch)
            if self._start_epoch is not None
            else _compute_swa_start_epoch(
                int(getattr(trainer, "max_epochs", 0) or 0),
                getattr(self.swa_callback, "_swa_epoch_start", 0.8),
            )
        )
        cur_epoch = int(getattr(trainer, "current_epoch", 0) or 0)
        if cur_epoch < start_epoch:
            return

        try:
            if getattr(self.swa_callback, "_swa_scheduler", None) is not None:
                # SWA already initialized; nothing left to do.
                self._done = True
                return
        except Exception:
            # Best-effort; continue and try to set lrs once.
            pass

        # Read current optimizer param-group LRs (after warmup/cosine scheduling).
        opt = None
        try:
            opts = getattr(trainer, "optimizers", []) or []
            if opts:
                opt = opts[0]
        except Exception:
            opt = None
        if opt is None:
            return

        # Lightning sometimes wraps optimizers; support both.
        opt_unwrapped = getattr(opt, "optimizer", opt)
        groups = list(getattr(opt_unwrapped, "param_groups", []) or [])
        if not groups:
            return

        lrs: list[float] = []
        debug_groups: list[tuple[str, str, float]] = []
        for g in groups:
            lr_raw = g.get("lr", None)
            try:
                lr = float(lr_raw)
            except Exception:
                lr = float("nan")
            if not (lr > 0.0):
                lr = float(self.eps)
            lr = max(float(lr), float(self.eps))
            lrs.append(lr)
            try:
                gname = str(g.get("name", "") or "")
                gtype = str(g.get("group_type", "") or "")
            except Exception:
                gname, gtype = "", ""
            debug_groups.append((gname, gtype, lr))

        # Hot-patch SWA callback's target LRs.
        try:
            setattr(self.swa_callback, "_swa_lrs", lrs)
            if self.log:
                try:
                    logger.info(
                        "Adaptive SWA LRs applied at epoch {} (start_epoch={}): {}",
                        cur_epoch,
                        start_epoch,
                        [
                            {"name": n, "group_type": t, "swa_lr": lr}
                            for (n, t, lr) in debug_groups
                        ],
                    )
                except Exception:
                    pass
        except Exception as e:
            if self.log:
                try:
                    logger.warning(f"Failed to set adaptive SWA LRs (non-fatal): {e}")
                except Exception:
                    pass
            return

        self._done = True


