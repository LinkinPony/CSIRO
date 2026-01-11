from __future__ import annotations

from typing import Any, Dict, Optional

import torch
from lightning.pytorch.callbacks import Callback
from loguru import logger


class ExponentialMovingAverage(Callback):
    """
    Maintain an exponential moving average (EMA) of model parameters.

    Design goals for this repo:
    - Track only trainable parameters by default (backbone is frozen; avoids duplicating huge weights).
    - Optionally evaluate validation metrics using EMA weights (so HPO optimizes the averaged model).
    - Optionally apply EMA weights at the end of training (so exported head weights use EMA).
    """

    def __init__(
        self,
        *,
        decay: float = 0.999,
        update_every_n_steps: int = 1,
        start_step: int = 0,
        eval_with_ema: bool = True,
        apply_at_end: bool = True,
        trainable_only: bool = True,
    ) -> None:
        super().__init__()
        self.decay = float(decay)
        if not (0.0 < self.decay < 1.0):
            raise ValueError(f"EMA decay must be in (0, 1), got: {self.decay}")

        self.update_every_n_steps = int(update_every_n_steps)
        if self.update_every_n_steps < 1:
            self.update_every_n_steps = 1

        self.start_step = int(start_step)
        if self.start_step < 0:
            self.start_step = 0

        self.eval_with_ema = bool(eval_with_ema)
        self.apply_at_end = bool(apply_at_end)
        self.trainable_only = bool(trainable_only)

        self._ema: Dict[str, torch.Tensor] = {}
        self._backup: Optional[Dict[str, torch.Tensor]] = None
        self._last_updated_step: int = -1
        self._num_updates: int = 0

    def _iter_tracked_params(self, pl_module) -> Any:  # type: ignore[no-untyped-def]
        for name, p in pl_module.named_parameters():
            try:
                if self.trainable_only and not bool(getattr(p, "requires_grad", False)):
                    continue
            except Exception:
                continue
            if not isinstance(p, torch.Tensor):
                continue
            if not torch.is_floating_point(p):
                continue
            yield str(name), p

    def _ensure_initialized(self, pl_module) -> None:  # type: ignore[no-untyped-def]
        if self._ema:
            return
        with torch.no_grad():
            for name, p in self._iter_tracked_params(pl_module):
                self._ema[name] = p.detach().clone()
        logger.info(
            "EMA initialized: tracked_params={}, decay={}, trainable_only={}",
            len(self._ema),
            self.decay,
            self.trainable_only,
        )

    def _update(self, pl_module) -> None:  # type: ignore[no-untyped-def]
        self._ensure_initialized(pl_module)
        if not self._ema:
            return
        with torch.no_grad():
            for name, p in self._iter_tracked_params(pl_module):
                ema_t = self._ema.get(name, None)
                if ema_t is None:
                    self._ema[name] = p.detach().clone()
                    continue
                # Keep EMA tensor on the same device/dtype as the live parameter.
                if ema_t.device != p.device or ema_t.dtype != p.dtype:
                    ema_t = ema_t.to(device=p.device, dtype=p.dtype)
                    self._ema[name] = ema_t
                # ema = decay * ema + (1 - decay) * param
                ema_t.mul_(self.decay).add_(p.detach(), alpha=(1.0 - self.decay))
        self._num_updates += 1

    def _swap_to_ema(self, pl_module) -> None:  # type: ignore[no-untyped-def]
        self._ensure_initialized(pl_module)
        if not self._ema:
            return
        if self._backup is not None:
            return
        backup: Dict[str, torch.Tensor] = {}
        with torch.no_grad():
            for name, p in self._iter_tracked_params(pl_module):
                ema_t = self._ema.get(name, None)
                if ema_t is None:
                    continue
                if ema_t.device != p.device or ema_t.dtype != p.dtype:
                    ema_t = ema_t.to(device=p.device, dtype=p.dtype)
                    self._ema[name] = ema_t
                backup[name] = p.detach().clone()
                p.detach().copy_(ema_t)
        self._backup = backup

    def _restore_from_backup(self, pl_module) -> None:  # type: ignore[no-untyped-def]
        if self._backup is None:
            return
        with torch.no_grad():
            for name, p in self._iter_tracked_params(pl_module):
                b = self._backup.get(name, None)
                if b is None:
                    continue
                if b.device != p.device or b.dtype != p.dtype:
                    b = b.to(device=p.device, dtype=p.dtype)
                p.detach().copy_(b)
        self._backup = None

    # -----------------
    # Lightning hooks
    # -----------------

    def on_fit_start(self, trainer, pl_module) -> None:  # type: ignore[override]
        # Lazily initialize on demand, but create immediately if possible.
        try:
            self._ensure_initialized(pl_module)
            self._last_updated_step = int(getattr(trainer, "global_step", 0))
        except Exception:
            pass

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx) -> None:  # type: ignore[override]
        try:
            if bool(getattr(trainer, "sanity_checking", False)):
                return
        except Exception:
            pass

        step = int(getattr(trainer, "global_step", 0))
        if step == self._last_updated_step:
            return
        if step < self.start_step:
            self._last_updated_step = step
            return
        if (step % self.update_every_n_steps) != 0:
            self._last_updated_step = step
            return

        self._update(pl_module)
        self._last_updated_step = step

    def on_validation_epoch_start(self, trainer, pl_module) -> None:  # type: ignore[override]
        if not self.eval_with_ema:
            return
        self._swap_to_ema(pl_module)

    def on_validation_epoch_end(self, trainer, pl_module) -> None:  # type: ignore[override]
        if not self.eval_with_ema:
            return
        # Restore AFTER other callbacks (place this callback last in callbacks list).
        self._restore_from_backup(pl_module)

    def on_test_epoch_start(self, trainer, pl_module) -> None:  # type: ignore[override]
        if not self.eval_with_ema:
            return
        self._swap_to_ema(pl_module)

    def on_test_epoch_end(self, trainer, pl_module) -> None:  # type: ignore[override]
        if not self.eval_with_ema:
            return
        self._restore_from_backup(pl_module)

    def on_train_end(self, trainer, pl_module) -> None:  # type: ignore[override]
        # Ensure we don't leave the model in a swapped (backup) state.
        self._restore_from_backup(pl_module)
        if self.apply_at_end:
            # Persist EMA weights on the module so subsequent checkpoint/export uses EMA.
            self._swap_to_ema(pl_module)
            # Do NOT restore.

    # Lightning will persist callback state in checkpoints when stateful callbacks implement this API.
    def state_dict(self) -> dict[str, Any]:
        try:
            ema_cpu = {k: v.detach().cpu() for k, v in (self._ema or {}).items()}
        except Exception:
            ema_cpu = {}
        return {
            "decay": float(self.decay),
            "update_every_n_steps": int(self.update_every_n_steps),
            "start_step": int(self.start_step),
            "eval_with_ema": bool(self.eval_with_ema),
            "apply_at_end": bool(self.apply_at_end),
            "trainable_only": bool(self.trainable_only),
            "num_updates": int(self._num_updates),
            "last_updated_step": int(self._last_updated_step),
            "ema": ema_cpu,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        try:
            self._num_updates = int(state_dict.get("num_updates", 0))
        except Exception:
            self._num_updates = 0
        try:
            self._last_updated_step = int(state_dict.get("last_updated_step", -1))
        except Exception:
            self._last_updated_step = -1
        ema = state_dict.get("ema", None)
        if isinstance(ema, dict):
            out: Dict[str, torch.Tensor] = {}
            for k, v in ema.items():
                if isinstance(v, torch.Tensor):
                    out[str(k)] = v.detach().clone()
            self._ema = out


