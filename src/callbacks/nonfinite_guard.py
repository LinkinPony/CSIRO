from __future__ import annotations

from typing import Any, Optional

import torch
from lightning.pytorch.callbacks import Callback
from loguru import logger


class NonFiniteGuard(Callback):
    """
    Fail-fast guard for NaN/Inf during training/validation.

    Why this exists:
    - Some hyperparameter samples (e.g., overly large LR) can cause loss/gradients to become non-finite.
    - Under Ray Tune this wastes substantial compute because the trial keeps running while metrics are NaN.

    Behavior:
    - If a non-finite train loss is observed for a batch, raise RuntimeError to terminate the run/trial.
    - Optionally check gradients after backward (disabled by default to avoid overhead).
    """

    def __init__(
        self,
        *,
        enabled: bool = True,
        check_grads: bool = False,
        check_grads_every_n_steps: int = 1,
    ) -> None:
        super().__init__()
        self.enabled = bool(enabled)
        self.check_grads = bool(check_grads)
        self.check_grads_every_n_steps = int(max(1, check_grads_every_n_steps))

    @staticmethod
    def _is_finite(x: Any) -> bool:
        if not isinstance(x, torch.Tensor):
            return True
        try:
            return bool(torch.isfinite(x).all().item())
        except Exception:
            return True

    def _fail(self, *, stage: str, what: str, value: Optional[torch.Tensor], trainer) -> None:  # type: ignore[no-untyped-def]
        step = int(getattr(trainer, "global_step", -1))
        epoch = int(getattr(trainer, "current_epoch", -1))
        v_str = None
        try:
            if isinstance(value, torch.Tensor):
                v_str = str(value.detach().float().cpu().item()) if value.numel() == 1 else f"shape={tuple(value.shape)}"
        except Exception:
            v_str = None
        logger.error("Non-finite detected: stage={} what={} epoch={} step={} value={}", stage, what, epoch, step, v_str)
        raise RuntimeError(f"Non-finite {what} detected at {stage} (epoch={epoch}, step={step}).")

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx) -> None:  # type: ignore[override]
        if not self.enabled:
            return
        # In this repo, training_step returns a scalar loss tensor.
        if isinstance(outputs, torch.Tensor) and (not self._is_finite(outputs)):
            self._fail(stage="train", what="loss", value=outputs, trainer=trainer)

    def on_validation_epoch_end(self, trainer, pl_module) -> None:  # type: ignore[override]
        if not self.enabled:
            return
        # Check common aggregated metrics.
        metrics = getattr(trainer, "callback_metrics", {}) or {}
        for key in ("val_loss", "train_loss"):
            v = metrics.get(key, None)
            if isinstance(v, torch.Tensor) and (not self._is_finite(v)):
                self._fail(stage="val", what=key, value=v, trainer=trainer)

    def on_after_backward(self, trainer, pl_module) -> None:  # type: ignore[override]
        if not self.enabled or not self.check_grads:
            return
        step = int(getattr(trainer, "global_step", 0))
        if (step % self.check_grads_every_n_steps) != 0:
            return
        # Only scan trainable params; stop at first non-finite grad to keep overhead bounded.
        try:
            for name, p in pl_module.named_parameters():
                if not bool(getattr(p, "requires_grad", False)):
                    continue
                g = getattr(p, "grad", None)
                if isinstance(g, torch.Tensor) and (not self._is_finite(g)):
                    logger.error("Non-finite gradient: {} (epoch={}, step={})", name, int(getattr(trainer, "current_epoch", -1)), step)
                    raise RuntimeError(f"Non-finite gradient detected for {name} (epoch={int(getattr(trainer, 'current_epoch', -1))}, step={step}).")
        except RuntimeError:
            raise
        except Exception:
            # Best-effort: never crash due to guard internals unless we actually detected non-finite.
            return

