from __future__ import annotations

from typing import Any, Iterable, Optional

from lightning.pytorch.callbacks import Callback


def _to_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    try:
        # Lightning often stores tensors in callback_metrics.
        if hasattr(x, "detach"):
            x = x.detach()
        if hasattr(x, "cpu"):
            x = x.cpu()
        if hasattr(x, "item"):
            return float(x.item())
        return float(x)
    except Exception:
        return None


def _pick_metric(metrics: dict, keys: Iterable[str]) -> Optional[float]:
    for k in keys:
        if k in metrics:
            v = _to_float(metrics.get(k))
            if v is not None:
                return v
    return None


class RayTuneReportCallback(Callback):
    """
    Report Lightning metrics to Ray Tune (per validation epoch).

    This is a lightweight alternative to Ray's PL integration callbacks and keeps
    our dependency surface small. It's safe to import even when Ray isn't installed
    (it will no-op).
    """

    def __init__(self) -> None:
        super().__init__()

    def on_validation_epoch_end(self, trainer, pl_module) -> None:  # type: ignore[override]
        # Lazy import so training still works without Ray installed.
        try:
            from ray.air import session
        except Exception:
            return

        # Only report when inside a Tune session.
        try:
            _ = session.get_trial_id()
        except Exception:
            return

        metrics = dict(getattr(trainer, "callback_metrics", {}) or {})

        val_loss = _pick_metric(metrics, ["val_loss", "val_loss/dataloader_idx_0"])
        val_r2 = _pick_metric(metrics, ["val_r2", "val_r2/dataloader_idx_0"])

        payload: dict[str, Any] = {"epoch": int(getattr(trainer, "current_epoch", 0))}
        if val_loss is not None:
            payload["val_loss"] = float(val_loss)
        if val_r2 is not None:
            payload["val_r2"] = float(val_r2)

        if len(payload) > 1:
            session.report(payload)


