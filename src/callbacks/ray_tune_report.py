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

        # Core Tune metrics
        val_loss = _pick_metric(metrics, ["val_loss", "val_loss/dataloader_idx_0"])
        val_r2 = _pick_metric(metrics, ["val_r2", "val_r2/dataloader_idx_0"])

        # Extra metrics for more flexible HPO (optional in config).
        val_r2_global = _pick_metric(metrics, ["val_r2_global", "val_r2_global/dataloader_idx_0"])
        val_loss_reg3_mse = _pick_metric(metrics, ["val_loss_reg3_mse", "val_loss_reg3_mse/dataloader_idx_0"])
        val_loss_5d_weighted = _pick_metric(
            metrics, ["val_loss_5d_weighted", "val_loss_5d_weighted/dataloader_idx_0"]
        )
        val_loss_ratio_mse = _pick_metric(metrics, ["val_loss_ratio_mse", "val_loss_ratio_mse/dataloader_idx_0"])
        val_loss_height = _pick_metric(metrics, ["val_loss_height", "val_loss_height/dataloader_idx_0"])
        val_loss_ndvi = _pick_metric(metrics, ["val_loss_ndvi", "val_loss_ndvi/dataloader_idx_0"])
        val_loss_state = _pick_metric(metrics, ["val_loss_state", "val_loss_state/dataloader_idx_0"])

        # IMPORTANT: report epoch as 1-based "epochs completed" so Ray ASHA grace_period
        # semantics match user expectations (e.g., grace_period=15 => run at least 15 epochs).
        payload: dict[str, Any] = {"epoch": int(getattr(trainer, "current_epoch", 0)) + 1}
        if val_loss is not None:
            payload["val_loss"] = float(val_loss)
        if val_r2 is not None:
            payload["val_r2"] = float(val_r2)
        if val_r2_global is not None:
            payload["val_r2_global"] = float(val_r2_global)
        if val_loss_reg3_mse is not None:
            payload["val_loss_reg3_mse"] = float(val_loss_reg3_mse)
        if val_loss_5d_weighted is not None:
            payload["val_loss_5d_weighted"] = float(val_loss_5d_weighted)
        if val_loss_ratio_mse is not None:
            payload["val_loss_ratio_mse"] = float(val_loss_ratio_mse)
        if val_loss_height is not None:
            payload["val_loss_height"] = float(val_loss_height)
        if val_loss_ndvi is not None:
            payload["val_loss_ndvi"] = float(val_loss_ndvi)
        if val_loss_state is not None:
            payload["val_loss_state"] = float(val_loss_state)

        if len(payload) > 1:
            session.report(payload)


