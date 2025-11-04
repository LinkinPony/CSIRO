from __future__ import annotations

import os
from typing import Any, Dict

import torch
from lightning.pytorch.callbacks import Callback


class HeadCheckpoint(Callback):
    def __init__(self, output_dir: str) -> None:
        super().__init__()
        self.output_dir = os.path.abspath(output_dir)
        os.makedirs(self.output_dir, exist_ok=True)

    def on_validation_epoch_end(self, trainer, pl_module) -> None:  # type: ignore[override]
        epoch = trainer.current_epoch
        # Collect metrics for filename suffix if available
        metrics = getattr(trainer, "callback_metrics", {}) or {}
        def _get_float(name: str):
            try:
                v = metrics.get(name, None)
                if v is None:
                    return None
                # torch.Tensor -> float
                try:
                    import torch as _t
                    if isinstance(v, _t.Tensor):
                        v = v.detach().cpu().item()
                except Exception:
                    pass
                return float(v)
            except Exception:
                return None

        val_loss = _get_float("val_loss")
        train_loss = _get_float("train_loss")
        val_r2 = _get_float("val_r2")

        state: Dict[str, Any] = {
            "state_dict": pl_module.head.state_dict(),
            "meta": {
                "backbone": getattr(pl_module.hparams, "backbone_name", None) if hasattr(pl_module, "hparams") else None,
                "embedding_dim": int(getattr(pl_module.hparams, "embedding_dim", 1024)) if hasattr(pl_module, "hparams") else 1024,
                "num_outputs": int(getattr(pl_module.hparams, "num_outputs", 3)) if hasattr(pl_module, "hparams") else 3,
                "head_hidden_dims": list(getattr(pl_module.hparams, "head_hidden_dims", [])) if hasattr(pl_module, "hparams") else [],
                "head_activation": getattr(pl_module.hparams, "head_activation", "relu") if hasattr(pl_module, "hparams") else "relu",
                "head_dropout": float(getattr(pl_module.hparams, "dropout", 0.0)) if hasattr(pl_module, "hparams") else 0.0,
                "use_output_softplus": bool(getattr(pl_module.hparams, "use_output_softplus", True)) if hasattr(pl_module, "hparams") else True,
            },
        }
        # Build filename with optional metric suffixes
        suffix_parts: list[str] = []
        if val_loss is not None:
            suffix_parts.append(f"val_loss{val_loss:.6f}")
        if train_loss is not None:
            suffix_parts.append(f"train_loss{train_loss:.6f}")
        if val_r2 is not None:
            suffix_parts.append(f"val_r2{val_r2:.6f}")
        metrics_suffix = ("-" + "-".join(suffix_parts)) if suffix_parts else ""
        out_path = os.path.join(self.output_dir, f"head-epoch{epoch:03d}{metrics_suffix}.pt")
        torch.save(state, out_path)


