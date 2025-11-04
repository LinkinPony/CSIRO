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
        out_path = os.path.join(self.output_dir, f"head-epoch{epoch:03d}.pt")
        torch.save(state, out_path)


