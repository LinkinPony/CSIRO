from __future__ import annotations

from typing import Any, Dict, List

import torch
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    LinearLR,
    SequentialLR,
    _LRScheduler,
)

from ..peft_integration import get_lora_param_list
from src.training.sam import SAM


class RegressorOptimMixin:
    # Guard optimizer stepping to avoid AMP GradScaler assertion when no grads were produced
    def optimizer_step(
        self,
        epoch: int,
        batch_idx: int,
        optimizer: Optimizer,
        optimizer_closure,
        **kwargs: Any,
    ) -> None:
        # Lightning automatic optimization passes in an `optimizer_closure` that runs:
        #   training_step -> (optional) zero_grad -> backward
        #
        # When overriding this hook, Lightning requires that the closure gets executed, otherwise the loop will
        # error when consuming the closure result. Importantly, we must execute the closure **exactly once** per
        # optimizer step. Calling `optimizer_closure()` manually and then calling `optimizer.step(closure=...)`
        # would execute it twice under common precision plugins (and under most torch optimizers that call the
        # closure), causing an extra forward/backward and changing gradients.
        #
        # We therefore wrap the closure to (a) execute it once, (b) detect the "no grads" case, and (c) return
        # `None` to signal AMP precision plugins to skip `scaler.step()` (avoids GradScaler assertions).
        def _has_any_grad() -> bool:
            for group in optimizer.param_groups:
                for p in group.get("params", []):
                    if getattr(p, "grad", None) is not None:
                        return True
            return False

        # SAM requires a two-step update with an extra forward-backward pass.
        if isinstance(optimizer, SAM):
            optimizer_closure()
            if not _has_any_grad():
                # No gradients this step; avoid scaler.step assertion and skip SAM update
                return
            optimizer.first_step(zero_grad=True)
            optimizer_closure()
            optimizer.second_step(zero_grad=True)
        else:
            def _closure():
                loss = optimizer_closure()
                # If no gradients were produced, return None so AMP plugins will skip stepping
                return None if not _has_any_grad() else loss

            optimizer.step(closure=_closure)
            optimizer.zero_grad(set_to_none=True)

    def configure_optimizers(self):
        # Separate parameter groups: LoRA adapters (smaller LR) and the rest (head, optionally others)
        lora_params: List[torch.nn.Parameter] = []
        try:
            lora_params = get_lora_param_list(self.feature_extractor.backbone)
        except Exception:
            lora_params = []

        all_params = [p for p in self.parameters() if p.requires_grad]
        lora_set = set(lora_params)
        # Uncertainty weighting parameters (separate group if present)
        uw_params: List[torch.nn.Parameter] = []
        try:
            if self._uw_task_params is not None:
                uw_params = [p for p in self._uw_task_params.parameters() if p.requires_grad]
        except Exception:
            uw_params = []
        uw_set = set(uw_params)
        other_params = [p for p in all_params if p not in lora_set and p not in uw_set]

        param_groups: List[Dict[str, Any]] = []
        if len(other_params) > 0:
            param_groups.append({
                "params": other_params,
                "lr": self.hparams.learning_rate,
                "weight_decay": self.hparams.weight_decay,
            })
        if len(uw_params) > 0:
            uw_lr = float(self._uw_learning_rate) if self._uw_learning_rate is not None else float(self.hparams.learning_rate)
            uw_wd = float(self._uw_weight_decay) if self._uw_weight_decay is not None else float(self.hparams.weight_decay)
            param_groups.append({
                "params": uw_params,
                "lr": uw_lr,
                "weight_decay": uw_wd,
            })
        if len(lora_params) > 0:
            lora_lr = float(self._peft_lora_lr or (self.hparams.learning_rate * 0.1))
            lora_wd = float(self._peft_lora_weight_decay if self._peft_lora_weight_decay is not None else self.hparams.weight_decay)
            param_groups.append({
                "params": lora_params,
                "lr": lora_lr,
                "weight_decay": lora_wd,
            })

        # Optimizer selection: plain AdamW or SAM-wrapped AdamW.
        opt_name = str(getattr(self.hparams, "optimizer_name", "adamw")).lower()
        use_sam_flag = bool(getattr(self.hparams, "use_sam", False))
        if opt_name in ("sam", "sam_adamw", "adamw_sam"):
            use_sam_flag = True

        if use_sam_flag:
            sam_rho = float(getattr(self.hparams, "sam_rho", 0.05))
            sam_adaptive = bool(getattr(self.hparams, "sam_adaptive", False))
            optimizer: Optimizer = SAM(param_groups, AdamW, rho=sam_rho, adaptive=sam_adaptive)
        else:
            optimizer = AdamW(param_groups)

        if self.hparams.scheduler_name and self.hparams.scheduler_name.lower() == "cosine":
            max_epochs: int = int(self.hparams.max_epochs or 10)
            warmup_epochs: int = int(getattr(self.hparams, "scheduler_warmup_epochs", 0) or 0)
            start_factor: float = float(getattr(self.hparams, "scheduler_warmup_start_factor", 0.1))

            if warmup_epochs > 0:
                # Linear warmup for the first N epochs, then cosine annealing
                warmup = LinearLR(optimizer, start_factor=start_factor, total_iters=warmup_epochs)
                cosine = CosineAnnealingLR(optimizer, T_max=max(1, max_epochs - warmup_epochs))
                scheduler: _LRScheduler = SequentialLR(
                    optimizer,
                    schedulers=[warmup, cosine],
                    milestones=[warmup_epochs],
                )
            else:
                scheduler = CosineAnnealingLR(optimizer, T_max=max_epochs)

            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "epoch",
                    "frequency": 1,
                    "monitor": "val_loss",
                },
            }
        return optimizer


