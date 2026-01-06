from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
from torch import nn
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    LinearLR,
    SequentialLR,
    _LRScheduler,
)

from src.training.sam import SAM


def _is_lora_param_name(name: str) -> bool:
    return ("lora_" in name) or ("lora_magnitude_vector" in name)


def _iter_lora_named_params(module: nn.Module) -> Iterable[Tuple[str, nn.Parameter]]:
    for name, p in module.named_parameters():
        if p.requires_grad and _is_lora_param_name(name):
            yield name, p


def _extract_block_idx(param_name: str, *, layers_pattern: str = "blocks") -> Optional[int]:
    """
    Extract transformer block index from a dotted parameter name.

    Example matches:
      - "...blocks.31.attn.qkv.lora_A.default.weight" -> 31
      - "...base_model.model.blocks.0.mlp.w1.lora_B..." -> 0
    """
    key = str(layers_pattern or "blocks")
    parts = param_name.split(".")
    for i, p in enumerate(parts):
        if p == key and (i + 1) < len(parts):
            try:
                return int(parts[i + 1])
            except Exception:
                return None
    # Best-effort fallback to the common key used by DINO-style ViTs.
    if key != "blocks":
        for i, p in enumerate(parts):
            if p == "blocks" and (i + 1) < len(parts):
                try:
                    return int(parts[i + 1])
                except Exception:
                    return None
    return None


def _iter_module_candidates(root: nn.Module) -> Iterable[nn.Module]:
    """
    Yield a small set of plausible "real backbone" modules for PEFT-wrapped models.
    """
    seen: set[int] = set()
    stack: List[nn.Module] = [root]
    while stack:
        m = stack.pop()
        mid = id(m)
        if mid in seen:
            continue
        seen.add(mid)
        yield m
        for attr in ("base_model", "model", "backbone"):
            child = getattr(m, attr, None)
            if isinstance(child, nn.Module):
                stack.append(child)


def _infer_backbone_depth(backbone: nn.Module, *, layers_pattern: str = "blocks") -> int:
    """
    Infer transformer depth (number of blocks) for LLRD scaling.
    Works for both raw DINOv3 backbones and PEFT-wrapped models.
    """
    key = str(layers_pattern or "blocks")
    for m in _iter_module_candidates(backbone):
        for attr in (key, "blocks"):
            try:
                ml = getattr(m, attr, None)
                if isinstance(ml, (nn.ModuleList, list)):
                    return int(len(ml))
            except Exception:
                continue
    return 0


def _build_lora_param_groups(
    *,
    backbone: nn.Module,
    peft_cfg: Dict[str, Any],
    base_lr: float,
    weight_decay: float,
) -> Tuple[List[Dict[str, Any]], List[nn.Parameter]]:
    """
    Build LoRA optimizer param groups.

    Supports layer-wise LoRA LR via LLRD (layer-wise lr decay), similar to lightly-train's
    DINOv3 EoMT strategy, but applied only to trainable LoRA parameters.
    """
    lora_named = list(_iter_lora_named_params(backbone))
    if not lora_named:
        return [], []

    # Defaults preserve legacy behavior unless explicitly enabled.
    layers_pattern = str(peft_cfg.get("layers_pattern", "blocks") or "blocks")
    try:
        llrd = float(peft_cfg.get("lora_llrd", peft_cfg.get("llrd", 1.0)))
    except Exception:
        llrd = 1.0
    try:
        group_size = int(peft_cfg.get("lora_group_size", 1) or 1)
    except Exception:
        group_size = 1
    group_size = max(1, group_size)

    # Depth for exponent: depth-1 - block_idx (so deepest block gets exponent 0 -> lr=base_lr).
    depth = _infer_backbone_depth(backbone, layers_pattern=layers_pattern)
    # Fallback if we can't infer depth: approximate with max seen block index + 1.
    if depth <= 0:
        max_idx = -1
        for name, _ in lora_named:
            bi = _extract_block_idx(name, layers_pattern=layers_pattern)
            if bi is not None:
                max_idx = max(max_idx, int(bi))
        depth = max(1, max_idx + 1)

    # If llrd==1 (or depth is degenerate), keep a single LoRA group (legacy behavior).
    if (not (llrd > 0.0)) or math.isclose(llrd, 1.0, rel_tol=0.0, abs_tol=1e-12) or depth <= 1:
        params = [p for _, p in lora_named]
        return (
            [
                {
                    "params": params,
                    "lr": float(base_lr),
                    "weight_decay": float(weight_decay),
                    "name": "lora",
                    "group_type": "lora",
                }
            ],
            params,
        )

    bucket_to_params: dict[int, List[nn.Parameter]] = defaultdict(list)
    misc_params: List[nn.Parameter] = []
    for name, p in lora_named:
        bi = _extract_block_idx(name, layers_pattern=layers_pattern)
        if bi is None:
            misc_params.append(p)
            continue
        bucket = int(bi) // int(group_size)
        bucket_to_params[bucket].append(p)

    groups: List[Dict[str, Any]] = []
    # Deterministic order: increasing bucket -> increasing block indices.
    for bucket in sorted(bucket_to_params.keys()):
        start = bucket * group_size
        end = min(depth - 1, (bucket + 1) * group_size - 1)
        rep_block = end  # representative index (deepest in the bucket)
        exponent = max(0, int(depth - 1 - rep_block))
        lr = float(base_lr) * (float(llrd) ** float(exponent))
        name = (
            f"lora_{layers_pattern}_{rep_block:03d}"
            if group_size == 1
            else f"lora_{layers_pattern}_{start:03d}-{end:03d}"
        )
        groups.append(
            {
                "params": bucket_to_params[bucket],
                "lr": lr,
                "weight_decay": float(weight_decay),
                "name": name,
                "group_type": "lora",
                "block_start": int(start),
                "block_end": int(end),
            }
        )

    if misc_params:
        # Treat misc params as the earliest block (smallest LR) to avoid over-updating.
        exponent = max(0, int(depth - 1))
        lr = float(base_lr) * (float(llrd) ** float(exponent))
        groups.append(
            {
                "params": misc_params,
                "lr": lr,
                "weight_decay": float(weight_decay),
                "name": "lora_misc",
                "group_type": "lora",
            }
        )

    # Flatten for convenience
    all_params: List[nn.Parameter] = []
    for g in groups:
        all_params.extend(list(g.get("params", [])))
    return groups, all_params


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
        # Separate parameter groups: head/other, UW (optional), and LoRA (optionally layer-wise LLRD).
        try:
            peft_cfg_raw = getattr(self.hparams, "peft_cfg", None)
            peft_cfg: Dict[str, Any] = dict(peft_cfg_raw or {}) if isinstance(peft_cfg_raw, dict) else {}
        except Exception:
            peft_cfg = {}

        lora_groups: List[Dict[str, Any]] = []
        lora_params: List[torch.nn.Parameter] = []
        try:
            base_lr = float(self._peft_lora_lr or (self.hparams.learning_rate * 0.1))
            base_wd = float(
                self._peft_lora_weight_decay
                if self._peft_lora_weight_decay is not None
                else self.hparams.weight_decay
            )
            lora_groups, lora_params = _build_lora_param_groups(
                backbone=self.feature_extractor.backbone,
                peft_cfg=peft_cfg,
                base_lr=base_lr,
                weight_decay=base_wd,
            )
        except Exception:
            lora_groups, lora_params = [], []

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
                "name": "head",
                "group_type": "head",
            })
        if len(uw_params) > 0:
            uw_lr = float(self._uw_learning_rate) if self._uw_learning_rate is not None else float(self.hparams.learning_rate)
            uw_wd = float(self._uw_weight_decay) if self._uw_weight_decay is not None else float(self.hparams.weight_decay)
            param_groups.append({
                "params": uw_params,
                "lr": uw_lr,
                "weight_decay": uw_wd,
                "name": "uw",
                "group_type": "uw",
            })
        if lora_groups:
            param_groups.extend(lora_groups)

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


