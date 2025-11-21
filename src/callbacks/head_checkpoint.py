from __future__ import annotations

import os
from typing import Any, Dict

import torch
from lightning.pytorch.callbacks import Callback
from torch import nn

from src.models.head_builder import build_head_layer
from src.models.peft_integration import export_lora_payload_if_any


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

        # Build an inference head state_dict: either a dedicated pl_module.head
        # (new-style) or a composed shared_bottleneck + main reg head (+ optional
        # ratio head, packed into a single linear layer).
        state_dict_to_save: Dict[str, Any]
        # Default values used for meta; will be populated in the try-block below.
        num_outputs_main = 1
        num_ratio_outputs = 0
        head_total_outputs = 1
        try:
            # Core head shape information
            num_outputs_main = int(getattr(pl_module.hparams, "num_outputs", 1))
            ratio_head = getattr(pl_module, "ratio_head", None)
            num_ratio_outputs = int(ratio_head.out_features) if isinstance(ratio_head, nn.Linear) else 0
            head_total_outputs = int(num_outputs_main + num_ratio_outputs)

            head_attr = getattr(pl_module, "head", None)
            if isinstance(head_attr, nn.Module):
                # New-style explicit head module (e.g., PLEHead). Export directly.
                state_dict_to_save = head_attr.state_dict()
            else:
                # Prefer composing a fresh head module to ensure compatibility with inference script.
                # When a ratio head is present, we pack both Dry_Total_g (main regression)
                # and ratio logits into a single linear layer with out_features =
                # num_outputs_main + num_ratio_outputs.
                use_output_softplus_eff = False
                head_module = build_head_layer(
                    embedding_dim=int(getattr(pl_module.hparams, "embedding_dim", 1024)),
                    num_outputs=head_total_outputs,
                    head_hidden_dims=list(getattr(pl_module.hparams, "head_hidden_dims", [])),
                    head_activation=str(getattr(pl_module.hparams, "head_activation", "relu")),
                    dropout=float(getattr(pl_module.hparams, "dropout", 0.0)),
                    use_output_softplus=use_output_softplus_eff,
                )

                def collect_linears(m: nn.Module):
                    return [mod for mod in m.modules() if isinstance(mod, nn.Linear)]

                # New-style model: shared_bottleneck + reg3_heads (one or more independent 1-d heads)
                if hasattr(pl_module, "shared_bottleneck") and hasattr(pl_module, "reg3_heads"):
                    bottleneck_linears = collect_linears(pl_module.shared_bottleneck)
                    tgt_linears = collect_linears(head_module)
                    if len(tgt_linears) != len(bottleneck_linears) + 1:
                        raise RuntimeError("Mismatch in linear layers between bottleneck and inference head")

                    # Copy shared bottleneck linear layers 1:1
                    for s, t in zip(bottleneck_linears, tgt_linears[:-1]):
                        with torch.no_grad():
                            t.weight.copy_(s.weight.data)
                            if t.bias is not None and s.bias is not None:
                                t.bias.copy_(s.bias.data)

                    # Aggregate final layer from scalar reg heads and optional ratio head
                    # into a single Linear(out_features=head_total_outputs).
                    final_linear = tgt_linears[-1]
                    reg3_heads = list(getattr(pl_module, "reg3_heads"))
                    if final_linear.out_features != head_total_outputs:
                        raise RuntimeError("Final inference head out_features does not match packed outputs")
                    with torch.no_grad():
                        row = 0
                        # Pack main reg3 scalar heads first
                        for h in reg3_heads:
                            if not isinstance(h, nn.Linear) or h.out_features != 1:
                                raise RuntimeError("reg3_heads must contain nn.Linear modules with out_features=1")
                            final_linear.weight[row : row + 1, :].copy_(h.weight.data)
                            if final_linear.bias is not None and h.bias is not None:
                                final_linear.bias[row] = h.bias.data[0]
                            row += 1
                        # Pack ratio head logits (if present) into remaining rows
                        if num_ratio_outputs > 0 and isinstance(ratio_head, nn.Linear):
                            if ratio_head.out_features != num_ratio_outputs:
                                raise RuntimeError("ratio_head.out_features mismatch")
                            final_linear.weight[row : row + num_ratio_outputs, :].copy_(ratio_head.weight.data)
                            if final_linear.bias is not None and ratio_head.bias is not None:
                                final_linear.bias[row : row + num_ratio_outputs].copy_(ratio_head.bias.data)
                    state_dict_to_save = head_module.state_dict()

                else:
                    # Legacy path: shared_bottleneck + single reg_head or a monolithic pl_module.head
                    src_linears = []
                    if hasattr(pl_module, "shared_bottleneck") and hasattr(pl_module, "reg_head"):
                        src_linears.extend(collect_linears(pl_module.shared_bottleneck))
                        src_linears.append(pl_module.reg_head)
                    else:
                        # fallback to legacy single head
                        src_linears = collect_linears(pl_module.head)

                    tgt_linears = collect_linears(head_module)
                    if len(src_linears) != len(tgt_linears):
                        raise RuntimeError("Mismatch in linear layers between source heads and inference head")
                    for s, t in zip(src_linears, tgt_linears):
                        with torch.no_grad():
                            t.weight.copy_(s.weight.data)
                            if t.bias is not None and s.bias is not None:
                                t.bias.copy_(s.bias.data)
                    state_dict_to_save = head_module.state_dict()
        except Exception:
            # ultimate fallback: try existing pl_module.head if present, otherwise fall back to composed head_module
            head_attr = getattr(pl_module, "head", None)
            if head_attr is not None:
                state_dict_to_save = head_attr.state_dict()
            else:
                # if a composed head_module exists, use it; otherwise re-raise
                if "head_module" in locals():
                    state_dict_to_save = head_module.state_dict()
                else:
                    raise

        state: Dict[str, Any] = {
            "state_dict": state_dict_to_save,
            "meta": {
                "backbone": getattr(pl_module.hparams, "backbone_name", None) if hasattr(pl_module, "hparams") else None,
                "embedding_dim": int(getattr(pl_module.hparams, "embedding_dim", 1024)) if hasattr(pl_module, "hparams") else 1024,
                # num_outputs_main: number of primary regression outputs (e.g., Dry_Total_g only)
                "num_outputs_main": int(getattr(pl_module.hparams, "num_outputs", 1)) if hasattr(pl_module, "hparams") else 1,
                # num_outputs_ratio: number of ratio logits packed after the main outputs
                "num_outputs_ratio": int(num_ratio_outputs),
                # head_total_outputs: total outputs in the packed head module
                "head_total_outputs": int(head_total_outputs),
                # Head architecture type (e.g., "mlp", "ple")
                "head_type": getattr(pl_module.hparams, "head_type", "mlp") if hasattr(pl_module, "hparams") else "mlp",
                "head_hidden_dims": list(getattr(pl_module.hparams, "head_hidden_dims", [])) if hasattr(pl_module, "hparams") else [],
                "head_activation": getattr(pl_module.hparams, "head_activation", "relu") if hasattr(pl_module, "hparams") else "relu",
                "head_dropout": float(getattr(pl_module.hparams, "dropout", 0.0)) if hasattr(pl_module, "hparams") else 0.0,
                # Optional PLE expert hidden dimension (0 when unused)
                "ple_expert_hidden_dim": int(getattr(pl_module.hparams, "ple_expert_hidden_dim", 0)) if hasattr(pl_module, "hparams") else 0,
                # Softplus is applied (if needed) in the training module; the packed head
                # used for offline inference is exported without a terminal Softplus.
                "use_output_softplus": False,
                "log_scale_targets": bool(getattr(pl_module.hparams, "log_scale_targets", False)) if hasattr(pl_module, "hparams") else False,
                # Order of ratio components packed in the head (if any)
                "ratio_components": ["Dry_Clover_g", "Dry_Dead_g", "Dry_Green_g"] if num_ratio_outputs > 0 else [],
            },
        }
        # Optionally bundle LoRA adapter payload alongside the head (to preserve the two-inputs rule at inference)
        try:
            fe = getattr(pl_module, "feature_extractor", None)
            if fe is not None and hasattr(fe, "backbone"):
                peft_payload = export_lora_payload_if_any(fe.backbone)
                if peft_payload is not None:
                    state["peft"] = peft_payload
        except Exception:
            pass
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

        # Save NDVI dense head separately if present
        try:
            ndvi_head = getattr(pl_module, "ndvi_dense_head", None)
            fe = getattr(pl_module, "feature_extractor", None)
            if ndvi_head is not None:
                ndvi_state = {
                    "state_dict": ndvi_head.state_dict(),
                    "meta": {
                        "backbone": getattr(pl_module.hparams, "backbone_name", None) if hasattr(pl_module, "hparams") else None,
                        "embedding_dim": int(getattr(pl_module.hparams, "embedding_dim", 1024)) if hasattr(pl_module, "hparams") else 1024,
                        "out_channels": 1,
                        "head_type": "ndvi_dense_linear",
                    },
                }
                try:
                    if fe is not None and hasattr(fe, "backbone"):
                        peft_payload = export_lora_payload_if_any(fe.backbone)
                        if peft_payload is not None:
                            ndvi_state["peft"] = peft_payload
                except Exception:
                    pass
                ndvi_path = os.path.join(self.output_dir, f"head-ndvi-epoch{epoch:03d}.pt")
                torch.save(ndvi_state, ndvi_path)
        except Exception:
            pass


