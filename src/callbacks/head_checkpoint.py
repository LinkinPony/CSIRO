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

        # Build an inference head state_dict: shared bottleneck + 3-d reg head (+ optional softplus)
        state_dict_to_save: Dict[str, Any]
        try:
            # Prefer composing a fresh head module to ensure compatibility with inference script
            head_module = build_head_layer(
                embedding_dim=int(getattr(pl_module.hparams, "embedding_dim", 1024)),
                num_outputs=int(getattr(pl_module.hparams, "num_outputs", 3)),
                head_hidden_dims=list(getattr(pl_module.hparams, "head_hidden_dims", [])),
                head_activation=str(getattr(pl_module.hparams, "head_activation", "relu")),
                dropout=float(getattr(pl_module.hparams, "dropout", 0.0)),
                use_output_softplus=bool(getattr(pl_module.hparams, "use_output_softplus", True)),
            )

            # copy weights: from pl_module.shared_bottleneck + pl_module.reg_head -> head_module linears
            def collect_linears(m: nn.Module):
                return [mod for mod in m.modules() if isinstance(mod, nn.Linear)]

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
            # ultimate fallback: try existing pl_module.head
            state_dict_to_save = getattr(pl_module, "head").state_dict()

        state: Dict[str, Any] = {
            "state_dict": state_dict_to_save,
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


