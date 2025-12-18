from __future__ import annotations

import os
from typing import Any, Dict

import torch
from lightning.pytorch.callbacks import Callback
from torch import nn

from src.models.head_builder import build_head_layer, MultiLayerHeadExport
from src.models.spatial_fpn import FPNHeadConfig, FPNScalarHead
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

        # Build an inference head state_dict: shared bottleneck + main reg head
        # (+ optional ratio head, packed into a single linear layer).
        state_dict_to_save: Dict[str, Any]
        try:
            head_type = str(getattr(pl_module.hparams, "head_type", "mlp")) if hasattr(pl_module, "hparams") else "mlp"
            head_type = head_type.strip().lower()
            if head_type in ("fpn", "fpn_scalar", "spatial_fpn"):
                # ---------------------------
                # FPN head (Phase A) export
                # ---------------------------
                num_outputs_main = int(getattr(pl_module.hparams, "num_outputs", 1)) if hasattr(pl_module, "hparams") else 1
                # Enable ratio outputs if the training module had ratio head enabled
                enable_ratio = bool(getattr(pl_module, "enable_ratio_head", False))
                num_ratio_outputs = 3 if enable_ratio else 0
                head_total_outputs = int(num_outputs_main + num_ratio_outputs)

                embedding_dim = int(getattr(pl_module.hparams, "embedding_dim", 1024)) if hasattr(pl_module, "hparams") else 1024
                fpn_dim = int(getattr(pl_module.hparams, "fpn_dim", 256)) if hasattr(pl_module, "hparams") else 256
                fpn_num_levels = int(getattr(pl_module.hparams, "fpn_num_levels", 3)) if hasattr(pl_module, "hparams") else 3
                fpn_patch_size = int(getattr(pl_module.hparams, "fpn_patch_size", 16)) if hasattr(pl_module, "hparams") else 16

                head_hidden_dims = list(getattr(pl_module.hparams, "head_hidden_dims", [])) if hasattr(pl_module, "hparams") else []
                head_activation = str(getattr(pl_module.hparams, "head_activation", "relu")) if hasattr(pl_module, "hparams") else "relu"
                dropout = float(getattr(pl_module.hparams, "dropout", 0.0)) if hasattr(pl_module, "hparams") else 0.0

                # Prefer the module attributes (normalized) over raw hparams.
                use_layerwise_heads = bool(getattr(pl_module, "use_layerwise_heads", False))
                backbone_layer_indices = list(getattr(pl_module, "backbone_layer_indices", []))
                use_separate_bottlenecks = bool(getattr(pl_module, "use_separate_bottlenecks", False))

                num_layers_eff = max(1, len(backbone_layer_indices)) if use_layerwise_heads else 1
                enable_ndvi = bool(getattr(pl_module, "enable_ndvi", False))

                # Save the head module state_dict directly (it is already lightweight).
                fpn_head = getattr(pl_module, "fpn_head", None)
                if fpn_head is None or not isinstance(fpn_head, nn.Module):
                    raise RuntimeError(
                        "FPN head export failed: expected `pl_module.fpn_head` to be an nn.Module. "
                        "This likely indicates a mismatched head_type or an unexpected LightningModule structure."
                    )
                state_dict_to_save = fpn_head.state_dict()

                # Build minimal meta and return early.
                state: Dict[str, Any] = {
                    "state_dict": state_dict_to_save,
                    "meta": {
                        "head_type": "fpn",
                        "backbone": getattr(pl_module.hparams, "backbone_name", None) if hasattr(pl_module, "hparams") else None,
                        "embedding_dim": embedding_dim,
                        "fpn_dim": fpn_dim,
                        "fpn_num_levels": fpn_num_levels,
                        "fpn_patch_size": fpn_patch_size,
                        "enable_ndvi": bool(enable_ndvi),
                        "num_outputs_main": int(num_outputs_main),
                        "num_outputs_ratio": int(num_ratio_outputs),
                        "head_total_outputs": int(head_total_outputs),
                        "head_hidden_dims": list(head_hidden_dims),
                        "head_activation": head_activation,
                        "head_dropout": float(dropout),
                        # Export without terminal Softplus; inference handles main outputs.
                        "use_output_softplus": False,
                        "log_scale_targets": bool(getattr(pl_module.hparams, "log_scale_targets", False)) if hasattr(pl_module, "hparams") else False,
                        # Patch tokens are required for FPN heads.
                        "use_patch_reg3": True,
                        "use_cls_token": bool(getattr(pl_module.hparams, "use_cls_token", True)) if hasattr(pl_module, "hparams") else True,
                        "use_layerwise_heads": bool(use_layerwise_heads),
                        "backbone_layer_indices": list(backbone_layer_indices),
                        "use_separate_bottlenecks": bool(use_separate_bottlenecks),
                        "ratio_components": ["Dry_Clover_g", "Dry_Dead_g", "Dry_Green_g"] if num_ratio_outputs > 0 else [],
                    },
                }
                # Optionally bundle LoRA payload
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
                return

            # Prefer composing a fresh head module to ensure compatibility with inference script.
            # When a ratio head is present, we pack both Dry_Total_g (main regression)
            # and ratio logits into a single linear layer with out_features =
            # num_outputs_main + num_ratio_outputs.
            num_outputs_main = int(getattr(pl_module.hparams, "num_outputs", 1))
            ratio_head = getattr(pl_module, "ratio_head", None)
            num_ratio_outputs = int(ratio_head.out_features) if isinstance(ratio_head, nn.Linear) else 0
            head_total_outputs = int(num_outputs_main + num_ratio_outputs)

            # For the packed head we disable Softplus in the module itself and
            # leave any required non-linearity to the inference script, which
            # handles only the main regression output.
            use_output_softplus_eff = False
            # Patch-mode heads use patch-token dimensionality as input (embedding_dim),
            # whereas legacy heads expect CLS+mean(patch) with 2 * embedding_dim.
            use_patch_reg3 = bool(getattr(pl_module.hparams, "use_patch_reg3", False)) if hasattr(pl_module, "hparams") else False
            # Whether the global feature includes CLS token (2C) or uses patch-mean only (C).
            use_cls_token = bool(getattr(pl_module.hparams, "use_cls_token", True)) if hasattr(pl_module, "hparams") else True
            use_layerwise_heads = bool(getattr(pl_module.hparams, "use_layerwise_heads", False)) if hasattr(pl_module, "hparams") else False
            backbone_layer_indices = list(getattr(pl_module.hparams, "backbone_layer_indices", [])) if hasattr(pl_module, "hparams") else []
            use_separate_bottlenecks = bool(
                getattr(pl_module.hparams, "use_separate_bottlenecks", False)
            ) if hasattr(pl_module, "hparams") else False

            # For backward compatibility, when layer-wise heads are disabled we continue
            # to use a simple MLP head built via build_head_layer. When layer-wise heads
            # are enabled and separate bottlenecks are *not* used, we still export a
            # packed MLP head whose final Linear contains concatenated per-layer weights.
            # When both layer-wise heads and separate bottlenecks are enabled, we export
            # a dedicated MultiLayerHeadExport that mirrors the training-time structure.
            if use_layerwise_heads and use_separate_bottlenecks and hasattr(pl_module, "layer_bottlenecks"):
                # Multi-layer + per-layer bottlenecks: export explicit per-layer MLPs.
                embedding_dim = int(getattr(pl_module.hparams, "embedding_dim", 1024))
                head_hidden_dims = list(getattr(pl_module.hparams, "head_hidden_dims", []))
                head_activation = str(getattr(pl_module.hparams, "head_activation", "relu"))
                dropout = float(getattr(pl_module.hparams, "dropout", 0.0))

                layer_bottlenecks = list(getattr(pl_module, "layer_bottlenecks", []))
                layer_reg3_heads = list(getattr(pl_module, "layer_reg3_heads", []))
                layer_ratio_heads = (
                    list(getattr(pl_module, "layer_ratio_heads"))
                    if hasattr(pl_module, "layer_ratio_heads") and getattr(pl_module, "layer_ratio_heads") is not None
                    else []
                )
                num_layers = len(layer_bottlenecks)
                if num_layers == 0:
                    raise RuntimeError(
                        "use_layerwise_heads and use_separate_bottlenecks are True, but layer_bottlenecks is empty."
                    )

                head_module = MultiLayerHeadExport(
                    embedding_dim=embedding_dim,
                    num_outputs_main=num_outputs_main,
                    num_outputs_ratio=num_ratio_outputs,
                    head_hidden_dims=head_hidden_dims,
                    head_activation=head_activation,
                    dropout=dropout,
                    use_patch_reg3=use_patch_reg3,
                    use_cls_token=use_cls_token,
                    num_layers=num_layers,
                )

                # Copy per-layer bottlenecks and heads 1:1 into the export module.
                with torch.no_grad():
                    # Bottlenecks
                    for idx in range(num_layers):
                        src_b = layer_bottlenecks[idx]
                        dst_b = head_module.layer_bottlenecks[idx]
                        dst_b.load_state_dict(src_b.state_dict())

                    # reg3 heads
                    for l_idx in range(num_layers):
                        src_reg3_list = list(layer_reg3_heads[l_idx])
                        dst_reg3_list = list(head_module.layer_reg3_heads[l_idx])
                        if len(src_reg3_list) != len(dst_reg3_list):
                            raise RuntimeError("Mismatch in number of reg3 heads per layer during export")
                        for s, t in zip(src_reg3_list, dst_reg3_list):
                            if not isinstance(s, nn.Linear) or not isinstance(t, nn.Linear) or s.out_features != t.out_features:
                                raise RuntimeError("Expected Linear reg3 heads when exporting MultiLayerHeadExport")
                            t.weight.copy_(s.weight.data)
                            if t.bias is not None and s.bias is not None:
                                t.bias.copy_(s.bias.data)

                    # Ratio heads (optional)
                    if num_ratio_outputs > 0 and head_module.layer_ratio_heads is not None:
                        if layer_ratio_heads and len(layer_ratio_heads) != num_layers:
                            raise RuntimeError("Mismatch in number of ratio heads per layer during export")
                        for l_idx in range(num_layers):
                            # Prefer layer-specific ratio heads when available; otherwise fall back to shared ratio_head.
                            if l_idx < len(layer_ratio_heads) and isinstance(layer_ratio_heads[l_idx], nn.Linear):
                                src_r = layer_ratio_heads[l_idx]
                            elif isinstance(ratio_head, nn.Linear):
                                src_r = ratio_head
                            else:
                                raise RuntimeError("Missing ratio head for MultiLayerHeadExport")
                            dst_r = head_module.layer_ratio_heads[l_idx]
                            if not isinstance(dst_r, nn.Linear) or src_r.out_features != dst_r.out_features:
                                raise RuntimeError("ratio_head out_features mismatch during export")
                            dst_r.weight.copy_(src_r.weight.data)
                            if dst_r.bias is not None and src_r.bias is not None:
                                dst_r.bias.copy_(src_r.bias.data)

                state_dict_to_save = head_module.state_dict()

            else:
                head_module = build_head_layer(
                    embedding_dim=int(getattr(pl_module.hparams, "embedding_dim", 1024)),
                    num_outputs=head_total_outputs
                    if not use_layerwise_heads
                    else head_total_outputs * max(1, len(backbone_layer_indices)),
                    head_hidden_dims=list(getattr(pl_module.hparams, "head_hidden_dims", [])),
                    head_activation=str(getattr(pl_module.hparams, "head_activation", "relu")),
                    dropout=float(getattr(pl_module.hparams, "dropout", 0.0)),
                    use_output_softplus=use_output_softplus_eff,
                    # Patch-mode uses C. Global mode uses 2C when CLS is included, otherwise C.
                    input_dim=int(getattr(pl_module.hparams, "embedding_dim", 1024))
                    if (use_patch_reg3 or (not use_cls_token))
                    else None,
                )

            def collect_linears(m: nn.Module):
                return [mod for mod in m.modules() if isinstance(mod, nn.Linear)]

            # New-style model: shared_bottleneck + reg3_heads (one or more independent 1-d heads)
            if hasattr(pl_module, "shared_bottleneck") and hasattr(pl_module, "reg3_heads") and not (
                use_layerwise_heads and use_separate_bottlenecks and hasattr(pl_module, "layer_bottlenecks")
            ):
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
                # into a single Linear(out_features=head_total_outputs * num_layers_when_applicable).
                final_linear = tgt_linears[-1]
                if use_layerwise_heads and hasattr(pl_module, "layer_reg3_heads") and getattr(pl_module, "layer_reg3_heads") is not None:
                    # Layer-wise packing: we stack per-layer heads along the output dimension.
                    layer_reg3_heads = list(getattr(pl_module, "layer_reg3_heads"))
                    layer_ratio_heads = list(getattr(pl_module, "layer_ratio_heads")) if hasattr(pl_module, "layer_ratio_heads") and getattr(pl_module, "layer_ratio_heads") is not None else []
                    num_layers = len(layer_reg3_heads)
                    expected_out = head_total_outputs * num_layers
                    if final_linear.out_features != expected_out:
                        raise RuntimeError("Final inference head out_features does not match packed layer-wise outputs")
                    with torch.no_grad():
                        row = 0
                        # For each layer, pack its reg3 heads then optional ratio head.
                        for l_idx, reg3_list in enumerate(layer_reg3_heads):
                            for h in reg3_list:
                                if not isinstance(h, nn.Linear) or h.out_features != 1:
                                    raise RuntimeError("layer_reg3_heads must contain nn.Linear modules with out_features=1")
                                final_linear.weight[row : row + 1, :].copy_(h.weight.data)
                                if final_linear.bias is not None and h.bias is not None:
                                    final_linear.bias[row] = h.bias.data[0]
                                row += 1
                            if num_ratio_outputs > 0:
                                # Use layer-specific ratio heads when available; otherwise fall back to shared ratio_head.
                                ratio_src = None
                                if l_idx < len(layer_ratio_heads) and isinstance(layer_ratio_heads[l_idx], nn.Linear):
                                    ratio_src = layer_ratio_heads[l_idx]
                                elif isinstance(ratio_head, nn.Linear):
                                    ratio_src = ratio_head
                                if ratio_src is None or ratio_src.out_features != num_ratio_outputs:
                                    raise RuntimeError("ratio_head / layer_ratio_heads out_features mismatch for packing")
                                final_linear.weight[row : row + num_ratio_outputs, :].copy_(ratio_src.weight.data)
                                if final_linear.bias is not None and ratio_src.bias is not None:
                                    final_linear.bias[row : row + num_ratio_outputs].copy_(ratio_src.bias.data)
                                row += num_ratio_outputs
                    state_dict_to_save = head_module.state_dict()
                else:
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
                "head_hidden_dims": list(getattr(pl_module.hparams, "head_hidden_dims", [])) if hasattr(pl_module, "hparams") else [],
                "head_activation": getattr(pl_module.hparams, "head_activation", "relu") if hasattr(pl_module, "hparams") else "relu",
                "head_dropout": float(getattr(pl_module.hparams, "dropout", 0.0)) if hasattr(pl_module, "hparams") else 0.0,
                # Softplus is applied (if needed) in the training module; the packed head
                # used for offline inference is exported without a terminal Softplus.
                "use_output_softplus": False,
                "log_scale_targets": bool(getattr(pl_module.hparams, "log_scale_targets", False)) if hasattr(pl_module, "hparams") else False,
                # Whether the main regression head was trained using per-patch predictions
                # averaged over patches (scheme A), or using a single CLS+mean(patch) feature.
                "use_patch_reg3": bool(getattr(pl_module.hparams, "use_patch_reg3", False)) if hasattr(pl_module, "hparams") else False,
                # Whether global features included CLS token during training.
                "use_cls_token": bool(getattr(pl_module.hparams, "use_cls_token", True)) if hasattr(pl_module, "hparams") else True,
                # Multi-layer configuration: whether layer-wise heads were used and which
                # backbone blocks were selected.
                "use_layerwise_heads": bool(getattr(pl_module.hparams, "use_layerwise_heads", False)) if hasattr(pl_module, "hparams") else False,
                "backbone_layer_indices": list(getattr(pl_module.hparams, "backbone_layer_indices", [])) if hasattr(pl_module, "hparams") else [],
                # Whether per-layer bottlenecks were used together with layer-wise heads.
                "use_separate_bottlenecks": bool(getattr(pl_module.hparams, "use_separate_bottlenecks", False)) if hasattr(pl_module, "hparams") else False,
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


