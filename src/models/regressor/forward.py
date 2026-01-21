from __future__ import annotations

from typing import Any, List, Optional, Tuple

import torch
from torch import Tensor, nn

from ..layer_utils import fuse_layerwise_predictions


class RegressorForwardMixin:
    def _get_backbone_layer_fusion_weights(
        self,
        *,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> Tensor:
        """
        Return fusion weights over selected backbone layers.

        - mean    : uniform weights
        - learned : softmax(mlp_layer_logits) for the MLP multi-layer path

        For non-MLP heads, fusion is typically implemented inside the head module
        (e.g. ViTDetMultiLayerScalarHead), so this method falls back to uniform weights.
        """
        try:
            L = int(getattr(self, "num_layers", 0) or 0)
        except Exception:
            L = 0
        mode = str(getattr(self, "backbone_layers_fusion", "mean") or "mean").strip().lower()
        logits = getattr(self, "mlp_layer_logits", None)

        if L <= 1:
            w = torch.ones(1, dtype=torch.float32)
        elif mode == "learned" and isinstance(logits, torch.Tensor) and int(logits.numel()) == int(L):
            # Keep weights computation in fp32 for numerical stability, then cast as requested.
            w = torch.softmax(logits.to(dtype=torch.float32), dim=0)
        else:
            w = torch.full((int(L),), 1.0 / float(L), dtype=torch.float32)

        if device is not None:
            w = w.to(device=device)
        if dtype is not None:
            w = w.to(dtype=dtype)
        return w

    def _forward_reg3_logits_for_heads(self, z: Tensor, heads: List[nn.Linear]) -> Tensor:
        """
        Compute main reg3 prediction in normalized domain (g/m^2 or z-score),
        by aggregating three independent scalar heads into a (B, num_outputs) tensor.
        """
        preds: List[Tensor] = []
        for head in heads:
            preds.append(head(z))
        return torch.cat(preds, dim=-1)

    def _forward_reg3_logits(self, z: Tensor) -> Tensor:
        """
        Convenience wrapper for single set of reg3 heads (no layer-wise structure).
        """
        return self._forward_reg3_logits_for_heads(z, list(self.reg3_heads))

    def _compute_reg3_from_images(
        self,
        images: Optional[Tensor] = None,
        pt_tokens: Optional[Tensor] = None,
        cls_token: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Compute main reg3 prediction logits (before optional Softplus) and global bottleneck
        features from input images.

        Returns:
            pred_reg3_logits: (B, num_outputs) in normalized domain (g/m^2 or z-score)
            z_global:         (B, bottleneck_dim) shared bottleneck from CLS+mean(patch)
        """
        if not self.use_patch_reg3:
            # Legacy/global path:
            #  - use_cls_token=True:  CLS concat mean(patch) -> shared_bottleneck -> reg3_heads
            #  - use_cls_token=False: mean(patch)           -> shared_bottleneck -> reg3_heads
            if images is None:
                raise ValueError("images must be provided when use_patch_reg3 is False")
            if self.use_cls_token:
                features = self.feature_extractor(images)
            else:
                # Avoid feeding CLS into the bottleneck by constructing a patch-mean-only feature.
                _, pt = self.feature_extractor.forward_cls_and_tokens(images)
                features = pt.mean(dim=1)  # (B, C)
            z = self.shared_bottleneck(features)
            pred_reg3_logits = self._forward_reg3_logits(z)
            return pred_reg3_logits, z

        # Patch-based path for main regression. When pt_tokens are provided, reuse them
        # directly (e.g., after manifold mixup on backbone patch tokens); otherwise,
        # obtain CLS and patch tokens from the backbone.
        if pt_tokens is None:
            if images is None:
                raise ValueError(
                    "Either images or pt_tokens must be provided in patch-mode reg3"
                )
            cls_tok, pt = self.feature_extractor.forward_cls_and_tokens(images)
        else:
            pt = pt_tokens
            cls_tok = cls_token
        if pt.dim() != 3:
            raise RuntimeError(
                f"Unexpected patch tokens shape in patch-mode reg3: {tuple(pt.shape)}"
            )
        B, N, C = pt.shape
        # Global bottleneck uses only the mean patch token (patch-only, C-dim).
        patch_mean = pt.mean(dim=1)  # (B, C)
        z_patch = self.shared_bottleneck(patch_mean)  # (B, bottleneck_dim)

        # Build per-patch features for the main regression path: each patch token (C-dim)
        # is fed through the shared bottleneck, and predictions are averaged over patches.
        patch_features_flat = pt.reshape(B * N, C)  # (B*N, C)
        z_patches_flat = self.shared_bottleneck(patch_features_flat)  # (B*N, bottleneck_dim)
        pred_patches_flat = self._forward_reg3_logits(z_patches_flat)  # (B*N, num_outputs)
        pred_patches = pred_patches_flat.view(B, N, self.num_outputs)  # (B, N, num_outputs)
        # Average over patches to obtain per-image logits.
        pred_patch = pred_patches.mean(dim=1)  # (B, num_outputs)

        # Optional dual-branch fusion: fuse patch prediction with a global prediction from
        # CLS+mean(patch) using a learnable alpha (stored as a logit).
        try:
            dual_enabled = bool(getattr(self, "dual_branch_enabled", False))
        except Exception:
            dual_enabled = False
        bott_global = getattr(self, "shared_bottleneck_global", None)
        alpha_logit = getattr(self, "dual_branch_alpha_logit", None)
        if dual_enabled and isinstance(bott_global, nn.Module):
            feats_global: Optional[Tensor] = None
            if bool(getattr(self, "use_cls_token", True)):
                if isinstance(cls_tok, Tensor):
                    feats_global = torch.cat([cls_tok, patch_mean], dim=-1)
            else:
                feats_global = patch_mean
            if feats_global is None:
                # Fallback: cannot construct CLS+mean(patch) features without CLS.
                return pred_patch, z_patch

            z_global = bott_global(feats_global)
            pred_global = self._forward_reg3_logits(z_global)
            if isinstance(alpha_logit, torch.Tensor):
                a = torch.sigmoid(alpha_logit).to(device=pred_patch.device, dtype=pred_patch.dtype)
            else:
                a = pred_patch.new_tensor([0.5])
            pred_fused = (a * pred_global) + ((1.0 - a) * pred_patch)
            z_fused = (a.to(device=z_patch.device, dtype=z_patch.dtype) * z_global) + ((1.0 - a.to(device=z_patch.device, dtype=z_patch.dtype)) * z_patch)
            return pred_fused, z_fused

        return pred_patch, z_patch

    def _compute_reg3_and_z_multilayer(
        self,
        images: Optional[Tensor] = None,
        cls_list: Optional[List[Tensor]] = None,
        pt_list: Optional[List[Tensor]] = None,
        feats_list: Optional[List[Tensor]] = None,
    ) -> Tuple[Tensor, Tensor, List[Tensor]]:
        """
        Multi-layer main regression path:
          - For each selected backbone layer, build a bottleneck feature z_l,
          - Apply a layer-specific reg3 head on z_l (or per-patch z_l when use_patch_reg3 is enabled),
          - Average predictions and bottleneck features over layers.

        Args:
            images:     Optional input images (used when cls/pt/feats are not provided).
            cls_list:   Optional list of per-layer CLS tokens (B, C).
            pt_list:    Optional list of per-layer patch tokens (B, N, C).
            feats_list: Optional list of per-layer *global feature vectors* (B, F),
                        where F is either C (mean(patch)) when use_cls_token=False,
                        or 2C ([CLS; mean(patch)]) when use_cls_token=True.

                        This is only supported when use_patch_reg3 is False.

        Returns:
            pred_reg3_logits: (B, num_outputs) averaged over layers
            z_global:         (B, bottleneck_dim) averaged bottleneck over layers
            z_layers:         list of (B, bottleneck_dim) per-layer bottlenecks
        """
        if not self.use_layerwise_heads:
            raise RuntimeError(
                "_compute_reg3_and_z_multilayer called but use_layerwise_heads is False"
            )
        if len(self.backbone_layer_indices) == 0:
            raise RuntimeError("backbone_layer_indices is empty in multi-layer path")

        # Option A: caller provided precomputed per-layer global features (only for non-patch mode).
        if feats_list is not None:
            if self.use_patch_reg3:
                raise RuntimeError("feats_list is unsupported when use_patch_reg3 is True")
            if len(feats_list) != len(self.backbone_layer_indices):
                raise RuntimeError(
                    "Mismatch between feats_list length and backbone_layer_indices in multi-layer path"
                )
            z_layers: List[Tensor] = []
            pred_layers: List[Tensor] = []
            for layer_idx, feats in enumerate(feats_list):
                # Select per-layer bottleneck if available; otherwise fall back to shared one.
                if self.layer_bottlenecks is not None:
                    bottleneck = self.layer_bottlenecks[layer_idx]
                else:
                    bottleneck = self.shared_bottleneck
                z_l = bottleneck(feats)  # (B, bottleneck_dim)
                z_layers.append(z_l)
                # Select layer-specific reg3 heads if available; otherwise fall back to shared heads.
                if self.layer_reg3_heads is not None:
                    heads_l = list(self.layer_reg3_heads[layer_idx])
                else:
                    heads_l = list(self.reg3_heads)
                pred_l = self._forward_reg3_logits_for_heads(z_l, heads_l)  # (B, num_outputs)
                pred_layers.append(pred_l)
            w = self._get_backbone_layer_fusion_weights(
                device=pred_layers[0].device, dtype=pred_layers[0].dtype
            )
            pred_reg3_logits = fuse_layerwise_predictions(pred_layers, weights=w)
            z_global = fuse_layerwise_predictions(z_layers, weights=w)
            return pred_reg3_logits, z_global, z_layers

        # Option B: obtain CLS/patch tokens for all requested layers in a single backbone forward,
        # unless they were already provided (e.g., after manifold mixup on backbone outputs).
        if cls_list is None or pt_list is None:
            if images is None:
                raise ValueError(
                    "Either images or (cls_list, pt_list) must be provided for multi-layer reg3"
                )
            cls_list, pt_list = self.feature_extractor.forward_layers_cls_and_tokens(
                images, self.backbone_layer_indices
            )
        if len(cls_list) != len(pt_list):
            raise RuntimeError("Mismatch between CLS and patch token lists in multi-layer path")

        z_layers: List[Tensor] = []
        pred_layers: List[Tensor] = []

        if not self.use_patch_reg3:
            # Global feature per layer:
            #  - use_cls_token=True:  [CLS ; mean(patch)] -> (B, 2C)
            #  - use_cls_token=False: mean(patch)         -> (B, C)
            for layer_idx, (cls, pt) in enumerate(zip(cls_list, pt_list)):
                if pt.dim() != 3:
                    raise RuntimeError(
                        f"Unexpected patch tokens shape in multi-layer reg3: {tuple(pt.shape)}"
                    )
                patch_mean = pt.mean(dim=1)  # (B, C)
                feats = (
                    torch.cat([cls, patch_mean], dim=-1)
                    if self.use_cls_token
                    else patch_mean
                )
                # Select per-layer bottleneck if available; otherwise fall back to shared one.
                if self.layer_bottlenecks is not None:
                    bottleneck = self.layer_bottlenecks[layer_idx]
                else:
                    bottleneck = self.shared_bottleneck
                z_l = bottleneck(feats)  # (B, bottleneck_dim)
                z_layers.append(z_l)
                # Select layer-specific reg3 heads if available; otherwise fall back to shared heads.
                if self.layer_reg3_heads is not None:
                    heads_l = list(self.layer_reg3_heads[layer_idx])
                else:
                    heads_l = list(self.reg3_heads)
                pred_l = self._forward_reg3_logits_for_heads(z_l, heads_l)  # (B, num_outputs)
                pred_layers.append(pred_l)
        else:
            # Patch-based reg3 per layer: per-patch predictions then averaged, then averaged over layers.
            for layer_idx, (cls, pt) in enumerate(zip(cls_list, pt_list)):
                if pt.dim() != 3:
                    raise RuntimeError(
                        f"Unexpected patch tokens shape in multi-layer patch-mode reg3: {tuple(pt.shape)}"
                    )
                B, N, C = pt.shape
                patch_mean = pt.mean(dim=1)  # (B, C)
                # Select per-layer bottleneck if available; otherwise fall back to shared one.
                if self.layer_bottlenecks is not None:
                    bottleneck = self.layer_bottlenecks[layer_idx]
                else:
                    bottleneck = self.shared_bottleneck
                z_patch_l = bottleneck(patch_mean)  # (B, bottleneck_dim)

                patch_features_flat = pt.reshape(B * N, C)
                z_patches_flat = bottleneck(patch_features_flat)  # (B*N, bottleneck_dim)
                if self.layer_reg3_heads is not None:
                    heads_l = list(self.layer_reg3_heads[layer_idx])
                else:
                    heads_l = list(self.reg3_heads)
                pred_patches_flat = self._forward_reg3_logits_for_heads(z_patches_flat, heads_l)  # (B*N, num_outputs)
                pred_patches = pred_patches_flat.view(B, N, self.num_outputs)
                pred_patch_l = pred_patches.mean(dim=1)  # (B, num_outputs)

                # Optional dual-branch fusion (per-layer): global prediction from CLS+mean(patch)
                # fused with patch prediction.
                try:
                    dual_enabled = bool(getattr(self, "dual_branch_enabled", False))
                except Exception:
                    dual_enabled = False
                bott_global = None
                if dual_enabled:
                    lb_g = getattr(self, "layer_bottlenecks_global", None)
                    if lb_g is not None:
                        try:
                            bott_global = lb_g[layer_idx]
                        except Exception:
                            bott_global = None
                    if bott_global is None:
                        bott_global = getattr(self, "shared_bottleneck_global", None)
                if dual_enabled and isinstance(bott_global, nn.Module):
                    if bool(getattr(self, "use_cls_token", True)):
                        feats_global_l = torch.cat([cls, patch_mean], dim=-1)
                    else:
                        feats_global_l = patch_mean
                    z_global_l = bott_global(feats_global_l)
                    pred_global_l = self._forward_reg3_logits_for_heads(z_global_l, heads_l)
                    alpha_logit = getattr(self, "dual_branch_alpha_logit", None)
                    if isinstance(alpha_logit, torch.Tensor):
                        a = torch.sigmoid(alpha_logit).to(device=pred_patch_l.device, dtype=pred_patch_l.dtype)
                    else:
                        a = pred_patch_l.new_tensor([0.5])
                    pred_l = (a * pred_global_l) + ((1.0 - a) * pred_patch_l)
                    a_z = a.to(device=z_patch_l.device, dtype=z_patch_l.dtype)
                    z_l = (a_z * z_global_l) + ((1.0 - a_z) * z_patch_l)
                else:
                    pred_l = pred_patch_l
                    z_l = z_patch_l

                z_layers.append(z_l)
                pred_layers.append(pred_l)

        w = self._get_backbone_layer_fusion_weights(
            device=pred_layers[0].device, dtype=pred_layers[0].dtype
        )
        pred_reg3_logits = fuse_layerwise_predictions(pred_layers, weights=w)
        z_global = fuse_layerwise_predictions(z_layers, weights=w)
        return pred_reg3_logits, z_global, z_layers

    def forward(self, images: Tensor) -> Tensor:
        # Return main regression prediction in original grams (g).
        # When using the FPN/DPT heads, main reg3 is computed from patch tokens.
        head_type = str(getattr(self, "_head_type", "mlp")).lower()
        if head_type == "fpn":
            if self.use_layerwise_heads and len(self.backbone_layer_indices) > 0:
                try:
                    _, pt_list = self.feature_extractor.forward_layers_cls_and_tokens(  # type: ignore[misc]
                        images, self.backbone_layer_indices
                    )
                except Exception as e:
                    backbone_name = getattr(getattr(self, "hparams", None), "backbone_name", None)
                    raise RuntimeError(
                        "Failed to extract multi-layer patch tokens for FPN head in forward(). "
                        f"Check model.backbone_layers.indices={self.backbone_layer_indices} "
                        f"are valid for backbone={backbone_name!r}. Original error: {e}"
                    ) from e
                if not pt_list:
                    raise RuntimeError(
                        "Multi-layer token extraction returned an empty pt_list for FPN head in forward(). "
                        f"indices={self.backbone_layer_indices}"
                    )
                pt_in: Any = pt_list
            else:
                _, pt = self.feature_extractor.forward_cls_and_tokens(images)
                pt_in = pt
            out_dict = self.fpn_head(pt_in, image_hw=(int(images.shape[-2]), int(images.shape[-1])))  # type: ignore[attr-defined]
            pred_reg3_logits = out_dict["reg3"]  # type: ignore[assignment]
            if pred_reg3_logits is None:
                raise RuntimeError("FPN head did not return reg3 logits")
        elif head_type == "vitdet":
            image_hw = (int(images.shape[-2]), int(images.shape[-1]))
            if self.use_layerwise_heads and len(self.backbone_layer_indices) > 0:
                try:
                    _, pt_list = self.feature_extractor.forward_layers_cls_and_tokens(  # type: ignore[misc]
                        images, self.backbone_layer_indices
                    )
                except Exception as e:
                    backbone_name = getattr(getattr(self, "hparams", None), "backbone_name", None)
                    raise RuntimeError(
                        "Failed to extract multi-layer patch tokens for ViTDet head in forward(). "
                        f"Check model.backbone_layers.indices={self.backbone_layer_indices} "
                        f"are valid for backbone={backbone_name!r}. Original error: {e}"
                    ) from e
                if not pt_list:
                    raise RuntimeError(
                        "Multi-layer token extraction returned an empty pt_list for ViTDet head in forward(). "
                        f"indices={self.backbone_layer_indices}"
                    )
                pt_in: Any = pt_list
            else:
                _, pt = self.feature_extractor.forward_cls_and_tokens(images)
                pt_in = pt
            out_dict = self.vitdet_head(pt_in, image_hw=image_hw)  # type: ignore[attr-defined]
            pred_reg3_logits = out_dict.get("reg3", None)  # type: ignore[assignment]
            if pred_reg3_logits is None:
                raise RuntimeError("ViTDet head did not return reg3 logits")
        elif head_type == "mamba":
            image_hw = (int(images.shape[-2]), int(images.shape[-1]))
            if self.use_layerwise_heads and len(self.backbone_layer_indices) > 0:
                try:
                    _, pt_list = self.feature_extractor.forward_layers_cls_and_tokens(  # type: ignore[misc]
                        images, self.backbone_layer_indices
                    )
                except Exception as e:
                    backbone_name = getattr(getattr(self, "hparams", None), "backbone_name", None)
                    raise RuntimeError(
                        "Failed to extract multi-layer patch tokens for Mamba head in forward(). "
                        f"Check model.backbone_layers.indices={self.backbone_layer_indices} "
                        f"are valid for backbone={backbone_name!r}. Original error: {e}"
                    ) from e
                if not pt_list:
                    raise RuntimeError(
                        "Multi-layer token extraction returned an empty pt_list for Mamba head in forward(). "
                        f"indices={self.backbone_layer_indices}"
                    )
                pt_in: Any = pt_list
            else:
                _, pt = self.feature_extractor.forward_cls_and_tokens(images)
                pt_in = pt
            out_dict = self.mamba_head(pt_in, image_hw=image_hw)  # type: ignore[attr-defined]
            pred_reg3_logits = out_dict.get("reg3", None)  # type: ignore[assignment]
            if pred_reg3_logits is None:
                raise RuntimeError("Mamba head did not return reg3 logits")
        elif head_type == "dpt":
            image_hw = (int(images.shape[-2]), int(images.shape[-1]))
            if self.use_layerwise_heads and len(self.backbone_layer_indices) > 0:
                try:
                    cls_list, pt_list = self.feature_extractor.forward_layers_cls_and_tokens(  # type: ignore[misc]
                        images, self.backbone_layer_indices
                    )
                except Exception as e:
                    backbone_name = getattr(getattr(self, "hparams", None), "backbone_name", None)
                    raise RuntimeError(
                        "Failed to extract multi-layer CLS/patch tokens for DPT head in forward(). "
                        f"Check model.backbone_layers.indices={self.backbone_layer_indices} "
                        f"are valid for backbone={backbone_name!r}. Original error: {e}"
                    ) from e
                if not pt_list:
                    raise RuntimeError(
                        "Multi-layer token extraction returned an empty pt_list for DPT head in forward(). "
                        f"indices={self.backbone_layer_indices}"
                    )
                out_dict = self.dpt_head(cls_list, pt_list, image_hw=image_hw)  # type: ignore[call-arg]
            else:
                cls_tok, pt = self.feature_extractor.forward_cls_and_tokens(images)
                out_dict = self.dpt_head(cls_tok, pt, image_hw=image_hw)  # type: ignore[call-arg]
            pred_reg3_logits = out_dict.get("reg3", None)  # type: ignore[assignment]
            if pred_reg3_logits is None:
                raise RuntimeError("DPT head did not return reg3 logits")
        elif head_type == "eomt":
            # EoMT-style injected-query head (matches `third_party/eomt`):
            # run backbone up to (depth - k) blocks, prepend queries, then run last-k blocks jointly.
            try:
                bb = getattr(self.feature_extractor, "backbone", None)
                bb0 = bb
                if bb0 is not None and (not hasattr(bb0, "blocks")):
                    base_model = getattr(bb0, "base_model", None)
                    if isinstance(base_model, nn.Module):
                        cand = getattr(base_model, "model", None)
                        if isinstance(cand, nn.Module) and hasattr(cand, "blocks"):
                            bb0 = cand
                        elif hasattr(base_model, "blocks"):
                            bb0 = base_model
                    cand2 = getattr(bb0, "model", None)
                    if isinstance(cand2, nn.Module) and hasattr(cand2, "blocks"):
                        bb0 = cand2
                blocks = getattr(bb0, "blocks", None)
                depth = len(blocks) if isinstance(blocks, (nn.ModuleList, list)) else 0
            except Exception:
                depth = 0
            if depth <= 0:
                raise RuntimeError("EoMT injected-query head requires a DINOv3 ViT backbone with `.blocks`")

            k_blocks = int(getattr(self, "eomt_num_layers", 4) or 4)
            k_blocks = int(max(0, min(k_blocks, depth)))
            start_block = int(depth - k_blocks)

            x_base, (H_p, W_p) = self.feature_extractor.forward_tokens_until_block(
                images, block_idx=int(start_block)
            )
            out_dict = self.eomt_head(  # type: ignore[attr-defined]
                x_base,
                backbone=getattr(self.feature_extractor, "backbone"),
                patch_hw=(int(H_p), int(W_p)),
            )
            pred_reg3_logits = out_dict.get("reg3", None)  # type: ignore[assignment]
            if pred_reg3_logits is None:
                raise RuntimeError("EoMT head did not return reg3 logits")
        else:
            # Legacy behavior:
            # When use_patch_reg3 is enabled, this corresponds to the per-patch prediction
            # averaged over all patches; otherwise it is the legacy CLS+mean(patch) head.
            if self.use_layerwise_heads:
                pred_reg3_logits, _, _ = self._compute_reg3_and_z_multilayer(images)
            else:
                pred_reg3_logits, _ = self._compute_reg3_from_images(images)
        out = pred_reg3_logits
        if self.out_softplus is not None:
            out = self.out_softplus(out)
        # Invert normalization and scaling to grams
        out_g = self._invert_reg3_to_grams(out)
        return out_g

    def _invert_reg3_to_g_per_m2(self, vals: Tensor) -> Tensor:
        x = vals
        if self._use_reg3_zscore and self._reg3_mean is not None and self._reg3_std is not None:
            safe_std = torch.clamp(self._reg3_std.to(x.device, dtype=x.dtype), min=1e-8)
            x = x * safe_std + self._reg3_mean.to(x.device, dtype=x.dtype)
        if self.log_scale_targets:
            x = torch.expm1(x).clamp_min(0.0)
        return x

    def _invert_reg3_to_grams(self, vals: Tensor) -> Tensor:
        gm2 = self._invert_reg3_to_g_per_m2(vals)
        return gm2 * float(self._area_m2)


