from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch
from torch import Tensor, nn


class PredictionMixin:
    """
    Compute model predictions (reg3 logits + shared representation z) for a batch.

    This contains the head-type specific forward paths used during training/validation
    steps, including optional manifold mixup on backbone tokens/features.
    """

    @staticmethod
    def _tensor_meta(t: Tensor) -> str:
        return f"shape={tuple(t.shape)}, dtype={t.dtype}, device={t.device}"

    @staticmethod
    def _list_tensor_shapes(xs: Optional[List[Tensor]]) -> str:
        if xs is None:
            return "None"
        try:
            return "[" + ", ".join(str(tuple(x.shape)) for x in xs) + "]"
        except Exception:
            return "<unavailable>"

    @staticmethod
    def _batch_target_shapes(batch: Dict[str, Tensor]) -> str:
        keys = (
            "y_reg3",
            "reg3_mask",
            "y_height",
            "y_ndvi",
            "ndvi_mask",
            "y_biomass_5d_g",
            "biomass_5d_mask",
            "y_ratio",
            "ratio_mask",
        )
        parts: List[str] = []
        for k in keys:
            v = batch.get(k, None)
            if isinstance(v, torch.Tensor):
                parts.append(f"{k}={tuple(v.shape)}")
        return "{" + ", ".join(parts) + "}"

    def _manifold_mixup_cfg_str(self) -> str:
        mm = getattr(self, "_manifold_mixup", None)
        if mm is None:
            return "<none>"
        apply_on = getattr(self, "_manifold_mixup_apply_on", None)
        return (
            f"enabled={getattr(mm, 'enabled', None)}, "
            f"prob={getattr(mm, 'prob', None)}, "
            f"alpha={getattr(mm, 'alpha', None)}, "
            f"mix_cls_token={getattr(mm, 'mix_cls_token', None)}, "
            f"detach_pair={getattr(mm, 'detach_pair', None)}, "
            f"apply_on={apply_on}"
        )

    def _raise_manifold_mixup_error(
        self,
        *,
        context: str,
        err: Exception,
        batch: Dict[str, Tensor],
        x: Optional[Tensor] = None,
        pt_list: Optional[List[Tensor]] = None,
        cls_list: Optional[List[Tensor]] = None,
    ) -> None:
        head_type = str(getattr(self, "_head_type", "mlp")).lower()
        try:
            layer_indices: object = list(getattr(self, "backbone_layer_indices", []))
        except Exception:
            layer_indices = "<unavailable>"
        flags = (
            f"use_layerwise_heads={bool(getattr(self, 'use_layerwise_heads', False))}, "
            f"use_patch_reg3={bool(getattr(self, 'use_patch_reg3', False))}, "
            f"use_cls_token={bool(getattr(self, 'use_cls_token', False))}, "
            f"backbone_layer_indices={layer_indices}"
        )
        msg = (
            "Manifold mixup failed. "
            f"context={context!r}, head_type={head_type!r}, "
            f"flags=({flags}), mixup_cfg=({self._manifold_mixup_cfg_str()}). "
        )
        if x is not None:
            msg += f"x({self._tensor_meta(x)}). "
        if pt_list is not None:
            msg += f"pt_list_shapes={self._list_tensor_shapes(pt_list)}. "
        if cls_list is not None:
            msg += f"cls_list_shapes={self._list_tensor_shapes(cls_list)}. "
        msg += f"batch_targets={self._batch_target_shapes(batch)}. original_error={err!r}"
        raise RuntimeError(msg) from err

    def _predict_reg3_and_z(
        self,
        *,
        images: Tensor,
        batch: Dict[str, Tensor],
        stage: str,
        use_mixup: bool,
        is_ndvi_only: bool,
        mixup_lam: Optional[float] = None,
        mixup_perm: Optional[Tensor] = None,
        mixup_mix_labels: bool = True,
    ) -> Tuple[
        Tensor,
        Tensor,
        Optional[List[Tensor]],
        Optional[Tensor],
        Optional[Tensor],
        Dict[str, Tensor],
        Optional[List[Tensor]],
        Optional[List[Tensor]],
    ]:
        """
        Returns:
            pred_reg3: (B, num_outputs) logits in normalized domain (pre-softplus)
            z:         (B, D) shared global representation
            z_layers:  Optional[list[(B, D)]] per-layer z (layerwise heads)
            ratio_logits_pred: Optional[(B,R)] ratio outputs for FPN/DPT heads
            ndvi_pred: Optional[(B,1)] NDVI pred for FPN/DPT heads
            batch: possibly-updated batch (mixup mutates labels)
            pred_reg3_layers: Optional[list[(B, num_outputs)]] per-layer reg3 logits (for multi-layer ViTDet)
            ratio_logits_layers: Optional[list[(B,R)]] per-layer ratio outputs (for multi-layer ViTDet)
        """
        pred_reg3: Tensor
        z: Tensor
        z_layers: Optional[List[Tensor]] = None
        ratio_logits_pred: Optional[Tensor] = None
        ndvi_pred: Optional[Tensor] = None
        pred_reg3_layers: Optional[List[Tensor]] = None
        ratio_logits_layers: Optional[List[Tensor]] = None

        head_type = str(getattr(self, "_head_type", "mlp")).lower()

        if head_type == "fpn":
            # Phase-A FPN path: always consume patch tokens (single-layer or multi-layer),
            # optionally apply manifold mixup on patch tokens, then predict reg3/ratio/ndvi.
            image_hw = (int(images.shape[-2]), int(images.shape[-1]))
            pt_in: Any
            if self.use_layerwise_heads and len(self.backbone_layer_indices) > 0:
                try:
                    _cls_list, pt_list = self.feature_extractor.forward_layers_cls_and_tokens(
                        images, self.backbone_layer_indices
                    )
                except Exception as e:
                    backbone_name = getattr(getattr(self, "hparams", None), "backbone_name", None)
                    raise RuntimeError(
                        "Failed to extract multi-layer patch tokens for FPN head. "
                        f"Check model.backbone_layers.indices={self.backbone_layer_indices} "
                        f"are valid for backbone={backbone_name!r}. Original error: {e}"
                    ) from e
                if not pt_list:
                    raise RuntimeError(
                        "Multi-layer token extraction returned an empty pt_list for FPN head. "
                        f"indices={self.backbone_layer_indices}"
                    )
                if (
                    use_mixup
                    and self._manifold_mixup is not None
                    and (stage == "train")
                    and (not is_ndvi_only)
                    and pt_list
                ):
                    try:
                        pt_stack = torch.stack(pt_list, dim=1)  # (B, L, N, C)
                        pt_stack, batch, _ = self._manifold_mixup.apply(
                            pt_stack,
                            batch,
                            force=True,
                            lam=mixup_lam,
                            perm=mixup_perm,
                            mix_labels=mixup_mix_labels,
                        )
                        pt_list = [pt_stack[:, i] for i in range(pt_stack.shape[1])]
                    except Exception as e:
                        self._raise_manifold_mixup_error(
                            context="fpn/multilayer/patch_tokens",
                            err=e,
                            batch=batch,
                            pt_list=pt_list,
                        )
                pt_in = pt_list
            else:
                _, pt_tokens = self.feature_extractor.forward_cls_and_tokens(images)
                if (
                    use_mixup
                    and self._manifold_mixup is not None
                    and (stage == "train")
                    and (not is_ndvi_only)
                ):
                    try:
                        pt_tokens, batch, _ = self._manifold_mixup.apply(
                            pt_tokens,
                            batch,
                            force=True,
                            lam=mixup_lam,
                            perm=mixup_perm,
                            mix_labels=mixup_mix_labels,
                        )
                    except Exception as e:
                        self._raise_manifold_mixup_error(
                            context="fpn/singlelayer/patch_tokens",
                            err=e,
                            batch=batch,
                            x=pt_tokens,
                        )
                pt_in = pt_tokens

            out_fpn = self.fpn_head(pt_in, image_hw=image_hw)  # type: ignore[attr-defined]
            pred_reg3 = out_fpn.get("reg3", None)  # type: ignore[assignment]
            z = out_fpn.get("z", None)  # type: ignore[assignment]
            ratio_logits_pred = out_fpn.get("ratio", None)
            ndvi_pred = out_fpn.get("ndvi", None)
            if pred_reg3 is None or z is None:
                raise RuntimeError("FPN head did not return required outputs (reg3/z)")

        elif head_type == "vitdet":
            # ViTDet-style simple feature pyramid head:
            # consume patch tokens (single-layer or multi-layer), optionally apply manifold mixup,
            # then predict reg3/ratio/ndvi.
            image_hw = (int(images.shape[-2]), int(images.shape[-1]))
            pt_in: Any
            if self.use_layerwise_heads and len(self.backbone_layer_indices) > 0:
                try:
                    _cls_list, pt_list = self.feature_extractor.forward_layers_cls_and_tokens(
                        images, self.backbone_layer_indices
                    )
                except Exception as e:
                    backbone_name = getattr(getattr(self, "hparams", None), "backbone_name", None)
                    raise RuntimeError(
                        "Failed to extract multi-layer patch tokens for ViTDet head. "
                        f"Check model.backbone_layers.indices={self.backbone_layer_indices} "
                        f"are valid for backbone={backbone_name!r}. Original error: {e}"
                    ) from e
                if not pt_list:
                    raise RuntimeError(
                        "Multi-layer token extraction returned an empty pt_list for ViTDet head. "
                        f"indices={self.backbone_layer_indices}"
                    )
                if (
                    use_mixup
                    and self._manifold_mixup is not None
                    and (stage == "train")
                    and (not is_ndvi_only)
                    and pt_list
                ):
                    try:
                        pt_stack = torch.stack(pt_list, dim=1)  # (B, L, N, C)
                        pt_stack, batch, _ = self._manifold_mixup.apply(
                            pt_stack,
                            batch,
                            force=True,
                            lam=mixup_lam,
                            perm=mixup_perm,
                            mix_labels=mixup_mix_labels,
                        )
                        pt_list = [pt_stack[:, i] for i in range(pt_stack.shape[1])]
                    except Exception as e:
                        self._raise_manifold_mixup_error(
                            context="vitdet/multilayer/patch_tokens",
                            err=e,
                            batch=batch,
                            pt_list=pt_list,
                        )
                pt_in = pt_list
            else:
                _, pt_tokens = self.feature_extractor.forward_cls_and_tokens(images)
                if (
                    use_mixup
                    and self._manifold_mixup is not None
                    and (stage == "train")
                    and (not is_ndvi_only)
                ):
                    try:
                        pt_tokens, batch, _ = self._manifold_mixup.apply(
                            pt_tokens,
                            batch,
                            force=True,
                            lam=mixup_lam,
                            perm=mixup_perm,
                            mix_labels=mixup_mix_labels,
                        )
                    except Exception as e:
                        self._raise_manifold_mixup_error(
                            context="vitdet/singlelayer/patch_tokens",
                            err=e,
                            batch=batch,
                            x=pt_tokens,
                        )
                pt_in = pt_tokens

            out_vitdet = self.vitdet_head(pt_in, image_hw=image_hw)  # type: ignore[attr-defined]
            pred_reg3 = out_vitdet.get("reg3", None)  # type: ignore[assignment]
            z = out_vitdet.get("z", None)  # type: ignore[assignment]
            ratio_logits_pred = out_vitdet.get("ratio", None)
            ndvi_pred = out_vitdet.get("ndvi", None)
            # Optional per-layer outputs for constraint-aware fusion downstream.
            try:
                pred_reg3_layers = out_vitdet.get("reg3_layers", None)
                ratio_logits_layers = out_vitdet.get("ratio_layers", None)
                if not isinstance(pred_reg3_layers, list):
                    pred_reg3_layers = None
                if not isinstance(ratio_logits_layers, list):
                    ratio_logits_layers = None
            except Exception:
                pred_reg3_layers = None
                ratio_logits_layers = None
            if pred_reg3 is None or z is None:
                raise RuntimeError("ViTDet head did not return required outputs (reg3/z)")

        elif head_type == "dpt":
            # DPT-style dense prediction head: consume multi-layer CLS+patch tokens (recommended),
            # optionally apply manifold mixup on tokens, then predict reg3/ratio/ndvi.
            image_hw = (int(images.shape[-2]), int(images.shape[-1]))

            # Force CLS+patch mixup when using DPT readout=project (regardless of mix_cls_token flag),
            # because project-readout explicitly fuses CLS into patch tokens.
            try:
                dpt_readout_mode = str(getattr(self.hparams, "dpt_readout", "ignore")).strip().lower()
            except Exception:
                dpt_readout_mode = "ignore"

            if self.use_layerwise_heads and len(self.backbone_layer_indices) > 0:
                try:
                    cls_list, pt_list = self.feature_extractor.forward_layers_cls_and_tokens(
                        images, self.backbone_layer_indices
                    )
                except Exception as e:
                    backbone_name = getattr(getattr(self, "hparams", None), "backbone_name", None)
                    raise RuntimeError(
                        "Failed to extract multi-layer CLS/patch tokens for DPT head. "
                        f"Check model.backbone_layers.indices={self.backbone_layer_indices} "
                        f"are valid for backbone={backbone_name!r}. Original error: {e}"
                    ) from e
                if not pt_list:
                    raise RuntimeError(
                        "Multi-layer token extraction returned an empty pt_list for DPT head. "
                        f"indices={self.backbone_layer_indices}"
                    )
                if (
                    use_mixup
                    and self._manifold_mixup is not None
                    and (stage == "train")
                    and (not is_ndvi_only)
                    and pt_list
                ):
                    try:
                        if dpt_readout_mode == "project":
                            # Stack CLS+patch into a single token tensor so mixup affects both.
                            tok_stack = torch.stack(
                                [
                                    torch.cat([cls_l.unsqueeze(1), pt_l], dim=1)
                                    for cls_l, pt_l in zip(cls_list, pt_list)
                                ],
                                dim=1,
                            )  # (B, L, N+1, C)
                            tok_stack, batch, _ = self._manifold_mixup.apply(
                                tok_stack,
                                batch,
                                force=True,
                                lam=mixup_lam,
                                perm=mixup_perm,
                                mix_labels=mixup_mix_labels,
                            )
                            cls_list = [tok_stack[:, i, 0] for i in range(tok_stack.shape[1])]
                            pt_list = [tok_stack[:, i, 1:] for i in range(tok_stack.shape[1])]
                        else:
                            pt_stack = torch.stack(pt_list, dim=1)  # (B, L, N, C)
                            pt_stack, batch, _ = self._manifold_mixup.apply(
                                pt_stack,
                                batch,
                                force=True,
                                lam=mixup_lam,
                                perm=mixup_perm,
                                mix_labels=mixup_mix_labels,
                            )
                            pt_list = [pt_stack[:, i] for i in range(pt_stack.shape[1])]
                    except Exception as e:
                        self._raise_manifold_mixup_error(
                            context=f"dpt/multilayer/tokens (readout={dpt_readout_mode})",
                            err=e,
                            batch=batch,
                            pt_list=pt_list,
                            cls_list=cls_list,
                        )
                out_dpt = self.dpt_head(cls_list, pt_list, image_hw=image_hw)  # type: ignore[call-arg]
            else:
                cls_tok, pt_tokens = self.feature_extractor.forward_cls_and_tokens(images)
                if (
                    use_mixup
                    and self._manifold_mixup is not None
                    and (stage == "train")
                    and (not is_ndvi_only)
                ):
                    try:
                        if dpt_readout_mode == "project":
                            tok = torch.cat(
                                [cls_tok.unsqueeze(1), pt_tokens], dim=1
                            )  # (B, N+1, C)
                            tok, batch, _ = self._manifold_mixup.apply(
                                tok,
                                batch,
                                force=True,
                                lam=mixup_lam,
                                perm=mixup_perm,
                                mix_labels=mixup_mix_labels,
                            )
                            cls_tok = tok[:, 0]
                            pt_tokens = tok[:, 1:]
                        else:
                            pt_tokens, batch, _ = self._manifold_mixup.apply(
                                pt_tokens,
                                batch,
                                force=True,
                                lam=mixup_lam,
                                perm=mixup_perm,
                                mix_labels=mixup_mix_labels,
                            )
                    except Exception as e:
                        self._raise_manifold_mixup_error(
                            context=f"dpt/singlelayer/tokens (readout={dpt_readout_mode})",
                            err=e,
                            batch=batch,
                            x=pt_tokens,
                            pt_list=[pt_tokens],
                            cls_list=[cls_tok],
                        )
                out_dpt = self.dpt_head(cls_tok, pt_tokens, image_hw=image_hw)  # type: ignore[call-arg]

            pred_reg3 = out_dpt.get("reg3", None)  # type: ignore[assignment]
            z = out_dpt.get("z", None)  # type: ignore[assignment]
            ratio_logits_pred = out_dpt.get("ratio", None)
            ndvi_pred = out_dpt.get("ndvi", None)
            if pred_reg3 is None or z is None:
                raise RuntimeError("DPT head did not return required outputs (reg3/z)")

        elif head_type == "eomt":
            # EoMT-style injected-query head (matches `third_party/eomt`):
            #   1) run backbone up to (depth - k) blocks WITHOUT queries
            #   2) run the remaining k blocks jointly with injected queries
            #   3) (optional) apply manifold mixup on the *concatenated pooled representation*
            #      (mean_query / mean_patch / cls) so labels are mixed consistently with what the head consumes
            #
            # NOTE: This path ignores `use_layerwise_heads` because the injected-query
            # design is defined by "last-k blocks" rather than arbitrary layer indices.
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

            # Stage A: run up to start_block and return token sequence + patch-grid HW for RoPE.
            x_base, (H_p, W_p) = self.feature_extractor.forward_tokens_until_block(
                images, block_idx=int(start_block)
            )

            # Run injected-query transformer to obtain the normalized token sequence
            # (queries + base tokens). We then build the pooled concat representation
            # inside the head so it matches inference behavior.
            try:
                x_norm = self.eomt_head._forward_x_norm(  # type: ignore[attr-defined]
                    x_base,
                    backbone=getattr(self.feature_extractor, "backbone"),
                    patch_hw=(int(H_p), int(W_p)),
                )
            except Exception as e:
                raise RuntimeError(
                    f"Failed to compute EoMT injected tokens (head_type='eomt'). Original error: {e!r}"
                ) from e

            fused_concat = self.eomt_head.build_concat_features_from_tokens(  # type: ignore[attr-defined]
                x_norm, backbone=getattr(self.feature_extractor, "backbone")
            )

            # Optional manifold mixup on the pooled concat representation.
            if (
                use_mixup
                and self._manifold_mixup is not None
                and (stage == "train")
                and (not is_ndvi_only)
            ):
                try:
                    fused_concat, batch, _ = self._manifold_mixup.apply(
                        fused_concat,
                        batch,
                        force=True,
                        lam=mixup_lam,
                        perm=mixup_perm,
                        mix_labels=mixup_mix_labels,
                    )
                except Exception as e:
                    self._raise_manifold_mixup_error(
                        context="eomt/injected/fused_concat",
                        err=e,
                        batch=batch,
                        x=fused_concat,
                    )

            out_eomt = self.eomt_head.forward_from_concat_features(  # type: ignore[attr-defined]
                fused_concat
            )
            pred_reg3 = out_eomt.get("reg3", None)  # type: ignore[assignment]
            z = out_eomt.get("z", None)  # type: ignore[assignment]
            ratio_logits_pred = out_eomt.get("ratio", None)
            ndvi_pred = out_eomt.get("ndvi", None)
            if pred_reg3 is None or z is None:
                raise RuntimeError("EoMT head did not return required outputs (reg3/z)")

        elif self.use_layerwise_heads:
            # Multi-layer path: obtain per-layer CLS and patch tokens from the backbone.
            cls_list, pt_list = self.feature_extractor.forward_layers_cls_and_tokens(
                images, self.backbone_layer_indices
            )
            if not self.use_patch_reg3:
                apply_mm = bool(
                    use_mixup
                    and self._manifold_mixup is not None
                    and (stage == "train")
                    and (not is_ndvi_only)
                )

                apply_on = str(getattr(self, "_manifold_mixup_apply_on", "features") or "features").strip().lower()
                tokens_mode = apply_on in {"tokens", "patch_tokens", "pt_tokens", "pt"}

                # v186 behavior for the multi-layer MLP path: apply manifold mixup directly on
                # backbone patch tokens (CLS tokens remain intact), then compute global features.
                if tokens_mode:
                    if apply_mm and self._manifold_mixup is not None and pt_list:
                        try:
                            pt_stack = torch.stack(pt_list, dim=1)  # (B, L, N, C)
                            pt_stack, batch, _ = self._manifold_mixup.apply(
                                pt_stack,
                                batch,
                                force=True,
                                lam=mixup_lam,
                                perm=mixup_perm,
                                mix_labels=mixup_mix_labels,
                            )
                            pt_list = [pt_stack[:, i] for i in range(pt_stack.shape[1])]
                        except Exception as e:
                            self._raise_manifold_mixup_error(
                                context="mlp/multilayer/patch_tokens (apply_on=tokens)",
                                err=e,
                                batch=batch,
                                cls_list=cls_list,
                                pt_list=pt_list,
                            )
                    pred_reg3, z, z_layers = self._compute_reg3_and_z_multilayer(
                        images=None, cls_list=cls_list, pt_list=pt_list
                    )
                else:
                    # Current default: build per-layer patch means (B, C), optionally apply
                    # manifold mixup on global features or patch-mean features, then run heads.
                    patch_mean_list: List[Tensor] = [pt.mean(dim=1) for pt in pt_list]

                    feats_list: List[Tensor] = []
                    if apply_mm and self._manifold_mixup is not None:
                        try:
                            # If requested, keep CLS tokens intact and mix only patch features.
                            if self.use_cls_token and (
                                not bool(getattr(self._manifold_mixup, "mix_cls_token", True))
                            ):
                                # (B, L, C): mix along batch dim, preserve layer structure.
                                pm_stack = torch.stack(patch_mean_list, dim=1)
                                pm_stack, batch, _ = self._manifold_mixup.apply(
                                    pm_stack,
                                    batch,
                                    force=True,
                                    lam=mixup_lam,
                                    perm=mixup_perm,
                                    mix_labels=mixup_mix_labels,
                                )
                                patch_mean_list = [
                                    pm_stack[:, i] for i in range(pm_stack.shape[1])
                                ]
                                feats_list = [
                                    torch.cat([cls_l, pm_l], dim=-1)
                                    for cls_l, pm_l in zip(cls_list, patch_mean_list)
                                ]
                            else:
                                for cls_l, pm_l in zip(cls_list, patch_mean_list):
                                    feats_list.append(
                                        torch.cat([cls_l, pm_l], dim=-1)
                                        if self.use_cls_token
                                        else pm_l
                                    )
                                feats_stack = torch.stack(feats_list, dim=1)
                                feats_stack, batch, _ = self._manifold_mixup.apply(
                                    feats_stack,
                                    batch,
                                    force=True,
                                    lam=mixup_lam,
                                    perm=mixup_perm,
                                    mix_labels=mixup_mix_labels,
                                )
                                feats_list = [
                                    feats_stack[:, i] for i in range(feats_stack.shape[1])
                                ]
                        except Exception as e:
                            self._raise_manifold_mixup_error(
                                context="mlp/multilayer/global_features",
                                err=e,
                                batch=batch,
                                cls_list=cls_list,
                                pt_list=pt_list,
                            )

                    if not feats_list:
                        for cls_l, pm_l in zip(cls_list, patch_mean_list):
                            feats_list.append(
                                torch.cat([cls_l, pm_l], dim=-1)
                                if self.use_cls_token
                                else pm_l
                            )

                    pred_reg3, z, z_layers = self._compute_reg3_and_z_multilayer(
                        images=None, feats_list=feats_list
                    )
            else:
                # Patch-mode: optionally apply manifold mixup on the DINO patch tokens.
                if (
                    use_mixup
                    and self._manifold_mixup is not None
                    and (stage == "train")
                    and (not is_ndvi_only)
                ):
                    try:
                        mix_cls = bool(getattr(self._manifold_mixup, "mix_cls_token", True))
                        if mix_cls and cls_list is not None:
                            # Mix CLS + patch tokens together (per-layer) for consistency.
                            tok_stack = torch.stack(
                                [
                                    torch.cat([cls_l.unsqueeze(1), pt_l], dim=1)
                                    for cls_l, pt_l in zip(cls_list, pt_list)
                                ],
                                dim=1,
                            )  # (B, L, N+1, C)
                            tok_stack, batch, _ = self._manifold_mixup.apply(
                                tok_stack,
                                batch,
                                force=True,
                                lam=mixup_lam,
                                perm=mixup_perm,
                                mix_labels=mixup_mix_labels,
                            )
                            cls_list = [tok_stack[:, i, 0] for i in range(tok_stack.shape[1])]
                            pt_list = [tok_stack[:, i, 1:] for i in range(tok_stack.shape[1])]
                        else:
                            # Legacy behavior: mix patch tokens only; keep CLS intact.
                            pt_stack = torch.stack(pt_list, dim=1)
                            pt_stack, batch, _ = self._manifold_mixup.apply(
                                pt_stack,
                                batch,
                                force=True,
                                lam=mixup_lam,
                                perm=mixup_perm,
                                mix_labels=mixup_mix_labels,
                            )
                            pt_list = [pt_stack[:, i] for i in range(pt_stack.shape[1])]
                    except Exception as e:
                        self._raise_manifold_mixup_error(
                            context="mlp/multilayer/patch_tokens",
                            err=e,
                            batch=batch,
                            pt_list=pt_list,
                            cls_list=cls_list,
                        )
                pred_reg3, z, z_layers = self._compute_reg3_and_z_multilayer(images=None, cls_list=cls_list, pt_list=pt_list)

        else:
            if self.use_patch_reg3:
                cls_tok, pt_tokens = self.feature_extractor.forward_cls_and_tokens(images)
                if (
                    use_mixup
                    and self._manifold_mixup is not None
                    and (stage == "train")
                    and (not is_ndvi_only)
                ):
                    try:
                        mix_cls = bool(getattr(self._manifold_mixup, "mix_cls_token", True))
                        if mix_cls:
                            tok = torch.cat([cls_tok.unsqueeze(1), pt_tokens], dim=1)  # (B, N+1, C)
                            tok, batch, _ = self._manifold_mixup.apply(
                                tok,
                                batch,
                                force=True,
                                lam=mixup_lam,
                                perm=mixup_perm,
                                mix_labels=mixup_mix_labels,
                            )
                            cls_tok = tok[:, 0]
                            pt_tokens = tok[:, 1:]
                        else:
                            pt_tokens, batch, _ = self._manifold_mixup.apply(
                                pt_tokens,
                                batch,
                                force=True,
                                lam=mixup_lam,
                                perm=mixup_perm,
                                mix_labels=mixup_mix_labels,
                            )
                    except Exception as e:
                        self._raise_manifold_mixup_error(
                            context="mlp/singlelayer/patch_tokens",
                            err=e,
                            batch=batch,
                            x=pt_tokens,
                        )
                pred_reg3, z = self._compute_reg3_from_images(
                    images=None, pt_tokens=pt_tokens, cls_token=cls_tok
                )
            else:
                apply_mm = bool(
                    use_mixup
                    and self._manifold_mixup is not None
                    and (stage == "train")
                    and (not is_ndvi_only)
                )

                if (
                    self.use_cls_token
                    and apply_mm
                    and self._manifold_mixup is not None
                    and (not bool(getattr(self._manifold_mixup, "mix_cls_token", True)))
                ):
                    cls_tok, pt_tokens = self.feature_extractor.forward_cls_and_tokens(images)
                    patch_mean = pt_tokens.mean(dim=1)
                    try:
                        patch_mean, batch, _ = self._manifold_mixup.apply(
                            patch_mean,
                            batch,
                            force=True,
                            lam=mixup_lam,
                            perm=mixup_perm,
                            mix_labels=mixup_mix_labels,
                        )
                    except Exception as e:
                        self._raise_manifold_mixup_error(
                            context="mlp/singlelayer/patch_mean (mix_cls_token=false)",
                            err=e,
                            batch=batch,
                            x=patch_mean,
                        )
                    features = torch.cat([cls_tok, patch_mean], dim=-1)
                else:
                    if self.use_cls_token:
                        features = self.feature_extractor(images)
                    else:
                        _, pt_tokens = self.feature_extractor.forward_cls_and_tokens(images)
                        features = pt_tokens.mean(dim=1)
                    if apply_mm and self._manifold_mixup is not None:
                        try:
                            features, batch, _ = self._manifold_mixup.apply(
                            features,
                            batch,
                            force=True,
                            lam=mixup_lam,
                            perm=mixup_perm,
                            mix_labels=mixup_mix_labels,
                            )
                        except Exception as e:
                            self._raise_manifold_mixup_error(
                                context="mlp/singlelayer/global_features",
                                err=e,
                                batch=batch,
                                x=features,
                            )
                z = self.shared_bottleneck(features)
                pred_reg3 = self._forward_reg3_logits(z)

        return (
            pred_reg3,
            z,
            z_layers,
            ratio_logits_pred,
            ndvi_pred,
            batch,
            pred_reg3_layers,
            ratio_logits_layers,
        )


