from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch
from torch import Tensor


class PredictionMixin:
    """
    Compute model predictions (reg3 logits + shared representation z) for a batch.

    This contains the head-type specific forward paths used during training/validation
    steps, including optional manifold mixup on backbone tokens/features.
    """

    def _predict_reg3_and_z(
        self,
        *,
        images: Tensor,
        batch: Dict[str, Tensor],
        stage: str,
        use_mixup: bool,
        is_ndvi_only: bool,
    ) -> Tuple[Tensor, Tensor, Optional[List[Tensor]], Optional[Tensor], Optional[Tensor], Dict[str, Tensor]]:
        """
        Returns:
            pred_reg3: (B, num_outputs) logits in normalized domain (pre-softplus)
            z:         (B, D) shared global representation
            z_layers:  Optional[list[(B, D)]] per-layer z (layerwise heads)
            ratio_logits_pred: Optional[(B,3)] ratio logits for FPN/DPT heads
            ndvi_pred: Optional[(B,1)] NDVI pred for FPN/DPT heads
            batch: possibly-updated batch (mixup mutates labels)
        """
        pred_reg3: Tensor
        z: Tensor
        z_layers: Optional[List[Tensor]] = None
        ratio_logits_pred: Optional[Tensor] = None
        ndvi_pred: Optional[Tensor] = None

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
                        pt_stack, batch, _ = self._manifold_mixup.apply(pt_stack, batch, force=True)
                        pt_list = [pt_stack[:, i] for i in range(pt_stack.shape[1])]
                    except Exception:
                        pass
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
                        pt_tokens, batch, _ = self._manifold_mixup.apply(pt_tokens, batch, force=True)
                    except Exception:
                        pass
                pt_in = pt_tokens

            out_fpn = self.fpn_head(pt_in, image_hw=image_hw)  # type: ignore[attr-defined]
            pred_reg3 = out_fpn.get("reg3", None)  # type: ignore[assignment]
            z = out_fpn.get("z", None)  # type: ignore[assignment]
            ratio_logits_pred = out_fpn.get("ratio", None)
            ndvi_pred = out_fpn.get("ndvi", None)
            if pred_reg3 is None or z is None:
                raise RuntimeError("FPN head did not return required outputs (reg3/z)")

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
                            tok_stack, batch, _ = self._manifold_mixup.apply(tok_stack, batch, force=True)
                            cls_list = [tok_stack[:, i, 0] for i in range(tok_stack.shape[1])]
                            pt_list = [tok_stack[:, i, 1:] for i in range(tok_stack.shape[1])]
                        else:
                            pt_stack = torch.stack(pt_list, dim=1)  # (B, L, N, C)
                            pt_stack, batch, _ = self._manifold_mixup.apply(pt_stack, batch, force=True)
                            pt_list = [pt_stack[:, i] for i in range(pt_stack.shape[1])]
                    except Exception:
                        pass
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
                            tok = torch.cat([cls_tok.unsqueeze(1), pt_tokens], dim=1)  # (B, N+1, C)
                            tok, batch, _ = self._manifold_mixup.apply(tok, batch, force=True)
                            cls_tok = tok[:, 0]
                            pt_tokens = tok[:, 1:]
                        else:
                            pt_tokens, batch, _ = self._manifold_mixup.apply(pt_tokens, batch, force=True)
                    except Exception:
                        pass
                out_dpt = self.dpt_head(cls_tok, pt_tokens, image_hw=image_hw)  # type: ignore[call-arg]

            pred_reg3 = out_dpt.get("reg3", None)  # type: ignore[assignment]
            z = out_dpt.get("z", None)  # type: ignore[assignment]
            ratio_logits_pred = out_dpt.get("ratio", None)
            ndvi_pred = out_dpt.get("ndvi", None)
            if pred_reg3 is None or z is None:
                raise RuntimeError("DPT head did not return required outputs (reg3/z)")

        elif self.use_layerwise_heads:
            # Multi-layer path: obtain per-layer CLS and patch tokens from the backbone.
            cls_list, pt_list = self.feature_extractor.forward_layers_cls_and_tokens(
                images, self.backbone_layer_indices
            )
            if not self.use_patch_reg3:
                # Build per-layer patch means (B, C).
                patch_mean_list: List[Tensor] = [pt.mean(dim=1) for pt in pt_list]

                apply_mm = bool(
                    use_mixup
                    and self._manifold_mixup is not None
                    and (stage == "train")
                    and (not is_ndvi_only)
                )

                feats_list: List[Tensor] = []
                if apply_mm and self._manifold_mixup is not None:
                    try:
                        # If requested, keep CLS tokens intact and mix only patch features.
                        if self.use_cls_token and (not bool(getattr(self._manifold_mixup, "mix_cls_token", True))):
                            # (B, L, C): mix along batch dim, preserve layer structure.
                            pm_stack = torch.stack(patch_mean_list, dim=1)
                            pm_stack, batch, _ = self._manifold_mixup.apply(
                                pm_stack, batch, force=True
                            )
                            patch_mean_list = [pm_stack[:, i] for i in range(pm_stack.shape[1])]
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
                                feats_stack, batch, force=True
                            )
                            feats_list = [feats_stack[:, i] for i in range(feats_stack.shape[1])]
                    except Exception:
                        feats_list = []

                if not feats_list:
                    for cls_l, pm_l in zip(cls_list, patch_mean_list):
                        feats_list.append(
                            torch.cat([cls_l, pm_l], dim=-1) if self.use_cls_token else pm_l
                        )

                pred_reg3, z, z_layers = self._compute_reg3_and_z_multilayer(images=None, feats_list=feats_list)
            else:
                # Patch-mode: optionally apply manifold mixup on the DINO patch tokens.
                if (
                    use_mixup
                    and self._manifold_mixup is not None
                    and (stage == "train")
                    and (not is_ndvi_only)
                ):
                    try:
                        pt_stack = torch.stack(pt_list, dim=1)
                        pt_stack, batch, _ = self._manifold_mixup.apply(pt_stack, batch, force=True)
                        pt_list = [pt_stack[:, i] for i in range(pt_stack.shape[1])]
                    except Exception:
                        pass
                pred_reg3, z, z_layers = self._compute_reg3_and_z_multilayer(images=None, cls_list=cls_list, pt_list=pt_list)

        else:
            if self.use_patch_reg3:
                _, pt_tokens = self.feature_extractor.forward_cls_and_tokens(images)
                if (
                    use_mixup
                    and self._manifold_mixup is not None
                    and (stage == "train")
                    and (not is_ndvi_only)
                ):
                    try:
                        pt_tokens, batch, _ = self._manifold_mixup.apply(pt_tokens, batch, force=True)
                    except Exception:
                        pass
                pred_reg3, z = self._compute_reg3_from_images(images=None, pt_tokens=pt_tokens)
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
                        patch_mean, batch, _ = self._manifold_mixup.apply(patch_mean, batch, force=True)
                    except Exception:
                        pass
                    features = torch.cat([cls_tok, patch_mean], dim=-1)
                else:
                    if self.use_cls_token:
                        features = self.feature_extractor(images)
                    else:
                        _, pt_tokens = self.feature_extractor.forward_cls_and_tokens(images)
                        features = pt_tokens.mean(dim=1)
                    if apply_mm and self._manifold_mixup is not None:
                        try:
                            features, batch, _ = self._manifold_mixup.apply(features, batch, force=True)
                        except Exception:
                            pass
                z = self.shared_bottleneck(features)
                pred_reg3 = self._forward_reg3_logits(z)

        return pred_reg3, z, z_layers, ratio_logits_pred, ndvi_pred, batch


