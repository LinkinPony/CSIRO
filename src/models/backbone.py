from typing import Optional, Tuple
import math

import torch
from torch import Tensor, nn

from .layer_utils import (
    normalize_layer_indices,
    split_cls_and_patches_from_intermediate,
)


class DinoV3FeatureExtractor(nn.Module):
    def __init__(self, backbone: nn.Module, *, inference_only: bool = True) -> None:
        super().__init__()
        self.backbone = backbone
        self.inference_only = bool(inference_only)
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False
        self.backbone.eval()

    def _forward_features_dict(self, images: Tensor):
        # Ensure AMP half inputs match backbone (float32) to avoid dtype mismatch in conv2d
        try:
            backbone_dtype = next(self.backbone.parameters()).dtype
        except StopIteration:
            backbone_dtype = images.dtype
        if images.dtype != backbone_dtype:
            images = images.to(dtype=backbone_dtype)
        # Support both raw DINOv3 and PEFT-wrapped backbones
        try:
            forward_features = getattr(self.backbone, "forward_features", None)
            if forward_features is None and hasattr(self.backbone, "base_model"):
                forward_features = getattr(self.backbone.base_model, "forward_features", None)
            if forward_features is None:
                # Fallback: call forward and expect a dict-like output
                out = self.backbone(images)
                feats = out if isinstance(out, dict) else {"x_norm_clstoken": out}
            else:
                feats = forward_features(images)
        except Exception:
            feats = self.backbone.forward_features(images)
        return feats

    def _get_intermediate_layers_raw(self, images: Tensor, layer_indices):
        """
        Call DINOv3-style get_intermediate_layers on the underlying backbone,
        handling PEFT-wrapped models where the method may live on base_model.
        """
        # Ensure dtype consistency with backbone parameters
        try:
            backbone_dtype = next(self.backbone.parameters()).dtype
        except StopIteration:
            backbone_dtype = images.dtype
        if images.dtype != backbone_dtype:
            images = images.to(dtype=backbone_dtype)

        backbone = self.backbone
        get_intermediate = getattr(backbone, "get_intermediate_layers", None)
        if get_intermediate is None and hasattr(backbone, "base_model"):
            get_intermediate = getattr(
                backbone.base_model, "get_intermediate_layers", None  # type: ignore[attr-defined]
            )
        if get_intermediate is None:
            raise RuntimeError(
                "Backbone does not implement get_intermediate_layers; "
                "multi-layer feature extraction is unsupported for this backbone."
            )

        # DINOv3 accepts either an int (last n layers) or a sequence of indices.
        # Here we always pass a normalized list of indices defined by the caller.
        outs = get_intermediate(
            images,
            n=layer_indices,
            reshape=False,
            return_class_token=True,
            return_extra_tokens=False,
            norm=True,
        )
        return outs

    def _extract_cls_and_pt(self, feats):
        """
        Helper to extract CLS token and patch tokens from a forward_features-style output.

        Returns:
            cls: Tensor of shape (B, C)
            pt:  Tensor of shape (B, N, C)
        """
        # CLS token (B, C)
        cls = feats.get("x_norm_clstoken", None)
        if cls is None:
            raise RuntimeError("Backbone did not return 'x_norm_clstoken' in forward_features output")
        # Patch tokens (B, N, C), try common keys and fallbacks
        pt = None
        for k in ("x_norm_patchtokens", "x_norm_patch_tokens", "x_patch_tokens", "x_tokens"):
            if isinstance(feats, dict) and k in feats:
                pt = feats[k]
                break
        if pt is None and isinstance(feats, (list, tuple)) and len(feats) >= 2:
            # Fallback: some implementations return tuple (cls, patches)
            pt = feats[1]
        if pt is None:
            raise RuntimeError("Backbone did not return patch tokens in forward_features output")
        if pt.dim() != 3:
            raise RuntimeError(f"Unexpected patch tokens shape: {tuple(pt.shape)}")
        return cls, pt

    def forward_layers_cls_and_tokens(
        self,
        images: Tensor,
        layer_indices,
    ):
        """
        Return CLS and patch tokens for a set of backbone layers.

        Args:
            images:        (B, 3, H, W)
            layer_indices: iterable of int, backbone block indices

        Returns:
            cls_list: list of Tensors, each (B, C)
            pt_list : list of Tensors, each (B, N, C)
        """
        indices = normalize_layer_indices(layer_indices)
        if len(indices) == 0:
            raise ValueError("layer_indices must contain at least one index")

        if self.inference_only:
            with torch.inference_mode():
                outs = self._get_intermediate_layers_raw(images, indices)
        else:
            outs = self._get_intermediate_layers_raw(images, indices)

        cls_list, pt_list = split_cls_and_patches_from_intermediate(outs)
        return cls_list, pt_list

    def forward_layers_patch_tokens(
        self,
        images: Tensor,
        layer_indices,
    ):
        """
        Convenience helper to obtain per-layer patch feature maps:

        Returns:
            patch_feats_per_layer: list of tensors, each (B, C, Hp, Wp)
            patch_stride:          int, approximate patch stride (shared across layers)
        """
        cls_list, pt_list = self.forward_layers_cls_and_tokens(images, layer_indices)
        # Use the first layer to infer spatial layout; assume it matches others.
        if len(pt_list) == 0:
            raise RuntimeError("No patch tokens returned from intermediate layers")
        sample_pt = pt_list[0]
        if sample_pt.dim() != 3:
            raise RuntimeError(f"Unexpected patch tokens shape in multi-layer path: {tuple(sample_pt.shape)}")
        B, N, C = sample_pt.shape
        side = int(math.sqrt(N))
        if side * side != N:
            side = int(round(math.sqrt(N)))
        if side <= 0:
            raise RuntimeError(f"Cannot infer patch grid from N={N}")
        patch_feats_per_layer = []
        for pt in pt_list:
            if pt.shape[1] != N or pt.shape[2] != C:
                raise RuntimeError(
                    f"Inconsistent patch token shape across layers: expected (B,{N},{C}), got {tuple(pt.shape)}"
                )
            feat = pt.transpose(1, 2).contiguous().view(B, C, side, side)
            patch_feats_per_layer.append(feat)

        in_h = images.shape[-2]
        patch_stride = max(1, in_h // side)
        return patch_feats_per_layer, patch_stride

    def _forward_impl(self, images: Tensor) -> Tensor:
        feats = self._forward_features_dict(images)
        cls, pt = self._extract_cls_and_pt(feats)
        # Mean over patch tokens (B, C)
        patch_mean = pt.mean(dim=1)
        # Concatenate CLS with mean patch token -> (B, 2C)
        return torch.cat([cls, patch_mean], dim=-1)

    def forward(self, images: Tensor) -> Tensor:
        if self.inference_only:
            with torch.inference_mode():
                return self._forward_impl(images)
        return self._forward_impl(images)

    def forward_cls_and_tokens(self, images: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Return CLS token and patch tokens from the backbone in a single forward pass.

        Returns:
            cls: Tensor of shape (B, C)
            pt:  Tensor of shape (B, N, C)
        """
        if self.inference_only:
            with torch.inference_mode():
                feats = self._forward_features_dict(images)
        else:
            feats = self._forward_features_dict(images)
        cls, pt = self._extract_cls_and_pt(feats)
        return cls, pt

    def forward_patch_tokens(self, images: Tensor) -> Tuple[Tensor, int]:
        """
        Returns:
            patch_feats: Tensor of shape (B, C, H_p, W_p)
            patch_stride: int, approximate patch size (input_size / W_p)
        """
        # Respect inference_only: allow gradients in training to enable LoRA updates,
        # but run inference-mode for memory/throughput when not training.
        if self.inference_only:
            with torch.inference_mode():
                return self._forward_patch_tokens_impl(images)
        return self._forward_patch_tokens_impl(images)

    def _forward_patch_tokens_impl(self, images: Tensor) -> Tuple[Tensor, int]:
        feats = self._forward_features_dict(images)
        _, pt = self._extract_cls_and_pt(feats)
        # pt: (B, N, C)
        B, N, C = pt.shape
        # Assume square grid
        side = int(math.sqrt(N))
        if side * side != N:
            # Try to fall back to input size if divisible by 16
            # This assumes square input tiles
            side = int(round(math.sqrt(N)))
        if side <= 0:
            raise RuntimeError(f"Cannot infer patch grid from N={N}")
        patch_feats = pt.transpose(1, 2).contiguous().view(B, C, side, side)
        # Approximate stride from input size (square tile assumption)
        in_h = images.shape[-2]
        patch_stride = max(1, in_h // side)
        return patch_feats, patch_stride


def load_dinov3_vitl16(
    pretrained: bool = True,
    weights_url: Optional[str] = None,
    weights_path: Optional[str] = None,
    check_hash: bool = False,
) -> nn.Module:
    # Prefer explicit offline weights if provided
    if weights_path:
        model = torch.hub.load(
            repo_or_dir="facebookresearch/dinov3",
            model="dinov3_vitl16",
            pretrained=False,
        )
        state = torch.load(weights_path, map_location="cpu")
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        # Load with strict=False to be robust to minor key mismatches
        model.load_state_dict(state, strict=False)
        return model

    if weights_url:
        model = torch.hub.load(
            repo_or_dir="facebookresearch/dinov3",
            model="dinov3_vitl16",
            pretrained=pretrained,
            weights=weights_url,
            check_hash=check_hash,
        )
        return model

    model = torch.hub.load(
        repo_or_dir="facebookresearch/dinov3",
        model="dinov3_vitl16",
        pretrained=pretrained,
    )
    return model


def load_dinov3_vith16plus(
    pretrained: bool = True,
    weights_url: Optional[str] = None,
    weights_path: Optional[str] = None,
    check_hash: bool = False,
) -> nn.Module:
    # Prefer explicit offline weights if provided
    if weights_path:
        model = torch.hub.load(
            repo_or_dir="facebookresearch/dinov3",
            model="dinov3_vith16plus",
            pretrained=False,
        )
        state = torch.load(weights_path, map_location="cpu")
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        model.load_state_dict(state, strict=False)
        return model

    if weights_url:
        model = torch.hub.load(
            repo_or_dir="facebookresearch/dinov3",
            model="dinov3_vith16plus",
            pretrained=pretrained,
            weights=weights_url,
            check_hash=check_hash,
        )
        return model

    model = torch.hub.load(
        repo_or_dir="facebookresearch/dinov3",
        model="dinov3_vith16plus",
        pretrained=pretrained,
    )
    return model

def build_feature_extractor(
    backbone_name: str,
    pretrained: bool = True,
    weights_url: Optional[str] = None,
    weights_path: Optional[str] = None,
) -> nn.Module:
    # When weights_path is provided, force pretrained=False to avoid online fetch
    use_pretrained = False if weights_path else pretrained
    if backbone_name == "dinov3_vitl16":
        backbone = load_dinov3_vitl16(
            pretrained=use_pretrained,
            weights_url=weights_url if not weights_path else None,
            weights_path=weights_path,
        )
    elif backbone_name == "dinov3_vith16plus":
        backbone = load_dinov3_vith16plus(
            pretrained=use_pretrained,
            weights_url=weights_url if not weights_path else None,
            weights_path=weights_path,
        )
    else:
        raise ValueError(f"Unsupported backbone: {backbone_name}")
    return DinoV3FeatureExtractor(backbone)


