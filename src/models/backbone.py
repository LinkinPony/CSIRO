from typing import Optional

import torch
from torch import Tensor, nn


class DinoV3FeatureExtractor(nn.Module):
    def __init__(self, backbone: nn.Module, *, inference_only: bool = True) -> None:
        super().__init__()
        self.backbone = backbone
        self.inference_only = bool(inference_only)
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False
        self.backbone.eval()

    def _forward_impl(self, images: Tensor) -> Tensor:
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
        return feats["x_norm_clstoken"]

    def forward(self, images: Tensor) -> Tensor:
        if self.inference_only:
            with torch.inference_mode():
                return self._forward_impl(images)
        return self._forward_impl(images)


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


