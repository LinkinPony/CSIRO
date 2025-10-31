from typing import Optional

import torch
from torch import Tensor, nn


class DinoV3FeatureExtractor(nn.Module):
    def __init__(self, backbone: nn.Module) -> None:
        super().__init__()
        self.backbone = backbone
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False
        self.backbone.eval()

    @torch.inference_mode()
    def forward(self, images: Tensor) -> Tensor:
        # Ensure AMP half inputs match backbone (float32) to avoid dtype mismatch in conv2d
        try:
            backbone_dtype = next(self.backbone.parameters()).dtype
        except StopIteration:
            backbone_dtype = images.dtype
        if images.dtype != backbone_dtype:
            images = images.to(dtype=backbone_dtype)
        feats = self.backbone.forward_features(images)
        return feats["x_norm_clstoken"]


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


def build_feature_extractor(
    backbone_name: str,
    pretrained: bool = True,
    weights_url: Optional[str] = None,
    weights_path: Optional[str] = None,
) -> nn.Module:
    if backbone_name != "dinov3_vitl16":
        raise ValueError(f"Unsupported backbone: {backbone_name}")
    # When weights_path is provided, force pretrained=False to avoid online fetch
    use_pretrained = False if weights_path else pretrained
    backbone = load_dinov3_vitl16(
        pretrained=use_pretrained,
        weights_url=weights_url if not weights_path else None,
        weights_path=weights_path,
    )
    return DinoV3FeatureExtractor(backbone)


