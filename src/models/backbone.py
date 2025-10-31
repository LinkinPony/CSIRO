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
        feats = self.backbone.forward_features(images)
        return feats["x_norm_clstoken"]


def load_dinov3_vitl16(pretrained: bool = True, weights_url: Optional[str] = None, check_hash: bool = False) -> nn.Module:
    if weights_url:
        model = torch.hub.load(
            repo_or_dir="facebookresearch/dinov3",
            model="dinov3_vitl16",
            pretrained=pretrained,
            weights=weights_url,
            check_hash=check_hash,
        )
    else:
        model = torch.hub.load(
            repo_or_dir="facebookresearch/dinov3",
            model="dinov3_vitl16",
            pretrained=pretrained,
        )
    return model


def build_feature_extractor(backbone_name: str, pretrained: bool = True, weights_url: Optional[str] = None) -> nn.Module:
    if backbone_name != "dinov3_vitl16":
        raise ValueError(f"Unsupported backbone: {backbone_name}")
    backbone = load_dinov3_vitl16(pretrained=pretrained, weights_url=weights_url)
    return DinoV3FeatureExtractor(backbone)


