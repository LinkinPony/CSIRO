from __future__ import annotations

from typing import Optional

import torch
from torch import nn
import torch.nn.functional as F


class NdviDenseHead(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int = 1,
        use_batchnorm: bool = True,
    ) -> None:
        super().__init__()
        layers = []
        if use_batchnorm:
            layers.append(nn.BatchNorm2d(in_channels))
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=True))
        self.proj = nn.Sequential(*layers)

    def forward(self, patch_feats: torch.Tensor) -> torch.Tensor:
        """
        Args:
            patch_feats: B x C x Hp x Wp
        Returns:
            pred: B x 1 x Hp x Wp
        """
        return self.proj(patch_feats)



