from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple

import torch
from torch import Tensor


def normalize_layer_indices(indices: Iterable[int]) -> List[int]:
    """
    Normalize a user-provided collection of layer indices:
      - Cast to int
      - Remove duplicates
      - Sort ascending
    """
    uniq = sorted({int(i) for i in indices})
    return uniq


def average_layerwise_predictions(preds_per_layer: Sequence[Tensor]) -> Tensor:
    """
    Given a sequence of per-layer predictions (each of shape (B, D)),
    stack them and average over the layer dimension.
    """
    if len(preds_per_layer) == 0:
        raise ValueError("preds_per_layer must contain at least one tensor")
    stacked = torch.stack(preds_per_layer, dim=0)  # (L, B, D)
    return stacked.mean(dim=0)


def split_cls_and_patches_from_intermediate(
    outputs: Sequence[Tuple[Tensor, Tensor]]
) -> Tuple[List[Tensor], List[Tensor]]:
    """
    Helper for DINOv3 get_intermediate_layers output when called with
    return_class_token=True, return_extra_tokens=False.

    Each element in `outputs` is expected to be:
        (patch_tokens, cls_token)
    where:
        patch_tokens: (B, N, C)
        cls_token   : (B, C)

    Returns:
        cls_list: list of (B, C)
        pt_list : list of (B, N, C)
    """
    cls_list: List[Tensor] = []
    pt_list: List[Tensor] = []
    for patches, cls in outputs:
        pt_list.append(patches)
        cls_list.append(cls)
    return cls_list, pt_list


