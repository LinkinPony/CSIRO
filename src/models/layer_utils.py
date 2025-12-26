from __future__ import annotations

from typing import Iterable, List, Optional, Sequence, Tuple

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


def fuse_layerwise_predictions(
    preds_per_layer: Sequence[Tensor],
    *,
    weights: Optional[Tensor] = None,
) -> Tensor:
    """
    Fuse a sequence of per-layer predictions (each of shape (B, ...)) into a single tensor.

    - If `weights` is None: uniform average over layers (same as `average_layerwise_predictions`).
    - If `weights` is provided: compute a weighted average over layers.

    Args:
        preds_per_layer: Sequence of tensors, each shaped (B, ...).
        weights: Optional tensor of shape (L,), where L == len(preds_per_layer).
                 Weights are renormalized to sum-to-1 for numerical safety.
    """
    if len(preds_per_layer) == 0:
        raise ValueError("preds_per_layer must contain at least one tensor")
    stacked = torch.stack(preds_per_layer, dim=0)  # (L, B, ...)
    if weights is None:
        return stacked.mean(dim=0)

    if not isinstance(weights, torch.Tensor):
        raise TypeError("weights must be a torch.Tensor when provided")
    if weights.dim() != 1:
        raise ValueError(f"weights must be 1D (L,), got shape={tuple(weights.shape)}")
    if int(weights.numel()) != int(stacked.shape[0]):
        raise ValueError(
            f"weights length must match number of layers: got {int(weights.numel())}, expected {int(stacked.shape[0])}"
        )

    w = weights.to(device=stacked.device, dtype=stacked.dtype)
    w = w / w.sum().clamp_min(1e-8)
    # Broadcast to (L, 1, 1, ...)
    view_shape = [int(w.shape[0])] + [1] * (stacked.dim() - 1)
    w = w.view(*view_shape)
    return (w * stacked).sum(dim=0)


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


