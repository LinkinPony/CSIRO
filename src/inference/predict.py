from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
import torchvision.transforms as T
from torch import nn
from torch.utils.data import DataLoader

from src.inference.data import TestImageDataset, build_transforms
from src.inference.mp import module_param_dtype, mp_get_devices_from_backbone
from src.models.head_builder import MultiLayerHeadExport


class DinoV3FeatureExtractor(nn.Module):
    def __init__(self, backbone: nn.Module) -> None:
        super().__init__()
        self.backbone = backbone
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.backbone.eval()

    def _forward_features_dict(self, images: torch.Tensor):
        try:
            backbone_dtype = next(self.backbone.parameters()).dtype
        except StopIteration:
            backbone_dtype = images.dtype
        if images.dtype != backbone_dtype:
            images = images.to(dtype=backbone_dtype)
        # Support PEFT-wrapped backbones (forward_features may live on base_model)
        try:
            forward_features = getattr(self.backbone, "forward_features", None)
            if forward_features is None and hasattr(self.backbone, "base_model"):
                forward_features = getattr(self.backbone.base_model, "forward_features", None)
            if forward_features is None:
                out = self.backbone(images)
                feats = out if isinstance(out, dict) else {"x_norm_clstoken": out}
            else:
                feats = forward_features(images)
        except Exception:
            feats = self.backbone.forward_features(images)
        return feats

    def _get_intermediate_layers_raw(self, images: torch.Tensor, layer_indices):
        """
        Call DINOv3-style get_intermediate_layers on the underlying backbone,
        handling PEFT-wrapped models where the method may live on base_model.
        """
        try:
            backbone_dtype = next(self.backbone.parameters()).dtype
        except StopIteration:
            backbone_dtype = images.dtype
        if images.dtype != backbone_dtype:
            images = images.to(dtype=backbone_dtype)

        backbone = self.backbone
        get_intermediate = getattr(backbone, "get_intermediate_layers", None)
        if get_intermediate is None and hasattr(backbone, "base_model"):
            get_intermediate = getattr(backbone.base_model, "get_intermediate_layers", None)  # type: ignore[attr-defined]
        if get_intermediate is None:
            raise RuntimeError(
                "Backbone does not implement get_intermediate_layers; "
                "multi-layer feature extraction is unsupported for this backbone."
            )

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
        cls = feats.get("x_norm_clstoken", None)
        if cls is None:
            raise RuntimeError("Backbone did not return 'x_norm_clstoken' in forward_features output")
        pt = None
        for k in ("x_norm_patchtokens", "x_norm_patch_tokens", "x_patch_tokens", "x_tokens"):
            if isinstance(feats, dict) and k in feats:
                pt = feats[k]
                break
        if pt is None and isinstance(feats, (list, tuple)) and len(feats) >= 2:
            pt = feats[1]
        if pt is None:
            raise RuntimeError("Backbone did not return patch tokens in forward_features output")
        if pt.dim() != 3:
            raise RuntimeError(f"Unexpected patch tokens shape: {tuple(pt.shape)}")
        return cls, pt

    @torch.inference_mode()
    def forward_layers_cls_and_tokens(self, images: torch.Tensor, layer_indices):
        """
        Return CLS and patch tokens for a set of backbone layers.

        Args:
            images:        (B, 3, H, W)
            layer_indices: iterable of int, backbone block indices

        Returns:
            cls_list: list of Tensors, each (B, C)
            pt_list : list of Tensors, each (B, N, C)
        """
        indices = sorted({int(i) for i in layer_indices})
        if len(indices) == 0:
            raise ValueError("layer_indices must contain at least one index")
        outs = self._get_intermediate_layers_raw(images, indices)
        cls_list: List[torch.Tensor] = []
        pt_list: List[torch.Tensor] = []
        for out in outs:
            # get_intermediate_layers with return_class_token=True returns tuples
            # of the form (patch_tokens, class_token).
            if isinstance(out, (list, tuple)) and len(out) >= 2:
                pt, cls = out[0], out[1]
            else:
                raise RuntimeError("Unexpected output format from get_intermediate_layers")
            pt_list.append(pt)
            cls_list.append(cls)
        return cls_list, pt_list

    @torch.inference_mode()
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Default forward used by legacy heads: returns CLS + mean(patch) features
        of shape (B, 2 * C), matching the training-time feature extractor.
        """
        feats = self._forward_features_dict(images)
        cls, pt = self._extract_cls_and_pt(feats)
        patch_mean = pt.mean(dim=1)
        return torch.cat([cls, patch_mean], dim=-1)

    @torch.inference_mode()
    def forward_cls_and_tokens(self, images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return CLS token and patch tokens from the backbone in a single forward pass.

        Returns:
            cls: Tensor of shape (B, C)
            pt:  Tensor of shape (B, N, C)
        """
        feats = self._forward_features_dict(images)
        cls, pt = self._extract_cls_and_pt(feats)
        return cls, pt


def extract_features_for_images(
    feature_extractor: nn.Module,
    dataset_root: str,
    image_paths: List[str],
    image_size: Tuple[int, int],
    mean: List[float],
    std: List[float],
    batch_size: int,
    num_workers: int,
    *,
    use_cls_token: bool = True,
) -> Tuple[List[str], torch.Tensor]:
    mp_devs = mp_get_devices_from_backbone(feature_extractor) if isinstance(feature_extractor, nn.Module) else None
    device = mp_devs[0] if mp_devs is not None else ("cuda" if torch.cuda.is_available() else "cpu")
    tf = build_transforms(image_size=image_size, mean=mean, std=std)
    ds = TestImageDataset(image_paths, root_dir=dataset_root, transform=tf)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=torch.cuda.is_available())

    # In model-parallel mode we MUST NOT move the full module to a single device.
    if mp_devs is not None:
        feature_extractor.eval()
    else:
        feature_extractor.eval().to(device)

    rels: List[str] = []
    feats_cpu: List[torch.Tensor] = []
    with torch.inference_mode():
        for images, rel_paths in dl:
            images = images.to(device, non_blocking=True)
            if use_cls_token:
                feats = feature_extractor(images)
            else:
                # Use patch-mean only (no CLS) for global features.
                _, pt = feature_extractor.forward_cls_and_tokens(images)  # type: ignore[attr-defined]
                feats = pt.mean(dim=1)
            feats_cpu.append(feats.detach().cpu().float())
            rels.extend(list(rel_paths))
    features = torch.cat(feats_cpu, dim=0) if feats_cpu else torch.empty((0, 0), dtype=torch.float32)
    return rels, features


def predict_from_features(
    features_cpu: torch.Tensor,
    head: nn.Module,
    batch_size: int,
    *,
    head_num_main: int,
    head_num_ratio: int,
    head_total: int,
    use_layerwise_heads: bool,
    num_layers: int,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    # For model-parallel backbone inference, prefer placing the head on stage1.
    mp_devs = mp_get_devices_from_backbone(head) if isinstance(head, nn.Module) else None
    device = mp_devs[1] if mp_devs is not None else ("cuda" if torch.cuda.is_available() else "cpu")
    head = head.eval().to(device)
    N = features_cpu.shape[0]
    preds_list: List[torch.Tensor] = []
    head_dtype = module_param_dtype(head, default=torch.float32)
    with torch.inference_mode():
        for i in range(0, N, max(1, batch_size)):
            chunk = features_cpu[i : i + max(1, batch_size)].to(device, non_blocking=True, dtype=head_dtype)
            out = head(chunk)
            preds_list.append(out.detach().cpu().float())
    if not preds_list:
        empty_main = torch.empty((0, head_num_main), dtype=torch.float32)
        empty_ratio = torch.empty((0, head_num_ratio), dtype=torch.float32) if head_num_ratio > 0 else None
        return empty_main, empty_ratio

    preds_all = torch.cat(preds_list, dim=0)  # (N, D)

    # Legacy_single-layer: interpret outputs directly.
    if not use_layerwise_heads or num_layers <= 1:
        if head_num_ratio > 0 and head_total == head_num_main + head_num_ratio:
            preds_main = preds_all[:, :head_num_main]
            preds_ratio = preds_all[:, head_num_main : head_num_main + head_num_ratio]
        else:
            preds_main = preds_all
            preds_ratio = None
        return preds_main, preds_ratio

    # Layer-wise packed outputs: final linear layer concatenates per-layer
    # [main, ratio] predictions along the feature dimension.
    if head_num_ratio > 0 and head_total == head_num_main + head_num_ratio:
        # preds_all: (N, head_total * num_layers)
        if preds_all.shape[1] != head_total * num_layers:
            raise RuntimeError(
                f"Unexpected packed head dimension: got {preds_all.shape[1]}, "
                f"expected {head_total * num_layers}"
            )
        preds_all_L = preds_all.view(N, num_layers, head_total)
        main_layers = preds_all_L[:, :, :head_num_main]  # (N, L, head_num_main)
        ratio_layers = preds_all_L[:, :, head_num_main : head_num_main + head_num_ratio]  # (N, L, head_num_ratio)
        preds_main = main_layers.mean(dim=1)  # (N, head_num_main)
        preds_ratio = ratio_layers.mean(dim=1)  # (N, head_num_ratio)
    else:
        # No dedicated ratio outputs: only main predictions are packed.
        if preds_all.shape[1] != head_num_main * num_layers:
            raise RuntimeError(
                f"Unexpected packed head dimension (no-ratio): got {preds_all.shape[1]}, "
                f"expected {head_num_main * num_layers}"
            )
        preds_all_L = preds_all.view(N, num_layers, head_num_main)
        preds_main = preds_all_L.mean(dim=1)
        preds_ratio = None

    return preds_main, preds_ratio


def predict_main_and_ratio_patch_mode(
    backbone: nn.Module,
    head: nn.Module,
    dataset_root: str,
    image_paths: List[str],
    image_size: Tuple[int, int],
    mean: List[float],
    std: List[float],
    batch_size: int,
    num_workers: int,
    head_num_main: int,
    head_num_ratio: int,
    head_total: int,
    use_layerwise_heads: bool,
    num_layers: int,
    use_separate_bottlenecks: bool,
    layer_indices: Optional[List[int]] = None,
) -> Tuple[List[str], torch.Tensor, Optional[torch.Tensor]]:
    """
    Patch-mode inference for a single head:
      - For each image, obtain CLS and patch tokens from the DINO backbone.
      - For each patch, apply the packed head on the patch token only (embedding_dim channels)
        and average per-patch main outputs to obtain image-level main predictions.
      - For ratio outputs (if present), apply the same head on the mean patch token.

    Returns:
        rels_in_order: list of image paths in dataloader order
        preds_main:    Tensor of shape (N_images, head_num_main)
        preds_ratio:   Tensor of shape (N_images, head_num_ratio) or None when head_num_ratio == 0
    """
    mp_devs = mp_get_devices_from_backbone(backbone) if isinstance(backbone, nn.Module) else None
    device0 = mp_devs[0] if mp_devs is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device1 = mp_devs[1] if mp_devs is not None else device0
    # Convenience alias for any tensors that must live with the head.
    device = device1
    tf = build_transforms(image_size=image_size, mean=mean, std=std)
    ds = TestImageDataset(image_paths, root_dir=dataset_root, transform=tf)
    dl = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    feature_extractor = DinoV3FeatureExtractor(backbone)
    # Do NOT move feature_extractor in model-parallel mode.
    feature_extractor.eval() if mp_devs is not None else feature_extractor.eval().to(device0)
    # Place head on stage1 to avoid copying patch tokens back to cuda:0.
    head = head.eval().to(device1)
    head_dtype = module_param_dtype(head, default=torch.float32)

    rels: List[str] = []
    preds_main_list: List[torch.Tensor] = []
    preds_ratio_list: List[torch.Tensor] = []

    with torch.inference_mode():
        for images, rel_paths in dl:
            images = images.to(device0, non_blocking=True)

            # Multi-layer path
            if use_layerwise_heads and num_layers > 1 and layer_indices is not None and len(layer_indices) > 0:
                cls_list, pt_list = feature_extractor.forward_layers_cls_and_tokens(images, layer_indices)
                if len(cls_list) != len(pt_list) or len(cls_list) != num_layers:
                    raise RuntimeError("Mismatch between CLS/patch token lists and num_layers in patch-mode multi-layer inference")

                main_layers_batch: List[torch.Tensor] = []
                ratio_layers_batch: List[torch.Tensor] = []

                for l_idx, (cls_l, pt_l) in enumerate(zip(cls_list, pt_list)):
                    _ = cls_l  # CLS not used in patch-mode
                    if pt_l.dim() != 3:
                        raise RuntimeError(f"Unexpected patch tokens shape in patch-mode inference: {tuple(pt_l.shape)}")
                    B, N, C = pt_l.shape
                    if use_separate_bottlenecks and isinstance(head, MultiLayerHeadExport):
                        # New path: explicit per-layer bottlenecks encoded in head.
                        pt_l = pt_l.to(device1, non_blocking=True, dtype=head_dtype)
                        layer_main, layer_ratio = head.forward_patch_layer(pt_l, l_idx)
                    else:
                        expected_dim = head_total * num_layers
                        offset = l_idx * head_total

                        if head_num_main > 0:
                            patch_features_flat = pt_l.reshape(B * N, C).to(device1, non_blocking=True, dtype=head_dtype)  # (B*N, C)
                            out_all_patch = head(patch_features_flat)  # (B*N, head_total * L)
                            if out_all_patch.shape[1] != expected_dim:
                                raise RuntimeError(
                                    f"Unexpected packed head dimension in patch-mode multi-layer: got {out_all_patch.shape[1]}, "
                                    f"expected {expected_dim}"
                                )
                            layer_slice = out_all_patch[:, offset : offset + head_total]  # (B*N, head_total)
                            layer_main = layer_slice[:, :head_num_main].view(B, N, head_num_main).mean(dim=1)  # (B, head_num_main)
                        else:
                            layer_main = torch.empty((B, 0), dtype=torch.float32, device=device)

                        if head_num_ratio > 0:
                            patch_mean_l = pt_l.mean(dim=1).to(device1, non_blocking=True, dtype=head_dtype)  # (B, C)
                            out_all_global = head(patch_mean_l)  # (B, head_total * L)
                            if out_all_global.shape[1] != expected_dim:
                                raise RuntimeError(
                                    f"Unexpected packed head dimension for ratio logits in patch-mode multi-layer: got {out_all_global.shape[1]}, "
                                    f"expected {expected_dim}"
                                )
                            layer_slice_g = out_all_global[:, offset : offset + head_total]  # (B, head_total)
                            layer_ratio = layer_slice_g[:, head_num_main : head_num_main + head_num_ratio]
                        else:
                            layer_ratio = None

                    main_layers_batch.append(layer_main)
                    if layer_ratio is not None:
                        ratio_layers_batch.append(layer_ratio)

                # Average over layers
                out_main_patch = (
                    torch.stack(main_layers_batch, dim=0).mean(dim=0)
                    if len(main_layers_batch) > 0
                    else torch.empty((images.size(0), 0), dtype=torch.float32, device=device)
                )
                if head_num_ratio > 0 and len(ratio_layers_batch) > 0:
                    out_ratio = torch.stack(ratio_layers_batch, dim=0).mean(dim=0)
                else:
                    out_ratio = None

            else:
                # Legacy single-layer path: use last-layer patch tokens only.
                _cls, pt = feature_extractor.forward_cls_and_tokens(images)
                if pt.dim() != 3:
                    raise RuntimeError(f"Unexpected patch tokens shape in patch-mode inference: {tuple(pt.shape)}")
                B, N, C = pt.shape

                if head_num_main > 0:
                    patch_features_flat = pt.reshape(B * N, C).to(device1, non_blocking=True, dtype=head_dtype)  # (B*N, C)
                    out_all_patch = head(patch_features_flat)  # (B*N, head_total)
                    out_main_patch = out_all_patch[:, :head_num_main].view(B, N, head_num_main).mean(dim=1)
                else:
                    out_main_patch = torch.empty((B, 0), dtype=torch.float32, device=device)

                # Ratio logits from global patch-mean
                if head_num_ratio > 0:
                    patch_mean = pt.mean(dim=1).to(device1, non_blocking=True, dtype=head_dtype)  # (B, C)
                    out_all_global = head(patch_mean)  # (B, head_total)
                    out_ratio = out_all_global[:, head_num_main : head_num_main + head_num_ratio]
                else:
                    out_ratio = None

            preds_main_list.append(out_main_patch.detach().cpu().float())
            if out_ratio is not None:
                preds_ratio_list.append(out_ratio.detach().cpu().float())
            rels.extend(list(rel_paths))

    preds_main = torch.cat(preds_main_list, dim=0) if preds_main_list else torch.empty((0, head_num_main), dtype=torch.float32)
    preds_ratio: Optional[torch.Tensor]
    if head_num_ratio > 0 and preds_ratio_list:
        preds_ratio = torch.cat(preds_ratio_list, dim=0)
    else:
        preds_ratio = None
    return rels, preds_main, preds_ratio


def predict_main_and_ratio_fpn(
    backbone: nn.Module,
    head: nn.Module,
    dataset_root: str,
    image_paths: List[str],
    image_size: Tuple[int, int],
    mean: List[float],
    std: List[float],
    batch_size: int,
    num_workers: int,
    head_num_main: int,
    head_num_ratio: int,
    *,
    use_layerwise_heads: bool,
    layer_indices: Optional[List[int]] = None,
) -> Tuple[List[str], torch.Tensor, Optional[torch.Tensor]]:
    """
    FPN-head inference (Phase A).
    """
    mp_devs = mp_get_devices_from_backbone(backbone) if isinstance(backbone, nn.Module) else None
    device0 = mp_devs[0] if mp_devs is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device1 = mp_devs[1] if mp_devs is not None else device0
    device = device1

    tf = build_transforms(image_size=image_size, mean=mean, std=std)
    ds = TestImageDataset(image_paths, root_dir=dataset_root, transform=tf)
    dl = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    feature_extractor = DinoV3FeatureExtractor(backbone)
    feature_extractor.eval() if mp_devs is not None else feature_extractor.eval().to(device0)
    head = head.eval().to(device1)
    head_dtype = module_param_dtype(head, default=torch.float32)

    rels: List[str] = []
    preds_main_list: List[torch.Tensor] = []
    preds_ratio_list: List[torch.Tensor] = []

    with torch.inference_mode():
        for images, rel_paths in dl:
            images = images.to(device0, non_blocking=True)
            H = int(images.shape[-2])
            W = int(images.shape[-1])

            if use_layerwise_heads and layer_indices is not None and len(layer_indices) > 0:
                _cls_list, pt_list = feature_extractor.forward_layers_cls_and_tokens(images, layer_indices)
                pt_list = [pt.to(device1, non_blocking=True, dtype=head_dtype) for pt in pt_list]
                out = head(pt_list, image_hw=(H, W))  # type: ignore[call-arg]
            else:
                _cls, pt = feature_extractor.forward_cls_and_tokens(images)
                pt = pt.to(device1, non_blocking=True, dtype=head_dtype)
                out = head(pt, image_hw=(H, W))  # type: ignore[call-arg]

            if not isinstance(out, dict):
                raise RuntimeError("FPN head forward must return a dict")
            reg3 = out.get("reg3", None)
            ratio = out.get("ratio", None)
            if reg3 is None:
                raise RuntimeError("FPN head did not return 'reg3'")
            preds_main_list.append(reg3.detach().cpu().float())
            if head_num_ratio > 0 and ratio is not None:
                preds_ratio_list.append(ratio.detach().cpu().float())

            rels.extend(list(rel_paths))

    preds_main = torch.cat(preds_main_list, dim=0) if preds_main_list else torch.empty((0, head_num_main), dtype=torch.float32)
    preds_ratio: Optional[torch.Tensor]
    if head_num_ratio > 0 and preds_ratio_list:
        preds_ratio = torch.cat(preds_ratio_list, dim=0)
    else:
        preds_ratio = None
    return rels, preds_main, preds_ratio


def predict_main_and_ratio_dpt(
    backbone: nn.Module,
    head: nn.Module,
    dataset_root: str,
    image_paths: List[str],
    image_size: Tuple[int, int],
    mean: List[float],
    std: List[float],
    batch_size: int,
    num_workers: int,
    head_num_main: int,
    head_num_ratio: int,
    *,
    use_layerwise_heads: bool,
    layer_indices: Optional[List[int]] = None,
) -> Tuple[List[str], torch.Tensor, Optional[torch.Tensor]]:
    """
    DPT-head inference.
    """
    mp_devs = mp_get_devices_from_backbone(backbone) if isinstance(backbone, nn.Module) else None
    device0 = mp_devs[0] if mp_devs is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device1 = mp_devs[1] if mp_devs is not None else device0

    tf = build_transforms(image_size=image_size, mean=mean, std=std)
    ds = TestImageDataset(image_paths, root_dir=dataset_root, transform=tf)
    dl = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    feature_extractor = DinoV3FeatureExtractor(backbone)
    feature_extractor.eval() if mp_devs is not None else feature_extractor.eval().to(device0)
    head = head.eval().to(device1)
    head_dtype = module_param_dtype(head, default=torch.float32)

    rels: List[str] = []
    preds_main_list: List[torch.Tensor] = []
    preds_ratio_list: List[torch.Tensor] = []

    with torch.inference_mode():
        for images, rel_paths in dl:
            images = images.to(device0, non_blocking=True)
            H = int(images.shape[-2])
            W = int(images.shape[-1])

            if use_layerwise_heads and layer_indices is not None and len(layer_indices) > 0:
                cls_list, pt_list = feature_extractor.forward_layers_cls_and_tokens(images, layer_indices)
                cls_list = [c.to(device1, non_blocking=True, dtype=head_dtype) for c in cls_list]
                pt_list = [pt.to(device1, non_blocking=True, dtype=head_dtype) for pt in pt_list]
                out = head(cls_list, pt_list, image_hw=(H, W))  # type: ignore[call-arg]
            else:
                cls, pt = feature_extractor.forward_cls_and_tokens(images)
                cls = cls.to(device1, non_blocking=True, dtype=head_dtype)
                pt = pt.to(device1, non_blocking=True, dtype=head_dtype)
                out = head(cls, pt, image_hw=(H, W))  # type: ignore[call-arg]

            if not isinstance(out, dict):
                raise RuntimeError("DPT head forward must return a dict")
            reg3 = out.get("reg3", None)
            ratio = out.get("ratio", None)
            if reg3 is None:
                raise RuntimeError("DPT head did not return 'reg3'")
            preds_main_list.append(reg3.detach().cpu().float())
            if head_num_ratio > 0 and ratio is not None:
                preds_ratio_list.append(ratio.detach().cpu().float())

            rels.extend(list(rel_paths))

    preds_main = torch.cat(preds_main_list, dim=0) if preds_main_list else torch.empty((0, head_num_main), dtype=torch.float32)
    preds_ratio: Optional[torch.Tensor]
    if head_num_ratio > 0 and preds_ratio_list:
        preds_ratio = torch.cat(preds_ratio_list, dim=0)
    else:
        preds_ratio = None
    return rels, preds_main, preds_ratio


def predict_main_and_ratio_global_multilayer(
    backbone: nn.Module,
    head: nn.Module,
    dataset_root: str,
    image_paths: List[str],
    image_size: Tuple[int, int],
    mean: List[float],
    std: List[float],
    batch_size: int,
    num_workers: int,
    head_num_main: int,
    head_num_ratio: int,
    head_total: int,
    layer_indices: List[int],
    *,
    use_separate_bottlenecks: bool = False,
    use_cls_token: bool = True,
) -> Tuple[List[str], torch.Tensor, Optional[torch.Tensor]]:
    """
    Global multi-layer inference for a single head.
    """
    mp_devs = mp_get_devices_from_backbone(backbone) if isinstance(backbone, nn.Module) else None
    device0 = mp_devs[0] if mp_devs is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device1 = mp_devs[1] if mp_devs is not None else device0
    device = device1
    tf = build_transforms(image_size=image_size, mean=mean, std=std)
    ds = TestImageDataset(image_paths, root_dir=dataset_root, transform=tf)
    dl = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    feature_extractor = DinoV3FeatureExtractor(backbone)
    feature_extractor.eval() if mp_devs is not None else feature_extractor.eval().to(device0)
    head = head.eval().to(device1)
    head_dtype = module_param_dtype(head, default=torch.float32)

    num_layers = len(layer_indices)
    if num_layers <= 0:
        raise ValueError("layer_indices must contain at least one layer for multi-layer inference")

    rels: List[str] = []
    preds_main_list: List[torch.Tensor] = []
    preds_ratio_list: List[torch.Tensor] = []

    with torch.inference_mode():
        for images, rel_paths in dl:
            images = images.to(device0, non_blocking=True)
            cls_list, pt_list = feature_extractor.forward_layers_cls_and_tokens(images, layer_indices)
            if len(cls_list) != len(pt_list) or len(cls_list) != num_layers:
                raise RuntimeError("Mismatch between CLS/patch token lists and num_layers in global multi-layer inference")

            main_layers_batch: List[torch.Tensor] = []
            ratio_layers_batch: List[torch.Tensor] = []

            for l_idx, (cls_l, pt_l) in enumerate(zip(cls_list, pt_list)):
                if pt_l.dim() != 3:
                    raise RuntimeError(f"Unexpected patch tokens shape in global multi-layer inference: {tuple(pt_l.shape)}")
                B, N, C = pt_l.shape
                cls_l = cls_l.to(device1, non_blocking=True, dtype=head_dtype)
                patch_mean_l = pt_l.mean(dim=1).to(device1, non_blocking=True, dtype=head_dtype)  # (B, C)
                if use_separate_bottlenecks and isinstance(head, MultiLayerHeadExport):
                    layer_main, layer_ratio = head.forward_global_layer(cls_l, patch_mean_l, l_idx)
                else:
                    feats_l = torch.cat([cls_l, patch_mean_l], dim=-1) if use_cls_token else patch_mean_l
                    feats_l = feats_l.to(device1, non_blocking=True, dtype=head_dtype)

                    out_all = head(feats_l)  # (B, head_total * num_layers)
                    expected_dim = head_total * num_layers
                    if out_all.shape[1] != expected_dim:
                        raise RuntimeError(
                            f"Unexpected packed head dimension in global multi-layer: got {out_all.shape[1]}, "
                            f"expected {expected_dim}"
                        )
                    offset = l_idx * head_total
                    layer_slice = out_all[:, offset : offset + head_total]  # (B, head_total)

                    if head_num_main > 0:
                        layer_main = layer_slice[:, :head_num_main]  # (B, head_num_main)
                    else:
                        layer_main = torch.empty((B, 0), dtype=torch.float32, device=device)

                    if head_num_ratio > 0:
                        layer_ratio = layer_slice[:, head_num_main : head_num_main + head_num_ratio]
                    else:
                        layer_ratio = None

                main_layers_batch.append(layer_main)
                if layer_ratio is not None:
                    ratio_layers_batch.append(layer_ratio)

            B = images.size(0)
            preds_main_batch = (
                torch.stack(main_layers_batch, dim=0).mean(dim=0)
                if len(main_layers_batch) > 0
                else torch.empty((B, 0), dtype=torch.float32, device=device)
            )
            if head_num_ratio > 0 and len(ratio_layers_batch) > 0:
                preds_ratio_batch = torch.stack(ratio_layers_batch, dim=0).mean(dim=0)
            else:
                preds_ratio_batch = None

            preds_main_list.append(preds_main_batch.detach().cpu().float())
            if preds_ratio_batch is not None:
                preds_ratio_list.append(preds_ratio_batch.detach().cpu().float())
            rels.extend(list(rel_paths))

    preds_main = torch.cat(preds_main_list, dim=0) if preds_main_list else torch.empty((0, head_num_main), dtype=torch.float32)
    preds_ratio: Optional[torch.Tensor]
    if head_num_ratio > 0 and preds_ratio_list:
        preds_ratio = torch.cat(preds_ratio_list, dim=0)
    else:
        preds_ratio = None

    return rels, preds_main, preds_ratio


