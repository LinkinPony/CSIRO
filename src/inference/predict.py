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


def _enable_dropout_only_train_mode(module: nn.Module, *, enabled: bool) -> int:
    """
    Enable/disable Dropout layers while leaving the rest of the module in eval().

    Why:
      - MC Dropout requires Dropout layers to run in training mode.
      - Some heads (e.g., DPT scalar head) contain BatchNorm; calling head.train()
        would change behavior and can update running stats. We avoid that by only
        toggling Dropout submodules.

    Returns:
        Number of dropout modules toggled.
    """
    n = 0
    for m in module.modules():
        if isinstance(
            m,
            (
                nn.Dropout,
                nn.Dropout1d,
                nn.Dropout2d,
                nn.Dropout3d,
                nn.AlphaDropout,
                nn.FeatureAlphaDropout,
            ),
        ):
            m.train(enabled)
            n += 1
    return n


def _maybe_seed_mc_dropout(seed: int) -> None:
    """
    Best-effort seeding for reproducible MC dropout sampling.
    """
    if int(seed) < 0:
        return
    s = int(seed)
    torch.manual_seed(s)
    try:
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(s)
    except Exception:
        pass


def _normalize_layer_weights(
    layer_weights: Optional[torch.Tensor],
    *,
    num_layers: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Optional[torch.Tensor]:
    """
    Normalize optional fusion weights for multi-layer inference.

    Returns:
        A tensor of shape (L,) on (device,dtype) summing to 1, or None to indicate
        uniform averaging should be used.
    """
    if layer_weights is None:
        return None
    if not isinstance(layer_weights, torch.Tensor):
        return None
    if layer_weights.dim() != 1 or int(layer_weights.numel()) != int(num_layers):
        return None
    w = layer_weights.to(device=device, dtype=dtype)
    w = w / w.sum().clamp_min(1e-8)
    return w


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

    @torch.inference_mode()
    def forward_tokens_until_block(
        self,
        images: torch.Tensor,
        *,
        block_idx: int,
    ) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """
        Run the DINOv3 ViT up to (but excluding) `block_idx` and return the *full* token sequence.

        Returns:
            x:  (B, N_tokens, C) token sequence containing:
                  [CLS, (storage tokens...), patch tokens]
            hw: (H_p, W_p) patch grid size for RoPE (NOT input image size).
        """
        bb: nn.Module = self.backbone
        # Best-effort unwrap PEFT / wrapper structures to the actual DINOv3 model exposing `.blocks`.
        if not hasattr(bb, "blocks"):
            base_model = getattr(bb, "base_model", None)
            if isinstance(base_model, nn.Module):
                cand = getattr(base_model, "model", None)
                if isinstance(cand, nn.Module) and hasattr(cand, "blocks"):
                    bb = cand
                elif hasattr(base_model, "blocks"):
                    bb = base_model
            cand2 = getattr(bb, "model", None)
            if isinstance(cand2, nn.Module) and hasattr(cand2, "blocks"):
                bb = cand2

        # Cast image dtype to patch-embed weight dtype when possible (conv2d dtype safety).
        try:
            patch_embed = getattr(bb, "patch_embed", None)
            proj = getattr(patch_embed, "proj", None)
            w = getattr(proj, "weight", None)
            backbone_dtype = w.dtype if isinstance(w, torch.Tensor) and w.is_floating_point() else images.dtype
        except Exception:
            backbone_dtype = images.dtype
        if images.dtype != backbone_dtype:
            images = images.to(dtype=backbone_dtype)

        blocks = getattr(bb, "blocks", None)
        if not isinstance(blocks, (nn.ModuleList, list)):
            raise RuntimeError("forward_tokens_until_block requires a DINOv3 ViT-style backbone with `.blocks`")
        depth = len(blocks)
        bi = int(block_idx)
        if bi < 0 or bi > depth:
            raise ValueError(f"block_idx must be in [0, {depth}] but got {bi} (depth={depth})")

        prepare = getattr(bb, "prepare_tokens_with_masks", None)
        if prepare is None:
            raise RuntimeError("Backbone does not expose prepare_tokens_with_masks(images, masks=None)")

        x, hw = prepare(images, None)  # type: ignore[misc]
        H_p, W_p = int(hw[0]), int(hw[1])
        rope_embed = getattr(bb, "rope_embed", None)
        rope = rope_embed(H=H_p, W=W_p) if rope_embed is not None else None
        for i in range(bi):
            x = blocks[i](x, rope)  # type: ignore[call-arg]
        return x, (H_p, W_p)


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
    hflip: bool = False,
    vflip: bool = False,
) -> Tuple[List[str], torch.Tensor]:
    mp_devs = mp_get_devices_from_backbone(feature_extractor) if isinstance(feature_extractor, nn.Module) else None
    device = mp_devs[0] if mp_devs is not None else ("cuda" if torch.cuda.is_available() else "cpu")
    tf = build_transforms(
        image_size=image_size,
        mean=mean,
        std=std,
        hflip=bool(hflip),
        vflip=bool(vflip),
    )
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
    layer_weights: Optional[torch.Tensor] = None,
    mc_dropout_enabled: bool = False,
    mc_dropout_samples: int = 1,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
    # For model-parallel backbone inference, prefer placing the head on stage1.
    mp_devs = mp_get_devices_from_backbone(head) if isinstance(head, nn.Module) else None
    device = mp_devs[1] if mp_devs is not None else ("cuda" if torch.cuda.is_available() else "cpu")
    head = head.eval().to(device)
    compute_var = bool(mc_dropout_enabled) and int(mc_dropout_samples) > 1
    if compute_var:
        _enable_dropout_only_train_mode(head, enabled=True)
    N = features_cpu.shape[0]
    preds_main_list: List[torch.Tensor] = []
    preds_ratio_list: List[torch.Tensor] = []
    preds_main_var_list: List[torch.Tensor] = []
    preds_ratio_var_list: List[torch.Tensor] = []
    head_dtype = module_param_dtype(head, default=torch.float32)
    with torch.inference_mode():
        for i in range(0, N, max(1, batch_size)):
            chunk = features_cpu[i : i + max(1, batch_size)].to(device, non_blocking=True, dtype=head_dtype)
            k = max(1, int(mc_dropout_samples)) if bool(mc_dropout_enabled) else 1
            B = int(chunk.shape[0])
            has_ratio = bool(head_num_ratio > 0 and head_total == head_num_main + head_num_ratio)
            use_layers = bool(use_layerwise_heads) and int(num_layers) > 1

            # Normalize layer fusion weights (float32 for stable var accumulation).
            w = (
                _normalize_layer_weights(
                    layer_weights,
                    num_layers=int(num_layers),
                    device=torch.device(device) if not isinstance(device, torch.device) else device,
                    dtype=torch.float32,
                )
                if use_layers
                else None
            )

            main_sum = None
            main_sq_sum = None
            ratio_sum = None
            ratio_sq_sum = None

            for _ in range(k):
                out_k = head(chunk)
                out_f = out_k.float()

                if not use_layers:
                    # Legacy single-layer outputs.
                    if has_ratio:
                        main_k = out_f[:, :head_num_main]
                        ratio_k = out_f[:, head_num_main : head_num_main + head_num_ratio]
                    else:
                        main_k = out_f
                        ratio_k = None
                else:
                    # Layer-wise packed outputs: fuse per-sample so MC var reflects the fused prediction.
                    if has_ratio:
                        expected_dim = head_total * num_layers
                        if out_f.shape[1] != expected_dim:
                            raise RuntimeError(
                                f"Unexpected packed head dimension: got {out_f.shape[1]}, expected {expected_dim}"
                            )
                        out_L = out_f.view(B, num_layers, head_total)
                        main_layers = out_L[:, :, :head_num_main]
                        ratio_layers = out_L[:, :, head_num_main : head_num_main + head_num_ratio]
                        if w is None:
                            main_k = main_layers.mean(dim=1)
                            ratio_k = ratio_layers.mean(dim=1)
                        else:
                            main_k = (main_layers * w.view(1, num_layers, 1)).sum(dim=1)
                            ratio_k = (ratio_layers * w.view(1, num_layers, 1)).sum(dim=1)
                    else:
                        expected_dim = head_num_main * num_layers
                        if out_f.shape[1] != expected_dim:
                            raise RuntimeError(
                                f"Unexpected packed head dimension (no-ratio): got {out_f.shape[1]}, expected {expected_dim}"
                            )
                        out_L = out_f.view(B, num_layers, head_num_main)
                        if w is None:
                            main_k = out_L.mean(dim=1)
                        else:
                            main_k = (out_L * w.view(1, num_layers, 1)).sum(dim=1)
                        ratio_k = None

                main_sum = main_k if main_sum is None else (main_sum + main_k)
                if compute_var:
                    main_sq = main_k * main_k
                    main_sq_sum = main_sq if main_sq_sum is None else (main_sq_sum + main_sq)

                if ratio_k is not None:
                    ratio_sum = ratio_k if ratio_sum is None else (ratio_sum + ratio_k)
                    if compute_var:
                        ratio_sq = ratio_k * ratio_k
                        ratio_sq_sum = ratio_sq if ratio_sq_sum is None else (ratio_sq_sum + ratio_sq)

            main_mean = main_sum / float(k)  # type: ignore[operator]
            preds_main_list.append(main_mean.detach().cpu().float())
            if compute_var and main_sq_sum is not None:
                main_var = (main_sq_sum / float(k)) - (main_mean * main_mean)
                preds_main_var_list.append(main_var.clamp_min(0.0).detach().cpu().float())

            if ratio_sum is not None:
                ratio_mean = ratio_sum / float(k)
                preds_ratio_list.append(ratio_mean.detach().cpu().float())
                if compute_var and ratio_sq_sum is not None:
                    ratio_var = (ratio_sq_sum / float(k)) - (ratio_mean * ratio_mean)
                    preds_ratio_var_list.append(ratio_var.clamp_min(0.0).detach().cpu().float())

    if not preds_main_list:
        empty_main = torch.empty((0, head_num_main), dtype=torch.float32)
        empty_ratio = torch.empty((0, head_num_ratio), dtype=torch.float32) if head_num_ratio > 0 else None
        return empty_main, empty_ratio, None, None

    preds_main = torch.cat(preds_main_list, dim=0)
    preds_ratio: Optional[torch.Tensor] = None
    if head_num_ratio > 0 and preds_ratio_list:
        preds_ratio = torch.cat(preds_ratio_list, dim=0)

    preds_main_var: Optional[torch.Tensor] = None
    preds_ratio_var: Optional[torch.Tensor] = None
    if compute_var and preds_main_var_list:
        preds_main_var = torch.cat(preds_main_var_list, dim=0)
    if compute_var and head_num_ratio > 0 and preds_ratio_var_list:
        preds_ratio_var = torch.cat(preds_ratio_var_list, dim=0)

    return preds_main, preds_ratio, preds_main_var, preds_ratio_var


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
    layer_weights: Optional[torch.Tensor] = None,
    *,
    mc_dropout_enabled: bool = False,
    mc_dropout_samples: int = 1,
    hflip: bool = False,
    vflip: bool = False,
) -> Tuple[List[str], torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
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
    tf = build_transforms(
        image_size=image_size,
        mean=mean,
        std=std,
        hflip=bool(hflip),
        vflip=bool(vflip),
    )
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
    compute_var = bool(mc_dropout_enabled) and int(mc_dropout_samples) > 1
    if compute_var:
        _enable_dropout_only_train_mode(head, enabled=True)
    head_dtype = module_param_dtype(head, default=torch.float32)

    rels: List[str] = []
    preds_main_list: List[torch.Tensor] = []
    preds_ratio_list: List[torch.Tensor] = []
    preds_main_var_list: List[torch.Tensor] = []
    preds_ratio_var_list: List[torch.Tensor] = []

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
                main_layers_var_batch: List[torch.Tensor] = []
                ratio_layers_var_batch: List[torch.Tensor] = []

                for l_idx, (cls_l, pt_l) in enumerate(zip(cls_list, pt_list)):
                    _ = cls_l  # CLS not used in patch-mode
                    if pt_l.dim() != 3:
                        raise RuntimeError(f"Unexpected patch tokens shape in patch-mode inference: {tuple(pt_l.shape)}")
                    B, N, C = pt_l.shape
                    if use_separate_bottlenecks and isinstance(head, MultiLayerHeadExport):
                        # New path: explicit per-layer bottlenecks encoded in head.
                        pt_l = pt_l.to(device1, non_blocking=True, dtype=head_dtype)
                        k = max(1, int(mc_dropout_samples)) if bool(mc_dropout_enabled) else 1
                        main_sum = None
                        main_sq_sum = None
                        ratio_sum = None
                        ratio_sq_sum = None
                        for _ in range(k):
                            layer_main_k, layer_ratio_k = head.forward_patch_layer(pt_l, l_idx)
                            mk = layer_main_k.float()
                            main_sum = mk if main_sum is None else (main_sum + mk)
                            if compute_var:
                                msq = mk * mk
                                main_sq_sum = msq if main_sq_sum is None else (main_sq_sum + msq)
                            if head_num_ratio > 0 and layer_ratio_k is not None:
                                rk = layer_ratio_k.float()
                                ratio_sum = rk if ratio_sum is None else (ratio_sum + rk)
                                if compute_var:
                                    rsq = rk * rk
                                    ratio_sq_sum = rsq if ratio_sq_sum is None else (ratio_sq_sum + rsq)
                        layer_main = main_sum / float(k)  # type: ignore[operator]
                        layer_main_var = None
                        if compute_var and main_sq_sum is not None:
                            layer_main_var = ((main_sq_sum / float(k)) - (layer_main * layer_main)).clamp_min(0.0)
                        layer_ratio = (ratio_sum / float(k)) if ratio_sum is not None else None
                        layer_ratio_var = None
                        if compute_var and ratio_sq_sum is not None and layer_ratio is not None:
                            layer_ratio_var = ((ratio_sq_sum / float(k)) - (layer_ratio * layer_ratio)).clamp_min(0.0)
                    else:
                        expected_dim = head_total * num_layers
                        offset = l_idx * head_total

                        if head_num_main > 0:
                            patch_features_flat = pt_l.reshape(B * N, C).to(device1, non_blocking=True, dtype=head_dtype)  # (B*N, C)
                            k = max(1, int(mc_dropout_samples)) if bool(mc_dropout_enabled) else 1
                            main_sum = None
                            main_sq_sum = None
                            for _ in range(k):
                                out_k = head(patch_features_flat)  # (B*N, head_total * L)
                                out_f = out_k.float()
                                if out_f.shape[1] != expected_dim:
                                    raise RuntimeError(
                                        f"Unexpected packed head dimension in patch-mode multi-layer: got {out_f.shape[1]}, "
                                        f"expected {expected_dim}"
                                    )
                                layer_slice = out_f[:, offset : offset + head_total]  # (B*N, head_total)
                                main_k = layer_slice[:, :head_num_main].view(B, N, head_num_main).mean(dim=1)  # (B, head_num_main)
                                main_sum = main_k if main_sum is None else (main_sum + main_k)
                                if compute_var:
                                    msq = main_k * main_k
                                    main_sq_sum = msq if main_sq_sum is None else (main_sq_sum + msq)
                            layer_main = main_sum / float(k)  # type: ignore[operator]
                            layer_main_var = None
                            if compute_var and main_sq_sum is not None:
                                layer_main_var = ((main_sq_sum / float(k)) - (layer_main * layer_main)).clamp_min(0.0)
                        else:
                            layer_main = torch.empty((B, 0), dtype=torch.float32, device=device)
                            layer_main_var = torch.empty((B, 0), dtype=torch.float32, device=device) if compute_var else None

                        if head_num_ratio > 0:
                            patch_mean_l = pt_l.mean(dim=1).to(device1, non_blocking=True, dtype=head_dtype)  # (B, C)
                            k = max(1, int(mc_dropout_samples)) if bool(mc_dropout_enabled) else 1
                            ratio_sum = None
                            ratio_sq_sum = None
                            for _ in range(k):
                                out_k_g = head(patch_mean_l)  # (B, head_total * L)
                                out_f_g = out_k_g.float()
                                if out_f_g.shape[1] != expected_dim:
                                    raise RuntimeError(
                                        f"Unexpected packed head dimension for ratio logits in patch-mode multi-layer: got {out_f_g.shape[1]}, "
                                        f"expected {expected_dim}"
                                    )
                                layer_slice_g = out_f_g[:, offset : offset + head_total]  # (B, head_total)
                                ratio_k = layer_slice_g[:, head_num_main : head_num_main + head_num_ratio]
                                ratio_sum = ratio_k if ratio_sum is None else (ratio_sum + ratio_k)
                                if compute_var:
                                    rsq = ratio_k * ratio_k
                                    ratio_sq_sum = rsq if ratio_sq_sum is None else (ratio_sq_sum + rsq)
                            layer_ratio = ratio_sum / float(k)  # type: ignore[operator]
                            layer_ratio_var = None
                            if compute_var and ratio_sq_sum is not None:
                                layer_ratio_var = ((ratio_sq_sum / float(k)) - (layer_ratio * layer_ratio)).clamp_min(0.0)
                        else:
                            layer_ratio = None
                            layer_ratio_var = None

                    main_layers_batch.append(layer_main)
                    if compute_var and layer_main_var is not None:
                        main_layers_var_batch.append(layer_main_var)
                    if layer_ratio is not None:
                        ratio_layers_batch.append(layer_ratio)
                        if compute_var and layer_ratio_var is not None:
                            ratio_layers_var_batch.append(layer_ratio_var)

                # Average over layers
                if len(main_layers_batch) > 0:
                    main_stack = torch.stack(main_layers_batch, dim=0)  # (L,B,D)
                    w = _normalize_layer_weights(
                        layer_weights,
                        num_layers=int(main_stack.shape[0]),
                        device=main_stack.device,
                        dtype=main_stack.dtype,
                    )
                    out_main_patch = main_stack.mean(dim=0) if w is None else (main_stack * w.view(-1, 1, 1)).sum(dim=0)
                else:
                    out_main_patch = torch.empty((images.size(0), 0), dtype=torch.float32, device=device)
                    w = None

                out_main_var = None
                if compute_var and len(main_layers_var_batch) == len(main_layers_batch) and len(main_layers_var_batch) > 0:
                    main_var_stack = torch.stack(main_layers_var_batch, dim=0)  # (L,B,D)
                    L = int(main_var_stack.shape[0])
                    if w is None:
                        out_main_var = main_var_stack.mean(dim=0) / float(max(1, L))
                    else:
                        out_main_var = (main_var_stack * (w.view(-1, 1, 1) ** 2)).sum(dim=0)

                if head_num_ratio > 0 and len(ratio_layers_batch) > 0:
                    ratio_stack = torch.stack(ratio_layers_batch, dim=0)  # (K,B,R)
                    w = _normalize_layer_weights(
                        layer_weights,
                        num_layers=int(ratio_stack.shape[0]),
                        device=ratio_stack.device,
                        dtype=ratio_stack.dtype,
                    )
                    out_ratio = ratio_stack.mean(dim=0) if w is None else (ratio_stack * w.view(-1, 1, 1)).sum(dim=0)
                else:
                    out_ratio = None
                    w = None

                out_ratio_var = None
                if (
                    compute_var
                    and out_ratio is not None
                    and len(ratio_layers_var_batch) == len(ratio_layers_batch)
                    and len(ratio_layers_var_batch) > 0
                ):
                    ratio_var_stack = torch.stack(ratio_layers_var_batch, dim=0)  # (L,B,R)
                    Lr = int(ratio_var_stack.shape[0])
                    if w is None:
                        out_ratio_var = ratio_var_stack.mean(dim=0) / float(max(1, Lr))
                    else:
                        out_ratio_var = (ratio_var_stack * (w.view(-1, 1, 1) ** 2)).sum(dim=0)

            else:
                # Legacy single-layer path: use last-layer patch tokens only.
                _cls, pt = feature_extractor.forward_cls_and_tokens(images)
                if pt.dim() != 3:
                    raise RuntimeError(f"Unexpected patch tokens shape in patch-mode inference: {tuple(pt.shape)}")
                B, N, C = pt.shape

                if head_num_main > 0:
                    patch_features_flat = pt.reshape(B * N, C).to(device1, non_blocking=True, dtype=head_dtype)  # (B*N, C)
                    k = max(1, int(mc_dropout_samples)) if bool(mc_dropout_enabled) else 1
                    main_sum = None
                    main_sq_sum = None
                    for _ in range(k):
                        out_k = head(patch_features_flat)  # (B*N, head_total)
                        out_f = out_k.float()
                        main_k = out_f[:, :head_num_main].view(B, N, head_num_main).mean(dim=1)
                        main_sum = main_k if main_sum is None else (main_sum + main_k)
                        if compute_var:
                            msq = main_k * main_k
                            main_sq_sum = msq if main_sq_sum is None else (main_sq_sum + msq)
                    out_main_patch = main_sum / float(k)  # type: ignore[operator]
                    out_main_var = None
                    if compute_var and main_sq_sum is not None:
                        out_main_var = ((main_sq_sum / float(k)) - (out_main_patch * out_main_patch)).clamp_min(0.0)
                else:
                    out_main_patch = torch.empty((B, 0), dtype=torch.float32, device=device)
                    out_main_var = torch.empty((B, 0), dtype=torch.float32, device=device) if compute_var else None

                # Ratio logits from global patch-mean
                if head_num_ratio > 0:
                    patch_mean = pt.mean(dim=1).to(device1, non_blocking=True, dtype=head_dtype)  # (B, C)
                    k = max(1, int(mc_dropout_samples)) if bool(mc_dropout_enabled) else 1
                    ratio_sum = None
                    ratio_sq_sum = None
                    for _ in range(k):
                        out_k_g = head(patch_mean)  # (B, head_total)
                        out_f_g = out_k_g.float()
                        ratio_k = out_f_g[:, head_num_main : head_num_main + head_num_ratio]
                        ratio_sum = ratio_k if ratio_sum is None else (ratio_sum + ratio_k)
                        if compute_var:
                            rsq = ratio_k * ratio_k
                            ratio_sq_sum = rsq if ratio_sq_sum is None else (ratio_sq_sum + rsq)
                    out_ratio = ratio_sum / float(k)  # type: ignore[operator]
                    out_ratio_var = None
                    if compute_var and ratio_sq_sum is not None:
                        out_ratio_var = ((ratio_sq_sum / float(k)) - (out_ratio * out_ratio)).clamp_min(0.0)
                else:
                    out_ratio = None
                    out_ratio_var = None

            preds_main_list.append(out_main_patch.detach().cpu().float())
            if compute_var and out_main_var is not None:
                preds_main_var_list.append(out_main_var.detach().cpu().float())
            if out_ratio is not None:
                preds_ratio_list.append(out_ratio.detach().cpu().float())
                if compute_var and out_ratio_var is not None:
                    preds_ratio_var_list.append(out_ratio_var.detach().cpu().float())
            rels.extend(list(rel_paths))

    preds_main = torch.cat(preds_main_list, dim=0) if preds_main_list else torch.empty((0, head_num_main), dtype=torch.float32)
    preds_ratio: Optional[torch.Tensor]
    if head_num_ratio > 0 and preds_ratio_list:
        preds_ratio = torch.cat(preds_ratio_list, dim=0)
    else:
        preds_ratio = None
    preds_main_var: Optional[torch.Tensor] = None
    preds_ratio_var: Optional[torch.Tensor] = None
    if compute_var and preds_main_var_list:
        preds_main_var = torch.cat(preds_main_var_list, dim=0)
    if compute_var and head_num_ratio > 0 and preds_ratio_var_list:
        preds_ratio_var = torch.cat(preds_ratio_var_list, dim=0)
    return rels, preds_main, preds_ratio, preds_main_var, preds_ratio_var


def predict_main_and_ratio_dual_branch(
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
    num_layers: int,
    layer_indices: Optional[List[int]] = None,
    layer_weights: Optional[torch.Tensor] = None,
    mc_dropout_enabled: bool = False,
    mc_dropout_samples: int = 1,
    hflip: bool = False,
    vflip: bool = False,
) -> Tuple[List[str], torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    Dual-branch inference (MLP patch-mode + global CLS+mean(patch) branch) for a single head:
      - Patch branch : per-patch predictions averaged over patches
      - Global branch: prediction from CLS+mean(patch)
      - Fusion       : y = a*y_global + (1-a)*y_patch (alpha lives inside the head module)

    Notes:
      - Requires an exported head that implements `forward_fused_layer(cls, pt, layer_idx)`.
      - Supports multi-layer backbones by fusing layer outputs using `layer_weights` when provided.
    """
    mp_devs = mp_get_devices_from_backbone(backbone) if isinstance(backbone, nn.Module) else None
    device0 = mp_devs[0] if mp_devs is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device1 = mp_devs[1] if mp_devs is not None else device0
    device = device1

    tf = build_transforms(
        image_size=image_size,
        mean=mean,
        std=std,
        hflip=bool(hflip),
        vflip=bool(vflip),
    )
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
    compute_var = bool(mc_dropout_enabled) and int(mc_dropout_samples) > 1
    if compute_var:
        _enable_dropout_only_train_mode(head, enabled=True)
    head_dtype = module_param_dtype(head, default=torch.float32)

    rels: List[str] = []
    preds_main_list: List[torch.Tensor] = []
    preds_ratio_list: List[torch.Tensor] = []
    preds_main_var_list: List[torch.Tensor] = []
    preds_ratio_var_list: List[torch.Tensor] = []

    with torch.inference_mode():
        for images, rel_paths in dl:
            images = images.to(device0, non_blocking=True)

            if bool(use_layerwise_heads) and int(num_layers) > 1 and layer_indices is not None and len(layer_indices) > 0:
                cls_list, pt_list = feature_extractor.forward_layers_cls_and_tokens(images, layer_indices)
                if len(cls_list) != len(pt_list) or len(cls_list) != int(num_layers):
                    raise RuntimeError("Mismatch between CLS/patch token lists and num_layers in dual-branch inference")

                main_layers_batch: List[torch.Tensor] = []
                ratio_layers_batch: List[torch.Tensor] = []
                main_layers_var_batch: List[torch.Tensor] = []
                ratio_layers_var_batch: List[torch.Tensor] = []

                for l_idx, (cls_l, pt_l) in enumerate(zip(cls_list, pt_list)):
                    if pt_l.dim() != 3:
                        raise RuntimeError(f"Unexpected patch tokens shape in dual-branch inference: {tuple(pt_l.shape)}")

                    cls_l = cls_l.to(device1, non_blocking=True, dtype=head_dtype)
                    pt_l = pt_l.to(device1, non_blocking=True, dtype=head_dtype)

                    k = max(1, int(mc_dropout_samples)) if bool(mc_dropout_enabled) else 1
                    main_sum = None
                    main_sq_sum = None
                    ratio_sum = None
                    ratio_sq_sum = None

                    for _ in range(k):
                        main_k, ratio_k = head.forward_fused_layer(cls_l, pt_l, int(l_idx))  # type: ignore[attr-defined]
                        mk = main_k.float()
                        main_sum = mk if main_sum is None else (main_sum + mk)
                        if compute_var:
                            msq = mk * mk
                            main_sq_sum = msq if main_sq_sum is None else (main_sq_sum + msq)
                        if head_num_ratio > 0 and ratio_k is not None:
                            rk = ratio_k.float()
                            ratio_sum = rk if ratio_sum is None else (ratio_sum + rk)
                            if compute_var:
                                rsq = rk * rk
                                ratio_sq_sum = rsq if ratio_sq_sum is None else (ratio_sq_sum + rsq)

                    layer_main = main_sum / float(k)  # type: ignore[operator]
                    layer_main_var = None
                    if compute_var and main_sq_sum is not None:
                        layer_main_var = ((main_sq_sum / float(k)) - (layer_main * layer_main)).clamp_min(0.0)

                    layer_ratio = (ratio_sum / float(k)) if ratio_sum is not None else None
                    layer_ratio_var = None
                    if compute_var and ratio_sq_sum is not None and layer_ratio is not None:
                        layer_ratio_var = ((ratio_sq_sum / float(k)) - (layer_ratio * layer_ratio)).clamp_min(0.0)

                    main_layers_batch.append(layer_main)
                    if compute_var and layer_main_var is not None:
                        main_layers_var_batch.append(layer_main_var)
                    if layer_ratio is not None:
                        ratio_layers_batch.append(layer_ratio)
                        if compute_var and layer_ratio_var is not None:
                            ratio_layers_var_batch.append(layer_ratio_var)

                # Fuse across layers
                if main_layers_batch:
                    main_stack = torch.stack(main_layers_batch, dim=0)  # (L,B,D)
                    w = _normalize_layer_weights(
                        layer_weights,
                        num_layers=int(main_stack.shape[0]),
                        device=main_stack.device,
                        dtype=main_stack.dtype,
                    )
                    out_main = main_stack.mean(dim=0) if w is None else (main_stack * w.view(-1, 1, 1)).sum(dim=0)
                else:
                    out_main = torch.empty((images.size(0), 0), dtype=torch.float32, device=device)
                    w = None

                out_main_var = None
                if compute_var and len(main_layers_var_batch) == len(main_layers_batch) and main_layers_var_batch:
                    main_var_stack = torch.stack(main_layers_var_batch, dim=0)  # (L,B,D)
                    L = int(main_var_stack.shape[0])
                    if w is None:
                        out_main_var = main_var_stack.mean(dim=0) / float(max(1, L))
                    else:
                        out_main_var = (main_var_stack * (w.view(-1, 1, 1) ** 2)).sum(dim=0)

                if head_num_ratio > 0 and ratio_layers_batch:
                    ratio_stack = torch.stack(ratio_layers_batch, dim=0)  # (K,B,R)
                    w_r = _normalize_layer_weights(
                        layer_weights,
                        num_layers=int(ratio_stack.shape[0]),
                        device=ratio_stack.device,
                        dtype=ratio_stack.dtype,
                    )
                    out_ratio = ratio_stack.mean(dim=0) if w_r is None else (ratio_stack * w_r.view(-1, 1, 1)).sum(dim=0)
                else:
                    out_ratio = None
                    w_r = None

                out_ratio_var = None
                if (
                    compute_var
                    and out_ratio is not None
                    and len(ratio_layers_var_batch) == len(ratio_layers_batch)
                    and ratio_layers_var_batch
                ):
                    ratio_var_stack = torch.stack(ratio_layers_var_batch, dim=0)  # (L,B,R)
                    Lr = int(ratio_var_stack.shape[0])
                    if w_r is None:
                        out_ratio_var = ratio_var_stack.mean(dim=0) / float(max(1, Lr))
                    else:
                        out_ratio_var = (ratio_var_stack * (w_r.view(-1, 1, 1) ** 2)).sum(dim=0)

            else:
                # Single-layer path
                cls, pt = feature_extractor.forward_cls_and_tokens(images)
                if pt.dim() != 3:
                    raise RuntimeError(f"Unexpected patch tokens shape in dual-branch inference: {tuple(pt.shape)}")
                cls = cls.to(device1, non_blocking=True, dtype=head_dtype)
                pt = pt.to(device1, non_blocking=True, dtype=head_dtype)

                k = max(1, int(mc_dropout_samples)) if bool(mc_dropout_enabled) else 1
                main_sum = None
                main_sq_sum = None
                ratio_sum = None
                ratio_sq_sum = None
                for _ in range(k):
                    main_k, ratio_k = head.forward_fused_layer(cls, pt, 0)  # type: ignore[attr-defined]
                    mk = main_k.float()
                    main_sum = mk if main_sum is None else (main_sum + mk)
                    if compute_var:
                        msq = mk * mk
                        main_sq_sum = msq if main_sq_sum is None else (main_sq_sum + msq)
                    if head_num_ratio > 0 and ratio_k is not None:
                        rk = ratio_k.float()
                        ratio_sum = rk if ratio_sum is None else (ratio_sum + rk)
                        if compute_var:
                            rsq = rk * rk
                            ratio_sq_sum = rsq if ratio_sq_sum is None else (ratio_sq_sum + rsq)

                out_main = main_sum / float(k)  # type: ignore[operator]
                out_main_var = None
                if compute_var and main_sq_sum is not None:
                    out_main_var = ((main_sq_sum / float(k)) - (out_main * out_main)).clamp_min(0.0)

                out_ratio = (ratio_sum / float(k)) if ratio_sum is not None else None
                out_ratio_var = None
                if compute_var and ratio_sq_sum is not None and out_ratio is not None:
                    out_ratio_var = ((ratio_sq_sum / float(k)) - (out_ratio * out_ratio)).clamp_min(0.0)

            preds_main_list.append(out_main.detach().cpu().float())
            if compute_var and out_main_var is not None:
                preds_main_var_list.append(out_main_var.detach().cpu().float())
            if out_ratio is not None:
                preds_ratio_list.append(out_ratio.detach().cpu().float())
                if compute_var and out_ratio_var is not None:
                    preds_ratio_var_list.append(out_ratio_var.detach().cpu().float())
            rels.extend(list(rel_paths))

    preds_main = torch.cat(preds_main_list, dim=0) if preds_main_list else torch.empty((0, head_num_main), dtype=torch.float32)
    preds_ratio: Optional[torch.Tensor]
    if head_num_ratio > 0 and preds_ratio_list:
        preds_ratio = torch.cat(preds_ratio_list, dim=0)
    else:
        preds_ratio = None
    preds_main_var: Optional[torch.Tensor] = None
    preds_ratio_var: Optional[torch.Tensor] = None
    if compute_var and preds_main_var_list:
        preds_main_var = torch.cat(preds_main_var_list, dim=0)
    if compute_var and head_num_ratio > 0 and preds_ratio_var_list:
        preds_ratio_var = torch.cat(preds_ratio_var_list, dim=0)
    return rels, preds_main, preds_ratio, preds_main_var, preds_ratio_var


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
    mc_dropout_enabled: bool = False,
    mc_dropout_samples: int = 1,
    hflip: bool = False,
    vflip: bool = False,
) -> Tuple[List[str], torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    FPN-head inference (Phase A).
    """
    mp_devs = mp_get_devices_from_backbone(backbone) if isinstance(backbone, nn.Module) else None
    device0 = mp_devs[0] if mp_devs is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device1 = mp_devs[1] if mp_devs is not None else device0
    device = device1

    tf = build_transforms(
        image_size=image_size,
        mean=mean,
        std=std,
        hflip=bool(hflip),
        vflip=bool(vflip),
    )
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
    compute_var = bool(mc_dropout_enabled) and int(mc_dropout_samples) > 1
    if compute_var:
        _enable_dropout_only_train_mode(head, enabled=True)
    head_dtype = module_param_dtype(head, default=torch.float32)

    rels: List[str] = []
    preds_main_list: List[torch.Tensor] = []
    preds_ratio_list: List[torch.Tensor] = []
    preds_main_var_list: List[torch.Tensor] = []
    preds_ratio_var_list: List[torch.Tensor] = []

    with torch.inference_mode():
        for images, rel_paths in dl:
            images = images.to(device0, non_blocking=True)
            H = int(images.shape[-2])
            W = int(images.shape[-1])

            # Backbone forward ONCE per batch; head is sampled K times on the same tokens.
            if use_layerwise_heads and layer_indices is not None and len(layer_indices) > 0:
                _cls_list, pt_list = feature_extractor.forward_layers_cls_and_tokens(images, layer_indices)
                pt_list = [pt.to(device1, non_blocking=True, dtype=head_dtype) for pt in pt_list]
                head_in = ("list", pt_list)
            else:
                _cls, pt = feature_extractor.forward_cls_and_tokens(images)
                pt = pt.to(device1, non_blocking=True, dtype=head_dtype)
                head_in = ("single", pt)

            k = max(1, int(mc_dropout_samples)) if bool(mc_dropout_enabled) else 1
            reg3_sum = None
            reg3_sq_sum = None
            ratio_sum = None
            ratio_sq_sum = None
            for _ in range(k):
                if head_in[0] == "list":
                    out = head(head_in[1], image_hw=(H, W))  # type: ignore[call-arg]
                else:
                    out = head(head_in[1], image_hw=(H, W))  # type: ignore[call-arg]
                if not isinstance(out, dict):
                    raise RuntimeError("FPN head forward must return a dict")
                reg3 = out.get("reg3", None)
                ratio = out.get("ratio", None)
                if reg3 is None:
                    raise RuntimeError("FPN head did not return 'reg3'")
                reg3_f = reg3.float()
                reg3_sum = reg3_f if reg3_sum is None else (reg3_sum + reg3_f)
                if compute_var:
                    reg3_sq = reg3_f * reg3_f
                    reg3_sq_sum = reg3_sq if reg3_sq_sum is None else (reg3_sq_sum + reg3_sq)
                if head_num_ratio > 0 and ratio is not None:
                    ratio_f = ratio.float()
                    ratio_sum = ratio_f if ratio_sum is None else (ratio_sum + ratio_f)
                    if compute_var:
                        ratio_sq = ratio_f * ratio_f
                        ratio_sq_sum = ratio_sq if ratio_sq_sum is None else (ratio_sq_sum + ratio_sq)

            reg3_mean = reg3_sum / float(k)  # type: ignore[operator]
            preds_main_list.append(reg3_mean.detach().cpu().float())
            if compute_var and reg3_sq_sum is not None:
                reg3_var = (reg3_sq_sum / float(k)) - (reg3_mean * reg3_mean)
                preds_main_var_list.append(reg3_var.clamp_min(0.0).detach().cpu().float())
            if head_num_ratio > 0 and ratio_sum is not None:
                ratio_mean = ratio_sum / float(k)
                preds_ratio_list.append(ratio_mean.detach().cpu().float())
                if compute_var and ratio_sq_sum is not None:
                    ratio_var = (ratio_sq_sum / float(k)) - (ratio_mean * ratio_mean)
                    preds_ratio_var_list.append(ratio_var.clamp_min(0.0).detach().cpu().float())

            rels.extend(list(rel_paths))

    preds_main = torch.cat(preds_main_list, dim=0) if preds_main_list else torch.empty((0, head_num_main), dtype=torch.float32)
    preds_ratio: Optional[torch.Tensor]
    if head_num_ratio > 0 and preds_ratio_list:
        preds_ratio = torch.cat(preds_ratio_list, dim=0)
    else:
        preds_ratio = None
    preds_main_var: Optional[torch.Tensor] = None
    preds_ratio_var: Optional[torch.Tensor] = None
    if compute_var and preds_main_var_list:
        preds_main_var = torch.cat(preds_main_var_list, dim=0)
    if compute_var and head_num_ratio > 0 and preds_ratio_var_list:
        preds_ratio_var = torch.cat(preds_ratio_var_list, dim=0)
    return rels, preds_main, preds_ratio, preds_main_var, preds_ratio_var


def predict_main_and_ratio_eomt(
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
    mc_dropout_enabled: bool = False,
    mc_dropout_samples: int = 1,
    hflip: bool = False,
    vflip: bool = False,
) -> Tuple[List[str], torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    EoMT injected-query head inference.

    Unlike FPN/DPT/ViTDet query-pooling variants, this head must run the backbone
    blocks jointly with learnable query tokens (inserted into the last-k blocks),
    so we cannot reuse the patch-token-only inference loop.
    """
    mp_devs = mp_get_devices_from_backbone(backbone) if isinstance(backbone, nn.Module) else None
    if mp_devs is not None and torch.device(mp_devs[0]) != torch.device(mp_devs[1]):
        raise RuntimeError(
            "EoMT injected-query head inference currently does not support 2-GPU model-parallel backbones. "
            "Please run on a single GPU/CPU or use a non-injected head type."
        )
    device = mp_devs[0] if mp_devs is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tf = build_transforms(
        image_size=image_size,
        mean=mean,
        std=std,
        hflip=bool(hflip),
        vflip=bool(vflip),
    )
    ds = TestImageDataset(image_paths, root_dir=dataset_root, transform=tf)
    dl = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    feature_extractor = DinoV3FeatureExtractor(backbone)
    # In model-parallel mode we MUST NOT move the full module to a single device.
    if mp_devs is not None:
        feature_extractor.eval()
    else:
        feature_extractor.eval().to(device)

    # Keep the injected head on the SAME device/dtype as the backbone blocks it will call.
    bb: nn.Module = backbone
    if not hasattr(bb, "blocks"):
        base_model = getattr(bb, "base_model", None)
        if isinstance(base_model, nn.Module):
            cand = getattr(base_model, "model", None)
            if isinstance(cand, nn.Module) and hasattr(cand, "blocks"):
                bb = cand
            elif hasattr(base_model, "blocks"):
                bb = base_model
        cand2 = getattr(bb, "model", None)
        if isinstance(cand2, nn.Module) and hasattr(cand2, "blocks"):
            bb = cand2
    try:
        patch_embed = getattr(bb, "patch_embed", None)
        proj = getattr(patch_embed, "proj", None)
        w = getattr(proj, "weight", None)
        bb_dtype = w.dtype if isinstance(w, torch.Tensor) and w.is_floating_point() else module_param_dtype(bb, default=torch.float32)
    except Exception:
        bb_dtype = module_param_dtype(bb, default=torch.float32)
    head = head.eval().to(device).to(dtype=bb_dtype)

    compute_var = bool(mc_dropout_enabled) and int(mc_dropout_samples) > 1
    if compute_var:
        _enable_dropout_only_train_mode(head, enabled=True)

    # Compute injection boundary from head config (num_blocks = last-k blocks).
    try:
        blocks = getattr(bb, "blocks", None)
        depth = len(blocks) if isinstance(blocks, (nn.ModuleList, list)) else 0
    except Exception:
        depth = 0
    if depth <= 0:
        raise RuntimeError("EoMT injected-query head inference requires a DINOv3 ViT backbone with `.blocks`")
    k_blocks = int(getattr(head, "num_blocks", 4) or 4)
    k_blocks = int(max(0, min(k_blocks, depth)))
    start_block = int(depth - k_blocks)

    rels: List[str] = []
    preds_main_list: List[torch.Tensor] = []
    preds_ratio_list: List[torch.Tensor] = []
    preds_main_var_list: List[torch.Tensor] = []
    preds_ratio_var_list: List[torch.Tensor] = []

    with torch.inference_mode():
        for images, rel_paths in dl:
            images = images.to(device, non_blocking=True)

            # Stage A: run up to (depth - k) blocks once per batch.
            x_base, patch_hw = feature_extractor.forward_tokens_until_block(images, block_idx=int(start_block))

            k = max(1, int(mc_dropout_samples)) if bool(mc_dropout_enabled) else 1
            reg3_sum = None
            reg3_sq_sum = None
            ratio_sum = None
            ratio_sq_sum = None
            for _ in range(k):
                out = head(  # type: ignore[call-arg]
                    x_base,
                    backbone=feature_extractor.backbone,
                    patch_hw=patch_hw,
                )
                if not isinstance(out, dict):
                    raise RuntimeError("EoMT head forward must return a dict")
                reg3 = out.get("reg3", None)
                ratio = out.get("ratio", None)
                if reg3 is None:
                    raise RuntimeError("EoMT head did not return 'reg3'")

                reg3_f = reg3.float()
                reg3_sum = reg3_f if reg3_sum is None else (reg3_sum + reg3_f)
                if compute_var:
                    reg3_sq = reg3_f * reg3_f
                    reg3_sq_sum = reg3_sq if reg3_sq_sum is None else (reg3_sq_sum + reg3_sq)

                if head_num_ratio > 0 and ratio is not None:
                    ratio_f = ratio.float()
                    ratio_sum = ratio_f if ratio_sum is None else (ratio_sum + ratio_f)
                    if compute_var:
                        ratio_sq = ratio_f * ratio_f
                        ratio_sq_sum = ratio_sq if ratio_sq_sum is None else (ratio_sq_sum + ratio_sq)

            reg3_mean = reg3_sum / float(k)  # type: ignore[operator]
            preds_main_list.append(reg3_mean.detach().cpu().float())
            if compute_var and reg3_sq_sum is not None:
                reg3_var = (reg3_sq_sum / float(k)) - (reg3_mean * reg3_mean)
                preds_main_var_list.append(reg3_var.clamp_min(0.0).detach().cpu().float())

            if head_num_ratio > 0 and ratio_sum is not None:
                ratio_mean = ratio_sum / float(k)
                preds_ratio_list.append(ratio_mean.detach().cpu().float())
                if compute_var and ratio_sq_sum is not None:
                    ratio_var = (ratio_sq_sum / float(k)) - (ratio_mean * ratio_mean)
                    preds_ratio_var_list.append(ratio_var.clamp_min(0.0).detach().cpu().float())

            rels.extend(list(rel_paths))

    preds_main = (
        torch.cat(preds_main_list, dim=0)
        if preds_main_list
        else torch.empty((0, head_num_main), dtype=torch.float32)
    )
    preds_ratio: Optional[torch.Tensor]
    if head_num_ratio > 0 and preds_ratio_list:
        preds_ratio = torch.cat(preds_ratio_list, dim=0)
    else:
        preds_ratio = None
    preds_main_var: Optional[torch.Tensor] = None
    preds_ratio_var: Optional[torch.Tensor] = None
    if compute_var and preds_main_var_list:
        preds_main_var = torch.cat(preds_main_var_list, dim=0)
    if compute_var and head_num_ratio > 0 and preds_ratio_var_list:
        preds_ratio_var = torch.cat(preds_ratio_var_list, dim=0)
    return rels, preds_main, preds_ratio, preds_main_var, preds_ratio_var


def predict_main_and_ratio_vitdet(
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
    mc_dropout_enabled: bool = False,
    mc_dropout_samples: int = 1,
    hflip: bool = False,
    vflip: bool = False,
) -> Tuple[
    List[str],
    torch.Tensor,
    Optional[torch.Tensor],
    Optional[torch.Tensor],
    Optional[torch.Tensor],
    Optional[torch.Tensor],
    Optional[torch.Tensor],
]:
    """
    ViTDet-head inference (SimpleFeaturePyramid-style scalar head).
    """
    mp_devs = mp_get_devices_from_backbone(backbone) if isinstance(backbone, nn.Module) else None
    device0 = mp_devs[0] if mp_devs is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device1 = mp_devs[1] if mp_devs is not None else device0
    device = device1

    tf = build_transforms(
        image_size=image_size,
        mean=mean,
        std=std,
        hflip=bool(hflip),
        vflip=bool(vflip),
    )
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
    compute_var = bool(mc_dropout_enabled) and int(mc_dropout_samples) > 1
    if compute_var:
        _enable_dropout_only_train_mode(head, enabled=True)
    head_dtype = module_param_dtype(head, default=torch.float32)

    rels: List[str] = []
    preds_main_list: List[torch.Tensor] = []
    preds_ratio_list: List[torch.Tensor] = []
    preds_main_var_list: List[torch.Tensor] = []
    preds_ratio_var_list: List[torch.Tensor] = []
    preds_main_layers_list: List[torch.Tensor] = []
    preds_ratio_layers_list: List[torch.Tensor] = []

    with torch.inference_mode():
        for images, rel_paths in dl:
            images = images.to(device0, non_blocking=True)
            H = int(images.shape[-2])
            W = int(images.shape[-1])

            # Backbone forward ONCE per batch; head is sampled K times on the same tokens.
            if use_layerwise_heads and layer_indices is not None and len(layer_indices) > 0:
                _cls_list, pt_list = feature_extractor.forward_layers_cls_and_tokens(images, layer_indices)
                pt_list = [pt.to(device1, non_blocking=True, dtype=head_dtype) for pt in pt_list]
                head_in = ("list", pt_list)
            else:
                _cls, pt = feature_extractor.forward_cls_and_tokens(images)
                pt = pt.to(device1, non_blocking=True, dtype=head_dtype)
                head_in = ("single", pt)

            k = max(1, int(mc_dropout_samples)) if bool(mc_dropout_enabled) else 1
            reg3_sum = None
            reg3_sq_sum = None
            ratio_sum = None
            ratio_sq_sum = None
            reg3_layers_sum = None
            ratio_layers_sum = None
            for _ in range(k):
                if head_in[0] == "list":
                    out = head(head_in[1], image_hw=(H, W))  # type: ignore[call-arg]
                else:
                    out = head(head_in[1], image_hw=(H, W))  # type: ignore[call-arg]
                if not isinstance(out, dict):
                    raise RuntimeError("ViTDet head forward must return a dict")
                reg3 = out.get("reg3", None)
                ratio = out.get("ratio", None)
                reg3_layers = out.get("reg3_layers", None)
                ratio_layers = out.get("ratio_layers", None)
                if reg3 is None:
                    raise RuntimeError("ViTDet head did not return 'reg3'")
                reg3_f = reg3.float()
                reg3_sum = reg3_f if reg3_sum is None else (reg3_sum + reg3_f)
                if compute_var:
                    reg3_sq = reg3_f * reg3_f
                    reg3_sq_sum = reg3_sq if reg3_sq_sum is None else (reg3_sq_sum + reg3_sq)
                if head_num_ratio > 0 and ratio is not None:
                    ratio_f = ratio.float()
                    ratio_sum = ratio_f if ratio_sum is None else (ratio_sum + ratio_f)
                    if compute_var:
                        ratio_sq = ratio_f * ratio_f
                        ratio_sq_sum = ratio_sq if ratio_sq_sum is None else (ratio_sq_sum + ratio_sq)
                # Optional per-layer outputs (multi-layer ViTDet): keep as (B, L, D).
                if isinstance(reg3_layers, list) and len(reg3_layers) > 0:
                    try:
                        reg3_layers_t = [t for t in reg3_layers if isinstance(t, torch.Tensor)]
                        if reg3_layers_t:
                            reg3_layers_stack = torch.stack(reg3_layers_t, dim=1)
                            reg3_layers_sum = (
                                reg3_layers_stack
                                if reg3_layers_sum is None
                                else (reg3_layers_sum + reg3_layers_stack)
                            )
                    except Exception:
                        pass
                if head_num_ratio > 0 and isinstance(ratio_layers, list) and len(ratio_layers) > 0:
                    try:
                        ratio_layers_t = [t for t in ratio_layers if isinstance(t, torch.Tensor)]
                        if ratio_layers_t:
                            ratio_layers_stack = torch.stack(ratio_layers_t, dim=1)
                            ratio_layers_sum = (
                                ratio_layers_stack
                                if ratio_layers_sum is None
                                else (ratio_layers_sum + ratio_layers_stack)
                            )
                    except Exception:
                        pass

            reg3_mean = reg3_sum / float(k)  # type: ignore[operator]
            preds_main_list.append(reg3_mean.detach().cpu().float())
            if compute_var and reg3_sq_sum is not None:
                reg3_var = (reg3_sq_sum / float(k)) - (reg3_mean * reg3_mean)
                preds_main_var_list.append(reg3_var.clamp_min(0.0).detach().cpu().float())
            if head_num_ratio > 0 and ratio_sum is not None:
                ratio_mean = ratio_sum / float(k)
                preds_ratio_list.append(ratio_mean.detach().cpu().float())
                if compute_var and ratio_sq_sum is not None:
                    ratio_var = (ratio_sq_sum / float(k)) - (ratio_mean * ratio_mean)
                    preds_ratio_var_list.append(ratio_var.clamp_min(0.0).detach().cpu().float())
            if reg3_layers_sum is not None:
                reg3_layers_mean = reg3_layers_sum / float(k)
                preds_main_layers_list.append(reg3_layers_mean.detach().cpu().float())
            if head_num_ratio > 0 and ratio_layers_sum is not None:
                ratio_layers_mean = ratio_layers_sum / float(k)
                preds_ratio_layers_list.append(ratio_layers_mean.detach().cpu().float())

            rels.extend(list(rel_paths))

    preds_main = (
        torch.cat(preds_main_list, dim=0)
        if preds_main_list
        else torch.empty((0, head_num_main), dtype=torch.float32)
    )
    preds_ratio: Optional[torch.Tensor]
    if head_num_ratio > 0 and preds_ratio_list:
        preds_ratio = torch.cat(preds_ratio_list, dim=0)
    else:
        preds_ratio = None
    preds_main_layers: Optional[torch.Tensor]
    if preds_main_layers_list:
        preds_main_layers = torch.cat(preds_main_layers_list, dim=0)
    else:
        preds_main_layers = None
    preds_ratio_layers: Optional[torch.Tensor]
    if head_num_ratio > 0 and preds_ratio_layers_list:
        preds_ratio_layers = torch.cat(preds_ratio_layers_list, dim=0)
    else:
        preds_ratio_layers = None
    preds_main_var: Optional[torch.Tensor] = None
    preds_ratio_var: Optional[torch.Tensor] = None
    if compute_var and preds_main_var_list:
        preds_main_var = torch.cat(preds_main_var_list, dim=0)
    if compute_var and head_num_ratio > 0 and preds_ratio_var_list:
        preds_ratio_var = torch.cat(preds_ratio_var_list, dim=0)
    return rels, preds_main, preds_ratio, preds_main_layers, preds_ratio_layers, preds_main_var, preds_ratio_var


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
    mc_dropout_enabled: bool = False,
    mc_dropout_samples: int = 1,
    hflip: bool = False,
    vflip: bool = False,
) -> Tuple[List[str], torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    DPT-head inference.
    """
    mp_devs = mp_get_devices_from_backbone(backbone) if isinstance(backbone, nn.Module) else None
    device0 = mp_devs[0] if mp_devs is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device1 = mp_devs[1] if mp_devs is not None else device0

    tf = build_transforms(
        image_size=image_size,
        mean=mean,
        std=std,
        hflip=bool(hflip),
        vflip=bool(vflip),
    )
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
    compute_var = bool(mc_dropout_enabled) and int(mc_dropout_samples) > 1
    if compute_var:
        _enable_dropout_only_train_mode(head, enabled=True)
    head_dtype = module_param_dtype(head, default=torch.float32)

    rels: List[str] = []
    preds_main_list: List[torch.Tensor] = []
    preds_ratio_list: List[torch.Tensor] = []
    preds_main_var_list: List[torch.Tensor] = []
    preds_ratio_var_list: List[torch.Tensor] = []

    with torch.inference_mode():
        for images, rel_paths in dl:
            images = images.to(device0, non_blocking=True)
            H = int(images.shape[-2])
            W = int(images.shape[-1])

            # Backbone forward ONCE per batch; head is sampled K times on the same tokens.
            if use_layerwise_heads and layer_indices is not None and len(layer_indices) > 0:
                cls_list, pt_list = feature_extractor.forward_layers_cls_and_tokens(images, layer_indices)
                cls_list = [c.to(device1, non_blocking=True, dtype=head_dtype) for c in cls_list]
                pt_list = [pt.to(device1, non_blocking=True, dtype=head_dtype) for pt in pt_list]
                head_in = ("list", (cls_list, pt_list))
            else:
                cls, pt = feature_extractor.forward_cls_and_tokens(images)
                cls = cls.to(device1, non_blocking=True, dtype=head_dtype)
                pt = pt.to(device1, non_blocking=True, dtype=head_dtype)
                head_in = ("single", (cls, pt))

            k = max(1, int(mc_dropout_samples)) if bool(mc_dropout_enabled) else 1
            reg3_sum = None
            reg3_sq_sum = None
            ratio_sum = None
            ratio_sq_sum = None
            for _ in range(k):
                if head_in[0] == "list":
                    cls_list_i, pt_list_i = head_in[1]
                    out = head(cls_list_i, pt_list_i, image_hw=(H, W))  # type: ignore[call-arg]
                else:
                    cls_i, pt_i = head_in[1]
                    out = head(cls_i, pt_i, image_hw=(H, W))  # type: ignore[call-arg]
                if not isinstance(out, dict):
                    raise RuntimeError("DPT head forward must return a dict")
                reg3 = out.get("reg3", None)
                ratio = out.get("ratio", None)
                if reg3 is None:
                    raise RuntimeError("DPT head did not return 'reg3'")
                reg3_f = reg3.float()
                reg3_sum = reg3_f if reg3_sum is None else (reg3_sum + reg3_f)
                if compute_var:
                    reg3_sq = reg3_f * reg3_f
                    reg3_sq_sum = reg3_sq if reg3_sq_sum is None else (reg3_sq_sum + reg3_sq)
                if head_num_ratio > 0 and ratio is not None:
                    ratio_f = ratio.float()
                    ratio_sum = ratio_f if ratio_sum is None else (ratio_sum + ratio_f)
                    if compute_var:
                        ratio_sq = ratio_f * ratio_f
                        ratio_sq_sum = ratio_sq if ratio_sq_sum is None else (ratio_sq_sum + ratio_sq)

            reg3_mean = reg3_sum / float(k)  # type: ignore[operator]
            preds_main_list.append(reg3_mean.detach().cpu().float())
            if compute_var and reg3_sq_sum is not None:
                reg3_var = (reg3_sq_sum / float(k)) - (reg3_mean * reg3_mean)
                preds_main_var_list.append(reg3_var.clamp_min(0.0).detach().cpu().float())
            if head_num_ratio > 0 and ratio_sum is not None:
                ratio_mean = ratio_sum / float(k)  # type: ignore[operator]
                preds_ratio_list.append(ratio_mean.detach().cpu().float())
                if compute_var and ratio_sq_sum is not None:
                    ratio_var = (ratio_sq_sum / float(k)) - (ratio_mean * ratio_mean)
                    preds_ratio_var_list.append(ratio_var.clamp_min(0.0).detach().cpu().float())

            rels.extend(list(rel_paths))

    preds_main = torch.cat(preds_main_list, dim=0) if preds_main_list else torch.empty((0, head_num_main), dtype=torch.float32)
    preds_ratio: Optional[torch.Tensor]
    if head_num_ratio > 0 and preds_ratio_list:
        preds_ratio = torch.cat(preds_ratio_list, dim=0)
    else:
        preds_ratio = None
    preds_main_var: Optional[torch.Tensor] = None
    preds_ratio_var: Optional[torch.Tensor] = None
    if compute_var and preds_main_var_list:
        preds_main_var = torch.cat(preds_main_var_list, dim=0)
    if compute_var and head_num_ratio > 0 and preds_ratio_var_list:
        preds_ratio_var = torch.cat(preds_ratio_var_list, dim=0)
    return rels, preds_main, preds_ratio, preds_main_var, preds_ratio_var


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
    layer_weights: Optional[torch.Tensor] = None,
    mc_dropout_enabled: bool = False,
    mc_dropout_samples: int = 1,
    hflip: bool = False,
    vflip: bool = False,
) -> Tuple[List[str], torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    Global multi-layer inference for a single head.
    """
    mp_devs = mp_get_devices_from_backbone(backbone) if isinstance(backbone, nn.Module) else None
    device0 = mp_devs[0] if mp_devs is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device1 = mp_devs[1] if mp_devs is not None else device0
    device = device1
    tf = build_transforms(
        image_size=image_size,
        mean=mean,
        std=std,
        hflip=bool(hflip),
        vflip=bool(vflip),
    )
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
    compute_var = bool(mc_dropout_enabled) and int(mc_dropout_samples) > 1
    if compute_var:
        _enable_dropout_only_train_mode(head, enabled=True)
    head_dtype = module_param_dtype(head, default=torch.float32)

    num_layers = len(layer_indices)
    if num_layers <= 0:
        raise ValueError("layer_indices must contain at least one layer for multi-layer inference")

    rels: List[str] = []
    preds_main_list: List[torch.Tensor] = []
    preds_ratio_list: List[torch.Tensor] = []
    preds_main_var_list: List[torch.Tensor] = []
    preds_ratio_var_list: List[torch.Tensor] = []

    with torch.inference_mode():
        for images, rel_paths in dl:
            images = images.to(device0, non_blocking=True)
            cls_list, pt_list = feature_extractor.forward_layers_cls_and_tokens(images, layer_indices)
            if len(cls_list) != len(pt_list) or len(cls_list) != num_layers:
                raise RuntimeError("Mismatch between CLS/patch token lists and num_layers in global multi-layer inference")

            main_layers_batch: List[torch.Tensor] = []
            ratio_layers_batch: List[torch.Tensor] = []
            main_layers_var_batch: List[torch.Tensor] = []
            ratio_layers_var_batch: List[torch.Tensor] = []

            for l_idx, (cls_l, pt_l) in enumerate(zip(cls_list, pt_list)):
                if pt_l.dim() != 3:
                    raise RuntimeError(f"Unexpected patch tokens shape in global multi-layer inference: {tuple(pt_l.shape)}")
                B, N, C = pt_l.shape
                cls_l = cls_l.to(device1, non_blocking=True, dtype=head_dtype)
                patch_mean_l = pt_l.mean(dim=1).to(device1, non_blocking=True, dtype=head_dtype)  # (B, C)
                if use_separate_bottlenecks and isinstance(head, MultiLayerHeadExport):
                    k = max(1, int(mc_dropout_samples)) if bool(mc_dropout_enabled) else 1
                    main_sum = None
                    main_sq_sum = None
                    ratio_sum = None
                    ratio_sq_sum = None
                    for _ in range(k):
                        layer_main_k, layer_ratio_k = head.forward_global_layer(cls_l, patch_mean_l, l_idx)
                        mk = layer_main_k.float()
                        main_sum = mk if main_sum is None else (main_sum + mk)
                        if compute_var:
                            msq = mk * mk
                            main_sq_sum = msq if main_sq_sum is None else (main_sq_sum + msq)
                        if head_num_ratio > 0 and layer_ratio_k is not None:
                            rk = layer_ratio_k.float()
                            ratio_sum = rk if ratio_sum is None else (ratio_sum + rk)
                            if compute_var:
                                rsq = rk * rk
                                ratio_sq_sum = rsq if ratio_sq_sum is None else (ratio_sq_sum + rsq)
                    layer_main = main_sum / float(k)  # type: ignore[operator]
                    layer_main_var = None
                    if compute_var and main_sq_sum is not None:
                        layer_main_var = ((main_sq_sum / float(k)) - (layer_main * layer_main)).clamp_min(0.0)
                    layer_ratio = (ratio_sum / float(k)) if ratio_sum is not None else None
                    layer_ratio_var = None
                    if compute_var and ratio_sq_sum is not None and layer_ratio is not None:
                        layer_ratio_var = ((ratio_sq_sum / float(k)) - (layer_ratio * layer_ratio)).clamp_min(0.0)
                else:
                    feats_l = torch.cat([cls_l, patch_mean_l], dim=-1) if use_cls_token else patch_mean_l
                    feats_l = feats_l.to(device1, non_blocking=True, dtype=head_dtype)

                    k = max(1, int(mc_dropout_samples)) if bool(mc_dropout_enabled) else 1
                    expected_dim = head_total * num_layers
                    offset = l_idx * head_total
                    main_sum = None
                    main_sq_sum = None
                    ratio_sum = None
                    ratio_sq_sum = None
                    for _ in range(k):
                        out_k = head(feats_l)  # (B, head_total * num_layers)
                        out_f = out_k.float()
                        if out_f.shape[1] != expected_dim:
                            raise RuntimeError(
                                f"Unexpected packed head dimension in global multi-layer: got {out_f.shape[1]}, "
                                f"expected {expected_dim}"
                            )
                        layer_slice = out_f[:, offset : offset + head_total]  # (B, head_total)
                        if head_num_main > 0:
                            main_k = layer_slice[:, :head_num_main]
                            main_sum = main_k if main_sum is None else (main_sum + main_k)
                            if compute_var:
                                msq = main_k * main_k
                                main_sq_sum = msq if main_sq_sum is None else (main_sq_sum + msq)
                        if head_num_ratio > 0:
                            ratio_k = layer_slice[:, head_num_main : head_num_main + head_num_ratio]
                            ratio_sum = ratio_k if ratio_sum is None else (ratio_sum + ratio_k)
                            if compute_var:
                                rsq = ratio_k * ratio_k
                                ratio_sq_sum = rsq if ratio_sq_sum is None else (ratio_sq_sum + rsq)

                    if head_num_main > 0 and main_sum is not None:
                        layer_main = main_sum / float(k)  # type: ignore[operator]
                        layer_main_var = None
                        if compute_var and main_sq_sum is not None:
                            layer_main_var = ((main_sq_sum / float(k)) - (layer_main * layer_main)).clamp_min(0.0)
                    else:
                        layer_main = torch.empty((B, 0), dtype=torch.float32, device=device)
                        layer_main_var = torch.empty((B, 0), dtype=torch.float32, device=device) if compute_var else None

                    if head_num_ratio > 0 and ratio_sum is not None:
                        layer_ratio = ratio_sum / float(k)  # type: ignore[operator]
                        layer_ratio_var = None
                        if compute_var and ratio_sq_sum is not None:
                            layer_ratio_var = ((ratio_sq_sum / float(k)) - (layer_ratio * layer_ratio)).clamp_min(0.0)
                    else:
                        layer_ratio = None
                        layer_ratio_var = None

                main_layers_batch.append(layer_main)
                if compute_var and layer_main_var is not None:
                    main_layers_var_batch.append(layer_main_var)
                if layer_ratio is not None:
                    ratio_layers_batch.append(layer_ratio)
                    if compute_var and layer_ratio_var is not None:
                        ratio_layers_var_batch.append(layer_ratio_var)

            B = images.size(0)
            if len(main_layers_batch) > 0:
                main_stack = torch.stack(main_layers_batch, dim=0)  # (L,B,D)
                w = _normalize_layer_weights(
                    layer_weights,
                    num_layers=int(main_stack.shape[0]),
                    device=main_stack.device,
                    dtype=main_stack.dtype,
                )
                preds_main_batch = main_stack.mean(dim=0) if w is None else (main_stack * w.view(-1, 1, 1)).sum(dim=0)
            else:
                preds_main_batch = torch.empty((B, 0), dtype=torch.float32, device=device)
                w = None

            preds_main_var_batch = None
            if compute_var and len(main_layers_var_batch) == len(main_layers_batch) and len(main_layers_var_batch) > 0:
                main_var_stack = torch.stack(main_layers_var_batch, dim=0)  # (L,B,D)
                L = int(main_var_stack.shape[0])
                if w is None:
                    preds_main_var_batch = main_var_stack.mean(dim=0) / float(max(1, L))
                else:
                    preds_main_var_batch = (main_var_stack * (w.view(-1, 1, 1) ** 2)).sum(dim=0)

            if head_num_ratio > 0 and len(ratio_layers_batch) > 0:
                ratio_stack = torch.stack(ratio_layers_batch, dim=0)  # (K,B,R)
                w = _normalize_layer_weights(
                    layer_weights,
                    num_layers=int(ratio_stack.shape[0]),
                    device=ratio_stack.device,
                    dtype=ratio_stack.dtype,
                )
                preds_ratio_batch = ratio_stack.mean(dim=0) if w is None else (ratio_stack * w.view(-1, 1, 1)).sum(dim=0)
            else:
                preds_ratio_batch = None
                w = None

            preds_ratio_var_batch = None
            if (
                compute_var
                and preds_ratio_batch is not None
                and len(ratio_layers_var_batch) == len(ratio_layers_batch)
                and len(ratio_layers_var_batch) > 0
            ):
                ratio_var_stack = torch.stack(ratio_layers_var_batch, dim=0)  # (L,B,R)
                Lr = int(ratio_var_stack.shape[0])
                if w is None:
                    preds_ratio_var_batch = ratio_var_stack.mean(dim=0) / float(max(1, Lr))
                else:
                    preds_ratio_var_batch = (ratio_var_stack * (w.view(-1, 1, 1) ** 2)).sum(dim=0)

            preds_main_list.append(preds_main_batch.detach().cpu().float())
            if compute_var and preds_main_var_batch is not None:
                preds_main_var_list.append(preds_main_var_batch.detach().cpu().float())
            if preds_ratio_batch is not None:
                preds_ratio_list.append(preds_ratio_batch.detach().cpu().float())
                if compute_var and preds_ratio_var_batch is not None:
                    preds_ratio_var_list.append(preds_ratio_var_batch.detach().cpu().float())
            rels.extend(list(rel_paths))

    preds_main = torch.cat(preds_main_list, dim=0) if preds_main_list else torch.empty((0, head_num_main), dtype=torch.float32)
    preds_ratio: Optional[torch.Tensor]
    if head_num_ratio > 0 and preds_ratio_list:
        preds_ratio = torch.cat(preds_ratio_list, dim=0)
    else:
        preds_ratio = None

    preds_main_var: Optional[torch.Tensor] = None
    preds_ratio_var: Optional[torch.Tensor] = None
    if compute_var and preds_main_var_list:
        preds_main_var = torch.cat(preds_main_var_list, dim=0)
    if compute_var and head_num_ratio > 0 and preds_ratio_var_list:
        preds_ratio_var = torch.cat(preds_ratio_var_list, dim=0)

    return rels, preds_main, preds_ratio, preds_main_var, preds_ratio_var


