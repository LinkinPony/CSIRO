from typing import Optional, Tuple
from pathlib import Path
import os
import math
import warnings

import torch
from torch import Tensor, nn

from .layer_utils import (
    normalize_layer_indices,
    split_cls_and_patches_from_intermediate,
)


def _resolve_dinov3_repo_or_dir() -> str:
    """
    Resolve the torch.hub repo_or_dir for DINOv3.

    Prefer the vendored copy under third_party/ for offline and reproducible runs.
    A manual override can be provided via the DINOV3_REPO_OR_DIR environment variable.
    """
    env = os.environ.get("DINOV3_REPO_OR_DIR", "").strip()
    if env:
        return env
    try:
        repo_root = Path(__file__).resolve().parents[2]
        local_repo = repo_root / "third_party" / "dinov3"
        if (local_repo / "hubconf.py").is_file():
            return str(local_repo)
    except Exception:
        pass
    return "facebookresearch/dinov3"


_DINOV3_REPO_OR_DIR = _resolve_dinov3_repo_or_dir()
_DINOV3_HUB_SOURCE = "local" if Path(_DINOV3_REPO_OR_DIR).is_dir() else "github"


class DinoV3FeatureExtractor(nn.Module):
    def __init__(self, backbone: nn.Module, *, inference_only: bool = True) -> None:
        super().__init__()
        self.backbone = backbone
        self.inference_only = bool(inference_only)
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False
        self.backbone.eval()

    def _infer_backbone_input_dtype(self, fallback: torch.dtype) -> torch.dtype:
        """
        Infer the dtype that the *image tensor* should be cast to before entering the
        DINOv3 backbone.

        Why not just use `next(self.backbone.parameters()).dtype`?
        - When PEFT/LoRA is enabled we may end up with mixed parameter dtypes
          (e.g., frozen base weights cast to fp16 for VRAM savings while trainable
          LoRA params remain fp32). In that case, the first parameter dtype can be
          misleading and cause conv2d dtype mismatch.

        We therefore prefer the patch-embed conv weight dtype when available.
        """
        backbone = self.backbone
        # Support PEFT-wrapped backbones: the real model may live under `base_model`.
        if hasattr(backbone, "base_model") and isinstance(getattr(backbone, "base_model"), nn.Module):
            backbone = getattr(backbone, "base_model")  # type: ignore[assignment]

        try:
            patch_embed = getattr(backbone, "patch_embed", None)
            proj = getattr(patch_embed, "proj", None)
            w = getattr(proj, "weight", None)
            if isinstance(w, torch.Tensor) and w.is_floating_point():
                return w.dtype
        except Exception:
            pass

        # Fallback: first floating-point parameter dtype
        try:
            for p in backbone.parameters():
                if isinstance(p, torch.Tensor) and p.is_floating_point():
                    return p.dtype
        except Exception:
            pass
        return fallback

    def _forward_features_dict(self, images: Tensor):
        # Ensure inputs match backbone dtype (avoid conv2d dtype mismatch).
        backbone_dtype = self._infer_backbone_input_dtype(images.dtype)
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
        # Ensure dtype consistency with backbone (avoid conv2d dtype mismatch)
        backbone_dtype = self._infer_backbone_input_dtype(images.dtype)
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

        # Validate indices early to avoid silent fallback downstream.
        # Prefer `blocks` length when available (common in ViT backbones).
        try:
            depth_model = backbone
            if hasattr(depth_model, "base_model") and isinstance(getattr(depth_model, "base_model"), nn.Module):
                depth_model = getattr(depth_model, "base_model")  # type: ignore[assignment]
            blocks = getattr(depth_model, "blocks", None)
            if isinstance(blocks, (nn.ModuleList, list)) and len(blocks) > 0:
                depth = len(blocks)
                bad = [int(i) for i in layer_indices if int(i) < 0 or int(i) >= depth]
                if bad:
                    raise ValueError(
                        f"Invalid backbone layer indices: {bad}. "
                        f"Backbone depth={depth} so valid indices are [0, {depth - 1}]."
                    )
        except ValueError:
            raise
        except Exception:
            # Best-effort only; if we cannot infer depth, defer to the backbone implementation.
            pass

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
            repo_or_dir=_DINOV3_REPO_OR_DIR,
            model="dinov3_vitl16",
            source=_DINOV3_HUB_SOURCE,
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
            repo_or_dir=_DINOV3_REPO_OR_DIR,
            model="dinov3_vitl16",
            source=_DINOV3_HUB_SOURCE,
            pretrained=pretrained,
            weights=weights_url,
            check_hash=check_hash,
        )
        return model

    model = torch.hub.load(
        repo_or_dir=_DINOV3_REPO_OR_DIR,
        model="dinov3_vitl16",
        source=_DINOV3_HUB_SOURCE,
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
            repo_or_dir=_DINOV3_REPO_OR_DIR,
            model="dinov3_vith16plus",
            source=_DINOV3_HUB_SOURCE,
            pretrained=False,
        )
        state = torch.load(weights_path, map_location="cpu")
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        model.load_state_dict(state, strict=False)
        return model

    if weights_url:
        model = torch.hub.load(
            repo_or_dir=_DINOV3_REPO_OR_DIR,
            model="dinov3_vith16plus",
            source=_DINOV3_HUB_SOURCE,
            pretrained=pretrained,
            weights=weights_url,
            check_hash=check_hash,
        )
        return model

    model = torch.hub.load(
        repo_or_dir=_DINOV3_REPO_OR_DIR,
        model="dinov3_vith16plus",
        source=_DINOV3_HUB_SOURCE,
        pretrained=pretrained,
    )
    return model


def load_dinov3_vit7b16(
    pretrained: bool = True,
    weights_url: Optional[str] = None,
    weights_path: Optional[str] = None,
    check_hash: bool = False,
) -> nn.Module:
    """
    DINOv3 ViT-7B/16 backbone (embed_dim=4096, depth=40).

    Official LVD1689M pretrain weights filename:
      dinov3_vit7b16_pretrain_lvd1689m-a955f4ea.pth
    """
    # Prefer explicit offline weights if provided
    if weights_path:
        model = torch.hub.load(
            repo_or_dir=_DINOV3_REPO_OR_DIR,
            model="dinov3_vit7b16",
            source=_DINOV3_HUB_SOURCE,
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
            repo_or_dir=_DINOV3_REPO_OR_DIR,
            model="dinov3_vit7b16",
            source=_DINOV3_HUB_SOURCE,
            pretrained=pretrained,
            weights=weights_url,
            check_hash=check_hash,
        )
        return model

    model = torch.hub.load(
        repo_or_dir=_DINOV3_REPO_OR_DIR,
        model="dinov3_vit7b16",
        source=_DINOV3_HUB_SOURCE,
        pretrained=pretrained,
    )
    return model


def build_feature_extractor(
    backbone_name: str,
    pretrained: bool = True,
    weights_url: Optional[str] = None,
    weights_path: Optional[str] = None,
    gradient_checkpointing: bool = False,
) -> nn.Module:
    """
    Build a DINOv3 feature extractor wrapper.

    Args:
        gradient_checkpointing: When True, apply activation/gradient checkpointing to the
            DINOv3 ViT transformer blocks (memory saver; slower backward). This is useful
            when training LoRA adapters with limited GPU memory.
    """

    def _enable_activation_checkpointing_on_blocks(m: nn.Module) -> bool:
        """
        Try to enable activation checkpointing by wrapping `m.blocks[i]`.
        Returns True if at least one block was wrapped.
        """
        blocks = getattr(m, "blocks", None)
        if blocks is None:
            return False
        if not isinstance(blocks, (nn.ModuleList, list)):
            return False
        if len(blocks) == 0:
            return False

        # Preferred: PyTorch checkpoint_wrapper (supports non-reentrant checkpointing and complex inputs).
        try:
            from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (  # type: ignore
                CheckpointImpl,
                checkpoint_wrapper,
            )

            wrapped = 0
            for i, b in enumerate(blocks):
                # Avoid double-wrapping
                if hasattr(b, "_checkpoint_wrapped_module"):
                    continue
                try:
                    blocks[i] = checkpoint_wrapper(  # type: ignore[index]
                        b,
                        checkpoint_impl=CheckpointImpl.NO_REENTRANT,
                        preserve_rng_state=True,
                    )
                except TypeError:
                    # Older signature fallback
                    blocks[i] = checkpoint_wrapper(  # type: ignore[index]
                        b,
                        preserve_rng_state=True,
                    )
                wrapped += 1
            return wrapped > 0
        except Exception:
            pass

        # Fallback: torch.utils.checkpoint.checkpoint with a small wrapper to handle list inputs.
        try:
            import inspect
            from torch.utils.checkpoint import checkpoint

            supports_use_reentrant = "use_reentrant" in inspect.signature(checkpoint).parameters
        except Exception:
            return False

        class _CheckpointBlock(nn.Module):
            def __init__(self, inner: nn.Module) -> None:
                super().__init__()
                self.inner = inner

            def forward(self, x_or_x_list, rope_or_rope_list=None):
                # Case A: tensor input (used by some DINO paths, e.g., intermediate layers)
                if isinstance(x_or_x_list, Tensor):
                    x = x_or_x_list
                    rope = rope_or_rope_list

                    def _fn(x_in: Tensor, sin: Optional[Tensor] = None, cos: Optional[Tensor] = None):
                        if sin is None or cos is None:
                            return self.inner(x_in, None)
                        return self.inner(x_in, (sin, cos))

                    if rope is None:
                        if supports_use_reentrant:
                            return checkpoint(_fn, x, None, None, use_reentrant=False, preserve_rng_state=True)
                        return checkpoint(_fn, x, None, None, preserve_rng_state=True)

                    sin, cos = rope
                    if supports_use_reentrant:
                        return checkpoint(_fn, x, sin, cos, use_reentrant=False, preserve_rng_state=True)
                    return checkpoint(_fn, x, sin, cos, preserve_rng_state=True)

                # Case B: list input (DINOv3 forward_features_list uses list-of-tensors)
                x_list = x_or_x_list
                rope_list = rope_or_rope_list
                if rope_list is None:
                    rope_list = [None for _ in x_list]
                L = len(x_list)
                rope_is_none = [r is None for r in rope_list]

                def _fn(*flat: Tensor):
                    xs = list(flat[:L])
                    rope_t = flat[L:]
                    idx = 0
                    rl = []
                    for is_none in rope_is_none:
                        if is_none:
                            rl.append(None)
                        else:
                            sin = rope_t[idx]
                            cos = rope_t[idx + 1]
                            idx += 2
                            rl.append((sin, cos))
                    out_list = self.inner(xs, rl)
                    return tuple(out_list)

                args = list(x_list)
                rope_args: list[Tensor] = []
                for r in rope_list:
                    if r is None:
                        continue
                    rope_args.extend([r[0], r[1]])

                if supports_use_reentrant:
                    outs = checkpoint(_fn, *args, *rope_args, use_reentrant=False, preserve_rng_state=True)
                else:
                    outs = checkpoint(_fn, *args, *rope_args, preserve_rng_state=True)
                return list(outs)

        wrapped = 0
        for i, b in enumerate(blocks):
            if isinstance(b, _CheckpointBlock) or hasattr(b, "_checkpoint_wrapped_module"):
                continue
            blocks[i] = _CheckpointBlock(b)  # type: ignore[index]
            wrapped += 1
        return wrapped > 0

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
    elif backbone_name in ("dinov3_vit7b16", "dinov3_vit7b", "vit7b16", "vit7b"):
        backbone = load_dinov3_vit7b16(
            pretrained=use_pretrained,
            weights_url=weights_url if not weights_path else None,
            weights_path=weights_path,
        )
    else:
        raise ValueError(f"Unsupported backbone: {backbone_name}")

    if bool(gradient_checkpointing):
        applied = _enable_activation_checkpointing_on_blocks(backbone)
        # Some wrappers (e.g., PEFT) may expose the actual model under base_model
        if (not applied) and hasattr(backbone, "base_model") and isinstance(getattr(backbone, "base_model"), nn.Module):
            applied = _enable_activation_checkpointing_on_blocks(backbone.base_model)  # type: ignore[attr-defined]
        if not applied:
            warnings.warn(
                "gradient_checkpointing was requested but could not be enabled on this backbone "
                "(no `blocks` attribute found or checkpoint wrapper unavailable)."
            )
    return DinoV3FeatureExtractor(backbone)


