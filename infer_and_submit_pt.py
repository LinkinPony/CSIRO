# ===== Required user variables =====
# Backward-compat: WEIGHTS_PT_PATH is ignored when HEAD_WEIGHTS_PT_PATH is provided.
HEAD_WEIGHTS_PT_PATH = "weights/head/"  # regression head-only weights (.pt)
# Backbone weights path can be EITHER:
#  - a single weights file (.pt or .pth), OR
#  - a directory that contains the official DINOv3 weights files.
# In directory mode, the script will auto-select the correct file based on the backbone name.
DINO_WEIGHTS_PT_PATH = "dinov3_weights"  # file or directory containing DINOv3 weights
INPUT_PATH = "data"  # dir containing test.csv & images, or a direct test.csv path
OUTPUT_SUBMISSION_PATH = "submission.csv"
DINOV3_PATH = "third_party/dinov3/dinov3"  # path to dinov3 source folder (contains dinov3/*)
PEFT_PATH = "third_party/peft/src"  # path to peft source folder (contains peft/*)

# New: specify the project directory that contains both `configs/` and `src/` folders.
# Example: PROJECT_DIR = "/media/dl/dataset/Git/CSIRO"
PROJECT_DIR = "."

# ===== Multi-GPU model-parallel inference (Scheme B) =====
# When running the VERY large dinov3_vit7b16 backbone on 2x16GB GPUs (e.g., Kaggle T4),
# we split the transformer blocks across cuda:0 and cuda:1 and move the token activations
# across devices once at the split boundary.
#
# Notes:
# - This is NOT data-parallel; it aims to *split model weights* across GPUs to fit in memory.
# - Only enabled when >= 2 CUDA devices are available and the backbone is dinov3_vit7b16.
USE_2GPU_MODEL_PARALLEL_FOR_VIT7B = True
# Number of transformer blocks to place on cuda:0 (the remaining go to cuda:1).
# For ViT7B (40 blocks), a 20/20 split is a reasonable default.
VIT7B_MP_SPLIT_IDX = 20
# Use fp16 weights for the backbone/head in model-parallel mode to fit on 2x16GB GPUs.
VIT7B_MP_DTYPE = "fp16"  # one of: "fp16", "fp32"
# ===================================
# ==========================================================
# INFERENCE SCRIPT (UPDATED REQUIREMENTS)
# - Allowed to import this project's source code from `src/` and configuration from `configs/`.
# - Ensures single source of truth: model head settings, image transforms, etc. are read from YAML config.
# - A valid PROJECT_DIR must be provided, and it must contain both `configs/` and `src/`.
# - Inference requires two weights when using the new format:
#   1) DINOv3 backbone weights: DINO_WEIGHTS_PT_PATH (frozen, shared across runs)
#   2) Regression head weights: HEAD_WEIGHTS_PT_PATH (packaged as weights/head/infer_head.pt)
# - Legacy support: if HEAD_WEIGHTS_PT_PATH is empty, fall back to WEIGHTS_PT_PATH.
# ==========================================================


import os
import re
import sys
import types
import glob
import tempfile
from typing import Dict, List, Tuple, Optional

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from PIL import Image
import pandas as pd
import json

from torch import nn
import torch.nn.functional as F


import yaml

# ===== Add local dinov3 to import path (optional, for offline use) =====
_DINOV3_DIR = os.path.abspath(DINOV3_PATH) if DINOV3_PATH else ""
if _DINOV3_DIR and os.path.isdir(_DINOV3_DIR) and _DINOV3_DIR not in sys.path:
    sys.path.insert(0, _DINOV3_DIR)

# ===== Prefer vendored PEFT over system installation (if available) =====
_PEFT_DIR = os.path.abspath(PEFT_PATH) if PEFT_PATH else ""
if _PEFT_DIR and os.path.isdir(_PEFT_DIR) and _PEFT_DIR not in sys.path:
    sys.path.insert(0, _PEFT_DIR)

# ===== Validate project directory and import project modules =====
_PROJECT_DIR_ABS = os.path.abspath(PROJECT_DIR) if PROJECT_DIR else ""
_CONFIGS_DIR = os.path.join(_PROJECT_DIR_ABS, "configs")
_SRC_DIR = os.path.join(_PROJECT_DIR_ABS, "src")
if not (_PROJECT_DIR_ABS and os.path.isdir(_PROJECT_DIR_ABS)):
    raise RuntimeError("PROJECT_DIR must point to the repository root containing `configs/` and `src/`.")
if not os.path.isdir(_CONFIGS_DIR):
    raise RuntimeError(f"configs/ not found under PROJECT_DIR: {_CONFIGS_DIR}")
if not os.path.isdir(_SRC_DIR):
    raise RuntimeError(f"src/ not found under PROJECT_DIR: {_SRC_DIR}")
if _PROJECT_DIR_ABS not in sys.path:
    sys.path.insert(0, _PROJECT_DIR_ABS)

from src.models.head_builder import build_head_layer, MultiLayerHeadExport  # noqa: E402
from src.models.peft_integration import _import_peft  # noqa: E402
from src.data.augmentations import build_eval_transform  # noqa: E402


# ===== Weights loader (TorchScript supported, state_dict also supported) =====
# Allowlist PEFT types for PyTorch 2.6+ safe deserialization (weights_only=True)
try:
    from torch.serialization import add_safe_globals  # type: ignore
except Exception:
    add_safe_globals = None  # type: ignore

try:
    from peft.utils.peft_types import PeftType  # type: ignore
except Exception:
    try:
        # Ensure third_party/peft is importable if bundled
        _import_peft()
        from peft.utils.peft_types import PeftType  # type: ignore
    except Exception:
        PeftType = None  # type: ignore

if add_safe_globals is not None and PeftType is not None:  # type: ignore
    try:
        add_safe_globals([PeftType])  # type: ignore
    except Exception:
        pass

def _torch_load_cpu(
    path: str,
    *,
    mmap: Optional[bool] = None,
    weights_only: Optional[bool] = True,
) -> object:
    """
    Kaggle-friendly torch.load wrapper:
    - Prefer weights_only=True to avoid pickle/object loading and silence warnings.
    - Prefer mmap=True (if supported) to avoid reading huge checkpoints fully into RAM.
    Falls back gracefully on older torch or if flags are unsupported.
    """
    # Always load onto CPU; we copy into GPU params via load_state_dict.
    kwargs = {"map_location": "cpu"}
    if weights_only is not None:
        kwargs["weights_only"] = weights_only
    if mmap is not None:
        kwargs["mmap"] = mmap
    try:
        return torch.load(path, **kwargs)
    except TypeError:
        # Older torch without weights_only/mmap
        return torch.load(path, map_location="cpu")
    except Exception:
        # If weights_only=True fails due to safe-unpickling constraints, fall back.
        if weights_only is True:
            try:
                kwargs2 = {"map_location": "cpu"}
                if mmap is not None:
                    kwargs2["mmap"] = mmap
                return torch.load(path, **kwargs2)  # weights_only omitted (defaults to legacy)
            except Exception:
                return torch.load(path, map_location="cpu")
        raise

def load_model_or_state(pt_path: str) -> Tuple[Optional[nn.Module], Optional[dict], dict]:
    if not os.path.isfile(pt_path):
        raise FileNotFoundError(f"Weights not found: {pt_path}")
    # 1) Try TorchScript (best for offline single-file inference)
    try:
        scripted = torch.jit.load(pt_path, map_location="cpu")
        return scripted, None, {"format": "torchscript"}
    except Exception:
        pass
    # 2) Fallback to torch.load objects (may be state_dict or checkpoint dict)
    obj = _torch_load_cpu(pt_path, mmap=None, weights_only=True)
    if isinstance(obj, dict) and "state_dict" in obj:
        return None, obj["state_dict"], obj.get("meta", {})
    if isinstance(obj, dict):
        # raw state_dict
        return None, obj, {}
    if isinstance(obj, nn.Module):
        # Pickled module (works only if class definitions are available)
        return obj, None, {"format": "pickled_module"}
    raise RuntimeError("Unsupported weights file format. Provide a TorchScript .pt for offline inference.")



def resolve_paths(input_path: str) -> Tuple[str, str]:
    if os.path.isdir(input_path):
        dataset_root = input_path
        test_csv = os.path.join(input_path, "test.csv")
    else:
        dataset_root = os.path.dirname(os.path.abspath(input_path))
        test_csv = input_path
    if not os.path.isfile(test_csv):
        raise FileNotFoundError(f"test.csv not found at: {test_csv}")
    return dataset_root, test_csv


class TestImageDataset(Dataset):
    def __init__(self, image_paths: List[str], root_dir: str, transform: T.Compose) -> None:
        self.image_paths = image_paths
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int):
        rel_path = self.image_paths[index]
        img_path = os.path.join(self.root_dir, rel_path)
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)
        return image, rel_path


def build_transforms(image_size: Tuple[int, int], mean: List[float], std: List[float]) -> T.Compose:
    return build_eval_transform(image_size=image_size, mean=mean, std=std)


def load_state_and_meta(pt_path: str):
    # Deprecated: kept for backward compatibility with older code paths
    model, state_dict, meta = load_model_or_state(pt_path)
    if model is not None:
        return model, meta
    return state_dict, meta


def load_head_state(pt_path: str) -> Tuple[dict, dict, Optional[dict]]:
    if not os.path.isfile(pt_path):
        raise FileNotFoundError(f"Head weights not found: {pt_path}")
    obj = _torch_load_cpu(pt_path, mmap=None, weights_only=True)
    if isinstance(obj, dict) and "state_dict" in obj:
        return obj["state_dict"], obj.get("meta", {}), obj.get("peft", None)
    if isinstance(obj, dict):
        return obj, {}, obj.get("peft", None)
    raise RuntimeError("Unsupported head weights file format. Expect a dict with 'state_dict'.")


# ==========================================================
# 2-GPU model-parallel helpers for dinov3_vit7b16 (Scheme B)
# ==========================================================
def _mp_get_devices() -> Optional[Tuple[torch.device, torch.device]]:
    try:
        if torch.cuda.is_available() and torch.cuda.device_count() >= 2:
            return torch.device("cuda:0"), torch.device("cuda:1")
    except Exception:
        pass
    return None


def _mp_resolve_dtype(dtype_str: str) -> torch.dtype:
    s = str(dtype_str or "").strip().lower()
    if s in ("fp16", "float16", "half"):
        return torch.float16
    if s in ("fp32", "float32"):
        return torch.float32
    # Default: fp16 for memory
    return torch.float16


def _mp_get_attr_chain(obj, names: List[str]):
    cur = obj
    for n in names:
        if cur is None:
            return None
        cur = getattr(cur, n, None)
    return cur


def _mp_get_backbone_for_attrs(backbone: nn.Module) -> nn.Module:
    """
    Best-effort resolve the underlying dinov3 backbone when wrapped (e.g., PEFT).
    For PEFT, common patterns:
      - peft_model.get_base_model()
      - peft_model.base_model.model
    We only need access to .blocks / .patch_embed / .norm / .rope_embed / tokens.
    """
    # 1) Direct attributes
    if hasattr(backbone, "blocks"):
        return backbone
    # 2) PEFT-style: .base_model.model
    base_model = getattr(backbone, "base_model", None)
    if base_model is not None:
        cand = getattr(base_model, "model", None)
        if isinstance(cand, nn.Module) and hasattr(cand, "blocks"):
            return cand
        if isinstance(base_model, nn.Module) and hasattr(base_model, "blocks"):
            return base_model
    # 3) Some wrappers use .model
    cand2 = getattr(backbone, "model", None)
    if isinstance(cand2, nn.Module) and hasattr(cand2, "blocks"):
        return cand2
    return backbone


def _mp_attach_flags(backbone: nn.Module, device0: torch.device, device1: torch.device, split_idx: int) -> None:
    try:
        backbone._mp_devices = (device0, device1)  # type: ignore[attr-defined]
        backbone._mp_split_idx = int(split_idx)  # type: ignore[attr-defined]
        backbone._mp_enabled = True  # type: ignore[attr-defined]
    except Exception:
        pass


def _mp_get_devices_from_backbone(backbone: nn.Module) -> Optional[Tuple[torch.device, torch.device]]:
    # Check common locations (feature_extractor wrapper, PEFT wrapper, underlying model).
    for cand in (
        backbone,
        getattr(backbone, "backbone", None),  # e.g., DinoV3FeatureExtractor
        getattr(backbone, "base_model", None),  # e.g., PEFT wrapper
        _mp_get_attr_chain(backbone, ["base_model", "model"]),
        getattr(backbone, "model", None),
    ):
        if cand is None:
            continue
        try:
            mp = getattr(cand, "_mp_devices", None)
            if isinstance(mp, (tuple, list)) and len(mp) == 2:
                d0 = torch.device(mp[0])
                d1 = torch.device(mp[1])
                return d0, d1
        except Exception:
            continue
    return None


def _mp_materialize_param(module: nn.Module, name: str, device: torch.device) -> None:
    """
    Materialize a top-level nn.Parameter living on 'meta' onto a real device.
    """
    p = getattr(module, name, None)
    if not isinstance(p, nn.Parameter):
        return
    if getattr(p, "device", None) is None:
        return
    if p.device.type != "meta":
        return
    new = torch.empty(p.shape, device=device, dtype=p.dtype)
    setattr(module, name, nn.Parameter(new, requires_grad=p.requires_grad))


def _mp_materialize_buffer(module: nn.Module, name: str, device: torch.device) -> None:
    """
    Materialize a registered buffer living on 'meta' onto a real device.
    """
    buf = module._buffers.get(name, None)  # type: ignore[attr-defined]
    if buf is None or not isinstance(buf, torch.Tensor):
        return
    if buf.device.type != "meta":
        return
    module._buffers[name] = torch.empty(buf.shape, device=device, dtype=buf.dtype)  # type: ignore[attr-defined]


def _mp_prepare_vit7b_backbone_two_gpu(
    backbone: nn.Module,
    *,
    split_idx: int,
    dtype: torch.dtype,
    device0: torch.device,
    device1: torch.device,
) -> nn.Module:
    """
    Take a DinoVisionTransformer instantiated on device='meta' and materialize its
    submodules across cuda:0 and cuda:1 (Scheme B).
    """
    # Underlying backbone (in case caller passes a wrapper, but here we usually pass the raw backbone)
    m = _mp_get_backbone_for_attrs(backbone)

    # 1) Set dtype on meta tensors before materialization (no real memory yet).
    try:
        m = m.to(dtype=dtype)
    except Exception:
        pass
    # Keep RoPE periods in fp32 when present (matches official dinov3 defaults; tiny memory).
    try:
        rope = getattr(m, "rope_embed", None)
        if rope is not None and hasattr(rope, "_buffers") and "periods" in rope._buffers:
            rope._buffers["periods"] = rope._buffers["periods"].to(dtype=torch.float32)  # type: ignore[attr-defined]
    except Exception:
        pass

    # 2) Materialize "token" parameters on device0
    for pname in ("cls_token", "mask_token", "storage_tokens"):
        try:
            _mp_materialize_param(m, pname, device0)
        except Exception:
            pass

    # 3) Materialize patch_embed + rope_embed on device0
    try:
        if hasattr(m, "patch_embed") and isinstance(m.patch_embed, nn.Module):
            m.patch_embed.to_empty(device=device0)
    except Exception:
        pass
    try:
        rope = getattr(m, "rope_embed", None)
        if isinstance(rope, nn.Module):
            # rope_embed has buffers, not params
            try:
                rope.to_empty(device=device0)  # type: ignore[attr-defined]
            except Exception:
                # fallback: materialize buffer(s) manually
                _mp_materialize_buffer(rope, "periods", device0)
    except Exception:
        pass

    # 4) Materialize transformer blocks split across devices
    if not hasattr(m, "blocks") or not isinstance(m.blocks, nn.ModuleList):
        raise RuntimeError("Backbone does not expose a ModuleList named 'blocks'; cannot model-parallelize.")
    n_blocks = len(m.blocks)
    split_idx = int(max(0, min(split_idx, n_blocks)))
    for i, blk in enumerate(m.blocks):
        target = device0 if i < split_idx else device1
        try:
            blk.to_empty(device=target)
        except Exception:
            # As a fallback, try moving (may allocate), but keep going
            blk.to(device=target)

    # 5) Materialize norms on device1 (final stage)
    for norm_name in ("norm", "cls_norm", "local_cls_norm"):
        try:
            sub = getattr(m, norm_name, None)
            if isinstance(sub, nn.Module):
                sub.to_empty(device=device1)
        except Exception:
            pass

    # 6) Attach flags for downstream code
    _mp_attach_flags(m, device0, device1, split_idx)
    return backbone


def _mp_patch_dinov3_methods(backbone: nn.Module, *, split_idx: int, device0: torch.device, device1: torch.device) -> None:
    """
    Monkeypatch forward_features + get_intermediate_layers on the underlying dinov3 backbone
    so it can run with blocks split across cuda:0/1.
    """
    m = _mp_get_backbone_for_attrs(backbone)

    def _forward_features_mp(self, x, masks: Optional[torch.Tensor] = None):  # noqa: ANN001
        # Only the Tensor path is needed for this repo's inference.
        if not isinstance(x, torch.Tensor):
            raise TypeError("Model-parallel forward_features only supports Tensor inputs")

        sp = int(max(0, min(int(split_idx), len(self.blocks))))
        # Ensure inputs land on stage0 device
        if x.device != device0:
            x = x.to(device0, non_blocking=True)

        # Prepare tokens on stage0
        tokens, (H, W) = self.prepare_tokens_with_masks(x, masks)

        # Pre-compute RoPE for both stages (same values; different device)
        rope0 = None
        rope1 = None
        if getattr(self, "rope_embed", None) is not None:
            rope0 = self.rope_embed(H=H, W=W)
            # Copy sin/cos to stage1 once (cheap)
            try:
                sin0, cos0 = rope0
                rope1 = (sin0.to(device1, non_blocking=True), cos0.to(device1, non_blocking=True))
            except Exception:
                rope1 = None

        out = tokens
        # Run blocks with a single activation transfer at the boundary
        for i, blk in enumerate(self.blocks):
            if i == sp:
                out = out.to(device1, non_blocking=True)
            rope = rope0 if i < sp else rope1
            out = blk(out, rope)

        # Ensure the post-block activations are on stage1 for final norm + downstream heads
        if out.device != device1:
            out = out.to(device1, non_blocking=True)

        # Final norm(s) (on stage1)
        if getattr(self, "untie_cls_and_patch_norms", False):
            x_norm_cls_reg = self.cls_norm(out[:, : self.n_storage_tokens + 1])
            x_norm_patch = self.norm(out[:, self.n_storage_tokens + 1 :])
        else:
            x_norm = self.norm(out)
            x_norm_cls_reg = x_norm[:, : self.n_storage_tokens + 1]
            x_norm_patch = x_norm[:, self.n_storage_tokens + 1 :]

        return {
            "x_norm_clstoken": x_norm_cls_reg[:, 0],
            "x_storage_tokens": x_norm_cls_reg[:, 1:],
            "x_norm_patchtokens": x_norm_patch,
            "x_prenorm": out,
            "masks": masks,
        }

    def _get_intermediate_layers_mp(
        self,
        x: torch.Tensor,
        *,
        n=1,
        reshape: bool = False,
        return_class_token: bool = False,
        return_extra_tokens: bool = False,
        norm: bool = True,
    ):
        if not isinstance(x, torch.Tensor):
            raise TypeError("Model-parallel get_intermediate_layers expects a Tensor input")

        sp = int(max(0, min(int(split_idx), len(self.blocks))))
        # Prepare tokens on stage0
        if x.device != device0:
            x = x.to(device0, non_blocking=True)
        tokens, (H, W) = self.prepare_tokens_with_masks(x)

        total_block_len = len(self.blocks)
        blocks_to_take = range(total_block_len - n, total_block_len) if isinstance(n, int) else n
        blocks_to_take = set(int(i) for i in blocks_to_take)

        # Pre-compute RoPE for both stages
        rope0 = None
        rope1 = None
        if getattr(self, "rope_embed", None) is not None:
            rope0 = self.rope_embed(H=H, W=W)
            try:
                sin0, cos0 = rope0
                rope1 = (sin0.to(device1, non_blocking=True), cos0.to(device1, non_blocking=True))
            except Exception:
                rope1 = None

        outs_full: List[torch.Tensor] = []
        out = tokens
        for i, blk in enumerate(self.blocks):
            if i == sp:
                out = out.to(device1, non_blocking=True)
            rope = rope0 if i < sp else rope1
            out = blk(out, rope)
            if i in blocks_to_take:
                # Ensure captured outputs live on stage1 for downstream heads
                if out.device != device1:
                    outs_full.append(out.to(device1, non_blocking=True))
                else:
                    outs_full.append(out)

        if len(outs_full) != len(blocks_to_take):
            raise RuntimeError(f"only {len(outs_full)} / {len(blocks_to_take)} blocks found")

        outputs = outs_full
        if norm:
            outputs_normed: List[torch.Tensor] = []
            for out_i in outputs:
                if getattr(self, "untie_cls_and_patch_norms", False):
                    x_norm_cls_reg = self.cls_norm(out_i[:, : self.n_storage_tokens + 1])
                    x_norm_patch = self.norm(out_i[:, self.n_storage_tokens + 1 :])
                    outputs_normed.append(torch.cat((x_norm_cls_reg, x_norm_patch), dim=1))
                else:
                    outputs_normed.append(self.norm(out_i))
            outputs = outputs_normed

        class_tokens = [out_i[:, 0] for out_i in outputs]
        extra_tokens = [out_i[:, 1 : self.n_storage_tokens + 1] for out_i in outputs]
        patch_tokens = [out_i[:, self.n_storage_tokens + 1 :] for out_i in outputs]

        if reshape:
            B, _, h, w = x.shape
            patch_tokens = [
                out.reshape(B, h // self.patch_size, w // self.patch_size, -1).permute(0, 3, 1, 2).contiguous()
                for out in patch_tokens
            ]

        if not return_class_token and not return_extra_tokens:
            return tuple(patch_tokens)
        if return_class_token and not return_extra_tokens:
            return tuple(zip(patch_tokens, class_tokens))
        if (not return_class_token) and return_extra_tokens:
            return tuple(zip(patch_tokens, extra_tokens))
        return tuple(zip(patch_tokens, class_tokens, extra_tokens))

    # Bind methods to the underlying module instance
    m.forward_features = types.MethodType(_forward_features_mp, m)  # type: ignore[method-assign]
    m.get_intermediate_layers = types.MethodType(_get_intermediate_layers_mp, m)  # type: ignore[method-assign]
    _mp_attach_flags(m, device0, device1, split_idx)


def _module_param_dtype(m: nn.Module, *, default: torch.dtype = torch.float32) -> torch.dtype:
    """
    Best-effort infer module parameter dtype (for casting inputs to match weights).
    """
    try:
        return next(m.parameters()).dtype
    except Exception:
        return default




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


class OfflineRegressor(nn.Module):
    def __init__(self, feature_extractor: nn.Module, head: nn.Module) -> None:
        super().__init__()
        self.feature_extractor = feature_extractor
        self.head = head

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.feature_extractor(images)
        return self.head(features)


def load_config(project_dir: str) -> Dict:
    config_path = os.path.join(project_dir, "configs", "train.yaml")
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Config YAML not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_config_file(config_path: str) -> Dict:
    """
    Load a YAML config from an explicit path (used for per-model ensemble configs).
    """
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Config YAML not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
def _parse_image_size(value) -> Tuple[int, int]:
    try:
        if isinstance(value, (list, tuple)) and len(value) == 2:
            w, h = int(value[0]), int(value[1])
            return (int(h), int(w))
        v = int(value)
        return (v, v)
    except Exception:
        v = int(value)
        return (v, v)



def discover_head_weight_paths(path: str) -> List[str]:
    # Accept single-file or directory containing a preferred single head or per-fold heads.
    if os.path.isfile(path):
        return [path]
    if os.path.isdir(path):
        # Preferred: explicit single-head file weights/head/infer_head.pt
        single = os.path.join(path, "infer_head.pt")
        if os.path.isfile(single):
            return [single]
        # Fallback: per-fold heads under weights/head/fold_*/infer_head.pt
        fold_paths: List[str] = []
        try:
            for name in sorted(os.listdir(path)):
                if name.startswith("fold_"):
                    cand = os.path.join(path, name, "infer_head.pt")
                    if os.path.isfile(cand):
                        fold_paths.append(cand)
        except Exception:
            pass
        if fold_paths:
            return fold_paths
        # Fallback: any .pt directly under directory
        pts = [os.path.join(path, n) for n in os.listdir(path) if n.endswith('.pt')]
        pts.sort()
        if pts:
            return pts
    raise FileNotFoundError(f"Cannot find head weights at: {path}")


def _read_ensemble_manifest_if_exists(
    project_dir: str,
    head_base: str,
) -> Optional[Tuple[List[Tuple[str, float]], str]]:
    """
    Read configs/ensemble.json if present and return:
      - list of (absolute_head_path, weight)
      - aggregation strategy: 'mean' or 'weighted_mean'
    Relative head paths are resolved against:
      1) HEAD_WEIGHTS_PT_PATH if it's a directory
      2) dirname(HEAD_WEIGHTS_PT_PATH) if it's a file
      3) project_dir as last resort
    """
    try:
        manifest_path = os.path.join(project_dir, "configs", "ensemble.json")
        if not os.path.isfile(manifest_path):
            return None
        with open(manifest_path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        if not isinstance(obj, dict):
            return None
        heads = obj.get("heads", [])
        if not isinstance(heads, list) or len(heads) == 0:
            # Empty heads list -> ignore manifest
            return None
        # Determine resolution base for relative paths
        if os.path.isdir(head_base):
            base_dir = head_base
        elif os.path.isfile(head_base):
            base_dir = os.path.dirname(head_base)
        else:
            base_dir = project_dir
        entries: List[Tuple[str, float]] = []
        for h in heads:
            if not isinstance(h, dict):
                continue
            p = h.get("path", None)
            if not isinstance(p, str) or len(p.strip()) == 0:
                continue
            w = h.get("weight", 1.0)
            try:
                w = float(w)
            except Exception:
                w = 1.0
            abs_path = p if os.path.isabs(p) else os.path.abspath(os.path.join(base_dir, p))
            if os.path.isfile(abs_path):
                entries.append((abs_path, w))
        if not entries:
            return None
        agg = str(obj.get("aggregation", "") or "").strip().lower()
        if agg not in ("mean", "weighted_mean"):
            # Default: weighted_mean when any weight != 1, else mean
            has_non_one = any(abs(w - 1.0) > 1e-8 for _, w in entries)
            agg = "weighted_mean" if has_non_one else "mean"
        return entries, agg
    except Exception:
        return None


def _read_ensemble_enabled_flag(project_dir: str) -> bool:
    """
    Read configs/ensemble.json and return 'enabled' flag (default: False).
    """
    try:
        manifest_path = os.path.join(project_dir, "configs", "ensemble.json")
        if not os.path.isfile(manifest_path):
            return False
        with open(manifest_path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        if not isinstance(obj, dict):
            return False
        return bool(obj.get("enabled", False))
    except Exception:
        return False


def _read_ensemble_cfg_obj(project_dir: str) -> Optional[dict]:
    """
    Read configs/ensemble.json and return the parsed dict (or None).

    Note: This repo historically used configs/ensemble.json for packaging-time "versions".
    We keep that schema and additionally support a newer "models" list schema.
    """
    try:
        manifest_path = os.path.join(project_dir, "configs", "ensemble.json")
        if not os.path.isfile(manifest_path):
            return None
        with open(manifest_path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def _safe_slug(value: str) -> str:
    s = str(value or "").strip()
    if not s:
        return "model"
    # Replace path separators and other unsafe characters.
    s = s.replace(os.sep, "_").replace("/", "_")
    s = re.sub(r"[^A-Za-z0-9_.-]+", "_", s)
    s = s.strip("._-")
    return s or "model"


def _resolve_path_best_effort(project_dir: str, p: str) -> str:
    """
    Resolve a user-provided path against PROJECT_DIR, with a small compatibility
    shim for packaged 'weights/' directories.
    """
    if not isinstance(p, str) or len(p.strip()) == 0:
        return ""
    p = p.strip()
    if os.path.isabs(p):
        return p
    # First: relative to project_dir
    cand = os.path.abspath(os.path.join(project_dir, p))
    if os.path.exists(cand):
        return cand
    # Compatibility: if a path starts with "weights/" but PROJECT_DIR already *is* the weights dir,
    # try stripping the prefix.
    if p.startswith("weights/") or p.startswith("weights" + os.sep):
        p2 = p.split("/", 1)[1] if "/" in p else p.split(os.sep, 1)[1]
        cand2 = os.path.abspath(os.path.join(project_dir, p2))
        if os.path.exists(cand2):
            return cand2
    return cand


def _resolve_version_train_yaml(project_dir: str, version: str) -> str:
    """
    Resolve a per-version train.yaml snapshot (best effort).

    Search order:
      1) <PROJECT_DIR>/weights/configs/versions/<ver>/train.yaml   (repo-root running)
      2) <PROJECT_DIR>/configs/versions/<ver>/train.yaml           (running inside packaged weights dir)
      3) <PROJECT_DIR>/outputs/<ver>/train.yaml
      4) <PROJECT_DIR>/outputs/<ver>/fold_0/train.yaml
      5) <PROJECT_DIR>/configs/train.yaml                          (fallback)
    """
    ver = str(version or "").strip()
    candidates: List[str] = []
    if ver:
        candidates.extend(
            [
                os.path.join(project_dir, "weights", "configs", "versions", ver, "train.yaml"),
                os.path.join(project_dir, "configs", "versions", ver, "train.yaml"),
                os.path.join(project_dir, "outputs", ver, "train.yaml"),
                os.path.join(project_dir, "outputs", ver, "fold_0", "train.yaml"),
            ]
        )
    candidates.append(os.path.join(project_dir, "configs", "train.yaml"))
    for c in candidates:
        if os.path.isfile(c):
            return c
    # Return the last fallback even if missing to preserve error context.
    return candidates[-1]


def _resolve_version_head_base(project_dir: str, version: str) -> str:
    """
    Resolve the packaged head directory for a given version (best effort).

    Search order:
      1) <PROJECT_DIR>/weights/head/<ver>   (repo-root running)
      2) <PROJECT_DIR>/head/<ver>           (running inside packaged weights dir)
    """
    ver = str(version or "").strip()
    if not ver:
        return ""
    candidates = [
        os.path.join(project_dir, "weights", "head", ver),
        os.path.join(project_dir, "head", ver),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return candidates[0]


def _normalize_ensemble_models(project_dir: str) -> List[dict]:
    """
    Normalize configs/ensemble.json into a list of model dicts.

    Supported schemas:
      - New: { enabled: true, models: [ {id?, version?, weight?, head_weights?, config? ...}, ... ] }
      - Legacy: { enabled: true, versions: [ "ver1", "ver2", ... ] }  (or single "version")
    """
    obj = _read_ensemble_cfg_obj(project_dir)
    if not isinstance(obj, dict) or not bool(obj.get("enabled", False)):
        return []

    models_raw = obj.get("models", None)
    models: List[dict] = []
    if isinstance(models_raw, list) and len(models_raw) > 0:
        for idx, m in enumerate(models_raw):
            if isinstance(m, str):
                ver = m.strip()
                if not ver:
                    continue
                models.append({"id": ver, "version": ver, "weight": 1.0})
                continue
            if not isinstance(m, dict):
                continue
            mm = dict(m)
            # Backward-compat for key names
            if "id" not in mm and isinstance(mm.get("name", None), str):
                mm["id"] = mm.get("name")
            if "version" not in mm and isinstance(mm.get("ver", None), str):
                mm["version"] = mm.get("ver")
            # Default id
            if not isinstance(mm.get("id", None), str) or not str(mm.get("id")).strip():
                v = mm.get("version", None)
                mm["id"] = str(v).strip() if isinstance(v, (str, int, float)) and str(v).strip() else f"model_{idx}"
            # Default weight
            try:
                mm["weight"] = float(mm.get("weight", 1.0))
            except Exception:
                mm["weight"] = 1.0
            models.append(mm)
        return models

    # Legacy: versions list
    versions = obj.get("versions", None)
    if versions is None:
        v = obj.get("version", None)
        versions = [v] if v is not None else []
    if not isinstance(versions, list):
        versions = []
    for v in versions:
        if isinstance(v, str) and v.strip():
            ver = v.strip()
            models.append({"id": ver, "version": ver, "weight": 1.0})
        elif isinstance(v, dict):
            # Allow list of dicts in versions for extra flexibility
            mm = dict(v)
            ver = str(mm.get("version", mm.get("id", "")) or "").strip()
            if not ver:
                continue
            mm.setdefault("id", ver)
            mm.setdefault("version", ver)
            try:
                mm["weight"] = float(mm.get("weight", 1.0))
            except Exception:
                mm["weight"] = 1.0
            models.append(mm)
    return models


def _resolve_dino_weights_path_for_model(
    project_dir: str,
    *,
    backbone_name: str,
    cfg: Optional[Dict] = None,
    model_cfg: Optional[dict] = None,
) -> str:
    """
    Resolve backbone weights path for a model.

    Priority:
      1) model_cfg['dino_weights_pt' | 'dino_weights' | 'backbone_weights']
      2) cfg['model']['weights_path'] (if it exists on disk)
      3) best-effort glob search under <PROJECT_DIR>/dinov3_weights/ (by backbone_name)
      4) global DINO_WEIGHTS_PT_PATH (if it exists on disk)
    """
    model_cfg = model_cfg or {}
    cfg = cfg or {}

    def _find_in_dir(dir_path: str, backbone: str) -> str:
        """
        Find backbone weights file in a directory by known filename/patterns.
        Assumes the base filenames are stable; extension may be .pt or .pth.
        """
        if not (isinstance(dir_path, str) and dir_path and os.path.isdir(dir_path)):
            return ""
        bn2 = str(backbone or "").strip().lower()
        # Prefer exact known filenames first (fast + unambiguous).
        exact_bases: List[str] = []
        if bn2 in ("dinov3_vit7b16", "dinov3_vit7b", "vit7b16", "vit7b"):
            exact_bases.append("dinov3_vit7b16_pretrain_lvd1689m-a955f4ea")
        elif bn2 == "dinov3_vith16plus":
            exact_bases.append("dinov3_vith16plus_pretrain_lvd1689m-7c1da9a5")
        elif bn2 == "dinov3_vitl16":
            exact_bases.append("dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd")
        for base in exact_bases:
            for ext in (".pt", ".pth"):
                cand = os.path.join(dir_path, base + ext)
                if os.path.isfile(cand):
                    return os.path.abspath(cand)

        # Fallback: glob patterns (still scoped to the directory).
        patterns2: List[str] = []
        if bn2 in ("dinov3_vit7b16", "dinov3_vit7b", "vit7b16", "vit7b"):
            patterns2.extend(["dinov3_vit7b16_pretrain_*.pt", "dinov3_vit7b16_pretrain_*.pth"])
        elif bn2 == "dinov3_vith16plus":
            patterns2.extend(["dinov3_vith16plus_pretrain_*.pt", "dinov3_vith16plus_pretrain_*.pth"])
        elif bn2 == "dinov3_vitl16":
            patterns2.extend(["dinov3_vitl16_pretrain_*.pt", "dinov3_vitl16_pretrain_*.pth"])
        if bn2:
            patterns2.extend([f"{bn2}*.pt", f"{bn2}*.pth"])
        for pat in patterns2:
            try:
                for cand in sorted(glob.glob(os.path.join(dir_path, pat))):
                    if os.path.isfile(cand):
                        return os.path.abspath(cand)
            except Exception:
                continue
        return ""

    # 1) Explicit override in ensemble config
    for k in ("dino_weights_pt", "dino_weights", "backbone_weights", "backbone_weights_pt"):
        v = model_cfg.get(k, None)
        if isinstance(v, str) and v.strip():
            p = _resolve_path_best_effort(project_dir, v)
            if os.path.isfile(p):
                return p
            if os.path.isdir(p):
                found = _find_in_dir(p, backbone_name)
                if found:
                    return found

    # 2) weights_path from the model's YAML config (only if it exists)
    try:
        v = cfg.get("model", {}).get("weights_path", None)
        if isinstance(v, str) and v.strip():
            p = _resolve_path_best_effort(project_dir, v)
            if os.path.isfile(p):
                return p
            if os.path.isdir(p):
                found = _find_in_dir(p, backbone_name)
                if found:
                    return found
    except Exception:
        pass

    # 3) Try to locate weights by backbone name under dinov3_weights/
    dinodir = os.path.join(project_dir, "dinov3_weights")
    found = _find_in_dir(dinodir, backbone_name)
    if found:
        return found

    # 4) Global script-level fallback (user-editable at top of file).
    # IMPORTANT: this is a last resort; for multi-backbone ensembles you should either:
    #   - set per-model dino_weights_pt in configs/ensemble.json, or
    #   - place the correct backbone weights under dinov3_weights/ with standard names.
    try:
        if isinstance(DINO_WEIGHTS_PT_PATH, str) and DINO_WEIGHTS_PT_PATH.strip():
            p = _resolve_path_best_effort(project_dir, DINO_WEIGHTS_PT_PATH)
            if os.path.isdir(p):
                found = _find_in_dir(p, backbone_name)
                if found:
                    return found
            if os.path.isfile(p):
                # Safety: avoid silently using a mismatched backbone weights file for multi-backbone ensembles.
                # If the filename doesn't look like the requested backbone, force the user to provide
                # the correct weights via dinov3_weights/ or per-model dino_weights_pt.
                token = ""
                bn_l = str(backbone_name or "").strip().lower()
                if bn_l in ("dinov3_vit7b16", "dinov3_vit7b", "vit7b16", "vit7b"):
                    token = "vit7b"
                elif bn_l == "dinov3_vith16plus":
                    token = "vith16plus"
                elif bn_l == "dinov3_vitl16":
                    token = "vitl16"
                if token:
                    base = os.path.basename(p).lower()
                    if token not in base:
                        raise FileNotFoundError(
                            f"Global DINO_WEIGHTS_PT_PATH appears to mismatch backbone '{backbone_name}': {p}. "
                            f"Expected filename to contain '{token}'. "
                            "Provide the correct weights under dinov3_weights/ or set per-model dino_weights_pt in configs/ensemble.json."
                        )
                return p
    except FileNotFoundError:
        raise
    except Exception:
        pass

    raise FileNotFoundError(
        "Backbone weights not found. Provide an explicit path via ensemble.json "
        "('dino_weights_pt') or set DINO_WEIGHTS_PT_PATH at the top of infer_and_submit_pt.py."
    )


def _save_ensemble_cache(
    cache_path: str,
    *,
    model_id: str,
    model_weight: float,
    rels_in_order: List[str],
    comps_5d_g: torch.Tensor,
    meta: Optional[dict] = None,
) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(cache_path)), exist_ok=True)
    payload = {
        "schema_version": 1,
        "model_id": str(model_id),
        "model_weight": float(model_weight),
        "components_order": ["Dry_Clover_g", "Dry_Dead_g", "Dry_Green_g", "GDM_g", "Dry_Total_g"],
        "rels_in_order": list(rels_in_order),
        "comps_5d_g": comps_5d_g.detach().cpu().float(),
        "meta": meta or {},
    }
    torch.save(payload, cache_path)


def _load_ensemble_cache(cache_path: str) -> Tuple[List[str], torch.Tensor, dict]:
    obj = _torch_load_cpu(cache_path, mmap=None, weights_only=True)
    if not isinstance(obj, dict):
        raise RuntimeError(f"Invalid cache format (expected dict): {cache_path}")
    rels = obj.get("rels_in_order", None)
    comps = obj.get("comps_5d_g", None)
    meta = obj.get("meta", {}) if isinstance(obj.get("meta", {}), dict) else {}
    if not isinstance(rels, list) or comps is None:
        raise RuntimeError(f"Invalid cache payload (missing rels/comps): {cache_path}")
    if not isinstance(comps, torch.Tensor) or comps.dim() != 2 or comps.size(-1) != 5:
        raise RuntimeError(f"Invalid comps_5d_g shape in cache: {cache_path}")
    return [str(r) for r in rels], comps.detach().cpu().float(), meta

def _read_packaged_ensemble_manifest_if_exists(
    head_base: str,
) -> Optional[Tuple[List[Tuple[str, float]], str]]:
    """
    Read weights/head/ensemble.json if present and return:
      - list of (absolute_head_path, weight=1.0)
      - aggregation strategy ('mean', default)
    Paths inside packaged manifest are expected to be relative to head_base.
    """
    try:
        if not isinstance(head_base, str) or len(head_base) == 0:
            return None
        base_dir = head_base if os.path.isdir(head_base) else os.path.dirname(head_base)
        manifest_path = os.path.join(base_dir, "ensemble.json")
        if not os.path.isfile(manifest_path):
            return None
        with open(manifest_path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        if not isinstance(obj, dict):
            return None
        heads = obj.get("heads", [])
        if not isinstance(heads, list) or len(heads) == 0:
            return None
        entries: List[Tuple[str, float]] = []
        for h in heads:
            if not isinstance(h, dict):
                continue
            p = h.get("path", None)
            if not isinstance(p, str) or len(p.strip()) == 0:
                continue
            abs_path = p if os.path.isabs(p) else os.path.abspath(os.path.join(base_dir, p))
            if os.path.isfile(abs_path):
                entries.append((abs_path, 1.0))
        if not entries:
            return None
        agg = str(obj.get("aggregation", "") or "").strip().lower()
        if agg not in ("mean", "weighted_mean"):
            agg = "mean"
        return entries, agg
    except Exception:
        return None

def _load_zscore_json_for_head(head_pt_path: str) -> Optional[dict]:
    """
    Try to locate z_score.json for a given head.
    Priority:
      1) Same directory as head .pt (e.g., weights/head/fold_i/z_score.json)
      2) Parent of head directory (e.g., weights/head/z_score.json)
      3) Parent of head parent (e.g., weights/z_score.json)
    """
    import json
    candidates: List[str] = []
    d = os.path.dirname(head_pt_path)
    candidates.append(os.path.join(d, "z_score.json"))
    parent = os.path.dirname(d)
    if parent and parent != d:
        candidates.append(os.path.join(parent, "z_score.json"))
        gp = os.path.dirname(parent)
        if gp and gp not in (parent, d):
            candidates.append(os.path.join(gp, "z_score.json"))
    for c in candidates:
        if os.path.isfile(c):
            try:
                with open(c, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                continue
    return None

def extract_features_for_images(
    feature_extractor: nn.Module,
    dataset_root: str,
    image_paths: List[str],
    image_size: int,
    mean: List[float],
    std: List[float],
    batch_size: int,
    num_workers: int,
    *,
    use_cls_token: bool = True,
) -> Tuple[List[str], torch.Tensor]:
    mp_devs = _mp_get_devices_from_backbone(feature_extractor) if isinstance(feature_extractor, nn.Module) else None
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
    mp_devs = _mp_get_devices_from_backbone(head) if isinstance(head, nn.Module) else None
    device = mp_devs[1] if mp_devs is not None else ("cuda" if torch.cuda.is_available() else "cpu")
    head = head.eval().to(device)
    N = features_cpu.shape[0]
    preds_list: List[torch.Tensor] = []
    head_dtype = _module_param_dtype(head, default=torch.float32)
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
    image_size: int,
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
    mp_devs = _mp_get_devices_from_backbone(backbone) if isinstance(backbone, nn.Module) else None
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
    head_dtype = _module_param_dtype(head, default=torch.float32)

    rels: List[str] = []
    preds_main_list: List[torch.Tensor] = []
    preds_ratio_list: List[torch.Tensor] = []

    with torch.inference_mode():
        for images, rel_paths in dl:
            images = images.to(device0, non_blocking=True)

            # Multi-layer path: use DINO get_intermediate_layers to obtain per-layer
            # CLS and patch tokens. For legacy packed heads, we apply the shared head
            # on each layer-specific patch feature and slice that layer's segment from
            # the final linear. For MultiLayerHeadExport (separate bottlenecks), we call
            # its explicit per-layer forward.
            if use_layerwise_heads and num_layers > 1 and layer_indices is not None and len(layer_indices) > 0:
                cls_list, pt_list = feature_extractor.forward_layers_cls_and_tokens(images, layer_indices)
                if len(cls_list) != len(pt_list) or len(cls_list) != num_layers:
                    raise RuntimeError("Mismatch between CLS/patch token lists and num_layers in patch-mode multi-layer inference")

                main_layers_batch: List[torch.Tensor] = []
                ratio_layers_batch: List[torch.Tensor] = []

                for l_idx, (cls_l, pt_l) in enumerate(zip(cls_list, pt_list)):
                    if pt_l.dim() != 3:
                        raise RuntimeError(f"Unexpected patch tokens shape in patch-mode inference: {tuple(pt_l.shape)}")
                    B, N, C = pt_l.shape
                    if use_separate_bottlenecks and isinstance(head, MultiLayerHeadExport):
                        # New path: explicit per-layer bottlenecks encoded in head.
                        pt_l = pt_l.to(device1, non_blocking=True, dtype=head_dtype)
                        layer_main, layer_ratio = head.forward_patch_layer(pt_l, l_idx)
                    else:
                        if head_num_main > 0:
                            # Ensure features live on head device
                            patch_features_flat = pt_l.reshape(B * N, C).to(device1, non_blocking=True, dtype=head_dtype)  # (B*N, C)
                            out_all_patch = head(patch_features_flat)  # (B*N, head_total * L)
                            expected_dim = head_total * num_layers
                            if out_all_patch.shape[1] != expected_dim:
                                raise RuntimeError(
                                    f"Unexpected packed head dimension in patch-mode multi-layer: got {out_all_patch.shape[1]}, "
                                    f"expected {expected_dim}"
                                )
                            offset = l_idx * head_total
                            layer_slice = out_all_patch[:, offset : offset + head_total]  # (B*N, head_total)
                            layer_main = layer_slice[:, :head_num_main]  # (B*N, head_num_main)
                            layer_main = layer_main.view(B, N, head_num_main).mean(dim=1)  # (B, head_num_main)
                        else:
                            # No main outputs; keep empty tensor for consistency
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
                            layer_ratio = layer_slice_g[
                                :, head_num_main : head_num_main + head_num_ratio
                            ]  # (B, head_num_ratio)
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
                # Legacy single-layer path: use last-layer CLS and patch tokens only.
                cls, pt = feature_extractor.forward_cls_and_tokens(images)
                if pt.dim() != 3:
                    raise RuntimeError(f"Unexpected patch tokens shape in patch-mode inference: {tuple(pt.shape)}")
                B, N, C = pt.shape

                if head_num_main > 0:
                    patch_features_flat = pt.reshape(B * N, C).to(device1, non_blocking=True, dtype=head_dtype)  # (B*N, C)
                    out_all_patch = head(patch_features_flat)  # (B*N, head_total)
                    out_main_patch = out_all_patch[:, :head_num_main]
                    out_main_patch = out_main_patch.view(B, N, head_num_main).mean(dim=1)  # (B, head_num_main)
                else:
                    out_main_patch = torch.empty((B, 0), dtype=torch.float32, device=device)

                # --- Ratio logits (if any) from global patch-mean features ---
                if head_num_ratio > 0:
                    patch_mean = pt.mean(dim=1).to(device1, non_blocking=True, dtype=head_dtype)  # (B, C)
                    out_all_global = head(patch_mean)  # (B, head_total)
                    out_ratio = out_all_global[:, head_num_main : head_num_main + head_num_ratio]  # (B, head_num_ratio)
                else:
                    out_ratio = None

            preds_main_list.append(out_main_patch.detach().cpu().float())
            if out_ratio is not None:
                preds_ratio_list.append(out_ratio.detach().cpu().float())
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
    return rels, preds_main, preds_ratio


def predict_main_and_ratio_global_multilayer(
    backbone: nn.Module,
    head: nn.Module,
    dataset_root: str,
    image_paths: List[str],
    image_size: int,
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
    Global multi-layer inference for a single head:
      - For each image, obtain per-layer CLS and patch tokens from the DINO backbone
        using get_intermediate_layers.
      - For each layer, build the global feature vector:
          - use_cls_token=True:  [CLS ; mean(patch)]
          - use_cls_token=False: mean(patch)
        apply the packed head,
        slice that layer's segment, and then average predictions over layers.
    """
    mp_devs = _mp_get_devices_from_backbone(backbone) if isinstance(backbone, nn.Module) else None
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
    head_dtype = _module_param_dtype(head, default=torch.float32)

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
                    # Explicit per-layer bottlenecks: call dedicated global-layer path.
                    layer_main, layer_ratio = head.forward_global_layer(cls_l, patch_mean_l, l_idx)
                else:
                    feats_l = (
                        torch.cat([cls_l, patch_mean_l], dim=-1)
                        if use_cls_token
                        else patch_mean_l
                    )
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
                        layer_ratio = layer_slice[
                            :, head_num_main : head_num_main + head_num_ratio
                        ]  # (B, head_num_ratio)
                    else:
                        layer_ratio = None

                main_layers_batch.append(layer_main)
                if layer_ratio is not None:
                    ratio_layers_batch.append(layer_ratio)

            # Average over layers
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

    return rels, preds_main, preds_ratio


def predict_for_images(
    model: nn.Module,
    dataset_root: str,
    image_paths: List[str],
    image_size: int,
    mean: List[float],
    std: List[float],
    batch_size: int,
    num_workers: int,
) -> Dict[str, Tuple[float, float, float]]:
    mp_devs = _mp_get_devices_from_backbone(model) if isinstance(model, nn.Module) else None
    device = mp_devs[0] if mp_devs is not None else ("cuda" if torch.cuda.is_available() else "cpu")

    tf = build_transforms(image_size=image_size, mean=mean, std=std)
    ds = TestImageDataset(image_paths, root_dir=dataset_root, transform=tf)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=torch.cuda.is_available())

    model.eval() if mp_devs is not None else model.eval().to(device)

    preds: Dict[str, Tuple[float, float, float]] = {}
    with torch.inference_mode():
        for images, rel_paths in dl:
            images = images.to(device, non_blocking=True)
            outputs = model(images)
            outputs = outputs.detach().cpu().float().tolist()
            for rel_path, vec in zip(rel_paths, outputs):
                v0, v1, v2 = float(vec[0]), float(vec[1]), float(vec[2])
                preds[rel_path] = (v0, v1, v2)
    return preds


def infer_components_5d_for_model(
    *,
    project_dir: str,
    cfg: Dict,
    head_base: str,
    dino_weights_pt_path: str,
    dataset_root: str,
    image_paths: List[str],
) -> Tuple[List[str], torch.Tensor, dict]:
    """
    Run inference for a *single model* (one backbone + one or more head weights) and return:
      - rels_in_order: list of image paths (same order as image_paths argument)
      - comps_5d_g:    Tensor (N, 5) in grams, order:
                       [Dry_Clover_g, Dry_Dead_g, Dry_Green_g, GDM_g, Dry_Total_g]
      - meta:          small metadata dict (for debugging / cache introspection)

    This function is the core building block for flexible ensembles: callers can invoke it
    multiple times (with different backbones) and ensemble the returned 5D components.
    """
    # --- Data settings (from this model's config) ---
    image_size = _parse_image_size(cfg["data"]["image_size"])
    mean = list(cfg["data"]["normalization"]["mean"])
    std = list(cfg["data"]["normalization"]["std"])
    target_bases = list(cfg["data"]["target_order"])
    batch_size = int(cfg["data"].get("batch_size", 32))
    num_workers = int(cfg["data"].get("num_workers", 4))

    # Dataset area (m^2) to convert g/m^2 to grams
    ds_name = str(cfg["data"].get("dataset", "csiro"))
    ds_map = dict(cfg["data"].get("datasets", {}))
    ds_info = dict(ds_map.get(ds_name, {}))
    try:
        width_m = float(ds_info.get("width_m", ds_info.get("width", 1.0)))
    except Exception:
        width_m = 1.0
    try:
        length_m = float(ds_info.get("length_m", ds_info.get("length", 1.0)))
    except Exception:
        length_m = 1.0
    try:
        area_m2 = float(ds_info.get("area_m2", width_m * length_m))
    except Exception:
        area_m2 = max(1.0, width_m * length_m if (width_m > 0.0 and length_m > 0.0) else 1.0)
    if not (area_m2 > 0.0):
        area_m2 = 1.0

    # --- Backbone selector ---
    backbone_name = str(cfg["model"]["backbone"]).strip()
    try:
        if backbone_name == "dinov3_vith16plus":
            from dinov3.hub.backbones import dinov3_vith16plus as _make_backbone  # type: ignore
        elif backbone_name == "dinov3_vitl16":
            from dinov3.hub.backbones import dinov3_vitl16 as _make_backbone  # type: ignore
        elif backbone_name in ("dinov3_vit7b16", "dinov3_vit7b", "vit7b16", "vit7b"):
            from dinov3.hub.backbones import dinov3_vit7b16 as _make_backbone  # type: ignore
        else:
            raise ImportError(f"Unsupported backbone in config: {backbone_name}")
    except Exception:
        raise ImportError(
            "dinov3 is not available locally. Ensure DINOV3_PATH points to third_party/dinov3/dinov3."
        )

    # --- Head weights discovery ---
    head_base_abs = _resolve_path_best_effort(project_dir, head_base)
    if not head_base_abs:
        raise FileNotFoundError("head_base is empty for model inference.")
    head_entries: Optional[Tuple[List[Tuple[str, float]], str]] = None
    try:
        head_entries = _read_packaged_ensemble_manifest_if_exists(head_base_abs)
    except Exception:
        head_entries = None
    if head_entries is not None:
        entries, aggregation = head_entries
        head_weight_paths = [p for (p, _w) in entries]
        weight_map = {p: float(w) for (p, w) in entries}
        _ = aggregation
    else:
        head_weight_paths = discover_head_weight_paths(head_base_abs)
        weight_map = {p: 1.0 for p in head_weight_paths}

    if not head_weight_paths:
        raise RuntimeError(f"No head weights found under: {head_base_abs}")

    # Inspect first head file to infer defaults
    _first_state, first_meta, _ = load_head_state(head_weight_paths[0])
    if not isinstance(first_meta, dict):
        first_meta = {}
    num_outputs_main_default = int(first_meta.get("num_outputs_main", first_meta.get("num_outputs", 3)))
    num_outputs_ratio_default = int(first_meta.get("num_outputs_ratio", 0))
    head_total_outputs_default = int(first_meta.get("head_total_outputs", num_outputs_main_default + num_outputs_ratio_default))
    use_cls_token_default = bool(first_meta.get("use_cls_token", True))
    use_layerwise_heads_default = bool(first_meta.get("use_layerwise_heads", False))
    backbone_layer_indices_default = list(first_meta.get("backbone_layer_indices", []))
    use_separate_bottlenecks_default = bool(first_meta.get("use_separate_bottlenecks", False))

    # Embedding dim: prefer head meta (authoritative), fall back to config.
    cfg_embedding_dim = int(cfg.get("model", {}).get("embedding_dim", int(first_meta.get("embedding_dim", 0) or 0) or 0))
    expected_embedding_dim = int(first_meta.get("embedding_dim", cfg_embedding_dim))
    if expected_embedding_dim <= 0:
        expected_embedding_dim = int(cfg_embedding_dim) if cfg_embedding_dim > 0 else 0

    # Preload backbone state (mmap for RAM friendliness).
    dino_weights_abs = _resolve_path_best_effort(project_dir, dino_weights_pt_path)
    if not (dino_weights_abs and os.path.isfile(dino_weights_abs)):
        raise FileNotFoundError(f"DINO weights not found: {dino_weights_pt_path}")
    dino_state = _torch_load_cpu(dino_weights_abs, mmap=True, weights_only=True)
    if isinstance(dino_state, dict) and "state_dict" in dino_state:
        dino_state = dino_state["state_dict"]

    # Accumulate per-head 5D components in the common rel order.
    rels_in_order_ref: Optional[List[str]] = None
    comps_sum: Optional[torch.Tensor] = None
    weight_sum: float = 0.0

    for head_pt in head_weight_paths:
        w = float(weight_map.get(head_pt, 1.0))
        if not (w > 0.0):
            # Skip zero/negative weights
            continue

        # Load head state/meta (and optional PEFT payload)
        state, meta, peft_payload = load_head_state(head_pt)
        if not isinstance(meta, dict):
            meta = {}

        head_num_main = int(meta.get("num_outputs_main", meta.get("num_outputs", num_outputs_main_default)))
        head_num_ratio = int(meta.get("num_outputs_ratio", num_outputs_ratio_default))
        head_total = int(meta.get("head_total_outputs", head_num_main + head_num_ratio))
        head_hidden_dims = list(
            meta.get("head_hidden_dims", first_meta.get("head_hidden_dims", list(cfg["model"]["head"].get("hidden_dims", [512, 256]))))
        )
        head_activation = str(meta.get("head_activation", first_meta.get("head_activation", str(cfg["model"]["head"].get("activation", "relu")))))
        head_dropout = float(meta.get("head_dropout", first_meta.get("head_dropout", float(cfg["model"]["head"].get("dropout", 0.0)))))
        head_embedding_dim = int(meta.get("embedding_dim", expected_embedding_dim))

        if expected_embedding_dim > 0 and head_embedding_dim != expected_embedding_dim:
            raise RuntimeError(
                f"Head {head_pt} embedding_dim({head_embedding_dim}) != expected_embedding_dim({expected_embedding_dim}); "
                "mixing heads trained on different backbones inside one model entry is unsupported."
            )

        # Build a fresh backbone per head (needed when per-head LoRA payload differs)
        use_mp = (
            bool(USE_2GPU_MODEL_PARALLEL_FOR_VIT7B)
            and (backbone_name in ("dinov3_vit7b16", "dinov3_vit7b", "vit7b16", "vit7b"))
            and (_mp_get_devices() is not None)
        )
        if use_mp:
            dev0, dev1 = _mp_get_devices()  # type: ignore[assignment]
            split_idx = int(VIT7B_MP_SPLIT_IDX)
            mp_dtype = _mp_resolve_dtype(VIT7B_MP_DTYPE)
            backbone = _make_backbone(pretrained=False, device="meta")
            _mp_prepare_vit7b_backbone_two_gpu(
                backbone,
                split_idx=split_idx,
                dtype=mp_dtype,
                device0=dev0,
                device1=dev1,
            )
            try:
                backbone.load_state_dict(dino_state, strict=True)
            except Exception:
                backbone.load_state_dict(dino_state, strict=False)
            _mp_patch_dinov3_methods(backbone, split_idx=split_idx, device0=dev0, device1=dev1)
        else:
            backbone = _make_backbone(pretrained=False)
            try:
                backbone.load_state_dict(dino_state, strict=True)
            except Exception:
                backbone.load_state_dict(dino_state, strict=False)

        # Inject per-head LoRA adapters (optional)
        try:
            if peft_payload is not None and isinstance(peft_payload, dict):
                peft_cfg_dict = peft_payload.get("config", None)
                peft_state = peft_payload.get("state_dict", None)
                if peft_cfg_dict and peft_state:
                    try:
                        from peft.config import PeftConfig  # type: ignore
                        from peft.mapping_func import get_peft_model  # type: ignore
                        from peft.utils.save_and_load import set_peft_model_state_dict  # type: ignore
                    except Exception:
                        LoraConfig, get_peft_model_alt, _, _ = _import_peft()  # noqa: F841
                        from peft.config import PeftConfig  # type: ignore
                        from peft.utils.save_and_load import set_peft_model_state_dict  # type: ignore
                        get_peft_model = get_peft_model_alt  # type: ignore
                    peft_config = PeftConfig.from_peft_type(**peft_cfg_dict)
                    backbone = get_peft_model(backbone, peft_config)
                    set_peft_model_state_dict(backbone, peft_state, adapter_name="default")
                    backbone.eval()
                    if use_mp:
                        try:
                            _mp_attach_flags(backbone, dev0, dev1, split_idx)  # type: ignore[arg-type]
                        except Exception:
                            pass
        except Exception as _e:
            print(f"[WARN] PEFT injection skipped for {head_pt}: {_e}")

        # Build head module according to head meta
        use_patch_reg3_head = bool(meta.get("use_patch_reg3", False))
        use_cls_token_head = bool(meta.get("use_cls_token", use_cls_token_default))
        use_layerwise_heads_head = bool(meta.get("use_layerwise_heads", use_layerwise_heads_default))
        backbone_layer_indices_head = list(meta.get("backbone_layer_indices", backbone_layer_indices_default))
        use_separate_bottlenecks_head = bool(meta.get("use_separate_bottlenecks", use_separate_bottlenecks_default))
        head_is_ratio = bool(head_num_ratio > 0 and head_total == (head_num_main + head_num_ratio))

        if use_layerwise_heads_head and use_separate_bottlenecks_head:
            num_layers_eff = max(1, len(backbone_layer_indices_head))
            head_module = MultiLayerHeadExport(
                embedding_dim=head_embedding_dim,
                num_outputs_main=head_num_main,
                num_outputs_ratio=head_num_ratio if head_is_ratio else 0,
                head_hidden_dims=head_hidden_dims,
                head_activation=head_activation,
                dropout=head_dropout,
                use_patch_reg3=use_patch_reg3_head,
                use_cls_token=use_cls_token_head,
                num_layers=num_layers_eff,
            )
        else:
            effective_outputs = head_total if not use_layerwise_heads_head else head_total * max(1, len(backbone_layer_indices_head))
            head_module = build_head_layer(
                embedding_dim=head_embedding_dim,
                num_outputs=effective_outputs
                if head_is_ratio
                else (
                    head_num_main
                    if not use_layerwise_heads_head
                    else head_num_main * max(1, len(backbone_layer_indices_head))
                ),
                head_hidden_dims=head_hidden_dims,
                head_activation=head_activation,
                dropout=head_dropout,
                use_output_softplus=False,
                input_dim=head_embedding_dim if (use_patch_reg3_head or (not use_cls_token_head)) else None,
            )
        head_module.load_state_dict(state, strict=True)

        if use_mp:
            try:
                head_module = head_module.to(device=dev1, dtype=mp_dtype)  # type: ignore[arg-type]
            except Exception:
                try:
                    head_module = head_module.to(device=dev1)  # type: ignore[arg-type]
                except Exception:
                    pass

        # --- Run inference for this head ---
        if use_patch_reg3_head:
            rels_in_order, preds_main, preds_ratio = predict_main_and_ratio_patch_mode(
                backbone=backbone,
                head=head_module,
                dataset_root=dataset_root,
                image_paths=image_paths,
                image_size=image_size,
                mean=mean,
                std=std,
                batch_size=batch_size,
                num_workers=num_workers,
                head_num_main=head_num_main,
                head_num_ratio=head_num_ratio if head_is_ratio else 0,
                head_total=head_total,
                use_layerwise_heads=use_layerwise_heads_head,
                num_layers=max(1, len(backbone_layer_indices_head)) if use_layerwise_heads_head else 1,
                use_separate_bottlenecks=use_separate_bottlenecks_head,
                layer_indices=backbone_layer_indices_head if use_layerwise_heads_head else None,
            )
        else:
            if use_layerwise_heads_head and len(backbone_layer_indices_head) > 0:
                rels_in_order, preds_main, preds_ratio = predict_main_and_ratio_global_multilayer(
                    backbone=backbone,
                    head=head_module,
                    dataset_root=dataset_root,
                    image_paths=image_paths,
                    image_size=image_size,
                    mean=mean,
                    std=std,
                    batch_size=batch_size,
                    num_workers=num_workers,
                    head_num_main=head_num_main,
                    head_num_ratio=head_num_ratio if head_is_ratio else 0,
                    head_total=head_total,
                    layer_indices=backbone_layer_indices_head,
                    use_separate_bottlenecks=use_separate_bottlenecks_head,
                    use_cls_token=use_cls_token_head,
                )
            else:
                feature_extractor = DinoV3FeatureExtractor(backbone)
                rels_in_order, features_cpu = extract_features_for_images(
                    feature_extractor=feature_extractor,
                    dataset_root=dataset_root,
                    image_paths=image_paths,
                    image_size=image_size,
                    mean=mean,
                    std=std,
                    batch_size=batch_size,
                    num_workers=num_workers,
                    use_cls_token=use_cls_token_head,
                )
                preds_main, preds_ratio = predict_from_features(
                    features_cpu=features_cpu,
                    head=head_module,
                    batch_size=batch_size,
                    head_num_main=head_num_main,
                    head_num_ratio=head_num_ratio if head_is_ratio else 0,
                    head_total=head_total,
                    use_layerwise_heads=False,
                    num_layers=1,
                )

        if rels_in_order_ref is None:
            rels_in_order_ref = rels_in_order
        elif rels_in_order_ref != rels_in_order:
            raise RuntimeError("Image order mismatch across heads within a model entry; aborting.")

        # --- Per-head normalization inversion (main outputs only) ---
        log_scale_cfg = bool(cfg["model"].get("log_scale_targets", False))
        log_scale_meta = bool(meta.get("log_scale_targets", log_scale_cfg))
        zscore = _load_zscore_json_for_head(head_pt)
        reg3_mean = None
        reg3_std = None
        if isinstance(zscore, dict) and "reg3" in zscore:
            try:
                reg3_mean = torch.tensor(zscore["reg3"]["mean"], dtype=torch.float32)
                reg3_std = torch.tensor(zscore["reg3"]["std"], dtype=torch.float32).clamp_min(1e-8)
            except Exception:
                reg3_mean, reg3_std = None, None
        if reg3_mean is not None and reg3_std is not None:
            preds_main = preds_main * reg3_std[:head_num_main] + reg3_mean[:head_num_main]  # type: ignore[index]
        if log_scale_meta:
            preds_main = torch.expm1(preds_main).clamp_min(0.0)

        # Convert from g/m^2 to grams
        preds_main_g = preds_main * float(area_m2)

        # Build this head's 5D components in grams
        N = preds_main_g.shape[0]
        if head_is_ratio and preds_ratio is not None and head_num_main >= 1 and head_num_ratio >= 1:
            total_g = preds_main_g[:, 0].view(N)
            p_ratio = F.softmax(preds_ratio, dim=-1)
            zeros = torch.zeros_like(total_g)
            comp_clover = (total_g * p_ratio[:, 0]) if head_num_ratio > 0 else zeros
            comp_dead = (total_g * p_ratio[:, 1]) if head_num_ratio > 1 else zeros
            comp_green = (total_g * p_ratio[:, 2]) if head_num_ratio > 2 else zeros
            comp_gdm = comp_clover + comp_green
            comps_5d = torch.stack([comp_clover, comp_dead, comp_green, comp_gdm, total_g], dim=-1)
        else:
            comps_list: List[torch.Tensor] = []
            for idx_row in range(N):
                base_map: Dict[str, float] = {}
                vec_row = preds_main_g[idx_row].tolist()
                for k, name in enumerate(target_bases):
                    if k < len(vec_row):
                        base_map[name] = float(vec_row[k])
                total = base_map.get("Dry_Total_g", None)
                clover = base_map.get("Dry_Clover_g", None)
                dead = base_map.get("Dry_Dead_g", None)
                green = base_map.get("Dry_Green_g", None)
                if total is None:
                    total = (clover or 0.0) + (dead or 0.0) + (green or 0.0)
                if dead is None and total is not None and clover is not None and green is not None:
                    dead = total - clover - green
                if clover is None:
                    clover = 0.0
                if dead is None:
                    dead = 0.0
                if green is None:
                    green = 0.0
                gdm_val = clover + green
                comps_list.append(torch.tensor([clover, dead, green, gdm_val, total], dtype=torch.float32))
            comps_5d = torch.stack(comps_list, dim=0) if comps_list else torch.zeros((0, 5), dtype=torch.float32)

        comps_5d = comps_5d.detach().cpu().float()
        if comps_sum is None:
            comps_sum = comps_5d * float(w)
        else:
            comps_sum = comps_sum + (comps_5d * float(w))
        weight_sum += float(w)

        # Free per-head GPU memory early
        try:
            del backbone
            del head_module
        except Exception:
            pass
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    if rels_in_order_ref is None or comps_sum is None or not (weight_sum > 0.0):
        raise RuntimeError("No valid heads were used for this model.")

    comps_avg = comps_sum / float(weight_sum)
    meta_out = {
        "backbone": backbone_name,
        "dino_weights": dino_weights_abs,
        "head_base": head_base_abs,
        "num_heads": len(head_weight_paths),
        "area_m2": float(area_m2),
        "image_size": list(image_size) if isinstance(image_size, (list, tuple)) else image_size,
    }
    return rels_in_order_ref, comps_avg, meta_out


def main():
    # Load configuration from project
    cfg = load_config(_PROJECT_DIR_ABS)

    # Data settings from config (single source of truth)
    image_size = _parse_image_size(cfg["data"]["image_size"])  # (H, W), e.g., (640, 640) or (640, 1280)
    mean = list(cfg["data"]["normalization"]["mean"])  # e.g., [0.485, 0.456, 0.406]
    std = list(cfg["data"]["normalization"]["std"])    # e.g., [0.229, 0.224, 0.225]
    target_bases = list(cfg["data"]["target_order"])    # legacy: base targets when using 3-d head
    # Dataset area (m^2) to convert g/m^2 (model output) back to grams for submission
    ds_name = str(cfg["data"].get("dataset", "csiro"))
    ds_map = dict(cfg["data"].get("datasets", {}))
    ds_info = dict(ds_map.get(ds_name, {}))
    try:
        width_m = float(ds_info.get("width_m", ds_info.get("width", 1.0)))
    except Exception:
        width_m = 1.0
    try:
        length_m = float(ds_info.get("length_m", ds_info.get("length", 1.0)))
    except Exception:
        length_m = 1.0
    try:
        area_m2 = float(ds_info.get("area_m2", width_m * length_m))
    except Exception:
        area_m2 = max(1.0, width_m * length_m if (width_m > 0.0 and length_m > 0.0) else 1.0)
    if not (area_m2 > 0.0):
        area_m2 = 1.0
    batch_size = int(cfg["data"].get("batch_size", 32))
    num_workers = int(cfg["data"].get("num_workers", 4))

    # Read test.csv
    dataset_root, test_csv = resolve_paths(INPUT_PATH)
    df = pd.read_csv(test_csv)
    if not {"sample_id", "image_path", "target_name"}.issubset(df.columns):
        raise ValueError("test.csv must contain columns: sample_id, image_path, target_name")
    unique_image_paths = df["image_path"].astype(str).unique().tolist()

    # ==========================================================
    # Multi-model ensemble path (supports mixing ViT/backbone types)
    # Strategy:
    #  - Run inference per model sequentially (per-model config + weights)
    #  - Cache each model's 5D predictions to disk
    #  - Read cached predictions and ensemble at the end
    # ==========================================================
    ensemble_models = _normalize_ensemble_models(_PROJECT_DIR_ABS)
    if len(ensemble_models) > 0:
        ensemble_obj = _read_ensemble_cfg_obj(_PROJECT_DIR_ABS) or {}
        cache_dir_cfg = ensemble_obj.get("cache_dir", None)
        if isinstance(cache_dir_cfg, str) and cache_dir_cfg.strip():
            cache_dir = _resolve_path_best_effort(_PROJECT_DIR_ABS, cache_dir_cfg.strip())
        else:
            cache_dir = os.path.join(_PROJECT_DIR_ABS, "outputs", "ensemble_cache")
        # Ensure cache_dir is writable (Kaggle note: /kaggle/input is read-only).
        try:
            os.makedirs(cache_dir, exist_ok=True)
        except Exception:
            pass
        if not (os.path.isdir(cache_dir) and os.access(cache_dir, os.W_OK)):
            # Prefer Kaggle working dir if present, otherwise fall back to system temp.
            fallback_roots = []
            try:
                env_root = os.environ.get("KAGGLE_WORKING_DIR", "").strip()
                if env_root:
                    fallback_roots.append(env_root)
            except Exception:
                pass
            fallback_roots.extend(
                [
                    "/kaggle/working",
                    tempfile.gettempdir(),
                    _PROJECT_DIR_ABS,
                ]
            )
            chosen_root = None
            for root in fallback_roots:
                try:
                    if root and os.path.isdir(root) and os.access(root, os.W_OK):
                        chosen_root = root
                        break
                except Exception:
                    continue
            if chosen_root is None:
                raise PermissionError(f"cache_dir is not writable and no fallback dir found: {cache_dir}")
            cache_dir_fallback = os.path.join(chosen_root, "ensemble_cache")
            os.makedirs(cache_dir_fallback, exist_ok=True)
            print(f"[ENSEMBLE] cache_dir not writable: {cache_dir} -> {cache_dir_fallback}")
            cache_dir = cache_dir_fallback
        else:
            print(f"[ENSEMBLE] cache_dir: {cache_dir}")

        cache_items: List[Tuple[str, float]] = []
        used_models: List[str] = []
        for idx, m in enumerate(ensemble_models):
            if not isinstance(m, dict):
                continue
            model_id = str(m.get("id", f"model_{idx}") or f"model_{idx}")
            version = m.get("version", None)
            try:
                model_weight = float(m.get("weight", 1.0))
            except Exception:
                model_weight = 1.0
            if not (model_weight > 0.0):
                continue

            # Resolve per-model YAML config
            cfg_path = None
            for k in ("config", "config_path", "train_yaml", "train_config"):
                v = m.get(k, None)
                if isinstance(v, str) and v.strip():
                    cfg_path = _resolve_path_best_effort(_PROJECT_DIR_ABS, v.strip())
                    break
            if cfg_path is None:
                if isinstance(version, (str, int, float)) and str(version).strip():
                    cfg_path = _resolve_version_train_yaml(_PROJECT_DIR_ABS, str(version).strip())
                else:
                    cfg_path = os.path.join(_PROJECT_DIR_ABS, "configs", "train.yaml")
            cfg_model = load_config_file(cfg_path)

            # Allow overriding backbone name at model level (optional)
            backbone_override = m.get("backbone", None)
            if isinstance(backbone_override, str) and backbone_override.strip():
                try:
                    if "model" not in cfg_model or not isinstance(cfg_model["model"], dict):
                        cfg_model["model"] = {}
                    cfg_model["model"]["backbone"] = backbone_override.strip()
                except Exception:
                    pass

            # Optional per-model inference overrides (do not affect training).
            try:
                if "data" not in cfg_model or not isinstance(cfg_model["data"], dict):
                    cfg_model["data"] = {}
                if "batch_size" in m:
                    cfg_model["data"]["batch_size"] = int(m.get("batch_size"))
                if "num_workers" in m:
                    cfg_model["data"]["num_workers"] = int(m.get("num_workers"))
            except Exception:
                pass

            # Resolve head weights base path
            head_base = None
            for k in ("head_base", "head_weights", "head_weights_path", "head_path"):
                v = m.get(k, None)
                if isinstance(v, str) and v.strip():
                    head_base = v.strip()
                    break
            if head_base is None:
                if isinstance(version, (str, int, float)) and str(version).strip():
                    head_base = _resolve_version_head_base(_PROJECT_DIR_ABS, str(version).strip())
                else:
                    head_base = HEAD_WEIGHTS_PT_PATH

            # Resolve backbone weights path
            backbone_name_eff = str(cfg_model.get("model", {}).get("backbone", "") or "").strip()
            dino_weights_path = _resolve_dino_weights_path_for_model(
                _PROJECT_DIR_ABS,
                backbone_name=backbone_name_eff,
                cfg=cfg_model,
                model_cfg=m,
            )
            print(f"[ENSEMBLE] Model '{model_id}': backbone={backbone_name_eff}, dino_weights={dino_weights_path}")

            # Cache file path
            cache_name = str(m.get("cache_name", "") or "").strip()
            if not cache_name:
                cache_name = _safe_slug(str(version).strip() if version is not None else model_id)
            cache_path = os.path.join(cache_dir, f"{cache_name}.pt")

            print(f"[ENSEMBLE] Running model '{model_id}' (version={version}) -> cache: {cache_path}")
            rels_in_order, comps_5d_g, meta = infer_components_5d_for_model(
                project_dir=_PROJECT_DIR_ABS,
                cfg=cfg_model,
                head_base=head_base,
                dino_weights_pt_path=dino_weights_path,
                dataset_root=dataset_root,
                image_paths=unique_image_paths,
            )
            meta = dict(meta or {})
            meta.update(
                {
                    "model_id": model_id,
                    "version": str(version) if version is not None else None,
                    "cfg_path": cfg_path,
                    "head_base": head_base,
                }
            )
            _save_ensemble_cache(
                cache_path,
                model_id=model_id,
                model_weight=model_weight,
                rels_in_order=rels_in_order,
                comps_5d_g=comps_5d_g,
                meta=meta,
            )
            cache_items.append((cache_path, float(model_weight)))
            used_models.append(model_id)

            # Release memory between models
            try:
                del comps_5d_g
            except Exception:
                pass
            try:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass

        if not cache_items:
            raise RuntimeError("Ensemble is enabled but no valid models were executed.")

        # Load cached predictions and ensemble
        rels_ref: Optional[List[str]] = None
        comps_sum: Optional[torch.Tensor] = None
        w_sum: float = 0.0
        for cache_path, w in cache_items:
            if not (w > 0.0):
                continue
            rels, comps, _meta = _load_ensemble_cache(cache_path)
            if rels_ref is None:
                rels_ref = rels
            elif rels_ref != rels:
                raise RuntimeError("Image order mismatch across cached models; aborting ensemble.")
            comps_sum = (comps * float(w)) if comps_sum is None else (comps_sum + comps * float(w))
            w_sum += float(w)

        if rels_ref is None or comps_sum is None or not (w_sum > 0.0):
            raise RuntimeError("Failed to build an ensemble from cached predictions.")

        comps_avg = comps_sum / float(w_sum)  # (N,5)
        # Build image -> component mapping
        image_to_components: Dict[str, Dict[str, float]] = {}
        for rel_path, vec in zip(rels_ref, comps_avg.tolist()):
            clover_g, dead_g, green_g, gdm_g, total_g = vec
            image_to_components[str(rel_path)] = {
                "Dry_Total_g": float(total_g),
                "Dry_Clover_g": float(clover_g),
                "Dry_Dead_g": float(dead_g),
                "Dry_Green_g": float(green_g),
                "GDM_g": float(gdm_g),
            }

        # Build submission
        rows = []
        for _, r in df.iterrows():
            sample_id = str(r["sample_id"])
            rel_path = str(r["image_path"])
            target_name = str(r["target_name"])
            comps = image_to_components.get(rel_path, {})
            value = comps.get(target_name, 0.0)
            value = max(0.0, float(value))
            rows.append((sample_id, value))

        out_dir = os.path.dirname(os.path.abspath(OUTPUT_SUBMISSION_PATH))
        if out_dir and not os.path.isdir(out_dir):
            os.makedirs(out_dir, exist_ok=True)
        with open(OUTPUT_SUBMISSION_PATH, "w", encoding="utf-8") as f:
            f.write("sample_id,target\n")
            for sample_id, value in rows:
                f.write(f"{sample_id},{value}\n")

        print(f"[ENSEMBLE] Models ensembled: {used_models}")
        print(f"Submission written to: {OUTPUT_SUBMISSION_PATH}")
        return

    # Build model and load weights (supports single head or k-fold heads under a directory)
    # Strictly offline path: require both backbone and head weights
    if not HEAD_WEIGHTS_PT_PATH:
        raise FileNotFoundError("HEAD_WEIGHTS_PT_PATH must be set to a valid head file or directory.")

    # Import correct DINOv3 constructor based on config
    try:
        backbone_name = str(cfg["model"]["backbone"]).strip()
        if backbone_name == "dinov3_vith16plus":
            from dinov3.hub.backbones import dinov3_vith16plus as _make_backbone  # type: ignore
        elif backbone_name == "dinov3_vitl16":
            from dinov3.hub.backbones import dinov3_vitl16 as _make_backbone  # type: ignore
        elif backbone_name in ("dinov3_vit7b16", "dinov3_vit7b", "vit7b16", "vit7b"):
            from dinov3.hub.backbones import dinov3_vit7b16 as _make_backbone  # type: ignore
        else:
            raise ImportError(f"Unsupported backbone in config: {backbone_name}")
    except Exception:
        raise ImportError(
            "dinov3 is not available locally. Ensure DINOV3_PATH points to third_party/dinov3/dinov3."
        )

    # Resolve DINOv3 backbone weights:
    # Allow DINO_WEIGHTS_PT_PATH to be either a file or a directory containing the official DINOv3 weights.
    dino_weights_pt_path = ""
    try:
        # Backward-compat: if user points directly to an existing file, use it as-is.
        if isinstance(DINO_WEIGHTS_PT_PATH, str) and DINO_WEIGHTS_PT_PATH.strip():
            p0 = _resolve_path_best_effort(_PROJECT_DIR_ABS, DINO_WEIGHTS_PT_PATH)
            if os.path.isfile(p0):
                dino_weights_pt_path = os.path.abspath(p0)
        # Otherwise, run best-effort discovery based on backbone name + standard filenames.
        if not dino_weights_pt_path:
            dino_weights_pt_path = _resolve_dino_weights_path_for_model(
                _PROJECT_DIR_ABS,
                backbone_name=backbone_name,
                cfg=cfg,
                model_cfg={},
            )
    except FileNotFoundError:
        dino_weights_pt_path = ""
    if not (dino_weights_pt_path and os.path.isfile(dino_weights_pt_path)):
        raise FileNotFoundError(
            "DINO_WEIGHTS_PT_PATH must point to a valid backbone .pt file, or a directory containing the official DINOv3 weights."
        )
    # Make it visible in logs for debugging (especially when DINO_WEIGHTS_PT_PATH is a directory).
    print(f"[WEIGHTS] Using DINO backbone weights: {dino_weights_pt_path}")

    # Discover head weights:
    # If configs/ensemble.json enabled, prefer packaged manifest weights/head/ensemble.json; otherwise scan directory.
    ensemble_enabled = _read_ensemble_enabled_flag(_PROJECT_DIR_ABS)
    if ensemble_enabled:
        pkg_manifest = _read_packaged_ensemble_manifest_if_exists(HEAD_WEIGHTS_PT_PATH)
        if pkg_manifest is not None:
            head_entries, aggregation = pkg_manifest
            head_weight_paths = [p for (p, _w) in head_entries]
            weight_map = {p: 1.0 for p in head_weight_paths}
            use_weighted = False
        else:
            head_weight_paths = discover_head_weight_paths(HEAD_WEIGHTS_PT_PATH)
            weight_map = {p: 1.0 for p in head_weight_paths}
            use_weighted = False
    else:
        head_weight_paths = discover_head_weight_paths(HEAD_WEIGHTS_PT_PATH)
        weight_map = {p: 1.0 for p in head_weight_paths}
        use_weighted = False

    # Inspect first head file to infer packed head shape (main + optional ratio outputs)
    first_state, first_meta, _ = load_head_state(head_weight_paths[0])
    if not isinstance(first_meta, dict):
        first_meta = {}
    # Default head shape (fallback when individual head meta is missing)
    num_outputs_main_default = int(first_meta.get("num_outputs_main", first_meta.get("num_outputs", 3)))
    num_outputs_ratio_default = int(first_meta.get("num_outputs_ratio", 0))
    head_total_outputs_default = int(first_meta.get("head_total_outputs", num_outputs_main_default + num_outputs_ratio_default))
    ratio_components_default = first_meta.get("ratio_components", [])
    use_patch_reg3_default = bool(first_meta.get("use_patch_reg3", False))
    use_cls_token_default = bool(first_meta.get("use_cls_token", True))
    use_layerwise_heads_default = bool(first_meta.get("use_layerwise_heads", False))
    backbone_layer_indices_default = list(first_meta.get("backbone_layer_indices", []))
    use_separate_bottlenecks_default = bool(first_meta.get("use_separate_bottlenecks", False))

    # Build weighted/mean ensemble by running per-head feature extraction (supports per-head LoRA).
    # Only require that all heads share the same embedding_dim (backbone choice); ratio/patch
    # settings are handled independently per head.
    cfg_embedding_dim = int(cfg["model"]["embedding_dim"])

    # Reference image order and per-image accumulators of 5D components:
    # [Dry_Clover_g, Dry_Dead_g, Dry_Green_g, GDM_g, Dry_Total_g]
    rels_in_order_ref: Optional[List[str]] = None
    image_to_sum: Dict[str, torch.Tensor] = {}
    image_to_weight: Dict[str, float] = {}
    num_heads_used: int = 0

    # Preload DINO backbone state (reused per-head)
    # IMPORTANT for Kaggle (31GB RAM): ViT7B fp32 checkpoints are ~26GB on disk and can OOM if fully loaded into RAM.
    # Use mmap=True (if supported) to memory-map the checkpoint and stream tensors into GPU params during load_state_dict.
    dino_state = _torch_load_cpu(dino_weights_pt_path, mmap=True, weights_only=True)
    if isinstance(dino_state, dict) and "state_dict" in dino_state:
        dino_state = dino_state["state_dict"]

    # Run per-head (each head can independently choose ratio/patch configuration)
    for head_pt in head_weight_paths:
        w = float(weight_map.get(head_pt, 1.0))
        # Load head state and meta
        state, meta, peft_payload = load_head_state(head_pt)
        if not isinstance(meta, dict):
            meta = {}

        # Use meta to build head architecture (do not rely on YAML defaults)
        head_num_main = int(meta.get("num_outputs_main", meta.get("num_outputs", num_outputs_main_default)))
        head_num_ratio = int(meta.get("num_outputs_ratio", num_outputs_ratio_default))
        head_total = int(meta.get("head_total_outputs", head_num_main + head_num_ratio))
        head_hidden_dims = list(meta.get("head_hidden_dims", first_meta.get("head_hidden_dims", list(cfg["model"]["head"].get("hidden_dims", [512, 256])))))
        head_activation = str(meta.get("head_activation", first_meta.get("head_activation", str(cfg["model"]["head"].get("activation", "relu")))))
        head_dropout = float(meta.get("head_dropout", first_meta.get("head_dropout", float(cfg["model"]["head"].get("dropout", 0.0)))))
        head_embedding_dim = int(meta.get("embedding_dim", cfg_embedding_dim))

        # Sanity: require embedding_dim in meta to match config embedding (backbone choice)
        if head_embedding_dim != cfg_embedding_dim:
            raise RuntimeError(
                f"Head {head_pt} embedding_dim({head_embedding_dim}) != cfg.model.embedding_dim({cfg_embedding_dim}); "
                "mixing heads trained on different backbones is unsupported in a single run."
            )

        # Build a fresh backbone per head, load base DINO weights.
        # For the VERY large vit7b16, use Scheme B (2-GPU model parallel) to fit on 2x16GB.
        use_mp = (
            bool(USE_2GPU_MODEL_PARALLEL_FOR_VIT7B)
            and (backbone_name in ("dinov3_vit7b16", "dinov3_vit7b", "vit7b16", "vit7b"))
            and (_mp_get_devices() is not None)
        )
        if use_mp:
            dev0, dev1 = _mp_get_devices()  # type: ignore[assignment]
            split_idx = int(VIT7B_MP_SPLIT_IDX)
            mp_dtype = _mp_resolve_dtype(VIT7B_MP_DTYPE)
            # Instantiate on meta to avoid CPU RAM spikes
            backbone = _make_backbone(pretrained=False, device="meta")
            _mp_prepare_vit7b_backbone_two_gpu(
                backbone,
                split_idx=split_idx,
                dtype=mp_dtype,
                device0=dev0,
                device1=dev1,
            )
            # Load weights into already-sharded modules
            try:
                backbone.load_state_dict(dino_state, strict=True)
            except Exception:
                backbone.load_state_dict(dino_state, strict=False)
            # Patch methods to support cross-device execution
            _mp_patch_dinov3_methods(backbone, split_idx=split_idx, device0=dev0, device1=dev1)
        else:
            backbone = _make_backbone(pretrained=False)
            try:
                backbone.load_state_dict(dino_state, strict=True)
            except Exception:
                backbone.load_state_dict(dino_state, strict=False)

        # Inject per-head LoRA adapters if bundle contains PEFT payload
        try:
            if peft_payload is not None and isinstance(peft_payload, dict):
                peft_cfg_dict = peft_payload.get("config", None)
                peft_state = peft_payload.get("state_dict", None)
                if peft_cfg_dict and peft_state:
                    try:
                        from peft.config import PeftConfig  # type: ignore
                        from peft.mapping_func import get_peft_model  # type: ignore
                        from peft.utils.save_and_load import set_peft_model_state_dict  # type: ignore
                    except Exception:
                        LoraConfig, get_peft_model_alt, _, _ = _import_peft()  # noqa: F841
                        from peft.config import PeftConfig  # type: ignore
                        from peft.utils.save_and_load import set_peft_model_state_dict  # type: ignore
                        get_peft_model = get_peft_model_alt  # type: ignore
                    peft_config = PeftConfig.from_peft_type(**peft_cfg_dict)
                    backbone = get_peft_model(backbone, peft_config)
                    set_peft_model_state_dict(backbone, peft_state, adapter_name="default")
                    backbone.eval()
                    # Preserve model-parallel metadata when wrapping
                    if use_mp:
                        try:
                            _mp_attach_flags(backbone, dev0, dev1, split_idx)  # type: ignore[arg-type]
                        except Exception:
                            pass
        except Exception as _e:
            print(f"[WARN] PEFT injection skipped for {head_pt}: {_e}")

        # Build head module according to its own meta (packed main + optional ratio outputs)
        # Determine whether this specific head was trained with patch-based main regression.
        # IMPORTANT: some older heads (e.g., pure ratio MLPs trained on global CLS+mean(patch))
        # do not store `use_patch_reg3` in their meta. For such heads we must *not* inherit
        # the default from the first (typically patch-based) head, otherwise we would
        # incorrectly build a patch-mode MLP with input_dim=embedding_dim instead of
        # 2 * embedding_dim, leading to shape mismatches when loading their checkpoints.
        #
        # Therefore, only heads that explicitly declare `use_patch_reg3: true` in meta
        # are treated as patch-mode heads; all others default to the legacy global path.
        use_patch_reg3_head = bool(meta.get("use_patch_reg3", False))
        # Whether this head expects CLS in global features.
        use_cls_token_head = bool(meta.get("use_cls_token", use_cls_token_default))
        # Determine whether this head uses layer-wise heads and which backbone layers.
        use_layerwise_heads_head = bool(meta.get("use_layerwise_heads", use_layerwise_heads_default))
        backbone_layer_indices_head = list(meta.get("backbone_layer_indices", backbone_layer_indices_default))
        use_separate_bottlenecks_head = bool(
            meta.get("use_separate_bottlenecks", use_separate_bottlenecks_default)
        )
        # Determine whether this head uses the ratio format.
        head_is_ratio = bool(head_num_ratio > 0 and head_total == (head_num_main + head_num_ratio))
        # When layer-wise heads are used, the packed head normally stores concatenated
        # per-layer outputs in its final linear layer. When separate bottlenecks are
        # also enabled, we instead export an explicit MultiLayerHeadExport that mirrors
        # the training-time per-layer structure.
        if use_layerwise_heads_head and use_separate_bottlenecks_head:
            num_layers_eff = max(1, len(backbone_layer_indices_head))
            head_module = MultiLayerHeadExport(
                embedding_dim=head_embedding_dim,
                num_outputs_main=head_num_main,
                num_outputs_ratio=head_num_ratio if head_is_ratio else 0,
                head_hidden_dims=head_hidden_dims,
                head_activation=head_activation,
                dropout=head_dropout,
                use_patch_reg3=use_patch_reg3_head,
                use_cls_token=use_cls_token_head,
                num_layers=num_layers_eff,
            )
        else:
            # Legacy / shared-bottleneck packed head.
            effective_outputs = head_total if not use_layerwise_heads_head else head_total * max(
                1, len(backbone_layer_indices_head)
            )
            head_module = build_head_layer(
                embedding_dim=head_embedding_dim,
                num_outputs=effective_outputs
                if head_is_ratio
                else (
                    head_num_main
                    if not use_layerwise_heads_head
                    else head_num_main * max(1, len(backbone_layer_indices_head))
                ),
                head_hidden_dims=head_hidden_dims,
                head_activation=head_activation,
                dropout=head_dropout,
                use_output_softplus=False,
                # For patch-mode heads, the packed MLP expects patch-token dimensionality
                # (embedding_dim) as input. Global heads default to CLS+mean(patch) with
                # 2 * embedding_dim; when CLS is disabled, they use mean(patch) with embedding_dim.
                input_dim=head_embedding_dim if (use_patch_reg3_head or (not use_cls_token_head)) else None,
            )
        head_module.load_state_dict(state, strict=True)
        # When using 2-GPU model-parallel ViT7B, default to fp16 head weights on stage1
        # to match backbone output dtype and reduce memory/compute on Kaggle T4.
        if use_mp:
            try:
                head_module = head_module.to(device=dev1, dtype=mp_dtype)  # type: ignore[arg-type]
            except Exception:
                try:
                    head_module = head_module.to(device=dev1)  # type: ignore[arg-type]
                except Exception:
                    pass

        if use_patch_reg3_head:
            # Patch-based main regression: compute per-patch predictions and average.
            # When layer-wise heads are enabled, use multiple backbone layers with
            # per-layer predictions averaged across layers; otherwise fall back to
            # last-layer-only behavior.
            rels_in_order, preds_main, preds_ratio = predict_main_and_ratio_patch_mode(
                backbone=backbone,
                head=head_module,
                dataset_root=dataset_root,
                image_paths=unique_image_paths,
                image_size=image_size,
                mean=mean,
                std=std,
                batch_size=batch_size,
                num_workers=num_workers,
                head_num_main=head_num_main,
                head_num_ratio=head_num_ratio if head_is_ratio else 0,
                head_total=head_total,
                use_layerwise_heads=use_layerwise_heads_head,
                num_layers=max(1, len(backbone_layer_indices_head)) if use_layerwise_heads_head else 1,
                use_separate_bottlenecks=use_separate_bottlenecks_head,
                layer_indices=backbone_layer_indices_head if use_layerwise_heads_head else None,
            )
        else:
            if use_layerwise_heads_head and len(backbone_layer_indices_head) > 0:
                # Global multi-layer path: per-layer global features (CLS+mean(patch) or mean(patch)).
                rels_in_order, preds_main, preds_ratio = predict_main_and_ratio_global_multilayer(
                    backbone=backbone,
                    head=head_module,
                    dataset_root=dataset_root,
                    image_paths=unique_image_paths,
                    image_size=image_size,
                    mean=mean,
                    std=std,
                    batch_size=batch_size,
                    num_workers=num_workers,
                    head_num_main=head_num_main,
                    head_num_ratio=head_num_ratio if head_is_ratio else 0,
                    head_total=head_total,
                    layer_indices=backbone_layer_indices_head,
                    use_separate_bottlenecks=use_separate_bottlenecks_head,
                    use_cls_token=use_cls_token_head,
                )
            else:
                # Legacy single-layer path: extract global features from last layer.
                feature_extractor = DinoV3FeatureExtractor(backbone)
                rels_in_order, features_cpu = extract_features_for_images(
                    feature_extractor=feature_extractor,
                    dataset_root=dataset_root,
                    image_paths=unique_image_paths,
                    image_size=image_size,
                    mean=mean,
                    std=std,
                    batch_size=batch_size,
                    num_workers=num_workers,
                    use_cls_token=use_cls_token_head,
                )
                preds_main, preds_ratio = predict_from_features(
                    features_cpu=features_cpu,
                    head=head_module,
                    batch_size=batch_size,
                    head_num_main=head_num_main,
                    head_num_ratio=head_num_ratio if head_is_ratio else 0,
                    head_total=head_total,
                    use_layerwise_heads=False,
                    num_layers=1,
                )

        if rels_in_order_ref is None:
            rels_in_order_ref = rels_in_order
        else:
            # Basic consistency check
            if rels_in_order_ref != rels_in_order:
                raise RuntimeError("Image order mismatch across heads; aborting.")

        # Per-head normalization inversion (main outputs only)
        # Decide log-scale for main reg3 outputs
        log_scale_cfg = bool(cfg["model"].get("log_scale_targets", False))
        log_scale_meta = bool(meta.get("log_scale_targets", log_scale_cfg))
        zscore = _load_zscore_json_for_head(head_pt)
        reg3_mean = None
        reg3_std = None
        if isinstance(zscore, dict) and "reg3" in zscore:
            try:
                reg3_mean = torch.tensor(zscore["reg3"]["mean"], dtype=torch.float32)
                reg3_std = torch.tensor(zscore["reg3"]["std"], dtype=torch.float32).clamp_min(1e-8)
            except Exception:
                reg3_mean, reg3_std = None, None
        zscore_enabled = reg3_mean is not None and reg3_std is not None
        if zscore_enabled:
            preds_main = preds_main * reg3_std[:head_num_main] + reg3_mean[:head_num_main]  # type: ignore[index]
        if log_scale_meta:
            preds_main = torch.expm1(preds_main).clamp_min(0.0)

        # Convert main predictions from g/m^2 to grams for this head
        preds_main_g = preds_main * float(area_m2)  # (N, head_num_main)

        # Build this head's 5D components in grams:
        # order: [Dry_Clover_g, Dry_Dead_g, Dry_Green_g, GDM_g, Dry_Total_g]
        N = preds_main_g.shape[0]
        if head_is_ratio and preds_ratio is not None and head_num_main >= 1 and head_num_ratio >= 1:
            # Ratio-format head: main_g (Dry_Total_g) + ratio logits for 3 components.
            total_g = preds_main_g[:, 0].view(N)  # (N,)
            p_ratio = F.softmax(preds_ratio, dim=-1)  # (N, head_num_ratio)
            zeros = torch.zeros_like(total_g)
            comp_clover = (total_g * p_ratio[:, 0]) if head_num_ratio > 0 else zeros
            comp_dead = (total_g * p_ratio[:, 1]) if head_num_ratio > 1 else zeros
            comp_green = (total_g * p_ratio[:, 2]) if head_num_ratio > 2 else zeros
            comp_gdm = comp_clover + comp_green
            comps_5d = torch.stack(
                [comp_clover, comp_dead, comp_green, comp_gdm, total_g],
                dim=-1,
            )  # (N, 5)
        else:
            # Legacy-format head: directly predict base targets (e.g., Dry_Total_g or 3D components).
            comps_list: List[torch.Tensor] = []
            for idx_row in range(N):
                base_map: Dict[str, float] = {}
                vec_row = preds_main_g[idx_row].tolist()
                # Map available outputs onto target_bases
                for k, name in enumerate(target_bases):
                    if k < len(vec_row):
                        base_map[name] = float(vec_row[k])
                total = base_map.get("Dry_Total_g", None)
                clover = base_map.get("Dry_Clover_g", None)
                dead = base_map.get("Dry_Dead_g", None)
                green = base_map.get("Dry_Green_g", None)
                if total is None:
                    total = (clover or 0.0) + (dead or 0.0) + (green or 0.0)
                if dead is None and total is not None and clover is not None and green is not None:
                    dead = total - clover - green
                if clover is None:
                    clover = 0.0
                if dead is None:
                    dead = 0.0
                if green is None:
                    green = 0.0
                gdm_val = clover + green
                comps_list.append(
                    torch.tensor(
                        [clover, dead, green, gdm_val, total],
                        dtype=torch.float32,
                    )
                )
            comps_5d = torch.stack(comps_list, dim=0) if comps_list else torch.zeros((0, 5), dtype=torch.float32)

        # Accumulate this head's 5D components into global ensemble sums.
        w_eff = float(w if use_weighted else 1.0)
        for idx_row, rel in enumerate(rels_in_order):
            comp_vec = comps_5d[idx_row]
            if rel not in image_to_sum:
                image_to_sum[rel] = torch.zeros(5, dtype=torch.float32)
                image_to_weight[rel] = 0.0
            image_to_sum[rel] += (comp_vec * w_eff)
            image_to_weight[rel] += w_eff

        num_heads_used += 1

    if num_heads_used == 0:
        raise RuntimeError("No valid head weights found for inference.")

    # Aggregate per-image 5D components across heads.
    image_to_components: Dict[str, Dict[str, float]] = {}
    for rel_path, sum_vec in image_to_sum.items():
        denom = float(image_to_weight.get(rel_path, 0.0))
        if denom <= 0.0:
            continue
        avg_vec = (sum_vec / denom).tolist()
        clover_g, dead_g, green_g, gdm_g, total_g = avg_vec
        image_to_components[rel_path] = {
            "Dry_Total_g": float(total_g),
            "Dry_Clover_g": float(clover_g),
            "Dry_Dead_g": float(dead_g),
            "Dry_Green_g": float(green_g),
            "GDM_g": float(gdm_g),
        }

    # Build submission
    rows = []
    for _, r in df.iterrows():
        sample_id = str(r["sample_id"])  # e.g., IDxxxx__Dry_Clover_g
        rel_path = str(r["image_path"])  # e.g., test/IDxxxx.jpg
        target_name = str(r["target_name"])  # one of 5
        comps = image_to_components.get(rel_path, {})
        value = comps.get(target_name, 0.0)
        # Clamp final physical predictions to be non-negative for submission
        value = max(0.0, float(value))
        rows.append((sample_id, value))

    out_dir = os.path.dirname(os.path.abspath(OUTPUT_SUBMISSION_PATH))
    if out_dir and not os.path.isdir(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    with open(OUTPUT_SUBMISSION_PATH, "w", encoding="utf-8") as f:
        f.write("sample_id,target\n")
        for sample_id, value in rows:
            f.write(f"{sample_id},{value}\n")
    print(f"Submission written to: {OUTPUT_SUBMISSION_PATH}")


if __name__ == "__main__":
    main()


