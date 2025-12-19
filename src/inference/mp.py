from __future__ import annotations

import types
from typing import List, Optional, Tuple

import torch
from torch import nn


# ==========================================================
# 2-GPU model-parallel helpers for dinov3_vit7b16 (Scheme B)
# ==========================================================
def mp_get_devices() -> Optional[Tuple[torch.device, torch.device]]:
    try:
        if torch.cuda.is_available() and torch.cuda.device_count() >= 2:
            return torch.device("cuda:0"), torch.device("cuda:1")
    except Exception:
        pass
    return None


def mp_resolve_dtype(dtype_str: str) -> torch.dtype:
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


def mp_get_backbone_for_attrs(backbone: nn.Module) -> nn.Module:
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


def mp_attach_flags(backbone: nn.Module, device0: torch.device, device1: torch.device, split_idx: int) -> None:
    try:
        backbone._mp_devices = (device0, device1)  # type: ignore[attr-defined]
        backbone._mp_split_idx = int(split_idx)  # type: ignore[attr-defined]
        backbone._mp_enabled = True  # type: ignore[attr-defined]
    except Exception:
        pass


def mp_get_devices_from_backbone(backbone: nn.Module) -> Optional[Tuple[torch.device, torch.device]]:
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


def mp_prepare_vit7b_backbone_two_gpu(
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
    m = mp_get_backbone_for_attrs(backbone)

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
    mp_attach_flags(m, device0, device1, split_idx)
    return backbone


def mp_patch_dinov3_methods(backbone: nn.Module, *, split_idx: int, device0: torch.device, device1: torch.device) -> None:
    """
    Monkeypatch forward_features + get_intermediate_layers on the underlying dinov3 backbone
    so it can run with blocks split across cuda:0/1.
    """
    m = mp_get_backbone_for_attrs(backbone)

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
    mp_attach_flags(m, device0, device1, split_idx)


def module_param_dtype(m: nn.Module, *, default: torch.dtype = torch.float32) -> torch.dtype:
    """
    Best-effort infer module parameter dtype (for casting inputs to match weights).
    """
    try:
        return next(m.parameters()).dtype
    except Exception:
        return default


