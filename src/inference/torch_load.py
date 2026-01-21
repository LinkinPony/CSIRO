from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch import nn


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
        # Prefer vendored PEFT if present (mirrors infer_and_submit_pt.py behavior).
        from src.models.peft_integration import _import_peft  # type: ignore

        _import_peft()
        from peft.utils.peft_types import PeftType  # type: ignore
    except Exception:
        PeftType = None  # type: ignore

if add_safe_globals is not None and PeftType is not None:  # type: ignore
    try:
        add_safe_globals([PeftType])  # type: ignore
    except Exception:
        pass


def torch_load_cpu(
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
    """
    Load a TorchScript module or a state_dict/checkpoint dict.

    Returns:
      (model, state_dict, meta)
    """
    if not os.path.isfile(pt_path):
        raise FileNotFoundError(f"Weights not found: {pt_path}")
    # 1) Try TorchScript (best for offline single-file inference)
    try:
        scripted = torch.jit.load(pt_path, map_location="cpu")
        return scripted, None, {"format": "torchscript"}
    except Exception:
        pass
    # 2) Fallback to torch.load objects (may be state_dict or checkpoint dict)
    obj = torch_load_cpu(pt_path, mmap=None, weights_only=True)
    if isinstance(obj, dict) and "state_dict" in obj:
        return None, obj["state_dict"], obj.get("meta", {})
    if isinstance(obj, dict):
        # raw state_dict
        return None, obj, {}
    if isinstance(obj, nn.Module):
        # Pickled module (works only if class definitions are available)
        return obj, None, {"format": "pickled_module"}
    raise RuntimeError("Unsupported weights file format. Provide a TorchScript .pt for offline inference.")


def load_head_state(pt_path: str) -> Tuple[dict, dict, Optional[dict]]:
    """
    Load head-only weights package (dict with 'state_dict', 'meta', optional 'peft').
    """
    if not os.path.isfile(pt_path):
        raise FileNotFoundError(f"Head weights not found: {pt_path}")
    obj = torch_load_cpu(pt_path, mmap=None, weights_only=True)
    if isinstance(obj, dict) and "state_dict" in obj:
        state = obj["state_dict"]
        meta_raw = obj.get("meta", {})
        meta: Dict[str, Any] = dict(meta_raw) if isinstance(meta_raw, dict) else {}
        return state, _reconcile_head_meta_with_state_dict(meta, state), obj.get("peft", None)
    if isinstance(obj, dict):
        return obj, {}, obj.get("peft", None)
    raise RuntimeError("Unsupported head weights file format. Expect a dict with 'state_dict'.")


def _infer_sequential_linear_hidden_dims_from_state_dict(
    state_dict: Dict[str, Any],
    *,
    root_prefix: Tuple[str, ...],
    seq_name: str,
) -> List[int]:
    """
    Best-effort infer MLP hidden dims from Sequential Linear weights in a state_dict.

    We look for keys like:
      - scalar_mlp.<idx>.weight
      - heads.0.scalar_mlp.<idx>.weight

    Returns:
      A list of out_features in ascending module index order, e.g. [512, 256].
      Returns [] if no matching Linear weights are found.
    """
    dims_by_idx: Dict[int, int] = {}
    for k, v in state_dict.items():
        if not isinstance(k, str):
            continue
        if not (isinstance(v, torch.Tensor) and v.ndim == 2):
            continue
        parts = k.split(".")
        # Apply optional prefix (e.g., ("heads","0") for ViTDetMultiLayerScalarHead)
        if root_prefix:
            if tuple(parts[: len(root_prefix)]) != tuple(root_prefix):
                continue
            parts = parts[len(root_prefix) :]
        # Expect: <seq_name>.<idx>.weight
        if len(parts) != 3:
            continue
        if parts[0] != seq_name:
            continue
        if parts[2] != "weight":
            continue
        idx_s = parts[1]
        if not idx_s.isdigit():
            continue
        dims_by_idx[int(idx_s)] = int(v.shape[0])
    if not dims_by_idx:
        return []
    return [dims_by_idx[i] for i in sorted(dims_by_idx)]


def _infer_vitdet_head_hidden_dims(state_dict: Dict[str, Any]) -> List[int]:
    """
    Infer bottleneck `head_hidden_dims` from an exported head `state_dict`.

    This is used for backward compatibility with older exported heads where
    `meta['head_hidden_dims']` was recorded as [] but the actual head weights
    include an MLP (e.g., 960 -> 512 -> 256).

    Note:
      This logic also applies to other heads that store an MLP under `scalar_mlp`
      (including the PyTorch-only Mamba head), because the state_dict key layout
      is the same: either `scalar_mlp.<idx>.weight` or `heads.<i>.scalar_mlp.<idx>.weight`.
    """
    # Prefer the first sub-head if this is a multi-layer wrapper.
    head_idxs: List[int] = []
    for k in state_dict.keys():
        if not isinstance(k, str):
            continue
        parts = k.split(".")
        if len(parts) >= 3 and parts[0] == "heads" and parts[1].isdigit() and parts[2] == "scalar_mlp":
            try:
                head_idxs.append(int(parts[1]))
            except Exception:
                continue
    root_prefix: Tuple[str, ...]
    if head_idxs:
        root_prefix = ("heads", str(min(head_idxs)))
    else:
        root_prefix = ()
    return _infer_sequential_linear_hidden_dims_from_state_dict(state_dict, root_prefix=root_prefix, seq_name="scalar_mlp")


def _reconcile_head_meta_with_state_dict(meta: Dict[str, Any], state_dict: dict) -> Dict[str, Any]:
    """
    Backward-compatible meta reconciliation for strict head loading.

    Problem we guard against:
      - Some older training runs treated `head_hidden_dims=[]` as "unset" and silently
        fell back to the default [512, 256] *during model construction*.
      - The exported head checkpoint stored the actual weights (with the MLP) but wrote
        meta `head_hidden_dims: []`, causing strict load failures in inference/packaging.

    Strategy:
      - For ViTDet and Mamba heads, if meta declares an empty `head_hidden_dims` but the state_dict
        contains Linear weights inside `scalar_mlp`, infer the hidden dims and override meta.
      - If there are no `scalar_mlp` Linear weights, keep [] (this corresponds to a true linear head).
    """
    try:
        head_type = str(meta.get("head_type", "") or "").strip().lower()
    except Exception:
        head_type = ""
    if head_type not in ("vitdet", "mamba"):
        return meta

    hd = meta.get("head_hidden_dims", None)
    try:
        hd_list = list(hd) if isinstance(hd, (list, tuple)) else None
    except Exception:
        hd_list = None

    # Only override when meta *explicitly* says "no hidden dims" but weights clearly contain an MLP.
    if hd_list is not None and len(hd_list) == 0 and isinstance(state_dict, dict):
        inferred = _infer_vitdet_head_hidden_dims(state_dict)
        if inferred:
            meta2 = dict(meta)
            meta2["head_hidden_dims"] = [int(x) for x in inferred]
            # Optional breadcrumb for debugging (not relied upon by inference code)
            meta2["head_hidden_dims_inferred_from_state_dict"] = True
            return meta2

    return meta


