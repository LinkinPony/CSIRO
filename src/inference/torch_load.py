from __future__ import annotations

import os
from typing import Optional, Tuple

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
        return obj["state_dict"], obj.get("meta", {}), obj.get("peft", None)
    if isinstance(obj, dict):
        return obj, {}, obj.get("peft", None)
    raise RuntimeError("Unsupported head weights file format. Expect a dict with 'state_dict'.")


