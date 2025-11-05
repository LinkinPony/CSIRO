from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Tuple

import torch
from torch import nn


def _import_peft():
    try:
        from peft import LoraConfig, get_peft_model  # type: ignore
        from peft.tuners.lora import initialize_lora_eva_weights  # type: ignore
        from peft.utils.save_and_load import get_peft_model_state_dict  # type: ignore
        return LoraConfig, get_peft_model, initialize_lora_eva_weights, get_peft_model_state_dict
    except Exception:
        import os
        import sys
        third_party = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "third_party", "peft", "src"))
        if third_party not in sys.path:
            sys.path.append(third_party)
        from peft import LoraConfig, get_peft_model  # type: ignore
        from peft.tuners.lora import initialize_lora_eva_weights  # type: ignore
        from peft.utils.save_and_load import get_peft_model_state_dict  # type: ignore
        return LoraConfig, get_peft_model, initialize_lora_eva_weights, get_peft_model_state_dict


LoraConfig, get_peft_model, initialize_lora_eva_weights, get_peft_model_state_dict = _import_peft()


def _compute_last_k_indices(module: nn.Module, *, modulelist_name: str, last_k: int) -> List[int]:
    depth = 0
    if hasattr(module, modulelist_name):
        ml = getattr(module, modulelist_name)
        if isinstance(ml, nn.ModuleList):
            depth = len(ml)
    if depth == 0:
        return []
    k = max(0, min(last_k, depth))
    return list(range(depth - k, depth))


def _mark_only_lora_trainable(m: nn.Module) -> None:
    for name, p in m.named_parameters():
        is_lora = ("lora_" in name) or ("lora_magnitude_vector" in name)
        p.requires_grad = is_lora


def inject_lora_into_feature_extractor(
    feature_extractor: nn.Module,
    peft_cfg: Dict,
) -> Tuple[nn.Module, Optional[LoraConfig]]:
    """
    Inject LoRA into the underlying DINOv3 backbone contained in a DinoV3FeatureExtractor instance.

    Returns the possibly wrapped backbone and the LoraConfig used (or None if disabled).
    """
    enabled = bool(peft_cfg.get("enabled", False))
    if not enabled:
        return feature_extractor, None

    # Pull config with sensible defaults
    method = str(peft_cfg.get("method", "lora")).lower()
    if method != "lora":
        return feature_extractor, None

    r = int(peft_cfg.get("r", 8))
    lora_alpha = int(peft_cfg.get("lora_alpha", r))
    lora_dropout = float(peft_cfg.get("lora_dropout", 0.05))
    use_dora = bool(peft_cfg.get("use_dora", True))
    init_lora_weights = peft_cfg.get("init", "eva")
    last_k_blocks = int(peft_cfg.get("last_k_blocks", 6))
    layers_pattern = str(peft_cfg.get("layers_pattern", "blocks"))
    target_modules = peft_cfg.get("target_modules", ["qkv", "proj"])  # attn-only

    backbone: nn.Module = feature_extractor.backbone

    # Compute layer indices: last K of ModuleList named by layers_pattern
    layer_indices = _compute_last_k_indices(backbone, modulelist_name=layers_pattern, last_k=last_k_blocks)

    lora_config = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        use_dora=use_dora,
        target_modules=list(target_modules) if isinstance(target_modules, (list, tuple)) else target_modules,
        layers_to_transform=layer_indices if len(layer_indices) > 0 else None,
        layers_pattern=layers_pattern if len(layer_indices) > 0 else None,
        init_lora_weights=init_lora_weights,
    )

    # Wrap backbone with PEFT model (in-place on return)
    peft_model = get_peft_model(backbone, lora_config)

    # Enable training for adapters only; keep base frozen
    _mark_only_lora_trainable(peft_model)

    # Make sure feature extractor runs in training mode when needed (LoRA dropout)
    if hasattr(feature_extractor, "inference_only"):
        try:
            feature_extractor.inference_only = False
        except Exception:
            pass
    peft_model.train()

    # Re-attach
    feature_extractor.backbone = peft_model
    return feature_extractor, lora_config


def get_lora_param_list(module: nn.Module) -> List[nn.Parameter]:
    params: List[nn.Parameter] = []
    for name, p in module.named_parameters():
        if p.requires_grad and ("lora_" in name or "lora_magnitude_vector" in name):
            params.append(p)
    return params


@torch.no_grad()
def maybe_initialize_eva(
    peft_backbone: nn.Module,
    dataloader: Optional[Iterable] = None,
    *,
    enabled: bool = False,
    show_progress_bar: bool = True,
) -> None:
    """
    Optionally run EVA initialization for LoRA weights. If disabled or dataloader is None, no-op.
    """
    if not enabled or dataloader is None:
        return

    def _forward_fn(model: nn.Module, inputs):
        # inputs expected to be mapping with key 'image'
        if isinstance(inputs, dict):
            x = inputs.get("image")
        else:
            x = inputs
        # Many DINOv3 wrappers expect NCHW float images
        return model(x)

    try:
        initialize_lora_eva_weights(
            peft_backbone,
            dataloader=dataloader,
            eva_state_dict=None,
            forward_fn=_forward_fn,
            prepare_model_inputs_fn=None,
            prepare_layer_inputs_fn=None,
            adapter_name="default",
            gather_distributed_inputs=True,
            show_progress_bar=show_progress_bar,
        )
    except Exception:
        # EVA is optional; if it fails, keep default init
        pass


def export_lora_payload_if_any(backbone: nn.Module) -> Optional[Dict]:
    """
    If the backbone is a PEFT-wrapped model, return a dict containing minimal LoRA config and adapter state.
    Otherwise return None.
    """
    peft_cfg = getattr(backbone, "peft_config", None)
    if peft_cfg is None:
        return None
    try:
        # Single-adapter case under key 'default'
        adapter_name = "default"
        cfg_obj = backbone.peft_config[adapter_name]
        cfg_dict = cfg_obj.to_dict() if hasattr(cfg_obj, "to_dict") else None
        state = get_peft_model_state_dict(backbone, adapter_name=adapter_name)
        return {"config": cfg_dict, "state_dict": state}
    except Exception:
        return None


