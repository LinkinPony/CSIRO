from __future__ import annotations

import json
import os
import re
import tempfile
from typing import Dict, List, Optional, Tuple

import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from torch import nn

from src.inference.data import resolve_paths
from src.inference.ensemble import (
    load_ensemble_cache,
    normalize_ensemble_models,
    read_ensemble_cfg_obj,
    read_ensemble_enabled_flag,
    read_packaged_ensemble_manifest_if_exists,
    resolve_cache_dir,
    save_ensemble_cache,
)
from src.inference.mp import (
    mp_attach_flags,
    mp_get_devices,
    mp_patch_dinov3_methods,
    mp_prepare_vit7b_backbone_two_gpu,
    mp_resolve_dtype,
)
from src.inference.paths import (
    find_dino_weights_in_dir,
    resolve_path_best_effort,
    resolve_version_head_base,
    resolve_version_train_yaml,
    safe_slug,
)
from src.inference.predict import (
    DinoV3FeatureExtractor,
    extract_features_for_images,
    predict_from_features,
    predict_main_and_ratio_dpt,
    predict_main_and_ratio_fpn,
    predict_main_and_ratio_vitdet,
    predict_main_and_ratio_global_multilayer,
    predict_main_and_ratio_patch_mode,
)
from src.inference.settings import InferenceSettings
from src.inference.torch_load import load_head_state, torch_load_cpu
from src.models.dpt_scalar_head import DPTHeadConfig, DPTScalarHead
from src.models.head_builder import MultiLayerHeadExport, build_head_layer
from src.models.peft_integration import _import_peft
from src.models.spatial_fpn import FPNHeadConfig, FPNScalarHead
from src.models.vitdet_head import ViTDetHeadConfig, ViTDetMultiLayerScalarHead, ViTDetScalarHead


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


def parse_image_size(value) -> Tuple[int, int]:
    """
    Accept int (square) or [width, height]; return (height, width).
    """
    try:
        if isinstance(value, (list, tuple)) and len(value) == 2:
            w, h = int(value[0]), int(value[1])
            return (int(h), int(w))
        v = int(value)
        return (v, v)
    except Exception:
        v = int(value)
        return (v, v)


def _round_to_multiple_int(v: float, multiple: int) -> int:
    """
    Round a value to the nearest positive integer multiple (min = multiple).

    Used for multi-scale TTA so that ViT patch grids remain aligned (DINOv3 uses 16x16 patches).
    """
    m = int(multiple)
    if m <= 0:
        try:
            return max(1, int(round(float(v))))
        except Exception:
            return 1
    try:
        x = float(v)
    except Exception:
        x = float(m)
    out = int(round(x / float(m))) * m
    if out < m:
        out = m
    return int(out)


def _resolve_tta_views(
    settings: InferenceSettings,
    *,
    base_image_size: Tuple[int, int],
    patch_multiple: int = 16,
) -> List[Tuple[Tuple[int, int], bool, bool]]:
    """
    Resolve a list of TTA views as (image_size_hw, hflip, vflip).

    - Always includes the base (scale=1.0, no flip) view.
    - When enabled, expands to: scales x {no-flip, hflip?}.
    - Scaled sizes are rounded to multiples of `patch_multiple` for ViT patch alignment.
    """
    base_h, base_w = int(base_image_size[0]), int(base_image_size[1])

    enabled = bool(getattr(settings, "tta_enabled", False))
    hflip_enabled = bool(getattr(settings, "tta_hflip", True))
    vflip_enabled = bool(getattr(settings, "tta_vflip", False))
    scales_raw = getattr(settings, "tta_scales", (1.0,))

    if not enabled:
        # Keep return type consistent: (image_size_hw, hflip, vflip)
        return [((base_h, base_w), False, False)]

    # Sanitize scales (keep order, ensure 1.0 exists).
    scales: List[float] = []
    try:
        for s in (scales_raw or (1.0,)):
            try:
                sf = float(s)
            except Exception:
                continue
            if not (sf > 0.0):
                continue
            scales.append(sf)
    except Exception:
        scales = [1.0]
    if not scales:
        scales = [1.0]
    if all(abs(sf - 1.0) > 1e-8 for sf in scales):
        scales.append(1.0)

    # Flip combinations:
    # - always include identity (False, False)
    # - include hflip if enabled
    # - include vflip if enabled
    # - include hvflip if both enabled (dihedral closure for flips)
    flip_pairs: List[Tuple[bool, bool]] = [(False, False)]
    if hflip_enabled:
        flip_pairs.append((True, False))
    if vflip_enabled:
        flip_pairs.append((False, True))
    if hflip_enabled and vflip_enabled:
        flip_pairs.append((True, True))

    out: List[Tuple[Tuple[int, int], bool, bool]] = []
    seen = set()
    for sf in scales:
        h = _round_to_multiple_int(float(base_h) * float(sf), patch_multiple)
        w = _round_to_multiple_int(float(base_w) * float(sf), patch_multiple)
        for hf, vf in flip_pairs:
            key = (int(h), int(w), bool(hf), bool(vf))
            if key in seen:
                continue
            seen.add(key)
            out.append(((int(h), int(w)), bool(hf), bool(vf)))
    return out


def discover_head_weight_paths(path: str) -> List[str]:
    """
    Accept single-file or directory containing a preferred single head or per-fold heads.
    """
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
        pts = [os.path.join(path, n) for n in os.listdir(path) if n.endswith(".pt")]
        pts.sort()
        if pts:
            return pts
    raise FileNotFoundError(f"Cannot find head weights at: {path}")


def resolve_dataset_area_m2(cfg: Dict) -> float:
    """
    Resolve dataset area (m^2) from config for unit conversion g <-> g/m^2.
    """
    ds_name = str(cfg.get("data", {}).get("dataset", "csiro"))
    ds_map = dict(cfg.get("data", {}).get("datasets", {}))
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
    return area_m2


def resolve_dino_weights_path_for_model(
    project_dir: str,
    *,
    backbone_name: str,
    cfg: Optional[Dict] = None,
    model_cfg: Optional[dict] = None,
    global_dino_weights: Optional[str] = None,
) -> str:
    """
    Resolve backbone weights path for a model.

    Priority:
      1) model_cfg['dino_weights_pt' | 'dino_weights' | 'backbone_weights']
      2) cfg['model']['weights_path'] (if it exists on disk)
      3) best-effort glob search under <PROJECT_DIR>/dinov3_weights/ (by backbone_name)
      4) global_dino_weights (if it exists on disk)
    """
    model_cfg = model_cfg or {}
    cfg = cfg or {}

    # 1) Explicit override in ensemble config
    for k in ("dino_weights_pt", "dino_weights", "backbone_weights", "backbone_weights_pt"):
        v = model_cfg.get(k, None)
        if isinstance(v, str) and v.strip():
            p = resolve_path_best_effort(project_dir, v)
            if os.path.isfile(p):
                return p
            if os.path.isdir(p):
                found = find_dino_weights_in_dir(p, backbone_name)
                if found:
                    return found

    # 2) weights_path from the model's YAML config (only if it exists)
    try:
        v = cfg.get("model", {}).get("weights_path", None)
        if isinstance(v, str) and v.strip():
            p = resolve_path_best_effort(project_dir, v)
            if os.path.isfile(p):
                return p
            if os.path.isdir(p):
                found = find_dino_weights_in_dir(p, backbone_name)
                if found:
                    return found
    except Exception:
        pass

    # 3) Try to locate weights by backbone name under dinov3_weights/
    dinodir = os.path.join(project_dir, "dinov3_weights")
    found = find_dino_weights_in_dir(dinodir, backbone_name)
    if found:
        return found

    # 4) Global fallback (user-editable at entrypoint).
    try:
        if isinstance(global_dino_weights, str) and global_dino_weights.strip():
            p = resolve_path_best_effort(project_dir, global_dino_weights)
            if os.path.isdir(p):
                found = find_dino_weights_in_dir(p, backbone_name)
                if found:
                    return found
            if os.path.isfile(p):
                # Safety: avoid silently using a mismatched backbone weights file for multi-backbone ensembles.
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
                            f"Global DINO weights appear to mismatch backbone '{backbone_name}': {p}. "
                            f"Expected filename to contain '{token}'. "
                            "Provide the correct weights under dinov3_weights/ or set per-model dino_weights_pt."
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


def _load_zscore_json_for_head(head_pt_path: str) -> Optional[dict]:
    """
    Try to locate z_score.json for a given head.
    Priority:
      1) Same directory as head .pt (e.g., weights/head/fold_i/z_score.json)
      2) Parent of head directory (e.g., weights/head/z_score.json)
      3) Parent of head parent (e.g., weights/z_score.json)
    """
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


def infer_components_5d_for_model(
    *,
    settings: InferenceSettings,
    project_dir: str,
    cfg: Dict,
    head_base: str,
    dino_weights_pt_path: str,
    dataset_root: str,
    image_paths: List[str],
    prefer_packaged_head_manifest: bool = True,
) -> Tuple[List[str], torch.Tensor, dict]:
    """
    Run inference for a *single model* (one backbone + one or more head weights) and return:
      - rels_in_order: list of image paths (same order as image_paths argument)
      - comps_5d_g:    Tensor (N, 5) in grams, order:
                       [Dry_Clover_g, Dry_Dead_g, Dry_Green_g, GDM_g, Dry_Total_g]
      - meta:          small metadata dict (for debugging / cache introspection)
    """
    # --- Data settings (transforms/targets come from this model's config) ---
    image_size = parse_image_size(cfg["data"]["image_size"])
    mean = list(cfg["data"]["normalization"]["mean"])
    std = list(cfg["data"]["normalization"]["std"])
    target_bases = list(cfg["data"]["target_order"])

    # Inference batch size is configured at entrypoint (not via YAML).
    try:
        batch_size = int(settings.infer_batch_size)
    except Exception:
        batch_size = 1
    batch_size = max(1, batch_size)
    num_workers = int(cfg["data"].get("num_workers", 4))

    # Dataset area (m^2) to convert g/m^2 to grams
    area_m2 = float(resolve_dataset_area_m2(cfg))

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
    except Exception as e:
        raise ImportError(
            "dinov3 is not available locally. Ensure DINOV3_PATH points to third_party/dinov3/dinov3."
        ) from e

    # --- Head weights discovery ---
    head_base_abs = resolve_path_best_effort(project_dir, head_base)
    if not head_base_abs:
        raise FileNotFoundError("head_base is empty for model inference.")

    head_entries: Optional[Tuple[List[Tuple[str, float]], str]] = None
    if prefer_packaged_head_manifest:
        try:
            head_entries = read_packaged_ensemble_manifest_if_exists(head_base_abs)
        except Exception:
            head_entries = None

    if head_entries is not None:
        entries, _aggregation = head_entries
        head_weight_paths = [p for (p, _w) in entries]
        weight_map = {p: float(w) for (p, w) in entries}
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
    dino_weights_abs = resolve_path_best_effort(project_dir, dino_weights_pt_path)
    if not (dino_weights_abs and os.path.isfile(dino_weights_abs)):
        raise FileNotFoundError(f"DINO weights not found: {dino_weights_pt_path}")
    dino_state = torch_load_cpu(dino_weights_abs, mmap=True, weights_only=True)
    if isinstance(dino_state, dict) and "state_dict" in dino_state:
        dino_state = dino_state["state_dict"]

    # Accumulate per-head 5D components in the common rel order.
    rels_in_order_ref: Optional[List[str]] = None
    comps_sum: Optional[torch.Tensor] = None
    weight_sum: float = 0.0

    for head_i, head_pt in enumerate(head_weight_paths):
        # NOTE: keep head ensemble weight separate from any internal per-layer fusion weights.
        head_w = float(weight_map.get(head_pt, 1.0))
        if not (head_w > 0.0):
            continue
        # Optional per-layer outputs (available for some multi-layer heads, e.g. ViTDet multi-layer).
        preds_main_layers: Optional[torch.Tensor] = None
        preds_ratio_layers: Optional[torch.Tensor] = None

        # Load head state/meta (and optional PEFT payload)
        state, meta, peft_payload = load_head_state(head_pt)
        if not isinstance(meta, dict):
            meta = {}

        head_num_main = int(meta.get("num_outputs_main", meta.get("num_outputs", num_outputs_main_default)))
        head_num_ratio = int(meta.get("num_outputs_ratio", num_outputs_ratio_default))
        head_total = int(meta.get("head_total_outputs", head_num_main + head_num_ratio))
        head_hidden_dims = list(meta.get("head_hidden_dims", first_meta.get("head_hidden_dims", list(cfg["model"]["head"].get("hidden_dims", [512, 256])))))
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
            bool(settings.use_2gpu_model_parallel_for_vit7b)
            and (backbone_name in ("dinov3_vit7b16", "dinov3_vit7b", "vit7b16", "vit7b"))
            and (mp_get_devices() is not None)
        )
        if use_mp:
            dev0, dev1 = mp_get_devices()  # type: ignore[assignment]
            split_idx = int(settings.vit7b_mp_split_idx)
            mp_dtype = mp_resolve_dtype(settings.vit7b_mp_dtype)
            backbone = _make_backbone(pretrained=False, device="meta")
            mp_prepare_vit7b_backbone_two_gpu(
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
            mp_patch_dinov3_methods(backbone, split_idx=split_idx, device0=dev0, device1=dev1)
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
                        _ = _import_peft()
                        from peft.config import PeftConfig  # type: ignore
                        from peft.mapping_func import get_peft_model  # type: ignore
                        from peft.utils.save_and_load import set_peft_model_state_dict  # type: ignore
                    peft_config = PeftConfig.from_peft_type(**peft_cfg_dict)
                    backbone = get_peft_model(backbone, peft_config)
                    set_peft_model_state_dict(backbone, peft_state, adapter_name="default")
                    backbone.eval()
                    if use_mp:
                        try:
                            mp_attach_flags(backbone, dev0, dev1, split_idx)  # type: ignore[arg-type]
                        except Exception:
                            pass
        except Exception as _e:
            print(f"[WARN] PEFT injection skipped for {head_pt}: {_e}")

        # Build head module according to head meta
        head_type_meta = str(meta.get("head_type", first_meta.get("head_type", "mlp"))).strip().lower()
        use_patch_reg3_head = bool(meta.get("use_patch_reg3", False))
        use_cls_token_head = bool(meta.get("use_cls_token", use_cls_token_default))
        use_layerwise_heads_head = bool(meta.get("use_layerwise_heads", use_layerwise_heads_default))
        backbone_layer_indices_head = list(meta.get("backbone_layer_indices", backbone_layer_indices_default))
        use_separate_bottlenecks_head = bool(meta.get("use_separate_bottlenecks", use_separate_bottlenecks_default))
        head_is_ratio = bool(head_num_ratio > 0 and head_total == (head_num_main + head_num_ratio))

        # Multi-layer fusion mode (mean or learned). For MLP multi-layer inference the learned
        # fusion logits (if any) are exported in head meta as `mlp_layer_logits`.
        fusion_mode_meta = str(
            meta.get(
                "backbone_layers_fusion",
                meta.get("layer_fusion", first_meta.get("backbone_layers_fusion", "mean")),
            )
            or "mean"
        ).strip().lower()
        layer_weights_head: Optional[torch.Tensor] = None
        if use_layerwise_heads_head and fusion_mode_meta == "learned":
            try:
                logits_meta = meta.get("mlp_layer_logits", None)
                if isinstance(logits_meta, (list, tuple)) and len(logits_meta) == len(backbone_layer_indices_head):
                    logits_t = torch.tensor([float(x) for x in logits_meta], dtype=torch.float32)
                    layer_weights_head = torch.softmax(logits_t, dim=0)
            except Exception:
                layer_weights_head = None

        # Ratio head coupling mode (enum). Prefer `ratio_head_mode` when present, but
        # fall back to legacy boolean flags for older exported heads.
        try:
            from src.models.regressor.heads.ratio_mode import resolve_ratio_head_mode, flags_from_ratio_head_mode

            ratio_head_mode_eff = resolve_ratio_head_mode(
                meta.get("ratio_head_mode", first_meta.get("ratio_head_mode", None)),
                separate_ratio_head=meta.get("separate_ratio_head", first_meta.get("separate_ratio_head", None)),
                separate_ratio_spatial_head=meta.get(
                    "separate_ratio_spatial_head", first_meta.get("separate_ratio_spatial_head", None)
                ),
            )
            separate_ratio_head_eff, separate_ratio_spatial_eff = flags_from_ratio_head_mode(ratio_head_mode_eff)
        except Exception:
            ratio_head_mode_eff = "shared"
            separate_ratio_head_eff, separate_ratio_spatial_eff = False, False

        if head_type_meta == "fpn":
            fpn_dim_meta = int(meta.get("fpn_dim", first_meta.get("fpn_dim", int(cfg["model"]["head"].get("fpn_dim", 256)))))
            fpn_levels_meta = int(meta.get("fpn_num_levels", first_meta.get("fpn_num_levels", int(cfg["model"]["head"].get("fpn_num_levels", 3)))))
            fpn_patch_size_meta = int(meta.get("fpn_patch_size", first_meta.get("fpn_patch_size", int(cfg["model"]["head"].get("fpn_patch_size", 16)))))
            fpn_reverse_level_order_meta = bool(meta.get("fpn_reverse_level_order", first_meta.get("fpn_reverse_level_order", bool(cfg["model"]["head"].get("fpn_reverse_level_order", True)))))
            enable_ndvi_meta = bool(meta.get("enable_ndvi", False))
            num_layers_eff = max(1, len(backbone_layer_indices_head)) if use_layerwise_heads_head else 1
            head_module = FPNScalarHead(
                FPNHeadConfig(
                    embedding_dim=head_embedding_dim,
                    fpn_dim=fpn_dim_meta,
                    num_levels=fpn_levels_meta,
                    num_layers=num_layers_eff,
                    use_separate_bottlenecks=use_separate_bottlenecks_head,
                    head_hidden_dims=head_hidden_dims,
                    head_activation=head_activation,
                    dropout=head_dropout,
                    num_outputs_main=head_num_main,
                    num_outputs_ratio=head_num_ratio if head_is_ratio else 0,
                    enable_ndvi=enable_ndvi_meta,
                    separate_ratio_head=bool(separate_ratio_head_eff),
                    separate_ratio_spatial_head=bool(separate_ratio_spatial_eff),
                    patch_size=fpn_patch_size_meta,
                    reverse_level_order=fpn_reverse_level_order_meta,
                )
            )
        elif head_type_meta == "vitdet":
            vitdet_dim_meta = int(meta.get("vitdet_dim", first_meta.get("vitdet_dim", int(cfg["model"]["head"].get("vitdet_dim", 256)))))
            vitdet_patch_size_meta = int(
                meta.get(
                    "vitdet_patch_size",
                    first_meta.get(
                        "vitdet_patch_size",
                        int(cfg["model"]["head"].get("vitdet_patch_size", cfg["model"]["head"].get("fpn_patch_size", 16))),
                    ),
                )
            )
            # Default scale_factors: [2.0, 1.0, 0.5] (repo default; less memory than ViTDet's 4.0)
            sf_default = list(cfg["model"]["head"].get("vitdet_scale_factors", [2.0, 1.0, 0.5]))
            vitdet_scale_factors_meta = list(
                meta.get(
                    "vitdet_scale_factors",
                    first_meta.get("vitdet_scale_factors", sf_default),
                )
            )
            enable_ndvi_meta = bool(meta.get("enable_ndvi", False))
            num_layers_eff = max(1, len(backbone_layer_indices_head)) if use_layerwise_heads_head else 1
            vitdet_cfg = ViTDetHeadConfig(
                embedding_dim=head_embedding_dim,
                vitdet_dim=vitdet_dim_meta,
                scale_factors=vitdet_scale_factors_meta,
                patch_size=vitdet_patch_size_meta,
                num_outputs_main=head_num_main,
                num_outputs_ratio=head_num_ratio if head_is_ratio else 0,
                enable_ndvi=enable_ndvi_meta,
                separate_ratio_head=bool(separate_ratio_head_eff),
                separate_ratio_spatial_head=bool(separate_ratio_spatial_eff),
                head_hidden_dims=head_hidden_dims,
                head_activation=head_activation,
                dropout=head_dropout,
            )
            if use_layerwise_heads_head:
                head_module = ViTDetMultiLayerScalarHead(  # type: ignore[assignment]
                    vitdet_cfg, num_layers=num_layers_eff, layer_fusion=fusion_mode_meta
                )
            else:
                head_module = ViTDetScalarHead(vitdet_cfg)  # type: ignore[assignment]
        elif head_type_meta == "dpt":
            dpt_features_meta = int(meta.get("dpt_features", first_meta.get("dpt_features", int(cfg["model"]["head"].get("dpt_features", 256)))))
            dpt_patch_size_meta = int(meta.get("dpt_patch_size", first_meta.get("dpt_patch_size", int(cfg["model"]["head"].get("dpt_patch_size", cfg["model"]["head"].get("fpn_patch_size", 16))))))
            dpt_readout_meta = str(meta.get("dpt_readout", first_meta.get("dpt_readout", str(cfg["model"]["head"].get("dpt_readout", "ignore"))))).strip().lower()
            enable_ndvi_meta = bool(meta.get("enable_ndvi", False))
            num_layers_eff = max(1, len(backbone_layer_indices_head)) if use_layerwise_heads_head else 1
            head_module = DPTScalarHead(
                DPTHeadConfig(
                    embedding_dim=head_embedding_dim,
                    features=dpt_features_meta,
                    patch_size=dpt_patch_size_meta,
                    readout=dpt_readout_meta,
                    num_layers=num_layers_eff,
                    num_outputs_main=head_num_main,
                    num_outputs_ratio=head_num_ratio if head_is_ratio else 0,
                    enable_ndvi=enable_ndvi_meta,
                    separate_ratio_head=bool(separate_ratio_head_eff),
                    separate_ratio_spatial_head=bool(separate_ratio_spatial_eff),
                    head_hidden_dims=head_hidden_dims,
                    head_activation=head_activation,
                    dropout=head_dropout,
                )
            )
        elif use_layerwise_heads_head and use_separate_bottlenecks_head:
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
                num_outputs=effective_outputs if head_is_ratio else (head_num_main if not use_layerwise_heads_head else head_num_main * max(1, len(backbone_layer_indices_head))),
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
        mc_enabled = bool(getattr(settings, "mc_dropout_enabled", False)) and int(getattr(settings, "mc_dropout_samples", 1)) > 1
        if mc_enabled:
            # Optional deterministic seeding (per-head) for reproducible MC dropout.
            try:
                seed0 = int(getattr(settings, "mc_dropout_seed", -1))
            except Exception:
                seed0 = -1
            if seed0 >= 0:
                seed_eff = int(seed0) + int(head_i)
                torch.manual_seed(seed_eff)
                try:
                    if torch.cuda.is_available():
                        torch.cuda.manual_seed_all(seed_eff)
                except Exception:
                    pass

        # Optional TTA: run multiple deterministic views and average raw head outputs.
        #
        # Default (when enabled): {identity, hflip}. Optional multi-scale resize is supported
        # via settings.tta_scales.
        preds_main_layers: Optional[torch.Tensor] = None
        preds_ratio_layers: Optional[torch.Tensor] = None

        tta_views = _resolve_tta_views(settings, base_image_size=image_size, patch_multiple=16)
        rels_view_ref: Optional[List[str]] = None
        preds_main_sum: Optional[torch.Tensor] = None
        preds_ratio_sum: Optional[torch.Tensor] = None
        preds_main_layers_sum: Optional[torch.Tensor] = None
        preds_ratio_layers_sum: Optional[torch.Tensor] = None
        n_views: int = 0

        for _view_idx, (image_size_view, hflip_view, vflip_view) in enumerate(tta_views):
            preds_main_layers_v: Optional[torch.Tensor] = None
            preds_ratio_layers_v: Optional[torch.Tensor] = None

            if head_type_meta == "fpn":
                rels_v, preds_main_v, preds_ratio_v = predict_main_and_ratio_fpn(
                    backbone=backbone,
                    head=head_module,
                    dataset_root=dataset_root,
                    image_paths=image_paths,
                    image_size=image_size_view,
                    mean=mean,
                    std=std,
                    batch_size=batch_size,
                    num_workers=num_workers,
                    head_num_main=head_num_main,
                    head_num_ratio=head_num_ratio if head_is_ratio else 0,
                    use_layerwise_heads=use_layerwise_heads_head,
                    layer_indices=backbone_layer_indices_head if use_layerwise_heads_head else None,
                    mc_dropout_enabled=mc_enabled,
                    mc_dropout_samples=int(getattr(settings, "mc_dropout_samples", 1)),
                    hflip=bool(hflip_view),
                    vflip=bool(vflip_view),
                )
            elif head_type_meta == "vitdet":
                (
                    rels_v,
                    preds_main_v,
                    preds_ratio_v,
                    preds_main_layers_v,
                    preds_ratio_layers_v,
                ) = predict_main_and_ratio_vitdet(
                    backbone=backbone,
                    head=head_module,
                    dataset_root=dataset_root,
                    image_paths=image_paths,
                    image_size=image_size_view,
                    mean=mean,
                    std=std,
                    batch_size=batch_size,
                    num_workers=num_workers,
                    head_num_main=head_num_main,
                    head_num_ratio=head_num_ratio if head_is_ratio else 0,
                    use_layerwise_heads=use_layerwise_heads_head,
                    layer_indices=backbone_layer_indices_head if use_layerwise_heads_head else None,
                    mc_dropout_enabled=mc_enabled,
                    mc_dropout_samples=int(getattr(settings, "mc_dropout_samples", 1)),
                    hflip=bool(hflip_view),
                    vflip=bool(vflip_view),
                )
            elif head_type_meta == "dpt":
                rels_v, preds_main_v, preds_ratio_v = predict_main_and_ratio_dpt(
                    backbone=backbone,
                    head=head_module,
                    dataset_root=dataset_root,
                    image_paths=image_paths,
                    image_size=image_size_view,
                    mean=mean,
                    std=std,
                    batch_size=batch_size,
                    num_workers=num_workers,
                    head_num_main=head_num_main,
                    head_num_ratio=head_num_ratio if head_is_ratio else 0,
                    use_layerwise_heads=use_layerwise_heads_head,
                    layer_indices=backbone_layer_indices_head if use_layerwise_heads_head else None,
                    mc_dropout_enabled=mc_enabled,
                    mc_dropout_samples=int(getattr(settings, "mc_dropout_samples", 1)),
                    hflip=bool(hflip_view),
                    vflip=bool(vflip_view),
                )
            elif use_patch_reg3_head:
                rels_v, preds_main_v, preds_ratio_v = predict_main_and_ratio_patch_mode(
                    backbone=backbone,
                    head=head_module,
                    dataset_root=dataset_root,
                    image_paths=image_paths,
                    image_size=image_size_view,
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
                    layer_weights=layer_weights_head,
                    mc_dropout_enabled=mc_enabled,
                    mc_dropout_samples=int(getattr(settings, "mc_dropout_samples", 1)),
                    hflip=bool(hflip_view),
                    vflip=bool(vflip_view),
                )
            else:
                if use_layerwise_heads_head and len(backbone_layer_indices_head) > 0:
                    rels_v, preds_main_v, preds_ratio_v = predict_main_and_ratio_global_multilayer(
                        backbone=backbone,
                        head=head_module,
                        dataset_root=dataset_root,
                        image_paths=image_paths,
                        image_size=image_size_view,
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
                        layer_weights=layer_weights_head,
                        mc_dropout_enabled=mc_enabled,
                        mc_dropout_samples=int(getattr(settings, "mc_dropout_samples", 1)),
                        hflip=bool(hflip_view),
                        vflip=bool(vflip_view),
                    )
                else:
                    feature_extractor = DinoV3FeatureExtractor(backbone)
                    rels_v, features_cpu = extract_features_for_images(
                        feature_extractor=feature_extractor,
                        dataset_root=dataset_root,
                        image_paths=image_paths,
                        image_size=image_size_view,
                        mean=mean,
                        std=std,
                        batch_size=batch_size,
                        num_workers=num_workers,
                        use_cls_token=use_cls_token_head,
                        hflip=bool(hflip_view),
                        vflip=bool(vflip_view),
                    )
                    preds_main_v, preds_ratio_v = predict_from_features(
                        features_cpu=features_cpu,
                        head=head_module,
                        batch_size=batch_size,
                        head_num_main=head_num_main,
                        head_num_ratio=head_num_ratio if head_is_ratio else 0,
                        head_total=head_total,
                        use_layerwise_heads=False,
                        num_layers=1,
                        mc_dropout_enabled=mc_enabled,
                        mc_dropout_samples=int(getattr(settings, "mc_dropout_samples", 1)),
                    )

            if rels_view_ref is None:
                rels_view_ref = rels_v
            elif rels_view_ref != rels_v:
                raise RuntimeError("Image order mismatch across TTA views for a head; aborting.")

            preds_main_sum = preds_main_v if preds_main_sum is None else (preds_main_sum + preds_main_v)
            if head_is_ratio and head_num_ratio > 0 and preds_ratio_v is not None:
                preds_ratio_sum = preds_ratio_v if preds_ratio_sum is None else (preds_ratio_sum + preds_ratio_v)

            if preds_main_layers_v is not None:
                preds_main_layers_sum = (
                    preds_main_layers_v
                    if preds_main_layers_sum is None
                    else (preds_main_layers_sum + preds_main_layers_v)
                )
            if preds_ratio_layers_v is not None:
                preds_ratio_layers_sum = (
                    preds_ratio_layers_v
                    if preds_ratio_layers_sum is None
                    else (preds_ratio_layers_sum + preds_ratio_layers_v)
                )

            n_views += 1

        if rels_view_ref is None or preds_main_sum is None or n_views <= 0:
            raise RuntimeError("TTA produced no predictions for this head.")

        rels_in_order = rels_view_ref
        preds_main = preds_main_sum / float(n_views)
        preds_ratio = (preds_ratio_sum / float(n_views)) if preds_ratio_sum is not None else None
        preds_main_layers = (preds_main_layers_sum / float(n_views)) if preds_main_layers_sum is not None else None
        preds_ratio_layers = (preds_ratio_layers_sum / float(n_views)) if preds_ratio_layers_sum is not None else None

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
        zscore_enabled = reg3_mean is not None and reg3_std is not None

        # Optional Softplus on main outputs (matches training behavior):
        use_output_softplus_cfg = bool(cfg.get("model", {}).get("head", {}).get("use_output_softplus", True))
        if use_output_softplus_cfg and (not log_scale_meta) and (not zscore_enabled):
            preds_main = F.softplus(preds_main)
            if preds_main_layers is not None:
                preds_main_layers = F.softplus(preds_main_layers)

        if zscore_enabled:
            preds_main = preds_main * reg3_std[:head_num_main] + reg3_mean[:head_num_main]  # type: ignore[index]
            if preds_main_layers is not None:
                m = reg3_mean[:head_num_main].view(1, 1, -1)  # type: ignore[index]
                s = reg3_std[:head_num_main].view(1, 1, -1)  # type: ignore[index]
                preds_main_layers = preds_main_layers * s + m
        if log_scale_meta:
            preds_main = torch.expm1(preds_main).clamp_min(0.0)
            if preds_main_layers is not None:
                preds_main_layers = torch.expm1(preds_main_layers).clamp_min(0.0)

        # Convert from g/m^2 to grams
        preds_main_g = preds_main * float(area_m2)
        preds_main_layers_g = (preds_main_layers * float(area_m2)) if preds_main_layers is not None else None

        # Build this head's 5D components in grams
        N = preds_main_g.shape[0]
        if head_is_ratio and preds_ratio is not None and head_num_main >= 1 and head_num_ratio >= 1:
            # Prefer constraint-aware fusion when we have multi-layer per-layer outputs:
            # c_l = softmax(ratio_l) * total_l, then average c across layers.
            comps_5d = None
            try:
                if (
                    preds_main_layers_g is not None
                    and (preds_ratio_layers is not None)
                    and preds_main_layers_g.dim() == 3
                    and preds_ratio_layers.dim() == 3
                    and preds_main_layers_g.shape[0] == N
                    and preds_main_layers_g.shape[1] == preds_ratio_layers.shape[1]
                ):
                    L = int(preds_main_layers_g.shape[1])
                    total_layers_g = preds_main_layers_g[:, :, 0].clamp_min(0.0)  # (N, L)
                    p_layers = F.softmax(preds_ratio_layers, dim=-1)  # (N, L, 3)
                    comp_layers = p_layers * total_layers_g.unsqueeze(-1)  # (N, L, 3)
                    # Optional learned fusion weights from the head module (fallback to uniform).
                    layer_w: Optional[torch.Tensor] = None
                    try:
                        if head_module is not None and hasattr(head_module, "get_layer_weights"):
                            w_full = head_module.get_layer_weights(device=comp_layers.device, dtype=comp_layers.dtype)  # type: ignore[attr-defined]
                            if isinstance(w_full, torch.Tensor) and int(w_full.numel()) == L:
                                layer_w = w_full.detach()
                    except Exception:
                        layer_w = None
                    if layer_w is None:
                        comp_bar = comp_layers.mean(dim=1)  # (N,3)
                    else:
                        layer_w = layer_w / layer_w.sum().clamp_min(1e-8)
                        comp_bar = (comp_layers * layer_w.view(1, L, 1)).sum(dim=1)
                    comp_clover = comp_bar[:, 0]
                    comp_dead = comp_bar[:, 1]
                    comp_green = comp_bar[:, 2]
                    total_g = comp_bar.sum(dim=-1)
                    comp_gdm = comp_clover + comp_green
                    comps_5d = torch.stack([comp_clover, comp_dead, comp_green, comp_gdm, total_g], dim=-1)
            except Exception:
                comps_5d = None

            if comps_5d is None:
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
            comps_sum = comps_5d * head_w
        else:
            comps_sum = comps_sum + (comps_5d * head_w)
        weight_sum += head_w

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


def _write_submission(df: pd.DataFrame, image_to_components: Dict[str, Dict[str, float]], output_path: str) -> None:
    rows = []
    for _, r in df.iterrows():
        sample_id = str(r["sample_id"])
        rel_path = str(r["image_path"])
        target_name = str(r["target_name"])
        comps = image_to_components.get(rel_path, {})
        value = comps.get(target_name, 0.0)
        value = max(0.0, float(value))
        rows.append((sample_id, value))

    out_dir = os.path.dirname(os.path.abspath(output_path))
    if out_dir and not os.path.isdir(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("sample_id,target\n")
        for sample_id, value in rows:
            f.write(f"{sample_id},{value}\n")


def run(settings: InferenceSettings) -> None:
    """
    Entry point for offline inference + submission generation.
    """
    project_dir_abs = os.path.abspath(settings.project_dir) if settings.project_dir else ""
    configs_dir = os.path.join(project_dir_abs, "configs")
    src_dir = os.path.join(project_dir_abs, "src")
    if not (project_dir_abs and os.path.isdir(project_dir_abs)):
        raise RuntimeError("PROJECT_DIR must point to the repository root containing `configs/` and `src/`.")
    if not os.path.isdir(configs_dir):
        raise RuntimeError(f"configs/ not found under PROJECT_DIR: {configs_dir}")
    if not os.path.isdir(src_dir):
        raise RuntimeError(f"src/ not found under PROJECT_DIR: {src_dir}")

    cfg = load_config(project_dir_abs)

    # Read test.csv
    dataset_root, test_csv = resolve_paths(settings.input_path)
    df = pd.read_csv(test_csv)
    if not {"sample_id", "image_path", "target_name"}.issubset(df.columns):
        raise ValueError("test.csv must contain columns: sample_id, image_path, target_name")
    unique_image_paths = df["image_path"].astype(str).unique().tolist()

    # ==========================================================
    # Multi-model ensemble path (supports mixing ViT/backbone types)
    # ==========================================================
    ensemble_models = normalize_ensemble_models(project_dir_abs)
    if len(ensemble_models) > 0:
        ensemble_obj = read_ensemble_cfg_obj(project_dir_abs) or {}
        cache_dir_cfg = ensemble_obj.get("cache_dir", None)
        cache_dir = resolve_cache_dir(project_dir_abs, cache_dir_cfg)

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
                    cfg_path = resolve_path_best_effort(project_dir_abs, v.strip())
                    break
            if cfg_path is None:
                if isinstance(version, (str, int, float)) and str(version).strip():
                    cfg_path = resolve_version_train_yaml(project_dir_abs, str(version).strip())
                else:
                    cfg_path = os.path.join(project_dir_abs, "configs", "train.yaml")
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
                    head_base = resolve_version_head_base(project_dir_abs, str(version).strip())
                else:
                    head_base = settings.head_weights_pt_path

            # Resolve backbone weights path
            backbone_name_eff = str(cfg_model.get("model", {}).get("backbone", "") or "").strip()
            dino_weights_path = resolve_dino_weights_path_for_model(
                project_dir_abs,
                backbone_name=backbone_name_eff,
                cfg=cfg_model,
                model_cfg=m,
                global_dino_weights=settings.dino_weights_pt_path,
            )
            print(f"[ENSEMBLE] Model '{model_id}': backbone={backbone_name_eff}, dino_weights={dino_weights_path}")

            # Cache file path
            cache_name = str(m.get("cache_name", "") or "").strip()
            if not cache_name:
                cache_name = safe_slug(str(version).strip() if version is not None else model_id)
            cache_path = os.path.join(cache_dir, f"{cache_name}.pt")

            print(f"[ENSEMBLE] Running model '{model_id}' (version={version}) -> cache: {cache_path}")
            rels_in_order, comps_5d_g, meta = infer_components_5d_for_model(
                settings=settings,
                project_dir=project_dir_abs,
                cfg=cfg_model,
                head_base=head_base,
                dino_weights_pt_path=dino_weights_path,
                dataset_root=dataset_root,
                image_paths=unique_image_paths,
                prefer_packaged_head_manifest=True,
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
            save_ensemble_cache(
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
            rels, comps, _meta = load_ensemble_cache(cache_path)
            if rels_ref is None:
                rels_ref = rels
            elif rels_ref != rels:
                raise RuntimeError("Image order mismatch across cached models; aborting ensemble.")
            comps_sum = (comps * float(w)) if comps_sum is None else (comps_sum + comps * float(w))
            w_sum += float(w)

        if rels_ref is None or comps_sum is None or not (w_sum > 0.0):
            raise RuntimeError("Failed to build an ensemble from cached predictions.")

        comps_avg = comps_sum / float(w_sum)  # (N,5)
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

        _write_submission(df, image_to_components, settings.output_submission_path)
        print(f"[ENSEMBLE] Models ensembled: {used_models}")
        print(f"Submission written to: {settings.output_submission_path}")
        return

    # ==========================================================
    # Single-model path (may still ensemble multiple head files under a directory)
    # ==========================================================
    if not settings.head_weights_pt_path:
        raise FileNotFoundError("HEAD_WEIGHTS_PT_PATH must be set to a valid head file or directory.")

    # Resolve DINO weights based on backbone name in config.
    backbone_name = str(cfg["model"]["backbone"]).strip()
    dino_weights_pt_path = ""
    try:
        if isinstance(settings.dino_weights_pt_path, str) and settings.dino_weights_pt_path.strip():
            p0 = resolve_path_best_effort(project_dir_abs, settings.dino_weights_pt_path)
            if os.path.isfile(p0):
                dino_weights_pt_path = os.path.abspath(p0)
        if not dino_weights_pt_path:
            dino_weights_pt_path = resolve_dino_weights_path_for_model(
                project_dir_abs,
                backbone_name=backbone_name,
                cfg=cfg,
                model_cfg={},
                global_dino_weights=settings.dino_weights_pt_path,
            )
    except FileNotFoundError:
        dino_weights_pt_path = ""
    if not (dino_weights_pt_path and os.path.isfile(dino_weights_pt_path)):
        raise FileNotFoundError(
            "DINO_WEIGHTS_PT_PATH must point to a valid backbone .pt file, or a directory containing the official DINOv3 weights."
        )
    print(f"[WEIGHTS] Using DINO backbone weights: {dino_weights_pt_path}")

    # Preserve original script semantics: only prefer weights/head/ensemble.json when configs/ensemble.json is enabled.
    prefer_packaged_manifest = bool(read_ensemble_enabled_flag(project_dir_abs))

    rels_in_order, comps_5d_g, _meta = infer_components_5d_for_model(
        settings=settings,
        project_dir=project_dir_abs,
        cfg=cfg,
        head_base=settings.head_weights_pt_path,
        dino_weights_pt_path=dino_weights_pt_path,
        dataset_root=dataset_root,
        image_paths=unique_image_paths,
        prefer_packaged_head_manifest=prefer_packaged_manifest,
    )

    # Build image -> component mapping
    image_to_components: Dict[str, Dict[str, float]] = {}
    for rel_path, vec in zip(rels_in_order, comps_5d_g.tolist()):
        clover_g, dead_g, green_g, gdm_g, total_g = vec
        image_to_components[str(rel_path)] = {
            "Dry_Total_g": float(total_g),
            "Dry_Clover_g": float(clover_g),
            "Dry_Dead_g": float(dead_g),
            "Dry_Green_g": float(green_g),
            "GDM_g": float(gdm_g),
        }

    _write_submission(df, image_to_components, settings.output_submission_path)
    print(f"Submission written to: {settings.output_submission_path}")


