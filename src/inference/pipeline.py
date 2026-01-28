from __future__ import annotations

import json
import os
import re
import tempfile
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from torch import nn

from src.data.csiro_pivot import read_and_pivot_csiro_train_csv
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
from src.inference.two_gpu_parallel import (
    run_two_processes_spawn,
    split_even_odd_indices,
    two_gpu_parallel_enabled,
    worker_guarded,
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
    predict_main_and_ratio_eomt,
    predict_main_and_ratio_fpn,
    predict_main_and_ratio_mamba,
    predict_main_and_ratio_vitdet,
    predict_main_and_ratio_global_multilayer,
    predict_main_and_ratio_patch_mode,
    predict_main_and_ratio_dual_branch,
)
from src.inference.settings import InferenceSettings
from src.inference.torch_load import load_head_state, torch_load_cpu
from src.models.dpt_scalar_head import DPTHeadConfig, DPTScalarHead
from src.models.eomt_injected_head import EoMTInjectedQueryHeadConfig, EoMTInjectedQueryScalarHead
from src.models.head_builder import DualBranchHeadExport, MultiLayerHeadExport, build_head_layer
from src.models.peft_integration import _import_peft
from src.models.spatial_fpn import FPNHeadConfig, FPNScalarHead
from src.models.vitdet_head import ViTDetHeadConfig, ViTDetMultiLayerScalarHead, ViTDetScalarHead
from src.models.mamba_head import MambaHeadConfig, Mamba2DScalarHead, MambaMultiLayerScalarHead


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


def _parse_head_epoch_from_filename(name: str) -> int:
    """
    Extract epoch integer from filenames like:
      - head-epoch034.pt
      - head-epoch034-val_loss0.123456.pt
    Returns -1 if not found.
    """
    try:
        m = re.search(r"head-epoch(\d+)", str(name or ""))
        return int(m.group(1)) if m else -1
    except Exception:
        return -1


def _pick_latest_pt(paths: List[str]) -> Optional[str]:
    """
    Pick a "latest" .pt file from a list, preferring larger epoch, then mtime.
    Returns None when list is empty.
    """
    if not paths:
        return None
    scored: List[Tuple[int, float, str]] = []
    for p in paths:
        if not (isinstance(p, str) and p and os.path.isfile(p)):
            continue
        try:
            epoch = _parse_head_epoch_from_filename(os.path.basename(p))
        except Exception:
            epoch = -1
        try:
            mtime = float(os.path.getmtime(p))
        except Exception:
            mtime = 0.0
        scored.append((int(epoch), float(mtime), str(p)))
    if not scored:
        return None
    scored.sort(key=lambda t: (t[0], t[1]), reverse=True)
    return scored[0][2]


def _discover_head_pt_for_run_dir(run_dir: str) -> Optional[str]:
    """
    Best-effort: locate a head weights .pt file for a single run directory.

    Supports both:
      - training outputs:  <run_dir>/head/head-epoch*.pt
      - packaged weights:  <run_dir>/infer_head.pt
    """
    if not (isinstance(run_dir, str) and run_dir and os.path.isdir(run_dir)):
        return None

    # Packaged convention: <run_dir>/infer_head.pt
    p0 = os.path.join(run_dir, "infer_head.pt")
    if os.path.isfile(p0):
        return p0

    # Some layouts may have <run_dir>/head/infer_head.pt
    p1 = os.path.join(run_dir, "head", "infer_head.pt")
    if os.path.isfile(p1):
        return p1

    # Training convention: <run_dir>/head/head-epoch*.pt
    head_dir = os.path.join(run_dir, "head")
    if os.path.isdir(head_dir):
        try:
            pts = [
                os.path.join(head_dir, n)
                for n in os.listdir(head_dir)
                if n.startswith("head-epoch") and n.endswith(".pt")
            ]
        except Exception:
            pts = []
        chosen = _pick_latest_pt(pts)
        if chosen is not None:
            return chosen

        # Fallback: any .pt under head_dir
        try:
            pts2 = [os.path.join(head_dir, n) for n in os.listdir(head_dir) if n.endswith(".pt")]
        except Exception:
            pts2 = []
        chosen = _pick_latest_pt(pts2)
        if chosen is not None:
            return chosen

    # Fallback: any .pt directly under run_dir
    try:
        pts3 = [os.path.join(run_dir, n) for n in os.listdir(run_dir) if n.endswith(".pt")]
    except Exception:
        pts3 = []
    chosen = _pick_latest_pt(pts3)
    return chosen


def _expand_kfold_train_all_model(project_dir: str, *, model_cfg: dict) -> List[dict]:
    """
    Expand a single ensemble model entry with:
      mode: "kfold_train_all"
    into multiple sub-model entries (fold_* + train_all), best-effort.

    This is used to ensemble *within a single version* while reusing all completed runs.
    """
    if not isinstance(model_cfg, dict):
        return []

    mode = str(model_cfg.get("mode", "") or "").strip().lower()
    if mode not in ("kfold_train_all", "kfold+train_all", "kfold_trainall"):
        return []

    version = model_cfg.get("version", None)
    ver = str(version).strip() if isinstance(version, (str, int, float)) else ""
    if not ver:
        print("[ENSEMBLE] mode=kfold_train_all requires a non-empty 'version'; skipping expansion.")
        return []

    # Resolve the per-version train.yaml snapshot (needed to resolve log_dir/ckpt_dir).
    cfg_path = None
    for k in ("config", "config_path", "train_yaml", "train_config"):
        v = model_cfg.get(k, None)
        if isinstance(v, str) and v.strip():
            cfg_path = resolve_path_best_effort(project_dir, v.strip())
            break
    if cfg_path is None:
        cfg_path = resolve_version_train_yaml(project_dir, ver)
    if not (isinstance(cfg_path, str) and cfg_path and os.path.isfile(cfg_path)):
        print(f"[ENSEMBLE] mode=kfold_train_all: train.yaml not found for version={ver}: {cfg_path}")
        return []

    try:
        cfg_model = load_config_file(cfg_path)
    except Exception as e:
        print(f"[ENSEMBLE] mode=kfold_train_all: failed to load config {cfg_path}: {e}")
        return []

    log_cfg = cfg_model.get("logging", {}) if isinstance(cfg_model.get("logging", {}), dict) else {}
    log_root = resolve_path_best_effort(project_dir, str(log_cfg.get("log_dir", "outputs") or "outputs"))
    ckpt_root = resolve_path_best_effort(
        project_dir, str(log_cfg.get("ckpt_dir", "outputs/checkpoints") or "outputs/checkpoints")
    )

    ckpt_ver_dir = os.path.join(str(ckpt_root), ver) if ver else str(ckpt_root)
    log_ver_dir = os.path.join(str(log_root), ver) if ver else str(log_root)

    include_cfg = model_cfg.get("include", None)
    include_cfg = include_cfg if isinstance(include_cfg, dict) else {}
    include_kfold = bool(include_cfg.get("kfold", True))
    include_train_all = bool(include_cfg.get("train_all", True))

    weights_cfg = model_cfg.get("weights", None)
    weights_cfg = weights_cfg if isinstance(weights_cfg, dict) else {}
    try:
        base_model_weight = float(model_cfg.get("weight", 1.0))
    except Exception:
        base_model_weight = 1.0
    try:
        fold_w = float(weights_cfg.get("fold", 1.0))
    except Exception:
        fold_w = 1.0
    try:
        train_all_w = float(weights_cfg.get("train_all", 1.0))
    except Exception:
        train_all_w = 1.0
    per_fold_w = weights_cfg.get("per_fold", None)
    per_fold_w = per_fold_w if isinstance(per_fold_w, dict) else {}

    members: List[Tuple[str, str, float]] = []

    # 1) k-fold members
    if include_kfold:
        fold_dirs: List[Tuple[str, str]] = []
        # Prefer training checkpoints layout
        if os.path.isdir(ckpt_ver_dir):
            try:
                for name in sorted(os.listdir(ckpt_ver_dir)):
                    if name.startswith("fold_"):
                        d = os.path.join(ckpt_ver_dir, name)
                        if os.path.isdir(d):
                            fold_dirs.append((name, d))
            except Exception:
                fold_dirs = []

        # Fallback: packaged heads under weights/head/<ver>/fold_*
        if not fold_dirs:
            hb = resolve_version_head_base(project_dir, ver)
            if hb and os.path.isdir(hb):
                try:
                    for name in sorted(os.listdir(hb)):
                        if name.startswith("fold_"):
                            d = os.path.join(hb, name)
                            if os.path.isdir(d):
                                fold_dirs.append((name, d))
                except Exception:
                    fold_dirs = []

        for fold_name, fold_dir in fold_dirs:
            head_pt = _discover_head_pt_for_run_dir(fold_dir)
            if head_pt is None:
                continue
            try:
                w_eff = float(per_fold_w.get(fold_name, fold_w))
            except Exception:
                w_eff = float(fold_w)
            if w_eff <= 0.0:
                continue
            members.append((fold_name, head_pt, float(base_model_weight) * float(w_eff)))

    # 2) train_all member
    if include_train_all:
        head_pt = None
        if os.path.isdir(os.path.join(ckpt_ver_dir, "train_all")):
            head_pt = _discover_head_pt_for_run_dir(os.path.join(ckpt_ver_dir, "train_all"))
        if head_pt is None:
            hb = resolve_version_head_base(project_dir, ver)
            if hb and os.path.isdir(os.path.join(hb, "train_all")):
                head_pt = _discover_head_pt_for_run_dir(os.path.join(hb, "train_all"))
        if head_pt is not None and float(train_all_w) > 0.0:
            members.append(("train_all", head_pt, float(base_model_weight) * float(train_all_w)))

    if not members:
        print(f"[ENSEMBLE] mode=kfold_train_all: no members found for version={ver} (ckpt_dir={ckpt_ver_dir}).")
        return []

    # Build expanded sub-model dicts
    model_id = str(model_cfg.get("id", ver) or ver)
    cache_prefix = str(model_cfg.get("cache_name", "") or "").strip()
    out: List[dict] = []
    for member_name, head_pt, w_eff in members:
        mm = dict(model_cfg)
        mm["id"] = f"{model_id}__{member_name}"
        mm["version"] = ver
        mm["weight"] = float(w_eff)
        mm["head_base"] = str(head_pt)
        # Ensure per-member config path is explicit (avoid repeated resolve and keep provenance)
        mm["config_path"] = str(cfg_path)
        # Ensure cache paths are unique across members
        if cache_prefix:
            mm["cache_name"] = safe_slug(f"{cache_prefix}__{member_name}")
        else:
            mm["cache_name"] = safe_slug(f"{ver}__{member_name}")
        # Prevent re-expansion if this dict is processed again downstream.
        mm["mode"] = "single"
        # Helpful context for z_score discovery / debugging
        mm["_kfold_train_all"] = {
            "member": str(member_name),
            "log_dir": str(log_ver_dir),
            "ckpt_dir": str(ckpt_ver_dir),
        }
        out.append(mm)
    print(f"[ENSEMBLE] Expanded mode=kfold_train_all version={ver} -> {len(out)} members")
    return out


def _expand_ensemble_models(project_dir: str, models: List[dict]) -> List[dict]:
    """
    Expand any model entries that request within-version expansion.
    """
    out: List[dict] = []
    for m in models:
        if not isinstance(m, dict):
            continue
        expanded = _expand_kfold_train_all_model(project_dir, model_cfg=m)
        if expanded:
            out.extend(expanded)
        else:
            out.append(m)
    return out


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
                elif bn_l == "dinov3_vits16":
                    token = "vits16"
                elif bn_l == "dinov3_vits16plus":
                    token = "vits16plus"
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


def _load_zscore_json_for_head(
    head_pt_path: str,
    *,
    cfg: Optional[Dict] = None,
    project_dir: Optional[str] = None,
) -> Optional[dict]:
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

    # ==========================================================
    # Training-output mapping: checkpoints -> logs
    #
    # When running "directly from outputs/", head weights may come from:
    #   outputs/checkpoints/<version>/<run>/head/head-epoch*.pt
    # while z_score.json is written under:
    #   outputs/<version>/<run>/z_score.json
    #
    # We map ckpt_dir -> log_dir using cfg.logging.{ckpt_dir,log_dir}.
    # ==========================================================
    if not (isinstance(cfg, dict) and isinstance(project_dir, str) and project_dir.strip()):
        return None

    try:
        log_cfg = cfg.get("logging", {}) if isinstance(cfg.get("logging", {}), dict) else {}
        ckpt_dir_cfg = str(log_cfg.get("ckpt_dir", "") or "").strip()
        log_dir_cfg = str(log_cfg.get("log_dir", "") or "").strip()
        if not (ckpt_dir_cfg and log_dir_cfg):
            return None
        ckpt_root = resolve_path_best_effort(project_dir, ckpt_dir_cfg)
        log_root = resolve_path_best_effort(project_dir, log_dir_cfg)
        if not (ckpt_root and log_root and os.path.isdir(ckpt_root) and os.path.isdir(log_root)):
            return None

        abs_head = os.path.abspath(head_pt_path)
        ckpt_root_abs = os.path.abspath(ckpt_root)
        log_root_abs = os.path.abspath(log_root)

        # Ensure head is under ckpt_root
        try:
            if os.path.commonpath([abs_head, ckpt_root_abs]) != ckpt_root_abs:
                return None
        except Exception:
            # Fallback: relpath test
            rel_test = os.path.relpath(abs_head, ckpt_root_abs)
            if rel_test.startswith(".."):
                return None

        # Map:
        #   <ckpt_root>/<ver>/<run>/head/<file>.pt -> <log_root>/<ver>/<run>/z_score.json
        rel_head_dir = os.path.relpath(os.path.dirname(abs_head), ckpt_root_abs)
        rel_run_dir = rel_head_dir
        if os.path.basename(rel_head_dir) == "head":
            rel_run_dir = os.path.dirname(rel_head_dir)
        zcand = os.path.join(log_root_abs, rel_run_dir, "z_score.json")
        if os.path.isfile(zcand):
            with open(zcand, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        return None
    return None


def _infer_components_5d_for_model_2gpu_worker(q, shard_id: int, device_id: int, payload: dict) -> None:
    """
    Worker for 2-GPU data-parallel sharding of `infer_components_5d_for_model`.

    Notes:
    - Each worker is pinned to a single GPU via torch.cuda.set_device(device_id).
    - Recursion is prevented by ENV_DISABLE_2GPU_PARALLEL inside `worker_guarded`.
    """

    def _fn(pl: dict) -> dict:
        rels, comps, meta, mc_var = infer_components_5d_for_model(
            settings=pl["settings"],
            project_dir=pl["project_dir"],
            cfg=pl["cfg"],
            head_base=pl["head_base"],
            dino_weights_pt_path=pl["dino_weights_pt_path"],
            dataset_root=pl["dataset_root"],
            image_paths=pl["image_paths"],
            prefer_packaged_head_manifest=pl.get("prefer_packaged_head_manifest", True),
            compute_mc_var=bool(pl.get("compute_mc_var", False)),
            mc_dropout_samples_override=pl.get("mc_dropout_samples_override", None),
            mc_var_mode=str(pl.get("mc_var_mode", "relative")),
            mc_include_ratio=bool(pl.get("mc_include_ratio", False)),
            mc_ratio_weight=float(pl.get("mc_ratio_weight", 1.0)),
            mc_eps=float(pl.get("mc_eps", 1e-8)),
        )
        return {
            "rels_in_order": list(rels),
            "comps_5d_g": comps.detach().cpu().float() if isinstance(comps, torch.Tensor) else comps,
            "meta": dict(meta or {}),
            "mc_var": mc_var.detach().cpu().float() if isinstance(mc_var, torch.Tensor) else None,
        }

    worker_guarded(q, shard_id, device_id, payload, fn=_fn)


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
    compute_mc_var: bool = False,
    mc_dropout_samples_override: Optional[int] = None,
    mc_var_mode: str = "relative",
    mc_include_ratio: bool = False,
    mc_ratio_weight: float = 1.0,
    mc_eps: float = 1e-8,
 ) -> Tuple[List[str], torch.Tensor, dict, Optional[torch.Tensor]]:
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

    # ==========================================================
    # 2-GPU data-parallel inference (each GPU runs independent images)
    #
    # - Only used when >=2 GPUs are available.
    # - Disabled automatically when ViT7B model-parallel is enabled (it already uses both GPUs).
    # - Uses spawn+2 processes with even/odd sharding to preserve order on merge.
    # ==========================================================
    try:
        can_two_gpu = bool(two_gpu_parallel_enabled()) and int(len(image_paths)) >= 2
    except Exception:
        can_two_gpu = False
    if can_two_gpu:
        bn_l = str(backbone_name or "").strip().lower()
        wants_vit7b_mp = (
            bool(getattr(settings, "use_2gpu_model_parallel_for_vit7b", False))
            and (bn_l in ("dinov3_vit7b16", "dinov3_vit7b", "vit7b16", "vit7b"))
            and (mp_get_devices() is not None)
        )
        if not bool(wants_vit7b_mp):
            idx0, idx1 = split_even_odd_indices(int(len(image_paths)))
            if idx0 and idx1:
                image_paths0 = [image_paths[i] for i in idx0]
                image_paths1 = [image_paths[i] for i in idx1]

                payload_common = {
                    "settings": settings,
                    "project_dir": str(project_dir),
                    "cfg": cfg,
                    "head_base": str(head_base),
                    "dino_weights_pt_path": str(dino_weights_pt_path),
                    "dataset_root": str(dataset_root),
                    "prefer_packaged_head_manifest": bool(prefer_packaged_head_manifest),
                    "compute_mc_var": bool(compute_mc_var),
                    "mc_dropout_samples_override": mc_dropout_samples_override,
                    "mc_var_mode": str(mc_var_mode),
                    "mc_include_ratio": bool(mc_include_ratio),
                    "mc_ratio_weight": float(mc_ratio_weight),
                    "mc_eps": float(mc_eps),
                }

                try:
                    res0, res1 = run_two_processes_spawn(
                        worker=_infer_components_5d_for_model_2gpu_worker,
                        payload0={**payload_common, "image_paths": image_paths0},
                        payload1={**payload_common, "image_paths": image_paths1},
                        device0=0,
                        device1=1,
                    )
                except Exception as e:
                    print(f"[WARN] 2-GPU data-parallel disabled (fallback to single process): {e}")
                else:
                    rels0 = [str(r) for r in (res0.get("rels_in_order", []) or [])]
                    rels1 = [str(r) for r in (res1.get("rels_in_order", []) or [])]
                    if rels0 != list(image_paths0):
                        raise RuntimeError("2-GPU shard0 image order mismatch; aborting.")
                    if rels1 != list(image_paths1):
                        raise RuntimeError("2-GPU shard1 image order mismatch; aborting.")

                    comps0 = res0.get("comps_5d_g", None)
                    comps1 = res1.get("comps_5d_g", None)
                    if not (isinstance(comps0, torch.Tensor) and comps0.dim() == 2 and int(comps0.shape[1]) == 5):
                        raise RuntimeError(
                            f"2-GPU shard0 invalid comps tensor: {type(comps0)} shape={getattr(comps0, 'shape', None)}"
                        )
                    if not (isinstance(comps1, torch.Tensor) and comps1.dim() == 2 and int(comps1.shape[1]) == 5):
                        raise RuntimeError(
                            f"2-GPU shard1 invalid comps tensor: {type(comps1)} shape={getattr(comps1, 'shape', None)}"
                        )
                    if int(comps0.shape[0]) != int(len(image_paths0)):
                        raise RuntimeError(
                            f"2-GPU shard0 N mismatch: got {int(comps0.shape[0])}, expected {len(image_paths0)}"
                        )
                    if int(comps1.shape[0]) != int(len(image_paths1)):
                        raise RuntimeError(
                            f"2-GPU shard1 N mismatch: got {int(comps1.shape[0])}, expected {len(image_paths1)}"
                        )

                    N = int(len(image_paths))
                    comps_full = torch.empty((N, 5), dtype=torch.float32)
                    comps_full[idx0] = comps0.detach().cpu().float()
                    comps_full[idx1] = comps1.detach().cpu().float()

                    mc0 = res0.get("mc_var", None)
                    mc1 = res1.get("mc_var", None)
                    mc_full: Optional[torch.Tensor] = None
                    if isinstance(mc0, torch.Tensor) or isinstance(mc1, torch.Tensor):
                        if not (isinstance(mc0, torch.Tensor) and isinstance(mc1, torch.Tensor)):
                            raise RuntimeError("2-GPU MC variance mismatch across shards (one shard missing mc_var).")
                        mc0 = mc0.detach().cpu().float().view(-1)
                        mc1 = mc1.detach().cpu().float().view(-1)
                        if int(mc0.numel()) != int(len(image_paths0)) or int(mc1.numel()) != int(len(image_paths1)):
                            raise RuntimeError("2-GPU MC variance length mismatch across shards.")
                        mc_full = torch.empty((N,), dtype=torch.float32)
                        mc_full[idx0] = mc0
                        mc_full[idx1] = mc1

                    meta_out = dict(res0.get("meta", {}) or {})
                    meta_out["two_gpu_data_parallel"] = True
                    meta_out["two_gpu_shard_mode"] = "even_odd"
                    meta_out["two_gpu_devices"] = [0, 1]
                    meta_out["two_gpu_num_images"] = int(N)

                    return list(image_paths), comps_full, meta_out, mc_full

    try:
        if backbone_name == "dinov3_vith16plus":
            from dinov3.hub.backbones import dinov3_vith16plus as _make_backbone  # type: ignore
        elif backbone_name == "dinov3_vits16":
            from dinov3.hub.backbones import dinov3_vits16 as _make_backbone  # type: ignore
        elif backbone_name == "dinov3_vits16plus":
            from dinov3.hub.backbones import dinov3_vits16plus as _make_backbone  # type: ignore
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
    # Optional: accumulate per-head MC-dropout variance scalars for per-model uncertainty.
    # Stored as weighted variance of the weighted-mean prediction (assumes independence).
    mc_var_total_w2_sum: Optional[torch.Tensor] = None
    mc_var_ratio_w2_sum: Optional[torch.Tensor] = None

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
        # Helpful one-liner for debugging weight provenance (EMA vs raw).
        try:
            ws = str(meta.get("weights_source", "") or "").strip().lower()
            if not ws:
                ws = "unknown"
            ema_info = meta.get("ema", None)
            if ws == "ema" and isinstance(ema_info, dict) and ema_info.get("decay", None) is not None:
                print(f"[HEAD] {os.path.basename(str(head_pt))} weights_source=ema decay={ema_info.get('decay')}")
            else:
                print(f"[HEAD] {os.path.basename(str(head_pt))} weights_source={ws}")
        except Exception:
            pass

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
        dual_branch_enabled_head = bool(meta.get("dual_branch_enabled", first_meta.get("dual_branch_enabled", False)))
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
        elif head_type_meta == "mamba":
            # PyTorch-only Mamba-like axial scan head.
            mamba_dim_meta = int(
                meta.get(
                    "mamba_dim",
                    first_meta.get(
                        "mamba_dim",
                        int(
                            (cfg.get("model", {}).get("head", {}).get("mamba", {}) or {}).get(
                                "dim", cfg["model"]["head"].get("vitdet_dim", 320)
                            )
                        ),
                    ),
                )
            )
            mamba_depth_meta = int(
                meta.get(
                    "mamba_depth",
                    first_meta.get(
                        "mamba_depth",
                        int((cfg.get("model", {}).get("head", {}).get("mamba", {}) or {}).get("depth", 4)),
                    ),
                )
            )
            mamba_patch_size_meta = int(
                meta.get(
                    "mamba_patch_size",
                    first_meta.get(
                        "mamba_patch_size",
                        int(
                            (cfg.get("model", {}).get("head", {}).get("mamba", {}) or {}).get(
                                "patch_size", cfg["model"]["head"].get("fpn_patch_size", 16)
                            )
                        ),
                    ),
                )
            )
            mamba_d_conv_meta = int(
                meta.get(
                    "mamba_d_conv",
                    first_meta.get(
                        "mamba_d_conv",
                        int((cfg.get("model", {}).get("head", {}).get("mamba", {}) or {}).get("d_conv", 3)),
                    ),
                )
            )
            mamba_bidirectional_meta = bool(
                meta.get(
                    "mamba_bidirectional",
                    first_meta.get(
                        "mamba_bidirectional",
                        bool((cfg.get("model", {}).get("head", {}).get("mamba", {}) or {}).get("bidirectional", True)),
                    ),
                )
            )
            enable_ndvi_meta = bool(meta.get("enable_ndvi", False))
            num_layers_eff = max(1, len(backbone_layer_indices_head)) if use_layerwise_heads_head else 1
            mamba_cfg = MambaHeadConfig(
                embedding_dim=head_embedding_dim,
                mamba_dim=int(mamba_dim_meta),
                depth=int(mamba_depth_meta),
                patch_size=int(mamba_patch_size_meta),
                d_conv=int(mamba_d_conv_meta),
                bidirectional=bool(mamba_bidirectional_meta),
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
                head_module = MambaMultiLayerScalarHead(  # type: ignore[assignment]
                    mamba_cfg, num_layers=num_layers_eff, layer_fusion=fusion_mode_meta
                )
            else:
                head_module = Mamba2DScalarHead(mamba_cfg)  # type: ignore[assignment]
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
        elif head_type_meta == "eomt":
            # EoMT-style injected-query head (matches `third_party/eomt`):
            # run backbone for the first (depth - k) blocks without queries, then prepend
            # Q learnable query tokens and run the last-k blocks jointly.
            eomt_cfg = cfg.get("model", {}).get("head", {}).get("eomt", {})
            if not isinstance(eomt_cfg, dict):
                eomt_cfg = {}
            eomt_num_queries_meta = int(
                meta.get("eomt_num_queries", first_meta.get("eomt_num_queries", int(eomt_cfg.get("num_queries", 16))))
            )
            # Backward compatibility:
            # - older configs use `num_layers` for this value (from the previous query-decoder variant)
            # - older exported head meta uses `eomt_num_layers`
            eomt_num_blocks_meta = int(
                meta.get(
                    "eomt_num_blocks",
                    meta.get(
                        "eomt_num_layers",
                        first_meta.get(
                            "eomt_num_blocks",
                            first_meta.get(
                                "eomt_num_layers",
                                int(eomt_cfg.get("num_blocks", eomt_cfg.get("num_layers", 4))),
                            ),
                        ),
                    ),
                )
            )
            eomt_query_pool_meta = str(
                meta.get("eomt_query_pool", first_meta.get("eomt_query_pool", str(eomt_cfg.get("query_pool", "mean"))))
            ).strip().lower()
            # New pooled feature construction (backward-compatible defaults)
            eomt_use_mean_query_meta = bool(
                meta.get(
                    "eomt_use_mean_query",
                    first_meta.get(
                        "eomt_use_mean_query",
                        bool(eomt_cfg.get("use_mean_query", True)),
                    ),
                )
            )
            eomt_use_mean_patch_meta = bool(
                meta.get(
                    "eomt_use_mean_patch",
                    first_meta.get(
                        "eomt_use_mean_patch",
                        bool(eomt_cfg.get("use_mean_patch", False)),
                    ),
                )
            )
            eomt_use_cls_token_meta = bool(
                meta.get(
                    "eomt_use_cls_token",
                    first_meta.get(
                        "eomt_use_cls_token",
                        bool(eomt_cfg.get("use_cls_token", eomt_cfg.get("use_cls", False))),
                    ),
                )
            )
            eomt_proj_dim_meta = int(
                meta.get(
                    "eomt_proj_dim",
                    first_meta.get("eomt_proj_dim", int(eomt_cfg.get("proj_dim", 0))),
                )
            )
            eomt_proj_activation_meta = str(
                meta.get(
                    "eomt_proj_activation",
                    first_meta.get(
                        "eomt_proj_activation",
                        str(eomt_cfg.get("proj_activation", "relu")),
                    ),
                )
            ).strip().lower()
            try:
                eomt_proj_dropout_meta = float(
                    meta.get(
                        "eomt_proj_dropout",
                        first_meta.get(
                            "eomt_proj_dropout",
                            float(eomt_cfg.get("proj_dropout", 0.0)),
                        ),
                    )
                )
            except Exception:
                eomt_proj_dropout_meta = float(eomt_cfg.get("proj_dropout", 0.0) or 0.0)
            enable_ndvi_meta = bool(meta.get("enable_ndvi", False))
            head_module = EoMTInjectedQueryScalarHead(
                EoMTInjectedQueryHeadConfig(
                    embedding_dim=head_embedding_dim,
                    num_queries=eomt_num_queries_meta,
                    num_blocks=eomt_num_blocks_meta,
                    dropout=head_dropout,
                    query_pool=eomt_query_pool_meta,
                    use_mean_query=bool(eomt_use_mean_query_meta),
                    use_mean_patch=bool(eomt_use_mean_patch_meta),
                    use_cls_token=bool(eomt_use_cls_token_meta),
                    proj_dim=int(eomt_proj_dim_meta),
                    proj_activation=str(eomt_proj_activation_meta),
                    proj_dropout=float(eomt_proj_dropout_meta),
                    head_hidden_dims=head_hidden_dims,
                    head_activation=head_activation,
                    num_outputs_main=head_num_main,
                    num_outputs_ratio=head_num_ratio if head_is_ratio else 0,
                    enable_ndvi=enable_ndvi_meta,
                )
            )
        elif dual_branch_enabled_head and use_patch_reg3_head:
            # Dual-branch MLP patch-mode head: patch + global prediction fused by alpha.
            num_layers_eff = max(1, len(backbone_layer_indices_head)) if use_layerwise_heads_head else 1
            try:
                alpha_init_meta = float(meta.get("dual_branch_alpha_init", first_meta.get("dual_branch_alpha_init", 0.2)))
            except Exception:
                alpha_init_meta = 0.2
            head_module = DualBranchHeadExport(
                embedding_dim=head_embedding_dim,
                num_outputs_main=head_num_main,
                num_outputs_ratio=head_num_ratio if head_is_ratio else 0,
                head_hidden_dims=head_hidden_dims,
                head_activation=head_activation,
                dropout=head_dropout,
                use_cls_token=use_cls_token_head,
                num_layers=int(num_layers_eff),
                alpha_init=float(alpha_init_meta),
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
        try:
            mc_samples_eff = (
                int(mc_dropout_samples_override)
                if mc_dropout_samples_override is not None
                else int(getattr(settings, "mc_dropout_samples", 1))
            )
        except Exception:
            mc_samples_eff = int(getattr(settings, "mc_dropout_samples", 1) or 1)
        mc_samples_eff = max(1, int(mc_samples_eff))
        if bool(compute_mc_var) and mc_samples_eff < 2:
            # Sensible default if user requested uncertainty but did not provide samples.
            mc_samples_eff = 8
        mc_enabled = (bool(getattr(settings, "mc_dropout_enabled", False)) or bool(compute_mc_var)) and mc_samples_eff > 1
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
        preds_main_var_sum: Optional[torch.Tensor] = None
        preds_ratio_var_sum: Optional[torch.Tensor] = None
        preds_main_layers_sum: Optional[torch.Tensor] = None
        preds_ratio_layers_sum: Optional[torch.Tensor] = None
        n_views: int = 0

        for _view_idx, (image_size_view, hflip_view, vflip_view) in enumerate(tta_views):
            preds_main_layers_v: Optional[torch.Tensor] = None
            preds_ratio_layers_v: Optional[torch.Tensor] = None
            preds_main_var_v: Optional[torch.Tensor] = None
            preds_ratio_var_v: Optional[torch.Tensor] = None

            if head_type_meta == "fpn":
                rels_v, preds_main_v, preds_ratio_v, preds_main_var_v, preds_ratio_var_v = predict_main_and_ratio_fpn(
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
                    mc_dropout_samples=int(mc_samples_eff),
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
                    preds_main_var_v,
                    preds_ratio_var_v,
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
                    mc_dropout_samples=int(mc_samples_eff),
                    hflip=bool(hflip_view),
                    vflip=bool(vflip_view),
                )
            elif head_type_meta == "mamba":
                (
                    rels_v,
                    preds_main_v,
                    preds_ratio_v,
                    preds_main_layers_v,
                    preds_ratio_layers_v,
                    preds_main_var_v,
                    preds_ratio_var_v,
                ) = predict_main_and_ratio_mamba(
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
                    mc_dropout_samples=int(mc_samples_eff),
                    hflip=bool(hflip_view),
                    vflip=bool(vflip_view),
                )
            elif head_type_meta == "dpt":
                rels_v, preds_main_v, preds_ratio_v, preds_main_var_v, preds_ratio_var_v = predict_main_and_ratio_dpt(
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
                    mc_dropout_samples=int(mc_samples_eff),
                    hflip=bool(hflip_view),
                    vflip=bool(vflip_view),
                )
            elif head_type_meta == "eomt":
                # EoMT-style injected-query head: requires running the backbone blocks jointly
                # with learnable query tokens (cannot reuse the patch-token-only loop).
                rels_v, preds_main_v, preds_ratio_v, preds_main_var_v, preds_ratio_var_v = predict_main_and_ratio_eomt(
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
                    mc_dropout_enabled=mc_enabled,
                    mc_dropout_samples=int(mc_samples_eff),
                    hflip=bool(hflip_view),
                    vflip=bool(vflip_view),
                )
            elif dual_branch_enabled_head and use_patch_reg3_head:
                rels_v, preds_main_v, preds_ratio_v, preds_main_var_v, preds_ratio_var_v = predict_main_and_ratio_dual_branch(
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
                    num_layers=max(1, len(backbone_layer_indices_head)) if use_layerwise_heads_head else 1,
                    layer_indices=backbone_layer_indices_head if use_layerwise_heads_head else None,
                    layer_weights=layer_weights_head,
                    mc_dropout_enabled=mc_enabled,
                    mc_dropout_samples=int(mc_samples_eff),
                    hflip=bool(hflip_view),
                    vflip=bool(vflip_view),
                )
            elif use_patch_reg3_head:
                rels_v, preds_main_v, preds_ratio_v, preds_main_var_v, preds_ratio_var_v = predict_main_and_ratio_patch_mode(
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
                    mc_dropout_samples=int(mc_samples_eff),
                    hflip=bool(hflip_view),
                    vflip=bool(vflip_view),
                )
            else:
                if use_layerwise_heads_head and len(backbone_layer_indices_head) > 0:
                    rels_v, preds_main_v, preds_ratio_v, preds_main_var_v, preds_ratio_var_v = predict_main_and_ratio_global_multilayer(
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
                        mc_dropout_samples=int(mc_samples_eff),
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
                    preds_main_v, preds_ratio_v, preds_main_var_v, preds_ratio_var_v = predict_from_features(
                        features_cpu=features_cpu,
                        head=head_module,
                        batch_size=batch_size,
                        head_num_main=head_num_main,
                        head_num_ratio=head_num_ratio if head_is_ratio else 0,
                        head_total=head_total,
                        use_layerwise_heads=False,
                        num_layers=1,
                        mc_dropout_enabled=mc_enabled,
                        mc_dropout_samples=int(mc_samples_eff),
                    )

            if rels_view_ref is None:
                rels_view_ref = rels_v
            elif rels_view_ref != rels_v:
                raise RuntimeError("Image order mismatch across TTA views for a head; aborting.")

            preds_main_sum = preds_main_v if preds_main_sum is None else (preds_main_sum + preds_main_v)
            if head_is_ratio and head_num_ratio > 0 and preds_ratio_v is not None:
                preds_ratio_sum = preds_ratio_v if preds_ratio_sum is None else (preds_ratio_sum + preds_ratio_v)
            if preds_main_var_v is not None:
                preds_main_var_sum = (
                    preds_main_var_v if preds_main_var_sum is None else (preds_main_var_sum + preds_main_var_v)
                )
            if preds_ratio_var_v is not None:
                preds_ratio_var_sum = (
                    preds_ratio_var_v if preds_ratio_var_sum is None else (preds_ratio_var_sum + preds_ratio_var_v)
                )

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
        preds_main_var = (
            (preds_main_var_sum / float(n_views * n_views)) if preds_main_var_sum is not None else None
        )
        preds_ratio_var = (
            (preds_ratio_var_sum / float(n_views * n_views)) if preds_ratio_var_sum is not None else None
        )
        preds_main_layers = (preds_main_layers_sum / float(n_views)) if preds_main_layers_sum is not None else None
        preds_ratio_layers = (preds_ratio_layers_sum / float(n_views)) if preds_ratio_layers_sum is not None else None

        if rels_in_order_ref is None:
            rels_in_order_ref = rels_in_order
        elif rels_in_order_ref != rels_in_order:
            raise RuntimeError("Image order mismatch across heads within a model entry; aborting.")

        # --- Per-head normalization inversion (main outputs only) ---
        log_scale_cfg = bool(cfg["model"].get("log_scale_targets", False))
        log_scale_meta = bool(meta.get("log_scale_targets", log_scale_cfg))
        zscore = _load_zscore_json_for_head(head_pt, cfg=cfg, project_dir=project_dir)
        reg3_mean = None
        reg3_std = None
        if isinstance(zscore, dict) and "reg3" in zscore:
            try:
                reg3_mean = torch.tensor(zscore["reg3"]["mean"], dtype=torch.float32)
                reg3_std = torch.tensor(zscore["reg3"]["std"], dtype=torch.float32).clamp_min(1e-8)
            except Exception:
                reg3_mean, reg3_std = None, None
        zscore_enabled = reg3_mean is not None and reg3_std is not None

        # Optional MC-dropout variance propagation for main outputs:
        # preds_main_var is variance in the *raw head output space* after TTA averaging.
        preds_main_var_eff: Optional[torch.Tensor] = preds_main_var

        # Optional Softplus on main outputs (matches training behavior):
        use_output_softplus_cfg = bool(cfg.get("model", {}).get("head", {}).get("use_output_softplus", True))
        if use_output_softplus_cfg and (not log_scale_meta) and (not zscore_enabled):
            if preds_main_var_eff is not None:
                # Delta-method approximation: Var(softplus(x)) ≈ (sigmoid(E[x])^2) * Var(x)
                sig = torch.sigmoid(preds_main)
                preds_main_var_eff = preds_main_var_eff * (sig * sig)
            preds_main = F.softplus(preds_main)
            if preds_main_layers is not None:
                preds_main_layers = F.softplus(preds_main_layers)

        if zscore_enabled:
            if preds_main_var_eff is not None:
                s = reg3_std[:head_num_main]  # type: ignore[index]
                preds_main_var_eff = preds_main_var_eff * (s * s)
            preds_main = preds_main * reg3_std[:head_num_main] + reg3_mean[:head_num_main]  # type: ignore[index]
            if preds_main_layers is not None:
                m = reg3_mean[:head_num_main].view(1, 1, -1)  # type: ignore[index]
                s = reg3_std[:head_num_main].view(1, 1, -1)  # type: ignore[index]
                preds_main_layers = preds_main_layers * s + m
        if log_scale_meta:
            if preds_main_var_eff is not None:
                # Delta-method approximation: Var(expm1(x)) ≈ (exp(E[x])^2) * Var(x)
                d = torch.exp(preds_main)
                preds_main_var_eff = preds_main_var_eff * (d * d)
            preds_main = torch.expm1(preds_main).clamp_min(0.0)
            if preds_main_layers is not None:
                preds_main_layers = torch.expm1(preds_main_layers).clamp_min(0.0)

        # Convert from g/m^2 to grams
        preds_main_g = preds_main * float(area_m2)
        preds_main_g_var: Optional[torch.Tensor] = None
        if preds_main_var_eff is not None:
            preds_main_g_var = preds_main_var_eff * float(area_m2) * float(area_m2)
        preds_main_layers_g = (preds_main_layers * float(area_m2)) if preds_main_layers is not None else None

        # Build this head's 5D components in grams
        N = preds_main_g.shape[0]
        if head_is_ratio and preds_ratio is not None and head_num_main >= 1 and head_num_ratio >= 1:
            # Ratio head output format (legacy):
            # - logits (.., >=3) -> softmax over first 3 dims -> proportions
            #
            # NOTE: if an older head exported extra dims (e.g., hurdle/presence logits),
            # we intentionally ignore them (no gating).
            def _ratio_logits_to_probs_inf(rlog: torch.Tensor) -> torch.Tensor:
                return F.softmax(rlog[..., :3], dim=-1)

            # Prefer constraint-aware fusion when we have multi-layer per-layer outputs:
            # c_l = p(ratio_l) * total_l, then average c across layers.
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
                    p_layers = _ratio_logits_to_probs_inf(preds_ratio_layers)  # (N, L, 3)
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
                p_ratio = _ratio_logits_to_probs_inf(preds_ratio)
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

        # --- Per-head MC-dropout uncertainty scalar (optional) ---
        #
        # We store a single per-image scalar to drive per-sample weighting across ensemble models.
        # By default this is the predictive variance of Dry_Total_g (in grams) for this head.
        if bool(compute_mc_var) and isinstance(preds_main_g_var, torch.Tensor) and preds_main_g_var.numel() > 0:
            try:
                idx_total = 0 if bool(head_is_ratio) else int(target_bases.index("Dry_Total_g"))
            except Exception:
                idx_total = 0
            var_total_head: Optional[torch.Tensor] = None
            try:
                if preds_main_g_var.dim() == 2 and preds_main_g_var.shape[1] > 0:
                    if 0 <= int(idx_total) < int(preds_main_g_var.shape[1]):
                        var_total_head = preds_main_g_var[:, int(idx_total)].view(-1)
                    else:
                        var_total_head = preds_main_g_var.mean(dim=-1)
            except Exception:
                var_total_head = None
            if var_total_head is not None:
                contrib = var_total_head.detach().cpu().float() * float(head_w * head_w)
                mc_var_total_w2_sum = contrib if mc_var_total_w2_sum is None else (mc_var_total_w2_sum + contrib)

        if bool(compute_mc_var) and bool(mc_include_ratio) and isinstance(preds_ratio_var, torch.Tensor):
            # Ratio variance is in logits space (dimensionless); treated as an auxiliary uncertainty term.
            ratio_var_scalar = preds_ratio_var.detach().cpu().float().mean(dim=-1)
            contrib = ratio_var_scalar * float(head_w * head_w)
            mc_var_ratio_w2_sum = contrib if mc_var_ratio_w2_sum is None else (mc_var_ratio_w2_sum + contrib)

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
    mc_var_out: Optional[torch.Tensor] = None
    if bool(compute_mc_var) and mc_var_total_w2_sum is not None and float(weight_sum) > 0.0:
        var_total_model = mc_var_total_w2_sum / float(weight_sum * weight_sum)
        mode = str(mc_var_mode or "").strip().lower()
        if mode in ("relative", "rel", "cv2", "relative_total"):
            denom = comps_avg[:, 4].clamp_min(0.0).pow(2) + float(mc_eps)
            mc_var_out = var_total_model / denom
        else:
            mc_var_out = var_total_model
        if bool(mc_include_ratio) and mc_var_ratio_w2_sum is not None:
            ratio_var_model = mc_var_ratio_w2_sum / float(weight_sum * weight_sum)
            mc_var_out = mc_var_out + float(mc_ratio_weight) * ratio_var_model

    if mc_var_out is not None:
        meta_out["mc_var_mode"] = str(mc_var_mode)
        meta_out["mc_include_ratio"] = bool(mc_include_ratio)
        meta_out["mc_ratio_weight"] = float(mc_ratio_weight)

    return rels_in_order_ref, comps_avg, meta_out, mc_var_out


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


def _resolve_train_csv_path_for_calibration(*, project_dir: str, cfg: Dict, dataset_root: str) -> Optional[str]:
    """
    Resolve train.csv path for transductive calibration.

    We try:
      1) Under the inference input root (dataset_root)
      2) Under cfg.data.root relative to the repo root (project_dir)
      3) As an absolute cfg.data.train_csv path
    """
    train_csv_name = str(cfg.get("data", {}).get("train_csv", "train.csv") or "train.csv")
    candidates: List[str] = []

    # 1) dataset_root (resolved from settings.input_path)
    try:
        candidates.append(os.path.join(str(dataset_root), train_csv_name))
    except Exception:
        pass

    # 2) cfg.data.root relative to project_dir
    data_root_cfg = cfg.get("data", {}).get("root", None)
    if isinstance(data_root_cfg, str) and data_root_cfg.strip():
        try:
            candidates.append(os.path.join(str(project_dir), data_root_cfg.strip(), train_csv_name))
        except Exception:
            pass

    # 3) absolute override
    if os.path.isabs(train_csv_name):
        candidates.append(train_csv_name)

    for p in candidates:
        p = str(p or "").strip()
        if p and os.path.isfile(p):
            return os.path.abspath(p)
    return None


def _compute_train_stats_total_and_ratio(
    train_csv_path: str,
    *,
    q_clip: Tuple[float, float] = (0.01, 0.99),
    std_eps: float = 1e-8,
    p_eps: float = 1e-6,
) -> Tuple[float, float, Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Compute training distribution stats for the recommended calibration space:

    - Total: u = log1p(max(Total_g, 0))
    - Ratio logits: r = log(p), where p = normalize([Clover, Dead, Green])

    Returns:
      (mu_u, std_u, mu_r(3,), std_r(3,))
    """
    # Keep target_order strict so we only use rows where all required columns exist.
    df = read_and_pivot_csiro_train_csv(
        data_root="",
        train_csv=str(train_csv_path),
        target_order=("Dry_Total_g", "Dry_Clover_g", "Dry_Dead_g", "Dry_Green_g"),
    )

    cols = ["Dry_Clover_g", "Dry_Dead_g", "Dry_Green_g", "Dry_Total_g"]
    for c in cols:
        if c not in df.columns:
            raise RuntimeError(f"Missing column in train.csv pivot: {c}")
    arr = df[cols].to_numpy(dtype=np.float64, copy=True)
    if arr.ndim != 2 or arr.shape[1] != 4:
        raise RuntimeError(f"Unexpected train array shape: {arr.shape}")

    comp = np.maximum(arr[:, :3], 0.0)
    total = np.maximum(arr[:, 3], 0.0)

    # --- Total stats (log1p) ---
    u = np.log1p(total)
    u_stat = u
    if q_clip is not None:
        try:
            lo = float(np.nanquantile(u, float(q_clip[0])))
            hi = float(np.nanquantile(u, float(q_clip[1])))
            u_stat = np.clip(u, lo, hi)
        except Exception:
            u_stat = u
    mu_u = float(np.nanmean(u_stat))
    std_u = float(np.nanstd(u_stat))
    std_u = max(float(std_u), float(std_eps))
    if not (np.isfinite(mu_u) and np.isfinite(std_u)):
        raise RuntimeError("Non-finite train stats for Total (log1p).")

    # --- Ratio stats (log p) ---
    mu_r: Optional[np.ndarray] = None
    std_r: Optional[np.ndarray] = None
    comp_sum = np.sum(comp, axis=1)
    mask = np.isfinite(comp_sum) & (comp_sum > 0.0)
    if int(np.sum(mask)) >= 8:
        comp_m = comp[mask]
        comp_sum_m = comp_sum[mask]
        p = comp_m / np.maximum(comp_sum_m[:, None], float(p_eps))
        p = np.clip(p, float(p_eps), 1.0)
        # Re-normalize to ensure sum-to-one after clipping.
        p = p / np.maximum(np.sum(p, axis=1, keepdims=True), float(p_eps))
        r = np.log(p)

        r_stat = r
        if q_clip is not None:
            try:
                lo = np.nanquantile(r, float(q_clip[0]), axis=0)
                hi = np.nanquantile(r, float(q_clip[1]), axis=0)
                r_stat = np.clip(r, lo, hi)
            except Exception:
                r_stat = r
        mu_r = np.nanmean(r_stat, axis=0).astype(np.float64)
        std_r = np.nanstd(r_stat, axis=0).astype(np.float64)
        std_r = np.maximum(std_r, float(std_eps))
        if mu_r.shape != (3,) or std_r.shape != (3,):
            mu_r = None
            std_r = None

    return mu_u, std_u, mu_r, std_r


def _transductive_affine_calibrate_total_and_ratio(
    comps_5d_g: torch.Tensor,
    *,
    project_dir: str,
    dataset_root: str,
    cfg: Dict,
    q_clip: Tuple[float, float] = (0.01, 0.99),
    lam: float = 0.3,
    a_clip_total: Tuple[float, float] = (0.7, 1.3),
    a_clip_ratio: Tuple[float, float] = (0.7, 1.3),
    calibrate_ratio: bool = True,
    std_eps: float = 1e-8,
    p_eps: float = 1e-6,
) -> torch.Tensor:
    """
    Recommended transductive affine calibration that preserves hard constraints by construction.

    - Calibrate Total in u = log1p(grams) space with u' = a*u + b.
    - Calibrate composition in logits space over [Clover, Dead, Green] with r' = a*r + b,
      where r = log(p) and p sums to 1.

    Reconstruct per sample:
      total' = expm1(u')
      p' = softmax(r')
      [C',D',G'] = total' * p'
      GDM' = C' + G'

    Output is (N,5) in grams with constraints:
      Total' = C'+D'+G',  GDM' = C'+G'
    """
    if not isinstance(comps_5d_g, torch.Tensor):
        return comps_5d_g
    if comps_5d_g.numel() == 0:
        return comps_5d_g
    if comps_5d_g.dim() != 2 or int(comps_5d_g.shape[1]) != 5:
        raise RuntimeError(f"Expected comps_5d_g shape (N,5), got: {tuple(comps_5d_g.shape)}")

    train_csv_path = _resolve_train_csv_path_for_calibration(project_dir=project_dir, cfg=cfg, dataset_root=dataset_root)
    if not (train_csv_path and os.path.isfile(train_csv_path)):
        print("[CALIB] train.csv not found; skipping transductive calibration.")
        return comps_5d_g

    try:
        mu_u_t, std_u_t, mu_r_t, std_r_t = _compute_train_stats_total_and_ratio(
            train_csv_path, q_clip=q_clip, std_eps=float(std_eps), p_eps=float(p_eps)
        )
    except Exception as e:
        print(f"[CALIB] Failed to compute training stats from: {train_csv_path}. Skipping. Error: {e}")
        return comps_5d_g

    x = comps_5d_g.detach().cpu()
    out_dtype = x.dtype
    x64 = x.to(torch.float64)

    # ----- Build calibration variables from predictions -----
    total = x64[:, 4].clamp_min(0.0)
    u = torch.log1p(total)

    comp = x64[:, :3].clamp_min(0.0)
    comp_sum = comp.sum(dim=-1, keepdim=True)
    p = comp / comp_sum.clamp_min(float(p_eps))
    zero_mask = (comp_sum.view(-1) <= float(p_eps)) | (~torch.isfinite(comp_sum.view(-1)))
    if bool(zero_mask.any().item()):
        p[zero_mask] = 1.0 / 3.0
    p = p.clamp_min(float(p_eps))
    p = p / p.sum(dim=-1, keepdim=True).clamp_min(float(p_eps))
    # log(p) is a valid logits parameterization since softmax(log(p)) == p.
    r = torch.log(p)

    # ----- Compute prediction stats (with optional robust clipping) -----
    u_stat = u
    if q_clip is not None:
        try:
            u_lo = torch.quantile(u, float(q_clip[0]))
            u_hi = torch.quantile(u, float(q_clip[1]))
            u_stat = u.clamp(u_lo, u_hi)
        except Exception:
            u_stat = u
    mu_u_p = u_stat.mean()
    std_u_p = u_stat.std(unbiased=False).clamp_min(float(std_eps))

    # ----- Total affine params (shrink towards identity) -----
    mu_u_t_t = torch.tensor(float(mu_u_t), dtype=torch.float64)
    std_u_t_t = torch.tensor(float(std_u_t), dtype=torch.float64)
    a_u = (std_u_t_t / std_u_p).clamp(min=float(a_clip_total[0]), max=float(a_clip_total[1]))
    b_u = mu_u_t_t - a_u * mu_u_p
    a_u = 1.0 + float(lam) * (a_u - 1.0)
    b_u = float(lam) * b_u

    u_cal = a_u * u + b_u
    total_cal = torch.expm1(u_cal).clamp_min(0.0)

    # ----- Ratio affine params (optional if we have valid train stats) -----
    ratio_ok = bool(calibrate_ratio) and (
        isinstance(mu_r_t, np.ndarray)
        and isinstance(std_r_t, np.ndarray)
        and mu_r_t.shape == (3,)
        and std_r_t.shape == (3,)
        and bool(np.isfinite(mu_r_t).all())
        and bool(np.isfinite(std_r_t).all())
    )
    r_cal = r
    if ratio_ok:
        r_stat = r
        if q_clip is not None:
            try:
                r_lo = torch.quantile(r, float(q_clip[0]), dim=0)
                r_hi = torch.quantile(r, float(q_clip[1]), dim=0)
                r_stat = r.clamp(r_lo.view(1, 3), r_hi.view(1, 3))
            except Exception:
                r_stat = r
        mu_r_p = r_stat.mean(dim=0)
        std_r_p = r_stat.std(dim=0, unbiased=False).clamp_min(float(std_eps))
        mu_r_t_t = torch.as_tensor(mu_r_t, dtype=torch.float64)
        std_r_t_t = torch.as_tensor(std_r_t, dtype=torch.float64)
        a_r = (std_r_t_t / std_r_p).clamp(min=float(a_clip_ratio[0]), max=float(a_clip_ratio[1]))
        b_r = mu_r_t_t - a_r * mu_r_p
        a_r = 1.0 + float(lam) * (a_r - 1.0)
        b_r = float(lam) * b_r
        r_cal = (r * a_r.view(1, 3)) + b_r.view(1, 3)

    # ----- Reconstruct constraint-consistent 5D outputs -----
    p_cal = torch.softmax(r_cal, dim=-1)
    comp_cal = p_cal * total_cal.view(-1, 1)
    clover = comp_cal[:, 0]
    dead = comp_cal[:, 1]
    green = comp_cal[:, 2]
    gdm = clover + green
    out = torch.stack([clover, dead, green, gdm, total_cal], dim=-1).to(dtype=out_dtype)

    # ----- Logging -----
    try:
        print(f"[CALIB] Enabled (train_csv={train_csv_path})")
        print(
            "[CALIB] Total log1p: "
            f"mu_pred={float(mu_u_p.item()):.4f}, std_pred={float(std_u_p.item()):.4f} "
            f"-> mu_train={float(mu_u_t):.4f}, std_train={float(std_u_t):.4f} "
            f"(a={float(a_u.item()):.4f}, b={float(b_u.item()):.4f}, lam={float(lam):.3f})"
        )
        if ratio_ok:
            # Report averages for readability (3 dims).
            mu_r_p0 = r.mean(dim=0)
            std_r_p0 = r.std(dim=0, unbiased=False)
            print(
                "[CALIB] Ratio logits: "
                f"mu_pred_mean={float(mu_r_p0.mean().item()):.4f}, std_pred_mean={float(std_r_p0.mean().item()):.4f} "
                f"-> mu_train_mean={float(np.mean(mu_r_t)):.4f}, std_train_mean={float(np.mean(std_r_t)):.4f} "
                f"(lam={float(lam):.3f})"
            )
        else:
            print("[CALIB] Ratio logits: train stats unavailable; keeping original composition.")
    except Exception:
        pass

    return out


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
    # Optional: within-version expansion (kfold + train_all -> multiple members)
    if ensemble_models:
        ensemble_models = _expand_ensemble_models(project_dir_abs, ensemble_models)
    if len(ensemble_models) > 0:
        ensemble_obj = read_ensemble_cfg_obj(project_dir_abs) or {}
        cache_dir_cfg = ensemble_obj.get("cache_dir", None)
        cache_dir = resolve_cache_dir(project_dir_abs, cache_dir_cfg)

        # Optional: per-sample dynamic weighting across ensemble models.
        agg = str(ensemble_obj.get("aggregation", "") or "").strip().lower()
        psw = ensemble_obj.get("per_sample_weighting", None)
        psw = psw if isinstance(psw, dict) else {}
        psw_enabled = bool(psw.get("enabled", False)) or agg in (
            "per_sample_weighted_mean",
            "dynamic_weighted_mean",
            "dynamic",
        )
        psw_method = str(psw.get("method", "mc_and_disagreement") or "mc_and_disagreement").strip().lower()

        mc_cfg = psw.get("mc_dropout", None)
        mc_cfg = mc_cfg if isinstance(mc_cfg, dict) else {}
        dis_cfg = psw.get("disagreement", None)
        dis_cfg = dis_cfg if isinstance(dis_cfg, dict) else {}

        use_mc = psw_enabled and (psw_method in ("mc_dropout", "mc_and_disagreement")) and bool(mc_cfg.get("enabled", True))
        use_dis = psw_enabled and (psw_method in ("disagreement", "mc_and_disagreement")) and bool(dis_cfg.get("enabled", True))

        # MC-dropout config (used during model inference to estimate uncertainty).
        mc_samples_override = None
        try:
            v = mc_cfg.get("samples", mc_cfg.get("mc_dropout_samples", None))
            if v is not None:
                mc_samples_override = int(v)
        except Exception:
            mc_samples_override = None
        mc_var_mode = str(mc_cfg.get("var_mode", "relative") or "relative").strip().lower()
        mc_include_ratio = bool(mc_cfg.get("include_ratio", False))
        try:
            mc_ratio_weight = float(mc_cfg.get("ratio_weight", 1.0))
        except Exception:
            mc_ratio_weight = 1.0
        try:
            mc_eps = float(mc_cfg.get("eps", 1e-8))
        except Exception:
            mc_eps = 1e-8

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
            rels_in_order, comps_5d_g, meta, mc_var = infer_components_5d_for_model(
                settings=settings,
                project_dir=project_dir_abs,
                cfg=cfg_model,
                head_base=head_base,
                dino_weights_pt_path=dino_weights_path,
                dataset_root=dataset_root,
                image_paths=unique_image_paths,
                prefer_packaged_head_manifest=True,
                compute_mc_var=bool(use_mc),
                mc_dropout_samples_override=mc_samples_override,
                mc_var_mode=str(mc_var_mode),
                mc_include_ratio=bool(mc_include_ratio),
                mc_ratio_weight=float(mc_ratio_weight),
                mc_eps=float(mc_eps),
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
                mc_var=mc_var,
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
        comps_list: List[torch.Tensor] = []
        mc_var_list: List[Optional[torch.Tensor]] = []
        w_list: List[float] = []
        for cache_path, w in cache_items:
            if not (w > 0.0):
                continue
            rels, comps, _meta, mc_var = load_ensemble_cache(cache_path)
            if rels_ref is None:
                rels_ref = rels
            elif rels_ref != rels:
                raise RuntimeError("Image order mismatch across cached models; aborting ensemble.")
            comps_list.append(comps.detach().cpu().float())
            mc_var_list.append(mc_var.detach().cpu().float() if isinstance(mc_var, torch.Tensor) else None)
            w_list.append(float(w))

        if rels_ref is None or not comps_list or not w_list:
            raise RuntimeError("Failed to build an ensemble from cached predictions.")

        comps_stack = torch.stack(comps_list, dim=0)  # (M,N,5)
        w_base = torch.tensor(w_list, dtype=torch.float32).view(-1, 1)  # (M,1)
        w_base_sum = float(w_base.sum().item())
        if not (w_base_sum > 0.0):
            raise RuntimeError("Invalid ensemble weights (sum <= 0).")

        # Default: weighted mean (legacy behavior).
        comps_avg = (comps_stack * w_base.view(-1, 1, 1)).sum(dim=0) / float(w_base_sum)  # (N,5)

        # Optional per-sample weighting: scale model weights by uncertainty and/or agreement per image.
        if bool(psw_enabled) and (bool(use_mc) or bool(use_dis)) and comps_stack.numel() > 0:
            M, N, D = int(comps_stack.shape[0]), int(comps_stack.shape[1]), int(comps_stack.shape[2])
            if D != 5:
                raise RuntimeError(f"Unexpected comps dimension for ensemble: D={D} (expected 5)")

            w_dyn = w_base.expand(M, N).clone()  # (M,N)

            # MC-dropout uncertainty weighting (requires per-model mc_var scalars).
            if bool(use_mc):
                mv_rows: List[torch.Tensor] = []
                for mv in mc_var_list:
                    if isinstance(mv, torch.Tensor) and mv.dim() == 1 and int(mv.numel()) == N:
                        mv_rows.append(mv.view(1, N))
                    else:
                        mv_rows.append(torch.full((1, N), float("nan"), dtype=torch.float32))
                mv_mat = torch.cat(mv_rows, dim=0)  # (M,N)
                finite = torch.isfinite(mv_mat)
                if not bool(finite.any().item()):
                    mv_mat = torch.zeros((M, N), dtype=torch.float32)
                else:
                    mv_filled = mv_mat.clone()
                    mv_filled[~finite] = 0.0
                    try:
                        mv_med = torch.nanmedian(mv_mat, dim=0).values  # (N,)
                    except Exception:
                        mv_med = mv_filled.mean(dim=0)
                    mv_mat = torch.where(finite, mv_mat, mv_med.view(1, N))

                mc_weight_fn = str(mc_cfg.get("weight_fn", "inv") or "inv").strip().lower()
                try:
                    mc_power = float(mc_cfg.get("power", 1.0))
                except Exception:
                    mc_power = 1.0
                try:
                    mc_weight_eps = float(mc_cfg.get("weight_eps", mc_eps))
                except Exception:
                    mc_weight_eps = float(mc_eps)
                clip = mc_cfg.get("clip", None)
                if isinstance(clip, (list, tuple)) and len(clip) == 2:
                    try:
                        c0 = float(clip[0])
                        c1 = float(clip[1])
                        mv_mat = mv_mat.clamp(min=c0, max=c1)
                    except Exception:
                        pass

                if mc_weight_fn in ("exp", "softmax"):
                    try:
                        temp = float(mc_cfg.get("temperature", 1.0))
                    except Exception:
                        temp = 1.0
                    temp = max(1e-8, float(temp))
                    w_mc = torch.exp(-mv_mat / temp)
                else:
                    w_mc = (mv_mat + float(mc_weight_eps)).clamp_min(0.0).pow(-float(mc_power))

                w_dyn = w_dyn * w_mc

            # Model disagreement / consistency weighting (based on distance to the base-weight mean).
            if bool(use_dis):
                dis_space = str(dis_cfg.get("space", "log1p") or "log1p").strip().lower()
                x = comps_stack.clamp_min(0.0)
                if dis_space in ("log", "log1p"):
                    x = torch.log1p(x)

                mean_ref = (x * w_base.view(-1, 1, 1)).sum(dim=0) / float(w_base_sum)  # (N,5)
                dis = ((x - mean_ref.view(1, N, 5)) ** 2).mean(dim=-1)  # (M,N)

                dis_weight_fn = str(dis_cfg.get("weight_fn", "inv") or "inv").strip().lower()
                try:
                    dis_power = float(dis_cfg.get("power", 1.0))
                except Exception:
                    dis_power = 1.0
                try:
                    dis_eps = float(dis_cfg.get("eps", 1e-8))
                except Exception:
                    dis_eps = 1e-8
                if dis_weight_fn in ("exp", "softmax"):
                    try:
                        temp = float(dis_cfg.get("temperature", 1.0))
                    except Exception:
                        temp = 1.0
                    temp = max(1e-8, float(temp))
                    w_dis = torch.exp(-dis / temp)
                else:
                    w_dis = (dis + float(dis_eps)).clamp_min(0.0).pow(-float(dis_power))

                w_dyn = w_dyn * w_dis

            # Normalize weights per sample and ensemble.
            w_sum_dyn = w_dyn.sum(dim=0)  # (N,)
            bad = (~torch.isfinite(w_sum_dyn)) | (w_sum_dyn <= 0.0)
            if bool(bad.any().item()):
                w_dyn = torch.where(bad.view(1, N), w_base.expand(M, N), w_dyn)
                w_sum_dyn = w_dyn.sum(dim=0).clamp_min(1e-12)
            else:
                w_sum_dyn = w_sum_dyn.clamp_min(1e-12)
            w_norm = w_dyn / w_sum_dyn.view(1, N)
            comps_avg = (comps_stack * w_norm.view(M, N, 1)).sum(dim=0)

        if bool(getattr(settings, "transductive_calibration_enabled", False)):
            comps_avg = _transductive_affine_calibrate_total_and_ratio(
                comps_avg,
                project_dir=project_dir_abs,
                dataset_root=dataset_root,
                cfg=cfg,
                q_clip=tuple(getattr(settings, "transductive_calibration_q_clip", (0.01, 0.99))),
                lam=float(getattr(settings, "transductive_calibration_lam", 0.3)),
                a_clip_total=tuple(getattr(settings, "transductive_calibration_a_clip_total", (0.7, 1.3))),
                a_clip_ratio=tuple(getattr(settings, "transductive_calibration_a_clip_ratio", (0.7, 1.3))),
                calibrate_ratio=bool(getattr(settings, "transductive_calibration_calibrate_ratio", True)),
                std_eps=float(getattr(settings, "transductive_calibration_std_eps", 1e-8)),
                p_eps=float(getattr(settings, "transductive_calibration_p_eps", 1e-6)),
            )
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

    rels_in_order, comps_5d_g, _meta, _mc_var = infer_components_5d_for_model(
        settings=settings,
        project_dir=project_dir_abs,
        cfg=cfg,
        head_base=settings.head_weights_pt_path,
        dino_weights_pt_path=dino_weights_pt_path,
        dataset_root=dataset_root,
        image_paths=unique_image_paths,
        prefer_packaged_head_manifest=prefer_packaged_manifest,
    )

    if bool(getattr(settings, "transductive_calibration_enabled", False)):
        comps_5d_g = _transductive_affine_calibrate_total_and_ratio(
            comps_5d_g,
            project_dir=project_dir_abs,
            dataset_root=dataset_root,
            cfg=cfg,
            q_clip=tuple(getattr(settings, "transductive_calibration_q_clip", (0.01, 0.99))),
            lam=float(getattr(settings, "transductive_calibration_lam", 0.3)),
            a_clip_total=tuple(getattr(settings, "transductive_calibration_a_clip_total", (0.7, 1.3))),
            a_clip_ratio=tuple(getattr(settings, "transductive_calibration_a_clip_ratio", (0.7, 1.3))),
            calibrate_ratio=bool(getattr(settings, "transductive_calibration_calibrate_ratio", True)),
            std_eps=float(getattr(settings, "transductive_calibration_std_eps", 1e-8)),
            p_eps=float(getattr(settings, "transductive_calibration_p_eps", 1e-6)),
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


