from __future__ import annotations

import json
import os
from typing import List, Optional, Tuple

import torch

from src.inference.paths import resolve_path_best_effort
from src.inference.torch_load import torch_load_cpu


def read_ensemble_cfg_obj(project_dir: str) -> Optional[dict]:
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


def read_ensemble_enabled_flag(project_dir: str) -> bool:
    """
    Read configs/ensemble.json and return 'enabled' flag (default: False).
    """
    try:
        obj = read_ensemble_cfg_obj(project_dir)
        if not isinstance(obj, dict):
            return False
        return bool(obj.get("enabled", False))
    except Exception:
        return False


def normalize_ensemble_models(project_dir: str) -> List[dict]:
    """
    Normalize configs/ensemble.json into a list of model dicts.

    Supported schemas:
      - New: { enabled: true, models: [ {id?, version?, weight?, head_weights?, config? ...}, ... ] }
      - Legacy: { enabled: true, versions: [ "ver1", "ver2", ... ] }  (or single "version")
    """
    obj = read_ensemble_cfg_obj(project_dir)
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


def read_packaged_ensemble_manifest_if_exists(head_base: str) -> Optional[Tuple[List[Tuple[str, float]], str]]:
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


def save_ensemble_cache(
    cache_path: str,
    *,
    model_id: str,
    model_weight: float,
    rels_in_order: List[str],
    comps_5d_g: torch.Tensor,
    mc_var: Optional[torch.Tensor] = None,
    meta: Optional[dict] = None,
) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(cache_path)), exist_ok=True)
    payload = {
        "schema_version": 2,
        "model_id": str(model_id),
        "model_weight": float(model_weight),
        "components_order": ["Dry_Clover_g", "Dry_Dead_g", "Dry_Green_g", "GDM_g", "Dry_Total_g"],
        "rels_in_order": list(rels_in_order),
        "comps_5d_g": comps_5d_g.detach().cpu().float(),
        "meta": meta or {},
    }
    if isinstance(mc_var, torch.Tensor):
        # MC-dropout uncertainty scalar per image (N,) or (N,1).
        mv = mc_var.detach().cpu().float()
        if mv.dim() == 2 and mv.size(-1) == 1:
            mv = mv.view(-1)
        payload["mc_var"] = mv
    torch.save(payload, cache_path)


def load_ensemble_cache(cache_path: str) -> Tuple[List[str], torch.Tensor, dict, Optional[torch.Tensor]]:
    obj = torch_load_cpu(cache_path, mmap=None, weights_only=True)
    if not isinstance(obj, dict):
        raise RuntimeError(f"Invalid cache format (expected dict): {cache_path}")
    rels = obj.get("rels_in_order", None)
    comps = obj.get("comps_5d_g", None)
    meta = obj.get("meta", {}) if isinstance(obj.get("meta", {}), dict) else {}
    mc_var = obj.get("mc_var", None)
    if not isinstance(rels, list) or comps is None:
        raise RuntimeError(f"Invalid cache payload (missing rels/comps): {cache_path}")
    if not isinstance(comps, torch.Tensor) or comps.dim() != 2 or comps.size(-1) != 5:
        raise RuntimeError(f"Invalid comps_5d_g shape in cache: {cache_path}")
    mc_var_out: Optional[torch.Tensor] = None
    if isinstance(mc_var, torch.Tensor):
        mv = mc_var.detach().cpu().float()
        if mv.dim() == 2 and mv.size(-1) == 1:
            mv = mv.view(-1)
        if mv.dim() == 1 and mv.numel() == int(comps.shape[0]):
            mc_var_out = mv
    return [str(r) for r in rels], comps.detach().cpu().float(), meta, mc_var_out


def resolve_cache_dir(project_dir: str, cache_dir_cfg: Optional[str]) -> str:
    """
    Resolve a writable cache dir (Kaggle-safe).
    """
    if isinstance(cache_dir_cfg, str) and cache_dir_cfg.strip():
        cache_dir = resolve_path_best_effort(project_dir, cache_dir_cfg.strip())
    else:
        cache_dir = os.path.join(project_dir, "outputs", "ensemble_cache")
    # Ensure cache_dir is writable (Kaggle note: /kaggle/input is read-only).
    try:
        os.makedirs(cache_dir, exist_ok=True)
    except Exception:
        pass
    if not (os.path.isdir(cache_dir) and os.access(cache_dir, os.W_OK)):
        # Prefer Kaggle working dir if present, otherwise fall back to system temp.
        import tempfile

        fallback_roots: List[str] = []
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
                project_dir,
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
    return cache_dir


