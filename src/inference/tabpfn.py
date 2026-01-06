from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from src.inference.two_gpu_parallel import (
    run_two_processes_spawn,
    split_even_odd_indices,
    two_gpu_parallel_enabled,
    worker_guarded,
)

if TYPE_CHECKING:
    import numpy as np

    from src.inference.settings import InferenceSettings


@dataclass(frozen=True)
class TabPFNSubmissionSettings:
    """
    Settings for TabPFN-based offline inference.

    Important: TabPFN weights are loaded LOCAL-ONLY from `weights_ckpt_path`.
    """

    # Local TabPFN 2.5 regressor checkpoint (.ckpt). Required.
    weights_ckpt_path: str

    # Optional: path to TabPFN python package source (offline-friendly).
    tabpfn_python_path: str = ""

    # TabPFN runtime params
    device: str = "auto"
    n_estimators: int = 8
    fit_mode: str = "fit_preprocessors"
    inference_precision: str = "auto"
    ignore_pretraining_limits: bool = True
    n_jobs: int = 1  # MultiOutputRegressor parallelism
    enable_telemetry: bool = False
    model_cache_dir: str = ""

    # Training data for TabPFN:
    # - Default: <dataset_root>/train.csv, where dataset_root is resolved from input_path.
    # - Override: explicit train.csv path.
    train_csv_path: str = ""

    # Directory (relative to `settings.project_dir`) containing packaged TabPFN fit-state
    # bundles produced by `package_artifacts.py`.
    #
    # Inference never fits TabPFN online; it requires this directory to exist.
    fit_state_dir: str = "tabpfn_fit"

    # Head-penultimate feature extraction settings
    # Feature source mode (matches configs/train_tabpfn.yaml):
    # - "head_penultimate": use regression head's penultimate (pre-final-linear) features (default; existing behavior)
    # - "dinov3_only"     : use frozen DINOv3 CLS token features (no LoRA, no head)
    feature_mode: str = "head_penultimate"
    feature_fusion: str = "mean"  # "mean" | "concat"
    feature_batch_size: int = 8
    feature_num_workers: int = 8
    feature_cache_path_train: str = ""  # optional .pt cache
    feature_cache_path_test: str = ""  # optional .pt cache

    # Post-processing constraint for 5D outputs (see `src.tabular.ratio_strict.apply_ratio_strict_5d`).
    ratio_strict: bool = False


def load_tabpfn_submission_settings_from_yaml(
    *,
    project_dir: str,
    config_path: str = "configs/train_tabpfn.yaml",
) -> TabPFNSubmissionSettings:
    """
    Load TabPFN submission settings from `configs/train_tabpfn.yaml` (or a compatible file).

    This keeps `infer_and_submit_pt.py` free of TabPFN parameter clutter as requested.
    """
    from src.inference.paths import resolve_path_best_effort

    project_dir_abs = os.path.abspath(project_dir) if project_dir else ""
    if not (project_dir_abs and os.path.isdir(project_dir_abs)):
        raise RuntimeError("project_dir must point to the repository root containing `configs/` and `src/`.")

    cfg_path = resolve_path_best_effort(project_dir_abs, str(config_path or "").strip() or "configs/train_tabpfn.yaml")
    if not os.path.isfile(cfg_path):
        raise FileNotFoundError(f"TabPFN config YAML not found: {cfg_path}")

    try:
        import yaml
    except Exception as e:  # pragma: no cover
        raise RuntimeError("PyYAML is required to load configs/train_tabpfn.yaml") from e

    with open(cfg_path, "r", encoding="utf-8") as f:
        obj = yaml.safe_load(f)
    cfg: Dict[str, Any] = dict(obj or {})

    tab = cfg.get("tabpfn", {}) if isinstance(cfg.get("tabpfn", {}), dict) else {}
    imgf = cfg.get("image_features", {}) if isinstance(cfg.get("image_features", {}), dict) else {}
    artifacts = cfg.get("artifacts", {}) if isinstance(cfg.get("artifacts", {}), dict) else {}

    # TabPFN base checkpoint (local-only)
    weights_ckpt_path = str(tab.get("model_path", tab.get("weights_ckpt_path", "")) or "").strip()
    if not weights_ckpt_path:
        raise ValueError("configs/train_tabpfn.yaml missing required: tabpfn.model_path")

    return TabPFNSubmissionSettings(
        weights_ckpt_path=weights_ckpt_path,
        tabpfn_python_path=str(tab.get("python_path", tab.get("tabpfn_python_path", "")) or "").strip(),
        device=str(tab.get("device", "auto") or "auto"),
        n_estimators=int(tab.get("n_estimators", 8)),
        fit_mode=str(tab.get("fit_mode", "fit_preprocessors") or "fit_preprocessors"),
        inference_precision=str(tab.get("inference_precision", "auto") or "auto"),
        ignore_pretraining_limits=bool(tab.get("ignore_pretraining_limits", True)),
        n_jobs=int(tab.get("n_jobs", 1)),
        enable_telemetry=bool(tab.get("enable_telemetry", False)),
        model_cache_dir=str(tab.get("model_cache_dir", "") or ""),
        # train_csv_path is intentionally kept for backward compatibility but is not used in inference.
        train_csv_path=str(tab.get("train_csv_path", "") or ""),
        feature_mode=str(imgf.get("mode", "head_penultimate") or "head_penultimate"),
        feature_fusion=str(imgf.get("fusion", "mean") or "mean"),
        feature_batch_size=int(imgf.get("batch_size", 8)),
        feature_num_workers=int(imgf.get("num_workers", 8)),
        # Optional (mostly useful for debugging); inference is safe when left empty.
        feature_cache_path_train=str(imgf.get("cache_path_train", "") or ""),
        feature_cache_path_test=str(imgf.get("cache_path_test", "") or ""),
        ratio_strict=bool(tab.get("ratio_strict", False)),
        fit_state_dir=str(artifacts.get("fit_state_dir", "tabpfn_fit") or "tabpfn_fit"),
    )


_TABPFN_FIT_EXT = ".tabpfn_fit"


class _TabPFNMultiOutputPredictor:
    """
    Lightweight multi-output predictor wrapper that avoids pickling sklearn's
    MultiOutputRegressor and instead loads each fitted TabPFNRegressor from disk.
    """

    def __init__(self, estimators: list[Any], targets: list[str]) -> None:
        self.estimators = list(estimators)
        self.targets = [str(t) for t in (targets or [])]
        if not self.targets:
            raise ValueError("Empty targets list for TabPFN multi-output predictor.")
        if len(self.estimators) != len(self.targets):
            raise ValueError(
                f"Estimator count mismatch: got {len(self.estimators)} estimators for {len(self.targets)} targets."
            )

    def predict(self, X: "np.ndarray") -> "np.ndarray":
        import numpy as np

        preds: list[np.ndarray] = []
        for est in self.estimators:
            y = est.predict(X)
            y = np.asarray(y).reshape(-1, 1)
            preds.append(y)
        return np.concatenate(preds, axis=1)


def _load_json(path: str) -> dict:
    try:
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        return dict(obj or {}) if isinstance(obj, dict) else {}
    except Exception:
        return {}


def _resolve_fit_state_subdir(*, project_dir_abs: str, tabpfn: TabPFNSubmissionSettings, subdir: str) -> str:
    from src.inference.paths import resolve_path_best_effort

    raw = str(tabpfn.fit_state_dir or "tabpfn_fit").strip() or "tabpfn_fit"
    base = resolve_path_best_effort(project_dir_abs, raw)
    if os.path.isdir(base):
        return os.path.join(base, str(subdir))

    # Backward/packaged layout compatibility:
    # - running from repo root: artifacts live under <PROJECT_DIR>/weights/<raw>
    # - running inside packaged weights dir: artifacts live under <PROJECT_DIR>/<raw>
    alt = resolve_path_best_effort(project_dir_abs, os.path.join("weights", raw)) if not raw.startswith("weights") else ""
    if alt and os.path.isdir(alt):
        return os.path.join(alt, str(subdir))

    # Return the best-effort path (even if missing) for clearer error messages.
    return os.path.join(base, str(subdir))


def _load_tabpfn_predictor_from_fit_state(
    *,
    project_dir_abs: str,
    tabpfn: TabPFNSubmissionSettings,
    subdir: str,
    expected_targets: list[str],
) -> _TabPFNMultiOutputPredictor:
    """
    Load a packaged multi-output TabPFN predictor from:
        <project_dir>/<tabpfn.fit_state_dir>/<subdir>/
            meta.json (optional)
            <target>.tabpfn_fit  (one per target)
    """
    fit_dir = _resolve_fit_state_subdir(project_dir_abs=project_dir_abs, tabpfn=tabpfn, subdir=subdir)
    if not os.path.isdir(fit_dir):
        raise FileNotFoundError(
            "Packaged TabPFN fit-state directory not found. "
            f"Expected: {fit_dir} (did you run `package_artifacts.py` with TabPFN enabled?)"
        )

    meta = _load_json(os.path.join(fit_dir, "meta.json"))
    targets = meta.get("targets", None)
    if not isinstance(targets, list) or not targets:
        targets = list(expected_targets)
    targets = [str(t) for t in targets]

    # Determine file mapping
    files_map = meta.get("files", None)
    if not isinstance(files_map, dict):
        files_map = {}

    paths: list[str] = []
    for t in targets:
        cand = files_map.get(t, None)
        if isinstance(cand, str) and cand.strip():
            p = os.path.join(fit_dir, cand.strip()) if not os.path.isabs(cand) else cand
        else:
            p = os.path.join(fit_dir, f"{t}{_TABPFN_FIT_EXT}")
        if not os.path.isfile(p):
            # Backward-compat: allow "target__<name>.tabpfn_fit"
            p2 = os.path.join(fit_dir, f"target__{t}{_TABPFN_FIT_EXT}")
            if os.path.isfile(p2):
                p = p2
        if not os.path.isfile(p):
            raise FileNotFoundError(f"Missing TabPFN fit-state for target={t!r}: {p}")
        paths.append(p)

    # Ensure TabPFN is importable (offline-friendly) before loading fit states.
    TabPFNRegressor = _import_tabpfn_regressor(
        tabpfn_python_path=str(tabpfn.tabpfn_python_path or "").strip(),
        project_dir=project_dir_abs,
    )

    # TabPFN fit-states store `model_path` as-is; keep relative paths working by loading from project_dir.
    cwd = os.getcwd()
    try:
        os.chdir(project_dir_abs)
        estimators = [TabPFNRegressor.load_from_fit_state(p, device=str(tabpfn.device)) for p in paths]
    finally:
        try:
            os.chdir(cwd)
        except Exception:
            pass

    return _TabPFNMultiOutputPredictor(estimators=estimators, targets=targets)


def _add_import_root(path: str, *, package_name: str | None = None) -> None:
    """
    Add a directory to sys.path such that `import <package_name>` works.

    - If `path` is a *repo root* containing `<package_name>/__init__.py`, we add `path`.
    - If `path` is the *package directory* itself (basename == package_name), we add its parent.
    """
    p = os.path.abspath(path) if path else ""
    if not (p and os.path.isdir(p)):
        return

    root = p
    if package_name:
        pkg_init = os.path.join(p, package_name, "__init__.py")
        if os.path.isfile(pkg_init):
            root = p
        else:
            if os.path.basename(p) == package_name and os.path.isfile(os.path.join(p, "__init__.py")):
                root = os.path.dirname(p)

    if root and os.path.isdir(root) and root not in sys.path:
        sys.path.insert(0, root)


def _import_tabpfn_regressor(*, tabpfn_python_path: str = "", project_dir: str = "") -> type:
    """
    Import TabPFNRegressor (prefer installed package; fallback to vendored source path).
    """
    # Ensure the optional `tabpfn-common-utils` dependency is available (offline no-op shim)
    # before importing TabPFN.
    from src.tabular.tabpfn_patches import apply_tabpfn_runtime_patches, install_tabpfn_common_utils_shim

    install_tabpfn_common_utils_shim()

    try:
        from tabpfn import TabPFNRegressor  # type: ignore

        apply_tabpfn_runtime_patches()
        return TabPFNRegressor
    except Exception as e:
        # 1) Explicit user-provided source path (offline-friendly)
        if tabpfn_python_path:
            _add_import_root(tabpfn_python_path, package_name="tabpfn")
            from tabpfn import TabPFNRegressor  # type: ignore

            apply_tabpfn_runtime_patches()
            return TabPFNRegressor

        # 2) Auto-fallback: packaged/vendored TabPFN under project_dir/third_party/TabPFN/src
        project_dir_abs = os.path.abspath(project_dir) if project_dir else ""
        if project_dir_abs:
            vendored = os.path.join(project_dir_abs, "third_party", "TabPFN", "src")
            if os.path.isdir(vendored):
                _add_import_root(vendored, package_name="tabpfn")
                from tabpfn import TabPFNRegressor  # type: ignore

                apply_tabpfn_runtime_patches()
                return TabPFNRegressor

        raise e


def _resolve_local_tabpfn_ckpt(project_dir: str, p: str) -> str:
    from src.inference.paths import resolve_path_best_effort

    raw = str(p or "").strip()
    if not raw:
        raise RuntimeError("TabPFN weights_ckpt_path is empty.")

    abs_p = resolve_path_best_effort(project_dir, raw)
    if os.path.isfile(abs_p):
        return os.path.abspath(abs_p)
    if os.path.isdir(abs_p):
        # Accept a directory and pick a deterministic .ckpt inside.
        try:
            cands = [os.path.join(abs_p, n) for n in os.listdir(abs_p) if n.endswith(".ckpt")]
        except Exception:
            cands = []
        cands = [c for c in cands if os.path.isfile(c)]
        cands.sort()
        if len(cands) == 1:
            return os.path.abspath(cands[0])
        if len(cands) > 1:
            # Prefer the default v2.5 filename if present.
            for c in cands:
                if "tabpfn" in os.path.basename(c).lower() and "v2.5" in os.path.basename(c).lower():
                    return os.path.abspath(c)
            return os.path.abspath(cands[-1])
    raise FileNotFoundError(f"TabPFN checkpoint not found (local): {abs_p}")


def _derive_shard_cache_path(cache_path: str, shard_id: int) -> str:
    """
    Create a deterministic per-shard cache path from a base cache file path.
    """
    base = str(cache_path or "").strip()
    if not base:
        return base
    root, ext = os.path.splitext(base)
    ext_eff = ext if ext else ".pt"
    return f"{root}__shard{int(shard_id)}{ext_eff}"


def _extract_head_penultimate_features_2gpu_worker(q, shard_id: int, device_id: int, payload: dict) -> None:
    """
    Worker for 2-GPU data-parallel feature extraction (head penultimate).
    """

    def _fn(pl: dict) -> dict:
        import torch

        rels, feats_np = extract_head_penultimate_features(
            project_dir=pl["project_dir"],
            dataset_root=pl["dataset_root"],
            cfg_train_yaml=pl["cfg_train_yaml"],
            dino_weights_pt_file=pl["dino_weights_pt_file"],
            head_weights_pt_path=pl["head_weights_pt_path"],
            image_paths=pl["image_paths"],
            fusion=pl["fusion"],
            batch_size=int(pl["batch_size"]),
            num_workers=int(pl["num_workers"]),
            cache_path=pl.get("cache_path", None),
            image_size_hw=pl.get("image_size_hw", None),
            hflip=bool(pl.get("hflip", False)),
            vflip=bool(pl.get("vflip", False)),
        )
        feats_t = torch.from_numpy(feats_np).contiguous().cpu().float()
        return {"rels_in_order": list(rels), "features": feats_t}

    worker_guarded(q, shard_id, device_id, payload, fn=_fn)


def _extract_dinov3_cls_features_2gpu_worker(q, shard_id: int, device_id: int, payload: dict) -> None:
    """
    Worker for 2-GPU data-parallel feature extraction (DINOv3 CLS token).
    """

    def _fn(pl: dict) -> dict:
        import torch

        rels, feats_np = extract_dinov3_cls_features(
            project_dir=pl["project_dir"],
            dataset_root=pl["dataset_root"],
            cfg_train_yaml=pl["cfg_train_yaml"],
            dino_weights_pt_file=pl["dino_weights_pt_file"],
            image_paths=pl["image_paths"],
            batch_size=int(pl["batch_size"]),
            num_workers=int(pl["num_workers"]),
            cache_path=pl.get("cache_path", None),
            image_size_hw=pl.get("image_size_hw", None),
            hflip=bool(pl.get("hflip", False)),
            vflip=bool(pl.get("vflip", False)),
        )
        feats_t = torch.from_numpy(feats_np).contiguous().cpu().float()
        return {"rels_in_order": list(rels), "features": feats_t}

    worker_guarded(q, shard_id, device_id, payload, fn=_fn)


def extract_head_penultimate_features(
    *,
    project_dir: str,
    dataset_root: str,
    cfg_train_yaml: dict,
    dino_weights_pt_file: str,
    head_weights_pt_path: str,
    image_paths: List[str],
    fusion: str,
    batch_size: int,
    num_workers: int,
    cache_path: Optional[str],
    image_size_hw: Optional[Tuple[int, int]] = None,
    hflip: bool = False,
    vflip: bool = False,
) -> Tuple[List[str], np.ndarray]:
    """
    Extract head-penultimate (pre-final-linear) features for a list of image rel paths.

    Returns:
        (rels_in_order, features_np) where features_np is a numpy array (N, D).
    """
    import numpy as np
    import torch
    from torch import nn
    from torch.utils.data import DataLoader

    from src.inference.data import TestImageDataset, build_transforms
    from src.inference.paths import resolve_path_best_effort
    from src.inference.torch_load import load_head_state
    from src.models.backbone import build_feature_extractor
    from src.models.head_builder import DualBranchHeadExport, MultiLayerHeadExport, build_head_layer
    from src.models.vitdet_head import ViTDetHeadConfig, ViTDetMultiLayerScalarHead, ViTDetScalarHead

    fusion_eff = str(fusion or "mean").strip().lower()
    if fusion_eff not in ("mean", "concat"):
        raise ValueError(f"Unsupported feature_fusion: {fusion_eff!r} (expected 'mean' or 'concat')")

    project_dir_abs = os.path.abspath(project_dir) if project_dir else ""
    dataset_root_abs = os.path.abspath(dataset_root) if dataset_root else ""

    # Resolve + load head weights (accept file or directory).
    head_base = str(head_weights_pt_path or "").strip()
    if not head_base:
        raise FileNotFoundError("head_weights_pt_path is empty (cannot load head weights).")
    head_base_abs = resolve_path_best_effort(project_dir_abs, head_base)

    head_candidates: List[str] = []
    if os.path.isfile(head_base_abs):
        head_candidates.append(head_base_abs)
    elif os.path.isdir(head_base_abs):
        # 1) Prefer packaged single-head file
        cand = os.path.join(head_base_abs, "infer_head.pt")
        if os.path.isfile(cand):
            head_candidates.append(cand)
        # 2) Per-fold packaged heads
        try:
            for name in sorted(os.listdir(head_base_abs)):
                if name.startswith("fold_"):
                    c = os.path.join(head_base_abs, name, "infer_head.pt")
                    if os.path.isfile(c):
                        head_candidates.append(c)
        except Exception:
            pass
        # 3) Any .pt directly under directory
        try:
            pts = [os.path.join(head_base_abs, n) for n in os.listdir(head_base_abs) if n.endswith(".pt")]
            pts.sort()
            head_candidates.extend([p for p in pts if p not in head_candidates])
        except Exception:
            pass
    else:
        raise FileNotFoundError(f"Head weights path not found: {head_base_abs}")

    head_pt: Optional[str] = None
    head_state: Optional[dict] = None
    head_meta: Optional[dict] = None
    peft_payload: Optional[dict] = None
    last_err: Optional[Exception] = None

    for cand in head_candidates:
        try:
            st, meta, peft = load_head_state(str(cand))
            head_pt = str(cand)
            head_state, head_meta, peft_payload = st, (meta if isinstance(meta, dict) else {}), peft  # type: ignore[assignment]
            break
        except Exception as e:
            last_err = e
            continue

    # Fallback: latest per-epoch head checkpoint from outputs/checkpoints/<version>/...
    if head_pt is None or head_state is None or head_meta is None:
        ver = str(cfg_train_yaml.get("version", "") or "").strip()
        fb_dirs: List[str] = []
        if ver and project_dir_abs:
            fb_dirs.append(os.path.join(project_dir_abs, "outputs", "checkpoints", ver, "train_all", "head"))
            fb_dirs.append(os.path.join(project_dir_abs, "outputs", "checkpoints", ver, "head"))

        def _epoch_num(path: str) -> int:
            import re

            m = re.search(r"head-epoch(\\d+)", os.path.basename(path))
            if not m:
                return -1
            try:
                return int(m.group(1))
            except Exception:
                return -1

        fb_files: List[str] = []
        for d in fb_dirs:
            if not os.path.isdir(d):
                continue
            try:
                for n in os.listdir(d):
                    if n.startswith("head-epoch") and n.endswith(".pt"):
                        fb_files.append(os.path.join(d, n))
            except Exception:
                continue

        fb_files = [p for p in fb_files if os.path.isfile(p)]
        fb_files.sort(key=lambda p: (_epoch_num(p), os.path.getmtime(p) if os.path.exists(p) else 0.0))

        for cand in reversed(fb_files):
            try:
                st, meta, peft = load_head_state(str(cand))
                head_pt = str(cand)
                head_state, head_meta, peft_payload = st, (meta if isinstance(meta, dict) else {}), peft  # type: ignore[assignment]
                print(f"[TABPFN][WARN] Falling back to loadable head checkpoint: {head_pt}")
                break
            except Exception as e:
                last_err = e
                continue

    if head_pt is None or head_state is None or head_meta is None:
        raise RuntimeError(
            "Failed to load any head checkpoint. "
            f"head_weights_pt_path={head_weights_pt_path!r} resolved={head_base_abs!r}. "
            f"Last error: {last_err}"
        )

    head_meta = dict(head_meta or {})
    head_type = str(head_meta.get("head_type", "mlp") or "mlp").strip().lower()
    if head_type not in ("mlp", "vitdet"):
        raise RuntimeError(
            f"Unsupported head_type={head_type!r} for TabPFN penultimate feature extraction. "
            "Currently supported: mlp, vitdet."
        )

    backbone_name = str(cfg_train_yaml.get("model", {}).get("backbone", "") or "").strip()
    if not backbone_name:
        raise RuntimeError("configs/train.yaml missing model.backbone")

    # Preprocessing (reuse train.yaml to match model training, unless overridden for TTA).
    if image_size_hw is not None:
        try:
            image_size = (int(image_size_hw[0]), int(image_size_hw[1]))  # (H,W)
        except Exception:
            image_size = (int(image_size_hw[0]), int(image_size_hw[0]))  # type: ignore[index]
    else:
        image_size_raw = cfg_train_yaml.get("data", {}).get("image_size", 224)
        try:
            if isinstance(image_size_raw, (list, tuple)) and len(image_size_raw) == 2:
                image_size = (int(image_size_raw[1]), int(image_size_raw[0]))  # (H,W)
            else:
                v = int(image_size_raw)
                image_size = (v, v)
        except Exception:
            v = int(image_size_raw)
            image_size = (v, v)
    mean = list(cfg_train_yaml.get("data", {}).get("normalization", {}).get("mean", [0.485, 0.456, 0.406]))
    std = list(cfg_train_yaml.get("data", {}).get("normalization", {}).get("std", [0.229, 0.224, 0.225]))

    # Optional cache
    cache_path_eff = str(cache_path).strip() if isinstance(cache_path, str) and str(cache_path).strip() else ""
    if cache_path_eff and project_dir_abs:
        cache_path_eff = resolve_path_best_effort(project_dir_abs, cache_path_eff)
        if os.path.isfile(cache_path_eff):
            try:
                obj = torch.load(cache_path_eff, map_location="cpu")
                if isinstance(obj, dict) and "image_paths" in obj and "features" in obj:
                    cached_paths = list(obj["image_paths"])
                    feats = obj["features"]
                    cached_meta = dict(obj.get("meta", {}) or {})
                    if (
                        cached_paths == list(image_paths)
                        and isinstance(feats, torch.Tensor)
                        and feats.dim() == 2
                        and str(cached_meta.get("head_weights_path", "")) == str(head_pt)
                        and str(cached_meta.get("dino_weights_pt_file", "")) == str(dino_weights_pt_file)
                        and str(cached_meta.get("fusion", "")) == str(fusion_eff)
                        and tuple(cached_meta.get("image_size", ())) == tuple(image_size)
                        and bool(cached_meta.get("hflip", False)) == bool(hflip)
                        and bool(cached_meta.get("vflip", False)) == bool(vflip)
                    ):
                        return cached_paths, feats.cpu().numpy()
            except Exception:
                pass

    # ==========================================================
    # 2-GPU data-parallel feature extraction (each GPU runs independent images)
    #
    # - Only used when >=2 GPUs are available AND cache miss (handled above).
    # - Uses spawn+2 processes with even/odd sharding to preserve order on merge.
    # - Writes per-shard cache files and then writes the combined cache (same schema).
    # ==========================================================
    try:
        can_two_gpu = bool(two_gpu_parallel_enabled()) and int(len(image_paths)) >= 2
    except Exception:
        can_two_gpu = False
    if can_two_gpu:
        idx0, idx1 = split_even_odd_indices(int(len(image_paths)))
        if idx0 and idx1:
            image_paths0 = [image_paths[i] for i in idx0]
            image_paths1 = [image_paths[i] for i in idx1]
            cache0 = _derive_shard_cache_path(cache_path_eff, 0) if cache_path_eff else ""
            cache1 = _derive_shard_cache_path(cache_path_eff, 1) if cache_path_eff else ""

            payload_common = {
                "project_dir": str(project_dir_abs),
                "dataset_root": str(dataset_root_abs),
                "cfg_train_yaml": cfg_train_yaml,
                "dino_weights_pt_file": str(dino_weights_pt_file),
                "head_weights_pt_path": str(head_weights_pt_path),
                "fusion": str(fusion_eff),
                "batch_size": int(batch_size),
                "num_workers": int(num_workers),
                "image_size_hw": tuple(image_size) if isinstance(image_size, (list, tuple)) else None,
                "hflip": bool(hflip),
                "vflip": bool(vflip),
            }

            try:
                res0, res1 = run_two_processes_spawn(
                    worker=_extract_head_penultimate_features_2gpu_worker,
                    payload0={**payload_common, "image_paths": image_paths0, "cache_path": (cache0 if cache0 else None)},
                    payload1={**payload_common, "image_paths": image_paths1, "cache_path": (cache1 if cache1 else None)},
                    device0=0,
                    device1=1,
                )
            except Exception as e:
                print(f"[TABPFN][WARN] 2-GPU feature extraction disabled (fallback to single process): {e}")
            else:
                rels0 = [str(r) for r in (res0.get("rels_in_order", []) or [])]
                rels1 = [str(r) for r in (res1.get("rels_in_order", []) or [])]
                if rels0 != list(image_paths0):
                    raise RuntimeError("2-GPU shard0 feature extraction order mismatch (head_penultimate).")
                if rels1 != list(image_paths1):
                    raise RuntimeError("2-GPU shard1 feature extraction order mismatch (head_penultimate).")

                feats0 = res0.get("features", None)
                feats1 = res1.get("features", None)
                if not (isinstance(feats0, torch.Tensor) and feats0.dim() == 2):
                    raise RuntimeError(
                        f"2-GPU shard0 invalid features tensor: {type(feats0)} shape={getattr(feats0, 'shape', None)}"
                    )
                if not (isinstance(feats1, torch.Tensor) and feats1.dim() == 2):
                    raise RuntimeError(
                        f"2-GPU shard1 invalid features tensor: {type(feats1)} shape={getattr(feats1, 'shape', None)}"
                    )
                if int(feats0.shape[0]) != int(len(image_paths0)) or int(feats1.shape[0]) != int(len(image_paths1)):
                    raise RuntimeError("2-GPU feature extraction N mismatch across shards (head_penultimate).")
                if int(feats0.shape[1]) != int(feats1.shape[1]):
                    raise RuntimeError(
                        f"2-GPU feature extraction D mismatch across shards (head_penultimate): {int(feats0.shape[1])} vs {int(feats1.shape[1])}"
                    )

                N = int(len(image_paths))
                D = int(feats0.shape[1])
                features_full = torch.empty((N, D), dtype=torch.float32)
                features_full[idx0] = feats0.detach().cpu().float()
                features_full[idx1] = feats1.detach().cpu().float()

                if cache_path_eff:
                    try:
                        os.makedirs(os.path.dirname(cache_path_eff), exist_ok=True)
                        torch.save(
                            {
                                "image_paths": list(image_paths),
                                "features": features_full.cpu(),
                                "meta": {
                                    "head_type": head_type,
                                    "head_weights_path": str(head_pt),
                                    "dino_weights_pt_file": str(dino_weights_pt_file),
                                    "fusion": str(fusion_eff),
                                    "image_size": tuple(image_size),
                                    "hflip": bool(hflip),
                                    "vflip": bool(vflip),
                                    "two_gpu_data_parallel": True,
                                },
                            },
                            cache_path_eff,
                        )
                        print(f"[TABPFN] Saved feature cache -> {cache_path_eff}")
                    except Exception as e:
                        print(f"[TABPFN][WARN] Saving feature cache failed: {e}")

                return list(image_paths), features_full.cpu().numpy()

    tf = build_transforms(image_size=image_size, mean=mean, std=std, hflip=bool(hflip), vflip=bool(vflip))
    ds = TestImageDataset(list(image_paths), root_dir=str(dataset_root_abs), transform=tf)
    dl = DataLoader(
        ds,
        batch_size=max(1, int(batch_size)),
        shuffle=False,
        num_workers=max(0, int(num_workers)),
        pin_memory=torch.cuda.is_available(),
    )

    # Build extractor (weights_path forces pretrained=False inside helper; offline-friendly)
    feature_extractor = build_feature_extractor(
        backbone_name=backbone_name,
        pretrained=bool(cfg_train_yaml.get("model", {}).get("pretrained", True)),
        weights_url=str(cfg_train_yaml.get("model", {}).get("weights_url", "") or "") or None,
        weights_path=str(dino_weights_pt_file),
        gradient_checkpointing=bool(cfg_train_yaml.get("model", {}).get("gradient_checkpointing", False)),
    )

    # Inject per-head LoRA adapters (optional) so features match the specified trained model.
    if peft_payload is not None and isinstance(peft_payload, dict):
        peft_cfg_dict = peft_payload.get("config", None)
        peft_state = peft_payload.get("state_dict", None)
        if peft_cfg_dict and peft_state:
            try:
                from peft.config import PeftConfig  # type: ignore
                from peft.mapping_func import get_peft_model  # type: ignore
                from peft.utils.save_and_load import set_peft_model_state_dict  # type: ignore
            except Exception:
                from src.models.peft_integration import _import_peft

                _import_peft()
                from peft.config import PeftConfig  # type: ignore
                from peft.mapping_func import get_peft_model  # type: ignore
                from peft.utils.save_and_load import set_peft_model_state_dict  # type: ignore

            peft_config = PeftConfig.from_peft_type(**peft_cfg_dict)
            feature_extractor.backbone = get_peft_model(feature_extractor.backbone, peft_config)  # type: ignore[assignment]
            set_peft_model_state_dict(feature_extractor.backbone, peft_state, adapter_name="default")  # type: ignore[arg-type]
            feature_extractor.backbone.eval()

    # Build head module from meta (must match exported state_dict).
    head_module: nn.Module
    if head_type == "vitdet":
        use_layerwise_heads = bool(head_meta.get("use_layerwise_heads", False))
        layer_indices = list(head_meta.get("backbone_layer_indices", []))
        num_layers_eff = max(1, len(layer_indices)) if use_layerwise_heads else 1
        vitdet_cfg = ViTDetHeadConfig(
            embedding_dim=int(head_meta.get("embedding_dim", int(cfg_train_yaml.get("model", {}).get("embedding_dim", 1024)))),
            vitdet_dim=int(head_meta.get("vitdet_dim", int(cfg_train_yaml.get("model", {}).get("head", {}).get("vitdet_dim", 256)))),
            scale_factors=list(
                head_meta.get(
                    "vitdet_scale_factors",
                    cfg_train_yaml.get("model", {}).get("head", {}).get("vitdet_scale_factors", [2.0, 1.0, 0.5]),
                )
            ),
            patch_size=int(head_meta.get("vitdet_patch_size", int(cfg_train_yaml.get("model", {}).get("head", {}).get("vitdet_patch_size", 16)))),
            num_outputs_main=int(head_meta.get("num_outputs_main", 1)),
            num_outputs_ratio=int(head_meta.get("num_outputs_ratio", 0)),
            enable_ndvi=bool(head_meta.get("enable_ndvi", False)),
            separate_ratio_head=bool(head_meta.get("separate_ratio_head", False)),
            separate_ratio_spatial_head=bool(head_meta.get("separate_ratio_spatial_head", False)),
            head_hidden_dims=list(head_meta.get("head_hidden_dims", [])),
            head_activation=str(head_meta.get("head_activation", "relu")),
            dropout=float(head_meta.get("head_dropout", 0.0)),
        )
        if use_layerwise_heads:
            fusion_mode = str(head_meta.get("backbone_layers_fusion", head_meta.get("layer_fusion", "mean")) or "mean").strip().lower()
            head_module = ViTDetMultiLayerScalarHead(vitdet_cfg, num_layers=num_layers_eff, layer_fusion=fusion_mode)
        else:
            head_module = ViTDetScalarHead(vitdet_cfg)
    else:
        use_patch_reg3 = bool(head_meta.get("use_patch_reg3", False))
        use_cls_token = bool(head_meta.get("use_cls_token", True))
        dual_branch_enabled = bool(head_meta.get("dual_branch_enabled", False))
        use_layerwise_heads = bool(head_meta.get("use_layerwise_heads", False))
        layer_indices = list(head_meta.get("backbone_layer_indices", []))
        use_separate_bottlenecks = bool(head_meta.get("use_separate_bottlenecks", False))
        num_layers_eff = max(1, len(layer_indices)) if use_layerwise_heads else 1

        head_total = int(
            head_meta.get(
                "head_total_outputs",
                int(head_meta.get("num_outputs_main", 1)) + int(head_meta.get("num_outputs_ratio", 0)),
            )
        )
        embedding_dim = int(head_meta.get("embedding_dim", int(cfg_train_yaml.get("model", {}).get("embedding_dim", 1024))))
        head_hidden_dims = list(head_meta.get("head_hidden_dims", []))
        head_activation = str(head_meta.get("head_activation", "relu"))
        head_dropout = float(head_meta.get("head_dropout", 0.0))

        if dual_branch_enabled and use_patch_reg3:
            try:
                alpha_init = float(head_meta.get("dual_branch_alpha_init", 0.2))
            except Exception:
                alpha_init = 0.2
            head_module = DualBranchHeadExport(
                embedding_dim=embedding_dim,
                num_outputs_main=int(head_meta.get("num_outputs_main", 1)),
                num_outputs_ratio=int(head_meta.get("num_outputs_ratio", 0)),
                head_hidden_dims=head_hidden_dims,
                head_activation=head_activation,
                dropout=head_dropout,
                use_cls_token=use_cls_token,
                num_layers=num_layers_eff,
                alpha_init=float(alpha_init),
            )
        elif use_layerwise_heads and use_separate_bottlenecks:
            head_module = MultiLayerHeadExport(
                embedding_dim=embedding_dim,
                num_outputs_main=int(head_meta.get("num_outputs_main", 1)),
                num_outputs_ratio=int(head_meta.get("num_outputs_ratio", 0)),
                head_hidden_dims=head_hidden_dims,
                head_activation=head_activation,
                dropout=head_dropout,
                use_patch_reg3=use_patch_reg3,
                use_cls_token=use_cls_token,
                num_layers=num_layers_eff,
            )
        else:
            head_module = build_head_layer(
                embedding_dim=embedding_dim,
                num_outputs=head_total if not use_layerwise_heads else head_total * num_layers_eff,
                head_hidden_dims=head_hidden_dims,
                head_activation=head_activation,
                dropout=head_dropout,
                use_output_softplus=False,
                input_dim=embedding_dim if (use_patch_reg3 or (not use_cls_token)) else None,
            )

    head_module.load_state_dict(head_state, strict=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    feature_extractor = feature_extractor.eval().to(device)
    head_module = head_module.eval().to(device)

    def _penultimate_from_sequential(head_seq: nn.Sequential, x: torch.Tensor) -> torch.Tensor:
        mods = list(head_seq)
        last_lin = None
        for i in range(len(mods) - 1, -1, -1):
            if isinstance(mods[i], nn.Linear):
                last_lin = i
                break
        if last_lin is None:
            raise RuntimeError("Cannot find final Linear layer in MLP head module.")
        z = x
        for i, m in enumerate(mods):
            if i == last_lin:
                break
            z = m(z)
        return z

    def _fuse_layers(z_list: List[torch.Tensor], *, weights: Optional[torch.Tensor]) -> torch.Tensor:
        if not z_list:
            raise RuntimeError("Empty z_list for fusion")
        if fusion_eff == "concat":
            return torch.cat(z_list, dim=-1)
        if weights is None:
            return torch.stack(z_list, dim=0).mean(dim=0)
        w = weights.to(device=z_list[0].device, dtype=z_list[0].dtype)
        w = w / w.sum().clamp_min(1e-8)
        while w.dim() < z_list[0].dim() + 1:
            w = w.view(*w.shape, 1)
        stacked = torch.stack(z_list, dim=0)
        return (w * stacked).sum(dim=0)

    feats_cpu: List[torch.Tensor] = []
    rels: List[str] = []
    with torch.inference_mode():
        for images, rel_paths in dl:
            images = images.to(device, non_blocking=True)
            H = int(images.shape[-2])
            W = int(images.shape[-1])

            if head_type == "vitdet":
                use_layerwise_heads = bool(head_meta.get("use_layerwise_heads", False))
                layer_indices = list(head_meta.get("backbone_layer_indices", []))
                if use_layerwise_heads and layer_indices:
                    _cls_list, pt_list = feature_extractor.forward_layers_cls_and_tokens(images, layer_indices)  # type: ignore[attr-defined]
                    out = head_module(pt_list, image_hw=(H, W))  # type: ignore[call-arg]
                    z_layers = out.get("z_layers", None) if isinstance(out, dict) else None
                    z = out.get("z", None) if isinstance(out, dict) else None
                    if fusion_eff == "concat" and isinstance(z_layers, list) and z_layers:
                        feats = torch.cat([t for t in z_layers if isinstance(t, torch.Tensor)], dim=-1)
                    else:
                        if not isinstance(z, torch.Tensor):
                            raise RuntimeError("ViTDet head did not return 'z'")
                        feats = z
                else:
                    _cls, pt = feature_extractor.forward_cls_and_tokens(images)  # type: ignore[attr-defined]
                    out = head_module(pt, image_hw=(H, W))  # type: ignore[call-arg]
                    z = out.get("z", None) if isinstance(out, dict) else None
                    if not isinstance(z, torch.Tensor):
                        raise RuntimeError("ViTDet head did not return 'z'")
                    feats = z
            else:
                use_patch_reg3 = bool(head_meta.get("use_patch_reg3", False))
                use_cls_token = bool(head_meta.get("use_cls_token", True))
                dual_branch_enabled = bool(head_meta.get("dual_branch_enabled", False))
                use_layerwise_heads = bool(head_meta.get("use_layerwise_heads", False))
                layer_indices = list(head_meta.get("backbone_layer_indices", []))
                use_separate_bottlenecks = bool(head_meta.get("use_separate_bottlenecks", False))

                weights = None
                fusion_mode_meta = str(
                    head_meta.get("backbone_layers_fusion", head_meta.get("layer_fusion", "mean")) or "mean"
                ).strip().lower()
                if use_layerwise_heads and fusion_eff == "mean" and fusion_mode_meta == "learned":
                    logits_meta = head_meta.get("mlp_layer_logits", None)
                    if isinstance(logits_meta, (list, tuple)) and len(logits_meta) == len(layer_indices):
                        try:
                            logits_t = torch.tensor([float(x) for x in logits_meta], device=device, dtype=torch.float32)
                            weights = torch.softmax(logits_t, dim=0)
                        except Exception:
                            weights = None

                if use_layerwise_heads and layer_indices:
                    cls_list, pt_list = feature_extractor.forward_layers_cls_and_tokens(images, layer_indices)  # type: ignore[attr-defined]
                    z_list: List[torch.Tensor] = []
                    for l_idx, (cls_l, pt_l) in enumerate(zip(cls_list, pt_list)):
                        if use_patch_reg3:
                            if pt_l.dim() != 3:
                                raise RuntimeError(f"Unexpected patch token shape: {tuple(pt_l.shape)}")
                            B, N, C = pt_l.shape
                            if dual_branch_enabled and isinstance(head_module, DualBranchHeadExport):
                                # Dual-branch: fuse patch-bottleneck features with global-bottleneck features.
                                z_flat = head_module.layer_bottlenecks_patch[l_idx](pt_l.reshape(B * N, C).to(device))  # type: ignore[attr-defined]
                                z_patch = z_flat.view(B, N, -1).mean(dim=1)
                                patch_mean = pt_l.mean(dim=1)
                                feats_l = torch.cat([cls_l, patch_mean], dim=-1) if use_cls_token else patch_mean
                                z_global = head_module.layer_bottlenecks_global[l_idx](feats_l.to(device))  # type: ignore[attr-defined]
                                a = torch.sigmoid(head_module.alpha_logit).to(device=z_patch.device, dtype=z_patch.dtype)  # type: ignore[attr-defined]
                                z_l = (a * z_global) + ((1.0 - a) * z_patch)
                            elif use_separate_bottlenecks and isinstance(head_module, MultiLayerHeadExport):
                                bottleneck = head_module.layer_bottlenecks[l_idx]
                                z_flat = bottleneck(pt_l.reshape(B * N, C).to(device))
                                z_l = z_flat.view(B, N, -1).mean(dim=1)
                            else:
                                if not isinstance(head_module, nn.Sequential):
                                    raise RuntimeError("Expected nn.Sequential MLP head for packed layerwise path.")
                                z_flat = _penultimate_from_sequential(head_module, pt_l.reshape(B * N, C).to(device))
                                z_l = z_flat.view(B, N, -1).mean(dim=1)
                        else:
                            patch_mean = pt_l.mean(dim=1)
                            feats_l = torch.cat([cls_l, patch_mean], dim=-1) if use_cls_token else patch_mean
                            if use_separate_bottlenecks and isinstance(head_module, MultiLayerHeadExport):
                                bottleneck = head_module.layer_bottlenecks[l_idx]
                                z_l = bottleneck(feats_l.to(device))
                            else:
                                if not isinstance(head_module, nn.Sequential):
                                    raise RuntimeError("Expected nn.Sequential MLP head for packed layerwise path.")
                                z_l = _penultimate_from_sequential(head_module, feats_l.to(device))
                        z_list.append(z_l)
                    feats = _fuse_layers(z_list, weights=weights)
                else:
                    cls, pt = feature_extractor.forward_cls_and_tokens(images)  # type: ignore[attr-defined]
                    if use_patch_reg3:
                        if pt.dim() != 3:
                            raise RuntimeError(f"Unexpected patch token shape: {tuple(pt.shape)}")
                        B, N, C = pt.shape
                        if dual_branch_enabled and isinstance(head_module, DualBranchHeadExport):
                            z_flat = head_module.layer_bottlenecks_patch[0](pt.reshape(B * N, C).to(device))  # type: ignore[attr-defined]
                            z_patch = z_flat.view(B, N, -1).mean(dim=1)
                            patch_mean = pt.mean(dim=1)
                            feats_in = torch.cat([cls, patch_mean], dim=-1) if use_cls_token else patch_mean
                            z_global = head_module.layer_bottlenecks_global[0](feats_in.to(device))  # type: ignore[attr-defined]
                            a = torch.sigmoid(head_module.alpha_logit).to(device=z_patch.device, dtype=z_patch.dtype)  # type: ignore[attr-defined]
                            feats = (a * z_global) + ((1.0 - a) * z_patch)
                        elif isinstance(head_module, nn.Sequential):
                            z_flat = _penultimate_from_sequential(head_module, pt.reshape(B * N, C).to(device))
                            feats = z_flat.view(B, N, -1).mean(dim=1)
                        else:
                            raise RuntimeError("Expected nn.Sequential MLP head in single-layer patch-mode.")
                    else:
                        patch_mean = pt.mean(dim=1)
                        feats_in = torch.cat([cls, patch_mean], dim=-1) if use_cls_token else patch_mean
                        if isinstance(head_module, nn.Sequential):
                            feats = _penultimate_from_sequential(head_module, feats_in.to(device))
                        else:
                            raise RuntimeError("Expected nn.Sequential MLP head in single-layer global mode.")

            feats_cpu.append(feats.detach().cpu().float())
            rels.extend(list(rel_paths))

    if rels != list(image_paths):
        raise RuntimeError("Feature extraction order mismatch (unexpected dataloader ordering).")

    features = torch.cat(feats_cpu, dim=0) if feats_cpu else torch.empty((0, 0), dtype=torch.float32)
    if int(features.shape[0]) != int(len(image_paths)):
        raise RuntimeError(f"Feature extraction produced wrong N: got {features.shape[0]}, expected {len(image_paths)}")

    feats_np = features.cpu().numpy().astype(np.float32, copy=False)

    if cache_path_eff:
        try:
            os.makedirs(os.path.dirname(cache_path_eff), exist_ok=True)
            torch.save(
                {
                    "image_paths": list(image_paths),
                    "features": features.cpu(),
                    "meta": {
                        "head_type": head_type,
                        "head_weights_path": str(head_pt),
                        "dino_weights_pt_file": str(dino_weights_pt_file),
                        "fusion": str(fusion_eff),
                        "image_size": tuple(image_size),
                        "hflip": bool(hflip),
                        "vflip": bool(vflip),
                    },
                },
                cache_path_eff,
            )
            print(f"[TABPFN] Saved feature cache -> {cache_path_eff}")
        except Exception as e:
            print(f"[TABPFN][WARN] Saving feature cache failed: {e}")

    return list(image_paths), feats_np


def extract_dinov3_cls_features(
    *,
    project_dir: str,
    dataset_root: str,
    cfg_train_yaml: dict,
    dino_weights_pt_file: str,
    image_paths: List[str],
    batch_size: int,
    num_workers: int,
    cache_path: Optional[str],
    image_size_hw: Optional[Tuple[int, int]] = None,
    hflip: bool = False,
    vflip: bool = False,
) -> Tuple[List[str], np.ndarray]:
    """
    Extract frozen DINOv3 CLS token features for a list of image rel paths.

    This matches `configs/train_tabpfn.yaml` / `tabpfn_train.py` mode: `dinov3_only`:
    - no LoRA injection
    - no regression head usage
    - TabPFN X is the backbone CLS token
    """
    import numpy as np
    import torch
    from torch.utils.data import DataLoader

    from src.inference.data import TestImageDataset, build_transforms
    from src.inference.paths import resolve_path_best_effort
    from src.models.backbone import build_feature_extractor

    project_dir_abs = os.path.abspath(project_dir) if project_dir else ""
    dataset_root_abs = os.path.abspath(dataset_root) if dataset_root else ""

    backbone_name = str(cfg_train_yaml.get("model", {}).get("backbone", "") or "").strip()
    if not backbone_name:
        raise RuntimeError("configs/train.yaml missing model.backbone")

    # Preprocessing (reuse train.yaml to match model training, unless overridden for TTA).
    if image_size_hw is not None:
        try:
            image_size = (int(image_size_hw[0]), int(image_size_hw[1]))  # (H,W)
        except Exception:
            image_size = (int(image_size_hw[0]), int(image_size_hw[0]))  # type: ignore[index]
    else:
        image_size_raw = cfg_train_yaml.get("data", {}).get("image_size", 224)
        try:
            if isinstance(image_size_raw, (list, tuple)) and len(image_size_raw) == 2:
                image_size = (int(image_size_raw[1]), int(image_size_raw[0]))  # (H,W)
            else:
                v = int(image_size_raw)
                image_size = (v, v)
        except Exception:
            v = int(image_size_raw)
            image_size = (v, v)
    mean = list(cfg_train_yaml.get("data", {}).get("normalization", {}).get("mean", [0.485, 0.456, 0.406]))
    std = list(cfg_train_yaml.get("data", {}).get("normalization", {}).get("std", [0.229, 0.224, 0.225]))

    # Optional cache
    cache_path_eff = str(cache_path).strip() if isinstance(cache_path, str) and str(cache_path).strip() else ""
    if cache_path_eff and project_dir_abs:
        cache_path_eff = resolve_path_best_effort(project_dir_abs, cache_path_eff)
        if os.path.isfile(cache_path_eff):
            try:
                obj = torch.load(cache_path_eff, map_location="cpu")
                if isinstance(obj, dict) and "image_paths" in obj and "features" in obj:
                    cached_paths = list(obj["image_paths"])
                    feats = obj["features"]
                    cached_meta = dict(obj.get("meta", {}) or {})
                    if (
                        cached_paths == list(image_paths)
                        and isinstance(feats, torch.Tensor)
                        and feats.dim() == 2
                        and str(cached_meta.get("mode", "")) == "dinov3_only"
                        and str(cached_meta.get("backbone", "")) == str(backbone_name)
                        and str(cached_meta.get("weights_path", "")) == str(dino_weights_pt_file)
                        and tuple(cached_meta.get("image_size", ())) == tuple(image_size)
                        and bool(cached_meta.get("hflip", False)) == bool(hflip)
                        and bool(cached_meta.get("vflip", False)) == bool(vflip)
                    ):
                        return cached_paths, feats.cpu().numpy()
            except Exception:
                pass

    # ==========================================================
    # 2-GPU data-parallel feature extraction (each GPU runs independent images)
    #
    # - Only used when >=2 GPUs are available AND cache miss (handled above).
    # - Uses spawn+2 processes with even/odd sharding to preserve order on merge.
    # - Writes per-shard cache files and then writes the combined cache (same schema).
    # ==========================================================
    try:
        can_two_gpu = bool(two_gpu_parallel_enabled()) and int(len(image_paths)) >= 2
    except Exception:
        can_two_gpu = False
    if can_two_gpu:
        idx0, idx1 = split_even_odd_indices(int(len(image_paths)))
        if idx0 and idx1:
            image_paths0 = [image_paths[i] for i in idx0]
            image_paths1 = [image_paths[i] for i in idx1]
            cache0 = _derive_shard_cache_path(cache_path_eff, 0) if cache_path_eff else ""
            cache1 = _derive_shard_cache_path(cache_path_eff, 1) if cache_path_eff else ""

            payload_common = {
                "project_dir": str(project_dir_abs),
                "dataset_root": str(dataset_root_abs),
                "cfg_train_yaml": cfg_train_yaml,
                "dino_weights_pt_file": str(dino_weights_pt_file),
                "batch_size": int(batch_size),
                "num_workers": int(num_workers),
                "image_size_hw": tuple(image_size) if isinstance(image_size, (list, tuple)) else None,
                "hflip": bool(hflip),
                "vflip": bool(vflip),
            }

            try:
                res0, res1 = run_two_processes_spawn(
                    worker=_extract_dinov3_cls_features_2gpu_worker,
                    payload0={**payload_common, "image_paths": image_paths0, "cache_path": (cache0 if cache0 else None)},
                    payload1={**payload_common, "image_paths": image_paths1, "cache_path": (cache1 if cache1 else None)},
                    device0=0,
                    device1=1,
                )
            except Exception as e:
                print(f"[TABPFN][WARN] 2-GPU feature extraction disabled (fallback to single process): {e}")
            else:
                rels0 = [str(r) for r in (res0.get("rels_in_order", []) or [])]
                rels1 = [str(r) for r in (res1.get("rels_in_order", []) or [])]
                if rels0 != list(image_paths0):
                    raise RuntimeError("2-GPU shard0 feature extraction order mismatch (dinov3_only).")
                if rels1 != list(image_paths1):
                    raise RuntimeError("2-GPU shard1 feature extraction order mismatch (dinov3_only).")

                feats0 = res0.get("features", None)
                feats1 = res1.get("features", None)
                if not (isinstance(feats0, torch.Tensor) and feats0.dim() == 2):
                    raise RuntimeError(
                        f"2-GPU shard0 invalid CLS features tensor: {type(feats0)} shape={getattr(feats0, 'shape', None)}"
                    )
                if not (isinstance(feats1, torch.Tensor) and feats1.dim() == 2):
                    raise RuntimeError(
                        f"2-GPU shard1 invalid CLS features tensor: {type(feats1)} shape={getattr(feats1, 'shape', None)}"
                    )
                if int(feats0.shape[0]) != int(len(image_paths0)) or int(feats1.shape[0]) != int(len(image_paths1)):
                    raise RuntimeError("2-GPU CLS feature extraction N mismatch across shards (dinov3_only).")
                if int(feats0.shape[1]) != int(feats1.shape[1]):
                    raise RuntimeError(
                        f"2-GPU CLS feature extraction D mismatch across shards (dinov3_only): {int(feats0.shape[1])} vs {int(feats1.shape[1])}"
                    )

                N = int(len(image_paths))
                D = int(feats0.shape[1])
                features_full = torch.empty((N, D), dtype=torch.float32)
                features_full[idx0] = feats0.detach().cpu().float()
                features_full[idx1] = feats1.detach().cpu().float()

                if cache_path_eff:
                    try:
                        os.makedirs(os.path.dirname(cache_path_eff), exist_ok=True)
                        torch.save(
                            {
                                "image_paths": list(image_paths),
                                "features": features_full.cpu(),
                                "meta": {
                                    "mode": "dinov3_only",
                                    "backbone": str(backbone_name),
                                    "weights_path": str(dino_weights_pt_file),
                                    "image_size": tuple(image_size),
                                    "hflip": bool(hflip),
                                    "vflip": bool(vflip),
                                    "two_gpu_data_parallel": True,
                                },
                            },
                            cache_path_eff,
                        )
                        print(f"[TABPFN] Saved feature cache -> {cache_path_eff}")
                    except Exception as e:
                        print(f"[TABPFN][WARN] Saving feature cache failed: {e}")

                return list(image_paths), features_full.cpu().numpy()

    tf = build_transforms(image_size=image_size, mean=mean, std=std, hflip=bool(hflip), vflip=bool(vflip))
    ds = TestImageDataset(list(image_paths), root_dir=str(dataset_root_abs), transform=tf)
    dl = DataLoader(
        ds,
        batch_size=max(1, int(batch_size)),
        shuffle=False,
        num_workers=max(0, int(num_workers)),
        pin_memory=torch.cuda.is_available(),
    )

    print(
        f"[TABPFN] Extracting DINOv3 CLS features (dinov3_only): backbone={backbone_name} weights={dino_weights_pt_file} image_size={tuple(image_size)}"
    )

    # IMPORTANT: no LoRA injection here (plain frozen DINOv3 only)
    feature_extractor = build_feature_extractor(
        backbone_name=backbone_name,
        pretrained=bool(cfg_train_yaml.get("model", {}).get("pretrained", True)),
        weights_url=str(cfg_train_yaml.get("model", {}).get("weights_url", "") or "") or None,
        weights_path=str(dino_weights_pt_file),
        gradient_checkpointing=False,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    feature_extractor = feature_extractor.eval().to(device)

    feats_cpu: List[torch.Tensor] = []
    rels: List[str] = []
    with torch.inference_mode():
        for images, rel_paths in dl:
            images = images.to(device, non_blocking=True)
            cls, _pt = feature_extractor.forward_cls_and_tokens(images)  # type: ignore[attr-defined]
            feats_cpu.append(cls.detach().cpu().float())
            rels.extend(list(rel_paths))

    if rels != list(image_paths):
        raise RuntimeError("Feature extraction order mismatch (unexpected dataloader ordering).")

    features = torch.cat(feats_cpu, dim=0) if feats_cpu else torch.empty((0, 0), dtype=torch.float32)
    if int(features.shape[0]) != int(len(image_paths)):
        raise RuntimeError(f"Feature extraction produced wrong N: got {features.shape[0]}, expected {len(image_paths)}")

    feats_np = features.cpu().numpy().astype(np.float32, copy=False)

    if cache_path_eff:
        try:
            os.makedirs(os.path.dirname(cache_path_eff), exist_ok=True)
            torch.save(
                {
                    "image_paths": list(image_paths),
                    "features": features.cpu(),
                    "meta": {
                        "mode": "dinov3_only",
                        "backbone": str(backbone_name),
                        "weights_path": str(dino_weights_pt_file),
                        "image_size": tuple(image_size),
                        "hflip": bool(hflip),
                        "vflip": bool(vflip),
                    },
                },
                cache_path_eff,
            )
            print(f"[TABPFN] Saved feature cache -> {cache_path_eff}")
        except Exception as e:
            print(f"[TABPFN][WARN] Saving feature cache failed: {e}")

    return list(image_paths), feats_np


def _write_submission_csv(df_test, image_to_components: dict[str, dict[str, float]], output_path: str) -> None:
    rows = []
    for _, r in df_test.iterrows():
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


def run_tabpfn_submission(
    *,
    settings: InferenceSettings,
    tabpfn: TabPFNSubmissionSettings,
) -> None:
    """
    Offline submission generation using **packaged** TabPFN fit-state (no online training):
      1) Load test.csv
      2) Load fitted TabPFN state from <project_dir>/<tabpfn.fit_state_dir>/
      3) Extract image features for test (supports ensemble + TTA)
      4) Predict and write submission.csv
    """
    import pandas as pd

    from src.inference.data import resolve_paths
    from src.inference.ensemble import normalize_ensemble_models
    from src.inference.pipeline import (
        _resolve_tta_views,
        load_config,
        load_config_file,
        parse_image_size,
        resolve_dino_weights_path_for_model,
    )
    from src.inference.paths import (
        resolve_path_best_effort,
        resolve_version_head_base,
        resolve_version_train_yaml,
        safe_slug,
    )

    project_dir_abs = os.path.abspath(settings.project_dir) if settings.project_dir else ""
    if not (project_dir_abs and os.path.isdir(project_dir_abs)):
        raise RuntimeError("settings.project_dir must point to the repository root containing `configs/` and `src/`.")

    # Validate TabPFN checkpoint path (local-only)
    ckpt_path = _resolve_local_tabpfn_ckpt(project_dir_abs, tabpfn.weights_ckpt_path)

    # Ensure TabPFN env settings are applied BEFORE importing tabpfn
    if not bool(tabpfn.enable_telemetry):
        os.environ["TABPFN_DISABLE_TELEMETRY"] = "1"
    else:
        os.environ.pop("TABPFN_DISABLE_TELEMETRY", None)
    if str(tabpfn.model_cache_dir or "").strip():
        os.environ["TABPFN_MODEL_CACHE_DIR"] = str(resolve_path_best_effort(project_dir_abs, str(tabpfn.model_cache_dir)))

    # Project config (train.yaml) for backbone + transforms
    cfg = load_config(project_dir_abs)

    dataset_root, test_csv = resolve_paths(str(settings.input_path))
    df_test = pd.read_csv(test_csv)
    if not {"sample_id", "image_path", "target_name"}.issubset(df_test.columns):
        raise ValueError("test.csv must contain columns: sample_id, image_path, target_name")
    unique_test_paths = df_test["image_path"].astype(str).unique().tolist()

    # ==========================================================
    # Ensemble + TabPFN (feature-level) path
    #
    # Strategy (as requested):
    # - Extract + cache per-model features for ALL ensemble models
    # - Fit a SINGLE TabPFN model on stacked train features (repeat y per model)
    # - For test: run that single TabPFN on each model's test features and average predictions
    # ==========================================================
    ensemble_models = normalize_ensemble_models(project_dir_abs)
    if len(ensemble_models) > 0:
        import numpy as np

        feature_mode = str(getattr(tabpfn, "feature_mode", "head_penultimate") or "head_penultimate").strip().lower()
        if feature_mode not in ("head_penultimate", "dinov3_only"):
            raise ValueError(
                f"Unsupported TabPFN feature_mode: {feature_mode!r} (expected 'head_penultimate' or 'dinov3_only')"
            )

        def _derive_model_cache_tag(*, model_id: str, version: object) -> str:
            base = str(model_id or "").strip() or "model"
            if isinstance(version, (str, int, float)) and str(version).strip():
                base = str(version).strip()
            parts = [base, feature_mode]
            if feature_mode != "dinov3_only":
                parts.append(str(tabpfn.feature_fusion or "mean").strip().lower())
            return safe_slug("__".join(parts))

        def _resolve_cache_path_for_model(*, base_path: str, split: str, model_tag: str) -> str:
            """
            Resolve a per-model cache path.

            - If base_path is empty: caching disabled (returns empty string)
            - If base_path is a directory: write <dir>/<split>__<model_tag>.pt
            - If base_path is a file: insert __<model_tag> before extension
            """
            raw = str(base_path or "").strip()
            if not raw:
                return ""

            p = resolve_path_best_effort(project_dir_abs, raw)
            # Treat as directory if:
            # - user passed an existing directory
            # - user passed a path ending with a separator
            # - user passed a path without an extension (common for "directory-style" base paths)
            if (p.endswith(os.sep) or p.endswith("/")) or os.path.isdir(p) or (os.path.splitext(p)[1] == ""):
                d = p.rstrip("/").rstrip(os.sep)
                return os.path.join(d, f"{split}__{model_tag}.pt")

            root, ext = os.path.splitext(p)
            ext_eff = ext if ext else ".pt"
            return f"{root}__{model_tag}{ext_eff}"

        # Normalize ensemble model entries (reuse inference.pipeline's semantics).
        models_eff: list[dict] = []
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

            # Base image size for TTA view resolution (H,W)
            try:
                base_image_size_hw = parse_image_size(cfg_model.get("data", {}).get("image_size", 224))
            except Exception:
                base_image_size_hw = (224, 224)

            # Allow overriding backbone name at model level (optional)
            backbone_override = m.get("backbone", None)
            if isinstance(backbone_override, str) and backbone_override.strip():
                try:
                    if "model" not in cfg_model or not isinstance(cfg_model["model"], dict):
                        cfg_model["model"] = {}
                    cfg_model["model"]["backbone"] = backbone_override.strip()
                except Exception:
                    pass

            # Resolve head weights base path (unused for dinov3_only)
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
                    head_base = str(settings.head_weights_pt_path)

            # Resolve backbone weights file
            backbone_name_eff = str(cfg_model.get("model", {}).get("backbone", "") or "").strip()
            dino_weights_file_i = resolve_dino_weights_path_for_model(
                project_dir_abs,
                backbone_name=backbone_name_eff,
                cfg=cfg_model,
                model_cfg=m,
                global_dino_weights=str(settings.dino_weights_pt_path),
            )
            if not (dino_weights_file_i and os.path.isfile(dino_weights_file_i)):
                p0 = resolve_path_best_effort(project_dir_abs, str(settings.dino_weights_pt_path))
                if os.path.isfile(p0):
                    dino_weights_file_i = os.path.abspath(p0)
            if not (dino_weights_file_i and os.path.isfile(dino_weights_file_i)):
                raise FileNotFoundError(
                    "Cannot resolve DINO backbone weights file for an ensemble model: "
                    f"model_id={model_id!r} version={version!r} backbone={backbone_name_eff!r}"
                )

            model_tag = _derive_model_cache_tag(model_id=model_id, version=version)
            models_eff.append(
                {
                    "model_id": model_id,
                    "version": str(version).strip() if isinstance(version, (str, int, float)) else None,
                    "weight": float(model_weight),
                    "cfg_path": str(cfg_path),
                    "cfg": cfg_model,
                    "backbone": backbone_name_eff,
                    "base_image_size_hw": tuple(base_image_size_hw),
                    "dino_weights_file": str(dino_weights_file_i),
                    "head_base": str(head_base),
                    "model_tag": str(model_tag),
                }
            )

        if not models_eff:
            raise RuntimeError("Ensemble is enabled but no valid models were found for TabPFN.")

        print(f"[TABPFN][ENSEMBLE] Models: {len(models_eff)} (single TabPFN fit; feature-level averaging)")
        if feature_mode == "dinov3_only":
            print("[TABPFN][ENSEMBLE] Feature mode: dinov3_only (CLS token; no head)")
        else:
            print(f"[TABPFN][ENSEMBLE] Feature mode: head_penultimate (fusion={str(tabpfn.feature_fusion)})")

        # Load packaged ensemble fit-state (no online fitting).
        from src.metrics import TARGETS_5D_ORDER

        predictor = _load_tabpfn_predictor_from_fit_state(
            project_dir_abs=project_dir_abs,
            tabpfn=tabpfn,
            subdir="ensemble",
            expected_targets=list(TARGETS_5D_ORDER),
        )
        D_ref: Optional[int] = None

        # Predict test set:
        # - For each model: average predictions across its TTA views (if enabled).
        # - Then ensemble models by ensemble.json weights.
        test_image_paths = list(unique_test_paths)
        y_pred_sum: Optional[np.ndarray] = None
        w_sum: float = 0.0
        for mi in models_eff:
            cfg_i = mi["cfg"]
            dino_w_i = mi["dino_weights_file"]
            head_base_i = mi["head_base"]
            model_id_i = mi["model_id"]
            version_i = mi.get("version", None)
            w_i = float(mi.get("weight", 1.0))
            if not (w_i > 0.0):
                continue
            base_hw_i = tuple(mi.get("base_image_size_hw", (224, 224)))
            model_tag_i = str(mi.get("model_tag", "") or safe_slug(str(model_id_i)))

            print(
                f"[TABPFN][ENSEMBLE] Predict test feats: model_id={model_id_i!r} version={version_i!r} "
                f"backbone={mi['backbone']!r} weight={w_i}"
            )

            # TTA views are resolved per-model (base image_size differs across versions).
            tta_views = _resolve_tta_views(settings, base_image_size=base_hw_i, patch_multiple=16)
            multi_view = len(tta_views) > 1

            y_pred_model_sum: Optional[np.ndarray] = None
            n_views_eff: int = 0
            for (image_size_view, hflip_view, vflip_view) in tta_views:
                view_tag = safe_slug(
                    f"{model_tag_i}__tta_{int(image_size_view[0])}x{int(image_size_view[1])}__hf{int(bool(hflip_view))}__vf{int(bool(vflip_view))}"
                )
                cache_test_view = _resolve_cache_path_for_model(
                    base_path=str(tabpfn.feature_cache_path_test or ""),
                    split="test",
                    model_tag=(view_tag if multi_view else model_tag_i),
                )

                if feature_mode == "dinov3_only":
                    _rels_test_i, X_test_i = extract_dinov3_cls_features(
                        project_dir=project_dir_abs,
                        dataset_root=str(dataset_root),
                        cfg_train_yaml=cfg_i,
                        dino_weights_pt_file=str(dino_w_i),
                        image_paths=test_image_paths,
                        batch_size=int(tabpfn.feature_batch_size),
                        num_workers=int(tabpfn.feature_num_workers),
                        cache_path=str(cache_test_view),
                        image_size_hw=tuple(image_size_view),
                        hflip=bool(hflip_view),
                        vflip=bool(vflip_view),
                    )
                else:
                    _rels_test_i, X_test_i = extract_head_penultimate_features(
                        project_dir=project_dir_abs,
                        dataset_root=str(dataset_root),
                        cfg_train_yaml=cfg_i,
                        dino_weights_pt_file=str(dino_w_i),
                        head_weights_pt_path=str(head_base_i),
                        image_paths=test_image_paths,
                        fusion=str(tabpfn.feature_fusion),
                        batch_size=int(tabpfn.feature_batch_size),
                        num_workers=int(tabpfn.feature_num_workers),
                        cache_path=str(cache_test_view),
                        image_size_hw=tuple(image_size_view),
                        hflip=bool(hflip_view),
                        vflip=bool(vflip_view),
                    )

                if not (
                    isinstance(X_test_i, np.ndarray)
                    and X_test_i.ndim == 2
                    and X_test_i.shape[0] == len(test_image_paths)
                ):
                    raise RuntimeError(
                        f"Invalid test features array for model_id={model_id_i!r}: shape={getattr(X_test_i, 'shape', None)}"
                    )
                if D_ref is None:
                    D_ref = int(X_test_i.shape[1])
                elif int(X_test_i.shape[1]) != int(D_ref):
                    raise RuntimeError(
                        "TabPFN ensemble requires the SAME feature dimension across models. "
                        f"Got D_ref={D_ref} but model_id={model_id_i!r} produced D={int(X_test_i.shape[1])}."
                    )

                y_pred_view = predictor.predict(X_test_i)
                y_pred_model_sum = y_pred_view if y_pred_model_sum is None else (y_pred_model_sum + y_pred_view)
                n_views_eff += 1

            if y_pred_model_sum is None or n_views_eff <= 0:
                raise RuntimeError(f"No TTA views produced predictions for model_id={model_id_i!r}")
            y_pred_model = y_pred_model_sum / float(n_views_eff)

            y_pred_sum = (y_pred_model * float(w_i)) if y_pred_sum is None else (y_pred_sum + y_pred_model * float(w_i))
            w_sum += float(w_i)

        if y_pred_sum is None or not (w_sum > 0.0):
            raise RuntimeError("No valid ensemble models produced predictions for TabPFN.")
        y_pred = y_pred_sum / float(w_sum)

        if bool(getattr(tabpfn, "ratio_strict", False)):
            from src.tabular.ratio_strict import apply_ratio_strict_5d

            y_pred = apply_ratio_strict_5d(y_pred)

        # Map predictions to per-image components dict
        image_to_components: dict[str, dict[str, float]] = {}
        for rel_path, vec in zip(unique_test_paths, y_pred.tolist()):
            comps = {name: float(vec[i]) for i, name in enumerate(TARGETS_5D_ORDER) if i < len(vec)}
            for k in list(comps.keys()):
                comps[k] = max(0.0, float(comps[k]))
            image_to_components[str(rel_path)] = comps

        _write_submission_csv(df_test, image_to_components, str(settings.output_submission_path))
        used_models = [str(mi.get("model_id", "")) for mi in models_eff]
        print(f"[TABPFN][ENSEMBLE] Models used: {used_models}")
        print(f"[TABPFN][ENSEMBLE] Submission written to: {settings.output_submission_path}")
        return

    # Resolve DINO backbone weights file (same strategy as inference pipeline)
    backbone_name = str(cfg.get("model", {}).get("backbone", "") or "").strip()
    dino_weights_file = resolve_dino_weights_path_for_model(
        project_dir_abs,
        backbone_name=backbone_name,
        cfg=cfg,
        model_cfg={},
        global_dino_weights=str(settings.dino_weights_pt_path),
    )
    if not (dino_weights_file and os.path.isfile(dino_weights_file)):
        p0 = resolve_path_best_effort(project_dir_abs, str(settings.dino_weights_pt_path))
        if os.path.isfile(p0):
            dino_weights_file = os.path.abspath(p0)
    if not (dino_weights_file and os.path.isfile(dino_weights_file)):
        raise FileNotFoundError(
            f"Cannot resolve DINO backbone weights file from dino_weights_pt_path={settings.dino_weights_pt_path!r}"
        )

    from src.metrics import TARGETS_5D_ORDER

    # Load packaged single-model fit-state (no online fitting).
    predictor = _load_tabpfn_predictor_from_fit_state(
        project_dir_abs=project_dir_abs,
        tabpfn=tabpfn,
        subdir="single",
        expected_targets=list(TARGETS_5D_ORDER),
    )

    print("[TABPFN] Loaded packaged fit-state (single).")
    print("[TABPFN] DINO weights:", dino_weights_file)
    feature_mode = str(getattr(tabpfn, "feature_mode", "head_penultimate") or "head_penultimate").strip().lower()
    if feature_mode not in ("head_penultimate", "dinov3_only"):
        raise ValueError(f"Unsupported TabPFN feature_mode: {feature_mode!r} (expected 'head_penultimate' or 'dinov3_only')")

    if feature_mode == "dinov3_only":
        print("[TABPFN] Feature mode: dinov3_only (CLS token; no head)")
        print("[TABPFN] Head weights: (unused for dinov3_only)")
    else:
        print("[TABPFN] Feature mode: head_penultimate (pre-final-linear)")
        print("[TABPFN] Head weights:", str(settings.head_weights_pt_path))
    print("[TABPFN] TabPFN ckpt (local):", ckpt_path)

    # Resolve base image size for TTA view generation (H,W)
    try:
        base_image_size_hw = parse_image_size(cfg.get("data", {}).get("image_size", 224))
    except Exception:
        base_image_size_hw = (224, 224)

    # Predict on test features with optional TTA (single packaged TabPFN).
    test_image_paths = list(unique_test_paths)
    tta_views = _resolve_tta_views(settings, base_image_size=tuple(base_image_size_hw), patch_multiple=16)
    multi_view = len(tta_views) > 1

    cache_test_base_raw = str(tabpfn.feature_cache_path_test or "").strip()
    cache_test_base_resolved = resolve_path_best_effort(project_dir_abs, cache_test_base_raw) if cache_test_base_raw else ""

    def _resolve_test_cache_path_for_view(*, view_tag: str) -> Optional[str]:
        if not cache_test_base_resolved:
            return None
        p = str(cache_test_base_resolved)
        is_dir_style = (p.endswith(os.sep) or p.endswith("/")) or os.path.isdir(p) or (os.path.splitext(p)[1] == "")
        if (not multi_view) and (not is_dir_style):
            # Backward-compatible: single view uses the raw file path.
            return p
        if is_dir_style:
            d = p.rstrip("/").rstrip(os.sep)
            return os.path.join(d, f"test__{view_tag}.pt")
        root, ext = os.path.splitext(p)
        ext_eff = ext if ext else ".pt"
        return f"{root}__{view_tag}{ext_eff}"

    y_pred_sum: Optional["np.ndarray"] = None
    n_views_eff: int = 0
    for (image_size_view, hflip_view, vflip_view) in tta_views:
        view_tag = safe_slug(
            f"tta_{int(image_size_view[0])}x{int(image_size_view[1])}__hf{int(bool(hflip_view))}__vf{int(bool(vflip_view))}"
        )
        cache_test_view = _resolve_test_cache_path_for_view(view_tag=view_tag)
        if feature_mode == "dinov3_only":
            _test_rels, X_test_view = extract_dinov3_cls_features(
                project_dir=project_dir_abs,
                dataset_root=str(dataset_root),
                cfg_train_yaml=cfg,
                dino_weights_pt_file=str(dino_weights_file),
                image_paths=test_image_paths,
                batch_size=int(tabpfn.feature_batch_size),
                num_workers=int(tabpfn.feature_num_workers),
                cache_path=str(cache_test_view) if cache_test_view else None,
                image_size_hw=tuple(image_size_view),
                hflip=bool(hflip_view),
                vflip=bool(vflip_view),
            )
        else:
            _test_rels, X_test_view = extract_head_penultimate_features(
                project_dir=project_dir_abs,
                dataset_root=str(dataset_root),
                cfg_train_yaml=cfg,
                dino_weights_pt_file=str(dino_weights_file),
                head_weights_pt_path=str(settings.head_weights_pt_path),
                image_paths=test_image_paths,
                fusion=str(tabpfn.feature_fusion),
                batch_size=int(tabpfn.feature_batch_size),
                num_workers=int(tabpfn.feature_num_workers),
                cache_path=str(cache_test_view) if cache_test_view else None,
                image_size_hw=tuple(image_size_view),
                hflip=bool(hflip_view),
                vflip=bool(vflip_view),
            )

        y_pred_view = predictor.predict(X_test_view)
        y_pred_sum = y_pred_view if y_pred_sum is None else (y_pred_sum + y_pred_view)
        n_views_eff += 1

    if y_pred_sum is None or n_views_eff <= 0:
        raise RuntimeError("TabPFN produced no predictions for test set (TTA views empty).")
    y_pred = y_pred_sum / float(n_views_eff)
    if bool(getattr(tabpfn, "ratio_strict", False)):
        from src.tabular.ratio_strict import apply_ratio_strict_5d

        y_pred = apply_ratio_strict_5d(y_pred)

    # Map predictions to per-image components dict
    image_to_components: dict[str, dict[str, float]] = {}
    for rel_path, vec in zip(unique_test_paths, y_pred.tolist()):
        comps = {name: float(vec[i]) for i, name in enumerate(TARGETS_5D_ORDER) if i < len(vec)}
        for k in list(comps.keys()):
            comps[k] = max(0.0, float(comps[k]))
        image_to_components[str(rel_path)] = comps

    _write_submission_csv(df_test, image_to_components, str(settings.output_submission_path))
    print(f"[TABPFN] Submission written to: {settings.output_submission_path}")


