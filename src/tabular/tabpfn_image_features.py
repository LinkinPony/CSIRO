from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import torch
from loguru import logger

from src.tabular.tabpfn_utils import parse_image_size_hw, resolve_under_repo


def _cfg_version(cfg_img: Dict[str, Any]) -> str:
    try:
        return str(cfg_img.get("version", "") or "").strip()
    except Exception:
        return ""


def _file_fingerprint(path: Path) -> dict:
    try:
        st = os.stat(str(path))
        return {"size": int(getattr(st, "st_size", 0)), "mtime_ns": int(getattr(st, "st_mtime_ns", 0))}
    except Exception:
        return {"size": None, "mtime_ns": None}


def _pick_ckpt_for_swa(ckpt_root: Path) -> Optional[Path]:
    """
    Pick a Lightning `.ckpt` file to use as the source for SWA head export.

    Priority:
      1) <ckpt_root>/last.ckpt
      2) newest *.ckpt under <ckpt_root>/
    """
    last_ckpt = (ckpt_root / "last.ckpt").resolve()
    if last_ckpt.is_file():
        return last_ckpt
    ckpt_candidates = [p for p in ckpt_root.glob("*.ckpt") if p.is_file()]
    ckpt_candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return ckpt_candidates[0] if ckpt_candidates else None


def _try_export_swa_head_from_ckpt(
    *,
    ckpt_root: Path,
    dst_dir: Path,
) -> Optional[Path]:
    """
    Best-effort export a packed SWA head weights file (infer_head.pt) from a Lightning checkpoint.

    This reuses the same implementation used by `package_artifacts.py` (which itself mirrors
    how `kfold_swa_metrics.json` is evaluated via `swa_eval_kfold.py`).

    The destination directory MUST be safe to overwrite (the exporter clears it).
    """
    ckpt_for_swa = _pick_ckpt_for_swa(ckpt_root)
    if ckpt_for_swa is None:
        return None
    try:
        from package_artifacts import _export_swa_head_from_checkpoint  # type: ignore
    except Exception:
        return None
    try:
        res = _export_swa_head_from_checkpoint(ckpt_for_swa, dst_dir)
    except Exception as e:
        logger.warning(
            "SWA head export failed (ckpt_root={} -> dst_dir={}): {}",
            str(ckpt_root),
            str(dst_dir),
            str(e),
        )
        return None
    if res is None:
        return None
    _src_ckpt, dst_path = res
    return Path(dst_path).resolve()


def resolve_fold_head_weights_path(
    *,
    repo_root: Path,
    cfg_img: Dict[str, Any],
    fold_idx: int,
) -> Optional[Path]:
    """
    Resolve a *fold-specific* head checkpoint for leakage-free CV.

    We intentionally DO NOT fall back to train_all weights, since that would leak
    validation labels into the feature extractor.

    Search order (best-effort):
      1) weights/head/fold_<i>/infer_head.pt
      2) weights/head/<version>/fold_<i>/infer_head.pt                (ensemble packaging layout)
      3) outputs/checkpoints/<version>/fold_<i>/head_swa/infer_head.pt (auto-exported SWA head)
      4) outputs/checkpoints/<version>/fold_<i>/head/head-epoch*.pt    (latest loadable; raw weights)
      5) outputs/checkpoints/<version>/fold_<i>/head/infer_head.pt
    """
    from src.inference.torch_load import load_head_state

    ver = str(cfg_img.get("version", "") or "").strip()
    fold_name = f"fold_{int(fold_idx)}"

    candidates: list[Path] = []

    # Packaged layout(s)
    candidates.append((repo_root / "weights" / "head" / fold_name / "infer_head.pt").resolve())
    if ver:
        candidates.append((repo_root / "weights" / "head" / ver / fold_name / "infer_head.pt").resolve())

    # Training outputs (per-fold heads are stored here)
    if ver:
        fold_root = (repo_root / "outputs" / "checkpoints" / ver / fold_name).resolve()

        # Prefer (and optionally auto-generate) an SWA-exported head that matches
        # the kfold_swa_metrics.json evaluation behaviour.
        head_swa_dir = (fold_root / "head_swa").resolve()
        head_swa_path = (head_swa_dir / "infer_head.pt").resolve()
        if head_swa_path.is_file():
            candidates.append(head_swa_path)
        else:
            if fold_root.is_dir():
                exported = _try_export_swa_head_from_ckpt(ckpt_root=fold_root, dst_dir=head_swa_dir)
                if exported is not None and exported.is_file():
                    candidates.append(exported)

        head_dir = (repo_root / "outputs" / "checkpoints" / ver / fold_name / "head").resolve()
        if head_dir.is_dir():
            # Prefer epoch checkpoints (choose latest *loadable*).
            head_files = [p for p in head_dir.glob("head-epoch*.pt") if p.is_file()]

            def _epoch_num(p: Path) -> int:
                name = p.name
                try:
                    # head-epochXYZ...
                    s = name.split("head-epoch", 1)[1]
                    digits = ""
                    for ch in s:
                        if ch.isdigit():
                            digits += ch
                        else:
                            break
                    return int(digits) if digits else -1
                except Exception:
                    return -1

            head_files.sort(key=lambda p: (_epoch_num(p), p.stat().st_mtime))

            # Try newest -> oldest until one loads.
            for cand in reversed(head_files):
                try:
                    _ = load_head_state(str(cand))
                    candidates.append(cand)
                    break
                except Exception:
                    continue

            # Some export flows may place infer_head.pt inside the fold head dir.
            candidates.append((head_dir / "infer_head.pt").resolve())

    for cand in candidates:
        if not cand.is_file():
            continue
        try:
            _ = load_head_state(str(cand))
            return cand
        except Exception:
            continue
    return None


def resolve_train_all_head_weights_path(
    *,
    repo_root: Path,
    cfg_img: Dict[str, Any],
) -> Optional[Path]:
    """
    Resolve a **train_all** head checkpoint (for feature extraction outside CV).

    Unlike `resolve_fold_head_weights_path`, this function *may* use train_all weights
    because there is no outer-fold validation leakage concern in train_all mode.

    Search order (best-effort):
      1) weights/head/infer_head.pt
      2) weights/head/<version>/train_all/infer_head.pt                (ensemble packaging layout)
      3) outputs/checkpoints/<version>/train_all/head_swa/infer_head.pt (auto-exported SWA head)
      4) outputs/checkpoints/<version>/train_all/head/head-epoch*.pt    (latest loadable; raw weights)
      5) outputs/checkpoints/<version>/train_all/head/infer_head.pt
      6) outputs/checkpoints/<version>/head_swa/infer_head.pt           (auto-exported SWA head; legacy single-run)
      7) outputs/checkpoints/<version>/head/head-epoch*.pt              (legacy single-run layout)
      8) outputs/checkpoints/<version>/head/infer_head.pt
    """
    from src.inference.torch_load import load_head_state

    ver = str(cfg_img.get("version", "") or "").strip()

    candidates: list[Path] = []

    # Packaged layout(s)
    candidates.append((repo_root / "weights" / "head" / "infer_head.pt").resolve())
    if ver:
        candidates.append((repo_root / "weights" / "head" / ver / "train_all" / "infer_head.pt").resolve())

    def _append_latest_epoch_ckpt(head_dir: Path) -> None:
        if not head_dir.is_dir():
            return
        head_files = [p for p in head_dir.glob("head-epoch*.pt") if p.is_file()]

        def _epoch_num(p: Path) -> int:
            name = p.name
            try:
                s = name.split("head-epoch", 1)[1]
                digits = ""
                for ch in s:
                    if ch.isdigit():
                        digits += ch
                    else:
                        break
                return int(digits) if digits else -1
            except Exception:
                return -1

        head_files.sort(key=lambda p: (_epoch_num(p), p.stat().st_mtime))
        for cand in reversed(head_files):
            try:
                _ = load_head_state(str(cand))
                candidates.append(cand)
                break
            except Exception:
                continue
        candidates.append((head_dir / "infer_head.pt").resolve())

    # Training outputs
    if ver:
        # train_all SWA head (preferred)
        train_all_root = (repo_root / "outputs" / "checkpoints" / ver / "train_all").resolve()
        train_all_swa_dir = (train_all_root / "head_swa").resolve()
        train_all_swa_path = (train_all_swa_dir / "infer_head.pt").resolve()
        if train_all_swa_path.is_file():
            candidates.append(train_all_swa_path)
        else:
            if train_all_root.is_dir():
                exported = _try_export_swa_head_from_ckpt(ckpt_root=train_all_root, dst_dir=train_all_swa_dir)
                if exported is not None and exported.is_file():
                    candidates.append(exported)

        _append_latest_epoch_ckpt((repo_root / "outputs" / "checkpoints" / ver / "train_all" / "head").resolve())
        # Legacy single-run layout (non-train_all)
        legacy_root = (repo_root / "outputs" / "checkpoints" / ver).resolve()
        legacy_swa_dir = (legacy_root / "head_swa").resolve()
        legacy_swa_path = (legacy_swa_dir / "infer_head.pt").resolve()
        if legacy_swa_path.is_file():
            candidates.append(legacy_swa_path)
        else:
            if legacy_root.is_dir():
                exported = _try_export_swa_head_from_ckpt(ckpt_root=legacy_root, dst_dir=legacy_swa_dir)
                if exported is not None and exported.is_file():
                    candidates.append(exported)
        _append_latest_epoch_ckpt((repo_root / "outputs" / "checkpoints" / ver / "head").resolve())

    for cand in candidates:
        if not cand.is_file():
            continue
        try:
            _ = load_head_state(str(cand))
            return cand
        except Exception:
            continue
    return None


def extract_head_penultimate_features_as_tabular(
    *,
    repo_root: Path,
    cfg_img: Dict[str, Any],
    df: pd.DataFrame,
    image_features_cfg: Dict[str, Any],
    head_weights_path: Path,
    cache_path: Optional[Path],
) -> np.ndarray:
    """
    Extract **head-internal penultimate features** (the tensor fed into the final Linear regressor)
    and return a 2D numpy array (N, D) for TabPFN.

    We intentionally do NOT use any tabular columns from train.csv as inputs here.
    """
    from torch import nn
    from torch.utils.data import DataLoader

    from src.inference.data import TestImageDataset, build_transforms
    from src.inference.torch_load import load_head_state
    from src.models.backbone import build_feature_extractor
    from src.models.head_builder import DualBranchHeadExport, MultiLayerHeadExport, build_head_layer
    from src.models.vitdet_head import ViTDetHeadConfig, ViTDetMultiLayerScalarHead, ViTDetScalarHead

    if "image_path" not in df.columns:
        raise KeyError("Pivoted dataframe missing required column: image_path")

    backbone_name = str(cfg_img.get("model", {}).get("backbone", "")).strip()
    if not backbone_name:
        raise ValueError("image model config missing model.backbone")

    weights_path_raw = cfg_img.get("model", {}).get("weights_path", None)
    if not isinstance(weights_path_raw, str) or not weights_path_raw.strip():
        raise ValueError("image model config missing model.weights_path (must be a local .pt/.pth)")
    weights_path = resolve_under_repo(repo_root, weights_path_raw)
    if not weights_path.is_file():
        raise FileNotFoundError(f"Backbone weights not found: {weights_path}")

    image_size = parse_image_size_hw(cfg_img.get("data", {}).get("image_size", 224))
    mean = list(cfg_img.get("data", {}).get("normalization", {}).get("mean", [0.485, 0.456, 0.406]))
    std = list(cfg_img.get("data", {}).get("normalization", {}).get("std", [0.229, 0.224, 0.225]))

    # Dataloader params
    batch_size = int(image_features_cfg.get("batch_size", 8))
    num_workers = int(image_features_cfg.get("num_workers", 8))

    # Head weights are resolved by the caller (per-fold) to avoid train_all leakage.
    if not isinstance(head_weights_path, Path):
        raise TypeError("head_weights_path must be a pathlib.Path")
    if not head_weights_path.is_file():
        raise FileNotFoundError(f"Head weights not found: {head_weights_path}")

    # Load head weights/meta (+ optional LoRA payload for backbone)
    head_state, head_meta, peft_payload = load_head_state(str(head_weights_path))
    if not isinstance(head_meta, dict):
        head_meta = {}
    head_type = str(head_meta.get("head_type", "mlp") or "mlp").strip().lower()
    if head_type not in ("mlp", "vitdet", "eomt"):
        raise SystemExit(
            f"Unsupported head_type={head_type!r} for penultimate-feature extraction. "
            "Currently supported: mlp, vitdet, eomt."
        )

    # Penultimate feature source selection
    # - mean  : fuse per-layer features by average/learned weights when applicable
    # - concat: concatenate per-layer features (only when layerwise heads are enabled)
    fusion = str(image_features_cfg.get("fusion", "mean")).strip().lower()
    if fusion not in ("mean", "concat"):
        raise ValueError(f"Unsupported image_features.fusion: {fusion}")

    image_paths: list[str] = df["image_path"].astype(str).tolist()
    dataset_root = str(resolve_under_repo(repo_root, cfg_img.get("data", {}).get("root", "data")))

    # Try cache first
    if cache_path is not None and cache_path.is_file():
        try:
            obj = torch.load(str(cache_path), map_location="cpu")
            if isinstance(obj, dict) and "image_paths" in obj and "features" in obj:
                cached_paths = list(obj["image_paths"])
                feats = obj["features"]
                cached_meta = dict(obj.get("meta", {}) or {})
                cfg_ver = _cfg_version(cfg_img)
                head_fp = _file_fingerprint(head_weights_path)
                bb_fp = _file_fingerprint(weights_path)
                if (
                    cached_paths == image_paths
                    and isinstance(feats, torch.Tensor)
                    and feats.dim() == 2
                    and str(cached_meta.get("cfg_version", "")) == str(cfg_ver)
                    and str(cached_meta.get("head_weights_path", "")) == str(head_weights_path)
                    and dict(cached_meta.get("head_weights_fingerprint", {}) or {}) == dict(head_fp)
                    and str(cached_meta.get("backbone", "")) == str(backbone_name)
                    and str(cached_meta.get("weights_path", "")) == str(weights_path)
                    and dict(cached_meta.get("backbone_weights_fingerprint", {}) or {}) == dict(bb_fp)
                    and tuple(cached_meta.get("image_size", ())) == tuple(image_size)
                    and str(cached_meta.get("fusion", "")) == str(fusion)
                ):
                    return feats.cpu().numpy()
        except Exception:
            pass

    tf = build_transforms(image_size=image_size, mean=mean, std=std, hflip=False, vflip=False)
    ds = TestImageDataset(image_paths, root_dir=dataset_root, transform=tf)
    dl = DataLoader(
        ds,
        batch_size=max(1, batch_size),
        shuffle=False,
        num_workers=max(0, num_workers),
        pin_memory=torch.cuda.is_available(),
    )

    logger.info(
        "Extracting head-penultimate features: head_type={} head_weights={} backbone={} weights={} image_size={} fusion={}",
        head_type,
        str(head_weights_path),
        backbone_name,
        str(weights_path),
        tuple(image_size),
        fusion,
    )

    # Build extractor (offline; weights_path forces pretrained=False inside helper)
    feature_extractor = build_feature_extractor(
        backbone_name=backbone_name,
        pretrained=bool(cfg_img.get("model", {}).get("pretrained", True)),
        weights_url=str(cfg_img.get("model", {}).get("weights_url", "") or "") or None,
        weights_path=str(weights_path),
        gradient_checkpointing=bool(cfg_img.get("model", {}).get("gradient_checkpointing", False)),
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
            embedding_dim=int(head_meta.get("embedding_dim", int(cfg_img.get("model", {}).get("embedding_dim", 1024)))),
            vitdet_dim=int(head_meta.get("vitdet_dim", int(cfg_img.get("model", {}).get("head", {}).get("vitdet_dim", 256)))),
            scale_factors=list(head_meta.get("vitdet_scale_factors", cfg_img.get("model", {}).get("head", {}).get("vitdet_scale_factors", [2.0, 1.0, 0.5]))),
            patch_size=int(head_meta.get("vitdet_patch_size", int(cfg_img.get("model", {}).get("head", {}).get("vitdet_patch_size", 16)))),
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
    elif head_type == "eomt":
        from src.models.eomt_injected_head import EoMTInjectedQueryHeadConfig, EoMTInjectedQueryScalarHead

        eomt_cfg = cfg_img.get("model", {}).get("head", {}).get("eomt", {})
        if not isinstance(eomt_cfg, dict):
            eomt_cfg = {}

        eomt_num_queries = int(head_meta.get("eomt_num_queries", int(eomt_cfg.get("num_queries", 16))))
        eomt_num_blocks = int(
            head_meta.get(
                "eomt_num_blocks",
                head_meta.get(
                    "eomt_num_layers",
                    int(eomt_cfg.get("num_blocks", eomt_cfg.get("num_layers", 4))),
                ),
            )
        )
        eomt_query_pool = str(head_meta.get("eomt_query_pool", str(eomt_cfg.get("query_pool", "mean")))).strip().lower()
        # New pooled feature construction (backward-compatible defaults)
        eomt_use_mean_query = bool(head_meta.get("eomt_use_mean_query", bool(eomt_cfg.get("use_mean_query", True))))
        eomt_use_mean_patch = bool(head_meta.get("eomt_use_mean_patch", bool(eomt_cfg.get("use_mean_patch", False))))
        eomt_use_cls_token = bool(
            head_meta.get("eomt_use_cls_token", bool(eomt_cfg.get("use_cls_token", eomt_cfg.get("use_cls", False))))
        )
        eomt_proj_dim = int(head_meta.get("eomt_proj_dim", int(eomt_cfg.get("proj_dim", 0))))
        eomt_proj_activation = str(head_meta.get("eomt_proj_activation", str(eomt_cfg.get("proj_activation", "relu")))).strip().lower()
        try:
            eomt_proj_dropout = float(head_meta.get("eomt_proj_dropout", float(eomt_cfg.get("proj_dropout", 0.0))))
        except Exception:
            eomt_proj_dropout = float(eomt_cfg.get("proj_dropout", 0.0) or 0.0)
        enable_ndvi = bool(head_meta.get("enable_ndvi", False))

        embedding_dim = int(head_meta.get("embedding_dim", int(cfg_img.get("model", {}).get("embedding_dim", 1024))))
        head_hidden_dims = list(head_meta.get("head_hidden_dims", []))
        head_activation = str(head_meta.get("head_activation", "relu"))
        head_dropout = float(head_meta.get("head_dropout", 0.0))

        head_module = EoMTInjectedQueryScalarHead(
            EoMTInjectedQueryHeadConfig(
                embedding_dim=int(embedding_dim),
                num_queries=int(eomt_num_queries),
                num_blocks=int(eomt_num_blocks),
                dropout=float(head_dropout),
                query_pool=str(eomt_query_pool),
                use_mean_query=bool(eomt_use_mean_query),
                use_mean_patch=bool(eomt_use_mean_patch),
                use_cls_token=bool(eomt_use_cls_token),
                proj_dim=int(eomt_proj_dim),
                proj_activation=str(eomt_proj_activation),
                proj_dropout=float(eomt_proj_dropout),
                head_hidden_dims=list(head_hidden_dims),
                head_activation=str(head_activation),
                num_outputs_main=int(head_meta.get("num_outputs_main", 1)),
                num_outputs_ratio=int(head_meta.get("num_outputs_ratio", 0)),
                enable_ndvi=bool(enable_ndvi),
            )
        )
    else:
        # MLP-style exported head module:
        # - Either MultiLayerHeadExport (when layerwise heads + separate bottlenecks)
        # - Or a packed MLP nn.Sequential built by build_head_layer
        use_patch_reg3 = bool(head_meta.get("use_patch_reg3", False))
        use_cls_token = bool(head_meta.get("use_cls_token", True))
        dual_branch_enabled = bool(head_meta.get("dual_branch_enabled", False))
        use_layerwise_heads = bool(head_meta.get("use_layerwise_heads", False))
        layer_indices = list(head_meta.get("backbone_layer_indices", []))
        use_separate_bottlenecks = bool(head_meta.get("use_separate_bottlenecks", False))
        num_layers_eff = max(1, len(layer_indices)) if use_layerwise_heads else 1
        head_total = int(head_meta.get("head_total_outputs", int(head_meta.get("num_outputs_main", 1)) + int(head_meta.get("num_outputs_ratio", 0))))
        embedding_dim = int(head_meta.get("embedding_dim", int(cfg_img.get("model", {}).get("embedding_dim", 1024))))
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

    eomt_start_block: Optional[int] = None
    if head_type == "eomt":
        try:
            bb0 = EoMTInjectedQueryScalarHead._resolve_base_backbone(feature_extractor.backbone)  # type: ignore[attr-defined]
            blocks = getattr(bb0, "blocks", None)
            depth = len(blocks) if isinstance(blocks, (nn.ModuleList, list)) else 0
        except Exception:
            depth = 0
        if depth <= 0:
            raise RuntimeError("EoMT injected-query feature extraction requires a DINOv3 ViT backbone with `.blocks`")
        try:
            k_blocks = int(getattr(head_module, "num_blocks", int(head_meta.get("eomt_num_blocks", head_meta.get("eomt_num_layers", 4)))))
        except Exception:
            k_blocks = int(head_meta.get("eomt_num_blocks", head_meta.get("eomt_num_layers", 4)) or 4)
        k_blocks = int(max(0, min(k_blocks, depth)))
        eomt_start_block = int(depth - k_blocks)

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

    def _fuse_layers(z_list: list[torch.Tensor], *, weights: Optional[torch.Tensor]) -> torch.Tensor:
        if not z_list:
            raise RuntimeError("Empty z_list for fusion")
        if fusion == "concat":
            return torch.cat(z_list, dim=-1)
        # mean / learned weights
        if weights is None:
            return torch.stack(z_list, dim=0).mean(dim=0)
        w = weights.to(device=z_list[0].device, dtype=z_list[0].dtype)
        w = w / w.sum().clamp_min(1e-8)
        while w.dim() < z_list[0].dim() + 1:
            w = w.view(*w.shape, 1)
        stacked = torch.stack(z_list, dim=0)
        return (w * stacked).sum(dim=0)

    feats_cpu: list[torch.Tensor] = []
    rels: list[str] = []
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
                    if fusion == "concat" and isinstance(z_layers, list) and z_layers:
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
            elif head_type == "eomt":
                if eomt_start_block is None:
                    raise RuntimeError("Internal error: eomt_start_block not initialized")
                x_base, (H_p, W_p) = feature_extractor.forward_tokens_until_block(  # type: ignore[attr-defined]
                    images, block_idx=int(eomt_start_block)
                )
                out = head_module(  # type: ignore[call-arg]
                    x_base,
                    backbone=getattr(feature_extractor, "backbone"),
                    patch_hw=(int(H_p), int(W_p)),
                )
                z = out.get("z", None) if isinstance(out, dict) else None
                if not isinstance(z, torch.Tensor):
                    raise RuntimeError("EoMT head did not return 'z'")
                feats = z
            else:
                # MLP head: extract the tensor right before the final Linear layer.
                use_patch_reg3 = bool(head_meta.get("use_patch_reg3", False))
                use_cls_token = bool(head_meta.get("use_cls_token", True))
                dual_branch_enabled = bool(head_meta.get("dual_branch_enabled", False))
                use_layerwise_heads = bool(head_meta.get("use_layerwise_heads", False))
                layer_indices = list(head_meta.get("backbone_layer_indices", []))
                use_separate_bottlenecks = bool(head_meta.get("use_separate_bottlenecks", False))

                # Optional learned fusion weights (only for mean-fusion; concat ignores weights)
                weights = None
                fusion_mode_meta = str(
                    head_meta.get("backbone_layers_fusion", head_meta.get("layer_fusion", "mean")) or "mean"
                ).strip().lower()
                if use_layerwise_heads and fusion == "mean" and fusion_mode_meta == "learned":
                    logits_meta = head_meta.get("mlp_layer_logits", None)
                    if isinstance(logits_meta, (list, tuple)) and len(logits_meta) == len(layer_indices):
                        try:
                            logits_t = torch.tensor([float(x) for x in logits_meta], device=device, dtype=torch.float32)
                            weights = torch.softmax(logits_t, dim=0)
                        except Exception:
                            weights = None

                if use_layerwise_heads and layer_indices:
                    cls_list, pt_list = feature_extractor.forward_layers_cls_and_tokens(images, layer_indices)  # type: ignore[attr-defined]
                    z_list: list[torch.Tensor] = []
                    for l_idx, (cls_l, pt_l) in enumerate(zip(cls_list, pt_list)):
                        if use_patch_reg3:
                            # Patch-mode: bottleneck runs per patch, then average.
                            if pt_l.dim() != 3:
                                raise RuntimeError(f"Unexpected patch token shape: {tuple(pt_l.shape)}")
                            B, N, C = pt_l.shape
                            if dual_branch_enabled and isinstance(head_module, DualBranchHeadExport):
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

    if rels != image_paths:
        raise RuntimeError("Feature extraction order mismatch (unexpected dataloader ordering).")

    features = torch.cat(feats_cpu, dim=0) if feats_cpu else torch.empty((0, 0), dtype=torch.float32)
    if features.shape[0] != len(image_paths):
        raise RuntimeError(f"Feature extraction produced wrong N: got {features.shape[0]}, expected {len(image_paths)}")

    # Save cache
    if cache_path is not None:
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cfg_ver = _cfg_version(cfg_img)
            head_fp = _file_fingerprint(head_weights_path)
            bb_fp = _file_fingerprint(weights_path)
            torch.save(
                {
                    "image_paths": image_paths,
                    "features": features,
                    "meta": {
                        "schema_version": 2,
                        "cfg_version": str(cfg_ver),
                        "head_type": head_type,
                        "head_weights_path": str(head_weights_path),
                        "head_weights_fingerprint": dict(head_fp),
                        "backbone": backbone_name,
                        "weights_path": str(weights_path),
                        "backbone_weights_fingerprint": dict(bb_fp),
                        "image_size": tuple(image_size),
                        "fusion": fusion,
                    },
                },
                str(cache_path),
            )
            logger.info("Saved image feature cache -> {}", cache_path)
        except Exception as e:
            logger.warning(f"Saving feature cache failed: {e}")

    return features.numpy()


def extract_dinov3_cls_features_as_tabular(
    *,
    repo_root: Path,
    cfg_img: Dict[str, Any],
    df: pd.DataFrame,
    image_features_cfg: Dict[str, Any],
    cache_path: Optional[Path],
) -> np.ndarray:
    """
    Extract **raw DINOv3 CLS token features** (no LoRA, no head) and return a 2D numpy array (N, C).
    """
    from torch.utils.data import DataLoader

    from src.inference.data import TestImageDataset, build_transforms
    from src.models.backbone import build_feature_extractor

    if "image_path" not in df.columns:
        raise KeyError("Pivoted dataframe missing required column: image_path")

    backbone_name = str(cfg_img.get("model", {}).get("backbone", "")).strip()
    if not backbone_name:
        raise ValueError("image model config missing model.backbone")

    weights_path_raw = cfg_img.get("model", {}).get("weights_path", None)
    if not isinstance(weights_path_raw, str) or not weights_path_raw.strip():
        raise ValueError("image model config missing model.weights_path (must be a local .pt/.pth)")
    weights_path = resolve_under_repo(repo_root, weights_path_raw)
    if not weights_path.is_file():
        raise FileNotFoundError(f"Backbone weights not found: {weights_path}")

    image_size = parse_image_size_hw(cfg_img.get("data", {}).get("image_size", 224))
    mean = list(cfg_img.get("data", {}).get("normalization", {}).get("mean", [0.485, 0.456, 0.406]))
    std = list(cfg_img.get("data", {}).get("normalization", {}).get("std", [0.229, 0.224, 0.225]))

    # Dataloader params
    batch_size = int(image_features_cfg.get("batch_size", 8))
    num_workers = int(image_features_cfg.get("num_workers", 8))

    image_paths: list[str] = df["image_path"].astype(str).tolist()
    dataset_root = str(resolve_under_repo(repo_root, cfg_img.get("data", {}).get("root", "data")))

    # Try cache first
    if cache_path is not None and cache_path.is_file():
        try:
            obj = torch.load(str(cache_path), map_location="cpu")
            if isinstance(obj, dict) and "image_paths" in obj and "features" in obj:
                cached_paths = list(obj["image_paths"])
                feats = obj["features"]
                cached_meta = dict(obj.get("meta", {}) or {})
                cfg_ver = _cfg_version(cfg_img)
                bb_fp = _file_fingerprint(weights_path)
                if (
                    cached_paths == image_paths
                    and isinstance(feats, torch.Tensor)
                    and feats.dim() == 2
                    and str(cached_meta.get("backbone", "")) == str(backbone_name)
                    and str(cached_meta.get("weights_path", "")) == str(weights_path)
                    and dict(cached_meta.get("weights_fingerprint", {}) or {}) == dict(bb_fp)
                    and str(cached_meta.get("cfg_version", "")) == str(cfg_ver)
                    and tuple(cached_meta.get("image_size", ())) == tuple(image_size)
                ):
                    return feats.cpu().numpy()
        except Exception:
            pass

    tf = build_transforms(image_size=image_size, mean=mean, std=std, hflip=False, vflip=False)
    ds = TestImageDataset(image_paths, root_dir=dataset_root, transform=tf)
    dl = DataLoader(
        ds,
        batch_size=max(1, batch_size),
        shuffle=False,
        num_workers=max(0, num_workers),
        pin_memory=torch.cuda.is_available(),
    )

    logger.info(
        "Extracting DINOv3 CLS features (dinov3_only): backbone={} weights={} image_size={}",
        backbone_name,
        str(weights_path),
        tuple(image_size),
    )

    # IMPORTANT: no LoRA injection here (plain frozen DINOv3 only)
    feature_extractor = build_feature_extractor(
        backbone_name=backbone_name,
        pretrained=bool(cfg_img.get("model", {}).get("pretrained", True)),
        weights_url=str(cfg_img.get("model", {}).get("weights_url", "") or "") or None,
        weights_path=str(weights_path),
        gradient_checkpointing=False,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    feature_extractor = feature_extractor.eval().to(device)

    feats_cpu: list[torch.Tensor] = []
    rels: list[str] = []
    with torch.inference_mode():
        for images, rel_paths in dl:
            images = images.to(device, non_blocking=True)
            cls, _pt = feature_extractor.forward_cls_and_tokens(images)  # type: ignore[attr-defined]
            feats_cpu.append(cls.detach().cpu().float())
            rels.extend(list(rel_paths))

    if rels != image_paths:
        raise RuntimeError("Feature extraction order mismatch (unexpected dataloader ordering).")

    features = torch.cat(feats_cpu, dim=0) if feats_cpu else torch.empty((0, 0), dtype=torch.float32)
    if features.shape[0] != len(image_paths):
        raise RuntimeError(f"Feature extraction produced wrong N: got {features.shape[0]}, expected {len(image_paths)}")

    # Save cache
    if cache_path is not None:
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cfg_ver = _cfg_version(cfg_img)
            bb_fp = _file_fingerprint(weights_path)
            torch.save(
                {
                    "image_paths": image_paths,
                    "features": features,
                    "meta": {
                        "schema_version": 2,
                        "mode": "dinov3_only",
                        "cfg_version": str(cfg_ver),
                        "backbone": backbone_name,
                        "weights_path": str(weights_path),
                        "weights_fingerprint": dict(bb_fp),
                        "image_size": tuple(image_size),
                    },
                },
                str(cache_path),
            )
            logger.info("Saved image feature cache -> {}", cache_path)
        except Exception as e:
            logger.warning(f"Saving feature cache failed: {e}")

    return features.numpy()


