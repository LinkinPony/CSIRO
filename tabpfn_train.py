from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import yaml
from loguru import logger

from src.data.csiro_pivot import read_and_pivot_csiro_train_csv
from src.metrics import TARGETS_5D_ORDER, weighted_r2_logspace
from src.tabular.features import build_y_5d
from src.training.splits import build_kfold_splits


def resolve_repo_root() -> Path:
    """
    Resolve the project root directory that contains both `configs/` and `src/`.

    Mirrors `train.py` so this script also works in packaged contexts.
    """
    here = Path(__file__).resolve().parent
    if (here / "configs").is_dir() and (here / "src").is_dir():
        return here
    if (here.parent / "configs").is_dir() and (here.parent / "src").is_dir():
        return here.parent
    return here


def _resolve_under_repo(repo_root: Path, p: str | Path) -> Path:
    pp = Path(str(p)).expanduser()
    if pp.is_absolute():
        return pp
    return (repo_root / pp).resolve()


def _parse_image_size_hw(value: Any) -> Tuple[int, int]:
    """
    Accept int (square) or [W, H]; return (H, W).
    Mirrors the project's convention used in training/inference helpers.
    """
    try:
        if isinstance(value, (list, tuple)) and len(value) == 2:
            w, h = int(value[0]), int(value[1])
            return int(h), int(w)
        v = int(value)
        return int(v), int(v)
    except Exception:
        v = int(value)
        return int(v), int(v)


def _import_tabpfn_regressor(repo_root: Path):
    """
    Import TabPFN from either an installed package or the vendored third_party copy.

    We do **not** write anything into `third_party/` (per workspace rules).
    """
    try:
        from tabpfn import TabPFNRegressor  # type: ignore

        return TabPFNRegressor
    except Exception:
        third_party_src = repo_root / "third_party" / "TabPFN" / "src"
        if third_party_src.is_dir():
            sys.path.insert(0, str(third_party_src))
        from tabpfn import TabPFNRegressor  # type: ignore

        return TabPFNRegressor


def parse_args(repo_root: Path) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "TabPFN 2.5 demo on CSIRO train.csv using the project's exact pivot + k-fold splitting "
            "and competition-style weighted R^2 in log-space."
        )
    )
    p.add_argument(
        "--config",
        type=str,
        default=str(repo_root / "configs" / "train_tabpfn.yaml"),
        help="Path to YAML config file (defaults to configs/train_tabpfn.yaml).",
    )
    p.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Output directory. If omitted, uses <logging.log_dir>/tabpfn/<version>-<timestamp>/",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed override (default: config seed).",
    )

    p.add_argument("--k", type=int, default=None, help="Number of folds override.")

    g_even = p.add_mutually_exclusive_group()
    g_even.add_argument(
        "--even-split",
        dest="even_split",
        action="store_true",
        help="Use the project's even_split mode (fresh random 50/50 split per fold).",
    )
    g_even.add_argument(
        "--no-even-split",
        dest="even_split",
        action="store_false",
        help="Disable even_split (classic k-fold style).",
    )
    p.set_defaults(even_split=None)

    g_grp = p.add_mutually_exclusive_group()
    g_grp.add_argument(
        "--group-by-date-state",
        dest="group_by_date_state",
        action="store_true",
        help="Group folds by (Sampling_Date, State) (default).",
    )
    g_grp.add_argument(
        "--no-group-by-date-state",
        dest="group_by_date_state",
        action="store_false",
        help="Disable grouped splitting and split per-sample.",
    )
    p.set_defaults(group_by_date_state=None)

    p.add_argument(
        "--feature-cols",
        type=str,
        default=None,
        help=(
            "(Legacy/ignored) Comma-separated train.csv feature columns. "
            "This script uses image features as X (see `image_features.mode`)."
        ),
    )
    p.add_argument(
        "--image-features-mode",
        type=str,
        default=None,
        choices=["head_penultimate", "dinov3_only"],
        help=(
            "Override `image_features.mode` from the config. "
            "Supported: head_penultimate (default; per-fold head penultimate pre-linear features), "
            "dinov3_only (no LoRA, no head; CLS token from the frozen DINOv3 backbone)."
        ),
    )

    p.add_argument(
        "--n-estimators",
        type=int,
        default=None,
        help="TabPFN ensemble size (overrides config tabpfn.n_estimators).",
    )
    p.add_argument(
        "--device",
        type=str,
        default=None,
        help="TabPFN device setting (overrides config tabpfn.device).",
    )
    p.add_argument(
        "--fit-mode",
        type=str,
        default=None,
        help="TabPFN fit_mode (overrides config tabpfn.fit_mode).",
    )
    p.add_argument(
        "--inference-precision",
        type=str,
        default=None,
        help="TabPFN inference_precision (overrides config tabpfn.inference_precision).",
    )
    p.add_argument(
        "--ignore-pretraining-limits",
        action="store_true",
        help="Pass ignore_pretraining_limits=True to TabPFN (also overrides CPU > 1000 samples guard).",
    )
    p.add_argument(
        "--n-jobs",
        type=int,
        default=None,
        help="Parallelism for multi-output wrapper (overrides config tabpfn.n_jobs).",
    )
    p.add_argument(
        "--max-folds",
        type=int,
        default=None,
        help="Debug: run only the first N folds.",
    )

    p.add_argument(
        "--enable-telemetry",
        action="store_true",
        help="Enable TabPFN telemetry (default: disabled).",
    )
    p.add_argument(
        "--model-cache-dir",
        type=str,
        default=None,
        help="Optional TabPFN model cache dir (sets TABPFN_MODEL_CACHE_DIR).",
    )
    p.add_argument(
        "--model-path",
        type=str,
        default=None,
        help=(
            "Explicit local TabPFN regressor checkpoint path (overrides config tabpfn.model_path). "
            "If omitted and config also does not provide tabpfn.model_path, the script will error."
        ),
    )

    return p.parse_args()


def _save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _resolve_fold_head_weights_path(
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
      3) outputs/checkpoints/<version>/fold_<i>/head/head-epoch*.pt   (latest loadable)
      4) outputs/checkpoints/<version>/fold_<i>/head/infer_head.pt
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


def _extract_head_penultimate_features_as_tabular(
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
    from src.inference.data import TestImageDataset, build_transforms
    from src.inference.torch_load import load_head_state
    from src.models.backbone import build_feature_extractor
    from src.models.head_builder import MultiLayerHeadExport, build_head_layer
    from src.models.vitdet_head import ViTDetHeadConfig, ViTDetMultiLayerScalarHead, ViTDetScalarHead
    from torch.utils.data import DataLoader

    if "image_path" not in df.columns:
        raise KeyError("Pivoted dataframe missing required column: image_path")

    backbone_name = str(cfg_img.get("model", {}).get("backbone", "")).strip()
    if not backbone_name:
        raise ValueError("image model config missing model.backbone")

    weights_path_raw = cfg_img.get("model", {}).get("weights_path", None)
    if not isinstance(weights_path_raw, str) or not weights_path_raw.strip():
        raise ValueError("image model config missing model.weights_path (must be a local .pt/.pth)")
    weights_path = _resolve_under_repo(repo_root, weights_path_raw)
    if not weights_path.is_file():
        raise FileNotFoundError(f"Backbone weights not found: {weights_path}")

    image_size = _parse_image_size_hw(cfg_img.get("data", {}).get("image_size", 224))
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
    if head_type not in ("mlp", "vitdet"):
        raise SystemExit(
            f"Unsupported head_type={head_type!r} for penultimate-feature extraction. "
            "Currently supported: mlp, vitdet."
        )

    # Penultimate feature source selection
    # - mean  : fuse per-layer features by average/learned weights when applicable
    # - concat: concatenate per-layer features (only when layerwise heads are enabled)
    fusion = str(image_features_cfg.get("fusion", "mean")).strip().lower()
    if fusion not in ("mean", "concat"):
        raise ValueError(f"Unsupported image_features.fusion: {fusion}")

    image_paths: list[str] = df["image_path"].astype(str).tolist()
    dataset_root = str(_resolve_under_repo(repo_root, cfg_img.get("data", {}).get("root", "data")))

    # Try cache first
    if cache_path is not None and cache_path.is_file():
        try:
            obj = torch.load(str(cache_path), map_location="cpu")
            if isinstance(obj, dict) and "image_paths" in obj and "features" in obj:
                cached_paths = list(obj["image_paths"])
                feats = obj["features"]
                cached_meta = dict(obj.get("meta", {}) or {})
                if (
                    cached_paths == image_paths
                    and isinstance(feats, torch.Tensor)
                    and feats.dim() == 2
                    and str(cached_meta.get("head_weights_path", "")) == str(head_weights_path)
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
    else:
        # MLP-style exported head module:
        # - Either MultiLayerHeadExport (when layerwise heads + separate bottlenecks)
        # - Or a packed MLP nn.Sequential built by build_head_layer
        use_patch_reg3 = bool(head_meta.get("use_patch_reg3", False))
        use_cls_token = bool(head_meta.get("use_cls_token", True))
        use_layerwise_heads = bool(head_meta.get("use_layerwise_heads", False))
        layer_indices = list(head_meta.get("backbone_layer_indices", []))
        use_separate_bottlenecks = bool(head_meta.get("use_separate_bottlenecks", False))
        num_layers_eff = max(1, len(layer_indices)) if use_layerwise_heads else 1
        head_total = int(head_meta.get("head_total_outputs", int(head_meta.get("num_outputs_main", 1)) + int(head_meta.get("num_outputs_ratio", 0))))
        embedding_dim = int(head_meta.get("embedding_dim", int(cfg_img.get("model", {}).get("embedding_dim", 1024))))
        head_hidden_dims = list(head_meta.get("head_hidden_dims", []))
        head_activation = str(head_meta.get("head_activation", "relu"))
        head_dropout = float(head_meta.get("head_dropout", 0.0))

        if use_layerwise_heads and use_separate_bottlenecks:
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
            else:
                # MLP head: extract the tensor right before the final Linear layer.
                use_patch_reg3 = bool(head_meta.get("use_patch_reg3", False))
                use_cls_token = bool(head_meta.get("use_cls_token", True))
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
                            if use_separate_bottlenecks and isinstance(head_module, MultiLayerHeadExport):
                                bottleneck = head_module.layer_bottlenecks[l_idx]
                                z_flat = bottleneck(pt_l.reshape(B * N, C).to(device))
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
                        if isinstance(head_module, nn.Sequential):
                            z_flat = _penultimate_from_sequential(head_module, pt.reshape(B * N, C).to(device))
                        else:
                            raise RuntimeError("Expected nn.Sequential MLP head in single-layer patch-mode.")
                        feats = z_flat.view(B, N, -1).mean(dim=1)
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
            torch.save(
                {
                    "image_paths": image_paths,
                    "features": features,
                    "meta": {
                        "head_type": head_type,
                        "head_weights_path": str(head_weights_path),
                        "backbone": backbone_name,
                        "weights_path": str(weights_path),
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


def _extract_dinov3_cls_features_as_tabular(
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
    from src.inference.data import TestImageDataset, build_transforms
    from src.models.backbone import build_feature_extractor
    from torch.utils.data import DataLoader

    if "image_path" not in df.columns:
        raise KeyError("Pivoted dataframe missing required column: image_path")

    backbone_name = str(cfg_img.get("model", {}).get("backbone", "")).strip()
    if not backbone_name:
        raise ValueError("image model config missing model.backbone")

    weights_path_raw = cfg_img.get("model", {}).get("weights_path", None)
    if not isinstance(weights_path_raw, str) or not weights_path_raw.strip():
        raise ValueError("image model config missing model.weights_path (must be a local .pt/.pth)")
    weights_path = _resolve_under_repo(repo_root, weights_path_raw)
    if not weights_path.is_file():
        raise FileNotFoundError(f"Backbone weights not found: {weights_path}")

    image_size = _parse_image_size_hw(cfg_img.get("data", {}).get("image_size", 224))
    mean = list(cfg_img.get("data", {}).get("normalization", {}).get("mean", [0.485, 0.456, 0.406]))
    std = list(cfg_img.get("data", {}).get("normalization", {}).get("std", [0.229, 0.224, 0.225]))

    # Dataloader params
    batch_size = int(image_features_cfg.get("batch_size", 8))
    num_workers = int(image_features_cfg.get("num_workers", 8))

    image_paths: list[str] = df["image_path"].astype(str).tolist()
    dataset_root = str(_resolve_under_repo(repo_root, cfg_img.get("data", {}).get("root", "data")))

    # Try cache first
    if cache_path is not None and cache_path.is_file():
        try:
            obj = torch.load(str(cache_path), map_location="cpu")
            if isinstance(obj, dict) and "image_paths" in obj and "features" in obj:
                cached_paths = list(obj["image_paths"])
                feats = obj["features"]
                cached_meta = dict(obj.get("meta", {}) or {})
                if (
                    cached_paths == image_paths
                    and isinstance(feats, torch.Tensor)
                    and feats.dim() == 2
                    and str(cached_meta.get("backbone", "")) == str(backbone_name)
                    and str(cached_meta.get("weights_path", "")) == str(weights_path)
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
            torch.save(
                {
                    "image_paths": image_paths,
                    "features": features,
                    "meta": {
                        "mode": "dinov3_only",
                        "backbone": backbone_name,
                        "weights_path": str(weights_path),
                        "image_size": tuple(image_size),
                    },
                },
                str(cache_path),
            )
            logger.info("Saved image feature cache -> {}", cache_path)
        except Exception as e:
            logger.warning(f"Saving feature cache failed: {e}")

    return features.numpy()


def main() -> None:
    repo_root = resolve_repo_root()
    args = parse_args(repo_root)

    cfg_path = Path(args.config).expanduser()
    if not cfg_path.is_absolute():
        cfg_path = (repo_root / cfg_path).resolve()
    if not cfg_path.is_file():
        raise FileNotFoundError(f"Config YAML not found: {cfg_path}")

    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg: Dict[str, Any] = yaml.safe_load(f)

    version = cfg.get("version", None)
    version = None if version in (None, "", "null") else str(version)

    seed = int(args.seed if args.seed is not None else cfg.get("seed", 42))

    tabpfn_cfg = dict(cfg.get("tabpfn", {}) or {})
    image_features_cfg = dict(cfg.get("image_features", {}) or {})

    # Load image-model config (defines backbone weights + preprocessing)
    img_cfg_path_raw = image_features_cfg.get("train_config", "configs/train.yaml")
    img_cfg_path = _resolve_under_repo(repo_root, str(img_cfg_path_raw))
    if not img_cfg_path.is_file():
        raise FileNotFoundError(f"Image model config not found: {img_cfg_path}")
    with open(img_cfg_path, "r", encoding="utf-8") as f:
        cfg_img: Dict[str, Any] = yaml.safe_load(f)

    # Feature extraction mode (default keeps backward compatible behaviour)
    image_features_mode = (
        str(args.image_features_mode).strip().lower()
        if getattr(args, "image_features_mode", None) is not None
        else str(image_features_cfg.get("mode", "head_penultimate")).strip().lower()
    )
    if image_features_mode not in ("head_penultimate", "dinov3_only"):
        raise SystemExit(f"Unsupported image_features.mode: {image_features_mode!r}")

    # ==========================================================
    # IMPORTANT (Leakage prevention):
    # - Fold splits MUST match the image training config (train.yaml).
    # - Per-fold feature extraction MUST use the corresponding fold head weights.
    # ==========================================================
    split_seed = int(cfg_img.get("seed", 42))
    if args.seed is not None and int(args.seed) != split_seed:
        logger.warning(
            "--seed={} is ignored for k-fold splitting to match configs/train.yaml exactly; using seed={} from {}.",
            int(args.seed),
            int(split_seed),
            str(img_cfg_path),
        )
    # Use image-config seed for deterministic behaviour + TabPFN random_state.
    seed = int(split_seed)
    np.random.seed(seed)

    # --- Resolve effective TabPFN settings (config-first, CLI overrides) ---
    # NOTE: train.csv tabular columns are NOT used as inputs anymore. We keep the CLI arg
    # for backward compatibility but ignore it.
    if args.feature_cols:
        if image_features_mode == "dinov3_only":
            logger.warning(
                "--feature-cols is ignored: TabPFN input is DINOv3 CLS token features (dinov3_only), not train.csv columns."
            )
        else:
            logger.warning(
                "--feature-cols is ignored: TabPFN input is head penultimate features (pre-linear), not train.csv columns."
            )

    # Numeric / string hyperparams
    n_estimators = int(args.n_estimators) if args.n_estimators is not None else int(tabpfn_cfg.get("n_estimators", 8))
    device = str(args.device) if args.device is not None else str(tabpfn_cfg.get("device", "auto"))
    fit_mode = str(args.fit_mode) if args.fit_mode is not None else str(tabpfn_cfg.get("fit_mode", "fit_preprocessors"))
    inference_precision = (
        str(args.inference_precision)
        if args.inference_precision is not None
        else str(tabpfn_cfg.get("inference_precision", "auto"))
    )
    n_jobs = int(args.n_jobs) if args.n_jobs is not None else int(tabpfn_cfg.get("n_jobs", 1))

    # Boolean flags (CLI can only force-enable)
    ignore_pretraining_limits = bool(args.ignore_pretraining_limits) or bool(tabpfn_cfg.get("ignore_pretraining_limits", False))
    enable_telemetry = bool(args.enable_telemetry) or bool(tabpfn_cfg.get("enable_telemetry", False))

    # Cache dir
    model_cache_dir = args.model_cache_dir if args.model_cache_dir is not None else tabpfn_cfg.get("model_cache_dir", None)

    # Model checkpoint path (MUST be local; do not fall back to downloading).
    model_path_raw = args.model_path if args.model_path is not None else tabpfn_cfg.get("model_path", None)
    if model_path_raw is None or (isinstance(model_path_raw, str) and not model_path_raw.strip()):
        raise SystemExit(
            "Missing TabPFN local checkpoint path. Please set `tabpfn.model_path` in the config "
            "or pass `--model-path /path/to/tabpfn-v2.5-regressor-*.ckpt`."
        )
    model_path_p = Path(str(model_path_raw)).expanduser()
    if not model_path_p.is_absolute():
        model_path_p = (repo_root / model_path_p).resolve()
    if not model_path_p.is_file():
        raise SystemExit(f"TabPFN model checkpoint not found: {model_path_p}")
    model_path = str(model_path_p)

    # TabPFN settings via env (must be set before importing tabpfn)
    if not bool(enable_telemetry):
        os.environ["TABPFN_DISABLE_TELEMETRY"] = "1"
    else:
        os.environ.pop("TABPFN_DISABLE_TELEMETRY", None)
    if model_cache_dir:
        os.environ["TABPFN_MODEL_CACHE_DIR"] = str(Path(str(model_cache_dir)).expanduser())

    TabPFNRegressor = _import_tabpfn_regressor(repo_root)

    # Lazy import (sklearn is already a dependency of tabpfn; we keep it local)
    from sklearn.multioutput import MultiOutputRegressor

    # Resolve output directory (align with project conventions)
    base_log_dir = Path(cfg.get("logging", {}).get("log_dir", "outputs")).expanduser()
    base_log_dir = _resolve_under_repo(repo_root, base_log_dir)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    if args.out_dir is not None:
        out_dir = _resolve_under_repo(repo_root, args.out_dir)
    else:
        slug = f"{version}" if version else "default"
        out_dir = base_log_dir / "tabpfn" / f"{slug}-{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # IMPORTANT (Leakage prevention): use the SAME dataset config as the image model.
    # Fold weights were trained on cfg_img.data.*, so we must pivot/split the exact same data.
    data_root = _resolve_under_repo(repo_root, cfg_img["data"]["root"])
    train_csv = str(cfg_img["data"]["train_csv"])

    full_df = read_and_pivot_csiro_train_csv(
        data_root=str(data_root),
        train_csv=train_csv,
        target_order=list(cfg_img["data"]["target_order"]),
    )
    if len(full_df) < 2:
        raise ValueError(f"Not enough samples after pivoting: {len(full_df)}")

    # Build splits EXACTLY like the image training pipeline (src/training/kfold_runner.py),
    # i.e. using cfg_img (configs/train.yaml).
    if args.k is not None or args.even_split is not None or args.group_by_date_state is not None:
        logger.warning(
            "CLI k-fold overrides (--k/--even-split/--group-by-date-state) are ignored to avoid leakage. "
            "Using k-fold settings from image config: {}",
            str(img_cfg_path),
        )
    splits = build_kfold_splits(full_df, cfg_img, seed=seed)

    # For classic k-fold (even_split=False), each sample should ideally be validated exactly once.
    kfold_img_cfg = dict(cfg_img.get("kfold", {}) or {})
    even_split_effective = bool(kfold_img_cfg.get("even_split", False))
    group_by_date_state_effective = bool(kfold_img_cfg.get("group_by_date_state", True))
    oof_pred: Optional[np.ndarray]
    if even_split_effective:
        oof_pred = None
    else:
        oof_pred = np.full((len(full_df), len(TARGETS_5D_ORDER)), np.nan, dtype=np.float64)
    val_counts = np.zeros((len(full_df),), dtype=np.int64)

    y_true_full = build_y_5d(full_df, fillna=0.0)

    fold_metrics: list[dict[str, Any]] = []
    fold_pred_frames: list[pd.DataFrame] = []

    logger.info("Repo root: {}", repo_root)
    logger.info("Config: {}", cfg_path)
    logger.info("Output dir: {}", out_dir)
    logger.info("Data: {} / {}", data_root, train_csv)
    logger.info("Full pivoted samples: {}", len(full_df))
    logger.info(
        "Folds: {} (even_split={} group_by_date_state={} split_seed={})",
        len(splits),
        even_split_effective,
        group_by_date_state_effective,
        int(seed),
    )
    if image_features_mode == "dinov3_only":
        logger.info("TabPFN X: dinov3_only CLS token (no LoRA, no head) from {}", img_cfg_path)
    else:
        logger.info("TabPFN X: head penultimate features (per-fold, pre-linear) from {}", img_cfg_path)
    logger.info("Targets (5D): {}", TARGETS_5D_ORDER)
    logger.info(
        "TabPFN params: n_estimators={}, device={}, fit_mode={}, inference_precision={}, n_jobs={}",
        int(n_estimators),
        str(device),
        str(fit_mode),
        str(inference_precision),
        int(n_jobs),
    )
    logger.info("TabPFN model_path (local): {}", model_path)

    fold_feature_meta: list[dict[str, Any]] = []

    # Pre-extract global features once when they are fold-invariant (dinov3_only).
    X_all_global: Optional[np.ndarray] = None
    global_cache_path: Optional[Path] = None
    if image_features_mode == "dinov3_only":
        cache_path_raw = image_features_cfg.get("cache_path", None)
        if isinstance(cache_path_raw, str) and cache_path_raw.strip():
            base = _resolve_under_repo(repo_root, cache_path_raw)
            if base.is_dir():
                global_cache_path = (base / "dinov3_cls_features.pt").resolve()
            else:
                global_cache_path = base

        X_all_global = _extract_dinov3_cls_features_as_tabular(
            repo_root=repo_root,
            cfg_img=cfg_img,
            df=full_df,
            image_features_cfg=image_features_cfg,
            cache_path=global_cache_path,
        )
        if X_all_global.ndim != 2 or X_all_global.shape[0] != len(full_df):
            raise RuntimeError(f"Invalid extracted DINOv3 CLS feature matrix shape: {X_all_global.shape}")
        fold_feature_meta.append(
            {
                "mode": "dinov3_only",
                "shape": list(X_all_global.shape),
                "cache_path": str(global_cache_path) if global_cache_path is not None else None,
            }
        )

    for fold_idx, (train_idx, val_idx) in enumerate(splits):
        if args.max_folds is not None and fold_idx >= int(args.max_folds):
            logger.info("Stopping early due to --max-folds={} (completed {} folds).", args.max_folds, fold_idx)
            break

        head_weights_path: Optional[Path] = None
        X_all: np.ndarray
        if image_features_mode == "dinov3_only":
            if X_all_global is None:
                raise RuntimeError("Internal error: X_all_global is None for dinov3_only mode.")
            X_all = X_all_global
        else:
            # Resolve fold-specific head weights; skip missing folds.
            head_weights_path = _resolve_fold_head_weights_path(repo_root=repo_root, cfg_img=cfg_img, fold_idx=int(fold_idx))
            if head_weights_path is None:
                logger.warning(
                    "Skipping fold {}: no fold-specific head weights found (expected under outputs/checkpoints/{}/fold_{}/head/ or weights/head/fold_{}/).",
                    int(fold_idx),
                    str(cfg_img.get("version", "") or "").strip(),
                    int(fold_idx),
                    int(fold_idx),
                )
                continue

            # Optional per-fold feature cache (avoid overwriting across folds).
            cache_path: Optional[Path] = None
            cache_path_raw = image_features_cfg.get("cache_path", None)
            if isinstance(cache_path_raw, str) and cache_path_raw.strip():
                base = _resolve_under_repo(repo_root, cache_path_raw)
                if base.is_dir():
                    cache_path = (base / f"head_penultimate_features.fold_{fold_idx}.pt").resolve()
                else:
                    # Insert fold suffix before extension
                    cache_path = base.with_name(f"{base.stem}.fold_{fold_idx}{base.suffix}")

            # Extract features using THIS fold's image head (prevents train_all leakage).
            X_all = _extract_head_penultimate_features_as_tabular(
                repo_root=repo_root,
                cfg_img=cfg_img,
                df=full_df,
                image_features_cfg=image_features_cfg,
                head_weights_path=head_weights_path,
                cache_path=cache_path,
            )
            if X_all.ndim != 2 or X_all.shape[0] != len(full_df):
                raise RuntimeError(f"Invalid extracted feature matrix shape for fold {fold_idx}: {X_all.shape}")

            fold_feature_meta.append(
                {
                    "fold": int(fold_idx),
                    "mode": "head_penultimate",
                    "head_weights_path": str(head_weights_path),
                    "shape": list(X_all.shape),
                    "cache_path": str(cache_path) if cache_path is not None else None,
                }
            )

        train_df = full_df.iloc[train_idx].reset_index(drop=True)
        val_df = full_df.iloc[val_idx].reset_index(drop=True)

        X_train = X_all[train_idx]
        X_val = X_all[val_idx]
        y_train = build_y_5d(train_df, fillna=0.0)
        y_val = build_y_5d(val_df, fillna=0.0)

        base = TabPFNRegressor(
            n_estimators=int(n_estimators),
            device=str(device),
            fit_mode=str(fit_mode),
            inference_precision=str(inference_precision),
            random_state=int(seed + fold_idx),
            ignore_pretraining_limits=bool(ignore_pretraining_limits),
            model_path=str(model_path),
        )
        model = MultiOutputRegressor(base, n_jobs=int(n_jobs))
        try:
            model.fit(X_train, y_train)
        except RuntimeError as e:
            msg = str(e)
            msg_l = msg.lower()
            if (
                "gated" in msg_l
                or "authentication error" in msg_l
                or "hf auth login" in msg_l
                or "hf_token" in msg_l
                or "unauthorized" in msg_l
            ):
                logger.error(
                    "TabPFN 仍然尝试从 HuggingFace 获取权重/配置，但本脚本已被设置为只用本地 ckpt。\n"
                    "请确认 `tabpfn.model_path` / `--model-path` 指向一个有效的本地 `.ckpt` 文件：{}\n"
                    "原始错误：{}",
                    model_path,
                    msg.strip(),
                )
                raise SystemExit(2) from e
            raise
        y_pred = model.predict(X_val)

        r2, r2_per = weighted_r2_logspace(y_val, y_pred, return_per_target=True)
        m = {
            "fold": int(fold_idx),
            "feature_mode": str(image_features_mode),
            "head_weights_path": str(head_weights_path) if head_weights_path is not None else None,
            "num_train": int(len(train_df)),
            "num_val": int(len(val_df)),
            "r2_weighted": float(r2),
            "r2_per_target_logspace": {
                name: float(r2_per[i]) for i, name in enumerate(TARGETS_5D_ORDER)
            },
        }
        fold_metrics.append(m)
        logger.info(
            "Fold {}: train={} val={} weighted_r2(log)={:.6f}",
            fold_idx,
            len(train_df),
            len(val_df),
            float(r2),
        )

        # Store predictions for inspection
        pred_df = val_df[["image_id"]].copy()
        pred_df["fold"] = int(fold_idx)
        for j, name in enumerate(TARGETS_5D_ORDER):
            pred_df[f"pred_{name}"] = y_pred[:, j]
            pred_df[f"true_{name}"] = y_val[:, j]
        fold_pred_frames.append(pred_df)

        # Track validation coverage / OOF predictions when applicable
        val_counts[val_idx] += 1
        if oof_pred is not None:
            oof_pred[val_idx] = y_pred

    # Save artifacts
    _save_json(
        out_dir / "run_args.json",
        {k: getattr(args, k) for k in vars(args).keys()},
    )
    _save_json(
        out_dir / "summary_config.json",
        {
            "seed": int(seed),
            "config_path": str(cfg_path),
            "version": version,
            "tabpfn_x": {
                "mode": str(image_features_mode),
                "source": (
                    "dinov3_cls_token_no_lora_no_head"
                    if image_features_mode == "dinov3_only"
                    else "image_head_penultimate_pre_linear_per_fold"
                ),
                "folds": fold_feature_meta,
                "image_train_config": str(img_cfg_path),
                "fusion": image_features_cfg.get("fusion", "mean") if image_features_mode != "dinov3_only" else None,
            },
            "targets_5d_order": list(TARGETS_5D_ORDER),
            "kfold": {
                "k": int(len(splits)),
                "even_split": bool(even_split_effective),
                "group_by_date_state": bool(group_by_date_state_effective),
            },
        },
    )
    _save_json(out_dir / "fold_metrics.json", fold_metrics)

    preds_all = pd.concat(fold_pred_frames, axis=0, ignore_index=True) if fold_pred_frames else pd.DataFrame()
    preds_all.to_csv(out_dir / "val_predictions.csv", index=False)

    # Coverage / aggregate metrics
    coverage = {
        "num_samples": int(len(full_df)),
        "val_counts_min": int(val_counts.min()) if len(val_counts) else 0,
        "val_counts_max": int(val_counts.max()) if len(val_counts) else 0,
        "val_counts_mean": float(val_counts.mean()) if len(val_counts) else 0.0,
    }

    summary: dict[str, Any] = {
        "folds_r2_weighted_mean": float(np.mean([m["r2_weighted"] for m in fold_metrics])) if fold_metrics else None,
        "folds_r2_weighted_std": float(np.std([m["r2_weighted"] for m in fold_metrics])) if fold_metrics else None,
        "coverage": coverage,
    }

    if oof_pred is not None:
        ok = np.isfinite(oof_pred).all(axis=1)
        summary["oof_coverage_frac"] = float(ok.mean())
        if ok.any():
            summary["oof_r2_weighted"] = float(
                weighted_r2_logspace(y_true_full[ok], oof_pred[ok])
            )
        else:
            summary["oof_r2_weighted"] = None

    _save_json(out_dir / "summary.json", summary)

    logger.info("Saved fold metrics -> {}", out_dir / "fold_metrics.json")
    logger.info("Saved val predictions -> {}", out_dir / "val_predictions.csv")
    logger.info("Saved summary -> {}", out_dir / "summary.json")

    if summary.get("oof_r2_weighted", None) is not None:
        logger.info("OOF weighted R2 (log-space): {:.6f}", float(summary["oof_r2_weighted"]))
    else:
        logger.info(
            "Mean weighted R2 across folds (log-space): {:.6f}",
            float(summary["folds_r2_weighted_mean"]) if summary.get("folds_r2_weighted_mean") is not None else float("nan"),
        )


if __name__ == "__main__":
    main()


