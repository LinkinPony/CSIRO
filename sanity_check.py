import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import yaml
from loguru import logger
import time

# Allowlist PEFT types for PyTorch 2.6+ safe deserialization (weights_only=True)
try:
    from torch.serialization import add_safe_globals  # type: ignore
except Exception:
    add_safe_globals = None  # type: ignore

try:
    from peft.utils.peft_types import PeftType  # type: ignore
except Exception:
    try:
        # Try to enable third_party peft import path via project setup (later)
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


def parse_args():
    p = argparse.ArgumentParser(description="Sanity check runner (config-driven)")
    p.add_argument(
        "--config",
        type=str,
        default=str(Path(__file__).parent / "configs" / "sanity.yaml"),
        help="Path to sanity check YAML config file (not the training config)",
    )
    return p.parse_args()


def load_cfg(config_path: str) -> Dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_sanity_cfg(config_path: str) -> Dict:
    y = load_cfg(config_path)
    s = y.get("sanity", y)
    return s


def _to_abs(base_dir: str, p: str) -> str:
    if not p:
        return p
    pp = Path(p)
    return str(pp if pp.is_absolute() else (Path(base_dir) / pp))


def setup_project_imports(project_dir: str) -> None:
    import sys
    project_dir_abs = str(Path(project_dir).resolve())
    configs_dir = str(Path(project_dir_abs) / "configs")
    src_dir = str(Path(project_dir_abs) / "src")
    if not os.path.isdir(configs_dir):
        raise RuntimeError(f"configs/ not found under project_dir: {configs_dir}")
    if not os.path.isdir(src_dir):
        raise RuntimeError(f"src/ not found under project_dir: {src_dir}")
    if project_dir_abs not in sys.path:
        sys.path.insert(0, project_dir_abs)


def resolve_dirs_for_project(cfg: Dict, project_dir: str) -> Tuple[Path, Path, str]:
    def _project_path(p: str) -> Path:
        pp = Path(p)
        return pp if pp.is_absolute() else Path(project_dir) / pp

    base_log_dir = _project_path(str(cfg["logging"]["log_dir"]))
    base_ckpt_dir = _project_path(str(cfg["logging"]["ckpt_dir"]))
    version = cfg.get("version", None)
    version = None if version in (None, "", "null") else str(version)
    log_dir = base_log_dir / version if version else base_log_dir
    ckpt_dir = base_ckpt_dir / version if version else base_ckpt_dir
    return log_dir, ckpt_dir, (version or "")


def _parse_image_size(value) -> Tuple[int, int]:
    try:
        if isinstance(value, (list, tuple)) and len(value) == 2:
            w, h = int(value[0]), int(value[1])
            return (int(h), int(w))
        v = int(value)
        return (v, v)
    except Exception:
        v = int(value)
        return (v, v)


class SimpleImageDataset(Dataset):
    def __init__(self, image_paths: List[str], root_dir: str, transform: T.Compose) -> None:
        self.image_paths = list(image_paths)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int):
        rel_path = self.image_paths[index]
        full_path = os.path.join(self.root_dir, rel_path)
        from PIL import Image  # deferred import

        image = Image.open(full_path).convert("RGB")
        image = self.transform(image)
        return image, rel_path


def build_val_transforms(image_size: Tuple[int, int], mean: List[float], std: List[float]) -> T.Compose:
    return T.Compose([
        T.Resize(image_size, interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
    ])


def ensure_packaged_weights(config_path: str, weights_dir: Path) -> None:
    # Import and invoke packager to export head weights and copy project sources under weights/
    import importlib.util
    spec = importlib.util.spec_from_file_location("package_artifacts", str(Path(__file__).parent / "package_artifacts.py"))
    if spec is None or spec.loader is None:
        raise RuntimeError("Failed to import package_artifacts.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[attr-defined]

    cfg = module.load_cfg(config_path)
    _ = module.resolve_dirs(cfg)
    # Emulate CLI args by temporarily patching sys.argv for defaults
    import sys
    argv_backup = list(sys.argv)
    try:
        sys.argv = [argv_backup[0], "--config", config_path, "--weights-dir", str(weights_dir)]
        module.main()
    finally:
        sys.argv = argv_backup


def discover_fold_head_paths(weights_dir: Path, k: int) -> List[Path]:
    paths: List[Path] = []
    for i in range(k):
        p = weights_dir / "head" / f"fold_{i}" / "infer_head.pt"
        if not p.is_file():
            raise FileNotFoundError(f"Missing head weights for fold {i}: {p}")
        paths.append(p)
    return paths


def discover_all_head_paths(weights_dir: Path) -> List[Path]:
    head_root = weights_dir / "head"
    if not head_root.exists():
        return []
    # Prefer per-fold heads
    fold_paths: List[Path] = []
    for name in sorted(os.listdir(head_root)):
        if name.startswith("fold_"):
            cand = head_root / name / "infer_head.pt"
            if cand.is_file():
                fold_paths.append(cand)
    if fold_paths:
        return fold_paths
    single = head_root / "infer_head.pt"
    return [single] if single.is_file() else []


def load_head_module(cfg: Dict) -> nn.Module:
    from src.models.head_builder import build_head_layer

    return build_head_layer(
        embedding_dim=int(cfg["model"]["embedding_dim"]),
        num_outputs=3,
        head_hidden_dims=list(cfg["model"]["head"].get("hidden_dims", [512, 256])),
        # Head activation is configurable via YAML; default to ReLU when unspecified.
        head_activation=str(cfg["model"]["head"].get("activation", "relu")),
        dropout=float(cfg["model"]["head"].get("dropout", 0.0)),
        use_output_softplus=bool(cfg["model"]["head"].get("use_output_softplus", True)),
    )


def load_head_state_dict(pt_path: Path) -> Dict[str, torch.Tensor]:
    obj = torch.load(str(pt_path), map_location="cpu")
    if isinstance(obj, dict) and "state_dict" in obj:
        return obj["state_dict"]
    if isinstance(obj, dict):
        return obj
    raise RuntimeError(f"Unsupported head weights file: {pt_path}")


def build_feature_extractor_offline(dinov3_source_dir: str, dino_weights_path: str, backbone_name: str = "dinov3_vitl16") -> nn.Module:
    # Use the same offline approach as infer_and_submit_pt.py to avoid torch.hub
    import sys
    if dinov3_source_dir and os.path.isdir(dinov3_source_dir) and dinov3_source_dir not in sys.path:
        sys.path.insert(0, dinov3_source_dir)
    try:
        if backbone_name == "dinov3_vith16plus":
            from dinov3.hub.backbones import dinov3_vith16plus as _make_backbone  # type: ignore
        elif backbone_name == "dinov3_vitl16":
            from dinov3.hub.backbones import dinov3_vitl16 as _make_backbone  # type: ignore
        elif backbone_name in ("dinov3_vit7b16", "dinov3_vit7b", "vit7b16", "vit7b"):
            from dinov3.hub.backbones import dinov3_vit7b16 as _make_backbone  # type: ignore
        else:
            raise ImportError(f"Unsupported backbone: {backbone_name}")
    except Exception as e:
        raise ImportError(
            f"Failed to import dinov3 backbones from '{dinov3_source_dir}'. Please ensure the path is correct. Error: {e}"
        )

    backbone = _make_backbone(pretrained=False)
    state = torch.load(dino_weights_path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    try:
        backbone.load_state_dict(state, strict=True)
    except Exception:
        backbone.load_state_dict(state, strict=False)

    from src.models.backbone import DinoV3FeatureExtractor as _FX
    return _FX(backbone)


def extract_features(
    feature_extractor: nn.Module,
    root_dir: str,
    image_paths: List[str],
    transform: T.Compose,
    batch_size: int,
    num_workers: int,
) -> Tuple[List[str], torch.Tensor]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ds = SimpleImageDataset(image_paths=image_paths, root_dir=root_dir, transform=transform)
    dl = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=bool(num_workers > 0),
    )

    feature_extractor.eval().to(device)
    rels: List[str] = []
    feats_cpu: List[torch.Tensor] = []
    with torch.inference_mode():
        for images, rel_paths in dl:
            images = images.to(device, non_blocking=True)
            feats = feature_extractor(images)
            feats_cpu.append(feats.detach().cpu().float())
            rels.extend(list(rel_paths))
    features = torch.cat(feats_cpu, dim=0) if feats_cpu else torch.empty((0, 0), dtype=torch.float32)
    return rels, features


def predict_from_features(features_cpu: torch.Tensor, head: nn.Module, batch_size: int) -> torch.Tensor:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    head = head.eval().to(device)
    n = features_cpu.shape[0]
    outputs: List[torch.Tensor] = []
    with torch.inference_mode():
        for i in range(0, n, max(1, batch_size)):
            chunk = features_cpu[i : i + max(1, batch_size)].to(device, non_blocking=True)
            out = head(chunk)
            outputs.append(out.detach().cpu().float())
    return torch.cat(outputs, dim=0) if outputs else torch.empty((0, 3), dtype=torch.float32)


def compute_metrics(
    preds: torch.Tensor,
    targets: torch.Tensor,
    *,
    global_targets: Optional[torch.Tensor] = None,
    target_names: Optional[List[str]] = None,
) -> Dict[str, float]:
    """
    Compute regression metrics with R^2 defined in log space and using a global
    dataset mean for the R^2 baseline, following the competition DESCRIPTION.

    Args:
        preds:   (N, D) predictions in grams.
        targets: (N, D) ground-truth in grams for the current evaluation subset.
        global_targets: Optional (M, D) tensor of ground-truth grams over the
            full dataset used to define the baseline mean for R^2.
        target_names: Optional list of D target names used to map weights.
    """
    preds = preds.float()
    targets = targets.float()

    # Basic metrics (linear space, grams)
    diff = preds - targets
    mse = torch.mean(diff ** 2).item()
    mae = torch.mean(torch.abs(diff)).item()
    rmse = float(np.sqrt(mse))

    # --- R^2 in log space ---
    eps = 1e-8
    preds_clamp = preds.clamp_min(0.0)
    targets_clamp = targets.clamp_min(0.0)
    preds_log = torch.log1p(preds_clamp)
    targets_log = torch.log1p(targets_clamp)

    # Baseline: constant predictor equal to the global dataset mean (in grams)
    # evaluated in log-space.
    if global_targets is not None:
        g = global_targets.float().clamp_min(0.0)
        g_mean = torch.mean(g, dim=0)  # (D,) mean in grams over full dataset
        mean_log = torch.log1p(g_mean)
    else:
        # Fallback: mean over the current evaluation subset
        mean_log = torch.mean(targets_log, dim=0)

    # Per-target R^2 in log space
    diff_log = preds_log - targets_log
    ss_res_per = torch.sum(diff_log ** 2, dim=0)  # (D,)
    ss_tot_per = torch.sum((targets_log - mean_log) ** 2, dim=0)
    r2_per = 1.0 - (ss_res_per / (ss_tot_per + eps))

    # Map DESCRIPTION weights onto the provided target names
    if target_names is None:
        target_names = [f"dim_{i}" for i in range(r2_per.shape[0])]
    weight_map = {
        "Dry_Green_g": 0.1,
        "Dry_Dead_g": 0.1,
        "Dry_Clover_g": 0.1,
        "GDM_g": 0.2,
        "Dry_Total_g": 0.5,
    }
    weights_list: List[float] = []
    for name in target_names:
        w = float(weight_map.get(str(name), 1.0))
        weights_list.append(w)
    w = torch.tensor(weights_list, dtype=r2_per.dtype)
    w = w.to(device=r2_per.device)

    # Weighted R^2 across dimensions (competition-style aggregation)
    valid = torch.isfinite(r2_per)
    w_eff = w * valid.to(dtype=w.dtype)
    denom = w_eff.sum().clamp_min(eps)
    r2_weighted = float((w_eff * r2_per).sum() / denom)

    # Per-target metrics in linear space (kept for backward compatibility)
    per_target_mse = torch.mean(diff ** 2, dim=0).tolist()
    per_target_mae = torch.mean(torch.abs(diff), dim=0).tolist()

    out: Dict[str, float] = {
        "mse": mse,
        "mae": mae,
        "rmse": rmse,
        "r2": r2_weighted,
    }
    # Legacy naming for 3 base components when available
    if len(per_target_mse) >= 3 and len(per_target_mae) >= 3:
        out.update(
            {
                "mse_dc": float(per_target_mse[0]),
                "mse_dd": float(per_target_mse[1]),
                "mse_dg": float(per_target_mse[2]),
                "mae_dc": float(per_target_mae[0]),
                "mae_dd": float(per_target_mae[1]),
                "mae_dg": float(per_target_mae[2]),
            }
        )
    return out


def _write_test_csv_for_images(
    image_paths: List[str],
    target_order: List[str],
    out_csv_path: Path,
) -> None:
    rows: List[Dict[str, str]] = []
    # Build rows: one per (image, target) for the three base targets
    for rel_path in image_paths:
        for t in target_order:
            # Use a stable sample_id that preserves mapping back
            sid = f"{rel_path}__{t}"
            rows.append({
                "sample_id": sid,
                "image_path": rel_path,
                "target_name": t,
            })
    df = pd.DataFrame(rows, columns=["sample_id", "image_path", "target_name"])  # type: ignore
    out_csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv_path, index=False)


def _run_infer_module(
    *,
    project_dir: str,
    head_path_or_dir: Path,
    dino_weights_path: str,
    input_csv_path: Path,
    output_submission_path: Path,
) -> None:
    # Import once and reuse the single inference implementation
    import importlib
    infer = importlib.import_module("infer_and_submit_pt")

    # Override module variables BEFORE calling main()
    infer.PROJECT_DIR = project_dir
    # Also update derived absolute path used internally
    try:
        infer._PROJECT_DIR_ABS = os.path.abspath(project_dir)  # type: ignore[attr-defined]
    except Exception:
        pass

    infer.HEAD_WEIGHTS_PT_PATH = str(head_path_or_dir)
    infer.DINO_WEIGHTS_PT_PATH = str(dino_weights_path)
    infer.INPUT_PATH = str(input_csv_path)
    infer.OUTPUT_SUBMISSION_PATH = str(output_submission_path)

    # Execute the single source of truth inference
    infer.main()


def _read_submission_as_wide(
    submission_csv: Path,
    test_csv: Path,
) -> pd.DataFrame:
    # Join back to recover image_path and target_name, then pivot to wide per image_path
    sub = pd.read_csv(str(submission_csv))  # columns: sample_id, target
    test_df = pd.read_csv(str(test_csv))    # columns: sample_id, image_path, target_name
    merged = sub.merge(test_df, on="sample_id", how="inner")
    # Pivot to columns pred_dc, pred_dd, pred_dg in train target order
    pv = merged.pivot_table(index="image_path", columns="target_name", values="target", aggfunc="first").reset_index()
    return pv


def main():
    args = parse_args()
    sanity = load_sanity_cfg(args.config)

    repo_root = Path(__file__).parent
    project_dir = _to_abs(str(repo_root), sanity.get("project_dir", "."))
    setup_project_imports(project_dir)

    # Load training config (source of truth for data/model)
    train_config_path = _to_abs(project_dir, sanity.get("train_config", "configs/train.yaml"))
    train_cfg = load_cfg(train_config_path)
    log_dir, ckpt_dir, version = resolve_dirs_for_project(train_cfg, project_dir)

    weights_dir = Path(_to_abs(project_dir, sanity.get("weights_dir", "weights"))).expanduser().resolve()
    out_dir = Path(_to_abs(project_dir, sanity.get("out_dir", "outputs/sanity"))).expanduser().resolve()
    out_ver_dir = out_dir / (version or "default")
    out_ver_dir.mkdir(parents=True, exist_ok=True)

    # Setup logging sinks
    logger.add(out_ver_dir / "sanity.log", rotation="10 MB", retention="7 days", level="DEBUG")
    logger.info("Sanity check started")
    logger.info("Using project_dir: {}", project_dir)
    logger.info("Sanity config: {}", args.config)
    logger.info("Train config: {}", train_config_path)
    logger.info("Version: {}  log_dir: {}  ckpt_dir: {}", version or "default", str(log_dir), str(ckpt_dir))
    logger.info("Weights dir: {}  Outputs dir: {}", str(weights_dir), str(out_ver_dir))

    # 1) Package weights
    logger.info("[1/4] Packaging weights using package_artifacts.py ...")
    t0 = time.time()
    ensure_packaged_weights(train_config_path, weights_dir)
    logger.success("Weights packaged in {:.2f}s", time.time() - t0)

    # Data settings from training config
    data_root = _to_abs(project_dir, str(train_cfg["data"]["root"]))
    train_csv = str(train_cfg["data"]["train_csv"])  # used only for reading; path in data_root
    image_size = _parse_image_size(train_cfg["data"]["image_size"])
    mean = list(train_cfg["data"]["normalization"]["mean"])
    std = list(train_cfg["data"]["normalization"]["std"])
    dm_target_order = list(train_cfg["data"]["target_order"])
    # Fixed 5D targets used for all R^2 computations (competition definition)
    score_targets = ["Dry_Clover_g", "Dry_Dead_g", "Dry_Green_g", "GDM_g", "Dry_Total_g"]
    val_batch_size = int(train_cfg["data"].get("val_batch_size", train_cfg["data"]["batch_size"]))
    num_workers = int(train_cfg["data"].get("num_workers", 4))
    logger.info(
        "Data settings: root={} image_size={} dm_targets={} score_targets={} val_bs={} workers={}",
        data_root,
        image_size,
        dm_target_order,
        score_targets,
        val_batch_size,
        num_workers,
    )

    # 2) Build full dataframe using project datamodule logic
    logger.info("[2/4] Building full dataframe from training CSV ...")
    from src.data.datamodule import PastureDataModule

    dm = PastureDataModule(
        data_root=data_root,
        train_csv=train_csv,
        image_size=image_size,
        batch_size=val_batch_size,
        val_batch_size=val_batch_size,
        num_workers=num_workers,
        val_split=float(train_cfg["data"]["val_split"]),
        target_order=dm_target_order,
        mean=mean,
        std=std,
        train_scale=tuple(train_cfg["data"]["augment"]["random_resized_crop_scale"]),
        hflip_prob=float(train_cfg["data"]["augment"]["horizontal_flip_prob"]),
        shuffle=bool(train_cfg["data"].get("shuffle", True)),
    )
    full_df = dm.build_full_dataframe()
    logger.info("Full dataset rows: {} (unique images)", len(full_df))

    # 3) Configure shared resources for inference module (single source of truth)
    logger.info("[3/4] Preparing single-source inference via infer_and_submit_pt.py ...")
    cfg_dino_weights = sanity.get("dino_weights_path", "")
    dino_weights_path = _to_abs(project_dir, cfg_dino_weights) if cfg_dino_weights else ""
    if not dino_weights_path or not os.path.isfile(dino_weights_path):
        cfg_weights_path = train_cfg["model"].get("weights_path", None)
        if not (cfg_weights_path and os.path.isfile(_to_abs(project_dir, str(cfg_weights_path)))):
            raise FileNotFoundError(
                "DINOv3 weights not found. Provide sanity.dino_weights_path or model.weights_path in training config."
            )
        dino_weights_path = _to_abs(project_dir, str(cfg_weights_path))
    logger.info("DINO weights: {}", dino_weights_path)
    dinov3_source = _to_abs(project_dir, sanity.get("dinov3_source", "third_party/dinov3/dinov3"))
    logger.info("dinov3 source: {}", dinov3_source)

    # 4) Per-fold validation inference using each fold's head (if kfold is enabled)
    k_cfg = int(train_cfg.get("kfold", {}).get("k", 5))
    fold_splits_root = Path(log_dir) / "folds"
    if not fold_splits_root.is_dir():
        raise FileNotFoundError(f"Fold splits directory not found: {fold_splits_root}. Ensure k-fold training ran.")

    fold_head_paths = discover_fold_head_paths(weights_dir, k=k_cfg)
    logger.info("Found {} fold heads", len(fold_head_paths))

    per_fold_metrics: List[Dict[str, float]] = []
    per_fold_rows: List[Dict[str, object]] = []
    all_fold_preds: List[torch.Tensor] = []
    all_fold_targets: List[torch.Tensor] = []

    logger.info("[4/4] Running per-fold validation inference via infer_and_submit_pt ...")
    # Pre-compute global targets (in grams) for metric baselines (fixed 5D order).
    full_targets_tensor = torch.from_numpy(
        full_df[score_targets].to_numpy(dtype=np.float32)
    )

    for fold_idx in range(k_cfg):
        val_csv = fold_splits_root / f"fold_{fold_idx}" / "val.csv"
        if not val_csv.is_file():
            raise FileNotFoundError(f"Missing val.csv for fold {fold_idx}: {val_csv}")
        vdf = pd.read_csv(str(val_csv))  # columns: image_id, image_path
        merged = vdf.merge(full_df, on=["image_id", "image_path"], how="inner")
        if len(merged) == 0:
            raise RuntimeError(f"Fold {fold_idx}: No overlapping rows between val.csv and full_df")
        logger.info("Fold {}: val rows={} ({})", fold_idx, len(merged), val_csv.name)

        image_paths = merged["image_path"].astype(str).tolist()
        targets = torch.from_numpy(merged[score_targets].to_numpy(dtype=np.float32))

        tmp_dir = out_ver_dir / "tmp"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        # IMPORTANT: place test.csv inside the true dataset_root so image paths resolve
        test_csv_path = Path(data_root) / f"sanity_fold_{fold_idx}_test.csv"
        sub_csv_path = tmp_dir / f"fold_{fold_idx}_submission.csv"

        # Build test.csv for this fold and run the unified inference
        _write_test_csv_for_images(
            image_paths=image_paths,
            target_order=score_targets,
            out_csv_path=test_csv_path,
        )

        # Point the inference module to the specific fold head directory
        head_dir_for_fold = fold_head_paths[fold_idx].parent
        _run_infer_module(
            project_dir=str(project_dir),
            head_path_or_dir=head_dir_for_fold,
            dino_weights_path=str(dino_weights_path),
            input_csv_path=test_csv_path,
            output_submission_path=sub_csv_path,
        )

        # Read predictions and compute metrics
        wide_preds = _read_submission_as_wide(sub_csv_path, test_csv_path)
        # Ensure order matches image_paths; collect predictions in fixed 5D order.
        pred_map = {
            r["image_path"]: [r.get(name, 0.0) for name in score_targets]
            for _, r in wide_preds.iterrows()
        }
        preds_mat = torch.tensor([pred_map[p] for p in image_paths], dtype=torch.float32)

        # Accumulate for global cross-fold metrics (5D).
        all_fold_preds.append(preds_mat)
        all_fold_targets.append(targets)

        m = compute_metrics(
            preds=preds_mat,
            targets=targets,
            global_targets=full_targets_tensor,
            target_names=score_targets,
        )
        per_fold_metrics.append({"fold": float(fold_idx), **m})
        logger.info(
            "Fold {} metrics: MSE={:.6f} MAE={:.6f} RMSE={:.6f} R2={:.4f}",
            fold_idx,
            m["mse"],
            m["mae"],
            m["rmse"],
            m["r2"],
        )

        for i in range(len(merged)):
            per_fold_rows.append(
                {
                    "fold": fold_idx,
                    "fold": fold_idx,
                    "image_id": merged.iloc[i]["image_id"],
                    "image_path": merged.iloc[i]["image_path"],
                    "target_dc": float(merged.iloc[i]["Dry_Clover_g"]),
                    "target_dd": float(merged.iloc[i]["Dry_Dead_g"]),
                    "target_dg": float(merged.iloc[i]["Dry_Green_g"]),
                    "target_gdm": float(merged.iloc[i]["GDM_g"]),
                    "target_dt": float(merged.iloc[i]["Dry_Total_g"]),
                    "pred_dc": float(preds_mat[i, 0].item()),
                    "pred_dd": float(preds_mat[i, 1].item()),
                    "pred_dg": float(preds_mat[i, 2].item()),
                    "pred_gdm": float(preds_mat[i, 3].item()),
                    "pred_dt": float(preds_mat[i, 4].item()),
                }
            )

    per_fold_pred_df = pd.DataFrame(per_fold_rows)
    per_fold_pred_path = out_ver_dir / "per_fold_val_predictions.csv"
    per_fold_metrics_path = out_ver_dir / "per_fold_metrics.json"
    per_fold_pred_df.to_csv(per_fold_pred_path, index=False)
    logger.success("Saved per-fold predictions: {} rows -> {}", len(per_fold_pred_df), per_fold_pred_path)
    with open(per_fold_metrics_path, "w", encoding="utf-8") as f:
        json.dump(per_fold_metrics, f, indent=2)
    logger.success("Saved per-fold metrics JSON -> {}", per_fold_metrics_path)

    # Aggregate across all folds using concatenated 5D predictions/targets.
    all_preds_tensor = torch.cat(all_fold_preds, dim=0)
    all_targets_tensor = torch.cat(all_fold_targets, dim=0)
    agg_per_fold_metrics = compute_metrics(
        all_preds_tensor,
        all_targets_tensor,
        global_targets=full_targets_tensor,
        target_names=score_targets,
    )
    logger.info(
        "Aggregated per-fold metrics: MSE={:.6f} MAE={:.6f} RMSE={:.6f} R2={:.4f}",
        agg_per_fold_metrics["mse"],
        agg_per_fold_metrics["mae"],
        agg_per_fold_metrics["rmse"],
        agg_per_fold_metrics["r2"],
    )

    # Ensemble over all available heads by pointing inference to weights/head/
    all_head_paths = discover_all_head_paths(weights_dir)
    if not all_head_paths:
        raise FileNotFoundError(f"No head weights found under: {weights_dir / 'head'}")
    used_heads = len(all_head_paths)
    logger.info("Ensemble heads discovered: {}", used_heads)

    unique_paths = full_df["image_path"].astype(str).tolist()

    tmp_dir = out_ver_dir / "tmp"
    # IMPORTANT: place test.csv inside the true dataset_root so image paths resolve
    test_csv_path = Path(data_root) / "sanity_full_test.csv"
    sub_csv_path = tmp_dir / "ensemble_full_submission.csv"
    _write_test_csv_for_images(
        image_paths=unique_paths,
        target_order=score_targets,
        out_csv_path=test_csv_path,
    )

    _run_infer_module(
        project_dir=str(project_dir),
        head_path_or_dir=weights_dir / "head",
        dino_weights_path=str(dino_weights_path),
        input_csv_path=test_csv_path,
        output_submission_path=sub_csv_path,
    )

    wide_preds = _read_submission_as_wide(sub_csv_path, test_csv_path)
    # Rename predictions for export while keeping 5D info
    df_full_preds = wide_preds.rename(
        columns={
            "Dry_Clover_g": "pred_dc",
            "Dry_Dead_g": "pred_dd",
            "Dry_Green_g": "pred_dg",
            "GDM_g": "pred_gdm",
            "Dry_Total_g": "pred_dt",
        }
    )

    merged_full = full_df.merge(df_full_preds, on="image_path", how="inner")
    ens_targets = torch.from_numpy(
        merged_full[score_targets].to_numpy(dtype=np.float32)
    )
    ens_preds = torch.from_numpy(
        merged_full[["pred_dc", "pred_dd", "pred_dg", "pred_gdm", "pred_dt"]].to_numpy(
            dtype=np.float32
        )
    )
    ensemble_metrics = compute_metrics(
        ens_preds,
        ens_targets,
        global_targets=full_targets_tensor,
        target_names=score_targets,
    )
    logger.info(
        "Ensemble full-dataset metrics: MSE={:.6f} MAE={:.6f} RMSE={:.6f} R2={:.4f}",
        ensemble_metrics["mse"],
        ensemble_metrics["mae"],
        ensemble_metrics["rmse"],
        ensemble_metrics["r2"],
    )

    merged_full_out = merged_full[
        [
            "image_id",
            "image_path",
            *score_targets,
            "pred_dc",
            "pred_dd",
            "pred_dg",
            "pred_gdm",
            "pred_dt",
        ]
    ].copy()
    ens_pred_path = out_ver_dir / "ensemble_full_predictions.csv"
    merged_full_out.to_csv(ens_pred_path, index=False)
    logger.success("Saved ensemble predictions -> {}", ens_pred_path)

    comp_left = per_fold_pred_df[
        [
            "image_id",
            "image_path",
            "target_dc",
            "target_dd",
            "target_dg",
            "target_gdm",
            "target_dt",
            "pred_dc",
            "pred_dd",
            "pred_dg",
            "pred_gdm",
            "pred_dt",
        ]
    ]
    comp_right = merged_full_out[
        ["image_id", "image_path", "pred_dc", "pred_dd", "pred_dg", "pred_gdm", "pred_dt"]
    ]
    comp_right = comp_right.rename(
        columns={
            "pred_dc": "ens_pred_dc",
            "pred_dd": "ens_pred_dd",
            "pred_dg": "ens_pred_dg",
            "pred_gdm": "ens_pred_gdm",
            "pred_dt": "ens_pred_dt",
        }
    )
    comp = comp_left.merge(comp_right, on=["image_id", "image_path"], how="inner")
    comp["abs_diff_dc"] = (comp["pred_dc"] - comp["ens_pred_dc"]).abs()
    comp["abs_diff_dd"] = (comp["pred_dd"] - comp["ens_pred_dd"]).abs()
    comp["abs_diff_dg"] = (comp["pred_dg"] - comp["ens_pred_dg"]).abs()
    comp["abs_diff_gdm"] = (comp["pred_gdm"] - comp["ens_pred_gdm"]).abs()
    comp["abs_diff_dt"] = (comp["pred_dt"] - comp["ens_pred_dt"]).abs()
    comp_path = out_ver_dir / "comparison_per_fold_vs_ensemble.csv"
    comp.to_csv(comp_path, index=False)
    logger.success("Saved comparison CSV ({} rows) -> {}", len(comp), comp_path)

    diff_stats = {
        "mean_abs_diff_dc": float(comp["abs_diff_dc"].mean()),
        "mean_abs_diff_dd": float(comp["abs_diff_dd"].mean()),
        "mean_abs_diff_dg": float(comp["abs_diff_dg"].mean()),
        "mean_abs_diff_gdm": float(comp["abs_diff_gdm"].mean()),
        "mean_abs_diff_dt": float(comp["abs_diff_dt"].mean()),
        "max_abs_diff_dc": float(comp["abs_diff_dc"].max()),
        "max_abs_diff_dd": float(comp["abs_diff_dd"].max()),
        "max_abs_diff_dg": float(comp["abs_diff_dg"].max()),
        "max_abs_diff_gdm": float(comp["abs_diff_gdm"].max()),
        "max_abs_diff_dt": float(comp["abs_diff_dt"].max()),
        "num_rows_compared": int(len(comp)),
    }

    summary = {
        "version": version,
        "per_fold_metrics_aggregated": agg_per_fold_metrics,
        "ensemble_full_metrics": ensemble_metrics,
        "difference_stats": diff_stats,
        "used_heads": used_heads,
        "folds": k_cfg,
        "log_dir": str(log_dir),
        "weights_dir": str(weights_dir),
        "outputs": {
            "per_fold_predictions_csv": str(out_dir / f"per_fold_val_predictions_{version or 'default'}.csv"),
            "ensemble_predictions_csv": str(out_dir / f"ensemble_full_predictions_{version or 'default'}.csv"),
            "comparison_csv": str(out_dir / f"comparison_per_fold_vs_ensemble_{version or 'default'}.csv"),
        },
    }

    summary["outputs"] = {
        "per_fold_predictions_csv": str(per_fold_pred_path),
        "ensemble_predictions_csv": str(ens_pred_path),
        "comparison_csv": str(comp_path),
    }
    summary_path = out_ver_dir / "sanity_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    logger.success("Summary JSON -> {}", summary_path)
    logger.info("Sanity check completed.")


if __name__ == "__main__":
    main()


