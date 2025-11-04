import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import yaml
from loguru import logger
import time


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


def build_feature_extractor_offline(dinov3_source_dir: str, dino_weights_path: str) -> nn.Module:
    # Use the same offline approach as infer_and_submit_pt.py to avoid torch.hub
    import sys
    if dinov3_source_dir and os.path.isdir(dinov3_source_dir) and dinov3_source_dir not in sys.path:
        sys.path.insert(0, dinov3_source_dir)
    try:
        from dinov3.hub.backbones import dinov3_vitl16 as _dinov3_vitl16  # type: ignore
    except Exception as e:
        raise ImportError(
            f"Failed to import dinov3 backbones from '{dinov3_source_dir}'. Please ensure the path is correct. Error: {e}"
        )

    backbone = _dinov3_vitl16(pretrained=False)
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


def compute_metrics(preds: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
    preds = preds.float()
    targets = targets.float()
    diff = preds - targets
    mse = torch.mean(diff ** 2).item()
    mae = torch.mean(torch.abs(diff)).item()
    rmse = float(np.sqrt(mse))
    # Global R^2 across all outputs (same as training computation)
    ss_res = torch.sum((targets - preds) ** 2)
    mean_t = torch.mean(targets)
    ss_tot = torch.sum((targets - mean_t) ** 2)
    r2 = (1.0 - (ss_res / (ss_tot + 1e-8))).item()
    # Per-target metrics
    per_target_mse = torch.mean(diff ** 2, dim=0).tolist()
    per_target_mae = torch.mean(torch.abs(diff), dim=0).tolist()
    return {
        "mse": mse,
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "mse_dc": float(per_target_mse[0]),
        "mse_dd": float(per_target_mse[1]),
        "mse_dg": float(per_target_mse[2]),
        "mae_dc": float(per_target_mae[0]),
        "mae_dd": float(per_target_mae[1]),
        "mae_dg": float(per_target_mae[2]),
    }


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
    target_order = list(train_cfg["data"]["target_order"])  # [Dry_Clover_g, Dry_Dead_g, Dry_Green_g]
    val_batch_size = int(train_cfg["data"].get("val_batch_size", train_cfg["data"]["batch_size"]))
    num_workers = int(train_cfg["data"].get("num_workers", 4))
    logger.info(
        "Data settings: root={} image_size={} targets={} val_bs={} workers={}",
        data_root,
        image_size,
        target_order,
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
        target_order=target_order,
        mean=mean,
        std=std,
        train_scale=tuple(train_cfg["data"]["augment"]["random_resized_crop_scale"]),
        hflip_prob=float(train_cfg["data"]["augment"]["horizontal_flip_prob"]),
        shuffle=bool(train_cfg["data"].get("shuffle", True)),
    )
    full_df = dm.build_full_dataframe()
    logger.info("Full dataset rows: {} (unique images)", len(full_df))

    # 3) Build offline DINO feature extractor
    logger.info("[3/4] Loading DINOv3 backbone and preparing transforms ...")
    cfg_dino_weights = sanity.get("dino_weights_path", "")
    dino_weights_path = _to_abs(project_dir, cfg_dino_weights) if cfg_dino_weights else ""
    if not dino_weights_path or not os.path.isfile(dino_weights_path):
        # Fallback to training config
        cfg_weights_path = train_cfg["model"].get("weights_path", None)
        if not (cfg_weights_path and os.path.isfile(_to_abs(project_dir, str(cfg_weights_path)))):
            raise FileNotFoundError(
                "DINOv3 weights not found. Provide sanity.dino_weights_path or model.weights_path in training config."
            )
        dino_weights_path = _to_abs(project_dir, str(cfg_weights_path))
    logger.info("DINO weights: {}", dino_weights_path)
    dinov3_source = _to_abs(project_dir, sanity.get("dinov3_source", "third_party/dinov3/dinov3"))
    logger.info("dinov3 source: {}", dinov3_source)
    t0 = time.time()
    feature_extractor = build_feature_extractor_offline(dinov3_source, dino_weights_path)
    logger.success("Backbone ready in {:.2f}s", time.time() - t0)
    val_tf = build_val_transforms(image_size=image_size, mean=mean, std=std)

    # 4) Per-fold validation inference using each fold's head (if kfold is enabled)
    k_cfg = int(train_cfg.get("kfold", {}).get("k", 5))
    fold_splits_root = Path(log_dir) / "folds"
    if not fold_splits_root.is_dir():
        raise FileNotFoundError(f"Fold splits directory not found: {fold_splits_root}. Ensure k-fold training ran.")

    fold_head_paths = discover_fold_head_paths(weights_dir, k=k_cfg)
    logger.info("Found {} fold heads", len(fold_head_paths))

    per_fold_metrics: List[Dict[str, float]] = []
    per_fold_rows: List[Dict[str, object]] = []

    logger.info("[4/4] Running per-fold validation inference ...")
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
        targets_np = merged[target_order].to_numpy(dtype=np.float32)
        t0 = time.time()
        _, feats = extract_features(
            feature_extractor=feature_extractor,
            root_dir=data_root,
            image_paths=image_paths,
            transform=val_tf,
            batch_size=val_batch_size,
            num_workers=num_workers,
        )
        logger.debug("Fold {}: features shape={}  extracted in {:.2f}s", fold_idx, tuple(feats.shape), time.time() - t0)

        head = load_head_module(train_cfg)
        state = load_head_state_dict(fold_head_paths[fold_idx])
        head.load_state_dict(state, strict=True)
        t0 = time.time()
        preds = predict_from_features(features_cpu=feats, head=head, batch_size=val_batch_size)
        logger.debug("Fold {}: head preds shape={}  in {:.2f}s", fold_idx, tuple(preds.shape), time.time() - t0)

        targets = torch.from_numpy(targets_np)
        m = compute_metrics(preds=preds, targets=targets)
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
                    "image_id": merged.iloc[i]["image_id"],
                    "image_path": merged.iloc[i]["image_path"],
                    "target_dc": float(merged.iloc[i][target_order[0]]),
                    "target_dd": float(merged.iloc[i][target_order[1]]),
                    "target_dg": float(merged.iloc[i][target_order[2]]),
                    "pred_dc": float(preds[i, 0].item()),
                    "pred_dd": float(preds[i, 1].item()),
                    "pred_dg": float(preds[i, 2].item()),
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

    all_preds_tensor = torch.from_numpy(per_fold_pred_df[["pred_dc", "pred_dd", "pred_dg"]].to_numpy(dtype=np.float32))
    all_targets_tensor = torch.from_numpy(per_fold_pred_df[["target_dc", "target_dd", "target_dg"]].to_numpy(dtype=np.float32))
    agg_per_fold_metrics = compute_metrics(all_preds_tensor, all_targets_tensor)
    logger.info(
        "Aggregated per-fold metrics: MSE={:.6f} MAE={:.6f} RMSE={:.6f} R2={:.4f}",
        agg_per_fold_metrics["mse"],
        agg_per_fold_metrics["mae"],
        agg_per_fold_metrics["rmse"],
        agg_per_fold_metrics["r2"],
    )

    all_head_paths = discover_all_head_paths(weights_dir)
    if not all_head_paths:
        raise FileNotFoundError(f"No head weights found under: {weights_dir / 'head'}")
    logger.info("Ensemble heads discovered: {}", len(all_head_paths))

    unique_paths = full_df["image_path"].astype(str).tolist()
    t0 = time.time()
    rels_in_order, full_feats = extract_features(
        feature_extractor=feature_extractor,
        root_dir=data_root,
        image_paths=unique_paths,
        transform=val_tf,
        batch_size=val_batch_size,
        num_workers=num_workers,
    )
    logger.info("Full features extracted: shape={} in {:.2f}s", tuple(full_feats.shape), time.time() - t0)

    ensemble_sum = torch.zeros((full_feats.shape[0], 3), dtype=torch.float32)
    used_heads = 0
    for head_pt in all_head_paths:
        head = load_head_module(train_cfg)
        state = load_head_state_dict(head_pt)
        head.load_state_dict(state, strict=True)
        preds = predict_from_features(features_cpu=full_feats, head=head, batch_size=val_batch_size)
        ensemble_sum += preds
        used_heads += 1
    if used_heads == 0:
        raise RuntimeError("No valid head weights for ensemble inference.")
    ensemble_preds = ensemble_sum / float(used_heads)
    logger.info("Ensemble predictions computed using {} heads", used_heads)

    df_full_preds = pd.DataFrame({"image_path": rels_in_order})
    df_full_preds["pred_dc"] = ensemble_preds[:, 0].numpy()
    df_full_preds["pred_dd"] = ensemble_preds[:, 1].numpy()
    df_full_preds["pred_dg"] = ensemble_preds[:, 2].numpy()

    merged_full = full_df.merge(df_full_preds, on="image_path", how="inner")
    ens_targets = torch.from_numpy(merged_full[target_order].to_numpy(dtype=np.float32))
    ens_preds = torch.from_numpy(merged_full[["pred_dc", "pred_dd", "pred_dg"]].to_numpy(dtype=np.float32))
    ensemble_metrics = compute_metrics(ens_preds, ens_targets)
    logger.info(
        "Ensemble full-dataset metrics: MSE={:.6f} MAE={:.6f} RMSE={:.6f} R2={:.4f}",
        ensemble_metrics["mse"],
        ensemble_metrics["mae"],
        ensemble_metrics["rmse"],
        ensemble_metrics["r2"],
    )

    merged_full_out = merged_full[["image_id", "image_path", *target_order, "pred_dc", "pred_dd", "pred_dg"]].copy()
    ens_pred_path = out_ver_dir / "ensemble_full_predictions.csv"
    merged_full_out.to_csv(ens_pred_path, index=False)
    logger.success("Saved ensemble predictions -> {}", ens_pred_path)

    comp_left = per_fold_pred_df[["image_id", "image_path", "target_dc", "target_dd", "target_dg", "pred_dc", "pred_dd", "pred_dg"]]
    comp_right = merged_full_out[["image_id", "image_path", "pred_dc", "pred_dd", "pred_dg"]]
    comp_right = comp_right.rename(columns={
        "pred_dc": "ens_pred_dc",
        "pred_dd": "ens_pred_dd",
        "pred_dg": "ens_pred_dg",
    })
    comp = comp_left.merge(comp_right, on=["image_id", "image_path"], how="inner")
    comp["abs_diff_dc"] = (comp["pred_dc"] - comp["ens_pred_dc"]).abs()
    comp["abs_diff_dd"] = (comp["pred_dd"] - comp["ens_pred_dd"]).abs()
    comp["abs_diff_dg"] = (comp["pred_dg"] - comp["ens_pred_dg"]).abs()
    comp_path = out_ver_dir / "comparison_per_fold_vs_ensemble.csv"
    comp.to_csv(comp_path, index=False)
    logger.success("Saved comparison CSV ({} rows) -> {}", len(comp), comp_path)

    diff_stats = {
        "mean_abs_diff_dc": float(comp["abs_diff_dc"].mean()),
        "mean_abs_diff_dd": float(comp["abs_diff_dd"].mean()),
        "mean_abs_diff_dg": float(comp["abs_diff_dg"].mean()),
        "max_abs_diff_dc": float(comp["abs_diff_dc"].max()),
        "max_abs_diff_dd": float(comp["abs_diff_dd"].max()),
        "max_abs_diff_dg": float(comp["abs_diff_dg"].max()),
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


