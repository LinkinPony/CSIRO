import argparse
import csv
import json
import os
import re
import shutil
from pathlib import Path
import math

import yaml


def resolve_repo_root() -> Path:
    """
    Resolve the project root directory that contains both `configs/` and `src/`.

    This script may be executed either from the repository root (during training/dev)
    or from a packaged `weights/scripts/` directory. In the packaged case, the repo
    root is the parent of the scripts directory.
    """
    here = Path(__file__).resolve().parent
    if (here / "configs").is_dir() and (here / "src").is_dir():
        return here
    if (here.parent / "configs").is_dir() and (here.parent / "src").is_dir():
        return here.parent
    return here


def parse_args():
    p = argparse.ArgumentParser(description="Package head weights and project sources into weights/ folder")
    p.add_argument(
        "--config",
        type=str,
        default=str(resolve_repo_root() / "configs" / "train.yaml"),
        help="Path to YAML config file (to resolve version and output dirs)",
    )
    p.add_argument(
        "--weights-dir",
        type=str,
        default=str(resolve_repo_root() / "weights"),
        help="Destination weights directory",
    )
    p.add_argument(
        "--best",
        action="store_true",
        help="Select best checkpoint by val_loss (default: use latest-epoch checkpoint)",
    )
    p.add_argument(
        "--no-swa",
        action="store_true",
        help="Disable SWA-averaged weights and use raw head-epoch checkpoints (legacy behaviour).",
    )
    return p.parse_args()


def load_cfg(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _find_config_snapshot_for_version(
    base_cfg: dict,
    repo_root: Path,
    version: str,
) -> Path | None:
    """
    Locate a snapshot of the training config for a given version.
    Priority:
      1) outputs/<version>/train.yaml
      2) outputs/<version>/fold_0/train.yaml   (train_all or special cases)
    Falls back to None if not found.
    """
    base_log_dir = Path(base_cfg["logging"]["log_dir"]).expanduser()
    # Root version log dir (without train_all folding)
    v_root = base_log_dir / version if version else base_log_dir
    cand1 = v_root / "train.yaml"
    if cand1.is_file():
        return cand1
    cand2 = v_root / "fold_0" / "train.yaml"
    if cand2.is_file():
        return cand2
    return None


def _copy_snapshot_into_weights(
    snapshot_path: Path | None,
    repo_root: Path,
    weights_dir: Path,
) -> None:
    """
    Copy config snapshot into weights/configs/train.yaml, overriding the repo copy.
    If snapshot_path is None or missing, leave the repo copy as-is.
    """
    try:
        dst_cfg_dir = weights_dir / "configs"
        dst_cfg_dir.mkdir(parents=True, exist_ok=True)
        if snapshot_path is not None and snapshot_path.is_file():
            import shutil as _shutil

            dst_cfg_path = dst_cfg_dir / "train.yaml"
            _shutil.copyfile(str(snapshot_path), str(dst_cfg_path))
            print(f"[CFG] Using snapshot config for inference: {snapshot_path} -> {dst_cfg_path}")
    except Exception as e:
        print(f"[CFG] Failed to copy config snapshot into weights/: {e}")


def _copy_version_snapshot_into_weights(
    snapshot_path: Path | None,
    *,
    weights_dir: Path,
    version: str,
) -> None:
    """
    Copy a per-version training config snapshot into:
      weights/configs/versions/<version>/train.yaml

    This is used for multi-model / multi-backbone ensembles so that inference can
    load the correct backbone + data settings per model, instead of relying on a
    single global weights/configs/train.yaml.
    """
    try:
        ver = str(version or "").strip()
        if not ver:
            return
        dst_cfg_dir = weights_dir / "configs" / "versions" / ver
        dst_cfg_dir.mkdir(parents=True, exist_ok=True)
        if snapshot_path is not None and snapshot_path.is_file():
            import shutil as _shutil

            dst_cfg_path = dst_cfg_dir / "train.yaml"
            _shutil.copyfile(str(snapshot_path), str(dst_cfg_path))
            print(f"[CFG] Copied per-version snapshot config: {snapshot_path} -> {dst_cfg_path}")
    except Exception as e:
        print(f"[CFG] Failed to copy per-version snapshot config into weights/: {e}")


def load_ensemble_cfg(repo_root: Path) -> dict:
    """
    Read configs/ensemble.json if present. Returns dict with at least:
      - enabled: bool
      - versions: list[str]   (legacy)
      - models: list[dict]    (new; each item may include 'version', 'weight', 'backbone', etc.)
    """
    try:
        path = repo_root / "configs" / "ensemble.json"
        if not path.is_file():
            return {"enabled": False, "versions": [], "models": []}
        import json as _json
        with open(path, "r", encoding="utf-8") as f:
            obj = _json.load(f)
        if not isinstance(obj, dict):
            return {"enabled": False, "versions": [], "models": []}
        enabled = bool(obj.get("enabled", False))
        # New schema: explicit model objects
        models = obj.get("models", [])
        if not isinstance(models, list):
            models = []
        models = [m for m in models if isinstance(m, dict)]

        # Backward-compat: accept single 'version' or list 'versions'
        versions = obj.get("versions", None)
        if versions is None:
            v = obj.get("version", "")
            versions = [str(v)] if isinstance(v, str) and v else []
        else:
            if isinstance(versions, list):
                versions = [str(v) for v in versions if isinstance(v, (str, int, float)) and str(v)]
            else:
                versions = []
        return {"enabled": enabled, "versions": versions, "models": models}
    except Exception:
        return {"enabled": False, "versions": [], "models": []}


def resolve_dirs(cfg: dict) -> tuple[Path, Path]:
    base_log_dir = Path(cfg["logging"]["log_dir"]).expanduser()
    base_ckpt_dir = Path(cfg["logging"]["ckpt_dir"]).expanduser()
    version = cfg.get("version", None)
    version = None if version in (None, "", "null") else str(version)
    log_dir = base_log_dir / version if version else base_log_dir
    ckpt_dir = base_ckpt_dir / version if version else base_ckpt_dir
    return log_dir, ckpt_dir


def find_latest_metrics_csv(log_dir: Path) -> Path | None:
    candidate_files: list[Path] = []
    if not log_dir.exists():
        return None
    for root, _, files in os.walk(log_dir):
        for name in files:
            if name == "metrics.csv":
                candidate_files.append(Path(root) / name)
    if not candidate_files:
        return None
    candidate_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidate_files[0]


def pick_best_epoch_from_metrics(metrics_csv: Path) -> int | None:
    try:
        with open(metrics_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            last_val_loss_by_epoch: dict[int, float] = {}
            for row in reader:
                if "epoch" not in row:
                    continue
                try:
                    epoch = int(float(row["epoch"]))
                except Exception:
                    continue
                if "val_loss" in row and row["val_loss"] not in ("", None):
                    try:
                        val_loss = float(row["val_loss"])
                        last_val_loss_by_epoch[epoch] = val_loss
                    except Exception:
                        pass
        if not last_val_loss_by_epoch:
            return None
        # select epoch with minimal val_loss
        best_epoch = min(last_val_loss_by_epoch.items(), key=lambda kv: kv[1])[0]
        return best_epoch
    except Exception:
        return None


def list_head_checkpoints(head_dir: Path) -> list[Path]:
    if not head_dir.exists():
        return []
    files = [p for p in head_dir.iterdir() if p.is_file() and p.name.startswith("head-epoch") and p.suffix == ".pt"]
    return files


def find_head_by_epoch(head_dir: Path, epoch: int) -> Path | None:
    # Support metric-suffixed filenames like head-epoch003-val_loss0.123456-...pt
    candidates = []
    prefix = f"head-epoch{epoch:03d}"
    for p in list_head_checkpoints(head_dir):
        name = p.name
        if name.startswith(prefix) and name.endswith(".pt"):
            candidates.append(p)
    if not candidates:
        return None
    # If multiple, prefer the most recent
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def pick_latest_head(head_files: list[Path]) -> Path | None:
    if not head_files:
        return None
    def parse_epoch(p: Path) -> int:
        m = re.search(r"head-epoch(\d+)(?:[^/]*)\.pt$", p.name)
        return int(m.group(1)) if m else -1
    head_files.sort(key=lambda p: (parse_epoch(p), p.stat().st_mtime), reverse=True)
    return head_files[0]


def copy_head_to_weights(head_src: Path, weights_dir: Path) -> Path:
    dst_dir = weights_dir / "head"
    # Clean destination directory before copying
    if dst_dir.exists() and dst_dir.is_dir():
        shutil.rmtree(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst_path = dst_dir / "infer_head.pt"
    shutil.copyfile(str(head_src), str(dst_path))
    return dst_path


def copy_tree(src_dir: Path, dst_dir: Path) -> None:
    # Clean destination directory before copying
    if dst_dir.exists() and dst_dir.is_dir():
        shutil.rmtree(dst_dir)
    shutil.copytree(
        src_dir,
        dst_dir,
        dirs_exist_ok=True,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo", "*.pyd", ".DS_Store"),
    )


def copy_optional_third_party(repo_root: Path, weights_dir: Path) -> None:
    # Copy PEFT sources for offline PEFT inference (optional)
    peft_src = repo_root / "third_party" / "peft"
    if peft_src.exists() and peft_src.is_dir():
        dst = weights_dir / "third_party" / "peft"
        if dst.exists() and dst.is_dir():
            shutil.rmtree(dst)
        shutil.copytree(
            peft_src,
            dst,
            dirs_exist_ok=True,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo", "*.pyd", ".git", ".DS_Store"),
        )

    # Copy TabPFN sources for offline TabPFN inference (optional)
    #
    # This supports `infer_and_submit_pt.py` using:
    #   TABPFN_PATH = "third_party/TabPFN/src"
    tabpfn_src = repo_root / "third_party" / "TabPFN"
    if tabpfn_src.exists() and tabpfn_src.is_dir():
        dst = weights_dir / "third_party" / "TabPFN"
        if dst.exists() and dst.is_dir():
            shutil.rmtree(dst)
        shutil.copytree(
            tabpfn_src,
            dst,
            dirs_exist_ok=True,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo", "*.pyd", ".git", ".DS_Store"),
        )


def _package_tabpfn_inference_artifacts(*, repo_root: Path, weights_dir: Path) -> None:
    """
    Build + package TabPFN artifacts for inference (no online training):
      - Copy TabPFN foundation weights into the package (optional; controlled by configs/train_tabpfn.yaml).
      - Extract train image features and fit TabPFN once (single or ensemble mode).
      - Save fitted-state bundles under: <weights_dir>/<artifacts.fit_state_dir>/{single|ensemble}/
    """
    cfg_path = (weights_dir / "configs" / "train_tabpfn.yaml").resolve()
    if not cfg_path.is_file():
        cfg_path = (repo_root / "configs" / "train_tabpfn.yaml").resolve()
    if not cfg_path.is_file():
        print("[TABPFN][PKG] configs/train_tabpfn.yaml not found; skipping TabPFN packaging.")
        return

    cfg = load_cfg(str(cfg_path))
    if not isinstance(cfg, dict):
        print("[TABPFN][PKG] Invalid train_tabpfn.yaml (not a dict); skipping TabPFN packaging.")
        return

    tab_cfg = cfg.get("tabpfn", {}) if isinstance(cfg.get("tabpfn", {}), dict) else {}
    imgf_cfg = cfg.get("image_features", {}) if isinstance(cfg.get("image_features", {}), dict) else {}
    data_cfg = cfg.get("data", {}) if isinstance(cfg.get("data", {}), dict) else {}
    art_cfg = cfg.get("artifacts", {}) if isinstance(cfg.get("artifacts", {}), dict) else {}

    fit_state_dir_name = str(art_cfg.get("fit_state_dir", "tabpfn_fit") or "tabpfn_fit").strip() or "tabpfn_fit"
    train_feat_cache_dir_name = (
        str(art_cfg.get("train_feature_cache_dir", "tabpfn_train_features") or "tabpfn_train_features").strip()
        or "tabpfn_train_features"
    )
    copy_tabpfn_weights = bool(art_cfg.get("copy_tabpfn_weights", False))

    model_path_raw = str(tab_cfg.get("model_path", tab_cfg.get("weights_ckpt_path", "")) or "").strip()
    if not model_path_raw:
        raise RuntimeError("[TABPFN][PKG] Missing required tabpfn.model_path in configs/train_tabpfn.yaml")

    # Copy TabPFN ckpt into weights/ so fit-states can load offline (relative model_path).
    if copy_tabpfn_weights:
        src_ckpt = Path(model_path_raw).expanduser()
        if not src_ckpt.is_absolute():
            src_ckpt = (repo_root / src_ckpt).resolve()
        if not src_ckpt.exists():
            raise FileNotFoundError(f"[TABPFN][PKG] TabPFN model_path not found: {src_ckpt}")

        rel = Path(model_path_raw)
        if rel.is_absolute():
            rel = Path("tabpfn_weights") / src_ckpt.name
        if not str(rel).startswith("tabpfn_weights"):
            rel = Path("tabpfn_weights") / src_ckpt.name
        dst_ckpt = (weights_dir / rel).resolve()

        if src_ckpt.is_file():
            dst_ckpt.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(str(src_ckpt), str(dst_ckpt))
            print(f"[TABPFN][PKG] Copied TabPFN ckpt: {src_ckpt} -> {dst_ckpt}")
        elif src_ckpt.is_dir():
            if dst_ckpt.exists() and dst_ckpt.is_dir():
                shutil.rmtree(dst_ckpt)
            shutil.copytree(str(src_ckpt), str(dst_ckpt), dirs_exist_ok=True)
            print(f"[TABPFN][PKG] Copied TabPFN ckpt dir: {src_ckpt} -> {dst_ckpt}")

    # Resolve training data (not copied into weights/)
    data_root = str(data_cfg.get("root", "data") or "data").strip() or "data"
    train_csv_name = str(data_cfg.get("train_csv", "train.csv") or "train.csv").strip() or "train.csv"
    dataset_root = Path(data_root).expanduser()
    if not dataset_root.is_absolute():
        dataset_root = (repo_root / dataset_root).resolve()
    train_csv_path = (dataset_root / train_csv_name).resolve()
    if not train_csv_path.is_file():
        raise FileNotFoundError(f"[TABPFN][PKG] train.csv not found: {train_csv_path}")

    target_order = data_cfg.get("target_order", ["Dry_Total_g"])
    if not isinstance(target_order, list) or not target_order:
        target_order = ["Dry_Total_g"]

    from src.data.csiro_pivot import read_and_pivot_csiro_train_csv
    from src.metrics import TARGETS_5D_ORDER
    from src.tabular.features import build_y_5d

    df_train = read_and_pivot_csiro_train_csv(
        data_root=str(dataset_root),
        train_csv=str(train_csv_path),
        target_order=list(target_order),
    )
    if len(df_train) < 2:
        raise RuntimeError(f"[TABPFN][PKG] Not enough training samples after pivoting: {len(df_train)}")
    y_train = build_y_5d(df_train, targets_5d_order=list(TARGETS_5D_ORDER), fillna=0.0)
    train_image_paths = df_train["image_path"].astype(str).tolist()

    # Feature extraction config (matches inference semantics)
    feature_mode = str(imgf_cfg.get("mode", "head_penultimate") or "head_penultimate").strip().lower()
    feature_fusion = str(imgf_cfg.get("fusion", "mean") or "mean").strip().lower()
    feature_batch_size = int(imgf_cfg.get("batch_size", 8))
    feature_num_workers = int(imgf_cfg.get("num_workers", 8))
    if feature_mode not in ("head_penultimate", "dinov3_only"):
        raise ValueError(f"[TABPFN][PKG] Unsupported image_features.mode={feature_mode!r}")

    fit_mode = str(tab_cfg.get("fit_mode", "fit_preprocessors") or "fit_preprocessors").strip()
    if fit_mode.lower() == "fit_with_cache":
        raise RuntimeError(
            "[TABPFN][PKG] tabpfn.fit_mode='fit_with_cache' cannot be packaged via save_fit_state. "
            "Use fit_preprocessors or low_memory."
        )

    # Configure TabPFN env before importing
    enable_telemetry = bool(tab_cfg.get("enable_telemetry", False))
    if not enable_telemetry:
        os.environ["TABPFN_DISABLE_TELEMETRY"] = "1"
    else:
        os.environ.pop("TABPFN_DISABLE_TELEMETRY", None)
    model_cache_dir = tab_cfg.get("model_cache_dir", None)
    if isinstance(model_cache_dir, str) and model_cache_dir.strip():
        os.environ["TABPFN_MODEL_CACHE_DIR"] = str((weights_dir / model_cache_dir.strip()).resolve())

    # Import TabPFN + feature extractors
    from src.inference.tabpfn import _import_tabpfn_regressor, extract_dinov3_cls_features, extract_head_penultimate_features

    TabPFNRegressor = _import_tabpfn_regressor(
        tabpfn_python_path=str(tab_cfg.get("python_path", "") or "").strip(),
        project_dir=str(weights_dir),
    )

    import numpy as np
    from sklearn.multioutput import MultiOutputRegressor

    # Determine single vs ensemble mode based on packaged configs/ensemble.json
    from src.inference.ensemble import normalize_ensemble_models
    from src.inference.pipeline import load_config, load_config_file, parse_image_size, resolve_dino_weights_path_for_model
    from src.inference.paths import resolve_path_best_effort, resolve_version_head_base, resolve_version_train_yaml, safe_slug

    ensemble_models = normalize_ensemble_models(str(weights_dir))
    use_ensemble = len(ensemble_models) > 0
    subdir = "ensemble" if use_ensemble else "single"

    out_fit_dir = (weights_dir / fit_state_dir_name / subdir).resolve()
    if out_fit_dir.exists() and out_fit_dir.is_dir():
        shutil.rmtree(out_fit_dir)
    out_fit_dir.mkdir(parents=True, exist_ok=True)

    train_cache_dir = (weights_dir / train_feat_cache_dir_name).resolve()
    train_cache_dir.mkdir(parents=True, exist_ok=True)

    # DINO weights dir (not packaged by default; used only for feature extraction during packaging)
    dino_weights_root = (repo_root / "dinov3_weights").resolve()
    if not dino_weights_root.is_dir():
        raise FileNotFoundError(f"[TABPFN][PKG] dinov3_weights directory not found: {dino_weights_root}")

    def _fit_and_save(*, X: np.ndarray, y: np.ndarray, meta_extra: dict) -> None:
        # Fit inside weights_dir so model_path (relative) resolves against packaged tabpfn_weights/.
        cwd = os.getcwd()
        try:
            os.chdir(str(weights_dir))
            base = TabPFNRegressor(
                n_estimators=int(tab_cfg.get("n_estimators", 8)),
                device=str(tab_cfg.get("device", "auto")),
                fit_mode=str(fit_mode),
                inference_precision=str(tab_cfg.get("inference_precision", "auto")),
                random_state=42,
                ignore_pretraining_limits=bool(tab_cfg.get("ignore_pretraining_limits", True)),
                model_path=str(model_path_raw),
            )
            model = MultiOutputRegressor(base, n_jobs=int(tab_cfg.get("n_jobs", 1)))
            model.fit(X, y)
        finally:
            try:
                os.chdir(cwd)
            except Exception:
                pass

        files: dict[str, str] = {}
        for i, tname in enumerate(list(TARGETS_5D_ORDER)):
            out_path = out_fit_dir / f"{tname}.tabpfn_fit"
            model.estimators_[i].save_fit_state(str(out_path))  # type: ignore[attr-defined]
            files[str(tname)] = out_path.name

        meta = {
            "mode": str(subdir),
            "targets": list(TARGETS_5D_ORDER),
            "files": files,
            "feature_mode": str(feature_mode),
            "feature_fusion": str(feature_fusion),
            "feature_dim": int(X.shape[1]) if isinstance(X, np.ndarray) and X.ndim == 2 else None,
            "tabpfn": {
                "model_path": str(model_path_raw),
                "n_estimators": int(tab_cfg.get("n_estimators", 8)),
                "device": str(tab_cfg.get("device", "auto")),
                "fit_mode": str(fit_mode),
                "inference_precision": str(tab_cfg.get("inference_precision", "auto")),
                "ignore_pretraining_limits": bool(tab_cfg.get("ignore_pretraining_limits", True)),
            },
        }
        if meta_extra:
            meta.update(meta_extra)
        with open(out_fit_dir / "meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        print(f"[TABPFN][PKG] Saved fit-state -> {out_fit_dir}")

    if use_ensemble:
        print(f"[TABPFN][PKG] Ensemble enabled ({len(ensemble_models)} models): training ensemble fit-state.")
        models_eff: list[dict] = []
        for idx, m in enumerate(ensemble_models):
            if not isinstance(m, dict):
                continue
            model_id = str(m.get("id", f"model_{idx}") or f"model_{idx}")
            version = m.get("version", None)
            # Resolve config path for model
            cfg_path = None
            for k in ("config", "config_path", "train_yaml", "train_config"):
                v = m.get(k, None)
                if isinstance(v, str) and v.strip():
                    cfg_path = resolve_path_best_effort(str(weights_dir), v.strip())
                    break
            if cfg_path is None:
                if isinstance(version, (str, int, float)) and str(version).strip():
                    cfg_path = resolve_version_train_yaml(str(weights_dir), str(version).strip())
                else:
                    cfg_path = str((weights_dir / "configs" / "train.yaml").resolve())
            cfg_model = load_config_file(str(cfg_path))

            # Base image size (H,W)
            try:
                base_image_size_hw = parse_image_size(cfg_model.get("data", {}).get("image_size", 224))
            except Exception:
                base_image_size_hw = (224, 224)

            # Head base for this model (unused for dinov3_only)
            if isinstance(version, (str, int, float)) and str(version).strip():
                head_base = resolve_version_head_base(str(weights_dir), str(version).strip())
            else:
                head_base = str((weights_dir / "head").resolve())

            backbone_name = str(cfg_model.get("model", {}).get("backbone", "") or "").strip()
            dino_weights_file = resolve_dino_weights_path_for_model(
                str(weights_dir),
                backbone_name=backbone_name,
                cfg=cfg_model,
                model_cfg=m,
                global_dino_weights=str(dino_weights_root),
            )
            if not (dino_weights_file and os.path.isfile(dino_weights_file)):
                raise FileNotFoundError(f"[TABPFN][PKG] Cannot resolve DINO weights for ensemble model: {model_id!r}")

            base_tag = str(version).strip() if isinstance(version, (str, int, float)) and str(version).strip() else model_id
            model_tag = safe_slug("__".join([base_tag, feature_mode, feature_fusion if feature_mode != "dinov3_only" else ""]))
            cache_train = str((train_cache_dir / f"train__{model_tag}.pt").resolve())

            models_eff.append(
                {
                    "model_id": model_id,
                    "version": str(version).strip() if isinstance(version, (str, int, float)) else None,
                    "cfg": cfg_model,
                    "head_base": str(head_base),
                    "dino_weights_file": str(dino_weights_file),
                    "base_image_size_hw": tuple(base_image_size_hw),
                    "cache_train": cache_train,
                }
            )

        if not models_eff:
            raise RuntimeError("[TABPFN][PKG] Ensemble enabled but no valid models found for TabPFN packaging.")

        X_list: list[np.ndarray] = []
        D_ref: int | None = None
        for mi in models_eff:
            cfg_i = mi["cfg"]
            dino_w = mi["dino_weights_file"]
            head_base = mi["head_base"]
            cache_train = mi["cache_train"]
            base_hw = tuple(mi.get("base_image_size_hw", (224, 224)))
            if feature_mode == "dinov3_only":
                _rels, X_i = extract_dinov3_cls_features(
                    project_dir=str(weights_dir),
                    dataset_root=str(dataset_root),
                    cfg_train_yaml=cfg_i,
                    dino_weights_pt_file=str(dino_w),
                    image_paths=train_image_paths,
                    batch_size=int(feature_batch_size),
                    num_workers=int(feature_num_workers),
                    cache_path=str(cache_train),
                    image_size_hw=base_hw,
                    hflip=False,
                    vflip=False,
                )
            else:
                _rels, X_i = extract_head_penultimate_features(
                    project_dir=str(weights_dir),
                    dataset_root=str(dataset_root),
                    cfg_train_yaml=cfg_i,
                    dino_weights_pt_file=str(dino_w),
                    head_weights_pt_path=str(head_base),
                    image_paths=train_image_paths,
                    fusion=str(feature_fusion),
                    batch_size=int(feature_batch_size),
                    num_workers=int(feature_num_workers),
                    cache_path=str(cache_train),
                    image_size_hw=base_hw,
                    hflip=False,
                    vflip=False,
                )
            if not (isinstance(X_i, np.ndarray) and X_i.ndim == 2 and X_i.shape[0] == len(train_image_paths)):
                raise RuntimeError(f"[TABPFN][PKG] Bad train feature array: shape={getattr(X_i, 'shape', None)}")
            if D_ref is None:
                D_ref = int(X_i.shape[1])
            elif int(X_i.shape[1]) != int(D_ref):
                raise RuntimeError(f"[TABPFN][PKG] Feature dim mismatch across models (expected {D_ref}, got {int(X_i.shape[1])}).")
            X_list.append(X_i)

        X_train_all = np.concatenate(X_list, axis=0)
        y_train_all = np.concatenate([y_train for _ in range(len(X_list))], axis=0)
        image_versions = [
            str(m.get("version")).strip()
            for m in models_eff
            if isinstance(m.get("version", None), (str, int, float)) and str(m.get("version")).strip()
        ]
        # Deduplicate while preserving order
        _seen = set()
        image_versions = [v for v in image_versions if not (v in _seen or _seen.add(v))]
        _fit_and_save(
            X=X_train_all,
            y=y_train_all,
            meta_extra={
                "image_versions": list(image_versions),
                "ensemble_models": [{"version": m.get("version")} for m in models_eff],
            },
        )
    else:
        print("[TABPFN][PKG] Ensemble disabled: training single-model fit-state.")
        cfg_img = load_config(str(weights_dir))
        backbone_name = str(cfg_img.get("model", {}).get("backbone", "") or "").strip()
        dino_weights_file = resolve_dino_weights_path_for_model(
            str(weights_dir),
            backbone_name=backbone_name,
            cfg=cfg_img,
            model_cfg={},
            global_dino_weights=str(dino_weights_root),
        )
        if not (dino_weights_file and os.path.isfile(dino_weights_file)):
            raise FileNotFoundError(f"[TABPFN][PKG] Cannot resolve DINO weights file for backbone={backbone_name!r}")

        try:
            base_image_size_hw = parse_image_size(cfg_img.get("data", {}).get("image_size", 224))
        except Exception:
            base_image_size_hw = (224, 224)

        cache_train = str((train_cache_dir / f"train__single__{safe_slug(feature_mode)}.pt").resolve())
        if feature_mode == "dinov3_only":
            _rels, X_train = extract_dinov3_cls_features(
                project_dir=str(weights_dir),
                dataset_root=str(dataset_root),
                cfg_train_yaml=cfg_img,
                dino_weights_pt_file=str(dino_weights_file),
                image_paths=train_image_paths,
                batch_size=int(feature_batch_size),
                num_workers=int(feature_num_workers),
                cache_path=str(cache_train),
                image_size_hw=tuple(base_image_size_hw),
                hflip=False,
                vflip=False,
            )
        else:
            _rels, X_train = extract_head_penultimate_features(
                project_dir=str(weights_dir),
                dataset_root=str(dataset_root),
                cfg_train_yaml=cfg_img,
                dino_weights_pt_file=str(dino_weights_file),
                head_weights_pt_path=str((weights_dir / "head").resolve()),
                image_paths=train_image_paths,
                fusion=str(feature_fusion),
                batch_size=int(feature_batch_size),
                num_workers=int(feature_num_workers),
                cache_path=str(cache_train),
                image_size_hw=tuple(base_image_size_hw),
                hflip=False,
                vflip=False,
            )
        if not (isinstance(X_train, np.ndarray) and X_train.ndim == 2 and X_train.shape[0] == len(train_image_paths)):
            raise RuntimeError(f"[TABPFN][PKG] Bad train feature array: shape={getattr(X_train, 'shape', None)}")
        image_version_single = str(cfg_img.get("version", "") or "").strip()
        _fit_and_save(
            X=X_train,
            y=y_train,
            meta_extra={
                # New schema (preferred)
                "image_versions": ([image_version_single] if image_version_single else []),
                # Legacy single-version key (kept for readability/backward-compat)
                "image_version": (image_version_single if image_version_single else None),
            },
        )


def copy_top_level_scripts(repo_root: Path, weights_dir: Path) -> list[Path]:
    # Copy selected top-level scripts into weights/scripts/ for portability
    script_names = [
        "infer_and_submit_pt.py",
        "package_artifacts.py",
        "train.py",
        "sanity_check.py",
    ]
    copied: list[Path] = []
    scripts_dir = weights_dir / "scripts"
    scripts_dir.mkdir(parents=True, exist_ok=True)
    for name in script_names:
        src = repo_root / name
        if src.is_file():
            dst = scripts_dir / name
            shutil.copyfile(str(src), str(dst))
            copied.append(dst)
        # Clean legacy location (weights/<script>.py) if it exists from older packaging runs.
        legacy_dst = weights_dir / name
        if legacy_dst.is_file():
            try:
                legacy_dst.unlink()
            except Exception:
                pass
    return copied


def _load_checkpoint(path: Path):
    import torch

    try:
        # PyTorch 2.6+ defaults to weights_only=True internally, which can fail for
        # Lightning checkpoints that store callback state. If the safe loader fails,
        # retry with weights_only=False (trusted local checkpoint).
        return torch.load(str(path), map_location="cpu")
    except Exception:
        try:
            print(f"[SWA] Retrying torch.load with weights_only=False for checkpoint: {path}")
            return torch.load(str(path), map_location="cpu", weights_only=False)  # type: ignore[call-arg]
        except Exception as e:
            print(f"[SWA] torch.load failed even with weights_only=False: {e}")
            raise


def _find_swa_average_model_state(ckpt_obj: dict) -> dict | None:
    """
    Locate the StochasticWeightAveraging callback state in a Lightning checkpoint
    and return its `average_model_state` (if present).
    """
    callbacks_state = ckpt_obj.get("callbacks", None)
    if not isinstance(callbacks_state, dict):
        return None
    for cb_name, cb_state in callbacks_state.items():
        if not isinstance(cb_state, dict):
            continue
        avg_state = cb_state.get("average_model_state", None)
        if not (isinstance(avg_state, dict) and avg_state):
            continue

        # IMPORTANT:
        # Lightning checkpoints can contain a non-empty `average_model_state`
        # even before SWA has actually started averaging (e.g., when
        # swa_epoch_start is late in training). In that case the callback state
        # typically reports `n_averaged == 0` and `latest_update_epoch == -1`,
        # and `average_model_state` is effectively an initial copy (often close
        # to epoch-0 init). Exporting such a head would look like "SWA == epoch0".
        n_averaged = cb_state.get("n_averaged", cb_state.get("num_averaged", None))
        latest_update_epoch = cb_state.get("latest_update_epoch", None)
        try:
            n_avg_int = int(n_averaged) if n_averaged is not None else None
        except Exception:
            n_avg_int = None
        try:
            latest_update_int = int(latest_update_epoch) if latest_update_epoch is not None else None
        except Exception:
            latest_update_int = None

        if (n_avg_int is not None and n_avg_int <= 0) or (latest_update_int is not None and latest_update_int < 0):
            print(
                f"[SWA] Found average_model_state in callback '{cb_name}', but SWA has not averaged any weights yet "
                f"(n_averaged={n_averaged}, latest_update_epoch={latest_update_epoch}); skipping."
            )
            continue

        return avg_state
    return None


def _export_swa_head_from_checkpoint(ckpt_path: Path, dst_dir: Path) -> tuple[Path, Path] | None:
    """
    Build an inference head using SWA-averaged weights stored inside a Lightning
    checkpoint. Returns (ckpt_path, dst_head_path) on success, or None if SWA
    information is not available in the checkpoint.
    """
    from src.models.regressor import BiomassRegressor
    from src.callbacks.head_checkpoint import HeadCheckpoint
    import shutil as _shutil

    if not ckpt_path.is_file():
        print(f"[SWA] Checkpoint not found on disk, skipping SWA export: {ckpt_path}")
        return None

    print(f"[SWA] Attempting to export SWA head from checkpoint: {ckpt_path}")

    try:
        ckpt_obj = _load_checkpoint(ckpt_path)
    except Exception as e:
        print(f"[SWA] Failed to load checkpoint ({ckpt_path}): {e}")
        return None

    avg_state = _find_swa_average_model_state(ckpt_obj)
    if not isinstance(avg_state, dict) or not avg_state:
        # No SWA information in this checkpoint
        print(
            "[SWA] No usable average_model_state found in checkpoint callbacks "
            "(SWA may be disabled, not started yet, or not checkpointed)."
        )
        return None

    # Instantiate model from checkpoint hyperparameters and then load SWA-averaged weights
    try:
        model = BiomassRegressor.load_from_checkpoint(str(ckpt_path), map_location="cpu")
    except Exception as e:
        print(f"[SWA] Failed to construct BiomassRegressor from checkpoint ({ckpt_path}): {e}")
        return None

    try:
        missing, unexpected = model.load_state_dict(avg_state, strict=False)
        if missing or unexpected:
            print(f"[SWA] Warning: loading SWA state into model had missing={len(missing)}, unexpected={len(unexpected)} keys.")
    except Exception as e:
        print(f"[SWA] Failed to load SWA average_model_state into model: {e}")
        return None

    # Use the existing HeadCheckpoint logic to compose a packed inference head
    # from the SWA-averaged model.
    # IMPORTANT: place the temporary directory OUTSIDE dst_dir, because we may
    # clear dst_dir before copying the final head file.
    tmp_dir = dst_dir.parent / ".tmp_swa_head"
    if tmp_dir.exists() and tmp_dir.is_dir():
        _shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    # IMPORTANT:
    # We invoke the callback manually with a dummy trainer, so it must save unconditionally.
    # The default HeadCheckpoint behavior (save_only_last_epoch=True) relies on Trainer metadata
    # like max_epochs/should_stop to decide "last epoch", which our dummy trainer does not provide.
    head_cb = HeadCheckpoint(output_dir=str(tmp_dir), save_only_last_epoch=False)

    class _DummyTrainer:
        def __init__(self, epoch: int):
            self.current_epoch = epoch
            # No metrics suffix needed for packaging; filenames will be simple.
            self.callback_metrics = {}

    epoch_int = 0
    try:
        epoch_int = int(ckpt_obj.get("epoch", 0))
    except Exception:
        epoch_int = 0

    dummy_trainer = _DummyTrainer(epoch=epoch_int)
    try:
        head_cb.on_validation_epoch_end(dummy_trainer, model)
    except Exception as e:
        print(f"[SWA] HeadCheckpoint export failed when using SWA-averaged model: {e}")
        _shutil.rmtree(tmp_dir, ignore_errors=True)
        return None

    # Locate the produced head-epoch*.pt under the temporary directory.
    head_files = list_head_checkpoints(tmp_dir)
    if not head_files:
        print("[SWA] No head-epoch*.pt files were produced by HeadCheckpoint in temporary SWA export dir.")
        _shutil.rmtree(tmp_dir, ignore_errors=True)
        return None

    chosen = pick_latest_head(head_files)
    if chosen is None:
        print("[SWA] Failed to select a head checkpoint from temporary SWA export dir.")
        _shutil.rmtree(tmp_dir, ignore_errors=True)
        return None

    # Copy into destination as infer_head.pt
    if dst_dir.exists() and dst_dir.is_dir():
        _shutil.rmtree(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst_path = dst_dir / "infer_head.pt"
    _shutil.copyfile(str(chosen), str(dst_path))

    # Clean temporary directory
    _shutil.rmtree(tmp_dir, ignore_errors=True)
    print(f"[SWA] Successfully exported SWA head: {ckpt_path} -> {dst_path}")
    return ckpt_path, dst_path


def main():
    args = parse_args()
    cfg = load_cfg(args.config)
    repo_root = resolve_repo_root()
    ensemble_cfg = load_ensemble_cfg(repo_root)
    log_dir, ckpt_dir = resolve_dirs(cfg)

    weights_dir = Path(args.weights_dir).expanduser()
    weights_dir.mkdir(parents=True, exist_ok=True)

    # Multi-version ensemble packaging path
    enabled = bool(ensemble_cfg.get("enabled", False))
    models_cfg = ensemble_cfg.get("models", [])
    if not isinstance(models_cfg, list):
        models_cfg = []
    # Prefer new schema: explicit models list. Fallback to legacy versions list.
    versions: list[str] = []
    if enabled and len(models_cfg) > 0:
        for m in models_cfg:
            if not isinstance(m, dict):
                continue
            v = m.get("version", None)
            if isinstance(v, (str, int, float)) and str(v).strip():
                versions.append(str(v).strip())
        # Deduplicate while preserving order
        seen = set()
        versions = [v for v in versions if not (v in seen or seen.add(v))]
        print(f"[ENSEMBLE] Multi-model packaging enabled. Versions: {versions}")
    else:
        versions = list(ensemble_cfg.get("versions", []))
        if enabled and len(versions) > 0:
            print(f"[ENSEMBLE] Multi-version packaging enabled. Versions: {versions}")

    if enabled and len(versions) > 0:
        select_best = bool(getattr(args, "best", False))
        use_swa = not bool(getattr(args, "no_swa", False))

        packaged_head_rel_paths: list[str] = []
        # Track a canonical version for config snapshot (first in list)
        canonical_ver: str = versions[0]
        canonical_snapshot: Path | None = _find_config_snapshot_for_version(cfg, repo_root, canonical_ver)
        for ver in versions:
            # Resolve per-version dirs without mutating base cfg
            tmp_cfg = dict(cfg or {})
            tmp_cfg["version"] = ver
            v_log_dir, v_ckpt_dir = resolve_dirs(tmp_cfg)

            # If train_all exists for this version, prefer packaging ONLY train_all.
            # This avoids exporting fold_*/ heads when a consolidated train_all head exists.
            train_all_dir = v_ckpt_dir / "train_all"
            if train_all_dir.exists() and train_all_dir.is_dir():
                dst_dir = weights_dir / "head" / ver
                exported: list[tuple[Path, Path]] = []

                chosen_ckpt: Path | None = None
                copied_head: Path | None = None
                if use_swa:
                    last_ckpt = train_all_dir / "last.ckpt"
                    if last_ckpt.is_file():
                        ckpt_for_swa = last_ckpt
                    else:
                        ckpt_candidates = [p for p in train_all_dir.glob("*.ckpt") if p.is_file()]
                        ckpt_candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
                        ckpt_for_swa = ckpt_candidates[0] if ckpt_candidates else None
                    if ckpt_for_swa is not None:
                        print(f"[SWA] {ver} train_all: attempting SWA export from {ckpt_for_swa}")
                        res = _export_swa_head_from_checkpoint(ckpt_for_swa, dst_dir)
                        if res is not None:
                            chosen_ckpt, copied_head = res
                            exported.append((chosen_ckpt, copied_head))
                            try:
                                rel = copied_head.relative_to(weights_dir / "head")
                                packaged_head_rel_paths.append(str(rel).replace("\\", "/"))
                            except Exception:
                                pass
                        else:
                            print(f"[SWA] {ver} train_all: SWA export failed; fallback to raw head-epoch*.pt.")

                if copied_head is None:
                    # Fallback to raw head-epoch*.pt under train_all/head
                    head_dir = train_all_dir / "head"
                    head_files = list_head_checkpoints(head_dir)
                    if not head_files:
                        raise FileNotFoundError(f"No head checkpoints found under: {head_dir}")
                    chosen_head: Path | None = None
                    if select_best:
                        metrics_csv = find_latest_metrics_csv(v_log_dir / "train_all")
                        # Fallback to root metrics.csv if train_all metrics are absent
                        if metrics_csv is None:
                            metrics_csv = find_latest_metrics_csv(v_log_dir)
                        if metrics_csv is not None:
                            best_epoch = pick_best_epoch_from_metrics(metrics_csv)
                            if best_epoch is not None:
                                p = find_head_by_epoch(head_dir, best_epoch)
                                if p is not None:
                                    chosen_head = p
                        if chosen_head is None:
                            chosen_head = pick_latest_head(head_files)
                    else:
                        chosen_head = pick_latest_head(head_files)
                    if chosen_head is None:
                        raise FileNotFoundError(f"Failed to determine a train_all head checkpoint for version {ver}.")
                    if dst_dir.exists() and dst_dir.is_dir():
                        shutil.rmtree(dst_dir)
                    dst_dir.mkdir(parents=True, exist_ok=True)
                    copied_head = dst_dir / "infer_head.pt"
                    shutil.copyfile(str(chosen_head), str(copied_head))
                    exported.append((chosen_head, copied_head))
                    try:
                        rel = copied_head.relative_to(weights_dir / "head")
                        packaged_head_rel_paths.append(str(rel).replace("\\", "/"))
                    except Exception:
                        pass

                # Copy z_score.json for train_all (best effort)
                zsrc = v_log_dir / "train_all" / "z_score.json"
                if not zsrc.is_file():
                    zsrc = v_log_dir / "z_score.json"
                if zsrc.is_file():
                    try:
                        shutil.copyfile(str(zsrc), str(dst_dir / "z_score.json"))
                    except Exception:
                        pass

                print(f"[ENSEMBLE] Packaged train_all head for version '{ver}':")
                for src_path, dst_path in exported:
                    print(f" - {src_path} -> {dst_path}")

            else:
                # Detect kfold by listing fold_* directories
                fold_dirs = [p for p in v_ckpt_dir.glob("fold_*") if p.is_dir()]
                if fold_dirs:
                    fold_dirs.sort(key=lambda p: p.name)
                    exported = []
                    for fold_dir in fold_dirs:
                        try:
                            fold_idx = int(str(fold_dir.name).split("_", 1)[1])
                        except Exception:
                            # fallback: enumerate
                            fold_idx = None
                        dst_dir = weights_dir / "head" / ver / (f"fold_{fold_idx}" if fold_idx is not None else fold_dir.name)

                        # Try SWA export
                        chosen: Path | None = None
                        dst_path: Path | None = None
                        if use_swa:
                            last_ckpt = fold_dir / "last.ckpt"
                            if last_ckpt.is_file():
                                ckpt_for_swa = last_ckpt
                            else:
                                ckpt_candidates = [p for p in fold_dir.glob("*.ckpt") if p.is_file()]
                                ckpt_candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
                                ckpt_for_swa = ckpt_candidates[0] if ckpt_candidates else None
                            if ckpt_for_swa is not None:
                                print(f"[SWA] {ver} fold {fold_dir.name}: attempting SWA export from {ckpt_for_swa}")
                                res = _export_swa_head_from_checkpoint(ckpt_for_swa, dst_dir)
                                if res is not None:
                                    chosen, dst_path = res
                                    exported.append((chosen, dst_path))
                                    try:
                                        rel = dst_path.relative_to(weights_dir / "head")
                                        packaged_head_rel_paths.append(str(rel).replace("\\", "/"))
                                    except Exception:
                                        pass
                                    # Copy z_score.json for this version/fold if present
                                    zsrc = v_log_dir / fold_dir.name / "z_score.json"
                                    if zsrc.is_file():
                                        try:
                                            shutil.copyfile(str(zsrc), str(dst_dir / "z_score.json"))
                                        except Exception:
                                            pass
                                    # done this fold
                                    continue
                                else:
                                    print(f"[SWA] {ver} fold {fold_dir.name}: SWA export failed; fallback to raw head-epoch*.pt.")

                        # Fallback to raw head-epoch*.pt under fold_dir/head
                        fold_ckpt_head_dir = fold_dir / "head"
                        head_files = list_head_checkpoints(fold_ckpt_head_dir)
                        if not head_files:
                            raise FileNotFoundError(f"No head checkpoints found under: {fold_ckpt_head_dir}")
                        if select_best:
                            metrics_csv = find_latest_metrics_csv(v_log_dir / fold_dir.name)
                            chosen = None
                            if metrics_csv is not None:
                                best_epoch = pick_best_epoch_from_metrics(metrics_csv)
                                if best_epoch is not None:
                                    p = find_head_by_epoch(fold_ckpt_head_dir, best_epoch)
                                    if p is not None:
                                        chosen = p
                            if chosen is None:
                                chosen = pick_latest_head(head_files)
                        else:
                            chosen = pick_latest_head(head_files)
                        if chosen is None:
                            raise FileNotFoundError(f"Failed to determine a head checkpoint for {ver}/{fold_dir.name}.")
                        if dst_dir.exists() and dst_dir.is_dir():
                            shutil.rmtree(dst_dir)
                        dst_dir.mkdir(parents=True, exist_ok=True)
                        dst_path = dst_dir / "infer_head.pt"
                        shutil.copyfile(str(chosen), str(dst_path))
                        exported.append((chosen, dst_path))
                        try:
                            rel = dst_path.relative_to(weights_dir / "head")
                            packaged_head_rel_paths.append(str(rel).replace("\\", "/"))
                        except Exception:
                            pass
                        zsrc = v_log_dir / fold_dir.name / "z_score.json"
                        if zsrc.is_file():
                            try:
                                shutil.copyfile(str(zsrc), str(dst_dir / "z_score.json"))
                            except Exception:
                                pass

                    print(f"[ENSEMBLE] Packaged heads for version '{ver}':")
                    for src_path, dst_path in exported:
                        print(f" - {src_path} -> {dst_path}")
                else:
                    # Single-run directory
                    dst_dir = weights_dir / "head" / ver
                    # Try SWA
                    chosen: Path | None = None
                    if use_swa:
                        main_last_ckpt = v_ckpt_dir / "last.ckpt"
                        if main_last_ckpt.is_file():
                            ckpt_for_swa = main_last_ckpt
                        else:
                            ckpt_candidates = [p for p in v_ckpt_dir.glob("*.ckpt") if p.is_file()]
                            ckpt_candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
                            ckpt_for_swa = ckpt_candidates[0] if ckpt_candidates else None
                        if ckpt_for_swa is not None:
                            print(f"[SWA] {ver}: attempting SWA export from {ckpt_for_swa}")
                            res = _export_swa_head_from_checkpoint(ckpt_for_swa, dst_dir)
                            if res is not None:
                                chosen, copied_head = res
                                print(f"Copied SWA head checkpoint: {chosen} -> {copied_head}")
                                try:
                                    rel = copied_head.relative_to(weights_dir / "head")
                                    packaged_head_rel_paths.append(str(rel).replace("\\", "/"))
                                except Exception:
                                    pass
                                # Copy z_score.json for this version if present
                                zsrc = v_log_dir / "z_score.json"
                                if zsrc.is_file():
                                    try:
                                        shutil.copyfile(str(zsrc), str(dst_dir / "z_score.json"))
                                    except Exception:
                                        pass
                            else:
                                print(f"[SWA] {ver}: SWA export failed; fallback to raw head-epoch*.pt.")
                    if chosen is None:
                        # Legacy raw copy
                        head_dir = v_ckpt_dir / "head"
                        head_files = list_head_checkpoints(head_dir)
                        if not head_files:
                            raise FileNotFoundError(f"No head checkpoints found under: {head_dir}")
                        if select_best:
                            metrics_csv = find_latest_metrics_csv(v_log_dir)
                            chosen = None
                            if metrics_csv is not None:
                                best_epoch = pick_best_epoch_from_metrics(metrics_csv)
                                if best_epoch is not None:
                                    p = find_head_by_epoch(head_dir, best_epoch)
                                    if p is not None:
                                        chosen = p
                            if chosen is None:
                                chosen = pick_latest_head(head_files)
                        else:
                            chosen = pick_latest_head(head_files)
                        if chosen is None:
                            raise FileNotFoundError(f"Failed to determine a head checkpoint for version {ver}.")
                        if dst_dir.exists() and dst_dir.is_dir():
                            shutil.rmtree(dst_dir)
                        dst_dir.mkdir(parents=True, exist_ok=True)
                        copied_head = dst_dir / "infer_head.pt"
                        shutil.copyfile(str(chosen), str(copied_head))
                        print(f"Copied head checkpoint: {chosen} -> {copied_head}")
                        try:
                            rel = copied_head.relative_to(weights_dir / "head")
                            packaged_head_rel_paths.append(str(rel).replace("\\", "/"))
                        except Exception:
                            pass
                        zsrc = v_log_dir / "z_score.json"
                        if zsrc.is_file():
                            try:
                                shutil.copyfile(str(zsrc), str(dst_dir / "z_score.json"))
                            except Exception:
                                pass

        # Write packaged ensemble manifest
        if packaged_head_rel_paths:
            try:
                import json as _json
                manifest = {
                    "aggregation": "mean",
                    "heads": [{"path": p} for p in packaged_head_rel_paths],
                }
                dst = weights_dir / "head" / "ensemble.json"
                dst.parent.mkdir(parents=True, exist_ok=True)
                with open(dst, "w", encoding="utf-8") as f:
                    _json.dump(manifest, f, indent=2)
                print(f"[ENSEMBLE] Wrote packaged manifest with {len(packaged_head_rel_paths)} heads: {dst}")
            except Exception as e:
                print(f"[ENSEMBLE] Failed to write packaged manifest: {e}")

        # Copy configs and src into weights/
        copy_tree(repo_root / "configs", weights_dir / "configs")
        # Override configs/train.yaml with snapshot from canonical version if available
        _copy_snapshot_into_weights(canonical_snapshot, repo_root, weights_dir)
        # Also store per-version snapshots so inference can pick backbone/data per model.
        for ver in versions:
            _copy_version_snapshot_into_weights(
                _find_config_snapshot_for_version(cfg, repo_root, ver),
                weights_dir=weights_dir,
                version=ver,
            )
        copy_tree(repo_root / "src", weights_dir / "src")
        copy_optional_third_party(repo_root, weights_dir)
        scripts_copied = copy_top_level_scripts(repo_root, weights_dir)
        # NEW: package TabPFN fit-state + feature caches for offline inference
        _package_tabpfn_inference_artifacts(repo_root=repo_root, weights_dir=weights_dir)
        print(f"Copied configs/ and src/ to: {weights_dir}")
        if scripts_copied:
            print("Copied scripts to weights/scripts/:")
            for p in scripts_copied:
                print(f" - {p}")
        return

    # If kfold is enabled, export one head per fold under weights/head/fold_*/infer_head.pt.
    # If train_all is enabled, additionally export a preferred single head to
    # weights/head/infer_head.pt (used as the default for offline inference).
    kfold_cfg = cfg.get("kfold", {})
    kfold_enabled = bool(kfold_cfg.get("enabled", False))

    train_all_cfg = cfg.get("train_all", {})
    train_all_enabled = bool(train_all_cfg.get("enabled", False))

    select_best = bool(getattr(args, "best", False))
    use_swa = not bool(getattr(args, "no_swa", False))

    # Collect relative paths (under weights/head) for packaged heads to emit a packaged manifest
    packaged_head_rel_paths: list[str] = []
    emit_packaged_manifest: bool = bool(ensemble_cfg.get("enabled", False))

    # 1) Per-fold heads for k-fold training (if enabled **and** train_all is disabled).
    # When both kfold and train_all are enabled, we now only package the train_all
    # head as the preferred single model for offline inference, and skip exporting
    # individual per-fold heads here.
    if kfold_enabled and not train_all_enabled:
        k = int(kfold_cfg.get("k", 5))
        exported = []
        for fold_idx in range(k):
            fold_root_ckpt_dir = ckpt_dir / f"fold_{fold_idx}"
            fold_ckpt_head_dir = fold_root_ckpt_dir / "head"
            fold_log_dir = log_dir / f"fold_{fold_idx}"

            # When SWA is enabled, attempt to build the head from the SWA-averaged
            # weights stored in this fold's Lightning checkpoint (prefer last.ckpt).
            chosen: Path | None = None
            dst_path: Path | None = None
            if use_swa:
                last_ckpt = fold_root_ckpt_dir / "last.ckpt"
                ckpt_for_swa: Path | None
                if last_ckpt.is_file():
                    ckpt_for_swa = last_ckpt
                else:
                    # Fallback: any .ckpt in the fold directory, preferring the most recent.
                    ckpt_candidates = [p for p in fold_root_ckpt_dir.glob("*.ckpt") if p.is_file()]
                    ckpt_candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
                    ckpt_for_swa = ckpt_candidates[0] if ckpt_candidates else None

                if ckpt_for_swa is not None:
                    print(f"[SWA] Fold {fold_idx}: attempting SWA export from checkpoint: {ckpt_for_swa}")
                    dst_dir = weights_dir / "head" / f"fold_{fold_idx}"
                    res = _export_swa_head_from_checkpoint(ckpt_for_swa, dst_dir)
                    if res is not None:
                        chosen, dst_path = res
                        exported.append((chosen, dst_path))
                        try:
                            rel = dst_path.relative_to(weights_dir / "head")
                            packaged_head_rel_paths.append(str(rel).replace("\\", "/"))
                        except Exception:
                            pass
                        # Copy z_score.json for this fold if present
                        zsrc = fold_log_dir / "z_score.json"
                        if zsrc.is_file():
                            try:
                                shutil.copyfile(str(zsrc), str(dst_dir / "z_score.json"))
                            except Exception:
                                pass
                        # SWA export for this fold succeeded; continue to next fold.
                        continue
                    else:
                        print(f"[SWA] Fold {fold_idx}: SWA export failed or unavailable, falling back to raw head-epoch*.pt.")
                else:
                    print(f"[SWA] Fold {fold_idx}: no .ckpt files found for SWA export, falling back to raw head-epoch*.pt.")

            # Legacy behaviour (or SWA unavailable): select a head checkpoint file
            # under fold_{i}/head/ as before.
            head_files = list_head_checkpoints(fold_ckpt_head_dir)
            if not head_files:
                raise FileNotFoundError(f"No head checkpoints found under: {fold_ckpt_head_dir}")

            if select_best:
                # Pick best by val_loss using this fold's latest metrics.csv
                metrics_csv = find_latest_metrics_csv(fold_log_dir)
                chosen = None
                if metrics_csv is not None:
                    best_epoch = pick_best_epoch_from_metrics(metrics_csv)
                    if best_epoch is not None:
                        p = find_head_by_epoch(fold_ckpt_head_dir, best_epoch)
                        if p is not None:
                            chosen = p
                # Fallback to latest-epoch checkpoint if best could not be determined
                if chosen is None:
                    chosen = pick_latest_head(head_files)
            else:
                # Default: purely latest-epoch checkpoint
                chosen = pick_latest_head(head_files)

            if chosen is None:
                raise FileNotFoundError(f"Failed to determine a head checkpoint for fold {fold_idx}.")

            # Copy into weights/head/fold_{i}/infer_head.pt (clean per-fold dir)
            dst_dir = weights_dir / "head" / f"fold_{fold_idx}"
            if dst_dir.exists() and dst_dir.is_dir():
                shutil.rmtree(dst_dir)
            dst_dir.mkdir(parents=True, exist_ok=True)
            dst_path = dst_dir / "infer_head.pt"
            shutil.copyfile(str(chosen), str(dst_path))
            exported.append((chosen, dst_path))
            try:
                rel = dst_path.relative_to(weights_dir / "head")
                packaged_head_rel_paths.append(str(rel).replace("\\", "/"))
            except Exception:
                pass
            # Copy z_score.json for this fold if present
            zsrc = fold_log_dir / "z_score.json"
            if zsrc.is_file():
                try:
                    shutil.copyfile(str(zsrc), str(dst_dir / "z_score.json"))
                except Exception:
                    pass

        print("Copied per-fold head checkpoints (src -> dst):")
        for src_path, dst_path in exported:
            print(f" - {src_path} -> {dst_path}")

    # 2) Preferred train_all head exported to weights/head/infer_head.pt (if enabled).
    train_all_head_dst: Path | None = None
    if train_all_enabled:
        train_all_root_ckpt_dir = ckpt_dir / "train_all"
        train_all_head_dir = train_all_root_ckpt_dir / "head"
        train_all_log_dir = log_dir / "train_all"

        chosen: Path | None = None
        # When SWA is enabled, attempt to build the head from SWA-averaged weights first.
        if use_swa:
            main_last_ckpt = train_all_root_ckpt_dir / "last.ckpt"
            ckpt_for_swa: Path | None
            if main_last_ckpt.is_file():
                ckpt_for_swa = main_last_ckpt
            else:
                ckpt_candidates = [p for p in train_all_root_ckpt_dir.glob("*.ckpt") if p.is_file()]
                ckpt_candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
                ckpt_for_swa = ckpt_candidates[0] if ckpt_candidates else None

            if ckpt_for_swa is not None:
                print(f"[SWA] train_all: attempting SWA export from checkpoint: {ckpt_for_swa}")
                swa_dst_dir = weights_dir / "head" / "train_all"
                res = _export_swa_head_from_checkpoint(ckpt_for_swa, swa_dst_dir)
                if res is not None:
                    chosen, swa_head_path = res
                    # Copy to the preferred root location without removing existing fold_* dirs.
                    head_dst_dir = weights_dir / "head"
                    head_dst_dir.mkdir(parents=True, exist_ok=True)
                    train_all_head_dst = head_dst_dir / "infer_head.pt"
                    shutil.copyfile(str(swa_head_path), str(train_all_head_dst))
                    # Remove the temporary train_all/ copy to avoid redundant packaging.
                    try:
                        shutil.rmtree(swa_dst_dir, ignore_errors=True)
                    except Exception:
                        pass
                    try:
                        rel = train_all_head_dst.relative_to(weights_dir / "head")
                        packaged_head_rel_paths.append(str(rel).replace("\\", "/"))
                    except Exception:
                        pass
                    # Copy z_score.json for train_all if present
                    zsrc = train_all_log_dir / "z_score.json"
                    if zsrc.is_file():
                        try:
                            shutil.copyfile(str(zsrc), str(head_dst_dir / "z_score.json"))
                        except Exception:
                            pass
                else:
                    print("[SWA] train_all: SWA export failed; falling back to raw head-epoch*.pt.")
            else:
                print("[SWA] train_all: no .ckpt files found for SWA export, falling back to raw head-epoch*.pt.")

        # Fallback: latest/best raw head checkpoint under train_all/head
        if train_all_head_dst is None:
            head_files = list_head_checkpoints(train_all_head_dir)
            if not head_files:
                raise FileNotFoundError(f"No head checkpoints found under: {train_all_head_dir}")

            if select_best:
                metrics_csv = find_latest_metrics_csv(train_all_log_dir)
                chosen = None
                if metrics_csv is not None:
                    best_epoch = pick_best_epoch_from_metrics(metrics_csv)
                    if best_epoch is not None:
                        p = find_head_by_epoch(train_all_head_dir, best_epoch)
                        if p is not None:
                            chosen = p
                if chosen is None:
                    chosen = pick_latest_head(head_files)
            else:
                chosen = pick_latest_head(head_files)

            if chosen is None:
                raise FileNotFoundError("Failed to determine a head checkpoint for train_all.")

            head_dst_dir = weights_dir / "head"
            head_dst_dir.mkdir(parents=True, exist_ok=True)
            train_all_head_dst = head_dst_dir / "infer_head.pt"
            shutil.copyfile(str(chosen), str(train_all_head_dst))
            try:
                rel = train_all_head_dst.relative_to(weights_dir / "head")
                packaged_head_rel_paths.append(str(rel).replace("\\", "/"))
            except Exception:
                pass
            # Copy z_score.json for train_all if present
            zsrc = train_all_log_dir / "z_score.json"
            if zsrc.is_file():
                try:
                    shutil.copyfile(str(zsrc), str(head_dst_dir / "z_score.json"))
                except Exception:
                    pass

        if train_all_head_dst is not None:
            print(f"Copied train_all head checkpoint -> {train_all_head_dst}")

    # 3) Single-run export (non-kfold, non-train_all) for backward-compatibility.
    if not kfold_enabled and not train_all_enabled:
        head_dir = ckpt_dir / "head"

        # When SWA is enabled, first attempt to build the head from SWA-averaged
        # weights stored in the main checkpoint directory (prefer last.ckpt).
        if use_swa:
            main_last_ckpt = ckpt_dir / "last.ckpt"
            ckpt_for_swa: Path | None
            if main_last_ckpt.is_file():
                ckpt_for_swa = main_last_ckpt
            else:
                ckpt_candidates = [p for p in ckpt_dir.glob("*.ckpt") if p.is_file()]
                ckpt_candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
                ckpt_for_swa = ckpt_candidates[0] if ckpt_candidates else None

            if ckpt_for_swa is not None:
                print(f"[SWA] Single-run: attempting SWA export from checkpoint: {ckpt_for_swa}")
                dst_dir = weights_dir / "head"
                res = _export_swa_head_from_checkpoint(ckpt_for_swa, dst_dir)
                if res is not None:
                    chosen, copied_head = res
                    print(f"Copied SWA head checkpoint: {chosen} -> {copied_head}")
                    try:
                        rel = copied_head.relative_to(weights_dir / "head")
                        packaged_head_rel_paths.append(str(rel).replace("\\", "/"))
                    except Exception:
                        pass
                    # Copy z_score.json from run log dir if present
                    zsrc = log_dir / "z_score.json"
                    if zsrc.is_file():
                        try:
                            shutil.copyfile(str(zsrc), str(dst_dir / "z_score.json"))
                        except Exception:
                            pass
                    # SWA export succeeded; skip legacy head selection.
                    head_dir = None  # type: ignore[assignment]
                else:
                    # Fall back to legacy behaviour below
                    print("[SWA] Single-run: SWA export failed or unavailable, falling back to raw head-epoch*.pt.")
            else:
                print("[SWA] Single-run: no .ckpt files found for SWA export, falling back to raw head-epoch*.pt.")

        if head_dir is not None:
            head_files = list_head_checkpoints(head_dir)
            if not head_files:
                raise FileNotFoundError(f"No head checkpoints found under: {head_dir}")

            if select_best:
                metrics_csv = find_latest_metrics_csv(log_dir)
                chosen = None
                if metrics_csv is not None:
                    best_epoch = pick_best_epoch_from_metrics(metrics_csv)
                    if best_epoch is not None:
                        p = find_head_by_epoch(head_dir, best_epoch)
                        if p is not None:
                            chosen = p
                if chosen is None:
                    chosen = pick_latest_head(head_files)
            else:
                chosen = pick_latest_head(head_files)

            if chosen is None:
                raise FileNotFoundError("Failed to determine a head checkpoint to package.")

            copied_head = copy_head_to_weights(chosen, weights_dir)
            print(f"Copied head checkpoint: {chosen} -> {copied_head}")
            try:
                rel = copied_head.relative_to(weights_dir / "head")
                packaged_head_rel_paths.append(str(rel).replace("\\", "/"))
            except Exception:
                pass
        # Copy z_score.json from run log dir if present
        zsrc = log_dir / "z_score.json"
        if zsrc.is_file():
            try:
                shutil.copyfile(str(zsrc), str(weights_dir / "z_score.json"))
            except Exception:
                pass

    # If ensemble is enabled, emit a packaged manifest at weights/head/ensemble.json
    if emit_packaged_manifest and packaged_head_rel_paths:
        try:
            import json as _json
            manifest = {
                "aggregation": "mean",
                "heads": [{"path": p} for p in packaged_head_rel_paths],
            }
            dst = weights_dir / "head" / "ensemble.json"
            dst.parent.mkdir(parents=True, exist_ok=True)
            with open(dst, "w", encoding="utf-8") as f:
                _json.dump(manifest, f, indent=2)
            print(f"[ENSEMBLE] Wrote packaged manifest: {dst}")
        except Exception as e:
            print(f"[ENSEMBLE] Failed to write packaged manifest: {e}")

    # Copy configs and src into weights/
    copy_tree(repo_root / "configs", weights_dir / "configs")
    # For single-version packaging, prefer snapshot from this cfg.version if available
    snapshot_ver = cfg.get("version", None)
    snapshot_ver = None if snapshot_ver in (None, "", "null") else str(snapshot_ver)
    snapshot_path = _find_config_snapshot_for_version(cfg, repo_root, snapshot_ver) if snapshot_ver else None
    _copy_snapshot_into_weights(snapshot_path, repo_root, weights_dir)
    copy_tree(repo_root / "src", weights_dir / "src")
    copy_optional_third_party(repo_root, weights_dir)
    scripts_copied = copy_top_level_scripts(repo_root, weights_dir)
    # NEW: package TabPFN fit-state + feature caches for offline inference
    _package_tabpfn_inference_artifacts(repo_root=repo_root, weights_dir=weights_dir)
    print(f"Copied configs/ and src/ to: {weights_dir}")
    if scripts_copied:
        print("Copied scripts to weights/scripts/:")
        for p in scripts_copied:
            print(f" - {p}")


if __name__ == "__main__":
    main()


