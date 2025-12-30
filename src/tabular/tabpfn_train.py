from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import yaml
from loguru import logger
import shutil

from src.data.csiro_pivot import read_and_pivot_csiro_train_csv
from src.metrics import TARGETS_5D_ORDER, weighted_r2_logspace
from src.tabular.features import build_y_5d
from src.tabular.tabpfn_image_features import (
    extract_dinov3_cls_features_as_tabular,
    extract_head_penultimate_features_as_tabular,
    resolve_fold_head_weights_path,
    resolve_train_all_head_weights_path,
)
from src.tabular.tabpfn_finetune import (
    finetune_tabpfn_regressor_on_fold,
    parse_finetune_config,
)
from src.tabular.tabpfn_utils import (
    configure_tabpfn_env,
    ensure_on_sys_path,
    import_tabpfn_regressor,
    parse_tabpfn_inference_precision,
    resolve_repo_root,
    resolve_under_repo,
)
from src.training.splits import build_kfold_splits


@dataclass(frozen=True)
class TabPFNRunParams:
    """Resolved TabPFN hyperparameters for a single run."""

    n_estimators: int
    device: str
    fit_mode: str
    inference_precision: str
    n_jobs: int
    ignore_pretraining_limits: bool
    enable_telemetry: bool
    model_cache_dir: Optional[str]
    model_path: str


def parse_args(repo_root: Path, argv: Optional[list[str]] = None) -> argparse.Namespace:
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
        help=(
            "Legacy override for log_dir. If omitted, uses <logging.log_dir>/<version>/ (train.yaml-style). "
            "Checkpoints always go to <logging.ckpt_dir>/<version>/."
        ),
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

    return p.parse_args(args=argv)


def _load_yaml(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        obj = yaml.safe_load(f)
    return dict(obj or {})


def _save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _resolve_tabpfn_params(
    *,
    repo_root: Path,
    args: argparse.Namespace,
    tabpfn_cfg: Dict[str, Any],
) -> TabPFNRunParams:
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

    return TabPFNRunParams(
        n_estimators=int(n_estimators),
        device=str(device),
        fit_mode=str(fit_mode),
        inference_precision=str(inference_precision),
        n_jobs=int(n_jobs),
        ignore_pretraining_limits=bool(ignore_pretraining_limits),
        enable_telemetry=bool(enable_telemetry),
        model_cache_dir=str(model_cache_dir) if model_cache_dir is not None else None,
        model_path=str(model_path_p),
    )


def _resolve_versioned_dirs(
    *,
    repo_root: Path,
    cfg: Dict[str, Any],
    version: Optional[str],
    args: argparse.Namespace,
) -> tuple[Path, Path]:
    """
    Resolve (log_dir, ckpt_dir) with the same semantics as `train.py`:
      - log_dir  = <logging.log_dir>/<version>  (if version is set)
      - ckpt_dir = <logging.ckpt_dir>/<version> (if version is set)

    NOTE: `--out-dir` is treated as a legacy override for log_dir only.
    """
    logging_cfg = dict(cfg.get("logging", {}) or {})
    base_log_dir = Path(logging_cfg.get("log_dir", "outputs")).expanduser()
    base_ckpt_dir = Path(logging_cfg.get("ckpt_dir", "outputs/checkpoints")).expanduser()

    base_log_dir = resolve_under_repo(repo_root, base_log_dir)
    base_ckpt_dir = resolve_under_repo(repo_root, base_ckpt_dir)

    log_dir = (base_log_dir / version) if version else base_log_dir
    ckpt_dir = (base_ckpt_dir / version) if version else base_ckpt_dir

    if args.out_dir is not None:
        log_dir = resolve_under_repo(repo_root, args.out_dir)
        logger.warning(
            "--out-dir overrides the versioned log_dir layout; checkpoints will still be written under {}",
            str(ckpt_dir),
        )

    log_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    return log_dir, ckpt_dir


def run_tabpfn_cv(*, repo_root: Path, args: argparse.Namespace) -> Path:
    cfg_path = Path(args.config).expanduser()
    if not cfg_path.is_absolute():
        cfg_path = (repo_root / cfg_path).resolve()
    if not cfg_path.is_file():
        raise FileNotFoundError(f"Config YAML not found: {cfg_path}")

    cfg = _load_yaml(cfg_path)
    version = cfg.get("version", None)
    version = None if version in (None, "", "null") else str(version)

    # Resolve versioned log/ckpt dirs (mirrors train.py)
    log_dir, ckpt_dir = _resolve_versioned_dirs(
        repo_root=repo_root, cfg=cfg, version=version, args=args
    )

    # Snapshot config for reproducibility (mirrors train.py)
    try:
        shutil.copyfile(str(cfg_path), str(log_dir / "train_tabpfn.yaml"))
    except Exception as e:
        logger.warning(f"Failed to snapshot train_tabpfn config to {log_dir}/train_tabpfn.yaml: {e}")

    # Optional TensorBoard (mirror train.yaml layout: log_dir/tensorboard/)
    tb_writer = None
    try:
        tb_cfg = dict((cfg.get("logging", {}) or {}).get("tensorboard", {}) or {})
        tb_enabled = bool(tb_cfg.get("enabled", True))
        if tb_enabled:
            from torch.utils.tensorboard import SummaryWriter  # type: ignore

            tb_log_dir = (log_dir / "tensorboard").resolve()
            tb_log_dir.mkdir(parents=True, exist_ok=True)
            tb_writer = SummaryWriter(log_dir=str(tb_log_dir), filename_suffix="")
            # best-effort: record config as text
            try:
                tb_writer.add_text("run/config_path", str(cfg_path))
                tb_writer.add_text("run/version", str(version))
            except Exception:
                pass
    except Exception as e:
        logger.warning(f"[TabPFN] TensorBoard disabled (SummaryWriter unavailable): {e}")
        tb_writer = None

    seed = int(args.seed if args.seed is not None else cfg.get("seed", 42))
    np.random.seed(seed)

    tabpfn_cfg = dict(cfg.get("tabpfn", {}) or {})
    image_features_cfg = dict(cfg.get("image_features", {}) or {})
    finetune_cfg_raw = dict(cfg.get("finetune", {}) or {})

    # Load image-model config (defines backbone weights + preprocessing)
    img_cfg_path_raw = image_features_cfg.get("train_config", "configs/train.yaml")
    img_cfg_path = resolve_under_repo(repo_root, str(img_cfg_path_raw))
    if not img_cfg_path.is_file():
        raise FileNotFoundError(f"Image model config not found: {img_cfg_path}")
    cfg_img = _load_yaml(img_cfg_path)

    # Feature extraction mode (default keeps backward compatible behaviour)
    image_features_mode = (
        str(args.image_features_mode).strip().lower()
        if getattr(args, "image_features_mode", None) is not None
        else str(image_features_cfg.get("mode", "head_penultimate")).strip().lower()
    )
    if image_features_mode not in ("head_penultimate", "dinov3_only"):
        raise SystemExit(f"Unsupported image_features.mode: {image_features_mode!r}")

    # NOTE: When using head_penultimate features, make sure the fold head weights correspond
    # to the same k-fold configuration (seed/k/even_split/grouping) used here. Otherwise you
    # may introduce leakage or degrade metrics.
    if image_features_mode == "head_penultimate":
        logger.warning(
            "head_penultimate mode assumes fold head weights match this run's k-fold config "
            "(seed/k/even_split/group_by_date_state). Please ensure they were trained the same way."
        )

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

    tabpfn_params = _resolve_tabpfn_params(repo_root=repo_root, args=args, tabpfn_cfg=tabpfn_cfg)

    # TabPFN settings via env (must be set before importing tabpfn)
    configure_tabpfn_env(
        repo_root=repo_root,
        enable_telemetry=tabpfn_params.enable_telemetry,
        model_cache_dir=tabpfn_params.model_cache_dir,
    )

    TabPFNRegressor = import_tabpfn_regressor(repo_root)

    # Lazy import (sklearn is already a dependency of tabpfn; we keep it local)
    from sklearn.multioutput import MultiOutputRegressor

    # Load + pivot train.csv (controlled by this config, not the image config)
    data_cfg = dict(cfg.get("data", {}) or {})
    data_root = resolve_under_repo(repo_root, str(data_cfg.get("root", "data")))
    train_csv = str(data_cfg.get("train_csv", "train.csv"))
    target_order = list(data_cfg.get("target_order", ["Dry_Total_g"]))

    full_df = read_and_pivot_csiro_train_csv(
        data_root=str(data_root),
        train_csv=train_csv,
        target_order=target_order,
    )
    if len(full_df) < 2:
        raise ValueError(f"Not enough samples after pivoting: {len(full_df)}")

    # Optional k-fold / train-all configuration (mirrors `train.py` semantics)
    kfold_cfg = dict(cfg.get("kfold", {}) or {})
    kfold_enabled = bool(kfold_cfg.get("enabled", True))

    train_all_cfg = dict(cfg.get("train_all", {}) or {})
    train_all_enabled = bool(train_all_cfg.get("enabled", False))

    if not kfold_enabled and not train_all_enabled:
        raise SystemExit("Nothing to run: both `kfold.enabled` and `train_all.enabled` are false.")

    # Resolve k-fold split knobs (only used when kfold_enabled=true).
    k_eff = int(args.k) if args.k is not None else int(kfold_cfg.get("k", 5))
    even_split_effective = bool(args.even_split) if args.even_split is not None else bool(kfold_cfg.get("even_split", False))
    group_by_date_state_effective = bool(args.group_by_date_state) if args.group_by_date_state is not None else bool(kfold_cfg.get("group_by_date_state", True))

    splits: list[tuple[np.ndarray, np.ndarray]] = []
    oof_pred: Optional[np.ndarray] = None
    val_counts = np.zeros((len(full_df),), dtype=np.int64)
    if kfold_enabled:
        splits = build_kfold_splits(
            full_df,
            cfg,
            k=k_eff,
            even_split=even_split_effective,
            group_by_date_state=group_by_date_state_effective,
            seed=int(seed),
        )
        if even_split_effective:
            oof_pred = None
        else:
            oof_pred = np.full((len(full_df), len(TARGETS_5D_ORDER)), np.nan, dtype=np.float64)

    y_true_full = build_y_5d(full_df, fillna=0.0)

    fold_metrics: list[dict[str, Any]] = []
    fold_pred_frames: list[pd.DataFrame] = []

    logger.info("Repo root: {}", repo_root)
    logger.info("Config: {}", cfg_path)
    logger.info("Log dir: {}", log_dir)
    logger.info("Ckpt dir: {}", ckpt_dir)
    logger.info("Data: {} / {}", data_root, train_csv)
    logger.info("Full pivoted samples: {}", len(full_df))
    if kfold_enabled:
        logger.info(
            "Folds: {} (k={} even_split={} group_by_date_state={} split_seed={})",
            len(splits),
            int(k_eff),
            even_split_effective,
            group_by_date_state_effective,
            int(seed),
        )
    else:
        logger.info("Folds: (kfold disabled)")
    logger.info("train_all: enabled={}", bool(train_all_enabled))
    if image_features_mode == "dinov3_only":
        logger.info("TabPFN X: dinov3_only CLS token (no LoRA, no head) from {}", img_cfg_path)
    else:
        logger.info("TabPFN X: head penultimate features (per-fold, pre-linear) from {}", img_cfg_path)
    logger.info("Targets (5D): {}", TARGETS_5D_ORDER)
    logger.info(
        "TabPFN params: n_estimators={}, device={}, fit_mode={}, inference_precision={}, n_jobs={}",
        int(tabpfn_params.n_estimators),
        str(tabpfn_params.device),
        str(tabpfn_params.fit_mode),
        str(tabpfn_params.inference_precision),
        int(tabpfn_params.n_jobs),
    )
    logger.info("TabPFN model_path (local): {}", tabpfn_params.model_path)

    finetune_cfg = parse_finetune_config(finetune_cfg_raw)
    if finetune_cfg.enabled:
        logger.info(
            "TabPFN finetune: enabled=true (epochs={} n_estimators_train={} lr={})",
            int(finetune_cfg.max_epochs),
            int(finetune_cfg.n_estimators),
            float(finetune_cfg.optimizer.lr),
        )
    else:
        logger.info("TabPFN finetune: enabled=false")

    fold_feature_meta: list[dict[str, Any]] = []

    # Pre-extract global features once when they are fold-invariant (dinov3_only).
    X_all_global: Optional[np.ndarray] = None
    global_cache_path: Optional[Path] = None
    if image_features_mode == "dinov3_only":
        cache_path_raw = image_features_cfg.get("cache_path", None)
        if isinstance(cache_path_raw, str) and cache_path_raw.strip():
            base = resolve_under_repo(repo_root, cache_path_raw)
            global_cache_path = (base / "dinov3_cls_features.pt").resolve() if base.is_dir() else base

        X_all_global = extract_dinov3_cls_features_as_tabular(
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

        fold_log_dir = (log_dir / f"fold_{fold_idx}").resolve()
        fold_ckpt_dir = (ckpt_dir / f"fold_{fold_idx}").resolve()
        fold_log_dir.mkdir(parents=True, exist_ok=True)
        fold_ckpt_dir.mkdir(parents=True, exist_ok=True)

        head_weights_path: Optional[Path] = None
        X_all: np.ndarray
        if image_features_mode == "dinov3_only":
            if X_all_global is None:
                raise RuntimeError("Internal error: X_all_global is None for dinov3_only mode.")
            X_all = X_all_global
        else:
            # Resolve fold-specific head weights; skip missing folds.
            head_weights_path = resolve_fold_head_weights_path(repo_root=repo_root, cfg_img=cfg_img, fold_idx=int(fold_idx))
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
                base = resolve_under_repo(repo_root, cache_path_raw)
                if base.is_dir():
                    cache_path = (base / f"head_penultimate_features.fold_{fold_idx}.pt").resolve()
                else:
                    # Insert fold suffix before extension
                    cache_path = base.with_name(f"{base.stem}.fold_{fold_idx}{base.suffix}")

            # Extract features using THIS fold's image head (prevents train_all leakage).
            X_all = extract_head_penultimate_features_as_tabular(
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

        # Optional: fine-tune TabPFN weights on this fold's training data.
        model_path_for_fold = str(tabpfn_params.model_path)
        finetune_summary: Optional[dict[str, Any]] = None
        if finetune_cfg.enabled:
            try:
                model_path_for_fold, finetune_summary = finetune_tabpfn_regressor_on_fold(
                    TabPFNRegressor=TabPFNRegressor,
                    base_model_path=str(tabpfn_params.model_path),
                    device=str(tabpfn_params.device),
                    ignore_pretraining_limits=bool(tabpfn_params.ignore_pretraining_limits),
                    seed=int(seed),
                    fold_idx=int(fold_idx),
                    finetune_cfg=finetune_cfg,
                    eval_n_estimators=int(getattr(finetune_cfg, "eval_n_estimators", 8)),
                    eval_fit_mode=str(tabpfn_params.fit_mode),
                    eval_inference_precision=str(tabpfn_params.inference_precision),
                    X_train=X_train,
                    y_train_5d=y_train,
                    X_val=X_val,
                    y_val_5d=y_val,
                    fold_log_dir=fold_log_dir,
                    fold_ckpt_dir=fold_ckpt_dir,
                    trainer_cfg_raw=dict(cfg.get("trainer", {}) or {}),
                    tb_log_dir=(log_dir / "tensorboard").resolve() if tb_writer is not None else None,
                    tb_run_name=f"kfold/fold_{int(fold_idx)}/finetune",
                )
            except Exception as e:
                logger.error(f"[TabPFN][finetune] fold={fold_idx} failed: {e}")
                raise

        base = TabPFNRegressor(
            n_estimators=int(tabpfn_params.n_estimators),
            device=str(tabpfn_params.device),
            fit_mode=str(tabpfn_params.fit_mode),
            inference_precision=parse_tabpfn_inference_precision(tabpfn_params.inference_precision),
            random_state=int(seed + fold_idx),
            ignore_pretraining_limits=bool(tabpfn_params.ignore_pretraining_limits),
            model_path=str(model_path_for_fold),
        )
        model = MultiOutputRegressor(base, n_jobs=int(tabpfn_params.n_jobs))
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
                    tabpfn_params.model_path,
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
            "r2_per_target_logspace": {name: float(r2_per[i]) for i, name in enumerate(TARGETS_5D_ORDER)},
            "tabpfn_model_path_used": str(model_path_for_fold),
            "finetune": finetune_summary,
        }
        fold_metrics.append(m)
        logger.info(
            "Fold {}: train={} val={} weighted_r2(log)={:.6f}",
            fold_idx,
            len(train_df),
            len(val_df),
            float(r2),
        )
        if tb_writer is not None:
            try:
                tb_writer.add_scalar("kfold/val/weighted_r2_log", float(r2), int(fold_idx))
                for j, name in enumerate(TARGETS_5D_ORDER):
                    tb_writer.add_scalar(f"kfold/val/r2_logspace/{name}", float(r2_per[j]), int(fold_idx))
            except Exception:
                pass

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

    # Save common run metadata at the root (mirrors train.py snapshot style).
    _save_json(log_dir / "run_args.json", {k: getattr(args, k) for k in vars(args).keys()})

    summary_cfg: dict[str, Any] = {
        "seed": int(seed),
        "config_path": str(cfg_path),
        "version": version,
        "log_dir": str(log_dir),
        "ckpt_dir": str(ckpt_dir),
        "modes": {"kfold": bool(kfold_enabled), "train_all": bool(train_all_enabled)},
        "tabpfn_x": {
            "mode": str(image_features_mode),
            "source": (
                "dinov3_cls_token_no_lora_no_head" if image_features_mode == "dinov3_only" else "image_head_penultimate_pre_linear"
            ),
            "runs": fold_feature_meta,
            "image_train_config": str(img_cfg_path),
            "fusion": image_features_cfg.get("fusion", "mean") if image_features_mode != "dinov3_only" else None,
        },
        "targets_5d_order": list(TARGETS_5D_ORDER),
        "kfold": {
            "enabled": bool(kfold_enabled),
            "k": int(len(splits)) if kfold_enabled else int(k_eff),
            "even_split": bool(even_split_effective),
            "group_by_date_state": bool(group_by_date_state_effective),
        },
        "train_all": {"enabled": bool(train_all_enabled)},
        "tabpfn": {
            "n_estimators": int(tabpfn_params.n_estimators),
            "device": str(tabpfn_params.device),
            "fit_mode": str(tabpfn_params.fit_mode),
            "inference_precision": str(tabpfn_params.inference_precision),
            "n_jobs": int(tabpfn_params.n_jobs),
            "ignore_pretraining_limits": bool(tabpfn_params.ignore_pretraining_limits),
            "model_path": str(tabpfn_params.model_path),
            "model_cache_dir": str(tabpfn_params.model_cache_dir) if tabpfn_params.model_cache_dir is not None else None,
            "enable_telemetry": bool(tabpfn_params.enable_telemetry),
        },
        "finetune": finetune_cfg_raw,
    }

    # -------------------------
    # 1) K-fold evaluation run
    # -------------------------
    kfold_summary: Optional[dict[str, Any]] = None
    if kfold_enabled:
        _save_json(log_dir / "fold_metrics.json", fold_metrics)

        preds_all = pd.concat(fold_pred_frames, axis=0, ignore_index=True) if fold_pred_frames else pd.DataFrame()
        preds_all.to_csv(log_dir / "val_predictions.csv", index=False)

        # Coverage / aggregate metrics
        coverage = {
            "num_samples": int(len(full_df)),
            "val_counts_min": int(val_counts.min()) if len(val_counts) else 0,
            "val_counts_max": int(val_counts.max()) if len(val_counts) else 0,
            "val_counts_mean": float(val_counts.mean()) if len(val_counts) else 0.0,
        }

        kfold_summary = {
            "folds_r2_weighted_mean": float(np.mean([m["r2_weighted"] for m in fold_metrics])) if fold_metrics else None,
            "folds_r2_weighted_std": float(np.std([m["r2_weighted"] for m in fold_metrics])) if fold_metrics else None,
            "coverage": coverage,
        }

        if oof_pred is not None:
            ok = np.isfinite(oof_pred).all(axis=1)
            kfold_summary["oof_coverage_frac"] = float(ok.mean())
            if ok.any():
                kfold_summary["oof_r2_weighted"] = float(weighted_r2_logspace(y_true_full[ok], oof_pred[ok]))
            else:
                kfold_summary["oof_r2_weighted"] = None

        _save_json(log_dir / "summary.json", kfold_summary)

        logger.info("Saved fold metrics -> {}", log_dir / "fold_metrics.json")
        logger.info("Saved val predictions -> {}", log_dir / "val_predictions.csv")
        logger.info("Saved summary -> {}", log_dir / "summary.json")

        if kfold_summary.get("oof_r2_weighted", None) is not None:
            logger.info("OOF weighted R2 (log-space): {:.6f}", float(kfold_summary["oof_r2_weighted"]))
        else:
            logger.info(
                "Mean weighted R2 across folds (log-space): {:.6f}",
                float(kfold_summary["folds_r2_weighted_mean"])
                if kfold_summary.get("folds_r2_weighted_mean") is not None
                else float("nan"),
            )
        summary_cfg["kfold"]["artifacts"] = {
            "fold_metrics_json": str((log_dir / "fold_metrics.json").resolve()),
            "val_predictions_csv": str((log_dir / "val_predictions.csv").resolve()),
            "summary_json": str((log_dir / "summary.json").resolve()),
        }
    else:
        logger.info("kfold: skipped (kfold.enabled=false)")

    # -------------------------
    # 2) Train-all run
    # -------------------------
    train_all_summary: Optional[dict[str, Any]] = None
    if train_all_enabled:
        train_all_log_dir = (log_dir / "train_all").resolve()
        train_all_ckpt_dir = (ckpt_dir / "train_all").resolve()
        train_all_log_dir.mkdir(parents=True, exist_ok=True)
        train_all_ckpt_dir.mkdir(parents=True, exist_ok=True)

        # Dummy validation row (mirrors train.py semantics; not used for real evaluation).
        dummy = full_df.sample(n=1, random_state=int(seed))
        val_idx_arr = np.asarray(dummy.index.tolist(), dtype=np.int64)
        if val_idx_arr.size != 1:
            raise RuntimeError("Failed to construct a 1-row dummy validation split for train_all.")

        head_weights_path_train_all: Optional[Path] = None
        cache_path_train_all: Optional[Path] = None

        if image_features_mode == "dinov3_only":
            if X_all_global is None:
                raise RuntimeError("Internal error: X_all_global is None for dinov3_only mode.")
            X_all_train_all = X_all_global
        else:
            # Resolve a train_all head checkpoint (explicit path wins; otherwise best-effort auto-resolve).
            head_weights_raw = image_features_cfg.get("head_weights_path", None)
            if isinstance(head_weights_raw, str) and head_weights_raw.strip():
                head_weights_path_train_all = resolve_under_repo(repo_root, head_weights_raw)
            else:
                head_weights_path_train_all = resolve_train_all_head_weights_path(repo_root=repo_root, cfg_img=cfg_img)

            if head_weights_path_train_all is None or not head_weights_path_train_all.is_file():
                raise RuntimeError(
                    "train_all is enabled and image_features.mode=head_penultimate, but no train_all head weights could be resolved. "
                    "Set `image_features.head_weights_path` explicitly or ensure image training outputs exist under "
                    f"outputs/checkpoints/{str(cfg_img.get('version', '') or '').strip()}/train_all/head/."
                )

            # Optional train_all feature cache
            cache_path_raw = image_features_cfg.get("cache_path", None)
            if isinstance(cache_path_raw, str) and cache_path_raw.strip():
                base = resolve_under_repo(repo_root, cache_path_raw)
                if base.is_dir():
                    cache_path_train_all = (base / "head_penultimate_features.train_all.pt").resolve()
                else:
                    cache_path_train_all = base.with_name(f"{base.stem}.train_all{base.suffix}")

            X_all_train_all = extract_head_penultimate_features_as_tabular(
                repo_root=repo_root,
                cfg_img=cfg_img,
                df=full_df,
                image_features_cfg=image_features_cfg,
                head_weights_path=head_weights_path_train_all,
                cache_path=cache_path_train_all,
            )
            if X_all_train_all.ndim != 2 or X_all_train_all.shape[0] != len(full_df):
                raise RuntimeError(f"Invalid extracted feature matrix shape for train_all: {X_all_train_all.shape}")

            fold_feature_meta.append(
                {
                    "run": "train_all",
                    "mode": "head_penultimate",
                    "head_weights_path": str(head_weights_path_train_all),
                    "shape": list(X_all_train_all.shape),
                    "cache_path": str(cache_path_train_all) if cache_path_train_all is not None else None,
                }
            )

        X_train_all = X_all_train_all
        y_train_all = y_true_full
        X_val_dummy = X_all_train_all[val_idx_arr]
        y_val_dummy = y_true_full[val_idx_arr]

        # Optional: fine-tune TabPFN weights on the full dataset (train_all).
        model_path_for_train_all = str(tabpfn_params.model_path)
        finetune_summary_train_all: Optional[dict[str, Any]] = None
        if finetune_cfg.enabled:
            model_path_for_train_all, finetune_summary_train_all = finetune_tabpfn_regressor_on_fold(
                TabPFNRegressor=TabPFNRegressor,
                base_model_path=str(tabpfn_params.model_path),
                device=str(tabpfn_params.device),
                ignore_pretraining_limits=bool(tabpfn_params.ignore_pretraining_limits),
                seed=int(seed),
                fold_idx=0,
                finetune_cfg=finetune_cfg,
                eval_n_estimators=int(getattr(finetune_cfg, "eval_n_estimators", 8)),
                eval_fit_mode=str(tabpfn_params.fit_mode),
                eval_inference_precision=str(tabpfn_params.inference_precision),
                X_train=X_train_all,
                y_train_5d=y_train_all,
                X_val=X_val_dummy,
                y_val_5d=y_val_dummy,
                fold_log_dir=train_all_log_dir,
                fold_ckpt_dir=train_all_ckpt_dir,
                trainer_cfg_raw=dict(cfg.get("trainer", {}) or {}),
                tb_log_dir=(log_dir / "tensorboard").resolve() if tb_writer is not None else None,
                tb_run_name="train_all/finetune",
            )

        base = TabPFNRegressor(
            n_estimators=int(tabpfn_params.n_estimators),
            device=str(tabpfn_params.device),
            fit_mode=str(tabpfn_params.fit_mode),
            inference_precision=parse_tabpfn_inference_precision(tabpfn_params.inference_precision),
            random_state=int(seed),
            ignore_pretraining_limits=bool(tabpfn_params.ignore_pretraining_limits),
            model_path=str(model_path_for_train_all),
        )
        model = MultiOutputRegressor(base, n_jobs=int(tabpfn_params.n_jobs))
        model.fit(X_train_all, y_train_all)
        y_pred_dummy = model.predict(X_val_dummy)

        r2_dummy = float(weighted_r2_logspace(y_val_dummy, y_pred_dummy))
        if not np.isfinite(r2_dummy):
            r2_dummy = float("nan")

        dummy_row = full_df.iloc[int(val_idx_arr[0])]
        train_all_summary = {
            "feature_mode": str(image_features_mode),
            "head_weights_path": str(head_weights_path_train_all) if head_weights_path_train_all is not None else None,
            "num_train": int(len(full_df)),
            "num_val_dummy": int(val_idx_arr.size),
            "dummy_val_image_id": str(dummy_row.get("image_id", "")),
            "tabpfn_model_path_used": str(model_path_for_train_all),
            "val_weighted_r2_log_dummy": (None if not np.isfinite(r2_dummy) else float(r2_dummy)),
            "finetune": finetune_summary_train_all,
        }
        if tb_writer is not None and np.isfinite(r2_dummy):
            try:
                tb_writer.add_scalar("train_all/val/weighted_r2_log_dummy", float(r2_dummy), 0)
            except Exception:
                pass

        # Store dummy-val predictions (debug convenience)
        pred_df = full_df.iloc[val_idx_arr].reset_index(drop=True)[["image_id"]].copy()
        for j, name in enumerate(TARGETS_5D_ORDER):
            pred_df[f"pred_{name}"] = y_pred_dummy[:, j]
            pred_df[f"true_{name}"] = y_val_dummy[:, j]
        pred_df.to_csv(train_all_log_dir / "val_predictions.csv", index=False)

        _save_json(train_all_log_dir / "summary.json", train_all_summary)
        logger.info("Saved train_all summary -> {}", train_all_log_dir / "summary.json")

        summary_cfg["train_all"]["log_dir"] = str(train_all_log_dir)
        summary_cfg["train_all"]["ckpt_dir"] = str(train_all_ckpt_dir)
        summary_cfg["train_all"]["artifacts"] = {
            "val_predictions_csv": str((train_all_log_dir / "val_predictions.csv").resolve()),
            "summary_json": str((train_all_log_dir / "summary.json").resolve()),
        }
    else:
        logger.info("train_all: skipped (train_all.enabled=false)")

    # Persist the combined run summary_config (best-effort overwrite).
    _save_json(log_dir / "summary_config.json", summary_cfg)

    if tb_writer is not None:
        try:
            tb_writer.flush()
            tb_writer.close()
        except Exception:
            pass

    return log_dir


def main(argv: Optional[list[str]] = None) -> None:
    repo_root = resolve_repo_root()
    ensure_on_sys_path(repo_root)
    args = parse_args(repo_root, argv=argv)
    run_tabpfn_cv(repo_root=repo_root, args=args)


if __name__ == "__main__":
    main()


