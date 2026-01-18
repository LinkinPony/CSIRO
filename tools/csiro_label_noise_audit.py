from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd


def _try_import_loguru():
    try:
        from loguru import logger  # type: ignore

        return logger
    except Exception:
        return None


LOGGER = _try_import_loguru()


def _log(msg: str, *args: Any) -> None:
    if LOGGER is not None:
        try:
            LOGGER.info(msg, *args)
            return
        except Exception:
            pass
    # Fallback
    if args:
        try:
            msg = msg.format(*args)
        except Exception:
            pass
    print(msg, flush=True)


@dataclass(frozen=True)
class AuditConfig:
    # I/O
    repo_root: Path
    data_root: Path
    train_csv: Path
    out_dir: Path
    # task
    target: str
    log1p: bool
    # models
    kfold: int
    seed: int
    ridge_lambda: float
    # features
    resize_short_side: int
    knn_k: int
    # constraints
    constraint_tol: float


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="CSIRO label-noise audit: hard constraints + CV residuals + kNN consistency (no human review)."
    )
    p.add_argument("--repo-root", type=str, default=None, help="Repo root (default: parent of this script).")
    p.add_argument("--data-root", type=str, default="data", help="Data root under repo (default: data).")
    p.add_argument("--train-csv", type=str, default="train.csv", help="Train CSV under data_root (default: train.csv).")
    p.add_argument(
        "--out-dir",
        type=str,
        default="outputs/label_audit/csiro",
        help="Output directory under repo (default: outputs/label_audit/csiro).",
    )
    p.add_argument(
        "--target",
        type=str,
        default="Dry_Total_g",
        help="Which target to audit (per-image wide column name). Default: Dry_Total_g.",
    )
    p.add_argument(
        "--no-log1p",
        action="store_true",
        help="Disable log1p transform on targets before fitting baselines (not recommended).",
    )
    p.add_argument("--kfold", type=int, default=5, help="K for out-of-fold residuals (default: 5).")
    p.add_argument("--seed", type=int, default=42, help="RNG seed for fold split (default: 42).")
    p.add_argument(
        "--ridge-lambda",
        type=float,
        default=10.0,
        help="Ridge regularization strength (default: 10.0).",
    )
    p.add_argument(
        "--resize-short-side",
        type=int,
        default=320,
        help="Resize short side before engineered image feats (default: 320).",
    )
    p.add_argument("--knn-k", type=int, default=10, help="k for kNN label-consistency in image-feature space.")
    p.add_argument(
        "--constraint-tol",
        type=float,
        default=1e-3,
        help="Tolerance for strict label constraints in grams (default: 1e-3).",
    )
    return p.parse_args(args=argv)


def _safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _parse_date_series(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s.astype(str), errors="coerce")


def _sin_cos_dayofyear(dt: pd.Series) -> Tuple[pd.Series, pd.Series]:
    doy = dt.dt.dayofyear.astype("float64")
    angle = 2.0 * math.pi * (doy / 365.25)
    return np.sin(angle), np.cos(angle)


def _robust_z(x: np.ndarray, *, eps: float = 1e-12) -> np.ndarray:
    """
    Robust z-score using MAD, returns 0 for non-finite values.
    """
    x = np.asarray(x, dtype=np.float64)
    out = np.zeros_like(x, dtype=np.float64)
    mask = np.isfinite(x)
    if not np.any(mask):
        return out
    v = x[mask]
    med = float(np.median(v))
    mad = float(np.median(np.abs(v - med)))
    scale = max(mad * 1.4826, eps)
    out[mask] = (v - med) / scale
    out[~mask] = 0.0
    return out


def _standardize_cols(X: np.ndarray, *, eps: float = 1e-12) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Standardize columns to mean 0 / std 1.
    Returns (X_std, mean, std).
    """
    X = np.asarray(X, dtype=np.float64)
    mu = np.nanmean(X, axis=0)
    sd = np.nanstd(X, axis=0)
    sd = np.where(sd < eps, 1.0, sd)
    Xs = (X - mu) / sd
    # Replace NaNs/Infs with 0 after standardization
    Xs = np.nan_to_num(Xs, nan=0.0, posinf=0.0, neginf=0.0)
    return Xs, mu, sd


def _drop_constant_and_allnan_cols(df: pd.DataFrame, cols: list[str]) -> list[str]:
    keep: list[str] = []
    for c in cols:
        if c not in df.columns:
            continue
        s = df[c]
        if not pd.api.types.is_numeric_dtype(s):
            continue
        v = pd.to_numeric(s, errors="coerce")
        if v.notna().sum() == 0:
            continue
        # constant?
        if float(v.max() - v.min()) == 0.0:
            continue
        keep.append(c)
    return keep


def _fit_ridge(X: np.ndarray, y: np.ndarray, lam: float) -> np.ndarray:
    """
    Solve ridge regression in closed form:
      w = (X^T X + lam I)^(-1) X^T y
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    p = X.shape[1]
    XtX = X.T @ X
    XtX.flat[:: p + 1] += float(lam)
    Xty = X.T @ y
    return np.linalg.solve(XtX, Xty)


def _kfold_indices(n: int, *, k: int, seed: int) -> list[np.ndarray]:
    k_eff = max(2, int(k))
    rng = np.random.default_rng(int(seed))
    idx = np.arange(int(n))
    rng.shuffle(idx)
    return [a.astype(np.int64) for a in np.array_split(idx, k_eff)]


def _oof_ridge_predict(X: np.ndarray, y: np.ndarray, *, kfold: int, seed: int, lam: float) -> np.ndarray:
    n = int(X.shape[0])
    folds = _kfold_indices(n, k=kfold, seed=seed)
    pred = np.full((n,), np.nan, dtype=np.float64)
    all_idx = np.arange(n, dtype=np.int64)
    for val_idx in folds:
        tr_idx = np.setdiff1d(all_idx, val_idx, assume_unique=False)
        if tr_idx.size == 0 or val_idx.size == 0:
            continue
        w = _fit_ridge(X[tr_idx], y[tr_idx], lam)
        pred[val_idx] = X[val_idx] @ w
    return pred


def _build_meta_design(df: pd.DataFrame) -> Tuple[np.ndarray, list[str]]:
    df = df.copy()
    # numeric
    for col in ["Pre_GSHH_NDVI", "Height_Ave_cm"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    dt = _parse_date_series(df.get("Sampling_Date", pd.Series([None] * len(df))))
    sin_doy, cos_doy = _sin_cos_dayofyear(dt)
    df["date_doy_sin"] = sin_doy
    df["date_doy_cos"] = cos_doy
    # Light interactions
    eps = 1e-8
    if "Pre_GSHH_NDVI" in df.columns and "Height_Ave_cm" in df.columns:
        df["ndvi_x_height"] = df["Pre_GSHH_NDVI"] * df["Height_Ave_cm"]
        df["height_over_ndvi"] = df["Height_Ave_cm"] / (df["Pre_GSHH_NDVI"] + eps)

    num_cols = [c for c in ["Pre_GSHH_NDVI", "Height_Ave_cm", "date_doy_sin", "date_doy_cos", "ndvi_x_height", "height_over_ndvi"] if c in df.columns]
    # fill numeric
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
        med = float(df[c].median()) if df[c].notna().any() else 0.0
        df[c] = df[c].fillna(med)

    # categoricals
    cat_df = df[["State", "Species"]].copy() if set(["State", "Species"]).issubset(df.columns) else pd.DataFrame(index=df.index)
    if not cat_df.empty:
        cat_df = cat_df.fillna("UNK").astype(str)
        cat_oh = pd.get_dummies(cat_df, prefix=cat_df.columns.tolist(), drop_first=False)
    else:
        cat_oh = pd.DataFrame(index=df.index)

    X_df = pd.concat([df[num_cols], cat_oh], axis=1)
    cols = [str(c) for c in X_df.columns]
    X = X_df.to_numpy(dtype=np.float64)
    Xs, _, _ = _standardize_cols(X)
    # intercept
    Xs = np.concatenate([np.ones((len(df), 1), dtype=np.float64), Xs], axis=1)
    cols = ["bias"] + cols
    return Xs, cols


def _build_image_features(
    df: pd.DataFrame, *, data_root: Path, resize_short_side: int
) -> Tuple[pd.DataFrame, list[str]]:
    # Reuse the project's engineered image features (PIL+numpy only).
    from tools.csiro_feature_engineering import ImageFeatConfig, extract_image_features

    cfg_img = ImageFeatConfig(resize_short_side=int(resize_short_side))
    rel_paths = df["image_path"].astype(str).tolist()
    abs_paths = [(data_root / p).resolve() for p in rel_paths]

    feat_rows: list[dict[str, Any]] = []
    for i, (rel, ap) in enumerate(zip(rel_paths, abs_paths)):
        if (i % 50) == 0:
            _log("Extracting image feats: {}/{}", i, len(abs_paths))
        row: dict[str, Any] = {"image_path": str(rel)}
        row.update(extract_image_features(ap, cfg=cfg_img))
        feat_rows.append(row)

    df_img = pd.DataFrame(feat_rows)
    # Keep numeric cols only, drop obvious non-features.
    drop_cols = {"image_path", "image_abspath", "error"}
    num_cols = [c for c in df_img.columns if c not in drop_cols and pd.api.types.is_numeric_dtype(df_img[c])]
    num_cols = _drop_constant_and_allnan_cols(df_img, num_cols)
    # fill nans
    df_num = df_img[["image_path"] + num_cols].copy()
    for c in num_cols:
        v = pd.to_numeric(df_num[c], errors="coerce")
        med = float(v.median()) if v.notna().any() else 0.0
        df_num[c] = v.fillna(med)
    return df_num, num_cols


def _build_image_design(df_img: pd.DataFrame, feature_cols: list[str]) -> Tuple[np.ndarray, list[str]]:
    X = df_img[feature_cols].to_numpy(dtype=np.float64)
    Xs, _, _ = _standardize_cols(X)
    Xs = np.concatenate([np.ones((len(df_img), 1), dtype=np.float64), Xs], axis=1)
    cols = ["bias"] + [str(c) for c in feature_cols]
    return Xs, cols


def _knn_label_residuals(X: np.ndarray, y: np.ndarray, *, k: int) -> np.ndarray:
    """
    kNN label consistency residual: |y_i - median(y_{NN(i)})| in feature space.
    Complexity O(N^2) which is fine for CSIRO (N~357).
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n = int(X.shape[0])
    k_eff = max(1, min(int(k), n - 1))
    # Pairwise squared distances
    # d(i,j) = ||x_i - x_j||^2 = x_i^2 + x_j^2 - 2 x_i x_j
    x2 = np.sum(X * X, axis=1, keepdims=True)
    d2 = x2 + x2.T - 2.0 * (X @ X.T)
    # numerical noise
    d2 = np.maximum(d2, 0.0)
    # exclude self
    np.fill_diagonal(d2, np.inf)
    resid = np.zeros((n,), dtype=np.float64)
    for i in range(n):
        nn = np.argpartition(d2[i], k_eff)[:k_eff]
        med = float(np.median(y[nn])) if nn.size > 0 else float("nan")
        resid[i] = abs(float(y[i]) - med) if np.isfinite(med) else 0.0
    return resid


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv=argv)
    repo_root = Path(args.repo_root).expanduser() if args.repo_root else Path(__file__).resolve().parents[1]
    repo_root = repo_root.resolve()
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    # Local imports after sys.path fix
    from src.data.csiro_pivot import read_and_pivot_csiro_train_csv
    from src.metrics import TARGETS_5D_ORDER

    cfg = AuditConfig(
        repo_root=repo_root,
        data_root=(repo_root / str(args.data_root)).resolve(),
        train_csv=(repo_root / str(args.data_root) / str(args.train_csv)).resolve(),
        out_dir=(repo_root / str(args.out_dir)).resolve(),
        target=str(args.target),
        log1p=not bool(args.no_log1p),
        kfold=int(args.kfold),
        seed=int(args.seed),
        ridge_lambda=float(args.ridge_lambda),
        resize_short_side=int(args.resize_short_side),
        knn_k=int(args.knn_k),
        constraint_tol=float(args.constraint_tol),
    )

    _safe_mkdir(cfg.out_dir)
    if not cfg.train_csv.is_file():
        raise FileNotFoundError(f"train.csv not found: {cfg.train_csv}")

    _log("repo_root={}", str(cfg.repo_root))
    _log("data_root={}", str(cfg.data_root))
    _log("train_csv={}", str(cfg.train_csv))
    _log("out_dir={}", str(cfg.out_dir))

    # Pivot to per-image view (guarantees canonical 5D target columns exist)
    df = read_and_pivot_csiro_train_csv(
        data_root=str(cfg.data_root),
        train_csv=str(cfg.train_csv),
        target_order=list(TARGETS_5D_ORDER),
    )
    if cfg.target not in df.columns:
        raise KeyError(f"Target column '{cfg.target}' not found in pivoted df. Available: {sorted(df.columns.tolist())}")

    # Hard constraints (labels themselves)
    for t in TARGETS_5D_ORDER:
        df[t] = pd.to_numeric(df[t], errors="coerce")
    df["sum_components"] = df["Dry_Clover_g"] + df["Dry_Dead_g"] + df["Dry_Green_g"]
    df["res_total_minus_sum"] = df["Dry_Total_g"] - df["sum_components"]
    df["res_gdm_minus_total_minus_dead"] = df["GDM_g"] - (df["Dry_Total_g"] - df["Dry_Dead_g"])
    df["res_gdm_minus_greenclover"] = df["GDM_g"] - (df["Dry_Green_g"] + df["Dry_Clover_g"])
    tol = float(cfg.constraint_tol)
    df["flag_total_constraint"] = df["res_total_minus_sum"].abs() > tol
    df["flag_gdm_constraint"] = df["res_gdm_minus_total_minus_dead"].abs() > tol
    df["flag_any_constraint"] = df["flag_total_constraint"] | df["flag_gdm_constraint"]

    # Target vector
    y_raw = pd.to_numeric(df[cfg.target], errors="coerce").to_numpy(dtype=np.float64)
    if not np.isfinite(y_raw).all():
        bad = int((~np.isfinite(y_raw)).sum())
        raise ValueError(f"Non-finite values in target '{cfg.target}': {bad} rows")
    y = np.log1p(np.maximum(y_raw, 0.0)) if cfg.log1p else y_raw.copy()

    # Build baselines
    X_meta, _ = _build_meta_design(df)
    pred_meta = _oof_ridge_predict(X_meta, y, kfold=cfg.kfold, seed=cfg.seed, lam=cfg.ridge_lambda)

    df_img, img_cols = _build_image_features(df, data_root=cfg.data_root, resize_short_side=cfg.resize_short_side)
    X_img, _ = _build_image_design(df_img, img_cols)
    pred_img = _oof_ridge_predict(X_img, y, kfold=cfg.kfold, seed=cfg.seed + 1, lam=cfg.ridge_lambda)

    # Combined
    X_comb = np.concatenate([X_meta, X_img[:, 1:]], axis=1)  # keep one intercept
    pred_comb = _oof_ridge_predict(X_comb, y, kfold=cfg.kfold, seed=cfg.seed + 2, lam=cfg.ridge_lambda)

    # Residuals (in y-space)
    res_meta = np.abs(pred_meta - y)
    res_img = np.abs(pred_img - y)
    res_comb = np.abs(pred_comb - y)
    pred_cons = 0.5 * (pred_meta + pred_img)
    res_cons = np.abs(pred_cons - y)
    pred_disagree = np.abs(pred_meta - pred_img)

    # kNN consistency (use image features, *not* the intercept)
    knn_res = _knn_label_residuals(X_img[:, 1:], y, k=cfg.knn_k)

    # Robust z (bigger => more suspicious)
    z_cons = np.maximum(0.0, _robust_z(res_cons))
    z_knn = np.maximum(0.0, _robust_z(knn_res))
    z_disagree = np.maximum(0.0, _robust_z(pred_disagree))

    # High-confidence noise: models agree (low disagreement) but label is far (high residual).
    # Score is positive-only; hard constraints get a big bump.
    score = (2.0 * z_cons) + (1.0 * z_knn) - (0.5 * z_disagree)
    score = np.maximum(score, 0.0)
    score = score + df["flag_any_constraint"].to_numpy(dtype=np.float64) * 10.0

    out = df[["image_id", "image_path", "Sampling_Date", "State", "Species", "Pre_GSHH_NDVI", "Height_Ave_cm"]].copy()
    out["y_raw"] = y_raw
    out["y_used"] = y
    out["pred_meta"] = pred_meta
    out["pred_img"] = pred_img
    out["pred_comb"] = pred_comb
    out["res_meta"] = res_meta
    out["res_img"] = res_img
    out["res_comb"] = res_comb
    out["pred_cons"] = pred_cons
    out["res_cons"] = res_cons
    out["pred_disagree"] = pred_disagree
    out["knn_res"] = knn_res
    out["z_cons"] = z_cons
    out["z_knn"] = z_knn
    out["z_disagree"] = z_disagree
    out["score"] = score
    out["flag_any_constraint"] = df["flag_any_constraint"].astype(bool).to_numpy()
    out["flag_total_constraint"] = df["flag_total_constraint"].astype(bool).to_numpy()
    out["flag_gdm_constraint"] = df["flag_gdm_constraint"].astype(bool).to_numpy()
    out["res_total_minus_sum"] = df["res_total_minus_sum"].to_numpy(dtype=np.float64)
    out["res_gdm_minus_total_minus_dead"] = df["res_gdm_minus_total_minus_dead"].to_numpy(dtype=np.float64)

    out = out.sort_values(["score", "res_cons"], ascending=[False, False]).reset_index(drop=True)

    out_path = (cfg.out_dir / "suspects.csv").resolve()
    out.to_csv(out_path, index=False)
    _log("Wrote {}", str(out_path))

    # A small summary for quick inspection
    n = int(len(out))
    n_hard = int(out["flag_any_constraint"].sum())
    _log("N images: {}", n)
    _log("Hard-constraint violations: {}", n_hard)
    _log("Top-10 suspects (image_id, score, y_raw, image_path):")
    for i in range(min(10, n)):
        r = out.iloc[i]
        _log(
            "  {}  score={:.3f}  y={:.3f}  path={}",
            str(r["image_id"]),
            float(r["score"]),
            float(r["y_raw"]),
            str(r["image_path"]),
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

