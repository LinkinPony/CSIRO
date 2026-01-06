from __future__ import annotations

import argparse
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger
from PIL import Image


@dataclass(frozen=True)
class ImageFeatConfig:
    """
    Settings for lightweight image feature extraction (PIL + numpy only).
    """

    resize_short_side: int = 320
    max_pixels: int = 320 * 320  # fail-safe; used if resize_short_side <= 0
    rgb_hist_bins: int = 32


def _safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _parse_date_series(s: pd.Series) -> pd.Series:
    # train.csv uses "YYYY/M/D" (no zero padding); coerce invalid to NaT.
    # pandas handles that with format inference; we keep it explicit-ish.
    return pd.to_datetime(s.astype(str), errors="coerce")


def _sin_cos_dayofyear(dt: pd.Series) -> tuple[pd.Series, pd.Series]:
    doy = dt.dt.dayofyear.astype("float64")
    # 365.25 for leap-years smoothing; keep NaN where dt is NaT.
    angle = 2.0 * math.pi * (doy / 365.25)
    return np.sin(angle), np.cos(angle)


def _rgb_to_gray_float(rgb_u8: np.ndarray) -> np.ndarray:
    """
    rgb_u8: uint8 array (..., 3) in RGB order.
    returns float32 in [0,255] approximately.
    """
    rgb = rgb_u8.astype(np.float32)
    return (0.2989 * rgb[..., 0]) + (0.5870 * rgb[..., 1]) + (0.1140 * rgb[..., 2])


def _conv2d_same(gray: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    Very small, dependency-free 2D convolution (same padding), for feature extraction only.
    gray: (H,W) float32
    kernel: (kh,kw) float32
    """
    if gray.ndim != 2:
        raise ValueError(f"Expected gray 2D, got shape={gray.shape}")
    if kernel.ndim != 2:
        raise ValueError(f"Expected kernel 2D, got shape={kernel.shape}")
    kh, kw = kernel.shape
    pad_h = kh // 2
    pad_w = kw // 2
    g = np.pad(gray, ((pad_h, pad_h), (pad_w, pad_w)), mode="edge")
    out = np.zeros_like(gray, dtype=np.float32)
    # Kernel is small (3x3), brute force is fine.
    for i in range(kh):
        for j in range(kw):
            out += kernel[i, j] * g[i : i + gray.shape[0], j : j + gray.shape[1]]
    return out


def _shannon_entropy_u8(gray_u8: np.ndarray) -> float:
    if gray_u8.size <= 0:
        return float("nan")
    hist = np.bincount(gray_u8.reshape(-1), minlength=256).astype(np.float64)
    p = hist / max(float(hist.sum()), 1.0)
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p)))


def _resize_for_feats(img: Image.Image, cfg: ImageFeatConfig) -> Image.Image:
    w, h = img.size
    if w <= 0 or h <= 0:
        return img
    if cfg.resize_short_side and cfg.resize_short_side > 0:
        short = min(w, h)
        if short <= cfg.resize_short_side:
            return img
        scale = float(cfg.resize_short_side) / float(short)
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        return img.resize((new_w, new_h), resample=Image.BILINEAR)

    # Fallback: cap total pixels
    max_pix = int(cfg.max_pixels) if cfg.max_pixels else 0
    if max_pix > 0 and (w * h) > max_pix:
        scale = math.sqrt(float(max_pix) / float(w * h))
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        return img.resize((new_w, new_h), resample=Image.BILINEAR)
    return img


def extract_image_features(
    img_path: Path,
    *,
    cfg: ImageFeatConfig,
) -> Dict[str, Any]:
    """
    Extract simple but surprisingly strong engineered features from an RGB image:
    - geometry + file size
    - RGB/HSV summary stats
    - visible vegetation indices (ExG, VARI, NGRDI)
    - texture proxies (Laplacian variance, gradient magnitude, entropy)
    """
    feats: Dict[str, Any] = {"image_abspath": str(img_path)}
    try:
        st = img_path.stat()
        feats["file_size_bytes"] = int(st.st_size)
    except Exception:
        feats["file_size_bytes"] = np.nan

    try:
        with Image.open(img_path) as im0:
            # Ensure RGB
            im = im0.convert("RGB")
            w, h = im.size
            feats["img_w"] = int(w)
            feats["img_h"] = int(h)
            feats["img_aspect"] = float(w) / float(h) if h else np.nan
            feats["img_pixels"] = int(w * h)
            feats["img_megapixels"] = float(w * h) / 1e6

            # EXIF (best-effort)
            try:
                exif = im0.getexif()
                feats["exif_orientation"] = int(exif.get(274)) if exif and exif.get(274) is not None else np.nan
            except Exception:
                feats["exif_orientation"] = np.nan

            im_small = _resize_for_feats(im, cfg)
            arr = np.asarray(im_small, dtype=np.uint8)  # (H,W,3)
    except Exception as e:
        feats["error"] = str(e)
        return feats

    # RGB stats
    rgb_img = arr.astype(np.float32)
    rgb = rgb_img.reshape(-1, 3)
    for c, name in enumerate(["r", "g", "b"]):
        v = rgb[:, c]
        feats[f"{name}_mean"] = float(np.mean(v))
        feats[f"{name}_std"] = float(np.std(v))
        feats[f"{name}_p05"] = float(np.percentile(v, 5))
        feats[f"{name}_p50"] = float(np.percentile(v, 50))
        feats[f"{name}_p95"] = float(np.percentile(v, 95))

    # Gray / brightness / contrast
    gray_f = _rgb_to_gray_float(arr).astype(np.float32)
    gray = gray_f.reshape(-1)
    feats["gray_mean"] = float(np.mean(gray))
    feats["gray_std"] = float(np.std(gray))
    feats["gray_p05"] = float(np.percentile(gray, 5))
    feats["gray_p50"] = float(np.percentile(gray, 50))
    feats["gray_p95"] = float(np.percentile(gray, 95))

    # Entropy (0-8-ish)
    gray_u8 = np.clip(gray_f, 0, 255).astype(np.uint8)
    feats["gray_entropy"] = _shannon_entropy_u8(gray_u8)

    # HSV via PIL (cheap + dependency-free)
    hsv_img_u8 = np.asarray(Image.fromarray(arr, mode="RGB").convert("HSV"), dtype=np.uint8)  # (H,W,3)
    hsv = hsv_img_u8.reshape(-1, 3).astype(np.float32)
    h = hsv[:, 0]  # 0..255 maps to 0..360deg
    s = hsv[:, 1]
    v = hsv[:, 2]
    feats["h_mean"] = float(np.mean(h))
    feats["s_mean"] = float(np.mean(s))
    feats["v_mean"] = float(np.mean(v))
    feats["s_std"] = float(np.std(s))
    feats["v_std"] = float(np.std(v))

    # Hue fractions: rough green / brown dominance proxies
    # H in [0,255] -> degrees ~ H*360/255.
    # Green-ish ~ [60, 170] deg => [43, 120] in 0..255
    green_mask = (h >= (60.0 / 360.0 * 255.0)) & (h <= (170.0 / 360.0 * 255.0)) & (s >= 40.0) & (v >= 40.0)
    # Brown/yellow-ish ~ [20, 60] deg => [14, 43]
    brown_mask = (h >= (20.0 / 360.0 * 255.0)) & (h <= (60.0 / 360.0 * 255.0)) & (s >= 20.0) & (v >= 20.0)
    feats["hue_green_frac"] = float(np.mean(green_mask.astype(np.float32)))
    feats["hue_brown_frac"] = float(np.mean(brown_mask.astype(np.float32)))
    feats["sat_high_frac"] = float(np.mean((s >= 120.0).astype(np.float32)))

    # Vegetation indices from RGB (visible-only proxies)
    eps = 1e-6
    R = rgb[:, 0]
    G = rgb[:, 1]
    B = rgb[:, 2]
    exg = 2.0 * G - R - B
    vari = (G - R) / (G + R - B + eps)
    ngrdi = (G - R) / (G + R + eps)
    for name, vec in [
        ("exg", exg),
        ("vari", vari),
        ("ngrdi", ngrdi),
        ("g_over_r", (G + eps) / (R + eps)),
        ("g_over_b", (G + eps) / (B + eps)),
    ]:
        feats[f"{name}_mean"] = float(np.mean(vec))
        feats[f"{name}_std"] = float(np.std(vec))
        feats[f"{name}_p05"] = float(np.percentile(vec, 5))
        feats[f"{name}_p50"] = float(np.percentile(vec, 50))
        feats[f"{name}_p95"] = float(np.percentile(vec, 95))

    # Texture proxies (blur / detail):
    # Laplacian variance: higher => sharper / more texture.
    lap_k = np.asarray([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)
    lap = _conv2d_same(gray_f, lap_k)
    feats["lap_var"] = float(np.var(lap))

    # Sobel gradient magnitude (mean + edge density)
    sx = np.asarray([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
    sy = np.asarray([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)
    gx = _conv2d_same(gray_f, sx)
    gy = _conv2d_same(gray_f, sy)
    gm = np.sqrt(gx * gx + gy * gy)
    feats["gradmag_mean"] = float(np.mean(gm))
    feats["gradmag_p90"] = float(np.percentile(gm, 90))
    # Edge density with fixed thresholds (more informative than percentile-based thresholds).
    feats["edge_frac_gt_20"] = float(np.mean((gm >= 20.0).astype(np.float32)))
    feats["edge_frac_gt_40"] = float(np.mean((gm >= 40.0).astype(np.float32)))

    # ----------------------------------------------------------
    # Spatial ("grid") features: 2x2 regions on the resized image
    # ----------------------------------------------------------
    H, W = arr.shape[0], arr.shape[1]
    h2 = max(1, H // 2)
    w2 = max(1, W // 2)

    # Per-pixel index images for slicing
    R_img = rgb_img[..., 0]
    G_img = rgb_img[..., 1]
    B_img = rgb_img[..., 2]
    exg_img = 2.0 * G_img - R_img - B_img
    ngrdi_img = (G_img - R_img) / (G_img + R_img + eps)

    # Hue masks precomputed for the whole image (then sliced)
    h_img = hsv_img_u8[..., 0].astype(np.float32)
    s_img = hsv_img_u8[..., 1].astype(np.float32)
    v_img = hsv_img_u8[..., 2].astype(np.float32)
    green_mask_img = (
        (h_img >= (60.0 / 360.0 * 255.0))
        & (h_img <= (170.0 / 360.0 * 255.0))
        & (s_img >= 40.0)
        & (v_img >= 40.0)
    )
    brown_mask_img = (
        (h_img >= (20.0 / 360.0 * 255.0))
        & (h_img <= (60.0 / 360.0 * 255.0))
        & (s_img >= 20.0)
        & (v_img >= 20.0)
    )

    def _region(prefix: str, sl_h: slice, sl_w: slice) -> None:
        gfrac = float(np.mean(green_mask_img[sl_h, sl_w].astype(np.float32)))
        bfrac = float(np.mean(brown_mask_img[sl_h, sl_w].astype(np.float32)))
        feats[f"{prefix}_green_frac"] = gfrac
        feats[f"{prefix}_brown_frac"] = bfrac
        feats[f"{prefix}_exg_mean"] = float(np.mean(exg_img[sl_h, sl_w]))
        feats[f"{prefix}_ngrdi_mean"] = float(np.mean(ngrdi_img[sl_h, sl_w]))
        feats[f"{prefix}_gray_mean"] = float(np.mean(gray_f[sl_h, sl_w]))
        feats[f"{prefix}_lap_var"] = float(np.var(lap[sl_h, sl_w]))

    _region("tl", slice(0, h2), slice(0, w2))
    _region("tr", slice(0, h2), slice(w2, W))
    _region("bl", slice(h2, H), slice(0, w2))
    _region("br", slice(h2, H), slice(w2, W))

    feats["green_frac_bottom_minus_top"] = float(0.5 * (feats["bl_green_frac"] + feats["br_green_frac"]) - 0.5 * (feats["tl_green_frac"] + feats["tr_green_frac"]))
    feats["brown_frac_bottom_minus_top"] = float(0.5 * (feats["bl_brown_frac"] + feats["br_brown_frac"]) - 0.5 * (feats["tl_brown_frac"] + feats["tr_brown_frac"]))

    return feats


def _corr_table(
    df: pd.DataFrame,
    *,
    y_col: str,
    feature_cols: list[str],
    log1p_y: bool,
) -> pd.DataFrame:
    y = df[y_col].astype(float)
    y_eff = np.log1p(np.maximum(y, 0.0)) if log1p_y else y
    out_rows = []
    for c in feature_cols:
        x = df[c]
        if not pd.api.types.is_numeric_dtype(x):
            continue
        x_eff = x.astype(float)
        # Drop rows where either is nan
        mask = np.isfinite(x_eff.to_numpy()) & np.isfinite(y_eff.to_numpy())
        if int(mask.sum()) < 10:
            continue
        xv = x_eff.to_numpy()[mask]
        yv = y_eff.to_numpy()[mask]
        # Pearson
        if np.std(xv) <= 1e-12 or np.std(yv) <= 1e-12:
            corr = np.nan
        else:
            corr = float(np.corrcoef(xv, yv)[0, 1])
        out_rows.append({"feature": c, "corr": corr, "abs_corr": (abs(corr) if np.isfinite(corr) else np.nan)})
    out = pd.DataFrame(out_rows).sort_values(["abs_corr", "feature"], ascending=[False, True]).reset_index(drop=True)
    return out


def _escape_md(s: str) -> str:
    return str(s).replace("|", "\\|").replace("\n", " ").replace("\r", " ")


def _format_md_cell(v: Any) -> str:
    try:
        if v is None:
            return ""
        if isinstance(v, float) and (not np.isfinite(v)):
            return ""
        # pandas uses numpy scalars frequently
        if isinstance(v, (np.floating,)):
            fv = float(v)
            return "" if (not np.isfinite(fv)) else f"{fv:.6g}"
        if isinstance(v, (np.integer,)):
            return str(int(v))
        if isinstance(v, float):
            return f"{float(v):.6g}"
        if isinstance(v, (int,)):
            return str(int(v))
        if isinstance(v, (str,)):
            return _escape_md(v)
        return _escape_md(str(v))
    except Exception:
        return _escape_md(str(v))


def df_to_markdown_table(df: pd.DataFrame, *, max_rows: int = 50) -> str:
    """
    Lightweight markdown table generator to avoid the optional `tabulate` dependency.
    """
    if df is None:
        return ""
    if len(df) > int(max_rows):
        df_eff = df.head(int(max_rows)).copy()
    else:
        df_eff = df.copy()
    cols = [str(c) for c in df_eff.columns]
    header = "| " + " | ".join(_escape_md(c) for c in cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    lines = [header, sep]
    for _, row in df_eff.iterrows():
        cells = [_format_md_cell(row.get(c, "")) for c in cols]
        lines.append("| " + " | ".join(cells) + " |")
    if len(df) > len(df_eff):
        lines.append(f"\n> 注：仅显示前 {len(df_eff)} 行（总计 {len(df)} 行）\n")
    return "\n".join(lines)


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CSIRO train.csv 深度特征工程 + 数据报告 (image+metadata).")
    p.add_argument(
        "--repo-root",
        type=str,
        default=None,
        help="Repo root (defaults to parent of this script).",
    )
    p.add_argument(
        "--data-root",
        type=str,
        default="data",
        help="Data root under repo (default: data).",
    )
    p.add_argument(
        "--train-csv",
        type=str,
        default="train.csv",
        help="Train CSV path under data_root (default: train.csv).",
    )
    p.add_argument(
        "--out-dir",
        type=str,
        default="outputs/feature_engineering/csiro_train",
        help="Output directory under repo (default: outputs/feature_engineering/csiro_train).",
    )
    p.add_argument(
        "--max-images",
        type=int,
        default=0,
        help="Debug: limit number of images to process (0=all).",
    )
    p.add_argument(
        "--resize-short-side",
        type=int,
        default=320,
        help="Resize short side before computing pixel features (default: 320).",
    )
    return p.parse_args(args=argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv=argv)

    repo_root = Path(args.repo_root).expanduser() if args.repo_root else Path(__file__).resolve().parents[1]
    if not repo_root.is_absolute():
        repo_root = repo_root.resolve()
    # Make `import src.*` work when running as a script.
    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)

    # Local imports after sys.path fix (repo is not installed as a package).
    from src.data.csiro_pivot import read_and_pivot_csiro_train_csv
    from src.metrics import TARGETS_5D_ORDER

    data_root = (repo_root / str(args.data_root)).resolve()
    train_csv_path = (data_root / str(args.train_csv)).resolve()
    test_csv_path = (data_root / "test.csv").resolve()
    out_dir = (repo_root / str(args.out_dir)).resolve()
    plots_dir = (out_dir / "plots").resolve()
    _safe_mkdir(out_dir)
    _safe_mkdir(plots_dir)

    if not train_csv_path.is_file():
        raise FileNotFoundError(f"train.csv not found: {train_csv_path}")

    logger.info("repo_root={}", str(repo_root))
    logger.info("train_csv={}", str(train_csv_path))
    logger.info("out_dir={}", str(out_dir))

    # -----------------------
    # 1) Load raw (long) CSV
    # -----------------------
    df_long = pd.read_csv(train_csv_path)
    required = {
        "sample_id",
        "image_path",
        "Sampling_Date",
        "State",
        "Species",
        "Pre_GSHH_NDVI",
        "Height_Ave_cm",
        "target_name",
        "target",
    }
    if not required.issubset(df_long.columns):
        missing = sorted(list(required - set(df_long.columns)))
        raise KeyError(f"train.csv missing columns: {missing}")

    df_long = df_long.copy()
    df_long["image_id"] = df_long["sample_id"].astype(str).str.split("__", n=1, expand=True)[0]

    n_rows = int(len(df_long))
    n_images = int(df_long["image_id"].nunique())
    target_names = sorted(df_long["target_name"].astype(str).unique().tolist())

    # Basic sanity: per-image consistency of metadata across target rows
    meta_cols = ["image_path", "Sampling_Date", "State", "Species", "Pre_GSHH_NDVI", "Height_Ave_cm"]
    meta_inconsistent = {}
    for c in meta_cols:
        nun = df_long.groupby("image_id")[c].nunique(dropna=False)
        bad = nun[nun > 1]
        if len(bad) > 0:
            meta_inconsistent[c] = int(len(bad))

    # Target table (long)
    tn_stats = []
    for tn, g in df_long.groupby("target_name"):
        y = g["target"].astype(float)
        tn_stats.append(
            {
                "target_name": str(tn),
                "n": int(y.count()),
                "zero_frac": float((y == 0).mean()),
                "min": float(np.nanmin(y.to_numpy())),
                "p50": float(np.nanmedian(y.to_numpy())),
                "mean": float(np.nanmean(y.to_numpy())),
                "p95": float(np.nanpercentile(y.to_numpy(), 95)),
                "max": float(np.nanmax(y.to_numpy())),
            }
        )
    df_tn_stats = pd.DataFrame(tn_stats).sort_values("target_name").reset_index(drop=True)

    # Best-effort: inspect test.csv schema (important for deciding which engineered features are usable at inference).
    test_cols: list[str] = []
    if test_csv_path.is_file():
        try:
            df_test_head = pd.read_csv(test_csv_path, nrows=5)
            test_cols = [str(c) for c in df_test_head.columns]
        except Exception:
            test_cols = []

    # -----------------------------------------
    # 2) Pivot to image-level (wide) dataframe
    # -----------------------------------------
    pivot_all = df_long.pivot_table(index="image_id", columns="target_name", values="target", aggfunc="first")
    meta_agg = df_long.groupby("image_id")[meta_cols].first()
    df_wide_all = pivot_all.join(meta_agg, how="left").reset_index(drop=False)

    # Project-canonical pivot (drop rows missing the requested targets)
    df_train = read_and_pivot_csiro_train_csv(
        data_root=str(data_root),
        train_csv=str(train_csv_path),
        target_order=list(TARGETS_5D_ORDER),
    )

    # -----------------------------------------
    # 3) Constraint checks (label consistency)
    # -----------------------------------------
    df_c = df_train.copy()
    eps = 1e-8
    for col in TARGETS_5D_ORDER:
        if col not in df_c.columns:
            df_c[col] = np.nan
        df_c[col] = df_c[col].astype(float)

    df_c["sum_components"] = df_c["Dry_Clover_g"] + df_c["Dry_Dead_g"] + df_c["Dry_Green_g"]
    df_c["res_total_minus_sum"] = df_c["Dry_Total_g"] - df_c["sum_components"]
    df_c["sum_green_clover"] = df_c["Dry_Green_g"] + df_c["Dry_Clover_g"]
    df_c["res_gdm_minus_greenclover"] = df_c["GDM_g"] - df_c["sum_green_clover"]

    def _res_stats(x: pd.Series) -> dict:
        v = x.to_numpy(dtype=np.float64)
        v = v[np.isfinite(v)]
        if v.size == 0:
            return {"n": 0}
        return {
            "n": int(v.size),
            "mean": float(np.mean(v)),
            "std": float(np.std(v)),
            "p01": float(np.percentile(v, 1)),
            "p50": float(np.percentile(v, 50)),
            "p99": float(np.percentile(v, 99)),
            "min": float(np.min(v)),
            "max": float(np.max(v)),
            "abs_p95": float(np.percentile(np.abs(v), 95)),
            "abs_max": float(np.max(np.abs(v))),
        }

    total_res = _res_stats(df_c["res_total_minus_sum"])
    gdm_res = _res_stats(df_c["res_gdm_minus_greenclover"])

    # -----------------------------------------
    # 4) Date feature engineering (metadata)
    # -----------------------------------------
    dt = _parse_date_series(df_train["Sampling_Date"])
    df_train = df_train.copy()
    df_train["date_parsed"] = dt
    df_train["date_year"] = dt.dt.year
    df_train["date_month"] = dt.dt.month
    df_train["date_day"] = dt.dt.day
    df_train["date_dayofyear"] = dt.dt.dayofyear
    df_train["date_weekofyear"] = dt.dt.isocalendar().week.astype("float64")
    sin_doy, cos_doy = _sin_cos_dayofyear(dt)
    df_train["date_doy_sin"] = sin_doy
    df_train["date_doy_cos"] = cos_doy

    # Simple numeric interactions
    df_train["ndvi_x_height"] = df_train["Pre_GSHH_NDVI"].astype(float) * df_train["Height_Ave_cm"].astype(float)
    df_train["height_over_ndvi"] = df_train["Height_Ave_cm"].astype(float) / (df_train["Pre_GSHH_NDVI"].astype(float) + eps)

    # -----------------------------------------
    # 5) Image feature extraction
    # -----------------------------------------
    cfg_img = ImageFeatConfig(resize_short_side=int(args.resize_short_side))
    img_paths = df_train["image_path"].astype(str).tolist()
    if args.max_images and int(args.max_images) > 0:
        img_paths = img_paths[: int(args.max_images)]
        df_train = df_train.iloc[: int(args.max_images)].reset_index(drop=True)

    abs_paths = [(data_root / p).resolve() for p in img_paths]
    logger.info("Extracting image engineered features: N={}", len(abs_paths))

    feat_rows: list[dict[str, Any]] = []
    for i, (rel, ap) in enumerate(zip(img_paths, abs_paths)):
        if (i % 50) == 0:
            logger.info("... {}/{} images", i, len(abs_paths))
        row = {"image_path": str(rel), "image_id": str(df_train.loc[i, "image_id"])}
        row.update(extract_image_features(ap, cfg=cfg_img))
        feat_rows.append(row)

    df_img = pd.DataFrame(feat_rows)
    # Track which columns are "image-only" engineered features (usable at inference),
    # versus train-only metadata (not present in test.csv in this repo).
    img_feature_cols_all = [
        c
        for c in df_img.columns
        if c not in {"image_path", "image_id", "image_abspath", "error"} and c.strip()
    ]
    df_merged = df_train.merge(df_img.drop(columns=["image_id"], errors="ignore"), on="image_path", how="left")

    # -----------------------------------------
    # 6) Correlation ranking (log1p targets)
    # -----------------------------------------
    # Numeric columns only; exclude targets themselves from feature list.
    numeric_cols = [c for c in df_merged.columns if pd.api.types.is_numeric_dtype(df_merged[c])]
    feature_cols = [c for c in numeric_cols if c not in set(TARGETS_5D_ORDER)]

    # Split correlation analysis into:
    # - all numeric features
    # - image-only features (usable at inference)
    # - metadata-only numeric features (train-only in this repo's test.csv)
    meta_numeric_cols = [
        "Pre_GSHH_NDVI",
        "Height_Ave_cm",
        "date_year",
        "date_month",
        "date_day",
        "date_dayofyear",
        "date_weekofyear",
        "date_doy_sin",
        "date_doy_cos",
        "ndvi_x_height",
        "height_over_ndvi",
    ]
    meta_numeric_cols = [c for c in meta_numeric_cols if c in df_merged.columns and pd.api.types.is_numeric_dtype(df_merged[c])]
    img_numeric_cols = [c for c in img_feature_cols_all if c in df_merged.columns and pd.api.types.is_numeric_dtype(df_merged[c])]

    corr_tables_all: dict[str, pd.DataFrame] = {}
    corr_tables_image: dict[str, pd.DataFrame] = {}
    corr_tables_meta: dict[str, pd.DataFrame] = {}
    for y in TARGETS_5D_ORDER:
        if y not in df_merged.columns:
            continue
        corr_tables_all[y] = _corr_table(df_merged, y_col=y, feature_cols=feature_cols, log1p_y=True)
        corr_tables_image[y] = _corr_table(df_merged, y_col=y, feature_cols=img_numeric_cols, log1p_y=True)
        corr_tables_meta[y] = _corr_table(df_merged, y_col=y, feature_cols=meta_numeric_cols, log1p_y=True)

    # -----------------------------------------
    # 7) Write artifacts
    # -----------------------------------------
    # 7.1) Save engineered feature table
    # Prefer parquet, fallback to csv.gz
    feat_out_base = out_dir / "train_engineered_features"
    saved_paths = []
    try:
        df_merged.to_parquet(str(feat_out_base.with_suffix(".parquet")), index=False)
        saved_paths.append(str(feat_out_base.with_suffix(".parquet")))
    except Exception as e:
        logger.warning("to_parquet failed (pyarrow likely missing): {}", str(e))
    try:
        df_merged.to_csv(str(feat_out_base.with_suffix(".csv.gz")), index=False, compression="gzip")
        saved_paths.append(str(feat_out_base.with_suffix(".csv.gz")))
    except Exception as e:
        logger.warning("to_csv(.gz) failed: {}", str(e))

    # 7.2) Save correlation tables
    corr_dir = (out_dir / "correlations").resolve()
    _safe_mkdir(corr_dir)
    for y in TARGETS_5D_ORDER:
        tab_all = corr_tables_all.get(y, None)
        tab_img = corr_tables_image.get(y, None)
        tab_meta = corr_tables_meta.get(y, None)
        if isinstance(tab_all, pd.DataFrame):
            tab_all.to_csv(str(corr_dir / f"corr_log1p__{y}.csv"), index=False)  # backward-compatible name
            tab_all.to_csv(str(corr_dir / f"corr_log1p__{y}__all.csv"), index=False)
        if isinstance(tab_img, pd.DataFrame):
            tab_img.to_csv(str(corr_dir / f"corr_log1p__{y}__image_only.csv"), index=False)
        if isinstance(tab_meta, pd.DataFrame):
            tab_meta.to_csv(str(corr_dir / f"corr_log1p__{y}__meta_only.csv"), index=False)

    # 7.3) Write markdown report (Chinese)
    report_path = (out_dir / "report.md").resolve()
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# CSIRO train.csv 特征工程报告（数据本身）\n\n")
        f.write(f"- repo_root: `{repo_root}`\n")
        f.write(f"- train_csv: `{train_csv_path}`\n")
        f.write(f"- 输出目录: `{out_dir}`\n\n")

        f.write("## 1) 数据概览（long format）\n\n")
        f.write(f"- 行数（含不同 target_name 的重复行）: **{n_rows}**\n")
        f.write(f"- 唯一图片数（image_id）: **{n_images}**\n")
        f.write(f"- target_name 种类: **{len(target_names)}**  -> `{target_names}`\n")
        if meta_inconsistent:
            f.write(f"- ⚠️ 同一 image_id 下元数据不一致（列->受影响图片数）: `{meta_inconsistent}`\n")
        else:
            f.write("- ✅ 同一 image_id 下元数据（日期/州/物种/NDVI/高度/路径）一致\n")
        f.write("\n")

        f.write("### 每个 target_name 的分布摘要\n\n")
        f.write(df_to_markdown_table(df_tn_stats))
        f.write("\n\n")

        f.write("## 1.5) 重要提示：哪些特征在测试集可用？\n\n")
        if test_cols:
            f.write(f"- 本 repo 的 `data/test.csv` 列: `{test_cols}`\n")
            f.write(
                "- 这意味着 `Sampling_Date/State/Species/Pre_GSHH_NDVI/Height_Ave_cm` **在 test.csv 中不存在**，"
                "因此这些字段（及其派生特征）不能直接作为最终提交模型的输入特征。\n"
            )
            f.write("- 但它们仍然非常有用：用于分组 CV（避免泄漏/过拟合）、训练分析、或训练时的辅助任务/重加权。\n\n")
        else:
            f.write("- 未能读取 `data/test.csv`，无法确认测试集特征可用性。\n\n")

        f.write("## 2) 数据概览（pivot to per-image）\n\n")
        f.write(f"- pivot 后样本数（用于训练，要求 5D targets 全部存在）: **{len(df_train)}**\n\n")

        # Missingness in pivot_all (before dropna)
        miss = df_wide_all[TARGETS_5D_ORDER].isna().mean() if set(TARGETS_5D_ORDER).issubset(df_wide_all.columns) else None
        if miss is not None:
            f.write("### 5D targets 缺失率（按 image_id）\n\n")
            miss_tbl = pd.DataFrame({"target": TARGETS_5D_ORDER, "missing_rate": [float(miss.get(t, np.nan)) for t in TARGETS_5D_ORDER]})
            f.write(df_to_markdown_table(miss_tbl))
            f.write("\n\n")

        f.write("## 3) 标签一致性/约束（非常关键的“可工程化结构”）\n\n")
        f.write("项目里已有 `ratio_strict` 约束，假设：\n\n")
        f.write("- `Dry_Total_g ≈ Dry_Clover_g + Dry_Dead_g + Dry_Green_g`\n")
        f.write("- `GDM_g ≈ Dry_Clover_g + Dry_Green_g`\n\n")
        f.write("在训练集真实标签上的残差统计（grams）：\n\n")
        f.write("### Dry_Total 残差：`Dry_Total_g - (Clover+Dead+Green)`\n\n")
        f.write(df_to_markdown_table(pd.DataFrame([total_res])))
        f.write("\n\n")
        f.write("### GDM 残差：`GDM_g - (Clover+Green)`\n\n")
        f.write(df_to_markdown_table(pd.DataFrame([gdm_res])))
        f.write("\n\n")

        f.write("## 4) 元数据特征工程（train.csv 自带）\n\n")
        f.write("- 日期：解析 `Sampling_Date` 并构造 year/month/day/dayofyear + `sin/cos(dayofyear)`（季节性）\n")
        f.write("- 交互项：`ndvi_x_height`、`height_over_ndvi`\n\n")

        # Category counts
        f.write("### 类别字段规模\n\n")
        state_vc = df_train["State"].astype(str).value_counts()
        species_vc = df_train["Species"].astype(str).value_counts()
        f.write(f"- State 唯一数: **{df_train['State'].nunique()}**\n")
        f.write(f"- Species 唯一数: **{df_train['Species'].nunique()}**\n\n")
        f.write("Top States:\n\n")
        f.write(df_to_markdown_table(state_vc.head(20).to_frame("count").reset_index(names=["State"])))
        f.write("\n\nTop Species:\n\n")
        f.write(df_to_markdown_table(species_vc.head(20).to_frame("count").reset_index(names=["Species"])))
        f.write("\n\n")

        f.write("## 5) 图像特征工程（从 `data/train/*` 提取）\n\n")
        f.write("提取的特征包含：\n\n")
        f.write("- 图像几何/文件：宽高、像素数、长宽比、文件大小、EXIF orientation\n")
        f.write("- 颜色：RGB/HSV 均值/方差/分位数，绿色/棕色 hue 占比，饱和度占比\n")
        f.write("- 可见光植被指数：ExG, VARI, NGRDI, G/R, G/B 的均值/方差/分位数\n")
        f.write("- 纹理：Laplacian variance、Sobel gradient magnitude、灰度熵、固定阈值边缘密度\n")
        f.write("- 空间网格（2x2）：分别计算 TL/TR/BL/BR 的 green/brown 占比、ExG/NGRDI、灰度均值、局部 lap_var，并提供 bottom-top 差分\n\n")

        f.write("## 6) 与目标的相关性（log1p 空间）\n\n")
        f.write("### 6.1 可用于推理的特征（image-only）Top-15\n\n")
        for y in TARGETS_5D_ORDER:
            tab = corr_tables_image.get(y, None)
            if tab is None or tab.empty:
                continue
            f.write(f"#### {y}\n\n")
            f.write(df_to_markdown_table(tab.head(15)))
            f.write("\n\n")

        f.write("### 6.2 仅训练可见的特征（metadata/date）Top-15（注意：test.csv 不含这些列）\n\n")
        for y in TARGETS_5D_ORDER:
            tab = corr_tables_meta.get(y, None)
            if tab is None or tab.empty:
                continue
            f.write(f"#### {y}\n\n")
            f.write(df_to_markdown_table(tab.head(15)))
            f.write("\n\n")

        f.write("## 7) 输出文件\n\n")
        f.write("- 工程化特征表：\n")
        for sp in saved_paths:
            f.write(f"  - `{sp}`\n")
        f.write(f"- 相关性表（每个 target 一份）：`{corr_dir}`\n")
        f.write(f"- 本报告：`{report_path}`\n")

    logger.info("Saved report -> {}", str(report_path))
    logger.info("Saved features -> {}", saved_paths)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


