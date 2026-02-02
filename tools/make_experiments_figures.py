#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import matplotlib.pyplot as plt
import yaml


def resolve_repo_root() -> Path:
    """
    Resolve the project root directory that contains both `configs/` and `src/`.
    """
    here = Path(__file__).resolve()
    for p in [here.parent] + list(here.parents):
        if (p / "configs").is_dir() and (p / "src").is_dir():
            return p
    return Path.cwd().resolve()


def _as_bool(x: Any) -> Optional[bool]:
    if x is None:
        return None
    if isinstance(x, bool):
        return bool(x)
    s = str(x).strip().lower()
    if s in ("true", "1", "yes", "y", "on"):
        return True
    if s in ("false", "0", "no", "n", "off"):
        return False
    return None


def _as_int(x: Any) -> Optional[int]:
    if x is None:
        return None
    try:
        return int(float(x))
    except Exception:
        return None


def _as_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    try:
        v = float(x)
        if not math.isfinite(v):
            return None
        return float(v)
    except Exception:
        return None


def _read_yaml(path: Path) -> dict[str, Any]:
    obj = yaml.safe_load(path.read_text(encoding="utf-8"))
    return obj if isinstance(obj, dict) else {}


def _read_json(path: Path) -> dict[str, Any]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    return obj if isinstance(obj, dict) else {}


@dataclass(frozen=True)
class KFoldR2:
    r2_mean: Optional[float]
    r2_std: Optional[float]
    num_folds: int


def _extract_kfold_swa_r2(run_root: Path) -> KFoldR2:
    p = run_root / "kfold_swa_metrics.json"
    if not p.is_file():
        return KFoldR2(r2_mean=None, r2_std=None, num_folds=0)

    obj = _read_json(p)
    per_fold = obj.get("per_fold", []) if isinstance(obj.get("per_fold", None), list) else []
    r2s: list[float] = []
    for m in per_fold:
        if not isinstance(m, dict):
            continue
        v = _as_float(m.get("r2", None))
        if v is None:
            continue
        r2s.append(float(v))

    num_folds = int(_as_int(obj.get("num_folds", None)) or len(r2s) or 0)

    # Prefer the JSON's precomputed average if present.
    r2_mean = None
    avg = obj.get("average", None)
    if isinstance(avg, dict):
        r2_mean = _as_float(avg.get("r2", None))
    if r2_mean is None and r2s:
        r2_mean = float(sum(r2s) / len(r2s))

    r2_std = None
    if len(r2s) >= 2:
        mu = float(sum(r2s) / len(r2s))
        r2_std = float(math.sqrt(sum((x - mu) ** 2 for x in r2s) / (len(r2s) - 1)))

    return KFoldR2(r2_mean=r2_mean, r2_std=r2_std, num_folds=num_folds)


def _extract_kfold_swa_per_fold_r2(run_root: Path) -> list[float]:
    p = run_root / "kfold_swa_metrics.json"
    if not p.is_file():
        return []
    obj = _read_json(p)
    per_fold = obj.get("per_fold", []) if isinstance(obj.get("per_fold", None), list) else []
    out: list[float] = []
    for m in per_fold:
        if not isinstance(m, dict):
            continue
        v = _as_float(m.get("r2", None))
        if v is None:
            continue
        out.append(float(v))
    return out


def _wrap_label(s: str, width: int = 16) -> str:
    # simple word-wrap for tick labels
    words = [w for w in str(s).replace("_", " ").split() if w]
    if not words:
        return str(s)
    lines: list[str] = []
    cur: list[str] = []
    n = 0
    for w in words:
        add = len(w) + (1 if cur else 0)
        if n + add > int(width) and cur:
            lines.append(" ".join(cur))
            cur = [w]
            n = len(w)
        else:
            cur.append(w)
            n += add
    if cur:
        lines.append(" ".join(cur))
    return "\n".join(lines)


def _plot_kfold_points(
    *,
    out_path_base: Path,
    title: str,
    xlabel: str,
    ylabel: str,
    items: list[dict[str, Any]],
    label_key: str,
    mean_key: str,
    per_fold_key: str,
) -> None:
    if not items:
        return

    plt.rcParams.update({"figure.dpi": 160})
    fig, ax = plt.subplots(figsize=(10.5, 4.2))

    xs = np.arange(len(items), dtype=np.float64)
    rng = np.random.default_rng(0)

    means: list[float] = []
    all_y: list[float] = []
    for i, it in enumerate(items):
        vals = list(it.get(per_fold_key) or [])
        vals_f = [float(v) for v in vals if v is not None and math.isfinite(float(v))]
        if vals_f:
            jitter = rng.uniform(-0.12, 0.12, size=len(vals_f))
            ax.scatter(
                np.full(len(vals_f), xs[i]) + jitter,
                np.asarray(vals_f, dtype=np.float64),
                s=22,
                alpha=0.75,
                linewidths=0.0,
                zorder=2,
            )
            mu = float(np.mean(np.asarray(vals_f, dtype=np.float64)))
            all_y.extend(vals_f)
        else:
            mu = _as_float(it.get(mean_key)) or float("nan")
        means.append(mu)
        if math.isfinite(mu):
            all_y.append(mu)

    ax.plot(xs, means, color="black", linewidth=1.6, marker="o", markersize=4.5, zorder=3)

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xticks(xs)
    ax.set_xticklabels([_wrap_label(str(it.get(label_key, ""))) for it in items])
    ax.grid(True, axis="y", alpha=0.35)

    # Keep a tight, interpretable range for R^2.
    finite_y = [v for v in all_y if math.isfinite(v)]
    if finite_y:
        lo = min(finite_y)
        hi = max(finite_y)
        pad = max(0.02, 0.08 * (hi - lo))
        ax.set_ylim(max(-0.6, lo - pad), min(1.0, hi + pad))

    fig.tight_layout()
    fig.savefig(str(out_path_base) + ".svg", bbox_inches="tight")
    fig.savefig(str(out_path_base) + ".png", bbox_inches="tight")
    plt.close(fig)


def _plot_kfold_sorted_intervals(
    *,
    out_path_base: Path,
    title: str,
    xlabel: str,
    ylabel: str,
    items: list[dict[str, Any]],
    label_key: str,
    per_fold_key: str,
) -> None:
    """
    Plot per-fold points + mean ± std for each item, sorted by mean.

    This is a higher-signal alternative to a category plot with a connecting mean line.
    """
    if not items:
        return

    rows: list[dict[str, Any]] = []
    for it in items:
        label = str(it.get(label_key, "") or "")
        vals = list(it.get(per_fold_key) or [])
        vals_f = [float(v) for v in vals if v is not None and math.isfinite(float(v))]
        mu = float(np.mean(np.asarray(vals_f, dtype=np.float64))) if vals_f else float("nan")
        std = float(np.std(np.asarray(vals_f, dtype=np.float64), ddof=1)) if len(vals_f) >= 2 else 0.0
        rows.append({"label": label, "vals": vals_f, "mu": mu, "std": std})

    # Sort by mean (descending); keep deterministic fallback by label.
    rows.sort(key=lambda r: (-(r["mu"] if math.isfinite(r["mu"]) else -1e9), r["label"]))

    plt.rcParams.update({"figure.dpi": 160})
    fig_h = max(2.2, 0.55 * len(rows) + 1.35)
    fig, ax = plt.subplots(figsize=(10.5, fig_h), dpi=160)
    rng = np.random.default_rng(0)

    all_x: list[float] = []
    for i, r in enumerate(rows):
        vals_f = r["vals"]
        if vals_f:
            jitter = rng.uniform(-0.18, 0.18, size=len(vals_f))
            ax.scatter(
                np.asarray(vals_f, dtype=np.float64),
                np.full(len(vals_f), i, dtype=np.float64) + jitter,
                s=26,
                alpha=0.78,
                linewidths=0.0,
                zorder=2,
            )
            all_x.extend(vals_f)

        if math.isfinite(r["mu"]):
            ax.errorbar(
                r["mu"],
                i,
                xerr=r["std"],
                fmt="o",
                color="black",
                ecolor="black",
                elinewidth=1.4,
                capsize=3.0,
                markersize=4.8,
                zorder=3,
            )
            all_x.append(float(r["mu"]))

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_yticks(np.arange(len(rows), dtype=np.float64))
    ax.set_yticklabels([_wrap_label(r["label"], width=28) for r in rows])
    ax.invert_yaxis()
    ax.grid(True, axis="x", alpha=0.35)

    # Tight, interpretable range for R^2.
    finite_x = [v for v in all_x if math.isfinite(v)]
    if finite_x:
        lo = min(finite_x)
        hi = max(finite_x)
        pad = max(0.02, 0.08 * (hi - lo))
        ax.set_xlim(max(-0.6, lo - pad), min(1.0, hi + pad))

    fig.tight_layout()
    fig.savefig(str(out_path_base) + ".svg", bbox_inches="tight")
    fig.savefig(str(out_path_base) + ".png", bbox_inches="tight")
    plt.close(fig)


def _plot_kfold_fold_heatmap(
    *,
    out_path_base: Path,
    title: str,
    items: list[dict[str, Any]],
    label_key: str,
    fold_map_key: str,
    fold_order: Optional[list[int]] = None,
) -> None:
    """
    Heatmap of fold-wise R² (rows=folds, cols=variants), annotated with values.
    """
    if not items:
        return

    # Build fold set.
    fold_ids: set[int] = set()
    for it in items:
        fm = it.get(fold_map_key) or {}
        if isinstance(fm, dict):
            for k in fm.keys():
                try:
                    fold_ids.add(int(k))
                except Exception:
                    continue
    folds = list(fold_order) if fold_order is not None else sorted(fold_ids)
    if not folds:
        return

    labels = [str(it.get(label_key, "") or "") for it in items]
    mat = np.full((len(folds), len(items)), np.nan, dtype=np.float64)
    for j, it in enumerate(items):
        fm = it.get(fold_map_key) or {}
        if not isinstance(fm, dict):
            continue
        for i, f in enumerate(folds):
            v = fm.get(f, fm.get(float(f), None))
            v = _as_float(v)
            if v is None:
                continue
            mat[i, j] = float(v)

    plt.rcParams.update({"figure.dpi": 160})
    fig_w = max(6.8, 2.2 + 1.2 * len(items))
    fig, ax = plt.subplots(figsize=(fig_w, 3.6), dpi=160)
    im = ax.imshow(mat, aspect="auto", interpolation="nearest")
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("Weighted R² (log1p grams; global mean baseline)")

    ax.set_title(title)
    ax.set_xlabel("ViTDet variant")
    ax.set_ylabel("Fold")
    ax.set_xticks(np.arange(len(items)))
    ax.set_xticklabels([_wrap_label(s, width=20) for s in labels], rotation=18, ha="right")
    ax.set_yticks(np.arange(len(folds)))
    ax.set_yticklabels([str(f) for f in folds])

    # Annotate each cell.
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat[i, j]
            if not math.isfinite(float(v)):
                continue
            # Choose a readable text color against the colormap.
            tcol = "white" if float(im.norm(float(v))) < 0.45 else "black"
            ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=8.5, color=tcol)

    fig.tight_layout()
    fig.savefig(str(out_path_base) + ".svg", bbox_inches="tight")
    fig.savefig(str(out_path_base) + ".png", bbox_inches="tight")
    plt.close(fig)


def _pick_metrics_csv(run_root: Path) -> Optional[Path]:
    """
    Pick a representative Lightning metrics.csv for this run.
    Prefer train_all/*/metrics.csv, else the newest metrics.csv found.
    """
    candidates: list[tuple[int, float, Path]] = []
    for p in run_root.rglob("metrics.csv"):
        if not p.is_file():
            continue
        rel = p.relative_to(run_root).as_posix()
        score = 0
        if rel.startswith("train_all/"):
            score += 100
        if "/lightning/" in rel:
            score += 10
        try:
            mtime = p.stat().st_mtime
        except Exception:
            mtime = 0.0
        candidates.append((score, mtime, p))
    if not candidates:
        return None
    candidates.sort(key=lambda x: (-x[0], -x[1], str(x[2])))
    return candidates[0][2]


@dataclass(frozen=True)
class CsvBestLast:
    metric_key: Optional[str]
    best_value: Optional[float]
    best_epoch: Optional[int]
    last_value: Optional[float]
    last_epoch: Optional[int]


def _extract_best_last_from_metrics_csv(run_root: Path, *, metric: str) -> CsvBestLast:
    p = _pick_metrics_csv(run_root)
    if p is None:
        return CsvBestLast(metric_key=None, best_value=None, best_epoch=None, last_value=None, last_epoch=None)

    with p.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames or []
        key_candidates = [metric, f"{metric}/dataloader_idx_0", f"{metric}/dataloader_idx_1"]
        key = None
        for k in key_candidates:
            if k in fields:
                key = k
                break
        if key is None:
            return CsvBestLast(metric_key=None, best_value=None, best_epoch=None, last_value=None, last_epoch=None)

        best_v: Optional[float] = None
        best_ep: Optional[int] = None
        last_v: Optional[float] = None
        last_ep: Optional[int] = None

        for row in reader:
            v = _as_float((row.get(key) or "").strip())
            if v is None:
                continue
            ep = _as_int((row.get("epoch") or "").strip())
            last_v, last_ep = float(v), ep
            if best_v is None or float(v) > best_v:
                best_v, best_ep = float(v), ep

    return CsvBestLast(metric_key=key, best_value=best_v, best_epoch=best_ep, last_value=last_v, last_epoch=last_ep)


def _extract_curve_from_metrics_csv(run_root: Path, *, metric: str) -> tuple[str | None, list[int], list[float]]:
    """
    Extract an (epoch -> metric) curve from a Lightning metrics.csv.
    Returns (resolved_metric_key, epochs, values).
    """
    p = _pick_metrics_csv(run_root)
    if p is None:
        return None, [], []

    epochs: list[int] = []
    vals: list[float] = []

    with p.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames or []
        key_candidates = [metric, f"{metric}/dataloader_idx_0", f"{metric}/dataloader_idx_1"]
        key = None
        for k in key_candidates:
            if k in fields:
                key = k
                break
        if key is None:
            return None, [], []

        # Keep the LAST observation per epoch (Lightning can emit multiple rows/epoch).
        per_epoch: dict[int, float] = {}
        for row in reader:
            ep = _as_int((row.get("epoch") or "").strip())
            if ep is None:
                continue
            v = _as_float((row.get(key) or "").strip())
            if v is None:
                continue
            per_epoch[int(ep)] = float(v)

        for ep in sorted(per_epoch.keys()):
            epochs.append(int(ep))
            vals.append(float(per_epoch[ep]))

    return key, epochs, vals


def _cfg_get(cfg: dict[str, Any], path: str) -> Any:
    """
    Minimal dotted-path getter for dict-of-dicts configs.
    """
    cur: Any = cfg
    for part in path.split("."):
        if not isinstance(cur, dict):
            return None
        if part not in cur:
            return None
        cur = cur.get(part)
    return cur


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--manifest",
        type=str,
        default="docs/figures/experiments/run_manifest.yaml",
        help="Path to the experiments run manifest (YAML).",
    )
    ap.add_argument(
        "--out-dir",
        type=str,
        default="docs/figures/experiments",
        help="Output directory for summary CSV and figures.",
    )
    ap.add_argument(
        "--make-fig-ablation",
        action="store_true",
        help="Generate Figure A (ViTDet k-fold overview) under out-dir.",
    )
    ap.add_argument(
        "--make-fig-head-compare",
        action="store_true",
        help="Generate Figure B (ViTDet fold heatmap) under out-dir.",
    )
    ap.add_argument(
        "--make-figures",
        action="store_true",
        help="Generate all figures (A + B).",
    )
    ap.add_argument(
        "--make-fig-learning-curve",
        action="store_true",
        help="Generate Figure C (learning curve; optional) under out-dir.",
    )
    args = ap.parse_args()

    repo_root = resolve_repo_root()
    manifest_path = (repo_root / str(args.manifest)).resolve()
    out_dir = (repo_root / str(args.out_dir)).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = _read_yaml(manifest_path)

    entries: list[dict[str, Any]] = []

    def add_entry(*, group: str, entry_id: str, label: str, run_root: str, source: str, metric: str) -> None:
        entries.append(
            {
                "group": group,
                "id": entry_id,
                "label": label,
                "run_root": run_root,
                "source": source,
                "metric": metric,
            }
        )

    # ViTDet k-fold runs (main results for §6)
    vk = manifest.get("vitdet_kfold", {}) if isinstance(manifest.get("vitdet_kfold", None), dict) else {}
    vk_metric = str(vk.get("metric", "r2_global") or "").strip()
    for r in vk.get("runs", []) if isinstance(vk.get("runs", None), list) else []:
        if not isinstance(r, dict):
            continue
        add_entry(
            group="vitdet_kfold",
            entry_id=str(r.get("id", "") or "").strip(),
            label=str(r.get("label", "") or "").strip(),
            run_root=str(r.get("run_root", "") or "").strip(),
            source=str(r.get("source", "") or "").strip(),
            metric=vk_metric,
        )

    # Enrich with metrics + config fields
    out_rows: list[dict[str, Any]] = []
    for e in entries:
        run_root_rel = str(e["run_root"]).strip()
        run_root = (repo_root / run_root_rel).resolve()
        if not run_root.exists():
            raise FileNotFoundError(f"run_root not found: {run_root_rel} -> {run_root}")

        cfg_path = run_root / "train.yaml"
        cfg = _read_yaml(cfg_path) if cfg_path.is_file() else {}

        # Metrics
        r2_kfold = _extract_kfold_swa_r2(run_root) if e["source"] == "kfold_swa_metrics" else None
        csv_stats = (
            _extract_best_last_from_metrics_csv(run_root, metric=str(e["metric"]))
            if e["source"] == "metrics_csv"
            else None
        )

        row: dict[str, Any] = dict(e)
        row["run_name"] = run_root.name
        row["train_yaml"] = str(cfg_path.relative_to(repo_root)) if cfg_path.is_file() else ""

        # Canonical metric outputs
        row["r2_global_mean"] = r2_kfold.r2_mean if r2_kfold is not None else None
        row["r2_global_std"] = r2_kfold.r2_std if r2_kfold is not None else None
        row["num_folds"] = r2_kfold.num_folds if r2_kfold is not None else None

        # Optional per-run CSV stats (for prototypes only)
        row["csv_metric_key"] = csv_stats.metric_key if csv_stats is not None else None
        row["csv_best_value"] = csv_stats.best_value if csv_stats is not None else None
        row["csv_best_epoch"] = csv_stats.best_epoch if csv_stats is not None else None
        row["csv_last_value"] = csv_stats.last_value if csv_stats is not None else None
        row["csv_last_epoch"] = csv_stats.last_epoch if csv_stats is not None else None

        # Key config fields (keep these sparse; this CSV is for paper tables/figures)
        row["version"] = cfg.get("version", None)
        row["data.image_size"] = _cfg_get(cfg, "data.image_size")
        row["trainer.max_epochs"] = _cfg_get(cfg, "trainer.max_epochs")
        row["model.backbone"] = _cfg_get(cfg, "model.backbone")
        row["model.head.type"] = _cfg_get(cfg, "model.head.type")
        row["model.log_scale_targets"] = _cfg_get(cfg, "model.log_scale_targets")
        row["model.backbone_layers.enabled"] = _cfg_get(cfg, "model.backbone_layers.enabled")
        row["model.backbone_layers.layer_fusion"] = _cfg_get(cfg, "model.backbone_layers.layer_fusion")
        row["model.head.vitdet_dim"] = _cfg_get(cfg, "model.head.vitdet_dim")
        row["model.head.vitdet_patch_size"] = _cfg_get(cfg, "model.head.vitdet_patch_size")
        row["model.head.vitdet_scale_factors"] = _cfg_get(cfg, "model.head.vitdet_scale_factors")
        row["model.head.hidden_dims"] = _cfg_get(cfg, "model.head.hidden_dims")
        row["model.head.activation"] = _cfg_get(cfg, "model.head.activation")
        row["model.head.dropout"] = _cfg_get(cfg, "model.head.dropout")
        row["model.head.ratio_head_mode"] = _cfg_get(cfg, "model.head.ratio_head_mode")
        row["model.head.use_cls_token"] = _cfg_get(cfg, "model.head.use_cls_token")
        row["model.head.use_patch_reg3"] = _cfg_get(cfg, "model.head.use_patch_reg3")
        row["data.augment.cutmix.enabled"] = _cfg_get(cfg, "data.augment.cutmix.enabled")
        row["data.augment.cutmix.prob"] = _cfg_get(cfg, "data.augment.cutmix.prob")
        row["data.augment.cutmix.alpha"] = _cfg_get(cfg, "data.augment.cutmix.alpha")
        row["peft.r"] = _cfg_get(cfg, "peft.r")
        row["peft.lora_alpha"] = _cfg_get(cfg, "peft.lora_alpha")
        row["peft.lora_dropout"] = _cfg_get(cfg, "peft.lora_dropout")
        row["peft.lora_lr"] = _cfg_get(cfg, "peft.lora_lr")
        row["peft.lora_llrd"] = _cfg_get(cfg, "peft.lora_llrd")
        row["optimizer.lr"] = _cfg_get(cfg, "optimizer.lr")
        row["optimizer.weight_decay"] = _cfg_get(cfg, "optimizer.weight_decay")
        row["data.augment.manifold_mixup.enabled"] = _cfg_get(cfg, "data.augment.manifold_mixup.enabled")
        row["data.augment.manifold_mixup.prob"] = _cfg_get(cfg, "data.augment.manifold_mixup.prob")
        row["data.augment.manifold_mixup.alpha"] = _cfg_get(cfg, "data.augment.manifold_mixup.alpha")
        row["peft.enabled"] = _cfg_get(cfg, "peft.enabled")
        row["peft.last_k_blocks"] = _cfg_get(cfg, "peft.last_k_blocks")
        row["mtl.enabled"] = _cfg_get(cfg, "mtl.enabled")
        row["pcgrad.enabled"] = _cfg_get(cfg, "pcgrad.enabled")
        row["trainer.ema.enabled"] = _cfg_get(cfg, "trainer.ema.enabled")
        row["trainer.ema.decay"] = _cfg_get(cfg, "trainer.ema.decay")

        out_rows.append(row)

    # Write CSV
    out_csv = out_dir / "experiments_summary.csv"
    # stable column order (important for diffs)
    cols = [
        "group",
        "id",
        "label",
        "run_name",
        "run_root",
        "source",
        "metric",
        "r2_global_mean",
        "r2_global_std",
        "num_folds",
        "csv_metric_key",
        "csv_best_value",
        "csv_best_epoch",
        "csv_last_value",
        "csv_last_epoch",
        "version",
        "model.backbone",
        "model.head.type",
        "model.log_scale_targets",
        "model.backbone_layers.enabled",
        "model.backbone_layers.layer_fusion",
        "model.head.vitdet_dim",
        "model.head.vitdet_patch_size",
        "model.head.vitdet_scale_factors",
        "model.head.hidden_dims",
        "model.head.activation",
        "model.head.dropout",
        "model.head.ratio_head_mode",
        "model.head.use_cls_token",
        "model.head.use_patch_reg3",
        "peft.enabled",
        "peft.r",
        "peft.lora_alpha",
        "peft.lora_dropout",
        "peft.last_k_blocks",
        "peft.lora_lr",
        "peft.lora_llrd",
        "optimizer.lr",
        "optimizer.weight_decay",
        "data.augment.cutmix.enabled",
        "data.augment.cutmix.prob",
        "data.augment.cutmix.alpha",
        "data.augment.manifold_mixup.enabled",
        "data.augment.manifold_mixup.prob",
        "data.augment.manifold_mixup.alpha",
        "mtl.enabled",
        "pcgrad.enabled",
        "trainer.ema.enabled",
        "trainer.ema.decay",
        "trainer.max_epochs",
        "data.image_size",
        "train_yaml",
    ]

    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(cols), extrasaction="ignore")
        w.writeheader()
        for r in out_rows:
            w.writerow(r)

    print(f"[make_experiments_figures] wrote: {out_csv}")

    # -----------------
    # Figures (optional)
    # -----------------
    make_ablation = bool(args.make_figures or args.make_fig_ablation)
    make_heads = bool(args.make_figures or args.make_fig_head_compare)
    make_curve = bool(args.make_fig_learning_curve)

    # Collect ViTDet k-fold items once (used by multiple plots).
    vitdet_items: list[dict[str, Any]] = []
    for e in entries:
        if e.get("group") != "vitdet_kfold":
            continue
        if e.get("source") != "kfold_swa_metrics":
            continue
        rr = (repo_root / str(e["run_root"])).resolve()
        obj = _read_json(rr / "kfold_swa_metrics.json")
        per_fold = obj.get("per_fold", []) if isinstance(obj.get("per_fold", None), list) else []
        fold_map: dict[int, float] = {}
        for m in per_fold:
            if not isinstance(m, dict):
                continue
            f = _as_int(m.get("fold", None))
            v = _as_float(m.get("r2", None))
            if f is None or v is None:
                continue
            fold_map[int(f)] = float(v)
        per = _extract_kfold_swa_per_fold_r2(rr)
        r2 = _extract_kfold_swa_r2(rr)
        vitdet_items.append(
            {
                "label": str(e.get("label", "")),
                "r2_global_mean": r2.r2_mean,
                "r2_global_std": r2.r2_std,
                "per_fold_r2": per,
                "fold_map": fold_map,
            }
        )

    # Figure A: ViTDet k-fold overview (sorted).
    if make_ablation:
        _plot_kfold_sorted_intervals(
            out_path_base=out_dir / "fig_vitdet_kfold_overview_r2_global",
            title="ViTDet variants (k-fold; global-baseline R²)",
            xlabel="Weighted R² (log1p grams; global mean baseline)",
            ylabel="Variant",
            items=vitdet_items,
            label_key="label",
            per_fold_key="per_fold_r2",
        )
        print(f"[make_experiments_figures] wrote: {out_dir / 'fig_vitdet_kfold_overview_r2_global.(svg|png)'}")

    # Figure B: Fold-wise heatmap.
    if make_heads:
        # Use the same sorted order as Figure A for consistency.
        vitdet_items_sorted = sorted(
            vitdet_items,
            key=lambda d: (-(d.get('r2_global_mean') if isinstance(d.get('r2_global_mean'), (int, float)) else -1e9), str(d.get('label', ''))),
        )
        _plot_kfold_fold_heatmap(
            out_path_base=out_dir / "fig_vitdet_kfold_fold_heatmap_r2_global",
            title="Fold-wise behavior (k-fold R²; global baseline)",
            items=vitdet_items_sorted,
            label_key="label",
            fold_map_key="fold_map",
            fold_order=[0, 1, 2, 3, 4],
        )
        print(f"[make_experiments_figures] wrote: {out_dir / 'fig_vitdet_kfold_fold_heatmap_r2_global.(svg|png)'}")

    if make_curve:
        lc = manifest.get("learning_curve", {}) if isinstance(manifest.get("learning_curve", None), dict) else {}
        metric = str(lc.get("metric", "val_r2_global") or "").strip()
        runs = lc.get("runs", []) if isinstance(lc.get("runs", None), list) else []

        fig, ax = plt.subplots(figsize=(10.0, 4.0), dpi=160)
        clip_lo = -1.5
        clip_hi = 1.0
        for r in runs:
            if not isinstance(r, dict):
                continue
            rr = (repo_root / str(r.get("run_root", "") or "")).resolve()
            label = str(r.get("label", rr.name) or rr.name)
            key, eps, vs = _extract_curve_from_metrics_csv(rr, metric=metric)
            if not eps:
                continue
            ys = np.asarray(vs, dtype=np.float64)
            ys = np.clip(ys, clip_lo, clip_hi)
            ax.plot(eps, ys.tolist(), marker="o", markersize=3.5, linewidth=1.6, label=label)

        ax.set_title(f"Learning curve (validation; clipped to [{clip_lo}, {clip_hi}])")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Weighted R² (log1p grams; global mean baseline)")
        ax.grid(True, axis="y", alpha=0.35)
        ax.legend(loc="best")
        ax.set_ylim(clip_lo - 0.05, clip_hi + 0.05)
        fig.tight_layout()
        out_base = out_dir / "fig_learning_curve_r2_global"
        fig.savefig(str(out_base) + ".svg", bbox_inches="tight")
        fig.savefig(str(out_base) + ".png", bbox_inches="tight")
        plt.close(fig)
        print(f"[make_experiments_figures] wrote: {out_base}.(svg|png)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

