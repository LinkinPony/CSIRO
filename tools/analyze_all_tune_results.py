#!/usr/bin/env python3
from __future__ import annotations

"""
Analyze all Ray Tune experiment results under a ray_results/ directory and produce:

- outputs/tune_analysis/experiments_summary.csv
- outputs/tune_analysis/trials_parquet/{trials.parquet, params_long.parquet}
- outputs/tune_analysis/param_importance.md

This script is intentionally "offline" (no Ray required). It parses Tune trial folders
containing both:
  - params.json (dict of dotted hyperparameter overrides)
  - result.json (JSONL of reported metrics)
"""

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd


def resolve_repo_root() -> Path:
    """
    Resolve the project root directory that contains both `conf/` and `src/`.
    """
    here = Path(__file__).resolve()
    for p in [here.parent] + list(here.parents):
        if (p / "conf").is_dir() and (p / "src").is_dir():
            return p
    return Path.cwd().resolve()


def _safe_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    try:
        s = str(x).strip()
        if s == "" or s.lower() in ("nan", "none", "null"):
            return None
        v = float(s)
        if not math.isfinite(v):
            return None
        return float(v)
    except Exception:
        return None


def _safe_int(x: Any) -> Optional[int]:
    if x is None:
        return None
    try:
        s = str(x).strip()
        if s == "" or s.lower() in ("nan", "none", "null"):
            return None
        return int(float(s))
    except Exception:
        return None


def _json_dumps_stable(x: Any) -> str:
    try:
        return json.dumps(x, sort_keys=True, ensure_ascii=False)
    except Exception:
        return str(x)


def _guess_trial_id_from_dir(trial_dir: Path) -> str:
    """
    Best-effort parse trial id from directory name:
      trainable_<exp>_<trialid>_<idx>_... -> <exp>_<trialid>
    """
    name = trial_dir.name
    if name.startswith("trainable_"):
        parts = name.split("_")
        if len(parts) >= 3:
            return f"{parts[1]}_{parts[2]}"
    return ""


def _iter_trial_dirs(exp_dir: Path) -> list[Path]:
    out: list[Path] = []
    try:
        for p in sorted(exp_dir.iterdir()):
            if not p.is_dir():
                continue
            if (p / "params.json").is_file() and (p / "result.json").is_file():
                out.append(p)
    except Exception:
        return []
    return out


def _read_params_json(trial_dir: Path) -> dict[str, Any]:
    p = trial_dir / "params.json"
    with open(p, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, dict):
        raise TypeError(f"params.json must be a dict, got: {type(obj)} ({p})")
    return obj


@dataclass(frozen=True)
class TrialMetricSummary:
    best_value: Optional[float]
    best_epoch: Optional[int]
    best_done: Optional[bool]
    last_value: Optional[float]
    last_epoch: Optional[int]
    last_done: Optional[bool]
    n_rows_scanned: int
    n_rows_with_metric: int
    trial_id_from_rows: str


def _summarize_result_jsonl(
    trial_dir: Path,
    *,
    metric: str,
    mode: str,
    min_epoch: int,
) -> TrialMetricSummary:
    mode_n = str(mode).strip().lower()
    if mode_n not in ("min", "max"):
        raise ValueError(f"Unsupported mode: {mode!r} (expected 'min' or 'max')")

    result_json = trial_dir / "result.json"
    best_val: Optional[float] = None
    best_epoch: Optional[int] = None
    best_done: Optional[bool] = None

    last_val: Optional[float] = None
    last_epoch: Optional[int] = None
    last_done: Optional[bool] = None

    n_rows = 0
    n_metric = 0
    trial_id: str = ""

    try:
        with open(result_json, "r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if not s:
                    continue
                n_rows += 1
                try:
                    row = json.loads(s)
                except Exception:
                    continue
                if not isinstance(row, dict):
                    continue

                if not trial_id:
                    tid = str(row.get("trial_id", "") or "").strip()
                    if tid:
                        trial_id = tid

                ep = _safe_int(row.get("epoch"))
                if ep is None or ep < int(min_epoch):
                    continue

                v = _safe_float(row.get(str(metric)))
                if v is None:
                    continue
                n_metric += 1

                done_raw = row.get("done", None)
                done = (bool(done_raw) if done_raw is not None else None)

                last_val, last_epoch, last_done = float(v), int(ep), done

                if best_val is None:
                    best_val, best_epoch, best_done = float(v), int(ep), done
                else:
                    better = (v < best_val) if mode_n == "min" else (v > best_val)
                    if better:
                        best_val, best_epoch, best_done = float(v), int(ep), done
    except Exception:
        # Keep best-effort semantics: return empty summary.
        return TrialMetricSummary(
            best_value=None,
            best_epoch=None,
            best_done=None,
            last_value=None,
            last_epoch=None,
            last_done=None,
            n_rows_scanned=n_rows,
            n_rows_with_metric=n_metric,
            trial_id_from_rows=trial_id,
        )

    return TrialMetricSummary(
        best_value=best_val,
        best_epoch=best_epoch,
        best_done=best_done,
        last_value=last_val,
        last_epoch=last_epoch,
        last_done=last_done,
        n_rows_scanned=n_rows,
        n_rows_with_metric=n_metric,
        trial_id_from_rows=trial_id,
    )


@dataclass(frozen=True)
class TuneCfgInfo:
    config_name: str
    tune_name: str
    tune_metric: str
    tune_mode: str
    scheduler_type: str
    grace_period: Optional[int]
    two_stage_enabled: bool
    two_stage_min_epoch: Optional[int]
    two_stage_top_n: Optional[int]
    two_stage_stage2_metric: Optional[str]
    two_stage_stage2_mode: Optional[str]


def _compose_hydra_cfg(repo_root: Path, *, config_name: str, overrides: list[str]) -> dict[str, Any]:
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra
    from omegaconf import OmegaConf

    try:
        GlobalHydra.instance().clear()
    except Exception:
        pass

    with initialize_config_dir(config_dir=str(repo_root / "conf"), version_base=None):
        cfg = compose(config_name=str(config_name), overrides=list(overrides or []))

    cfg_dict = OmegaConf.to_container(cfg, resolve=True)  # type: ignore[assignment]
    if not isinstance(cfg_dict, dict):
        raise TypeError(f"Hydra config did not resolve to a dict, got: {type(cfg_dict)}")
    return cfg_dict


def _load_tune_cfg_index(repo_root: Path) -> dict[str, TuneCfgInfo]:
    """
    Index conf/tune*.yaml by their resolved tune.name.
    """
    out: dict[str, TuneCfgInfo] = {}
    for p in sorted((repo_root / "conf").glob("tune*.yaml")):
        config_name = p.stem
        try:
            cfg = _compose_hydra_cfg(repo_root, config_name=config_name, overrides=[])
        except Exception:
            continue
        tune_cfg = dict(cfg.get("tune", {}) or {})
        name = str(tune_cfg.get("name", "") or "").strip()
        if not name:
            continue

        sched = dict(tune_cfg.get("scheduler", {}) or {})
        sched_type = str(sched.get("type", "") or "").strip().lower()
        grace = _safe_int(sched.get("grace_period")) if sched_type == "asha" else None

        two_stage_cfg = dict(tune_cfg.get("two_stage", {}) or {})
        ts_enabled = bool(two_stage_cfg.get("enabled", False))

        out[name] = TuneCfgInfo(
            config_name=config_name,
            tune_name=name,
            tune_metric=str(tune_cfg.get("metric", "") or "").strip(),
            tune_mode=str(tune_cfg.get("mode", "") or "").strip(),
            scheduler_type=sched_type,
            grace_period=grace,
            two_stage_enabled=ts_enabled,
            two_stage_min_epoch=_safe_int(two_stage_cfg.get("min_epoch")),
            two_stage_top_n=_safe_int(two_stage_cfg.get("top_n")),
            two_stage_stage2_metric=(str(two_stage_cfg.get("stage2_metric", "")).strip() or None),
            two_stage_stage2_mode=(str(two_stage_cfg.get("stage2_mode", "")).strip() or None),
        )
    return out


def _infer_min_epoch_for_experiment(
    exp_dir: Path,
    *,
    exp_name: str,
    tune_cfg_index: dict[str, TuneCfgInfo],
    default_min_epoch: int,
) -> int:
    # Highest priority: two_stage_candidates.json (written by tune.py)
    cand = exp_dir / "two_stage_candidates.json"
    if cand.is_file():
        try:
            with open(cand, "r", encoding="utf-8") as f:
                obj = json.load(f)
            if isinstance(obj, dict):
                stage1 = obj.get("stage1", {}) or {}
                if isinstance(stage1, dict):
                    me = _safe_int(stage1.get("min_epoch"))
                    if me is not None:
                        return int(me)
        except Exception:
            pass

    # Next: infer from conf index by tune.name.
    info = tune_cfg_index.get(exp_name)
    if info is not None:
        if info.two_stage_min_epoch is not None:
            return int(info.two_stage_min_epoch)
        if info.grace_period is not None:
            return int(info.grace_period)

    return int(default_min_epoch)


def _is_stage2_experiment_name(name: str) -> bool:
    n = str(name).lower()
    return ("stage2" in n) or n.endswith("-kfold") or n.endswith("-stage2") or n.endswith("-stage2-kfold")


def _spearmanr(x: np.ndarray, y: np.ndarray) -> Optional[float]:
    """
    Spearman correlation without scipy: corr(rank(x), rank(y)).
    Returns None when insufficient data / zero variance.
    """
    if x.size < 3 or y.size < 3:
        return None
    # rankdata (average ranks for ties)
    rx = pd.Series(x).rank(method="average").to_numpy(dtype=float)
    ry = pd.Series(y).rank(method="average").to_numpy(dtype=float)
    sx = float(np.std(rx))
    sy = float(np.std(ry))
    if sx <= 0.0 or sy <= 0.0:
        return None
    return float(np.corrcoef(rx, ry)[0, 1])


def _coerce_param_value(v: Any) -> tuple[Optional[float], str]:
    """
    Return (numeric_value_if_possible, stable_string_value).
    """
    if isinstance(v, (int, float)) and math.isfinite(float(v)):
        return float(v), str(v)
    if isinstance(v, (list, tuple)):
        # Make it a stable categorical token.
        return None, _json_dumps_stable(list(v))
    if isinstance(v, dict):
        return None, _json_dumps_stable(v)
    # Try parse numeric from strings
    fv = _safe_float(v)
    if fv is not None:
        return float(fv), str(v)
    return None, str(v)


def _build_param_table(trials: pd.DataFrame) -> pd.DataFrame:
    """
    Expand trials.params (dict) into a long table:
      experiment, trial_id, param_key, param_value_str, param_value_num
    """
    rows: list[dict[str, Any]] = []
    for rec in trials.itertuples(index=False):
        params = getattr(rec, "params", None)
        if not isinstance(params, dict):
            continue
        for k, v in params.items():
            num, s = _coerce_param_value(v)
            rows.append(
                {
                    "experiment": getattr(rec, "experiment"),
                    "trial_id": getattr(rec, "trial_id"),
                    "is_stage2_experiment": bool(getattr(rec, "is_stage2_experiment")),
                    "param_key": str(k),
                    "param_value_str": str(s),
                    "param_value_num": num,
                }
            )
    return pd.DataFrame(rows)


def _write_param_importance_md(
    *,
    out_path: Path,
    trials: pd.DataFrame,
    params_long: pd.DataFrame,
    metric: str,
    mode: str,
) -> None:
    """
    Write a lightweight, explainable importance report.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    mode_n = str(mode).strip().lower()
    is_min = mode_n == "min"

    # Dataset choice: focus on non-stage2 experiments (stage1) because stage2 trials pack params differently.
    base = trials[(trials["is_stage2_experiment"] == False)].copy()  # noqa: E712
    base = base[base["best_value"].notna()].copy()

    lines: list[str] = []
    lines.append("# Tune parameter importance (offline)\n")
    lines.append(f"- results_root: `{trials.attrs.get('results_root', '')}`\n")
    lines.append(f"- metric_analyzed: `{metric}` (mode: `{mode_n}`)\n")
    lines.append(f"- trials_total: {len(trials)}\n")
    lines.append(f"- stage1_trials_with_metric: {len(base)}\n")
    lines.append("\n---\n")

    if len(base) < 30:
        lines.append("Not enough stage-1 trials with this metric to compute stable importance.\n")
        out_path.write_text("".join(lines), encoding="utf-8")
        return

    # Keep params present in enough trials.
    params_stage1 = params_long[params_long["is_stage2_experiment"] == False].copy()  # noqa: E712
    freq = params_stage1.groupby("param_key")["trial_id"].nunique().sort_values(ascending=False)
    n_trials = base["trial_id"].nunique()
    keep_keys = [k for k, c in freq.items() if (c / max(1, n_trials)) >= 0.25]
    keep_keys = keep_keys[:80]  # cap to keep feature matrix sane

    lines.append("## Coverage-filtered parameters\n")
    lines.append(f"- unique_stage1_trials: {n_trials}\n")
    lines.append(f"- params_kept (coverage>=25%): {len(keep_keys)}\n\n")

    # Build wide feature frame for sklearn
    # Convert params to stable tokens
    feat_rows: list[dict[str, Any]] = []
    for rec in base.itertuples(index=False):
        params = rec.params if isinstance(rec.params, dict) else {}
        row: dict[str, Any] = {"trial_id": rec.trial_id, "y": float(rec.best_value)}
        for k in keep_keys:
            if k not in params:
                row[k] = None
            else:
                num, s = _coerce_param_value(params.get(k))
                row[k] = num if num is not None else s
        feat_rows.append(row)
    feat = pd.DataFrame(feat_rows).dropna(subset=["y"]).reset_index(drop=True)

    # Simple univariate stats
    lines.append("## Simple univariate signals (binned means + Spearman)\n\n")
    uni_rows: list[dict[str, Any]] = []
    for k in keep_keys:
        col = feat[k]
        if col.isna().all():
            continue
        # numeric?
        if pd.api.types.is_numeric_dtype(col):
            x = col.to_numpy(dtype=float)
            y = feat["y"].to_numpy(dtype=float)
            m = np.isfinite(x)
            if int(m.sum()) < 10:
                continue
            rho = _spearmanr(x[m], y[m])
            uni_rows.append(
                {
                    "param": k,
                    "type": "numeric",
                    "n": int(m.sum()),
                    "spearman_r": (None if rho is None else float(rho)),
                }
            )
        else:
            # categorical: group means by value
            vc = col.value_counts(dropna=True)
            if len(vc) <= 1:
                continue
            top_vals = vc.head(8).index.tolist()
            grp = (
                feat.loc[col.isin(top_vals)]
                .groupby(k)["y"]
                .agg(["count", "mean"])
                .reset_index()
                .sort_values("mean", ascending=is_min)
            )
            # keep a short string summary
            parts = []
            for _i, rr in grp.iterrows():
                try:
                    val = rr[k]
                except Exception:
                    val = None
                parts.append(f"{val} (n={int(rr['count'])}, mean={float(rr['mean']):.4g})")
            uni_rows.append({"param": k, "type": "categorical", "n": int(vc.sum()), "top_groups": "; ".join(parts[:6])})

    uni_df = pd.DataFrame(uni_rows)
    if not uni_df.empty:
        # Sort numeric by abs Spearman
        num_df = uni_df[uni_df["type"] == "numeric"].copy()
        if not num_df.empty:
            num_df["abs_spearman"] = num_df["spearman_r"].abs()
            num_df = num_df.sort_values("abs_spearman", ascending=False).head(20)
            lines.append("### Top numeric params by |Spearman|\n\n")
            for r in num_df.itertuples(index=False):
                lines.append(f"- `{r.param}`: spearman={r.spearman_r}, n={r.n}\n")
            lines.append("\n")

            # Add coarse binned means for the strongest numeric params (simple, explainable).
            lines.append("### Binned means (numeric params, 5-quantile bins)\n\n")
            top_for_bins = [str(x) for x in num_df["param"].head(8).tolist()]
            for k in top_for_bins:
                if k not in feat.columns:
                    continue
                dfk = feat[[k, "y"]].dropna().copy()
                if len(dfk) < 30:
                    continue
                try:
                    bins = pd.qcut(dfk[k], q=5, duplicates="drop")
                except Exception:
                    continue
                grp = dfk.groupby(bins)["y"].agg(["count", "mean"]).reset_index()
                grp = grp.sort_values("mean", ascending=is_min)
                lines.append(f"- `{k}`\n")
                for _j, rr in grp.iterrows():
                    lines.append(f"  - {rr.iloc[0]}: n={int(rr['count'])}, mean={float(rr['mean']):.4g}\n")
            lines.append("\n")

        cat_df = uni_df[uni_df["type"] == "categorical"].copy()
        if not cat_df.empty:
            cat_df = cat_df.sort_values("n", ascending=False).head(15)
            lines.append("### High-coverage categorical params (top value group means)\n\n")
            for r in cat_df.itertuples(index=False):
                lines.append(f"- `{r.param}`: {r.top_groups}\n")
            lines.append("\n")
    else:
        lines.append("No univariate signals computed (insufficient variation).\n\n")

    # Permutation importance (sklearn)
    try:
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.inspection import permutation_importance
        from sklearn.model_selection import train_test_split
        from sklearn.pipeline import Pipeline
        from sklearn.compose import ColumnTransformer
        from sklearn.preprocessing import OneHotEncoder
    except Exception as e:
        lines.append(f"## Permutation importance\n\nSkipped (sklearn import failed: {e}).\n")
        out_path.write_text("".join(lines), encoding="utf-8")
        return

    X = feat.drop(columns=["trial_id", "y"])
    y = feat["y"].to_numpy(dtype=float)

    # Separate numeric vs categorical
    num_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    cat_cols = [c for c in X.columns if c not in num_cols]

    pre = ColumnTransformer(
        transformers=[
            ("num", "passthrough", num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
        ],
        remainder="drop",
    )

    model = RandomForestRegressor(
        n_estimators=300,
        random_state=42,
        n_jobs=-1,
        min_samples_leaf=2,
    )

    pipe = Pipeline([("pre", pre), ("model", model)])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    pipe.fit(X_train, y_train)

    # Permutation importance on raw columns (before OHE) isn't directly supported when pipeline expands.
    # We therefore compute importance on *input columns* by permuting each column in X_test and scoring.
    def score_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return float(np.mean(np.abs(y_true - y_pred)))

    base_pred = pipe.predict(X_test)
    base_mae = score_mae(y_test, base_pred)

    rng = np.random.RandomState(42)
    importances: list[tuple[str, float]] = []
    for col in X_test.columns:
        deltas: list[float] = []
        for _ in range(5):
            Xp = X_test.copy()
            Xp[col] = rng.permutation(Xp[col].to_numpy())
            pred_p = pipe.predict(Xp)
            mae_p = score_mae(y_test, pred_p)
            deltas.append(mae_p - base_mae)
        importances.append((col, float(np.mean(deltas))))

    importances.sort(key=lambda x: x[1], reverse=True)

    lines.append("## Permutation importance (RandomForest, MAE increase)\n\n")
    lines.append(f"- baseline_MAE: {base_mae:.6f}\n")
    lines.append(f"- test_size: {len(X_test)}\n\n")
    for k, imp in importances[:25]:
        lines.append(f"- `{k}`: +MAE {imp:.6f}\n")

    out_path.write_text("".join(lines), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-root", type=str, default="/media/dl/dataset/ray_results", help="Ray results root dir")
    ap.add_argument("--output-root", type=str, default="", help="Output root (default: <repo>/outputs/tune_analysis)")
    ap.add_argument("--metric", type=str, default="val_loss_5d_weighted", help="Metric to analyze (from result.json)")
    ap.add_argument("--mode", type=str, default="min", choices=["min", "max"], help="Optimization direction for metric")
    ap.add_argument("--default-min-epoch", type=int, default=1, help="Fallback min_epoch when not inferable")
    ap.add_argument("--write-parquet", action="store_true", default=True)
    ap.add_argument("--no-parquet", action="store_true", default=False)
    ap.add_argument("--verbose", action="store_true", default=False)
    args = ap.parse_args()

    repo_root = resolve_repo_root()
    results_root = Path(str(args.results_root)).expanduser().resolve()
    if not results_root.is_dir():
        print(f"[ERR] results root does not exist or is not a directory: {results_root}", file=sys.stderr)
        return 2

    out_root = Path(str(args.output_root)).expanduser().resolve() if str(args.output_root).strip() else (repo_root / "outputs" / "tune_analysis")
    out_root.mkdir(parents=True, exist_ok=True)

    metric = str(args.metric).strip()
    mode = str(args.mode).strip().lower()
    default_min_epoch = int(args.default_min_epoch)

    # Compose tune config index (for min_epoch inference / metadata)
    tune_cfg_index = _load_tune_cfg_index(repo_root)

    exp_dirs = [p for p in sorted(results_root.iterdir()) if p.is_dir()]
    trial_rows: list[dict[str, Any]] = []

    for exp_dir in exp_dirs:
        exp_name = exp_dir.name
        is_stage2 = _is_stage2_experiment_name(exp_name)
        info = tune_cfg_index.get(exp_name)
        min_epoch = _infer_min_epoch_for_experiment(exp_dir, exp_name=exp_name, tune_cfg_index=tune_cfg_index, default_min_epoch=default_min_epoch)

        trial_dirs = _iter_trial_dirs(exp_dir)
        if args.verbose:
            print(f"[INFO] {exp_name}: trials={len(trial_dirs)} min_epoch={min_epoch} stage2={is_stage2}")

        for td in trial_dirs:
            try:
                params = _read_params_json(td)
            except Exception:
                params = {}

            summ = _summarize_result_jsonl(td, metric=metric, mode=mode, min_epoch=min_epoch)
            tid = summ.trial_id_from_rows or _guess_trial_id_from_dir(td)
            if not tid:
                tid = td.name

            trial_rows.append(
                {
                    "experiment": exp_name,
                    "trial_id": tid,
                    "trial_dir": str(td),
                    "is_stage2_experiment": bool(is_stage2),
                    "min_epoch_used": int(min_epoch),
                    "metric": metric,
                    "mode": mode,
                    "best_value": summ.best_value,
                    "best_epoch": summ.best_epoch,
                    "best_done": summ.best_done,
                    "last_value": summ.last_value,
                    "last_epoch": summ.last_epoch,
                    "last_done": summ.last_done,
                    "n_rows_scanned": int(summ.n_rows_scanned),
                    "n_rows_with_metric": int(summ.n_rows_with_metric),
                    "params": params,
                    # Optional metadata from conf if available
                    "conf_config_name": (info.config_name if info else None),
                    "conf_tune_metric": (info.tune_metric if info else None),
                    "conf_tune_mode": (info.tune_mode if info else None),
                    "conf_scheduler_type": (info.scheduler_type if info else None),
                    "conf_grace_period": (info.grace_period if info else None),
                    "conf_two_stage_enabled": (info.two_stage_enabled if info else None),
                    "conf_two_stage_min_epoch": (info.two_stage_min_epoch if info else None),
                    "conf_two_stage_top_n": (info.two_stage_top_n if info else None),
                    "conf_two_stage_stage2_metric": (info.two_stage_stage2_metric if info else None),
                    "conf_two_stage_stage2_mode": (info.two_stage_stage2_mode if info else None),
                }
            )

    trials = pd.DataFrame(trial_rows)
    trials.attrs["results_root"] = str(results_root)

    # Write trials parquet + params parquet
    parquet_ok = bool(args.write_parquet) and (not bool(args.no_parquet))
    parquet_dir = out_root / "trials_parquet"
    parquet_dir.mkdir(parents=True, exist_ok=True)

    # Save params in a long table (more stable than a wide schema).
    params_long = _build_param_table(trials)

    # Avoid storing raw dict in parquet (keep a stable JSON string for the trials table)
    trials_for_disk = trials.copy()
    trials_for_disk["params_json"] = trials_for_disk["params"].apply(_json_dumps_stable)
    trials_for_disk = trials_for_disk.drop(columns=["params"])

    if parquet_ok:
        trials_for_disk.to_parquet(parquet_dir / "trials.parquet", index=False)
        params_long.to_parquet(parquet_dir / "params_long.parquet", index=False)

    # Experiment summary
    def _best_agg(s: pd.Series) -> Optional[float]:
        s2 = s.dropna()
        if s2.empty:
            return None
        return float(s2.min()) if mode == "min" else float(s2.max())

    summary = (
        trials.groupby(["experiment", "is_stage2_experiment"], dropna=False)
        .agg(
            trials_total=("trial_id", "count"),
            trials_with_metric=("best_value", lambda x: int(pd.Series(x).notna().sum())),
            best_of_best=("best_value", _best_agg),
            best_of_last=("last_value", _best_agg),
            median_best=("best_value", lambda x: float(pd.Series(x).dropna().median()) if pd.Series(x).dropna().size else np.nan),
            p10_best=("best_value", lambda x: float(pd.Series(x).dropna().quantile(0.10)) if pd.Series(x).dropna().size else np.nan),
            p90_best=("best_value", lambda x: float(pd.Series(x).dropna().quantile(0.90)) if pd.Series(x).dropna().size else np.nan),
            min_epoch_used=("min_epoch_used", lambda x: int(pd.Series(x).dropna().min()) if pd.Series(x).dropna().size else default_min_epoch),
            conf_config_name=("conf_config_name", lambda x: pd.Series(x).dropna().iloc[0] if pd.Series(x).dropna().size else None),
            conf_tune_metric=("conf_tune_metric", lambda x: pd.Series(x).dropna().iloc[0] if pd.Series(x).dropna().size else None),
            conf_tune_mode=("conf_tune_mode", lambda x: pd.Series(x).dropna().iloc[0] if pd.Series(x).dropna().size else None),
            conf_two_stage_enabled=("conf_two_stage_enabled", lambda x: pd.Series(x).dropna().iloc[0] if pd.Series(x).dropna().size else None),
        )
        .reset_index()
    )

    # Link stage1 -> stage2 best when the conventional suffix directory exists.
    stage2_map: dict[str, str] = {}
    for r in summary.itertuples(index=False):
        if bool(r.is_stage2_experiment):
            continue
        stage2_name = f"{r.experiment}-stage2-kfold"
        if (results_root / stage2_name).is_dir():
            stage2_map[str(r.experiment)] = stage2_name

    stage2_best_lookup: dict[str, Optional[float]] = {}
    for stage1, stage2 in stage2_map.items():
        s2 = summary[(summary["experiment"] == stage2)]
        if len(s2) >= 1:
            stage2_best_lookup[stage1] = s2.iloc[0]["best_of_best"]
        else:
            stage2_best_lookup[stage1] = None

    summary["linked_stage2_experiment"] = summary["experiment"].map(lambda n: stage2_map.get(str(n), None))
    summary["linked_stage2_best_of_best"] = summary["experiment"].map(lambda n: stage2_best_lookup.get(str(n), None))

    summary_path = out_root / "experiments_summary.csv"
    summary.to_csv(summary_path, index=False)

    # Param importance markdown
    _write_param_importance_md(
        out_path=out_root / "param_importance.md",
        trials=trials,
        params_long=params_long,
        metric=metric,
        mode=mode,
    )

    print(f"[OK] wrote: {summary_path}")
    if parquet_ok:
        print(f"[OK] wrote: {parquet_dir / 'trials.parquet'}")
        print(f"[OK] wrote: {parquet_dir / 'params_long.parquet'}")
    print(f"[OK] wrote: {out_root / 'param_importance.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

