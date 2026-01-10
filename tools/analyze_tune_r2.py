#!/usr/bin/env python3
from __future__ import annotations

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


def _as_path_under_root(p: str, *, repo_root: Path) -> Path:
    pp = Path(str(p)).expanduser()
    if not pp.is_absolute():
        pp = (repo_root / pp).resolve()
    return pp


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


def _guess_trial_id_from_dir(trial_dir: Path) -> str:
    """
    Best-effort parse trial id from directory name:
      trainable_<exp>_<trialid>_<idx>_... -> <exp>_<trialid>
    """
    name = trial_dir.name
    if name.startswith("trainable_"):
        parts = name.split("_")
        # Example: trainable_d9a15_00057_57_...
        if len(parts) >= 3:
            return f"{parts[1]}_{parts[2]}"
    return ""


def _read_params(trial_dir: Path) -> dict[str, Any]:
    p = trial_dir / "params.json"
    with open(p, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, dict):
        raise TypeError(f"params.json must be a dict, got: {type(obj)} ({p})")
    return obj


def _iter_trial_dirs(exp_dir: Path) -> list[Path]:
    out: list[Path] = []
    for p in sorted(exp_dir.iterdir()):
        if not p.is_dir():
            continue
        if (p / "result.json").is_file() and (p / "params.json").is_file():
            out.append(p)
    return out


def _read_result_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
            except Exception:
                continue
            if isinstance(obj, dict):
                records.append(obj)
    return records


def _normalize_param_value(v: Any) -> Any:
    if isinstance(v, list):
        # Make hashable / groupable.
        return tuple(_normalize_param_value(x) for x in v)
    if isinstance(v, dict):
        # Stable string representation for grouping.
        try:
            return json.dumps(v, sort_keys=True, ensure_ascii=False)
        except Exception:
            return str(v)
    return v


def _compute_swa_start_epoch(max_epochs: int, swa_epoch_start: float | int) -> int:
    """
    Mirror Lightning's convention for `swa_epoch_start`:
      - if swa_epoch_start is a float in [0, 1), treat it as a fraction of max_epochs
      - otherwise treat it as an absolute (0-based) epoch index
    """
    try:
        v = float(swa_epoch_start)
    except Exception:
        try:
            return int(swa_epoch_start)  # type: ignore[arg-type]
        except Exception:
            return 0

    if v < 1.0:
        if not (max_epochs > 0):
            return 0
        if v <= 0.0:
            return 0
        return int(v * float(max_epochs))
    return int(v)


@dataclass(frozen=True)
class TuneEnv:
    exp_dir: Path
    metric: str
    mode: str
    max_epochs: int
    swa_enabled: bool
    swa_start_epoch0: int  # 0-based
    swa_start_epoch1: int  # 1-based (Ray-reported "epoch")
    grace_period: Optional[int]


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


def _build_env(
    *,
    repo_root: Path,
    config_name: str,
    hydra_overrides: list[str],
    exp_dir_arg: str,
    metric_arg: str,
    mode_arg: str,
) -> TuneEnv:
    cfg = _compose_hydra_cfg(repo_root, config_name=config_name, overrides=hydra_overrides)
    tune_cfg = dict(cfg.get("tune", {}) or {})
    trainer_cfg = dict(cfg.get("trainer", {}) or {})

    metric = str(metric_arg).strip() or str(tune_cfg.get("metric", "val_r2")).strip()
    mode = str(mode_arg).strip() or str(tune_cfg.get("mode", "max")).strip()

    if str(exp_dir_arg).strip():
        exp_dir = Path(str(exp_dir_arg).strip()).expanduser().resolve()
    else:
        name = str(tune_cfg.get("name", "tune")).strip()
        storage_path = _as_path_under_root(str(tune_cfg.get("storage_path", "ray_results")), repo_root=repo_root)
        exp_dir = (storage_path / name).resolve()

    max_epochs = int(trainer_cfg.get("max_epochs", 0) or 0)
    swa_cfg = dict(trainer_cfg.get("swa", {}) or {})
    swa_enabled = bool(swa_cfg.get("enabled", False))
    swa_epoch_start = swa_cfg.get("swa_epoch_start", 0.8)
    swa_start_epoch0 = _compute_swa_start_epoch(max_epochs, swa_epoch_start) if swa_enabled else max_epochs + 1
    swa_start_epoch1 = int(swa_start_epoch0) + 1

    grace: Optional[int] = None
    sched = dict(tune_cfg.get("scheduler", {}) or {})
    if str(sched.get("type", "")).strip().lower() == "asha":
        grace = _safe_int(sched.get("grace_period"))

    return TuneEnv(
        exp_dir=exp_dir,
        metric=metric,
        mode=str(mode).strip().lower(),
        max_epochs=max_epochs,
        swa_enabled=bool(swa_enabled),
        swa_start_epoch0=int(swa_start_epoch0),
        swa_start_epoch1=int(swa_start_epoch1),
        grace_period=grace,
    )


def _pick_better(a: float, b: float, mode: str) -> float:
    if mode == "min":
        return a if a <= b else b
    return a if a >= b else b


def _trial_summary_from_records(
    *,
    trial_dir: Path,
    trial_id_fallback: str,
    records: list[dict[str, Any]],
    env: TuneEnv,
    min_epoch_for_best: int,
) -> dict[str, Any]:
    # Note: Ray Tune may write multiple records per reported epoch (e.g., sanity check + real epoch 0 val).
    best_val: Optional[float] = None
    best_epoch: Optional[int] = None
    best_iter: Optional[int] = None

    best_pre_swa: Optional[float] = None
    best_pre_swa_epoch: Optional[int] = None
    best_pre_swa_iter: Optional[int] = None

    best_post_swa: Optional[float] = None
    best_post_swa_epoch: Optional[int] = None
    best_post_swa_iter: Optional[int] = None

    last_val: Optional[float] = None
    last_epoch: Optional[int] = None
    last_iter: Optional[int] = None
    last_done: Optional[bool] = None
    last_time_total_s: Optional[float] = None

    n_metric = 0
    n_records = 0

    trial_id: str = ""
    for rec in records:
        if not isinstance(rec, dict):
            continue
        n_records += 1
        if not trial_id:
            tid = str(rec.get("trial_id", "") or "").strip()
            if tid:
                trial_id = tid

        ep = _safe_int(rec.get("epoch"))
        it = _safe_int(rec.get("training_iteration"))
        done_raw = rec.get("done", None)
        done = bool(done_raw) if done_raw is not None else None
        tts = _safe_float(rec.get("time_total_s"))

        v = _safe_float(rec.get(env.metric))
        if v is not None:
            n_metric += 1
            # last observed metric in file order
            last_val, last_epoch, last_iter = float(v), ep, it
            last_done = done if done is not None else last_done
            last_time_total_s = tts if tts is not None else last_time_total_s

            if ep is not None and ep < int(min_epoch_for_best):
                # keep last, but ignore for "best" to reduce early noise
                continue

            if best_val is None:
                best_val, best_epoch, best_iter = float(v), ep, it
            else:
                better = (v < best_val) if env.mode == "min" else (v > best_val)
                if better:
                    best_val, best_epoch, best_iter = float(v), ep, it

            # Pre/Post SWA split uses Ray's 1-based epoch convention.
            if env.swa_enabled and ep is not None and ep >= int(env.swa_start_epoch1):
                if best_post_swa is None:
                    best_post_swa, best_post_swa_epoch, best_post_swa_iter = float(v), ep, it
                else:
                    better_post = (v < best_post_swa) if env.mode == "min" else (v > best_post_swa)
                    if better_post:
                        best_post_swa, best_post_swa_epoch, best_post_swa_iter = float(v), ep, it
            else:
                if best_pre_swa is None:
                    best_pre_swa, best_pre_swa_epoch, best_pre_swa_iter = float(v), ep, it
                else:
                    better_pre = (v < best_pre_swa) if env.mode == "min" else (v > best_pre_swa)
                    if better_pre:
                        best_pre_swa, best_pre_swa_epoch, best_pre_swa_iter = float(v), ep, it

        else:
            # Still track last epoch/done even if metric missing in early records.
            if ep is not None:
                last_epoch = ep
            if it is not None:
                last_iter = it
            if done is not None:
                last_done = done
            if tts is not None:
                last_time_total_s = tts

    if not trial_id:
        trial_id = trial_id_fallback

    # Degradation metrics (for max objective, positive means "got worse by the end").
    delta_best_to_last: Optional[float] = None
    delta_pre_swa_best_to_last: Optional[float] = None
    if best_val is not None and last_val is not None:
        delta_best_to_last = float(best_val - last_val) if env.mode == "max" else float(last_val - best_val)
    if best_pre_swa is not None and last_val is not None:
        delta_pre_swa_best_to_last = (
            float(best_pre_swa - last_val) if env.mode == "max" else float(last_val - best_pre_swa)
        )

    delta_pre_to_post_best: Optional[float] = None
    if best_pre_swa is not None and best_post_swa is not None:
        delta_pre_to_post_best = (
            float(best_pre_swa - best_post_swa) if env.mode == "max" else float(best_post_swa - best_pre_swa)
        )

    return {
        "trial_id": trial_id,
        "trial_dir": str(trial_dir),
        "n_records": int(n_records),
        "n_metric_records": int(n_metric),
        "done": (None if last_done is None else bool(last_done)),
        "last_epoch": (None if last_epoch is None else int(last_epoch)),
        "last_training_iteration": (None if last_iter is None else int(last_iter)),
        "time_total_s": (None if last_time_total_s is None else float(last_time_total_s)),
        f"{env.metric}_best": (None if best_val is None else float(best_val)),
        f"{env.metric}_best_epoch": (None if best_epoch is None else int(best_epoch)),
        f"{env.metric}_best_iter": (None if best_iter is None else int(best_iter)),
        f"{env.metric}_last": (None if last_val is None else float(last_val)),
        f"{env.metric}_last_epoch": (None if last_epoch is None else int(last_epoch)),
        f"{env.metric}_pre_swa_best": (None if best_pre_swa is None else float(best_pre_swa)),
        f"{env.metric}_pre_swa_best_epoch": (None if best_pre_swa_epoch is None else int(best_pre_swa_epoch)),
        f"{env.metric}_post_swa_best": (None if best_post_swa is None else float(best_post_swa)),
        f"{env.metric}_post_swa_best_epoch": (None if best_post_swa_epoch is None else int(best_post_swa_epoch)),
        "delta_best_to_last": (None if delta_best_to_last is None else float(delta_best_to_last)),
        "delta_pre_swa_best_to_last": (
            None if delta_pre_swa_best_to_last is None else float(delta_pre_swa_best_to_last)
        ),
        "delta_pre_swa_best_minus_post_swa_best": (
            None if delta_pre_to_post_best is None else float(delta_pre_to_post_best)
        ),
    }


def _infer_param_columns(df: pd.DataFrame) -> list[str]:
    fixed = {
        "trial_id",
        "trial_dir",
        "n_records",
        "n_metric_records",
        "done",
        "last_epoch",
        "last_training_iteration",
        "time_total_s",
        "delta_best_to_last",
        "delta_pre_swa_best_to_last",
        "delta_pre_swa_best_minus_post_swa_best",
    }
    return [c for c in df.columns if c not in fixed and not c.startswith("val_")]


def _param_effect_table(df: pd.DataFrame, *, y_col: str, param_cols: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    y = pd.to_numeric(df[y_col], errors="coerce")
    for col in param_cols:
        s = df[col]
        if s.dropna().empty:
            continue
        nunique = int(s.nunique(dropna=True))

        # Heuristic: treat small-cardinality as categorical.
        is_categorical = (s.dtype == object) or nunique <= 12
        if is_categorical:
            g = df.groupby(col, dropna=True)[y_col].agg(["count", "mean", "median", "max", "min"])
            g = g.sort_values("mean", ascending=False)
            if g.empty:
                continue
            best_val = g.index[0]
            worst_val = g.index[-1]
            delta_mean = float(g["mean"].iloc[0] - g["mean"].iloc[-1])
            rows.append(
                {
                    "param": col,
                    "kind": "categorical",
                    "n_unique": nunique,
                    "best_value": str(best_val),
                    "worst_value": str(worst_val),
                    "mean_best": float(g["mean"].iloc[0]),
                    "mean_worst": float(g["mean"].iloc[-1]),
                    "delta_mean": delta_mean,
                    "n_best": int(g["count"].iloc[0]),
                }
            )
        else:
            x = pd.to_numeric(s, errors="coerce")
            if x.dropna().empty or y.dropna().empty:
                continue
            # Spearman correlation is more robust than Pearson for loguniform params.
            x_eff = x.copy()
            if bool((x_eff > 0).all()):
                x_eff = np.log10(x_eff)
            corr = pd.concat([x_eff, y], axis=1).corr(method="spearman").iloc[0, 1]
            rows.append(
                {
                    "param": col,
                    "kind": "numeric",
                    "n_unique": nunique,
                    "spearman": (None if not np.isfinite(corr) else float(corr)),
                    "x_log10": bool((x > 0).all()),
                    "x_min": (None if x.dropna().empty else float(x.min())),
                    "x_max": (None if x.dropna().empty else float(x.max())),
                }
            )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    # Rank by absolute correlation (numeric) or delta_mean (categorical).
    out["score"] = 0.0
    out.loc[out["kind"] == "categorical", "score"] = out.loc[out["kind"] == "categorical", "delta_mean"].abs()
    out.loc[out["kind"] == "numeric", "score"] = out.loc[out["kind"] == "numeric", "spearman"].abs()
    out = out.sort_values("score", ascending=False)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze a Ray Tune experiment directory for R^2-oriented model selection.\n\n"
            "This script scans per-trial params.json + result.json (JSONL) and summarizes:\n"
            "  - best/last metric, pre-SWA best, post-SWA best\n"
            "  - SWA-related degradation (peak->last)\n"
            "  - simple hyperparam effect ranking\n\n"
            "It is designed for the CSIRO repo Tune outputs under tune.storage_path/tune.name."
        )
    )
    parser.add_argument("--config-name", type=str, default="tune_vitdet", help="Hydra config name under conf/.")
    parser.add_argument(
        "--exp-dir",
        type=str,
        default="",
        help=(
            "Ray Tune experiment directory. If omitted, derived from composed config: tune.storage_path/tune.name."
        ),
    )
    parser.add_argument("--metric", type=str, default="val_r2", help="Metric key in result.json to optimize.")
    parser.add_argument("--mode", type=str, default="max", choices=["min", "max"], help="min|max")
    parser.add_argument(
        "--min-epoch",
        type=int,
        default=1,
        help="Ignore records with epoch < min_epoch when computing best metrics (still keeps last).",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="",
        help="Output directory. Default: <exp-dir>/analysis_r2",
    )

    args, hydra_overrides = parser.parse_known_args()

    repo_root = resolve_repo_root().resolve()
    sys.path.insert(0, str(repo_root))

    env = _build_env(
        repo_root=repo_root,
        config_name=str(args.config_name),
        hydra_overrides=hydra_overrides,
        exp_dir_arg=str(args.exp_dir),
        metric_arg=str(args.metric),
        mode_arg=str(args.mode),
    )

    exp_dir = env.exp_dir
    if not exp_dir.exists():
        raise SystemExit(f"Experiment directory does not exist: {exp_dir}")

    trial_dirs = _iter_trial_dirs(exp_dir)
    if not trial_dirs:
        raise SystemExit(f"No trial dirs found under: {exp_dir} (expected params.json + result.json)")

    out_dir = Path(str(args.out_dir).strip()).expanduser().resolve() if str(args.out_dir).strip() else exp_dir / "analysis_r2"
    out_dir.mkdir(parents=True, exist_ok=True)

    trial_rows: list[dict[str, Any]] = []
    for td in trial_dirs:
        trial_id_guess = _guess_trial_id_from_dir(td)
        try:
            params = _read_params(td)
        except Exception:
            params = {}
        params = {str(k): _normalize_param_value(v) for k, v in (params or {}).items()}

        rec_path = td / "result.json"
        recs = _read_result_jsonl(rec_path)
        summ = _trial_summary_from_records(
            trial_dir=td,
            trial_id_fallback=trial_id_guess,
            records=recs,
            env=env,
            min_epoch_for_best=int(args.min_epoch),
        )
        summ.update(params)
        trial_rows.append(summ)

    df = pd.DataFrame(trial_rows)
    # Sort by chosen objective (best pre-SWA when available, else best overall).
    pre_best_col = f"{env.metric}_pre_swa_best"
    best_col = f"{env.metric}_best"
    sort_col = pre_best_col if pre_best_col in df.columns else best_col
    df_sorted = df.sort_values(sort_col, ascending=(env.mode == "min"), na_position="last")

    # Choose target for param effects.
    y_col = pre_best_col if df_sorted[pre_best_col].notna().any() else best_col
    # Infer from params.json keys: columns that start with "model."/"peft."/"optimizer."/"scheduler."
    param_cols = [
        c
        for c in df.columns
        if any(str(c).startswith(p) for p in ("model.", "peft.", "optimizer.", "scheduler."))
    ]
    param_cols = sorted(set(param_cols))

    effects = _param_effect_table(df_sorted, y_col=y_col, param_cols=param_cols)

    # Persist outputs.
    summary_csv = out_dir / "trial_summary.csv"
    df_sorted.to_csv(summary_csv, index=False)
    effects_csv = out_dir / "param_effects.csv"
    effects.to_csv(effects_csv, index=False)

    # Human-readable report (markdown).
    report_md = out_dir / "report.md"
    top_k = 15
    with open(report_md, "w", encoding="utf-8") as f:
        f.write("# Ray Tune R^2 analysis\n\n")
        f.write(f"- exp_dir: `{exp_dir}`\n")
        f.write(f"- metric/mode: **{env.metric} / {env.mode}**\n")
        f.write(f"- trainer.max_epochs: **{env.max_epochs}**\n")
        if env.swa_enabled:
            f.write(f"- SWA: enabled, start_epoch (0-based)={env.swa_start_epoch0}, reported epoch>={env.swa_start_epoch1}\n")
        else:
            f.write("- SWA: disabled\n")
        if env.grace_period is not None:
            f.write(f"- ASHA grace_period: **{env.grace_period}**\n")
        f.write(f"- min_epoch_for_best: **{int(args.min_epoch)}**\n\n")

        f.write(f"## Top {top_k} trials by `{y_col}`\n\n")
        cols_show = [
            "trial_id",
            y_col,
            f"{env.metric}_best_epoch" if f"{env.metric}_best_epoch" in df_sorted.columns else None,
            f"{env.metric}_last" if f"{env.metric}_last" in df_sorted.columns else None,
            "delta_pre_swa_best_to_last" if "delta_pre_swa_best_to_last" in df_sorted.columns else None,
            "last_epoch",
            "done",
        ]
        cols_show = [c for c in cols_show if c]
        f.write(df_sorted[cols_show].head(top_k).to_markdown(index=False))
        f.write("\n\n")

        if "delta_pre_swa_best_to_last" in df_sorted.columns:
            d = pd.to_numeric(df_sorted["delta_pre_swa_best_to_last"], errors="coerce")
            f.write("## SWA/late-epoch degradation (peak pre-SWA -> last)\n\n")
            f.write(
                "- trials with metric available: **{} / {}**\n".format(
                    int(df_sorted[best_col].notna().sum()), int(len(df_sorted))
                )
            )
            if d.dropna().shape[0] > 0:
                f.write(
                    "- delta_pre_swa_best_to_last ({} objective): mean={:.6f}, median={:.6f}, p90={:.6f}, max={:.6f}\n\n".format(
                        env.mode,
                        float(d.mean()),
                        float(d.median()),
                        float(d.quantile(0.9)),
                        float(d.max()),
                    )
                )
            else:
                f.write("- Not enough data to compute degradation stats.\n\n")

        if not effects.empty:
            f.write("## Hyperparameter effects (rough ranking)\n\n")
            f.write(
                "This is a lightweight ranking: categorical params use Δ(mean) across values; numeric params use |Spearman|.\n\n"
            )
            f.write(effects.head(25).to_markdown(index=False))
            f.write("\n\n")

        f.write("## Files written\n\n")
        f.write(f"- `{summary_csv}`\n")
        f.write(f"- `{effects_csv}`\n")
        f.write(f"- `{report_md}`\n")

    print(f"[analyze_tune_r2] exp_dir: {exp_dir}")
    print(f"[analyze_tune_r2] trials: {len(trial_dirs)}")
    print(f"[analyze_tune_r2] wrote: {summary_csv}")
    print(f"[analyze_tune_r2] wrote: {effects_csv}")
    print(f"[analyze_tune_r2] wrote: {report_md}")


if __name__ == "__main__":
    main()


