from __future__ import annotations

import copy
import os
import sys
from pathlib import Path
from typing import Any, Optional

import hydra
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from src.training.entrypoint import resolve_repo_root, run_training


def _set_by_dotted_path(d: dict, path: str, value: Any) -> None:
    parts = [p for p in str(path).split(".") if p]
    if not parts:
        return
    cur: Any = d
    for p in parts[:-1]:
        if not isinstance(cur, dict):
            return
        if p not in cur or not isinstance(cur[p], dict):
            cur[p] = {}
        cur = cur[p]
    if isinstance(cur, dict):
        cur[parts[-1]] = value


def _as_path_under_root(p: str, *, repo_root: Path) -> Path:
    pp = Path(str(p)).expanduser()
    if not pp.is_absolute():
        pp = (repo_root / pp).resolve()
    return pp


def _build_search_space(spec: dict) -> dict:
    from ray import tune

    out: dict[str, Any] = {}
    for k, v in (spec or {}).items():
        if not isinstance(v, dict):
            raise ValueError(f"search_space['{k}'] must be a dict, got: {type(v)}")
        t = str(v.get("type", "") or "").strip().lower()
        if t == "choice":
            vals = v.get("values", None)
            if not isinstance(vals, list) or len(vals) == 0:
                raise ValueError(f"choice space for '{k}' requires non-empty list 'values'")
            out[str(k)] = tune.choice(vals)
        elif t == "uniform":
            out[str(k)] = tune.uniform(float(v["lower"]), float(v["upper"]))
        elif t == "loguniform":
            out[str(k)] = tune.loguniform(float(v["lower"]), float(v["upper"]))
        elif t == "randint":
            # [lower, upper)
            out[str(k)] = tune.randint(int(v["lower"]), int(v["upper"]))
        else:
            raise ValueError(f"Unsupported search space type for '{k}': {t!r}")
    return out


def _try_extract_final_metrics(
    log_dir: Path,
) -> tuple[Optional[int], dict[str, float]]:
    """
    Best-effort: read Lightning CSVLogger metrics.csv under log_dir and return:
      - epoch_end (1-based epochs completed)
      - a dict of final validation metrics (keys match Lightning log names)

    NOTE: Lightning's CSVLogger writes multiple rows per epoch (e.g., one with val_* fields
    and another with train_* fields). We therefore select the last row where `val_loss`
    is present, rather than naïvely taking the last row for an epoch.
    """
    try:
        import pandas as pd
    except Exception:
        return None, {}

    try:
        candidates: list[Path] = []
        for root, _dirs, files in os.walk(str(log_dir)):
            for name in files:
                if name == "metrics.csv":
                    candidates.append(Path(root) / name)
        if not candidates:
            return None, {}
        candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        metrics_csv = candidates[0]

        df = pd.read_csv(str(metrics_csv))
        if "epoch" not in df.columns:
            return None, {}

        # Keep only rows with a numeric epoch.
        df = df[df["epoch"].notna()].reset_index(drop=True)
        if len(df) < 1:
            return None, {}

        # Prefer the last row that actually has validation metrics.
        if "val_loss" in df.columns:
            df_val = df[df["val_loss"].notna()].reset_index(drop=True)
        else:
            df_val = df
        if len(df_val) < 1:
            df_val = df
        last = df_val.iloc[-1]

        def pick(cols: list[str]) -> Optional[float]:
            for c in cols:
                if c in df_val.columns and last[c] is not None and str(last[c]) != "":
                    try:
                        v = float(last[c])
                        # Drop NaN/inf values (treat as missing).
                        if v != v or v in (float("inf"), float("-inf")):
                            continue
                        return v
                    except Exception:
                        continue
            return None

        try:
            # Lightning's CSV logger uses 0-based epoch indices. Convert to 1-based
            # "epochs completed" to stay consistent with Ray Tune reporting.
            epoch_end0 = int(float(last["epoch"]))
            epoch_end = int(epoch_end0 + 1)
        except Exception:
            epoch_end = None

        metrics: dict[str, float] = {}
        # Core metrics
        v = pick(["val_loss", "val_loss/dataloader_idx_0"])
        if v is not None:
            metrics["val_loss"] = v
        v = pick(["val_r2", "val_r2/dataloader_idx_0"])
        if v is not None:
            metrics["val_r2"] = v
        v = pick(["val_r2_global", "val_r2_global/dataloader_idx_0"])
        if v is not None:
            metrics["val_r2_global"] = v

        # Common sub-losses useful for HPO
        v = pick(["val_loss_reg3_mse", "val_loss_reg3_mse/dataloader_idx_0"])
        if v is not None:
            metrics["val_loss_reg3_mse"] = v
        v = pick(["val_loss_5d_weighted", "val_loss_5d_weighted/dataloader_idx_0"])
        if v is not None:
            metrics["val_loss_5d_weighted"] = v
        v = pick(["val_loss_ratio_mse", "val_loss_ratio_mse/dataloader_idx_0"])
        if v is not None:
            metrics["val_loss_ratio_mse"] = v
        v = pick(["val_loss_height", "val_loss_height/dataloader_idx_0"])
        if v is not None:
            metrics["val_loss_height"] = v
        v = pick(["val_loss_ndvi", "val_loss_ndvi/dataloader_idx_0"])
        if v is not None:
            metrics["val_loss_ndvi"] = v
        v = pick(["val_loss_state", "val_loss_state/dataloader_idx_0"])
        if v is not None:
            metrics["val_loss_state"] = v

        return epoch_end, metrics
    except Exception:
        return None, {}


def _safe_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    try:
        s = str(x).strip()
        if s == "" or s.lower() in ("nan", "none", "null"):
            return None
        v = float(s)
        if v != v or v in (float("inf"), float("-inf")):
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
    name = str(trial_dir.name)
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
    import json

    p = trial_dir / "params.json"
    with open(p, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, dict):
        raise TypeError(f"params.json must be a dict, got: {type(obj)} ({p})")
    return obj


def _score_trial_from_result_json(
    trial_dir: Path,
    *,
    metric: str,
    mode: str,
    scope: str,
    min_epoch: int,
) -> Optional[dict[str, Any]]:
    """
    Score a Tune trial by scanning its result.json (JSONL).

    - scope: "best" or "last" (within rows where epoch >= min_epoch and metric present)
    - mode: "min" or "max"
    """
    import json

    mode_n = str(mode).strip().lower()
    if mode_n not in ("min", "max"):
        raise ValueError(f"Unsupported mode: {mode!r} (expected 'min' or 'max')")
    scope_n = str(scope).strip().lower()
    if scope_n not in ("best", "last"):
        raise ValueError(f"Unsupported scope: {scope!r} (expected 'best' or 'last')")

    result_json = trial_dir / "result.json"
    if not result_json.is_file():
        return None

    best_val: Optional[float] = None
    best_epoch: Optional[int] = None
    best_done: Optional[bool] = None

    last_val: Optional[float] = None
    last_epoch: Optional[int] = None
    last_done: Optional[bool] = None

    trial_id: str = ""

    with open(result_json, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
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

            v = _safe_float(row.get(metric))
            if v is None:
                continue

            done_raw = row.get("done", None)
            done = (bool(done_raw) if done_raw is not None else None)

            last_val, last_epoch, last_done = float(v), int(ep), done

            if best_val is None:
                best_val, best_epoch, best_done = float(v), int(ep), done
            else:
                better = (v < best_val) if mode_n == "min" else (v > best_val)
                if better:
                    best_val, best_epoch, best_done = float(v), int(ep), done

    if not trial_id:
        trial_id = _guess_trial_id_from_dir(trial_dir)

    if scope_n == "last":
        if last_val is None or last_epoch is None:
            return None
        return {
            "trial_id": trial_id,
            "trial_dir": str(trial_dir),
            "value": float(last_val),
            "epoch": int(last_epoch),
            "done": (None if last_done is None else bool(last_done)),
        }

    # scope == "best"
    if best_val is None or best_epoch is None:
        return None
    return {
        "trial_id": trial_id,
        "trial_dir": str(trial_dir),
        "value": float(best_val),
        "epoch": int(best_epoch),
        "done": (None if best_done is None else bool(best_done)),
    }


def _missing_metric_sentinel(*, mode: str) -> float:
    return float(1.0e9) if str(mode).strip().lower() == "min" else float(-1.0e9)


def _try_extract_kfold_metrics(
    log_dir: Path,
) -> tuple[Optional[int], Optional[int], dict[str, float], dict[str, float]]:
    """
    Best-effort extract *aggregated* k-fold metrics under a single seed log_dir.

    Returns:
      (epoch_end, num_folds, avg_metrics, fold_std_metrics)

    Notes:
    - Prefer scanning fold_*/metrics.csv so we can aggregate *any* metric that Lightning logged
      (e.g., val_r2_global, val_loss_5d_weighted), not just the subset exported by kfold_runner.
    - Falls back to log_dir/kfold_metrics.json produced by `src/training/kfold_runner.py`.
    """
    import json
    import math

    # Prefer: average fold_*/metrics.csv (more complete metric set than kfold_metrics.json).
    try:
        fold_dirs = [p for p in log_dir.iterdir() if p.is_dir() and p.name.startswith("fold_")]
    except Exception:
        fold_dirs = []

    if fold_dirs:
        fold_dirs = sorted(fold_dirs)
        per_fold_metrics: list[dict[str, float]] = []
        epoch_end: Optional[int] = None
        for fd in fold_dirs:
            ep_i, m = _try_extract_final_metrics(fd)
            if ep_i is not None:
                epoch_end = int(ep_i) if epoch_end is None else max(int(epoch_end), int(ep_i))
            if m:
                per_fold_metrics.append(m)

        if per_fold_metrics:
            # Mean/std per key across folds.
            keys = set().union(*(m.keys() for m in per_fold_metrics))
            avg_metrics: dict[str, float] = {}
            fold_std: dict[str, float] = {}
            for k in sorted(keys):
                vals: list[float] = []
                for m in per_fold_metrics:
                    v = _safe_float(m.get(k, None))
                    if v is not None:
                        vals.append(float(v))
                if not vals:
                    continue
                mu = float(sum(vals) / len(vals))
                avg_metrics[str(k)] = mu
                if len(vals) >= 2:
                    var = float(sum((x - mu) ** 2 for x in vals) / float(len(vals)))
                    fold_std[str(k)] = float(var**0.5)
            # NOTE: report the number of folds actually contributing metrics.
            return epoch_end, int(len(per_fold_metrics)), avg_metrics, fold_std

    kfold_json = (log_dir / "kfold_metrics.json").resolve()
    if kfold_json.is_file():
        try:
            with open(kfold_json, "r", encoding="utf-8") as f:
                obj = json.load(f)
            if isinstance(obj, dict):
                num_folds_decl = _safe_int(obj.get("num_folds", None))
                avg_raw = obj.get("average", {}) or {}
                avg = dict(avg_raw) if isinstance(avg_raw, dict) else {}
                per_fold_raw = obj.get("per_fold", []) or []
                per_fold = list(per_fold_raw) if isinstance(per_fold_raw, list) else []
                # Prefer the number of actually evaluated folds when a subset is used (kfold.folds != null).
                num_folds = int(len(per_fold)) if per_fold else num_folds_decl

                # Avg metrics (from file)
                avg_metrics: dict[str, float] = {}
                for k, v in avg.items():
                    fv = _safe_float(v)
                    if fv is not None:
                        avg_metrics[str(k)] = float(fv)

                # Fold-level std (computed from per_fold)
                vals_by_k: dict[str, list[float]] = {}
                epoch_end: Optional[int] = None
                for rec in per_fold:
                    if not isinstance(rec, dict):
                        continue
                    ep = _safe_int(rec.get("epoch", None))
                    if ep is not None:
                        epoch_end = int(ep) if epoch_end is None else max(int(epoch_end), int(ep))
                    for k, v in rec.items():
                        if k in ("fold", "epoch"):
                            continue
                        fv = _safe_float(v)
                        if fv is None or not math.isfinite(fv):
                            continue
                        vals_by_k.setdefault(str(k), []).append(float(fv))

                fold_std: dict[str, float] = {}
                for k, vals in vals_by_k.items():
                    if len(vals) < 2:
                        continue
                    mu = float(sum(vals) / len(vals))
                    var = float(sum((x - mu) ** 2 for x in vals) / float(len(vals)))
                    fold_std[str(k)] = float(var**0.5)

                return epoch_end, num_folds, avg_metrics, fold_std
        except Exception:
            # Fall back to empty below.
            pass

    return None, None, {}, {}


@hydra.main(version_base=None, config_path="conf", config_name="tune")
def main(cfg: DictConfig) -> None:
    repo_root = resolve_repo_root().resolve()

    # Ensure our repo imports work even if Ray Tune changes the working directory.
    sys.path.insert(0, str(repo_root))

    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(cfg_dict, dict):
        raise TypeError(f"tune config must resolve to dict, got: {type(cfg_dict)}")

    # Extract Tune-specific sections, leaving the remaining keys as the training config.
    tune_cfg = dict(cfg_dict.get("tune", {}) or {})
    ray_cfg = dict(cfg_dict.get("ray", {}) or {})
    cfg_dict.pop("tune", None)
    cfg_dict.pop("ray", None)
    cfg_dict.pop("hydra", None)
    train_cfg = cfg_dict

    # Tune settings
    metric = str(tune_cfg.get("metric", "val_loss"))
    mode = str(tune_cfg.get("mode", "min"))
    name = str(tune_cfg.get("name", "tune"))
    num_samples = int(tune_cfg.get("num_samples", 10))
    max_concurrent_trials = int(tune_cfg.get("max_concurrent_trials", 1))
    resources_per_trial = dict(tune_cfg.get("resources_per_trial", {"cpu": 4, "gpu": 1}) or {})

    storage_path = _as_path_under_root(str(tune_cfg.get("storage_path", "ray_results")), repo_root=repo_root)
    storage_path.mkdir(parents=True, exist_ok=True)

    seeds_raw = tune_cfg.get("seeds", None)
    seeds = list(seeds_raw) if isinstance(seeds_raw, (list, tuple)) else None
    seeds = seeds if seeds else [int(train_cfg.get("seed", 42))]
    seeds = [int(s) for s in seeds]
    report_per_epoch = bool(tune_cfg.get("report_per_epoch", True))
    resume = bool(tune_cfg.get("resume", False))
    restore_path_raw = tune_cfg.get("restore_path", None)
    restore_path = str(restore_path_raw).strip() if restore_path_raw not in (None, "", "null") else ""
    resume_unfinished = bool(tune_cfg.get("resume_unfinished", True))
    resume_errored = bool(tune_cfg.get("resume_errored", False))
    restart_errored = bool(tune_cfg.get("restart_errored", False))

    # Build Ray Tune search space
    #
    # Note: some Hydra configs may carry merge directives such as `_delete_` inside `tune.search_space`.
    # These are not part of the Tune search space spec and would break `_build_search_space`.
    space_spec = dict(tune_cfg.get("search_space", {}) or {})
    try:
        space_spec = {
            str(k): v
            for k, v in (space_spec or {}).items()
            if str(k) and not str(k).startswith("_")
        }
    except Exception:
        space_spec = dict(tune_cfg.get("search_space", {}) or {})

    # Ray init
    ray_address = str(ray_cfg.get("address", "") or "").strip()
    try:
        import ray

        if not ray_address or ray_address.lower() in ("none", "null"):
            ray.init(ignore_reinit_error=True)
        else:
            ray.init(address=ray_address, ignore_reinit_error=True)
    except Exception as e:
        raise RuntimeError(f"Failed to init Ray (address={ray_address!r}): {e}") from e

    from ray import air, tune
    from ray.tune import CLIReporter
    from ray.tune.schedulers import ASHAScheduler

    param_space = _build_search_space(space_spec)

    # Scheduler
    sched_cfg = dict(tune_cfg.get("scheduler", {}) or {})
    scheduler_type = str(sched_cfg.get("type", "asha") or "asha").strip().lower()
    scheduler = None
    if scheduler_type == "asha":
        max_epochs = int(sched_cfg.get("max_epochs", (train_cfg.get("trainer", {}) or {}).get("max_epochs", 20)))
        grace_period = int(sched_cfg.get("grace_period", 3))
        reduction_factor = int(sched_cfg.get("reduction_factor", 2))
        scheduler = ASHAScheduler(
            time_attr="epoch",
            max_t=max_epochs,
            grace_period=grace_period,
            reduction_factor=reduction_factor,
        )
    elif scheduler_type in ("none", "fifo", ""):
        scheduler = None
    else:
        raise ValueError(f"Unsupported scheduler.type: {scheduler_type!r}")

    # Progress reporting (CLI). The Ray Dashboard also shows metrics per-trial, but this gives
    # a quick view of the current/best objective during execution (useful when watching logs).
    try:
        metric_cols: dict[str, str] = {
            "epoch": "epoch",
            str(metric): "obj",
            "val_loss": "val_loss",
            "val_r2": "val_r2",
            "val_r2_global": "val_r2_g",
        }
        param_cols: dict[str, str] = {}
        for k in sorted((space_spec or {}).keys()):
            kk = str(k)
            # Dotted keys are verbose; use the leaf name as the displayed column header.
            disp = kk.split(".")[-1] if "." in kk else kk
            param_cols[kk] = disp
        progress_reporter = CLIReporter(
            metric_columns=metric_cols,
            parameter_columns=param_cols,
            metric=str(metric),
            mode=str(mode),
            sort_by_metric=True,
        )
    except Exception:
        progress_reporter = None

    def trainable(config: dict) -> None:
        # Note: executed on Ray workers.
        from ray.air import session

        # Ensure this worker imports *this repo's* `src` package.
        #
        # Why this matters:
        # - The driver process inserts repo_root into sys.path, but Ray worker processes may
        #   start with a different CWD/sys.path (especially on remote nodes).
        # - Without this, `import src...` may resolve to an unrelated installed package or
        #   an out-of-date checkout on the worker, causing confusing behavior (e.g. head_type
        #   not recognizing "mamba" and silently falling back to MLP).
        rr_env = str(os.environ.get("CSIRO_REPO_ROOT", "") or "").strip()
        rr = Path(rr_env).resolve() if rr_env else Path(repo_root).resolve()
        if (rr / "src").is_dir():
            if str(rr) not in sys.path:
                sys.path.insert(0, str(rr))
        else:
            logger.warning(
                "Ray worker repo_root does not look valid (rr={}); `src` imports may resolve to a different package.",
                rr,
            )

        # Debug breadcrumb: log the actual module file used on this worker (helps catch stale installs/checkouts).
        try:
            import src.models.regressor.biomass_regressor as _br

            logger.info("Ray worker import: biomass_regressor={}", getattr(_br, "__file__", "?"))
        except Exception as e:
            logger.warning("Ray worker debug import failed (biomass_regressor): {}", e)

        trial_dir = Path(session.get_trial_dir()).resolve()
        trial_dir.mkdir(parents=True, exist_ok=True)

        # Build trial cfg = base training cfg + sampled hyperparams (dotted keys).
        cfg_trial = copy.deepcopy(train_cfg)
        for k, v in (config or {}).items():
            _set_by_dotted_path(cfg_trial, str(k), v)

        # Determine whether this trial runs k-fold training.
        #
        # Note on ASHA:
        # - Full k-fold (multiple folds) only reports a single aggregated result at the end (no per-epoch),
        #   which keeps ASHA semantics clean and avoids excessive Ray reports.
        # - A *single selected fold* behaves like a normal single-split run, so we allow per-epoch reporting
        #   (enabling ASHA) when kfold.enabled=true but kfold.folds selects exactly one fold.
        kfold_cfg_trial = cfg_trial.get("kfold", {}) or {}
        kfold_enabled_trial = bool(kfold_cfg_trial.get("enabled", False))

        def _selected_folds_count(kfold_cfg: dict) -> Optional[int]:
            """
            Best-effort count of selected folds from kfold.folds.

            Supported forms (mirrors src/training/kfold_runner.py semantics):
              - null / missing / \"all\": run all folds -> return None
              - int: single fold -> return 1
              - list/tuple/set: return len(unique_folds) if parseable
              - string: \"0,2,4\" -> return number of parsed indices
            """
            raw = None
            try:
                raw = kfold_cfg.get("folds", None)
            except Exception:
                raw = None

            if raw is None:
                return None
            if isinstance(raw, int):
                return 1
            if isinstance(raw, (list, tuple, set)):
                try:
                    vals = [int(x) for x in list(raw)]
                    return int(len(sorted(set(vals))))
                except Exception:
                    return None
            if isinstance(raw, str):
                s = raw.strip().lower()
                if s in ("", "none", "null", "all"):
                    return None
                parts = [p for p in s.replace(",", " ").split() if p]
                try:
                    vals = [int(p) for p in parts]
                    return int(len(sorted(set(vals)))) if vals else None
                except Exception:
                    return None
            return None

        selected_folds_count = _selected_folds_count(kfold_cfg_trial if isinstance(kfold_cfg_trial, dict) else {})
        single_fold_kfold = bool(kfold_enabled_trial and selected_folds_count == 1)

        # Enable per-epoch reporting only for single-seed trials and either:
        # - non-kfold, or
        # - kfold with exactly one fold selected (acts like a single split).
        enable_epoch_reporting = bool(report_per_epoch) and len(seeds) == 1 and (
            (not kfold_enabled_trial) or single_fold_kfold
        )

        from src.callbacks.ray_tune_report import RayTuneReportCallback

        # Aggregate final metrics across seeds (when multi-seed is enabled).
        metrics_by_name: dict[str, list[float]] = {}
        epoch_ends: list[int] = []
        fold_counts: list[int] = []

        for si, seed in enumerate(seeds):
            cfg_seed = copy.deepcopy(cfg_trial)
            cfg_seed["seed"] = int(seed)

            seed_root = trial_dir / (f"seed_{seed}" if len(seeds) > 1 else "run")
            log_dir = seed_root / "logs"
            ckpt_dir = seed_root / "checkpoints"

            extra_callbacks = [RayTuneReportCallback()] if enable_epoch_reporting else None

            run_training(
                cfg_seed,
                log_dir=log_dir,
                ckpt_dir=ckpt_dir,
                repo_root=rr,
                source_config_path=None,
                extra_callbacks=extra_callbacks,
                enable_post_kfold_swa_eval=False,
            )

            if kfold_enabled_trial:
                # Extract aggregated k-fold metrics under this seed_root/logs.
                ep_end, num_folds, avg_metrics, fold_std = _try_extract_kfold_metrics(log_dir)
                if ep_end is not None:
                    epoch_ends.append(int(ep_end))
                if num_folds is not None:
                    fold_counts.append(int(num_folds))

                # Mean across folds (per seed)
                for k, v in (avg_metrics or {}).items():
                    try:
                        metrics_by_name.setdefault(str(k), []).append(float(v))
                    except Exception:
                        continue

                # Std across folds (per seed), stored with a suffix for clarity.
                for k, v in (fold_std or {}).items():
                    try:
                        metrics_by_name.setdefault(f"{str(k)}_fold_std", []).append(float(v))
                    except Exception:
                        continue
            else:
                # Extract final metrics for aggregation; only emit per-seed reports when epoch reporting is disabled.
                epoch_end, final_metrics = _try_extract_final_metrics(log_dir)
                if epoch_end is not None:
                    epoch_ends.append(int(epoch_end))
                for k, v in (final_metrics or {}).items():
                    try:
                        metrics_by_name.setdefault(str(k), []).append(float(v))
                    except Exception:
                        continue

                # Optional per-seed report (useful when report_per_epoch is disabled).
                # IMPORTANT: Ray strict metric checking requires every report to include the objective metric.
                if not enable_epoch_reporting and (epoch_end is not None) and final_metrics:
                    per_seed_payload: dict[str, Any] = {"epoch": int(epoch_end), "seed": int(seed)}
                    # Objective metric (or sentinel) for Ray Tune robustness.
                    if str(metric) in final_metrics:
                        per_seed_payload[str(metric)] = float(final_metrics[str(metric)])
                    else:
                        per_seed_payload[str(metric)] = _missing_metric_sentinel(mode=mode)
                    # Keep historical per-seed keys for the most common metrics.
                    if "val_loss" in final_metrics:
                        per_seed_payload[f"val_loss_seed{si}"] = float(final_metrics["val_loss"])
                    if "val_r2" in final_metrics:
                        per_seed_payload[f"val_r2_seed{si}"] = float(final_metrics["val_r2"])
                    if "val_r2_global" in final_metrics:
                        per_seed_payload[f"val_r2_global_seed{si}"] = float(final_metrics["val_r2_global"])
                    session.report(per_seed_payload)

        # Final aggregated report (the metric optimized by Tune is `metric` from the Tune config).
        max_epochs = int(cfg_trial.get("trainer", {}).get("max_epochs", 0) or 0)
        # Keep epoch 1-based for consistency with per-epoch reports.
        epoch_final = int(max(epoch_ends) if epoch_ends else max(0, max_epochs))
        payload: dict[str, Any] = {"epoch": int(epoch_final)}
        for k, vs in (metrics_by_name or {}).items():
            if not vs:
                continue
            try:
                payload[str(k)] = float(sum(vs) / len(vs))
            except Exception:
                continue
        if len(seeds) > 1:
            payload["num_seeds"] = int(len(seeds))
        if kfold_enabled_trial:
            payload["kfold_num_folds"] = float(sum(fold_counts) / len(fold_counts)) if fold_counts else 0.0
        # Ray strict metric checking: always include the objective metric.
        if str(metric) not in payload:
            payload[str(metric)] = _missing_metric_sentinel(mode=mode)
        # Only report if we have at least one scalar metric in addition to epoch.
        if len(payload) > 1:
            session.report(payload)

    trainable_wrapped = tune.with_resources(trainable, resources=resources_per_trial)

    # Construct a new Tuner or restore an existing run.
    #
    # NOTE: Restoring expects the experiment directory (including checkpoints/state) to be present.
    # For local/NFS storage this is typically: <storage_path>/<name>.
    if resume:
        exp_path = restore_path or str(storage_path / name)
        # Auto-fallback: if the restore path doesn't exist yet, start a new run instead.
        # This keeps UX simple when users always enable resume (e.g., in a wrapper script).
        exp_exists = True
        try:
            # Best-effort: only check local filesystem paths (NFS counts as local here).
            if "://" not in exp_path:
                exp_exists = Path(exp_path).exists()
        except Exception:
            exp_exists = True

        if exp_exists:
            logger.info(
                "Restoring Tune run from: {} (resume_unfinished={}, resume_errored={}, restart_errored={})",
                exp_path,
                resume_unfinished,
                resume_errored,
                restart_errored,
            )
            tuner = tune.Tuner.restore(
                exp_path,
                trainable=trainable_wrapped,
                resume_unfinished=resume_unfinished,
                resume_errored=resume_errored,
                restart_errored=restart_errored,
                # Re-specify param_space for safety (must be unmodified for restore).
                param_space=param_space,
            )
            # Best-effort: inject the progress reporter for restored runs too.
            # Ray's public `Tuner.restore()` API doesn't accept a RunConfig override.
            if progress_reporter is not None:
                try:
                    local_tuner = getattr(tuner, "_local_tuner", None)
                    if local_tuner is not None:
                        rc = getattr(local_tuner, "_run_config", None)
                        if rc is not None:
                            rc.progress_reporter = progress_reporter
                except Exception:
                    pass
        else:
            logger.info(
                "Resume requested but restore path does not exist yet: {}. Starting a new run instead.",
                exp_path,
            )
            tuner = tune.Tuner(
                trainable_wrapped,
                param_space=param_space,
                tune_config=tune.TuneConfig(
                    metric=metric,
                    mode=mode,
                    num_samples=num_samples,
                    max_concurrent_trials=max_concurrent_trials,
                    scheduler=scheduler,
                ),
                run_config=air.RunConfig(
                    name=name,
                    storage_path=str(storage_path),
                    progress_reporter=progress_reporter,
                ),
            )
    else:
        tuner = tune.Tuner(
            trainable_wrapped,
            param_space=param_space,
            tune_config=tune.TuneConfig(
                metric=metric,
                mode=mode,
                num_samples=num_samples,
                max_concurrent_trials=max_concurrent_trials,
                scheduler=scheduler,
            ),
            run_config=air.RunConfig(
                name=name,
                storage_path=str(storage_path),
                progress_reporter=progress_reporter,
            ),
        )

    logger.info(
        "Starting Tune run: name={}, metric={}({}), num_samples={}, max_concurrent_trials={}, storage_path={}",
        name,
        metric,
        mode,
        num_samples,
        max_concurrent_trials,
        storage_path,
    )
    results = tuner.fit()

    # Save best config as a resolved training config YAML for convenience and log best metrics.
    try:
        best = results.get_best_result(metric=metric, mode=mode)
        best_metrics: dict[str, Any] = {}
        try:
            raw = getattr(best, "metrics", None)
            if isinstance(raw, dict):
                best_metrics = dict(raw)
        except Exception:
            best_metrics = {}

        best_value = best_metrics.get(str(metric), None)
        best_epoch = best_metrics.get("epoch", None)

        best_trial_dir = ""
        try:
            p = getattr(best, "path", None)
            best_trial_dir = str(p) if p is not None else ""
        except Exception:
            best_trial_dir = ""
        if not best_trial_dir:
            try:
                best_trial_dir = str(getattr(best, "log_dir", "") or "")
            except Exception:
                best_trial_dir = ""

        best_trial_id = best_metrics.get("trial_id", None)
        if best_trial_id is None:
            try:
                best_trial_id = getattr(best, "trial_id", None)
            except Exception:
                best_trial_id = None

        logger.info(
            "Tune best: metric={}({}) -> {} (epoch={}, trial_id={}, trial_dir={})",
            metric,
            mode,
            best_value,
            best_epoch,
            best_trial_id,
            best_trial_dir,
        )

        best_overrides = dict(best.config or {})
        best_cfg = copy.deepcopy(train_cfg)
        for k, v in best_overrides.items():
            _set_by_dotted_path(best_cfg, str(k), v)
        # Prefer the experiment directory reported by Tune (handles any internal naming/suffixing).
        exp_path = str(getattr(results, "experiment_path", "") or "").strip()
        out_dir = Path(exp_path) if exp_path else (storage_path / name)
        out_path = out_dir / "best_train_cfg.yaml"
        import yaml

        with open(out_path, "w", encoding="utf-8") as f:
            # Prepend a short summary as YAML comments so the best metric is visible
            # when browsing the config file directly.
            header_lines = [
                "# --- Ray Tune best result ---",
                f"# metric: {metric}",
                f"# mode: {mode}",
            ]
            if best_value is not None:
                header_lines.append(f"# best_{metric}: {best_value}")
            if best_epoch is not None:
                header_lines.append(f"# epoch: {best_epoch}")
            if best_trial_id not in (None, "", "null"):
                header_lines.append(f"# trial_id: {best_trial_id}")
            if best_trial_dir:
                header_lines.append(f"# trial_dir: {best_trial_dir}")
            header_lines.append("# ----------------------------")
            f.write("\n".join(header_lines) + "\n\n")
            yaml.safe_dump(best_cfg, f, sort_keys=False, allow_unicode=True)
        logger.info("Saved best resolved training config -> {}", out_path)
    except Exception as e:
        logger.warning(f"Failed to write best_train_cfg.yaml: {e}")

    # Optional: two-stage filtering (stage 1: single split HPO; stage 2: k-fold eval of top-N).
    try:
        two_stage_cfg = dict(tune_cfg.get("two_stage", {}) or {})
        two_stage_enabled = bool(two_stage_cfg.get("enabled", False))
    except Exception:
        two_stage_enabled = False

    if two_stage_enabled:
        import json

        # Stage-1 experiment directory
        exp_path = str(getattr(results, "experiment_path", "") or "").strip()
        exp_dir = (Path(exp_path).resolve() if exp_path else (storage_path / name).resolve())

        # Candidate selection knobs
        try:
            default_min_epoch = int((tune_cfg.get("scheduler", {}) or {}).get("grace_period", 1) or 1)
        except Exception:
            default_min_epoch = 1
        top_n = int(two_stage_cfg.get("top_n", 8) or 0)
        min_epoch = int(two_stage_cfg.get("min_epoch", default_min_epoch) or default_min_epoch)
        scope = str(two_stage_cfg.get("scope", "best") or "best").strip().lower()
        if top_n <= 0:
            logger.warning("two_stage.enabled=true but top_n<=0; skipping stage 2.")
            return

        # Select top-N from stage-1 results.json + params.json
        trial_dirs = _iter_trial_dirs(exp_dir)
        candidates: list[dict[str, Any]] = []
        for td in trial_dirs:
            try:
                score = _score_trial_from_result_json(
                    td,
                    metric=str(metric),
                    mode=str(mode),
                    scope=str(scope),
                    min_epoch=int(min_epoch),
                )
                if score is None:
                    continue
                params = _read_params_json(td)
                # Keep JSON-serializable payload only.
                score["params"] = dict(params)
                candidates.append(score)
            except Exception:
                continue

        # Sort and de-duplicate by params.json content (keeps stage2 budget predictable).
        def _params_key(c: dict) -> str:
            try:
                return json.dumps(c.get("params", {}), sort_keys=True, ensure_ascii=False)
            except Exception:
                return str(c.get("params", {}))

        reverse = str(mode).strip().lower() == "max"
        candidates = sorted(candidates, key=lambda c: float(c.get("value", _missing_metric_sentinel(mode=mode))), reverse=reverse)

        uniq: list[dict[str, Any]] = []
        seen: set[str] = set()
        for c in candidates:
            k = _params_key(c)
            if k in seen:
                continue
            seen.add(k)
            uniq.append(c)
            if len(uniq) >= int(top_n):
                break

        if not uniq:
            logger.warning(
                "two_stage: no candidates found under {} (metric={}, scope={}, min_epoch={}). Skipping stage 2.",
                exp_dir,
                metric,
                scope,
                min_epoch,
            )
            return

        # Persist candidate list for reproducibility
        try:
            out_candidates = exp_dir / "two_stage_candidates.json"
            payload = {
                "stage1": {
                    "exp_dir": str(exp_dir),
                    "metric": str(metric),
                    "mode": str(mode),
                    "scope": str(scope),
                    "min_epoch": int(min_epoch),
                },
                "stage2": {"top_n": int(top_n)},
                "candidates": [
                    {
                        "rank": int(i + 1),
                        "trial_id": str(c.get("trial_id", "")),
                        "trial_dir": str(c.get("trial_dir", "")),
                        "value": float(c.get("value")),
                        "epoch": int(c.get("epoch")),
                        "done": c.get("done", None),
                        "params": dict(c.get("params", {}) or {}),
                    }
                    for i, c in enumerate(uniq)
                ],
            }
            with open(out_candidates, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
            logger.info("two_stage: wrote candidates -> {}", out_candidates)
        except Exception as e:
            logger.warning(f"two_stage: failed to write candidates json (non-fatal): {e}")

        # Stage-2 settings
        stage2_name = str(two_stage_cfg.get("name", "") or "").strip()
        if not stage2_name:
            suffix = str(two_stage_cfg.get("name_suffix", "-stage2") or "-stage2").strip() or "-stage2"
            stage2_name = f"{name}{suffix}"

        stage2_max_concurrent = int(two_stage_cfg.get("max_concurrent_trials", max_concurrent_trials) or max_concurrent_trials)
        stage2_resources = dict(two_stage_cfg.get("resources_per_trial", resources_per_trial) or {})
        stage2_resume = bool(two_stage_cfg.get("resume", False))
        stage2_restore_path_raw = two_stage_cfg.get("restore_path", None)
        stage2_restore_path = (
            str(stage2_restore_path_raw).strip()
            if stage2_restore_path_raw not in (None, "", "null")
            else ""
        )
        stage2_resume_unfinished = bool(two_stage_cfg.get("resume_unfinished", True))
        stage2_resume_errored = bool(two_stage_cfg.get("resume_errored", True))
        stage2_restart_errored = bool(two_stage_cfg.get("restart_errored", False))

        # Optional: optimize a different metric in stage 2 (defaults to stage-1 tune.metric/mode).
        stage2_metric = str(two_stage_cfg.get("stage2_metric", metric) or metric).strip()
        stage2_mode = str(two_stage_cfg.get("stage2_mode", mode) or mode).strip()

        # Fixed training overrides applied in stage-2 (on top of base train_cfg + candidate params).
        stage2_train_overrides = dict(two_stage_cfg.get("train_overrides", {}) or {})

        # Build stage-2 param space: grid over the selected candidate override dicts.
        stage2_candidates = []
        for i, c in enumerate(uniq):
            stage2_candidates.append(
                {
                    "rank": int(i + 1),
                    "stage1_value": float(c.get("value")),
                    "stage1_epoch": int(c.get("epoch")),
                    "trial_id": str(c.get("trial_id", "")),
                    "trial_dir": str(c.get("trial_dir", "")),
                    "params": dict(c.get("params", {}) or {}),
                }
            )

        stage2_param_space = {"candidate": tune.grid_search(stage2_candidates)}

        def trainable_stage2(config: dict) -> None:
            # Note: executed on Ray workers.
            from ray.air import session

            # Ensure this worker imports *this repo's* `src` package (see rationale in stage-1 trainable).
            rr_env = str(os.environ.get("CSIRO_REPO_ROOT", "") or "").strip()
            rr = Path(rr_env).resolve() if rr_env else Path(repo_root).resolve()
            if (rr / "src").is_dir():
                if str(rr) not in sys.path:
                    sys.path.insert(0, str(rr))
            else:
                logger.warning(
                    "Ray worker repo_root does not look valid (rr={}); `src` imports may resolve to a different package.",
                    rr,
                )

            # Debug breadcrumb (once per stage-2 trial).
            try:
                import src.models.regressor.biomass_regressor as _br

                logger.info("Ray worker import: biomass_regressor={}", getattr(_br, "__file__", "?"))
            except Exception as e:
                logger.warning("Ray worker debug import failed (biomass_regressor): {}", e)

            trial_dir = Path(session.get_trial_dir()).resolve()
            trial_dir.mkdir(parents=True, exist_ok=True)

            cand = dict(config.get("candidate", {}) or {})
            cand_params = dict(cand.get("params", {}) or {})

            # Build cfg = base training cfg + sampled candidate hyperparams.
            cfg_trial = copy.deepcopy(train_cfg)
            for k, v in (cand_params or {}).items():
                _set_by_dotted_path(cfg_trial, str(k), v)

            # Apply stage-2 fixed overrides.
            for k, v in (stage2_train_overrides or {}).items():
                _set_by_dotted_path(cfg_trial, str(k), v)

            # Enforce stage-2 semantics: k-fold enabled, train_all disabled.
            _set_by_dotted_path(cfg_trial, "kfold.enabled", True)
            _set_by_dotted_path(cfg_trial, "train_all.enabled", False)

            metrics_by_name: dict[str, list[float]] = {}
            epoch_ends: list[int] = []
            fold_counts: list[int] = []

            for seed in seeds:
                cfg_seed = copy.deepcopy(cfg_trial)
                cfg_seed["seed"] = int(seed)

                seed_root = trial_dir / (f"seed_{seed}" if len(seeds) > 1 else "run")
                log_dir = seed_root / "logs"
                ckpt_dir = seed_root / "checkpoints"

                run_training(
                    cfg_seed,
                    log_dir=log_dir,
                    ckpt_dir=ckpt_dir,
                    repo_root=rr,
                    source_config_path=None,
                    extra_callbacks=None,
                    enable_post_kfold_swa_eval=False,
                )

                ep_end, num_folds, avg_metrics, fold_std = _try_extract_kfold_metrics(log_dir)
                if ep_end is not None:
                    epoch_ends.append(int(ep_end))
                if num_folds is not None:
                    fold_counts.append(int(num_folds))

                # Mean across folds (per seed)
                for k, v in (avg_metrics or {}).items():
                    try:
                        metrics_by_name.setdefault(str(k), []).append(float(v))
                    except Exception:
                        continue

                # Std across folds (per seed), stored with a suffix for clarity.
                for k, v in (fold_std or {}).items():
                    try:
                        metrics_by_name.setdefault(f"{str(k)}_fold_std", []).append(float(v))
                    except Exception:
                        continue

            # Final aggregated report (mean across seeds of the fold-mean metric).
            max_epochs_eff = int(cfg_trial.get("trainer", {}).get("max_epochs", 0) or 0)
            epoch_final = int(max(epoch_ends) if epoch_ends else max_epochs_eff)
            payload: dict[str, Any] = {"epoch": int(epoch_final)}

            for k, vs in (metrics_by_name or {}).items():
                if not vs:
                    continue
                try:
                    payload[str(k)] = float(sum(vs) / len(vs))
                except Exception:
                    continue

            # Ray strict metric checking: always include the objective metric.
            if str(stage2_metric) not in payload:
                payload[str(stage2_metric)] = _missing_metric_sentinel(mode=stage2_mode)

            # Lightweight metadata (numeric-only for robust CLI printing).
            payload["stage1_rank"] = float(cand.get("rank", 0) or 0)
            payload["stage1_value"] = float(
                cand.get("stage1_value", _missing_metric_sentinel(mode=mode))
            )
            payload["kfold_num_folds"] = float(sum(fold_counts) / len(fold_counts)) if fold_counts else 0.0
            if len(seeds) > 1:
                payload["num_seeds"] = float(len(seeds))

            session.report(payload)

        trainable_stage2_wrapped = tune.with_resources(trainable_stage2, resources=stage2_resources)

        # Stage-2: FIFO (no ASHA) because each trial is a fixed expensive evaluation.
        stage2_scheduler = None

        if stage2_resume:
            exp_path2 = stage2_restore_path or str(storage_path / stage2_name)
            exp_exists2 = True
            try:
                if "://" not in exp_path2:
                    exp_exists2 = Path(exp_path2).exists()
            except Exception:
                exp_exists2 = True

            if exp_exists2:
                logger.info(
                    "two_stage: restoring stage-2 Tune run from: {} (resume_unfinished={}, resume_errored={}, restart_errored={})",
                    exp_path2,
                    stage2_resume_unfinished,
                    stage2_resume_errored,
                    stage2_restart_errored,
                )
                tuner2 = tune.Tuner.restore(
                    exp_path2,
                    trainable=trainable_stage2_wrapped,
                    resume_unfinished=stage2_resume_unfinished,
                    resume_errored=stage2_resume_errored,
                    restart_errored=stage2_restart_errored,
                    param_space=stage2_param_space,
                )
            else:
                logger.info(
                    "two_stage: stage-2 resume requested but restore path does not exist yet: {}. Starting a new run instead.",
                    exp_path2,
                )
                tuner2 = tune.Tuner(
                    trainable_stage2_wrapped,
                    param_space=stage2_param_space,
                    tune_config=tune.TuneConfig(
                        metric=stage2_metric,
                        mode=stage2_mode,
                        num_samples=1,
                        max_concurrent_trials=stage2_max_concurrent,
                        scheduler=stage2_scheduler,
                    ),
                    run_config=air.RunConfig(
                        name=stage2_name,
                        storage_path=str(storage_path),
                        progress_reporter=progress_reporter,
                    ),
                )
        else:
            tuner2 = tune.Tuner(
                trainable_stage2_wrapped,
                param_space=stage2_param_space,
                tune_config=tune.TuneConfig(
                    metric=stage2_metric,
                    mode=stage2_mode,
                    num_samples=1,
                    max_concurrent_trials=stage2_max_concurrent,
                    scheduler=stage2_scheduler,
                ),
                run_config=air.RunConfig(
                    name=stage2_name,
                    storage_path=str(storage_path),
                    progress_reporter=progress_reporter,
                ),
            )

        logger.info(
            "two_stage: starting stage-2 k-fold eval: name={}, top_n={}, metric={}({}), kfold.enabled=true",
            stage2_name,
            int(len(stage2_candidates)),
            stage2_metric,
            stage2_mode,
        )
        results2 = tuner2.fit()

        # Save stage-2 best resolved training config (base + candidate params + stage2 overrides).
        try:
            best2 = results2.get_best_result(metric=stage2_metric, mode=stage2_mode)
            best_cfg2 = copy.deepcopy(train_cfg)

            best_config_obj = dict(getattr(best2, "config", {}) or {})
            cand_obj = dict(best_config_obj.get("candidate", {}) or {})
            cand_params_best = dict(cand_obj.get("params", {}) or {})
            for k, v in cand_params_best.items():
                _set_by_dotted_path(best_cfg2, str(k), v)
            for k, v in (stage2_train_overrides or {}).items():
                _set_by_dotted_path(best_cfg2, str(k), v)
            _set_by_dotted_path(best_cfg2, "kfold.enabled", True)
            _set_by_dotted_path(best_cfg2, "train_all.enabled", False)

            exp_path2 = str(getattr(results2, "experiment_path", "") or "").strip()
            out_dir2 = Path(exp_path2).resolve() if exp_path2 else (storage_path / stage2_name)
            out_path2 = out_dir2 / "best_train_cfg.yaml"

            best_metrics2: dict[str, Any] = {}
            try:
                raw2 = getattr(best2, "metrics", None)
                if isinstance(raw2, dict):
                    best_metrics2 = dict(raw2)
            except Exception:
                best_metrics2 = {}
            best_value2 = best_metrics2.get(str(stage2_metric), None)
            best_epoch2 = best_metrics2.get("epoch", None)
            best_rank2 = best_metrics2.get("stage1_rank", None)

            import yaml

            with open(out_path2, "w", encoding="utf-8") as f:
                header_lines = [
                    "# --- Ray Tune stage-2 best result (k-fold eval) ---",
                    f"# metric: {stage2_metric}",
                    f"# mode: {stage2_mode}",
                ]
                if best_value2 is not None:
                    header_lines.append(f"# best_{stage2_metric}: {best_value2}")
                if best_epoch2 is not None:
                    header_lines.append(f"# epoch: {best_epoch2}")
                if best_rank2 is not None:
                    header_lines.append(f"# stage1_rank: {best_rank2}")
                header_lines.append("# ------------------------------------------------")
                f.write("\n".join(header_lines) + "\n\n")
                yaml.safe_dump(best_cfg2, f, sort_keys=False, allow_unicode=True)
            logger.info("two_stage: saved stage-2 best resolved training config -> {}", out_path2)

            # Convenience: also copy a pointer file into stage-1 experiment dir.
            try:
                link_path = exp_dir / "best_train_cfg_stage2.yaml"
                with open(link_path, "w", encoding="utf-8") as f:
                    f.write(f"# stage2_best_cfg: {str(out_path2.resolve())}\n")
                logger.info("two_stage: wrote stage-1 pointer -> {}", link_path)
            except Exception:
                pass
        except Exception as e:
            logger.warning(f"two_stage: failed to write stage-2 best_train_cfg.yaml (non-fatal): {e}")


if __name__ == "__main__":
    main()


