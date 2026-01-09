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
) -> tuple[Optional[int], Optional[float], Optional[float]]:
    """
    Best-effort: read Lightning CSVLogger metrics.csv under log_dir and return (val_loss, val_r2).
    """
    try:
        import pandas as pd
    except Exception:
        return None, None, None

    try:
        candidates: list[Path] = []
        for root, _dirs, files in os.walk(str(log_dir)):
            for name in files:
                if name == "metrics.csv":
                    candidates.append(Path(root) / name)
        if not candidates:
            return None, None, None
        candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        metrics_csv = candidates[0]

        df = pd.read_csv(str(metrics_csv))
        if "epoch" not in df.columns:
            return None, None, None
        gb = df.groupby("epoch").tail(1).reset_index(drop=True)
        last = gb.iloc[-1]

        def pick(cols: list[str]) -> Optional[float]:
            for c in cols:
                if c in gb.columns and last[c] is not None and str(last[c]) != "":
                    try:
                        return float(last[c])
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

        val_loss = pick(["val_loss", "val_loss/dataloader_idx_0"])
        val_r2 = pick(["val_r2", "val_r2/dataloader_idx_0"])
        return epoch_end, val_loss, val_r2
    except Exception:
        return None, None, None


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

    def trainable(config: dict) -> None:
        # Note: executed on Ray workers.
        from ray.air import session

        trial_dir = Path(session.get_trial_dir()).resolve()
        trial_dir.mkdir(parents=True, exist_ok=True)

        # Build trial cfg = base training cfg + sampled hyperparams (dotted keys).
        cfg_trial = copy.deepcopy(train_cfg)
        for k, v in (config or {}).items():
            _set_by_dotted_path(cfg_trial, str(k), v)

        # Only enable per-epoch reporting for single-seed trials to keep ASHA semantics clean.
        enable_epoch_reporting = bool(report_per_epoch) and len(seeds) == 1

        from src.callbacks.ray_tune_report import RayTuneReportCallback

        val_losses: list[float] = []
        val_r2s: list[float] = []
        epoch_ends: list[int] = []

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
                repo_root=resolve_repo_root(),
                source_config_path=None,
                extra_callbacks=extra_callbacks,
                enable_post_kfold_swa_eval=False,
            )

            # Extract final metrics for aggregation; only emit per-seed reports when epoch reporting is disabled.
            epoch_end, vloss, vr2 = _try_extract_final_metrics(log_dir)
            if epoch_end is not None:
                epoch_ends.append(int(epoch_end))
            if vloss is not None:
                val_losses.append(float(vloss))
            if vr2 is not None:
                val_r2s.append(float(vr2))

            if not enable_epoch_reporting and (epoch_end is not None) and (vloss is not None or vr2 is not None):
                per_seed_payload: dict[str, Any] = {"epoch": int(epoch_end), "seed": int(seed)}
                if vloss is not None:
                    per_seed_payload[f"val_loss_seed{si}"] = float(vloss)
                if vr2 is not None:
                    per_seed_payload[f"val_r2_seed{si}"] = float(vr2)
                session.report(per_seed_payload)

        # Final aggregated report (the metric optimized by Tune is `val_loss`).
        max_epochs = int(cfg_trial.get("trainer", {}).get("max_epochs", 0) or 0)
        # Keep epoch 1-based for consistency with per-epoch reports.
        epoch_final = int(max(epoch_ends) if epoch_ends else max(0, max_epochs))
        payload: dict[str, Any] = {"epoch": int(epoch_final)}
        if val_losses:
            payload["val_loss"] = float(sum(val_losses) / len(val_losses))
        if val_r2s:
            payload["val_r2"] = float(sum(val_r2s) / len(val_r2s))
        if len(seeds) > 1:
            payload["num_seeds"] = int(len(seeds))
        if "val_loss" in payload or "val_r2" in payload:
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

    # Save best config as a resolved training config YAML for convenience.
    try:
        best = results.get_best_result(metric=metric, mode=mode)
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
            yaml.safe_dump(best_cfg, f, sort_keys=False, allow_unicode=True)
        logger.info("Saved best resolved training config -> {}", out_path)
    except Exception as e:
        logger.warning(f"Failed to write best_train_cfg.yaml: {e}")


if __name__ == "__main__":
    main()


