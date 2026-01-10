#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional


def resolve_repo_root() -> Path:
    """
    Resolve the project root directory that contains both `conf/` and `src/`.

    This script may be invoked from arbitrary working directories, so we can't
    rely on relative paths.
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


def _safe_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    try:
        s = str(x).strip()
        if s == "" or s.lower() in ("nan", "none", "null"):
            return None
        v = float(s)
        # Drop NaN/inf
        if v != v or v in (float("inf"), float("-inf")):
            return None
        return v
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


@dataclass(frozen=True)
class TrialScore:
    trial_dir: Path
    trial_id: str
    metric: str
    mode: str
    scope: str
    value: float
    epoch: int
    done: bool


def _iter_trial_dirs(exp_dir: Path) -> Iterable[Path]:
    for p in sorted(exp_dir.iterdir()):
        if not p.is_dir():
            continue
        # Ray Tune trial dirs in this repo start with trainable_*, but we keep the check loose.
        # NOTE: progress.csv headers can miss metrics that were absent in the first report
        # (e.g., val_r2). result.json (JSONL) is more reliable for metric scanning.
        if (p / "params.json").is_file() and (p / "result.json").is_file():
            yield p


def _read_params(trial_dir: Path) -> dict[str, Any]:
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
) -> Optional[TrialScore]:
    mode_n = str(mode).strip().lower()
    if mode_n not in ("min", "max"):
        raise ValueError(f"Unsupported mode: {mode!r} (expected 'min' or 'max')")
    scope_n = str(scope).strip().lower()
    if scope_n not in ("last", "best"):
        raise ValueError(f"Unsupported scope: {scope!r} (expected 'last' or 'best')")

    result_json = trial_dir / "result.json"
    if not result_json.is_file():
        return None

    best_val: Optional[float] = None
    best_epoch: Optional[int] = None
    best_done: bool = False
    best_trial_id: str = ""

    last_val: Optional[float] = None
    last_epoch: Optional[int] = None
    last_done: bool = False
    last_trial_id: str = ""

    try:
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
                ep = _safe_int(row.get("epoch"))
                if ep is None or ep < int(min_epoch):
                    continue
                v = _safe_float(row.get(metric))
                if v is None:
                    continue
                done_raw = row.get("done", False)
                done = bool(done_raw) if done_raw is not None else False
                tid = str(row.get("trial_id", "") or "").strip()

                # Track last seen
                last_val, last_epoch, last_done, last_trial_id = float(v), int(ep), bool(done), tid

                # Track best seen (across all rows)
                if best_val is None:
                    best_val, best_epoch, best_done, best_trial_id = float(v), int(ep), bool(done), tid
                else:
                    better = (v < best_val) if mode_n == "min" else (v > best_val)
                    if better:
                        best_val, best_epoch, best_done, best_trial_id = float(v), int(ep), bool(done), tid
    except Exception:
        return None

    if scope_n == "last":
        if last_val is None or last_epoch is None:
            return None
        trial_id = last_trial_id or _guess_trial_id_from_dir(trial_dir)
        return TrialScore(
            trial_dir=trial_dir,
            trial_id=trial_id,
            metric=str(metric),
            mode=mode_n,
            scope=scope_n,
            value=float(last_val),
            epoch=int(last_epoch),
            done=bool(last_done),
        )

    # scope == "best"
    if best_val is None or best_epoch is None:
        return None
    trial_id = best_trial_id or _guess_trial_id_from_dir(trial_dir)
    return TrialScore(
        trial_dir=trial_dir,
        trial_id=trial_id,
        metric=str(metric),
        mode=mode_n,
        scope=scope_n,
        value=float(best_val),
        epoch=int(best_epoch),
        done=bool(best_done),
    )


def _guess_trial_id_from_dir(trial_dir: Path) -> str:
    """
    Best-effort parse trial id from directory name:
      trainable_<exp>_<trialid>_<idx>_... -> trialid
    """
    name = trial_dir.name
    if name.startswith("trainable_"):
        parts = name.split("_")
        # Example: trainable_d9a15_00057_57_...
        if len(parts) >= 3:
            return f"{parts[1]}_{parts[2]}"
    return ""


def _pick_better(a: TrialScore, b: TrialScore) -> TrialScore:
    if a.mode == "min":
        return a if a.value <= b.value else b
    return a if a.value >= b.value else b


def _compose_hydra_cfg(repo_root: Path, *, config_name: str, overrides: list[str]) -> dict[str, Any]:
    # Lazy imports so this script stays importable even if Hydra isn't installed.
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra
    from omegaconf import OmegaConf

    # Avoid "Hydra is already initialized" issues when this is used in notebooks/repl.
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Export a resolved training YAML from an *in-progress* Ray Tune experiment by scanning "
            "trial result.json + params.json and writing a best-so-far config (no need to wait for tune.py)."
        )
    )
    parser.add_argument(
        "--config-name",
        type=str,
        default="tune_vitdet",
        help="Hydra config name under conf/ (without .yaml). Default: tune_vitdet",
    )
    parser.add_argument(
        "--exp-dir",
        type=str,
        default="",
        help=(
            "Path to the Ray Tune experiment directory (e.g., /mnt/csiro_nfs/ray_results/<tune.name>). "
            "If omitted, it is derived from the composed config: tune.storage_path/tune.name."
        ),
    )
    parser.add_argument(
        "--trial-dir",
        type=str,
        default="",
        help="Optional: export config for this specific trial directory instead of selecting best.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="",
        help="Output YAML path. Default: <exp-dir>/best_train_cfg_sofar.yaml",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="",
        help="Tune metric column name (defaults to tune.metric from the composed config).",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="",
        help="Optimization mode: min|max (defaults to tune.mode from the composed config).",
    )
    parser.add_argument(
        "--scope",
        type=str,
        default="best",
        choices=["best", "last"],
        help="How to score each trial from result.json: best (across all rows) or last row.",
    )
    parser.add_argument(
        "--min-epoch",
        type=int,
        default=1,
        help="Ignore rows with epoch < min_epoch (useful to match ASHA grace_period). Default: 1",
    )

    # Let users pass arbitrary Hydra overrides after our known args.
    args, hydra_overrides = parser.parse_known_args()

    repo_root = resolve_repo_root().resolve()
    sys.path.insert(0, str(repo_root))

    cfg_dict = _compose_hydra_cfg(repo_root, config_name=args.config_name, overrides=hydra_overrides)

    # Extract tune/ray sections and leave the rest as the training config (same convention as tune.py).
    tune_cfg = dict(cfg_dict.get("tune", {}) or {})
    cfg_dict.pop("tune", None)
    cfg_dict.pop("ray", None)
    cfg_dict.pop("hydra", None)
    train_cfg: dict[str, Any] = cfg_dict

    metric = str(args.metric).strip() or str(tune_cfg.get("metric", "val_loss")).strip()
    mode = str(args.mode).strip() or str(tune_cfg.get("mode", "min")).strip()

    exp_dir: Optional[Path] = None
    if str(args.trial_dir).strip():
        trial_dir = Path(str(args.trial_dir).strip()).expanduser().resolve()
        exp_dir = trial_dir.parent
        candidates = [trial_dir]
    else:
        if str(args.exp_dir).strip():
            exp_dir = Path(str(args.exp_dir).strip()).expanduser().resolve()
        else:
            name = str(tune_cfg.get("name", "tune")).strip()
            storage_path = _as_path_under_root(str(tune_cfg.get("storage_path", "ray_results")), repo_root=repo_root)
            exp_dir = (storage_path / name).resolve()
        candidates = list(_iter_trial_dirs(exp_dir))

    if exp_dir is None:
        raise SystemExit("Failed to determine exp_dir")
    if not exp_dir.exists():
        raise SystemExit(f"Experiment directory does not exist: {exp_dir}")
    if not candidates:
        raise SystemExit(f"No trial dirs with params.json + result.json found under: {exp_dir}")

    best: Optional[TrialScore] = None
    best_params: Optional[dict[str, Any]] = None

    for td in candidates:
        score = _score_trial_from_result_json(
            td,
            metric=metric,
            mode=mode,
            scope=str(args.scope),
            min_epoch=int(args.min_epoch),
        )
        if score is None:
            continue
        try:
            params = _read_params(td)
        except Exception:
            continue

        if best is None:
            best, best_params = score, params
        else:
            chosen = _pick_better(best, score)
            if chosen is score:
                best, best_params = score, params

    if best is None or best_params is None:
        raise SystemExit(
            f"No usable trial results found (metric={metric!r}). "
            f"Try a different --metric, or lower --min-epoch."
        )

    # Build resolved training cfg = base training cfg + sampled hyperparams.
    best_cfg = copy.deepcopy(train_cfg)
    for k, v in best_params.items():
        _set_by_dotted_path(best_cfg, str(k), v)

    out_path: Path
    if str(args.out).strip():
        out_path = Path(str(args.out).strip()).expanduser().resolve()
    else:
        out_path = (exp_dir / "best_train_cfg_sofar.yaml").resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    import yaml

    header_lines = [
        "# --- Ray Tune best-so-far export ---",
        f"# metric: {best.metric}",
        f"# mode: {best.mode}",
        f"# scope: {best.scope}",
        f"# min_epoch: {int(args.min_epoch)}",
        f"# best_{best.metric}: {best.value}",
        f"# epoch: {best.epoch}",
        f"# done: {bool(best.done)}",
        f"# trial_id: {best.trial_id}",
        f"# trial_dir: {best.trial_dir}",
        "# ----------------------------------",
    ]

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(header_lines) + "\n\n")
        yaml.safe_dump(best_cfg, f, sort_keys=False, allow_unicode=True)

    print(f"[export_best_tune_cfg] exp_dir: {exp_dir}")
    print(f"[export_best_tune_cfg] best trial: {best.trial_id} @ epoch {best.epoch} ({best.metric}={best.value})")
    print(f"[export_best_tune_cfg] wrote: {out_path}")
    print(f"[export_best_tune_cfg] run: python {repo_root/'train.py'} --config {out_path}")


if __name__ == "__main__":
    main()


