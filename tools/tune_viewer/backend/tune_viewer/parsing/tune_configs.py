from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class TuneConfigSummary:
    config_file: str
    name: str
    metric: str | None
    mode: str | None
    search_space: dict[str, Any]
    scheduler: dict[str, Any]


def _as_str(v: Any) -> str:
    return str(v).strip()


def _get(d: dict[str, Any], path: str, default: Any = None) -> Any:
    cur: Any = d
    for part in str(path).split("."):
        if not part:
            continue
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def load_tune_configs(conf_dir: Path) -> list[TuneConfigSummary]:
    """
    Load `conf/tune*.yaml` and extract tune metadata.

    Note: These YAML files can contain Hydra interpolations; we keep unresolved values as-is.
    """
    out: list[TuneConfigSummary] = []
    if not conf_dir.is_dir():
        return out

    for p in sorted(conf_dir.glob("tune*.yaml")):
        try:
            raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        except Exception:
            continue
        if not isinstance(raw, dict):
            continue

        tune = raw.get("tune", {}) or {}
        if not isinstance(tune, dict):
            tune = {}

        name = _as_str(tune.get("name") or p.stem)
        metric = _as_str(tune.get("metric")) if tune.get("metric") is not None else None
        mode = _as_str(tune.get("mode")) if tune.get("mode") is not None else None

        search_space = tune.get("search_space", {}) or {}
        if not isinstance(search_space, dict):
            search_space = {}

        sched = tune.get("scheduler", {}) or {}
        if not isinstance(sched, dict):
            sched = {}

        out.append(
            TuneConfigSummary(
                config_file=str(p.name),
                name=name,
                metric=metric,
                mode=mode,
                search_space=dict(search_space),
                scheduler=dict(sched),
            )
        )

    return out


def index_tune_configs_by_name(configs: list[TuneConfigSummary]) -> dict[str, TuneConfigSummary]:
    return {c.name: c for c in configs if c.name}


def infer_min_epoch_for_best(cfg: TuneConfigSummary) -> int:
    """
    Use ASHA grace_period if available; else 1.
    """
    sched_type = _as_str(cfg.scheduler.get("type", "")).lower()
    if sched_type == "asha":
        gp = cfg.scheduler.get("grace_period")
        try:
            if gp is None:
                return 1
            v = int(float(gp))
            return max(1, v)
        except Exception:
            return 1
    return 1

