from __future__ import annotations

import threading
import time
import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional

from tune_viewer.parsing.tune_configs import (
    TuneConfigSummary,
    index_tune_configs_by_name,
    infer_min_epoch_for_best,
    load_tune_configs,
)
from tune_viewer.parsing.utils import guess_trial_id_from_dir, read_json, read_jsonl, safe_float, safe_int


@dataclass(frozen=True)
class FileFingerprint:
    mtime_ns: int
    size: int


def _fingerprint(path: Path) -> Optional[FileFingerprint]:
    try:
        st = path.stat()
    except Exception:
        return None
    return FileFingerprint(mtime_ns=int(getattr(st, "st_mtime_ns", int(st.st_mtime * 1e9))), size=int(st.st_size))


def _is_trial_dir(p: Path) -> bool:
    return p.is_dir() and (p / "params.json").is_file() and (p / "result.json").is_file()


def _iter_trial_dirs(exp_dir: Path) -> list[Path]:
    out: list[Path] = []
    try:
        for p in exp_dir.iterdir():
            if _is_trial_dir(p):
                out.append(p)
    except Exception:
        return []
    out.sort(key=lambda x: x.name)
    return out


def _has_error_marker(trial_dir: Path) -> bool:
    # Ray Tune typically writes `error.txt` (and sometimes `error.pkl`) on failure.
    if (trial_dir / "error.txt").is_file():
        return True
    if (trial_dir / "error.pkl").is_file():
        return True
    # Loose fallback: any file starting with "error" in top-level of trial dir.
    try:
        for p in trial_dir.iterdir():
            if p.is_file() and p.name.lower().startswith("error"):
                return True
    except Exception:
        pass
    return False


@dataclass
class TrialCacheEntry:
    params_fp: Optional[FileFingerprint] = None
    params: dict[str, Any] = field(default_factory=dict)

    result_fp: Optional[FileFingerprint] = None
    records: list[dict[str, Any]] = field(default_factory=list)

    # Cache computed summaries keyed by (metric, mode, min_epoch).
    summaries: dict[tuple[str, str, int], dict[str, Any]] = field(default_factory=dict)
    available_metrics: list[str] = field(default_factory=list)

    lightning_fp: Optional[FileFingerprint] = None
    lightning_csv_relpath: Optional[str] = None
    lightning_available_columns: list[str] = field(default_factory=list)
    lightning_points_full: list[dict[str, Any]] = field(default_factory=list)


class TuneResultsIndex:
    def __init__(self, *, results_root: Path, conf_dir: Path) -> None:
        self.results_root = results_root.resolve()
        self.conf_dir = conf_dir.resolve()

        self._lock = threading.Lock()
        self._trial_cache: dict[str, TrialCacheEntry] = {}

        # Tune configs cache
        self._tune_cfg_fps: dict[str, FileFingerprint] = {}
        self._tune_cfg_index: dict[str, TuneConfigSummary] = {}

    def _refresh_tune_config_index(self) -> dict[str, TuneConfigSummary]:
        fps: dict[str, FileFingerprint] = {}
        try:
            tune_files = sorted(self.conf_dir.glob("tune*.yaml"))
        except Exception:
            tune_files = []

        changed = False
        for p in tune_files:
            fp = _fingerprint(p)
            if fp is None:
                continue
            fps[p.name] = fp
            if self._tune_cfg_fps.get(p.name) != fp:
                changed = True

        # Also detect deletions.
        if set(fps.keys()) != set(self._tune_cfg_fps.keys()):
            changed = True

        if not changed and self._tune_cfg_index:
            return self._tune_cfg_index

        configs = load_tune_configs(self.conf_dir)
        index = index_tune_configs_by_name(configs)
        self._tune_cfg_fps = fps
        self._tune_cfg_index = index
        return index

    def get_tune_configs(self) -> list[dict[str, Any]]:
        with self._lock:
            idx = self._refresh_tune_config_index()
            # Return in a stable order by name.
            out = []
            for name in sorted(idx.keys()):
                c = idx[name]
                out.append(
                    {
                        "config_file": c.config_file,
                        "name": c.name,
                        "metric": c.metric,
                        "mode": c.mode,
                        "search_space": c.search_space,
                        "scheduler": c.scheduler,
                        "min_epoch_for_best": infer_min_epoch_for_best(c),
                    }
                )
            return out

    def _get_or_create_trial_cache(self, trial_dir: Path) -> TrialCacheEntry:
        key = str(trial_dir.resolve())
        entry = self._trial_cache.get(key)
        if entry is None:
            entry = TrialCacheEntry()
            self._trial_cache[key] = entry
        return entry

    def _load_params_cached(self, trial_dir: Path) -> dict[str, Any]:
        params_path = trial_dir / "params.json"
        fp = _fingerprint(params_path)
        with self._lock:
            entry = self._get_or_create_trial_cache(trial_dir)
            if fp is not None and entry.params_fp == fp and entry.params:
                return entry.params

        params: dict[str, Any] = {}
        if params_path.is_file():
            try:
                params = read_json(params_path)
            except Exception:
                params = {}

        with self._lock:
            entry = self._get_or_create_trial_cache(trial_dir)
            entry.params_fp = fp
            entry.params = params
        return params

    def _load_records_cached(self, trial_dir: Path) -> list[dict[str, Any]]:
        result_path = trial_dir / "result.json"
        fp = _fingerprint(result_path)
        with self._lock:
            entry = self._get_or_create_trial_cache(trial_dir)
            if fp is not None and entry.result_fp == fp:
                return entry.records

        records = read_jsonl(result_path) if result_path.is_file() else []

        # Infer available metric keys (numeric, excluding Ray housekeeping keys).
        metric_keys: set[str] = set()
        skip = {
            "trial_id",
            "timestamp",
            "date",
            "time_this_iter_s",
            "time_total_s",
            "training_iteration",
            "iterations_since_restore",
            "time_since_restore",
            "pid",
            "hostname",
            "node_ip",
            "done",
            "config",
            "checkpoint_dir_name",
            "epoch",
        }
        for rec in records:
            for k, v in (rec or {}).items():
                if k in skip:
                    continue
                if safe_float(v) is not None:
                    metric_keys.add(str(k))
        metrics_sorted = sorted(metric_keys)

        with self._lock:
            entry = self._get_or_create_trial_cache(trial_dir)
            entry.result_fp = fp
            entry.records = records
            entry.available_metrics = metrics_sorted
            # Invalidate summaries for this trial (because records changed).
            entry.summaries = {}

        return records

    def _find_lightning_metrics_csv(self, trial_dir: Path) -> Optional[Path]:
        """
        Find the most recent Lightning metrics CSV under:
          run/logs/lightning/**/metrics.csv
        """
        base = trial_dir / "run" / "logs" / "lightning"
        if not base.is_dir():
            return None
        candidates: list[Path] = []
        try:
            candidates = list(base.glob("**/metrics.csv"))
        except Exception:
            candidates = []
        if not candidates:
            return None
        candidates = [p for p in candidates if p.is_file()]
        if not candidates:
            return None
        candidates.sort(key=lambda p: _fingerprint(p).mtime_ns if _fingerprint(p) is not None else 0, reverse=True)
        return candidates[0]

    def _parse_lightning_metrics_csv(self, csv_path: Path) -> tuple[list[str], list[dict[str, Any]]]:
        """
        Parse a Lightning `metrics.csv` into:
          - available scalar columns (numeric, excluding epoch/step)
          - points: each is {row_idx, step?, epoch?, <metric>: float}
        """
        available: set[str] = set()
        points: list[dict[str, Any]] = []

        with open(csv_path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None:
                return [], []
            fieldnames = [str(x) for x in reader.fieldnames if x is not None]

            for idx, row in enumerate(reader):
                if not isinstance(row, dict):
                    continue

                ep = safe_int(row.get("epoch"))
                step = safe_int(row.get("step"))

                out: dict[str, Any] = {"row_idx": int(idx), "epoch": ep, "step": step}
                any_metric = False

                for k in fieldnames:
                    if k in ("epoch", "step"):
                        continue
                    v = safe_float(row.get(k))
                    if v is None:
                        continue
                    out[k] = float(v)
                    available.add(k)
                    any_metric = True

                # Keep rows that have an x-axis value or any metric.
                if any_metric or ep is not None or step is not None:
                    points.append(out)

        cols = sorted(available)
        return cols, points

    def _load_lightning_cached(self, trial_dir: Path) -> tuple[Optional[str], list[str], list[dict[str, Any]]]:
        csv_path = self._find_lightning_metrics_csv(trial_dir)
        if csv_path is None:
            with self._lock:
                entry = self._get_or_create_trial_cache(trial_dir)
                entry.lightning_fp = None
                entry.lightning_csv_relpath = None
                entry.lightning_available_columns = []
                entry.lightning_points_full = []
            return None, [], []

        fp = _fingerprint(csv_path)
        rel = None
        try:
            rel = str(csv_path.relative_to(trial_dir))
        except Exception:
            rel = str(csv_path.name)

        with self._lock:
            entry = self._get_or_create_trial_cache(trial_dir)
            if fp is not None and entry.lightning_fp == fp and entry.lightning_points_full:
                return entry.lightning_csv_relpath, entry.lightning_available_columns, entry.lightning_points_full

        cols, points = self._parse_lightning_metrics_csv(csv_path)

        with self._lock:
            entry = self._get_or_create_trial_cache(trial_dir)
            entry.lightning_fp = fp
            entry.lightning_csv_relpath = rel
            entry.lightning_available_columns = cols
            entry.lightning_points_full = points

        return rel, cols, points

    def _compute_trial_summary(
        self,
        *,
        exp_name: str,
        trial_dir: Path,
        metric: str,
        mode: str,
        min_epoch_for_best: int,
    ) -> dict[str, Any]:
        # Cached per (metric, mode, min_epoch) as long as result.json fingerprint is unchanged.
        key = (str(metric), str(mode).lower().strip(), int(min_epoch_for_best))

        with self._lock:
            entry = self._get_or_create_trial_cache(trial_dir)
            cached = entry.summaries.get(key)
            if cached is not None:
                return cached

        records = self._load_records_cached(trial_dir)
        params = self._load_params_cached(trial_dir)
        result_fp = _fingerprint(trial_dir / "result.json")

        trial_id = ""
        n_metric = 0
        n_records = 0

        last_epoch: Optional[int] = None
        last_iter: Optional[int] = None
        last_done: Optional[bool] = None
        last_time_total_s: Optional[float] = None
        last_val: Optional[float] = None

        best_val: Optional[float] = None
        best_epoch: Optional[int] = None
        best_iter: Optional[int] = None

        mode_n = str(mode).lower().strip()
        metric_n = str(metric).strip()

        # Always compute these pinned metrics as well (regardless of Tune objective).
        pinned_jobs = [
            ("val_r2", "max"),
            ("val_loss_5d_weighted", "min"),
            ("train_loss_5d_weighted", "min"),
        ]

        # Compute objective + pinned in one pass over records.
        metric_jobs: list[tuple[str, str]] = []
        seen_jobs: set[tuple[str, str]] = set()
        for job in [(metric_n, mode_n), *pinned_jobs]:
            if job in seen_jobs:
                continue
            seen_jobs.add(job)
            metric_jobs.append(job)

        metric_stats: dict[tuple[str, str], dict[str, Any]] = {
            (m, md): {
                "n": 0,
                "last": None,
                "last_epoch": None,
                "last_iter": None,
                "best": None,
                "best_epoch": None,
                "best_iter": None,
            }
            for (m, md) in metric_jobs
        }

        for rec in records:
            if not isinstance(rec, dict):
                continue
            n_records += 1

            if not trial_id:
                tid = str(rec.get("trial_id", "") or "").strip()
                if tid:
                    trial_id = tid

            ep = safe_int(rec.get("epoch"))
            it = safe_int(rec.get("training_iteration"))
            if ep is not None:
                last_epoch = ep
            if it is not None:
                last_iter = it

            done_raw = rec.get("done", None)
            if done_raw is not None:
                last_done = bool(done_raw)
            tts = safe_float(rec.get("time_total_s"))
            if tts is not None:
                last_time_total_s = float(tts)

            for (m, md) in metric_jobs:
                v = safe_float(rec.get(m))
                if v is None:
                    continue
                v_f = float(v)

                st = metric_stats[(m, md)]
                st["n"] = int(st["n"]) + 1
                st["last"] = v_f
                if ep is not None:
                    st["last_epoch"] = int(ep)
                if it is not None:
                    st["last_iter"] = int(it)

                if ep is not None and ep < int(min_epoch_for_best):
                    continue

                best_now = st.get("best")
                if best_now is None:
                    st["best"] = v_f
                    st["best_epoch"] = (None if ep is None else int(ep))
                    st["best_iter"] = (None if it is None else int(it))
                else:
                    better = (v_f < float(best_now)) if md == "min" else (v_f > float(best_now))
                    if better:
                        st["best"] = v_f
                        st["best_epoch"] = (None if ep is None else int(ep))
                        st["best_iter"] = (None if it is None else int(it))

        # Extract objective metric stats.
        obj_st = metric_stats.get((metric_n, mode_n), None)
        if obj_st is not None:
            n_metric = int(obj_st.get("n") or 0)
            last_val = obj_st.get("last")
            best_val = obj_st.get("best")
            best_epoch = obj_st.get("best_epoch")
            best_iter = obj_st.get("best_iter")

        def _pack_metric_summary(m: str, md: str) -> dict[str, Any]:
            st = metric_stats.get((m, md), None)
            if st is None:
                return {
                    "metric": str(m),
                    "mode": str(md),
                    "min_epoch_for_best": int(min_epoch_for_best),
                    "n_metric_records": 0,
                    "best": None,
                    "best_epoch": None,
                    "best_training_iteration": None,
                    "last": None,
                    "last_epoch": None,
                    "last_training_iteration": None,
                }
            return {
                "metric": str(m),
                "mode": str(md),
                "min_epoch_for_best": int(min_epoch_for_best),
                "n_metric_records": int(st.get("n") or 0),
                "best": st.get("best"),
                "best_epoch": st.get("best_epoch"),
                "best_training_iteration": st.get("best_iter"),
                "last": st.get("last"),
                "last_epoch": st.get("last_epoch"),
                "last_training_iteration": st.get("last_iter"),
            }

        if not trial_id:
            trial_id = guess_trial_id_from_dir(trial_dir) or trial_dir.name

        status: str
        if _has_error_marker(trial_dir):
            status = "ERROR"
        elif last_done is True:
            status = "TERMINATED"
        elif n_records == 0:
            status = "NO_DATA"
        else:
            status = "RUNNING"

        out = {
            "exp_name": str(exp_name),
            "trial_dirname": str(trial_dir.name),
            "trial_dir": str(trial_dir),
            "trial_id": str(trial_id),
            "status": status,
            "n_records": int(n_records),
            "n_metric_records": int(n_metric),
            "metric": str(metric),
            "mode": mode_n,
            "min_epoch_for_best": int(min_epoch_for_best),
            "best": (None if best_val is None else float(best_val)),
            "best_epoch": (None if best_epoch is None else int(best_epoch)),
            "best_training_iteration": (None if best_iter is None else int(best_iter)),
            "last": (None if last_val is None else float(last_val)),
            "last_epoch": (None if last_epoch is None else int(last_epoch)),
            "last_training_iteration": (None if last_iter is None else int(last_iter)),
            "time_total_s": (None if last_time_total_s is None else float(last_time_total_s)),
            "result_mtime_ns": (None if result_fp is None else int(result_fp.mtime_ns)),
            "result_size": (None if result_fp is None else int(result_fp.size)),
            "params": params,
            "pinned_metrics": {
                # Always show these regardless of objective metric (if present in result.json).
                "val_r2": _pack_metric_summary("val_r2", "max"),
                "val_loss_5d_weighted": _pack_metric_summary("val_loss_5d_weighted", "min"),
                "train_loss_5d_weighted": _pack_metric_summary("train_loss_5d_weighted", "min"),
            },
        }

        with self._lock:
            entry = self._get_or_create_trial_cache(trial_dir)
            entry.summaries[key] = out
        return out

    def _infer_metric_mode_for_experiment(self, exp_name: str) -> tuple[str, str, int, Optional[TuneConfigSummary]]:
        with self._lock:
            cfg_index = self._refresh_tune_config_index()
            cfg = cfg_index.get(exp_name)

        if cfg is not None and cfg.metric and cfg.mode:
            return str(cfg.metric), str(cfg.mode), infer_min_epoch_for_best(cfg), cfg

        # Fallback defaults.
        return "val_r2", "max", 1, None

    def list_experiments(self) -> list[dict[str, Any]]:
        now = time.time()
        out: list[dict[str, Any]] = []

        try:
            exp_dirs = [p for p in self.results_root.iterdir() if p.is_dir() and not p.name.startswith(".")]
        except Exception:
            exp_dirs = []
        exp_dirs.sort(key=lambda p: p.name)

        for exp_dir in exp_dirs:
            exp_name = exp_dir.name
            metric, mode, min_epoch_for_best, cfg = self._infer_metric_mode_for_experiment(exp_name)

            trials = _iter_trial_dirs(exp_dir)
            n_trials = len(trials)

            counts = {"RUNNING": 0, "TERMINATED": 0, "ERROR": 0, "NO_DATA": 0}
            best_trial: Optional[dict[str, Any]] = None
            last_update_ns: Optional[int] = None
            pinned_best: dict[str, dict[str, Any]] = {
                "train_loss_5d_weighted": {"best": None, "best_trial_id": None, "best_trial_dirname": None},
                "val_loss_5d_weighted": {"best": None, "best_trial_id": None, "best_trial_dirname": None},
            }

            for td in trials:
                summ = self._compute_trial_summary(
                    exp_name=exp_name,
                    trial_dir=td,
                    metric=metric,
                    mode=mode,
                    min_epoch_for_best=min_epoch_for_best,
                )
                st = str(summ.get("status") or "")
                if st in counts:
                    counts[st] += 1
                else:
                    counts[st] = counts.get(st, 0) + 1

                mtime_ns = summ.get("result_mtime_ns")
                if isinstance(mtime_ns, int):
                    last_update_ns = mtime_ns if last_update_ns is None else max(last_update_ns, mtime_ns)

                # Track best values for pinned metrics across trials (for quick experiment-level display).
                pm = summ.get("pinned_metrics") or {}
                if isinstance(pm, dict):
                    for mk in ("train_loss_5d_weighted", "val_loss_5d_weighted"):
                        ms = pm.get(mk) if isinstance(pm.get(mk), dict) else {}
                        vv = (ms or {}).get("best") if isinstance(ms, dict) else None
                        if vv is None:
                            continue
                        try:
                            vv_f = float(vv)
                        except Exception:
                            continue
                        md = str((ms or {}).get("mode") or "min").lower().strip()
                        cur = pinned_best.get(mk, {}).get("best")
                        better_p = (cur is None) or ((vv_f < float(cur)) if md == "min" else (vv_f > float(cur)))
                        if better_p:
                            pinned_best[mk]["best"] = vv_f
                            pinned_best[mk]["best_trial_id"] = summ.get("trial_id")
                            pinned_best[mk]["best_trial_dirname"] = summ.get("trial_dirname")

                v = summ.get("best")
                if v is None:
                    continue
                if best_trial is None:
                    best_trial = summ
                else:
                    a = float(best_trial["best"])  # type: ignore[arg-type]
                    b = float(v)
                    better = (b < a) if str(mode).lower().strip() == "min" else (b > a)
                    if better:
                        best_trial = summ

            out.append(
                {
                    "name": exp_name,
                    "path": str(exp_dir),
                    "n_trials": int(n_trials),
                    "metric": metric,
                    "mode": str(mode).lower().strip(),
                    "min_epoch_for_best": int(min_epoch_for_best),
                    "counts": counts,
                    "best": (None if best_trial is None else best_trial.get("best")),
                    "best_trial_id": (None if best_trial is None else best_trial.get("trial_id")),
                    "best_trial_dirname": (None if best_trial is None else best_trial.get("trial_dirname")),
                    "pinned_best": pinned_best,
                    "last_update_ns": last_update_ns,
                    "now_s": float(now),
                    "tune_config_file": (None if cfg is None else cfg.config_file),
                }
            )

        return out

    def list_trials(self, *, exp_name: str) -> list[dict[str, Any]]:
        exp_dir = (self.results_root / exp_name).resolve()
        if not exp_dir.is_dir():
            return []

        metric, mode, min_epoch_for_best, _cfg = self._infer_metric_mode_for_experiment(exp_name)
        trials = _iter_trial_dirs(exp_dir)
        out: list[dict[str, Any]] = []
        for td in trials:
            out.append(
                self._compute_trial_summary(
                    exp_name=exp_name,
                    trial_dir=td,
                    metric=metric,
                    mode=mode,
                    min_epoch_for_best=min_epoch_for_best,
                )
            )
        # Sort by best metric (None last)
        out.sort(
            key=lambda r: (r.get("best") is None, r.get("best")),
            reverse=(str(mode).lower().strip() != "min"),
        )
        return out

    def get_trial_timeseries(
        self,
        *,
        exp_name: str,
        trial_dirname: str,
        metrics: Iterable[str],
    ) -> dict[str, Any]:
        trial_dir = (self.results_root / exp_name / trial_dirname).resolve()
        if not _is_trial_dir(trial_dir):
            return {"points": [], "available_metrics": []}

        records = self._load_records_cached(trial_dir)
        params = self._load_params_cached(trial_dir)

        # Include requested metrics (and only numeric values).
        metrics_req = [str(m).strip() for m in metrics if str(m).strip()]
        metrics_req = list(dict.fromkeys(metrics_req))  # stable unique

        points: list[dict[str, Any]] = []
        for idx, rec in enumerate(records):
            if not isinstance(rec, dict):
                continue
            row: dict[str, Any] = {
                "row_idx": int(idx),
                "epoch": safe_int(rec.get("epoch")),
                "training_iteration": safe_int(rec.get("training_iteration")),
                "time_total_s": safe_float(rec.get("time_total_s")),
                "done": (None if rec.get("done") is None else bool(rec.get("done"))),
            }
            for m in metrics_req:
                v = safe_float(rec.get(m))
                if v is not None:
                    row[m] = float(v)
            points.append(row)

        with self._lock:
            available = list(self._get_or_create_trial_cache(trial_dir).available_metrics)

        return {
            "exp_name": exp_name,
            "trial_dirname": trial_dirname,
            "trial_dir": str(trial_dir),
            "params": params,
            "available_metrics": available,
            "points": points,
        }

    def get_trial_lightning_metrics(
        self,
        *,
        exp_name: str,
        trial_dirname: str,
        columns: Iterable[str],
        max_points: int,
    ) -> dict[str, Any]:
        trial_dir = (self.results_root / exp_name / trial_dirname).resolve()
        if not _is_trial_dir(trial_dir):
            return {"csv_relpath": None, "available_columns": [], "points": []}

        rel, available_cols, points_full = self._load_lightning_cached(trial_dir)

        cols_req = [str(c).strip() for c in columns if str(c).strip()]
        cols_req = list(dict.fromkeys(cols_req))
        if not cols_req:
            # Default: try some common tags, otherwise take first few.
            defaults = [
                "train_loss",
                "train_loss_5d_weighted",
                "val_loss",
                "val_loss_5d_weighted",
                "val_r2",
                "lr-AdamW/head",
            ]
            cols_req = [c for c in defaults if c in set(available_cols)]
            if not cols_req:
                cols_req = available_cols[:6]

        # Project full points to requested columns (keep row_idx, step, epoch).
        points: list[dict[str, Any]] = []
        for p in points_full:
            row = {"row_idx": p.get("row_idx"), "step": p.get("step"), "epoch": p.get("epoch")}
            for c in cols_req:
                if c in p:
                    row[c] = p[c]
            points.append(row)

        # Downsample if needed.
        try:
            max_n = int(max_points)
        except Exception:
            max_n = 5000
        max_n = max(200, min(200_000, max_n))
        if len(points) > max_n:
            stride = (len(points) + max_n - 1) // max_n
            sampled = points[::stride]
            if points and sampled and sampled[-1] is not points[-1]:
                sampled.append(points[-1])
            points = sampled

        return {
            "exp_name": exp_name,
            "trial_dirname": trial_dirname,
            "trial_dir": str(trial_dir),
            "csv_relpath": rel,
            "available_columns": available_cols,
            "requested_columns": cols_req,
            "max_points": max_n,
            "points": points,
        }

    def get_trial_train_yaml(
        self,
        *,
        exp_name: str,
        trial_dirname: str,
        max_bytes: int,
    ) -> dict[str, Any]:
        """
        Return a trial's `train.yaml` snapshot.

        - Primary location (single-seed):   <trial>/run/logs/train.yaml
        - Multi-seed fallback:              <trial>/seed_*/logs/train.yaml  (first one)
        - If missing, infer from another trial in the same experiment that has a snapshot,
          and apply this trial's sampled params.json (dotted keys) onto that config.
        """
        trial_dir = (self.results_root / exp_name / trial_dirname).resolve()
        if not _is_trial_dir(trial_dir):
            return {
                "exp_name": exp_name,
                "trial_dirname": trial_dirname,
                "trial_dir": str(trial_dir),
                "yaml": "",
                "inferred": False,
                "source_kind": None,
                "source_trial_dirname": None,
                "source_relpath": None,
                "applied_params": False,
                "applied_params_count": 0,
            }

        def _read_text(path: Path) -> str:
            try:
                raw = path.read_text(encoding="utf-8", errors="replace")
            except Exception:
                return ""
            if max_bytes and len(raw.encode("utf-8", errors="replace")) > int(max_bytes):
                # Mirror the API's cap behavior roughly (bytes are approximate due to encoding).
                # We keep it simple here to avoid depending on tune_viewer.core.fs from parsing/.
                b = raw.encode("utf-8", errors="replace")[: int(max_bytes)]
                return b.decode("utf-8", errors="replace")
            return raw

        def _find_train_yaml(td: Path) -> tuple[Optional[Path], Optional[str], Optional[str]]:
            p = td / "run" / "logs" / "train.yaml"
            if p.is_file():
                return p, "run/logs/train.yaml", "run"
            # Multi-seed trials: seed_*/logs/train.yaml
            try:
                seed_paths = sorted(td.glob("seed_*/logs/train.yaml"))
            except Exception:
                seed_paths = []
            for sp in seed_paths:
                if sp.is_file():
                    try:
                        rel = str(sp.relative_to(td))
                    except Exception:
                        rel = str(sp.name)
                    return sp, rel, "seed"
            return None, None, None

        # 1) Use this trial's own snapshot if present.
        p_self, rel_self, kind_self = _find_train_yaml(trial_dir)
        if p_self is not None:
            return {
                "exp_name": exp_name,
                "trial_dirname": trial_dirname,
                "trial_dir": str(trial_dir),
                "yaml": _read_text(p_self),
                "inferred": False,
                "source_kind": str(kind_self or "self"),
                "source_trial_dirname": str(trial_dirname),
                "source_relpath": rel_self,
                "applied_params": False,
                "applied_params_count": 0,
            }

        # 2) Infer from another trial in the same experiment.
        params = self._load_params_cached(trial_dir)
        exp_dir = (self.results_root / exp_name).resolve()

        def _set_by_dotted_path(d: dict, path: str, value: Any) -> bool:
            parts = [p for p in str(path).split(".") if p]
            if not parts:
                return False
            cur: Any = d
            for p in parts[:-1]:
                if not isinstance(cur, dict):
                    return False
                if p not in cur or not isinstance(cur[p], dict):
                    cur[p] = {}
                cur = cur[p]
            if isinstance(cur, dict):
                cur[parts[-1]] = value
                return True
            return False

        try:
            import yaml  # type: ignore
        except Exception:
            yaml = None  # type: ignore[assignment]

        if yaml is not None and exp_dir.is_dir():
            for other_td in _iter_trial_dirs(exp_dir):
                if other_td.name == str(trial_dirname):
                    continue
                p_other, rel_other, kind_other = _find_train_yaml(other_td)
                if p_other is None:
                    continue
                base_text = _read_text(p_other)
                if not base_text.strip():
                    continue
                try:
                    base_cfg = yaml.safe_load(base_text)
                except Exception:
                    continue
                if not isinstance(base_cfg, dict):
                    continue

                applied = 0
                for k, v in (params or {}).items():
                    if _set_by_dotted_path(base_cfg, str(k), v):
                        applied += 1

                try:
                    out_yaml = yaml.safe_dump(base_cfg, sort_keys=False, allow_unicode=True)
                except Exception:
                    continue

                return {
                    "exp_name": exp_name,
                    "trial_dirname": trial_dirname,
                    "trial_dir": str(trial_dir),
                    "yaml": str(out_yaml or ""),
                    "inferred": True,
                    "source_kind": "other_trial",
                    "source_trial_dirname": str(other_td.name),
                    "source_relpath": rel_other,
                    "applied_params": True,
                    "applied_params_count": int(applied),
                }

            # Repo fallback: if the experiment has no other snapshots, fall back to configs/train.yaml
            # (best-effort; may not include all Hydra overrides for the Tune experiment).
            repo_root = self.conf_dir.parent
            for cfg_path in [
                repo_root / "configs" / "train.yaml",
                repo_root / "configs" / "train_backup.yaml",
            ]:
                if not cfg_path.is_file():
                    continue
                base_text = _read_text(cfg_path)
                if not base_text.strip():
                    continue
                try:
                    base_cfg = yaml.safe_load(base_text)
                except Exception:
                    continue
                if not isinstance(base_cfg, dict):
                    continue

                applied = 0
                for k, v in (params or {}).items():
                    if _set_by_dotted_path(base_cfg, str(k), v):
                        applied += 1

                try:
                    out_yaml = yaml.safe_dump(base_cfg, sort_keys=False, allow_unicode=True)
                except Exception:
                    continue

                rel_cfg = None
                try:
                    rel_cfg = str(cfg_path.relative_to(repo_root))
                except Exception:
                    rel_cfg = str(cfg_path.name)

                return {
                    "exp_name": exp_name,
                    "trial_dirname": trial_dirname,
                    "trial_dir": str(trial_dir),
                    "yaml": str(out_yaml or ""),
                    "inferred": True,
                    "source_kind": "repo_config",
                    "source_trial_dirname": None,
                    "source_relpath": rel_cfg,
                    "applied_params": True,
                    "applied_params_count": int(applied),
                }

        return {
            "exp_name": exp_name,
            "trial_dirname": trial_dirname,
            "trial_dir": str(trial_dir),
            "yaml": "",
            "inferred": False,
            "source_kind": None,
            "source_trial_dirname": None,
            "source_relpath": None,
            "applied_params": False,
            "applied_params_count": 0,
        }

