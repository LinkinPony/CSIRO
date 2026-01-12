from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from tune_viewer.core.settings import get_settings
from tune_viewer.core.fs import PathTraversalError, read_text_capped, resolve_under_root, tail_text_lines
from tune_viewer.parsing.index import TuneResultsIndex


def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title="CSIRO Tune Viewer",
        version="0.1.0",
    )

    # Dev-friendly CORS (local viewer). Restrict later if needed.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/api/health")
    def health() -> dict[str, object]:
        return {
            "ok": True,
            "results_root": str(settings.results_root),
            "repo_root": str(settings.repo_root),
            "poll_seconds": int(settings.poll_seconds),
        }

    index = TuneResultsIndex(results_root=settings.results_root, conf_dir=(settings.repo_root / "conf"))

    @app.get("/api/tune-configs")
    def tune_configs() -> list[dict[str, Any]]:
        return index.get_tune_configs()

    @app.get("/api/experiments")
    def experiments() -> list[dict[str, Any]]:
        return index.list_experiments()

    @app.get("/api/experiments/{exp_name}/trials")
    def experiment_trials(exp_name: str) -> list[dict[str, Any]]:
        # Path safety: exp_name is used for filesystem access.
        try:
            _ = resolve_under_root(root=settings.results_root, rel_path=exp_name)
        except PathTraversalError as e:
            raise HTTPException(status_code=400, detail=str(e))
        return index.list_trials(exp_name=str(exp_name))

    @app.get("/api/experiments/{exp_name}/trials/{trial_dirname}/timeseries")
    def trial_timeseries(
        exp_name: str,
        trial_dirname: str,
        metrics: str = Query(default="val_r2", description="Comma-separated metrics, e.g. val_r2,val_loss"),
    ) -> dict[str, Any]:
        try:
            _ = resolve_under_root(root=settings.results_root, rel_path=f"{exp_name}/{trial_dirname}")
        except PathTraversalError as e:
            raise HTTPException(status_code=400, detail=str(e))

        metric_list = [m.strip() for m in str(metrics).split(",") if m.strip()]
        return index.get_trial_timeseries(exp_name=str(exp_name), trial_dirname=str(trial_dirname), metrics=metric_list)

    @app.get("/api/experiments/{exp_name}/trials/{trial_dirname}/lightning/metrics")
    def trial_lightning_metrics(
        exp_name: str,
        trial_dirname: str,
        columns: str = Query(default="", description="Comma-separated scalar columns (tags) from metrics.csv"),
        max_points: int = Query(default=5000, ge=200, le=200000),
    ) -> dict[str, Any]:
        try:
            _ = resolve_under_root(root=settings.results_root, rel_path=f"{exp_name}/{trial_dirname}")
        except PathTraversalError as e:
            raise HTTPException(status_code=400, detail=str(e))
        cols = [c.strip() for c in str(columns).split(",") if c.strip()]
        return index.get_trial_lightning_metrics(
            exp_name=str(exp_name),
            trial_dirname=str(trial_dirname),
            columns=cols,
            max_points=int(max_points),
        )

    def _list_files_under(dir_path: Path, *, max_entries: int = 2000) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        queue: list[Path] = [dir_path]
        entries = 0

        def should_recurse(rel_path: str) -> bool:
            # Keep traversal bounded and focused on the usual artifact locations.
            if rel_path == "run":
                return True
            if rel_path.startswith("run/checkpoints"):
                # Typically only one level of files (e.g. last.ckpt).
                return rel_path.count("/") <= 2
            if rel_path.startswith("run/logs"):
                # Allow a couple of levels for lightning/tensorboard version dirs.
                return rel_path.count("/") <= 5
            return False

        while queue and entries < max_entries:
            cur = queue.pop(0)
            try:
                children = sorted(cur.iterdir(), key=lambda p: (not p.is_dir(), p.name))
            except Exception:
                continue
            for p in children:
                try:
                    st = p.stat()
                except Exception:
                    continue
                rel = str(p.relative_to(dir_path))
                out.append(
                    {
                        "path": rel,
                        "name": p.name,
                        "is_dir": p.is_dir(),
                        "size": int(st.st_size),
                        "mtime_ns": int(getattr(st, "st_mtime_ns", int(st.st_mtime * 1e9))),
                    }
                )
                entries += 1
                if entries >= max_entries:
                    break
                if p.is_dir() and should_recurse(rel):
                    queue.append(p)
        return out

    @app.get("/api/experiments/{exp_name}/trials/{trial_dirname}/files")
    def trial_files(exp_name: str, trial_dirname: str) -> list[dict[str, Any]]:
        try:
            trial_dir = resolve_under_root(root=settings.results_root, rel_path=f"{exp_name}/{trial_dirname}")
        except PathTraversalError as e:
            raise HTTPException(status_code=400, detail=str(e))
        if not trial_dir.is_dir():
            raise HTTPException(status_code=404, detail="Trial directory not found.")
        return _list_files_under(trial_dir)

    @app.get("/api/experiments/{exp_name}/trials/{trial_dirname}/file")
    def trial_file(
        exp_name: str,
        trial_dirname: str,
        path: str = Query(..., description="Relative path under the trial dir, e.g. run/logs/train.yaml"),
        tail_lines: Optional[int] = Query(default=None, ge=1, le=20000),
    ) -> dict[str, Any]:
        try:
            trial_dir = resolve_under_root(root=settings.results_root, rel_path=f"{exp_name}/{trial_dirname}")
        except PathTraversalError as e:
            raise HTTPException(status_code=400, detail=str(e))
        if not trial_dir.is_dir():
            raise HTTPException(status_code=404, detail="Trial directory not found.")

        try:
            file_path = resolve_under_root(root=trial_dir, rel_path=str(path))
        except PathTraversalError as e:
            raise HTTPException(status_code=400, detail=str(e))
        if not file_path.is_file():
            raise HTTPException(status_code=404, detail="File not found.")

        max_bytes = int(settings.max_file_bytes)
        if tail_lines is not None:
            text = tail_text_lines(file_path, tail_lines=int(tail_lines), max_bytes=max_bytes)
        else:
            text = read_text_capped(file_path, max_bytes=max_bytes)

        try:
            st = file_path.stat()
        except Exception:
            st = None

        return {
            "path": str(path),
            "abs_path": str(file_path),
            "size": (None if st is None else int(st.st_size)),
            "mtime_ns": (None if st is None else int(getattr(st, "st_mtime_ns", int(st.st_mtime * 1e9)))),
            "max_bytes": int(max_bytes),
            "tail_lines": (None if tail_lines is None else int(tail_lines)),
            "content": text,
        }

    return app


app = create_app()

