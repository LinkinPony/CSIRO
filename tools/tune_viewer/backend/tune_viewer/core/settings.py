from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from tune_viewer.core.repo import resolve_repo_root


def _env_int(key: str, default: int) -> int:
    raw = str(os.getenv(key, "")).strip()
    if not raw:
        return int(default)
    try:
        return int(float(raw))
    except Exception:
        return int(default)


def _env_path(key: str, default: str) -> Path:
    raw = str(os.getenv(key, "")).strip()
    if not raw:
        raw = str(default)
    p = Path(raw).expanduser()
    return p.resolve() if p.is_absolute() else p


@dataclass(frozen=True)
class Settings:
    results_root: Path
    repo_root: Path
    poll_seconds: int
    max_file_bytes: int
    max_tail_lines: int


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    # Default to the NFS location used in this project.
    results_root = _env_path("CSIRO_TUNE_VIEWER_RESULTS_ROOT", "/mnt/csiro_nfs/ray_results")
    if not results_root.is_absolute():
        # Treat as relative to repo root if provided relative.
        repo_root = resolve_repo_root()
        results_root = (repo_root / results_root).resolve()
    else:
        repo_root = resolve_repo_root()

    return Settings(
        results_root=results_root,
        repo_root=repo_root,
        poll_seconds=max(1, _env_int("CSIRO_TUNE_VIEWER_POLL_SECONDS", 8)),
        max_file_bytes=max(16_384, _env_int("CSIRO_TUNE_VIEWER_MAX_FILE_BYTES", 2_000_000)),
        max_tail_lines=max(50, _env_int("CSIRO_TUNE_VIEWER_MAX_TAIL_LINES", 4000)),
    )

