from __future__ import annotations

from pathlib import Path


def resolve_repo_root(*, start: Path | None = None) -> Path:
    """
    Best-effort resolve the project root directory that contains both `conf/` and `src/`.

    This mirrors the convention used by the existing tools scripts in this repo.
    """
    start_path = (start or Path(__file__)).resolve()

    # If `start` is a file, search from its parent.
    base = start_path if start_path.is_dir() else start_path.parent

    for p in [base] + list(base.parents):
        if (p / "conf").is_dir() and (p / "src").is_dir():
            return p

    cwd = Path.cwd().resolve()
    for p in [cwd] + list(cwd.parents):
        if (p / "conf").is_dir() and (p / "src").is_dir():
            return p

    return cwd

