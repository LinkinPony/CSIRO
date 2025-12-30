from __future__ import annotations

import sys
from pathlib import Path


def _resolve_repo_root_for_wrapper() -> Path:
    """
    Minimal repo-root resolver for this wrapper script.

    We keep it here (instead of importing from `src.*`) so the wrapper can run even
    when executed from packaged layouts like `weights/scripts/`, where `src/` is
    located in the parent directory and is not on sys.path by default.
    """
    here = Path(__file__).resolve().parent
    for cand in (here, here.parent):
        if (cand / "configs").is_dir() and (cand / "src").is_dir():
            return cand
    return here


def main() -> None:
    repo_root = _resolve_repo_root_for_wrapper()
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from src.tabular.tabpfn_train import main as _main

    _main()


if __name__ == "__main__":
    main()


