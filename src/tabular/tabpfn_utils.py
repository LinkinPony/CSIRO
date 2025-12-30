from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Optional, Tuple


def resolve_repo_root(start: Optional[Path] = None) -> Path:
    """
    Best-effort resolve the project root directory containing both `configs/` and `src/`.

    This works for:
    - running from the repo root
    - running from packaged layouts like `weights/scripts/*` (where parent has configs/src)
    - importing from inside `src/` (walk parents until we find the root)
    """
    p = (start or Path(__file__).resolve()).resolve()
    here = p if p.is_dir() else p.parent
    for cand in [here, *here.parents]:
        if (cand / "configs").is_dir() and (cand / "src").is_dir():
            return cand
    return here


def ensure_on_sys_path(path: Path) -> None:
    """Ensure `path` is on sys.path (prepended)."""
    p = str(path)
    if p and p not in sys.path:
        sys.path.insert(0, p)


def resolve_under_repo(repo_root: Path, p: str | Path) -> Path:
    """Resolve `p` under `repo_root` if it is not already absolute."""
    pp = Path(str(p)).expanduser()
    if pp.is_absolute():
        return pp
    return (repo_root / pp).resolve()


def parse_image_size_hw(value: Any) -> Tuple[int, int]:
    """
    Accept int (square) or [W, H]; return (H, W).
    Mirrors the project's convention used in training/inference helpers.
    """
    try:
        if isinstance(value, (list, tuple)) and len(value) == 2:
            w, h = int(value[0]), int(value[1])
            return int(h), int(w)
        v = int(value)
        return int(v), int(v)
    except Exception:
        v = int(value)
        return int(v), int(v)


def configure_tabpfn_env(*, repo_root: Path, enable_telemetry: bool, model_cache_dir: str | None) -> None:
    """
    Configure TabPFN env vars.

    Must be called BEFORE importing `tabpfn` if you want the settings to apply.
    """
    if not bool(enable_telemetry):
        os.environ["TABPFN_DISABLE_TELEMETRY"] = "1"
    else:
        os.environ.pop("TABPFN_DISABLE_TELEMETRY", None)

    cache_dir = str(model_cache_dir or "").strip()
    if cache_dir:
        os.environ["TABPFN_MODEL_CACHE_DIR"] = str(resolve_under_repo(repo_root, cache_dir))


def import_tabpfn_regressor(repo_root: Path):
    """
    Import TabPFN from either an installed package or the vendored third_party copy.

    We do **not** write anything into `third_party/` (per workspace rules).
    """
    from src.tabular.tabpfn_patches import (  # local import to avoid hard dependency at module import time
        apply_tabpfn_runtime_patches,
        install_tabpfn_common_utils_shim,
    )

    # Ensure TabPFN optional deps are available as offline shims before importing.
    install_tabpfn_common_utils_shim()

    try:
        from tabpfn import TabPFNRegressor  # type: ignore

        apply_tabpfn_runtime_patches()
        return TabPFNRegressor
    except Exception:
        third_party_src = repo_root / "third_party" / "TabPFN" / "src"
        if third_party_src.is_dir():
            sys.path.insert(0, str(third_party_src))
        from tabpfn import TabPFNRegressor  # type: ignore

        apply_tabpfn_runtime_patches()
        return TabPFNRegressor


def parse_tabpfn_inference_precision(value: Any):
    """
    Normalize a user-facing inference precision config value into what TabPFN expects.

    TabPFN 2.5 accepts:
      - "auto" | "autocast" (strings)
      - torch.dtype (e.g., torch.float32 / torch.float16 / torch.bfloat16)

    We additionally accept common string aliases like "float32", "fp16", "bf16", etc.
    """
    import torch

    if value in (None, "", "null"):
        return "auto"

    # Pass-through for correct types
    if isinstance(value, torch.dtype):
        return value

    s = str(value).strip()
    if not s:
        return "auto"
    key = s.strip().lower().replace("torch.", "")

    if key in ("auto", "autocast"):
        return key

    dtype_map = {
        # fp32
        "fp32": torch.float32,
        "float32": torch.float32,
        "f32": torch.float32,
        "32": torch.float32,
        # fp16
        "fp16": torch.float16,
        "float16": torch.float16,
        "f16": torch.float16,
        "16": torch.float16,
        "half": torch.float16,
        # bf16
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
    }
    if key in dtype_map:
        return dtype_map[key]

    raise ValueError(
        f"Unsupported TabPFN inference_precision={value!r}. "
        "Expected one of: auto|autocast|fp32|float32|fp16|float16|bf16|bfloat16."
    )

