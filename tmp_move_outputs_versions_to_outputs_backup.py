#!/usr/bin/env python3
"""
TEMP TOOL: Move selected training runs from `outputs/` to `outputs_backup/`.

Selection source:
  - A JSON file whose top-level keys are version names (e.g. an average.r2 summary):
      {
        "eomt-renew-v2": 0.59,
        "vitdet-512-v3": 0.73,
        ...
      }

For each version, this script attempts to move:
  - outputs/<version>/            -> outputs_backup/<version>/
  - outputs/checkpoints/<version>/ -> outputs_backup/checkpoints/<version>/

Safety:
  - Dry-run by default (unless `--yes` is provided).
  - If destination already exists, a unique suffix is appended to avoid overwriting.
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _resolve_path(repo_root: Path, p: str) -> Path:
    path = Path(p).expanduser()
    if not path.is_absolute():
        path = repo_root / path
    return path.resolve()


def _load_versions(summary_json: Path) -> List[str]:
    with summary_json.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, dict):
        raise TypeError(
            f"Expected top-level JSON object (dict) in {summary_json}, got {type(obj)}"
        )
    # keys are versions
    versions = [str(k) for k in obj.keys()]
    versions = sorted(set(versions))
    return versions


def _unique_dest(dest: Path, *, suffix: Optional[str] = None) -> Path:
    if not dest.exists():
        return dest
    if suffix is None:
        suffix = time.strftime("%Y%m%d-%H%M%S")
    base = dest.name + f"__moved_{suffix}"
    candidate = dest.with_name(base)
    i = 1
    while candidate.exists():
        candidate = dest.with_name(f"{base}_{i}")
        i += 1
    return candidate


def _move(src: Path, dest: Path, *, dry_run: bool) -> Tuple[bool, Path]:
    """
    Returns (did_move, actual_dest).
    """
    if not src.exists():
        print(f"[SKIP] Missing: {src}")
        return (False, dest)

    actual_dest = _unique_dest(dest)
    if dry_run:
        if actual_dest != dest:
            print(f"[PLAN] MOVE {src}  ->  {actual_dest}  (dest exists, suffix applied)")
        else:
            print(f"[PLAN] MOVE {src}  ->  {actual_dest}")
        return (True, actual_dest)

    actual_dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        # Fast path when same filesystem.
        src.rename(actual_dest)
    except Exception:
        # Fallback for cross-device moves.
        shutil.move(str(src), str(actual_dest))
    print(f"[DONE] MOVED {src}  ->  {actual_dest}")
    return (True, actual_dest)


def main() -> int:
    repo_root = Path(__file__).resolve().parent

    p = argparse.ArgumentParser(
        description="Move runs listed in a summary JSON from outputs/ to outputs_backup/."
    )
    p.add_argument(
        "--summary-json",
        type=str,
        default="outputs_kfold_swa_metrics_average_r2_summary_50_50.json",
        help="JSON whose keys are versions to move (default: outputs_kfold_swa_metrics_average_r2_summary_50_50.json).",
    )
    p.add_argument(
        "--outputs-dir",
        type=str,
        default="outputs",
        help="Source outputs directory (default: outputs).",
    )
    p.add_argument(
        "--backup-dir",
        type=str,
        default="outputs_backup",
        help="Destination backup directory (default: outputs_backup).",
    )
    p.add_argument(
        "--no-checkpoints",
        action="store_true",
        help="Do not move outputs/checkpoints/<version> directories.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned moves only (always enabled unless --yes).",
    )
    p.add_argument(
        "--yes",
        action="store_true",
        help="Actually perform the move (otherwise dry-run).",
    )
    args = p.parse_args()

    summary_json = _resolve_path(repo_root, args.summary_json)
    outputs_dir = _resolve_path(repo_root, args.outputs_dir)
    backup_dir = _resolve_path(repo_root, args.backup_dir)

    effective_dry_run = args.dry_run or (not args.yes)
    print(f"[INFO] summary_json={summary_json}")
    print(f"[INFO] outputs_dir={outputs_dir}")
    print(f"[INFO] backup_dir={backup_dir}")
    print(f"[INFO] mode={'DRY_RUN' if effective_dry_run else 'EXECUTE'}")

    if not summary_json.is_file():
        raise FileNotFoundError(f"summary_json not found: {summary_json}")
    if not outputs_dir.is_dir():
        raise FileNotFoundError(f"outputs_dir not found: {outputs_dir}")
    backup_dir.mkdir(parents=True, exist_ok=True)

    versions = _load_versions(summary_json)
    if not versions:
        print("[INFO] No versions found in summary JSON. Nothing to do.")
        return 0

    print(f"[INFO] Versions to move: {len(versions)}")

    moved_any = 0
    planned_any = 0
    for v in versions:
        # 1) outputs/<version>
        src_run_dir = outputs_dir / v
        dst_run_dir = backup_dir / v
        did, _ = _move(src_run_dir, dst_run_dir, dry_run=effective_dry_run)
        if did:
            planned_any += 1
            if not effective_dry_run:
                moved_any += 1

        # 2) outputs/checkpoints/<version>
        if not args.no_checkpoints:
            src_ckpt_dir = outputs_dir / "checkpoints" / v
            dst_ckpt_dir = backup_dir / "checkpoints" / v
            did2, _ = _move(src_ckpt_dir, dst_ckpt_dir, dry_run=effective_dry_run)
            if did2:
                planned_any += 1
                if not effective_dry_run:
                    moved_any += 1

    if effective_dry_run:
        print(f"[OK] Planned {planned_any} move operations (dry-run). Re-run with --yes to execute.")
    else:
        print(f"[OK] Executed {moved_any} move operations.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


