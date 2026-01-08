#!/usr/bin/env python3
"""
Summarize all `outputs/**/kfold_swa_metrics.json` into two aggregated JSON files.

Outputs:
  1) Full summary JSON: { "<version>": <full metrics JSON>, ... }
  2) R2-only summary JSON: { "<version>": <average.r2 | null>, ... }

Where:
  - <version> is inferred as the first directory under the outputs directory:
      outputs/<version>/kfold_swa_metrics.json  -> version == "<version>"
  - If multiple metrics files exist for the same version, the newest (by mtime) is used.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _resolve_path(repo_root: Path, p: str) -> Path:
    path = Path(p).expanduser()
    if not path.is_absolute():
        path = repo_root / path
    return path.resolve()


def _infer_version(outputs_dir: Path, metrics_path: Path) -> str:
    rel = metrics_path.relative_to(outputs_dir)
    # outputs/<version>/... -> rel.parts[0] is version
    return rel.parts[0] if rel.parts else metrics_path.parent.name


def _safe_load_json(path: Path) -> Optional[Any]:
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"[WARN] Failed to load JSON: {path} ({e})")
        return None


def _extract_average_r2(metrics: Any) -> Optional[float]:
    if not isinstance(metrics, dict):
        return None
    avg = metrics.get("average", None)
    if not isinstance(avg, dict):
        return None
    r2 = avg.get("r2", None)
    if isinstance(r2, (int, float)):
        return float(r2)
    return None


def _sort_r2_items(items: List[Tuple[str, Optional[float]]]) -> List[Tuple[str, Optional[float]]]:
    # Put valid r2 values first (descending), then None at the bottom.
    def key_fn(x: Tuple[str, Optional[float]]) -> Tuple[int, float]:
        _, r2 = x
        if r2 is None:
            return (0, float("-inf"))
        return (1, float(r2))

    return sorted(items, key=key_fn, reverse=True)


def main() -> int:
    repo_root = Path(__file__).resolve().parent

    p = argparse.ArgumentParser(
        description=(
            "Aggregate outputs/**/kfold_swa_metrics.json into two summary JSON files "
            "(full metrics + average.r2 only)."
        )
    )
    p.add_argument(
        "--outputs-dir",
        type=str,
        default="outputs",
        help="Outputs directory to scan (default: outputs).",
    )
    p.add_argument(
        "--metrics-filename",
        type=str,
        default="kfold_swa_metrics.json",
        help="Metrics filename to look for (default: kfold_swa_metrics.json).",
    )
    p.add_argument(
        "--out-full",
        type=str,
        default="outputs_kfold_swa_metrics_summary.json",
        help="Path to write the full summary JSON (default: repo root).",
    )
    p.add_argument(
        "--out-r2",
        type=str,
        default="outputs_kfold_swa_metrics_average_r2_summary.json",
        help="Path to write the average.r2-only summary JSON (default: repo root).",
    )
    args = p.parse_args()

    outputs_dir = _resolve_path(repo_root, args.outputs_dir)
    out_full = _resolve_path(repo_root, args.out_full)
    out_r2 = _resolve_path(repo_root, args.out_r2)

    if not outputs_dir.is_dir():
        raise FileNotFoundError(f"outputs_dir does not exist or is not a directory: {outputs_dir}")

    metric_files = sorted([fp for fp in outputs_dir.rglob(args.metrics_filename) if fp.is_file()])
    if not metric_files:
        print(f"[INFO] No metrics files found under {outputs_dir} (filename={args.metrics_filename}).")
        # Still write empty outputs for determinism.
        out_full.write_text("{}\n", encoding="utf-8")
        out_r2.write_text("{}\n", encoding="utf-8")
        print(f"[INFO] Wrote empty: {out_full}")
        print(f"[INFO] Wrote empty: {out_r2}")
        return 0

    by_version: Dict[str, List[Path]] = defaultdict(list)
    for fp in metric_files:
        try:
            version = _infer_version(outputs_dir, fp)
        except Exception as e:
            print(f"[WARN] Skipping file with unresolvable version: {fp} ({e})")
            continue
        by_version[version].append(fp)

    full_summary: Dict[str, Any] = {}
    used_paths: Dict[str, Path] = {}
    for version, paths in sorted(by_version.items(), key=lambda kv: kv[0]):
        if not paths:
            continue
        if len(paths) > 1:
            newest = max(paths, key=lambda pth: pth.stat().st_mtime)
            print(
                f"[WARN] Multiple metrics files found for version={version}; "
                f"using newest by mtime: {newest} (candidates={len(paths)})"
            )
            chosen = newest
        else:
            chosen = paths[0]

        metrics = _safe_load_json(chosen)
        if metrics is None:
            print(f"[WARN] Skipping version={version} due to JSON load failure: {chosen}")
            continue

        full_summary[version] = metrics
        used_paths[version] = chosen

    # Write full summary (version-sorted)
    out_full.parent.mkdir(parents=True, exist_ok=True)
    with out_full.open("w", encoding="utf-8") as f:
        json.dump(full_summary, f, ensure_ascii=False, indent=2)
        f.write("\n")

    # Build + write r2 summary (sorted by r2 desc, None last)
    r2_items: List[Tuple[str, Optional[float]]] = [
        (version, _extract_average_r2(metrics)) for version, metrics in full_summary.items()
    ]
    r2_items = _sort_r2_items(r2_items)
    r2_summary: Dict[str, Optional[float]] = {version: r2 for version, r2 in r2_items}

    out_r2.parent.mkdir(parents=True, exist_ok=True)
    with out_r2.open("w", encoding="utf-8") as f:
        json.dump(r2_summary, f, ensure_ascii=False, indent=2)
        f.write("\n")

    # Print a small console summary.
    print(f"[OK] Found {len(metric_files)} metrics files, summarized {len(full_summary)} versions.")
    print(f"[OK] Wrote full summary: {out_full}")
    print(f"[OK] Wrote r2 summary:   {out_r2}")
    # Show top 10 by r2
    top = [(v, r2) for v, r2 in r2_items if r2 is not None][:10]
    if top:
        print("[TOP] average.r2 (desc):")
        for v, r2 in top:
            src = used_paths.get(v)
            src_str = f"  ({src})" if src is not None else ""
            print(f"  - {v}: {r2:.6f}{src_str}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


